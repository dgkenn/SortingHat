"""Restart safety of the long steps, end to end on SYNTHETIC data: kill each step part-way (test hook), rerun with the same
arguments, and assert (1) the outputs are byte/value-identical to an uninterrupted run, (2) work stored before the kill was
not redone (pyarrow row-group reads / model fits are counted), (3) ``--no-resume`` redoes everything, (4) a stale
checkpoint (other code version) is not reused."""
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow.parquet as pq
import pytest

from sortinghat import checkpoint as ck
from sortinghat import omop_cache
from sortinghat.checkpoint import SimulatedKill
from test_checkpoint import RG, Counter  # noqa: F401  (tests/ on sys.path)
from test_silver_feasibility_inputs import argv_for, load_script, make_inputs, rsf

bc = load_script("build_cohort")
bb = rsf.bb


@pytest.fixture(scope="module")
def store_dir(synth_dir, tmp_path_factory):
    d = tmp_path_factory.mktemp("steps") / "store"
    shutil.copytree(synth_dir, d)
    for f in (d / "OMOP" / "Merged").rglob("*.parquet"):
        pq.write_table(pq.read_table(f), f, row_group_size=RG)
    return d


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.delenv(ck.ENV_FAIL_AFTER, raising=False)
    monkeypatch.setenv(ck.ENV_VERSION, "steps-v1")

    def use_root(name: str) -> Path:
        r = tmp_path / name
        monkeypatch.setenv(ck.ENV_ROOT, str(r))
        monkeypatch.setenv(omop_cache.ENV_ROOT, str(r / "omop"))      # each run owns its OMOP cache (cold start)
        return r
    return SimpleNamespace(use_root=use_root, mp=monkeypatch, tmp=tmp_path)


@pytest.fixture
def counter(monkeypatch):
    return Counter(monkeypatch)


def kill_then_resume(main, argv, mp, kill_after, counter=None):
    """Run ``main`` until the hook kills it after ``kill_after`` stored units, then rerun it. With a ``Counter`` the row-group
    reads of the two runs are kept apart (``counter.killed``, ``counter.rg`` = the resumed run only)."""
    mp.setenv(ck.ENV_FAIL_AFTER, str(kill_after))
    with pytest.raises(SimulatedKill):
        main(argv)
    mp.delenv(ck.ENV_FAIL_AFTER)
    if counter is not None:
        counter.killed = counter.rg
        counter.reset()
    return main(argv)


def same_files(a: Path, b: Path, names) -> None:
    for n in names:
        assert (a / n).read_bytes() == (b / n).read_bytes(), n


# ======================================================================================================== cohort
def test_cohort_build_resumes_to_identical_outputs(store_dir, env, counter, capsys):
    ref, run = env.tmp / "ref", env.tmp / "run"
    env.use_root("ck_ref")
    assert bc.main(["--data", str(store_dir), "--out", str(ref)]) == 0
    full = counter.rg
    assert full > 20
    files = ["local_only/cohort_study1.csv", "local_only/recording_keys.csv", "cohort/flow.json", "cohort/flow.md"]

    env.use_root("ck_run")
    counter.reset()
    assert kill_then_resume(bc.main, ["--data", str(store_dir), "--out", str(run)], env.mp, kill_after=14, counter=counter) == 0
    same_files(ref, run, files)
    assert 0 < counter.killed and 0 < counter.rg < full
    assert counter.killed + counter.rg == full                          # every row group was read exactly once overall
    out = capsys.readouterr().out
    assert "omop_visit_occurrence:" in out and "omop_measurement:" in out and "row groups (cached" in out

    counter.reset()
    assert bc.main(["--data", str(store_dir), "--out", str(run)]) == 0   # relaunch after success: nothing is read
    assert counter.rg == 0
    same_files(ref, run, files)

    counter.reset()
    assert bc.main(["--data", str(store_dir), "--out", str(run), "--no-resume"]) == 0
    assert counter.rg == full                                           # --no-resume: from nothing
    same_files(ref, run, files)

    env.mp.setenv(ck.ENV_VERSION, "steps-v2")                           # other code version: the step's checkpoint is not reused
    capsys.readouterr()
    counter.reset()
    assert bc.main(["--data", str(store_dir), "--out", str(run)]) == 0
    out = capsys.readouterr().out
    assert "fresh run" in out and "sessions-S0001 done" in out and "loaded from checkpoint" not in out
    assert counter.rg == 0                                              # ... but the shared OMOP cache (source data only) is
    same_files(ref, run, files)


@pytest.fixture(scope="module")
def cohort_csv(store_dir, tmp_path_factory):
    """A cohort table (record-level, under local_only/) for the downstream steps."""
    out = tmp_path_factory.mktemp("cohort_for_steps")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(ck.ENV_ROOT, str(out / "ck"))
        mp.setenv(omop_cache.ENV_ROOT, str(out / "omop"))
        assert bc.main(["--data", str(store_dir), "--out", str(out), "--all-sites"]) == 0
    return out / "local_only" / "cohort_study1.csv"


# ============================================================================================== silver labels
def test_silver_labels_resume_to_identical_outputs(store_dir, cohort_csv, env, counter):
    from sortinghat.labels import extract
    ref, run = env.tmp / "ref", env.tmp / "run"

    def argv(out):
        return ["--data", str(store_dir), "--cohort", str(cohort_csv), "--out", str(out / "silver"),
                "--labels-out", str(out / "local_only" / "silver_labels.csv")]
    env.use_root("ck_ref")
    assert extract.main(argv(ref)) == 0
    full = counter.rg
    files = ["silver/silver_report.json", "local_only/silver_labels.csv"]
    assert full > 10

    env.use_root("ck_run")
    counter.reset()
    assert kill_then_resume(extract.main, argv(run), env.mp, kill_after=9, counter=counter) == 0
    same_files(ref, run, files)
    assert 0 < counter.killed and 0 < counter.rg < full
    assert counter.killed + counter.rg == full                          # every row group was read exactly once overall

    counter.reset()
    assert extract.main(argv(run)) == 0
    assert counter.rg == 0
    same_files(ref, run, files)
    counter.reset()
    assert extract.main(argv(run) + ["--no-resume"]) == 0
    assert counter.rg == full
    same_files(ref, run, files)


# ================================================================================================== baselines
def test_baselines_resume_to_identical_outputs(store_dir, cohort_csv, env, counter):
    ref, run = env.tmp / "ref" / "local_only", env.tmp / "run" / "local_only"

    def argv(d):
        return ["--data", str(store_dir), "--cohort", str(cohort_csv), "--sites", "all", "--out", str(d / "bl.parquet")]
    env.use_root("ck_ref")
    assert bb.main(argv(ref)) == 0
    full = counter.rg
    assert full > 10

    env.use_root("ck_run")
    counter.reset()
    assert kill_then_resume(bb.main, argv(run), env.mp, kill_after=25, counter=counter) == 0
    pd.testing.assert_frame_equal(pd.read_parquet(run / "bl.parquet"), pd.read_parquet(ref / "bl.parquet"))
    assert json.loads((run / "bl.columns.json").read_text()) == json.loads((ref / "bl.columns.json").read_text())
    assert 0 < counter.killed and 0 < counter.rg < full
    assert counter.killed + counter.rg == full                          # every row group was read exactly once overall

    counter.reset()
    assert bb.main(argv(run)) == 0
    assert counter.rg == 0
    pd.testing.assert_frame_equal(pd.read_parquet(run / "bl.parquet"), pd.read_parquet(ref / "bl.parquet"))
    counter.reset()
    assert bb.main(argv(run) + ["--no-resume"]) == 0
    assert counter.rg == full
    pd.testing.assert_frame_equal(pd.read_parquet(run / "bl.parquet"), pd.read_parquet(ref / "bl.parquet"))


# ================================================================================================ field audit
def test_field_audit_resumes_to_identical_outputs(store_dir, env, counter):
    from sortinghat.audit import field_audit
    ref, run = env.tmp / "ref", env.tmp / "run"
    env.use_root("ck_ref")
    assert field_audit.main(["--data", str(store_dir), "--out", str(ref)]) == 0
    full = counter.rg
    assert full > 10
    files = ["field_audit.json", "field_audit.md", "local_only/handcheck_sample_ids.csv"]

    env.use_root("ck_run")
    counter.reset()
    assert kill_then_resume(field_audit.main, ["--data", str(store_dir), "--out", str(run)], env.mp, kill_after=30, counter=counter) == 0
    same_files(ref, run, files)
    assert 0 < counter.killed and 0 < counter.rg < full
    assert counter.killed + counter.rg == full                          # every row group was read exactly once overall

    counter.reset()
    assert field_audit.main(["--data", str(store_dir), "--out", str(run), "--seed", "0"]) == 0
    assert counter.rg == 0
    same_files(ref, run, files)
    counter.reset()
    assert field_audit.main(["--data", str(store_dir), "--out", str(run), "--no-resume"]) == 0
    assert counter.rg == full
    same_files(ref, run, files)


# ============================================================================== silver-feasibility model fits
def test_feasibility_fits_resume_per_scheme_fold_and_rung(tmp_path, env, monkeypatch):
    from sortinghat.models import ladder
    paths, _ms, _cohort = make_inputs(tmp_path / "in", signal=1.0, n_per_site=140, seed=5)
    fits = {"n": 0}
    real = ladder.fit_model

    def counting(*a, **kw):
        fits["n"] += 1
        return real(*a, **kw)
    monkeypatch.setattr(ladder, "fit_model", counting)

    def argv(out):
        return argv_for(paths, out, "--splits", "loso", "temporal", "--null-reps", "1")
    env.use_root("ck_ref")
    assert rsf.main(argv(tmp_path / "ref")) == 0
    full = fits["n"]
    assert full > 12
    ref_json = (tmp_path / "ref" / "report.json").read_text()
    ref_md = (tmp_path / "ref" / "report.md").read_text()

    env.use_root("ck_run")
    fits["n"] = 0
    monkeypatch.setenv(ck.ENV_FAIL_AFTER, "7")
    with pytest.raises(SimulatedKill):
        rsf.main(argv(tmp_path / "run"))
    monkeypatch.delenv(ck.ENV_FAIL_AFTER)
    assert fits["n"] == 7
    fits["n"] = 0
    assert rsf.main(argv(tmp_path / "run")) == 0
    assert (tmp_path / "run" / "report.json").read_text() == ref_json
    assert (tmp_path / "run" / "report.md").read_text() == ref_md
    assert fits["n"] == full - 7                                       # 7 fits were stored before the kill, none redone

    fits["n"] = 0
    assert rsf.main(argv(tmp_path / "run")) == 0
    assert fits["n"] == 0
    assert (tmp_path / "run" / "report.json").read_text() == ref_json
    assert rsf.main(argv(tmp_path / "run") + ["--no-resume"]) == 0
    assert fits["n"] == full
    assert (tmp_path / "run" / "report.json").read_text() == ref_json

    monkeypatch.setenv(ck.ENV_VERSION, "steps-v2")                     # stale (other code version): refit everything
    fits["n"] = 0
    assert rsf.main(argv(tmp_path / "run")) == 0
    assert fits["n"] == full
