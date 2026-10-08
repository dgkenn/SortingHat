"""run_silver_feasibility.py with the --embeddings and --morgoth rungs and the "Commercial-clean gap" section, on SYNTHETIC inputs
only: fake CBraMod-embedding and MORGOTH-findings parquet parts and a fake exposure flag file (every identifier is fake)."""
import contextlib
import io
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sortinghat.safe_output import assert_aggregate_only
from test_silver_feasibility_inputs import argv_for, make_inputs, rsf  # noqa: F401  (tests/ on sys.path)

BAD_TOKENS = re.compile(r"(\b71\d{6}\b|\brec[0-9a-f]{20}\b|sub-|ses-|\d{4}-\d{2}-\d{2}|local_only|\.edf)")
GAP_TITLE = "Commercial-clean gap"
IU_TITLE = "Intended use: undifferentiated AMS"


def _rids(cohort: pd.DataFrame) -> list[str]:
    return [rsf.opaque_recording_id(rsf.data_io.edf_key_for_row(s, b, sid, e if isinstance(e, str) else None))
            for s, b, sid, e in zip(cohort["SiteID"], cohort["BidsFolder"], cohort["SessionID"], cohort["EEGFolder"])]


def write_fake_parts(lo: Path, ms, cohort: pd.DataFrame, seed: int = 5) -> None:
    """Fake extractor parts. Embedding: pure noise (no planted signal). MORGOTH findings: the planted-label signal. Both carry
    QC failures, a nested-window junk row and a duplicate that must never be used."""
    rng = np.random.default_rng(seed)
    n = len(cohort)
    rid = _rids(cohort)
    (lo / "embeddings").mkdir(exist_ok=True)
    (lo / "morgoth").mkdir(exist_ok=True)
    emb_ok = rng.random(n) > 0.05
    emb = pd.DataFrame(rng.normal(0, 1, (n, 6)).astype("float32"), columns=[f"emb.cbramod.{j}" for j in range(6)])
    base = pd.DataFrame({"recording_id": rid, "window": "primary", "qc_pass": True, "emb_ok": emb_ok,
                         "onset_offset_s": 0.0})
    junk = pd.concat([base.iloc[:40].assign(window="20s"), emb.iloc[:40] * 0 + 99.0], axis=1)
    pd.concat([pd.concat([base, emb], axis=1), junk]).to_parquet(lo / "embeddings" / "part-s0-0000000000.parquet", index=False)

    qc = rng.random(n) > 0.05
    cols = {}
    for k in ("E1", "E2", "E5"):
        j = ms.label_names.index(k)
        yc = ms.y_gold[:, j] - ms.y_gold[:, j].mean()
        cols[f"morgoth.finding.{k}.mean"] = 1.2 * yc + rng.normal(0, 1, n)
        cols[f"morgoth.finding.{k}.max"] = 0.8 * yc + rng.normal(0, 1, n)
    mg = pd.DataFrame(cols)
    mbase = pd.DataFrame({"recording_id": rid, "window": "primary", "qc_pass": qc, "onset_offset_s": 0.0})
    mjunk = pd.concat([mbase.iloc[:40].assign(window="20s"), mg.iloc[:40] * 0 + 99.0], axis=1)
    both = pd.concat([pd.concat([mbase, mg], axis=1), mjunk])
    for k, ix in enumerate(np.array_split(np.arange(len(both)), 2)):
        both.iloc[ix].to_parquet(lo / "morgoth" / f"part-s{k}of2-{k:010d}.parquet", index=False)
    dup = pd.concat([mbase, mg], axis=1).iloc[:30]                                         # duplicate rows across parts
    dup.to_parquet(lo / "morgoth" / "part-dup-0000000009.parquet", index=False)


def write_exposure(path: Path, cohort: pd.DataFrame, frac: float, with_split: bool, seed: int = 6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    flag = rng.random(len(cohort)) < frac
    df = pd.DataFrame({"person_id": cohort["person_id"].to_numpy(), "SiteID": cohort["SiteID"].to_numpy(),
                       "in_morgoth_lists": flag, "in_morgoth_train_split": np.nan, "n_morgoth_lists": flag.astype(int)})
    if with_split:
        df["in_morgoth_train_split"] = flag & (rng.random(len(cohort)) < 0.6)
    df.to_parquet(path, index=False)
    return df["in_morgoth_train_split"].to_numpy() if with_split else flag


def run(paths, out, *extra):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = rsf.main(argv_for(paths, out, "--skip-controls", "--splits", "loso", "temporal", *extra))
    assert rc == 0
    out = Path(out)
    return json.loads((out / "report.json").read_text()), (out / "report.md").read_text(), buf.getvalue()


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("mg")
    paths, ms, cohort = make_inputs(d, signal=1.0, seed=11, n_per_site=300)
    lo = d / "local_only"
    write_fake_parts(lo, ms, cohort)
    none = ["--embeddings", str(lo / "no_emb"), "--morgoth", str(lo / "no_mg")]
    r = SimpleNamespace(d=d, lo=lo, paths=paths, ms=ms, cohort=cohort)
    r.none = SimpleNamespace(zip=run(paths, d / "out_none", *none))
    # default locations (<features>/../embeddings and ../morgoth) + no exposure file there
    r.noexp = SimpleNamespace(zip=run(paths, d / "out_noexp"))
    flags = write_exposure(lo / "morgoth" / "exposure.parquet", cohort, 0.3, with_split=False)
    r.expo_lists = SimpleNamespace(zip=run(paths, d / "out_expo"), flags=flags)
    flags = write_exposure(lo / "morgoth" / "exposure.parquet", cohort, 0.5, with_split=True)
    r.expo_split = SimpleNamespace(zip=run(paths, d / "out_split"), flags=flags)
    return r


def J(x):
    return x.zip[0]


def MD(x):
    return x.zip[1]


def test_no_parts_means_both_rungs_skipped_and_no_gap_section(world):
    j, md, _ = world.none.zip
    assert set(j["settings"]["skipped_rungs"]) == {"morgoth", "embeddings (CBraMod)", "dynamics"}
    assert set(j["settings"]["rungs_skipped"]) == {"dynamics", "cbramod_frozen", "combined_cbramod", "morgoth_findings",
                                                   "combined_morgoth"}
    assert j["settings"]["rungs_ran"] == ["prior", "qeeg", "connectivity", "combined"]
    assert j["commercial_clean_gap"]["available"] is False and GAP_TITLE not in md
    assert "skipped" in j["morgoth_findings"] and "skipped" in j["frozen_cbramod"]
    for lad in j["ladder"].values():
        assert "morgoth_findings" not in lad["rungs"]["A"] or not lad["rungs"]["A"]["morgoth_findings"].get("available")


def test_rungs_run_and_are_recorded(world):
    j, md, _ = world.noexp.zip
    s = j["settings"]
    assert s["rungs_ran"] == ["prior", "qeeg", "connectivity", "combined", "cbramod_frozen", "combined_cbramod",
                              "morgoth_findings", "combined_morgoth"]
    assert s["rungs_skipped"] == ["dynamics"] and s["skipped_rungs"] == ["dynamics"]
    assert j["headline_rung"] == "combined"                                   # the headline rung is unchanged
    assert j["morgoth_findings"]["dim"] == 6 and j["frozen_cbramod"]["dim"] == 6
    for split, lad in j["ladder"].items():
        for b, rungs in lad["rungs"].items():
            for nm in ("cbramod_frozen", "combined_cbramod", "morgoth_findings", "combined_morgoth"):
                assert rungs[nm]["available"], (split, b, nm)
    assert "Rungs that ran:" in md and "morgoth_findings" in md and "combined_morgoth" in md
    assert "| morgoth_findings |" in md and "| combined_morgoth |" in md


def test_morgoth_rung_recovers_its_planted_signal_and_noise_embedding_does_not(world):
    j = J(world.noexp)
    for split in ("loso", "temporal"):
        r = j["ladder"][split]["rungs"]["A"]
        assert r["morgoth_findings"]["delta"] < -0.005, split
        assert r["morgoth_findings"]["delta"] < r["cbramod_frozen"]["delta"]


def test_gap_section_comes_after_the_intended_use_block(world):
    md = MD(world.noexp)
    assert md.index(f"## {IU_TITLE}") < md.index(f"## {GAP_TITLE}")
    assert md.index("## Headline Delta") < md.index(f"## {GAP_TITLE}") < md.index("## Per-label Delta")
    assert md.index(f"## {IU_TITLE}") < md.index("## Data flow")
    assert list(J(world.noexp)).index("intended_use") < list(J(world.noexp)).index("commercial_clean_gap")


def test_gap_is_the_difference_of_the_two_deltas_with_ordered_intervals(world):
    g = J(world.noexp)["commercial_clean_gap"]
    assert g["available"] and g["rungs"] == ["cbramod_frozen", "morgoth_findings"]
    assert set(g["by_scheme"]) == {"loso", "temporal"}
    for split, per in g["by_scheme"].items():
        assert set(per) == {"A", "C"}
        for b, blk in per.items():
            lad = J(world.noexp)["ladder"][split]["rungs"][b]
            p = blk["pooled"]
            assert p["delta_a"] == pytest.approx(lad["cbramod_frozen"]["delta"], abs=5e-5)
            assert p["delta_b"] == pytest.approx(lad["morgoth_findings"]["delta"], abs=5e-5)
            assert p["gap"] == pytest.approx(p["delta_a"] - p["delta_b"], abs=5e-5)
            assert p["gap"] > 0                                                   # noise embedding helps less than planted findings
            c99, c95 = p["ci_99_within_site"], p["ci_95_within_site"]
            assert c99["lo"] <= p["gap"] <= c99["hi"] and c95["lo"] <= p["gap"] <= c95["hi"]
            assert c99["hi"] - c99["lo"] > c95["hi"] - c95["lo"]                  # 99% wider than 95%: same levels as the rest
            assert set(blk["per_site"]) == {"site_1", "site_2"}
            for s_, v in blk["per_site"].items():
                assert v["gap"] == pytest.approx(v["delta_a"] - v["delta_b"], abs=5e-5)
                assert v["ci_99_within_site"]["lo"] <= v["gap"] <= v["ci_99_within_site"]["hi"]
            # the pooled gap is the n-weighted mean of the per-site gaps (equal weights per row)
            ns = [v["n"] for v in blk["per_site"].values()]
            assert all(isinstance(x, int) and x >= 11 for x in ns)
            assert sum(v["gap"] * v["n"] for v in blk["per_site"].values()) / sum(ns) == pytest.approx(p["gap"], abs=5e-5)
    md = MD(world.noexp)
    assert "| all sites |" in md and "| site_1 |" in md and "| site_2 |" in md


def test_gap_uses_the_patient_level_bootstrap_of_the_ladder_cfg(world):
    """Same bootstrap function and levels as the ladder's own Delta intervals: a re-run on identical d_i gives identical bounds."""
    a = np.random.default_rng(0).normal(0.0, 1.0, 400)
    b = np.random.default_rng(1).normal(0.1, 1.0, 400)
    sites = np.array(["x"] * 200 + ["y"] * 200)
    s1 = rsf._gap_stats(a, b, sites, 300, 0, 0.01, 0.05)
    s2 = rsf._gap_stats(a, b, sites, 300, 0, 0.01, 0.05)
    assert s1 == s2 and s1["gap"] == pytest.approx(float(np.mean(a - b)))
    small = rsf._gap_stats(a[:5], b[:5], sites[:5], 300, 0, 0.01, 0.05)
    assert small["gap"] == rsf.SUPPRESSED and small["n"] == rsf.SUPPRESSED


def test_without_an_exposure_file_the_sensitivity_row_is_reported_as_not_run(world):
    g = J(world.noexp)["commercial_clean_gap"]
    for per in g["by_scheme"].values():
        for blk in per.values():
            assert "not_run" in blk["exposure_sensitivity"]
    assert "no MORGOTH exposure flag file found" in MD(world.noexp)
    assert J(world.noexp)["settings"]["morgoth_exposure_flags"].startswith("none")


def test_exposure_file_adds_a_sensitivity_row_that_drops_flagged_patients(world):
    x = world.expo_lists
    j, md, _ = x.zip
    assert "any MORGOTH data list" in j["settings"]["morgoth_exposure_flags"]
    base = J(world.noexp)["commercial_clean_gap"]["by_scheme"]
    for split, per in j["commercial_clean_gap"]["by_scheme"].items():
        for b, blk in per.items():
            ex = blk["exposure_sensitivity"]
            assert "any MORGOTH data list" in ex["basis"]
            n_all, n_exc = blk["pooled"]["n"], ex["n_excluded"]
            assert isinstance(n_exc, int) and 11 <= n_exc < n_all
            assert ex["n_kept"] == n_all - n_exc
            assert ex["pooled"]["n"] == n_all - n_exc
            assert blk["pooled"] == base[split][b]["pooled"]                      # the main rows do not depend on the flag file
            p = ex["pooled"]
            assert p["gap"] == pytest.approx(p["delta_a"] - p["delta_b"], abs=5e-5) and p["gap"] != blk["pooled"]["gap"]
            assert p["ci_99_within_site"]["lo"] <= p["gap"] <= p["ci_99_within_site"]["hi"]
            assert set(ex["per_site"]) == {"site_1", "site_2"}
    assert "excluding MORGOTH-exposed patients" in md and "flag basis: any MORGOTH data list" in md


def test_training_split_flag_is_preferred_when_the_lists_carry_a_split_column(world):
    j, md, _ = world.expo_split.zip
    assert "MORGOTH training split" in j["settings"]["morgoth_exposure_flags"]
    lists = J(world.expo_lists)["commercial_clean_gap"]["by_scheme"]["loso"]["A"]["exposure_sensitivity"]["n_excluded"]
    split = j["commercial_clean_gap"]["by_scheme"]["loso"]["A"]["exposure_sensitivity"]
    assert split["basis"] == "MORGOTH training split" and isinstance(split["n_excluded"], int)
    assert md.count("flag basis: MORGOTH training split") == 4                      # 2 schemes x 2 baselines
    assert isinstance(lists, int)


def test_exposure_loader_semantics(tmp_path):
    lo = tmp_path / "local_only"
    lo.mkdir()
    pd.DataFrame({"person_id": [1, 2, 3, 3], "in_morgoth_lists": [True, False, False, True],
                  "in_morgoth_train_split": [np.nan] * 4}).to_parquet(lo / "e.parquet")
    f, basis = rsf.load_morgoth_exposure(lo / "e.parquet")
    assert f.to_dict() == {1: 1.0, 2: 0.0, 3: 1.0} and "list" in basis
    pd.DataFrame({"person_id": [1, 2], "in_morgoth_lists": [True, True],
                  "in_morgoth_train_split": [False, True]}).to_parquet(lo / "s.parquet")
    f, basis = rsf.load_morgoth_exposure(lo / "s.parquet")
    assert f.to_dict() == {1: 0.0, 2: 1.0} and "training split" in basis
    assert rsf.load_morgoth_exposure(lo / "missing.parquet") is None
    with pytest.raises(SystemExit):
        rsf.load_morgoth_exposure(tmp_path / "e.parquet")                           # not under local_only/


def test_morgoth_loader_filters_window_and_qc_and_requires_local_only(world, tmp_path):
    c = rsf.bb.load_cohort(world.paths["cohort"], None, "table")
    rids = set(rsf.recording_ids(c))
    df, st = rsf.load_morgoth_primary(world.lo / "morgoth", rids)
    assert df.index.is_unique and set(df.index) <= rids and all(str(c).startswith("morgoth.") for c in df.columns)
    assert (df.abs() < 50).all().all()                                              # the 20s-window junk (99.0) never appears
    assert st["n_recordings"] == len(df) and 0 < len(df) < len(rids)               # QC failures dropped
    empty, st0 = rsf.load_morgoth_primary(world.lo / "nope", rids)
    assert empty.empty and st0["n_parts"] == 0
    (tmp_path / "mg").mkdir()
    with pytest.raises(SystemExit):
        rsf.load_morgoth_primary(tmp_path / "mg", rids)


def test_only_one_rung_family_means_no_gap(world, tmp_path):
    j, md, _ = run(world.paths, tmp_path / "o", "--embeddings", str(world.lo / "no_emb"))
    assert j["settings"]["rungs_ran"][-2:] == ["morgoth_findings", "combined_morgoth"]
    assert "cbramod_frozen" in j["settings"]["rungs_skipped"]
    assert j["commercial_clean_gap"]["available"] is False and GAP_TITLE not in md
    assert j["settings"]["skipped_rungs"] == ["dynamics", "embeddings (CBraMod)"]


@pytest.mark.parametrize("name", ["noexp", "expo_lists", "expo_split"])
def test_outputs_stay_aggregate_only(world, name):
    x = getattr(world, name)
    j, md, stdout = x.zip
    ids = {str(p) for p in world.cohort["person_id"]} | set(rsf.recording_ids(world.cohort))
    for text in (md, stdout, json.dumps(j)):
        assert not BAD_TOKENS.search(text), BAD_TOKENS.search(text)
        assert_aggregate_only(text, ids)
    assert_aggregate_only(j, ids)
    out = world.d / {"noexp": "out_noexp", "expo_lists": "out_expo", "expo_split": "out_split"}[name]
    assert sorted(p.name for p in out.iterdir()) == ["report.json", "report.md"]
    not_people = {"n_boot", "n_valid", "n_reps", "n_strata", "n_favorable", "n_eeg_features", "n_features_in", "n_features_out",
                  "n_reps_with_spurious_gain_95", "strata_with_delta_below_zero"}

    def counts(o, key=""):
        if isinstance(o, dict):
            for k, v in o.items():
                yield from counts(v, k)
        elif isinstance(o, int) and not isinstance(o, bool) and re.match(r"^n($|_)", key) and key not in not_people:
            yield key, o
    for k, v in counts(j):
        assert v >= 11, (k, v)
