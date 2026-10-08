"""The cbramod_frozen rung in scripts/run_silver_feasibility.py with a FAKE embedding parquet (no torch, no model): present
-> the rung and combined_cbramod are fitted and the headline rung is untouched; absent -> skipped cleanly."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sortinghat.embed import cbramod as cb
from sortinghat.embed.store import load_primary_embeddings, write_part
from sortinghat.models import CBRAMOD_FROZEN_RUNG, LadderConfig, run_ladder
from sortinghat.models.ladder import rung_columns
from sortinghat.models.synthetic import make_synthetic_study
from test_silver_feasibility_inputs import argv_for, make_inputs, rsf  # noqa: F401  (tests/ is on sys.path)

DIM = cb.EMB_DIM


def fake_rows(rids, X, window="primary", ok=True):
    rows = []
    for rid, x in zip(rids, X):
        row = {"recording_id": rid, "onset_offset_s": 12.0, "window": window, "qc_pass": True, "usable_fraction": 0.9,
               "qc_n_missing_or_dead_min": 0, "qc_coverage_fraction": 1.0, "emb_ok": ok, "emb_n_segments": 60,
               "emb_n_used": 60 if ok else 0, "emb_valid_token_frac": 1.0, "emb_frac_over_amp": 0.0, "emb_n_absent_channels": 0}
        row.update({c: float(v) if ok else float("nan") for c, v in zip(cb.emb_columns(), x)})
        rows.append(row)
    return rows


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    d = tmp_path_factory.mktemp("emb_ladder")
    paths, ms, cohort = make_inputs(d, signal=1.0, seed=3, n_per_site=260)
    keys = [rsf.data_io.edf_key_for_row(s, b, sid, e if isinstance(e, str) else None)
            for s, b, sid, e in zip(cohort["SiteID"], cohort["BidsFolder"], cohort["SessionID"], cohort["EEGFolder"])]
    rids = [rsf.opaque_recording_id(k) for k in keys]
    rng = np.random.default_rng(0)
    Z = ms.eeg.to_numpy(float)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    X = Z @ rng.normal(size=(Z.shape[1], DIM)) / np.sqrt(Z.shape[1]) + rng.normal(0, 1.0, (len(Z), DIM))   # label signal via ms.eeg
    emb_dir = d / "local_only" / "embeddings"
    emb_dir.mkdir()
    cfg = cb.EmbedConfig()
    rows = fake_rows(rids, X)
    junk = fake_rows(rids[:50], X[:50] * 0 + 999.0, window="20s")                  # nested windows must never be read
    failed = fake_rows(rids[50:70], X[50:70], ok=False)                            # emb_ok False duplicates of good recordings
    stray = fake_rows([f"rec{i:020d}" for i in range(30)], X[:30])                 # recordings not in the cohort
    for k, ch in enumerate(np.array_split(np.arange(len(rows)), 3)):
        write_part([rows[i] for i in ch] + (junk + failed + stray if k == 0 else []), cfg, emb_dir, f"s{k}of3")
    return d, paths, ms, rids, emb_dir


def run_report(paths, out, *extra):
    rc = rsf.main(argv_for(paths, out, *extra))
    assert rc == 0
    return json.loads((Path(out) / "report.json").read_text()), (Path(out) / "report.md").read_text()


def test_loader_reads_primary_ok_rows_only_and_dedupes(study):
    d, paths, ms, rids, emb_dir = study
    df, st = load_primary_embeddings(emb_dir, set(rids))
    assert df.index.is_unique and set(df.index) == set(rids) and st["n_emb_ok"] >= len(rids)
    assert df.filter(like="emb.cbramod.").shape[1] == DIM and df.filter(like="emb.cbramod.").abs().to_numpy().max() < 50
    assert (df.filter(like="emb.cbramod.").dtypes == np.float32).all()


def test_loader_requires_local_only(tmp_path):
    (tmp_path / "embeddings").mkdir()
    with pytest.raises(SystemExit):
        load_primary_embeddings(tmp_path / "embeddings")


def test_rung_spec_and_run_ladder_with_embedding_columns():
    ms = make_synthetic_study(n_per_site=200, signal=1.0, seed=1, embeddings=True, morgoth=False)
    assert rung_columns(list(ms.eeg.columns), CBRAMOD_FROZEN_RUNG) and all(c.startswith("emb.cbramod.") for c in
                                                                         rung_columns(list(ms.eeg.columns), CBRAMOD_FROZEN_RUNG))
    res = run_ladder(ms, rungs=[CBRAMOD_FROZEN_RUNG], cfg=LadderConfig(n_boot=100, per_label=False), split="loso")
    assert res.get("cbramod_frozen").available and res.get("cbramod_frozen").delta < 0
    no_emb = ms.replace(eeg=ms.eeg.loc[:, [c for c in ms.eeg.columns if not c.startswith("emb.")]])
    res = run_ladder(no_emb, rungs=[CBRAMOD_FROZEN_RUNG], cfg=LadderConfig(n_boot=100, per_label=False), split="loso")
    assert not res.get("cbramod_frozen").available                                     # skipped, not an error


@pytest.fixture(scope="module")
def with_emb(study, tmp_path_factory):
    d, paths, *_ = study
    return run_report(paths, d / "out_with")


@pytest.fixture(scope="module")
def without_emb(study):
    d, paths, *_ = study
    return run_report(paths, d / "out_without", "--embeddings", str(d / "local_only" / "no_such_dir"))


def test_cbramod_rungs_reported_when_the_parquet_exists(with_emb):
    J, md = with_emb
    for split in ("loso", "temporal"):
        for b in ("A", "C"):
            r = J["ladder"][split]["rungs"][b]
            assert r["cbramod_frozen"]["available"] and r["cbramod_frozen"]["n_eeg_features"] == DIM
            assert r["combined_cbramod"]["available"] and r["combined_cbramod"]["n_eeg_features"] > DIM
            assert r["cbramod_frozen"]["delta"] < 0                                      # the fake embedding carries the planted signal
    assert J["frozen_cbramod"]["dim"] == DIM and "n_analysed_with_embedding" in J["frozen_cbramod"]
    assert "cbramod_frozen" in md and "combined_cbramod" in md and "Frozen CBraMod embedding rung" in md
    assert "leakage_probes_cbramod_frozen" in J["controls"]
    assert "embeddings (CBraMod)" not in J["settings"]["skipped_rungs"]
    assert J["headline_rung"] == "combined"


def test_headline_rung_is_unchanged_by_the_embeddings(with_emb, without_emb):
    Jw, Jo = with_emb[0], without_emb[0]
    assert Jw["n_eeg_features"] == Jo["n_eeg_features"]                                   # counts qeeg./conn. columns only
    for split in ("loso", "temporal"):
        for b in ("A", "C"):
            rw, ro = Jw["ladder"][split]["rungs"][b]["combined"], Jo["ladder"][split]["rungs"][b]["combined"]
            assert rw["n_eeg_features"] == ro["n_eeg_features"]
            assert rw["delta"] == pytest.approx(ro["delta"], abs=1e-9)
    assert Jw["controls"]["leakage_probes"] == Jo["controls"]["leakage_probes"]


def test_skipped_cleanly_when_the_parquet_is_missing(without_emb):
    J, md = without_emb
    for split in J["ladder"]:
        for b, r in J["ladder"][split]["rungs"].items():
            assert "cbramod_frozen" not in r and "combined_cbramod" not in r
    assert J["frozen_cbramod"].get("skipped") and "embeddings (CBraMod)" in J["settings"]["skipped_rungs"]
    assert "leakage_probes_cbramod_frozen" not in J["controls"] and "Frozen CBraMod embedding rung" not in md


def test_embeddings_outside_local_only_are_refused(study, tmp_path):
    d, paths, *_ = study
    (tmp_path / "pub").mkdir()
    with pytest.raises(SystemExit):
        rsf.main(argv_for(paths, tmp_path / "o", "--embeddings", str(tmp_path / "pub")))


def test_report_holds_no_record_level_values(with_emb, study):
    J, md = with_emb
    _d, _p, _ms, rids, _e = study
    txt = json.dumps(J) + md
    assert not any(r in txt for r in rids[:50]) and "recording_id" not in txt
