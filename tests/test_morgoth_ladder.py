"""The morgoth_findings rung: loader returns None without parquet (rung skipped), and a fitted rung when present."""
import numpy as np
import pandas as pd
import pytest

from sortinghat.models import (MORGOTH_FINDINGS_RUNG, DEFAULT_RUNGS, LadderConfig, assemble_eeg_frame, eeg_frame_from_pipeline_rows,
                               load_morgoth_findings, run_ladder)
from sortinghat.models.ladder import rung_columns
from sortinghat.models.synthetic import make_synthetic_study
from sortinghat.morgoth.heads import feature_names


def write_parts(d, ids, rng, window="primary", qc=True):
    d.mkdir(parents=True, exist_ok=True)
    cols = feature_names(("normal", "slowing"))
    df = pd.DataFrame({"recording_id": ids, "window": window, "qc_pass": qc})
    for c in cols:
        df[c] = rng.uniform(0, 1, len(ids))
    df.to_parquet(d / "part-a-1.parquet", index=False)
    df.iloc[:2].to_parquet(d / "part-a-2.parquet", index=False)          # duplicated recordings (crash between part and ledger)
    return cols


def test_absent_parquet_gives_none_and_rung_unavailable(tmp_path):
    assert load_morgoth_findings(tmp_path / "nothing", ["a", "b"]) is None
    assert load_morgoth_findings(tmp_path, ["a"]) is None                   # empty directory
    assert rung_columns(["qeeg.x", "conn.y"], MORGOTH_FINDINGS_RUNG) == []
    base = pd.DataFrame({"qeeg.a": [1.0, 2.0]})
    assert list(assemble_eeg_frame(base, load_morgoth_findings(tmp_path, ["a", "b"])).columns) == ["qeeg.a"]


def test_loader_aligns_to_ids_dedups_and_prefixes(tmp_path):
    rng = np.random.default_rng(0)
    ids = [f"rec{i}" for i in range(5)]
    cols = write_parts(tmp_path / "mg", ids, rng)
    out = load_morgoth_findings(tmp_path / "mg", ["rec3", "rec0", "missing", "rec4"])
    assert list(out.columns) == cols and len(out) == 4 and all(c.startswith("morgoth.") for c in out.columns)
    assert out.iloc[2].isna().all() and not out.iloc[0].isna().any()
    df = pd.read_parquet(tmp_path / "mg" / "part-a-1.parquet").set_index("recording_id")
    assert out.iloc[0]["morgoth.normal.abnormal.mean"] == pytest.approx(df.loc["rec3", "morgoth.normal.abnormal.mean"])


def test_qc_failed_rows_are_dropped_by_default(tmp_path):
    rng = np.random.default_rng(1)
    write_parts(tmp_path / "mg", ["a", "b"], rng, qc=False)
    assert load_morgoth_findings(tmp_path / "mg", ["a", "b"]) is None
    assert load_morgoth_findings(tmp_path / "mg", ["a", "b"], require_qc_pass=False) is not None


def test_rung_runs_in_ladder_when_columns_exist_and_is_skipped_otherwise():
    cfg = LadderConfig(n_boot=50, per_label=False)
    rungs = (DEFAULT_RUNGS[0], MORGOTH_FINDINGS_RUNG)
    d = make_synthetic_study(n_per_site=250, seed=1, morgoth=True, embeddings=False)
    with_m = run_ladder(d, rungs=rungs, cfg=cfg)
    r = with_m.get("morgoth_findings")
    assert r.available and r.n_eeg_features > 0 and np.isfinite(r.delta)
    d2 = make_synthetic_study(n_per_site=250, seed=1, morgoth=False, embeddings=False)
    without = run_ladder(d2, rungs=rungs, cfg=cfg)
    r = without.get("morgoth_findings")
    assert not r.available and r.n_eeg_features == 0
