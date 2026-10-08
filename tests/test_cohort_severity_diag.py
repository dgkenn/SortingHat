"""--severity: per-site score availability, vocabulary, phenotype, onset and flow aggregates for the candidate set."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sortinghat.cohort import CohortConfig, FrameSources, build_cohort
from sortinghat.cohort.severity_diag import run_severity_diag
from sortinghat.data_io import LocalStore
from sortinghat.safe_output import SUPPRESSED, assert_aggregate_only

REPO = Path(__file__).resolve().parents[1]
SCORE_NAME = r"Glasgow|FOUR Score|RASS|Eye Opening"


@pytest.fixture(scope="module")
def sev(synth_dir):
    return run_severity_diag(LocalStore(synth_dir))


def test_shares_match_an_independent_recomputation(sev, synth):
    rep, _ = sev
    res = build_cohort(FrameSources(synth[0]), CohortConfig())
    cand = res.stages["first_eeg"]
    m = synth[0]["omop_measurement"]
    m = m[m["measurement_source_value"].astype(str).str.contains(SCORE_NAME)]
    for site in ("S0001", "S0002", "I0003"):
        c = cand[cand["SiteID"] == site]
        b = rep["sites"][site]
        assert b["n_candidates"] == len(c)
        mm = m[m["person_id"].isin(c["person_id"])].merge(c[["person_id", "t0"]], on="person_id")
        dh = (mm["measurement_datetime"] - mm["t0"]).dt.total_seconds() / 3600
        exp = {"any_gcs_four_rass_row": mm["person_id"].nunique(),
               "within_pm6h": mm.loc[dh.abs() <= 6, "person_id"].nunique(),
               "within_minus6h_plus1h": mm.loc[(dh >= -6) & (dh <= 1), "person_id"].nunique()}
        for k, v in exp.items():
            assert b["measurement"][k]["n"] == v, (site, k)
            assert b["either_table"][k]["n"] == v                              # synthetic observations hold no scores
            assert b["observation"][k]["n"] == SUPPRESSED or b["observation"][k]["n"] == 0
        assert b["measurement"]["within_minus6h_plus1h"]["n"] <= b["measurement"]["within_pm6h"]["n"] <= \
            b["measurement"]["any_gcs_four_rass_row"]["n"]


def test_vocabulary_unmatched_keywords_onset_and_flow(sev):
    rep, known = sev
    b = rep["sites"]["S0001"]
    top = {d["value"]: d["lexicon_key"] for d in b["score_vocabulary_matched_top20"]}
    assert top["Glasgow Coma Scale Score"] == "gcs" and top["FOUR Score"] == "four" and len(top) <= 20
    assert b["unmatched_keyword_source_values_top20"] == []                       # nothing keyword-like is missed here
    assert b["hours_onset_proxy_to_eeg"]["q50"] >= 0 and set(b["onset_proxy_basis_counts"]) <= {"abnormal_score", "visit_start", "none"}
    rows = b["flow_steps"]
    assert rows[0]["step"].startswith("EEG sessions") and any("first qualifying" in r["step"] or "first" in r["step"].lower() for r in rows)
    assert not any(" + " in r["step"] for r in rows)                              # unmerged
    assert b["condition"]["any_condition_row"]["of"] == b["n_candidates"]
    assert_aggregate_only(rep, known)


def test_unmatched_keyword_values_are_reported(synth_dir, tmp_path):
    import shutil
    import pyarrow as pa
    import pyarrow.parquet as pq
    d = tmp_path / "d"
    shutil.copytree(synth_dir, d)
    part = sorted((d / "OMOP/Merged/measurement").glob("*.parquet"))[0]
    t = pq.read_table(part).to_pandas()
    sel = t["measurement_source_value"].astype(str).str.contains("Glasgow")
    t.loc[sel, "measurement_source_value"] = "GCS-total odd label 12345"          # lexicon matches "gcs" -> matched, so rename
    t.loc[sel, "measurement_source_value"] = "Coma depth assessment 9"             # keyword 'coma', not in the lexicon
    pq.write_table(pa.Table.from_pandas(t, preserve_index=False), part)
    rep, _ = run_severity_diag(LocalStore(d))
    vals = [v for s in rep["sites"].values() for v in s["unmatched_keyword_source_values_top20"]]
    assert any(v["value"].startswith("Coma depth assessment") for v in vals)       # digits are masked in listed strings


def test_cli_severity_flag_prints_aggregates_only(synth_dir, synth, tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ("CLAUDECODE", "SORTINGHAT_AGENT_SESSION")}
    p = subprocess.run([sys.executable, str(REPO / "scripts" / "diag_cohort.py"), "--data", str(synth_dir), "--severity"],
                       capture_output=True, text=True, check=True, env=env, cwd=REPO)
    rep = json.loads(p.stdout)
    assert "severity" in rep and "S0001" in rep["severity"]["sites"]
    ids = {str(i) for i in synth[0]["omop_person"]["person_id"]}
    assert not any(i in p.stdout or i in p.stderr for i in ids)
