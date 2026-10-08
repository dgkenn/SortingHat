"""scripts/diag_cohort.py: aggregates only, correct on known synthetic counts."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from sortinghat.cohort.diag import run_diag
from sortinghat.data_io import LocalStore
from sortinghat.safe_output import SUPPRESSED, assert_aggregate_only

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def diag(synth_dir, tmp_path_factory):
    root = tmp_path_factory.mktemp("diag_store") / "d"
    shutil.copytree(synth_dir, root)
    (root / "PatientMergeHistory").mkdir()
    pd.DataFrame({"RetiredPatientID": [1], "SurvivingPatientID": [2]}).to_csv(root / "PatientMergeHistory" / "a.csv", index=False)
    pd.DataFrame({"x": [1]}).to_csv(root / "PatientMergeHistory" / "b.csv", index=False)
    report, known = run_diag(LocalStore(root))
    return root, report, known


def test_counts_match_the_tables(diag, synth):
    _, rep, _ = diag
    eeg = synth[0]["eeg_metadata"]
    rf = synth[0]["reports_findings"]
    for site in ("S0001", "S0002", "I0003"):
        b = rep["sites"][site]
        a = b["n_adult_sessions_with_start"]
        assert isinstance(a, int) and a > 100
        assert b["share_matched_by_current_rule"]["of"] == a
        assert b["share_covered_of_all_adult_eegs"]["closed"]["of"] == a
        # coverage can only grow when the interval is widened or a null end is given a horizon
        c = b["share_covered_of_all_adult_eegs"]
        assert c["widened_24h"]["n"] >= c["closed"]["n"] and c["null_end_30d"]["n"] >= c["closed"]["n"]
        assert c["widened_and_null_end_30d"]["n"] >= max(c["widened_24h"]["n"], c["null_end_30d"]["n"])
    s1 = rep["sites"]["S0001"]
    assert s1["id_join"]["share_with_visit_via_bdsp_id_number"]["n"] in (SUPPRESSED, 0) or \
        s1["id_join"]["share_with_visit_via_folder_tail"]["n"] > 100        # S-sites: BDSPPatientID blank, BidsFolder works
    assert s1["id_join"]["share_bdsp_id_blank"]["of"] == int((eeg["SiteID"] == "S0001").sum())
    assert s1["hours_eeg_minus_nearest_visit_start"]["q50"] > 0             # synthetic visits begin before the EEG


def test_value_counts_and_visit_table(diag):
    _, rep, _ = diag
    ids = {d["value"]: d["n"] for d in rep["visit_concept_id"]["values"]}
    assert {"9201", "32037", "9202", "9203"} <= set(ids) and all(n >= 11 for n in ids.values())
    assert len(rep["visit_source_value"]["values"]) <= 20
    assert rep["visit_table"]["pid_arrow_type"] in ("int64", "int32")


def test_merge_history_reports_column_names_only(diag):
    _, rep, _ = diag
    m = rep["merge_history_table"]
    assert m["n_files"] == SUPPRESSED
    assert {tuple(l["column_names"]) for l in m["layouts"]} == {("RetiredPatientID", "SurvivingPatientID"), ("x",)}


def test_duration_by_service_distinguishes_ltm_from_routine(diag):
    _, rep, _ = diag
    d = rep["duration_by_service"]["S0001"]
    assert set(d) == {"LTM", "Routine"} and d["LTM"]["clock_hours"]["q50"] > d["Routine"]["clock_hours"]["q50"]
    assert d["LTM"]["ratio_meta_over_clock"]["q50"] == pytest.approx(1.0, abs=0.01)


def test_report_is_aggregate_only_and_carries_no_identifier(diag):
    _, rep, known = diag
    assert_aggregate_only(rep, known)
    blob = json.dumps(rep)
    assert not any(i in blob for i in list(known)[:2000])


def test_cli_prints_aggregates_only(synth_dir, synth, tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ("CLAUDECODE", "SORTINGHAT_AGENT_SESSION")}
    out = tmp_path / "d.json"
    p = subprocess.run([sys.executable, str(REPO / "scripts" / "diag_cohort.py"), "--data", str(synth_dir),
                        "--out", str(out)], capture_output=True, text=True, check=True, env=env, cwd=REPO)
    ids = {str(i) for i in synth[0]["omop_person"]["person_id"]} | set(synth[0]["eeg_metadata"]["SessionID"].astype(str))
    assert not any(i in p.stdout or i in p.stderr for i in ids)
    assert json.loads(p.stdout) == json.loads(out.read_text())
    assert "sub-" not in p.stdout and "ses-" not in p.stdout
