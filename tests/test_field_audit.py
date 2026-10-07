import json
import re
import subprocess
import sys

import pandas as pd
import pytest

from sortinghat.audit import field_audit
from sortinghat.audit.field_audit import build_candidates, run_audit
from sortinghat.tableio import load_tables


@pytest.fixture(scope="module")
def audit_run(synth_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("audit_out")
    proc = subprocess.run(
        [sys.executable, "-m", "sortinghat.audit.field_audit", "--data", str(synth_dir), "--out", str(out)],
        capture_output=True, text=True, check=True)
    return out, proc


def test_expected_pass_fail(synth):
    report, ids = run_audit(synth[0])
    status = {r["field"]: r["passed"] for r in report["rows"]}
    assert len(status) == 8
    assert status["EEG start date and time of day"]
    assert status["Consistent within-patient date shift"]
    assert status["Timestamped notes"]
    assert status["Site identifier"]
    assert not status["Medication administration times (not just orders)"]   # injected MAR gaps
    assert not status["Imaging report finalization time"]                     # injected gaps
    assert report["gate_0a_automated_stop_rows_pass"] is True
    assert set(report["fallbacks_triggered"]) == {
        "Medication administration times (not just orders)", "Imaging report finalization time"}
    assert len(ids) == 20 and len(set(ids)) == 20


def test_date_shift_check_detects_injection(synth):
    tables, truth = synth
    report, _ = run_audit(tables)
    ds = next(r for r in report["rows"] if r["field"].startswith("Consistent"))
    cands = set(build_candidates(tables["eeg_metadata"])["BDSPPatientID"])
    injected = len(cands & set(truth["table_shift_ids"]))
    assert injected > 30
    rate = ds["observed"]["misaligned_proportion"]
    assert rate != "<11"
    assert rate == pytest.approx(injected / ds["observed"]["evaluated_n"], rel=0.25)


def test_monotonicity_violation_detected(synth):
    tables = {k: v.copy() for k, v in synth[0].items()}
    labs = tables["labs"]
    idx = labs.index[:400]
    labs.loc[idx, "result_time"] = labs.loc[idx, "collect_time"] - pd.Timedelta(hours=1)
    report, _ = run_audit(tables)
    ds = next(r for r in report["rows"] if r["field"].startswith("Consistent"))
    assert ds["observed"]["monotonicity"]["lab_collect_before_result"]["n_violations"] != "<11"
    assert not ds["passed"]


def test_small_site_is_suppressed(synth):
    tables = {k: v.copy() for k, v in synth[0].items()}
    eeg = tables["eeg_metadata"]
    keep = eeg[eeg["SiteID"] != "I0002"]
    # keep only 5 patients of I0002 so their cell must be suppressed
    few = eeg[eeg["SiteID"] == "I0002"]
    few_ids = few["BDSPPatientID"].drop_duplicates().head(5)
    tables["eeg_metadata"] = pd.concat([keep, few[few["BDSPPatientID"].isin(few_ids)]])
    tables["eeg_metadata"].loc[tables["eeg_metadata"]["SiteID"] == "I0002", ["AgeAtVisit", "PatientClass"]] = [60.0, "ICU"]
    report, _ = run_audit(tables)
    site_row = next(r for r in report["rows"] if r["field"] == "Site identifier")
    assert site_row["observed"]["candidates_per_site"]["I0002"] == "<11"
    start_row = report["rows"][0]
    assert start_row["observed"]["by_site"]["I0002"]["n_total"] == "<11"
    assert start_row["observed"]["by_site"]["I0002"]["proportion"] == "<11"


def test_outputs_contain_no_patient_ids(synth, audit_run):
    out, proc = audit_run
    all_ids = set(synth[0]["eeg_metadata"]["BDSPPatientID"])
    id_re = re.compile(r"SYN[SI]\d{10}|sub-|ses-")
    texts = [proc.stdout, proc.stderr, (out / "field_audit.md").read_text(),
             (out / "field_audit.json").read_text()]
    for t in texts:
        assert not id_re.search(t)
        assert not (set(re.findall(r"[A-Za-z0-9_\-]{6,}", t)) & all_ids)
    json.loads((out / "field_audit.json").read_text())


def test_handcheck_list_local_only(synth, audit_run):
    out, proc = audit_run
    f = out / "local_only" / "handcheck_sample_ids.csv"
    lines = f.read_text().split()
    assert lines[0] == "BDSPPatientID" and len(lines) == 21
    all_ids = set(synth[0]["eeg_metadata"]["BDSPPatientID"])
    assert set(lines[1:]) <= all_ids
    assert (f.stat().st_mode & 0o077) == 0
    assert "local file" in proc.stdout
    # IDs from the list appear nowhere outside that file
    for p in out.rglob("*"):
        if p.is_file() and p != f:
            assert not (set(lines[1:]) & set(re.findall(r"[A-Za-z0-9_\-]{6,}", p.read_text())))


def test_cli_refuses_restricted_path_in_agent_session(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("SORTINGHAT_RESTRICTED_ROOT", str(tmp_path / "secret"))
    (tmp_path / "secret").mkdir()
    from sortinghat.agent_safety import RestrictedDataError
    with pytest.raises(RestrictedDataError):
        field_audit.main(["--data", str(tmp_path / "secret"), "--out", str(tmp_path / "o")])
