import json
import re
import shutil
import subprocess
import sys

import pandas as pd
import pytest

from sortinghat import schema
from sortinghat.audit import field_audit
from sortinghat.audit.field_audit import build_candidates, from_raw_tables, run_audit


@pytest.fixture(scope="module")
def audit_run(synth_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("audit_out")
    proc = subprocess.run(
        [sys.executable, "-m", "sortinghat.audit.field_audit", "--data", str(synth_dir), "--out", str(out)],
        capture_output=True, text=True, check=True)
    return out, proc


@pytest.fixture(scope="module")
def analytic(synth):
    return from_raw_tables(synth[0])


def _copy(tables):
    return {k: v.copy() for k, v in tables.items()}


def test_expected_pass_fail(synth):
    report, ids = run_audit(synth[0])                # raw real-layout tables are converted automatically
    status = {r["field"]: r["passed"] for r in report["rows"]}
    assert len(status) == 8
    assert status["EEG start date and time of day"]
    assert status["Consistent within-patient date shift"]
    assert status["Timestamped notes"]
    assert status["Site identifier"]
    assert not status["Medication administration times (not just orders)"]   # injected MAR gaps
    assert not status["Imaging report finalization time"]                     # injected gaps
    assert not status["Lab result or verification time"]                      # real table has no result column
    assert report["gate_0a_automated_stop_rows_pass"] is True
    assert set(report["fallbacks_triggered"]) == {
        "Medication administration times (not just orders)", "Imaging report finalization time",
        "Lab result or verification time"}
    assert len(ids) == 20 and len(set(ids)) == 20


def test_cli_through_data_io_matches_in_memory(synth, audit_run):
    out, _ = audit_run
    cli = json.loads((out / "field_audit.json").read_text())
    mem, _ = run_audit(synth[0])
    assert [r["observed_text"] for r in cli["rows"]] == [r["observed_text"] for r in mem["rows"]]


def test_start_time_comes_from_findings_and_pid_from_bids_folder(synth, analytic):
    eeg = analytic["eeg_metadata"]
    raw = synth[0]["eeg_metadata"]
    md = raw["SiteID"].isin(["I0008", "I0009"])
    assert raw.loc[~md, "StartTime"].isna().all()                              # blank in the real table (S-sites, I0002/3)
    assert raw.loc[md, "StartTime"].notna().mean() > 0.9                       # I0008/I0009: the real start (StartDateTime)
    assert eeg["StartTime"].notna().mean() > 0.9
    assert eeg["person_id"].notna().all()                                      # BDSPPatientID blank for S0001/S0002
    assert set(eeg["person_id"]) == set(synth[0]["omop_person"]["person_id"])


def test_date_shift_check_detects_injection(synth, analytic):
    _, truth = synth
    report, _ = run_audit(analytic)
    ds = next(r for r in report["rows"] if r["field"].startswith("Consistent"))
    cands = set(build_candidates(analytic["eeg_metadata"])["person_id"])
    injected = len(cands & set(truth["table_shift_ids"]))
    assert injected > 30
    rate = ds["observed"]["misaligned_proportion"]
    assert rate != "<11"
    assert rate == pytest.approx(injected / ds["observed"]["evaluated_n"], rel=0.25)


def test_monotonicity_violation_detected(analytic):
    tables = _copy(analytic)
    dx = tables["omop_drug_exposure"]
    idx = dx.index[dx["drug_exposure_end_datetime"].notna()][:400]
    dx.loc[idx, "drug_exposure_end_datetime"] = dx.loc[idx, "drug_exposure_start_datetime"] - pd.Timedelta(hours=1)
    report, _ = run_audit(tables)
    ds = next(r for r in report["rows"] if r["field"].startswith("Consistent"))
    assert ds["observed"]["monotonicity"]["drug_start_before_end"]["n_violations"] != "<11"
    assert not ds["passed"]


def test_lab_result_time_passes_only_with_a_result_column(analytic):
    tables = _copy(analytic)
    m = tables["omop_measurement"]
    m["measurement_result_datetime"] = m["measurement_datetime"] + pd.Timedelta(minutes=30)
    report, _ = run_audit(tables)
    lab = next(r for r in report["rows"] if r["field"].startswith("Lab result"))
    assert lab["passed"]


def test_missing_patient_class_is_flagged_not_silent(analytic):
    tables = _copy(analytic)
    tables["eeg_metadata"] = tables["eeg_metadata"].drop(columns=["PatientClass"])
    report, _ = run_audit(tables)
    r0 = report["rows"][0]
    assert r0["acute_care_filter_applied"] is False and "PatientClass absent" in r0["observed_text"]


def test_small_site_is_suppressed(analytic):
    tables = _copy(analytic)
    eeg = tables["eeg_metadata"]
    keep = eeg[eeg["SiteID"] != "I0002"]
    few = eeg[eeg["SiteID"] == "I0002"]
    few_ids = few["person_id"].drop_duplicates().head(5)
    tables["eeg_metadata"] = pd.concat([keep, few[few["person_id"].isin(few_ids)]])
    sel = tables["eeg_metadata"]["SiteID"] == "I0002"
    tables["eeg_metadata"].loc[sel, "AgeAtVisit"] = 60.0
    tables["eeg_metadata"].loc[sel, "PatientClass"] = "ICU"
    report, _ = run_audit(tables)
    site_row = next(r for r in report["rows"] if r["field"] == "Site identifier")
    assert site_row["observed"]["candidates_per_site"]["I0002"] == "<11"
    start_row = report["rows"][0]
    assert start_row["observed"]["by_site"]["I0002"]["n_total"] == "<11"
    assert start_row["observed"]["by_site"]["I0002"]["proportion"] == "<11"


def test_outputs_contain_no_patient_ids(synth, audit_run):
    out, proc = audit_run
    all_ids = {str(p) for p in synth[0]["omop_person"]["person_id"]}
    id_re = re.compile(r"sub-|ses-")
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
    all_ids = {str(p) for p in synth[0]["omop_person"]["person_id"]}
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
    with pytest.raises(RestrictedDataError):                     # --s3 needs a plain terminal too
        field_audit.main(["--s3", "--dry-run-schema"])


# ------------------------------------------------------------ --dry-run-schema
def test_dry_run_schema_all_present(synth_dir, capsys):
    assert field_audit.main(["--data", str(synth_dir), "--dry-run-schema", "--strict"]) == 0
    out = capsys.readouterr().out
    assert "Missing CONFIRMED columns" in out and ": 0" in out


@pytest.fixture()
def broken_dir(synth_dir, tmp_path):
    d = tmp_path / "broken"
    shutil.copytree(synth_dir, d)
    meta = next((d / "EEG/eeg-metadata").glob("S0001_*.csv"))
    txt = meta.read_text()
    head, rest = txt.split("\n", 1)
    head = head.replace("DurationInSeconds", "DurationInSecond").replace(",PatientClass", "")
    # rows still have the old width; the dry run reads the header line only
    meta.write_text(head + "\n" + rest)
    rf = d / "EEG/HEEDB_Metadata/S0001_EEG__reports_findings.csv"
    t = rf.read_text().replace("StartTime(EEG)", "StartTime (EEG)", 1)
    rf.write_text(t)
    shutil.rmtree(d / "Imaging")
    shutil.rmtree(d / "OMOP/Merged/note_nlp")
    return d


def test_dry_run_schema_reports_missing_names_only(broken_dir, synth, tmp_path, capsys):
    out_dir = tmp_path / "dry"
    rc = field_audit.main(["--data", str(broken_dir), "--dry-run-schema", "--out", str(out_dir), "--strict"])
    text = capsys.readouterr().out
    assert rc == 1                                                      # CONFIRMED column missing / table not found
    rep = json.loads((out_dir / "schema_dry_run.json").read_text())
    units = {(u["table"], u["site"]): u for u in rep["units"]}
    miss = {m["column"]: m for m in units[("eeg_metadata", "S0001")]["missing"]}
    assert miss["DurationInSeconds"]["closest_actual_name"] == "DurationInSecond"
    assert miss["DurationInSeconds"]["provenance"] == "CONFIRMED"
    assert miss["PatientClass"]["provenance"] == "ASSUMED"
    assert [m["column"] for m in units[("reports_findings", "S0001")]["missing"]] == ["StartTime(EEG)"]
    assert units[("reports_findings", "S0001")]["missing"][0]["closest_actual_name"] == "StartTime (EEG)"
    assert not units[("imaging", None)]["found"] and not units[("omop_note_nlp", None)]["found"]
    assert units[("eeg_metadata", "S0002")]["missing"] == []
    assert rep["n_tables_not_found"] == 2 and rep["n_missing_confirmed"] == 2
    # names only: no patient ids, session ids, folders or timestamps anywhere in stdout or files
    all_ids = {str(p) for p in synth[0]["omop_person"]["person_id"]}
    blob = text + (out_dir / "schema_dry_run.json").read_text() + (out_dir / "schema_dry_run.md").read_text()
    assert not (set(re.findall(r"[A-Za-z0-9_\-]{6,}", blob)) & all_ids)
    assert not re.search(r"sub-|ses-|\d{4}-\d\d-\d\d", blob)
    assert "Still unknown" in text


def test_dry_run_does_not_read_rows(synth_dir, monkeypatch):
    """The probe must only ever read headers and parquet footers, never whole objects."""
    from sortinghat import data_io
    calls = []
    orig = data_io.LocalStore.get_object

    def spy(self, Bucket=None, Key="", Range=None):
        calls.append(Range)
        return orig(self, Bucket, Key, Range)
    monkeypatch.setattr(data_io.LocalStore, "get_object", spy)
    field_audit.probe_schema(data_io.LocalStore(synth_dir))
    assert calls and all(r is not None for r in calls)
