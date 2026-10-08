import json
import re
import shutil
import subprocess
import sys

import pandas as pd
import pytest

from sortinghat import schema
from sortinghat import data_io
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


SHIFT_TABLES = ("omop_visit_occurrence", "omop_measurement", "omop_observation", "omop_note", "omop_drug_exposure",
                "omop_condition_occurrence")


def _shift_site(raw, analytic, site, days, tables=SHIFT_TABLES):
    """Raw-table copy with every time column of ``tables`` shifted by ``days`` for the patients of ``site``."""
    out = _copy(raw)
    eeg = analytic["eeg_metadata"]
    pids = set(eeg.loc[eeg["SiteID"] == site, "person_id"].dropna().astype(int))
    for t in tables:
        d = out[t]
        m = d["person_id"].isin(pids)
        for c in d.columns:
            if pd.api.types.is_datetime64_any_dtype(d[c]):
                d.loc[m, c] = d.loc[m, c] + pd.Timedelta(days=days)
    return out


def _ds(report):
    return next(r for r in report["rows"] if r["field"].startswith("Consistent"))


def test_date_shift_gate_passes_on_aligned_synthetic_data(synth):
    ds = _ds(run_audit(synth[0])[0])
    o = ds["observed"]
    assert ds["passed"] and not o["sites_failed"] and not o["nonzero_day_modes_by_site"]
    assert set(o["sites_evaluated"]) == {"I0002", "I0003", "S0001", "S0002"}
    big = o["by_site_detail"]["S0001"]
    assert big["nearest_event_within_24h"]["n_pass"] != "<11"
    assert big["whole_day_gap_top5"][0]["day"] == 0
    sd = big["shift_detection"]
    assert sd["eeg_date_minus_visit_start_date_days"]["q50"] == 0          # EEG date inside the covering visit
    assert "per_table_within_24h" in big and set(big["per_table_within_24h"]) >= {"note", "measurement", "condition"}


def test_date_shift_gate_detects_a_site_level_constant_offset(synth, analytic):
    report, _ = run_audit(_shift_site(synth[0], analytic, "S0001", 10))
    ds = _ds(report)
    o = ds["observed"]
    assert not ds["passed"] and "Consistent within-patient date shift" in report["stop_rows_failed"]
    assert report["gate_0a_automated_stop_rows_pass"] is False
    assert o["sites_failed"] == ["S0001"]
    assert o["nonzero_day_modes_by_site"]["S0001"] and set(o["nonzero_day_modes_by_site"]["S0001"]) <= {6, 7, 8, 9, 10}  # 10 d minus the +-2 d spread of the synthetic EHR
    top = o["by_site_detail"]["S0001"]["whole_day_gap_top5"]
    assert len(top) <= 5 and top[0]["day"] in (6, 7, 8, 9) and top[0]["n"] != "<11"
    for other in ("I0003", "S0002"):                                        # the other sites are untouched
        assert o["by_site_detail"][other]["site_gate"] == "PASS"
    sd = o["by_site_detail"]["S0001"]["shift_detection"]                    # visits shifted 10 d: none covers the EEG now
    assert sd["candidates_with_covering_visit"] == "<11"
    assert o["by_site_detail"]["S0002"]["shift_detection"]["candidates_with_covering_visit"] != "<11"


def test_date_shift_gate_fails_when_too_few_candidates_have_a_nearby_event(synth, analytic):
    # a 40-day offset in EVERY ancillary table at one site: no event within +-24 h, and the gap mode is far from 0
    report, _ = run_audit(_shift_site(synth[0], analytic, "I0003", 40))
    o = _ds(report)["observed"]
    assert "I0003" in o["sites_failed"] and o["by_site_detail"]["I0003"]["site_gate"] == "FAIL"
    assert o["by_site_detail"]["I0003"]["nearest_event_within_24h"]["proportion"] in ("<11",) or \
        float(str(o["by_site_detail"]["I0003"]["nearest_event_within_24h"]["proportion"]).lstrip(">=")) < 0.5


def test_one_shifted_table_is_reported_per_table_but_does_not_fail_the_gate(synth, analytic):
    # the gate asks whether SOME event is near the EEG; one table off by 100 d is visible per table, not a gate failure
    report, _ = run_audit(_shift_site(synth[0], analytic, "S0001", 100, tables=("omop_note",)))
    ds = _ds(report)
    assert ds["passed"]
    pt = ds["observed"]["by_site_detail"]["S0001"]["per_table_within_24h"]
    assert pt["note"] == "<11" or float(str(pt["note"]).lstrip(">=")) < 0.2
    assert pt["measurement"] != "<11" and float(str(pt["measurement"]).lstrip(">=")) > 0.9


def test_old_median_proxy_is_gone(synth):
    ds = _ds(run_audit(synth[0])[0])
    assert "misaligned_proportion" not in ds["observed"] and "median" not in ds["pass_criterion"]
    assert "+-24 h" in ds["pass_criterion"] and "80%" in ds["pass_criterion"]


def test_day_modes_are_peaks_not_the_natural_tail():
    from sortinghat.audit.alignment import day_peaks
    natural = pd.Series([0] * 800 + [1] * 120 + [2] * 40 + [-1] * 40)        # 12% on day 1 beside a dominant day 0
    assert day_peaks(natural, 1000, 0.05) == []
    shifted = pd.Series([3] * 600 + [4] * 250 + [0] * 100 + [1] * 50)
    assert day_peaks(shifted, 1000, 0.05) == [3]


def test_ordering_violations_are_secondary_and_precisely_defined(synth, analytic):
    raw = _copy(synth[0])
    eeg = analytic["eeg_metadata"]
    pids = sorted(set(eeg["person_id"].dropna().astype(int)))[:600]
    note, per, death = raw["omop_note"], raw["omop_person"], raw["omop_death"]
    # birth AFTER every event of the first 300 patients
    sel = per["person_id"].isin(pids[:300])
    per.loc[sel, "birth_datetime"] = pd.Timestamp("2200-01-01")
    # death 30 days BEFORE the notes of the next 300 patients (more than 24 h => counted)
    nxt = pids[300:600]
    first_note = note[note["person_id"].isin(nxt)].groupby("person_id")["note_datetime"].min()
    newd = pd.DataFrame({"person_id": first_note.index, "death_datetime": first_note.values - pd.Timedelta(days=30),
                         "death_date": pd.NaT, "cause_source_value": None})
    raw["omop_death"] = pd.concat([death, newd], ignore_index=True)
    report, _ = run_audit(raw)
    ds = _ds(report)
    sec = ds["observed"]["ordering_violations_secondary"]
    assert sec["rows_by_source"]["note"]["n_before_birth"] != "<11"
    assert sec["rows_by_source"]["measurement"]["n_before_birth"] != "<11"
    assert sec["rows_by_source"]["note"]["n_after_death_plus_24h"] != "<11"
    assert ds["passed"]                                                       # secondary: never the gate
    base = _ds(run_audit(synth[0])[0])["observed"]["ordering_violations_secondary"]["rows_by_source"]["note"]
    assert base["n_before_birth"] == "<11"


def test_monotonicity_violation_detected_as_secondary_count(synth):
    raw = _copy(synth[0])
    dx = raw["omop_drug_exposure"]
    idx = dx.index[dx["drug_exposure_end_datetime"].notna()][:400]
    dx.loc[idx, "drug_exposure_end_datetime"] = dx.loc[idx, "drug_exposure_start_datetime"] - pd.Timedelta(hours=1)
    ds = _ds(run_audit(raw)[0])
    v = ds["observed"]["ordering_violations_secondary"]["within_record_end_before_start"]
    assert v["drug_start_before_end"]["n_violations"] != "<11"
    assert ds["passed"]                                                       # ordering is reported, not the gate


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
    head = head.replace("DurationInSeconds", "DurationInSecond").replace(",BidsFlag", "")
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
    assert miss["BidsFlag"]["provenance"] == "CONFIRMED"
    assert [m["column"] for m in units[("reports_findings", "S0001")]["missing"]] == ["StartTime(EEG)"]
    assert units[("reports_findings", "S0001")]["missing"][0]["closest_actual_name"] == "StartTime (EEG)"
    assert not units[("imaging", None)]["found"] and not units[("omop_note_nlp", None)]["found"]
    assert units[("eeg_metadata", "S0002")]["missing"] == []
    # I0008 / I0009 have no reports_findings file by design: absent, but not counted as a missing table
    assert units[("reports_findings", "I0008")]["expected_absent"] and not units[("reports_findings", "I0008")]["found"]
    assert rep["n_tables_not_found"] == 2 and rep["n_missing_confirmed"] == 3
    # names only: no patient ids, session ids, folders or timestamps anywhere in stdout or files
    all_ids = {str(p) for p in synth[0]["omop_person"]["person_id"]}
    blob = text + (out_dir / "schema_dry_run.json").read_text() + (out_dir / "schema_dry_run.md").read_text()
    assert not (set(re.findall(r"[A-Za-z0-9_\-]{6,}", blob)) & all_ids)
    assert not re.search(r"sub-|ses-|\d{4}-\d\d-\d\d", blob)
    assert "Still unknown" in text


def test_dry_run_per_site_variants_are_not_drift(synth_dir):
    """Each site is compared with ITS OWN real header: variants differ but nothing is missing or unlisted."""
    rep = field_audit.probe_schema(data_io.LocalStore(synth_dir), list_unlisted=True)
    for u in rep["units"]:
        if u["found"]:
            assert u["missing"] == [] and u["unlisted_columns"] == [], (u["table"], u["site"])
    assert rep["n_missing_confirmed"] == 0 and rep["n_tables_not_found"] == 0


def test_list_unlisted_prints_names_only(synth_dir, tmp_path, capsys):
    d = tmp_path / "extra"
    shutil.copytree(synth_dir, d)
    f = d / "EEG/HEEDB_Metadata/HEEDB_patients.csv"
    head, rest = f.read_text().split("\n", 1)
    f.write_text(head + ",SomeNewColumn,sub-S0001123456\n" + rest)
    assert field_audit.main(["--data", str(d), "--dry-run-schema", "--list-unlisted"]) == 0
    out = capsys.readouterr().out
    assert "Unlisted columns" in out and "heedb_patients: `SomeNewColumn`, `<id-like column>`" in out
    assert "sub-S0001123456" not in out


def test_probe_prefixes_lists_levels_and_never_enters_patient_folders(synth_dir, tmp_path, capsys):
    d = tmp_path / "px"
    shutil.copytree(synth_dir, d)
    (d / "EEG/bids/S0001/sub-S0001123456/ses-1/eeg").mkdir(parents=True)
    (d / "EEG/bids/S0001/sub-S0001123456/ses-1/eeg/x.edf").write_bytes(b"x")
    for i in range(60):                                                           # a population of folders
        (d / f"Imaging/I0004/Clinical/f{i}").mkdir(parents=True, exist_ok=True)
    (d / "Imaging/I0001/sub-123456").mkdir(parents=True)
    (d / "Imaging/I0001/BIDS").mkdir()
    (d / "Imaging/I0001/12345.csv").write_text("a\n")
    assert field_audit.main(["--data", str(d), "--dry-run-schema", "--probe-prefixes"]) == 0
    out = capsys.readouterr().out
    assert "## Access point prefixes" in out and "prefix `EEG/`" in out and "prefix `bids/`" in out
    assert "prefix `BIDS/`" in out
    assert "sub-S0001123456" not in out and "ses-1" not in out and "x.edf" not in out        # not listed
    assert "sub-123456" not in out and "12345.csv" not in out                                  # masked
    assert "<id-like prefix>" in out and "<id-like file>" in out
    assert "Imaging/I0004/Clinical/f0" not in out                                              # not descended


def test_id_like_guard_and_flags_need_dry_run():
    for bad in ("sub-S0001123", "ses-1", "123456", "1234567.csv", "2026_04_30", "18thAugust2025",
                "0123456789abcdef0123"):
        assert data_io.looks_id_like(bad), bad
    for ok in ("I0001", "S0002-EHR", "Clinical", "BIDS", "Non-BIDS", "part-00000.parquet", "HEEDB_Metadata"):
        assert not data_io.looks_id_like(ok), ok
    with pytest.raises(SystemExit):
        field_audit.main(["--s3", "--out", "x", "--list-unlisted"])


def test_local_store_delimiter_listing(synth_dir):
    s3 = data_io.LocalStore(synth_dir)
    root = s3.list_objects_v2(Bucket="x", Prefix="", Delimiter="/")
    assert {c["Prefix"] for c in root["CommonPrefixes"]} >= {"EEG/", "OMOP/"}
    eeg = s3.list_objects_v2(Bucket="x", Prefix="EEG/", Delimiter="/")
    assert {c["Prefix"] for c in eeg["CommonPrefixes"]} == {"EEG/HEEDB_Metadata/", "EEG/eeg-metadata/"}


# ------------------------------------------------------------ PatientClass is derived from visits on disk
def test_patient_class_derivation_matches_generated_class_where_a_time_exists(synth, synth_dir):
    tabs = field_audit.load_audit_tables(data_io.LocalStore(synth_dir))
    got = tabs["eeg_metadata"]
    assert got["PatientClass"].notna().mean() > 0.9
    mem = field_audit.from_raw_tables(synth[0])["eeg_metadata"]
    adult = got[(got["AgeAtVisit"] >= 18) & got["ClassTime"].notna()]       # visits are read for these only
    a = adult.set_index("SessionID")["PatientClass"]
    b = mem.set_index("SessionID")["PatientClass"].reindex(a.index)
    assert len(a) > 1000 and (a.fillna("-") == b.fillna("-")).all()


def test_visit_class_prefers_concept_then_text():
    vc = field_audit.visit_class
    assert vc(32037, "whatever") == "ICU" and vc(0, "Inpatient") == "Inpatient" and vc(0, "ED") == "ED"
    assert vc(0, "Outpatient") == "Outpatient" and vc(0, "") is None and vc(None, None) is None


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


# ------------------------------------------------------------ drug_type_concept_id semantics and score concepts by name
ADMIN_ID, ORDER_ID = 32818, 38000177            # ids used by the synthetic generator
GCS_CID, RASS_CID, FOUR_CID = 3000001, 3000002, 3000003


def _concepts(extra=()):
    rows = [(ADMIN_ID, "EHR administration record", "Type Concept", "Type Concept"),
            (ORDER_ID, "Prescription written", "Type Concept", "Type Concept"),
            (GCS_CID, "Glasgow coma score total", "Measurement", "LOINC"),
            (RASS_CID, "Richmond agitation sedation scale", "Observation", "LOINC"),
            (FOUR_CID, "Full outline of unresponsiveness score", "Measurement", "LOINC"),
            (3000009, "Glasgow outcome scale", "Condition", "SNOMED")] + list(extra)
    return pd.DataFrame(rows, columns=["concept_id", "concept_name", "domain_id", "vocabulary_id"])


def test_classify_drug_type_names():
    c = field_audit.classify_drug_type
    assert c("EHR administration record") == "administration" and c("Inpatient administration") == "administration"
    assert c("Prescription written") == "order" and c("Prescription dispensed in pharmacy") == "order"
    assert c("EHR order") == "order" and c("Claim") == "other" and c(None) == "unresolved"


def test_medication_uses_drug_type_concept_names_when_they_resolve(analytic):
    tables = _copy(analytic)
    tables["omop_concept"] = _concepts()
    dx = tables["omop_drug_exposure"]
    # administration-type rows WITHOUT an end datetime must still count (the old proxy would call them orders)
    dx["drug_exposure_end_datetime"] = pd.NaT
    dx["drug_type_concept_id"] = ADMIN_ID
    report, _ = run_audit(tables)
    med = next(r for r in report["rows"] if r["field"].startswith("Medication"))
    assert med["passed"] and med["details"]["drug_type_semantics_resolved"]
    assert "concept" in med["details"]["method"] and "FALLBACK" not in med["details"]["method"]
    # the same rows typed as orders although they carry end datetimes: FAIL (an end time is not administration)
    dx["drug_exposure_end_datetime"] = dx["drug_exposure_start_datetime"] + pd.Timedelta(hours=1)
    dx["drug_type_concept_id"] = ORDER_ID
    report, _ = run_audit(tables)
    med = next(r for r in report["rows"] if r["field"].startswith("Medication"))
    assert not med["passed"] and med["fallback_triggered"]
    assert {c["category"] for c in med["details"]["drug_type_concepts"]} == {"order"}


def test_medication_falls_back_to_end_time_proxy_when_type_ids_do_not_resolve(analytic):
    tables = _copy(analytic)
    tables["omop_drug_exposure"]["drug_type_concept_id"] = 0                  # zero-filled (rule 6)
    report, _ = run_audit(tables)
    med = next(r for r in report["rows"] if r["field"].startswith("Medication"))
    assert not med["details"]["drug_type_semantics_resolved"] and "FALLBACK" in med["details"]["method"]


def test_scores_found_by_concept_name_in_measurement_and_observation(analytic):
    tables = _copy(analytic)
    m = tables["omop_measurement"]
    sc = m["kind"] == "score"
    m.loc[sc, "measurement_source_value"] = None                              # text gives nothing: concept id only
    m.loc[sc, "measurement_concept_id"] = GCS_CID
    m.loc[sc, "score_class"] = None
    base, _ = run_audit(tables)                                               # no concept table: nothing identifiable
    assert not next(r for r in base["rows"] if r["field"].startswith("GCS"))["passed"]
    # rebuild the analytic frames from raw rows through the concept map, as the loaders do
    concept = _concepts()
    score_map, _ = field_audit.concept_maps(concept)
    assert score_map == {GCS_CID: "GCS", RASS_CID: "RASS", FOUR_CID: "FOUR"}     # outcome scale (Condition) excluded
    raw = m.drop(columns=["kind", "score_class"])
    tables["omop_measurement"] = field_audit.filter_measurements(raw, score_map)
    tables["omop_concept"] = concept
    report, _ = run_audit(tables)
    row = next(r for r in report["rows"] if r["field"].startswith("GCS"))
    assert row["passed"] and row["details"]["by_class_within_6h"]["GCS"]["proportion"] != "<11"
    assert row["details"]["n_score_concepts_found_by_name"]["GCS"] == 1
    # observation-table scores count too
    obs = tables["omop_measurement"][lambda d: d["kind"] == "score"].head(300)
    tables["omop_observation"] = pd.DataFrame({
        "person_id": obs["person_id"], "observation_datetime": obs["measurement_datetime"],
        "observation_source_value": "RASS score", "score_class": "RASS"})
    tables["omop_measurement"] = tables["omop_measurement"][lambda d: d["kind"] != "score"]
    report, _ = run_audit(tables)
    row = next(r for r in report["rows"] if r["field"].startswith("GCS"))
    assert row["details"]["by_source_within_6h"]["observation"]["n_candidates"] != 0
    assert row["details"]["by_source_within_6h"]["measurement"]["n_candidates"] == "<11"     # none left (0 -> <11)


def test_other_scales_do_not_count_towards_the_criterion(analytic):
    tables = _copy(analytic)
    m = tables["omop_measurement"]
    sc = m["kind"] == "score"
    m.loc[sc, "score_class"] = "OTHER"
    report, _ = run_audit(tables)
    row = next(r for r in report["rows"] if r["field"].startswith("GCS"))
    assert not row["passed"]


def test_loader_resolves_concepts_by_name_and_drug_types_on_disk(synth, synth_dir, tmp_path):
    """load_audit_tables on a store whose omop_concept has the type and score concepts: the streaming path
    (Arrow-side filters) finds score rows with NO source text and classifies drug types by name."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    d = tmp_path / "withconcepts"
    shutil.copytree(synth_dir, d)
    cdir = d / "OMOP/Merged/concept"
    part = sorted(cdir.glob("*.parquet"))[0]
    old = pq.read_table(part).to_pandas()
    pq.write_table(pa.Table.from_pandas(pd.concat([old, _concepts().assign(
        standard_concept="S", concept_class_id="x", concept_code="x")], ignore_index=True), preserve_index=False), part)
    mpart = sorted((d / "OMOP/Merged/measurement").glob("*.parquet"))[0]
    mt = pq.read_table(mpart).to_pandas()
    gcs = mt["measurement_source_value"].astype(str).str.contains("Glasgow")
    assert gcs.sum() > 50
    mt.loc[gcs, "measurement_concept_id"] = GCS_CID
    mt.loc[gcs, "measurement_source_value"] = "x"
    pq.write_table(pa.Table.from_pandas(mt, preserve_index=False), mpart)
    tabs = field_audit.load_audit_tables(data_io.LocalStore(d))
    sm = tabs["omop_measurement"]
    assert (sm["score_class"] == "GCS").sum() >= gcs.sum() * 0.5            # only cohort patients are kept
    assert set(tabs["omop_concept"]["concept_id"]) >= {GCS_CID, ADMIN_ID, ORDER_ID}
    assert 3000009 not in set(tabs["omop_concept"]["concept_id"])             # Condition-domain "Glasgow outcome": excluded
    report, _ = run_audit(tabs)
    med = next(r for r in report["rows"] if r["field"].startswith("Medication"))
    assert med["details"]["drug_type_semantics_resolved"] and "concept" in med["details"]["method"]
    assert not any("Inpatient" in str(k) for k in med["details"])            # names live in list values, not keys


def test_real_data_failure_prints_stage_and_class_only(synth_dir, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(data_io, "open_store", lambda *a, **k: data_io.LocalStore(synth_dir))

    def boom(*a, **k):
        raise ValueError("bad cell 2019-03-04 sub-S0001123456")
    monkeypatch.setattr(field_audit, "load_audit_tables", boom)
    assert field_audit.main(["--s3", "--out", str(tmp_path / "o")]) == 2
    out = capsys.readouterr().out
    assert "ValueError" in out and "load" in out and "sub-S0001" not in out and "2019" not in out


# ------------------------------------------------------------ the audit reuses the cohort's candidates and readers
def _real_like_store(synth_dir, tmp_path):
    """Synthetic store made to look like the real visit table: every visit_concept_id 0, no source text."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    d = tmp_path / "reallike"
    shutil.copytree(synth_dir, d)
    for part in (d / "OMOP/Merged/visit_occurrence").glob("*.parquet"):
        t = pq.read_table(part).to_pandas()
        t["visit_concept_id"] = 0
        t["visit_source_value"] = None
        pq.write_table(pa.Table.from_pandas(t, preserve_index=False), part)
    return d


def test_candidates_follow_the_cohort_rules_when_concept_ids_are_all_zero(synth_dir, tmp_path):
    from sortinghat.cohort import CohortConfig, StoreSources, build_cohort
    d = _real_like_store(synth_dir, tmp_path)
    store = data_io.LocalStore(d)
    tabs = field_audit.load_audit_tables(store)
    cands = field_audit.build_candidates(tabs["eeg_metadata"])
    assert set(cands["SiteID"]) <= {"I0002", "I0003", "S0001", "S0002"}                 # D-113
    assert tabs["eeg_metadata"]["PatientClass"].isin(["Inpatient", "Outpatient"]).sum() > 1000     # proxy-filled classes
    report, _ = run_audit(tabs)
    assert report["rows"][0]["acute_care_filter_applied"] is True                       # not the "all adult EEGs" fallback
    res = build_cohort(StoreSources(store), CohortConfig())                              # the cohort's own candidates
    cohort_ids = set(res.stages["first_eeg"]["person_id"])
    audit_ids = set(cands["person_id"])
    assert len(audit_ids) > 300 and len(cohort_ids & audit_ids) / len(cohort_ids | audit_ids) > 0.97


def test_staged_reader_matches_the_plain_reader(synth_dir):
    from sortinghat.cohort.sources import iter_filtered_batches
    store = data_io.LocalStore(synth_dir)
    ids = range(50_000_000, 50_001_500)
    pat = field_audit.SCORE_RE.pattern
    got = sum(t.num_rows for t in iter_filtered_batches(store, "measurement", ids, ["person_id", "measurement_datetime",
                                                        "measurement_source_value"], text_col="measurement_source_value",
                                                        pattern=pat))
    ref = 0
    for b in data_io.iter_omop_batches("measurement", person_ids=ids, columns=["person_id", "measurement_source_value"], s3=store):
        d = b.to_pandas()
        ref += int(d["measurement_source_value"].astype(str).str.contains(pat, case=False, regex=True).sum())
    assert got == ref > 100


def test_loader_applies_the_patient_merge_map(synth_dir, tmp_path):
    d = tmp_path / "m"
    shutil.copytree(synth_dir, d)
    base = field_audit.load_audit_tables(data_io.LocalStore(synth_dir))["eeg_metadata"]
    c = field_audit.build_candidates(base).sort_values("person_id")
    a, b = int(c["person_id"].iloc[0]), int(c["person_id"].iloc[1])
    (d / "PatientMergeHistory").mkdir()
    pd.DataFrame({"MergedBDSPPatientID": [b], "BDSPPatientID": [a], "LineNBR": [1],
                  "BDSPLastModifiedDTS": ["2020-01-01"]}).to_csv(d / "PatientMergeHistory" / "m.csv", index=False)
    merged = field_audit.load_audit_tables(data_io.LocalStore(d))["eeg_metadata"]
    assert b not in set(merged["person_id"]) and a in set(merged["person_id"])


# ---------------------------------------------------------------- per-site pass counts add up to the overall (bug B)
def test_pass_cell_never_hides_a_near_perfect_site():
    from sortinghat.audit.field_audit import pass_cell
    c = pass_cell(50925, 50931)                      # 6 failures: used to print "<11/50931" and read as "all fail"
    assert c["n_pass"] == ">=50921" and c["proportion"] == ">=0.9998" and c["n_total"] == 50931
    assert pass_cell(50900, 50931)["n_pass"] == 50900                       # 31 failures: exact
    assert pass_cell(5, 50931) == {"n_pass": "<11", "n_total": 50931, "proportion": "<11"}
    assert pass_cell(5, 8) == {"n_pass": "<11", "n_total": "<11", "proportion": "<11"}
    assert pass_cell(50931, 50931)["proportion"] == ">=0.9998"              # 0 failures: still bounded, never "<11"


def _bounds(cell):
    """(lo, hi) of the true pass count implied by a suppressed cell."""
    v, tot = cell["n_pass"], cell["n_total"]
    if isinstance(v, int):
        return v, v
    if v == "<11":
        return 0, 10
    return int(v[2:]), int(tot)


def _check_rows_add_up(report):
    checked = 0
    for r in report["rows"]:
        blocks = []
        o = r["observed"]
        if isinstance(o, dict) and "by_site" in o:
            blocks.append(o)
        if isinstance(o, dict) and "within_24h" in o:
            blocks += [o["within_24h"], o["within_72h"]]
        for blk in blocks:
            overall = blk["overall"]
            if overall["n_total"] == "<11":
                continue
            lo_o, hi_o = _bounds(overall)
            parts = [_bounds(b) for b in blk["by_site"].values()]
            lo, hi = sum(p[0] for p in parts), sum(p[1] for p in parts)
            assert lo <= hi_o and lo_o <= hi, (r["field"], overall, blk["by_site"])
            if all(p[0] == p[1] for p in parts) and lo_o == hi_o:
                assert lo == lo_o, (r["field"], overall, blk["by_site"])       # all exact: must be equal
            tot = [b["n_total"] for b in blk["by_site"].values()]
            if all(isinstance(t, int) for t in tot):
                assert sum(tot) == overall["n_total"], r["field"]
            checked += 1
    return checked


def test_per_site_pass_counts_add_up_to_the_overall_pass_count_for_every_row(synth, analytic):
    report, _ = run_audit(synth[0])
    assert _check_rows_add_up(report) >= 7                  # start, shift x2, medication, imaging, scores, notes
    tables = _copy(analytic)                                # and with a lab result column (the lab row has a by-site block)
    m = tables["omop_measurement"]
    m["measurement_result_datetime"] = m["measurement_datetime"] + pd.Timedelta(minutes=30)
    rep2, _ = run_audit(tables)
    assert any(r["field"].startswith("Lab result") and "by_site" in r["observed"] for r in rep2["rows"])
    assert _check_rows_add_up(rep2) >= 8


def test_start_time_row_per_site_matches_overall_exactly_on_a_large_synthetic_set():
    from sortinghat.synthetic import generate
    tables, _ = generate(12000, seed=7)
    report, _ = run_audit(tables)
    blk = report["rows"][0]["observed"]
    exact = [b for b in blk["by_site"].values() if isinstance(b["n_pass"], int)]
    assert exact, blk
    for b in blk["by_site"].values():                       # no site with a high pass rate reads "<11"
        assert not (b["n_pass"] == "<11" and isinstance(b["n_total"], int) and b["n_total"] > 50 and b["proportion"] == "<11"
                    and b["n_total"] >= 200), b
    if all(isinstance(b["n_pass"], int) for b in blk["by_site"].values()):
        assert sum(b["n_pass"] for b in blk["by_site"].values()) == blk["overall"]["n_pass"]
