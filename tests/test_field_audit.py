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
