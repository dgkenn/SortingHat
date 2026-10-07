"""Synthetic HEEDB-shaped dataset (numpy/pandas only). Entirely fake.

Emits the REAL layout (``docs/heedb_schema_real.md``) INCLUDING the per-site header variants that the first
names-only dry run found (``schema.SITE_VARIANTS``): S0001/S0002 (EEGFolder, DateOfDeath, ServiceName; blank
``BDSPPatientID``, blank eeg_metadata ``StartTime``/``EndTime``), I0002 (no ServiceName / DateOfDeath / EEGFolder),
I0003 (adds ``AgeInDaysAtVisit``), and I0008/I0009 (no ``SiteID`` but ``InstituteID``; ``StartDateTime`` /
``EndDateTime`` / ``RecordingDuration`` / ``DateOfBirth`` and no ``AgeAtVisit``; and NO reports_findings file).
The in-memory ``tables`` use CANONICAL names, with every column a site does not have blanked, so they equal what
``data_io.read_site_table`` returns after the files are written. ``PatientClass`` / ``ReferralIndication`` exist in
memory only (in no real header); the on-disk audit derives PatientClass from ``omop_visit_occurrence``.

Also written: the three wide per-patient CSVs and OMOP parquet tables (``person_id`` = int(BDSPPatientID); OMOP CDM
column names; datetimes as text). Columns that are still ASSUMED in ``sortinghat/schema.py`` (imaging finalization,
...) are filled so the audit logic can run on them.

Deliberately injected defects so the field audit has something to find:
  * StartTime(EEG) missing for ~2-5% of EEGs (site dependent; at I0008/I0009 the metadata StartDateTime).
  * drug_exposure_end_datetime (the administration-record proxy) absent for patients without a MAR feed.
  * Imaging report finalization time missing (worse at one site).
  * No lab result-time column (as in the real measurement table).
  * ~3% of patients have one ancillary table (note/measurement/imaging/drug_exposure) carrying an
    extra date offset, i.e. an inconsistent within-patient date shift.
  * Score documentation near the EEG is patchy; one pediatric site.
Every patient also receives a random per-patient date shift (consistent across tables), mirroring BDSP
de-identification.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import schema
from ..tableio import write_tables as _write_tables

SITE_WEIGHTS = {"S0001": 0.36, "S0002": 0.27, "I0003": 0.18, "I0002": 0.10, "I0008": 0.05, "I0009": 0.04}
PEDIATRIC_SITE = "I0002"
BLANK_PATIENT_ID_SITES = {"S0001", "S0002"}      # eeg_metadata.BDSPPatientID blank on some real releases
START_MISSING = {"S0001": 0.02, "S0002": 0.05, "I0003": 0.02, "I0002": 0.02, "I0008": 0.03, "I0009": 0.03}
MAR_COVERAGE = {"S0001": 0.93, "S0002": 0.55, "I0003": 0.85, "I0002": 0.70, "I0008": 0.60, "I0009": 0.60}
IMG_FINAL_MISSING = {"S0001": 0.04, "S0002": 0.06, "I0003": 0.20, "I0002": 0.08, "I0008": 0.10, "I0009": 0.10}
SCORE_NEAR = {"S0001": 0.75, "S0002": 0.60, "I0003": 0.35, "I0002": 0.50, "I0008": 0.40, "I0009": 0.40}
METADATA_START_SITES = {"I0008", "I0009"}        # real start/end live in eeg_metadata (StartDateTime/EndDateTime); no reports_findings
# canonical names mapped by ANY site variant: a site that lacks one gets it blanked (it has no such header column)
_MAPPED = {t: {c for v in schema.SITE_VARIANTS.values() for c in ((v.eeg_metadata if t == "eeg_metadata"
                                                               else (v.reports_findings or {})).values()) if c}
           for t in ("eeg_metadata", "reports_findings")}
TABLE_SHIFT_RATE = 0.03
ID_BASE = 50_000_000                             # synthetic BDSPPatientID / OMOP person_id range
SESSION_BASE = 900_000_000

CLASSES = ["ICU", "Inpatient", "ED", "Outpatient"]
CLASS_P = [0.25, 0.35, 0.10, 0.30]
ACUTE = {"ICU", "Inpatient", "ED"}
VISIT_CONCEPT = {"ICU": 32037, "Inpatient": 9201, "ED": 9203, "Outpatient": 9202}   # standard OMOP visit concepts
INDICATIONS = ["rule out NCSE", "post-arrest", "unexplained AMS", "seizure", "spell", "other"]
# drug_source_value is free text in the real table; sedation-class rows are matched by regex in the audit
DRUGS = [("PROPOFOL 10 MG/ML IV EMULSION", True), ("MIDAZOLAM 1 MG/ML INJ", True),
         ("DEXMEDETOMIDINE 4 MCG/ML IV", True), ("FENTANYL 50 MCG/ML INJ", True),
         ("LORAZEPAM 2 MG/ML INJ", False), ("LEVETIRACETAM 500 MG/5 ML IV", False),
         ("ACETAMINOPHEN 325 MG TAB", False), ("HEPARIN 5000 UNIT/ML INJ", False)]
LABS = [("LACTATE, WHOLE BLOOD", "mmol/L"), ("AMMONIA, PLASMA", "umol/L"), ("SODIUM", "mmol/L"),
        ("GLUCOSE", "mg/dL"), ("CREATININE", "mg/dL"), ("BUN", "mg/dL"), ("AST", "U/L"), ("ALT", "U/L"),
        ("WBC", "K/uL")]
SCORES = [("Glasgow Coma Scale Score", 0.5, (3, 16)), ("FOUR Score", 0.15, (0, 17)),
          ("RASS (Richmond Agitation Sedation Scale)", 0.25, (-5, 5)), ("Eye Opening", 0.1, (1, 5))]
MODALITIES = ["CT head", "MRI brain", "CTA head"]
NOTE_TYPES = [(1, "physician"), (2, "nursing"), (3, "eeg_report"), (4, "other")]   # ids are placeholders
ICD = ["I63.9", "G40.909", "G93.41", "R40.2", "I46.9", "E11.9", "I10", "N17.9", "A41.9"]
FLAGS_ALL = schema.FINDING_FLAGS_ALL
DRUG_TYPE_ORDER, DRUG_TYPE_ADMIN = 38000177, 32818    # OMOP drug-type concept ids (placeholders for the synthetic data)
MEAS_TYPE_LAB, MEAS_TYPE_EXAM = 44818702, 44818701
SHIFTABLE = ["omop_note", "omop_measurement", "imaging", "omop_drug_exposure"]
TIME_COLS = {"omop_note": ["note_datetime"], "omop_measurement": ["measurement_datetime"],
             "imaging": ["study_datetime", "report_final_datetime"],
             "omop_drug_exposure": ["drug_exposure_start_datetime", "drug_exposure_end_datetime"]}


def _only_available(table: str, site: str, row: dict) -> dict:
    """Drop canonical columns that the site's real header does not have (they stay NaN in the frame)."""
    have = schema.canonical_to_actual(table, site)
    return {k: v for k, v in row.items() if k in have or k not in _MAPPED[table] or k == "SiteID"}


def _ts(base: pd.Timestamp, hours: float) -> pd.Timestamp:
    return (base + pd.Timedelta(minutes=float(hours) * 60.0)).round("us")   # microsecond precision, like the real text


DATE_TWINS = {"omop_note": {"note_date": "note_datetime"}, "omop_measurement": {"measurement_date": "measurement_datetime"},
              "omop_drug_exposure": {"drug_exposure_start_date": "drug_exposure_start_datetime",
                                     "drug_exposure_end_date": "drug_exposure_end_datetime"}}


def _dates(rows, tbl):
    """Recompute the *_date twins of shifted datetime columns."""
    for dc, col in DATE_TWINS.get(tbl, {}).items():
        for r in rows:
            r[dc] = pd.NaT if pd.isna(r[col]) else r[col].normalize()


def generate(n_patients: int = 3000, seed: int = 20260101):
    """Return ``(tables, truth)``. ``tables`` is keyed by ``schema`` table name; ``truth`` is in-memory
    ground truth for tests only."""
    rng = np.random.default_rng(seed)
    sites = rng.choice(list(SITE_WEIGHTS), size=n_patients, p=list(SITE_WEIGHTS.values()))

    R: dict[str, list] = {t: [] for t in schema.TABLE_NAMES}
    truth = {"table_shift_ids": [], "table_shift_table": {}, "no_start_sessions": 0}
    note_counter = sess_counter = visit_counter = 0

    for i in range(n_patients):
        site = str(sites[i])
        pid = ID_BASE + i
        # --- per-patient demographics and date shift (consistent across tables)
        if site == PEDIATRIC_SITE and rng.random() < 0.9:
            age = float(rng.uniform(0.1, 17.9))
        else:
            age = float(np.clip(rng.normal(58, 18), 18, 95))
        sex = str(rng.choice(["Female", "Male"]))
        true_anchor = pd.Timestamp("2014-01-01") + pd.Timedelta(days=int(rng.integers(0, 3000)))
        shift = pd.Timedelta(days=int(rng.integers(-1800, 1800)))
        anchor = true_anchor + shift + pd.Timedelta(minutes=int((rng.normal(13, 5) % 24) * 60))

        # --- sessions
        n_sess = int(min(1 + rng.poisson(0.5), 4))
        classes = [str(rng.choice(CLASSES, p=CLASS_P)) for _ in range(n_sess)]
        classes[0] = str(rng.choice(CLASSES, p=[0.30, 0.40, 0.10, 0.20]))
        offsets = [0] + sorted(int(x) for x in rng.integers(1, 400, n_sess - 1))
        starts = [anchor + pd.Timedelta(days=o) for o in offsets]
        first_acute = next((k for k, c in enumerate(classes) if c in ACUTE), None)
        start_missing = [bool(rng.random() < START_MISSING[site]) for _ in range(n_sess)]
        if first_acute is not None and start_missing[first_acute]:
            # keep ancillary anchoring unambiguous: later sessions are non-acute
            for k in range(first_acute + 1, n_sess):
                classes[k] = "Outpatient"
        died = rng.random() < 0.08
        for k in range(n_sess):
            sess_counter += 1
            sid = str(SESSION_BASE + sess_counter)
            svc = "Routine" if classes[k] in ("Outpatient", "ED") else str(
                rng.choice(["Routine", "LTM"], p=[0.5, 0.5]))
            dur = float(rng.uniform(1200, 3600)) if svc == "Routine" else float(rng.uniform(12, 72) * 3600)
            st, miss = starts[k], start_missing[k]
            md_start = site in METADATA_START_SITES
            dob = (anchor.normalize() - pd.Timedelta(days=int(round(age * 365.25)))) if md_start else pd.NaT
            row = {
                "SiteID": site, "InstituteID": site if md_start else None,
                "BDSPPatientID": None if site in BLANK_PATIENT_ID_SITES else str(pid),
                "BidsFolder": f"sub-{site}{pid}", "SessionID": sid,
                "EEGFolder": "ceeg" if svc == "LTM" else "eeg",
                "DurationInSeconds": round(dur, 1), "ServiceName": svc,
                "AgeAtVisit": round(age, 2) if rng.random() < 0.1 else np.nan,       # largely empty
                "AgeInDaysAtVisit": float(round(age * 365.25)),
                "SexDSC": sex if rng.random() < 0.3 else None,                         # often empty
                "DateOfBirth": dob,
                "DateOfDeath": (st + pd.Timedelta(days=int(rng.integers(1, 60))) if died and site in
                                BLANK_PATIENT_ID_SITES and k == n_sess - 1 else pd.NaT),
                # S-sites: blank in the real table (real start is reports_findings StartTime(EEG)); I0008/I0009 have it
                "StartTime": (pd.NaT if miss else st) if md_start else pd.NaT,
                "EndTime": (st + pd.Timedelta(seconds=dur)) if md_start else pd.NaT,
                "CreationTime": pd.NaT if miss else st - pd.Timedelta(minutes=int(rng.integers(0, 6))),
                "HasXLTEKAnnotations": str(bool(rng.random() < 0.5)),
                "HasPersystAnnotations": str(bool(rng.random() < 0.3)),
                "BDSPLastModifiedDTS": pd.Timestamp("2026-04-30 06:00:00"),
                "BidsFlag": "True",
                # analytic only: in NO real eeg_metadata header (PatientClass is derived from visit_occurrence on disk)
                "PatientClass": classes[k], "ReferralIndication": str(rng.choice(INDICATIONS)),
            }
            R["eeg_metadata"].append(_only_available("eeg_metadata", site, row))
            truth["no_start_sessions"] += int(miss)
            if not md_start:                                    # I0008/I0009 have no reports_findings file
                end = st + pd.Timedelta(seconds=dur)
                rf = {"BDSPPatientID": str(pid), "SessionID": sid,
                      schema.START_EEG: pd.NaT if miss else st, schema.END_EEG: end,
                      "AgeAtVisit": round(age, 2), "AgeInDaysAtVisit": float(round(age * 365.25)), "SexDSC": sex,
                      schema.SERVICE_EEG: svc, "SiteID": site,
                      "ReportCreationTime": end + pd.Timedelta(hours=float(rng.uniform(1, 30))),
                      "ReportEEGDateTime": st.normalize() if site == "I0002" else st,
                      "ReportProcedureDate": st.normalize(),
                      "ReportEncounterDTS": st - pd.Timedelta(hours=float(rng.uniform(0, 6))),
                      "ReportBeginDTS": st, "ReportExamEndDTS": end,
                      "ReportProcedureDSC": "EEG ROUTINE" if svc == "Routine" else "EEG LONG TERM MONITORING"}
                for f in FLAGS_ALL:
                    v = rng.random()
                    rf[f] = "1" if v < 0.08 else ("None" if v < 0.12 else None)
                R["reports_findings"].append(_only_available("reports_findings", site, rf))
            vc = 0 if site == "I0003" else VISIT_CONCEPT[classes[k]]     # concept ids can be zero-filled (rule 6)
            visit_counter += 1
            R["omop_visit_occurrence"].append({
                "person_id": pid, "visit_occurrence_id": visit_counter,
                "visit_start_datetime": st - pd.Timedelta(hours=float(rng.uniform(0, 6))),
                "visit_end_datetime": st + pd.Timedelta(hours=float(rng.uniform(6, 240))),
                "visit_concept_id": vc, "visit_type_concept_id": 0, "visit_source_value": classes[k],
                "admitted_from_concept_id": 0, "admitted_from_source_value": None,
                "discharged_to_concept_id": 0, "discharged_to_source_value": None})

        # --- cohort-level tables
        R["omop_person"].append({
            "person_id": pid, "gender_concept_id": 8532 if sex == "Female" else 8507,
            "year_of_birth": int(true_anchor.year - age), "birth_datetime": pd.NaT,
            "race_concept_id": 0, "ethnicity_concept_id": 0, "gender_source_value": sex[0],
            "race_source_value": None, "ethnicity_source_value": None})
        R["heedb_patients"].append({
            "SiteID": site, "BDSPPatientID": str(pid), "Sex": sex, "AgeAtVisitAvg": round(age, 2),
            "RaceAndEthnicity": None, "RaceAndEthnicityDSC": None, "VisitCount": n_sess, "HasEEG": "True", "HasReports": "True",
            "MatchedEEGReports": n_sess, "ICD10Count": int(rng.poisson(4)), "MedicationCount": int(rng.poisson(6))})
        R["icd10_neurology"].append({
            "BDSPPatientID": str(pid), "Cerebrovascular Diseases": "I63.9 G45.9" if rng.random() < 0.15 else None,
            "Cerebral Degeneration": "G30.9" if rng.random() < 0.05 else None, "SiteID": site,
            "SexDSC": sex, "VisitCount": n_sess, "AgeAtVisitAvg": round(age, 2)})
        R["medication_atc"].append({
            "BDSPPatientID": str(pid), "Nervous System Drugs": int(rng.poisson(2))})
        if died:
            dd = starts[-1] + pd.Timedelta(days=int(rng.integers(1, 60)))
            R["omop_death"].append({"person_id": pid, "death_datetime": dd, "death_date": dd.normalize(),
                                    "cause_source_value": None})
        for _ in range(int(rng.integers(0, 4))):
            R["omop_condition_occurrence"].append({
                "person_id": pid, "condition_start_datetime": _ts(anchor, rng.uniform(-24 * 30, 24 * 30)),
                "condition_source_value": str(rng.choice(ICD)).replace(".", ""), "condition_concept_id": 0})
        for _ in range(int(rng.integers(0, 3))):
            pt = _ts(anchor, rng.uniform(-72, 72))
            R["omop_procedure_occurrence"].append({
                "person_id": pid, "procedure_datetime": pt, "procedure_date": pt.normalize(),
                "procedure_concept_id": 0, "procedure_source_value": str(int(rng.integers(90000, 99999)))})
        if rng.random() < 0.1:
            ot = _ts(anchor, rng.uniform(0, 72))
            R["omop_observation"].append({
                "person_id": pid, "observation_concept_id": 0, "observation_datetime": ot,
                "observation_date": ot.normalize(), "observation_source_value": "GOAL OF CARE",
                "value_as_string": "full code"})
        if first_acute is None:
            continue
        t0 = starts[first_acute]

        p = {t: [] for t in ("omop_drug_exposure", "omop_measurement", "imaging", "omop_note", "omop_note_nlp")}
        # --- drug exposure: end datetime (administration record proxy) only for patients with a MAR feed
        has_mar = rng.random() < MAR_COVERAGE[site]
        for _ in range(int(rng.poisson(2.5))):
            name, _sed = DRUGS[int(rng.integers(len(DRUGS)))]
            order = _ts(t0, rng.uniform(-48, 12))
            admin = has_mar and rng.random() < 0.9
            if admin:
                start = order + pd.Timedelta(minutes=float(5 + rng.exponential(40)))
                end = start + pd.Timedelta(minutes=float(30 + rng.exponential(240)))
            else:
                start, end = order, pd.NaT
            p["omop_drug_exposure"].append({
                "person_id": pid, "drug_exposure_start_datetime": start, "drug_exposure_end_datetime": end,
                "drug_source_value": name, "quantity": round(float(rng.exponential(2)), 2), "drug_concept_id": 0,
                "drug_type_concept_id": DRUG_TYPE_ADMIN if admin else DRUG_TYPE_ORDER,
                "route_source_value": "IV" if "IV" in name or "INJ" in name else "PO",
                "drug_exposure_start_date": start.normalize(),
                "drug_exposure_end_date": pd.NaT if pd.isna(end) else end.normalize(), "visit_occurrence_id": None})
        # --- measurement: labs (no result-time column exists) and scores
        for _ in range(int(2 + rng.poisson(3))):
            col = _ts(t0, rng.uniform(-48, 24))
            name, unit = LABS[int(rng.integers(len(LABS)))]
            p["omop_measurement"].append({
                "person_id": pid, "measurement_datetime": col, "measurement_date": col.normalize(),
                "measurement_time": col.strftime("%H:%M:%S"),
                "measurement_source_value": name, "value_as_number": round(float(rng.normal(10, 4)), 2),
                "unit_source_value": unit, "measurement_concept_id": 0,
                "measurement_type_concept_id": MEAS_TYPE_LAB, "visit_occurrence_id": None})

        def score_row(tm):
            j = int(rng.choice(len(SCORES), p=[x[1] for x in SCORES]))
            nm, _, (lo, hi) = SCORES[j]
            return {"person_id": pid, "measurement_datetime": tm, "measurement_date": tm.normalize(),
                    "measurement_time": tm.strftime("%H:%M:%S"),
                    "measurement_source_value": nm, "value_as_number": float(rng.integers(lo, hi)),
                    "unit_source_value": None, "measurement_concept_id": 0,
                    "measurement_type_concept_id": MEAS_TYPE_EXAM, "visit_occurrence_id": None}
        if rng.random() < SCORE_NEAR[site]:
            for _ in range(int(rng.integers(1, 4))):
                p["omop_measurement"].append(score_row(_ts(t0, rng.uniform(-5.5, 5.5))))
        for _ in range(int(rng.integers(0, 4))):
            sgn = 1 if rng.random() < 0.5 else -1
            p["omop_measurement"].append(score_row(_ts(t0, sgn * rng.uniform(12, 72))))
        # --- imaging (ASSUMED table)
        for _ in range(int(rng.choice([0, 1, 2], p=[0.35, 0.45, 0.20]))):
            st = _ts(t0, rng.uniform(-72, 24))
            fin = st + pd.Timedelta(minutes=float(20 + rng.exponential(120)))
            if rng.random() < IMG_FINAL_MISSING[site]:
                fin = pd.NaT
            p["imaging"].append({"person_id": pid, "modality": str(rng.choice(MODALITIES)),
                                 "study_datetime": st, "report_final_datetime": fin})
        # --- notes (metadata only; no note_text)
        if rng.random() > 0.02:
            for _ in range(int(1 + rng.poisson(4))):
                note_counter += 1
                tm = _ts(t0, rng.uniform(-72, 72))
                if rng.random() < 0.02:
                    tm = pd.NaT
                ntid, ntname = NOTE_TYPES[int(rng.integers(len(NOTE_TYPES)))]
                p["omop_note"].append({
                    "note_id": note_counter, "person_id": pid, "note_date": pd.NaT if pd.isna(tm) else tm.normalize(),
                    "note_datetime": tm, "note_type_concept_id": ntid, "note_class_concept_id": 0,
                    "note_title": "SYNTHETIC " + ntname, "note_source_value": ntname, "visit_occurrence_id": None})
                if rng.random() < 0.1:
                    p["omop_note_nlp"].append({
                        "note_nlp_id": note_counter, "note_id": note_counter, "section_concept_id": 0,
                        "note_nlp_concept_id": 0, "nlp_system": "synthetic", "nlp_datetime": tm,
                        "term_exists": "Y", "term_temporal": None})

        # --- inconsistent date shift injected into one ancillary table
        if rng.random() < TABLE_SHIFT_RATE:
            tbl = str(rng.choice(SHIFTABLE))
            off = pd.Timedelta(days=int(rng.integers(60, 400)) * int(rng.choice([-1, 1])))
            for r in p[tbl]:
                for c in TIME_COLS[tbl]:
                    if not pd.isna(r[c]):
                        r[c] = r[c] + off
            _dates(p[tbl], tbl)
            truth["table_shift_ids"].append(pid)
            truth["table_shift_table"][pid] = tbl
        for t, rows in p.items():
            R[t] += rows

    R["omop_concept"] = [
        {"concept_id": c, "concept_name": n, "domain_id": d, "vocabulary_id": v, "standard_concept": "S",
         "concept_class_id": cc, "concept_code": str(c)}
        for c, n, d, v, cc in [(32037, "Intensive Care", "Visit", "Visit", "Visit"),
                               (9201, "Inpatient Visit", "Visit", "Visit", "Visit"),
                               (9202, "Outpatient Visit", "Visit", "Visit", "Visit"),
                               (9203, "Emergency Room Visit", "Visit", "Visit", "Visit"),
                               (8532, "FEMALE", "Gender", "Gender", "Gender"),
                               (8507, "MALE", "Gender", "Gender", "Gender")]]

    def df(rows, table):
        d = pd.DataFrame(rows, columns=schema.columns(table))
        return schema.coerce_types(table, d)

    tables = {t: df(rows, t) for t, rows in R.items()}
    truth["seed"] = seed
    return tables, truth


def write_tables(tables, outdir, truth=None, fmt: str = "auto"):
    paths = _write_tables(tables, outdir, fmt)
    manifest = {"synthetic": True, "seed": None if truth is None else truth["seed"],
                "n_patients": int(tables["omop_person"]["person_id"].nunique()),
                "n_injected_table_shift_patients": None if truth is None else len(truth["table_shift_ids"]),
                "note": "Entirely fake data. No real patients."}
    (Path(outdir) / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return paths
