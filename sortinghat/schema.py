"""Expected HEEDB layout: table and column names as the earlier research code read them from the REAL bucket.

Authoritative evidence is ``docs/heedb_schema_real.md`` (column names used by code that ran against the real
BDSP tables; nothing here was checked against live data by this repo). Provenance of each column:

  CONFIRMED - read by analysis code that ran against the real table (or stated as observed in its docs).
  NAMED     - appears only in an extractor request list, constant or doc table; OMOP extractors silently skip
              columns that do not exist, so NAMED does not prove existence.
  ASSUMED   - placeholder. Nothing in the source touches it (imaging finalization time, note timestamps,
              PatientClass, ReferralIndication, OMOP ``person`` extras, ``note_nlp``...). Where the table is
              OMOP the placeholder uses the standard OMOP CDM v5.4 column name. A human remaps these after
              ``python -m sortinghat.audit.field_audit --dry-run-schema`` shows what actually exists.

Table keys equal ``sortinghat.data_io.TABLES`` keys (CSV tables by their short name, OMOP tables as
``omop_<table>`` under ``OMOP/Merged/<table>/*.parquet``). Column names are exact, including spaces and
parentheses (``StartTime(EEG)``, ``gen slowing``).

Dtype vocabulary: ``str``, ``int``, ``float``, ``datetime`` (stored as text ``YYYY-MM-DD HH:MM:SS[.ffffff]``
in the real files; parse with ``parse_datetimes``).
"""

from __future__ import annotations

import pandas as pd

CONFIRMED = "CONFIRMED"
NAMED = "NAMED"
ASSUMED = "ASSUMED"
PROVENANCES = (CONFIRMED, NAMED, ASSUMED)

SITE_ID = "SiteID"
PATIENT_ID = "BDSPPatientID"          # CSV tables; OMOP ``person_id`` is int(BDSPPatientID)
OMOP_PERSON_ID = "person_id"

# Real column names that differ from earlier assumptions (kept so errors can point at them).
START_EEG = "StartTime(EEG)"
END_EEG = "EndTime(EEG)"
SERVICE_EEG = "ServiceName(EEG)"

# Label columns of reports_findings (cell asserted when non-empty and not the strings None/nan).
FINDING_FLAGS_CONFIRMED = ["bs", "gen slowing", "foc slowing", "seizure", "gpd", "lpd", "pdr", "low voltage",
                           "status", "wicket", "breach", "bets"]
FINDING_FLAGS_NAMED = ["normal", "abnormal", "spikes", "lrda", "grda", "uninterpretable", "bipd", "eses",
                       "cjd", "ppr", "diffuse Beta"]

C, N, A = CONFIRMED, NAMED, ASSUMED

# table -> list of (column, dtype, provenance, description)
SCHEMA: dict[str, list[tuple[str, str, str, str]]] = {
    # EEG/eeg-metadata/{SITE}_eeg_metadata_<release>.csv: one row per EEG session
    "eeg_metadata": [
        ("SiteID", "str", C, "Site code"),
        ("BDSPPatientID", "str", C, "Patient id; BLANK in S0001/S0002 on some releases, derive from BidsFolder"),
        ("BidsFolder", "str", C, "sub-<SITE><BDSPPatientID>"),
        ("SessionID", "str", C, "Session id; joins reports_findings.SessionID"),
        ("EEGFolder", "str", C, "Starts with 'ceeg' => task token cEEG in the EDF key"),
        ("DurationInSeconds", "float", C, "Recording duration (plural Seconds)"),
        ("ServiceName", "str", C, "Routine / LTM / EMU / OR"),
        ("AgeAtVisit", "float", C, "Largely empty; prefer reports_findings.AgeAtVisit"),
        ("SexDSC", "str", C, "Often empty"),
        ("DateOfDeath", "datetime", C, "Outcome column; only meaningful for S0001/S0002"),
        ("StartTime", "datetime", C, "BLANK in the real table; real start is reports_findings StartTime(EEG)"),
        ("EndTime", "datetime", C, "BLANK in the real table; real end is reports_findings EndTime(EEG)"),
        ("CreationTime", "datetime", N, "Public BDSP page; never read by the source code"),
        ("HasXLTEKAnnotations", "str", N, "Public BDSP page; never read"),
        ("HasPersystAnnotations", "str", N, "Public BDSP page; never read"),
        ("BDSPLastModifiedDTS", "datetime", N, "Public BDSP page; never read"),
        ("PatientClass", "str", A, "ASSUMED: ICU / Inpatient / ED / Outpatient (defines acute care); not seen"),
        ("ReferralIndication", "str", A, "ASSUMED: EEG referral indication category; not seen"),
    ],
    # EEG/HEEDB_Metadata/{SITE}_EEG__reports_findings.csv: one row per EEG report
    "reports_findings": (
        [("BDSPPatientID", "str", C, "Patient id"),
         ("SessionID", "str", C, "Joins eeg_metadata.SessionID"),
         (START_EEG, "datetime", C, "Real EEG start timestamp"),
         (END_EEG, "datetime", C, "Real EEG end timestamp"),
         ("AgeAtVisit", "float", C, "Age at the study"),
         ("SexDSC", "str", C, "Sex"),
         (SERVICE_EEG, "str", C, "OR / EMU / Routine / LTM")]
        + [(f, "str", C, "Finding label") for f in FINDING_FLAGS_CONFIRMED]
        + [(f, "str", N, "Finding label (constant/docstring only)") for f in FINDING_FLAGS_NAMED]),
    # EEG/HEEDB_Metadata/HEEDB_patients.csv
    "heedb_patients": [
        ("SiteID", "str", C, "Site code"),
        ("BDSPPatientID", "str", C, "Patient id"),
        ("Sex", "str", C, "Sex"),
        ("AgeAtVisitAvg", "float", C, "Mean age at visit"),
        ("Race", "str", N, ""), ("VisitCount", "int", N, ""), ("HasEEG", "str", N, ""),
        ("HasReports", "str", N, ""), ("MatchedEEGReports", "int", N, ""), ("ICD10Count", "int", N, ""),
        ("MedicationCount", "int", N, ""),
    ],
    # EEG/HEEDB_Metadata/HEEDB_ICD10_for_Neurology.csv: wide, one row per patient, one column per ICD chapter
    "icd10_neurology": [
        ("BDSPPatientID", "str", C, "Patient id"),
        ("Cerebrovascular Diseases", "str", C, "Code strings or counts; no timestamps"),
        ("Cerebral Degeneration", "str", C, "Code strings or counts; no timestamps"),
        ("SiteID", "str", N, ""), ("SexDSC", "str", N, ""), ("VisitCount", "int", N, ""),
        ("AgeAtVisitAvg", "float", N, ""),
    ],
    # EEG/HEEDB_Metadata/HEEDB_Medication_ATC.csv: wide, one row per patient
    "medication_atc": [
        ("BDSPPatientID", "str", C, "Patient id"),
        ("Nervous System Drugs", "str", C, "Code strings or counts; no timestamps"),
    ],
    # OMOP/Merged/<table>/*.parquet (OMOP CDM v5-style). person_id == int(BDSPPatientID)
    "omop_person": [
        ("person_id", "int", C, "Integer BDSPPatientID"),
        ("gender_concept_id", "int", A, "ASSUMED OMOP CDM v5.4 person extra; never read by the source"),
        ("year_of_birth", "int", A, "ASSUMED OMOP CDM v5.4 person extra"),
        ("birth_datetime", "datetime", A, "ASSUMED OMOP CDM v5.4 person extra"),
        ("race_concept_id", "int", A, "ASSUMED OMOP CDM v5.4 person extra"),
        ("ethnicity_concept_id", "int", A, "ASSUMED OMOP CDM v5.4 person extra"),
    ],
    "omop_visit_occurrence": [
        ("person_id", "int", C, "Integer BDSPPatientID"),
        ("visit_start_datetime", "datetime", N, "Single ~9 GB part"),
        ("visit_end_datetime", "datetime", N, ""),
        ("visit_concept_id", "int", N, "May be zero-filled (catalogue rule 6)"),
        ("discharge_to_concept_id", "int", N, ""),
        ("discharge_to_source_value", "str", N, ""),
        ("visit_source_value", "str", N, ""),
    ],
    "omop_condition_occurrence": [
        ("person_id", "int", C, "Integer BDSPPatientID"),
        ("condition_start_datetime", "datetime", C, ""),
        ("condition_source_value", "str", C, "ICD-9/ICD-10 strings"),
        ("condition_concept_id", "int", N, ""),
    ],
    "omop_drug_exposure": [
        ("person_id", "int", C, "Integer BDSPPatientID"),
        ("drug_exposure_start_datetime", "datetime", C, "Order vs administration semantics UNKNOWN"),
        ("drug_exposure_end_datetime", "datetime", C, "Vasopressor ends may be stamped at death"),
        ("drug_source_value", "str", C, "Free text drug name"),
        ("quantity", "float", C, ""),
        ("drug_concept_id", "int", N, ""),
    ],
    "omop_measurement": [
        ("person_id", "int", C, "Integer BDSPPatientID"),
        ("measurement_datetime", "datetime", C, "Collection-versus-result semantics UNKNOWN"),
        ("measurement_date", "datetime", C, "Fallback when datetime is empty"),
        ("measurement_source_value", "str", C, "Text name of lab / vital / score"),
        ("value_as_number", "float", C, ""),
        ("unit_source_value", "str", C, "Mixed units possible (F and C)"),
        ("measurement_concept_id", "int", N, ""),
    ],
    "omop_death": [
        ("person_id", "int", C, "Integer BDSPPatientID"),
        ("death_datetime", "datetime", C, "Incomplete: no row is not survival"),
        ("cause_source_value", "str", N, ""),
    ],
    "omop_procedure_occurrence": [
        ("person_id", "int", C, "Integer BDSPPatientID"),
        ("procedure_datetime", "datetime", C, ""),
        ("procedure_date", "datetime", C, ""),
        ("procedure_concept_id", "int", C, ""),
        ("procedure_source_value", "str", C, "Numeric billing codes, not names"),
    ],
    "omop_observation": [
        ("person_id", "int", C, "Integer BDSPPatientID"),
        ("observation_concept_id", "int", C, "Catalogue rule 6: 100 percent zero"),
        ("observation_datetime", "datetime", N, ""),
        ("observation_date", "datetime", N, ""),
        ("observation_source_value", "str", N, ""),
        ("value_as_string", "str", N, ""),
    ],
    "omop_concept": [
        ("concept_id", "int", C, ""), ("concept_name", "str", C, ""), ("domain_id", "str", C, ""),
        ("vocabulary_id", "str", C, ""), ("standard_concept", "str", C, ""),
        ("concept_class_id", "str", N, ""), ("concept_code", "str", N, ""),
    ],
    # ASSUMED: nothing in the source touches note / note_nlp. Standard OMOP CDM v5.4 names; note_text is
    # deliberately NOT requested (text is never loaded by the audit; CLAUDE.md rule 4).
    "omop_note": [
        ("note_id", "int", A, "ASSUMED OMOP CDM v5.4"),
        ("person_id", "int", A, "ASSUMED OMOP CDM v5.4"),
        ("note_date", "datetime", A, "ASSUMED OMOP CDM v5.4"),
        ("note_datetime", "datetime", A, "ASSUMED OMOP CDM v5.4: note timestamp"),
        ("note_type_concept_id", "int", A, "ASSUMED OMOP CDM v5.4"),
        ("note_class_concept_id", "int", A, "ASSUMED OMOP CDM v5.4"),
        ("note_title", "str", A, "ASSUMED OMOP CDM v5.4"),
        ("note_source_value", "str", A, "ASSUMED OMOP CDM v5.4"),
        ("visit_occurrence_id", "int", A, "ASSUMED OMOP CDM v5.4"),
    ],
    "omop_note_nlp": [
        ("note_nlp_id", "int", A, "ASSUMED OMOP CDM v5.4"),
        ("note_id", "int", A, "ASSUMED OMOP CDM v5.4"),
        ("section_concept_id", "int", A, "ASSUMED OMOP CDM v5.4"),
        ("note_nlp_concept_id", "int", A, "ASSUMED OMOP CDM v5.4"),
        ("nlp_system", "str", A, "ASSUMED OMOP CDM v5.4"),
        ("nlp_datetime", "datetime", A, "ASSUMED OMOP CDM v5.4"),
        ("term_exists", "str", A, "ASSUMED OMOP CDM v5.4"),
        ("term_temporal", "str", A, "ASSUMED OMOP CDM v5.4"),
    ],
    # ASSUMED: the access point has an Imaging/ prefix but no source script reads it. Table location and
    # every column below are placeholders to be remapped.
    "imaging": [
        ("person_id", "int", A, "ASSUMED placeholder (location and layout of Imaging/ unknown)"),
        ("modality", "str", A, "ASSUMED: CT head / MRI brain / CTA"),
        ("study_datetime", "datetime", A, "ASSUMED: acquisition time"),
        ("report_final_datetime", "datetime", A, "ASSUMED: report finalization time (the 0a field)"),
    ],
}

TABLE_NAMES = list(SCHEMA)

# Candidate names to try for a canonical analytic field when loading real data (ASSUMED / UNKNOWN fields).
COLUMN_ALIASES: dict[str, list[str]] = {
    "imaging.report_final_datetime": ["report_final_datetime", "finalized_datetime", "ReportFinalDTS"],
    "measurement.result_datetime": ["measurement_result_datetime", "result_datetime", "ResultDTS"],
}

# Fields still UNKNOWN (no column named anywhere in the source); documented in docs/heedb_schema.md.
UNKNOWN_FIELDS = [
    "lab result / verification time (OMOP measurement has only measurement_datetime / measurement_date)",
    "medication order time vs administration time (drug_exposure_start_datetime semantics)",
    "imaging table location, modality and report finalization time (Imaging/ prefix)",
    "note and note_nlp tables (existence, note_datetime, note type coding)",
    "PatientClass (acute-care flag) and ReferralIndication; nearest: ServiceName, visit_occurrence.visit_concept_id",
    "OMOP person extras (gender, birth date); other ICD-chapter / ATC-group wide columns; FOUR score rows",
]


def columns(table: str) -> list[str]:
    return [c for c, *_ in SCHEMA[table]]


def dtype_of(table: str, column: str) -> str:
    for c, t, *_ in SCHEMA[table]:
        if c == column:
            return t
    raise KeyError(f"{table}.{column}")


def datetime_cols(table: str) -> list[str]:
    return [c for c, t, *_ in SCHEMA[table] if t == "datetime"]


def columns_by_provenance(table: str, *provs: str) -> list[str]:
    return [c for c, _, p, _ in SCHEMA[table] if p in provs]


def parse_datetimes(s: pd.Series) -> pd.Series:
    """Parse ``YYYY-MM-DD[ HH:MM:SS[.ffffff]]`` text (mixed precision allowed); bad cells become NaT."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.astype("datetime64[us]")
    out = pd.to_datetime(s.astype("object").where(s.notna(), None), format="mixed", errors="coerce")
    return out.astype("datetime64[us]")       # one resolution so mixed-source frames subtract cleanly


def coerce_types(table: str, df: pd.DataFrame) -> pd.DataFrame:
    """Cast the columns present in ``df`` to the schema dtypes (text -> datetime/float/Int64)."""
    out = df.copy()
    for c, t, *_ in SCHEMA[table]:
        if c not in out.columns:
            continue
        if t == "datetime":
            out[c] = parse_datetimes(out[c])
        elif t == "float":
            out[c] = pd.to_numeric(out[c], errors="coerce")
        elif t == "int":
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    return out


def provenance_markdown() -> str:
    lines = ["| Table | Column | Type | Provenance | Description |", "|---|---|---|---|---|"]
    for t, cols in SCHEMA.items():
        for c, ty, prov, desc in cols:
            lines.append(f"| {t} | {c} | {ty} | {prov} | {desc} |")
    return "\n".join(lines)
