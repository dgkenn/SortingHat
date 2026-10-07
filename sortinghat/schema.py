"""Schema used by the synthetic generator and the field audit.

Provenance of each column is marked:
  DOCUMENTED - listed on the public BDSP Harvard EEG Database page
               (bdsp.io/content/harvard-eeg-db), EEG metadata CSV.
  ASSUMED    - the public page does not document column names (EHR branches
               data_Structured / data_Unstructured / data_Imaging are only
               described, not itemised). These names are plausible placeholders
               and MUST be remapped against the real tables (see
               ``COLUMN_ALIASES``) before running on restricted data.
"""

from __future__ import annotations

DOCUMENTED = "DOCUMENTED"
ASSUMED = "ASSUMED"

PATIENT_ID = "BDSPPatientID"
SITE_ID = "SiteID"

# table -> list of (column, dtype, provenance, description)
SCHEMA: dict[str, list[tuple[str, str, str, str]]] = {
    "eeg_metadata": [
        ("SiteID", "str", DOCUMENTED, "Hospital where the EEG was recorded"),
        ("BDSPPatientID", "str", DOCUMENTED, "Patient identifier"),
        ("BidsFolder", "str", DOCUMENTED, "Folder holding a patient's studies"),
        ("SessionID", "str", DOCUMENTED, "Study/session identifier"),
        ("CreationTime", "datetime", DOCUMENTED, "De-identified (date-shifted) creation time"),
        ("StartTime", "datetime", DOCUMENTED, "De-identified (date-shifted) EEG start"),
        ("EndTime", "datetime", DOCUMENTED, "De-identified (date-shifted) EEG end"),
        ("DurationInSecond", "float", DOCUMENTED, "Recording duration"),
        ("ServiceName", "str", DOCUMENTED, "Routine / LTM / EMU"),
        ("AgeAtVisit", "float", DOCUMENTED, "Age at the study"),
        ("SexDSC", "str", DOCUMENTED, "Patient-reported gender"),
        ("PatientClass", "str", ASSUMED, "ICU / Inpatient / ED / Outpatient (defines acute care)"),
        ("ReferralIndication", "str", ASSUMED, "EEG referral indication category"),
    ],
    "medications": [
        ("BDSPPatientID", "str", ASSUMED, "Patient identifier"),
        ("med_name", "str", ASSUMED, "Medication name"),
        ("med_class", "str", ASSUMED, "sedative / analgesic / antiseizure / other"),
        ("order_time", "datetime", ASSUMED, "Order time"),
        ("admin_time", "datetime", ASSUMED, "Administration (MAR) time; null if only ordered"),
    ],
    "labs": [
        ("BDSPPatientID", "str", ASSUMED, "Patient identifier"),
        ("lab_name", "str", ASSUMED, "Assay"),
        ("collect_time", "datetime", ASSUMED, "Specimen collection time"),
        ("result_time", "datetime", ASSUMED, "Result / verification time"),
    ],
    "imaging": [
        ("BDSPPatientID", "str", ASSUMED, "Patient identifier"),
        ("modality", "str", ASSUMED, "CT head / MRI brain / CTA"),
        ("study_time", "datetime", ASSUMED, "Acquisition time"),
        ("report_final_time", "datetime", ASSUMED, "Report finalization time"),
    ],
    "clinical_scores": [
        ("BDSPPatientID", "str", ASSUMED, "Patient identifier"),
        ("score_type", "str", ASSUMED, "GCS / FOUR / RASS"),
        ("score_value", "float", ASSUMED, "Score value"),
        ("score_time", "datetime", ASSUMED, "Documentation time"),
    ],
    "notes": [
        ("BDSPPatientID", "str", ASSUMED, "Patient identifier"),
        ("note_id", "str", ASSUMED, "Note identifier"),
        ("note_type", "str", ASSUMED, "Physician / nursing / EEG report / other"),
        ("note_time", "datetime", ASSUMED, "Note timestamp (text body not modelled)"),
    ],
}

TABLE_NAMES = list(SCHEMA)

# Candidate column names to try per canonical column when loading real data.
COLUMN_ALIASES: dict[str, list[str]] = {
    "report_final_time": ["report_final_time", "finalized_time", "ReportFinalDTS"],
    "admin_time": ["admin_time", "administration_time", "MAR_time"],
    "result_time": ["result_time", "verified_time", "ResultDTS"],
}


def datetime_cols(table: str) -> list[str]:
    return [c for c, t, _, _ in SCHEMA[table] if t == "datetime"]


def columns(table: str) -> list[str]:
    return [c for c, *_ in SCHEMA[table]]


def provenance_markdown() -> str:
    lines = ["| Table | Column | Type | Provenance | Description |", "|---|---|---|---|---|"]
    for t, cols in SCHEMA.items():
        for c, ty, prov, desc in cols:
            lines.append(f"| {t} | {c} | {ty} | {prov} | {desc} |")
    return "\n".join(lines)
