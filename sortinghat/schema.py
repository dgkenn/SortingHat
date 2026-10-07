"""HEEDB layout: table and column names, per-site variants, and provenance.

Evidence, in order of strength:

  1. The first real NAMES-ONLY dry run (2026-10-07, ``docs/research/heedb_schema_dryrun_2026-10-07.md``):
     CSV header lines and parquet footers of the real BDSP tables, no values, no rows.
  2. Column names read by earlier research code that ran against the real tables (``docs/heedb_schema_real.md``).

Provenance of each column:

  CONFIRMED - the name was SEEN in a real header (dry run), or is read by code that ran against the real table.
              A mapping is CONFIRMED when both sides are real header names. Semantics (units, order versus
              administration, ...) are still unread unless a description says otherwise.
  NAMED     - appears only in an extractor list, constant or doc table, and was NOT found in the real header.
  ASSUMED   - placeholder. Not present in any real header: ``PatientClass`` and ``ReferralIndication`` (analytic
              columns; PatientClass is DERIVED from ``omop_visit_occurrence``), and the ``imaging`` table
              (the real ``Imaging/`` prefix layout differs from the placeholder; see ``IMAGING_REAL_LAYOUT``).

Table keys equal ``sortinghat.data_io.TABLES`` keys (CSV tables by their short name, OMOP tables as
``omop_<table>`` under ``OMOP/Merged/<table>/*.parquet``). ``SCHEMA`` holds CANONICAL column names (the analytic
frame). The two per-site CSV tables are NOT uniform across sites: ``SITE_VARIANTS`` records each site's real
header and the mapping actual name -> canonical name; ``sortinghat.data_io.normalise_site_table`` applies it.
Column names are exact, including spaces and parentheses (``StartTime(EEG)``, ``gen slowing``).

Dtype vocabulary: ``str``, ``int``, ``float``, ``datetime`` (stored as text ``YYYY-MM-DD HH:MM:SS[.ffffff]``
in the real files; parse with ``parse_datetimes``).
"""

from __future__ import annotations

from dataclasses import dataclass

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

ICD_CHAPTER_COLUMNS = ["Behavioral/Cognitive Syndromes", "Cerebral Lobe Dysfunction", "Cranial Nerve Disorders",
                       "Developmental Delay", "Genetic Disorders", "Headache Disorders", "Infections",
                       "Intracranial And Spinal Tumors", "Motor Neuron Diseases",
                       "Movement And Cerebellar Disorders", "Muscular Dystrophies And Other Myopathies",
                       "Other Neurological Disorders", "Peripheral Nervous System Disorders", "Seizure Disorders",
                       "Sleep Disorders", "Miscellaneous"]
ATC_GROUP_COLUMNS = ["Alimentary Tract And Metabolism Drugs", "Antiinfectives For Systemic Use",
                     "Antineoplastic And Immunomodulating Agents", "Antiparasitic Products, Insecticides And Repellents",
                     "Blood And Blood Forming Organ Drugs", "Cardiovascular System Drugs", "Dermatologicals",
                     "Genito Urinary System And Sex Hormones", "Musculo-Skeletal System Drugs",
                     "Respiratory System Drugs", "Sensory Organ Drugs",
                     "Systemic Hormonal Preparations, Excl. Sex Hormones And Insulins", "Various Drug Classes In Atc"]

# Label columns of reports_findings (cell asserted when non-empty and not the strings None/nan).
# The first 12 were read by code that ran; the next 11 were only named there. ALL are present in the real
# reports_findings headers of every site that has the file (dry run 2026-10-07). The last 16 are sleep / syndrome
# labels seen in the headers only (never used by the source code).
FINDING_FLAGS_CONFIRMED = ["bs", "gen slowing", "foc slowing", "seizure", "gpd", "lpd", "pdr", "low voltage",
                           "status", "wicket", "breach", "bets"]
FINDING_FLAGS_NAMED = ["normal", "abnormal", "spikes", "lrda", "grda", "uninterpretable", "bipd", "eses",
                       "cjd", "ppr", "diffuse Beta"]
FINDING_FLAGS_SLEEP_SYNDROME = ["spindles", "vertex wave", "k_complexes", "posts", "awake", "n1", "n2", "dravet",
                                "jeavons", "sunflower", "wham", "angelman", "fold", "jae", "jme", "bects"]
FINDING_FLAGS_ALL = FINDING_FLAGS_CONFIRMED + FINDING_FLAGS_NAMED + FINDING_FLAGS_SLEEP_SYNDROME

C, N, A = CONFIRMED, NAMED, ASSUMED

# table -> list of (CANONICAL column, dtype, provenance, description)
SCHEMA: dict[str, list[tuple[str, str, str, str]]] = {
    # EEG/eeg-metadata/{SITE}_eeg_metadata_<release>.csv: one row per EEG session. The header differs per site
    # (SITE_VARIANTS); this is the canonical (union) frame. "Sites" below are the sites whose real header has it.
    "eeg_metadata": [
        ("SiteID", "str", C, "Site code. In the header of S0001/S0002/I0002/I0003; ABSENT at I0008/I0009 (filled from the file name)"),
        ("InstituteID", "str", C, "I0008/I0009 only: the header's site/institute identifier (value format unread)"),
        ("BDSPPatientID", "str", C, "Patient id; BLANK in S0001/S0002 on some releases, derive from BidsFolder"),
        ("BidsFolder", "str", C, "sub-<SITE><BDSPPatientID>; present at every site"),
        ("SessionID", "str", C, "Session id; joins reports_findings.SessionID"),
        ("EEGFolder", "str", C, "S0001/S0002 only (NOT at I-sites; the dry run found BidsFolder there). 'ceeg' prefix => task token cEEG"),
        ("DurationInSeconds", "float", C, "Recording duration, seconds. I0002/I0003/S-sites; I0008/I0009 call it RecordingDuration (unit not stated by the name)"),
        ("ServiceName", "str", C, "Routine / LTM / EMU / OR. S-sites and I0003 only (absent at I0002, I0008, I0009)"),
        ("AgeAtVisit", "float", C, "Largely empty; prefer reports_findings.AgeAtVisit. Absent at I0008/I0009"),
        ("AgeInDaysAtVisit", "float", C, "I0003 only (also in its reports_findings); age in days"),
        ("SexDSC", "str", C, "Often empty"),
        ("DateOfBirth", "datetime", C, "I0008/I0009 only; with StartTime gives the age there (no AgeAtVisit column)"),
        ("DateOfDeath", "datetime", C, "Outcome column; S0001/S0002 only (absent at every I-site)"),
        ("StartTime", "datetime", C, "BLANK in S0001/S0002 (real start: reports_findings StartTime(EEG)); real start at I0008/I0009 (header StartDateTime)"),
        ("EndTime", "datetime", C, "BLANK in S0001/S0002; I0008/I0009 header name is EndDateTime"),
        ("CreationTime", "datetime", C, "S-sites, I0002, I0003; absent at I0008/I0009"),
        ("HasXLTEKAnnotations", "str", C, "Present at every site"),
        ("HasPersystAnnotations", "str", C, "S-sites and I0003 only"),
        ("BDSPLastModifiedDTS", "datetime", C, "I0002, I0003, I0008, I0009; absent at S0001/S0002"),
        ("BidsFlag", "str", C, "S-sites, I0008, I0009 (absent at I0002, I0003); semantics unread"),
        ("PatientClass", "str", A, "ASSUMED / DERIVED: in NO real eeg_metadata header. Derive from omop_visit_occurrence "
                                  "(field_audit.derive_patient_class); ICU / Inpatient / ED / Outpatient"),
        ("ReferralIndication", "str", A, "ASSUMED: in NO real header at any site (nearest: reports_findings "
                                         "ProcedureDSC(Reports) at S-sites, EEG type not clinical indication)"),
    ],
    # EEG/HEEDB_Metadata/{SITE}_EEG__reports_findings.csv: one row per EEG report. NO file for I0008/I0009.
    "reports_findings": (
        [("BDSPPatientID", "str", C, "Patient id"),
         ("SessionID", "str", C, "Joins eeg_metadata.SessionID"),
         (START_EEG, "datetime", C, "Real EEG start timestamp (every site that has the file)"),
         (END_EEG, "datetime", C, "Real EEG end timestamp"),
         ("AgeAtVisit", "float", C, "Age at the study"),
         ("AgeInDaysAtVisit", "float", C, "I0003 only"),
         ("SexDSC", "str", C, "Sex"),
         (SERVICE_EEG, "str", C, "OR / EMU / Routine / LTM. S-sites only (absent at I0002, I0003)"),
         ("SiteID", "str", C, "S-sites only"),
         ("ReportCreationTime", "datetime", C, "header CreationTime(EEG); every site that has the file"),
         ("ReportEEGDateTime", "datetime", C, "I0003 header EEGDateTime(Reports); I0002 header EEGDate(Reports) (date only)"),
         ("ReportProcedureDate", "datetime", C, "I0003 only; header ProcedureDate(Reports)"),
         ("ReportEncounterDTS", "datetime", C, "S-sites only; header EncounterDTS(Reports); encounter timestamp"),
         ("ReportBeginDTS", "datetime", C, "S-sites only; header BeginDTS(Reports); exam begin (possible start-time fallback)"),
         ("ReportExamEndDTS", "datetime", C, "S-sites only; header ExamEndDTS(Reports); exam end"),
         ("ReportProcedureDSC", "str", C, "S-sites only; header ProcedureDSC(Reports); procedure description (EEG type)")]
        + [(f, "str", C, "Finding label (read by code that ran)") for f in FINDING_FLAGS_CONFIRMED]
        + [(f, "str", C, "Finding label (named in code; now seen in the real headers)") for f in FINDING_FLAGS_NAMED]
        + [(f, "str", C, "Sleep / syndrome label (seen in the real headers only)")
           for f in FINDING_FLAGS_SLEEP_SYNDROME]),
    # EEG/HEEDB_Metadata/HEEDB_patients.csv
    "heedb_patients": [
        ("SiteID", "str", C, "Site code"),
        ("BDSPPatientID", "str", C, "Patient id"),
        ("Sex", "str", C, "Sex"),
        ("AgeAtVisitAvg", "float", C, "Mean age at visit"),
        ("RaceAndEthnicity", "str", C, "Real header (replaces the NAMED 'Race', which does not exist)"),
        ("RaceAndEthnicityDSC", "str", C, "Real header"),
        ("VisitCount", "int", C, ""), ("HasEEG", "str", C, ""),
        ("HasReports", "str", C, ""), ("MatchedEEGReports", "int", C, ""), ("ICD10Count", "int", C, ""),
        ("MedicationCount", "int", C, ""),
    ],
    # EEG/HEEDB_Metadata/HEEDB_ICD10_for_Neurology.csv: wide, one row per patient, one column per ICD chapter
    "icd10_neurology": (
        [("BDSPPatientID", "str", C, "Patient id"),
         ("Cerebrovascular Diseases", "str", C, "Code strings or counts; no timestamps"),
         ("Cerebral Degeneration", "str", C, "Code strings or counts; no timestamps"),
         ("SiteID", "str", C, ""), ("SexDSC", "str", C, ""), ("VisitCount", "int", C, ""),
         ("AgeAtVisitAvg", "float", C, "")]
        + [(c, "str", C, "ICD chapter column (seen in the real header)") for c in ICD_CHAPTER_COLUMNS]),
    # EEG/HEEDB_Metadata/HEEDB_Medication_ATC.csv: wide, one row per patient
    "medication_atc": (
        [("BDSPPatientID", "str", C, "Patient id"),
         ("Nervous System Drugs", "str", C, "Code strings or counts; no timestamps"),
         ("SiteID", "str", C, ""), ("SexDSC", "str", C, ""), ("VisitCount", "int", C, ""),
         ("AgeAtVisitAvg", "float", C, "")]
        + [(c, "str", C, "ATC group column (seen in the real header)") for c in ATC_GROUP_COLUMNS]),
    # OMOP/Merged/<table>/*.parquet (OMOP CDM v5.4-style). person_id == int(BDSPPatientID)
    "omop_person": [
        ("person_id", "int", C, "Integer BDSPPatientID"),
        ("gender_concept_id", "int", C, "In the real footer (concept ids may be zero-filled)"),
        ("year_of_birth", "int", C, "In the real footer"),
        ("birth_datetime", "datetime", C, "In the real footer"),
        ("race_concept_id", "int", C, "In the real footer"),
        ("ethnicity_concept_id", "int", C, "In the real footer"),
        ("gender_source_value", "str", C, "In the real footer; text fallback if gender_concept_id is zero-filled"),
        ("race_source_value", "str", C, "In the real footer"),
        ("ethnicity_source_value", "str", C, "In the real footer"),
    ],
    "omop_visit_occurrence": [
        ("person_id", "int", C, "Integer BDSPPatientID"),
        ("visit_occurrence_id", "int", C, "Join key to note / measurement / drug_exposure visit_occurrence_id"),
        ("visit_start_datetime", "datetime", C, "Single ~9 GB part"),
        ("visit_end_datetime", "datetime", C, ""),
        ("visit_concept_id", "int", C, "May be zero-filled (catalogue rule 6); the PatientClass source"),
        ("visit_type_concept_id", "int", C, ""),
        ("visit_source_value", "str", C, "Text fallback for PatientClass"),
        ("admitted_from_concept_id", "int", C, "Real footer"),
        ("admitted_from_source_value", "str", C, "Real footer (ED admit vs transfer)"),
        ("discharged_to_concept_id", "int", C, "REAL name (the NAMED 'discharge_to_concept_id' does not exist)"),
        ("discharged_to_source_value", "str", C, "REAL name (the NAMED 'discharge_to_source_value' does not exist)"),
    ],
    "omop_condition_occurrence": [
        ("person_id", "int", C, "Integer BDSPPatientID"),
        ("condition_start_datetime", "datetime", C, ""),
        ("condition_source_value", "str", C, "ICD-9/ICD-10 strings"),
        ("condition_concept_id", "int", C, ""),
    ],
    "omop_drug_exposure": [
        ("person_id", "int", C, "Integer BDSPPatientID"),
        ("drug_exposure_start_datetime", "datetime", C, "Order vs administration semantics UNKNOWN from names alone"),
        ("drug_exposure_end_datetime", "datetime", C, "Vasopressor ends may be stamped at death"),
        ("drug_source_value", "str", C, "Free text drug name"),
        ("quantity", "float", C, ""),
        ("drug_concept_id", "int", C, ""),
        ("drug_type_concept_id", "int", C, "Real footer. OMOP 'drug type' = provenance of the record (prescription "
                                            "written vs administration record): THE column that can settle "
                                            "order-vs-administration (needs a concept-name join; values unread)"),
        ("route_source_value", "str", C, "Real footer; route text (IV vs PO)"),
        ("drug_exposure_start_date", "datetime", C, "Real footer; date twin of the start datetime"),
        ("drug_exposure_end_date", "datetime", C, "Real footer; date twin of the end datetime"),
        ("visit_occurrence_id", "int", C, "Real footer"),
    ],
    "omop_measurement": [
        ("person_id", "int", C, "Integer BDSPPatientID"),
        ("measurement_datetime", "datetime", C, "Collection-versus-result semantics UNKNOWN (no result-time column exists)"),
        ("measurement_date", "datetime", C, "Fallback when datetime is empty"),
        ("measurement_time", "str", C, "Real footer; time-of-day twin (pair with measurement_date if datetime is empty)"),
        ("measurement_source_value", "str", C, "Text name of lab / vital / score"),
        ("value_as_number", "float", C, ""),
        ("unit_source_value", "str", C, "Mixed units possible (F and C)"),
        ("measurement_concept_id", "int", C, ""),
        ("measurement_type_concept_id", "int", C, "Real footer; record provenance (lab result vs vital vs exam)"),
        ("visit_occurrence_id", "int", C, "Real footer"),
    ],
    "omop_death": [
        ("person_id", "int", C, "Integer BDSPPatientID"),
        ("death_datetime", "datetime", C, "Incomplete: no row is not survival"),
        ("death_date", "datetime", C, "Real footer; date fallback"),
        ("cause_source_value", "str", C, ""),
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
        ("observation_datetime", "datetime", C, ""),
        ("observation_date", "datetime", C, ""),
        ("observation_source_value", "str", C, ""),
        ("value_as_string", "str", C, ""),
    ],
    "omop_concept": [
        ("concept_id", "int", C, ""), ("concept_name", "str", C, ""), ("domain_id", "str", C, ""),
        ("vocabulary_id", "str", C, ""), ("standard_concept", "str", C, ""),
        ("concept_class_id", "str", C, ""), ("concept_code", "str", C, ""),
    ],
    # OMOP note / note_nlp EXIST (all columns below are in the real footers). ``note_text`` is in the real note table
    # but is deliberately NOT in this schema, so it is never requested (CLAUDE.md rule 4).
    "omop_note": [
        ("note_id", "int", C, "Real footer"),
        ("person_id", "int", C, "Real footer"),
        ("note_date", "datetime", C, "Real footer"),
        ("note_datetime", "datetime", C, "Real footer: THE note timestamp (authoring vs service time unread)"),
        ("note_type_concept_id", "int", C, "Real footer: note type coding (may be zero-filled)"),
        ("note_class_concept_id", "int", C, "Real footer"),
        ("note_title", "str", C, "Real footer"),
        ("note_source_value", "str", C, "Real footer: text note type"),
        ("visit_occurrence_id", "int", C, "Real footer"),
    ],
    "omop_note_nlp": [
        ("note_nlp_id", "int", C, "Real footer"),
        ("note_id", "int", C, "Real footer"),
        ("section_concept_id", "int", C, "Real footer"),
        ("note_nlp_concept_id", "int", C, "Real footer"),
        ("nlp_system", "str", C, "Real footer"),
        ("nlp_datetime", "datetime", C, "Real footer"),
        ("term_exists", "str", C, "Real footer"),
        ("term_temporal", "str", C, "Real footer"),
    ],
    # ASSUMED: imaging is NOT at Imaging/imaging_metadata/. The real layout is Imaging/<SITE>/{BIDS,Clinical,Non-BIDS}/
    # for sites I0001 and I0004 only (IMAGING_REAL_LAYOUT); no table below the site level has been listed, so every
    # column here is a placeholder to be remapped after a names-only look inside Imaging/<SITE>/Clinical/.
    "imaging": [
        ("person_id", "int", A, "ASSUMED placeholder (table layout under Imaging/<SITE>/Clinical/ unknown)"),
        ("modality", "str", A, "ASSUMED: CT head / MRI brain / CTA"),
        ("study_datetime", "datetime", A, "ASSUMED: acquisition time"),
        ("report_final_datetime", "datetime", A, "ASSUMED: report finalization time (the 0a field)"),
    ],
}

TABLE_NAMES = list(SCHEMA)

# Real Imaging/ layout from prefix NAMES only (dry run 2026-10-07): sites with an Imaging/<SITE>/ prefix and the
# sub-prefixes below each. NONE of these sites has an eeg-metadata CSV in the six HEEDB EEG sites (I0002, I0003,
# I0008, I0009, S0001, S0002). There is no BIND/ prefix at the access point root.
IMAGING_REAL_LAYOUT = {"sites": ("I0001", "I0004"), "subprefixes": ("BIDS", "Clinical", "Non-BIDS")}


# ---------------------------------------------------------------------------------------------------------
# Per-site variants of the two CSV tables (REAL headers, dry run 2026-10-07; every name below was SEEN)
# ---------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SiteVariant:
    """One site group's real headers. ``eeg_metadata`` / ``reports_findings`` map ACTUAL header name -> CANONICAL
    name (a mapping to ``None`` is a column that exists but is deliberately never loaded or mapped).
    ``reports_findings=None`` means the site has no reports_findings file (I0008, I0009)."""
    key: str
    sites: tuple[str, ...]
    eeg_metadata: dict
    reports_findings: dict | None
    note: str = ""


def _ident(names) -> dict:
    return {n: n for n in names}


_RF_BASE = _ident(["BDSPPatientID", "SessionID", START_EEG, END_EEG, "AgeAtVisit", "SexDSC"]
                  + FINDING_FLAGS_CONFIRMED + FINDING_FLAGS_NAMED + FINDING_FLAGS_SLEEP_SYNDROME)

_S_EEG = _ident(["SiteID", "BDSPPatientID", "BidsFolder", "SessionID", "EEGFolder", "DurationInSeconds",
                 "ServiceName", "AgeAtVisit", "SexDSC", "DateOfDeath", "StartTime", "EndTime", "CreationTime",
                 "HasXLTEKAnnotations", "HasPersystAnnotations", "BidsFlag"])
_I0002_EEG = _ident(["SiteID", "BDSPPatientID", "BidsFolder", "SessionID", "DurationInSeconds", "AgeAtVisit",
                     "SexDSC", "StartTime", "EndTime", "CreationTime", "HasXLTEKAnnotations", "BDSPLastModifiedDTS"])
_I0003_EEG = _ident(["SiteID", "BDSPPatientID", "BidsFolder", "SessionID", "DurationInSeconds", "ServiceName",
                     "AgeAtVisit", "AgeInDaysAtVisit", "SexDSC", "StartTime", "EndTime", "CreationTime",
                     "HasXLTEKAnnotations", "HasPersystAnnotations", "BDSPLastModifiedDTS"])
_I0008_EEG = {**_ident(["InstituteID", "BDSPPatientID", "BidsFolder", "SessionID", "SexDSC", "DateOfBirth",
                        "HasXLTEKAnnotations", "BDSPLastModifiedDTS", "BidsFlag"]),
              "StartDateTime": "StartTime", "EndDateTime": "EndTime", "RecordingDuration": "DurationInSeconds"}

SITE_VARIANTS: dict[str, SiteVariant] = {v.key: v for v in (
    SiteVariant("S0001/S0002", ("S0001", "S0002"), _S_EEG,
                {**_RF_BASE, "SiteID": "SiteID", SERVICE_EEG: SERVICE_EEG, "CreationTime(EEG)": "ReportCreationTime",
                 "ProcedureDSC(Reports)": "ReportProcedureDSC", "EncounterDTS(Reports)": "ReportEncounterDTS",
                 "BeginDTS(Reports)": "ReportBeginDTS", "ExamEndDTS(Reports)": "ReportExamEndDTS",
                 "DeidentifiedName(Reports)": None},
                "BDSPPatientID blank in some releases (derive from BidsFolder). eeg_metadata StartTime/EndTime blank."),
    SiteVariant("I0002", ("I0002",), _I0002_EEG,
                {**_RF_BASE, "CreationTime(EEG)": "ReportCreationTime", "EEGDate(Reports)": "ReportEEGDateTime",
                 "DeidentifiedName(Reports)": None},
                "No ServiceName, no DateOfDeath, no EEGFolder; reports_findings has no ServiceName(EEG)."),
    SiteVariant("I0003", ("I0003",), _I0003_EEG,
                {**_RF_BASE, "AgeInDaysAtVisit": "AgeInDaysAtVisit", "CreationTime(EEG)": "ReportCreationTime",
                 "EEGDateTime(Reports)": "ReportEEGDateTime", "ProcedureDate(Reports)": "ReportProcedureDate",
                 "DeidentifiedName(Reports)": None, "N1": None},
                "No DateOfDeath, no EEGFolder. reports_findings carries both 'n1' and 'N1' (N1 is not mapped)."),
    SiteVariant("I0008/I0009", ("I0008", "I0009"), _I0008_EEG, None,
                "No SiteID (InstituteID), no AgeAtVisit (DateOfBirth), no ServiceName, no CreationTime, "
                "StartDateTime/EndDateTime/RecordingDuration; NO reports_findings file."),
)}


def variant_for(site: str | None) -> SiteVariant | None:
    for v in SITE_VARIANTS.values():
        if site in v.sites:
            return v
    return None


def _variant_map(table: str, site: str | None) -> dict | None:
    """actual -> canonical for a known site; ``None`` when the site is unknown (no variant) ."""
    v = variant_for(site)
    if v is None or table not in ("eeg_metadata", "reports_findings"):
        return None
    return v.eeg_metadata if table == "eeg_metadata" else (v.reports_findings if v.reports_findings is not None else {})


def never_load(table: str, site: str | None) -> list[str]:
    """Header names that exist but must not be read (name-like text: ``DeidentifiedName(Reports)``) or are duplicates."""
    m = _variant_map(table, site) or {}
    return [a for a, c in m.items() if c is None]


def expected_columns(table: str, site: str | None = None) -> list[tuple[str, str, str, str]]:
    """Expected columns in ACTUAL header-name space as ``(name, dtype, provenance, description)``.

    Known site: its real header (so a dry run reports drift, not variant differences). Unknown site on a
    per-site table: the union of every variant's header names. Other tables: ``SCHEMA``."""
    if table not in ("eeg_metadata", "reports_findings"):
        return SCHEMA[table]
    canon = {c: (t, p, d) for c, t, p, d in SCHEMA[table]}
    m = _variant_map(table, site)
    if m is None:
        m = {}
        for v in SITE_VARIANTS.values():
            for a, c in (v.eeg_metadata if table == "eeg_metadata" else (v.reports_findings or {})).items():
                m.setdefault(a, c)
    out = []
    for a, c in m.items():
        if c is None:
            out.append((a, "str", CONFIRMED, "present in the real header; deliberately not loaded/mapped"))
        else:
            t, p, d = canon[c]
            out.append((a, t, p, d + ("" if a == c else f" [header name {a!r} -> canonical {c!r}]")))
    return out


def canonical_to_actual(table: str, site: str) -> dict:
    """canonical -> actual header name for a known site (inverse of the variant mapping; skipped names omitted)."""
    return {c: a for a, c in (_variant_map(table, site) or {}).items() if c is not None}


# Candidate names to try for a canonical analytic field when loading real data (ASSUMED / UNKNOWN fields).
COLUMN_ALIASES: dict[str, list[str]] = {
    "imaging.report_final_datetime": ["report_final_datetime", "finalized_datetime", "ReportFinalDTS"],
    "measurement.result_datetime": ["measurement_result_datetime", "result_datetime", "ResultDTS"],
}

# Real columns that fill the roles the audit used to mark UNKNOWN (names only; semantics still unread).
ROLE_COLUMNS: dict[str, str] = {
    "note timestamp": "omop_note.note_datetime (note_date twin); note type: note_type_concept_id, note_source_value, note_title",
    "measurement time": "omop_measurement.measurement_datetime / measurement_date / measurement_time (time-of-day twin); no result-time column",
    "order vs administration": "omop_drug_exposure.drug_type_concept_id (+ route_source_value, drug_exposure_end_datetime); semantics need a concept join",
    "patient class": "DERIVED from omop_visit_occurrence.visit_concept_id / visit_source_value (+ admitted_from_*) by time overlap",
    "referral / indication": "NONE (nearest: reports_findings.ProcedureDSC(Reports) at S-sites, an EEG-type description)",
    "EEG start fallbacks": "reports_findings ReportBeginDTS (S-sites), ReportEEGDateTime (I0002 date only, I0003); eeg_metadata StartTime at I0008/I0009",
    "age at visit": "reports_findings.AgeAtVisit; I0003 AgeInDaysAtVisit; I0008/I0009 DateOfBirth with StartTime",
}

# Fields still UNKNOWN after the names-only dry run; documented in docs/heedb_schema.md.
UNKNOWN_FIELDS = [
    "lab result / verification time: NO result-time column in omop_measurement (only measurement_datetime / "
    "measurement_date / measurement_time)",
    "medication order time vs administration time: drug_exposure_start_datetime semantics unread; "
    "drug_type_concept_id is the discriminating column (values unread)",
    "imaging table contents: Imaging/<SITE>/{BIDS,Clinical,Non-BIDS}/ for I0001 and I0004 only (none of the six "
    "EEG-metadata sites); modality and report finalization columns not located",
    "PatientClass (acute-care flag): no column in any eeg_metadata header; derive from omop_visit_occurrence "
    "(visit_concept_id may be zero-filled)",
    "ReferralIndication: no column in any real header",
    "units of RecordingDuration (I0008/I0009), semantics of BidsFlag, note_datetime (authoring vs service time)",
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
