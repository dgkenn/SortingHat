# HEEDB schema used by SortingHat tooling

**Authoritative source: `docs/heedb_schema_real.md`** (real table and column names found in the earlier research
code that read the BDSP bucket; status CONFIRMED / NAMED / UNKNOWN per column). `sortinghat/schema.py` is the
machine-readable copy and `sortinghat/data_io.py` (`TABLES`) holds the matching file/prefix locations. The public
BDSP page (bdsp.io/content/harvard-eeg-db) only lists the EEG metadata CSV; nothing here has been checked against
live data by this repo.

Provenance vocabulary in `schema.py`:

- **CONFIRMED**: read by code that ran against the real table.
- **NAMED**: appears only in an extractor list, constant or doc table. OMOP extractors skip absent columns
  silently, so NAMED does not prove the column exists.
- **ASSUMED**: placeholder; nothing in the source touches it. OMOP tables use standard OMOP CDM v5.4 names.

## Layout (what the synthetic generator emits and the audit reads)

| Table key | Location under the access point | Notes |
|---|---|---|
| `eeg_metadata` | `EEG/eeg-metadata/{SITE}_eeg_metadata_<release>.csv` | `DurationInSeconds`; `StartTime`/`EndTime` blank; `BDSPPatientID` can be blank (use `BidsFolder` = `sub-<SITE><PID>`); `AgeAtVisit`/`SexDSC` sparse |
| `reports_findings` | `EEG/HEEDB_Metadata/{SITE}_EEG__reports_findings.csv` | real start/end are `StartTime(EEG)` / `EndTime(EEG)`; label columns (`bs`, `gen slowing`, ...); join on (`BDSPPatientID`, `SessionID`) |
| `heedb_patients`, `icd10_neurology`, `medication_atc` | `EEG/HEEDB_Metadata/*.csv` | per-patient, wide, no timestamps |
| `omop_<table>` | `OMOP/Merged/<table>/*.parquet` | `person_id` = `int(BDSPPatientID)`; datetimes are text `YYYY-MM-DD HH:MM:SS[.ffffff]` |
| `imaging` | `Imaging/imaging_metadata/*.parquet` | ASSUMED location and columns |

Analytic mapping used by the audit (`sortinghat/audit/field_audit.py`):

| Audit field | Real source |
|---|---|
| EEG start `t0` | `reports_findings."StartTime(EEG)"` |
| age | `reports_findings.AgeAtVisit`, else `eeg_metadata.AgeAtVisit` |
| medication timing | `omop_drug_exposure` (`drug_source_value` regex for sedatives; `drug_exposure_end_datetime` present as the administration-record proxy) |
| lab collection / result time | `omop_measurement.measurement_datetime`; **no result-time column known**, so the row fails unless one is found |
| GCS / FOUR / RASS | `omop_measurement` rows whose `measurement_source_value` matches the score regex |
| notes | `omop_note.note_datetime` (ASSUMED) |
| imaging finalization | `imaging.report_final_datetime` (ASSUMED) |
| acute care flag | `eeg_metadata.PatientClass` (ASSUMED); if absent the audit uses all adults and says so |

## Remaining unknowns (settle with `--dry-run-schema`, names only)

1. **Imaging**: table location under `Imaging/`, modality, report finalization time.
2. **Notes**: whether OMOP `note` / `note_nlp` exist in `OMOP/Merged/`, and the note timestamp and type coding.
3. **PatientClass** (acute-care flag) and **ReferralIndication**: not seen in any table. Nearest proxies are
   `ServiceName` and `visit_occurrence.visit_concept_id` (NAMED, may be zero-filled).
4. **Lab result / verification time**: OMOP `measurement` has only `measurement_datetime` / `measurement_date`.
5. **Medication order versus administration time**: semantics of `drug_exposure_start_datetime` are not stated.
6. **OMOP `person` extras** (gender, birth date, race): the table was verified to match EEG patients but its
   columns were never read.
7. Other ICD-chapter and ATC-group columns in the wide tables; FOUR-score rows in `measurement`;
   `eeg_metadata.BDSPPatientID` blankness per release; hospital names behind site codes.

## Column list

Generated from `schema.py` (`python -c "from sortinghat import schema; print(schema.provenance_markdown())"`).
