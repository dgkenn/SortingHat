# HEEDB schema used by SortingHat tooling

**Authoritative source: `docs/heedb_schema_real.md`** (real table and column names found in the earlier research
code that read the BDSP bucket; status CONFIRMED / NAMED / UNKNOWN per column). `sortinghat/schema.py` is the
machine-readable copy and `sortinghat/data_io.py` (`TABLES`) holds the matching file/prefix locations. The public
BDSP page (bdsp.io/content/harvard-eeg-db) only lists the EEG metadata CSV. Table and column NAMES were checked
against the real access point on 2026-10-07 (names-only dry run); no values have ever been read by this repo.

Provenance vocabulary in `schema.py`:

- **CONFIRMED**: the name was SEEN in a real header (first names-only dry run, 2026-10-07,
  `docs/research/heedb_schema_dryrun_2026-10-07.md`) or is read by code that ran against the real table. Units,
  semantics and fill rates are still unread.
- **NAMED**: only in an extractor list, constant or doc and NOT found in the header. (None remain after the dry run.)
- **ASSUMED**: placeholder, in no real header: `PatientClass` and `ReferralIndication` (analytic columns) and the
  `imaging` table.

## Layout (what the synthetic generator emits and the audit reads)

| Table key | Location under the access point | Notes |
|---|---|---|
| `eeg_metadata` | `EEG/eeg-metadata/{SITE}_eeg_metadata_<release>.csv` | **four per-site header variants** (below). `BidsFolder` = `sub-<SITE><PID>` is the one universal key; `BDSPPatientID` can be blank |
| `reports_findings` | `EEG/HEEDB_Metadata/{SITE}_EEG__reports_findings.csv` | **absent for I0008 and I0009**; real start/end are `StartTime(EEG)` / `EndTime(EEG)`; 39 label columns; join on (`BDSPPatientID`, `SessionID`) |
| `heedb_patients`, `icd10_neurology`, `medication_atc` | `EEG/HEEDB_Metadata/*.csv` | per-patient, wide, no timestamps |
| `omop_<table>` | `OMOP/Merged/<table>/*.parquet` | `person_id` = `int(BDSPPatientID)`; datetimes are text `YYYY-MM-DD HH:MM:SS[.ffffff]`; `note` / `note_nlp` exist |
| `imaging` | placeholder `Imaging/imaging_metadata/*.parquet` (does NOT exist) | real prefixes: `Imaging/{I0001,I0004}/{BIDS,Clinical,Non-BIDS}/`; columns ASSUMED |

Per-site variants of `eeg_metadata` (`schema.SITE_VARIANTS`; the normaliser `data_io.normalise_site_table` maps each to canonical names):

| Variant | Sites | Differences |
|---|---|---|
| S | S0001, S0002 | has `EEGFolder`, `DateOfDeath`, `ServiceName`, `HasPersystAnnotations`, `BidsFlag`; no `BDSPLastModifiedDTS`; metadata `StartTime`/`EndTime` blank; reports_findings has `ServiceName(EEG)`, `BeginDTS/ExamEndDTS/EncounterDTS/ProcedureDSC(Reports)` |
| I0002 | I0002 | no `ServiceName`, `DateOfDeath`, `EEGFolder`, `HasPersystAnnotations`; reports_findings has `EEGDate(Reports)` |
| I0003 | I0003 | adds `AgeInDaysAtVisit`; no `DateOfDeath`, `EEGFolder`; reports_findings has `EEGDateTime/ProcedureDate(Reports)` and `N1` |
| I0008/I0009 | I0008, I0009 | no `SiteID` (`InstituteID`), no `AgeAtVisit` (`DateOfBirth`), no `ServiceName`/`CreationTime`/`DurationInSeconds` (`RecordingDuration`); `StartDateTime`/`EndDateTime` instead of `StartTime`/`EndTime`; **no reports_findings** |

Analytic mapping used by the audit (`sortinghat/audit/field_audit.py`):

| Audit field | Real source |
|---|---|
| EEG start `t0` | `reports_findings."StartTime(EEG)"`; at I0008/I0009 `eeg_metadata.StartDateTime` (no findings file) |
| age | `reports_findings.AgeAtVisit`, else `eeg_metadata.AgeAtVisit`, else `AgeInDaysAtVisit / 365.25` (I0003), else `StartTime - DateOfBirth` (I0008/I0009) |
| medication timing | `omop_drug_exposure` (`drug_source_value` regex for sedatives; `drug_exposure_end_datetime` present as the administration-record proxy; `drug_type_concept_id` can settle order vs administration) |
| lab collection / result time | `omop_measurement.measurement_datetime` (+ `measurement_date`, `measurement_time`); **no result-time column exists**, so the row fails |
| GCS / FOUR / RASS | `omop_measurement` rows whose `measurement_source_value` matches the score regex |
| notes | `omop_note.note_datetime` (exists; CONFIRMED name) |
| imaging finalization | `imaging.report_final_datetime` (ASSUMED; imaging is not at the placeholder prefix and exists for I0001/I0004 only) |
| acute care flag | DERIVED: `omop_visit_occurrence` visit overlapping the EEG time (`visit_concept_id`, else `visit_source_value` text); not in any eeg_metadata header. If no class can be derived the audit uses all adults and says so |

## Remaining unknowns (names alone cannot settle them)

1. **Imaging contents**: what is inside `Imaging/{I0001,I0004}/Clinical/` (and whether it holds report finalization
   time); one more names-only listing would tell. None of the six EEG-metadata sites has an imaging prefix.
2. **Lab result / verification time**: no such column in OMOP `measurement`.
3. **Medication order versus administration time**: `drug_type_concept_id` is the discriminating column; its
   values (via the concept table) need one aggregate run.
4. **ReferralIndication**: in no real header; PatientClass must be derived from visits (`visit_concept_id` may be zero-filled).
5. Units of `RecordingDuration`, meaning of `BidsFlag`, whether `note_datetime` is authoring or service time,
   whether I0008/I0009 `DateOfBirth` is shifted consistently with `StartDateTime`.
6. FOUR-score rows in `measurement`; `eeg_metadata.BDSPPatientID` blankness per release; hospital names behind site codes.

## Column list

Generated from `schema.py` (`python -c "from sortinghat import schema; print(schema.provenance_markdown())"`).
