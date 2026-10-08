# Real HEEDB table and column names

Two sources of evidence:

1. **Names-only dry run of the real BDSP access point, 2026-10-07** (authorised by the project lead; CSV header lines,
   parquet footers and `Delimiter='/'` prefix listings only; no value, row, count, date or ID was read or printed).
   Output: `docs/research/heedb_schema_dryrun_2026-10-07.md`. Section 0 below summarises what it settled.
2. The earlier research code (`dgkenn/codex-playground-`, ref `origin/claude/research-program-continuation-5upe54`,
   2026-08-10): column names used by code that read the real bucket. Sections A to C are from that code and are
   annotated where the dry run changed the picture. See `docs/heedb_access.md` for key paths and access.

Status vocabulary (as in `sortinghat/schema.py`):

- **CONFIRMED**: the exact name is SEEN in a real header (dry run), or is read by analysis code that ran against
  the real table. A mapping is CONFIRMED when both sides are real header names. This says nothing about units,
  semantics or fill rates, which were not read.
- **NAMED**: the name appears only in an extractor request list, constant or doc, and was NOT found in the header.
- **ASSUMED**: placeholder, found in no real header (`PatientClass`, `ReferralIndication`, the `imaging` table).

## 0. What the 2026-10-07 names-only dry run settled

### 0.1 `eeg_metadata` is NOT uniform across sites: four header variants

`schema.SITE_VARIANTS` records them; `data_io.normalise_site_table` maps each onto canonical names. Every name
below was SEEN in the header (CONFIRMED). "-" means the column is not in that site's header.

| Canonical | S0001 / S0002 | I0002 | I0003 | I0008 / I0009 (actual header name) |
|---|---|---|---|---|
| SiteID | SiteID | SiteID | SiteID | - (header has **InstituteID**; SiteID is filled from the file name) |
| BDSPPatientID | yes (blank on some releases) | yes | yes | yes |
| BidsFolder | yes | yes | yes | yes |
| SessionID | yes | yes | yes | yes |
| EEGFolder | yes | - | - | - |
| DurationInSeconds | yes | yes | yes | **RecordingDuration** (unit not stated by the name; assumed seconds, verify with an aggregate check) |
| ServiceName | yes | - | yes | - |
| AgeAtVisit | yes (sparse) | yes | yes | - (**DateOfBirth** present; age = StartTime - DateOfBirth) |
| AgeInDaysAtVisit | - | - | yes | - |
| SexDSC | yes | yes | yes | yes |
| DateOfBirth | - | - | - | yes |
| DateOfDeath | yes | - | - | - |
| StartTime / EndTime | yes (blank on the 2026_04_30 release) | yes | yes | **StartDateTime / EndDateTime** |
| CreationTime | yes | yes | yes | - |
| HasXLTEKAnnotations | yes | yes | yes | yes |
| HasPersystAnnotations | yes | - | yes | - |
| BDSPLastModifiedDTS | - | yes | yes | yes |
| BidsFlag | yes | - | - | yes |
| InstituteID | - | - | - | yes |

`EEGFolder` (S0001/S0002 only) is why `BidsFolder` is the universal key: the `ceeg` task-token rule in
`data_io.bids_edf_key` cannot be applied at the I-sites (it defaults to `EEG` there, unverified).
**I0008 and I0009 have no `reports_findings` file at all**; their real start/end are in `eeg_metadata`
(`StartDateTime` / `EndDateTime`).

### 0.2 `reports_findings` variants (I0008 / I0009: no file)

All sites that have the file carry `BDSPPatientID, SessionID, StartTime(EEG), EndTime(EEG), AgeAtVisit, SexDSC` and
**all 39 label columns**: the 12 CONFIRMED and 11 NAMED flags of section A2 (now CONFIRMED) plus 16 sleep/syndrome
labels (`spindles, vertex wave, k_complexes, posts, awake, n1, n2, dravet, jeavons, sunflower, wham, angelman, fold,
jae, jme, bects`). Differences:

| Header name | Sites | Canonical | Role |
|---|---|---|---|
| ServiceName(EEG) | S0001, S0002 (NOT I0002, I0003) | ServiceName(EEG) | OR / EMU / Routine / LTM |
| SiteID | S0001, S0002 | SiteID | |
| CreationTime(EEG) | all three groups | ReportCreationTime | report creation |
| BeginDTS(Reports), ExamEndDTS(Reports) | S0001, S0002 | ReportBeginDTS, ReportExamEndDTS | exam begin/end; start-time fallback |
| EncounterDTS(Reports) | S0001, S0002 | ReportEncounterDTS | encounter time |
| ProcedureDSC(Reports) | S0001, S0002 | ReportProcedureDSC | procedure description (EEG type, not a clinical indication) |
| EEGDateTime(Reports), ProcedureDate(Reports) | I0003 | ReportEEGDateTime, ReportProcedureDate | start-time fallback |
| EEGDate(Reports) | I0002 | ReportEEGDate | date only (not a time proxy) |
| AgeInDaysAtVisit | I0003 | AgeInDaysAtVisit | |
| N1 | I0003 | (not mapped) | duplicate of `n1` in different case |
| DeidentifiedName(Reports) | all three groups | (never loaded) | name-like text; `data_io.read_site_table` does not read it |

### 0.3 Other tables

- `HEEDB_patients`: `Race` does not exist; the header has **RaceAndEthnicity** and **RaceAndEthnicityDSC**. The other
  NAMED columns (`VisitCount, HasEEG, HasReports, MatchedEEGReports, ICD10Count, MedicationCount`) exist.
- `HEEDB_ICD10_for_Neurology`: 16 chapter columns (`Behavioral/Cognitive Syndromes ... Miscellaneous`) plus the NAMED
  `SiteID, SexDSC, VisitCount, AgeAtVisitAvg` (all exist). `HEEDB_Medication_ATC`: 13 ATC-group columns plus the same four.
- OMOP `visit_occurrence`: the NAMED `discharge_to_*` are really **`discharged_to_concept_id` / `discharged_to_source_value`**.
  Also present: `visit_occurrence_id, visit_type_concept_id, admitted_from_concept_id, admitted_from_source_value,
  visit_start_date, visit_end_date, care_site_id, preceding_visit_occurrence_id`.
- OMOP `note` and `note_nlp` **exist** with the OMOP CDM v5.4 columns (`note_datetime`, `note_date`,
  `note_type_concept_id`, `note_class_concept_id`, `note_title`, `note_source_value`, `visit_occurrence_id`; `note_text`
  is in the table and is deliberately never requested). `person` has the CDM v5.4 extras (`gender_concept_id`,
  `year_of_birth`, `birth_datetime`, `race_concept_id`, `ethnicity_concept_id`, `*_source_value`).
- OMOP `drug_exposure` has `drug_type_concept_id` (record provenance: prescription written versus administration
  record, via a concept join), `route_source_value`, `drug_exposure_start_date`, `drug_exposure_end_date`,
  `verbatim_end_date`. OMOP `measurement` has `measurement_time`, `measurement_type_concept_id`, `value_source_value`,
  `range_low/high` but **no result/verification datetime**.

### 0.4 Access point layout (prefix names only)

Root: `ECG/ EEG/ EHR/ Imaging/ NAX/ OMOP/ PSG/ PatientMergeHistory/` (no `BIND/`). `EEG/` = `HEEDB_Metadata/ bids/
eeg-metadata/`. `OMOP/` = `I0001-OMOP/ I0002-OMOP/ I0003-OMOP/ Merged/ Merged_Fixed/`. `EHR/` = `I0001-EHR ... I0009-EHR`
(7 sites). `NAX/` = `NAX-epilepsy nax-gcs nax-mci-dementia nax-stroke-mrs nax-stroke-nihss`. **`Imaging/` holds only
`I0001/` and `I0004/`, each with `BIDS/ Clinical/ Non-BIDS/`**; none of the six HEEDB EEG-metadata sites (I0002, I0003,
I0008, I0009, S0001, S0002) has an imaging prefix. `Imaging/imaging_metadata/` does not exist. Nothing deeper than
`Imaging/<SITE>/{BIDS,Clinical,Non-BIDS}/` was listed.

## A. Real tables and columns

**Tally of columns read by the earlier code (before the dry run): 63** (eeg_metadata 12, reports_findings 19, HEEDB_patients 4, ICD10 table 3,
medication ATC table 2, OMOP 23 across person/condition/drug/measurement/death/procedure/observation/concept).
The NAMED-only columns are listed in the tables but not counted.

### A1. `EEG/eeg-metadata/{SITE}_eeg_metadata_<release date>.csv` (one row per EEG session)

Only `S0001` and `S0002` were read in the analyses; release date seen: `2026_04_30`. **The dry run (section 0.1) found four header variants; this table describes S0001/S0002.**

| Column | Status | Source | Note |
|---|---|---|---|
| SiteID | CONFIRMED | `analysis/heedb_normal_reference_cohort.py` | |
| BDSPPatientID | CONFIRMED (caveat) | `analysis/heedb_bs_quantify.py`, `heedb_bs_mortality.py` | The later `heedb_command_following.py` docstring says this column is **blank** in this table and the patient id must come from `BidsFolder` (`sub-<SITE><PID>`); `heedb_bs_calibrate.py` says the same. The earlier mortality scripts keyed `DateOfDeath` on it. Check on your release before relying on it. |
| BidsFolder | CONFIRMED | `pipeline/stream_fetch.py`, `heedb_bs_quantify.py` | `sub-<SITE><PID>`; PID is the numeric BDSPPatientID |
| SessionID | CONFIRMED | same | joins to `reports_findings.SessionID` |
| EEGFolder | CONFIRMED (S0001/S0002 only) | `heedb_bs_quantify.py`, `heedb_command_following.py` | value starting `ceeg` selects task token `cEEG` in the EDF key; NOT in the I-site headers |
| DurationInSeconds | CONFIRMED | `pipeline/stream_fetch.py`, `heedb_bs_calibrate.py` | plural **Seconds**; `sortinghat/schema.py` has `DurationInSecond` |
| ServiceName | CONFIRMED | `pipeline/stream_fetch.py` | values seen include Routine, LTM, EMU (config), OR (`heedb_bs_iatrogenic.py`) |
| AgeAtVisit | CONFIRMED (mostly empty) | `docs/HANDOFF.md`, `pipeline/stream_fetch.py` | "largely empty" in the catalog |
| SexDSC | CONFIRMED (often empty) | `pipeline/stream_fetch.py` | |
| DateOfDeath | CONFIRMED | `heedb_bs_mortality.py`, `heedb_bs_iatrogenic.py`, `pipeline/stream_fetch.py` | outcome column; only meaningful for S0001/S0002 |
| StartTime | CONFIRMED (blank) | `heedb_command_following.py` | blank in this table; real times are in `reports_findings` |
| EndTime | CONFIRMED (blank) | same | |
| CreationTime, HasXLTEKAnnotations, HasPersystAnnotations | CONFIRMED (header, 2026-10-07) | dry run | per-site, see 0.1 |
| BDSPLastModifiedDTS | CONFIRMED at I-sites only (header) | dry run | absent at S0001/S0002 |

### A2. `EEG/HEEDB_Metadata/{SITE}_EEG__reports_findings.csv` (one row per EEG report)

Sites read: `S0001`, `S0002`. A label cell counts as asserted when non-empty and not the strings `None`/`nan`
(`has()` in `heedb_bs_mortality.py`).

| Column | Status | Source |
|---|---|---|
| BDSPPatientID | CONFIRMED | 65 uses across `analysis/heedb_*.py`, e.g. `heedb_bs_mortality.py` |
| SessionID | CONFIRMED | `heedb_burden_validity.py`, `heedb_command_following.py` |
| StartTime(EEG) | CONFIRMED | 58 uses; the real EEG start timestamp |
| EndTime(EEG) | CONFIRMED | 57 uses |
| AgeAtVisit | CONFIRMED | `heedb_bs_mortality.py` |
| SexDSC | CONFIRMED | same (value `Female` seen) |
| ServiceName(EEG) | CONFIRMED | `heedb_bs_mortality.py`; values OR, EMU, Routine, LTM |
| bs (burst suppression) | CONFIRMED | `heedb_bs_mortality.py` |
| gen slowing, foc slowing, seizure, gpd, lpd, pdr, low voltage, status | CONFIRMED | same |
| wicket, breach, bets (benign variants) | CONFIRMED | `heedb_bs_iatrogenic.py` |
| normal, abnormal, spikes, lrda, grda, uninterpretable, bipd, eses, cjd, ppr, diffuse Beta | NAMED | `heedb_normal_reference_cohort.py` constant `ABNORMAL_COLS` and `normal` |
| spindles, vertex wave, k_complexes, posts, awake, n1, n2, dravet, jeavons, sunflower, wham, angelman, fold, jae, jme, bects | NAMED | same file, docstring only |

### A3. `EEG/HEEDB_Metadata/HEEDB_patients.csv`

| Column | Status | Source |
|---|---|---|
| SiteID, BDSPPatientID, Sex, AgeAtVisitAvg | CONFIRMED | `pipeline/stream_fetch.py` `_load_patients`; joined on (SiteID, numeric id) |
| VisitCount, HasEEG, HasReports, MatchedEEGReports, ICD10Count, MedicationCount | CONFIRMED (header, 2026-10-07) | `docs/HEEDB_UNLOCK.md` table + dry run |
| Race | does NOT exist; header has RaceAndEthnicity, RaceAndEthnicityDSC | dry run | |

### A4. `HEEDB_ICD10_for_Neurology.csv` and `HEEDB_Medication_ATC.csv`

Wide, **one row per patient, one column per category**, not long event tables. Cells hold code strings or
counts, not timestamps (`num()` in `heedb_bs_iatrogenic.py`: "ICD/med cells hold CODE STRINGS (e.g. 'I63.9
G45.9') or counts -> presence flag").

| Table.column | Status | Source |
|---|---|---|
| icd10_neurology.BDSPPatientID | CONFIRMED | `heedb_bs_iatrogenic.py`, `heedb_normal_reference_cohort.py` |
| icd10_neurology."Cerebrovascular Diseases", "Cerebral Degeneration" | CONFIRMED | `heedb_bs_iatrogenic.py` |
| icd10_neurology.SiteID, SexDSC, VisitCount, AgeAtVisitAvg | NAMED | exclusion list in `heedb_normal_reference_cohort.py` (every other column is an ICD "chapter") |
| icd10_neurology other chapter columns | UNKNOWN names | run `list(df.columns)` once locally |
| medication_atc.BDSPPatientID | CONFIRMED | `heedb_bs_iatrogenic.py` |
| medication_atc."Nervous System Drugs" | CONFIRMED | same |
| medication_atc other ATC-group columns | UNKNOWN names | |

### A5. `OMOP/Merged/{table}/*.parquet` (OMOP CDM v5-style, merged across sites)

`person_id` is the integer `BDSPPatientID`. Extraction pattern: `analysis/heedb_omop_extract.py` (the
`COLS` dict is the list of requested columns). Observed date format: `YYYY-MM-DD HH:MM:SS[.ffffff]`
(`dt()` helpers try `%Y-%m-%d %H:%M:%S.%f`, `%Y-%m-%d %H:%M:%S`, `%Y-%m-%d`).

| Table | Column | Status | Source (downstream use) |
|---|---|---|---|
| (all) | person_id | CONFIRMED | 61 analysis scripts |
| condition_occurrence | condition_start_datetime | CONFIRMED | 4 scripts |
| condition_occurrence | condition_source_value | CONFIRMED | 48 scripts; ICD-9 and ICD-10 strings, dots removed by `norm()` |
| condition_occurrence | condition_concept_id | NAMED | extractor only |
| drug_exposure | drug_exposure_start_datetime, drug_exposure_end_datetime | CONFIRMED | `heedb_bs_iatrogenic.py` family, `heedb_wlst_pressor.py` |
| drug_exposure | drug_source_value, quantity | CONFIRMED | 10 and 35 scripts |
| drug_exposure | drug_concept_id | NAMED | extractor only |
| measurement | measurement_datetime, measurement_date | CONFIRMED | `heedb_burden_nse.py` (falls back from datetime to date) |
| measurement | measurement_source_value, value_as_number, unit_source_value | CONFIRMED | `heedb_reversal_temperature.py` (units mix Fahrenheit and Celsius), `heedb_burden_nse.py` |
| measurement | measurement_concept_id | NAMED | extractor only |
| death | death_datetime | CONFIRMED | 58 scripts |
| death | cause_source_value | NAMED | extractor only |
| procedure_occurrence | procedure_datetime, procedure_date, procedure_concept_id | CONFIRMED | `heedb_wlst_procedure.py` |
| procedure_occurrence | procedure_source_value | CONFIRMED | `heedb_concept_select.py` (numeric billing codes, not names) |
| observation | observation_concept_id | CONFIRMED | catalogue rule 6: "100 % zero" |
| observation | observation_datetime, observation_date, observation_source_value, value_as_string | NAMED | extractor only (goals-of-care entries) |
| visit_occurrence | visit_start_datetime, visit_end_datetime, visit_concept_id, visit_source_value | CONFIRMED (footer, 2026-10-07) | extractor only (single 9 GB part, about 512 M rows) |
| visit_occurrence | discharge_to_concept_id, discharge_to_source_value | **renamed in reality: discharged_to_concept_id, discharged_to_source_value** | dry run |
| concept | concept_id, concept_name, domain_id, vocabulary_id, standard_concept | CONFIRMED | `heedb_concept_select.py`, `heedb_wlst_procedure.py` |
| concept | concept_class_id, concept_code | NAMED | extractor only |
| person | gender_concept_id, year_of_birth, birth_datetime, race_concept_id, ethnicity_concept_id (+ *_source_value) | CONFIRMED (footer, 2026-10-07) | see 0.3 |
| note, note_nlp | all columns of the schema | CONFIRMED (footer, 2026-10-07) | exist; see 0.3 |
| specimen, device_exposure, etc. | all | UNKNOWN | not probed |

Merged table sizes recorded in the source: `measurement` 66 GB / 554 parts, `drug_exposure` 59 GB / 375,
`condition_occurrence` 27 GB / 181. Cohort scale: 49,232 patients with S0001+S0002 EEG reports; 7,323
with a clinician burst-suppression label on 22,057 reports.

### A6. Signal files and I-CARE (context only)

- EDF key: `EEG/bids/{SITE}/{BidsFolder}/ses-{SessionID}/eeg/{BidsFolder}_ses-{SessionID}_task-{cEEG|EEG}_eeg.edf`
  (`analysis/heedb_bs_calibrate.py::bids_key`). Real EDFs seen: 256 Hz, 50 channels for a routine EEG
  (`docs/HEEDB_UNLOCK.md`); channel labels upper-cased match 10-20 names (FP1 ... PZ).
- I-CARE per-patient text fields (`analysis/icare_cohort.py`): `Hospital`, `Age`, `Sex`, `ROSC`, `OHCA`,
  `Shockable Rhythm`, `TTM`, `Outcome`, `CPC`.

## B. What SortingHat currently assumes, mapped to the real names

`sortinghat/schema.py` / `docs/heedb_schema.md` marks columns DOCUMENTED (public BDSP page) or ASSUMED.
"Real equivalent" is what the earlier code actually reads.

### B1. `eeg_metadata`

| SortingHat column | schema.py provenance | Real equivalent | Mapping status |
|---|---|---|---|
| SiteID | DOCUMENTED | eeg_metadata.SiteID | CONFIRMED |
| BDSPPatientID | DOCUMENTED | eeg_metadata.BDSPPatientID (may be blank); reports_findings.BDSPPatientID; derive from BidsFolder | CONFIRMED with caveat: verify non-blank on your release |
| BidsFolder | DOCUMENTED | eeg_metadata.BidsFolder | CONFIRMED |
| SessionID | DOCUMENTED | eeg_metadata.SessionID | CONFIRMED |
| CreationTime | DOCUMENTED | eeg_metadata.CreationTime (all but I0008/I0009) | CONFIRMED (header) |
| StartTime | DOCUMENTED | **reports_findings."StartTime(EEG)"**; eeg_metadata.StartTime is blank | CONFIRMED but in a different table and name |
| EndTime | DOCUMENTED | **reports_findings."EndTime(EEG)"**; eeg_metadata.EndTime is blank | CONFIRMED, different table and name |
| DurationInSecond | DOCUMENTED | **DurationInSeconds** (plural) | CONFIRMED, **name differs; schema.py will not match** |
| ServiceName | DOCUMENTED | eeg_metadata.ServiceName; reports_findings."ServiceName(EEG)" | CONFIRMED (values Routine, LTM, EMU, OR) |
| AgeAtVisit | DOCUMENTED | eeg_metadata.AgeAtVisit (largely empty); reports_findings.AgeAtVisit; HEEDB_patients.AgeAtVisitAvg | CONFIRMED; prefer findings or patients table |
| SexDSC | DOCUMENTED | eeg_metadata.SexDSC (often empty); reports_findings.SexDSC; HEEDB_patients.Sex | CONFIRMED |
| PatientClass (ICU/Inpatient/ED/Outpatient) | ASSUMED | **dry run: no such column in any eeg_metadata header; derive from OMOP visit_occurrence** (`field_audit.derive_patient_class`). Nearest: `ServiceName` (LTM suggests ICU monitoring, OR/EMU/Routine are other contexts) and OMOP `visit_occurrence.visit_concept_id` | UNKNOWN; visit_concept_id is NAMED only and may be zero-filled (rule 6) |
| ReferralIndication | ASSUMED | **dry run: in no real header at any site.** Nearest: `ProcedureDSC(Reports)` at S-sites (EEG type) | UNKNOWN |

Not in SortingHat's schema but real and useful: `DateOfDeath` (eeg_metadata), `EEGFolder`, the finding flags
(`bs`, `gen slowing`, `seizure`, ...), and OMOP `death.death_datetime`.

### B2. `medications`

| SortingHat column | Real equivalent | Mapping status |
|---|---|---|
| BDSPPatientID | OMOP drug_exposure.person_id (integer of BDSPPatientID) | CONFIRMED |
| med_name | OMOP drug_exposure.drug_source_value (free text; the source regex-matched propofol, midazolam, pentobarbital, dexmedetomidine, norepinephrine, ...) | CONFIRMED |
| med_class | none stored. The source derived classes by regex on drug_source_value, or used the per-patient ATC-group wide table (no times) | UNKNOWN as a column; derive it |
| order_time | `drug_type_concept_id` can distinguish prescription-written from administration records (values unread) | UNKNOWN until its values are summarised |
| admin_time | drug_exposure.drug_exposure_start_datetime (end: drug_exposure_end_datetime). Whether this is order or administration time is not stated; the source noted vasopressor end times are stamped at death by the charting system (rule 7 note) | CONFIRMED name; semantics UNKNOWN |

### B3. `labs`

| SortingHat column | Real equivalent | Mapping status |
|---|---|---|
| BDSPPatientID | measurement.person_id | CONFIRMED |
| lab_name | measurement.measurement_source_value (text; the source row-filtered it by regex, e.g. neuron-specific enolase) | CONFIRMED |
| collect_time | measurement.measurement_datetime (or measurement_date) | CONFIRMED name; collection-versus-result semantics UNKNOWN |
| result_time | dry run: no result/verification datetime in `measurement` (only measurement_datetime, measurement_date, measurement_time) | NOT FOUND |

Also seen: value_as_number, unit_source_value (mixed units possible), measurement_concept_id (NAMED).

### B4. `imaging`

| SortingHat column | Real equivalent | Mapping status |
|---|---|---|
| all (modality, study_time, report_final_time) | `Imaging/<SITE>/{BIDS,Clinical,Non-BIDS}/` for **I0001 and I0004 only** (dry run); table layout inside not listed; BIND/ does not exist at the root | UNKNOWN columns; location CONFIRMED to prefix level |

### B5. `clinical_scores`

| SortingHat column | Real equivalent | Mapping status |
|---|---|---|
| BDSPPatientID | measurement.person_id | CONFIRMED |
| score_type | measurement.measurement_source_value, row-filtered by the regex `glasgow|\bgcs\b|rass|richmond|sedation scale|level of consciousness|eye opening|best motor response|best verbal response|ramsay|arousal` (`heedb_omop_extract.py` pseudo-table `measurement_conscious`) | CONFIRMED (GCS components and RASS); FOUR score not seen |
| score_value | measurement.value_as_number | CONFIRMED |
| score_time | measurement.measurement_datetime | CONFIRMED |

### B6. `notes`

| SortingHat column | Real equivalent | Mapping status |
|---|---|---|
| note_id, note_type, note_time | OMOP `note`: `note_id`, `note_type_concept_id` / `note_source_value` / `note_title` / `note_class_concept_id`, `note_datetime` (+ `note_date`) | CONFIRMED (footer, 2026-10-07); timestamp semantics unread |

## C. Mismatches to fix when remapping (human action, not done here)

1. `DurationInSecond` must become `DurationInSeconds` (confirmed plural) at S/I0002/I0003; I0008/I0009 call it `RecordingDuration`.
2. EEG start and end times live in `reports_findings` as `StartTime(EEG)` / `EndTime(EEG)`; the metadata
   table's `StartTime`/`EndTime` were blank in `S0001`/`S0002` for release `2026_04_30`. The two tables join on
   (`BidsFolder` = `sub-<SITE><BDSPPatientID>`, `SessionID`).
3. Medications, labs and scores are **OMOP event tables** (`drug_exposure`, `measurement`) joined by integer
   `person_id`, not HEEDB-metadata tables with `med_name`/`lab_name`/`score_type` columns. `HEEDB_Medication_ATC`
   and `HEEDB_ICD10_for_Neurology` are wide per-patient presence tables with no timestamps, so they cannot
   support timing checks.
4. Done by the 2026-10-07 dry run (section 0): notes exist, `PatientClass` and `ReferralIndication` are in no header (PatientClass is
   derived from visits), imaging sits under `Imaging/<SITE>/` for I0001 and I0004 only.
5. Data-quality lessons that bear on the audit: `*_concept_id` columns can be all zero (rule 6); billing-code
   `*_source_value` columns need a `concept` join (rule 5); death rows are incomplete (about 45 percent in the
   burst-suppression cohort), so "no death row" is not survival; temperature units mix F and C; time-shifted
   dates were consistent enough across tables for minute-level EEG-to-measurement joins in
   `heedb_command_following.py::build_worklist`, but cross-table date alignment should still be checked
   (`field_audit`'s nearest-event alignment check, D-117).
