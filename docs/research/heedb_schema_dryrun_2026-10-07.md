# Phase 0a schema dry run (names only)

Sites checked: I0002, I0003, I0008, I0009, S0001, S0002. Only table and column NAMES are read and shown; no values, no rows.

- Tables/files not found: 3
- Missing CONFIRMED columns (schema.py is wrong or the release differs): 23
- Missing NAMED columns (may simply not exist): 10
- Missing ASSUMED columns (placeholders to remap): 12

| Table | Site | Found | Expected | Present | Missing | Unlisted cols |
|---|---|---|---|---|---|---|
| eeg_metadata | I0002 | yes | 18 | 12 | 6 | 0 |
| eeg_metadata | I0003 | yes | 18 | 14 | 4 | 1 |
| eeg_metadata | I0008 | yes | 18 | 6 | 12 | 6 |
| eeg_metadata | I0009 | yes | 18 | 6 | 12 | 6 |
| eeg_metadata | S0001 | yes | 18 | 15 | 3 | 1 |
| eeg_metadata | S0002 | yes | 18 | 15 | 3 | 1 |
| reports_findings | I0002 | yes | 30 | 29 | 1 | 19 |
| reports_findings | I0003 | yes | 30 | 29 | 1 | 22 |
| reports_findings | I0008 | NO | 30 | 0 | 0 |  |
| reports_findings | I0009 | NO | 30 | 0 | 0 |  |
| reports_findings | S0001 | yes | 30 | 30 | 0 | 23 |
| reports_findings | S0002 | yes | 30 | 30 | 0 | 23 |
| heedb_patients |  | yes | 11 | 10 | 1 | 2 |
| icd10_neurology |  | yes | 7 | 7 | 0 | 16 |
| medication_atc |  | yes | 2 | 2 | 0 | 17 |
| omop_person |  | yes | 6 | 6 | 0 | 12 |
| omop_visit_occurrence |  | yes | 7 | 5 | 2 | 12 |
| omop_condition_occurrence |  | yes | 4 | 4 | 0 | 12 |
| omop_drug_exposure |  | yes | 6 | 6 | 0 | 17 |
| omop_measurement |  | yes | 7 | 7 | 0 | 16 |
| omop_death |  | yes | 3 | 3 | 0 | 4 |
| omop_procedure_occurrence |  | yes | 5 | 5 | 0 | 11 |
| omop_observation |  | yes | 6 | 6 | 0 | 15 |
| omop_concept |  | yes | 7 | 7 | 0 | 3 |
| omop_note |  | yes | 9 | 9 | 0 | 7 |
| omop_note_nlp |  | yes | 8 | 8 | 0 | 6 |
| imaging |  | NO | 4 | 0 | 0 |  |

## Missing columns

- eeg_metadata / I0002: `EEGFolder` [CONFIRMED] (closest actual name: `BidsFolder`)
- eeg_metadata / I0002: `ServiceName` [CONFIRMED]
- eeg_metadata / I0002: `DateOfDeath` [CONFIRMED]
- eeg_metadata / I0002: `HasPersystAnnotations` [NAMED] (closest actual name: `HasXLTEKAnnotations`)
- eeg_metadata / I0002: `PatientClass` [ASSUMED]
- eeg_metadata / I0002: `ReferralIndication` [ASSUMED]
- eeg_metadata / I0003: `EEGFolder` [CONFIRMED] (closest actual name: `BidsFolder`)
- eeg_metadata / I0003: `DateOfDeath` [CONFIRMED]
- eeg_metadata / I0003: `PatientClass` [ASSUMED]
- eeg_metadata / I0003: `ReferralIndication` [ASSUMED]
- eeg_metadata / I0008: `SiteID` [CONFIRMED]
- eeg_metadata / I0008: `EEGFolder` [CONFIRMED] (closest actual name: `BidsFolder`)
- eeg_metadata / I0008: `DurationInSeconds` [CONFIRMED]
- eeg_metadata / I0008: `ServiceName` [CONFIRMED]
- eeg_metadata / I0008: `AgeAtVisit` [CONFIRMED]
- eeg_metadata / I0008: `DateOfDeath` [CONFIRMED] (closest actual name: `DateOfBirth`)
- eeg_metadata / I0008: `StartTime` [CONFIRMED] (closest actual name: `StartDateTime`)
- eeg_metadata / I0008: `EndTime` [CONFIRMED] (closest actual name: `EndDateTime`)
- eeg_metadata / I0008: `CreationTime` [NAMED]
- eeg_metadata / I0008: `HasPersystAnnotations` [NAMED] (closest actual name: `HasXLTEKAnnotations`)
- eeg_metadata / I0008: `PatientClass` [ASSUMED]
- eeg_metadata / I0008: `ReferralIndication` [ASSUMED]
- eeg_metadata / I0009: `SiteID` [CONFIRMED]
- eeg_metadata / I0009: `EEGFolder` [CONFIRMED] (closest actual name: `BidsFolder`)
- eeg_metadata / I0009: `DurationInSeconds` [CONFIRMED]
- eeg_metadata / I0009: `ServiceName` [CONFIRMED]
- eeg_metadata / I0009: `AgeAtVisit` [CONFIRMED]
- eeg_metadata / I0009: `DateOfDeath` [CONFIRMED] (closest actual name: `DateOfBirth`)
- eeg_metadata / I0009: `StartTime` [CONFIRMED] (closest actual name: `StartDateTime`)
- eeg_metadata / I0009: `EndTime` [CONFIRMED] (closest actual name: `EndDateTime`)
- eeg_metadata / I0009: `CreationTime` [NAMED]
- eeg_metadata / I0009: `HasPersystAnnotations` [NAMED] (closest actual name: `HasXLTEKAnnotations`)
- eeg_metadata / I0009: `PatientClass` [ASSUMED]
- eeg_metadata / I0009: `ReferralIndication` [ASSUMED]
- eeg_metadata / S0001: `BDSPLastModifiedDTS` [NAMED]
- eeg_metadata / S0001: `PatientClass` [ASSUMED]
- eeg_metadata / S0001: `ReferralIndication` [ASSUMED]
- eeg_metadata / S0002: `BDSPLastModifiedDTS` [NAMED]
- eeg_metadata / S0002: `PatientClass` [ASSUMED]
- eeg_metadata / S0002: `ReferralIndication` [ASSUMED]
- reports_findings / I0002: `ServiceName(EEG)` [CONFIRMED] (closest actual name: `CreationTime(EEG)`)
- reports_findings / I0003: `ServiceName(EEG)` [CONFIRMED] (closest actual name: `CreationTime(EEG)`)
- **reports_findings** / I0008: NOT FOUND at `EEG/HEEDB_Metadata/{site}_EEG__reports_findings.csv`
- **reports_findings** / I0009: NOT FOUND at `EEG/HEEDB_Metadata/{site}_EEG__reports_findings.csv`
- heedb_patients: `Race` [NAMED]
- omop_visit_occurrence: `discharge_to_concept_id` [NAMED] (closest actual name: `discharged_to_concept_id`)
- omop_visit_occurrence: `discharge_to_source_value` [NAMED] (closest actual name: `discharged_to_source_value`)
- **imaging**: NOT FOUND at `Imaging/imaging_metadata/`

## Still unknown (no column named anywhere in the source)

- lab result / verification time (OMOP measurement has only measurement_datetime / measurement_date)
- medication order time vs administration time (drug_exposure_start_datetime semantics)
- imaging table location, modality and report finalization time (Imaging/ prefix)
- note and note_nlp tables (existence, note_datetime, note type coding)
- PatientClass (acute-care flag) and ReferralIndication; nearest: ServiceName, visit_occurrence.visit_concept_id
- OMOP person extras (gender, birth date); other ICD-chapter / ATC-group wide columns; FOUR score rows
