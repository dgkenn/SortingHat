# Phase 0a schema dry run (names only)

Sites checked: I0002, I0003, I0008, I0009, S0001, S0002. Only table and column NAMES are read and shown; no values, no rows.

- Tables/files not found: 1
- Missing CONFIRMED columns (schema.py is wrong or the release differs): 0
- Missing NAMED columns (may simply not exist): 0
- Missing ASSUMED columns (placeholders to remap): 0

| Table | Site | Found | Expected | Present | Missing | Unlisted cols |
|---|---|---|---|---|---|---|
| eeg_metadata | I0002 | yes | 12 | 12 | 0 | 0 |
| eeg_metadata | I0003 | yes | 15 | 15 | 0 | 0 |
| eeg_metadata | I0008 | yes | 12 | 12 | 0 | 0 |
| eeg_metadata | I0009 | yes | 12 | 12 | 0 | 0 |
| eeg_metadata | S0001 | yes | 16 | 16 | 0 | 0 |
| eeg_metadata | S0002 | yes | 16 | 16 | 0 | 0 |
| reports_findings | I0002 | yes | 48 | 48 | 0 | 0 |
| reports_findings | I0003 | yes | 51 | 51 | 0 | 0 |
| reports_findings | I0008 | absent (expected) | 0 | 0 | 0 |  |
| reports_findings | I0009 | absent (expected) | 0 | 0 | 0 |  |
| reports_findings | S0001 | yes | 53 | 53 | 0 | 0 |
| reports_findings | S0002 | yes | 53 | 53 | 0 | 0 |
| heedb_patients |  | yes | 12 | 12 | 0 | 0 |
| icd10_neurology |  | yes | 23 | 23 | 0 | 0 |
| medication_atc |  | yes | 19 | 19 | 0 | 0 |
| omop_person |  | yes | 9 | 9 | 0 | 9 |
| omop_visit_occurrence |  | yes | 11 | 11 | 0 | 6 |
| omop_condition_occurrence |  | yes | 4 | 4 | 0 | 12 |
| omop_drug_exposure |  | yes | 11 | 11 | 0 | 12 |
| omop_measurement |  | yes | 10 | 10 | 0 | 13 |
| omop_death |  | yes | 4 | 4 | 0 | 3 |
| omop_procedure_occurrence |  | yes | 5 | 5 | 0 | 11 |
| omop_observation |  | yes | 6 | 6 | 0 | 15 |
| omop_concept |  | yes | 7 | 7 | 0 | 3 |
| omop_note |  | yes | 9 | 9 | 0 | 7 |
| omop_note_nlp |  | yes | 8 | 8 | 0 | 6 |
| imaging |  | NO | 4 | 0 | 0 |  |

## Missing columns

- reports_findings / I0008: no file at `EEG/HEEDB_Metadata/{site}_EEG__reports_findings.csv` (the site variant has none; expected)
- reports_findings / I0009: no file at `EEG/HEEDB_Metadata/{site}_EEG__reports_findings.csv` (the site variant has none; expected)
- **imaging**: NOT FOUND at `Imaging/imaging_metadata/`

## Unlisted columns (present in the real header, not in schema.py; names only)

- omop_person: `month_of_birth`, `day_of_birth`, `location_id`, `provider_id`, `care_site_id`, `person_source_value`, `gender_source_concept_id`, `race_source_concept_id`, `ethnicity_source_concept_id`
- omop_visit_occurrence: `visit_start_date`, `visit_end_date`, `provider_id`, `care_site_id`, `visit_source_concept_id`, `preceding_visit_occurrence_id`
- omop_condition_occurrence: `condition_occurrence_id`, `condition_start_date`, `condition_end_date`, `condition_end_datetime`, `condition_type_concept_id`, `condition_status_concept_id`, `stop_reason`, `provider_id`, `visit_occurrence_id`, `visit_detail_id`, `condition_source_concept_id`, `condition_status_source_value`
- omop_drug_exposure: `drug_exposure_id`, `verbatim_end_date`, `stop_reason`, `refills`, `days_supply`, `sig`, `route_concept_id`, `lot_number`, `provider_id`, `visit_detail_id`, `drug_source_concept_id`, `dose_unit_source_value`
- omop_measurement: `measurement_id`, `operator_concept_id`, `value_as_concept_id`, `unit_concept_id`, `range_low`, `range_high`, `provider_id`, `visit_detail_id`, `measurement_source_concept_id`, `unit_source_concept_id`, `value_source_value`, `measurement_event_id`, `meas_event_field_concept_id`
- omop_death: `death_type_concept_id`, `cause_concept_id`, `cause_source_concept_id`
- omop_procedure_occurrence: `procedure_occurrence_id`, `procedure_end_date`, `procedure_end_datetime`, `procedure_type_concept_id`, `modifier_concept_id`, `quantity`, `provider_id`, `visit_occurrence_id`, `visit_detail_id`, `procedure_source_concept_id`, `modifier_source_value`
- omop_observation: `observation_id`, `observation_type_concept_id`, `value_as_number`, `value_as_concept_id`, `qualifier_concept_id`, `unit_concept_id`, `provider_id`, `visit_occurrence_id`, `visit_detail_id`, `observation_source_concept_id`, `unit_source_value`, `qualifier_source_value`, `value_source_value`, `observation_event_id`, `obs_event_field_concept_id`
- omop_concept: `valid_start_date`, `valid_end_date`, `invalid_reason`
- omop_note: `note_text`, `encoding_concept_id`, `language_concept_id`, `provider_id`, `visit_detail_id`, `note_event_id`, `note_event_field_concept_id`
- omop_note_nlp: `snippet`, `offset`, `lexical_variant`, `note_nlp_source_concept_id`, `nlp_date`, `term_modifiers`

## Access point prefixes (names only; Delimiter='/', depth <= 2; per-patient folders never listed)

- `<root>`
    - prefix `ECG/`
    - prefix `EEG/`
    - prefix `EHR/`
    - prefix `Imaging/`
    - prefix `NAX/`
    - prefix `OMOP/`
    - prefix `PSG/`
    - prefix `PatientMergeHistory/`
- `ECG/`
    - prefix `I0001/`
    - prefix `I0006/`
- `EEG/`
    - prefix `HEEDB_Metadata/`
    - prefix `bids/`
    - prefix `eeg-metadata/`
- `EHR/`
    - prefix `I0001-EHR/`
    - prefix `I0002-EHR/`
    - prefix `I0003-EHR/`
    - prefix `I0004-EHR/`
    - prefix `I0006-EHR/`
    - prefix `I0007-EHR/`
    - prefix `I0009-EHR/`
- `Imaging/`
    - prefix `I0001/`
    - prefix `I0004/`
- `NAX/`
    - prefix `NAX-epilepsy/`
    - prefix `nax-gcs/`
    - prefix `nax-mci-dementia/`
    - prefix `nax-stroke-mrs/`
    - prefix `nax-stroke-nihss/`
- `OMOP/`
    - prefix `I0001-OMOP/`
    - prefix `I0002-OMOP/`
    - prefix `I0003-OMOP/`
    - prefix `Merged/`
    - prefix `Merged_Fixed/`
- `PSG/`
    - prefix `bids/`
    - prefix `caisr_output_snasiri/`
    - prefix `morpheus/`
    - prefix `prepared_snasiri/`
    - prefix `psg-metadata/`
- `PatientMergeHistory/`
    - file `<id-like file>`
    - file `<id-like file>`
    - file `<id-like file>`
- `Imaging/I0001/`
    - prefix `BIDS/`
    - prefix `Clinical/`
    - prefix `Non-BIDS/`
- `Imaging/I0004/`
    - prefix `BIDS/`
    - prefix `Clinical/`
    - prefix `Non-BIDS/`

## Still unknown after the names-only dry run

- lab result / verification time: NO result-time column in omop_measurement (only measurement_datetime / measurement_date / measurement_time)
- medication order time vs administration time: drug_exposure_start_datetime semantics unread; drug_type_concept_id is the discriminating column (values unread)
- imaging table contents: Imaging/<SITE>/{BIDS,Clinical,Non-BIDS}/ for I0001 and I0004 only (none of the six EEG-metadata sites); modality and report finalization columns not located
- PatientClass (acute-care flag): no column in any eeg_metadata header; derive from omop_visit_occurrence (visit_concept_id may be zero-filled)
- ReferralIndication: no column in any real header
- units of RecordingDuration (I0008/I0009), semantics of BidsFlag, note_datetime (authoring vs service time)
