# HEEDB schema used by the synthetic generator

Source: public BDSP page for the Harvard Electroencephalography Database
(bdsp.io/content/harvard-eeg-db). Note: "HEEDB" also names the unrelated
Harvard-Emory ECG Database; this project concerns the EEG database.

Documented publicly: an EEG metadata CSV (SiteID, BDSPPatientID, BidsFolder,
SessionID, CreationTime, StartTime, EndTime, DurationInSecond, HasXLTEKAnnotations,
HasPersystAnnotations, ServiceName, AgeAtVisit, SexDSC, BDSPLastModifiedDTS), BIDS
participants.tsv / scans.tsv, and EHR branches data_Structured (medications, labs,
vitals, problem lists, ICD, CPT, demographics as Parquet), data_Unstructured (notes,
EEG reports) and data_Imaging (imaging metadata and reports). Site IDs listed: S0001,
S0002, I0003, I0002 (hospital mapping not given). Column names inside the EHR
branches, clinical scores (GCS/FOUR/RASS) and acute-care/setting flags are NOT
publicly documented, so they are ASSUMED below and must be remapped by someone with
real access (see COLUMN_ALIASES in `sortinghat/schema.py`).

Operationalisations chosen by the audit (also assumptions): acute care = PatientClass
in {ICU, Inpatient, ED}; candidate = first acute-care adult EEG per patient with a
start time; "present" criteria for labs/imaging/notes = >=95%.

## Table and column provenance

| Table | Column | Type | Provenance | Description |
|---|---|---|---|---|
| eeg_metadata | SiteID | str | DOCUMENTED | Hospital where the EEG was recorded |
| eeg_metadata | BDSPPatientID | str | DOCUMENTED | Patient identifier |
| eeg_metadata | BidsFolder | str | DOCUMENTED | Folder holding a patient's studies |
| eeg_metadata | SessionID | str | DOCUMENTED | Study/session identifier |
| eeg_metadata | CreationTime | datetime | DOCUMENTED | De-identified (date-shifted) creation time |
| eeg_metadata | StartTime | datetime | DOCUMENTED | De-identified (date-shifted) EEG start |
| eeg_metadata | EndTime | datetime | DOCUMENTED | De-identified (date-shifted) EEG end |
| eeg_metadata | DurationInSecond | float | DOCUMENTED | Recording duration |
| eeg_metadata | ServiceName | str | DOCUMENTED | Routine / LTM / EMU |
| eeg_metadata | AgeAtVisit | float | DOCUMENTED | Age at the study |
| eeg_metadata | SexDSC | str | DOCUMENTED | Patient-reported gender |
| eeg_metadata | PatientClass | str | ASSUMED | ICU / Inpatient / ED / Outpatient (defines acute care) |
| eeg_metadata | ReferralIndication | str | ASSUMED | EEG referral indication category |
| medications | BDSPPatientID | str | ASSUMED | Patient identifier |
| medications | med_name | str | ASSUMED | Medication name |
| medications | med_class | str | ASSUMED | sedative / analgesic / antiseizure / other |
| medications | order_time | datetime | ASSUMED | Order time |
| medications | admin_time | datetime | ASSUMED | Administration (MAR) time; null if only ordered |
| labs | BDSPPatientID | str | ASSUMED | Patient identifier |
| labs | lab_name | str | ASSUMED | Assay |
| labs | collect_time | datetime | ASSUMED | Specimen collection time |
| labs | result_time | datetime | ASSUMED | Result / verification time |
| imaging | BDSPPatientID | str | ASSUMED | Patient identifier |
| imaging | modality | str | ASSUMED | CT head / MRI brain / CTA |
| imaging | study_time | datetime | ASSUMED | Acquisition time |
| imaging | report_final_time | datetime | ASSUMED | Report finalization time |
| clinical_scores | BDSPPatientID | str | ASSUMED | Patient identifier |
| clinical_scores | score_type | str | ASSUMED | GCS / FOUR / RASS |
| clinical_scores | score_value | float | ASSUMED | Score value |
| clinical_scores | score_time | datetime | ASSUMED | Documentation time |
| notes | BDSPPatientID | str | ASSUMED | Patient identifier |
| notes | note_id | str | ASSUMED | Note identifier |
| notes | note_type | str | ASSUMED | Physician / nursing / EEG report / other |
| notes | note_time | datetime | ASSUMED | Note timestamp (text body not modelled) |
