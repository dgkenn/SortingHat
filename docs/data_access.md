# Data access: every dataset the plan needs, as found in the earlier research code

Source: `github.com/dgkenn/codex-playground-`, ref `origin/claude/research-program-continuation-5upe54`
(2026-08-10). HEEDB detail stays in `docs/heedb_access.md` and `docs/heedb_schema_real.md`; secrets-free
credential layout is in `docs/credentials_template.md`; code is `sortinghat/data_io.py` (with
`sortinghat/heedb_io.py` as a re-export shim). Nothing here was verified live: the status of each dataset is

- **seen-in-code**: the source has a loader or script that read it, with real paths;
- **documented-only**: named in the source's notes or surveys, no loader;
- **not found**: nothing in the source.

Network facts below are what the source recorded. Where the source discovered a gotcha, the source's rule number
is quoted ("rule N" = `CLAUDE.md` error catalogue; `DAS` = `bsde/docs/DEPOSIT_ACCESS_STATUS.md`).

## Summary

| Dataset | Status | Method | Auth |
|---|---|---|---|
| HEEDB (EEG, reports, OMOP, EHR) | seen-in-code | S3 access point `bdsp-credentialed-access-point` | AWS profile (BDSP-registered keys) |
| I-CARE | seen-in-code (two routes) | BDSP restricted access point; also open on PhysioNet v2.1 | AWS profile / none |
| MORGOTH (data and checkpoints) | seen-in-code (data), documented-only (weights) | BDSP projects access point | AWS profile |
| GCS from EHR | seen-in-code | HEEDB OMOP `measurement` rows filtered by regex | AWS profile |
| Toxic-metabolic encephalopathy cohort | not found as a cohort | only proxies (ICD chapter columns, aetiology code groups) | n/a |
| BIND | not found | | |
| NESI | not found as data (named as a 2026 model/marker set) | | |
| SPaRCNet | documented-only | BDSP restricted prefix `sparcnet_data/` | AWS profile |
| PROPHET | not found | | |
| TUH/TUEG (TUAB, TUSZ, TUEV, TUAR) | seen-in-code (transport), never run | rsync over SSH to NEDC | NEDC-registered SSH key |
| TDBRAIN | not found | | |
| CBraMod weights | seen-in-code | Hugging Face `weighting666/CBraMod` | none |
| Other model weights (MORGOTH, LaBraM, BIOT, EEGPT) | MORGOTH documented-only; others not found as downloads | | |
| VitalDB | seen-in-code | public HTTPS API | none |
| PhysioNet DUA projects | seen-in-code | HTTPS with Django session login | PhysioNet account + per-project DUA |
| OpenNeuro / NEMAR (ds005385, ds004541, ds005620, ds006695, ds007554) | seen-in-code (OpenNeuro), NEMAR mentioned only | anonymous S3 over HTTPS | none |
| Chennu propofol sedation | seen-in-code | HTTP Range reads inside one remote ZIP | none |

Not found at all in the source: **BIND, PROPHET, TDBRAIN, TUAR, a toxic-metabolic encephalopathy cohort, and
NESI as a dataset.** Details in each section.

## HEEDB

Status: seen-in-code. Method, paths, 403 gotcha, setup and lessons are in `docs/heedb_access.md`; columns in
`docs/heedb_schema_real.md`.

- Path pattern: access point `arn:aws:s3:us-east-1:184438910517:accesspoint/bdsp-credentialed-access-point`
  as `Bucket=`; prefixes `EEG/eeg-metadata/`, `EEG/HEEDB_Metadata/`, `EEG/bids/`, `OMOP/Merged/`, `EHR/`, `PSG/`,
  `ECG/`, `Imaging/`, `NAX/`.
- Auth: BDSP-registered AWS key pair, profile name `physionet` (any name works).
- Code: `data_io.read_csv_table`, `iter_omop_batches`, `bids_edf_key`; run via `scripts/heedb_run.sh`.
- Gotcha headline: sandbox placeholder `AWS_ACCESS_KEY_ID` shadows the profile and produces a 403 that looks
  like expiry (rules 8 and 36).
- Also in the same bucket (credentialed AP, `EEG/bids/`): the source found **12 BIDS datasets**, among them
  `Neurotech` (4,915 subjects, EDF+ 256 Hz, CC BY-NC 4.0, `phenotype/` directory; age present for 2,691 and sex
  for 2,213 subjects; largely paediatric; shifted dates with times of day preserved). The anonymous view of
  `EEG/bids/Neurotech/` showed only a LICENSE file; with credentials the data is reachable. The other 11
  dataset names were not recorded.

## I-CARE

Status: seen-in-code, **two routes**.

1. BDSP route (what the analysis scripts used): access point
   `arn:aws:s3:us-east-1:184438910517:accesspoint/bdsp-restricted-access-point`, key prefix
   `ICARE_train/training/`. Per-patient directory `ICARE_train/training/<pid>/` with `<pid>.txt` (lines
   `Key: value`: Hospital, Age, Sex, ROSC, OHCA, Shockable Rhythm, TTM, Outcome, CPC) and hourly segments named
   `<pid>_<seg>_<hour>_EEG.mat` (about 60 MB, plus `_ECG` and `_OTHER`) with a WFDB `.hea` of the same stem.
   607 patients; the source sampled 15 patient directories at 48.6 GB, so pull a bounded, resumable subset.
   Profile: `physionet`. Code: `data_io.bdsp_list("icare", ...)`, `bdsp_get`, `parse_icare_txt`.
   Source scripts: `analysis/icare_cohort.py`, `icare_topography.py`, `icare_seq_exclusions.py`.
2. PhysioNet route: the source's dataset registry (`bsde/data_registry/DATASET_REGISTRY.csv`) records I-CARE
   v2.1 at `https://physionet.org/content/i-care/2.1/` as **OPEN, no credentialing, CC BY-NC-SA 4.0**
   (verified by the source on 2026-07-29 from the PhysioNet page). The source also had a WFDB-over-HTTPS adapter
   (`bsde/src/bsde/ingestion/physionet_wfdb.py`). Re-check the licence page yourself before relying on this.

Gotchas: CPC and Outcome are 3-6 month prognostic labels, never contemporaneous state labels. I-CARE is
entirely cardiac arrest, so it cannot test an aetiology contrast (source `CLAUDE.md`).

## MORGOTH

Status: data seen-in-code; model weights documented-only.

- Data: projects access point `arn:aws:s3:us-east-1:184438910517:accesspoint/bdsp-credentialed-projects-ap`
  (source docs also write `s3://bdsp-opendata-credentialed/morgoth1/`, the underlying bucket name),
  prefix `morgoth1/data/internal_dataset/<TASK>/...` (labelled task sets: SEIZURE, IIIC, GPD/LPD/LRDA/GRDA,
  SLOWING, BS, sleep and others) and `morgoth1/data/pretrain/` (SSL `.mat`). A concrete key read by code:
  `morgoth1/data/internal_dataset/GENSLOWING/list_events_gensloing_20241121.xlsx`
  (`analysis/heedb_flag_vs_expert.py`). Profile `physionet`.
- Weights: the 2026-08 notes (`docs/research/36_...md` section V.5) report `morgoth2/models/202605/morgoth/`
  exposing **38 checkpoints, 1.39 GB** (IIIC, SEIZURE, SPIKES, SLOWING, NORMAL, PD, GPD/LPD/GRDA/LRDA,
  SPIKE_LOCALIZATION, SLEEP/PSG variants) plus `baby_morgoth/model/checkpoint-20260516.pth`; the access point
  is not named there (the projects access point is the likely one). Earlier (`docs/MORGOTH_INTEGRATION.md`)
  the weights were unreachable and the public code repo returned 404, and the integration (a
  `MorgothEmbedder`) was never built. There is **no burst-suppression head** in those checkpoints.
- Lesson: MORGOTH's labels could not validate the source's slowing flag (`heedb_flag_vs_expert.py`, ledger R385).

## GCS from the EHR

Status: seen-in-code. Not a separate source: Glasgow Coma Scale and RASS are **HEEDB OMOP `measurement` rows**
selected by regex on `measurement_source_value`
(`glasgow|gcs|rass|richmond|sedation scale|level of consciousness|eye opening|best motor response|best verbal
response|ramsay|arousal`; pseudo-table `measurement_conscious` in `analysis/heedb_omop_extract.py`). Values in
`value_as_number`, time in `measurement_datetime`, patient via `person_id` = `int(BDSPPatientID)`.
`analysis/heedb_command_following.py` builds per-patient assessment worklists (GCS motor/total/eye/verbal and
RASS) and matches them to EEG sessions (loading about 15.4 M label rows took 372 s and 2.6 GB there).
Extract with `PIDS_FILE=<cohort ids> OMOP_OUT=<dir> python analysis/heedb_omop_extract.py measurement_conscious`
(source command; port or use `data_io.iter_omop_batches("measurement", person_ids=...)` and filter yourself).
Lessons: rule 86/92 (GCS-motor and RASS are co-charted by the same nurse in the same round, so RASS is not an
independent incumbent for GCS-motor); rule 6 (check `*_concept_id` is populated before using it).

## Toxic-metabolic encephalopathy cohort

Status: **not found** as a cohort or loader. What exists: HEEDB's `HEEDB_ICD10_for_Neurology.csv` has a chapter
column "Behavioral/Cognitive Syndromes" described as a superset of delirium/encephalopathy
(`docs/HEEDB_UNLOCK.md`, not read by code), the source's aetiology code groups
(`analysis/heedb_bs_ascertainment.py::AETIOLOGY`: anoxic, status, metabolic, sepsis, structural) use
`condition_source_value` prefixes, and BDSP's restricted AP lists `e-cam-s/` (374 delirium cEEG files). The
source commented that toxic/metabolic encephalopathy spares thalamocortical structure
(`heedb_thalamocortical_test.py`) but never built the cohort. Adversarial review on 2026-07-27 found that an ICD code the source had used for "anoxic" (34982) was
actually "Toxic encephalopathy" and removed it.

## BIND, NESI, PROPHET, TDBRAIN, TUAR

Status: **not found.** Case-insensitive and word-boundary searches of the whole source tree (all docs, code,
notes) found no dataset, loader or path for BIND, PROPHET, TDBRAIN or TUAR. NESI appears only as "NESI (2026)",
a published severity index treated as a comparator marker set (`vitaldb_aki/docs/PHASE2C_HEEDB_EEG_TOPOGRAPHIC.md`),
not data. If the plan needs them, access must be researched from scratch.

## SPaRCNet

Status: documented-only. BDSP restricted access point prefix `sparcnet_data/` (IIIC training data; listed in
`bsde/docs/MASTER_PLAN.md` section 9.27 and `docs/research/35_HEEDB_bs_context_findings.md`; no script reads it).
The source's SAP (`docs/research/26_eeg_iic_ncse_SAP.md`) cites SPaRCNet (Jing, Neurology 2023) and the Kaggle
HMS competition as comparators and planned a "HMS/SPaRCNet" cross-site validation that was never run. Use
`data_io.bdsp_list("sparcnet")` (restricted AP, profile `physionet`) to explore.

## TUH / TUEG (TUAB, TUSZ, TUEV, TUAR)

Status: transport seen-in-code (`pipeline/tuh_fetch.py`, `TUHRsyncClient`), **never executed against real data**.
Subcorpora TUAB (abnormal), TUSZ (seizure) and TUEV (events) are mentioned only in notes; TUAR is not mentioned.

- Method: rsync over SSH. User `nedc-tuh-eeg`, host `www.isip.piconepress.com`, remote root `data/tuh_eeg`,
  connectivity probe `data/tuh_eeg/TEST`. Command (from the NEDC email, as coded):
  `rsync -auvxL -e "ssh -i ~/.ssh/id_ed25519" nedc-tuh-eeg@www.isip.piconepress.com:data/tuh_eeg/TEST .`
  `data_io.tuh_rsync_argv` builds this argv without running it.
- Auth: an ed25519 key you registered with NEDC. Approval for the source's author came 2026-07-27.
- Gotchas: port 22 was blocked in the cloud sandbox (443 open), so TUH cannot be pulled from there; `rsync`/`ssh`
  binaries were also absent. Use your own machine. TUH carries **no linked outcomes or diagnoses** (manifest:
  `recording_id, patient_id, edf_path, sfreq, age, sex`; ledger R321), so it cannot replicate an outcome
  association; it can validate a measurement against clinician labels (TUAB/TUSZ). NEDC's rule: delete the data
  when finished, so the code fetched one EDF to scratch at a time.

## CBraMod and other model weights

- CBraMod: seen-in-code. Hugging Face repo `weighting666/CBraMod`, file `pretrained_weights.pth` (19.8 MB),
  sha256 pinned in the source `config.yaml` (copied into `data_io.CBRAMOD`). Verified loading into
  `braindecode.models.CBraMod(n_outputs, n_chans=19, sfreq=200, n_times=6000, patch_size=200,
  return_encoder_output=True)` with 0 missing keys; load with `torch.load(..., weights_only=True)`; verify the
  hash with `data_io.sha256_file`. No token was needed. **Input must be microvolts** (mne returns volts; per-channel
  z-norm destroyed performance: AUC 0.40 versus 0.62).
- MORGOTH: see above (documented-only, weights on BDSP projects AP).
- LaBraM, BIOT, EEGPT: mentioned in literature notes only; no download or loader. Not found.

## VitalDB

Status: seen-in-code (`bsde/src/bsde/ingestion/vitaldb.py`, `bsde/scripts/stream_vitaldb_*.py`).

- Method: public HTTPS API `https://api.vitaldb.net`: `/cases` (table), `/trks` (track list), `/labs`, and
  `/<track id>` (a waveform or numeric track as CSV text). No credentials. Code: `data_io.vitaldb_table`,
  `vitaldb_fetch`, `vitaldb_decode`.
- Licence: api.vitaldb.net is CC BY-NC-SA 4.0 (R&D only); the PhysioNet mirror is CC BY 4.0. The source kept
  derived feature tables out of git for this reason (`.gitignore`).
- Facts used: EEG track `BIS/EEG1_WAV` at 128 Hz in microvolts; monitor tracks `BIS/BIS`, `BIS/SR`, `BIS/SEF`,
  `BIS/EMG`, `BIS/SQI`; agent tracks `Orchestra/PPF20_CE` (propofol Ce), `Primus/INSP_SEVO`, `Primus/INSP_DES`;
  5,871 cases with EEG; one EEG track is about 9.4 MB per case and a full waveform pass is about 55 GB.
- Gotchas (source `LOCAL_SETUP.md` section 8, rules 87 and others): responses are **gzipped regardless of
  Accept-Encoding**; `/cases` starts with a **UTF-8 BOM** (use `utf-8-sig`); `BIS/BIS` emits a literal `0.0` when the
  sensor is detached, so validity is `BIS/SQI > 0` not a ban on 0; `subjectid` is not `caseid` (cluster on the
  patient); a track being present is a fact about the machine, not the patient (check `max(v) > 0`, rule 87).

## PhysioNet DUA projects (and open projects)

Status: seen-in-code (`bsde/scripts/extract_mimic_sedation.py`, `extract_eeg_power_anesthesia.py`,
`physionet_fetch.sh`, `bsde/src/bsde/ingestion/physionet_wfdb.py`).

- Method: HTTPS `https://physionet.org/files/<slug>/<version>/<path>`. Open projects (Sleep-EDF, EEGMMIDB, CAP
  sleep DB, I-CARE 2.1): anonymous, `wget -r -N -c -np -nH --cut-dirs=3` or ranged reads. Credentialed projects
  (MIMIC-IV 3.1, HiRID 1.1.1, SICdb 1.0.8, eICU-CRD, `eeg-power-anesthesia` 1.0.0, `eeg-gaba-anesthesia`):
  **log in for a Django session cookie** (GET `/login/`, read `csrfmiddlewaretoken`, POST username and password,
  keep cookies), then fetch. Code: `data_io.physionet_session`, `physionet_url`, `http_get`. Wrapper
  `bsde/scripts/physionet_fetch.sh <slug> <version> <dest>` is the source's shell version.
- Auth: PhysioNet account (username or email both worked), CITI training, a **signed DUA per project**. Credentials
  from `PHYSIONET_USER`/`PHYSIONET_PASSWORD` or `~/.netrc` entry `machine physionet.org`.
- Gotchas (DAS, measured): HTTP Basic auth (what `~/.netrc` gives `wget`/`curl`) returns 403 on file paths; four
  different situations all return unhelpful codes (wrong path 404, **stale version 403**, Basic-instead-of-session
  403, missing DUA 403), so only a directory listing of the **current version through a session** distinguishes
  them. Landing-page text such as "you must be a credentialed user" is boilerplate present on every restricted
  project page and says nothing about your grant: request a file and read the status. `sicdb/1.0.6/` returned 403
  while `sicdb/1.0.8/` returned 200. PhysioNet's AWS mirror `s3://physionet-open/` (250 projects) is readable but does
  not carry credentialed projects; its S3 grant names an IAM user (`physionet-user`), not the account root.
- Relevance: HiRID, SICdb, eICU and MIMIC carry no EEG (SICdb/HiRID/eICU overviews checked in DAS). The EEG-bearing
  PhysioNet projects the source identified are `eeg-power-anesthesia` (extracted in full), `eeg-gaba-anesthesia`
  (403, needs its own DUA), and the open sets above.

## OpenNeuro and NEMAR

Status: OpenNeuro seen-in-code (`bsde/src/bsde/ingestion/openneuro_s3.py`, `openneuro_brainvision.py`,
`analysis/ds005385_extract.py`, `openneuro_multicohort.py`); NEMAR named in `bsde/docs/SOP_DATA_ACQUISITION.md`
and `TIER1_TARGETS_FOUND.md` as reachable (HTTP 200) with no loader.

- Method: anonymous HTTPS to the public S3 mirror: listing
  `https://s3.amazonaws.com/openneuro.org/?list-type=2&prefix=<ds>/` (XML, paginated) and files
  `https://s3.amazonaws.com/openneuro.org/<ds>/<BIDS path>`. EDF is read with two ranged GETs (header, then a
  window) and BrainVision with the `.vhdr` plus one ranged read of the `.eeg`. Code: `data_io.openneuro_list`,
  `openneuro_url`, `http_get(byte_range=...)`. Auth: none. Version-specific facts via the OpenNeuro GraphQL API.
- Datasets the source used: **ds005385** (Dortmund Vital Study; 608 subjects, 20-70 y, two sessions about 5 y
  apart, four 184 s blocks per session: `task-EyesOpen`/`task-EyesClosed`, `acq-pre`/`acq-post`; 64 ch, 1000 Hz,
  FCz reference); **ds004541** (8 patients, anaesthetic agent not recorded, explicit `loc`/`roc` events,
  CC0); **ds005620** (21 subjects, BrainVision, tasks `awake`/`sed`/`sed2`); **ds006695** (sleep, 19 subjects used);
  **ds007554**.
- Gotchas: `task-EyesOpen` and `task-EyesClosed` are separate files; parse BIDS entities rather than substring
  matching (rule 61: `loc` matched `@loc-300`, `rest` matched `acq-rest`); HBN (`s3://fcp-indi/data/Projects/HBN/
  BIDS_EEG/`, anonymous) has unrecorded units, so only scale-invariant features are valid.

## Chennu propofol sedation

Status: seen-in-code (`bsde/src/bsde/ingestion/chennu.py`, `remote_zip.py`). University of Cambridge Apollo
repository: a 3.69 GB ZIP (166 members; 80 EEGLAB `.set`/`.fdt` pairs, 20 subjects x 4 sedation levels) served by
`api.repository.cam.ac.uk/server/api/core/bitstreams/<id>/content` after a 302 from the `www` host. Read with
HTTP Range requests (EOCD from the tail, central directory, then single members); nothing downloaded whole.
No auth; CC BY 2.0 UK. Gotcha: the sandbox's TLS check failed for `repository.cam.ac.uk` (certificate CN mismatch).
Subject = `name.split("-", 1)[0]` (verified 20 x 4). Units of `EEG.data` are an assumption, not documented.

## Other public deposits the source used (for completeness)

Zenodo (Krause dexmedetomidine/propofol, record 15497531; BCI database records), figshare (DoC resting-state
article 23552964 and others), PhysioNet open sets, BNCI Horizon, DOSE-I. Access is anonymous HTTPS via each
service's public API; see `bsde/data_registry/DATASET_REGISTRY.csv` and `LICENSE_TABLE.csv` for routes and
licence verification status. Bath (BATH-01632) and WBIC data were request-only and never obtained.

## Local path conventions in the source

Caches under `/tmp/eeg_probe/` (ephemeral in the sandbox; a cloud rollback wiped them, rule 38), PhysioNet pulls
under `/tmp/eeg_probe/physionet/`, scratch EDFs under `/tmp/heedb_scratch`, cohort id files
`/tmp/heedb_bs_patients.txt` and `/tmp/heedb_all_patients.txt`. In SortingHat use `data/restricted/` or `local_only/`
(gitignored) on durable disk, and `OMOP_OUT`-style explicit output paths.
