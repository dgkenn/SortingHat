# HEEDB / BDSP data access (ported from the earlier HEEDB research program)

> See also `docs/data_access.md` (all datasets) and `docs/credentials_template.md` (placeholders only).

Source: `github.com/dgkenn/codex-playground-`, branch `claude/research-program-continuation-5upe54`
(commit date 2026-08-10, newest of the three candidate refs; `research` and
`claude/heedb-eeg-phenotype-discovery-2mnwzx` are an older commit, 2026-08-04). Files read: `LOCAL_SETUP.md`,
`docs/CREDENTIALS.md`, `CLAUDE.md` (credentials section and error catalogue), `docs/HEEDB_UNLOCK.md`,
`docs/RUNBOOK.md`, `docs/LESSONS.md`, `scripts/heedb_run.sh`, `scripts/bdsp_bootstrap.sh`,
`common/awsenv.py`, `pipeline/stream_fetch.py`, `config.yaml`, `analysis/heedb_omop_extract.py` and the
`analysis/heedb_*.py` scripts. Nothing here was verified against live data by this port: every fact is
"seen in the source code or docs", and `docs/heedb_schema_real.md` marks the confirmation level per column.

**This document contains no secrets and no data values.** Credentials are described by variable name only.

## 1. How access works

HEEDB is not a plain public bucket. It is served from **S3 access points** that only the AWS identity you
registered with BDSP can read.

| Item | Value (from the source project) |
|---|---|
| Transport | boto3 (`list_objects_v2`, `get_object`, ranged `get_object`) over HTTPS; the AWS CLI works too |
| HEEDB clinical access point | `arn:aws:s3:us-east-1:184438910517:accesspoint/bdsp-credentialed-access-point`, used as the `Bucket=` argument |
| Bucket argument | The full ARN goes in `Bucket=`. A bare access-point name returns `NoSuchBucket`, which reads like a permissions problem (source `MASTER_PLAN.md` 9.27). |
| Region | `us-east-1` |
| Client config that worked | `Config(s3={"payload_signing_enabled": False})`, long read timeout (600 s), `retries={"max_attempts": 8, "mode": "standard"}` |
| Credentials | your own AWS key pair for the AWS account BDSP approved; resolved by the normal boto3 chain from `~/.aws/credentials` |
| Profile name used in the source | `physionet` (identical keys to `default`; it was just the name BDSP's instructions used). Any name works if you pass it. |
| Other BDSP access points in use | I-CARE: `.../accesspoint/bdsp-restricted-access-point`, prefix `ICARE_train/training/`. MORGOTH label sets: `.../accesspoint/bdsp-credentialed-projects-ap`, prefix `morgoth1/data/internal_dataset/`. |
| Your own alias | The BDSP Cloud Credentials dashboard issues each approved user an access-point alias (shape `bdsp-credentialed-ac-...-s3alias`). `config.yaml` and `docs/RUNBOOK.md` in the source say either that alias or the shared ARN works with your keys. In `sortinghat/data_io.py` set `HEEDB_ACCESS_POINT` to use an alias. |

Prerequisites (source `docs/RUNBOOK.md` section 0, from bdsp.io/about/howto_accessdata):
an AWS account whose 12-digit ID is registered in the BDSP Cloud Credentials dashboard, the signed
project-specific DUA, and CITI "Data or Specimens-Only Research" training with the certificate uploaded.

### Top level of the access point

`ECG/  EEG/  EHR/  Imaging/  NAX/  OMOP/  PSG/  PatientMergeHistory/` (source `docs/HEEDB_UNLOCK.md`).

### Key patterns used by the source scripts

| What | Key or prefix (under the access point) |
|---|---|
| Per-site EEG session catalog | `EEG/eeg-metadata/{SITE}_eeg_metadata_2026_04_30.csv` (scripts hard-code the date; `data_io.resolve_key` globs the prefix and takes the newest) |
| Per-site EEG report labels | `EEG/HEEDB_Metadata/{SITE}_EEG__reports_findings.csv` (double underscore; a single-underscore variant is also tried as a fallback in one script) |
| Patient master table | `EEG/HEEDB_Metadata/HEEDB_patients.csv` |
| Neurology ICD-10 chapters per patient | `EEG/HEEDB_Metadata/HEEDB_ICD10_for_Neurology.csv` |
| Medication ATC groups per patient | `EEG/HEEDB_Metadata/HEEDB_Medication_ATC.csv` |
| EDF signal for one session | `EEG/bids/{SITE}/{BidsFolder}/ses-{SessionID}/eeg/{BidsFolder}_ses-{SessionID}_task-{cEEG or EEG}_eeg.edf` |
| Merged OMOP tables (parquet parts) | `OMOP/Merged/{table}/*.parquet` (181 parts for `condition_occurrence`, 375 for `drug_exposure`, 554 for `measurement`; `visit_occurrence` is one 9 GB part) |
| Per-site OMOP / EHR | `OMOP/{SITE}-OMOP/`, `EHR/{SITE}-EHR/` (named in `HEEDB_UNLOCK.md`; the source scripts used `OMOP/Merged/` instead) |
| I-CARE | `ICARE_train/training/{pid}/{pid}.txt` (key: value lines) under the restricted access point |

`{SITE}` is a de-identified code. Seen: `S0001`, `S0002`, `I0002`, `I0003`, `I0009` (`config.yaml`), and
`I0001`-`I0009` as a range in `HEEDB_UNLOCK.md`. The source `config.yaml` never resolved codes to hospital
names ("TO-CONFIRM"). Report files and `DateOfDeath` were only ever used for `S0001` and `S0002`.

Task token in the EDF name: `cEEG` when the metadata `EEGFolder` value starts with `ceeg` (any case),
otherwise `EEG` (`heedb_bs_calibrate.bids_key`, "the working one"). `BidsFolder` is `sub-<SITE><BDSPPatientID>`.
Real-data finding (7 of 20 streamed recordings `not_found`): the documented name is not always the object name.
`data_io.resolve_edf_key` therefore tries the documented key, the other task token, a task-less name and SessionID spellings
(`12.0` -> `12`), then lists ONLY that recording's own `ses-<id>/eeg/` folder (`Delimiter='/'`) and takes the `.edf` whose name
starts with `<BidsFolder>_ses-<id>`. `scripts/diag_eeg_paths.py` reports, as aggregates, which pattern resolved how many recordings.

`OMOP person_id` equals `int(BDSPPatientID)`. The source verified a 100 percent match of all 34,620 `S0001`
EEG patients against a 15M-row `person` table (`heedb_omop_extract.py` docstring), and every aetiology
script joins by `int(BDSPPatientID)`.

## 2. Setup on your own machine

1. Get access (section 1 prerequisites). Keep the secret key out of the repo, out of shell history that is
   synced, and out of any agent session.
2. Install: `pip install boto3 pyarrow pandas` (also `mne` if you read EDFs).
3. Configure credentials, mode 600, outside the repo. `~/.aws/credentials`:

   ```
   [physionet]            # any profile name; the source used "physionet" and also wrote the same keys under [default]
   aws_access_key_id = <your key id>
   aws_secret_access_key = <your secret>
   ```

   `~/.aws/config`:

   ```
   [profile physionet]
   region = us-east-1
   s3 =
     payload_signing_enabled = false
   ```

   (Example values are placeholders. Never paste real keys into this repo, a chat, or an agent session.)
4. Smoke test the access (human-run, plain terminal; prints only a count, no keys or rows):

   ```bash
   HEEDB_AWS_PROFILE=physionet scripts/heedb_run.sh python - <<'PY'
   from sortinghat import data_io
   s3 = data_io.make_client()
   print(len(data_io.list_keys(s3, data_io.HEEDB_METADATA_PREFIX)), "objects under EEG/HEEDB_Metadata/")
   PY
   ```
5. Run every restricted job through `scripts/heedb_run.sh <command>` (see section 3). It refuses to run when
   `CLAUDECODE=1`, in line with `CLAUDE.md` rule 2.
6. Keep outputs aggregate-only (`CLAUDE.md` rule 3) and any local extracts under `data/restricted/` or
   `local_only/` (gitignored). The source project's caches lived in `/tmp/eeg_probe/` and a
   container reaper rolled them back repeatedly; on your own machine point them somewhere durable.

Disk and compute from the source's `LOCAL_SETUP.md`: Linux or WSL2, 200 GB+ free is comfortable for full
extractions, 16 GB RAM works (the 3.3 GB OMOP condition table is the largest single object), every
extraction is network-bound not compute-bound, GPU not needed for anything except foundation-model work.

## 3. The sandbox 403 placeholder-key gotcha

Some sandboxes (the Claude Code cloud container was the case) export **placeholder** `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY` for their outbound agent proxy: the key id was a 14-character token beginning `prox`,
not a real 20-character `AKIA...`/`ASIA...` id. boto3 resolves static environment credentials **before**
profile credentials, so every script silently authenticated as the stub. The BDSP access point then
returned 403 / `InvalidAccessKeyId`, which reads exactly like expired or revoked credentials. It cost the
source project hours, three separate times (catalogue rule 8; rule 36, "third occurrence").

What did not work: `unset` in `~/.bashrc` (agent tool shells are non-interactive and never source it).

What worked:

- `scripts/heedb_run.sh <cmd>`: runs the command with the stub variables removed (and leaves `AWS_CA_BUNDLE`
  and `HTTPS_PROXY` alone so the proxy still carries traffic). This port only drops them when the key id
  provably is not an AWS key id (not `^(AKIA|ASIA)[A-Z0-9]{16}$`); the source wrapper dropped them
  unconditionally and also did not unset `AWS_SESSION_TOKEN`.
- In Python, `sortinghat.data_io.drop_placeholder_env()` / `make_client()` do the same test as the source's
  `common/awsenv.py` (drop only if the key is not shaped like a real one and a credentials file exists).
- Store real keys under **non-standard variable names** (the source used `BDSP_AWS_ACCESS_KEY_ID` /
  `BDSP_AWS_SECRET_ACCESS_KEY` in the cloud environment settings and had a `SessionStart` bootstrap write
  `~/.aws/credentials` from them). Not needed on a normal machine.
- Diagnose per credential source, not globally. A 403 **with** the wrapper is a real credential or
  registration problem; a 403 **without** it is almost always the stub collision. Test with
  `sts get-caller-identity`:
  `scripts/heedb_run.sh python -c "import boto3; print(boto3.Session().client('sts').get_caller_identity()['Arn'])"`.
- The source bootstrap also probed the ambient environment as well as the profile, because "profile works"
  said nothing about what an ordinary script would see (2026-07-28: bootstrap reported WORKS, the analysis
  then died with `InvalidAccessKeyId`).

Other access facts from the source:

- PhysioNet DUA projects (HiRID etc.) are a **separate credential** (`~/.netrc` for physionet.org over HTTPS)
  and the BDSP keys do nothing for them. Not needed for HEEDB.
- `config.yaml` key `data.s3.catalog_key = "HEEDB/index.tsv"` was stale (404). The real catalog is the
  per-site `EEG/eeg-metadata/{SITE}_eeg_metadata_*.csv`.
- TUH (NEDC) uses rsync over SSH on port 22 with your own registered key; the cloud sandbox blocked port 22.
  TUH has no linked outcomes or diagnoses (manifest: `recording_id, patient_id, edf_path, sfreq, age, sex`;
  ledger R321), so it cannot replicate an outcome association even with access. Out of scope for HEEDB work.
- Shell heredocs failed in the source's cloud bash wrapper (`docs/LESSONS.md`); the smoke test above uses one,
  which is fine in a normal terminal.

## 4. HEEDB-specific lessons from the source error catalogue

Numbers are the rule numbers in the source `CLAUDE.md` section "SOP: the error catalogue" (100+ numbered
entries; these are the ones that bear on HEEDB/OMOP/S3 data handling, not statistics in general). Quoted text
is from the source, with dashes and typographic punctuation simplified.

**Access and environment**

- **Rule 8.** "An access failure may be credential *precedence*, not expiry. Diagnose per credential source."
- **Rule 36.** "Credential precedence, third occurrence. The sandbox exports `AWS_ACCESS_KEY_ID` as a
  14-character `prox...` proxy token that outranks `~/.aws/credentials`, and the failure reads as
  `InvalidAccessKeyId` - indistinguishable from expiry. ... An `unset` in `~/.bashrc` does nothing here - the
  tool's shells are non-interactive and never source it."
- **Rule 38.** A cloud container snapshot-restore rolled back `/tmp` and `.git`. "Before declaring work lost to
  a container rollback, `git fetch` and compare HEAD against the remote. The rollback restores an OLD COMMIT,
  not an empty tree." Caches in `/tmp/eeg_probe/` do not survive; extractions are resumable and the discipline
  was commit-and-push at every artifact boundary. Not relevant on a durable machine except as a habit.
- **Rule 56.** Two concurrent writers on one extraction CSV gave "931 rows where 710 were expected, 419
  duplicated `recording_id`s". "Before relaunching or resuming any background job, confirm it is dead ... any
  resumable extractor should de-duplicate on its key when it loads."

**Empty, missing and unpopulated data**

- **Rule 5.** "Empty is not evidence of absence until the filter has been shown capable of matching something.
  Assert non-empty, or assert the filter matches a known positive." (HEEDB case: an OMOP regex filter over
  `procedure_source_value`/`observation_source_value` matched nothing because those columns hold numeric
  billing codes such as 36415 or 99214, so the extraction returned a clean-looking empty file. Names live in
  the `concept` table and must be joined through `*_concept_id`; see `heedb_omop_extract.py` `ID_FILTER_*`.)
- **Rule 6.** "Check that a `concept_id` column is populated before designing around it.
  `observation_concept_id` was 100 % zero; one `Counter` over 100k rows would have shown it in seconds."
- **Rule 7.** "Before choosing an administrative table as an instrument, ask what makes a row appear in it.
  Billing tables see reimbursable acts; state tables see charted statuses; neither sees decisions.
  `procedure_occurrence` contains zero extubations because extubation is not separately billable."
  Related, from the extractor's comments: `drug_exposure` vasopressor end times are stamped at death by the
  charting system, and DNR codes in `condition_occurrence` document chronic status (median 42 days to death).
- **Rule 14.** "Report exclusions and check whether they are outcome-related. Both major exclusions in this
  project turned out to be." (HEEDB case: `heedb_omop/` was extracted only for burst-suppression patients with
  an ascertained death; `heedb_omop_quant/` is the full 49,232-patient cohort. Using the wrong one was "a
  scientific error".) Death is also ascertainment-limited: the OMOP `death` table covers about 45 percent of
  the 7,323-patient burst-suppression cohort (3,304), so absence of a death row is not survival.
- **Rule 32.** "Before comparing two predictors, check that both VARY in the stratum you will compare them in."
  A measurement's availability (a flag present in 100.0 percent of the patients who carry the measure) defines
  a stratum selected on the thing that makes the measurement possible.
- **Rule 61.** "Substring-matching a structured identifier is not parsing it". Parse BIDS/ID fields
  (`sub-`, `ses-`, `task-`) rather than testing `token in name`. The HEEDB analogue: parse
  `BidsFolder`/`SessionID`, do not search the key text.

**Time and per-patient structure**

- **Rule 10.** "Any per-patient aggregation over repeated measurements is look-ahead until proven otherwise.
  `max()`, `or` and bare assignment across rows all leaked. The pattern was in 17 scripts." The source
  analyses keep one record per patient (the first EEG) for outcome work, and drop rows where the death date is
  more than a day before the EEG as a data error.
- **Rule 27.** "A mask that compresses out bad samples glues time together. ... One recording in 24 had a
  1,817 s hole closed up." Verify the time axis before modelling temporal evolution.
- **Rule 28.** "Two measurements separated in space or time are not thereby measuring different things."
- **Rule 22.** "A validity figure living in a code comment is not a validity figure." A burden-detector
  "AUC 0.829" in a comment was irreproducible; measured properly on 27,948 matched recordings it was 0.749.

**Signal handling (source `docs/LESSONS.md` and `HEEDB_UNLOCK.md`, not in the numbered catalogue)**

- `mne` returns EDF in **volts**; foundation models (CBraMod) expect **microvolts**: multiply by 1e6.
  Per-channel z-scoring destroyed the amplitude scale (within-site AUC 0.40 versus 0.62 with correct scaling).
- Frozen embeddings encoded hospital almost perfectly (site-probe AUC 0.961). A cross-site claim needs a
  nonlinear site-invariance gate; a linear gate gave false assurance after ComBat/CORAL.
- EDFs are about 1 GB each for ICU cEEG. The source read a bounded window with HTTP range requests (EDF has a
  fixed-layout header) rather than downloading (`analysis/heedb_edf_range.py`).
- `AgeAtVisit` in the per-site metadata is "largely empty"; adult filtering must join `HEEDB_patients.csv`
  (`AgeAtVisitAvg`, `Sex`) on site + patient id (`docs/HANDOFF.md`, `tests/test_heedb_age_join.py`).

## 5. What `sortinghat/data_io.py` gives you for HEEDB (other datasets: `docs/data_access.md`)

`make_client(profile=None)` (refuses in an agent session), `list_keys`, `resolve_key`, `read_csv_table(name,
site)`, `finding_present` (the `None`/`nan`/empty convention), `bids_edf_key`, `bids_folder_for`, `omop_parts`,
`iter_omop_batches(table, person_ids=...)` (ranged parquet reads, filtered inside Arrow), `S3RangeFile`.
Table names: `eeg_metadata`, `reports_findings`, `heedb_patients`, `icd10_neurology`, `medication_atc`, and
`omop_<table>` for the merged OMOP tables. Unit tests (`tests/test_data_io.py`) use a fake client and
synthetic bytes only.
