# Public-repo exposure check: patient-level HEEDB / I-CARE results in the earlier research repo

Scope: `github.com/dgkenn/codex-playground-` (a **public** repo), tip of
`origin/claude/research-program-continuation-5upe54` (2026-08-10), alongside `origin/research` and
`origin/claude/heedb-eeg-phenotype-discovery-2mnwzx` (both at the older 2026-08-04 commit `4aff42c`). The same two suspicious-looking files exist on all three refs.

Method: read-only `git ls-tree` / `git cat-file`. For every tracked `csv`, `tsv`, `npz`, `npy`, `json`, `jsonl`,
`parquet`, `zip`, `pkl`, `h5`, `feather` file: CSV header line (column names) and line count; `npz`/`zip` member
names only; `json`/`jsonl` structure only (top-level type, and for the one HEEDB-named JSON its top-level field
names after checking none contained digits). **No data row, value or ID is reproduced in this file.** Limits:
the clone is shallow (50 commits), so files deleted in older history were not checked; JSON content was not
inspected beyond structure.

Inventory scanned on the newest ref: 650 files (205 csv, 423 json, 19 npz, 2 jsonl, 1 zip). Zero files of type
parquet, pkl, h5, feather, npy, tsv.

## Result: 1 suspicious file

| # | Path | Format | Rows | Header columns | Why suspicious |
|---|---|---|---|---|---|
| 1 | `bsde/results/icare_labels.csv` | CSV with 10 leading `#` comment lines (not plain CSV) | 607 data rows (618 lines in file) | `patient_id`, `hospital`, `age`, `sex`, `rosc`, `ohca`, `shockable_rhythm`, `ttm`, `outcome_prognostic`, `cpc_prognostic` | One row per I-CARE patient: patient identifier, hospital, age, sex and clinical outcome in a public repo |

Context for #1:

- The file's own comment block says "per-patient I-CARE metadata parsed from each patient's `<pid>.txt` file" and
  warns that outcome and CPC are prognostic (3-6 month) variables. It is **not** described as aggregate or derived.
- Mitigating, from the source's own `bsde/data_registry/DATASET_REGISTRY.csv` and `LICENSE_TABLE.csv`: I-CARE v2.1
  on PhysioNet is recorded as **open, no credentialing, CC BY-NC-SA 4.0, redistribution of raw data permitted**
  (verified by the source on 2026-07-29 from the PhysioNet pages; unconfirmed by this check). The identifiers are
  the dataset's own pseudonymous patient numbers. Note the source's analysis scripts read I-CARE through the
  **BDSP restricted access point**, so check which terms govern your copy before treating the open licence as
  settled.
- It was committed by an automatic checkpoint (`b3e0ca8`, 2026-08-07), not by a deliberate release decision.
- `bsde/results/.gitignore` states "Still never committed: raw EEG, credentials, anything patient-identifying",
  and the same policy block says feature tables are committed freely. File #1 sits in tension with the former.

## Not suspicious (checked)

- **No HEEDB patient-level results file is tracked.** No tracked csv/json/npz has HEEDB-style columns
  (`BDSPPatientID`, `SiteID`, `person_id`, `BidsFolder`, `StartTime(EEG)`) or a name containing heedb, omop, bdsp,
  morgoth, S000x/I000x, burden or aetiol, other than item below. The HEEDB analyses wrote to `/tmp/eeg_probe/`,
  which was never committed. `CLAUDE.md` says so: the cached tables are "credentialed patient-derived data ...
  Nothing there is in git and nothing should be". The root `.gitignore` carries "Raw data working area - PHI
  (HEEDB/MIMIC/eICU under DUA); NEVER commit" and also excludes `mimic_*.csv`, MGH-derived windows and VitalDB
  re-extractions. Analysis script docstrings repeat "De-identified outputs only (aggregate stats; no PII rows/dates
  printed or committed)".
- `bsde/results/e204_heedb_command_following.json` (1,362 bytes): a single experiment report; top-level fields
  are `experiment, features, perms, n_rows_raw, n_rows, n_patients, obeys_rate, minority_fraction, g1,
  g5_minutes_before, g2, g4, primary, descriptive_intercept_only, verdict, why`. Aggregate result summary, no ID
  keys.
- The other **189 CSVs with subject/recording/case-id-like columns** are derived features of **public** deposits
  (VitalDB, Chennu, Dreyer, Stieger, EEGMMIDB, Sleep-EDF/CAP, ds004541, ds005620, ds006695, ds007554, HBN, LEMON,
  DOSE-I, figshare DoC), not HEEDB or I-CARE. They carry their own licences (VitalDB's is CC BY-NC-SA 4.0; the
  source ignored some VitalDB re-extractions for that reason). Per-dataset exposure is outside this check.
  The 19 `npz` files and the JSON files under `bsde/results/ds006695_partial/` are ds006695 (OpenNeuro) epoch data.
- `vitaldb_aki/cache/*.csv` (9 files with `caseid`): VitalDB cases, public deposit.

## Other exposure items found while checking (not data files)

1. **HEEDB-format patient identifiers in code.** Two tracked source files contain a string shaped like a real
   HEEDB BIDS folder or patient id: `analysis/heedb_edf_range.py` (a default example key used in a `__main__`
   smoke test) and `pipeline/stream_fetch.py` (a docstring example). `tests/test_heedb_age_join.py` and
   `tests/test_heedb_bdsp.py` use clearly synthetic fixtures (sequential made-up ids). I did not verify whether the
   two non-test strings are real; treat as possibly real pseudonymous ids and scrub.
2. **Account and identity metadata in public docs** (values not reproduced here): the author's AWS account number
   and the fact that the BDSP keys resolve to the account root (`bsde/docs/DEPOSIT_ACCESS_STATUS.md`,
   `bsde/docs/MASTER_PLAN.md` section 9.27, `docs/HEEDB_UNLOCK.md`); the email address and NEDC SSH public-key
   fingerprint (`docs/CREDENTIALS.md`); the PhysioNet login names tried (`DEPOSIT_ACCESS_STATUS.md`). None is a secret
   key, but they identify the credentialed account. The same document records that a password was pasted into a
   chat transcript and "should be rotated". The shared BDSP access-point ARNs (account 184438910517) are BDSP's
   own public endpoints.
3. **Aggregate statistics about HEEDB in public prose** (cohort sizes such as the 49,232-patient EEG cohort,
   AUCs, counts by aetiology; `CLAUDE.md`, `docs/research/*.md`). These are aggregates; under SortingHat's
   small-cell rule (n < 11 suppressed) no cell this small was noticed, but only a skim was done.

## Do the docs claim these files are aggregate or derived?

- For `icare_labels.csv`: no. Its header calls it "per-patient"; no doc describes it as aggregate.
- For HEEDB generally: yes, by policy. `CLAUDE.md` (cache section), the root `.gitignore` and individual script
  docstrings all assert patient-derived data is kept out of git and outputs are de-identified aggregates. This check
  found that assertion **true for HEEDB** (no patient-level HEEDB file tracked) and **not applicable but in tension**
  for the I-CARE per-patient table.
- `bsde/results/.gitignore` records a policy change on 2026-07-29 to commit feature tables "freely, including
  partial ones", with the stated exception "anything patient-identifying". A future HEEDB-derived per-recording
  feature table written under `bsde/results/` would have been committed automatically by the checkpoint loop.
  None was found, but the guard is a convention, not a rule enforced in code.

## Suggested actions for the repo owner (not performed)

1. Decide whether `bsde/results/icare_labels.csv` should remain public (open-licence argument) or be removed and
   regenerated from `analysis/icare_cohort.py`; if removed, history still holds it (needs a history rewrite if that
   matters).
2. Scrub the two possibly-real identifier strings in code.
3. Consider rotating the credential that was pasted into a transcript, and review the account-identity details in
   the public docs.
4. For SortingHat itself: the existing `.gitignore` plus `data/restricted/`/`local_only/` convention and
   CLAUDE.md rule 6 already cover this; keep `scripts/heedb_run.sh` and `data_io` human-run only.
