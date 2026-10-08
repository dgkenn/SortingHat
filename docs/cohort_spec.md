# Study 1 cohort specification (operational)

**Status:** DRAFT, 2026-10-08; project-lead decisions D-104 to D-115 applied (section 7). Developed and tested on synthetic data only; the first real diagnostic run (aggregates) led to D-111 to D-115.
**Code:** `sortinghat/cohort/` (config, sources, rules, flow, build, output), `scripts/build_cohort.py`, tests `tests/test_cohort*.py`.
**Design source:** `docs/research_plan_v1.txt` ("Population and index time"), `docs/prereg_study1_sap.md` section 2.
**DECISION_LOG:** D-104 to D-107 and D-111 to D-115 record the choices below (section 7). Section 5 stays the reference list; ids `C-nn` are the spec's own.

## 1. What the plan fixes, and what this code does with it

| Plan statement | Implemented as | Where |
|---|---|---|
| Adults | `AgeAtVisit` (reports_findings, else eeg_metadata) -> `AgeInDaysAtVisit / 365.25` (I0003) -> `StartTime - DateOfBirth` (I0008/I0009), age >= 18.0 at the index EEG | `sources.site_sessions` via `field_audit.merge_eeg`, `build` |
| Acute-care encounter (ICU, inpatient, ED) | D-111/D-112. Real HEEDB has `visit_concept_id` = 0 on every visit and a mostly null `visit_source_value`, and visits are date-only, so the class cannot be read. **Cover**: a visit covers the EEG if the EEG's DATE lies in [visit_start_date - 24 h, visit_end_date + 24 h]. **Acute** = the covering visit is inpatient-length (`visit_end_date > visit_start_date`) OR `ServiceName` contains LTM, ICU, ED, INPATIENT or EMERGENCY (where ServiceName exists; EMU and OR stay excluded). Among covering visits: exact before slack-only cover, inpatient-length first, started on or before the EEG date, then the LATEST start. A concept-id/text class, if non-zero ids ever appear, still decides by itself (no-op today) | `rules.match_visits`, `build` |
| First qualifying EEG per patient | earliest `StartTime(EEG)` among the patient's adult, acute-care, non-OR/EMU sessions that have a start time; one row per patient | `build` |
| t0 = EEG start | `StartTime(EEG)` of reports_findings (S-sites, I0002, I0003); no fallback time is used. Study 1 sites are I0002, I0003, S0001, S0002 (D-113); I0008/I0009 have no EHR rows and are excluded from labelled analyses (their sessions are counted in the flow as a separate excluded block) | `sources`, `build` |
| Minutes 1-11 must exist | clock duration `EndTime - StartTime` >= 660 s (D-115; the metadata duration is not used). The streaming extractor confirms the length from the EDF header (n_records x record duration) and drops windows that do not fit; the key list carries `clock_duration_s` | `build` |
| EEG within 24 h of documented ACI onset; 6/12/48 h sensitivity | onset proxy (section 2); rows kept up to the widest window (48 h); flags `onset_le_6h/12h/24h/48h`; `in_strict` / `in_broad` use 24 h | `build` |
| Strict: GCS <= 11 or FOUR <= 12 within +-6 h of t0 | PRIMARY (D-105): nearest GCS/FOUR in [t0 - 6 h, t0 + 1 h] qualifies -> `severity_strict`, `in_strict`. SENSITIVITY `strict_pm6`: any qualifying score within +-6 h -> `severity_strict_pm6`, `in_strict_pm6` (section 3) | `rules.severity` |
| Broad EHR-phenotype cohort, reported separately | `phenotype` (section 3); `in_broad` contains `in_strict`; "broad only" = `in_broad & ~in_strict` | `rules.phenotype` |
| >= 60% usable data on the 10 hairline electrodes | NOT a metadata step. The key list carries the window; the streaming extractor applies the QC and its counts are appended to the flow (flow.md lists it as pending) | `output.PENDING_STEPS` |

## 2. ACI onset proxy

HEEDB has no structured onset time. The proxy uses timestamps that exist:

1. **Encounter start**: start of the matched visit, moved back to the earliest start of the same person's acute visits that overlap it or end within 6 h before it (an ED visit followed by an admission counts from ED arrival; one hop only).
2. **First abnormal score** (`score_then_visit`, the default): the first GCS total <= 14 or FOUR <= 15 charted in [encounter start, t0]. If there is none, the encounter start. Scores before the encounter start belong to an earlier encounter and are ignored.

`onset_basis` records which was used. Sensitivity rules: `visit_start` (encounter start only) and `score_only` (rows without an abnormal score have no onset and are excluded). `hours_since_onset = t0 - onset` is the time-since-onset covariate; by construction it is >= 0 (asserted).

Known weakness: the proxy is not the clinical onset. A patient admitted for another reason and comatose on day 1 is "onset day 1" even if the EEG question arose on day 6, and the first abnormal score can be later than true onset when nothing was charted. The 6/12/24/48 h flags and the two alternative rules are there to show how much this matters.

## 3. Strict severity and broad phenotype

**Scores.** Rows of `omop_measurement` whose `measurement_source_value` is classified `gcs`, `gcs_eye/motor/verbal` or `four` by `baselines.lexicon.classify_measurement` (the same rule the baselines use, so the cohort and Baseline A read the same rows). Concept ids are not used (they can be zero-filled). Values outside `lexicon.PLAUSIBLE` (GCS 3-15, FOUR 0-16) are dropped, never clipped. When eye, motor and verbal are all charted at the same timestamp and no total is, the total is their sum. A date without a time cannot place a score within +-6 h and is ignored.

**Strict (primary, D-105).** Per instrument, the score closest to t0 within [t0 - 6 h, t0 + 1 h] (the pre-t0 score on a tie) is taken; the patient is strict if the nearest GCS total is <= 11 or the nearest FOUR is <= 12. So only 1 h of post-t0 information defines membership, and the intended-use population is knowable at t0 (closes C-18). **`strict_pm6` (flagged sensitivity).** Any GCS <= 11 or FOUR <= 12 within +-6 h of t0, the plan's wording. A strict patient is always `strict_pm6`; the table keeps both flags, the key list carries both, and the flow shows both ("Cohort membership" and "Strict definitions" splits). Rows that are `strict_pm6` or phenotype-positive but not primary-strict stay in the table so the sensitivity cohort can be analysed; `in_broad` = (primary strict or phenotype) within 24 h. Baselines never see post-t0 rows because every feature goes through `baselines.asof.as_of`. Sedation is not adjusted for.

**Broad phenotype.** A condition starts in [encounter start, t0 + 6 h] with a symptom-level code for impaired consciousness: ICD-10-CM R40.0-R40.4, R41.0, R41.82; ICD-9-CM 780.01, 780.02, 780.09, 780.97 (dots removed, upper-cased). Deliberately NOT used: cardiac arrest, anoxic brain injury (E2-specific), G92/G93.4 encephalopathy (banned label evidence, D-007), seizure codes (E3 positive control), so the broad cohort does not select on a label family. EEG referral indication is not used (no real column; `ReferralIndication` is ASSUMED).

## 4. Flow, disclosure control, outputs

**Order of steps** (units change from EEG sessions to patients at the first-EEG step): 1 EEG sessions; 2 Study 1 site (D-113); 3 patient id resolvable; 4 start time present; 5 age present; 6 age >= 18; 7 a visit covers the EEG date (D-112); 8 acute-care proxy (D-111); 9 not OR/EMU; 10 first qualifying EEG per patient; 11 clock duration known; 12 clock duration >= 11 min (D-115); 13 onset proxy exists; 14 within 48 h of onset; 15 strict severity (primary, or `strict_pm6`) or phenotype. Then disjoint splits: strict / strict_pm6-only / broad-only / sensitivity-window-only; both strict definitions (both / primary only / pm6 only); acute-care basis (visit class / visit longer than a day / ServiceName only). Date-only visits make the encounter start (and so the visit-based onset proxy) accurate to a day only: the encounter start is the START of the visit's first day, so `hours_since_onset` is an upper bound; the 6/12 h windows are meaningful only for onsets taken from timestamped scores.

**Disclosure control** (`cohort/flow.py`; everything printed or written goes through `sortinghat.safe_output`):

- counts < 11 are shown as `<11`;
- a step with < 11 exclusions is merged into the next step (the last one into the previous row), so exact remaining counts cannot reveal it by subtraction; merged rows list all reasons; merging never crosses the sessions -> patients boundary;
- partitions: a hidden part also hides the smallest other part, and parts that are hidden but sum to < 11 also hide the smallest shown part;
- sites whose final count is < 11 are pooled with the smallest other site ("I0002+I0009"); an ALL partition cell is hidden when exactly one site cell for it is hidden; a group is withheld only if its pooled total is < 11;
- output is checked with `assert_aggregate_only` against all record identifiers before it is written. Exact zeros can still be inferred from a split whose other cells are all shown.

**Files** (`--out`, default `out/`, gitignored; `**/local_only/` is also ignored):

| File | Content | Mode |
|---|---|---|
| `out/local_only/cohort_study1.csv` | one row per patient: ids, t0, age, service, visit class, acute basis, duration, onset, onset basis, hours since onset, `onset_le_*h`, lowest GCS/FOUR in the window, nearest GCS/FOUR in the primary window, `severity_strict`, `severity_strict_pm6`, `phenotype`, `in_strict`, `in_strict_pm6`, `in_broad`, `person_id_source` (id before the merge map), `n_unstamped_sessions` | 0600 (dir 0700) |
| `out/local_only/recording_keys.csv` | `SiteID, person_id, SessionID, BidsFolder, EEGFolder, edf_key, task_token_assumed, window_start_s, window_duration_s, in_strict, in_broad`; read with `cohort.read_key_list`; `edf_key` is the documented BIDS EDF key under the access point (first candidate of `data_io.bids_edf_candidates`; the extractor can rebuild the others from the same row's `SiteID, BidsFolder, SessionID, EEGFolder`) | 0600 |
| `out/cohort/flow.md`, `flow.json` | aggregate flow, suppressed | normal |

The key list holds every table row (the 24-48 h rows are needed for the 48 h sensitivity analysis). The extractor reads `edf_key` and the window columns; the cohort table is never needed by it. `task_token_assumed` is True where `EEGFolder` does not exist (all I-sites): the task token defaults to `EEG` there, unverified for continuous EEG.

**Diagnostics.** `--debug-flow` also writes `out/cohort/flow_debug.md/json` and prints the ALL-sites table with every step on its own row (only counts < 11 suppressed; small exclusions are inferable, so it is not for sharing). `python scripts/diag_cohort.py --s3 --out out/cohort/diag.json` prints aggregates only for the visit join and matching (id forms and join rates, null/midnight visit ends, signed hours from the EEG to the nearest visit start/end, coverage under closed / +-24 h / null-end-30 d / both, concept-id and source-value counts, EEG date outside the visit date range, merge-history column NAMES, metadata/clock duration ratio by service). Once it names the merge-history columns, pass `--merge-cols OLD NEW`.

**Memory.** Both scripts are designed for a ~4 GB peak. OMOP `visit_occurrence`, `measurement` and `condition_occurrence` are read row group by row group (`data_io.iter_omop_batches`: only the needed columns, filtered to the adult EEG person ids in Arrow) and reduced per chunk before anything is kept: visits become a compact frame (int64 id, `datetime64[s]` start/end, categorical care setting, about 25 bytes a row) pruned to visits that can matter to the person's EEGs (`rules.prune_visits`); GCS/FOUR rows become (id, `datetime64[s]`, category, float32) pruned to each patient's onset/severity window; conditions are reduced to (id, time) inside the phenotype window. Chunks are concatenated column by column (`sources.concat_frames`), visit matching runs 20,000 sessions at a time, and the diagnostic keeps only per-session and per-site accumulators (exact counters and per-session running minima for the quantiles), never a visit frame. `--max-memory-gb N` sets `RLIMIT_AS` (address space, an upper bound on RSS, so choose it generously) and turns a `MemoryError` into one aggregate error line and exit code 3; both scripts print their peak RSS. `tests/test_cohort_scale.py` generates 2.4 M visit and 5.4 M measurement rows and asserts a peak RSS under 4 GB (about 1.7 GB measured). The real tables are far larger than the test, so read the peak RSS line of the first real run.

**Running.** Synthetic or a local mirror: `python scripts/build_cohort.py --data <dir> --out out`. Real data: `scripts/heedb_run.sh python scripts/build_cohort.py --s3 --out out`, human-run only (`make_client` refuses inside an agent session; the script also refuses restricted paths when `CLAUDECODE=1`). The first real run should be read for the `duration_over_clock_*` checks in flow.md before any other number (section 5, C-07).

## 5. Operational choices that need a DECISION_LOG entry

Entries are not written here. Ids are provisional. "Alt" is the sensitivity switch already in the code.

| Id | Choice | Why it needs a decision | Alt / switch |
|---|---|---|---|
| C-01 | **Index rule.** "First qualifying EEG" = the first adult, acute-care, non-OR/EMU EEG that has a start time. Duration, onset window and severity are then applied to THAT EEG; a patient is not rescued by a later EEG. | The plan does not say whether "qualifying" includes the EEG-level criteria. First-meeting-all-criteria would add patients and choose t0 by outcome-adjacent features. | First EEG meeting every EEG-level criterion (not implemented) |
| C-02 | **Acute-care proxy** (CHANGED by D-111): covering visit longer than a day OR acute ServiceName; class from `visit_concept_id`/text only if it ever becomes non-zero. A session needs a covering visit even when ServiceName is acute, because the onset proxy needs the encounter start. | Defines the population; the real concept ids are all 0. | `acute_classes`, `service_acute`, `--no-service-proxy` |
| C-03 | **ServiceName as an acute-care signal** (CHANGED by D-111): LTM, ICU, ED, INPATIENT, EMERGENCY (substring) count as acute where ServiceName exists (S-sites, I0003); I0002 has no ServiceName and relies on visit length. LTM is not necessarily ICU. | Asymmetric across sites. | `service_acute`, `use_service_proxy` |
| C-04 | **OR and EMU services excluded** where ServiceName exists (S-sites, I0003); no effect at I0002, I0008, I0009 (no ServiceName). | Intra-operative and epilepsy-unit EEGs are inpatient visits but not ACI work-ups; the exclusion is site-asymmetric. | `exclude_services` |
| C-05 | **Encounter chaining**: acute visits ending <= 6 h before the matched visit starts belong to the same encounter (one hop). | Sets ED arrival as encounter start. | `visit_chain_gap_h` |
| C-06 | **ACI onset proxy** = first abnormal score (GCS <= 14 or FOUR <= 15) in [encounter start, t0], else encounter start. No note-derived onset. | "Documented ACI onset" has no structured field. | `--onset-rule visit_start|score_only`; `abnormal_*_max` |
| C-07 | **Recording duration** (CHANGED by D-115) = EndTime - StartTime (clock); metadata `DurationInSeconds`/`RecordingDuration` is not used (I0003 metadata/clock median 2.35). flow.md still prints the metadata/clock ratio per site as information. Gaps and EDF+D remain the extractor's business. | Metadata duration is unreliable. | |
| C-08 | **Strict severity** (CHANGED by D-105): primary = nearest GCS/FOUR in [-6 h, +1 h] of t0, pre-t0 on ties, per instrument; `strict_pm6` = any score within +-6 h as sensitivity. Totals as charted, else the sum of three components at one timestamp; implausible values dropped; no sedation adjustment; RASS/NESI do not qualify. "Nearest" is read per instrument, not among qualifying scores only. | Plan gives thresholds, not which value or window. | `--score-rule any`; `score_before_h`, `score_after_h`, `pm6_window_h` |
| C-09 | **Windows**: rows kept to 48 h; primary flag at 24 h; 6/12/48 h are flags, not separate cohorts. | The 48 h analysis is wider than the 24 h primary. | `onset_primary_h`, `onset_sensitivity_h` |
| C-10 | **Broad phenotype proxy** = symptom codes R40.0-4, R41.0, R41.82 / 780.01, .02, .09, .97 in [encounter start, t0 + 6 h]; excludes arrest, anoxic, encephalopathy and seizure codes. Broad contains strict. Billing diagnoses may be stamped at discharge, so the window misses some. | Defines the secondary cohort and its spectrum. | `phenotype_after_h`, `rules.PHENOTYPE_CODES` |
| C-11 | **Patient identity** = BDSPPatientID (OMOP `person_id`), taken as globally unique, AFTER applying `PatientMergeHistory/` (D-114): `MergedBDSPPatientID` -> `BDSPPatientID` (real columns, with `LineNBR` and `BDSPLastModifiedDTS`); chains are followed; when a retired id has several survivors the row with the latest `BDSPLastModifiedDTS` wins (across files too). Visits/scores/conditions of retired ids are re-keyed. If the prefix is absent a one-line notice is printed and written to the flow checks; unrecognised columns are not applied (`--merge-cols OLD NEW` overrides). | Merged records would be counted twice and "first EEG" would be wrong. | `sources.merge_rows` |
| C-12 | **Missing start time**: such sessions are dropped before choosing the first EEG; no start fallback (ReportBeginDTS, ReportEEGDateTime, CreationTime are not t0). `n_unstamped_sessions` and a flow split show how many included patients have one. | A patient's true first EEG may be the unstamped one. | |
| C-13 | **Age**: missing age is excluded, not imputed from `HEEDB_patients`; age is at the index EEG. I0008/I0009 age depends on a shifted `DateOfBirth`. | | |
| C-14 | **Score identification by source text** (shared lexicon with the baselines); concept ids unused. | Concept ids may be zero-filled; the audit's newer concept-class mapping is not used here. | |
| C-15 | **Disclosure rules of the flow** (merge, pool, partition suppression; section 4). | Changes what the flow can show; small sites are pooled, so the per-site >= 300 check (D-024) must be read from the local table, not the flow. | |
| C-16 | **Flow stops before EEG QC.** The minimum-channel/usable-data step (10 hairline electrodes, >= 60% usable) is added by the extractor; the cohort size entering the models is the post-QC size. | Plan lists it as a population criterion. | |
| C-17 | **Baseline D is dropped** (D-107): HEEDB has no EEG referral-indication source (no column in any real header), so the Baseline D sensitivity analysis is removed from the SAP. The cohort does not record the indication. (`baselines/` still builds D from `ReferralIndication` for synthetic data; that code is not owned here.) | | |
| C-18 | **Selection on post-t0 information.** Resolved for the primary strict cohort by D-105 (only +1 h after t0); remains for `strict_pm6` (up to +6 h, labelled sensitivity) and for the broad phenotype window (t0 + 6 h). | | |
| C-19 | **Visit matching rules** (added after the first real run excluded 196,283 of 257,401 adult sessions at "no visit covering the EEG start"): (1) any covering visit, not only the one with the latest start; chosen exact-before-slack, then ICU > ED > Inpatient > Outpatient > unclassified, then latest start; (2) a null visit end is open for 30 days after the start (`--open-visit-days`; the old rule treated it as open forever); (3) a date-only end (a date column, or exactly 00:00:00) is the end of that day, and `visit_start_date`/`visit_end_date` fill null datetimes; (4) an end before the start is a zero-length visit; (5) `person_id` is cast to int64 (text and leading zeros tolerated); (6) slack (`--visit-slack-h`, default 24 since D-112): widen every interval; slack-only matches are labelled `visit_match = slack` and give an encounter start of t0 when the visit starts after the EEG. SUPERSEDED in part by D-112: visits are treated as date-only (`visit_dates_only`), the cover is [start date - 24 h, end date + 24 h], and ties go to the latest start on or before the EEG date. | Changes who is "acute-care"; slack trades precision for coverage. | `CohortConfig.visit_slack_h / open_visit_days / date_only_end_of_day` |
| C-20 | **Study 1 sites** (D-113) = I0002, I0003, S0001, S0002; I0008/I0009 have no OMOP rows for their patients, so they are excluded from labelled analyses. Their sessions are counted in the flow and shown as a separate excluded block (never pooled with a study site). `--all-sites` turns the restriction off (synthetic tests). | Changes the site count entering the >= 3 sites x >= 300 candidates check (D-024). | `CohortConfig.study_sites` |

## 6. Limits and what is not verified

- Synthetic data only. Real-data behaviour of the visit classifier, the score text rules, the duration unit and the phenotype window is unknown.
- `visit_occurrence` is about 512 M rows in one 9 GB part; the loader streams it filtered to adult candidate person ids inside Arrow, but memory and run time on the real table are untested.
- Overlapping visits are resolved by "latest start <= t0" (the audit's rule), not by any-overlap.
- The recording key list assumes the BIDS EDF key pattern; the `cEEG` task token cannot be derived outside S0001/S0002.
- The three tables `imaging`, `omop_note`, labs are not used by the cohort.

## 7. Project-lead decisions applied (2026-10-07)

| DECISION_LOG | Content | Spec rows |
|---|---|---|
| D-104 | Operational choices accepted as specified here: C-01 to C-07, C-09, C-10, C-12 to C-16 | those rows |
| D-105 | Strict window [-6 h, +1 h] primary, +-6 h sensitivity (`strict_pm6`) | C-08, C-18 |
| D-106 | Patient merge history applied when available | C-11 |
| D-107 | Baseline D dropped | C-17 |
| D-111 | Acute-care proxy: covering visit longer than a day OR acute ServiceName; concept-id class a no-op | C-02, C-03 |
| D-112 | Visit cover [visit_start_date - 24 h, visit_end_date + 24 h]; ties to the latest start on or before the EEG date | C-19 |
| D-113 | Study 1 sites I0002, I0003, S0001, S0002; I0008/I0009 an excluded block | C-20 |
| D-114 | Merge history `MergedBDSPPatientID` -> `BDSPPatientID`, latest `BDSPLastModifiedDTS` wins | C-11 |
| D-115 | Recording duration = clock EndTime - StartTime; the extractor confirms from the EDF header | C-07 |
