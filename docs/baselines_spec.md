# Clinical baselines A-D: feature specification (for the SAP)

Reference implementation: `sortinghat/baselines/`; tests: `tests/test_baselines.py`, `tests/test_baselines_leakage.py`.
Plan source: "Clinical baselines" table and "Population and index time" (`docs/research_plan_v1.txt`); SAP sections 4.1, 9 and 10;
DECISION_LOG D-014, D-015, D-032, D-034. Developed on synthetic data only (CLAUDE.md rule 1). Items marked **[OP]** are
operationalizations the plan does not state and need a DECISION_LOG entry before freeze; **[ASSUMED]** marks fields or codings
that no source script has touched and a human must verify on the real tables.

## 1. The t0 gate

t0 = `StartTime(EEG)` of the first qualifying EEG (same cohort rule as `field_audit.build_candidates`).

Every clinical observation becomes a row of one long event frame with an **availability time** `t_avail` (when a clinician could
have seen it). `asof.as_of(events, t0)` is the **only** place that compares an event with t0:

1. keep rows with `t_avail <= t0` (inclusive; a row with unknown availability is dropped, never assumed early);
2. censor intervals: a drug row whose recorded end is after t0 becomes `ongoing`, its end is removed and its `quantity` is set to
   NaN (the stop time and total dose of a running infusion are future information);
3. attach `t0`, `hours_since_event`, `hours_since_avail`, and drop the raw end column.

`build_feature_set` calls it exactly once; extractors receive only its output and `assert_masked` re-checks the invariant.
Static fields (age, sex, referral indication) are not events. Tests that enforce this: injected post-t0 events in every domain leave
the matrix bit-identical (offsets from 1 microsecond to 400 days); no sentinel value reaches any feature; editing a running
infusion's end/quantity is invisible; a lab collected before t0 but resulted after t0 is excluded; a test scans the package so that no
module other than `asof.py` compares an event time with t0; mutation checks of `as_of` fail the suite.

## 2. Baselines (nested column subsets; provenance table gives the baseline of each column)

| Baseline | Adds | Groups (feature prefix) |
|---|---|---|
| A | age, sex; GCS (+eye/motor/verbal), FOUR, RASS, NESI; t0 sedative/opioid exposure | `demo__`, `score__`, `sed__` |
| B | A + vitals, pupils, POC glucose, arrest/trauma history, witnessed convulsion | `vital__`, `pupil__`, `poc_glucose__`, `hx__` |
| C | B + every lab, toxicology, culture, imaging report available by t0 | `lab__`, `tox__`, `culture__`, `img__` |
| D | C + EEG referral indication category (`ReferralIndication` field only) | `ind__` |

Output: `FeatureSet.X` (one row per patient, index `person_id`, stable names `<group>__<key>__<stat>`, raw NaN kept),
`FeatureSet.index` (person_id, SiteID, SessionID, t0; never a model input), `FeatureSet.provenance` (columns: feature, baseline, group,
role, source_table, source_field, time_basis, window, imputation, approximate_time_possible, description),
`FeatureSet.matrix("A".."D")`, `FeatureSet.to_long()`. Names depend only on `BaselineConfig`, not on the data.
CLI: `python -m sortinghat.baselines --data data/synthetic --out out/baselines` (matrix only under `local_only/`, mode 0600).

### 2.1 Baseline A
* **Scores**: latest value in [t0-6 h, t0] (plan: +/-6 h means the 6 h before t0). Features `value`, `miss`, `age_h`. Plausibility
  ranges null out impossible values (GCS 3-15, FOUR 0-16, RASS -5..4); NESI is unconstrained (no documented range in the repo).
  Name matching is by regex on `measurement_source_value` (components before totals).
* **Sedatives/opioids**, by RxNorm ingredient name matched on word boundaries in `drug_source_value` plus the OMOP `concept_name`
  of `drug_concept_id` when the concept table has it (brand names included). Named ingredients: propofol, midazolam, lorazepam,
  dexmedetomidine, ketamine, fentanyl, hydromorphone, morphine, each with `on_t0`, `qty_6h`, `qty_24h`. Class aggregates
  (all sedatives incl. diazepam, clonazepam, pentobarbital, phenobarbital, etomidate; all opioids incl. remifentanil, sufentanil,
  oxycodone, hydrocodone, methadone, buprenorphine, meperidine, tramadol, codeine): `on_t0`, count started in 24 h; plus
  `sed__n_agents_24h` and `sed__approx_time`.
* **Administration vs order.** A row with a recorded end datetime is read as an administration record (start = administration);
  a row without one is read as an order (start = order time, `approx = True`; `sed__approx_time` = 1 if any such row falls in the
  24 h window). `on_t0`: administration row started with no stop recorded by t0; order row placed within 2 h of t0 **[OP]**.
  `qty_*`: recorded `quantity` of completed administrations, pro-rated by overlap with the window; order rows count if placed in the
  window; running infusions contribute 0 (total unknown at t0; `on_t0` carries them). **Units are not normalized** (free-text
  product, unknown `quantity` unit) **[OP]**; dose normalization to mg-equivalents needs the audit's unit evidence.
  `drug_time_basis = "order"` forces the order fallback for every row (SAP robustness row 8).

### 2.2 Baseline B
Vitals (HR, SBP, DBP, MAP, RR, SpO2, temperature in C; F converted) and pupils (size and reactivity per side, any non-reactive,
size asymmetry): latest value in 6 h **[OP]**. POC glucose (name must say POC/fingerstick/bedside/capillary/glucometer): latest in 24 h
**[OP]**, with `age_h`. Serum glucose is a Baseline C lab. History flags `hx__{arrest,trauma,head_trauma}_{any,recent}` and
`hx__convulsion_recent` (recent = 72 h **[OP]**) from ICD source values (I46, Z86.74, S06/S02, S00-S99, T07, T14, ICD-9 equivalents),
CPR procedure codes (CPT 92950, PCS 5A12012) and a "witnessed seizure/convulsion" observation or measurement (negative answers
ignored). Absence of a record is 0. **Limits:** diagnosis codes have an unknown coding time (often assigned at discharge); their
start datetime is used and the group is marked approximate. Seizure/G40/R56 diagnosis codes are deliberately not used for
convulsion, and nonspecific encephalopathy codes (G92, G93.4) are not used anywhere (banned evidence, D-007). Pupil reactivity coding
0 = non-reactive is **[ASSUMED]**.

### 2.3 Baseline C
Per lexicon key (36 chemistry/heme/gas/CSF labs, 11 tox keys, 4 cultures, 4 imaging classes): latest result available in
[t0-72 h, t0] **[OP]**; `value`+`miss` for labs/tox; cultures `resulted`, `positive` (1/0), `miss`; imaging `final_by_t0` and `n_by_t0`.
"Every lab" is met by a frozen vocabulary: `BaselineConfig.extra_labs` adds data-discovered names; `baselines.vocab.discover_lab_vocabulary`
(human-run, training sites only, counts suppressed, min 11 patients) lists candidates. Unmapped names are otherwise counted in the
diagnostics (`n_measurement_rows_unmapped`) and excluded. Unit canonicalization covers glucose, lactate, creatinine, calcium.
* **Availability.** Result time if a result-datetime column exists (aliases in `schema.COLUMN_ALIASES`; none in the real OMOP
  `measurement`), else collection time + assay lag (`lexicon`: 1 h chemistry/CBC/gas, 2 h LFT/ammonia/troponin/coag, 3 h serum tox,
  2 h urine screen, 1.5 h CSF, 6 h send-out endocrine, 24 h cultures) and `lab__approx_time` = 1 for that patient. Imaging uses the
  final-report time, else study time + read lag (CT 1 h, CTA 1.5 h, MRI 4 h), `img__approx_time` = 1. **All lags are [PLACEHOLDER]**
  textbook turnaround guesses; replace with Phase 0a observed medians. Date-only values are placed at the end of their day.
  `lab_time_basis = "collect_plus_lag"` / `imaging_time_basis = "study_plus_lag"` force the approximate versions;
  `"result"` keeps only rows with true result times (rows without one are dropped).
* The imaging table is **[ASSUMED]** (location and columns). Only modality and times exist, so no imaging finding content is a
  feature yet; remap and add finding flags after the dry-run schema check. The H5 "t0 precedes head-CT result" stratifier is not a
  feature (it needs the post-t0 result time) and must be computed separately as a subgroup variable.
* Pending-at-t0 studies (acquired, unread) are not features **[OP]**.

### 2.4 Baseline D
One-hot of `ReferralIndication` mapped to rule_out_ncse, post_arrest, unexplained_ams, seizure, spell, other, missing (keyword rules in
`lexicon.indication_category`; **[ASSUMED]** column and category wording, verify against the real field). Nothing else from the EEG
order or report enters D.

## 3. Missing-data handling (SAP section 9, item 3)

* Absence is information at t0: every variable that can be unobserved carries a `<name>__miss` indicator computed by the builder
  (deterministic, nothing fitted).
* Zero-fill by definition (not imputation): drug exposure/doses/counts, history flags, imaging availability, culture `resulted`.
* Remaining NaN in `value`/`age_h` columns: median of the **training** rows, fitted inside each training fold on training sites only
  (`MedianImputer.fit(train).transform(any)`); a column with no training observation gets 0; the held-out site never contributes.
  Fitted medians are exposed (`medians_`) for the SAP appendix. No multiple imputation.
* Time-quality flags: `sed__approx_time`, `lab__approx_time`, `img__approx_time` (and the group-level `approximate_time_possible`
  provenance column) mark rows that used the order or collection-plus-lag fallback. Analyses relying on them are labeled
  "approximate" (D-034). Decision rule (plan Phase 0a): medication administration times under 80% present => medication
  groups approximate; no true lab result time => all lab/tox/culture features approximate.

## 4. Operationalizations proposed for the DECISION_LOG (none yet logged)

Score window = 6 h before t0; vital/pupil window 6 h; POC glucose 24 h; lab lookback 72 h; recent-history window 72 h; order-time
"on infusion" = within 2 h; dose = raw recorded quantity, pro-rated, running infusions 0; indication category set; plausibility
ranges; the lag table; median imputation with indicators; history flags from codes at start datetime. All are `BaselineConfig`
fields or `lexicon` constants, so they can be changed before freeze without touching logic.

## 5. Items needing a human on the real tables

1. Whether `drug_exposure_start_datetime` is order or administration time (decides `drug_time_basis`), and the `quantity` units.
2. Real `measurement_source_value` strings for scores, vitals, pupils, POC glucose, cultures, tox (lexicon is regex guesses); FOUR/NESI naming.
3. Whether any result/verification datetime exists on `measurement`; observed assay lags to replace the placeholder table.
4. Imaging location, columns, finalization time, and finding fields.
5. `ReferralIndication` existence and values; pupil reactivity coding.
6. A cohort-filtered loader for restricted runs (`tableio.load_tables` reads whole tables; fine for synthetic scale only).
7. The synthetic generator lacks vitals, pupils, POC glucose, tox, cultures, arrest/trauma observations and several named drugs; the
   tests add them (`tests/test_baselines_helpers.py`) rather than editing the shared generator.
