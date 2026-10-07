# Statistical Analysis Plan: Study 1 (1A and 1B)

**Project:** Sorting Hat, EEG-based probabilistic etiologic differential for acute impaired consciousness
**Document:** Preregistration-ready SAP for Study 1A (scientific claim) and Study 1B (translational claim)
**Version:** 0.1 DRAFT, 2026-10-07
**Source of truth for design:** `docs/research_plan_v1.txt` (cited as "plan"), `DECISION_LOG.md` (cited as D-nnn), `docs/labels_spec.md`
**Reference implementation:** `sortinghat/metrics/` (cited as `metrics.<module>`); tests in `tests/test_metrics_*.py`
**Registry:** OSF or AsPredicted, to be filed at freeze. Registry ID: `[pending]`
**Outcome data seen at drafting:** none. No HEEDB table, Phase 0 audit result, pilot label, or model output has been viewed (consistent with the DECISION_LOG header).
**Sign-off required before freeze:** biostatistician `[pending]`; senior EEG or neurocritical-care co-investigator `[pending]`.

Items marked **[OP]** are operationalizations that the plan does not state; they are collected in section 16 and need a DECISION_LOG entry before freeze. Items marked **[PLACEHOLDER]** are numbers that the Phase 0 pilot must replace.

---

## 1. Objectives

- **Claim 1A (scientific).** Etiologic families of acute impaired consciousness leave reproducible, site-transportable EEG signatures beyond severity and known sedative exposure.
- **Claim 1B (translational).** Adding a short EEG to everything the team knows at t0 improves the calibrated, multi-label etiologic differential.
- **Not claimed** (plan, "Core question"): exact metabolic subtypes, a single cause per patient, superiority to a neurologist reading the full record, or any performance on reduced montages (that is Study 2).

## 2. Population and index time

- Adults, acute-care encounter (ICU, inpatient, ED), **first qualifying EEG per patient**. One row per patient throughout, so every index-level split is a patient-level split.
- **Strict cohort** (primary): GCS <= 11 or FOUR <= 12. **Broad EHR-phenotype cohort**: reported separately, never pooled into the primary estimate.
- **t0 = EEG start.** Primary window = minutes 1 to 11 of the recording, requiring >= 60% usable data on the minimum channel set.
- EEG within 24 h of documented acute-consciousness-impairment (ACI) onset; time since onset enters every model as a covariate. Sensitivity windows: <= 6 h, <= 12 h, <= 48 h.
- Record the **EEG referral indication** as a category (for example "rule out NCSE", "post-arrest", "unexplained AMS"). It is used only in Baseline D.
- Eligibility depends on the Phase 0a field audit passing its "Stop" rows (plan, Phase 0a). The site criterion is >= 3 adult sites with >= 300 candidates each (`metrics.splits.check_site_requirements`); if it fails, the plan falls back to a grouped split with a weaker claim (D-024).
- Flow of exclusions (counts only, small cells suppressed per `sortinghat.safe_output`) is reported in a STARD-AI / TRIPOD+AI flow diagram.

## 3. Labels and reference standard

### 3.1 Label set

| Label | Role in Study 1 |
|---|---|
| E1 Acute structural | Primary |
| E2 Global hypoxic-ischemic | Primary |
| E3 Epileptic contributor | **Excluded from primary endpoint.** Positive control (separate EEG-based adjudication); H4 ranking |
| E4a Exogenous intoxication | Primary |
| E4b Iatrogenic sedation | **Excluded from primary endpoint.** Covariate (Baseline A onward) and secondary label |
| E5 Metabolic/physiologic | Primary |
| E6 Systemic infection/inflammation (no CNS infection) | Primary |
| E7 CNS infection/inflammation | Primary only with >= 100 gold positives across >= 2 sites (`metrics.labels.e7_eligible`); otherwise exploratory via an enriched set |

The primary label set is therefore {E1, E2, E4a, E5, E6} plus E7 if eligible (`metrics.labels.PRIMARY_LABELS`). E7 eligibility is decided on the evaluation set's gold-positive counts before any model output is unblinded.

### 3.2 Gold states and binarization

Gold labels use the ordinal states absent / possible / probable / definite / unassessable, assigned blind to EEG, EEG reports and model outputs (plan, Reference standard; `docs/labels_spec.md`). **Binary positive = probable or definite** (`docs/labels_spec.md`, section 1). For scoring:

- y = 1 for probable or definite; y = 0 for absent or possible; **mask = False (unassessable) for "unassessable"**.
- Sensitivity: "possible" masked; "possible" scored as 1 (section 12). **[OP]** The plan keeps the handoff's ordinal states but the scoring of "possible" is not stated in the plan; this SAP follows `docs/labels_spec.md`.

### 3.3 Adjudication design

Evaluation set: about 1,000 consecutive gold cases, dual independent review with third-reader resolution. Development cases: single reviewer with a random 20% double-read. Packets are built by deterministic extraction plus an open-weight summarizer on approved compute, after an EEG-sentence filter. Silver labels train models and never score them. The silver-label circularity rules and the circularity audit are as in the plan (Study 1, "Silver labels") and `docs/labels_spec.md`.

## 4. Comparators, representations and models

### 4.1 Baselines

| Baseline | Contents | Used in |
|---|---|---|
| A Severity + sedation | Age, sex, GCS/FOUR/RASS, NESI, t0 sedative and opioid exposure | **H1** |
| B Bedside | A + vitals, pupils, witnessed convulsion, arrest/trauma history, point-of-care glucose | Secondary |
| C Full t0 | B + every lab, toxicology, imaging and culture result **available before t0** | **H2** |
| D Referral question | C + the EEG indication category (indication field only) | Sensitivity (proxy for the clinician's prior) |

"Available before t0" uses result-availability (verification) time, not collection time, wherever the Phase 0a audit shows it exists; otherwise Study 1B is labeled "approximate" (plan, Phase 0a).

### 4.2 EEG representations

The representation ladder is as in the plan: prior, qEEG, connectivity, MORGOTH findings, frozen embeddings, dynamics, combined, each with the same shallow multi-label head. Frozen CBraMod embeddings are run alongside MORGOTH.

- **Primary EEG model [OP]:** the top rung ("combined") on the primary 10-minute window, with the head family and hyperparameter budget fixed before any evaluation-set unblinding. The plan does not name the rung that carries H1 and H2; naming one avoids a best-of-ladder selection. All other rungs are descriptive ladder results with CIs, not hypothesis tests.
- **Commercial-clean gap** = Delta(CBraMod-based combined) minus Delta(MORGOTH-based combined), paired bootstrap CI, reported not tested.

### 4.3 Model for comparison

For each baseline X in {A, B, C, D}, the comparator is a baseline-only model with the same head family, training data, hyperparameter budget and recalibration as the "baseline + EEG" model. Delta compares the two on **identical test patients**. No model sees a held-out site in training, hyperparameter selection, imputation fitting, feature scaling or recalibration.

## 5. Primary endpoint

For patient i, let L_i be the set of labels assessable for i, y_ik in {0,1}, and p_ik the predicted probability clipped to [eps, 1 - eps]:

```
loss_i(M) = (1 / |L_i|) * sum_{k in L_i}  l(y_ik, p_ik(M)),     l(y, p) = -[ y ln p + (1 - y) ln(1 - p) ]
loss(M)   = (1 / N) * sum_i loss_i(M),   over patients with |L_i| >= 1
Delta     = loss(model with EEG) - loss(baseline)  =  (1 / N) * sum_i d_i,     d_i = loss_i(model) - loss_i(baseline)
```

- Negative Delta means EEG helps.
- The mean is over patients of the mean over that patient's assessable labels, **not** a pooled mean over cells (`metrics.loss.per_patient_loss`, `delta_log_loss`; a unit test pins the difference).
- Labels in L_i are restricted to the primary set (section 3.1) for the primary endpoint.
- **Clipping:** eps = 1e-4 (`DEFAULT_EPS`) **[OP]**. Sensitivity at 1e-6 and 1e-3. Without clipping a single confident miss can dominate the mean.
- Patients with no assessable primary label are excluded identically for model and baseline, and their number is reported.
- Pooled Delta is patient-weighted over the union of leave-one-site-out test sets. The unweighted mean of site-level Deltas (`metrics.loss.site_weighted_delta`) is a sensitivity.

## 6. Hypotheses and exact success criteria

alpha = 0.05 two-sided throughout (95% intervals). "CI" refers to the interval defined in section 7.2. Criteria are copied from the plan's hypotheses table (D-017 to D-022) and made exact where the plan leaves room.

| ID | Hypothesis | Estimand | Success criterion | Role |
|---|---|---|---|---|
| **H1** | EEG improves on severity + t0 sedative exposure | Delta vs Baseline A, strict cohort, leave-one-site-out | Pooled 95% CI for Delta **below 0**, **and** Delta point estimate **below 0 at every held-out site** (D-018, D-023) | Primary, 1A |
| **H2** | EEG improves on the full t0 clinical model | Delta vs Baseline C | Same as H1 | Primary, 1B |
| **H3** | H1 survives severity matching | Delta vs Baseline A within each of 3 severity strata (GCS/FOUR/NESI strata, cut points fixed before unblinding [OP]) | Delta point estimate **below 0 within at least 2 of 3 strata** (`metrics.hypotheses.h3_severity_stratified`). A stratum with < 50 patients is not estimable and counts as not favorable **[OP, PLACEHOLDER]**. Per-stratum CIs are reported but not part of the rule | Key secondary |
| **H4** | Identifiability ranking E3 > E2 > E1 > E4a > E5 > E6, with E6 vs E5 near chance | Per-label Delta_k (E3 from the positive-control analysis, others vs Baseline A) ranked most to least negative; Kendall tau-b between predicted and observed ranking with bootstrap CI; pairwise AUROC for E6 vs E5 | **Recorded before unblinding and reported whichever way it falls** (plan). No pass/fail threshold is set by the plan. Reported: observed ranking, tau-b with CI, and pairwise AUROC for E6 vs E5 with CI. "Near chance" is read as the pairwise-AUROC CI containing 0.5 **[OP]** | Boundary map, descriptive |
| **H5** | The 1B gain is larger when t0 precedes the head-CT result | Interaction = Delta(early subgroup) - Delta(rest), vs Baseline C. Early = t0 before head CT result time (finalization time per Phase 0a) | Interaction **estimate with 95% CI** (plan). "Supported" if the estimate is below 0 and the CI lies entirely below 0 **[OP]**; covariate-adjusted estimate (site, severity, time since onset) reported alongside the unadjusted one. **Dropped if Phase 0a shows no imaging finalization time** (D-021) | Intended-use proxy |
| **H6** | At least 70% of the 10-minute gain is present by 2 minutes | Ratio Delta(2 min) / Delta(10 min), nested windows from the start of the primary window | Ratio **with CI** (plan). "Met" if the point estimate is >= 0.70 and Delta(10 min) < 0 **[OP]**. The ratio is uninterpretable, and is reported as such, when the CI for Delta(10 min) includes 0 or fewer than 90% of bootstrap draws have Delta(10 min) < 0 **[OP, PLACEHOLDER]**. Also report Delta at 20 s, 1, 2, 5, 10 min and the earliest window reaching 70% (`first_window_reaching_fraction`) | Device design, exploratory |

Notes.

- H1 and H2 require both a CI excluding zero **and** the per-site condition. A pooled CI below zero with one unfavorable site is reported as "pooled effect not site-transportable" and H1/H2 are not met.
- With three sites the per-site condition is strict. Section 14 gives its power under stated assumptions.
- H5 and H6 describe how the gain varies; neither alters the pass/fail status of H1 or H2.

## 7. Estimation

### 7.1 Splits

- **Leave-one-site-out (LOSO)** across adult sites (`metrics.splits.leave_one_site_out`). Each patient's prediction comes from a model trained on the other sites. Per-site and pooled results are both reported.
- **Late-calendar temporal holdout within sites**: the latest 20% of cases in each site by calendar time are the test set; training uses earlier cases only, optionally with an embargo (`metrics.splits.late_temporal_holdout`). Fraction fixed at 0.20 **[OP]**.
  - HEEDB timestamps are date-shifted (`docs/heedb_schema.md`). The Phase 0a audit only requires that the shift is consistent **within patient**. If the shift does not preserve calendar order across patients within a site, the temporal holdout cannot be built and is dropped and recorded as such, not substituted with another construction.
- Hyperparameters and model selection use inner folds grouped by site inside the training sites.

### 7.2 Paired, site-aware bootstrap for Delta

Unit of resampling is the **patient**. The per-patient vector d_i carries both models' losses for the same patient, so resampling rows preserves the model-baseline pairing; the two models' predictions are never resampled independently (a unit test checks that the paired interval is much narrower than an unpaired one on correlated predictions).

| Mode | Resamples | Captures | Limitation |
|---|---|---|---|
| `within_site` (stratified) | Patients with replacement inside each site; site sizes fixed | Patient sampling variance, conditional on the observed sites | Cannot see between-site variance in Delta (the plan's stated concern) |
| `cluster` | Whole sites with replacement | Between-site variance | With S = 3 sites there are only 10 distinct draws and 3 of 27 draws use a single site; percentile intervals are coarse and unreliable |
| `two_stage` | Sites, then patients within drawn sites | Both | Same coarse-site limitation; the most conservative |
| `site_t` (sensitivity) | None. Equal-weight mean of site Deltas with a t(S-1) interval | Between-site spread | 2 degrees of freedom for S = 3 |

- **Which interval carries H1/H2:** the **within-site stratified bootstrap** CI, protected against hidden between-site heterogeneity by the per-site favorable condition. Reasoning: with S = 3 the cluster interval cannot support a coverage claim, and the plan answers the within-site concern with "require a favorable effect at every held-out site; report each separately" (plan, Critical evaluation table). **All four intervals are reported side by side** in the main results table (`metrics.bootstrap.delta_ci_all_modes`). If the within-site CI is below 0 but the cluster or two-stage CI includes 0, the report states that the effect is supported for patients at these sites but not shown for a population of sites; H1/H2 status is unchanged. **[OP]** The plan asks for both intervals to be reported but does not say which carries the decision.
- Percentile intervals; B = 10,000 replicates for final analyses (code default 2,000 for development); fixed seed recorded in the analysis log.
- Patients with undefined d_i (no assessable primary label) are dropped before resampling.

### 7.3 Per-site Delta

Per-site Delta and n are reported for every held-out site (`metrics.loss.per_site_delta`). "Favorable at every held-out site" means the point estimate is strictly below 0 at each (`favorable_at_every_site`; exactly 0 is not favorable). Site names appear in outputs only as pseudonymous labels, and counts under 11 are suppressed.

### 7.4 Per-label results

Per-label Delta_k (mean over patients assessable for label k) with percentile CI and a one-sided bootstrap p-value for H0: Delta_k >= 0 from a shared resample across labels (`metrics.hypotheses.per_label_delta_ci`). A label **"improves individually"** (used by G2) when its CI upper bound is below 0 **[OP]**.

### 7.5 Calibration, discrimination and selective prediction (per label)

All computed on patients assessable for that label, on the pooled LOSO predictions and per held-out site:

- **Calibration slope and intercept** by logistic recalibration, logit P(y=1) = a + b logit(p) (Van Calster 2016). b is the slope (ideal 1); a is the intercept of the joint model. **Calibration-in-the-large** = intercept with slope fixed at 1 (offset model). **O/E** = observed over expected events. Wald CIs from the Fisher information. Reported as NaN when a class is empty or the fit shows separation (`metrics.calibration.calibration_slope_intercept`).
- **ECE**: 10 equal-width bins (primary) and 10 equal-count bins (sensitivity). Binned ECE is biased upward in small samples, is reported with a bootstrap CI, and is never used for model selection (`ece`).
- **Brier score** and **AUROC** (`brier`, `auroc`).
- **Risk-coverage curves** per label: cases ranked by confidence max(p, 1 - p); risk = mean log loss (also Brier and 0/1 error) over the top-k most confident cases; AURC reported (`risk_coverage`, `per_label_risk_coverage`).
- Calibration metrics are **descriptive**. They do not enter H1-H6 or the gate rules. Section 14 shows why: the evaluation set cannot estimate per-label calibration slopes to the precision usually wanted.

## 8. Multiplicity

| Family | Hypotheses | Handling |
|---|---|---|
| Confirmatory | H1, H2 | Each tested once at alpha = 0.05 on one prespecified primary EEG model and one baseline. They address different claims (1A and 1B), feed different gates (G2 and G3) and are never combined into one claim, so no alpha split between them. A single primary representation (section 4.2) is what prevents a hidden multiplicity across the ladder |
| Gated secondary | H3, H5 | H3 is interpreted as confirmatory only if H1 is met; H5 only if H2 is met. Otherwise they are reported descriptively. No further alpha adjustment (decision rules are point-estimate rules or single CIs) **[OP]** |
| Per-label claims for G2 | "At least 3 primary families improve individually" | Holm adjustment across the primary labels on the one-sided bootstrap p-values (`metrics.bootstrap.holm_adjust`). A label counts for G2 if its CI upper bound is below 0 **and** its Holm-adjusted p < 0.05 **[OP]** |
| Exploratory | H4, H6, ladder rungs, other baselines, subgroups, reduced windows, sensitivity analyses | No multiplicity control. Intervals are nominal 95% and are labeled exploratory. Never used for a gate |

## 9. Missing data

1. **Unassessable labels** are missing by design, not missing at random. They are masked in the endpoint; the mask rate per label and per site is reported. If unassessability differs across sites by more than the pre-pilot expectation, report the Delta restricted to patients with all primary labels assessable as a sensitivity **[OP]**.
2. **EEG quality:** < 60% usable data on the minimum channel set means exclusion (plan). Exclusion counts by site are reported. Sensitivity: an inverse-probability-of-inclusion weighted Delta, with weights from a model on baseline covariates and site, fitted within training sites only.
3. **Baseline covariates:** missingness is a feature of the t0 state (a lab not yet drawn is information). Missing values are handled by an in-training-fold imputation plus a missingness indicator for each variable, with imputation parameters fitted on training sites only. No multiple imputation of predictors, because a deployed model cannot do it.
4. **Severity score** (GCS, FOUR or RASS) must fall within +/- 6 h of the EEG for inclusion (Phase 0a). Patients outside the window enter only the broad cohort through the BDSP GCS-from-EHR tool. Sensitivity: exclude patients whose severity score was derived rather than documented.
5. **Timing fields:** where the audit forces the fallback (orders instead of administration times, collection time plus assay lag instead of result time), the corresponding analysis is labeled "approximate" in every table and abstract.
6. **Silver-label gaps** affect training only. They do not enter the endpoint.

## 10. Leakage controls

Mandatory controls from the handoff are retained and operationalized here. Results of every probe are reported whether favorable or not.

1. **Timing:** a predictor is allowed only if its availability time is <= t0. Result-availability, medication administration and imaging finalization times are used where the audit confirms them (plan, Phase 0a).
2. **Label circularity:** the three banned evidence classes (EEG reports and EEG-mentioning sentences; nonspecific encephalopathy codes G92, G93.4 and free-text "toxic-metabolic encephalopathy" without an objective anchor; post-t0 neurology impressions unless the EEG filter has run) are excluded as positive evidence for E1, E2, E4a, E5, E6, E7. **Circularity audit:** if a silver-trained model agrees more with EEG-report impressions than with gold labels, the silver labels are rebuilt (plan).
3. **Silver labels never score;** gold evaluation labels are blind to EEG, EEG reports and model outputs; development gold cases only calibrate and select models and are disjoint from the evaluation set.
4. **Patient-level isolation:** first EEG per patient; cross-site duplicate-patient check; every split at the patient or site level.
5. **Fold hygiene:** feature scaling, imputation, embedding normalization, hyperparameter search and recalibration are fitted inside training sites only. The held-out site is touched once per fold.
6. **Site, duration and channel leakage probes:** a classifier predicting site, recording duration and channel set from the EEG features alone; reported with the Delta stratified by these factors. A probe that predicts site well while Delta is carried by that site is a failed control.
7. **Pretrained-model exposure accounting:** for MORGOTH and CBraMod, list which datasets, sites and patients each was trained on. CBraMod was pretrained on TUEG, and TUSZ/TUAB/TUEV are subsets of it (plan); these are treated as contaminated benchmarks. Delta is also reported on any held-out site shown or suspected to be in a pretrained component's training data.
8. **Negative controls:** (a) permute EEG features across patients within site and strata of the baseline score, expecting Delta ~ 0; (b) a label with no plausible EEG signature or a deliberately irrelevant outcome, expecting no gain; (c) E3 as the positive control, expecting a clear gain. Failure of (c) means the pipeline is broken (the plan's own wording for the E3 positive control in Study 2: "if not, the pipeline is broken") and no Delta is interpreted.
9. **Freeze order:** the code, primary EEG model, imputation, thresholds, section 6 criteria and this SAP are hashed and registered before any evaluation-set gold label or model output is opened.
10. **Sedative-excluded subset** and **severity matching** (section 12) are mandatory controls, not optional sensitivities.

## 11. Gate decision rules (Study 1)

The plan defines gates G0, G2, G3, G4 and G5.

| Gate | Passes when (plan) | Operational rule in this SAP | If it fails (plan) |
|---|---|---|---|
| **G0 Feasibility** | All "Stop" rows of the field audit pass; pilot kappa >= 0.6 in >= 4 primary families; >= 3 families at >= 10% prevalence | Kappa is read as the unweighted binary kappa (probable/definite positive) on the 200-case pilot (`docs/labels_spec.md`, section 6) **[OP]**; prevalence from pilot gold labels | Redesign around a smaller cohort or a CERTA collaboration; no full study |
| **G2 Etiologic information** | H1 met; survives severity matching and sedative adjustment; >= 3 primary families improve individually | (i) H1 met; (ii) H3 met; (iii) sedative-excluded subset has pooled Delta point estimate < 0 **[OP]** (Baseline A already contains sedation); (iv) >= 3 primary labels meet the per-label rule of section 8 | 1-2 families: narrow product (for example E1/E2 triage). None: publish the boundary map and stop the commercial track |
| **G3 Clinical increment** | H2 met at every held-out site | H2 met as defined in section 6 (pooled CI below 0 and every site below 0) | Academic paper only, unless H5 shows a clear early-EEG gain worth a narrower prospective test |

- G4 (deployable EEG) and G5 (prospective) depend on Study 2 and prospective work and are outside this SAP. H6 informs the duration half of G4 but does not decide it.
- G2 is evaluated on Baseline A and G3 on Baseline C; both are reported in the plan's sequence (G2 then G3), and a G3 result is reported whether or not G2 passes.
- **Gap, noted only:** the plan defines **no gate G1** (the sequence reads G0, G2, G3, G4, G5). DECISION_LOG D-056 keeps the IDs as written and flags the gap for the v1.1 revision. This SAP does not define or assume one.

## 12. Sensitivity analyses

Each is reported next to the primary result with the same Delta and CI machinery. None changes the status of H1 or H2.

| # | Sensitivity | Purpose |
|---|---|---|
| 1 | Clipping eps = 1e-6 and 1e-3 | Robustness to confident misses |
| 2 | "Possible" masked; "possible" scored 1 | Gold-state binarization |
| 3 | ACI-to-EEG windows <= 6 h, <= 12 h, <= 48 h | Spectrum (plan) |
| 4 | **Sedative-excluded subset** | Detect Delta carried by sedative signatures |
| 5 | **Severity matching** (H3 strata) and a within-stratum matched analysis | Detect Delta carried by depth of unconsciousness |
| 6 | Baselines B and D in place of A or C | Value over bedside data and over the referral question (proxy for the clinician's prior) |
| 7 | Broad EHR-phenotype cohort, reported separately | Spectrum |
| 8 | Approximate-time versions (orders for administrations; collection time plus lag for results) where the audit forces them | Robustness to timestamp quality |
| 9 | Site-weighted (equal-weight) Delta; `site_t` interval; cluster and two-stage bootstraps | Between-site variance |
| 10 | Late-calendar temporal holdout | Calendar drift within sites |
| 11 | Binned ECE with quantile bins | ECE bin sensitivity |
| 12 | Primary label set plus E7 (if not eligible) and plus E3 or E4b, labeled non-primary | Show what the exclusions change |
| 13 | MORGOTH-based vs CBraMod-based combined model | Commercial-clean gap |
| 14 | IPW-for-inclusion and complete-label analyses (section 9) | Missing-data robustness |
| 15 | Early-EEG subgroup (H5) with and without covariate adjustment | Intended-use proxy |

## 13. Reporting

TRIPOD+AI (PMID 38626948) for the model-evaluation components and STARD-AI (PMID 40954311) for the diagnostic-accuracy framing. Outputs from restricted data are aggregate-only with small-cell suppression (n < 11 shown as "<11"), through `sortinghat.safe_output`. Statistics code in `sortinghat/metrics/` never emits record-level rows; per-patient vectors stay in memory. Hypothesis outcomes are reported in the order of section 6, including every null result.

## 14. Sample size for the ~1,000-case evaluation set

### 14.1 Method

External-validation precision approach for binary outcomes (Riley et al., Stat Med 2021, PMID 34031906; summarized in Riley et al., BMJ 2024, PMID 38253388). Archer et al. (Stat Med 2021, PMID 33150684) give the matching calibration-slope precision approach for continuous outcomes; it is cited for the shared method and not used here because every label is binary. Sample size is chosen so the 95% CI width (2 x 1.96 x SE) of each performance measure meets a target:

1. **O/E:** SE(ln O/E) ~ sqrt((1 - phi) / (n phi)), phi = label prevalence among assessable patients. Target CI width 0.2 (O/E about 0.9 to 1.1).
2. **Calibration slope:** Var(slope-hat) = [I^-1]_22 / n, where I is the per-patient Fisher information of the recalibration model at the assumed truth (slope 1, intercept 0). Target CI width 0.2 (the value requested for this plan; Riley et al. use 0.3 in their worked example and note that 0.2 raises N sharply).
3. **C-statistic:** Newcombe SE, target CI width 0.1 (Riley et al.).

Linear-predictor distribution: no development-study distribution exists for these labels, so the documented last resort is used: LP | y ~ N(mu_y, s^2) with common variance, s = sqrt(2) Phi^-1(C), mu = logit(phi) -/+ s^2/2. This gives a well-calibrated model with the stated prevalence and C. The assumption "well calibrated" is the conservative default in Riley et al. Real LP distributions will differ, so **recompute after the Phase 0 pilot** (as D-027 requires).

**Check against the published example.** For prevalence 0.43 and C 0.77, Riley et al. report 423 (O/E, width 0.22), 949 (slope, width 0.3), 2,137 (slope, width 0.2), 347 (C, width 0.1) using a beta-distributed LP. The script gives 421, 1,000, 2,249 and 366, within 5.5% in every case; the difference is the LP distribution. A unit test pins agreement within 10%.

N below is the number of patients **assessable for that label**. Total patients needed is N divided by the label's assessable fraction.

### 14.2 Output of `python -m sortinghat.metrics.sample_size`

**Table S1. Minimum N (assessable patients per label), primary targets**

| Prevalence | C | N: O/E (CI width 0.2) | N: slope (CI width 0.2) | N: C-stat (CI width 0.1) | N required | Events at N | Binding |
|---|---|---|---|---|---|---|---|
| 5% | 0.70 | 7,299 | 15,739 | 2,737 | 15,739 | 787 | slope |
| 5% | 0.75 | 7,299 | 10,092 | 2,532 | 10,092 | 505 | slope |
| 5% | 0.80 | 7,299 | 7,135 | 2,230 | 7,299 | 365 | O/E |
| 10% | 0.70 | 3,458 | 8,714 | 1,413 | 8,714 | 872 | slope |
| 10% | 0.75 | 3,458 | 5,747 | 1,300 | 5,747 | 575 | slope |
| 10% | 0.80 | 3,458 | 4,196 | 1,140 | 4,196 | 420 | slope |
| 20% | 0.70 | 1,537 | 5,227 | 759 | 5,227 | 1,046 | slope |
| 20% | 0.75 | 1,537 | 3,550 | 691 | 3,550 | 710 | slope |
| 20% | 0.80 | 1,537 | 2,671 | 599 | 2,671 | 535 | slope |

**Table S2. Relaxed targets (O/E width 0.4, slope width 0.3)**

| Prevalence | C | N: O/E (CI width 0.4) | N: slope (CI width 0.3) | N: C-stat (CI width 0.1) | N required | Events at N | Binding |
|---|---|---|---|---|---|---|---|
| 5% | 0.70 | 1,825 | 6,995 | 2,737 | 6,995 | 350 | slope |
| 5% | 0.75 | 1,825 | 4,486 | 2,532 | 4,486 | 225 | slope |
| 5% | 0.80 | 1,825 | 3,171 | 2,230 | 3,171 | 159 | slope |
| 10% | 0.70 | 865 | 3,873 | 1,413 | 3,873 | 388 | slope |
| 10% | 0.75 | 865 | 2,554 | 1,300 | 2,554 | 256 | slope |
| 10% | 0.80 | 865 | 1,865 | 1,140 | 1,865 | 187 | slope |
| 20% | 0.70 | 385 | 2,324 | 759 | 2,324 | 465 | slope |
| 20% | 0.75 | 385 | 1,578 | 691 | 1,578 | 316 | slope |
| 20% | 0.80 | 385 | 1,187 | 599 | 1,187 | 238 | slope |

**Table S3. Expected 95% CI width at fixed N (assuming calibration slope 1, intercept 0)**

| Prevalence | C | slope width N=1000 | slope width N=800 | slope width N=500 | O/E width N=1000 | O/E width N=800 | O/E width N=500 |
|---|---|---|---|---|---|---|---|
| 5% | 0.70 | 0.79 | 0.89 | 1.12 | 0.54 | 0.60 | 0.76 |
| 5% | 0.75 | 0.64 | 0.71 | 0.90 | 0.54 | 0.60 | 0.76 |
| 5% | 0.80 | 0.53 | 0.60 | 0.76 | 0.54 | 0.60 | 0.76 |
| 10% | 0.70 | 0.59 | 0.66 | 0.83 | 0.37 | 0.42 | 0.53 |
| 10% | 0.75 | 0.48 | 0.54 | 0.68 | 0.37 | 0.42 | 0.53 |
| 10% | 0.80 | 0.41 | 0.46 | 0.58 | 0.37 | 0.42 | 0.53 |
| 20% | 0.70 | 0.46 | 0.51 | 0.65 | 0.25 | 0.28 | 0.35 |
| 20% | 0.75 | 0.38 | 0.42 | 0.53 | 0.25 | 0.28 | 0.35 |
| 20% | 0.80 | 0.33 | 0.37 | 0.46 | 0.25 | 0.28 | 0.35 |

**Table S4. Expected 95% CI half-width for primary-endpoint Delta (placeholder SDs of d_i)**

| SD of d_i | half-width N=1000 | half-width N=800 | half-width N=500 |
|---|---|---|---|
| 0.10 | 0.006 | 0.007 | 0.009 |
| 0.20 | 0.012 | 0.014 | 0.018 |
| 0.30 | 0.019 | 0.021 | 0.026 |

**Table S5. Power of the H1/H2 rule (CI upper < 0 and every site < 0): SD of d_i = 0.20 (placeholder), 3 sites x 333 patients**

| True Delta | P(all 3 sites < 0) | Power, H1/H2 rule (tau=0) | Power, H1/H2 rule (tau=0.01) |
|---|---|---|---|
| -0.01 | 0.55 | 0.31 | 0.30 |
| -0.02 | 0.90 | 0.84 | 0.70 |
| -0.03 | 0.99 | 0.99 | 0.93 |
| -0.05 | 1.00 | 1.00 | 1.00 |

### 14.3 What this means for the plan

All numbers below read from the tables above.

- **Per-label calibration slope is the binding criterion** in 8 of 9 prevalence/C cells. Reaching slope CI width 0.2 needs **about 2,700 to 15,700 assessable patients per label** (20% prevalence, C 0.80: 2,671; 5%, C 0.70: 15,739), and O/E width 0.2 needs **1,537 (20%) to 7,299 (5%)**. Relaxing to slope width 0.3 and O/E width 0.4 still needs 1,187 to 6,995 (Table S2).
- **At N = 1,000 assessable patients per label** the expected slope CI width is **0.33 to 0.79** and the O/E width **0.25 to 0.54** (Table S3). The 1,000-case set cannot support precise per-label calibration claims, least of all for labels at or below 10% prevalence. Events at N = 1,000: 50, 100 and 200 for prevalences of 5%, 10% and 20%. Collins et al. (Stat Med 2016, PMID 26553135) put the minimum at 100 events and ideally 200; only labels with >= 10% prevalence reach 100. The E7 rule (>= 100 gold positives) is the same threshold.
- **Consequence for this SAP:** calibration quantities (slope, intercept, ECE) are descriptive with reported CIs, not decision criteria (section 7.5). The decision rules rest on Delta (a patient-level mean over many cells) and on per-label Delta.
- **Primary endpoint precision (Table S4).** The CI half-width of Delta is 1.96 x SD(d_i) / sqrt(N). The SD of d_i is unknown before any data exist; the listed SDs (0.10, 0.20, 0.30) are **[PLACEHOLDER]** and the pilot or development data must replace them. With SD 0.20, N = 1,000 gives a half-width of about 0.012 log-loss units.
- **Per-site condition (Table S5).** With 3 sites of about 333 patients and SD 0.20, a true Delta of -0.02 gives about 0.90 probability that all three site estimates are negative and about 0.84 power for the full H1/H2 rule; -0.01 gives about 0.31. A true Delta of -0.03 gives about 0.99. Between-site heterogeneity (tau = 0.01) lowers the -0.02 case to 0.70. These are normal-theory approximations with placeholder SD and equal site sizes; the simulation is in `power_h1_rule`.
- **The "about 1,000" target** is therefore justified, if at all, by H1/H2 power for plausible effects and not by calibration precision. Whether to raise N, accept descriptive calibration, or pool sites differently is a decision for the biostatistician after the pilot measures prevalence, assessable fraction, SD(d_i) and LP spread. This SAP does not change N.
- E7: the 100-positive rule implies about 10% prevalence at N = 1,000 for E7 to qualify from the evaluation set; otherwise it is exploratory via the enriched set (plan).

## 15. Deviations and amendments

Any change after freeze is an amendment with a date, reason and a statement of whether outcome data had been seen (same rules as DECISION_LOG: append-only, supersede by citing the earlier ID). Deviations are listed in the final report.

## 16. Operationalizations the plan does not state (need DECISION_LOG entries before freeze)

1. Scoring of "possible" (section 3.2): y = 0, with sensitivities. Follows `docs/labels_spec.md`.
2. Probability clipping eps = 1e-4 (section 5).
3. Primary EEG model = top ladder rung on the 10-minute window (section 4.2).
4. The within-site stratified bootstrap carries the H1/H2 CI; cluster, two-stage and site-level t intervals are co-reported (section 7.2).
5. H3: "Delta below 0 within a stratum" = point estimate below 0; minimum stratum n of 50 (placeholder); severity cut points fixed before unblinding (section 6).
6. H4: "near chance" = pairwise E6-vs-E5 AUROC CI contains 0.5; Kendall tau-b reporting (section 6).
7. H5: "supported" = interaction estimate below 0 with CI entirely below 0 (section 6).
8. H6: "met" = ratio point estimate >= 0.70 with Delta(10 min) < 0; interpretability threshold of 90% of bootstrap draws (placeholder) (section 6).
9. "Improves individually" (G2) = per-label CI upper bound below 0 and Holm-adjusted p < 0.05 (sections 7.4 and 8).
10. G2 "sedative adjustment" = sedative-excluded subset Delta point estimate below 0 (section 11).
11. Temporal-holdout fraction 0.20 and its dependence on calendar-ordered timestamps (section 7.1).
12. E7 eligibility: >= 100 total positives and at least two sites with >= 1 positive (`metrics.labels.e7_eligible`).
13. Gating of H3 and H5 interpretation on H1 and H2 (section 8).
14. Bootstrap B = 10,000 for final analyses.
15. G0 kappa = unweighted binary kappa on the pilot (section 11).

## 17. References

PMIDs below were checked against PubMed on 2026-10-07 through the PubMed tools (title, authors, journal, year).

1. Riley RD, Debray TPA, Collins GS, Archer L, Ensor J, van Smeden M, Snell KIE. Minimum sample size for external validation of a clinical prediction model with a binary outcome. *Stat Med* 2021;40(19):4230-4251. PMID 34031906. doi:10.1002/sim.9025
2. Riley RD, Snell KIE, Archer L, Ensor J, Debray TPA, van Calster B, van Smeden M, Collins GS. Evaluation of clinical prediction models (part 3): calculating the sample size required for an external validation study. *BMJ* 2024;384:e074821. PMID 38253388. doi:10.1136/bmj-2023-074821
3. Archer L, Snell KIE, Ensor J, Hudda MT, Collins GS, Riley RD. Minimum sample size for external validation of a clinical prediction model with a continuous outcome. *Stat Med* 2021;40(1):133-146. PMID 33150684. doi:10.1002/sim.8766
4. Collins GS, Ogundimu EO, Altman DG. Sample size considerations for the external validation of a multivariable prognostic model: a resampling study. *Stat Med* 2016;35(2):214-226. PMID 26553135. doi:10.1002/sim.6787
5. Van Calster B, McLernon DJ, van Smeden M, Wynants L, Steyerberg EW. Calibration: the Achilles heel of predictive analytics. *BMC Med* 2019;17:230. PMID 31842878. doi:10.1186/s12916-019-1466-7
6. Van Calster B, Nieboer D, Vergouwe Y, De Cock B, Pencina MJ, Steyerberg EW. A calibration hierarchy for risk models was defined: from utopia to empirical data. *J Clin Epidemiol* 2016;74:167-176. PMID 26772608. doi:10.1016/j.jclinepi.2015.12.005
7. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ* 2024;385:e078378. PMID 38626948. doi:10.1136/bmj-2023-078378 (an erratum is indexed as PMID 38636956)
8. Sounderajah V, Guni A, Liu X, et al. The STARD-AI reporting guideline for diagnostic accuracy studies using artificial intelligence. *Nat Med* 2025;31(10):3283-3289. PMID 40954311. doi:10.1038/s41591-025-03953-8 (an author correction is indexed as PMID 42443516)

Not PubMed-indexed and therefore **not verified here** (verify before filing): Holm S. A simple sequentially rejective multiple test procedure. *Scand J Stat* 1979 (Holm adjustment, section 8); Gneiting T, Raftery AE. Strictly proper scoring rules, prediction, and estimation. *J Am Stat Assoc* 2007 (log loss as a proper scoring rule); Efron B, Tibshirani RJ. *An Introduction to the Bootstrap*. 1993 (percentile bootstrap). A PubMed search for cluster-bootstrap methods returned no relevant hit, so none is cited for section 7.2.

The worked example (covid-19 deterioration model, prevalence 0.43, C 0.77) and the target CI widths in section 14.1 were read from the open-access full text of reference 2 (PMC11778934). That text shows the criteria only as a figure, so the three SE formulas are implemented from the standard derivations in reference 1 and are not transcribed from it. Their correctness is supported by agreement with the worked example (within 5.5%) and by a Monte Carlo test of the slope SE in `tests/test_metrics_sample_size.py`.

## 18. Software map

| SAP element | Module |
|---|---|
| Masked log loss, Delta, per-site Delta, favorable check | `sortinghat/metrics/loss.py` |
| Label sets, E7 eligibility | `sortinghat/metrics/labels.py` |
| LOSO, temporal holdout, site requirements | `sortinghat/metrics/splits.py` |
| Paired site-aware bootstrap, Holm, one-sided p | `sortinghat/metrics/bootstrap.py` |
| Calibration slope/intercept/CITL/O-E, ECE, Brier, AUROC, risk-coverage | `sortinghat/metrics/calibration.py` |
| H3, H5, H6, per-label Delta, H4 helpers | `sortinghat/metrics/hypotheses.py` |
| Sample size and power tables | `sortinghat/metrics/sample_size.py` |
| Tests on synthetic predictions with known answers | `tests/test_metrics_*.py` |

