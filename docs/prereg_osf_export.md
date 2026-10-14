# OSF preregistration export: Sorting Hat Study 1

**Compiled 2026-10-14** from `docs/prereg_study1_sap.md` (v0.1 plus Amendment A1) and `DECISION_LOG.md` (through D-156). Structured to the sections of the OSF "OSF Preregistration" template (study information, design plan, sampling plan, variables, analysis plan, other) so each block can be pasted into the matching field. The SAP and the decision log remain the sources of truth; if this document and the SAP differ, the SAP governs and this document must be corrected before filing.

**Status: DRAFT for the project lead to file.** Nothing here has been filed. Registry ID: `[pending]`. Sign-off still required before freeze: biostatistician `[pending]`; senior EEG or neurocritical-care co-investigator `[pending]`. Items marked **[OP-A1]** await confirmation by the project lead and the biostatistician.

> **Read this first: this is not a registration made before all data were seen.** Aggregate, exploratory silver-label results had been seen when this document was written, and the amended primary endpoint and primary population were chosen with them in view. No gold label exists. Details are in the section "Deviations and data already seen", which should be pasted into the OSF fields for existing data and for any other explanation field.

---

## 1. Study information

**Title.** Sorting Hat Study 1: Does a short scalp EEG add calibrated information about the cause of acute impaired consciousness, beyond severity, sedation and the clinical picture at the time of the EEG? A preregistered retrospective multi-site evaluation on HEEDB.

**Authors / contributors.** `[Name], [Title], Boston University` (project lead); co-investigators and adjudicators `[pending]`.

**Description (suggested abstract).**
Acute impaired consciousness (coma, stupor, undifferentiated altered mental status) has many causes, and a clinician reading an unknown EEG often has little history. Study 1 is a retrospective evaluation on the Harvard Electroencephalography Database (HEEDB, BDSP) of whether the first 10 minutes of scalp EEG improve a calibrated, multi-label probability estimate of etiologic family. Claim 1A (scientific): etiologic families leave reproducible, site-transportable EEG signatures beyond severity and known sedative exposure. Claim 1B (translational): adding a short EEG to everything known at the time of the EEG improves the differential. Reference labels are ordinal, adjudicated blind to the EEG, the EEG report and model outputs. The primary endpoint is the patient-level masked log loss, compared between a baseline-only model and a baseline-plus-EEG model on identical held-out patients, with leave-one-site-out validation. The study does not claim exact metabolic subtypes, a single cause per patient, superiority to a neurologist with the full record, performance on reduced montages, or any ED triage benefit.

**Amendment context (paste into the description).** The SAP was drafted 2026-10-07. Amendment A1 (2026-10-14, before any gold label exists) narrows the primary label set to E1 (acute structural), E2 (global hypoxic-ischemic) and E5 (metabolic/physiologic) and makes the presentation-based stratum "IU-large" the primary population for the intended-use hypotheses. Both choices were informed by aggregate, exploratory silver-label feasibility results and are not independent of them (see "Deviations and data already seen").

**Hypotheses.** Alpha is 0.05 two-sided except where stated. "Delta" is the difference in masked patient-level log loss (baseline plus EEG minus baseline only); negative means the EEG helps. Baseline A = age, sex, GCS/FOUR/RASS, t0 sedative and opioid exposure. Baseline C = A plus every lab, toxicology and culture result available before t0. Each of H1, H2, H3, H5 and H6 is stated for the **primary population IU-large** (below) and, as a secondary replicate with the same criteria, for the whole cohort.

| ID | Hypothesis | Success criterion |
|---|---|---|
| H1 (claim 1A) | EEG improves on severity plus t0 sedative exposure | Delta vs Baseline A: pooled 99% within-site stratified bootstrap interval entirely below 0, **and** Delta point estimate below 0 at every held-out site |
| H2 (claim 1B) | EEG improves on the full t0 clinical model | Same criterion, Delta vs Baseline C |
| H3 | H1 survives severity matching | Delta vs A below 0 in at least 2 of 3 severity strata (GCS-equivalent <= 5, 6-8, >= 9, fixed in advance); a stratum with fewer than 50 patients is not estimable and counts as not favourable. Interpreted as confirmatory only if H1 is met |
| H4 | Identifiability ranking E3 > E2 > E1 > E4a > E5 > E6 with E6 vs E5 near chance | Recorded before unblinding and reported whichever way it falls: observed ranking, Kendall tau-b with bootstrap interval, pairwise AUROC E6 vs E5 with interval (near chance = interval contains 0.5). Descriptive. Estimated in the whole cohort (primary) and in IU-large (descriptive) |
| H5 | The gain over Baseline C is larger when t0 precedes the head-CT result | Interaction = Delta(early) minus Delta(rest) with 95% interval; supported if the estimate is below 0 and the interval lies entirely below 0. Dropped if no imaging finalization time exists in HEEDB. Interpreted only if H2 is met |
| H6 | At least 70% of the 10-minute gain is present by 2 minutes | Ratio Delta(2 min)/Delta(10 min) with interval; met if the point estimate is >= 0.70 and Delta(10 min) < 0; uninterpretable if the Delta(10 min) interval includes 0 or fewer than 90% of bootstrap draws have Delta(10 min) < 0. Exploratory |

The primary label set for all of these is {E1, E2, E5} (see Variables). H1 and H2 are the confirmatory pair; they address different claims, are never combined, and no alpha is split between them. H3 and H5 are gated. H4 and H6 are exploratory.

**Gates (decision rules for the project, not hypotheses).** G0 (feasibility): all field-audit "Stop" rows pass, pilot kappa >= 0.6 in >= 4 primary families, >= 3 families at >= 10% prevalence; unchanged and evaluated over the original five primary families. G2 (etiologic information): H1 met, H3 met, the sedative-excluded subset has a pooled Delta point estimate < 0 and the sedation-intensity reading is not (b) or (d) **[OP-A1]**, and all three of E1, E2, E5 improve individually (per-label interval upper bound below 0 and Holm-adjusted p < 0.05) **[OP-A1]**. G3 (clinical increment): H2 met at every held-out site. G2 and G3 are evaluated on IU-large, with whole-cohort results alongside. G4 and G5 depend on Study 2 and prospective work and are outside this registration.

---

## 2. Design plan

**Study type.** Observational, retrospective, multi-site diagnostic-prediction model evaluation (external-style validation by leave-one-site-out), with a planned frozen-model external validation at two outside sites.

**Blinding.** Gold etiologic labels are assigned by clinician adjudicators blind to the EEG, the EEG report and all model outputs; the packets given to adjudicators pass an EEG-sentence filter. Analysts do not open the evaluation-set labels or model outputs until the code, primary model, thresholds and this plan are hashed and registered (freeze order).

**Study design.**
- Index: first qualifying EEG per patient (one row per patient, so every split is a patient-level split). t0 = EEG start.
- Cohorts: **strict** (nearest GCS <= 11 or FOUR <= 12 within [-6 h, +1 h] of t0) and **broad** (EHR phenotype), reported separately and never pooled. The strict cohort cannot supply an intended-use stratum of useful size, so IU-large rows are drawn from the broad cohort.
- Validation: (i) leave-one-site-out across adult sites with labelled EHR data; (ii) late-calendar temporal holdout within sites (latest 20% of cases). Per-site and pooled results are both reported.
- Comparators: Baselines A and C (primary), B (bedside, secondary), P (presentation: age, sex, severity score nearest t0, first vitals and point-of-care glucose of the current encounter, no diagnosis codes, no history). Every baseline uses only events of the current encounter up to t0 by default; allowing prior-encounter events is a sensitivity analysis. A baseline-only comparator has the same head family, data, hyperparameter budget and recalibration as its baseline-plus-EEG counterpart, and Delta is computed on identical patients.
- EEG model: the top rung of a representation ladder (prior, qEEG, connectivity, MORGOTH findings, frozen CBraMod embeddings, dynamics, combined) on the primary 10-minute window (minutes 1 to 11), fixed before unblinding. Other rungs are descriptive. The commercial-clean gap (CBraMod-based minus MORGOTH-based combined model) is reported, not tested.
- Intended-use population: **IU-large**, defined by presentation only (no diagnosis-code, label or EEG-derived condition): t0 on encounter calendar day 0-1; no sedative or opioid continuous infusion running at t0; no ventilation proxy and no vasopressor proxy in the 24 h before t0; recording <= 12 h or not a continuous-EEG task folder. A condition that cannot be established counts as not met. IU-small (calendar day 0, not a continuous-EEG task folder) is a nested secondary stratum. HEEDB has no ED flag, so IU-large approximates the ED population and does not identify it.
- External validation (frozen model, model-to-data, aggregate return only): Careggi EMINENCE (Florence; adult ED patients with emergency EEG) with an EEG-blind re-adjudicated subset as the primary reference, and CERTA (Switzerland; adult ICU/ward patients with documented etiology groups). Success criteria and label mappings are fixed in a dated amendment (A3) before any external run.

**Randomization.** Not applicable. Random elements: 20% of rows are development rows within site (hyperparameter selection and recalibration only); bootstrap seeds are fixed and recorded.

---

## 3. Sampling plan

**Existing data.** Choose the option that matches the facts in "Deviations and data already seen". **Suggested selection: "Registration following analysis of the data"**, because HEEDB EEG and EHR data and silver (structured, EEG-blind) labels on the same patients have been accessed and analysed in aggregate. The confirmatory data, the gold adjudicated labels, **do not exist yet**, which should be stated in the explanation field. The project lead decides the option.

**Explanation of existing data.** See "Deviations and data already seen". In one sentence: aggregate-only outputs of the Phase 0a field audit, cohort flow, silver-label prevalence and the exploratory silver-label feasibility analysis (strict and broad cohorts) have been seen; no gold label, adjudication pilot, kappa or model output scored against gold has been seen or produced; no record-level data have been shown to any hosted model.

**Data collection procedures.**
- Source: HEEDB v4.1 (BDSP, restricted-use agreement), adult acute-care encounters (ICU, inpatient, ED proxy). Study 1 sites: I0002, I0003, S0001, S0002; I0008 and I0009 have no EHR rows and are excluded from all labelled analyses. At the time of writing only S0001 and S0002 have labelled patients, so the every-site rule is a two-site rule at present (a weaker claim, as the plan's fallback states).
- Acute-care proxy: a visit covers the EEG date [visit start - 24 h, visit end + 24 h] and is inpatient-length or its service names an ICU, ED, inpatient or long-term-monitoring setting; EMU and OR excluded. Recording clock duration >= 11 minutes; minimum channel set of the 10 hairline electrodes (Fp1, Fp2, F7, F8, T3, T4, T5, T6, O1, O2); >= 60% usable data in the window.
- Gold labels: ordinal states (absent / possible / probable / definite / unassessable), dual independent review with third-reader resolution for the evaluation set; development cases single-reviewed with a 20% double-read. Packets are built by deterministic extraction plus an open-weight summarizer on approved compute. Silver labels train models and never score them in the confirmatory analysis.
- All real-data processing runs on approved compute with aggregate-only outputs and small-cell suppression (n < 11 shown as "<11"). No notes are sent to a hosted model.

**Sample size.** About 1,000 consecutive gold-adjudicated evaluation cases (a 1,500-case reserve is optional) plus a development set. The number of IU-large cases is computed from aggregate cohort counts before sampling; if it is too small, a dated amendment (A2) may enrich IU-large with design weights. No estimate of the final analysed n for IU-large exists yet.

**Sample size rationale.** The size is justified by simulated power for the H1/H2 rule (about 0.89 at a true +0.04 AUROC and 0.99 at +0.06 at N = 1,000; a placeholder normal-theory table gives about 0.84 power for a true Delta of -0.02 log-loss units with SD(d_i) = 0.20), not by calibration precision. A per-label calibration slope CI width of 0.2 would need about 2,700 to 15,700 assessable patients per label; at N = 1,000 the expected width is 0.33 to 0.79. Calibration is therefore descriptive. All SDs and the power tables are placeholders until the pilot supplies prevalence, assessable fraction and SD(d_i), and will be recomputed for the {E1, E2, E5} endpoint and for IU-large (A1.7).

**Stopping rule.** Not applicable (retrospective, fixed evaluation set). The adjudication reserve is added only for reasons recorded before any outcome is seen.

---

## 4. Variables

**Manipulated variables.** None (observational). The contrast of interest is the model with versus without EEG features, with the same baseline covariates.

**Measured variables.**
- *Outcome (primary endpoint).* For patient i, let L_i be the set of labels assessable for i within the **primary label set {E1, E2, E5}**, y_ik in {0,1}, and p_ik the predicted probability clipped to [1e-4, 1 - 1e-4]. Patient loss = mean over k in L_i of the binary log loss; model loss = mean over patients with |L_i| >= 1. **Binary positive = probable or definite; absent or possible = 0; unassessable = masked.** The mean is over patients of the per-patient mean, not a pooled mean over cells.
- *Labels.* E1 acute structural, E2 global hypoxic-ischemic, E5 metabolic/physiologic: **primary**. E3 epileptic contributor: positive control, separate EEG-based adjudication, never in the primary endpoint. E4a exogenous intoxication, E6 systemic infection/inflammation (no CNS infection), E7 CNS infection/inflammation (E7 eligible only with >= 100 gold positives across >= 2 sites): **secondary/exploratory**. E4b iatrogenic sedation: covariate and secondary label.
- *Predictors.* EEG features from the primary window (qEEG, connectivity, MORGOTH findings, frozen CBraMod embeddings); baseline covariates as above, availability-timed at or before t0; time since documented onset (EEG within 24 h of onset) enters every model.
- *Stratifiers and controls.* Site; severity strata (GCS-equivalent); ordinal sedation-intensity grade 0-3 (from Baseline A `sed__*` columns only; cut points fixed, not re-tuned); IU-large and IU-small membership; EEG-to-onset windows (<= 6, 12, 48 h).
- *Excluded predictors and evidence.* EEG reports and EEG-mentioning sentences, nonspecific encephalopathy codes (G92, G93.4) and free-text "toxic-metabolic encephalopathy" without an objective anchor, and post-t0 neurology impressions (unless the EEG filter has run) are never label evidence. No EEG-derived quantity enters any baseline or any stratum definition.

**Indices.** Delta (primary); per-site Delta; per-label Delta with percentile interval and one-sided bootstrap p; calibration slope, intercept, calibration-in-the-large and O/E (descriptive); ECE (10 equal-width bins, with equal-count bins as sensitivity); Brier; AUROC; risk-coverage curves; Kendall tau-b (H4); interaction and ratio estimates (H5, H6).

---

## 5. Analysis plan

**Statistical models.** One shallow multi-label head family for every rung and baseline; hyperparameters and recalibration chosen on gold development cases from training sites only; no held-out site is used in training, selection, imputation, scaling or recalibration. Delta uses a paired, site-aware bootstrap resampling **patients** within site (the model-baseline pairing is preserved): the 99% within-site stratified percentile interval (B = 10,000 for final analyses, seed recorded) carries H1 and H2, protected from hidden between-site heterogeneity by the every-site condition. Cluster, two-stage and site-level t intervals, and the equal-weight site mean, are co-reported (the within-site result is then described as supported for these sites and not shown for a population of sites if the others include 0). All other intervals are 95%.

**Transformations.** Probability clipping at 1e-4 (1e-6 and 1e-3 as sensitivity). In-training-fold imputation with a missingness indicator for each variable; feature scaling and embedding normalization fitted on training sites only.

**Inference criteria.** See the hypothesis table. H1 and H2 require the 99% interval below 0 **and** a favourable point estimate at every held-out site; a pooled interval below 0 with one unfavourable site is reported as "pooled effect not site-transportable" and the hypothesis is not met. No multiplicity adjustment across H1 and H2 (distinct claims and gates); H3 and H5 are gated on H1 and H2; per-label claims for G2 use Holm adjustment across the three primary labels; H4, H6, other rungs, other baselines, subgroups and sensitivity analyses are exploratory with nominal 95% intervals and no gate role. In IU-large, a hypothesis is estimable only with >= 50 scored patients and >= 11 at each site; otherwise it is reported as not estimable and the whole-cohort result is the only evidence.

**Data exclusion.** Recordings with < 60% usable data on the minimum channel set; patients with no assessable primary label (dropped identically for model and baseline, and counted); sites without EHR data. Exclusion counts are reported by site in a TRIPOD+AI / STARD-AI flow diagram.

**Missing data.** Unassessable labels are missing by design and are masked, with mask rates reported per label and site (a complete-label analysis is a sensitivity). Missing predictors are imputed within training folds with indicators; no multiple imputation of predictors. EEG-quality exclusion is examined with an inverse-probability-of-inclusion sensitivity analysis. Where timing fields force a fallback (orders for administrations; collection time plus lag for results), the analysis is labelled "approximate".

**Mandatory controls (reported in full whatever the result).**
1. Timing: a predictor is allowed only if available at or before t0.
2. Circularity audit: if a silver-trained model agrees more with EEG-report impressions than with gold labels, the silver labels are rebuilt.
3. Leakage probes: classifiers predicting site, recording duration, channel set and sedation from the EEG features alone, with Delta stratified by these; a probe that predicts site well while Delta is carried by that site is a failed control. Pretrained-model exposure accounting for MORGOTH and CBraMod, with Delta on any site known or suspected to be in a pretrained component's training data.
4. Negative controls: EEG features permuted across patients within site and severity strata (expect Delta about 0); a label with no plausible EEG signature (expect no gain); **E3 as the positive control (expect a clear gain; if not, the pipeline is broken and no Delta is interpreted)**.
5. Sedative-excluded subset and severity matching.
6. **Sedation-intensity analysis:** Delta by grade 0-3 and light (0-1) vs heavy (2-3), heavy-minus-light difference, site-adjusted trend of the per-patient difference on grade with its 99% bootstrap interval, EEG-only sedation-intensity probe. Reading fixed in advance: (a) light-grade 99% interval below 0 and a flat trend: against a sedation explanation; (b) light interval including 0 and a trend below 0: consistent with a sedation contribution; (c) both below 0: a gain at every level that also grows with intensity, a sedation contribution cannot be excluded for the part that grows; (d) not estimable: inconclusive, never exculpatory. "Beyond sedation" wording is used only under reading (a).
7. **Label drift screen and data-feed availability:** test vs train prevalence among assessable rows in the temporal holdout, per site and pooled; a ratio > 3 or < 1/3 flags the label and its temporal results are not interpreted. Run on gold and silver labels. A case whose evidence source was absent in its era (for example no blood-culture rows) is unassessable, never negative.
8. Freeze order: the code, primary EEG model, imputation, thresholds, this plan and the amendments are hashed and registered before any evaluation-set gold label or model output is opened.

**Exploratory analyses.** The ladder, Baseline B, calibration and risk-coverage, per-label results, H4, H6, IU-small, the transported (inverse-odds weighted) IU-large estimate, with-history baselines, temporal holdout, and the original all-label endpoint (below). Sensitivity analyses reported next to the primary result: clipping; "possible" masked or scored 1; onset windows; sedative-excluded subset; severity matching; Baseline B; broad cohort; approximate-time versions; site-weighted and alternative intervals; temporal holdout; quantile-bin ECE; MORGOTH vs CBraMod; IPW and complete-label analyses; early-EEG subgroup; `strict_pm6`; with-history baselines; the **original all-label endpoint (E1, E2, E4a, E5, E6, plus E7 if eligible), reported with the same prominence as the primary**; the whole cohort; E4a, E6, E7 per label; leave-one-label-out within {E1, E2, E5}.

---

## 6. Other

**Deviations and data already seen.** (Paste this section into the "existing data" explanation and any other free-text field.)

*What has been seen, all aggregate-only with counts below 11 suppressed, and none of it at record level in a hosted-model session:*
1. **Schema and field audit (Phase 0a).** Table and column names, provenance, field completeness and timestamp-quality aggregates for the HEEDB tables used (names-only dry run, then the audit). These informed the acute-care proxy, the t0 definition, the severity window and the exclusion of I0008/I0009 (no EHR rows).
2. **Cohort flow.** Counts at each exclusion step for the strict and broad cohorts, EEG-QC pass counts and recording-length distributions.
3. **Silver-label yield and prevalence.** Anchor-firing counts and per-label silver prevalence by site, from structured, EEG-blind rules. A review of the silver rules (E2 shock operationalisation, E5/E6 mapping, report suppression) followed; the E6 blood-culture availability rule was added after a temporal-drift diagnostic.
4. **Silver-label feasibility analysis (exploratory; models trained, tuned and scored on silver labels, two validation schemes, mandatory controls).**
   - **Strict cohort**, several versions (v1 on silver labels produced before the yield review; a rerun on corrected labels; a run adding the frozen CBraMod and MORGOTH rungs; v3 and v4 with the sedation-confounding, E6 availability and drift screens, the sedation-intensity analysis and the IU-large/IU-small strata with transport weighting). *The project lead should confirm the version-to-change mapping against the run logs before filing.*
   - **Broad cohort run** (analysed n in the low thousands, two sites).
   - What was observed, qualitatively: a modest, replicated EEG increment for E1, E2 and E5 beyond Baseline A and a smaller one beyond Baseline C; persistence in patients with no recorded sedation and in IU-large (mostly through E1); no increment for E4a; consistently worse predictions with the EEG for E6 (a small but consistent loss of AUROC, leave-one-site-out) with a culture-feed availability problem; E7 too rare; commercial-clean gap about 0 leave-one-site-out; analyses of label drift, leakage probes, negative controls and the circularity audit (EEG-report comparator) at aggregate level.
5. **Intended-use fit diagnostic.** Aggregate comparison of the cohort with an ED-like presentation (setting, sedation, ventilation, vasopressors, timing), which found the strict cohort far from the ED population and the broad cohort mixed.

*What has NOT been seen:* any gold label; any adjudication pilot, kappa or prevalence from gold labels; any model output scored against gold labels; the evaluation set; any EEG report used as label evidence; any external (Careggi, CERTA) data.

*Deviations from the original SAP v0.1 (all in Amendment A1, 2026-10-14, before any gold label):*
- **Primary label set** narrowed from {E1, E2, E4a, E5, E6 (+E7)} to {E1, E2, E5}; E4a, E6, E7 become secondary/exploratory; E3 remains the separately reported positive control. This choice was informed by the silver feasibility results above, so it is not blind to them. The original endpoint is reported as a sensitivity analysis of equal prominence.
- **Primary population** for H1, H2, H3, H5, H6 and the gates changed from the strict cohort to IU-large (defined by presentation only; definition frozen in D-155 before any stratum-level result was computed); the strict and broad whole cohorts become secondary. H4 stays whole-cohort.
- **Controls** the sedation-intensity analysis (D-154) and drift screen with availability rule (D-153) added as mandatory.
- **Gate reading:** G2's "at least 3 primary families improve individually" now requires all of E1, E2, E5; G2 (iii) also requires a sedation-intensity reading other than (b) or (d) **[OP-A1]**.
- *Earlier exploratory deviations from the plan, recorded in DECISION_LOG and `docs/silver_feasibility.md`:* silver labels were used to train, tune and score (D-143); models were trained before Gate 0; two sites rather than three; MORGOTH/CBraMod/dynamics rungs added later; the temporal holdout may not be a true temporal scheme because HEEDB dates are shifted.

*Known threats to the primary result that follow from this history:* the amended endpoint has a higher prior chance of favourable results than the original; silver and gold labels measure different things; the two-site every-site rule is weak; IU-large may be small and approximates the ED only.

**External validation and claim limits.** Frozen model, preprocessing and scoring script hashed before any external data; no external data for training, selection or recalibration; collaborators run locally and return aggregates only with their suppression rules. Careggi: EEG-blind re-adjudicated subset as the primary reference (their discharge diagnoses were made with the EEG available and are secondary and labelled circular). CERTA: ICU check of transport of the E1, E2, E5 result. Default external readout: Delta over the best analogue of Baseline P, over age/sex and over the prevalence prior, pooled over the E1/E2/E5 counterparts, 95% interval, reported whichever way it falls; success criteria and label mappings fixed in amendment A3 before any run. **No claim about ED triage will be made from HEEDB alone;** IU-large results are described as "undifferentiated, early-presentation acute-care EEG (a HEEDB proxy for the ED)". A deployment or triage claim requires a prospective ED study and is outside this registration.

**Pending amendments.** A2: evaluation-set sampling frame (consecutive or IU-large-enriched with design weights), due before any case is sampled. A3: external-validation success criteria and label mappings, due before any external run. Any further label-set or population change requires a new dated amendment before the evaluation set is unblinded, stating what outcome data had been seen.

**Reporting guidelines.** TRIPOD+AI and STARD-AI. Hypothesis outcomes are reported in the order of the hypothesis table, including every null result, and E4a/E6 demotion is stated in the main text.

**Data and code availability.** The data are restricted (BDSP agreement) and cannot be shared; only aggregates with small-cell suppression (n < 11 shown as "<11") are reported. Code, the frozen analysis plan and the decision log are to be shared on registration or publication `[lead to decide]`. No record-level data, derived reports from restricted data or secrets are committed.

**Funding, conflicts, IP.** `[to be completed by the project lead]`. Declare any intellectual-property or commercial interest here as required by the registry and the institutions involved (the project lead decides the wording; do not paste internal IP correspondence).

---

## How to file (checklist for the human)

1. **Before you file.** Have the biostatistician and the senior EEG/neurocritical-care co-investigator read the SAP and Amendment A1; record their sign-off. Resolve every **[OP-A1]** item, and update SAP section 16 and the decision log to match. Confirm the version-to-change mapping of the silver runs against the run logs. Confirm the implementation of the amended label sets in `sortinghat/metrics/labels.py` (open item A1.7, not yet done), and, if you choose to quote exact figures in OSF (the analysed n and the E6 AUROC change), take them from the aggregate reports in `out/` (kept out of git, CLAUDE.md rule 6). Do not paste any suppressed or restricted-derived table into OSF; the qualitative statements above, plus at most those two aggregate figures, are the maximum.
2. **OSF account.** Sign in with an institutional ORCID-linked account at osf.io (create it if needed). Create a project for Sorting Hat Study 1 and add co-investigators as contributors with the right permissions. Keep the project private while drafting.
3. **Choose the form.** Start a registration from the project and select the "OSF Preregistration" template (or the "AsPredicted" form for a short version; the SAP names OSF or AsPredicted). Paste the sections of this document into the matching fields. Where a field has a fixed choice, use: existing data = "Registration following analysis of the data" (or the closest honest option) and put the disclosure above in the explanation field.
4. **Attach the files.** Upload the SAP with Amendment A1 (`docs/prereg_study1_sap.md`, as PDF or markdown), `DECISION_LOG.md` (optional; it contains internal commercial and IP entries, so review or redact it before uploading), and this document. Add the git commit hash and a SHA-256 of the SAP, the analysis code and the primary EEG model configuration, so the freeze is verifiable. Do not upload data, record-level files, or reports derived from restricted data.
5. **Freeze and embargo.** Submitting a registration creates an immutable time-stamped copy. OSF lets you choose public immediately or an **embargo of up to 4 years**, after which it becomes public automatically; you can end an embargo early. Check the current OSF embargo rules at filing, including what remains visible during an embargo. Consider IP timing and scoop risk when choosing; a public registration with a short embargo is the usual default, and the amendment history is part of the record either way.
6. **Registry ID.** Copy the registration DOI/ID into the SAP header (`Registry ID: [pending]`), the decision log and the outreach drafts. This edit is a note about the registration, not a change to the plan.
7. **After filing.** Any later change is a dated amendment filed as an OSF "update" or a new linked registration, stating whether outcome data had been seen. File A2 before the evaluation-set sampling and A3 before any external run.
8. **Do not file until** the freeze order of SAP section 10, item 9 can be honoured: no evaluation-set gold label or model output has been opened.
