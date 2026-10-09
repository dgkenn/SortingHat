# Sorting Hat Decision Log

**Purpose.** This log records every protocol change in the Research Plan (`docs/research_plan_v1.txt`) relative to the master handoff v1.0 (29 Sep 2026). It exists for preregistration integrity: each decision is dated and written down before any outcome data (pilot label or model output) is viewed; data-quality aggregates are noted separately (see Rules).

**Rules.**
- Append-only. Do not edit or delete an entry. If a decision changes, add a new entry that cites the superseded ID (e.g., "Supersedes D-014").
- Each entry records whether OUTCOME data (gold/silver label distributions by EEG feature, or any model performance) had been seen — none has been as of 2026-10-08. Data-quality aggregates (schema, field-audit, cohort-flow and EEG-QC counts) have been viewed since 2026-10-07 and are noted per entry where they informed a decision.
- Elements the plan keeps unchanged from the handoff (cohort definition, primary window, preprocessing, representation ladder, ordinal gold states, leakage probes, sedative-excluded subset) are not logged.

**Dating note.** Entries are dated 2026-10-07, the date this log was written. The plan has no internal date. Its "Next 30 days" list still shows the HEEDB metadata pull, the Phase 0 field audit and all modeling as future work, so no outcome data exists to have been seen.

---

## Ontology and primary endpoint

### D-001 Split toxic/pharmacologic into E4a and E4b
- **Date:** 2026-10-07 · **Area:** ontology · **Outcome data seen?** No
- **Decision:** Replace the handoff's combined toxic/pharmacologic label with E4a (exogenous intoxication, primary) and E4b (iatrogenic sedation, covariate and secondary label).
- **Rationale:** A severity-only baseline has no drug data, so EEG would gain log loss by recognizing sedative signatures the team already knows about.
- **Source:** Critical evaluation table, row "Toxic/pharmacologic label includes iatrogenic sedation"; Revised ontology.

### D-002 Exclude E3 from primary label set
- **Date:** 2026-10-07 · **Area:** ontology · **Outcome data seen?** No
- **Decision:** Remove E3 (epileptic contributor) from the primary endpoint and use it as the pipeline's positive control with separate EEG-based adjudication.
- **Rationale:** E3 is partly defined by EEG, so including it would guarantee a win on the endpoint.
- **Source:** Core question, Primary endpoint paragraph; Revised ontology, E3 row.

### D-003 Exclude E4b from primary endpoint
- **Date:** 2026-10-07 · **Area:** ontology · **Outcome data seen?** No
- **Decision:** E4b (iatrogenic sedation) is excluded from the primary label set.
- **Rationale:** The team knows about planned sedation at t0, so it is not a diagnostic question.
- **Source:** Core question, Primary endpoint paragraph.

### D-004 E7 enters primary endpoint only above a threshold
- **Date:** 2026-10-07 · **Area:** ontology · **Outcome data seen?** No
- **Decision:** E7 (CNS infection/inflammation) is a primary label only with at least 100 gold positives across at least 2 sites; otherwise it is exploratory via an enriched set.
- **Rationale:** The plan states the threshold but gives no reason; the likely purpose is that a site-held-out estimate needs positives at more than one site (inferred, not stated in the plan).
- **Source:** Revised ontology, E7 row.

---

## Labels and reference standard

### D-005 Silver labels train models and never score them
- **Date:** 2026-10-07 · **Area:** labels · **Outcome data seen?** No
- **Decision:** Silver labels are used only to train models; they are never used to score them.
- **Rationale:** Silver labels can carry EEG-influenced guesses, so scoring against them would reward imitating the EEG reader rather than the cause.
- **Source:** Silver labels: the circularity rules, opening paragraph.

### D-006 Ban EEG-derived evidence as positive evidence
- **Date:** 2026-10-07 · **Area:** labels · **Outcome data seen?** No
- **Decision:** EEG reports, and any note sentence mentioning EEG, cEEG, LTM, slowing, triphasic waves, burst suppression, or LPDs/GPDs/LRDA, are banned as positive evidence for E1, E2, E4a, E5, E6 and E7.
- **Rationale:** Discharge diagnoses are often written off the EEG report, so the model would learn to imitate the reader.
- **Source:** Silver labels, banned evidence rule 1; Critical evaluation table, silver-label row.

### D-007 Ban nonspecific encephalopathy codes as positive evidence
- **Date:** 2026-10-07 · **Area:** labels · **Outcome data seen?** No
- **Decision:** The G92 and G93.4 code families, and free-text "toxic-metabolic encephalopathy" without an objective anchor, are banned as positive evidence for the same labels.
- **Rationale:** These diagnoses are nonspecific and can be written from the EEG impression rather than from an independent finding.
- **Source:** Silver labels, banned evidence rule 2.

### D-008 Ban post-t0 neurology impressions unless EEG content is removed
- **Date:** 2026-10-07 · **Area:** labels · **Outcome data seen?** No
- **Decision:** Post-t0 neurology impressions are banned as positive evidence unless the sentence filter has removed EEG content.
- **Rationale:** Such impressions can repeat the EEG reading the model is meant to be tested against, which is the same circularity as D-006.
- **Source:** Silver labels, banned evidence rule 3.

### D-009 Require an objective anchor for every silver positive
- **Date:** 2026-10-07 · **Area:** labels · **Outcome data seen?** No
- **Decision:** Each silver positive needs an objective anchor: a timed lab value, imaging finding, culture or CSF result, arrest event, toxicology result, or antidote response.
- **Rationale:** Anchors tie each positive to evidence that does not depend on the EEG reading.
- **Source:** Silver labels, "Each positive needs an objective anchor."

### D-010 Circularity audit with rebuild trigger
- **Date:** 2026-10-07 · **Area:** labels · **Outcome data seen?** No
- **Decision:** If a silver-trained model agrees more with EEG-report impressions than with gold labels, the silver labels are rebuilt on objective anchors.
- **Rationale:** The exclusion rules can miss leakage, so the plan adds an empirical test of it.
- **Source:** Silver labels, circularity audit paragraph; Go/no-go gates and risk register, "Silver labels leak EEG interpretation."

### D-011 Dual review on evaluation set only, with 20% double-read in development
- **Date:** 2026-10-07 · **Area:** labels · **Outcome data seen?** No
- **Decision:** Evaluation-set gold cases get dual independent review plus third-reader resolution; development cases get one reviewer, with a random 20% double-read to track agreement.
- **Rationale:** Dual review of every case drove the handoff's roughly 4,500-review, 1,100–1,900 physician-hour estimate, and the 20% double-read tracks agreement at lower cost.
- **Source:** Reference standard; Critical evaluation table, adjudication-labor row.

### D-012 Pre-assembled packets with EEG sentence filter
- **Date:** 2026-10-07 · **Area:** labels · **Outcome data seen?** No
- **Decision:** Reviewer packets are assembled locally by deterministic extraction plus an open-weight summarizer on approved compute, with a sentence filter that removes EEG mentions before review; the summarizer must match full charts on 50 cases before it is relied on.
- **Rationale:** Pre-assembly cuts reviewer minutes, and the filter keeps gold labels blinded to EEG, reports and model outputs.
- **Source:** Reference standard, packet paragraph; Adjudication budget.

### D-013 Phase 0b pilot reports three additional outputs
- **Date:** 2026-10-07 · **Area:** labels · **Outcome data seen?** No
- **Decision:** The 200-case blinded pilot reports measured minutes per case, per-label kappa, and the share of positive silver labels whose only evidence was EEG-derived.
- **Rationale:** Measured minutes replace assumed ones, and the EEG-only share measures how much circularity would have hurt.
- **Source:** Phase 0b, label feasibility pilot.

---

## Clinical baselines

### D-014 Add t0 sedative and opioid exposure to Baseline A
- **Date:** 2026-10-07 · **Area:** baselines · **Outcome data seen?** No
- **Decision:** Baseline A includes t0 sedative and opioid exposure, so H1 compares EEG against severity plus sedation.
- **Rationale:** Without drug data the baseline cannot account for propofol or benzodiazepine signatures, which inflates EEG's apparent gain.
- **Source:** Clinical baselines table, Baseline A (H1 "changed: sedation added"); Critical evaluation table, row 1.

### D-015 Add Baseline D referral-question comparator
- **Date:** 2026-10-07 · **Area:** baselines · **Outcome data seen?** No
- **Decision:** Add Baseline D (Baseline C plus the EEG referral indication category, indication field only) as a sensitivity analysis.
- **Rationale:** It serves as a proxy for the clinician's prior, and it uses only the pre-EEG indication field rather than EEG findings.
- **Source:** Clinical baselines table, Baseline D; Population and index time.

### D-016 Add early-EEG subgroup (t0 before head CT result)
- **Date:** 2026-10-07 · **Area:** baselines · **Outcome data seen?** No
- **Decision:** Add an early-EEG subgroup in which t0 precedes the head CT result, used to test H5.
- **Rationale:** Clinical EEG often starts hours after CT and labs return, so measuring 1B value only there misses the early-triage moment.
- **Source:** Critical evaluation table, row "Study 1B comparator = everything known at t0"; Population and index time.

---

## Hypotheses

### D-017 Preregister H1–H6 with the plan's criteria and roles
- **Date:** 2026-10-07 · **Area:** hypotheses · **Outcome data seen?** No
- **Decision:** H1–H6 are preregistered with the success criteria and roles in the plan's table before any unblinding.
- **Rationale:** Writing predictions down in advance turns per-label results into tests of stated hypotheses rather than a story told afterward.
- **Source:** Core question, claims and hypotheses: Preregistered hypotheses table and the paragraph before H4.

### D-018 H1 and H2 success criterion
- **Date:** 2026-10-07 · **Area:** hypotheses · **Outcome data seen?** No
- **Decision:** H1 (Study 1A) and H2 (Study 1B) succeed only if the 95% CI for Δ log loss is below 0 and the point estimate is below 0, with the per-site condition in D-023.
- **Rationale:** Requiring both a CI excluding zero and a negative point estimate makes the primary claim directional and checkable.
- **Source:** Preregistered hypotheses table, H1–H2.

### D-019 H3 severity-stratified check
- **Date:** 2026-10-07 · **Area:** hypotheses · **Outcome data seen?** No
- **Decision:** H3 (key secondary) requires Δ below 0 within at least 2 of 3 severity strata defined by GCS, FOUR or NESI.
- **Rationale:** It tests the risk that the signal is only depth of unconsciousness, and its failure leads to a boundary-map paper.
- **Source:** Preregistered hypotheses table, H3; Critical evaluation table, "Signal is depth and sedation only."

### D-020 H4 identifiability ranking, reported whichever way it falls
- **Date:** 2026-10-07 · **Area:** hypotheses · **Outcome data seen?** No
- **Decision:** H4 predicts identifiability ranking E3 > E2 > E1 > E4a > E5 > E6, with E6 vs E5 near chance, recorded before unblinding and reported whichever way it falls.
- **Rationale:** Writing the ranking down first turns a messy per-label result into a test of a stated prediction.
- **Source:** Preregistered hypotheses table, H4; paragraph after the table.

### D-021 H5 early-EEG interaction
- **Date:** 2026-10-07 · **Area:** hypotheses · **Outcome data seen?** No
- **Decision:** H5 tests whether the 1B gain is larger when t0 precedes the head CT result, reported as an interaction estimate with 95% CI.
- **Rationale:** It is the intended-use proxy for early triage; H5 is dropped if imaging finalization times fail the Phase 0 audit.
- **Source:** Preregistered hypotheses table, H5; Phase 0a, imaging finalization row.

### D-022 H6 short-recording ratio (exploratory)
- **Date:** 2026-10-07 · **Area:** hypotheses · **Outcome data seen?** No
- **Decision:** H6 (device design, exploratory) tests whether at least 70% of the 10-minute gain is present by 2 minutes, reported as the ratio Δ(2 min)/Δ(10 min) with CI.
- **Rationale:** It informs short-recording device design and makes no claim about any reduced montage before Study 2.
- **Source:** Preregistered hypotheses table, H6; Not-claimed paragraph.

---

## Splits and analyses

### D-023 Every held-out site must show a favorable Δ
- **Date:** 2026-10-07 · **Area:** splits · **Outcome data seen?** No
- **Decision:** Every held-out site must show a favorable Δ, and each site is reported separately.
- **Rationale:** Three folds are thin, two sites may share a system or EEG vendor, and a within-site bootstrap hides between-site variance.
- **Source:** Critical evaluation table, row "Leave-one-site-out across 3 adult sites"; Splits and analyses.

### D-024 Site minimums and grouped-split fallback
- **Date:** 2026-10-07 · **Area:** splits · **Outcome data seen?** No
- **Decision:** Leave-one-site-out requires at least 3 adult sites with at least 300 candidates each; if site identifiers fail, a grouped split is used with a weaker claim.
- **Rationale:** The minimum makes each held-out fold estimable, and the grouped split is the plan's stated fallback.
- **Source:** Phase 0a, site identifier row; Study 1, Splits and analyses.

### D-025 Add late-calendar temporal holdout within sites
- **Date:** 2026-10-07 · **Area:** splits · **Outcome data seen?** No
- **Decision:** A late-calendar temporal holdout within sites is added alongside leave-one-site-out.
- **Rationale:** It is a second generalization check across calendar time within each site, complementing the between-site folds.
- **Source:** Splits and analyses, first paragraph.

### D-026 Per-label risk–coverage curves
- **Date:** 2026-10-07 · **Area:** splits · **Outcome data seen?** No
- **Decision:** Per-label risk–coverage curves are added to the analyses.
- **Rationale:** The prospective target requires probabilities calibrated well enough to say "likely" and "I don't know," which these curves show directly.
- **Source:** Splits and analyses, "Added" sentence; Prospective and regulatory program, closing paragraph.

### D-027 Evaluation size set by formal calculation after the pilot
- **Date:** 2026-10-07 · **Area:** splits · **Outcome data seen?** No
- **Decision:** The target of about 1,000 consecutive gold evaluation cases is recalculated after the pilot using formal external-validation sample-size methods.
- **Rationale:** Measured pilot minutes and kappa change what is feasible, and sizing by habit is the approach the plan rejects.
- **Source:** Splits and analyses, evaluation-size paragraph; Critical evaluation table, adjudication row.

### D-028 Each Study 2 dataset tests one thing and is never pooled
- **Date:** 2026-10-07 · **Area:** splits · **Outcome data seen?** No
- **Decision:** Each external dataset (CERTA, I-CARE 2.0, 2024 Korean NCSE cohort, BDSP toxic-metabolic cohort, SPaRCNet/TUSZ) tests one stated question and is never pooled into training or used as the source of a class.
- **Rationale:** Keeping external sets out of training preserves them as valid stress tests.
- **Source:** Study 2, External stress tests.

### D-029 Study 2 runs after Gate 2 on a frozen model
- **Date:** 2026-10-07 · **Area:** splits · **Outcome data seen?** No
- **Decision:** Study 2 runs only after Gate 2, on a model frozen at the end of Study 1, and the BDSP toxic-metabolic cohort stays untouched until that freeze.
- **Rationale:** A frozen model prevents tuning toward the external data.
- **Source:** Study 2 introduction; External stress tests table, BDSP row.

### D-030 Optional Study 1D only if H2 succeeds
- **Date:** 2026-10-07 · **Area:** labor · **Outcome data seen?** No
- **Decision:** Optional Study 1D (three clinicians giving differentials on 150 evaluation packets, then revising after seeing model output) runs only if H2 succeeds.
- **Rationale:** Clinician time is committed only after the primary increment is shown, and the study gives a clinician comparator early.
- **Source:** Optional Study 1D.

---

## Data, cohort and Phase 0 audit

### D-031 Strict cohort primary; broad EHR-phenotype cohort reported separately
- **Date:** 2026-10-07 · **Area:** data · **Outcome data seen?** No
- **Decision:** The strict cohort (GCS ≤11 or FOUR ≤12) is primary, and the broad EHR-phenotype cohort is reported separately.
- **Rationale:** The broad cohort depends on GCS-from-EHR tools, so its results are kept apart from the strict cohort.
- **Source:** Population and index time; Phase 0a, GCS row.

### D-032 Record EEG referral indication as a category
- **Date:** 2026-10-07 · **Area:** data · **Outcome data seen?** No
- **Decision:** The EEG referral indication is recorded as a category field (e.g., "rule out NCSE", "post-arrest", "unexplained AMS").
- **Rationale:** It defines the spectrum of the cohort and powers Baseline D.
- **Source:** Population and index time; Clinical baselines, Baseline D.

### D-033 Phase 0a stop rules
- **Date:** 2026-10-07 · **Area:** data · **Outcome data seen?** No
- **Decision:** Study 1 stops if EEG start date and time are not present for at least 95% of acute-care EEGs, if the within-patient date shift is inconsistent on 20 hand-checked cases, or if timestamped notes are missing.
- **Rationale:** Without these, t0, all timing logic and ACI onset cannot be trusted, and Study 1B is impossible.
- **Source:** Phase 0a field audit table, rows marked "Stop."

### D-034 Phase 0a fallbacks for non-stop fields
- **Date:** 2026-10-07 · **Area:** data · **Outcome data seen?** No
- **Decision:** Where a non-stop field fails: medication times fall back to orders with 1B labeled "approximate" (if administration times are under 80% present); lab times use collection time plus assay lag with 1B labeled "approximate"; failed imaging finalization drops H5; a score within ±6 h present for under 50% limits GCS to the broad cohort via the BDSP tool; a failed site identifier uses a grouped split.
- **Rationale:** Each fallback keeps a publishable result while stating the weaker claim.
- **Source:** Phase 0a field audit table, "If it fails" column.

### D-035 Metadata-first manifest with no recursive syncs
- **Date:** 2026-10-07 · **Area:** data · **Outcome data seen?** No
- **Decision:** Pull metadata first for every dataset, record exact version and size in the Phase 0 inventory, and never run a recursive sync of the HEEDB or BIND roots.
- **Rationale:** Corpus assembly, not access, is the bottleneck, and full syncs pull data the study does not need.
- **Source:** Data acquisition manifest, introduction.

### D-036 BIND pulls limited to candidates
- **Date:** 2026-10-07 · **Area:** data · **Outcome data seen?** No
- **Decision:** BIND pathology labels and report metadata are pulled for candidates only.
- **Rationale:** It limits restricted-data movement to the adjudication candidate set, consistent with D-035.
- **Source:** Data acquisition manifest, BIND row.

### D-037 Count TUEG once; TUH subsets are label sources and contaminated benchmarks
- **Date:** 2026-10-07 · **Area:** data · **Outcome data seen?** No
- **Decision:** TUEG is counted once; TUSZ, TUAB, TUEV and TUAR are treated as label sources and contaminated benchmarks, not as added hours.
- **Rationale:** They are subsets drawn from TUEG and were inside CBraMod's pretraining.
- **Source:** Critical evaluation table, row "TUSZ, TUAB, TUEV listed as extra hours."

### D-038 Re-verify unconfirmed citations before use
- **Date:** 2026-10-07 · **Area:** regulatory · **Outcome data seen?** No
- **Decision:** Before any use in a protocol, grant or Pre-Sub, re-verify the January 2026 CDS guidance, the August 2026 human-factors guidance, the CLEF preprint, and the claim that one BDSP restricted DUA covers every restricted dataset.
- **Rationale:** These citations came from another assistant and could not all be confirmed.
- **Source:** Critical evaluation of the source material, paragraph after the table.

### D-039 Begin no modeling before Phase 0 passes
- **Date:** 2026-10-07 · **Area:** data · **Outcome data seen?** No
- **Decision:** No model is trained until the field audit and the 200-case pilot both pass.
- **Rationale:** Phase 0 decides whether Study 1B is possible at all, and it costs weeks rather than months.
- **Source:** Phase 0 introduction.

---

## Montage and hardware

### D-040 Four deployable geometries replace the channel-count ablation
- **Date:** 2026-10-07 · **Area:** montage · **Outcome data seen?** No
- **Decision:** The generic 19→12→8→6→4-channel ablation is replaced by four geometries: full 19-channel reference; rapid-EEG hairline headband (frontal-temporal-occipital ring); rapid-EEG full-coverage headcap; and forehead-only array. Exact electrode lists and referencing are confirmed from vendor documentation before simulation.
- **Rationale:** Rapid-EEG hardware fixes the electrode geometry that will be deployed, so geometry rather than channel count is what needs testing.
- **Source:** Critical evaluation table, row "Generic 19→12→8→6→4-channel ablation"; Montage section, geometries 1–3.

### D-041 Data-driven 4–8 electrode set chosen inside training folds
- **Date:** 2026-10-07 · **Area:** montage · **Outcome data seen?** No
- **Decision:** A data-driven 4–8 electrode set is added as a fourth geometry, selected entirely inside training folds.
- **Rationale:** Selection inside training folds keeps held-out sites untouched.
- **Source:** Montage section, geometry 4.

### D-042 Nested duration windows
- **Date:** 2026-10-07 · **Area:** montage · **Outcome data seen?** No
- **Decision:** Δ is reported at prespecified nested windows of 20 s, 1, 2, 5 and 10 min from the start of the primary window.
- **Rationale:** These windows give the duration curve needed for H6 and Gate 4.
- **Source:** Montage section, "Duration" paragraph.

### D-043 Simulated electrode deletion requires paired prospective recording
- **Date:** 2026-10-07 · **Area:** montage · **Outcome data seen?** No
- **Decision:** Any reduced montage needs a paired prospective recording (rapid vs conventional) before it is claimed as deployable.
- **Rationale:** Simulated deletion ignores impedance, placement error and real artifact.
- **Source:** Montage section, closing paragraph; Prospective and regulatory program, P2.

---

## Labor and team

### D-044 Literature sweep and early BDSP collaboration offer
- **Date:** 2026-10-07 · **Area:** labor · **Outcome data seen?** No
- **Decision:** Phase 0 includes a 2024–2026 literature and preprint novelty sweep and a decision on whether to offer the BDSP group a collaboration.
- **Rationale:** The BDSP group built HEEDB, MORGOTH, NESI and the TME cohort and is the likeliest group to publish this analysis first.
- **Source:** Phase 0c; Go/no-go gates and risk register, "Scooped by the group that built HEEDB."

### D-045 Adjudication budget reduced
- **Date:** 2026-10-07 · **Area:** labor · **Outcome data seen?** No
- **Decision:** Development gold cases fall from about 800 to 400, total gold cases from about 2,100 to about 1,700 (enriched set about 300 retained), and reviews from about 4,725 to about 3,405.
- **Rationale:** Silver labels now do the training, so gold development cases only calibrate and select models.
- **Source:** Team, labor budget and timeline, adjudication budget table and assumptions.

### D-046 Physician-hour planning budget
- **Date:** 2026-10-07 · **Area:** labor · **Outcome data seen?** No
- **Decision:** Study 1 adjudication is planned at 570–850 physician-hours (10–15 min per review with pre-assembled packets), replacing the handoff's 1,180–1,970, and is subject to replacement by measured pilot minutes.
- **Rationale:** Pre-assembled packets and evaluation-only dual review are what bring the estimate down, and the pilot measures the actual minutes.
- **Source:** Team, labor budget and timeline, adjudication budget table; Phase 0b.

### D-047 Named adjudicator pool, authorship-based
- **Date:** 2026-10-07 · **Area:** labor · **Outcome data seen?** No
- **Decision:** Adjudication is done by a named pool of 6–10 neurology, neurocritical-care, EM or IM trainees and fellows, with authorship as the participation incentive.
- **Rationale:** Study 1 needs roughly 570–850 physician-hours and is not a solo project during residency.
- **Source:** Team, labor budget and timeline, team table (adjudicator pool row).

### D-048 Biostatistician and senior co-investigator roles
- **Date:** 2026-10-07 · **Area:** labor · **Outcome data seen?** No
- **Decision:** A part-time biostatistician (sample size, proper scoring, bootstrap design, preregistration) and a senior EEG or neurocritical-care co-investigator, ideally BDSP-affiliated, are added to the team.
- **Rationale:** Formal sample sizing and proper scoring need statistical ownership, and the ontology and E3 adjudication need clinician credibility.
- **Source:** Team, labor budget and timeline, team table.

### D-049 Scope to Studies 1A and 1B first; collaborators own adjudication
- **Date:** 2026-10-07 · **Area:** labor · **Outcome data seen?** No
- **Decision:** Work is scoped to Studies 1A and 1B first, with collaborators owning adjudication.
- **Rationale:** The clinical training schedule threatens the milestones, and this scope is the mitigation the plan's risk register names.
- **Source:** Go/no-go gates and risk register, "Clinical training schedule stalls the work."

### D-050 Timeline dates are a proposal to reset after Phase 0
- **Date:** 2026-10-07 · **Area:** labor · **Outcome data seen?** No
- **Decision:** The timeline is treated as a proposal, to be reset after Phase 0.
- **Rationale:** Phase 0 results (field audit, pilot minutes, kappa) determine the feasible schedule.
- **Source:** Team, labor budget and timeline, introduction.

---

## Gates

### D-051 G0 feasibility gate
- **Date:** 2026-10-07 · **Area:** data · **Outcome data seen?** No
- **Decision:** G0 passes only if every Phase 0a "Stop" row passes, pilot kappa is at least 0.6 for at least four primary families, and at least three families have at least 10% prevalence; failure leads to a redesign around a smaller cohort or CERTA collaboration, with no full study.
- **Rationale:** It confirms the labels and timestamps can support Study 1B before any modeling.
- **Source:** Go/no-go gates and risk register, G0; Phase 0 gate paragraph.

### D-052 G2 etiologic-information gate
- **Date:** 2026-10-07 · **Area:** hypotheses · **Outcome data seen?** No
- **Decision:** G2 passes if H1 is met, the result survives severity matching and sedative adjustment, and at least 3 primary families improve individually; 1–2 families yields a narrow product, and none yields a boundary-map publication with the commercial track stopped.
- **Rationale:** The commercial track is conditioned on evidence that EEG adds etiologic information beyond severity and sedation.
- **Source:** Go/no-go gates and risk register, G2.

### D-053 G3 clinical-increment gate
- **Date:** 2026-10-07 · **Area:** hypotheses · **Outcome data seen?** No
- **Decision:** G3 passes only if H2 is met at every held-out site; failure yields an academic paper only, unless H5 shows a clear early-EEG gain justifying a narrower prospective test.
- **Rationale:** The product's value depends on the increment over the full t0 clinical model.
- **Source:** Go/no-go gates and risk register, G3.

### D-054 G4 deployable-EEG gate
- **Date:** 2026-10-07 · **Area:** montage · **Outcome data seen?** No
- **Decision:** G4 passes if the hairline geometry keeps at least 70% of the full-montage Δ and the 2-minute recording keeps at least 70% of the 10-minute Δ; failure limits the product to a conventional-EEG software product.
- **Rationale:** The deployment path on rapid-EEG hardware is viable only if most of the signal survives on that hardware and in shorter recordings.
- **Source:** Go/no-go gates and risk register, G4.

### D-055 G5 prospective gate
- **Date:** 2026-10-07 · **Area:** prospective · **Outcome data seen?** No
- **Decision:** G5 passes only if P2 shows the four de-risking conditions at once (works in consecutive patients, at hospitals never used in training, beyond what clinicians already know, and calibrated enough to say "likely" and "I don't know"); failure means no FDA submission and reassessment of the indication.
- **Rationale:** The plan defines de-risking as these four conditions together, so a partial pass is not treated as de-risked.
- **Source:** Go/no-go gates and risk register, G5; Prospective and regulatory program, closing paragraph.

### D-056 Gate IDs kept as written; G1 not invented
- **Date:** 2026-10-07 · **Area:** hypotheses · **Outcome data seen?** No
- **Decision:** Gate IDs stay G0 and G2–G5 exactly as the plan writes them; no G1 is defined, and the gap is flagged for the v1.1 revision.
- **Rationale:** The plan defines no G1, and inventing one would change the decision structure it sets out.
- **Source:** Go/no-go gates and risk register; Phase 0 and Study 1 gate references.

---

## Prospective and regulatory program

### D-057 Prospective enrollment only at rapid-EEG sites; consent by waiver or alteration
- **Date:** 2026-10-07 · **Area:** prospective · **Outcome data seen?** No
- **Decision:** Prospective enrollment is limited to sites that already use rapid EEG for AMS as standard care, with research as data capture and a hidden model, and consent is sought by waiver or alteration rather than surrogate consent for research-only EEG.
- **Rationale:** Consecutive enrollment within 30–60 min is not achievable with conventional EEG at most hospitals, and surrogate consent biases who is enrolled; the IRB makes the actual consent determination.
- **Source:** Prospective and regulatory program, "Why rapid-EEG sites" and stage table.

### D-058 P1 silent run-in criterion
- **Date:** 2026-10-07 · **Area:** prospective · **Outcome data seen?** No
- **Decision:** P1 enrolls 100–200 consecutive patients at 1–2 sites and passes at 80% or more of eligible patients captured with usable EEG.
- **Rationale:** It proves enrollment, timing, QC, adjudication and latency work before the decisive study.
- **Source:** Prospective and regulatory program, stage table, P1.

### D-059 P2 multicenter silent validation as the decisive test
- **Date:** 2026-10-07 · **Area:** prospective · **Outcome data seen?** No
- **Decision:** P2 enrolls 1,500–3,000 consecutive patients at 5–10 sites, includes a paired rapid-vs-conventional substudy, and exits only with Δ below 0 at sites never used in training and a calibration slope of 0.8–1.2.
- **Rationale:** P2 is the test of whether the signal holds at unseen hospitals and is calibrated enough to be used.
- **Source:** Prospective and regulatory program, stage table, P2.

### D-060 P3 human-in-the-loop study and human-factors evidence
- **Date:** 2026-10-07 · **Area:** prospective · **Outcome data seen?** No
- **Decision:** P3 (100–300 patients) follows DECIDE-AI, and its human-factors work is counted as regulatory evidence.
- **Rationale:** Display, anchoring and abstention behavior must be understood before any utility trial.
- **Source:** Prospective and regulatory program, stage table, P3.

### D-061 P4 utility-RCT endpoint chosen from P2
- **Date:** 2026-10-07 · **Area:** prospective · **Outcome data seen?** No
- **Decision:** The P4 utility RCT endpoint is chosen from P2's strongest actionable signal and reported per SPIRIT-AI and CONSORT-AI.
- **Rationale:** It avoids fixing an endpoint before the prospective signal is known.
- **Source:** Prospective and regulatory program, stage table, P4.

### D-062 Plan for De Novo classification
- **Date:** 2026-10-07 · **Area:** regulatory · **Outcome data seen?** No
- **Decision:** The regulatory plan assumes De Novo classification using BrainScope DEN140025 as the template, not 510(k).
- **Rationale:** The only adjunctive interpretive EEG classification (21 CFR 882.1450) covers structural injury in trauma, so a multi-etiology aid is likely a new classification.
- **Source:** Prospective and regulatory program, "Regulatory framing"; Sources list (eCFR, FDA product classification).

### D-063 Narrow first claim
- **Date:** 2026-10-07 · **Area:** regulatory · **Outcome data seen?** No
- **Decision:** The first claim is framed as an adjunctive aid that estimates probabilities of predefined etiologic families, never as a diagnosis.
- **Rationale:** The plan's "Not claimed" list excludes exact metabolic subtypes and single-cause diagnosis, and the narrow claim matches it.
- **Source:** Prospective and regulatory program, "Regulatory framing"; Core question, "Not claimed" paragraph.

### D-064 Predetermined Change Control Plan and QMSR design controls
- **Date:** 2026-10-07 · **Area:** regulatory · **Outcome data seen?** No
- **Decision:** A Predetermined Change Control Plan is used if the model updates after authorization, and design controls run under the QMSR from the first commercial line of code.
- **Rationale:** Update and design-history expectations apply from the start of product work, not after authorization.
- **Source:** Prospective and regulatory program, "Regulatory framing."

### D-065 Pre-Sub: HEEDB results cited as literature, not submitted as data
- **Date:** 2026-10-07 · **Area:** regulatory · **Outcome data seen?** No
- **Decision:** The P0 FDA Pre-Sub agrees intended use, reference standard and pivotal design, and HEEDB results may be cited as literature but not submitted as data.
- **Rationale:** HEEDB is research-only data under a DUA, so its data cannot serve as submitted evidence.
- **Source:** Prospective and regulatory program, stage table, P0; Data acquisition manifest, HEEDB row.

---

## Commercial track

### D-066 No commercial build before Study 1 passes Gate 2
- **Date:** 2026-10-07 · **Area:** commercial · **Outcome data seen?** No
- **Decision:** Nothing commercial is built until Study 1 passes Gate 2, and the commercial set-up row starts only if G2 and G3 pass.
- **Rationale:** Commercial spending should wait for evidence that the signal exists.
- **Source:** Executive summary, sequence paragraph; Team, labor budget and timeline, introduction.

### D-067 Defer encoder retraining; provenance work only before Gate 2
- **Date:** 2026-10-07 · **Area:** commercial · **Outcome data seen?** No
- **Decision:** Encoder retraining is deferred until after Gate 2; pre-Gate work is limited to provenance (rights ledger, production-build refusal rule, TUEG manifest hash, CBraMod hash and training-data trace).
- **Rationale:** Retraining now would consume weeks of GPU and engineering before the signal is known, and the Study 1 gap decides whether it is needed.
- **Source:** Critical evaluation table, row "Commercial encoder retraining runs in parallel now"; Commercial track, "Now, before Gate 2."

### D-068 CBraMod frozen embeddings alongside MORGOTH; commercial-clean gap
- **Date:** 2026-10-07 · **Area:** commercial · **Outcome data seen?** No
- **Decision:** Frozen CBraMod embeddings are run on HEEDB alongside MORGOTH, and the difference in Δ between them is the "commercial-clean gap."
- **Rationale:** The plan identifies this gap as the single most useful number for the commercial encoder decision.
- **Source:** Representations and models.

### D-069 Post-Gate-2 encoder options C1–C3
- **Date:** 2026-10-07 · **Area:** commercial · **Outcome data seen?** No
- **Decision:** After Gate 2, choose C1 (frozen public CBraMod) if the gap is small and counsel clears the checkpoint; C2 (CBraMod architecture retrained on TUEG) if the gap is small but checkpoint provenance is unclear; or C3 (C2 plus continued pretraining on licensed acute-care EEG) if the gap is large.
- **Rationale:** It maps the gap and provenance outcomes to a specific encoder path.
- **Source:** Commercial track, "After Gate 2: encoder decision" table.

### D-070 Proprietary etiologic dataset routes
- **Date:** 2026-10-07 · **Area:** commercial · **Outcome data seen?** No
- **Decision:** Route 1 (licensed hospital EEG archive plus EHR, retrospective) supplies commercial training labels, route 3 (prospective commercial cohort) supplies validation, and route 2 (rapid-EEG vendor partnership) is kept only as a strategic option.
- **Rationale:** The cheapest first proprietary dataset is retrospective, and route 2 gives deployment-hardware EEG without etiologic labels.
- **Source:** Commercial track, "The proprietary etiologic dataset" table and paragraph.

### D-071 Required grants in every data agreement
- **Date:** 2026-10-07 · **Area:** commercial · **Outcome data seen?** No
- **Decision:** Every agreement must grant commercial model training, derivative weights, regulatory submission, continued improvement, sublicensing or spinout, and audit access.
- **Rationale:** Without these rights the commercial model cannot be trained, submitted or spun out.
- **Source:** Commercial track, paragraph after the route table.

### D-072 Dataset commercial-status posture
- **Date:** 2026-10-07 · **Area:** commercial · **Outcome data seen?** No
- **Decision:** NMT and VitalDB stay yellow or on hold until written clarification; MIMIC and eICU are excluded; CLEF stays excluded until its weights are confirmed obtainable under BDSP terms; the TUEG family and the CBraMod checkpoint are commercial candidates pending counsel review; TDBRAIN is a commercial candidate.
- **Rationale:** Share-alike and conflicting licence statements block commercial use, and MIMIC and eICU lack the raw clinical EEG pairing the project needs.
- **Source:** Data acquisition manifest, status column; closing paragraph of the manifest.

### D-073 Ceribell as default competitor; design for its hardware
- **Date:** 2026-10-07 · **Area:** commercial · **Outcome data seen?** No
- **Decision:** Ceribell is treated as the default competitor, partner or acquirer, and the product is designed to run on its rapid-EEG hardware.
- **Rationale:** Ceribell already sells rapid EEG with FDA-cleared seizure and delirium indications and added epileptiform detection in August 2026.
- **Source:** Critical evaluation table, row "No competitive landscape"; Go/no-go gates and risk register, "A rapid-EEG vendor builds this first."

### D-074 NIH SBIR/STTR funding, STTR preferred
- **Date:** 2026-10-07 · **Area:** commercial · **Outcome data seen?** No
- **Decision:** NIH SBIR/STTR funding is pursued for the commercial track once a company entity exists, with STTR preferred.
- **Rationale:** STTR pairs a small business with an academic partner, which matches a company licensing an academic collaborator's cohort.
- **Source:** Commercial track, "Entity and funding."

---

## Firewall and IP

### D-075 Commercial track uses only preprinted Study 1 and 2 conclusions
- **Date:** 2026-10-07 · **Area:** firewall · **Outcome data seen?** No
- **Decision:** The commercial track may use Study 1 and Study 2 conclusions only after they are preprinted.
- **Rationale:** It makes the research-to-commercial boundary auditable, since every commercial decision must trace to a public document.
- **Source:** Program architecture, Firewall rules (additions to handoff section 1.2), first bullet.

### D-076 Dated log of commercial design decisions
- **Date:** 2026-10-07 · **Area:** firewall · **Outcome data seen?** No
- **Decision:** A dated log is kept of every commercial design decision and the public source it rests on.
- **Rationale:** The log is the audit trail that makes the firewall checkable by anyone.
- **Source:** Program architecture, Firewall rules, second bullet.

### D-077 Provisional patent filed before preprint
- **Date:** 2026-10-07 · **Area:** firewall · **Outcome data seen?** No
- **Decision:** Any provisional patent is filed before the preprint.
- **Rationale:** The plan states the timing only; the reason (preserving patent position before public disclosure) is inferred.
- **Source:** Program architecture, Firewall rules, first bullet.

### D-078 Counsel sign-off on the firewall boundary
- **Date:** 2026-10-07 · **Area:** firewall · **Outcome data seen?** No
- **Decision:** The firewall boundary requires institutional counsel's sign-off, not only the lead's own judgment.
- **Rationale:** The lead is the conduit between tracks, so the lead cannot be the only check on what crosses.
- **Source:** Program architecture, Firewall rules, third bullet.

### D-079 IP review before entity, invention disclosure or data license
- **Date:** 2026-10-07 · **Area:** firewall · **Outcome data seen?** No
- **Decision:** IP ownership is settled with the current employer and the incoming residency institution before forming an entity, filing an invention disclosure, or signing a data license; any BDSP-affiliated collaborator's institutional IP interests are agreed in writing before Study 1 work starts; the Phase 0d review reads both agreements' IP clauses and asks who owns personal-time work using externally licensed data.
- **Rationale:** An employer IP claim could undermine the company and its licences, and the plan says the Phase 0 review is more urgent given the future residency health system as a likely archive.
- **Source:** Program architecture, Firewall rules, fourth bullet; Phase 0d; Commercial track, "Entity and funding" and closing paragraph.

---

## Agent safety for restricted data

### D-080 Agents develop against a synthetic HEEDB-schema dataset
- **Date:** 2026-10-07 · **Area:** safety · **Outcome data seen?** No
- **Decision:** Agents develop against a synthetic dataset with HEEDB's schema and never read the real tables.
- **Rationale:** Running df.head() on a HEEDB table sends record-level data into a hosted model's context, which the BDSP terms prohibit.
- **Source:** Phase 0e, first bullet.

### D-081 Aggregate-only outputs with small-cell suppression
- **Date:** 2026-10-07 · **Area:** safety · **Outcome data seen?** No
- **Decision:** Scripts that touch restricted data print only aggregates, with small-cell suppression (n < 11 shown as "<11").
- **Rationale:** It keeps record-level values out of agent context and logs.
- **Source:** Phase 0e, second bullet.

### D-082 Restricted-data jobs run outside agent sessions
- **Date:** 2026-10-07 · **Area:** safety · **Outcome data seen?** No
- **Decision:** Restricted-data jobs run from a plain terminal or scheduler, never inside an agent session.
- **Rationale:** An agent session that executes restricted jobs creates the same leakage path as D-080.
- **Source:** Phase 0e, third bullet; Risk register, "Restricted data reaches a hosted model."

### D-083 Open-weight LLMs on approved compute for note reading
- **Date:** 2026-10-07 · **Area:** safety · **Outcome data seen?** No
- **Decision:** Any LLM used to read notes is an open-weight model running on approved compute.
- **Rationale:** A hosted model would receive restricted note text, which the BDSP terms prohibit.
- **Source:** Phase 0e, fourth bullet; Reference standard, packet paragraph.

---

## Silver-anchor rule changes (labels)

### D-084 Remove creatinine from E5 anchors
- **Date:** 2026-10-07 · **Area:** labels · **Outcome data seen?** No
- **Decision:** Delete the E5_creatinine anchor (creatinine >= 6.0 mg/dL); BUN >= 100 mg/dL and the other E5 anchors stay.
- **Rationale:** A creatinine threshold flags patients with stable end-stage renal disease, whose level does not explain acute coma.
- **Source:** Project lead sign-off of the proposed silver-anchor rules (2026-10-07); Silver labels: the circularity rules; builds on D-009.

### D-085 Require acidemia for the E5 PaCO2 anchor
- **Date:** 2026-10-07 · **Area:** labels · **Outcome data seen?** No
- **Decision:** E5_paco2 becomes an all_of: PaCO2 > 70 mmHg AND arterial pH < 7.30, both within [-12, +6] h of t0 (pH within 1 h of the PaCO2 is ideal; implemented as both in window).
- **Rationale:** PaCO2 alone includes chronic CO2 retainers; acidemia separates acute hypercapnia from compensated chronic retention.
- **Source:** Project lead sign-off of the proposed silver-anchor rules (2026-10-07); Silver labels: the circularity rules; builds on D-009.

### D-086 End the E5 hypoglycemia window at +1 h
- **Date:** 2026-10-07 · **Area:** labels · **Outcome data seen?** No
- **Decision:** E5_glucose_lo (glucose < 50 mg/dL) window changes from [-12, +6] to [-12, +1] h.
- **Rationale:** Hypoglycemia found well after t0 may be iatrogenic (insulin, treatment of the unwell patient) and cannot be the cause of the state at t0.
- **Source:** Project lead sign-off of the proposed silver-anchor rules (2026-10-07); Silver labels: the circularity rules; builds on D-009.

### D-087 Replace E6 with a CDC Adult Sepsis Event style definition
- **Date:** 2026-10-07 · **Area:** labels · **Outcome data seen?** No
- **Decision:** E6 becomes all_of: blood culture drawn [-72, +24] h; >= 4 consecutive qualifying antimicrobial days starting within +-2 days of the culture (flag qad_ge4, computed upstream) [-72, +24] h; and any organ dysfunction within [-48, +24] h: vasopressor initiation, lactate >= 2.0 mmol/L, creatinine doubling vs encounter baseline excluding ESRD, bilirubin >= 2.0 mg/dL with doubling, or platelets < 100 x10^3/uL with >= 50% decline from a baseline >= 100. SIRS criteria, the suspected_infection item and the bacteremia branch are removed; new mechanical ventilation is explicitly not used; exclude_if_label E7 is kept.
- **Rationale:** SIRS is near-universal in comatose ICU patients and would destroy E6 specificity; the CDC Adult Sepsis Event definition is validated for EHR surveillance. Comatose patients are intubated for airway protection, so ventilation is not organ dysfunction evidence.
- **Source:** Project lead sign-off of the proposed silver-anchor rules (2026-10-07); Silver labels: the circularity rules; builds on D-009.

### D-088 Redefine E2 profound shock as sustained MAP < 50 mmHg for >= 30 min
- **Date:** 2026-10-07 · **Area:** labels · **Outcome data seen?** No
- **Decision:** The profound_shock flag (computed upstream) means sustained MAP < 50 mmHg for >= 30 minutes, window [-48, 0] h; the earlier alternative of >= 2 vasopressors is dropped.
- **Rationale:** Two or more vasopressors without sustained hypotension is ordinary septic shock, not hypoxic-ischemic injury.
- **Source:** Project lead sign-off of the proposed silver-anchor rules (2026-10-07); Silver labels: the circularity rules; builds on D-009.

### D-089 Require acute or subacute imaging for E1
- **Date:** 2026-10-07 · **Area:** labels · **Outcome data seen?** No
- **Decision:** Add a top-level note and item comments that E1 imaging flags must be acute or subacute findings (qualifiers acute, new, hyperacute, subacute; exclude chronic, old, remote, sequela). Add acuity_required: true to E1; anchors.py enforces it when the event table carries an optional boolean acute column (only acute == True rows count).
- **Rationale:** Chronic imaging findings do not explain acute coma.
- **Source:** Project lead sign-off of the proposed silver-anchor rules (2026-10-07); Silver labels: the circularity rules; builds on D-009.

### D-090 Apply the E7 CSF WBC threshold to the RBC-corrected count
- **Date:** 2026-10-07 · **Area:** labels · **Outcome data seen?** No
- **Decision:** Add item csf_rbc (no LOINC mapped; 26455-6 noted as an unverified candidate). Corrected WBC = csf_wbc - csf_rbc/500 when both come from the same tap (same timestamp); otherwise the uncorrected count is used.
- **Rationale:** Traumatic taps inflate CSF WBC and would create false E7 positives.
- **Source:** Project lead sign-off of the proposed silver-anchor rules (2026-10-07); Silver labels: the circularity rules; builds on D-009.

### D-091 Mark the ICD-9 banned prefixes as verified
- **Date:** 2026-10-07 · **Area:** labels · **Outcome data seen?** No
- **Decision:** The project lead verified that ICD-9 prefixes 348.30, 348.31, 348.39 and 349.82 are correct; the UNVERIFIED note in banned_evidence.py is changed accordingly.
- **Rationale:** The prefixes had been recalled from memory rather than tool-enumerated; the lead's check closes that open item.
- **Source:** Project lead sign-off of the proposed silver-anchor rules (2026-10-07); Silver labels: the circularity rules; builds on D-009.

### D-092 Delegate clinical-threshold sign-off to the project lead, with co-I re-review before lock
- **Date:** 2026-10-07 · **Area:** labels · **Outcome data seen?** No
- **Decision:** The silver_anchors.yaml status changes from "PROPOSED — needs co-investigator sign-off" to "SIGNED OFF by project lead 2026-10-07 (delegated decision); re-review with EEG/neurocritical-care co-investigator before Study 1 protocol lock". Per-item notes where clinically uncertain (hepatic and uremic acute-on-chronic, antibiotic pre-treatment for CSF) are kept.
- **Rationale:** The clinical thresholds and windows were decided by the project lead rather than a clinical co-investigator; the co-investigator's re-review is required before the Study 1 protocol is locked.
- **Source:** Project lead sign-off of the proposed silver-anchor rules (2026-10-07); Silver labels: the circularity rules; builds on D-009.

### D-093 Evaluation set stays at about 1,000 gold cases, with a 1,500-case optional reserve
- **Date:** 2026-10-07 · **Area:** hypotheses/labor · **Outcome data seen?** No
- **Decision:** The consecutive gold evaluation set remains about 1,000 cases (at least ~330 per site, each with at least one assessable primary label). A reserve extension to 1,500 cases is optional, to be decided after the Phase 0 pilot; it is not required for H1/H2.
- **Rationale:** Simulation (`sortinghat/metrics/power.py`, `docs/research/evaluation_sample_size.md`) gives H1/H2 power of 0.89 at +0.04 AUROC gain and 0.99 at +0.06 at N = 1,000 (80% from about 610), and only about 5 points more at N = 1,500; power at +0.02 is capped near 0.75 by between-site heterogeneity whatever N is. The 1,500 reserve costs 188 to 281 additional physician-hours (2.25 reviews per case at 10 to 15 minutes), so the 570 to 850 hour budget is unchanged. E7 stays exploratory through the enriched set (100 positives at 3% prevalence would need about 3,700 cases).
- **Source:** SAP section 14; research plan adjudication budget; `docs/research/evaluation_sample_size.md` sections 3, 5, 6.

### D-094 Calibration slope is a precision-reported secondary, not a gate, in Study 1
- **Date:** 2026-10-07 · **Area:** hypotheses · **Outcome data seen?** No
- **Decision:** Per-label calibration slope and O/E are reported with bootstrap CIs and calibration plots, with the expected CI width at the achieved N stated beforehand. They never pass or fail a hypothesis, model, site or gate, and no claim of "well calibrated" or slope near 1 is made from them. Slope is reported for labels with at least 100 expected positives (E1, E2, E5, E6); E4a and E7 get O/E only. Builds on SAP section 7.5.
- **Rationale:** Slope CI width 0.2 needs 2,671 to 15,739 assessable patients (SAP section 14); the expected width at N = 1,000 is 0.41 to 0.57 for E1 to E6. A calibration gate would need about 4,300 to 8,100 cases (roughly 1,600 to 4,600 evaluation-only physician-hours) to protect a property the primary claim does not rest on.
- **Source:** SAP sections 7.5 and 14; `docs/research/evaluation_sample_size.md` sections 4 and 5.

### D-095 H1/H2 use the 99% within-site bootstrap CI together with the every-site rule
- **Date:** 2026-10-07 · **Area:** hypotheses · **Outcome data seen?** No
- **Decision:** H1 and H2 are met when the 99% (two-sided, alpha = 0.01) within-site stratified bootstrap CI for Delta lies below 0 and the Delta point estimate is below 0 at every held-out site. All four intervals stay co-reported at 95%. Refines D-018 and D-023 (the CI level, previously 95%).
- **Rationale:** In simulation with site-level heterogeneity in the EEG gain (tau = 0.015 AUROC), the 95% rule rejects 6.2% (N = 1,000) and 6.9% (N = 1,500) at a true gain of 0; 97.5% gives 4.5% and 5.5%; 99% gives 3.1% and 3.7%, the only level at or below 5% at both. Power cost at +0.04 is 0.89 to 0.86 (N = 1,000) and 0.92 to 0.91 (N = 1,500); none at +0.06. At tau = 0.03 no level gets below 8.6%, so the co-reported two-stage interval remains the guard against between-site variance.
- **Source:** `docs/research/evaluation_sample_size.md` section 8; SAP sections 6 and 7.2.

### D-096 EEG minimum channel set is the 10 hairline electrodes
- **Date:** 2026-10-07 · **Area:** hypotheses · **Outcome data seen?** No
- **Decision:** The minimum channel set for the 60% usable-data rule is Fp1, Fp2, F7, F8, T3, T4, T5, T6, O1, O2 (the Ceribell-headband hairline electrodes), replacing the earlier assumed 16 non-midline 10-20 channels. `DEFAULT_MINIMUM_CHANNELS` (defined in `sortinghat/eeg/io.py`, imported by `sortinghat/eeg/window.py`) and its tests are updated. The epoch rule (90% of the minimum set clean, i.e. 9 of 10) is unchanged in form.
- **Rationale:** Every included recording then supports the Ceribell-headband simulation (Study 2) without a separate inclusion set. This can lower the exclusion rate relative to the 16-channel set and makes QC blind to the other nine channels; both are reported in the exclusion counts.
- **Source:** Project lead instruction (2026-10-07); `configs/montages.yaml` (ceribell_headband); SAP sections 2 and 9.

### D-097 Burst-suppression 5 uV threshold and EEG QC thresholds are provisional
- **Date:** 2026-10-07 · **Area:** hypotheses · **Outcome data seen?** No
- **Decision:** The 5 uV burst-suppression envelope threshold and the EEG QC thresholds (flat 0.5 uV, clipping 5%, extreme 500 uV, line-noise ratio 1.0, disconnected rule, 90% epoch-channel fraction) are provisional. They will be fixed from the Phase 0b pilot, using counts and distributions only and no outcome labels or model results, and frozen before any evaluation-set gold label or model output is opened.
- **Rationale:** The values are untuned defaults; low-voltage traces read as suppressed at 5 uV, and the pilot is the first time real recordings are available. Fixing them from signal properties alone keeps them independent of outcomes.
- **Source:** `docs/eeg_pipeline.md` (QC and features); SAP section 10, item 9 (freeze order).

### D-098 Model selection on training-site gold development cases
- **Date:** 2026-10-07 · **Area:** splits · **Outcome data seen?** No
- **Decision:** Hyperparameters, model selection and recalibration use the gold development cases from the training sites of each leave-one-site-out fold, replacing the SAP's inner site-grouped folds.
- **Rationale:** With three sites, each outer fold has only two training sites, so inner site-grouped folds would be two unstable folds. The held-out site still never influences any fitted parameter (tested in `tests/test_models_cv.py`).
- **Source:** `docs/models_spec.md`; SAP section on model selection.

### D-099 Separate shrinkage on the EEG feature block
- **Date:** 2026-10-07 · **Area:** hypotheses · **Outcome data seen?** No
- **Decision:** The shared multi-label head's selection grid includes a separate L2 penalty on the EEG feature block, so the baseline+EEG model can shrink toward the baseline-only model.
- **Rationale:** Without it, many weak EEG features inflate variance and push null Δ above 0; with it, synthetic no-signal runs keep |Δ| < 0.02 while planted signals are recovered.
- **Source:** `docs/models_spec.md`.

### D-100 Raw HEEDB EEG and tables are streamed and processed in memory; only derived rows are stored
- **Date:** 2026-10-07 · **Area:** data · **Outcome data seen?** No
- **Decision:** Raw HEEDB EEG and tables are streamed from S3 and processed in memory. For each EDF only the header and the minute 1 to 11 byte range (plus 10 s of filter padding either side) are fetched with ranged GETs, decoded in RAM and reduced to feature rows (`sortinghat/eeg/stream.py`, `scripts/extract_eeg_features.py`). Only derived feature rows and a small per-recording status ledger are stored, as parquet and CSV under gitignored `local_only/` paths (mode 0600). No raw signal or table is written to disk or committed, and run output is aggregate-only through `sortinghat.safe_output` (counts below 11 shown as "<11"). Failures are reduced to fixed ID-free reason codes.
- **Rationale:** ICU cEEG EDFs are about 1 GB each and Study 1 uses 10 minutes, so ranged reads cut transfer by roughly two orders of magnitude and avoid a raw-data copy that would need its own access control. Keeping raw bytes in memory only is consistent with CLAUDE.md rules 3, 5 and 6 and the BDSP terms.
- **Source:** `docs/eeg_pipeline.md`; `docs/heedb_access.md` section 1; CLAUDE.md rules 2, 3, 5, 6.

### D-101 Phase 0a "timestamped notes" row operationalised at >=80% of candidates
- **Date:** 2026-10-07 · **Area:** data · **Outcome data seen?** No
- **Decision:** The plan's "Present" criterion for timestamped notes (a Stop row) passes when >=80% of candidates have at least one note with a timestamp.
- **Rationale:** Notes are needed for ACI onset and silver labels in most, not all, patients; 95% would make a Stop row stricter than the plan's own EEG-start row intends for a secondary field, while 80% still supports both uses. Chosen before the audit was run.
- **Source:** Phase 0a table; `sortinghat/audit/field_audit.py`.

### D-102 E1 silver anchors from acute structural diagnosis and neurosurgical procedure codes
- **Date:** 2026-10-07 · **Area:** labels · **Outcome data seen?** No
- **Decision:** Because HEEDB imaging exists only at sites I0001/I0004 (no EEG there), E1 silver positives use an alternative branch: acute structural ICD-10 codes (I60.*, I61.*, I62.0*, I63.*, S06.2*–S06.6*, G93.5/G93.6 only with one of these) or neurosurgical procedures (craniotomy/craniectomy, EVD, thrombectomy, thrombolysis) within [-72, 24] h of t0; encounter-level code timing is flagged `timing_approximate`. Imaging flags remain defined for future sites.
- **Rationale:** These codes are objective, not EEG-derived, and are the only structural evidence available at the EEG sites; approximate timing is reported as a sensitivity stratum.
- **Source:** `docs/research/heedb_schema_dryrun_2026-10-07.md`; `configs/silver_anchors.yaml`; `docs/silver_extraction.md`.

### D-103 H5 (early-EEG subgroup) dropped per Phase 0a fallback
- **Date:** 2026-10-07 · **Area:** hypotheses · **Outcome data seen?** No
- **Decision:** H5 is dropped from Study 1: no imaging report finalization time exists at any EEG site. It moves to the prospective programme.
- **Rationale:** The plan's Phase 0a table specifies "Drop H5" when imaging finalization time is absent.
- **Source:** Phase 0a table; schema dry run.

### D-104 Study 1 cohort operational choices accepted as specified
- **Date:** 2026-10-07 · **Area:** data · **Outcome data seen?** No
- **Decision:** The cohort operationalisations C-01 to C-07, C-09, C-10 and C-12 to C-16 in `docs/cohort_spec.md` section 5 are accepted as written: index rule (first adult, acute-care, non-OR/EMU EEG with a start time; later criteria applied to that EEG only), the visit-based acute-care proxy with the ServiceName fallback off, OR/EMU exclusion, ED-to-admission encounter chaining, the ACI onset proxy (first abnormal GCS/FOUR, else encounter start), recording duration >= 660 s with the unit check, 48 h retention with 6/12/24/48 h flags, the symptom-code broad phenotype, missing-start handling, age rules, text-based score identification, the flow's disclosure rules, the EEG-QC step left to the extractor.
- **Rationale:** The plan names the population and index time but not these operational rules; each was chosen before any HEEDB data were read and has a sensitivity switch in `CohortConfig`.
- **Source:** `docs/cohort_spec.md`; Population and index time; `sortinghat/cohort/`.

### D-105 Strict severity window [-6 h, +1 h] primary; +-6 h as a flagged sensitivity
- **Date:** 2026-10-07 · **Area:** data · **Outcome data seen?** No
- **Decision:** The primary strict cohort uses the nearest GCS (<= 11) or FOUR (<= 12) per instrument within [t0 - 6 h, t0 + 1 h] (pre-t0 on a tie). The plan's +-6 h rule becomes the sensitivity cohort `strict_pm6` (any qualifying score within +-6 h), reported beside the primary in the flow and the cohort table. Amends the Phase 0a "score within +-6 h" inclusion row for the primary analysis.
- **Rationale:** The intended-use population must be knowable at t0; a window reaching 6 h after t0 defines membership with information a deployed system would not have. One hour after t0 allows a bedside score charted around the EEG start. Addresses spec item C-18 for the primary cohort.
- **Source:** Population and index time; `docs/cohort_spec.md` C-08, C-18; Phase 0a, GCS row.

### D-106 Patient merge history applied when available
- **Date:** 2026-10-07 · **Area:** data · **Outcome data seen?** No
- **Decision:** If a patient merge-history table exists (`PatientMergeHistory/` at the access-point root), retired ids are mapped to the surviving id (chains followed) before first-EEG selection, and OMOP rows of retired ids are re-keyed. If it is absent or its old/new id columns are not recognised, patient identity stays BDSPPatientID as given and a one-line aggregate notice is printed and written to the flow report. Its column names are unread and assumed.
- **Rationale:** Merged records would otherwise be counted as two patients and "first EEG per patient" would pick the wrong EEG, and a patient-level split could leak across merged ids.
- **Source:** `docs/research/heedb_schema_dryrun_2026-10-07.md` (access-point prefixes); `docs/cohort_spec.md` C-11.

### D-107 Baseline D (referral question) dropped
- **Date:** 2026-10-07 · **Area:** baselines · **Outcome data seen?** No
- **Decision:** Baseline D and its sensitivity analysis are removed from Study 1. Supersedes D-015.
- **Rationale:** No EEG referral-indication field exists in any real HEEDB header at any site (`ReferralIndication` is ASSUMED only), so Baseline D has no input and the plan's "indication field only" comparator cannot be built.
- **Source:** `docs/research/heedb_schema_dryrun_2026-10-07.md`; `docs/heedb_schema_real.md` section B1; `docs/cohort_spec.md` C-17.

### D-108 t0 is the EEG signal onset, not the file start
- **Date:** 2026-10-07 · **Area:** eeg · **Outcome data seen?** No (signal properties only)
- **Decision:** For every recording t0 = EEG signal onset = the start of the first 10-s block (grid from the file start) in which at least 8 of the 10 required electrodes (Fp1 Fp2 F7 F8 T3 T4 T5 T6 O1 O2) are non-constant, searched within the first 120 min of the file. Recordings with no such block are excluded with reason `no_signal_onset`. The primary window is onset + 1 min to onset + 11 min and the nested windows start at its start. The cohort / feature join uses t0 = metadata start + onset offset (`stream.find_signal_onset` / `onset_offset_s`; stored as `onset_offset_s` in the local_only feature parts).
- **Rationale:** A real-data diagnostic (S0001, n = 60) found 24 recordings with all 10 required channels digitally constant over the whole of minutes 1-11 and a median usable fraction of 0: files contain setup / gap padding before the real signal, so the file start is not the EEG start. Placing the window by signal onset, a property of the signal alone, restores minutes 1-11 of actual EEG without looking at any label or outcome.
- **Source:** `scripts/diag_eeg_signals.py` aggregate output (coordinator report); `docs/eeg_pipeline.md`; Study 1 window definition (minutes 1-11).

### D-109 t0 is the start of the first sustained live segment (supersedes the onset definition of D-108)
- **Date:** 2026-10-07 · **Area:** eeg · **Outcome data seen?** No (signal properties only)
- **Decision:** t0 = the start of the first 60-s period, scanned on a 10-s grid from the file start, in which at least 8 of the 10 required electrodes (Fp1 Fp2 F7 F8 T3 T4 T5 T6 O1 O2) are non-constant (digital samples) in at least 90% of their 2-s epochs (27 of 30), searched within the first 120 min of the file. The primary window is t0 + 1 min to t0 + 11 min and the nested windows start at its start. Recordings with no such segment are excluded with reason `no_sustained_signal` (replaces `no_signal_onset`). The offset is still `onset_offset_s` and the cohort / feature join uses t0 = metadata start + that offset.
- **Rationale:** A real-data diagnostic after D-108 (S0001, n = 60) showed the reader is exact (pyedflib agrees, max |difference| 0 in 57 of 57) and the constant epochs are genuine: about 74% held a non-zero value and 26% were zero, almost none sat at the rail, and 20 recordings' single-block onsets were not sustained (a short live blip, then a hold). A sustained-segment rule tied to the signal alone puts minutes 1-11 inside real recording, without looking at any label or outcome.
- **Source:** `scripts/diag_eeg_signals.py` aggregate output (coordinator report); `docs/eeg_pipeline.md`; D-108.

### D-110 EEG QC thresholds calibrated from signal properties: line ratio 10, extreme 1000 uV, 8 of 10 clean channels
- **Date:** 2026-10-08 · **Area:** eeg · **Outcome data seen?** No (signal properties only)
- **Decision:** Per D-097 the provisional EEG QC thresholds are changed: (a) line-noise ratio 1.0 to 10.0; (b) line-noise flags no longer feed the persistent-disconnection rule (only flat and clipping do); (c) `epoch_channel_frac` 0.9 to 0.8, so an epoch is usable when at least 8 of the 10 minimum-set channels are clean, and the features for that epoch use only the clean channels (a connectivity pair involving an unclean channel is NaN for it; region, global and pair-mean summaries are NaN-aware); (d) extreme amplitude 500 to 1000 uV. The flat (0.5 uV), clipping (5%) and 60% usable-window thresholds are unchanged. The window-level channel rule is aligned with (c): a window needs at least 8 of the 10 minimum-set electrodes present and not exactly constant over the window (a dead channel counts as missing), so with 1-2 missing or dead channels it can still pass on the usable-fraction rule (3 or more fail as `too_few_minimum_channels`); this relaxes the all-10-present requirement of D-096. The number of missing or dead minimum-set channels is recorded per window as a QC field and reported only in aggregate.
- **Rationale:** After D-109, real-data aggregates (S0001, n = 56) gave a median usable fraction of 0.888 but only 32 of 56 passing; among the 24 failing windows the `disconnected` rule flagged 75% of minimum-set cells, driven by line noise (45%) and flat (27%), while amplitudes were plausible (std median 82 uV, p99 294 uV) and the line-noise ratio median was 0.68 (q75 11.5). A ratio of 1.0 and the line-noise route to `disconnected` mislabel mains-contaminated but connected channels, which the 60 Hz notch handles; the amplitude and epoch-fraction limits were set from the same aggregates. No label, outcome or model result was used.
- **Source:** `scripts/diag_eeg_signals.py` aggregate output (coordinator report); D-097; `docs/eeg_pipeline.md`; `sortinghat/eeg/window.py`.

### D-111 Acute-care proxy: inpatient-length covering visit or acute ServiceName
- **Date:** 2026-10-08 · **Area:** data · **Outcome data seen?** No
- **Decision:** An EEG is acute-care when the visit covering its date is inpatient-length (`visit_end_date` after `visit_start_date`) OR `ServiceName` indicates ICU, ED, inpatient or LTM (in hospital) where ServiceName exists; EMU and OR services remain excluded. The `visit_concept_id` classification is dropped (kept only as a no-op in case non-zero ids appear). Supersedes the visit-class part of C-02/C-03 in `docs/cohort_spec.md` and the "ServiceName fallback off" choice accepted in D-104.
- **Rationale:** The first real diagnostic (aggregates only) showed `visit_concept_id` = 0 for all 28.8M visit rows and a mostly null `visit_source_value`, so the care setting cannot be read and the class-based step removed every patient. Visit length and ServiceName are the remaining acute-care signals.
- **Source:** `scripts/diag_cohort.py` aggregates; `docs/cohort_spec.md` C-02, C-03.

### D-112 Visit cover [visit_start_date - 24 h, visit_end_date + 24 h]
- **Date:** 2026-10-08 · **Area:** data · **Outcome data seen?** No
- **Decision:** Visits are treated as date-only. A visit covers an EEG when the EEG's date lies in [visit start date - 24 h, visit end date + 24 h]; among covering visits an exact cover is preferred to a slack-only cover, then an inpatient-length visit, then one that started on or before the EEG date, then the latest start. Amends the interval rule of D-104/C-19.
- **Rationale:** 91-100% of visit starts and ends are at midnight, and widening by 24 h raised the covered share to 78-99% by site, against a near-zero match with timestamp intervals.
- **Source:** `scripts/diag_cohort.py` aggregates; `docs/cohort_spec.md` C-19.

### D-113 Study 1 sites are I0002, I0003, S0001 and S0002
- **Date:** 2026-10-08 · **Area:** splits · **Outcome data seen?** No
- **Decision:** I0008 and I0009 are excluded from all labelled analyses because their patients have no OMOP (EHR) rows. The cohort flow reports them as a separate excluded block. The site-count check of D-024 (at least 3 adult sites with at least 300 candidates each) is applied to the four Study 1 sites.
- **Rationale:** Without EHR data there are no severity scores, visits, medications or labels to build baselines or reference standards at those sites.
- **Source:** `scripts/diag_cohort.py` aggregates; `docs/cohort_spec.md` C-20.

### D-114 Patient merge history applied: MergedBDSPPatientID to BDSPPatientID
- **Date:** 2026-10-08 · **Area:** data · **Outcome data seen?** No
- **Decision:** The merge-history files (columns `MergedBDSPPatientID`, `BDSPPatientID`, `LineNBR`, `BDSPLastModifiedDTS`) are applied before first-EEG selection: each merged id maps to `BDSPPatientID`, chains are followed, and when one merged id has several survivors the row with the latest `BDSPLastModifiedDTS` wins. Specifies the column names left open in D-106.
- **Rationale:** The diagnostic read the column names of the real files; the rule for conflicts avoids an arbitrary choice.
- **Source:** `scripts/diag_cohort.py` (column names only); `docs/cohort_spec.md` C-11.

### D-115 Recording duration is the clock duration; the extractor confirms from the EDF header
- **Date:** 2026-10-08 · **Area:** data · **Outcome data seen?** No
- **Decision:** For the cohort, recording duration is EndTime minus StartTime; the metadata duration (`DurationInSeconds`, `RecordingDuration`) is not used. The streaming extractor confirms the length from the EDF header (number of records times record duration) and excludes recordings or windows that do not fit minutes 1-11. Supersedes the metadata-first duration rule of C-07.
- **Rationale:** The metadata duration disagrees with the clock at I0003 (median ratio 2.35 with a heavy upper tail), so its unit or meaning is not reliable.
- **Source:** `scripts/diag_cohort.py` aggregates; `docs/cohort_spec.md` C-07.

### D-116 Sessions whose BDSPPatientID disagrees with the id in BidsFolder are excluded (ambiguous identity)
- **Date:** 2026-10-08 · **Area:** data · **Outcome data seen?** No
- **Decision:** A session whose `BDSPPatientID` differs from the id embedded in `BidsFolder` (`sub-<SITE><id>`) is removed from the cohort flow as its own step ("Patient id in BDSPPatientID disagrees with BidsFolder (ambiguous identity)", counted per site, counts < 11 suppressed), before the row-integrity check. The check itself accepts a start time equal to the metadata `StartTime` or to any `StartTime(EEG)` of the report rows for the same (patient id, SessionID), and its abort message breaks failures down by cause (key missing / patient-id mismatch / start-time mismatch), aggregate counts only.
- **Rationale:** Source identity conflict (219 sessions at I0002 in the first real build): it cannot be known which id links to the EHR, so the session cannot be assigned to a patient without risking a wrong link. A genuinely misaligned row (a folder belonging to another patient) must still abort the build.
- **Source:** `scripts/diag_cohort.py` / real build abort message (aggregate counts only); `sortinghat/cohort/integrity.py`.

### D-117 Date-shift gate: nearest-event alignment and whole-day gap modes replace the median-offset proxy
- **Date:** 2026-10-08 · **Area:** data · **Outcome data seen?** No
- **Decision:** The Phase 0a STOP row "Consistent within-patient date shift" is now computed per candidate (first qualifying EEG, `t0`) as the signed hours from `t0` to the NEAREST row of each ancillary table (visit start and visit end of `omop_visit_occurrence`, `measurement`, `observation`, `note`, `drug_exposure`, `condition_occurrence`; datetime, else its date twin), streamed so memory is O(candidates) (`sortinghat/audit/alignment.py`). **PASS** = at every Study 1 site with clinical data (a site where at least one candidate has an event; D-113), (a) at least 80% of the site's candidates have some ancillary event within +-24 h of EEG start, AND (b) no non-zero whole-day gap mode holds more than 5% of the site's candidates. **FAIL** otherwise (also when no site has clinical data). Whole-day gap = trunc(hours / 24) toward zero, taken on the nearest event over all tables, so day 0 means within +-24 h; a "mode" is a local peak of the day histogram (count at least that of both neighbouring days), so the natural tail on day +-1 beside a dominant day 0 is not a mode while a constant k-day site offset is. Reported per site (counts < 11 suppressed): shares within +-24 h and +-72 h, quantiles (q10 to q90) of the signed and absolute nearest gap overall and per table, the top-5 whole-day gaps with counts, and a shift-detection block for candidates with a covering visit (start - 24 h <= t0 <= end + 24 h): share with EEG date inside [visit start date, visit end date], before the start date, after the end date, and quantiles of (EEG date - visit start date) in days. The 20-case human hand-check sample is unchanged (local_only).
- **Ordering violations (secondary count, not the gate):** (1) an ancillary row (visit start, measurement, observation, note, drug exposure, condition) timed before the patient's birth (`birth_datetime`, else 1 January of `year_of_birth`); (2) a row timed more than 24 h after the death time (`death_datetime`, else the end of `death_date`; no death row is never counted); (3) a candidate whose EEG start is more than 24 h before the start of its earliest visit; (4) within-record end before start (EEG start/end, drug start/end, imaging). Supersedes the "ordering violations <= 1%" part of the old gate.
- **Rationale:** The first real run failed the old proxy (63.6% "misaligned", 20.45% ordering violations of 4.7M pairs) because it asked whether the MEDIAN event is near the EEG; patients have multi-year EHR histories, so the median is far from the EEG whatever the date shift. The valid question is whether some event is near the EEG, and a constant offset shows as a non-zero whole-day gap mode. Other real evidence already pointed to alignment (78 to 99% of EEG starts within a visit date range +-24 h; about 3,300 patients with a GCS within [-6, +1] h). The rule cannot detect a shift applied identically to every table; that remains the human hand-check.
- **Related fix (same change):** per-site cells of the field-audit table printed "<11/n" for a site where nearly everyone passes (the complement rule of `suppress_proportion` hid the pass count when fewer than 11 records FAILED). Such a cell now reads ">=n-10" with a floored lower-bound proportion; per-site pass counts add up to the overall count (exactly, or within the suppression bounds), asserted for every row in `tests/test_field_audit.py`.
- **Source:** `sortinghat/audit/field_audit.py`, `sortinghat/audit/alignment.py`; synthetic tests only.

---

## Compliance follow-up (2026-10-08)

Entries below follow `docs/research/plan_compliance_2026-10-08.md`. Dates are 2026-10-08 unless stated.

### D-118 Streaming aggregate-only real-data jobs may run from agent sessions (exception to Phase 0e rule 2)
- **Date:** 2026-10-08 · **Area:** safety · **Outcome data seen?** No
- **Decision:** The project lead explicitly authorised (2026-10-07/08, in session) running streaming, aggregate-only real-data jobs from the Claude Code agent session with the agent-session guard bypassed for those commands (`scripts/heedb_run.sh`, `scripts/overnight.sh`). This overrides Phase 0e rule 2 for those jobs. Record-level outputs stay in `local_only/` and are never printed; notes are never read by a hosted model.
- **Rationale:** The lead is unavailable to run jobs locally. DUA permission for cloud processing is to be confirmed by the lead.
- **Source:** Project-lead instruction in session (2026-10-07/08); CLAUDE.md hard rules 2 and 4; `scripts/overnight.sh`; `docs/research/plan_compliance_2026-10-08.md` s2 item 1 and s5 item 1.

### D-119 GCS/FOUR/RASS Phase 0a row judged per Study 1 site with clinical data
- **Date:** 2026-10-08 · **Area:** data · **Outcome data seen?** No · **Data-quality aggregates seen?** Yes
- **Decision:** The GCS/FOUR/RASS within ±6 h row (at least 50%) is judged per Study 1 site with clinical data. It passes at S0001 (53.8%) and S0002 (65.4%). I0002 and I0003 lack scores, notes and medication records. The GCS-from-EHR tool is not applied.
- **Rationale:** Pooled over all sites the row fails (42.1%, 21,572 of 51,260; D-034 fallback). The per-site judgement is the one recorded here.
- **Source:** `out/audit/field_audit.md`; `docs/research/plan_compliance_2026-10-08.md` s1 (0a GCS row) and s2 item 2; D-034; D-105.

### D-120 Only S0001 and S0002 contribute labelled patients; two-site validation schemes
- **Date:** 2026-10-08 · **Area:** splits · **Outcome data seen?** No · **Data-quality aggregates seen?** Yes
- **Decision:** Only S0001 and S0002 contribute labelled patients. The plan's fallback for the site-identifier row is applied: leave-one-site-out across 2 sites plus a late-calendar temporal holdout within each site are co-primary validation schemes. The cross-site claim is weakened accordingly. Power is to be recomputed for this design.
- **Rationale:** The strict primary cohort is 3,292 patients (2,002 at S0001, 1,290 at S0002); I0002 and I0003 each have fewer than 11. This is below the three-site minimum of D-024, which was applied to candidate counts, so the two-site design is recorded here.
- **Source:** `out/cohort/flow.md`; `docs/research/plan_compliance_2026-10-08.md` s1 (0a site row) and s5 item 3; D-024; D-025; D-113.

### D-121 Medication administration row denominator = candidates with a sedation-class exposure
- **Date:** 2026-10-08 · **Area:** data · **Outcome data seen?** No · **Data-quality aggregates seen?** Yes
- **Decision:** The medication administration row denominator is candidates with a sedation-class `drug_exposure` in the 48 h before t0, as implemented in `field_audit.py`.
- **Rationale:** The plan states the row over candidates. Baseline A uses t0 sedative exposure (D-014), so administration timing matters for the candidates with that exposure. D-034 logs only the fallback, not this denominator.
- **Source:** `sortinghat/audit/field_audit.py`; `docs/research/plan_compliance_2026-10-08.md` s1 (0a medication row) and s2 item 4; D-014; D-034.

### D-122 Baseline C has no imaging inputs; Baseline A has no NESI input
- **Date:** 2026-10-08 · **Area:** baselines · **Outcome data seen?** No · **Data-quality aggregates seen?** Yes
- **Decision:** Baseline C has no imaging inputs (no imaging at Study 1 sites). Baseline A has no NESI input (no NESI source found). Both are reported as limitations.
- **Rationale:** Imaging exists only at I0001 and I0004, which have no EEG (D-102) and are not Study 1 sites (D-113). `docs/data_access.md` marks NESI "not found".
- **Source:** `docs/research/heedb_schema_dryrun_2026-10-07.md`; `docs/data_access.md`; `docs/baselines_spec.md` s2.3; D-014; D-102; D-103; D-113; `docs/research/plan_compliance_2026-10-08.md` s2 item 5.

### D-123 Silver labels are structured-data only until an open-weight note model is available
- **Date:** 2026-10-08 · **Area:** labels · **Outcome data seen?** No
- **Decision:** Silver labels are structured-data only until an open-weight model on approved compute is available for notes.
- **Rationale:** The plan's silver sources include notes (ICD codes, clinical attribution and notes). A hosted model may not read notes (D-083), and no open-weight model on approved compute is in place, so note-derived silver evidence is deferred. This narrows the silver scope.
- **Source:** CLAUDE.md hard rule 4; D-083; `docs/silver_extraction.md`; `docs/research/plan_compliance_2026-10-08.md` s2 item 6.

### D-124 Feature windows use t0 = sustained EEG signal onset; cohort, baselines and silver use metadata start
- **Date:** 2026-10-08 · **Area:** data · **Outcome data seen?** No · **Data-quality aggregates seen?** Yes
- **Decision:** Feature windows use t0 = sustained EEG signal onset (D-109). Cohort, baselines and silver windows are anchored at the metadata EEG start, which is never later than the signal onset, so baselines are conservative with respect to leakage.
- **Rationale:** Baseline inputs all precede the signal onset, so they cannot contain information from the EEG-derived window. The join that applies the onset offset to the cohort is not yet implemented.
- **Source:** D-108; D-109; `docs/eeg_pipeline.md`; `docs/research/plan_compliance_2026-10-08.md` s2 item 7.

### D-125 Feature and structured-label extraction before Gate 0 is data preparation, not training
- **Date:** 2026-10-08 · **Area:** data · **Outcome data seen?** No · **Data-quality aggregates seen?** Yes
- **Decision:** Feature and structured-label extraction before Gate 0 is data preparation, not training. No model is fitted, and no label-by-feature summaries are computed, until Gate 0 passes.
- **Rationale:** D-039 bars model training before Phase 0 passes and does not name extraction. The plan's sequence ("no modeling yet"; nothing trained until the field audit and pilot pass) puts extraction after Gate 0; the order used is recorded here as a departure from that sequence.
- **Source:** D-039; D-051; `docs/research_plan_v1.txt` (Phase 0 introduction); `docs/research/plan_compliance_2026-10-08.md` s3 item 1.

### D-126 Evaluation size of about 1,000 cases is provisional
- **Date:** 2026-10-08 · **Area:** labor · **Outcome data seen?** No
- **Decision:** The ~1,000-case evaluation size (D-093) is provisional and will be recalculated with formal methods after the pilot, as the plan specifies.
- **Rationale:** D-027 requires the size to follow measured pilot minutes and kappa. D-093 fixed the figure before the pilot, and its simulations assume three sites, whereas D-120 uses two.
- **Source:** D-027; D-093; D-120; `sortinghat/metrics/power.py`; `docs/research/plan_compliance_2026-10-08.md` s1 (evaluation size row).

### D-127 Transparency note: D-117 replaced the date-shift proxy after the original proxy returned FAIL
- **Date:** 2026-10-08 · **Area:** data · **Outcome data seen?** No · **Data-quality aggregates seen?** Yes
- **Decision:** Transparency note: D-117 replaced the date-shift alignment proxy after the original proxy had returned FAIL (63.6% "misaligned", 20.45% ordering violations). The median-based proxy is invalid for multi-year EHR histories. The original result will be reported alongside the new one. No outcome data had been seen.
- **Rationale:** The median event is far from the EEG for patients with multi-year histories whatever the date shift, so the original proxy could not test alignment. The replacement is D-117's nearest-event gate.
- **Source:** D-117; `docs/research/plan_compliance_2026-10-08.md` s2 item 10 and s3 item 6.

### D-128 SAP s16.1: "possible" scoring
- **Date:** 2026-10-08 · **Area:** labels · **Outcome data seen?** No
- **Decision:** Primary scoring sets y = 0 for "possible" (binary positive = probable or definite). Sensitivities mask "possible" and score it as 1 (SAP s12).
- **Rationale:** The plan keeps the handoff's ordinal states but does not state how "possible" is scored. The SAP follows `docs/labels_spec.md`.
- **Source:** SAP s3.2, s12 and s16 item 1; `docs/labels_spec.md`.

### D-129 SAP s16.2: probability clipping eps = 1e-4
- **Date:** 2026-10-08 · **Area:** hypotheses · **Outcome data seen?** No
- **Decision:** Predicted probabilities are clipped to [eps, 1 - eps] with eps = 1e-4 (`DEFAULT_EPS`). Sensitivities use eps = 1e-6 and 1e-3.
- **Rationale:** Without clipping, a single confident miss can dominate the mean log loss.
- **Source:** SAP s5, s12 and s16 item 2.

### D-130 SAP s16.3: primary EEG model is the top ladder rung
- **Date:** 2026-10-08 · **Area:** hypotheses · **Outcome data seen?** No
- **Decision:** The primary EEG model is the top ladder rung ("combined") on the 10-minute window. Its head family and hyperparameter budget are fixed before evaluation-set unblinding. Other rungs are descriptive ladder results, not hypothesis tests.
- **Rationale:** The plan does not name the rung that carries H1 and H2. Naming one avoids best-of-ladder selection.
- **Source:** SAP s4.2 and s16 item 3; D-017.

### D-131 SAP s16.4: within-site stratified bootstrap carries the H1/H2 interval
- **Date:** 2026-10-08 · **Area:** hypotheses · **Outcome data seen?** No
- **Decision:** The within-site stratified bootstrap carries the H1/H2 CI (99%, D-095). Cluster, two-stage and site-level t intervals are co-reported.
- **Rationale:** SAP s7.2 gives the reason as S = 3 sites, for which a cluster interval cannot support a coverage claim. The two-site design of D-120 revisits that premise, and the choice is to be re-checked there.
- **Source:** SAP s7.2 and s16 item 4; D-095; D-120.

### D-132 SAP s16.5: H3 stratum rule and severity cut points
- **Date:** 2026-10-08 · **Area:** hypotheses · **Outcome data seen?** No
- **Decision:** "Delta below 0 within a stratum" means the point estimate is below 0. The minimum stratum size is n = 50 (placeholder). Severity cut points are fixed before unblinding.
- **Rationale:** The plan does not define a stratum-level rule or a minimum size. A stratum under 50 patients is treated as not estimable and not favourable, and the n = 50 minimum is a placeholder for the pilot to replace.
- **Source:** SAP s6 (H3 row) and s16 item 5; D-019.

### D-133 SAP s16.6: H4 "near chance" and rank statistic
- **Date:** 2026-10-08 · **Area:** hypotheses · **Outcome data seen?** No
- **Decision:** "Near chance" for E6 vs E5 means the pairwise AUROC CI contains 0.5. Kendall tau-b is reported between the predicted and observed label rankings, with a bootstrap CI.
- **Rationale:** The plan says "near chance" without a criterion. H4 is recorded before unblinding and reported whichever way it falls.
- **Source:** SAP s6 (H4 row) and s16 item 6; D-020.

### D-134 SAP s16.7: H5 "supported" rule (H5 currently dropped)
- **Date:** 2026-10-08 · **Area:** hypotheses · **Outcome data seen?** No
- **Decision:** H5 is "supported" if the interaction estimate is below 0 and its CI lies entirely below 0. The rule applies only if H5 is reinstated.
- **Rationale:** The plan gives an interaction estimate with a 95% CI but no decision rule. H5 is dropped from Study 1 under D-103, and the SAP has not yet been updated to say so.
- **Source:** SAP s6 (H5 row) and s16 item 7; D-021; D-103.

### D-135 SAP s16.8: H6 "met" rule and interpretability threshold
- **Date:** 2026-10-08 · **Area:** hypotheses · **Outcome data seen?** No
- **Decision:** H6 is "met" if the ratio Delta(2 min)/Delta(10 min) has a point estimate of at least 0.70 and Delta(10 min) < 0. The ratio is reported as uninterpretable when the CI for Delta(10 min) includes 0 or fewer than 90% of bootstrap draws have Delta(10 min) < 0 (90% is a placeholder).
- **Rationale:** The 70% threshold is the plan's. The sign condition and the interpretability threshold are operational, and the 90% level is a placeholder.
- **Source:** SAP s6 (H6 row) and s16 item 8; D-022.

### D-136 SAP s16.9: "improves individually" for G2
- **Date:** 2026-10-08 · **Area:** hypotheses · **Outcome data seen?** No
- **Decision:** A label "improves individually" for G2 when its per-label Delta CI upper bound is below 0 and its Holm-adjusted one-sided p-value is below 0.05.
- **Rationale:** The plan says "improve individually" without a test. Holm adjustment is applied across the primary labels.
- **Source:** SAP s7.4, s8 and s16 item 9; D-052.

### D-137 SAP s16.10: G2 sedative adjustment
- **Date:** 2026-10-08 · **Area:** hypotheses · **Outcome data seen?** No
- **Decision:** The G2 "sedative adjustment" condition is met when the sedative-excluded subset has a pooled Delta point estimate below 0.
- **Rationale:** Baseline A already contains sedation (D-014), so the subset check tests whether Delta is carried by sedative signatures.
- **Source:** SAP s11 and s16 item 10; D-052; D-014.

### D-138 SAP s16.11: temporal holdout fraction 0.20
- **Date:** 2026-10-08 · **Area:** splits · **Outcome data seen?** No
- **Decision:** The late-calendar temporal holdout is the latest 20% of cases in each site by calendar time. Training uses earlier cases only, optionally with an embargo. The fraction is fixed at 0.20.
- **Rationale:** The plan does not give a fraction. If the date shift does not preserve calendar order across patients within a site, the holdout is dropped and recorded, not replaced by another construction.
- **Source:** SAP s7.1 and s16 item 11; D-025; D-117.

### D-139 SAP s16.12: E7 eligibility reading
- **Date:** 2026-10-08 · **Area:** labels · **Outcome data seen?** No
- **Decision:** E7 is a primary label if it has at least 100 total gold positives and at least two sites with at least one positive (`metrics.labels.e7_eligible`). Eligibility is decided on evaluation-set gold counts before any model output is unblinded.
- **Rationale:** D-004 sets the 100-positive, two-site threshold. "At least one positive at each of two sites" is the operational reading of "across at least 2 sites".
- **Source:** SAP s3.1 and s16 item 12; D-004.

### D-140 SAP s16.13: gating of H3 and H5 interpretation
- **Date:** 2026-10-08 · **Area:** hypotheses · **Outcome data seen?** No
- **Decision:** H3 is interpreted as confirmatory only if H1 is met. H5 is interpreted as confirmatory only if H2 is met. Otherwise both are reported descriptively. No further alpha adjustment is applied.
- **Rationale:** The gating stops secondary tests from being read as confirmatory when the primary fails. H3 and H5 use point-estimate or single-CI rules, so no multiplicity adjustment is added.
- **Source:** SAP s8 and s16 item 13; D-018.

### D-141 SAP s16.14: bootstrap replicates B = 10,000 for final analyses
- **Date:** 2026-10-08 · **Area:** hypotheses · **Outcome data seen?** No
- **Decision:** Final analyses use B = 10,000 percentile-bootstrap replicates with a fixed seed recorded in the analysis log. The code default of 2,000 is for development only. The sub-items 14a to 14d are covered by D-093 to D-097.
- **Rationale:** The plan does not state a replicate count. The final-analysis value is set here so that the development default is not reported as final.
- **Source:** SAP s7.2 and s16 item 14; D-093; D-094; D-095; D-096; D-097.

### D-142 SAP s16.15: G0 kappa type and prevalence source
- **Date:** 2026-10-08 · **Area:** data · **Outcome data seen?** No
- **Decision:** G0 kappa is the unweighted binary kappa (probable or definite positive) on the 200-case pilot. Prevalence is taken from pilot gold labels.
- **Rationale:** The plan says "kappa" without a type. The binary positive definition follows `docs/labels_spec.md` section 6.
- **Source:** SAP s11 (G0 row) and s16 item 15; `docs/labels_spec.md` s6; D-051.

### D-143 Pre-Gate-0 silver-label feasibility analysis (exploratory)
- **Date:** 2026-10-08 · **Area:** hypotheses · **Outcome data seen?** No · **Data-quality aggregates seen?** Yes
- **Decision:** At the project lead's direction, before Gate 0 and before any gold labels, run an exploratory silver-label feasibility analysis: models trained AND evaluated on structured, EEG-blind silver labels, with both D-120 validation schemes and the plan's mandatory controls. Gold-label adjudication (Phase 0b pilot, kappa) is deferred, not dropped. EEG reports and any EEG-derived text are never used as label evidence (circularity rule, D-009); they serve only as the circularity-audit comparator. Clinical notes are not read by a hosted model. All outputs are labelled exploratory and do not test the preregistered H1–H6.
- **Rationale:** Obtain early evidence that EEG carries etiologic signal before investing adjudication labour; deviates from "silver labels train models; they never score them" and from "nothing is trained until Gate 0", both deliberately and for this exploratory purpose only.
- **Source:** Project lead instruction (2026-10-08); `docs/silver_feasibility.md`.

### D-144 Silver-anchor rules evidence-reviewed against the literature; ammonia threshold raised, NMDAR antibody rule tightened
- **Date:** 2026-10-08 · **Area:** labels · **Outcome data seen?** No
- **Decision:** Every anchor in `configs/silver_anchors.yaml` (E1, E2, E4a, E5, E6, E7), the EEG-sentence filter and the banned G92/G93.4 codes were checked against published definitions (table in `docs/research/anchor_evidence_review.md`). Changes: (1) E5 ammonia >= 100 -> >= 150 umol/L; (2) E7 autoimmune antibody: NMDAR counts only from CSF (serum-only positives do not), other neural antibodies CSF or serum; the extractor must apply the specimen rule. All other thresholds and windows are unchanged: glucose < 50 and Na < 120 are stricter than the consensus levels (< 54 mg/dL; profound hyponatraemia < 125), PaCO2 > 70 with pH < 7.30, pH < 7.10, Ca > 14, glucose > 600, lactate >= 2.0 and the CDC ASE organ-dysfunction items, CSF WBC >= 20 with RBC/500 correction. YAML status is now "Evidence-reviewed 2026-10-08 (literature-grounded; see docs/research/anchor_evidence_review.md); external co-I review before publication"; `anchors.is_signed_off()` keys on the old "SIGNED OFF" prefix and now returns False (not used elsewhere; to be widened by the code owner).
- **Rationale:** Prefer specificity for silver positives and cite only what was retrieved. Median admission ammonia in grade 3-4 HE was 116 umol/L (PMID 28859230), so 100 was not specific; ALF intracranial-hypertension risk separates near 146-150 (PMID 19465152). NMDAR antibody sensitivity is 100% in CSF vs 85.6% in serum (PMID 24360484) and Graus 2016 (PMID 26906964) requires CSF. I60/I61 PPV is 88.6% in any position vs 98.2% principal (PMID 33469381), so E1 dx codes should prefer principal position upstream. Unsupported items (hypernatraemia > 160, BUN >= 100, PaCO2 > 70, profound shock MAP < 50, ethanol >= 300) are flagged as expert thresholds for the co-I.
- **Source:** `docs/research/anchor_evidence_review.md`; tests updated (`tests/test_labels_anchors.py`, `tests/test_labels_extract.py`).

### D-145 Intended-use analyses: Baseline P (Presentation), current-encounter baselines, undifferentiated-AMS subgroup
- **Date:** 2026-10-08 · **Area:** hypotheses · **Outcome data seen?** No
- **Decision:** The intended use is undifferentiated altered mental status or unexplained unconsciousness in the ED ("given an unknown EEG, what is the cause?") with little or no history. HEEDB EEGs are often recorded hours or days into an admission, so Baselines A to C may encode workup information the intended user lacks. (1) New **Baseline P "Presentation"**: age, sex, the GCS / FOUR (and RASS if charted) nearest to t0 in [-6 h, +1 h] (the cohort's strict-severity window, D-105; earlier chart on a tie), and the FIRST vitals and point-of-care glucose of the current encounter before t0. No diagnosis codes, no past medical history, nothing from a prior encounter. P is not nested in A to C. Its +1 h score window is the one bounded exception to the t0 mask (a second call of the same gate, `score` domain only, `BaselineConfig.presentation_score_after_h`; 0 gives a strictly masked P). (2) **Current-encounter restriction**: by default every baseline (A, B, C, P) uses only events at or after the encounter start (the cohort's `encounter_start`, i.e. the covering visit chained back along acute visits, else the same rule re-run on the visit table, else t0 minus 3 days; the basis is counted in the build log), and `as_of` still limits them to t0. `scripts/build_baselines.py --with-history` builds the sensitivity variant that also allows prior-encounter events; its sidecar and the silver-feasibility report label it "with history". (3) `scripts/run_silver_feasibility.py` reports first a block "Intended use: undifferentiated AMS": (a) EEG-only vs the prevalence prior, (b) EEG + age/sex vs age/sex, (c) P + EEG vs P, as Delta masked log loss with the 99% within-site interval, per-site Delta, both validation schemes (leave-one-site-out and temporal holdout), then the existing A and C comparisons. (4) **Undifferentiated subgroup**: EEG on encounter day 0 to 1 (t0 within 36 h of the encounter start), no ICD code of the primary label families (acute structural dx, arrest, asphyxia) recorded at or before t0 in any encounter, and no sedative or opioid exposure in the 6 h before t0 (Baseline A sedation features). A component that cannot be established counts as not met. All comparisons are repeated in the subgroup (trained on all training-fold rows, scored on subgroup rows) and its size is reported suppressed (< 11); it is not estimable below 50 scored rows or 11 per site. EEG-derived information is never an input or label evidence. (5) `docs/silver_feasibility.md` and the SAP (section 4.4, "Intended-use analyses", with sensitivity row 17 and operationalization 16) are updated.
- **Rationale:** A clinician reading an unknown EEG in the ED has the presentation, not the in-hospital workup; a baseline built on labs, cultures and sedation from days of care answers a different question than the intended use, and a prior encounter's data is not available to that user. P is the comparator for the intended use; the current-encounter default removes a source of optimistic baselines; the subgroup is where the intended-use claim lives. Every change is exploratory (pre-Gate-0, silver labels, D-143) and moves no gate. Known limits: encounter starts are date-granular while visits are date-only (D-112); ICD codes are timed at their start date and an unknown-time code counts as already recorded; a sedative bolus with no recorded quantity that ended before t0 is invisible to the Baseline A sedation features; the subgroup is a restriction on the same silver labels, so it inherits their ascertainment bias.
- **Source:** Project lead instruction (2026-10-08); `sortinghat/baselines/` (`encounter.py`, `asof.as_of_presentation`, `features.py`), `scripts/build_baselines.py`, `scripts/run_silver_feasibility.py`; tests `tests/test_intended_use_baselines.py`, `tests/test_intended_use_analysis.py`; D-105, D-107, D-112, D-143.

### D-146 Baseline P is strictly t0-masked
- **Date:** 2026-10-08 · **Area:** baselines · **Outcome data seen?** No
- **Decision:** Baseline P (presentation) uses severity scores charted at or before t0 only (`presentation_score_after_h = 0`); the [-6, +1] h window remains a cohort-inclusion rule (D-105), never a model input.
- **Rationale:** A model input must be knowable by the ED clinician at t0; post-t0 scores are defensible for selection but not as features.
- **Source:** `sortinghat/baselines/config.py`.

### D-147 Tech-transfer contact deferred until a positive feasibility signal; research outreach decoupled
- **Date:** 2026-10-08 · **Area:** commercial · **Outcome data seen?** No
- **Decision:** The incoming institution's tech-transfer question (outreach draft #4) is deferred until the silver-label feasibility analysis (D-143) shows an EEG signal worth protecting. Academic outreach (BDSP collaboration, CERTA, Korean cohort; drafts #1–3) is no longer gated on it. The project lead will read the IP clauses of the current and incoming house-staff agreements in the meantime, since ownership is set by contract terms, not disclosure timing.
- **Rationale:** No invention to disclose yet; collaboration and scoop-risk mitigation (CLEF, BDSP) are time-sensitive.
- **Source:** Project lead instruction (2026-10-08); `docs/outreach_drafts.md`.

### D-148 CBraMod input amplitude policy: keep high-amplitude segments
- **Date:** 2026-10-08 · **Area:** representations · **Outcome data seen?** No
- **Decision:** CBraMod segments exceeding its nominal ~100 µV input range are kept (scaled per the model's normalisation), not dropped; `--amp-policy drop` is a sensitivity analysis.
- **Rationale:** ICU and encephalopathic EEG routinely exceeds 100 µV (high-voltage delta, triphasic waves, burst-suppression bursts); dropping it would bias against the most impaired patients and the etiologies of interest.
- **Source:** `docs/embeddings.md`.

### D-149 Column-pruned, candidate-filtered OMOP extracts are cached on the container disk (EHR tables only)
- **Date:** 2026-10-08 · **Area:** data · **Outcome data seen?** No
- **Decision:** Row groups of the OMOP EHR tables that the cohort build, Phase 0a audit, structured silver labels and baselines read are cached once on the container disk, shared by all those steps, under `out/local_only/omop_cache/` (directories mode 0700, files 0600, gitignored, never committed, never printed). Each cached unit is one source row group, column-pruned to the union of the columns any step reads and filtered to the candidate people (every adult at a Study 1 site, with merged ids, about 51k; the candidate id list is its own small file), stored as parquet and keyed by the S3 object key, size and ETag so a changed source object is never served stale. A step needing fewer people or columns filters further in memory; a request outside the candidate set or the column union reads S3 as before. Missing row groups are fetched by a bounded pool (default 4, env `SORTINGHAT_FETCH_WORKERS`) and handed over in order, so peak memory is about four row groups. `python -m sortinghat.omop_cache warm --s3` fills the cache once and resumes after a restart. EEG recordings stay streamed (D-100 unchanged for EEG); the note table is cached with timestamp columns only, never text. This partially supersedes D-100's "no raw table is written to disk" for the EHR tables only.
- **Rationale:** The network is the bottleneck: four steps re-streamed the same tables from S3 and the container restarts about hourly, so every restart began again from zero. Caching the filtered extract once removes the repeated transfer and survives restarts. The extract holds only rows of candidate people and only the columns the analysis uses, stays on the same restricted container disk and local_only convention as the cohort table and the baseline matrix, and outputs remain aggregate-only; results are identical to uncached reads (tested).
- **Source:** `sortinghat/omop_cache.py`, `sortinghat/checkpoint.py`, `tests/test_omop_cache.py`; CLAUDE.md rules 3, 5, 6; D-100, D-118.

### D-150 Frozen-rung extraction restricted to the strict cohort first
- **Date:** 2026-10-09 · **Area:** representations · **Outcome data seen?** No
- **Decision:** CBraMod + MORGOTH extraction (stage 3c) runs first on recordings in the strict cohort (primary or ±6 h sensitivity, 3,851 recordings), the population of the primary analyses; the broad cohort follows if time allows.
- **Rationale:** CPU-only inference runs at ~330 recordings/h; the full 14.5k list would take ~44 h. Throughput only, no outcome information involved.
- **Source:** `scripts/stage3c.sh`.

### D-151 Feasibility analysis released on pre-review silver labels; rerun after label fixes
- **Date:** 2026-10-09 · **Area:** labels · **Outcome data seen?** No
- **Decision:** The exploratory silver feasibility analysis (D-143) runs now on the silver labels produced at 03:58 UTC, before the silver-yield review (E2 shock, E5/E6 component mapping, report suppression) completes; it is rerun on the corrected labels and both versions are reported, the corrected one as primary.
- **Rationale:** Project lead asked to speed up all parts; the review fixes only mapping errors, so an early run is informative and the rerun is cheap (checkpoints keyed by the label file).
- **Source:** Project lead instruction (2026-10-09); `out/logs/silver_final.ok`.

### D-152 Silver-yield review: profound-shock operationalisation and mapping fixes (amends D-088)
- **Date:** 2026-10-09 · **Area:** labels · **Outcome data seen?** No (anchor-firing counts only; no label-by-EEG summaries) · **Data-quality aggregates seen?** Yes
- **Decision:** (1) `profound_shock` = MAP < 50 mmHg on consecutive readings with no normal reading between, successive low readings ≤ 60 min apart (was 15; matches charting cadence), first-to-last spanning ≥ 30 min, window [-48, 0] h; threshold and duration unchanged. MAP derived from SBP/DBP including combined "SBP/DBP" text rows. (2) Mapping fixes: combined BP text split; unit aliases "millimeter of mercury" and "@"; blank-unit BP rows kept. (3) Silver report: per-site prevalence suppressed whenever its count is suppressed, including complementary suppression. E5 and E6 components verified correct and left unchanged (strict by design; E6 capped at ~3.4% by blood-culture availability).
- **Rationale:** No MAP rows matched before the fix; the 15-min gap was incompatible with hourly charting. Expected shock yield remains <11, so E2 rests on arrest/asphyxia anchors.
- **Source:** `docs/research/silver_yield_review.md`.
