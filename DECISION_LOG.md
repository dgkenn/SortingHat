# Sorting Hat Decision Log

**Purpose.** This log records every protocol change in the Research Plan (`docs/research_plan_v1.txt`) relative to the master handoff v1.0 (29 Sep 2026). It exists for preregistration integrity: each decision is dated and written down before any HEEDB data, Phase 0 audit result, pilot label, or model output is viewed.

**Rules.**
- Append-only. Do not edit or delete an entry. If a decision changes, add a new entry that cites the superseded ID (e.g., "Supersedes D-014").
- Every entry records whether outcome data had been seen. All entries below are "No".
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
