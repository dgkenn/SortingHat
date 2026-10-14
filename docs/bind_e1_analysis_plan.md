# Imaging-anchored E1 (structural injury) test inside HEEDB, using BIND: pre-specified analysis plan

Status: PRE-SPECIFIED, written 2026-10-14 before any BIND label has been computed, counted or linked to an EEG case. DECISION_LOG D-158.
Parent documents: `docs/prereg_study1_sap.md` (Amendment A1; this plan is a secondary, exploratory, pre-Gate-0 analysis that adds no hypothesis to the SAP and enters no gate), `docs/silver_feasibility.md`, `docs/research/open_etiology_datasets_2026-10-14.md` (rank 2).
Code: `sortinghat/labels/bind.py`, `configs/bind_e1.yaml`, `tests/test_labels_bind.py` (synthetic fixtures only). Access and terms: `docs/bind_access_request.md`.

**Outcome data seen?** Yes, aggregate silver feasibility (see SAP A1.1). **No BIND data has been seen.** Only table and column NAMES of BIND were read (names-only probe, 2026-10-14). No value, count, date or ID of any BIND table has been opened.

## 1. Question and why it is not circular

Does the frozen EEG representation carry information about **acute structural brain injury** (SAP label E1) beyond the t0 clinical baselines, when "injury" is defined by **radiology**, not by our ICD-based silver anchors and not by any EEG-derived field?

The silver feasibility result for E1 (SAP A1.1; the gain was mostly E1 in IU-large) used labels built from ICD-10 diagnosis codes and procedure codes in [-72, +24] h of t0 (`configs/silver_anchors.yaml`, E1 structured alternative). Those labels share code-assignment behaviour with Baseline C features, and could share clinician behaviour with the EEG (an EEG read can precede a code). Imaging-E1 changes the reference standard to the radiologist's report of a CT or MRI that was read without the EEG. Circularity checks, all fixed now:

1. **Label independent of the EEG.** The label table is built only from BIND `Clinical/` findings and session dates. No EEG report, EEG feature or EEG-derived column is an input; `Report_txt` is never loaded (enforced in the loader and tested). The imaging reports are written by radiologists; their free text is not read by anyone in this plan except the optional human validation sample (step 0d, `local_only/`).
2. **Label independent of our anchors.** The class definitions mirror the silver imaging items (ICH, SAH, SDH/EDH, infarct, TBI/contusion, mass effect) only as *concepts*; the evidence is the radiology finding row, never a diagnosis code, procedure code, drug or lab. BIND is not an input to Baselines A, C or P (they are built from OMOP and HEEDB tables; a column-name scan asserts no BIND-derived column is a model input).
3. **Not independent of clinical reality.** A diagnosis code and a CT finding both measure the same lesion. That correlation is wanted; it is not circularity. The remaining risk is **the reverse direction**: an EEG finding may have prompted the imaging (verification bias). It is addressed by sensitivity analysis 4 (section 7) and by stating the estimand as conditional on imaging in the window.
4. **Circularity audit, imaging version.** Reuse `sortinghat/labels/circularity_audit.py`: compare the agreement of the EEG-model's predictions with imaging-E1 against their agreement with the EEG-report-impression comparator. If the model agrees more with EEG-report impressions than with imaging-E1, the EEG gain is read as partly reader-pattern learning and the imaging result is described that way.

## 2. Verified facts about BIND (primary sources, 2026-10-14)

| Item | Fact | Source |
|---|---|---|
| Name, version | Brain Imaging and Neurophysiology Database (BIND), v1.0, published 2025-09-09, DOI 10.60508/mby8-3a26 | https://bdsp.io/content/n1vba1x5qt62frfjem65/1.0/ |
| Paper | Maschke C, ... Westover MB. Sci Data 13:1176 (2026-05-21), doi:10.1038/s41597-026-07421-x, PMID 42168237, PMC13462082. Article licence CC BY-NC-ND 4.0. | PubMed / PMC |
| Scale | 38,942 patients; 1,791,885 scans (1,723,699 MRI; 54,137 CT; 5,093 PET; 526 SPECT); 96.54% of sessions have a full-text report; 84,960 reports judged brain-related | PMC13462082 |
| Sites | I-0001 = Mass General Brigham (28,438 patients; 81,297 encounters; all patients had imaging AND an EEG, routine or long-term); I-0004 = Stanford (10,504 patients; 27,440 encounters; imaging plus **PSG**, not EEG) | PMC13462082 |
| Linkage | "Patients' unique identification numbers are stable over encounters and datasets" so imaging links to EEG/PSG; 74.9% of BIND patients are also in HEEDB. The key name is not stated in the paper. | PMC13462082; bdsp.io page |
| Labels | Clinical metadata extracted from reports by **Bio-Medical-Llama-3-8B** (open-weight): per report brain-related yes/no/unknown and pathology yes/no/unknown; list of findings (type, general term, location, brain-related, magnitude, acute/subacute/chronic); each finding mapped to **10 overarching categories / 150 subcategories** (categories named in the paper: vascular 22.88% of brain findings, acquired and traumatic injuries 15.97%, neoplasms 12.17%, neurodegenerative, inflammatory, white matter, structural abnormalities, cyst-like lesions, other, unassigned); a self-validation pass asks whether each finding is explicitly in the report. 394,122 findings extracted. | PMC13462082 |
| Extraction accuracy | **Not reported.** A board-certified neurologist reviewed 1,000 random reports (1.2% of reports), but the paper gives no accuracy, sensitivity or agreement figure and states "we cannot guarantee the accuracy or completeness"; the metadata is "an entry point", "not clinical ground truth". The only accuracy figures are for MRI sequence naming (0.947 MGB, 0.846 Stanford). | PMC13462082 |
| Access | BDSP "Database Credentialed Access": credentialed user, signed credentialed DUA, CITI "Data or Specimens Only Research", AWS account ID in the BDSP profile. Imaging is one of the large repositories (EEG, ECG, PSG, EHR, Imaging) for which BDSP says the credentialed DUA is sufficient. | bdsp.io page; https://bdsp.io/about/howto_accessdata |
| Licence | Files: BDSP Credentialed Health Data License 1.5.0 (text linked, not read). The page is silent on commercial use. The paper: "Commercial entities may access the dataset through a separate institutional agreement with the participating hospitals". | bdsp.io page; PMC13462082 |
| Code | https://github.com/bdsp-core/BigBrainImagingDatabase: `ask_llama_open.py` (extraction), `ask_llama_category.py` (category selection), `mri_sequence_identification.ipynb`. Prompts are therefore inspectable. | GitHub page |
| Real tables (names-only probe, 2026-10-14) | `Imaging/{I0001,I0004}/Clinical/{site}_imaging_findings_De-id.csv` (21 columns, same at both sites), `{site}_reports.csv` (`bdsp_patient_id, session_id, Report_txt, Type`), `{site}_demographics.csv` (per-site header; has `ShiftedDateOfBirth`, `ShiftedStudyDate`). Findings columns: `BDSPPatientID, Session_id, ShiftedDateOfBirth, ShiftedStudyDate, Counts, Abnormality, Brain, FindingID, PresentPrior, Change, Name, General Name, Location, Brain-related, Magnitude, Acute/Chronic, Test, Label_manual, Nullfinding, Label_LLAMA_Level2, Label_LLAMA_Level1`. | `docs/heedb_schema_real.md` section B4 |

Consequences for the design:

* **The study date is a date, not a timestamp** (`ShiftedStudyDate`; no time-of-day column exists in these tables). The window is therefore in calendar days (section 3).
* **Only I0001 (MGB) is EEG-linked.** I0004 (Stanford, PSG) is outside the EEG cohort. Which HEEDB EEG site codes (S0001/S0002) contain the I0001 patients is not documented; it is measured in step 0a.
* **A manual-label column exists (`Label_manual`).** Its coverage is unknown. If it is populated for a subset (plausibly the 1,000 reviewed reports), step 0d uses it to estimate extraction accuracy.
* **How normal studies are represented is unknown** (a `Nullfinding` row, or no row). The loader treats a session as a brain study if it has a brain-related findings row or, optionally, a matching report `Type`. If normal studies leave no row, `Type` or the demographics session list must supply them (step 0c). Until then negatives are those with a `Nullfinding` or non-qualifying brain row.

## 3. Label definition (fixed)

Unit: one EEG case (cohort row; t0 = metadata EEG start, D-124). Notation: `offset = date(imaging study) - date(t0)` in calendar days, in the shifted time base.

* **Window (primary): offset in [-1, +3] days** (about -24 h to +72 h; the date-only resolution makes the clock span up to 48 h before and 96 h after). Fixed in `configs/bind_e1.yaml` (`window_days`).
* **Brain study:** a study with at least one brain-related findings row (or a brain report `Type`, if used).
* **Lesion classes, primary imaging-E1 = any of:** acute or subacute **infarct** (including restricted diffusion), **intraparenchymal haemorrhage** (not microbleeds, cavernoma, hemosiderin), **subarachnoid haemorrhage**, **subdural/epidural/extra-axial haemorrhage** (not chronic, hygroma), **traumatic contusion/diffuse axonal injury/TBI**, **mass effect** (midline shift, herniation, vasogenic oedema, tumour with oedema). These are the six silver imaging items.
* **Acuity:** the finding's `Acute/Chronic` qualifier must be acute, subacute, hyperacute or new. A chronic/remote/old/stable qualifier wins over an acute one. A class mention whose acuity is blank or unknown is **indeterminate**.
* **Secondary class (reported, not in the primary label):** acute **hydrocephalus** (also the broad variant, sensitivity 3).
* **Positive:** at least one brain study in the window with a qualifying finding row (class concept, acute, brain-related, not a `Nullfinding`, not negated).
* **Assessable negative:** at least one brain study in the window, no positive, and no indeterminate primary-class mention.
* **Not assessable (masked, never negative):** no brain study in the window (reason `no_brain_study_in_window`); or no positive but an indeterminate-acuity mention (`indeterminate_acuity`).
* **Negative-control outcome:** a **chronic** lesion of a primary class on a brain study in the window, with no acute positive (`ctrl_chronic`). The EEG model should not gain here beyond what it gains on any structural abnormality; a similar gain would point to age and chronic burden, not acute injury.
* **What is fixed and what may still move.** The clinical definitions above, the window, the classes and the masking rules are fixed now. The regular expressions that implement them in `configs/bind_e1.yaml` are ASSUMED because no value vocabulary has been seen. Step 0c lists the distinct values of the categorical columns (value strings only, no counts, no linkage to EEG); the expressions may be adapted to the observed strings only to implement these same definitions. The final YAML is hashed and logged in DECISION_LOG before the first label is linked to an EEG case (step 1).

## 4. Population and strata

* **Primary stratum: IU-large** (SAP A1.3; D-155; broad cohort at the labelled EEG sites). **Secondary: the whole broad cohort** (strict reported separately, never pooled). Estimand: conditional on having a brain study in the window (the imaged subpopulation).
* The cases are those with a stored held-out prediction from the frozen feasibility pipeline. No refit anywhere.
* Estimable: at least 50 imaging-positive cases in the stratum (stop rule, section 8) and, for a per-site statement, at least 50 at that site.

## 5. Estimands and inference (primary)

Predictions: the **stored held-out per-patient predictions** of the silver feasibility run for label E1, scheme **leave-one-site-out** (the primary scheme of the feasibility analysis; the temporal scheme is reported alongside), models fitted on silver E1 and **never refitted or tuned on imaging-E1**:

* EEG arm: the headline rung `combined` (qEEG + connectivity) added to Baseline A; secondary arms: added to Baseline C, `combined_cbramod` (commercial-clean), `combined_morgoth`.
* Baseline arm: Baseline A (severity plus sedation); secondary Baseline C and P.

**Primary endpoint (IU-large): the pair**
1. **ΔAUROC** = AUROC(A + EEG) - AUROC(A) for imaging-E1 (calibration-free, so robust to the difference between silver and imaging prevalence).
2. **Δ masked log loss** = log loss(A + EEG) - log loss(A), after an **intercept-only logit shift** that is estimated on the other site's imaging labels (cross-fitted; if only one site is estimable, estimated on the evaluation set and applied identically to both arms, and labelled in-sample). The unshifted Δ log loss is reported alongside.

Interval: 99% within-site patient bootstrap (B = 10,000), as in SAP A1. **Supportive** if the ΔAUROC interval lies above 0 **and** the shifted Δ log loss interval lies below 0 in IU-large, and the point estimates have the same sign at every site with at least 50 positives. Anything else is **not supportive**, reported as it falls; a point estimate in the wrong direction with an interval excluding 0 is reported as harm. The whole broad cohort is evaluated with the same criteria and reported in the same table (secondary, as in SAP A1.3). There is no multiplicity adjustment beyond this one pre-named primary pair; every other result is labelled exploratory.

**Interpretation rules (fixed now).**
* Supportive on imaging-E1 and silver E1 both positive: the silver E1 gain is not explained by ICD-code circularity alone. It is still not gold and not an ED claim (SAP A1.5).
* Not supportive on imaging-E1 while silver E1 was positive: the silver E1 gain is attributed to code or documentation behaviour until shown otherwise; this is reported with the same prominence. It is read as informative only if the label is validated (step 0d: lower 95% bound of PPV at least 0.70 and of NPV at least 0.90); otherwise it is "inconclusive, label unvalidated".
* Estimable but fewer than 200 positives: descriptive only (a paired ΔAUROC of 0.05 needs about 200 positives, 0.10 about 50, at assumed correlation 0.6; `docs/research/open_etiology_datasets_2026-10-14.md` section 2.2).

## 6. Secondary analyses (exploratory, same machinery)

1. **Concordance of silver E1 and imaging-E1** (`bind.concordance`): 2x2 table, sensitivity, specificity, PPV, NPV of silver against imaging, Cohen's kappa when all cells are at least 11; overall, IU-large, per site, per silver anchor family (dx-only versus procedure), and per imaging class. Imaging is the reference, not the truth.
2. **Per-class** ΔAUROC for infarct, ICH, SAH, SDH/EDH, TBI, mass effect, hydrocephalus, only where a class has at least 50 positives.
3. **EEG-only** AUROC for imaging-E1 (no baseline), and the sedation-intensity and drift controls below.
4. Comparators: Baseline C, P + EEG, `combined_cbramod`, `combined_morgoth`; temporal-holdout scheme.

## 7. Mandatory controls and sensitivity analyses

* **Sedation intensity (D-154).** The grade-0-3 analysis (cut points fixed) and the reading rules (a) to (d) of SAP A1.4(c1) apply unchanged to the imaging-E1 ΔAUROC: ΔAUROC by grade, light versus heavy, site-adjusted trend, EEG-only sedation probe. A stratum under 50 patients is not estimable.
* **Drift (D-153).** The label drift screen for imaging-E1: assessable fraction by calendar tenth within site (imaging coverage can change with the data feed), test-versus-train prevalence ratio (flag above 3 or below 1/3); a flagged label has no temporal interpretation, leave-one-site-out governs.
* **Negative-control outcome** (`ctrl_chronic`): ΔAUROC reported next to the primary.
* Sensitivities (all reported, none changes the primary): (1) windows [0, +1], [-3, +1] (the silver window), [-1, +7]; (2) indeterminate mapped to negative, then to positive; (3) hydrocephalus added to the label; (4) **EEG-influenced imaging**: imaging on or before the EEG day only (offset in [-1, 0]) and, separately, first imaging study only; (5) only classes with an ACUTE qualifier plus the specific class names (no mass effect); (6) MGB-only sites; (7) exclusion of cases whose silver E1 positivity rests only on a procedure code.
* **Selection.** Patients without a brain study are not in the estimand. The share of IU-large with a window study, by age band and site, is reported (suppressed) so the imaged subpopulation can be compared with IU-large; no weighting is applied in the primary.

## 8. Stop rules (fixed)

* **S1, linkage (step 0a):** if the imaging patients overlapping the analysed EEG cohort number fewer than 100 at I0001, stop; report the suppressed overlap only.
* **S2, time base (step 0b):** if the share of linked patients whose BIND `ShiftedDateOfBirth` equals the HEEDB/OMOP shifted date of birth to the day is below 95%, the BIND and HEEDB date shifts differ and day-level timing is not defensible: stop; report the agreement only.
* **S3, positives (step 1):** fewer than 50 imaging-positive cases in a stratum makes that stratum **not estimable**; if both IU-large and the whole broad cohort are below 50, stop and report suppressed counts only. The first count of positives is a feasibility count and is logged before any model-based number is computed.
* **S4, validation:** if step 0d shows PPV or NPV below the bounds in section 5, run and report the analysis but label every result "label-validity limited".

## 9. Order of work (nothing below has been run)

Each step is a human-launched, aggregate-only job via `scripts/heedb_run.sh` (CLAUDE.md rules 2, 3, D-118); step numbers are the order of execution.

* **0a Linkage count.** `bind.linkage_qc`: suppressed overlap of imaging patients with the analysed cohort, per HEEDB site.
* **0b Time base.** `bind.shift_agreement`: BIND `ShiftedDateOfBirth` versus OMOP `person.birth_datetime` for linked patients.
* **0c Vocabulary.** Distinct value strings (no counts) of `Brain`, `Brain-related`, `Acute/Chronic`, `PresentPrior`, `Change`, `Test`, `Nullfinding`, `Label_LLAMA_Level1`, `Label_LLAMA_Level2`, and `Type` of the reports table; the question of how normal studies are represented; the share of rows with a non-empty `Label_manual` (suppressed). No free-text column (`Name`, `General Name`, `Location`, `Magnitude`, `Report_txt`) is listed. The YAML is finalised and hashed; this plan, the YAML and the code hash are registered in DECISION_LOG (new entry) before step 1.
* **0d Label validity (optional but required for a strong reading).** A clinician reads 100 random window-positive and 100 random window-negative reports in `local_only/` (rule 5; no agent reads them, no hosted model sees them, rule 4), recording agreement with the LLM-derived label; PPV and NPV with Wilson 95% intervals. If `Label_manual` is populated, its agreement with `Label_LLAMA_*` is reported as well. An open-weight model on approved compute may re-extract as a further comparison.
* **1 Count.** `bind.build_label_table` on the cohort; `bind.summarise` (suppressed). Apply S3.
* **2 Analysis.** Section 5 to 7, from stored predictions. Report: aggregate-only, `out/` gitignored (rule 6); no number from this analysis is committed.

## 10. Limits, stated in advance

* The label is LLM-extracted (Bio-Medical-Llama-3-8B), with no published accuracy; it can miss acute findings or hallucinate them. Label noise biases ΔAUROC toward 0 and a null result is uninformative without step 0d.
* Imaging exists only for patients who were imaged and had an EEG; spectrum and verification bias apply. The estimand is not "all patients with altered mental status".
* Day-resolution timing cannot order a same-day CT and EEG.
* BIND covers MGB (I0001) only for EEG linkage; the other HEEDB sites have no imaging. External validity beyond the labelled sites is not addressed.
* The frozen silver-trained model optimises silver E1; imaging-E1 is a different construct (SAP A1.1 caveat). A gain on imaging-E1 is evidence about structural signal in the representation, not the SAP's gold-standard H-hypotheses.
* **Rights.** BIND is under the BDSP Credentialed Health Data License 1.5.0 (research; commercial use not addressed on the page; the paper routes commercial users to a separate hospital agreement). BIND-derived labels are used as research evidence only. **The commercial-clean path (CBraMod-based `combined_cbramod` and any later retraining) will not use BIND-derived labels for training or tuning of any product.** Ledger id `bind_v1_0` remains red for commercial purposes.
