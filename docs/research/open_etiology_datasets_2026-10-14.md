# Open data for "does EEG carry etiology information in acute encephalopathy?" - search, verification, ranked options

Date: 2026-10-14. Builds on `docs/research/dataset_alternatives_2026-10-09.md` (not repeated; this note goes deeper on OPEN data only). Label definitions: `docs/prereg_study1_sap.md` Amendment A1 (primary E1 structural, E2 hypoxic-ischemic, E5 metabolic; control E3; secondary E4a, E6, E7).

Method. Primary landing pages opened with WebFetch (PhysioNet, BDSP, AWS Open Data registry, NEDC, OpenNeuro GraphQL, Zenodo/Dryad/figshare/Harvard Dataverse/OSF REST APIs); PubMed (E-utilities) and Europe PMC full text for papers. No restricted data was read or used; `out/` was not read; no record-level data of any kind was opened. The TUEG, I-CARE and other corpora named below were NOT downloaded.

Tags. **[V]** read from a primary page or record in this session. **[U]** unverified (search snippet, memory, inference, or page silent). **[P]** project-internal statement carried over from `docs/`. **[M]** my own calculation or judgment.

---

## 1. Bottom line (candid)

1. **No adequate open dataset exists.** I found no public or credentialed-download corpus that has all of: raw adult EEG waveforms, several etiologies of encephalopathy/coma/delirium in one acquisition pipeline, per-patient diagnosis labels assigned by someone other than us, and a comparison group of non-etiologic encephalopathy or ICU controls. Repositories searched programmatically: Zenodo, Dryad, figshare, OpenNeuro, Harvard Dataverse, OSF, PhysioNet (all 28 "EEG" items, three listing pages), BDSP topic listings (partial, see section 6), NEDC, PubMed "Scientific Data" EEG-patient datasets. Searched by web search only (no hits, but not exhaustive): Mendeley Data, Kaggle, IEEE DataPort, Synapse, ScienceDB. Not searched: Korean/Japanese national repositories, DANS, CNGB.
2. **What open data does contain is single-etiology or severity-only**: post-arrest coma (I-CARE, E2 only, cause of arrest not even provided [V]); delirium severity (VE-CAM-S: no etiology [V]); DoC after chronic injury (tiny, etiology unknown on the landing page [V]); acute stroke and mild TBI during motor-imagery or cognitive tasks in conscious patients; and several clinically interesting sets that are tabular only (delirium Dryad deposit is a PDF; COVID-EEG Dryad deposit is a 20 KB CSV [V]). A "panel of single-etiology contrasts" cannot be assembled from open data without pooling different sites and devices, and pooling is where site/device confounding would dominate (section 4.2).
3. **The least-bad open route to independent etiology labels is TUEG's clinical-history text**, reachable only through NEDC's keyword-script service (reports are no longer distributed [V]). Labels would be written by the treating/reading neurologists, not by us, but they are keyword-derived, unvalidatable (we never see the text), and the TUEG case mix is epilepsy-heavy (87% of reports contained "epilep" in the 2016 release [V]). This is rank 1 below, with a count-first gate.
4. **Published (non-open) evidence that EEG carries etiology information exists and is moderate**: CERTA CNN 4-class etiology accuracy 54.5% overall (84.1% in the top-confidence subset) [V, PMID 36049354]; NCSE vs metabolic encephalopathy vs benzodiazepine intoxication AUC 0.905 with prospective maintenance [V, PMID 39033787]; pediatric cerebral malaria vs non-malarial febrile coma AUROC 0.85-0.90 [V, PMID 40736255]. None of these tests the SAP's estimand (increment over severity, sedation and clinical context), and none is open.
5. **An open-weight LLM can pre-screen but should not substitute for human gold labels.** Published validations against clinician adjudication show kappa/accuracy around 0.90 for explicit, rule-like facts, but 0.40-0.78 for etiologic inference (stroke subtype, cause of death), and I found no study validating LLM adjudication of encephalopathy etiology (section 7).
6. **Candid conclusion for the decision the lead faces.** Open data cannot produce "strong evidence" on the incremental-etiology claim. It can produce moderate, independent, construct-validity evidence for E1/E2/E5 (rank 1-3 below). If the funding rule is strictly "strong open evidence first", it cannot be met; if it is "is there any independent signal worth a pilot?", rank 1 plus the already-seen silver result plus the published literature is a defensible yes-with-caveats, and the pilot in the SAP (Gate G0) remains the only route to the pre-registered gold reference standard.

---

## 2. Ranked shortlist (single experiments, most convincing evidence soonest)

All three top options score a frozen model; none trains on the external data. Success criteria and label mappings must be fixed in a dated amendment (A3-style) before the first scoring run, and the run itself is a human-run, aggregate-only job (CLAUDE.md rules 1-3: synthetic data only in agent sessions; these corpora are real patient data even though de-identified) [P]. Because the SAP freeze order requires hashing before any gold label or model output is opened, any pre-freeze run must be logged in DECISION_LOG and disclosed as an exploratory go/no-go look (no gold labels exist, so nothing is unblinded) [M].

| Rank | Experiment | Data and access | Who assigned the labels | Expected n | Time to first result [U] | Main confounds | Licence / commercial |
|---|---|---|---|---|---|---|---|
| **1** | **Frozen-model transport test on TUEG with NEDC-run history-keyword labels** (E1, E2, E5 primary; E3 positive control; E4a, E6, E7 only if counts allow). EEG-only head scores vs matched within-TUEG controls; compare with an age/sex/medication-only baseline. | TUEG v2.0.2, 26,846 recordings [V]. Free download after signed form, credentials 24-48 h [V]. Labels via Python script run by NEDC [V]. | Report authors (treating/reading neurologists). Our regexes only extract. | **Unknown; count query first.** Case mix is epilepsy-weighted [V]. My guess is E1 and E2 reach >=100 positives and E5 only if hepatic/renal/sodium/glucose are pooled [U]. | Credentials 1-2 days [V]; script turnaround and count return unknown [U]; scoring days after counts. | Label precision unverifiable (we never see text); negation and "history of"; setting (ED/ICU/outpatient) unknowable [V: not stated]; sedation/antiseizure meds; year/device drift; CBraMod encoder was pretrained on all of TUEG (unlabeled) [P], so the CBraMod arm is not held-out at encoder level (MORGOTH arm is cleaner; it used TUH only as external test [V]). | NEDC v6.0: research and technology development, no redistribution, delete when finished [V]. No explicit NC clause, no explicit commercial grant; counsel confirmation needed [P]. Result is evidence for a go/no-go, not a training source. |
| **2** | **Imaging-anchored E1 test inside HEEDB** using BIND's imaging-report metadata as the structural reference (acute lesion present/absent within a time window of the EEG), scored against the frozen E1 head and against expert EEG-pattern labels. | BIND: 38,942 patients, 1.8 M scans, linked to HEEDB EEG; controlled access (DUA + CITI) [V]; metadata extracted from 84,960 reports with Bio-Medical-Llama-3-8B [V]. | Radiologists wrote the reports; BDSP's open-weight LLM extracted fields. Not us. | Large (thousands) [U]; exact EEG-imaging overlap and field list not read. | Weeks (access, linkage rules, time-window definition) [U]. | Extraction accuracy of the LLM not read [U]; imaging only for patients who had imaging (selection); same four Boston hospitals as training (not external); EEG reads may have influenced imaging orders. | BDSP controlled/credentialed; paper is CC BY-NC-ND [V]; registry text says non-commercial research use [V]. Not commercial-clean. |
| **3** | **I-CARE positive control for the E2 head**: does the frozen E2 output track post-arrest severity (CPC, burst-suppression-dominated hours) across the 607 open patients? | PhysioNet I-CARE v2.1, open, 607 patients, 80,809 segments, 32,712 h, 1.5 TB [V]. | Consortium clinicians (outcome CPC at 3-6 months [V]). | 607 (good vs poor outcome counts not read). | Days after download. | Tests a severity gradient within one etiology, not etiology discrimination (no non-arrest controls; cause of arrest not provided [V]); mixing I-CARE with other sources to get controls would make the source the strongest feature (section 4.2); cohort is the training reference for much of the field, so contamination of public encoders is possible [U]. | **CC BY-NC-SA 4.0** [V]: non-commercial and share-alike; fine for a go/no-go, red for any commercial training or release of derived weights. |
| 4 | Descriptive expert-pattern x etiology-proxy table inside HEEDB (MORGOTH internal_dataset expert labels: GPD, LPD, LRDA, GRDA, BS, focal/generalized slowing, vs structured etiology proxies and imaging) | Credentialed, already held [P] | EEG side: expert readers (not us). Etiology side: EHR anchors (our rules, circular per SAP A1.1). | Large | Days | Labels were sampled per pattern class (enriched), so only cross-class comparisons are valid, not prevalences [M]; adds little beyond what the field already knows (e.g., GPD with metabolic/hypoxic causes) [M]. | BDSP credentialed; MORGOTH code CC BY-NC 4.0 [V]. |
| 5 | Literature-only synthesis (no new data): CERTA 2022, Kim 2024, Malawi 2025, plus the project's own aggregate silver results | PubMed | n/a | n/a | Immediate | Different populations/labels; none tests increment over context | Citable |

### 2.1 Why rank 1, and its design in brief

- **Step 0 (count-first gate).** Obtain NEDC credentials. Submit one script per label family returning matching filenames; record only counts. If fewer than about 50 positives for a family, drop it from the primary analysis (power table below). If E1, E2 and E5 are all under 50, stop: the open-data route fails and the answer to the lead is "not feasible".
- **Label sources.** Use the CLINICAL HISTORY and MEDICATIONS sections only. Exclude IMPRESSION and CLINICAL CORRELATION, which are written after reading the EEG and would import the circularity the plan bans. Report sections confirmed from a published report example: clinical history, medications, description of the record, impression, clinical correlation [V, LREC 2020 paper, Figure 1 text]. Section header strings in the corpus itself are not confirmed [U].
- **Keyword families (fix in the amendment before scoring).** E1: stroke, infarct, hemorrhage, SAH, ICH, TBI, subdural, mass/tumor, hydrocephalus. E2: cardiac arrest, anoxic, hypoxic, asphyxia, drowning. E5: hepatic encephalopathy/cirrhosis/ammonia, uremia/renal failure/ESRD, hyponatremia, hypoglycemia, hypercapnia. E3 control: status epilepticus, NCSE (expect many). E4a: overdose, ingestion, intoxication, toxic. E6: sepsis, bacteremia. E7: encephalitis, meningitis, autoimmune, anti-NMDA. Negation and "history of" rules written by us; their precision cannot be measured with NEDC's service (filenames only) [V], so state the label as "history mentions X", not "has X".
- **Controls.** Same corpus. Primary: history mentions altered mental status / confusion / encephalopathy / unresponsive / coma and none of the keyword families. Secondary: any non-etiologic indication. Match on age band, sex, session year (TUEG session folders carry the year [U]), sampling rate and channel count, and sedative/antiseizure medication class from the medications section.
- **Scoring.** Frozen heads, no refit on TUEG. Report AUROC per family against matched controls with 95% intervals, and the delta over a baseline that sees only age, sex, year and medication class (the EEG heads and the baseline both frozen or both fit on HEEDB, never on TUEG).
- **Interpretation rule.** Supportive if the E3 control shows clear signal and at least two of {E1, E2, E5} have a lower 95% bound above 0.5 against encephalopathy-matched controls. A null result is uninformative (label noise can mask a real effect). No claim about E4a, E6, E7 unless counts allow.

### 2.2 Rough power [M] (Hanley-McNeil standard errors, 3 controls per positive)

Smallest AUROC distinguishable from 0.5 with 80% power:

| Positives (controls = 3x) | alpha 0.05 two-sided | alpha 0.05/6 (six contrasts) |
|---|---|---|
| 20 | 0.708 | 0.756 |
| 30 | 0.671 | 0.711 |
| 50 | 0.633 | 0.664 |
| 100 | 0.594 | 0.617 |
| 200 | 0.567 | 0.583 |
| 400 | 0.547 | 0.559 |

Smallest AUROC gain over a baseline AUROC of 0.75 (paired; assumed correlation 0.6 between the two AUROC estimates; 80% power, alpha 0.05 two-sided): 50 positives 0.10; 100 positives 0.073; 200 positives 0.053; 400 positives 0.038; 800 positives 0.027. With correlation 0.8 the figures are about 25% smaller. **Increment-over-baseline claims therefore need several hundred positives per label; TUEG is more likely to support "AUROC above chance vs matched controls" than "increment over clinical context".**

Label noise attenuates the observed AUROC roughly as 0.5 + precision x (true AUROC - 0.5) when negatives are clean [M]. Example: true 0.75 with 60% label precision observes about 0.65. Anchors from the literature (different questions, so only context): CERTA 4-class accuracy 54.5% overall [V], NCSE/metabolic/benzodiazepine AUC 0.905 [V], cerebral malaria AUROC 0.85-0.90 [V]. My guess for realistic TUEG keyword labels is observed AUROC 0.58-0.70 [U], i.e. detectable only if each family has about 50-200 positives.

---

## 3. Question A: multi-etiology open EEG sets (what exists, verified)

None meets the A-criteria. Everything found, with the reason it fails:

| Dataset | URL | Licence | Size, channels, rate | Label source | Why it does not meet A |
|---|---|---|---|---|---|
| I-CARE v2.1 (PhysioNet) | https://physionet.org/content/i-care/2.1/ | CC BY-NC-SA 4.0 [V] | 607 patients, 80,809 segments, 32,712 h, 1.5 TB; 7 academic hospitals (codes A-F in files); 22 named EEG leads (Fp1...F9) plus ECG; sampling rate not on page [V] | Consortium; age, sex, hospital, OHCA/IHCA, shockable rhythm, ROSC time, TTM, CPC outcome; **cause of arrest not provided** [V] | One etiology (E2). 413 further patients held back (challenge validation/test) [V]. |
| I-CARE Brain Continuity Index (Dryad) | doi:10.5061/dryad.2fqz612zv | CC0 [V, listing] | 52 MB derived | Derived index | Derived features only. |
| VE-CAM-S (BDSP open tier) | https://bdsp.io/content/ve-cam-s/1.0/ ; doi:10.60508/jfpq-mj80 | ODbL v1.0 on the dataset page [V]; the AWS registry page for the open tier says ODC-By and "non-commercial research use" [V]: **inconsistent, resolve with BDSP** | 404 analysed adult inpatients (132 neither, 143 delirium, 129 coma) from MGB, 2015-2019; size, channels, rate and whether raw EEG is included are not stated [V] | CAM-S Long Form, RASS, visually scored EEG features, GOS, mortality; **no etiology or diagnoses** (the paper says the score "is not specific to the cause of encephalopathy") [V, PMC8769081] | Severity, not etiology. Single MGB site, probably overlapping HEEDB patients [U]. Paper says data on request; the BDSP page says files are open [V both]. |
| E-CAM-S (BDSP restricted) | https://bdsp.io/content/e-cam-s/1.0/ | Restricted | 373 patients; 4 frontal channels (Fp1-Fp2, Fp1-F7, Fp2-F8, F7-F8) [V, PMC8678335] | CAM / CAM-S | "Various causes of altered mental status", no etiology breakdown [V]. |
| Delirium EEG slowing (Dryad) | doi:10.5061/dryad.tv06pt2 (PMID 31467255 [U on PMID]) | CC0 [V] | 200 patients; the only file is a 217 KB PDF supplement [V] | 3D-CAM, clinical EEG reports | No waveforms. |
| COVID-19 inpatient EEG (Dryad) | doi:10.5061/dryad.gmsbcc2mp | CC0 [V] | 192 adults, 4 hospitals; one 20 KB CSV [V] | Chart review (indication encephalopathy 54.7%, coma 17.2%, metabolic abnormalities recorded) [V] | Tabular only, one pathogen. |
| Prolonged DoC resting EEG (figshare) | https://figshare.com/articles/dataset/23552964 | CC BY 4.0 [V] | 59 patients + 32 controls; 62 electrodes, 250 Hz, BrainVision .dat, 60 two-second epochs per segment [V] | Behavioral DoC diagnosis; etiology not on the landing page [V] | Chronic, not acute; epochs only; etiology unknown. |
| DoC pilot (Zenodo) | https://zenodo.org/records/6951440 | CC BY 4.0 [V] | 4 EDF files (about 75 MB) [V]; the paper had 33 DoC patients, traumatic vs non-traumatic [V, PMID 36009445] | Rehab-unit clinicians | Four files only. |
| Bath DoC BCI set | https://researchdata.bath.ac.uk/1632/ | [U] | UWS 14 / MCS 17 / LIS 11 [P from prior survey] | Behavioral | Chronic, task-based. |
| Acute stroke motor-imagery EEG (Sci Data 2024, PMID 38272904) | doi:10.1038/s41597-023-02787-8 ; data on figshare "EEG datasets of stroke patients" (Liu H, Lv X, 2022) | Article CC BY 4.0 [V]; dataset licence not read [U] | 50 acute stroke patients (1-30 days), Xuanwu Hospital; 30 EEG + 2 EOG channels, 500 Hz [V] | Hospital diagnosis (stroke) | Conscious patients doing a task; no control group; one etiology. |
| Acute single-channel EEG after stroke (Dryad) | doi:10.5061/dryad.h6986 | CC0 [V] | 24 patients within 72 h; one xlsx of relative band power (Fp1 only) [V] | MoCA at 90 days | Features only. |
| Mild TBI acute/chronic (OpenNeuro ds003522, ds003523, ds005114) | https://openneuro.org/datasets/ds003522 | CC0 [V] | 91-96 subjects each, 27-60 GB; session 1 at 3-14 days post-injury [V] | Study group (control, sub-acute mTBI, chronic TBI) | Mild injury, conscious, tasks; not impaired consciousness. |
| COVID survivors EEG (OpenNeuro ds008082) | https://openneuro.org/datasets/ds008082 | CC0 [V] | 173 EEGs (86 healthy, 87 COVID); 1.2 GB; 8 min eyes closed [V] | Cohort | Survivors, not encephalopathy. |
| NMT / NMT-Events (Zenodo) | https://zenodo.org/records/21629135 ; /21630052 | CC BY 4.0 [V] | About 13 GB and 10 GB [V] | Normal/abnormal; event annotations | No etiology. |
| VitalDB | https://physionet.org/content/vitaldb/ ; Sci Data 2022 PMC9178032 | Paper CC BY 4.0 [V] | 6,388 surgical cases; BIS Vista EEG waveform 128 Hz; preoperative sodium, glucose, BUN, creatinine, AST/ALT, albumin, pH, PaCO2 [V] | Lab values | Anaesthetised EEG; a weak, objective metabolic-proxy test at best; project ledger already yellow/on hold [P]. |
| Minimal hepatic encephalopathy (Harvard Dataverse) | doi:10.7910/DVN/53W4LE | CC0 [V] | One 8.8 KB table | Psychometrics, fNIRS | Not EEG. |

Negative-search log (for the record): Zenodo/Dryad/figshare/Dataverse/OSF queries for hepatic, uremic, hypoglycaemia, sepsis, septic encephalopathy, autoimmune encephalitis, intoxication/overdose, ICU, status epilepticus returned only the items above or irrelevant records. A 2024 IEEE Access scoping review of open EEG healthcare datasets exists (Peres da Silva, Tedesco, O'Flynn, doi:10.1109/ACCESS.2024.3376254 [V citation]); I did not read its full dataset table.

---

## 4. Question B: per-etiology panel feasibility

### 4.1 Table

| Etiology | Best open data | Acute? Adult? | Controls available | Verdict |
|---|---|---|---|---|
| E1 structural (stroke, ICH, TBI) | Acute-stroke MI set (n=50) [V]; Dryad single-channel (n=24) [V]; OpenNeuro mild TBI (CC0) [V]; TUEG keyword [rank 1]; BDSP SAH-DCI (restricted) [V] | Acute yes but conscious, task-capable, adult | None of the right kind (no encephalopathic controls) | Only TUEG keyword and BIND imaging (rank 2) are usable. |
| E2 hypoxic-ischemic | I-CARE (open, NC-SA); BDSP I-CARE v2.0 (1,020 patients, more than 50,000 h, restricted [P]); BDSP 1,000-patient post-arrest EEG-pattern table (credentialed, tabular, July 2026) [V] | Acute coma, adult | None (all arrest) | Positive control only (rank 3). Licence NC-SA. |
| E3 seizure/NCSE | TUSZ v2.0.6 (NEDC) [V]; HMS Kaggle (1,950 patients, 17,089 EEGs, 106,800 labelled 50 s samples, 200 Hz; expert votes for seizure, LPD, GPD, LRDA, GRDA, other; licence **not verified**) [V arXiv 2412.07878]; SPaRCNet (2,711 patients, 6,095 EEGs, 50,697 samples; BDSP restricted) [V]; CHB-MIT, Siena (outpatient/EMU) | Mixed | Non-seizure segments | The most abundant independently labelled class, but EEG-defined, not etiology. Use as positive control. |
| E4a toxic | None open. BDSP NESI includes an ICANS (CAR-T neurotoxicity) cohort with matched outpatient controls (credentialed) [V] | Adult | Outpatient controls | No open set. ICANS is immune-mediated neurotoxicity and does not map cleanly to E4a [M]. |
| E5 metabolic (hepatic, uremic, glucose, sodium, hypercapnia) | None open with waveforms. UCL cirrhosis cohorts (264 patients + 98 controls; 226 + 137) published, data sharing not shown [V from search snippets, U]; hypoglycaemic-clamp EEG in 34 T1D patients (Rubega 2020, Entropy) data availability not read [U]; VitalDB labs [V] | Hepatic/clamp not acute encephalopathy | Healthy controls | Not feasible as a panel. TUEG keyword only. |
| E6 septic | None | n/a | n/a | None. Consistent with SAP A1 demoting E6. |
| E7 autoimmune/CNS infection | None | n/a | n/a | None. |
| Delirium (all-cause) | VE-CAM-S (open), E-CAM-S (restricted), JSMF ACCESS (intraoperative derived features, 646 adults, **PhysioNet restricted**, published 21 Sep 2026; raw EDF only on request) [V], ICU SLEEP (482 ICU adults, credentialed, **no EEG in release**) [V] | Postoperative or inpatient | Non-delirious | Severity/outcome labels only. |

### 4.2 The pooling-confound risk, stated explicitly [M]

Any design that takes positives from one dataset (e.g., I-CARE) and controls from another (e.g., TUEG, VE-CAM-S, HMS) lets a classifier learn the source: channel set and order (I-CARE lists 22 EEG leads including F9; TUEG's most common file has 31 EEG-only channels), sampling rate (TUEG 87% 250 Hz [V]), amplifier and filter chain, reference, mains frequency, impedance, record length, and era. Source identification is usually near-perfect and far easier than etiology. Rules for this project:

1. Contrast cases and controls **within one source** (TUEG, or one HEEDB site) whenever possible.
2. If sources must be mixed, first train a source-identification probe on **controls only**; if it reaches AUROC above about 0.8, the contrast is dead and must not be reported as etiologic.
3. Harmonise to the model's input contract (19 channels, 10-20, 200 Hz, 0.5-70 Hz, common average, as in MORGOTH [V]) and report with and without harmonisation.
4. Never pool I-CARE (NC-SA) with commercial-clean data in any artifact that ships.

---

## 5. Question C: TUH EEG corpus, current status (verified 2026-10-14)

- **Reports.** The NEDC page states "We no longer distribute reports with our corpora." Users may send a standard-library Python script that takes a file listing filenames and returns matching filenames; NEDC runs it on the anonymized reports and returns the matches; send to help@nedcdata.org with subject "Python Script" [V]. The page warns that reports are unstructured and searching is "not trivial". The page gives no report count and does not mention ICD codes [V].
- **Clinical-history / medication / impression sections.** They exist in the text: a report example in the LREC 2020 EEG-report paper shows CLINICAL HISTORY, MEDICATIONS, description, IMPRESSION, CLINICAL CORRELATION [V]; the 2016 corpus paper describes reports as "unstructured text that describes the patient, relevant history, medications, and clinical impression" [V]. A search snippet states EEGs since 2011 also carry ICD-9 codes [U]; neither the NEDC page nor the 2016 paper mentions ICD [V], so do not assume codes are reachable via the script service.
- **Subcorpora** (current releases): TUEG v2.0.2 (26,846 recordings, 2002-2017), TUAB v3.0.1 (normal/abnormal), TUAR v3.0.1 (artifacts), TUEP v3.1.0 (100 epilepsy + 100 non-epilepsy subjects with metadata on diseases and medications), TUEV v2.0.1 (six event classes), TUSZ v2.0.6 (seizures), TUSL v2.0.1 (slowing) [V]. None carries an etiology label; TUEP's disease/medication metadata covers only 200 subjects [V].
- **Case mix.** 2016 release: 16,986 sessions, 10,874 subjects, age 51.6 +/- 55.9, 51% female; 87% of reports contain "epilep"; most files 250 Hz (87%) and 31 EEG-only channels [V]. Clinical setting (ED/ICU/inpatient/outpatient) is not stated anywhere I read [V].
- **Licence** (NEDC form v6.0, 3 June 2025 [V], read from the PDF text): "offered free and without restriction, except for a few conditions": acknowledge the provider; no release to third parties or redistribution; no re-identification; no malicious use, "This data can be used for research and technology development, but not for uses beyond these broad classifications"; "delete the data from all computer systems when finished". No explicit non-commercial clause and no explicit commercial or derivative-weights grant. Consistent with `docs/data_rights.md` (yellow) [P].
- **Provenance flag.** The CBraMod checkpoint used by the project was pretrained on all of TUEG [P], so TUEG-derived evaluations are in-distribution for the encoder (unlabeled).

---

## 6. Question D: BDSP/HEEDB-linked resources beyond what we use

Listings read: BDSP EEG topic page 1 of 5 only (10 items), delirium topic (4), search "ICANS" (10), AWS registry pages for the open, credentialed and restricted tiers [V]. Other listing pages were not read; there may be more.

| Resource | Tier / licence as shown | Content | Etiology relevance |
|---|---|---|---|
| **Open tier (no credentialing)**: RHOFBAS (sleep), Human Burst Suppression EEG of Deep Hypothermia (11 patients, raw EEG, CC BY-SA 4.0 [V]), VE-CAM-S (ODbL [V]) | AWS registry: Open Data Commons Attribution License v1.0 and "non-commercial research use" [V]. Page-level licences differ from the registry text; unresolved | See section 3 | None. |
| HEEDB v4.1 | Credentialed (bdsp.io) vs "Restricted Health Data License 1.0.0" (AWS) [P/V] | Our primary source | n/a |
| **MORGOTH 1.0** (doi:10.60508/v7ky-5g40; June 2026; credentialed; BDSP Credentialed Health Data License 1.5.0; code CC BY-NC 4.0) [V] | Credentialed | `data/internal_dataset/` folders: NORMAL, ABNORMAL, BS, FOCALSLOWING, GENSLOWING, SEIZURE, LPD, GPD, LRDA, GRDA, IIIC, SEIZURE_BCH, SPIKES (+_BCH, _FOCAL_GEN, _HM), BIPD, BIRD, PDR, POSTS, BETS, WICKETS, VW, SPINDLES, AWAKE, N1/N2_19Channel, SLEEPSTAGING, MoE (multi-expert consensus), Morgoth_test_dataset_10s; 17 EEG-level binary findings; 19 ch, 200 Hz; development 18,677 patients, internal validation 13,334, external 1,573 from 48 institutions (external sets not in the S3 listing); pretraining 14,500 patients (7,793 routine, 2,942 ICU, 765 EMU) [V] | **No etiology or diagnosis field** ("No etiology or diagnosis breakdown is provided beyond the label categories") [V]. All labels are EEG-derived. They become etiology evidence only by linking sample IDs to OMOP (our silver layer) - rank 4. |
| SPaRCNet / IIIC | Restricted, "Controlled Access" [V] | 2,711 patients, 6,095 EEGs, 50,697 expert-labelled samples, 18 institutions [V] | EEG-derived. |
| HMS Harmful Brain Activity (Kaggle) | **Licence not verified**; BDSP-adjacent (Harvard/CDAC/Persyst/Jazz organisers) [V from search snippets] | 1,950 patients, 17,089 EEGs, 106,800 rows, 200 Hz [V] | EEG-derived; public download but terms unknown. I recall a non-commercial licence [U]; check the Kaggle Data tab before any use. |
| GROND v1.0.0, LENS v1.1.0 | Credentialed [V] | Periodic/rhythmic pattern characterisation; normative slowing | EEG-derived. |
| **NESI** (doi:10.60508/rrbg-ba24; June 2026) | Credentialed, License 1.5.0 [V] | HEEDB-derived 10-minute 19-channel 200 Hz segments with RASS (97,620 valid), GCS (110,836), CAM-S (526), ICANS grade (784), mortality (5,675 subjects, 1,333 deaths); propofol effect-site concentration and 24 h medications; folder `DiagnosisMetadtafiles/` with admission-diagnosis tables (contents not described) [V] | Closest "label not assigned by us" resource for severity, plus an ICANS (immune-effector neurotoxicity) cohort with matched outpatient controls. Etiology only via the diagnosis tables (unread). |
| ICU SLEEP | Credentialed [V] | 482 ICU adults, delirium-free days, no EEG in release [V] | None. |
| Post-arrest EEG patterns, 1,000 patients (doi:10.60508/3f4n-dz72; July 2026) | Credentialed [V] | Tabular REDCap CSV; CPC outcome; cause of arrest not stated [V] | E2 only. |
| I-CARE v2.0 (BDSP) | Restricted [V] | More than 50,000 h, 1,020 patients [P] | E2 only. |
| SAH delayed cerebral ischemia | Restricted [V] | Continuous EEG after SAH | E1 subtype only. |
| **BIND** (Sci Data 2026, doi:10.1038/s41597-026-07421-x) | Controlled access: DUA + CITI [V]; article CC BY-NC-ND [V] | 38,942 patients, 1.8 M scans (MGH, BWH, Stanford), full-text reports, Bio-Medical-Llama-3-8B-extracted metadata for 84,960 reports; linked to HEEDB EEG/PSG [V] | **Independent structural reference for E1.** Extraction accuracy not read [U]. |

Net for D: BDSP has no etiology-labelled EEG set beyond cohort-defined etiologies (arrest, SAH, CAR-T neurotoxicity). Every BDSP tier states non-commercial research use on the registry pages [V], so none of it is a commercial-clean training source regardless of how it is used for evidence.

---

## 7. Question E: open-weight LLM adjudication from notes - published validation

CLAUDE.md rule 4 allows only open-weight models on approved compute [P]; studies that used proprietary models (flagged) show the ceiling, not what we may run.

| Study | Task | Model | Reference standard | Result | Read |
|---|---|---|---|---|---|
| JAMA Netw Open 2025, PMID 41134571 [V] | Presenting signs/symptoms from admission notes (5 MGB hospitals) | **Llama 3 8B** (open-weight) | Blinded physician review, 303 notes | accuracy 99.3%, balanced accuracy 84.6%, sensitivity 69.7%, PPV 68.4% | Open-weight, same health system; extraction of documented items is only moderately sensitive. Accuracy is inflated by negatives. |
| Alzheimers Dement (DADM) 2026, PMID 42802831 [V] | Lecanemab eligibility variables from German reports | **gpt-oss-120b** on-premises | Expert consensus, 99 held-out | accuracy 94%, kappa 0.90 | Rule-like, explicit criteria; best open-weight evidence found. |
| npj Digit Med 2025, PMID 41388138 [V] | Clinically significant postoperative delirium from notes | **Llama-3-70B**, GPT-4o | Antipsychotic use or neurologist diagnosis | c-statistic 0.74 (Llama), 0.76 (GPT-4o); LLM-run Fleiss kappa 0.85 vs physicians 0.22 (what the raters were is not clear from the abstract; reliability, not validity) | Delirium is detectable from notes at modest discrimination. |
| Stroke 2025, PMID 40709446 [V] | Stroke type and ischemic subtype | GPT-4o (proprietary) | GWTG-Stroke registry abstractors, 4,123 hospitalizations | type accuracy 98%; subtype sensitivity 0.40 (cryptogenic) to 0.95 (small-vessel) | Etiologic subtyping is much weaker than detection. |
| medRxiv 2026, PMID 41646769 [V] | Infection presence/site in cirrhosis ED notes | Claude 3.5 Sonnet (proprietary; not allowed for our notes) | Physician-adjudicated, 1,000 encounters | accuracy 94.7%, sensitivity 99.5%, PPV 92.8% | Ceiling for E6-type "infection documented". Preprint. |
| Cureus 2025, PMID 40486463 [V] | Cause of death | Claude 3.5 Sonnet (proprietary) | Autopsy, 150 cases | underlying cause 69.3% (history) to 78.0% (history + PM-CT) | Etiologic inference from history ~70-78%. |
| J Clin Med 2026, PMID 42278954 [V] | 17 MS-note phenotypes | GPT-5.2 vs Llama-3.1 8B (open) and others | Two humans + adjudication, 100 notes | GPT-5.2 macro-F1 0.80; local open model weaker (numbers not read) | Small open models lag frontier ones. |
| medRxiv 2026, PMID 41646828 [V] | Dementia identification, MGB | RAG-LLM (model not read) | Manual chart review | F1 0.933, sensitivity 91.1%, PPV 95.5% | Documented-diagnosis detection works. |
| Nat Commun 2025, PMID 41326410 [V] | Trial outcome pre-adjudication from telephone vignettes (1,046) | Fu-LLM (domain model) | Trial adjudicators | Outperformed other LLMs and, in silico, human adjudicators | Event adjudication, not etiology. |

**Gap.** About six targeted searches (PubMed tiab and Europe PMC) found **no validation of LLM adjudication of encephalopathy/altered-mental-status etiology** against clinician adjudication. The nearest are etiologic subtyping (stroke) and cause-of-death studies, where accuracy falls to 0.40-0.78 on the inferential part.

**Judgment [M].**
- An LLM reading notes extracts **documented** etiology; it does not **adjudicate** the evidence (labs, imaging, course) the way the SAP's adjudicators do. Used on notes alone it is another silver label with different noise, and it shares the notes' documentation bias (notes may be written with the EEG known; blood-culture-style availability issues from D-153 remain).
- Fit for **pre-screening**: rank cases by LLM-extracted etiology and confidence; send the LLM-uncertain and LLM-disagreeing cases plus a random sample to humans; this can reduce reader load but I cannot quantify the saving from published data.
- Not fit to **substitute** for the pre-registered gold labels. At best, LLM labels can be a pre-registered secondary label set, validated against humans.
- Any validation still needs human labels. Approximate 95% kappa half-widths when the true kappa is 0.7 [M, normal approximation]: prevalence 0.10: n=100 +/-0.25, 200 +/-0.17, 300 +/-0.14, 500 +/-0.11. Prevalence 0.20: n=100 +/-0.18, 200 +/-0.13, 300 +/-0.10. Prevalence 0.35: n=100 +/-0.15, 200 +/-0.10. So a per-label validation to +/-0.10 needs roughly 300 cases for a 20%-prevalence label and about 500 for a 10% label, which is not far below the size of a human adjudication pilot. For rare labels (E7, E4a, E6) LLM validation is not cheaper than human adjudication.
- Compute and model: open-weight candidates with published clinical-note use are Llama-3-8B/70B, Llama-3.1-8B, gpt-oss-120b and Bio-Medical-Llama-3-8B (used by BDSP for BIND) [V]. Approved-compute and EEG-blind prompting (strip EEG reports and EEG-derived text) are prerequisites.

---

## 8. Published, non-open evidence that EEG carries etiology information (for the lead's context)

- CERTA (NCT03129438; 358 adults with acute consciousness impairment): CNN predicted etiology (four categories) with accuracy 54.5%, rising to 67.7%, 70.3% and 84.1% at certainty thresholds 50%, 60% and 75%; Grad-CAM suggested learned patterns resembled expert-recognised ones. Neuroimage Clin 2022, PMID 36049354 [V].
- Korea University: CNN on functional-connectivity matrices separated NCSE, metabolic encephalopathy and benzodiazepine intoxication, AUC 0.905, maintained in a prospective cohort; delta and theta bands carried the signal. NeuroImage 2024, PMID 39033787 [V].
- Malawi (children, Blantyre): EEG features separated cerebral malaria (n=203) from nonmalarial febrile coma (n=87), AUROC 0.85 (quantitative), 0.86 (visual), 0.90 combined. Clin Infect Dis 2025, PMID 40736255 [V]. Pediatric, not adult.
- DoC pilot: connectivity and dominant-frequency EEG features discriminated traumatic from non-traumatic etiologies and predicted outcome (accuracy 80-83%) in 33 patients. Biomedicines 2022, PMID 36009445 [V].
- Caveat common to all: group-level discrimination of selected etiologies, not increment over clinical context or sedation; none open.

---

## 9. Licence and commercial-clean summary

| Source | Licence as read | Commercial-clean? |
|---|---|---|
| TUEG family (NEDC v6.0) | Research and technology development; no redistribution; delete when finished [V] | Unresolved (yellow). Written NEDC/counsel confirmation required [P]. |
| I-CARE (PhysioNet) | **CC BY-NC-SA 4.0** [V] | **No** (NC and SA). |
| BDSP open tier | Registry: ODC-By + "non-commercial research use"; pages: ODbL (VE-CAM-S), CC BY-SA 4.0 (hypothermia) [V] | Unclear; ODbL/BY-SA share-alike also complicates derivatives. |
| BDSP credentialed / restricted (HEEDB, MORGOTH data, NESI, BIND, SPaRCNet, E-CAM-S) | "Credentialed Health Data License 1.5.0" / "Restricted Health Data License 1.0.0"; registry: non-commercial research use [V] | **No** for training; evidence-only use is a separate question for counsel. |
| MORGOTH code | CC BY-NC 4.0 [V] | **No.** |
| BIND article | CC BY-NC-ND 4.0 [V] | **No.** |
| HMS Kaggle | **Not verified** | Treat as no until read. |
| OpenNeuro mild-TBI sets, ds008082, Dryad deposits | CC0 [V] | Yes, but not fit for purpose. |
| Zenodo DoC pilot, figshare DoC, NMT, acute-stroke article, VitalDB article | CC BY 4.0 [V] (dataset terms for stroke set and VitalDB not read) | Likely yes; not fit for purpose. |
| JSMF ACCESS | PhysioNet Restricted Health Data License 1.5.0 [V] | No. |

---

## 10. Thin, unverified or inferred

- TUEG keyword-label counts: **unknown**; every statement about reaching 50-200 positives per family is my guess [U]. Report precision is unmeasurable via NEDC's service. NEDC's script turnaround time and whether they will run multiple scripts are not stated [V: page silent].
- Setting mix in TUEG (ED/ICU/inpatient/outpatient): not stated on any primary page I read.
- ICD-9 coding of post-2011 TUEG sessions: search snippet only [U].
- HMS Kaggle licence and the access terms: not retrievable (Kaggle pages are not machine-readable here).
- VE-CAM-S: whether raw EEG is in the open files, channel count and sampling rate are not stated on the page; the paper says data on request [V], the page says open [V]. The registry and page licences conflict.
- BIND: EEG-to-imaging time window, the field list, access timeline and LLM-extraction accuracy not read. BDSP topic listings: only part read.
- Rubega 2020 (hypoglycaemic clamp EEG, 34 T1D patients, PMC7516516) and the UCL cirrhosis cohorts: data availability not read; only search snippets [U].
- DoC figshare set's etiology breakdown: the linked paper was not identified.
- Dryad delirium record's PMID 31467255 was matched by title only [U].
- Time-to-result figures are my estimates [U]. Power tables are my calculation with standard approximations [M].
- Absence claims are "not found in the searches listed", not proof of absence.

---

## 11. Sources

- NEDC TUH EEG page: https://isip.piconepress.com/projects/nedc/html/tuh_eeg/ ; access form v6.0 (PDF text read): https://www.isip.piconepress.com/projects/nedc/forms/tuh_eeg.pdf ; TUEG 2016: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2016.00196/full ; EEG-report annotation (LREC 2020): https://aclanthology.org/2020.lrec-1.276.pdf
- PhysioNet I-CARE v2.1: https://physionet.org/content/i-care/2.1/ ; PhysioNet EEG listings: https://physionet.org/content/?topic=eeg ; JSMF ACCESS: https://physionet.org/content/jsmf-access/1.0.0/ (doi:10.13026/m2hn-w830) ; VitalDB: https://pmc.ncbi.nlm.nih.gov/articles/PMC9178032
- BDSP: open tier https://registry.opendata.aws/bdsp_open_projects/ ; credentialed https://registry.opendata.aws/bdsp_credentialed_projects/ ; restricted https://registry.opendata.aws/bdsp_restricted_projects/ ; VE-CAM-S https://bdsp.io/content/ve-cam-s/1.0/ ; hypothermia https://bdsp.io/content/dhypothermia/1.0/ ; MORGOTH https://bdsp.io/content/morgoth1/1.0.0/ ; NESI https://bdsp.io/content/1yizvr4l41wmiljc6lou/1.0.0/ ; ICU SLEEP https://bdsp.io/content/phgb6nut6vxcqew5tiek/1.0.0/ ; post-arrest 1,000 patients https://bdsp.io/content/yzzwxkgymmwioq3pet9l/1.0.0/ ; SPaRCNet https://registry.opendata.aws/bdsp-sparcnet/ ; delirium topic https://bdsp.io/content/?topic=delirium ; BIND (Sci Data 2026) doi:10.1038/s41597-026-07421-x, PMC13462082
- VE-CAM-S paper PMC8769081 (Crit Care Explor 2022, PMID 35072078); E-CAM-S paper PMC8678335; HMS competition description arXiv 2412.07878 and https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification
- Open repositories: Dryad doi:10.5061/dryad.tv06pt2, gmsbcc2mp, h6986, 2fqz612zv ; figshare https://figshare.com/articles/dataset/23552964 ; Zenodo https://zenodo.org/records/6951440 ; OpenNeuro ds003522, ds003523, ds005114, ds008082 ; NMT https://zenodo.org/records/21629135 ; acute stroke MI Sci Data doi:10.1038/s41597-023-02787-8 (PMC10811218)
- Etiology-from-EEG literature: PMID 36049354, 39033787, 40736255, 36009445
- LLM validation: PMID 41134571, 42802831, 41388138, 40709446, 41646769, 40486463, 42278954, 41646828, 41326410
- IEEE Access scoping review: doi:10.1109/ACCESS.2024.3376254
- Project documents: `docs/research/dataset_alternatives_2026-10-09.md`, `docs/prereg_study1_sap.md` (Amendment A1), `docs/research/cbramod_provenance.md`, `docs/data_rights.md`
