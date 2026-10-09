# Sorting Hat - alternative and complementary datasets and study designs

Date: 2026-10-09. Method: PubMed (E-utilities wrapper), WebSearch/WebFetch of primary pages (NEDC, BDSP, PhysioNet, ACNS, journal pages). No local restricted data was read or used; `out/` was not read. Context documents skimmed first: `docs/research_plan_v1.txt`, `docs/prereg_study1_sap.md`, `docs/data_access.md`, `docs/data_rights.md`, `docs/research/novelty_sweep.md`.

Verification tags: **[V]** the fact was read from a primary page or PubMed record in this session; **[U]** unverified (search snippet only, memory, or inferred); **[P]** project-internal statement carried over from `docs/` and not re-checked.

Fit scores are 0 (none) to 3 (strong) for (a) development, (b) external validation, (c) intended-use evaluation (adult ED undifferentiated AMS). They are my judgment, not measurements.

---

## 1. Bottom line

1. **Nothing public gives what HEEDB gives**: raw clinical EEG linked to structured EHR at scale, with commercial-clean rights. No public or credentialed corpus found combines adult ED undifferentiated AMS, EEG, and etiology labels. The honest answer is that the intended-use (ED) claim cannot be earned retrospectively from any dataset I could find; it needs a prospective study.
2. **The ED-like subgroup in HEEDB is small for a structural reason, not a sampling accident.** In ED AMS patients who get a conventional EEG, EEG is late: one academic ED reported median 6.4 h from ED presentation to EEG order and 20.9 h to EEG completion (Runcie et al., Int J Emerg Med 2026, PMID 41928084 [V]); the Careggi ED cohort reports a median 350 min from symptom onset to EEG [V]. Conventional EEG therefore selects patients after CT/labs/sedation. That is the argument for rapid-EEG sites in the prospective stage.
3. **Best external additions, in order**: (i) CERTA (already planned; ICU, 4 Swiss hospitals, 364 pts, etiology groups known; access by DUA) for external validation of the ICU/HEEDB-like claim; (ii) the **Careggi EMINENCE ED cohort** (Florence; 1,018 ED patients with emergency 19-channel EEG in 2023; ED final diagnoses; "data available on request") as the only identified adult ED undifferentiated-AMS EEG set with diagnosis categories; (iii) a rapid-EEG consortium route (Ceribell investigator-initiated program plus SAFER-EEG/DECIDE investigators, several of whom are BDSP-adjacent) as the bridge to the deployment hardware and to prospective sites.
4. **TUEG is a weak fit for this project's central question** and its commercial-clean status is weaker than the plan assumes: clinical reports are no longer distributed, there is no structured etiology, and the current NEDC terms say "research and technology development, but not for uses beyond these broad classifications" [V]. It remains useful for encoder provenance and E3 controls only.
5. **MIMIC contains no EEG** [V for the waveform module; no EEG-linked module found]. MIMIC-IV-ED (BIDMC, about 425k ED stays [V]) is useful only for ED AMS label-pipeline development and etiology priors, not for any EEG claim.

---

## 2. Candidate-by-candidate evidence

### 2.1 Credentialed or public EEG corpora

| Dataset | Size / population | Labels and EHR linkage | Access and license | Fit a / b / c | Notes |
|---|---|---|---|---|---|
| **TUEG** (Temple Univ. Hospital EEG Corpus) | 26,846 recordings 2002-2017 per NEDC page [V]; the 2016 paper described 16,986 sessions / 10,874 subjects at its release (PMC4865520 [V]). Mixed clinical population; age mean 51.6. ED/ICU/inpatient mix **not stated** on either page [V]. One secondary web description lists outpatient, ICU, EMU and ER sessions without counts [U]. | Reports are unstructured text: "patient, relevant history, medications, and clinical impression" [V]; ICD-9 codes exist for post-2011 EEGs per an LREC paper snippet [U]. **NEDC states "We no longer distribute reports with our corpora"**: a user submits a Python script and NEDC runs it on the anonymized reports and returns matching filenames [V]. No structured diagnoses, no outcomes. | NEDC form v6.0 (3 Jun 2025) [V]: free, acknowledge, **no redistribution, no re-identification, delete data when finished, "can be used for research and technology development, but not for uses beyond these broad classifications."** No explicit commercial grant and no statement on derivative weights. Credentials in 24-48 h per NEDC page [V]. | 2 / 0 / 0 | Use for encoder work and E3 controls. A "commercial candidate" flag in `configs/data_rights.yaml` is not supported by the text; get written NEDC/counsel confirmation (including delete-on-completion vs regulatory data retention). Reports as weak labels are only possible through NEDC-run scripts, and the setting (ED vs inpatient) is unknowable from public documentation. |
| TUH subcorpora (TUAB v3.0.1, TUAR v3.0.1, TUEP v3.1.0, TUEV v2.0.1, TUSZ v2.0.6, TUSL v2.0.1) | Subsets of TUEG [V]. TUEP is 100 epilepsy + 100 non-epilepsy subjects with disease and medication metadata [V]. | Event or normal/abnormal labels, not etiology. | Same NEDC terms. | 1 / 0 / 0 | Already treated as label sources and contaminated benchmarks in the plan. |
| **I-CARE** | PhysioNet v2.1 open: 607 patients, 32,712 h, 7 hospitals (hospital IDs A-F), published 14 Dec 2023 [V]. BDSP restricted v2.0 (29 Jul 2026) lists 1,020 patients, more than 50,000 h [V]. | Post-arrest only (E2). Age, sex, hospital, ROSC time, OHCA, shockable rhythm, TTM; CPC outcome (3-6 mo) [V]. No other etiologies. | PhysioNet: **CC BY-NC-SA 4.0**, open [V]. BDSP: restricted DUA [V]. | 1 / 1 / 0 | E2 stress test only (as in plan). Non-commercial license blocks any commercial use. |
| BDSP **Neurotech** v1.0 | 23,607 recordings, 4,914 patients, 212,186 h, 2021-2025; ambulatory/outpatient/prolonged monitoring from one service provider (Natus/Xltek) [V]. | ICD-10 referral diagnoses, meds, EEG findings for 4,812 pts; technician annotations [V]. No acute etiology adjudication; not ED. | BDSP Credentialed Health Data License 1.5.0 (page lists "restricted") [V]. `docs/data_access.md` recorded CC BY-NC 4.0 and "largely paediatric"; the BDSP page shows neither [P vs V conflict]. | 1 / 1 / 0 | Out-of-domain robustness at most. Re-check license. |
| BDSP SPaRCNet v1.1, GROND v1.0.0 (Aug 2026), LENS v1.1.0 (Sep 2026), ELUCID, BIND v1.0, MORGOTH 1.0 | ICU EEG pattern sets, normative slowing, imaging [V from BDSP listings]. | Pattern or imaging labels, no etiology. MORGOTH data page lists ICU but not ED [V]; code is CC BY-NC 4.0 [V]. | BDSP credentialed/restricted. | 1 / 1 / 0 | E3 control (SPaRCNet, GROND) and E1 imaging evidence (BIND). Not a path to ED. |
| **CCEMRC** database (ACNS) | ~4,700 patients analyzed in 2016; original database retired 2021; 27 centers listed [V/U]. | Structured cEEG reporting fields in REDCap (the public item is a template) [V]. Whether raw EEG is held centrally: not stated; I believe it is not [U]. | Template free by email; original data case-by-case with contributor-site approval [V]. | 0 / 0 / 0 | Useful as a recruiting network for a prospective study and for etiology-by-EEG-pattern priors, not as a signal dataset. |
| MIMIC-IV, MIMIC-IV-Waveform, MIMIC-IV-ED | MIMIC-IV-ED v2.2: ~425,000 BIDMC ED stays 2011-2019, triage/vitals/ICD/medrecon/pyxis [V]. Waveform DB: ECG, PPG, respiration, blood pressure; **no EEG listed** [V]. | MIMIC-IV-ED links to MIMIC-IV hospital data by subject_id/hadm_id [V]. **No MIMIC-EEG module or HEEDB linkage found** (PubMed and web searches empty) [U for absence]. | PhysioNet Credentialed Health Data License 1.5.0, CITI training [V]. PhysioNet bars sending credentialed data to third-party APIs unless provider terms fit; it lists Claude as no-training-by-default [V], but this repo's rule 1 and 3 still apply: treat as restricted, no row-level data in agent context. | 1 / 0 / 1 (label pipeline only) | Could estimate etiology prior and test the silver-label extraction on ED AMS without EEG. BIDMC overlaps a HEEDB site, so any cross-linking would need BDSP/PhysioNet governance; do not attempt. |
| CHB-MIT (22 pediatric epilepsy subjects, ODC-By) [V], Siena (14 adult epilepsy patients, CC BY 4.0) [V], Bonn, NMT, TDBRAIN, Healthy Brain Network, Dortmund | Seizure or research cohorts. | None relevant to acute etiology. | Open. | 0 / 0 / 0 | **Irrelevant** to this question (state it and move on). Bonn, NMT, TDBRAIN: not re-verified, background knowledge [U]. |
| Public disorders-of-consciousness sets (figshare 23552964: 32 healthy + 59 prolonged DoC; Bath BATH-01632: UWS 14 / MCS 17 / LIS 11) | Chronic DoC, resting or BCI tasks. | Behavioral diagnosis, etiology not confirmed. | Open or request-only. | 0 / 0 / 0 | Wrong phase (chronic, not acute). |
| Harvard-Emory ECG database | ECG only (medRxiv 2024.09.27.24314503) [V from search listing]. | n/a | n/a | 0 / 0 / 0 | Not EEG. |
| Europe/Asia hospital EEG sets with diagnoses | I found **no** open adult ICU/ED etiology-labeled EEG corpus outside those listed (CAUEEG: Korean normal/MCI/dementia, unverified license [U]; NEUROSKY-EPI: single-channel consumer EEG, size and link unstated [V]). | n/a | n/a | 0 / 0 / 0 | Absence from about 8 searches; not proof. A China/Netherlands-specific search was not exhausted. |

### 2.2 Research cohorts holding EEG plus etiology (negotiated access)

| Cohort | Size / population | Labels | Access route | Fit a / b / c | Notes |
|---|---|---|---|---|---|
| **CERTA** (NCT03129438; Rossetti, Zubler, Schindler et al.) | 364 adults, GCS <=11 or FOUR <=12, 4 Swiss tertiary hospitals, ICU and ward [P from novelty sweep; PMIDs 32716479, 36049354, 40718992, 42202614 listed there]. Etiology groups published: HIBI 112, ICH 85, ischemic stroke 28, TBI 48, toxic-metabolic 23, encephalitis 7, unknown/other 114. 67.6% received sedatives (Guinchard 2026). | Etiology plus 6-month outcome; sedation dosing documented in the trial (PMID 42202614). "14 etiologies" detail **not confirmed**. | No public release found. DUA with investigators (contacts in `novelty_sweep.md`). | 1 / 3 / 1 | Strongest true external check of the ICU claim for E1/E2/E5-like groups; cannot test E6/E7. Conventional 20-min routine or cEEG, not rapid, not ED. |
| **Careggi EMINENCE** (Florence) | 1,018 ED patients with emergency EEG in 2023 (J Med Syst 2026, PMID 42062615 [V]); 579 patients / 603 EEGs in the Diagnostics secondary analysis (PMID 40218213 [V]); 208/579 (36%) acute confusional state, rest hemispheric transient deficits [V]. 19-electrode 10-20 cap, 128 Hz, ~30 min, usually unsedated [V]. Median onset-to-EEG 350 min [V]. | ED-discharge final diagnosis by emergency physicians with neurologist input, **no outpatient follow-up**; categories: seizures 217, vascular 64, migraine 12, encephalopathies 78, other 120, unknown 88 [V]. Brain CT in 96%; some lab/CT fields recorded. Multicentre extension **EMINENCE-M**: protocol in BMJ Open 2026, planned 3,850 ED patients, retrospective, Italian centres [V per protocol listing]. | "Available on request from the corresponding author" (A. Grippo, antonello.grippo@unifi.it) [V]. Raw format (EDF or vendor) not stated [V]. No commercial terms exist; Italian/EU GDPR and a data-transfer agreement apply [U]. | 1 / 2 / 2 | Best identified adult ED AMS set. Weaknesses: label is an ED-discharge impression made with the EEG in hand (circularity risk, and exactly the leakage the plan bans); seizure-heavy (37%); few toxic/metabolic/infectious positives; no sedation records mentioned; 128 Hz; no OMOP. Usable as a locked intended-use check on E1/E3/E5-like groups and as a test of whether ED-time EEG shifts the posterior, if labels are re-adjudicated EEG-blind. |
| **Korea Univ. Anam** cohort (Kim JB et al., NeuroImage 2024, PMID 39033787) | N not in abstract [U]; NCSE vs metabolic vs benzodiazepine 3-way. | Single-label. | Email corresponding author (details in `novelty_sweep.md`). | 0 / 2 / 0 | Replication of the E3/E5/E4a branch only. |
| **Ziai / Hopkins ED EEG** (Clin Neurophysiol 2011, PMID 21978652 [V]) | 82 adult ED patients, standard EEG within 30 min of referral, 1 day/week. | Utility survey, cause categories (toxicologic, psychiatric, endocrine/metabolic associated with utility). | NIH-funded; raw EEG holder not identified [U]. | 0 / 1 / 2 | Old and small; shows an ED-time EEG protocol is feasible. |
| **Downstate microEEG** (Zehtabchi; Eur J Emerg Med 2013 PMID 22644284 [V]: first 50 pts; Acad Emerg Med 2014 PMID 24628753 [V]: RCT, 149 pts, two urban EDs, 30-min microEEG at presentation) | Adults with AMS; EEG abnormal 93%, NCS 5%. | Management-change outcomes; diagnoses likely recorded but not verified [U]. | NIH grant 1RC3NS070658 to Bio-Signal Group with SUNY subcontract [per search snippet, U]; device company status unknown. | 0 / 1 / 2 | Closest historical analog of the exact intended-use population. Data holder and format unknown; worth one email to the PI/co-authors (S. Zehtabchi, A. Grant). |
| **UCLA limited-EEG ED cohort** (Richard et al., Ann Emerg Med 2024, PMID 38888533 [V]) | 132 adults with unexplained mental status change; 108 interpretable limited EEGs; NCSE prevalence 2.9% in the cohort. | Epileptologist reads and AI read; clinical data collected. | Not stated. | 0 / 1 / 2 | Tiny for etiology. |
| **SAFER-EEG** (Kalkach-Aparicio, Neurology 2024, PMID 38875512 [V]; Parvizi, Crit Care Med 2026, PMID 42223304 [V]) | Retrospective, 4 tertiary centers: 240 rapid-response EEG with follow-on cEEG (2018-2022); secondary cohort 400 adults (359 with mRS) at 3 centers. Authors include Westover, Zafar, Hirsch, Struck [V]. | Seizure risk and discharge mRS; not etiology (chart etiology may exist [U]). | Investigator route; Ceribell data used. | 0 / 2 / 1 | Natural bridge: BDSP-adjacent authors, rapid-EEG hardware, adult acute care. |
| **DECIDE** (Vespa, Crit Care Med 2020, PMID 32618687 [V]) | 181 ICU patients with suspected nonconvulsive seizures, 5 US academic hospitals incl. MGH; 164 complete [V]. | Physician pre/post assessments; seizure focus. | Ceribell-sponsored. | 0 / 1 / 1 | ICU, seizure-focused. |
| PHIRE prehospital (Guterman, JACEP Open 2024, PMID 39281726 [V]) | 34 EMS encounters, mean age 69, 10 electrodes, median 10.5 min recording, 94% interpretable [V]. | Feasibility only. | UCSF. | 0 / 0 / 1 | Shows application time (2.5 min) and quality are achievable out of hospital. |
| Pediatric ED point-of-care EEG sets (Zurich, Kobe, Tokyo, Bologna) | 20-242 children [V]. | NCSE-centric. | Various. | 0 / 0 / 0 | Wrong population (adults required). |

### 2.3 Rapid-EEG vendors as data holders

- **Ceribell**: FDA-cleared seizure/ESE detection, delirium monitor K251936, and (reported Aug 2026) epileptiform abnormality detection [per `novelty_sweep.md`; the Aug 2026 item has a single secondary source in my searches]. Company says 600+ hospitals and 135+ publications [V from search result]. I found **no public dataset and no data-sharing program**; the only formal route is the **Investigator-Initiated Studies Program** (ceribell.com/?p=19469 [U: page title seen in search results only]). Whether Ceribell holds etiology labels: not known. Realistic role: hardware and site partner for prospective work, not a label source.
- **Zeto**: 21-electrode dry-electrode headset; ONE cleared 2024, New Wave April 2026 [V from press snippets]. Runs a clinical-trial sponsorship program (critical care, stroke, concussion) [V]. No peer-reviewed ED AMS study found [V].
- **BrainScope** (concussion/structural-injury, De Novo history): no clearance details retrieved this session [U].

### 2.4 BDSP-partner data (Emory, BCH, other sites)

No evidence found that Emory ICU EEG, an ED set, or any rapid-EEG set is hosted in BDSP. BDSP's EEG listings show HEEDB v4.1 (284,343 EEGs, 109,178 patients, 4 sites; settings listed: routine, EMU, ICU/LTM; **ED not mentioned**) [V], I-CARE, SPaRCNet, GROND, LENS, ELUCID, BIND, plus the Neurotech set [V]. HEEDB sites are MGH, BWH, BIDMC, BCH (site-ID mapping not given on the page) [V]. Emory appears only as an author affiliation on a BDSP platform abstract [V], not as a data source. Treat any Emory/BCH data claim as unverified until BDSP confirms.

One cheap internal action: the project excluded I0008/I0009 because they have no OMOP/EHR rows (D-113). Ask BDSP whether those sites' structured EHR is planned; that would add sites inside the existing DUA, with no new legal work.

---

## 3. Licensing and the commercial-clean path

| Artifact | What the primary text says | Consequence for `data_rights` ledger |
|---|---|---|
| HEEDB | bdsp.io: "BDSP Credentialed Health Data License 1.5.0"; AWS Registry: "BDSP Restricted Health Data License 1.0.0"; neither page reproduces license text or states commercial terms [V]. | Two license names for one dataset; get the actual text. Keep red. |
| TUEG family | NEDC v6.0: research and technology development; nothing beyond; no redistribution; delete when finished [V]. | Do not mark green for production or regulatory use without written NEDC confirmation. "Delete when finished" is a flag for regulatory data lineage. |
| I-CARE (PhysioNet) | CC BY-NC-SA 4.0 [V]. | Red for commercial. |
| MIMIC family | PhysioNet Credentialed Health Data License 1.5.0 [V]. | Research only. |
| MORGOTH code | CC BY-NC 4.0 [V]. | Red for commercial. |
| CERTA, Careggi, Korea Univ., Downstate | No standing terms; negotiable. | Only sources where a commercial-training grant could be negotiated, at a price. |
| Ceribell, Zeto | Commercial partners; terms unknown. | Strategic route (plan route 2). |

---

## 4. Study designs considered

| Design | What it buys | What it cannot do | Cost / time (my estimate, [U]) |
|---|---|---|---|
| **A. Stay on HEEDB, add the earliest-EEG subgroup** (plan H5, undifferentiated subgroup) | Largest, cheapest, already approved; supports G2/G3 on ICU-weighted populations. | The ED-like subgroup stays under 100 (consistent with Runcie's 6.4 h to order); LOSO has 4 sites at best. | Zero marginal. |
| **B. Frozen-model, code-to-data external evaluation** (ship the frozen model plus the repo's aggregate-only scoring script to CERTA/Careggi investigators; get back only suppressed aggregates) | Avoids moving raw EEG across borders and matches this repo's aggregate-only rules; faster legal path than a data transfer; works with GDPR constraints. | Needs a clean frozen model and a shared preprocessing contract (128 Hz Careggi vs 200 Hz model input; channel naming). | 2-4 months per site for DUA or collaboration agreement plus engineering. |
| **C. Retrospective single-ED chart review with EEG-blind re-adjudication** (Careggi-style, or an ED with rapid EEG already standard) | Gives ED-time EEG and an etiology reference standard; can be done under waiver of consent. | Still biased to patients clinicians chose to scan; ED-discharge labels are shaky; only one site. | 6-12 months incl. IRB and adjudication labor. |
| **D. Multi-site via BDSP partners** | Would give more EHR-linked sites. | No evidence such sites exist today. | Unknown. |
| **E. Prospective ED AMS cohort with protocolized rapid EEG** | The only design that answers the intended-use question. | Slowest and costliest; needs rapid-EEG hardware, adjudicators, funding. | 2-3 years (plan P1-P2). |

---

## 5. Ranked recommendation

### 5.1 What to keep using HEEDB for
1. Development and site-held-out evaluation for Study 1 (H1-H4): ICU/inpatient-weighted signal discovery, sedation-aware baselines, the commercial-clean gap, montage and duration simulation. This is the only dataset with the volume and EHR linkage for it.
2. The early-EEG and undifferentiated subgroups as **hypothesis-generating** analyses (H5, section 4.4 of the SAP), reported with explicit caveats, never as the intended-use claim.
3. Request BDSP structured EHR for I0008/I0009 (internal, cheap).

### 5.2 External datasets to pursue first (in order)
1. **CERTA**: already in the plan; do the outreach now. Best external test of E1/E2/E5-like transport, and it has sedation dosing.
2. **Careggi EMINENCE / EMINENCE-M**: one email to A. Grippo proposing design B (frozen model, aggregate return) plus an EEG-blind re-adjudication of a subset. It is the only identified adult ED AMS EEG set with diagnosis categories. Ask for: raw file format, sampling rate, medication/sedation fields, and whether discharge labels were assigned before or after the EEG was read.
3. **Rapid-EEG consortium route**: approach SAFER-EEG/DECIDE investigators already linked to the BDSP group (Westover, Zafar, Hirsch, Struck) and Ceribell's Investigator-Initiated Studies Program. Purpose: rapid-hardware transfer data and identification of prospective sites, not labels.
4. Low-cost opportunistic: email the Downstate/Zehtabchi group about the microEEG AMS data; Korea Univ. for the E3/E5/E4a replication.

Do **not** spend effort on: CHB-MIT, Siena, Bonn, NMT, TDBRAIN, public DoC sets, CCEMRC (as signal), MIMIC (as EEG source). TUEG stays at its present role (encoder provenance, E3 control) pending written NEDC clarification.

### 5.3 What prospective study is ultimately required
For a journal-grade or regulatory-grade ED claim, retrospective data cannot suffice because (i) conventional-EEG ED cohorts are selected and late, (ii) reduced-montage hardware needs real recordings (the plan already says this), and (iii) the reference standard must be independent of the EEG.

Minimum design (consistent with plan P1-P2; numbers beyond the plan are my arithmetic):
- **Population**: consecutive adults with undifferentiated AMS or unexplained decreased consciousness in the ED, at 5-10 sites where rapid EEG is already standard care; EEG within 60 min of enrollment, at least 10 min recorded; t0 clinician probabilistic differential captured before the model output is shown (silent).
- **Reference standard**: blinded multi-reader adjudication of etiology using ED and hospital data plus 30-day follow-up, **EEG-blind** (no EEG report, no model output), with explicit "unassessable" states, as in the SAP.
- **Primary endpoint**: change in masked multi-label log loss versus the baseline that includes clinician t0 differential and sedation, evaluated at sites never used in training; calibration slope 0.8-1.2.
- **Size**: 1,500-3,000 (plan). Arithmetic: 100 positives for a label of prevalence 5% needs about 2,000 patients, at 3% about 3,300, at 10% about 1,000. E7 and some E6 sub-labels will stay exploratory unless enrichment is allowed (which breaks calibration claims).
- **Paired substudy**: rapid vs conventional EEG in a subset to quantify hardware loss.
- **Consent**: rapid EEG as standard care with data capture only is the best route to an IRB waiver of consent; confirm with each IRB. A research-only EEG would need surrogate consent or an exception-from-informed-consent pathway, with associated selection bias. The SIREN network (16 hubs per a hub page, funding status in 2026 not stated) is a possible infrastructure for emergency-care trials, but this is [U] for neuro-EEG use.
- **Reporting/regulatory**: STARD-AI and TRIPOD+AI; FDA Pre-Sub first; De Novo expected (plan).

Fit summary of the four-tier stack for the final claim: HEEDB (develop) -> CERTA (external ICU) -> Careggi or equivalent (retrospective ED proxy) -> prospective P2 at rapid-EEG sites (the claim).

---

## 6. What is thin, unverified or inferred

- ED, ICU and inpatient proportions in TUEG: **not found** in any primary page. Do not assume ED content.
- Whether TUEG reports contain a labelled "clinical history" and "medications" section: the 2016 paper says the reports describe "patient, relevant history, medications, and clinical impression" [V]; exact section headers not confirmed. The ACNS guideline ordering (history then medications) came via an LREC snippet [U].
- BDSP commercial-use terms: neither license text was retrievable; the two pages cite different license names/versions.
- Whether Careggi will share raw EEG, in which format, and whether the final-diagnosis labels were assigned with the EEG known: **unknown**; the Diagnostics paper says diagnoses were made at ED discharge from "clinical features and instrumental data" [V], which plausibly includes EEG.
- Which center Runcie et al. (PMID 41928084) studied: the first author is at Michigan, a co-author at BWH; the center is not stated in the abstract [V]. If it is BWH, the cohort overlaps HEEDB and is not external.
- Ceribell Investigator-Initiated Program terms, Zeto trial-sponsorship terms: not read beyond titles/snippets.
- EMINENCE-M center count and data-sharing posture: only the protocol listing was read; I did not open the BMJ Open full text.
- Cost/time figures in section 4 are my estimates, not sourced.
- Search coverage: PubMed query translation was noisy; absence of Asian or Dutch ICU/ED etiology-labelled EEG sets is "not found in a few searches", not evidence of absence.
- No MIMIC-EEG linkage found; absence is inferred from empty searches.

## 7. Sources (primary pages and records used)

- NEDC TUH EEG page: https://isip.piconepress.com/projects/nedc/html/tuh_eeg/ ; access form v6.0: https://www.isip.piconepress.com/projects/nedc/forms/tuh_eeg.pdf ; Obeid and Picone 2016: https://pmc.ncbi.nlm.nih.gov/articles/PMC4865520
- BDSP: https://bdsp.io/ ; HEEDB v4.1 https://bdsp.io/content/harvard-eeg-db/4.1/ ; topic list https://bdsp.io/content/?topic=eeg ; Neurotech https://bdsp.io/content/nf89816gtxbon11kbr9a/1.0/ ; MORGOTH https://bdsp.io/content/morgoth1/1.0.0/ ; AWS registry https://registry.opendata.aws/bdsp-harvard-eeg
- I-CARE v2.1: https://physionet.org/content/i-care/2.1/
- MIMIC-IV-ED v2.2: https://physionet.org/content/mimic-iv-ed/2.2/ ; MIMIC-IV-Waveform: https://physionet.org/content/mimic4wdb/0.1.0/ ; PhysioNet LLM-use note: https://physionet.org/news/post/gpt-responsible-use
- CHB-MIT: https://physionet.org/content/chbmit/1.0.0/ ; Siena: https://physionet.org/content/siena-scalp-eeg/1.0.0/
- CCEMRC public database: https://www.acns.org/research/critical-care-eeg-monitoring-research-consortium-ccemrc/ccemrc-public-database
- EMINENCE: PMID 40218213 (https://pmc.ncbi.nlm.nih.gov/articles/PMC11989146/); PMID 42062615 (doi 10.1007/s10916-026-02397-y); protocol https://www.citedrive.com/en/discovery/diagnostic-yield-of-electroencephalographyin-the-emergency-department-protocol-for-the-eminence-m-multicentre-retrospective-observational-study/ (doi 10.1136/bmjopen-2026-116956)
- ED EEG studies: PMID 41928084 (Runcie 2026, doi 10.1186/s12245-026-01200-6); PMID 21978652 (Ziai 2011); PMID 22644284 (Zehtabchi 2013); PMID 24628753 (Zehtabchi 2014 RCT); PMID 38888533 (Richard 2024); PMID 39281726 (Guterman PHIRE 2024)
- Rapid EEG: PMID 32618687 (DECIDE); PMID 38875512 (SAFER-EEG); PMID 42223304 (SAFER-EEG secondary); PMID 41773897 (Gururangan review)
- Vendors: Ceribell investigator program https://ceribell.com/?p=19469 ; Zeto trial sponsorship https://www.newswise.com/articles/zeto-selects-clinical-trial-sponsorship-recipients-advancing-eeg-in-critical-care-stroke-and-concussion-research
- Prior project documents: `docs/research/novelty_sweep.md` (CERTA, Korea, CLEF, Ceribell clearances), `docs/data_access.md`, `docs/data_rights.md`
