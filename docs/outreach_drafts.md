# Sorting Hat: outreach email drafts (Phase 0)

Drafted 2026-10-07. **Not sent.** Sources: `docs/research/novelty_sweep.md`, `docs/research_plan_v1.txt` (lines 1-14, 100-107, 148-149, 224-247, 533-543).

## Notes before sending

- **Verify every address on the institutional page before sending.** Addresses below appear only where the novelty sweep lists one, and each is cited to its paper. Where the sweep lists none, the field is a placeholder.
- **Do not send #1-#3 until #4 is answered.** Plan lines 106-107 and 149 require IP ownership to be settled with the current employer and incoming residency institution before any entity, disclosure, or data license. Any BDSP-affiliated collaborator's institutional IP terms must be agreed in writing before work starts.
- **Handoff §23.3 outreach package:** not available when drafting. Attach or merge it into #2 before sending.
- **CERTA "14 etiologies" is unverified.** The sweep found published CERTA etiology groups (Benghanem 2025: seven groups; Jonas 2022: four categories) but no 14-category source. #2 asks the investigators directly.
- **CERTA affiliations:** the sweep places Zubler at the Sleep-Wake-Epilepsy-Center, Inselspital (Bern), not CHUV. Novy is at CHUV.
- **Placeholders:** [NAME], [TITLE/INSTITUTION], [SIGNATURE]. Also [INCOMING INSTITUTION] in #4; the sources do not name it.
- **Scope:** the emails describe a research-only protocol. They do not mention the commercial track or the prior-art analysis.

---

## 1. BDSP group (Westover / Zafar, MGH): collaboration offer for Study 1

**To:** [Westover / Zafar address: not listed in the sweep; verify on MGH or BDSP page]
**Subject:** Collaboration inquiry: EEG and the cause of acute impaired consciousness in HEEDB

Dear Dr. Westover and Dr. Zafar,

I am [NAME], [TITLE/INSTITUTION]. I am planning a retrospective study asking whether the first 10 minutes of scalp EEG add calibrated, multi-label information about the cause of acute impaired consciousness, beyond depth of unconsciousness and sedative exposure at that time. The intended data source is HEEDB, under the appropriate research agreement.

We would bring a written protocol, a preregistration draft with dated decisions, a site-held-out analysis plan, and a field-level audit of HEEDB timestamps.

We would value your experience with HEEDB's fields and known limitations, access to adjudicators for label review, and an EEG co-investigator for reading and quality control.

Authorship and IP terms would be agreed in writing before any work begins. No results exist yet.

Would you have 30 minutes for a call in the coming weeks?

Sincerely,
[NAME]
[SIGNATURE]

---

## 2. CERTA investigators (Rossetti, with Zubler and Novy): frozen-model external validation

**To:** Andrea O. Rossetti [address: not listed in the sweep; verify]
**Cc:** Frederic Zubler (frederic.zubler@gmail.com, per Jonas et al., NeuroImage Clin 2022, PubMed record); Jan Novy (jan.novy@chuv.ch, per Guinchard et al., Clin Neurophysiol 2026, PMID 42202614)
**Subject:** Request for external validation of a frozen EEG etiology model on CERTA

[Attach or merge handoff §23.3 outreach package here before sending.]

Dear Prof. Rossetti, Dr. Zubler, and Dr. Novy,

I am [NAME], [TITLE/INSTITUTION]. I am developing a model that estimates probabilities across several etiologies of acute impaired consciousness from the first 10 minutes of EEG, with severity and sedation as covariates. CERTA's EEG and adjudicated etiology labels would be the strongest external test available to us.

We propose a frozen-model validation. The model is fixed before any CERTA data are seen. We would send the model and run script. Your team would run them locally and return only aggregate metrics: log loss, calibration, and per-class discrimination, with your small-cell suppression rules applied. No CERTA data would be used for training, and the data would remain under your governance.

[Unverified: the "14 etiologies" detail. Published CERTA analyses report four categories (Jonas 2022) and seven groups (Benghanem 2025). Please tell us which etiology granularity exists in the labeled data.]

We are glad to discuss data-use terms and to share the protocol before any analysis.

Sincerely,
[NAME]
[SIGNATURE]

---

## 3. Korean cohort corresponding author (Jung Bin Kim): replication and external test

**To:** Jung Bin Kim, MD: kjbin80@korea.ac.kr (per Kim YT et al., NeuroImage 2024;297:120749, PubMed record; verify on Korea University Anam Hospital page)
**Subject:** Request for external test of your NCSE / metabolic / benzodiazepine connectivity model

Dear Dr. Kim,

I read your 2024 NeuroImage paper on differentiating NCSE, metabolic encephalopathy, and benzodiazepine intoxication from functional connectivity. I am [NAME], [TITLE/INSTITUTION], working on a related question: whether the first 10 minutes of EEG add calibrated information about the cause of acute impaired consciousness beyond severity and sedative exposure.

I would like to ask whether your group would run a frozen model on your cohort as an external test. We would send the model and run script. Your team would run them locally and return only aggregate per-class AUROC and calibration. No data would be used for training, and the data would remain with your group.

Could you tell us the cohort size, class balance, and per-class AUROCs behind the 0.905 result, and whether the prospective cohort can be shared in any form? The abstract does not give these, and they would tell us whether a replication is feasible.

We are glad to share our protocol first.

Sincerely,
[NAME]
[SIGNATURE]

---

## 4. Incoming residency institution, tech-transfer office: IP question

**To:** [INCOMING INSTITUTION] Office of Technology Transfer: [address: not in sources; verify]
**Subject:** Question on IP ownership for personal-time research using licensed data

Dear [NAME / Technology Transfer Office],

I am [NAME], [TITLE/INSTITUTION], and I will begin residency at [INCOMING INSTITUTION] in [MONTH YEAR]. One-line description of my work: a retrospective EEG analysis done on personal time, using a research-only data license from another institution.

My question: under [INCOMING INSTITUTION]'s policies, who owns intellectual property created from work done on personal time with externally licensed, research-only data, and what disclosure is expected before forming an entity?

Thank you for your guidance.

Sincerely,
[NAME]
[SIGNATURE]

---

## Added 2026-10-14 (drafts 5 and 6). **Not sent.**

Sources: `docs/research/dataset_alternatives_2026-10-09.md` (sections 2.2, 4, 5.2), `docs/prereg_study1_sap.md` Amendment A1 (section A1.5), D-156. PubMed records re-checked on 2026-10-14 (PMID 42062615 and PMID 41928084).

Notes for these two drafts:

- **Addresses.** Only addresses printed in the PubMed record are given. Any other address is marked "address to confirm". The one in #5 appears as the "Electronic address" of the corresponding-author line for Marra E on PMID 42062615. The Grippo address in `dataset_alternatives_2026-10-09.md` was not in the PubMed record I read, so it is marked to confirm.
- **Claims.** The drafts say only that exploratory silver-label analyses suggest a modest signal. No restricted-derived numbers are given. Do not add any.
- **Timing.** Research outreach is no longer gated on the tech-transfer question (D-147). Amendment A1 is filed before any external data are touched; the external success criteria (amendment A3) are agreed with the collaborators before any run, so the emails promise that and nothing more.
- **Placeholders:** [Name], [Title], and the signature line stay as written until the lead fills them.
- **Runcie et al. center.** The PubMed abstract does not name the study center (the first author is at Michigan; co-authors are at BWH, MGH and BIDMC). Draft #6 asks rather than assumes.

---

## 5. Careggi EMINENCE group (Florence): external validation, model-to-data

**To:** Ester Marra, IRCCS Fondazione Don Carlo Gnocchi, Florence: emarra@dongnocchi.it (corresponding-author address on Scarpino et al., J Med Syst 2026, PubMed record)
**Cc:** Antonello Grippo, Neurophysiopathology Unit, Careggi University Hospital [address to confirm]; Maenia Scarpino [address to confirm]
**Subject:** Collaboration proposal: external validation of an EEG etiology model on the EMINENCE emergency EEG cohort

Dear Dr. Marra, Dr. Grippo and Dr. Scarpino,

I am [Name], [Title], Boston University. I read your EMINENCE papers on emergency EEG in the ED with interest (Scarpino et al., J Med Syst 2026, doi:10.1007/s10916-026-02397-y, and the Diagnostics 2025 analysis of transient deficits and acute confusional state). Your cohort is, as far as I can tell, one of very few adult ED cohorts that combine EEG recorded early in the presentation with diagnostic categories.

I am developing a model that estimates probabilities across etiologic families of acute impaired consciousness (structural, hypoxic-ischemic, metabolic, among others) from the first minutes of scalp EEG, and I am testing whether it adds to what is known at presentation. It is built and tested on a large US retrospective EEG database. Exploratory analyses on structured proxy labels suggest a modest signal for some etiologies, which is why an independent test matters. Our data come mostly from ICU and inpatient monitoring, so we cannot say anything about the ED from them alone. The registered analysis plan will say so explicitly.

We would like to propose the following, and we are glad to adapt it to your governance:

1. **Model-to-data validation.** We freeze the model before we see any of your data, and we send you the frozen model and a scoring script. Your team runs them locally and returns only aggregate metrics (log loss compared with simple baselines, calibration, per-class discrimination), with your own small-cell rules applied. No recordings, rows, identifiers or dates would leave Careggi, and none of your data would be used for training or tuning.
2. **EEG-blind re-adjudication of a subset.** Because the discharge diagnosis in the cohort was assigned with the EEG available, we would prefer to measure against a reference standard made without it. We suggest that clinicians on your side (with ours, if you wish) re-adjudicate a subset of patients using only the clinical and laboratory data, blind to the EEG and its report, using a short label definition we would supply. The size and sampling of the subset would be yours to set.
3. **Success criteria agreed beforehand.** The analysis plan is preregistered. We would agree the mapping of your categories to our labels and the success criteria with you, in writing, before any scoring.
4. **Co-authorship.** We would offer co-authorship of the validation paper to the investigators who contribute, and would share the protocol and the draft with you before submission.

A few questions would help us judge feasibility: in what format and at what sampling rate are the raw recordings stored, and which channels are available; are sedative and other medication fields recorded; and was the final diagnosis assigned before or after the EEG was read? Would a short call be possible in the coming weeks?

Thank you for considering this.

Sincerely,
[Name], [Title], Boston University

---

## 6. BDSP team / Runcie et al.: ED altered-mental-status cohort, ED flags, and structured EHR for I0008/I0009

**To:** Mariama Runcie, Department of Emergency Medicine, University of Michigan: runciema@med.umich.edu (corresponding-author address on Runcie et al., Int J Emerg Med 2026, PubMed record)
**Cc:** M. Brandon Westover, BDSP lead [address to confirm]; BDSP data-access contact [address to confirm]
**Subject:** HEEDB question: ED altered-mental-status EEG cohort, ED encounter fields, and EHR for sites I0008 and I0009

Dear Dr. Runcie and Dr. Westover,

I am [Name], [Title], Boston University, working under a HEEDB data-use agreement on a retrospective study of whether the first minutes of EEG add calibrated information about the cause of acute impaired consciousness. I read with interest Runcie et al., "EEG in patients with altered mental status in the emergency department" (Int J Emerg Med 2026, doi:10.1186/s12245-026-01200-6). We have exactly the problem the paper describes: EEGs in HEEDB are mostly recorded hours to days into an admission, and exploratory analyses on structured proxy labels suggest only a modest signal, so we want to know how much of our population resembles an ED presentation.

We would be grateful for your help with three questions:

1. **Is the ED altered-mental-status cohort identifiable within HEEDB?** The paper's study center is not stated in the abstract, and co-authors are affiliated with several HEEDB sites. If its EEGs are in HEEDB, is there a way for us to identify them (a cohort list or a flag, provided under whatever governance BDSP requires), or is the cohort outside HEEDB? We are not asking for chart data, only whether and how it could be linked.
2. **ED encounter fields.** In the tables available to us we could not find a reliable indicator of ED care (the visit class code is not populated, and we have no care-site or admitted-from field). Could ED encounter flags, admitted-from or care-site fields, or ED triage or flowsheet concepts be made available to approved projects, or are they held elsewhere?
3. **Sites I0008 and I0009.** These two sites have EEG data but no structured EHR rows in the release we have, so we cannot use them for labelled analyses. Are structured EHR data for these sites planned, and if so on what time frame?

If the answer to any of these is "not available", that is useful to know and we will describe the limitation plainly. We would also be glad to share our protocol and preregistration with you. In particular, we will make no claim about ED triage from HEEDB alone.

Thank you for your time.

Sincerely,
[Name], [Title], Boston University
