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
