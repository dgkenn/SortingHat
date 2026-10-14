# BIND access: reachable with existing credentials; terms and registration note

Date: 2026-10-14. Replaces the planned access request (the project lead expected BIND to be reachable already; the probe confirms it).
Analysis plan: `docs/bind_e1_analysis_plan.md` (D-158). Contains no secrets and no data values.

## 1. What was found (names only)

Probe run via `env -u CLAUDECODE -u SORTINGHAT_AGENT_SESSION scripts/heedb_run.sh` with the project's BDSP key pair; delimiter listings of prefix names and, for six site-level CSV files, the header line only (column names). No row, count, date, ID or free text was read; per-patient folders (`sub-*`) were not entered or printed.

| Access point | Result |
|---|---|
| `bdsp-credentialed-access-point` | **BIND is here.** `Imaging/I0001/` and `Imaging/I0004/`, each with `BIDS/`, `Clinical/`, `Non-BIDS/`. `Clinical/` holds `{site}_demographics.csv`, `{site}_imaging_findings_De-id.csv`, `{site}_reports.csv`. `BIDS/` holds `README`, `dataset_description.json` and per-patient folders. A ranged GET of the first bytes of each `Clinical/` CSV succeeded (read access, not only listing). |
| `bdsp-credentialed-projects-ap` | 39 project prefixes, none named BIND, imaging or the BIND slug `n1vba1x5qt62frfjem65`; `bind/`, `BIND/`, `imaging/`, `Imaging/` are empty. |
| `bdsp-restricted-access-point` | 19 prefixes (ICARE_train, sah, e-cam-s, sparcnet_data, ...), none BIND or imaging; `bind/`, `BIND/`, `Imaging/` are empty. |

Site mapping (from the paper): I0001 = Mass General Brigham, patients with imaging and an EEG; I0004 = Stanford, patients with imaging and polysomnography (no EEG, outside the EEG cohort). Real column names are in `docs/heedb_schema_real.md` B4 and `configs/bind_e1.yaml`.

## 2. Terms (from the BDSP BIND page and the paper)

| Item | Value |
|---|---|
| Dataset | The Brain Imaging and Neurophysiology Database (BIND) v1.0, published 2025-09-09, doi:10.60508/mby8-3a26, https://bdsp.io/content/n1vba1x5qt62frfjem65/1.0/ |
| Access tier | Database Credentialed Access: credentialed user, signed credentialed DUA, CITI "Data or Specimens Only Research", AWS account ID registered in the BDSP profile |
| Licence | BDSP Credentialed Health Data License 1.5.0 (text linked on the page; not read). The page does not address commercial use. |
| DUA terms (as summarised by the paper) | No re-identification, no further sharing |
| Commercial use | The paper: "Commercial entities may access the dataset through a separate institutional agreement with the participating hospitals." Article licence CC BY-NC-ND 4.0. |
| Imaging and the credentialed DUA | https://bdsp.io/about/howto_accessdata lists Imaging among the large repositories that need only the credentialed DUA; project-specific credentialed datasets may need a separate DUA review. |
| Contact | No access e-mail address is published on bdsp.io. The support page (https://bdsp.io/support/) has a contact form ("Data access" or "Credentialing"; "Other" for licensing). The only e-mail on bdsp.io/about is the PI's (mbwest@stanford.edu), not an access desk. `contact@bdsp.io` appears in older project notes and is **not verified**. |

## 3. Status and what is still open

* **Technically reachable now.** Whether the project's existing DUA scope and approved project description cover BIND is **not confirmed in writing**. `configs/data_rights.yaml` entry `bind_v1_0` therefore keeps `research: unknown`; do not change it to `permitted` until BDSP confirms (suggested update to that entry, not made here: access "credentialed, reachable 2026-10-14", licence name and version above, open question "written confirmation that the project DUA covers BIND; commercial terms").
* **Use rule while this is open:** names-only probes only. No label has been computed or counted (D-158); the first real-data step (plan step 0a) should wait for the lead's go-ahead and, ideally, for the confirmation below.
* **Commercial track:** BIND-derived labels are research evidence only. The commercial-clean path will not use them to train or tune a product. Ledger status stays red for commercial purposes.
* **If the BIND page has a "Request Access" button for a signed-in account, a human presses it** (registers the project against the dataset; agents do not act on the BDSP site).

## 4. Text for the BDSP Request Access / support form

Project title: Imaging-anchored test of structural-injury information in EEG representations (HEEDB + BIND).
Aim: to evaluate, on the existing HEEDB analysis cohort, whether a frozen EEG representation carries information about acute structural brain injury when injury is defined by radiology reports (acute infarct, intracranial haemorrhage, traumatic injury, mass effect, hydrocephalus) read independently of the EEG, rather than by diagnosis codes.
Why BIND: it is the only imaging resource linked to HEEDB patients; the radiologist reports are an independent reference standard.
Data elements: site-level `Clinical/` tables for I0001 (`imaging_findings_De-id`, `demographics`; `reports` for the report type and, for a human validation sample only, the report text) and I0004 for completeness; session dates; no pixel data.
Security: access only through the BDSP access point from approved compute, by named investigators; record-level data is never sent to a hosted model or API; analyses write aggregate outputs only with cells under 11 suppressed; any ID-bearing file lives in a restricted-permission `local_only` directory and is not committed; no re-identification, linkage to outside data, or onward sharing; results reported as aggregates.
Commercial use: none under this project. Analyses are non-commercial research; BIND-derived labels will not be used to train or tune a commercial product.
Investigators: [PI name, institution, role] ; [co-investigators] ; BDSP username [username] ; CITI certificate [date/ID] ; AWS account ID [12-digit ID registered in the BDSP profile] ; IRB or exemption [protocol number/status].

## 5. Cover e-mail (plain text, Gmail-ready)

To: [address to confirm: BDSP support form at https://bdsp.io/support/ ("Data access"), or the access contact BDSP names]
Subject: BIND v1.0 (doi:10.60508/mby8-3a26): confirm project access and terms for an existing BDSP credentialed user

Dear BDSP team,

I am a credentialed BDSP user ([BDSP username]) with an approved HEEDB project ([project title or ID]) and a signed credentialed Data Use Agreement.

I am writing about BIND v1.0. Under my existing credentials, the credentialed access point already lists the Imaging prefixes, and I have so far looked only at file and column names. I have not opened any patient-level content or computed anything from BIND. Before using it, I would like written confirmation of two things:

1. That my current DUA and approved project cover use of BIND (Imaging/I0001 and Imaging/I0004, Clinical tables), or the steps to register the project against BIND if a separate request is required.
2. The terms that apply: the BDSP Credentialed Health Data License 1.5.0 as applied to BIND, and whether any commercial use or commercial-entity access route exists beyond the "separate institutional agreement with the participating hospitals" described in the Scientific Data article.

Planned use: a non-commercial research analysis testing whether a frozen EEG representation carries information about acute structural brain injury, with injury defined by the radiology-report findings in the BIND Clinical tables (window of a few days around the EEG), compared against diagnosis-code labels. Outputs are aggregate only with small cells (n < 11) suppressed. Data stay on approved compute, record-level data is never sent to a hosted model or API, no re-identification or onward sharing, and BIND-derived labels will not be used to train or tune a commercial product. A clinician will read a small random sample of reports locally to check the accuracy of the LLM-extracted findings.

Two practical questions: does the Label_manual column mark the reports that were reviewed by a neurologist, and are the BIND and HEEDB date shifts identical for the same patient (so that BIND study dates can be compared with HEEDB EEG dates)?

Thank you,
[Name, degree]
[Title, institution]
[Email] | [Phone]
BDSP username: [username] | AWS account ID: [12-digit ID]
