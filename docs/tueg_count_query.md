# TUEG keyword-label count query (NEDC script service) and the frozen-model transport test

Date: 2026-10-14. Status: DRAFT for the project lead. Nothing has been sent to NEDC and no TUEG data (waveform, report text, file list) has been seen.
Source note: `docs/research/open_etiology_datasets_2026-10-14.md` (rank 1, section 2.1). Label definitions: `docs/prereg_study1_sap.md` Amendment A1
(E1, E2, E5 primary; E3 positive control). Keyword file: `configs/tueg_keywords.yaml`. Code: `scripts/tueg_label_from_filelist.py`, tests `tests/test_tueg_label_from_filelist.py`.

Tags: **[V]** read from a primary page in this session (2026-10-14); **[P]** project-internal statement; **[U]** unverified or inferred.

## 0. Verified NEDC facts (what NEDC offers, what it does not)

| Fact | Source |
|---|---|
| Reports are not distributed: "We no longer distribute reports with our corpora. However, we do offer users the opportunity to search the anonymized reports for keywords or items of interest." [V] | https://isip.piconepress.com/projects/nedc/html/tuh_eeg/ |
| Mechanism: "If you provide a simple Python script that executes your search criteria, we can run it on the reports, and report which files matched your search criteria." The script must not use "any non-standard libraries"; it "takes as input a file containing a list of filenames, and returns filenames that match your search criteria." [V] | same page, section "Keyword search of reports" (RPRT) |
| Contact: **help@nedcdata.org**; the page's mailto link pre-fills the subject **"Python Script"**. The same address is used for access requests and general help. A second address printed on the page, joseph.picone@gmail.com, is for drive shipments and general corpus guidance; we do not use it for this request. [V] | same page |
| Template: `query.tar.gz` on that page (a ~30-line `query.py` taking the list file as `argv[1]`, opening each listed file and printing the file name on a match; plus `x.list`). The template stops after the first line of each file (a `break` inside the line loop); our generated scripts do not copy that behaviour. [V] | https://isip.piconepress.com/projects/nedc/html/tuh_eeg/query.tar.gz (read locally, not committed) |
| Caveat stated by NEDC: "Searching these reports is not trivial, since the reports are unstructured text"; false positives are likely. [V] | same page |
| **Not stated** on the page: any limit on the number of scripts, script size or turnaround time; whether counts-only output is acceptable; the number of reports; ICD codes; any ICU/ED/inpatient filter or field. [V that the page is silent] | same page |
| NEDC offers **no care-setting filter**. TUEG is described only as "clinical EEG recordings collected at Temple University Hospital", 2002-2017. [V] | same page |
| Current TUEG release: **v2.0.2** (server directory listing shows v2.0.1 dated 2026-01-19 and v2.0.2 dated 2026-05-21). Sub-corpora: TUAB v3.0.1, TUAR v3.0.1, TUEP v3.1.0, TUEV v2.0.1, TUSZ v2.0.6, TUSL v2.0.1. [V] | https://isip.piconepress.com/projects/nedc/html/tuh_eeg/ ; https://isip.piconepress.com/projects/nedc/data/tuh_eeg/tuh_eeg/ |
| **Path convention** (TUEG v2.0.2 AAREADME): `edf/000/aaaaaaaa/s001_2015_12_30/01_tcp_ar/aaaaaaaa_s001_t000.edf`. `000` groups 100 patients; `aaaaaaaa` is "a randomized subject ID" (8 characters); the session folder is `s<number>_<date>` with the session number "in chronological order" and the date "often not precise"; `01_tcp_ar` names the compatible montage definition; `t000` is "the first token generated from the EDF conversion process", i.e. several EDF tokens (`t000`, `t001`, ...) belong to one session. [V] | https://isip.piconepress.com/projects/nedc/data/tuh_eeg/tuh_eeg/v2.0.2/AAREADME.txt ; the TUAB AAREADME (https://isip.piconepress.com/projects/nedc/data/tuh_eeg/tuh_eeg_abnormal/v3.0.1/AAREADME.txt) gives the same subject_session_token file name |
| **Not documented**: where the report of a session sits, how it is named, and the section header strings inside reports. `scripts/tueg_label_from_filelist.py` therefore keys on subject + session parsed from any listed path (EDF or report, with or without a token) and the query scripts take whatever file names NEDC's list holds. [U] | (absence on the pages read) |
| Access: registration form v6.0 (3 June 2025) emailed to help@nedcdata.org, credentials in 24-48 h, then rsync over SSH. `docs/data_access.md` records that this project is already registered (approval 2026-07-27, `nedc-tuh-eeg` rsync key), so the request below is from an existing subscriber. [V for the process; P for our registration] | https://isip.piconepress.com/projects/nedc/html/tuh_eeg/ ; https://www.isip.piconepress.com/projects/nedc/forms/tuh_eeg.pdf |

Consequences: (i) we never see report text, only file names or counts, so label precision cannot be measured (section 3 limitation); (ii) setting (ED/ICU/inpatient) can only be approximated by words in the history text, so it is a proxy and "unknown" will be large; (iii) NEDC's turnaround and willingness to run ~25 scripts or a counts-only script are unknown ([U]; the email asks).

## 1. What we ask NEDC for

Design: the matcher is generated from `configs/tueg_keywords.yaml` into standard-library-only scripts (`python scripts/tueg_label_from_filelist.py emit-queries --out-dir <dir>`), so the exact keywords, cues and section rule that NEDC runs are the ones in the repository, hashed in a `MANIFEST.json`. Only the **CLINICAL HISTORY** and **MEDICATIONS** sections are read. IMPRESSION, CLINICAL CORRELATION, DESCRIPTION OF THE RECORD and INTRODUCTION are never searched: they are written after reading the EEG (or may restate it), so using them would import the circularity the plan bans.

| Round | What | Output | Purpose |
|---|---|---|---|
| 0 | One **profile** script (strict variant) | Counts only, tab-separated: reports read, sessions, sessions per set, label x setting, "pure" counts (exactly one of E1/E2/E5) overall and in acute context, label overlaps, clean controls | The count-first gate. No file names leave NEDC. If NEDC will only return file names, skip to round 1. |
| 1 | 13 list scripts (strict): `UNIVERSE_HISTORY` (reports with a recognised CLINICAL HISTORY section = denominator), `E1`, `E2`, `E5`, `E3`, `E3_broad`, `E5_nonspecific`, `CTRL`, `SETTING_ICU`, `SETTING_ED`, `SETTING_ADMIT`, `SETTING_CLINIC`, `SETTING_EMU` | Matching file names | Counts per label and per label x setting are then computed locally (intersections of the lists); no per-label setting script is needed. |
| 2 (only if the stop rule passes) | Loose variants of E1/E2/E5 (negation only; sensitivity), matching covariates: `UNIVERSE_MEDS`, `MED_SEDATIVE`, `MED_ASM`, `SEX_F`, `SEX_M`, age bands | Matching file names | Matched-control selection and the sensitivity analysis. Only for labels that passed. |

E4a, E6 and E7 are defined in the YAML (`query: false`) and are not requested unless the lead decides otherwise after the counts.

Label families (full lists in the YAML):

| Label | Role | Inclusion terms (abridged) | Policy |
|---|---|---|---|
| E1 structural | primary | stroke, infarct, intracranial/intracerebral/subarachnoid/subdural hemorrhage or hematoma, SAH/ICH/SDH/IVH/TBI/CVA (case-sensitive), head trauma/injury, brain tumor/glioma/metastases/"brain mass", hydrocephalus | negation, family, uncertain ("r/o", "rule out", "?", "vs"), remote, "history of" all excluded; "heat stroke", myocardial/bowel infarct, NPH ignored |
| E2 hypoxic-ischemic | primary | cardiac / cardiopulmonary / respiratory arrest, anoxic, hypoxic-ischemic, post-arrest, ROSC, PEA and VF (only with an arrest/CPR/code context), code blue, therapeutic hypothermia, asphyxia, drowning, "hypoxic" | "history of"/"s/p" **not** excluded (recent arrests are written that way); "years ago", "remote", "prior" excluded; "hypoxic respiratory failure" ignored |
| E5 metabolic | primary | hepatic encephalopathy/coma, hyperammonemia, liver failure, uremia/uremic, renal failure, ESRD, hyponatremia, hypoglycemia, hypercapnia, DKA, HHS, hypercalcemia | strict drops "history of ESRD"; loose variant keeps it |
| E3 | positive control | status epilepticus, NCSE, new-onset/breakthrough/witnessed seizure, seizure activity/clusters, postictal | as E1 |
| E3_broad, E5_nonspecific | indeterminate | any seizure/epilepsy wording; "metabolic/toxic encephalopathy", delirium | never positives; sessions are removed from the control set |
| CTRL (control set) | control | altered mental status, encephalopathy, unresponsive, coma, confusion, obtunded, lethargy | locally set-differenced: a control must match **no** label set |
| Settings (proxy) | stratifier | ICU / intubated / ventilator; ED / EMS / found down; "admitted" / inpatient; clinic / outpatient / routine EEG; EMU | acute = ICU or ED or admit; acute plus clinic/EMU = "mixed" (excluded from acute counts) |

Cue handling (`cues` in the YAML): a mention is dropped if a negation or family cue is in the preceding window ("no", "without", "denies", "negative for", "family history of", ...), if a post-cue follows ("ruled out", "negative"), and in the strict variant if an uncertainty ("r/o", "evaluate for", "concern for", "possible", "?", "vs"), remote ("years ago") or historical ("history of", "hx", "h/o", "PMH") cue applies. Scope breakers ("but", "now", "presented", "after", "s/p", "due to") stop a cue from reaching a term, so "history of HTN, now with acute stroke" counts. The heuristic is conservative on purpose (precision over recall) and is tested on synthetic sentences only; its true precision on TUEG reports is unknown.

## 2. Stop rule (decided before any count is seen)

Unit: **session**, restricted to sessions with exactly one primary label ("pure": E1, E2 or E5, not two of them) whose history text also carries an acute-setting word (ICU, ED or admission wording), and for which at least 3 clean acute controls per positive exist (clean = CTRL hit, no label-set hit, subject has no positive session; the control pool is the same setting stratum).

1. A primary label with **fewer than ~50** such positives is dropped from the primary analysis and reported "not testable" (survey, section 2.1: below about 50 positives a family leaves the primary analysis; the power table gives smallest detectable AUROC 0.633 at 50 positives, 0.594 at 100).
2. The open-data route **stops** ("not feasible") when fewer than 2 of {E1, E2, E5} remain testable. Reading note for the lead: the survey says stop if all three are under 50; the brief for this task says stop if "E1, E2 or E5" is under about 50. The two differ for the case of exactly one failing label. I implemented the survey rule plus the requirement of the survey's interpretation rule (at least two of three labels must be testable for a "supportive" verdict), and the code also reports the stricter "any primary below threshold" flag (`stop_if_any_primary_below`) so the lead can pick; **[OP-TUEG]: confirm which reading stands before the first request is sent.** SAP A1 G2 requires all three families to improve on HEEDB; that is a different test and is not changed here.
3. The threshold applies to the acute-context count because the intended-use claim concerns acute presentations. Caveat: setting words are a weak proxy, so a label could fail the acute count while having many positives overall; the summary reports all settings, and the lead may relax to "all settings" in the amendment if acute counts are small only because reports rarely state the setting.
4. Counts below 11 are shown as "<11"; the verdict flags are exact. The verdict is produced by `tueg_label_from_filelist.py profile-check` (round 0) or `build` (round 1).

## 3. Planned frozen-model transport test (pre-specification; to be fixed in a dated amendment before any scoring)

This section is a plan, not a result. Success criteria and label mappings must be registered (A3-style, as the survey requires) before the first scoring run; the counts rounds above score nothing and are logged in DECISION_LOG as an exploratory feasibility look.

1. **Model.** The frozen HEEDB-trained E1/E2/E5 (and E3) heads, no refit, no recalibration, no TUEG training, no TUEG-based tuning of preprocessing choices after the first look. Input harmonised to the model contract (19 channels, 10-20, 200 Hz, 0.5-70 Hz, common average reference); TUEG's own reference (AR/LE, `01_tcp_ar`, `02_tcp_le`) and 250 Hz are handled by that harmonisation, reported with and without. Whether the HEEDB heads transport across that shift is part of what the test measures.
2. **Rungs and what counts as primary.** The **hand-crafted-feature rung (qEEG) is primary** because CBraMod was pretrained on all of TUEG (paper, section 3.1: 26,846 sessions from 14,987 subjects; the version is not stated; `docs/research/cbramod_provenance.md`), so every TUEG recording scored here was seen by the CBraMod encoder, unlabeled. A CBraMod-embedding result on TUEG is therefore in-distribution at the encoder level, not held out, and is reported only as a labelled secondary ("encoder saw these EEGs"); a positive CBraMod result would not be read as transport evidence, a null one is still informative. The MORGOTH arm is cleaner: the survey records TUH as an external test set for MORGOTH, not training data **[P; `docs/models_spec.md` still lists MORGOTH exposure as UNVERIFIED for a human to fill from the model card; do that before relying on it]**. It is reported as a second secondary arm.
3. **Cases and controls.** Positives: "pure" sessions per label (section 2). Controls: clean CTRL sessions, same corpus (no cross-source pooling, so the source cannot be the signal), matched about 3:1 on age band, sex, session year (from the path), recording length and channel set, sedative / antiseizure medication class (round-2 covariates), and care-setting stratum. One session per subject; a subject that has a positive session never contributes a control (`subject_has_positive_session`); positives of two labels in different sessions of one subject are flagged and the subject is kept in one stratum by rule in the amendment. All splits and bootstraps are by subject.
4. **Estimand and tests.** AUROC of each frozen head against matched controls (positives vs controls), 95% subject-cluster bootstrap interval, compared with chance (0.5); no calibration claim. Secondary: the same AUROC for a baseline that sees only age, sex, year and medication class (both frozen or both fit on HEEDB, never on TUEG). Increment-over-baseline claims need several hundred positives per label (survey, section 2.2); TUEG is more likely to support "above chance" than "increment over context".
5. **Interpretation rule (from the survey).** Supportive if the E3 positive control shows clear signal **and** at least two of {E1, E2, E5} have a lower 95% bound above 0.5 against encephalopathy-matched controls. A null is uninformative because label noise can mask a real effect. No claim about E4a, E6, E7. The E3 control here is keyword-defined; it is a check that the pipeline can see a known EEG-visible state, not the EEG-adjudicated E3 of the SAP.
6. **Label-precision limitation (must appear wherever the result is quoted).** Labels are "history mentions X" by keyword, written by the treating or reading neurologist, not "has X". We never see the text, so precision, negation errors and misses are unmeasurable (NEDC returns file names). The case mix is epilepsy-heavy (87% of reports in the 2016 release contained "epilep" [P/V via survey]). Observed AUROC is attenuated roughly as 0.5 + precision x (true AUROC - 0.5) [M, survey]: a true 0.75 at 60% precision reads about 0.65, so the realistic range is 0.58-0.70 and detectability needs 50-200 positives per label. The CLINICAL HISTORY section may be edited by the reading neurologist after the EEG was read; EEG-blindness of that section is assumed, not verified **[U]**. An open-weight LLM check on a human-labelled subset is impossible here because we cannot read the reports.
7. **Execution rules.** Everything that touches TUEG waveforms is a human-launched, aggregate-only job (CLAUDE.md rules 1-3, applied by analogy: TUEG is de-identified and not under the HEEDB restriction, but its terms forbid release to third parties, and sending recordings or record-level labels into a hosted model's context would be that). The label table (session IDs) is written only under `local_only/` with mode 0600 and never read by an agent; only suppressed aggregates are printed or committed.

## 4. Licence implications (TUEG terms v6.0, 3 June 2025) [V]

Terms as read from the form text: (1) acknowledge the provider using the corpus publication; (2) no release to a third party or redistribution; (3) no re-identification; (4) "no malicious purposes. This data can be used for research and technology development, but not for uses beyond these broad classifications."; (5) "delete the data from all computer systems when finished with the data." The page itself states no further terms.

- The transport test is research use. There is **no explicit non-commercial clause and no explicit commercial or derivative-weights grant** (consistent with `docs/research/dataset_alternatives_2026-10-09.md` and `docs/data_rights.md`). Evidence from a TUEG scoring run is an input to a go/no-go decision, not a training source; no TUEG-derived weight or label table is shipped.
- **Delete when finished** has operational consequences: the downloaded EDFs for the scored sessions and the derived feature files must be deleted at completion with a dated deletion note (what is kept: aggregates and the keyword file; whether session-ID label tables are "data" under term 5 is a question for NEDC, included in the email). This conflicts in spirit with any regulatory requirement to retain data lineage; flag to counsel.
- The CBraMod question already open in `docs/data_rights.md` (weights pretrained on TUEG used in a regulated commercial product) is not answered by these terms.

**Flag for `configs/data_rights.yaml` (entry `tuh_tueg`; not edited here):** the entry says "Terms text not read in repo" and `evidence_file: docs/data_access.md`. The v6.0 text is now read (this note, section 4 and the verified form URL above). Suggested edits for the lead: record the v6.0 date and the five conditions in `licence`; keep `research: permitted`, keep `commercial_training`, `derivative_weights`, `regulatory_submission`, `sublicensing` as `unknown` (the text neither grants nor excludes them; "research and technology development ... not beyond these broad classifications" argues against assuming a grant); add an open question for the delete-when-finished condition versus retained label tables and lineage; add `docs/tueg_count_query.md` as evidence. Status stays yellow.

## 5. Ready-to-send email (plain text)

Send to: help@nedcdata.org (the address NEDC prints for the Python-script service). Subject: `Python Script`
Attach: a zip of the generated scripts (`tueg_query_*.py`) and `MANIFEST.json` produced by `emit-queries` (or only `tueg_query_PROFILE_strict.py` if you want the counts-only round first).

```
Subject: Python Script

Dear NEDC team,

I am an existing TUH EEG Corpus subscriber ([INSTITUTION]; registered on [DATE OF APPROVAL]; subscriber email [REGISTERED EMAIL]). I am writing to use your keyword-search service for the anonymized TUEG reports, as described at https://isip.piconepress.com/projects/nedc/html/tuh_eeg/ ("Python Script").

Purpose. We are testing, for research, whether a model trained on a different hospital's EEG transports to TUEG for a few clinical presentations (for example stroke/hemorrhage, post-cardiac-arrest, and metabolic encephalopathy, plus status epilepticus as a check). The model is not trained on TUEG. We do not need the report text; we need only the names of the matching files, and we will not redistribute them.

What the scripts do. Each is a standard-library-only Python script in your format: argv[1] is a file listing the report files; it prints the names of the files that match. They read only the CLINICAL HISTORY and MEDICATIONS sections of each report (not IMPRESSION, CLINICAL CORRELATION or the description) and apply simple negation and "history of" rules. The scripts are attached together with a manifest of SHA-256 hashes. The keyword lists are in the scripts as readable JSON.

Request, in two steps:
1. Counts first. The attached script tueg_query_PROFILE_strict.py runs all of the searches in one pass and prints only counts (reports read, sessions, and counts per search and per overlap). It prints no file names. If you are willing to run it, please return its output as it is. If you only run scripts that return file names, please say so and go straight to step 2.
2. File names. The 13 scripts named tueg_query_<SET>_strict.py (SET = UNIVERSE_HISTORY, E1, E2, E5, E3, E3_broad, E5_nonspecific, CTRL, SETTING_ICU, SETTING_ED, SETTING_ADMIT, SETTING_CLINIC, SETTING_EMU): please return the matching file names for each, as one text file per script, named after the script. If 13 scripts are too many, the priority order is UNIVERSE_HISTORY, E1, E2, E5, CTRL, SETTING_ICU, SETTING_ED, SETTING_ADMIT, E3, then the rest.
I may send a second batch (about 12 more scripts) after I have the first results.

Questions, so that I use the service correctly:
a. Which corpus version and report set do the scripts run on (we use TUEG v2.0.2)? Roughly how many reports are there, and is there one report per session? What are the report files called and where do they sit relative to the EDF session folders, so that the file names you return map to subject_session?
b. Is there a limit on script size or on the number of scripts, and what turnaround should I expect?
c. Do the returned file names (not report text) count as data that I must delete when finished (condition 5 of the v6.0 terms)? I plan to keep the file lists, labelled as derived from the reports, until the analysis is complete, then delete them together with any EDFs I download.
d. Do you prefer that I credit the corpus and the query service in a particular way beyond the corpus publication?
e. The reports are unstructured and I expect false positives and misses; if you can tell me whether the section headings CLINICAL HISTORY and MEDICATIONS appear in the same form in all years (2002-2017), that would help me interpret the counts.

Thank you for running the service and for maintaining the corpus.

Kind regards,

[NAME]
[TITLE]
[DEPARTMENT]
[INSTITUTION]
[POSTAL ADDRESS]
[TELEPHONE, with country code]
[REGISTERED EMAIL]
```

The commercial-use and derivative-weights questions are deliberately **not** in this email: they go to counsel with the CBraMod question (`docs/data_rights.md`), so a licence query does not delay the script request.

## 6. Workflow after NEDC replies

1. `python scripts/tueg_label_from_filelist.py emit-queries --out-dir out/tueg_queries` (done before sending; hash the manifest in the DECISION_LOG entry for the request).
2. Round 0 counts: save the reply as text; `python scripts/tueg_label_from_filelist.py profile-check --file <reply.txt> --summary-out out/tueg/profile_check.json`.
3. Round 1 lists: place `<SET>.txt` files in a local directory; `python scripts/tueg_label_from_filelist.py build --lists-dir <dir> --table-out out/local_only/tueg/labels.csv --summary-out out/tueg/summary.json`. The table (record-level) stays under `local_only/`; the summary is aggregate (n < 11 shown as "<11"), contains the per-label and label x setting counts, conflict-flag counts and the stop-rule verdict.
4. If the route continues: round 2, then the amendment fixing the success criteria and matching, then the human-run scoring job.
5. Delete the downloaded EDFs and derived features when the analysis is complete (term 5), and record the deletion.

## 7. Thin, unverified or inferred

- Whether NEDC will run ~25 scripts, a counts-only script, and how fast: unknown.
- Report file naming and location, and the section header strings in the corpus: not documented on the pages read; the query scripts and the YAML assume `CLINICAL HISTORY:` and `MEDICATIONS:` (upper case, colon or end of line). The profile reports how many sessions have a recognised CLINICAL HISTORY section; if that number is low, the header census comes before any label count is trusted.
- TUEG setting mix and how often reports state it: not stated anywhere read.
- All power and attenuation figures are the survey's estimates [M], not measurements.
- The matcher has been tested only on synthetic sentences. Its precision on real reports is unknown and unmeasurable through NEDC.
