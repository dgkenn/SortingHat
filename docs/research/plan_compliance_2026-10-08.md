# Plan compliance review: `docs/research_plan_v1.txt` vs repository state

Date: 2026-10-08 (review run late afternoon; the overnight driver was still running). Read-only review.
Sources read: the plan, `DECISION_LOG.md` (D-001..D-117, 117 unique entries, no deletions in git history), `docs/*.md`,
`docs/research/*.md`, `sortinghat/`, `scripts/` (incl. `overnight.sh`), `configs/`, `out/cohort/flow.md`,
`out/audit/field_audit.md`, `out/logs/*.log` (aggregate logs). Not opened: anything under `out/local_only/` and
`out/audit/local_only/` (only top-level file names and file modes were listed). `pytest tests/test_field_audit.py` passes.
No HEEDB access was made in this review.

Status key: done / in progress / not started / blocked-human / deviated. "NOT LOGGED" means no D-number covers it.

## 1. Compliance table

### Architecture, hypotheses, firewall

| Plan item | Status | Evidence | Deviation logged? | Next action |
|---|---|---|---|---|
| E4a/E4b split; E3 and E4b out of primary endpoint | done | D-001..D-003; `sortinghat/labels/ontology.py`; `metrics/labels.py` | n/a (plan change, logged) | none |
| Primary endpoint (masked multi-label log loss, delta) | done | D-002, SAP s5, `metrics/loss.py` | yes | none |
| H1-H6 preregistered | in progress | SAP v0.1 DRAFT (`docs/prereg_study1_sap.md`): registry ID `[pending]`, biostatistician and co-I sign-off `[pending]`, not hashed or filed | D-017..D-022 | Biostatistician review; log SAP s16 items; file and hash before any gold label is opened |
| H1/H2 CI 95% to 99% within-site plus every-site rule | deviated | D-095, `evaluation_sample_size.md` s8 | D-095 | Biostatistician to confirm |
| H5 (early-EEG) | deviated (dropped) | D-103; audit: no imaging finalization time at EEG sites | D-103 | Move to prospective programme |
| Baseline D / referral indication | deviated (dropped) | D-107; no field in any real header | D-107 | `baselines/` and SAP s4.3 still carry D; tidy |
| Firewall: dated log of commercial decisions, preprint-first, counsel sign-off | not started | no ledger or commercial decision file in repo | D-075..D-078 (intent only) | Create the log when first commercial decision is made; counsel review |
| IP review before entity, disclosure or data license | blocked-human | `next_30_days_checklist.md` (USER-ONLY); outreach draft #4 unsent | D-079 | See s3 |

### Phase 0 and Gate 0

| Plan item | Status | Evidence | Deviation logged? | Next action |
|---|---|---|---|---|
| 0a EEG start date/time, >=95% | done: PASS 99.9% (158407/158518) | `out/audit/field_audit.md` | criterion unchanged | none |
| 0a Consistent date shift (STOP) | in progress / deviated | Only real output is stale: old median-offset proxy FAIL (63.6% misaligned). D-117 replaced the gate after that FAIL. Rerun with the D-117 gate has not produced output: `out/logs/field_audit.log` shows `exit=124` (timeout, 15:11); `overnight.sh` step 2 pending. 20-case human hand-check pending. Within-record ordering: `drug_start_before_end` violations 963,348 of 4,380,317 (22.0%), now only a "secondary count" | D-117 (flag says outcome seen "No" although it cites the real FAIL) | Complete the rerun; human hand-check; investigate the 22% drug start/end inversion before trusting Baseline A timing |
| 0a Medication administration times (>=80% of candidates) | done: PASS under an operationalisation | 98.9% (12699/12847) among candidates with a sedation exposure in 48 h; only 24.8% of all candidates; I0002 and I0003 have <11 exposed candidates | NOT LOGGED (denominator change; D-034 logs only the fallback) | Log the operationalisation |
| 0a Lab result/verification time | done: FAIL, fallback | no result-time column in `omop_measurement` | D-034 (pre-logged fallback) | Baseline C labelled "approximate"; replace placeholder assay lags |
| 0a Imaging finalization time | done: FAIL, fallback | no imaging table for EEG sites (only I0001/I0004) | D-103 (H5), D-102 (E1 silver) | Log that Baseline C has no imaging features (NOT LOGGED) |
| 0a GCS/FOUR/RASS within +-6 h, >=50% | done: FAIL (42.1%, 21572/51260) | `field_audit.md`; per site: S0001 53.8%, S0002 65.4%, I0002 and I0003 <11 | fallback pre-logged (D-034); NOT LOGGED that the fallback ("GCS-from-EHR tool; broad cohort only") was not applied: the cohort keeps a structured-GCS strict cohort (D-105) | Project-lead decision; `NAX/nax-gcs/` prefix exists, unused |
| 0a Site identifier (>=3 adult sites with >=300 candidates) | done: PASS on pre-severity candidates, but see risk 3 | 4 of 4 sites (I0002 13040, I0003 1259, S0001 22508, S0002 14453); strict primary cohort 3292 = S0001 2002 + S0002 1290, I0002 and I0003 <11 (`flow.md`) | D-113, D-024 cover the site list, not the 2-site outcome (NOT LOGGED) | Decide: grouped split / 2-site LOSO / GCS-tool broad cohort |
| 0a Timestamped notes (STOP) | done: PASS 90.1% (46171/51260) | `field_audit.md`; threshold fixed at 80% before the run | D-101 | none |
| 0b Label pilot (200 cases; minutes, per-label kappa, EEG-only share) | not started / blocked-human | code ready (`labels/pilot.py`, `draw_pilot_sample`); no sample drawn, no reviewers, no packets | D-013 | Draw sample (local_only), recruit two pilot adjudicators |
| 0c Novelty sweep and collaboration decision | done / blocked-human | `novelty_sweep.md` (scoop risk LOW-MODERATE; CLEF on HEEDB); drafts unsent | D-044 | Decide on BDSP offer after IP answer |
| 0d IP and employer review | blocked-human | no work product | D-079 | Human reads both contracts; send email #4 |
| 0e Agent safety | deviated | Synthetic generator, `safe_output`, `.claude/settings.json` deny rules, `heedb_run.sh` guard exist. But value-level jobs are being launched with the guard unset (see s2 item 1) | D-080..D-083 (rules only) | See s2 item 1 |
| Gate 0 | not determined; cannot pass yet | Stop rows: EEG start PASS, notes PASS, date shift unresolved; kappa and prevalence criteria unmeasured | D-051 | Needs rerun, hand-check, pilot |
| `phase0a_field_audit_2026-10-07.md` | stale | says "NOT YET RUN", contradicting `out/audit/field_audit.md` | n/a | Update after the rerun |

### Study 1

| Plan item | Status | Evidence | Deviation logged? | Next action |
|---|---|---|---|---|
| Population: adults, acute care, first qualifying EEG | done (code and real flow) | `out/cohort/flow.md`: 341,228 sessions to 14,516 table rows; strict primary 3,292; strict_pm6 3,851; broad 8,204 (`build_cohort.log`) | D-104..D-106, D-111..D-116 | EEG-QC step still to append to flow |
| Strict cohort GCS <=11/FOUR <=12 | deviated | primary window [-6,+1] h; +-6 h demoted to sensitivity | D-105 | none |
| t0 = EEG start; window min 1-11 | deviated | t0 = start of first sustained live segment (median offset ~3-6 min, q90 ~29-40 min); QC 8/10 channels | D-108, D-109, D-110 | Implement and document the downstream join (risk 5) |
| EEG within 24 h of ACI onset | deviated | structured proxy only (first abnormal score, else encounter start); no note-derived onset | D-104 (C-06) | Revisit if notes are used |
| Referral indication | deviated (dropped) | D-107 | yes | none |
| Ontology E1-E7 | done in code | `labels/ontology.py`; E7 rule D-004 | D-001..D-004 | none |
| Gold reference standard (dual review, packets, 50-case summarizer validation) | not started | no packet builder or summarizer; sentence filter built (`labels/eeg_filter.py`); BIND not accessed | D-011, D-012 | Needs adjudicators and open-weight compute |
| E3 positive control with separate EEG-based adjudication | not started | no adjudication process; not in `models/` controls | D-002 | Design with EEG co-I |
| Silver rules (bans, anchors, circularity audit) | in progress | `labels/banned_evidence.py`, `configs/silver_anchors.yaml` (SIGNED OFF by lead, co-I re-review pending), `labels/extract.py`, `circularity_audit.py`; real silver run queued in `overnight.sh` step 3, not yet done | D-005..D-010, D-084..D-092, D-102 | Co-I re-review; run; EEG-impression flag mapping needs EEG-clinician sign-off |
| Silver from notes (open-weight LLM) | not started | `silver_extraction.md`: "Notes ... out of scope" | NOT LOGGED | Log scope narrowing or build |
| Baselines A-C | in progress | `sortinghat/baselines/` tested on synthetic only; no real-data baseline matrix; NESI has no data source; no imaging; lag table is placeholders | D-014, D-034; NESI and imaging gaps NOT LOGGED | Log; run on real data via human |
| Representations: qEEG, connectivity | in progress | `eeg/`, `scripts/extract_eeg_features.py`; 12 shards, about 14.5k recordings attempted; per-shard primary-window QC pass 0.78-0.83 (`extract12b_s*.log`) | D-100, D-108..D-110 | Wait for completion; append QC counts to flow |
| Representations: MORGOTH, frozen CBraMod, dynamics | not started | `models/inputs.py` placeholders raise `NotImplementedError`; no weights obtained | D-068 | Obtain MORGOTH (human); embedding step |
| Splits: LOSO, temporal holdout | in progress | `metrics/splits.py` (synthetic); only 2 usable strict sites; temporal holdout depends on cross-patient calendar order that per-patient date shift may not preserve | D-023..D-025, D-098 | See risk 3 |
| Controls (severity matching, sedative-excluded, leakage probes, negative controls, exposure accounting, risk-coverage) | in progress | `models/controls.py`; exposure accounting is a STUB; E3 positive control, H6 nested-window analysis not built (`models_spec.md`) | D-026 | Finish after pilot |
| Evaluation size (~1,000 gold, recalc after pilot) | deviated (ordering) | `metrics/power.py`, `sample_size.py`; N fixed at ~1,000 before the pilot; sims assume 3 sites | D-027, D-093, D-094 | Recompute with measured pilot inputs and the real site count |
| Study 1D reader study | not started (conditional on H2) | none | D-030 | none |
| Any model trained on real data before Gate 0 | none found | harness trained on synthetic only; `scripts/` has no real-data fitting script | D-039 | Keep so |

### Study 2, commercial, prospective

| Plan item | Status | Evidence | Deviation logged? | Next action |
|---|---|---|---|---|
| Study 2 external datasets, frozen model | not started (gated by G2) | no datasets; outreach unsent | D-028, D-029 | none |
| Montage simulation (real geometries) | in progress (early, code only) | `configs/montages.yaml`, `montage/`; Ceribell headband electrode list medium confidence, reference unknown; headcap electrodes unknown; forehead-only via BrainScope list | D-040, D-041 | Vendor IFUs (human) |
| Duration windows 20 s-10 min | in progress | extractor writes nested windows; H6 analysis not built | D-042 | none |
| Commercial "now": rights ledger and production-build refusal rule | not started | not in repo | D-067 | Create |
| Commercial "now": pull TUEG, hash manifest | blocked-human | checklist USER-ONLY | D-067 | Human |
| Commercial "now": hash CBraMod, trace training data | done; counsel step blocked-human | `cbramod_provenance.md` (SHA-256 recorded; checkpoint only in scratch, not repo) | D-067 | Counsel on TUH derivative-weights terms |
| Keep NMT, VitalDB yellow; no encoder retraining | done | D-072, D-067; no training code | yes | none |
| Proprietary dataset, entity, SBIR/STTR | not started (by design, post-G2) | none | D-070, D-071, D-074 | none |
| Prospective P0-P4, De Novo, PCCP, QMSR | not started (by design) | none | D-057..D-065 | none |
| Citation re-verification | done; plan text not yet corrected | `citation_verification.md` found: one-DUA claim unsupported; CLEF is MIT/BWH/MGH; CERTA "14 etiologies" not found (7 groups; toxic-metabolic 23, encephalitis 7); 882.1450 wording | D-038 (intent only); findings NOT LOGGED | Log; fold into handoff v1.1 |

### Data manifest, team, gates, next 30 days

| Plan item | Status | Evidence | Deviation logged? | Next action |
|---|---|---|---|---|
| HEEDB (names-only probe done; value-level audit/cohort done) | in progress | `heedb_schema_dryrun_2026-10-07.md`, `field_audit.md`, `flow.md` | D-035, D-100 | Phase 0 version/size inventory not found in repo (NOT LOGGED) |
| BIND, MORGOTH, NESI, GCS-from-EHR, TME, I-CARE, SPaRCNet: confirm DUA, pull | blocked-human | `data_access.md` marks BIND/NESI "not found"; DUA status is USER-ONLY | D-036 | See s3 |
| TDBRAIN | not started | citation check: Brainclinics source, "V3.1" unconfirmed | NOT LOGGED | Human download |
| Team: biostatistician, co-I, 6-10 adjudicators, counsel | blocked-human | no names or commitments in repo | D-047..D-049 | See s3 |
| Gates G0, G2-G5 | defined; none passed | SAP s11; no G1 | D-051..D-056 | none |
| Risk register | exists in plan; see s4 for live ranking | | | |
| Next 30 days: DUA check | blocked-human | checklist unchecked | | |
| Next 30 days: agent-safety setup | done, with the s2 caveat | | | |
| Next 30 days: HEEDB metadata pull and audit | in progress | see 0a | | |
| Next 30 days: IP clauses and tech-transfer question | blocked-human | | | |
| Next 30 days: novelty sweep | done | | | |
| Next 30 days: three outreach emails | done (drafted, unsent; #4 first) | `outreach_drafts.md`; handoff s23.3 package missing | | Human send |
| Next 30 days: recruit biostatistician and 2 pilot adjudicators | blocked-human | | | |
| Next 30 days: TUEG subsets, CBraMod hash | TUEG blocked-human; hash done | | | |
| Next 30 days: handoff v1.1 and log each change | in progress | log has 117 entries; v1.1 not issued; handoff v1.0 is not in the repo, so changes cannot be diffed | | Lead to issue v1.1 |

## 2. Deviations from the plan that are NOT in DECISION_LOG

1. **Restricted value-level jobs run from an agent session with the guard bypassed.** `scripts/overnight.sh` runs
   `env -u CLAUDECODE -u SORTINGHAT_AGENT_SESSION scripts/heedb_run.sh ...`; the running extraction processes are children of
   a Claude Code shell (`pgrep` shows the shell-snapshot wrapper), with `BDSP_AWS_*` secrets in the environment. CLAUDE.md rule 2 says restricted
   jobs run from a plain terminal or scheduler, and its only recorded authorisation covers the names-only probe.
   `phase0a_field_audit_2026-10-07.md` itself states that unsetting these variables "defeats that control". The `overnight.sh` header
   says "HUMAN-AUTHORISED" but no authorisation is recorded, and no D-number covers it. Whether the cloud container counts as
   "approved compute" is also unrecorded.
2. **GCS audit fallback not applied.** Row FAILED (42.1% < 50%); plan fallback is "BDSP GCS-from-EHR tool; broad cohort only".
   The cohort instead keeps a structured-GCS strict cohort (D-105 changes the window but not this).
3. **Site criterion met on a different population than the analysis cohort.** Audit counts 4 sites with >=300 pre-severity
   candidates; the strict primary cohort has 2 contributing sites. D-093's "~330 per site", the 3-site power simulations, and D-098's
   rationale all assume 3 sites.
4. **Medication-administration row operationalised** with a conditional denominator (candidates with a sedation exposure) instead of "candidates".
5. **Baseline composition gaps:** Baseline C has no imaging (no imaging table at EEG sites); Baseline A lists NESI but there is no NESI source in the data; lab
   and imaging lags are placeholders (`baselines_spec.md` s2.3).
6. **Silver labels are structured-only;** note-derived silver evidence (plan: "ICD codes, clinical attribution and notes") is out of scope with no log entry.
7. **t0 redefinition not propagated.** D-109 says the cohort/feature join uses t0 = metadata start + `onset_offset_s`, but `onset_offset_s`
   is consumed nowhere outside the extractor; cohort `t0`, baselines' `as_of`, silver extraction and the SAP (s2: "t0 = EEG start ... minutes 1 to 11 of the recording") still use metadata start.
   The new `no_sustained_signal` exclusion is not in the flow.
8. **Unlogged operationalisations that the SAP and specs themselves say need entries before freeze:** SAP s16 items 1-13 and 15
   (possible=0, eps=1e-4, primary rung, bootstrap choice, H3/H4/H5/H6 definitions, "improves individually", sedative adjustment, 0.20 temporal fraction,
   E7 eligibility, H3/H5 gating, G0 kappa type); `baselines_spec.md` s4 (all windows and lags); `models_spec.md` choices (Platt default, 30-dev minimum, 5 SD clip).
9. **Citation-check findings** that change plan text (single-DUA claim false; restricted/credentialed licence is non-commercial; CERTA categories 4/7 not 14; CLEF provenance;
   882.1450 wording; TUSZ not mentioned in the CBraMod paper although D-037 treats it as contaminated) are not logged or reflected in any decision.
10. **Log header and flags are inaccurate.** Header says no HEEDB data or Phase 0 result has been viewed and every entry is "Outcome data seen? No";
    D-110..D-117 were made after real aggregates and the failed audit. D-117 rewrote a Stop-row gate after the FAIL. The field is arguably defined as labels/model output,
    but the log never says so. Needs an annotation, not an edit (append-only).
11. **Stale status documents:** SAP header "Outcome data seen at drafting: none"; `phase0a_field_audit_2026-10-07.md` "NOT YET RUN";
    `baselines_spec.md`/SAP s4.3 still list Baseline D.

## 3. Steps skipped or out of order

1. **Real-data feature extraction (about 14.5k recordings) and structured silver extraction precede the Gate 0 verdict, the hand-check and the pilot.**
   Plan: "no modeling yet" and "nothing is trained until the field audit and the 200-case pilot pass". These are not model training and no
   real-data fit exists, but they are the largest compute spend and they fix QC and window rules on real EEG first.
2. **QC thresholds fixed from real S0001 signal diagnostics (D-110), not the Phase 0b pilot** that D-097 specified. Logged, but out of order.
3. **No gold labels, no pilot kappa, no pilot minutes, no EEG-only share exist.** G0 cannot be evaluated. The sample-size
   decision (D-093) was nonetheless fixed ahead of the pilot (D-027 said recalculate after).
4. **SAP not filed or hashed** while the project has already viewed per-site cohort and audit aggregates (freeze order, SAP s10.9).
5. **Downstream harness, Study 2 montage code and power simulations built before any Gate:** synthetic only; defensible, but effort is running ahead of Phase 0.
6. **Date-shift gate redefined after seeing its FAIL (D-117),** and the replacement has not yet run on real data; the human hand-check is still pending.
7. **Handoff v1.1 not issued** although the plan lists it in the first 30 days and the log is now 117 entries long.
8. No model was trained on real data before Gate 0; the E4b/E3/imaging-dependent pieces (E3 control, MORGOTH, CBraMod embeddings) are simply not built.

## 4. Items blocked on humans

- **Adjudicators:** 6-10 pool, two pilot adjudicators, third readers; none recruited. Pilot sample draw and chart-review packets (needs open-weight summarizer on approved compute).
- **Senior EEG/neurocritical-care co-I:** re-review of silver anchors before protocol lock (D-092), E1 code lists, EEG-impression flag mapping, E3 adjudication design.
- **Biostatistician:** SAP sign-off, sample-size and CI-level confirmation (D-093..D-095), SAP s16 items.
- **IP review (0d):** both contracts; email #4 to the incoming tech-transfer office. Outreach #1-#3 depend on it. Counsel: firewall sign-off, CBraMod/TUH derivative-weights question, rights ledger.
- **Outreach:** BDSP group (addresses not in sources), CERTA (handoff s23.3 package missing from repo), Korean cohort (Kim JB).
- **BDSP DUAs:** confirm each (HEEDB, BIND, MORGOTH, NESI, TME, GCS-from-EHR); I-CARE and SPaRCNet each need their own restricted DUA.
- **Date-shift hand-check** (20 cases, `out/audit/local_only/`, human only) and the Stop-row verdict.
- **Project-lead decisions:** GCS fallback vs structured strict cohort; 2-site design; ratify or stop agent-session HEEDB runs; issue handoff v1.1; register the SAP.
- **TUEG pull and manifest hash; TDBRAIN download; MORGOTH weights/splits; vendor IFUs; FDA originals.**

## 5. Top 5 risks to plan fidelity right now

1. **Governance of restricted data.** Value-level HEEDB jobs run from an agent session/cloud container with the agent guard env vars removed, no recorded
   authorisation, and `.claude/settings.json` does not deny `**/local_only/**` (rule 5 relies on convention). Exposure is the plan's own "restricted data reaches a hosted model" risk.
2. **Gate 0 is unresolved and the Stop-row gate was rewritten after a FAIL.** The only real audit output says FAIL; the D-117 rerun timed out; 22% of drug exposure rows end before they start,
   which undermines Baseline A (sedative exposure at t0) timing. The human hand-check remains the only real control.
3. **Effective site count is 2, not 3-4.** Strict cohort comes from S0001 and S0002 only; I0002/I0003 have <11 strict patients and <11 GCS/sedation-administration coverage.
   LOSO becomes one training site per fold, "every held-out site" rests on two estimates, S-sites may share a system or vendor (plan's own warning), and the temporal holdout may be infeasible.
4. **Label and labour bottleneck is unstarted.** No adjudicators, biostatistician or co-I; no imaging at EEG sites (E1 gold packets, Baseline C); silver is structured-only with a proxy antidote anchor;
   G0's kappa and prevalence criteria cannot be measured. This is the plan's own "adjudication labor never materializes" risk, now on the critical path.
5. **Pre-registration integrity and drift.** Many operational decisions (D-102..D-117, SAP s16) were taken after viewing real aggregates, the log header and flags are stale, the t0 definition differs across
   cohort/features/baselines/silver, the SAP is unregistered and unhashed, and handoff v1.0 is not in the repo. The claim "decided before any outcome was seen" needs an explicit definition and an annotation, then a freeze.

Also watch: scoop risk via CLEF-style HEEDB work (`novelty_sweep.md`) and the unstarted IP review, both of which the plan ranks as early-phase items.
