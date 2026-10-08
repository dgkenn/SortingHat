# Data-rights ledger and production-build refusal rule

Owner files: `configs/data_rights.yaml` (ledger), `sortinghat/rights.py` (gate and CLI),
`tests/test_rights.py`, `docs/rights_checks.log` (append-only check log, created on the first check).
Ledger as of 2026-10-08. Plan references: `docs/research_plan_v1.txt` lines 258-300 (commercial track)
and 335-440 (data manifest).

## Status

No artefact is green. Counsel has not reviewed any licence, and no written grant is held for any
commercial use. So every commercial purpose is refused today, and the refusal is the expected result.

| Lineage | Count | Meaning |
|---|---|---|
| `research_only` | 11 | BDSP credentialed or restricted terms, or NC licences. Commercial uses are `not_permitted`. |
| `commercial_candidate` | 8 | May be cleared for commercial use after counsel review and written terms. |
| `yellow` | 3 | Share-alike or conflicting licence statements. Hold until clarified in writing. |

Status: 13 red, 9 yellow, 0 green (22 artefacts).

## Gate: the production-build refusal rule

`python -m sortinghat.rights check --purpose <purpose> --manifest <file>` (exit 0 allowed, 3 refused, 2 error).
`assert_buildable(ids, purpose)` raises `RightsRefused` on refusal. `@requires_rights(purpose, ids)`
refuses before the wrapped function runs.

| Purpose | Each artefact must be |
|---|---|
| `research` | `permitted` for `research` (any status) |
| `commercial_training` | status `green` and `commercial_training` `permitted` |
| `production_build` | status `green` and `commercial_training`, `derivative_weights`, `sublicensing` `permitted` |
| `regulatory_submission` | status `green` and `commercial_training`, `derivative_weights`, `regulatory_submission` `permitted` |

Rules that apply to every purpose:

1. An id missing from the ledger is refused (fail closed).
2. An empty manifest is refused.
3. Partly cleared is not green. One `unknown` use makes the status `yellow`, which blocks every commercial purpose.
4. Every check appends one dated line to `docs/rights_checks.log`: timestamp, source, purpose, verdict, and
   artefact ids with reasons. It holds no patient data, no paths and no record-level values. The file is never truncated.

Permitted-use values: `permitted` (the recorded licence or grant allows it), `not_permitted` (excluded by
the licence, or no grant is held), `unknown` (not read, or sources conflict). Status is derived from the uses
and lineage, and the loader rejects any entry whose declared status does not match:
red if any use is `not_permitted` or lineage is `research_only`; green if all five uses are `permitted`,
lineage is not `yellow`, and there are no open questions; yellow otherwise.

## Ledger

| Id | Lineage | Status | Research | Commercial training | Derivative weights | Open questions |
|---|---|---|---|---|---|---|
| `heedb_v4_1` | research_only | red | permitted | not_permitted | not_permitted | 3 |
| `bind_v1_0` | research_only | red | unknown | not_permitted | not_permitted | 1 |
| `morgoth_1_0_0` | research_only | red | unknown | not_permitted | not_permitted | 3 |
| `nesi_1_0_0` | research_only | red | unknown | not_permitted | not_permitted | 2 |
| `gcs_from_ehr_1_0_0` | research_only | red | unknown | not_permitted | not_permitted | 2 |
| `tme_cohort_1_0_0` | research_only | red | unknown | not_permitted | not_permitted | 2 |
| `icare_2_0_bdsp` | research_only | red | unknown | not_permitted | not_permitted | 2 |
| `icare_2_1_physionet` | research_only | red | permitted | not_permitted | not_permitted | 2 |
| `sparcnet_1_1` | research_only | red | unknown | not_permitted | not_permitted | 1 |
| `prophet_1_0` | research_only | red | unknown | not_permitted | not_permitted | 1 |
| `tuh_tueg` | commercial_candidate | yellow | permitted | unknown | unknown | 3 |
| `tuh_tuab` | commercial_candidate | yellow | permitted | unknown | unknown | 1 |
| `tuh_tusz` | commercial_candidate | yellow | permitted | unknown | unknown | 1 |
| `tuh_tuev` | commercial_candidate | yellow | permitted | unknown | unknown | 1 |
| `tuh_tuar` | commercial_candidate | yellow | unknown | unknown | unknown | 1 |
| `tdbrain_v3` | commercial_candidate | yellow | permitted | unknown | unknown | 2 |
| `cbramod_checkpoint` | yellow | yellow | permitted | unknown | unknown | 5 |
| `nmt` | yellow | yellow | unknown | unknown | unknown | 2 |
| `vitaldb_physionet_1_0_0` | yellow | yellow | permitted | unknown | unknown | 2 |
| `vitaldb_api` | research_only | red | permitted | not_permitted | not_permitted | 1 |
| `certa_eeg` | commercial_candidate | red | unknown | not_permitted | not_permitted | 3 |
| `korean_ncse_cohort_2024` | commercial_candidate | red | unknown | not_permitted | not_permitted | 1 |

Each entry in the YAML also records the licence text as recorded, source URL, version or hash where known,
access tier, and evidence file (a repo file that exists; the loader checks this).

## Counsel questions (summary)

Full lists are in the YAML `open_questions`.

1. **TUH family (TUEG, TUAB, TUSZ, TUEV, TUAR).** Do the TUH terms cover derivative weights in a regulated
   commercial product, and do they restrict redistribution of weights? (plan line 263). Record the TUEG version and manifest hash first (line 262).
2. **CBraMod checkpoint.** Licence conflict: GitHub code MIT, Hugging Face card Apache-2.0, braindecode mirror
   bsd-3-clause. Which terms govern the weights, given that the pretraining data is TUEG? The HF owner
   (`weighting666`) differs from the GitHub owner (`wjq-learning`).
3. **BDSP restricted and credentialed sets (HEEDB, BIND, MORGOTH, NESI, I-CARE 2.0, SPaRCNet, PROPHET, TME, GCS-from-EHR).**
   Each restricted dataset needs its own signed DUA. The restricted licence (BDSP Restricted Health Data License 1.0.0)
   limits use to non-commercial research. The single-DUA claim in the plan is unsupported (`citation_verification.md` row 4).
   Confirm in writing with BDSP (contact@bdsp.io). Also confirm whether models trained on research-only data are restricted.
4. **I-CARE 2.1 (PhysioNet, CC BY-NC-SA 4.0).** NC bars commercial use. Share-alike may reach derived weights.
5. **NMT (share-alike) and VitalDB (conflict).** Written clarification (plan line 264). The PhysioNet page says CC BY 4.0,
   while `api.vitaldb.net` is recorded as CC BY-NC-SA 4.0. The API copy is ledgered separately as research-only.
6. **TDBRAIN.** Read the Brainclinics DUA V3 text. The CC BY 4.0 statement comes from a secondary page.
7. **CERTA and the Korean cohort.** No licence is held. The CERTA "14 etiologies" detail is unverified
   (published groups: 4 and 7). Terms must be negotiated before any use.

## Firewall rules

- The commercial track uses only Study 1 and Study 2 conclusions after they are preprinted (D-075).
- No record-level, patient-level or restricted data enters a commercial artefact or the commercial track.
  Restricted data is research-only under every DUA recorded here.
- Every commercial design decision is logged with its public source (D-076). Entries are append-only.
- The firewall boundary needs institutional counsel sign-off, not only the lead's judgement (D-078).
- IP ownership is settled with the employer and the residency institution before forming an entity,
  filing a disclosure or signing a data licence (D-079).
- Every data agreement must grant commercial training, derivative weights, regulatory submission,
  continued improvement, sublicensing or spinout, and audit access (D-071). The ledger's five uses cover the first five.

## Dated decision log (commercial design)

Append new entries at the bottom. Do not edit existing entries; supersede them with a new entry.
"Public source" is what can be checked outside the repo. "Internal plan" means the basis is the research
plan (not public) and no outside source is cited.

| Date | ID | Decision | Public source | Outcome data seen? |
|---|---|---|---|---|
| 2026-10-07 | D-066 | No commercial build before Study 1 passes Gate 2 | Internal plan | No |
| 2026-10-07 | D-067 | Defer encoder retraining; pre-Gate work limited to provenance (rights ledger, refusal rule, TUEG manifest hash, CBraMod hash and training trace) | Internal plan; CBraMod paper arXiv 2412.07236 (training-data statement) | No |
| 2026-10-07 | D-069 | After Gate 2: C1 frozen public CBraMod if the gap is small and counsel clears it; C2 retrain if provenance is unclear; C3 if the gap is large | Internal plan; CBraMod repo https://github.com/wjq-learning/CBraMod | No |
| 2026-10-07 | D-070 | Route 1 (licensed hospital archive) for training labels; route 3 (prospective cohort) for validation; route 2 (vendor) optional | Internal plan | No |
| 2026-10-07 | D-071 | Every agreement grants commercial training, derivative weights, regulatory submission, continued improvement, sublicensing or spinout, and audit access | Internal plan | No |
| 2026-10-07 | D-072 | NMT and VitalDB yellow or on hold; CLEF excluded until weights are confirmed; TUEG family, CBraMod and TDBRAIN commercial candidates pending counsel; MIMIC and eICU excluded | Internal plan; VitalDB PhysioNet page https://physionet.org/content/vitaldb/; BDSP restricted licence https://registry.opendata.aws/bdsp-sparcnet | No |
| 2026-10-07 | D-074 | Pursue NIH SBIR/STTR once a company entity exists, STTR preferred | Internal plan (NIH program page not checked in this repo) | No |
| 2026-10-07 | D-075 | Commercial track uses only preprinted Study 1 and 2 conclusions | Internal rule | No |
| 2026-10-07 | D-076 | Dated log of commercial design decisions with public sources (this table) | Internal rule | No |
| 2026-10-07 | D-078 | Firewall boundary needs counsel sign-off | Internal rule | No |
| 2026-10-07 | D-079 | IP review before entity, invention disclosure or data licence | Internal plan | No |
| 2026-10-07 | (unlogged) | Citation check: "one BDSP DUA covers all restricted datasets" is unsupported. Restricted licence is non-commercial. CBraMod licences conflict. | BDSP https://bdsp.io/about/database/ ; https://registry.opendata.aws/bdsp_restricted_projects/ ; https://github.com/wjq-learning/CBraMod (LICENSE) ; https://huggingface.co/weighting666/CBraMod | No |
| 2026-10-08 | (ledger) | Ledger and refusal rule added. No artefact green. Every gated purpose refused. | Repo files listed at the top of this document | No |

The 2026-10-07 citation-check row is not yet in `DECISION_LOG.md`. The plan-compliance note
(`docs/research/plan_compliance_2026-10-08.md`) records that those findings were not logged. Add them
to `DECISION_LOG.md` as a new entry, not an edit.
