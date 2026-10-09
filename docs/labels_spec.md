# Silver-label and reviewer-packet machinery (spec)

Code: `sortinghat/labels/`. Config: `configs/silver_anchors.yaml`. Tests: `tests/test_labels*.py`.
Source: `docs/research_plan_v1.txt` (Revised ontology, Reference standard, Silver labels: the circularity rules,
Phase 0b); `DECISION_LOG.md` D-001..D-013 and D-084..D-092 (anchor changes). Everything here was built and tested on tiny hand-made fixtures only;
no HEEDB data was seen. Every output that may leave a restricted-data job is aggregate-only with small-cell
suppression (n < 11 shown as `<11`) via `sortinghat.safe_output`. Record-level ID lists go only to `local_only/`.

## 1. Ontology (`ontology.py`)

| Label | Role | In primary endpoint |
|---|---|---|
| E1 acute structural (sublabels focal / diffuse) | primary | yes |
| E2 global hypoxic-ischemic | primary | yes |
| E3 epileptic contributor | positive control (separate EEG-based adjudication) | no (D-002) |
| E4a exogenous intoxication | primary | yes |
| E4b iatrogenic sedation | covariate / secondary | no (D-003) |
| E5 metabolic/physiologic | primary | yes |
| E6 systemic infection/inflammation (no CNS infection) | primary | yes |
| E7 CNS infection/inflammation | primary_conditional | only if >= 100 gold positives across >= 2 sites (D-004); otherwise exploratory |

`primary_endpoint_labels(e7_gold_positives, e7_sites)` returns E1, E2, E4a, E5, E6, plus E7 only when its rule
is met; it never returns E3 or E4b. Gold states (`GoldState`): absent 0, possible 1, probable 2, definite 3,
unassessable (off-scale, excluded from kappa). Binary positive = probable or definite. The circularity rules
apply to E1, E2, E4a, E5, E6, E7 (`SILVER_CIRCULARITY_LABELS`).

## 2. EEG sentence filter (`eeg_filter.py`)

`filter_text(text)` splits into sentences (newlines, `.!?;` followed by a capital/digit; clinical abbreviations
and decimals protected), drops every sentence matching an EEG-content pattern, and returns `FilterResult(text,
n_removed, n_kept)`. Removed text is never returned. `filter_notes` does the same for a list of notes.

Matched (word-bounded, case-insensitive unless stated): EEG in all forms (cEEG, c-EEG, vEEG, qEEG, aEEG, video-EEG,
EEGs, "E.E.G."), electroencephalogram/-graphy and misspellings, video/long-term monitoring, "brain waves",
slowing, slow waves, delta/theta slowing/activity, triphasic / tri-phasic, burst suppression (incl. "burst
supression", "suppression-burst"), LPD/GPD/LRDA/GRDA/PLEDs/GPEDs/BIPDs/SIRPIDs/FIRDA/TIRDA/ESES/IIC, periodic
discharges, rhythmic delta, epileptiform, spike-and-wave, sharp waves, polyspikes, background
attenuation/suppression/slowing/activity/reactivity, low voltage, isoelectric, electrocerebral silence,
posterior dominant rhythm, alpha coma, hypsarrhythmia, interictal, nonconvulsive status / NCSE, sleep spindles,
K-complexes, "reactivity ... background", and (case-sensitive upper case) `LTM`, `PDR`.

**Policy: err toward removal.** A removed benign sentence costs context; a leaked EEG sentence breaks blinding and
creates circularity. Deliberate over-removals, all tested: "slowing of heart rate" and any "slowing" (removed),
`LTM` in the sense of long-term memory (removed), "low voltage ECG" (removed), any sentence with "reactiv..."
within a few words of "background". Deliberately **not** matched: bare `BS` (blood sugar / bowel sounds are far
more common than burst suppression in notes; spelled-out burst suppression is caught), `PD` alone, `NCS`,
"postictal" and "seizure-like" (clinical observations), and words containing the letters (free, feeling, degree,
eggs, Delta-9 THC). Because `BS` is not filtered, a reviewer might see an abbreviated "BS" for burst suppression;
Phase 0b should count such leaks in a blinding check.

## 3. Banned positive evidence (`banned_evidence.py`)

Banned for E1, E2, E4a, E5, E6, E7: (1) EEG reports and EEG-mentioning sentences; (2) G92.* and G93.4* codes and
free-text encephalopathy (toxic-metabolic, metabolic, toxic, unspecified, septic, any "encephalopathy", TME,
SAE) without an anchor; (3) post-t0 (or untimed) neurology impressions unless EEG-filtered.
`classify_evidence(source, ...)` returns `Verdict(allowed, category, reason)`; `eeg_derived` covers EEG report,
EEG-mentioning sentence and unfiltered post-t0 neurology impression.

ICD-10-CM codes under the banned prefixes, enumerated 2026-10-07 from the ICD-10 MCP server (FY2027):

| Code | Description |
|---|---|
| G92 | Toxic encephalopathy (header) |
| G92.0 | Immune effector cell-associated neurotoxicity syndrome (header) |
| G92.00-G92.05 | ICANS, grade unspecified / 1 / 2 / 3 / 4 / 5 |
| G92.8 | Other toxic encephalopathy |
| G92.9 | Unspecified toxic encephalopathy |
| G93.4 | Other and unspecified encephalopathy (header) |
| G93.40 | Encephalopathy, unspecified |
| G93.41 | Metabolic encephalopathy |
| G93.42 | Megalencephalic leukoencephalopathy with subcortical cysts |
| G93.43 | Leukoencephalopathy with calcifications and cysts |
| G93.44 | Adult-onset leukodystrophy with axonal spheroids |
| G93.45 | Developmental and epileptic encephalopathy |
| G93.49 | Other encephalopathy |

The family rule bans by prefix (`G92`, `G934`), so future-year additions are caught; G92.0x and G93.42-G93.45 are
not "nonspecific" but are banned with the family (they are not independent anchors for these labels either).
Re-enumerate each October. Legacy ICD-9 prefixes 348.30/348.31/348.39/349.82 are also banned; the project lead
verified on 2026-10-07 that these four codes are correct (not tool-enumerated). HEEDB `condition_source_value`
mixes ICD-9 and ICD-10.

## 4. Objective anchors (`anchors.py`, `configs/silver_anchors.yaml`)

A silver positive needs at least one satisfied anchor; the YAML holds all rules as data. **Status: Evidence-reviewed
2026-10-08 (literature-grounded; see `docs/research/anchor_evidence_review.md`); external co-I review before publication**
(DECISION_LOG D-084..D-092 for the original sign-off, D-144 for the review). Input is a tidy event table
`case_id | item | value | hours_from_t0` (t0 = EEG start) plus an optional boolean `acute` column; items map from
OMOP via the LOINC table in the YAML (LOINC listed only where confident; empty list = map locally; verify against
the local concept map). Flag items (`loinc: []`) are computed upstream by the extractor.

| Label | Anchor | Window vs t0 (h) |
|---|---|---|
| E1 | ACUTE/SUBACUTE imaging finding: ICH, SAH, SDH/EDH, infarct, TBI/contusion, mass effect (`acuity_required: true`) | -72 to +24 |
| E2 | arrest event; asphyxia; profound shock (sustained MAP < 50 mmHg for >= 30 min, flag computed upstream) | -168 to 0; -168 to 0; -48 to 0 |
| E4a | antidote with documented response; non-therapeutic tox screen; ethanol >= 300 mg/dL; acetaminophen >= 150 ug/mL; salicylate >= 30 mg/dL | -6 to +6; -24 to +6 |
| E5 | ammonia >= 150 umol/L (was 100); BUN >= 100 mg/dL; glucose < 50 (window ends +1 h) or > 600 mg/dL; Na < 120 or > 160 mmol/L; PaCO2 > 70 mmHg AND arterial pH < 7.30 (both in window); arterial pH < 7.10; Ca > 14 mg/dL. Creatinine is not an anchor. | -24 to +6 (glucose<50: -12 to +1; PaCO2+pH, pH<7.10: -12 to +6) |
| E6 | CDC Adult Sepsis Event style: blood culture drawn AND `qad_ge4` (>= 4 consecutive qualifying antimicrobial days starting within +-2 d of the culture; upstream) AND any organ dysfunction: vasopressor initiation, lactate >= 2.0 mmol/L, `creatinine_doubling` (vs encounter baseline, excluding ESRD), `bilirubin_doubling_ge2` (>= 2.0 mg/dL and doubled), `platelets_drop` (< 100 x10^3/uL and >= 50% decline from baseline >= 100). Removed if any E7 anchor fires. No SIRS, no new mechanical ventilation. | culture and QAD -72 to +24; organ dysfunction -48 to +24 |
| E7 | CSF culture or PCR positive; autoimmune antibody (NMDAR: CSF only; other neural antibodies CSF or serum); or RBC-corrected CSF WBC >= 20/uL plus protein > 100, glucose < 40 or positive blood culture | -72 to +72 (antibody -168 to +168) |

Rule details:

- **E1 acuity.** Imaging flags must be acute or subacute findings (report qualifiers acute, new, hyperacute,
  subacute count; chronic, old, remote, sequela do not). When a label sets `acuity_required` and the event table
  has an `acute` column, only rows with `acute == True` count (missing or null acute does not count). If the
  column is absent the extractor must have applied the filter.
- **E5 PaCO2.** A composite `all_of`: PaCO2 > 70 mmHg and arterial pH < 7.30, both inside [-12, +6] h (pH within
  1 h of the PaCO2 is ideal; the implementation only requires both in the window). The acidemia requirement
  excludes chronic CO2 retention. Hepatic (ammonia) and uremic (BUN) thresholds keep a note that acute-on-chronic
  baselines need clinician review.
- **E6.** Mechanical ventilation is deliberately not used (comatose patients are intubated for airway
  protection). SIRS and the `suspected_infection` item were removed.
- **E7 RBC correction.** Corrected WBC = csf_wbc - csf_rbc / 500 (floored at 0) when a `csf_rbc` row has the same
  `hours_from_t0` as the `csf_wbc` row (same tap); otherwise the uncorrected count is used. `csf_rbc` has no
  LOINC in the YAML (26455-6 is a candidate, not verified).

**Data-source availability (D-153; `availability.py`, `availability_rules` in the YAML).** A label whose source table is nearly empty for part
of a site's record period cannot fire there, and coding those cases as confident negatives makes the label drift with the data feed (E6 and blood
cultures: a ~14x temporal observed/expected, `docs/research/e6_drift_2026-10-09.md`). The rule makes such cases NOT ASSESSABLE (NaN, never 0),
reason `culture_source_absent`. E6 today:

1. A case has the source when it has >= 1 `blood_culture_drawn` row in the E6 culture window ([-72, +24] h, the label's own leaf; configurable).
2. Within each site, cases are ordered by t0 (rank only) and cut into `round(1 / bin_fraction)` equal bins (default 10 bins of ~10% of the site).
3. A bin whose share of cases with the source is below `min_share` (default 10%) is source-absent; all its cases are not assessable. Contiguous
   source-absent bins form the "culture-free era" of the site. A site with fewer than 2 x `min_bin_cases` (default 50) cases is not evaluated.
4. A positive is never made not assessable (asserted in code and tested); by default the rest of an absent bin is NA, including negatives that
   happen to have a culture row (`keep_cases_with_source: true` keeps those, which keeps both classes but selects the sicker cases in the era).

The rule uses only structured data availability and the within-site t0 rank. It reads no outcome label (except to protect positives), no EEG feature
and no EEG report, so it is EEG-blind. It is label-level (applied after the anchors are evaluated): `extract_silver` returns NaN in `labels`, the
labels CSV holds an empty cell, `run_silver_feasibility.load_silver` reads that as not assessable and the masked losses skip it. The silver report
adds `availability.<label>` (per site: bins, source-absent bins, not-assessable count and share, positives kept, per-bin source share; all suppressed)
and `n_not_assessable` / assessable-based prevalence in `per_label`. `--no-availability-rules` on `python -m sortinghat.labels.extract` reproduces
the legacy labels. Limits: the threshold is a data-driven heuristic; a bin that is source-absent for a reason other than the feed (a genuinely
low-culture period) is also marked; the first and last bins can mix eras; culture timestamps that carry result time rather than collection time
(not yet verified by a human) make the early era look more absent than it is.

E4a note: toxicology positivity for a drug given in hospital is E4b, not E4a; the extractor must drop in-hospital
agents. `silver_anchor_table(events)` returns a boolean case-by-label table; fired anchor ids are kept in
`.attrs['fired']` for local audit only.

## 5. Circularity audit (`circularity_audit.py`)

On cases held out from silver training that have both gold and an EEG-report-impression label, compare the
silver-trained model's agreement with the EEG-report impression versus with gold (AUROC by default; Cohen's
kappa at a threshold as an option). **Rule (as written in the plan): leak = agreement(EEG impression) >
agreement(gold)**, per label; any leaking label sets `rebuild_silver_labels = True`. A seeded bootstrap CI of the
difference and `leak_confident` (lower bound > 0) are reported for context only. Overlap between silver-training
cases and audit cases raises `SilverScoringError` (silver labels train, never score).

## 6. Phase 0b pilot (`pilot.py`)

- `draw_pilot_sample`: seeded, site-stratified (proportional, largest remainder, >= 1 per site), n = 200. IDs are
  record-level: save with `save_pilot_ids` (`local_only/`, mode 0600), never print. Allocation by site is returned
  suppressed.
- `label_kappas`: per label, two reviewers: linear- and quadratic-weighted kappa on the 4-level ordinal scale,
  unweighted binary kappa (probable/definite positive), prevalence, unassessable count (excluded).
- `minutes_summary`: minutes per case as quantiles (10/25/50/75/90), overall and per reviewer; no min/max.
- `eeg_only_share`: among naive silver positives (any evidence row), the share whose evidence is entirely
  EEG-derived (`eeg_only_share`) and entirely banned (`banned_only_share`, adds G92/G93.4 and anchorless
  encephalopathy text). This measures how much circularity would have hurt (D-013).
- `gate0_pilot_check`: kappa >= 0.6 for >= 4 primary families and >= 3 families with >= 10% prevalence
  (primary families = `primary_endpoint_labels()`; E7 only if the caller lists it).
- `pilot_report` bundles all of the above.

## Open items

Anchor thresholds and windows were signed off by the project lead (delegated), then evidence-reviewed 2026-10-08
(`docs/research/anchor_evidence_review.md`); they still need external co-investigator review before publication, and
the extractor must enforce the NMDAR CSF-only specimen rule and prefer principal-diagnosis position for E1 dx codes; the OMOP-to-item mapper
(LOINC and drug/procedure concepts to `items`) is not built, since the real table contents are unseen; the
reviewer-packet summarizer (open-weight, 50-case validation) is out of scope here; `BS` abbreviation leakage
should be checked in the pilot.
