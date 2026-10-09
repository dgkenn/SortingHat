# Structured silver labels: yield review (E2 shock, E5, E6, report bug)

Aggregate numbers only (n < 11 shown as "<11"). Source: the real structured silver run (14,516 cases: S0001 8,032, S0002 4,968,
I0002 1,504, I0003 12), re-examined from the silver checkpoint and the shared OMOP cache with read-only probes (D-118).
Code changes: `sortinghat/labels/extract.py`, `configs/anchor_concepts.yaml`, `tests/test_labels_extract.py`.

## 1. Report bug: per-site prevalence shown while its count was hidden

Cause. The counts and prevalence were computed from the same label table. The count was hidden by complementary suppression:
I0003 has < 11 positives, so the smallest remaining cell (I0002) was also hidden to stop it being recovered by subtraction. The
prevalence was only suppressed on its own primary rule, so `0.262 x 1,504` gave back the hidden I0002 count (about 394). All
I0002 E1 anchor counts were "<11" for the same reason. This was a small disclosure leak, not a numerator/denominator mismatch.
The totals reconcile (S0001 2,912 + S0002 1,756 + I0002 about 394 + I0003 < 11 = 5,064).

Fix. `silver_report` now sets the per-site prevalence to "<11" whenever the per-site `n_positive` is suppressed, by the
primary or the complementary rule. Tests: `test_per_site_prevalence_is_suppressed_whenever_its_count_is_suppressed`,
`test_no_report_cell_shows_prevalence_without_its_count`. Note: a complementary-suppressed cell still prints "<11" although its
true value can be larger; that is the existing convention. Rerun to refresh `out/silver/silver_report.json`.

## 2. E2_shock (profound shock): the mapping was broken, but the definition is legitimately rare

Diagnosis (cohort 14,516):

| Finding | Result |
|---|---|
| MAP rows matched | 0 at every site. No row anywhere is named like a MAP (searched mean / MAP / arterial / NIBP / IBP names). |
| sbp / dbp rows matched | I0002 only: 660 cases, concept-id route, about 32 readings per case, median spacing about 200 min. S0001 / S0002: 0. |
| S0001 / S0002 BP content | `BLOOD PRESSURE` (concept 1003132, no number, text "SBP/DBP"): S0001 2.76M rows, 7,652 cases; S0002 1.80M rows, 4,636 cases. Same text in `Blood Pressure-Epic` and `Systolic/Diastolic-LFA3959.0`. |
| Numeric Epic rows | `Systolic-Epic` / `Diastolic-Epic` exist (4,362 / 2,862 cases) but are almost all outside [-48, 0] h (158 / 71 cases inside). Their unit is "millimeter of mercury", which was not a recognised alias (rows dropped as `unit_unrecognised`). The LFA numeric rows carry the unit "@". |
| Units | Values are mmHg (q05 / q50 / q95 of systolic 100 / 126 / 160). |

Root causes: (a) the bulk BP stream was a non-numeric "SBP/DBP" text that no rule parsed; (b) the unit spellings "millimeter of
mercury" and "@" were unrecognised; (c) blank-unit Epic rows were dropped (`unit_missing`).

Fixes (mapping only; no threshold changed). `bp_text` section in `anchor_concepts.yaml` plus a split in
`classify_measurements`: a strictly matching "NNN/NN" text becomes one sbp row and one dbp row at the same timestamp
(plausibility from the sbp / dbp specs, sbp > dbp; counters `bp_text_rows_split`, `bp_text_rows_unparsed`). Aliases
"millimeter of mercury" -> mmHg, "@" accepted for map / sbp / dbp, `unit_optional` for the three BP items, and name variants
"Syst BP" / "Dias BP". MAP is computed as DBP + (SBP - DBP) / 3 at the same minute, as before. A real-data classification of the
BP rows with the fixed code gave 966,191 split text pairs (4,172 unparsed text rows), 197,340 MAP readings in [-48, 0] h, and
MAP coverage in the shock window for 10,947 of 14,516 cases (S0001 6,356 of 8,032; S0002 4,327 of 4,968; I0002 264 of 1,504).

Cadence (why the old 15-min gap rule was also wrong). Consecutive-reading spacing in [-48, 0] h at S0001 / S0002: q25 10 min,
median 30 min, q75 60 min; share of intervals <= 15 / 30 / 60 min = 0.37-0.42 / 0.55-0.57 / 0.75-0.77. I0002 median about 200 min.
A 15-min maximum gap between consecutive low readings therefore breaks most genuine runs.

Operationalisation chosen (needs a DECISION_LOG entry, amending D-088): `profound_shock` = MAP < 50 mmHg on consecutive readings
with no normal reading between them, successive low readings at most 60 min apart (`max_gap_min` 15 -> 60), and first-to-last low
reading spanning >= 30 min, in the [-48, 0] h window. Threshold (50) and duration (30 min) are unchanged. The alternative "any MAP
< 50 with vasopressor initiation within 1 h" was measured and rejected (it adds nothing and is not what D-088 means).

Yield on the real data after the mapping fix (cases in [-48, 0] h):

| Measure | Cases |
|---|---|
| any MAP < 50 / 55 / 60 / 65 | 441 / 806 / 1,408 / 2,502 |
| >= 2 readings MAP < 50 (not necessarily consecutive) | 207 |
| low-MAP runs (< 50) with >= 2 readings | 183 runs, median span 5 min, q90 15 min |
| runs < 50 spanning >= 15 min | 26 cases |
| runs < 50 spanning >= 30 min, any max gap 15-120 min | <11 |
| any MAP < 50 reading and any vasopressor initiation in window / within +-1 h | 108 / <11 |

Conclusion: E2_shock stays < 11 cases (< 0.1%) after the fix, but now for a legitimate reason: MAP < 50 is mostly isolated
(about 43% of low readings have normal neighbours) or lasts a few minutes. For scale, runs at MAP < 55 spanning >= 30 min are 25
cases and at MAP < 60 are 105 cases (informational only; thresholds not changed). Expected E2 prevalence is unchanged at about
12.1% (arrest 1,546 and asphyxia 340 carry it). Items for the project lead: whether to keep 50 or move to 55 / 65 (a clinical
threshold, so not changed here); I0002 has too little BP data (median 3.5 readings in the window) for any sustained rule.

## 3a. E6 (CDC Adult Sepsis Event style) funnel, windows as in `silver_anchors.yaml`

The `anchor_firing` block of the report counts only cases that are already positive, so it is not a funnel. Funnel from the
event table (cases; per site S0001 / S0002):

| Component | Cases | S0001 / S0002 |
|---|---|---|
| blood culture drawn [-72, +24] h | 1,523 | 396 / 320 (I0002 805) |
| ... of which qad_ge4 [-72, +24] h | 497 | 290 / 206 |
| vasopressor initiation [-48, +24] h | 1,393 | 801 / 591 |
| lactate measured / lactate >= 2.0 | 8,600 / 4,674 | 4,746 / 2,772 measured |
| creatinine doubling | 159 | 52 / 82 |
| bilirubin >= 2.0 and doubled | 84 | 31 / 45 |
| platelets < 100 and >= 50% drop | 189 | 50 / 127 |
| any organ-dysfunction criterion | 5,723 | 3,250 / 1,884 |
| culture AND organ dysfunction | 793 | |
| QAD AND organ dysfunction (= E6, before the E7 exclusion) | 296 | 177 / 119 |

E6 positives by organ criterion among culture AND QAD cases: lactate 269, vasopressor 41, platelets 12, creatinine < 11, bilirubin
< 11; the sole qualifying criterion is lactate for 241 cases. So 60% of the 497 culture-plus-QAD cases pass; E6 cannot exceed
497 (3.4%) while the culture and QAD definitions stand. The binding step is the blood culture (10.5% of cases) followed by QAD
(33% of cultured cases).

Component checks (none broken):
- Blood culture. Matched by LOINC 600-7 as the source value (S0001 13,090 / S0002 11,045 rows) and by name for I0002. A scan of every
  culture-like name found no unmapped blood-culture test; the unmatched ones are urine, MRSA screen, respiratory, fungal, "ORGANISM"
  and "GRAM STAIN" rows. 28% of S0001 cases have a culture at some time in the prefilter window; only 5% have one in [-72, +24] h.
- Antimicrobials: 7,551 cases have an antimicrobial start in [-72, +24] h; 25,043 of 397,638 rows were rejected on route (about
  6%). I0002 has antimicrobial rows for only 61 cases, so QAD (and E6) is effectively unobservable at I0002.
- Vasopressors: 4,918 cases have a vasopressor row in [-48, +24] h but only 1,393 have an initiation. For most of the rest the
  vasopressor started a median of 17 h before the encounter start (the rule requires start >= encounter start and a 24 h washout).
  This is an operationalisation choice (E6 would gain at most the culture-and-QAD cases without lactate >= 2: about 200), not a
  mapping error; flagged for the project lead, not changed.
- Labs: units and ranges are sound (lactate q50 1.8 mmol/L, bilirubin q50 0.5 mg/dL, platelets q50 190, creatinine q50 0.88 mg/dL);
  coverage in window is 12,622 (creatinine), 11,145 (bilirubin), 12,370 (platelets) cases. Doubling and drop rules are legitimately strict.

## 3b. E5 (metabolic): all anchors map correctly; low yield is legitimate strictness

Per anchor, within its window: cases with the lab measured, cases firing, distribution of the per-case worst value.

| Anchor | Measured | Fire | Worst value q10 / q50 / q90 |
|---|---|---|---|
| ammonia >= 150 umol/L | 2,263 | 55 | 15 / 27 / 67 |
| BUN >= 100 mg/dL | 12,350 | 94 | 9 / 17 / 40 |
| glucose < 50 mg/dL | 10,673 | 39 | 86 / 119 / 202 |
| glucose > 600 mg/dL | 12,347 | 166 | 94 / 138 / 278 |
| sodium < 120 mmol/L | 12,358 | 77 | 132 / 138 / 143 |
| sodium > 160 mmol/L | 12,358 | 74 | 135 / 140 / 146 |
| PaCO2 > 70 mmHg | 3,183 | 90 | 30 / 38 / 52 |
| arterial pH < 7.30 (the paired criterion) | 3,590 | 792 | 7.21 / 7.38 / 7.47 |
| PaCO2 > 70 AND pH < 7.30 | | 81 | |
| arterial pH < 7.10 | 3,590 | 127 | |
| calcium > 14 mg/dL | 12,185 | 27 | 8.1 / 9.1 / 9.9 |
| E5 overall (union) | | 638 (4.4%) | |

Chemistry is measured in about 85% of cases, blood gases in 22-25%, ammonia in 16%. Values are in the stated units (calcium q50 9.1
mg/dL, glucose mg/dL, sodium mmol/L), match counts are non-zero for every anchor, and the E5 union reproduces the report (638). Nothing
to fix; the thresholds are extreme-value anchors by design. Informational sensitivity (not applied): glucose < 70 221 cases, Na < 125
200, Na > 155 171, BUN >= 80 201, ammonia >= 100 125, calcium > 13 44, pH < 7.20 283. The one mapping change that touches E5: the new
"millimeter of mercury" alias also lets Epic PaCO2 rows with that unit convert (they were dropped as `unit_unrecognised`; 32,900 rows
were dropped in total over all items); the size of that gain is unknown until the next run.

## Needs a DECISION_LOG entry (not written)

1. Amend D-088: `profound_shock` maximum gap between consecutive low MAP readings 15 -> 60 min (cadence-based); MAP is derived
   from sbp/dbp (including combined "SBP/DBP" text rows) at S0001 / S0002; expected yield < 11 cases, so E2_shock is practically
   inert and E2 rests on arrest / asphyxia.
2. Concept-map mapping fixes: combined BP text split, unit aliases "millimeter of mercury" and "@" for BP, blank-unit BP kept.
3. Report: per-site prevalence follows the suppression state of its count (complementary included).
4. Open for the project lead (no change made): vasopressor "initiation" requires start >= encounter start; MAP < 50 threshold.

## Expected effect after a rerun

E2 about 12.1% (unchanged; shock < 11). E5 4.4% (unchanged). E6 2.0% (unchanged; ceiling 3.4%). The rerun is needed for the new BP
rows and the report fix; the measurement checkpoints re-key because the concept map changed.
