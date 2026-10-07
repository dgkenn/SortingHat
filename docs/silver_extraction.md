# Structured silver-label extraction (spec)

Code: `sortinghat/labels/concepts.py`, `sortinghat/labels/extract.py`. Config: `configs/anchor_concepts.yaml` (concept map),
`configs/silver_anchors.yaml` (signed-off anchor rules). Tests: `tests/test_labels_extract.py`,
`tests/test_labels_extract_concepts.py`. Built and tested on synthetic / hand-built fixtures only; **no HEEDB data has been
seen and nothing here has been run on real data.**

Scope: silver labels E1, E2, E4a, E5, E6, E7 from **structured OMOP tables only**. Notes (free text) are out of scope: they need an
open-weight model on approved compute (CLAUDE.md rule 4). Silver labels are retrospective training targets and are never model
inputs; the baselines' `as_of` t0 gate is deliberately **not** applied (the anchor windows in `silver_anchors.yaml` bound the
information used, and may extend after t0).

## 1. Pipeline

```
cohort (person_id, t0, SiteID[, case_id]) ---------------------------------------------+
omop_concept -> ConceptIndex (LOINC/ICD/CPT/RxNorm code -> concept_id; UCUM units; Meas Value names)
omop_visit_occurrence -> encounter start per case (visit containing t0, + preceding_visit chain)
omop_measurement / drug_exposure / condition_occurrence / procedure_occurrence / observation
   -> column-pruned, cohort-filtered Arrow batches (data_io.iter_omop_batches), classified batch by batch
   -> flags computed upstream -> banned_evidence exclusions -> anchor event table
anchors.silver_anchor_table -> E1, E2, E4a, E5, E6, E7 (+ e4b_* covariate hints)
```

`extract_silver(source, cohort)`: `source` is a `data_io.LocalStore` / S3 client (streamed; only matched rows are kept in memory)
or a dict of in-memory tables. Result `SilverResult`: `labels` (nullable boolean per case; NA when a label is not computable),
`events` (the anchor event table plus audit columns `source, basis, timing_approximate, evidence_*`), `fired` (case -> label ->
anchor ids), `label_status`, `anchor_status`, `gaps`, `diagnostics` (counts). `labels`, `events` and `fired` are **record-level**:
keep them in memory or under `local_only/`; only `silver_report` output may leave a restricted job.

Tables read: `omop_concept, omop_visit_occurrence, omop_measurement, omop_drug_exposure, omop_condition_occurrence,
omop_procedure_occurrence, omop_observation` (`extract.STRUCTURED_TABLES`), with the columns listed in `extract.COLUMNS` (never
`note_text`). Never read: `reports_findings, omop_note, omop_note_nlp, imaging, eeg_metadata` (`extract.FORBIDDEN_TABLES`);
passing them in a dict is counted (`forbidden_tables_ignored`) and ignored; a spy test asserts the streaming path never opens them.

Run (human, restricted data; develop with `--data data/synthetic` only):
`python -m sortinghat.labels.extract --data DIR|--s3 --cohort cohort.csv --out out/silver [--labels-out local_only/silver.csv]`.
`--labels-out` must be under a `local_only/` directory (mode 0600). Throughput is bounded by `anchors.py`, which evaluates every rule per
case with pandas (about 50 ms per case with any in-window event; cases with none skip it).

## 2. How rows are matched (`anchor_concepts.yaml`)

Per row, first hit wins, recorded as `basis` and counted in the diagnostics:

| basis | meaning |
|---|---|
| `concept` | `*_concept_id` is in the ids that `omop_concept` resolves for the item's `(vocabulary_id, concept_code)` |
| `source_concept` | `*_source_concept_id` likewise (ICD / CPT live there) |
| `source_code` | the `*_source_value` text is the code (`I46.9`, `92950`, a LOINC string); concept ids may be zero-filled (catalogue rule 6) |
| `name` | regex over `*_source_value` (with `name_exclude` for specimen words: urine, CSF, venous, ionized ...); `ExtractConfig(allow_name_fallback=False)` turns it off |

Drugs: union of the concept route (RxNorm / RxNorm Extension `concept_name` regex, plus RxNorm ingredient codes for the five
pressors and naloxone) and a regex over `drug_source_value`. No `concept_ancestor` table is read, so ingredient codes only match rows
coded at the ingredient level.

## 3. Units, plausibility, result semantics

Each quantitative item has one canonical unit (the unit in `silver_anchors.yaml`). The unit comes from `unit_source_value`, else the UCUM
code of `unit_concept_id`, else the unit stored with the matched LOINC code. Conversions (`value = (x + pre_add) * mul + add`):

| item | canonical | converted from |
|---|---|---|
| ammonia | umol/L | ug/dL x 0.58718 (NH3 17.031 g/mol); mmol/L x 1000 |
| glucose, CSF glucose | mg/dL | mmol/L x 18.016; g/L x 100 |
| BUN | mg/dL | mmol/L (urea) x 2.8014 |
| creatinine | mg/dL | umol/L x 0.011312; mg/L x 0.1 |
| sodium | mmol/L | mEq/L x 1 |
| PaCO2 | mmHg | kPa x 7.50062 |
| calcium (total) | mg/dL | mmol/L x 4.008; mEq/L x 2.004 |
| lactate | mmol/L | mg/dL x 0.11101 |
| bilirubin (total) | mg/dL | umol/L x 0.05847 |
| platelets, WBC | 10*3/uL | /uL x 0.001 (K/uL, 10*9/L as is) |
| temperature | C | (F - 32) x 0.5556 (extracted only with `include_unused_items`) |
| ethanol | mg/dL | mmol/L x 4.607; g/L x 100; g/dL or % x 1000 |
| acetaminophen | ug/mL | umol/L x 0.15116; mg/dL x 10 |
| salicylate | mg/dL | mg/L or ug/mL x 0.1; mmol/L x 13.812 |
| CSF protein | mg/dL | g/L x 100; mg/L x 0.1 |

An **unrecognised or missing unit drops the row** (counted as `meas_dropped_unit_unrecognised` / `..._unit_missing`; pH is unitless and
exempt); a value outside the item's `plausible` range after conversion is dropped (`meas_dropped_implausible`). Nothing is guessed:
5.5 with no unit is never read as mg/dL.

Qualitative results (`value_source_value`, else the name of `value_as_concept_id`, else the number where allowed): negative text wins first
(`No growth`, `Not detected`), then pending, then positive. Blood and CSF culture rows carry the organism, so any non-negative,
non-pending text is positive **except common commensals** (coagulase-negative staph, diphtheroids, ...), which never count. `Gram
negative rods` is positive. Numbers make PCR / antibody / tox rows positive when > 0 but never a culture.

## 4. Flags computed upstream

All baselines use the **encounter**: the visit containing t0, extended back along `preceding_visit_occurrence_id` (3 hops, so an ED visit
before admission is included); `t0 - 7 days` when no visit covers t0 (`visit_fallback_cases`). Row prefilter window: `[t0 - 30 d, t0 + 10 d]`
(ESRD codes: unbounded lookback).

| item | rule as implemented |
|---|---|
| `arrest_event` | condition codes ICD-10-CM `I46.*`, ICD-9 `427.5*`, SNOMED 410429000; procedure codes CPT 92950, ICD-10-PCS 5A12012, ICD-9 99.60 / 99.63; observation / procedure wording (cardiac arrest, ROSC, CPR, code blue) minus history / negation wording. Z86.74 (history) never matches |
| `asphyxia_event` | `T71*`, `T75.1*`, `T58*`, `T17*`, `R09.01`, `W65-W70, W73-W81, W83, W84`, `X70`, `X71`; ICD-9 994.7, 994.1, 986, 933, 934 |
| `profound_shock` | MAP series (direct MAP, else DBP + (SBP - DBP) / 3 at the same minute). A run = consecutive readings < 50 mmHg, no normal reading between, gaps <= 15 min; qualifies when first-to-last low reading >= 30 min. Event time = run onset. MAP outside [10, 300] is dropped (0 = disconnected line) |
| `vasopressor_initiation` | norepinephrine, epinephrine, vasopressin, phenylephrine, dopamine, angiotensin II; a start counts when no pressor started in the previous 24 h (rate-change rows are not new starts) and the start is in the encounter. Oral / topical / local-anaesthetic products excluded |
| `blood_culture_drawn` | any blood-culture row (LOINC 600-7, 17928-3, 17934-1 or name) |
| `qad_ge4` | CDC-ASE style: a qualifying antimicrobial day = any calendar day with a qualifying agent (IV/IM, or any route for linezolid, fluoroquinolones, metronidazole, TMP-SMX, rifampin, azoles, doxycycline, minocycline); >= 4 consecutive QADs starting within +-2 calendar days of a culture day, starting with an agent not given the 2 previous days; one missing day between administrations of the same agent is bridged. Exposure days run start..end (capped at 14 d; end dates can be stamped at death). Event stamped at the culture time |
| `creatinine_doubling` | value >= 2 x the lowest EARLIER creatinine in the encounter; cases with ESRD codes (N18.6, Z99.2, ICD-9 585.6, V45.11) excluded |
| `bilirubin_doubling_ge2` | total bilirubin >= 2.0 mg/dL and >= 2 x the lowest earlier value in the encounter |
| `platelets_drop` | value < 100 and <= 50 % of the highest earlier value in the encounter, which must be >= 100 |
| `tox_screen_positive_nontherapeutic` | positive screen (or detected level) for an agent class (opiates, oxycodone, fentanyl, methadone, buprenorphine, benzodiazepines, barbiturates, PCP, amphetamines, cocaine, cannabinoids, TCA) **and** none of that class's in-hospital drugs was started in the encounter at or before the specimen time |
| `antidote_response` | an antidote **given** (naloxone, nalmefene, flumazenil, physostigmine, fomepizole, acetylcysteine, hydroxocobalamin, pralidoxime, digoxin immune Fab) |

### E4a versus E4b

A positive screen for an agent the team gave is **E4b, not E4a** (`silver_anchors.yaml` note). The class-to-drug lists are generous on
purpose (`tox_agents.*.in_hospital_drugs`): a missed E4a costs recall, a false E4a plants an iatrogenic case in the intoxication label.
The same split is applied to antidotes: naloxone / nalmefene after an in-hospital opioid, and flumazenil after an in-hospital
benzodiazepine, are **reversals** (E4b), not E4a evidence. The E4b side is exposed as case-level boolean hints, not anchors:
`e4b_tox_inhospital` (in-hospital-agent positive screen in [-24, +6] h), `e4b_antidote_reversal` ([-6, +6] h),
`e4b_sedative_exposure` (propofol, midazolam, lorazepam, dexmedetomidine, ketamine, barbiturates, etomidate started in [-24, 0] h).
Any drug row counts as an exposure (order versus administration is not distinguished: `drug_type_concept_id` values are unread), which errs
toward E4b. Outpatient prescriptions from before the encounter are **not** excluded by default (`exposure_lookback_days`
extends the exposure window backwards).

### Antidote response: limitation

The anchor name is `antidote_response` (antidote given AND documented response). The clinical response (arousal after naloxone) is not
codable from structured data, so the extractor records **antidote given** (event `source = drug:antidote_given`) and treats it as the
anchor. Consequence: E4a is over-called by antidotes given empirically with no response, and by naloxone / flumazenil / acetylcysteine given for
other reasons (the in-hospital pairing above removes the iatrogenic reversal case only). Status: `proxy`.

## 5. E1: structured alternative (no imaging for the Study 1 cohort)

The names-only probe found imaging only at I0001 and I0004, which have no EEG, so imaging-report flags (`imaging_ich`, `imaging_sah`,
`imaging_sdh_edh`, `imaging_infarct`, `imaging_tbi_contusion`, `imaging_mass_effect`) are **never emitted** here and are recorded as gaps
(`SilverResult.gaps`, `anchor_status = unavailable`). The imaging items stay in `silver_anchors.yaml` for future sites; `extract_silver(...,
extra_events=...)` merges externally produced imaging flags (an `acute` column is honoured; our rows get `acute=True`).

**Project-lead decision (relayed by the coordinator, 2026-10-07; DECISION_LOG entry needed, not written here):** E1 is anchored on

- (a) acute structural ICD-10 diagnosis codes in [-72, +24] h of t0: `I60.*` SAH, `I61.*` ICH, `I62.0*` nontraumatic subdural (chronic
  `I62.03` excluded), `I63.*` infarct, `S06.4*` epidural, `S06.5*` traumatic subdural, `S06.6*` traumatic SAH, `S06.2*` diffuse and `S06.3*`
  focal TBI / contusion (7th character S = sequela excluded), and `G93.5` compression of brain / `G93.6` cerebral edema **only when one of
  the former is present for the same case in the window**;
- (b) neurosurgical procedure codes in the same window: craniotomy hematoma evacuation and craniectomy (ICD-10-PCS `00C0-00C7`, open skull
  excision `0NB00ZZ ...`, CPT 61312-61315, 61154, 61322, 61323), EVD placement (`009600Z`, `009630Z`, `009640Z`, CPT 61210, 61107),
  intracranial thrombectomy (`03CG*`, CPT 61645), thrombolysis (`3E03317`, CPT 37195).

New items `dx_ich, dx_sah, dx_sdh_edh, dx_infarct, dx_tbi, dx_mass_effect, proc_neurosurgical` and leaf ids `E1_dx_*`, `E1_proc_neurosurg`
are an alternative `any_of` branch of E1 in `silver_anchors.yaml`; sublabels: focal = ICH, infarct, SDH/EDH, TBI; diffuse = SAH, mass effect.

Diagnosis timing is imprecise (encounter-level codes): event time = `condition_start_datetime` when present, else the start of the linked
visit (`visit_occurrence_id`), else `condition_start_date` (00:00); the two fallbacks set `timing_approximate` in the event table
(`dx_time_from_visit_start` counts them). The ICD-10-CM and ICD-10-PCS lists were verified against the ICD-10 MCP server (FY2027) on 2026-10-07.
ICD-9-CM equivalents (430, 431, 432.0/432.1, 433.x1/434.x1, 851, 852.x, 348.4, 348.5) and the CPT codes are author-confident and **not**
tool-verified. Caveats: thrombolysis codes also cover PE / MI use (indication not codable); `I62.1` (nontraumatic extradural) is not in the
lead's list and is not matched; E1 status is `partial` (every anchor is a `proxy`).

## 6. Banned evidence and reports_findings

- Events are screened with `banned_evidence`: a nonspecific-encephalopathy ICD code (`G92*`, `G93.4*`, ICD-9 348.30/.31/.39, 349.82) never
  anchors (none is in the concept map; the screen is a second guard on the `condition_source_value` of matched rows); any matched row whose
  text mentions EEG content is dropped (`banned_dropped_eeg_derived`); wording-derived events (observation text) with anchorless
  encephalopathy wording are dropped (`banned_dropped_nonspecific_dx`).
- **`reports_findings` never produces a positive.** It is not an argument of `extract_silver`, is in `FORBIDDEN_TABLES`, and is never
  read by the streaming path. Its only use is `eeg_impression_comparator(reports_findings, cohort)`, which returns `case_id, label, eeg_impr`
  (1 flag asserted, 0 report without flag, NaN no matching report) for the **circularity audit's EEG-impression comparator**. The flag-to-label
  mapping (`eeg_impression_comparator` in `anchor_concepts.yaml`: E1 <- foc slowing, lpd, lrda; E2 <- bs, low voltage; E5 <- gpd) is a
  provisional author guess needing EEG-clinician sign-off; labels with no mapped flag get no comparator rows. `load_reports_findings(store)`
  reads it through `data_io.read_site_table` (I0008 / I0009 have no file).

## 7. Aggregate-only report

`silver_report(result)` returns: overall and per-site silver positives and prevalence for each label, per-anchor firing counts (overall and per
site), label and anchor computability, the recorded gaps, and the diagnostics counters. Every count < 11 is `"<11"`; prevalence is suppressed when
either cell is < 11; if exactly one site cell of a row is suppressed while the total is shown, the smallest other cell is also suppressed
(complementary suppression). The report passes `assert_aggregate_only` (case ids as known ids) before it is returned, and `main()` writes it with
`safe_write_json`. Anchors that never fired are listed as `"<11"`.

## 8. What cannot be computed from structured data

| Item | Status |
|---|---|
| `imaging_ich, imaging_sah, imaging_sdh_edh, imaging_infarct, imaging_tbi_contusion, imaging_mass_effect` | **unavailable** (findings live in radiology report text; no imaging findings table at an EEG site). E1 uses the section 5 alternative |
| `antidote_response` (the response) | only "antidote given" is codable (proxy) |
| E4a / E4b boundary for outpatient prescriptions and order-vs-administration | approximated (section 4) |
| CDC ASE death / hospice / transfer shortcut for QAD < 4 days | not implemented (conservative: such cases are missed) |
| Positivity of cultures, PCR and antibodies | from coded result text only; pending results and unlabelled organisms are not positive; commensals excluded; antibody and PCR rows need a name that names the assay (and CSF for PCR) |
| Diagnosis timing (all `dx_*`, arrest and asphyxia codes) | approximate (coding time, not event time) |

## 9. Items needing a human before a real run

1. Verify every code in `anchor_concepts.yaml` against the local `omop_concept` (`concept_code` + `vocabulary_id`): LOINC (esp. the candidate CSF
   RBC code 26455-6, culture codes 600-7 / 17928-3 / 17934-1, 14749-6 and 14682-9 SI variants), SNOMED 410429000, RxNorm ingredient codes, all CPT
   and ICD-9 lists. Compare the `meas_match_basis_*`, `code_match_basis_*` and `meas_dropped_*` counts: a high `name` share or many dropped units
   means the map needs local codes.
2. Count how often `measurement_concept_id` / `unit_concept_id` are zero-filled; check units by item (aggregate only).
3. Review the antimicrobial list against CDC ASE, the tox class-to-drug lists, and the `exposure_lookback_days` default.
4. Co-investigator re-review of thresholds (already required by `silver_anchors.yaml`) plus the shock gap (15 min) and the E1 code lists.
5. Sign off the provisional EEG-impression flag mapping.

## 10. Tests

Synthetic only (hand-built OMOP fixtures in the test file; the generator has none of these items): unit conversions, the unit-drop rule, name and
concept routes, every flag above, E4a/E4b split (tox and antidote), E1 dx / procedure / mass-effect gating / timing fallback, banned evidence,
that `reports_findings`, notes and imaging never change the labels, the streaming path equals the in-memory path, a spy test that streaming reads
only `STRUCTURED_TABLES`, report suppression and aggregate-only guard, the CLI `local_only/` rule, and a smoke run on generator output.
