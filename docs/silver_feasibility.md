# Silver-label feasibility analysis (EXPLORATORY, pre-Gate-0)

**EXPLORATORY — silver-label feasibility; not a test of preregistered hypotheses.**

Status: built and tested on synthetic data only (CLAUDE.md rule 1). Authorised by the project lead; recorded as D-143. Intended-use
block, Baseline P and current-encounter baselines added by D-145 (section 2.3).
Code: `scripts/build_baselines.py`, `scripts/run_silver_feasibility.py`. Tests: `tests/test_silver_feasibility*.py`.
Reads the harness in `sortinghat/models/`, `sortinghat/metrics/`, `sortinghat/baselines/`, `sortinghat/labels/` unchanged.

## 1. Purpose

No gold labels exist yet, and gold adjudication (Phase 0b pilot, kappa, Gate 0) is the expensive step. Before spending it,
the lead wants early evidence on one question: **does the qEEG / connectivity representation carry etiologic signal beyond
what a t0-masked clinical baseline carries?** The analysis trains Study 1A/1B-style models on structured, EEG-blind silver
labels and scores them on the same silver labels, with both validation schemes of D-120 and the plan's mandatory controls.
Gold adjudication is deferred, not dropped.

It answers "is there anything here worth adjudicating for", never "does EEG help". Its numbers are not H1-H6 results, do
not count toward any gate, and must not be quoted without the banner above.

## 2. What it runs

```
cohort_study1.csv ─┬─> scripts/build_baselines.py ──> baselines_AC.parquet (+ .columns.json)   Baselines A, C, P (as_of at t0)
(local_only)       │
                   └─> scripts/run_silver_feasibility.py <── features/part-*.parquet          primary window, QC pass
                                    ^                    <── silver/silver_labels.csv         E1 E2 E4a E5 E6 (E7)
                                    └── reports_findings (circularity comparator only)
                                                         ──> out/silver_feasibility/report.md, report.json  (aggregate-only)
```

Real-data commands (streaming, aggregate-only stdout, D-118; record-level files stay in `out/local_only/`):

```
env -u CLAUDECODE -u SORTINGHAT_AGENT_SESSION scripts/heedb_run.sh python3 scripts/build_baselines.py --s3 \
    --cohort out/local_only/cohort_study1.csv --out out/local_only/baselines_AC.parquet
env -u CLAUDECODE -u SORTINGHAT_AGENT_SESSION scripts/heedb_run.sh python3 scripts/run_silver_feasibility.py --s3
```

(`--s3` on the second command is only for the circularity audit's EEG-report comparator; without `--data`/`--s3` that audit is
reported as skipped.) Synthetic dry run: `python3 -m pytest -o addopts="" -q tests/test_silver_feasibility*.py`.

### 2.1 Baselines (`build_baselines.py`)

* Index from the cohort table: t0 = **metadata EEG start** (D-124; never later than the signal onset, so conservative for
  leakage), age from the cohort, sex from `SexDSC` of the session's `reports_findings`, else `eeg_metadata`. No referral
  indication: Baseline D is dropped (D-107).
* OMOP measurement, drug_exposure, condition_occurrence, procedure_occurrence and observation rows of the cohort's people (and
  of ids merged into them, D-106/D-114) are streamed row group by row group through `data_io.iter_omop_batches`, filtered
  to the cohort inside Arrow, turned into events with the package's own builders, and kept as events only.
* The package's `build_feature_set` then runs unchanged: **`baselines.asof.as_of` stays the single t0 gate.** A memory
  prefilter removes events whose event time is after t0 and charted vitals/scores/pupils/POC glucose older than their
  window plus 1 h; it never admits a row. A test proves the matrix is identical with and without it, and equal to the
  package's in-memory build; a second test injects post-t0 events in every domain (1 s to 400 days) and shows the matrix does not move.
* Outputs: `baselines_AC.parquet` (person_id + the A-C and P columns and the two `meta__` subgroup columns, float64, mode 0600, under
  `local_only/`; the file name is kept for the stage-2 driver) and a names-only sidecar listing which column belongs to A, B, C, P and
  meta, plus the encounter scope. The imaging group is not written (no imaging at the Study 1 sites, D-122);
  NESI stays as a 100%-missing column (no source, D-122) and is dropped inside each training fold.
* Stdout: the encounter scope and how the encounter start was resolved; per baseline (A, B, C, P) the pooled missing rate of its value-type variables overall, by group and by site, and how many
  variables are observed in < 11 patients / missing at least 90% / 50-90% / 10-50% / under 10%. Counts and rates < 11 are suppressed.

**Encounter scope and Baseline P (D-145).** By default every baseline (A, B, C and the new P) uses only events of the CURRENT
encounter: events timed before the encounter start are removed when the event frame is built, so a prior visit's labs, drugs,
vitals, scores or diagnosis codes never reach a model input, and `as_of` then limits what is left to t0. The encounter start is the
cohort table's `encounter_start` (the covering visit chained back along acute visits, as in the cohort builder); a cohort table
written before D-145 has no such column, so `build_baselines.py` re-runs the cohort's visit rule on the visit table, and a person
no visit covers gets t0 minus 3 days. The build log counts the three sources and the removed events. `--with-history` writes the
labelled sensitivity variant that also allows prior-encounter events (`encounter_scope` in the sidecar; the report says "with
history"; a sidecar without the field is reported as "not recorded"). **Baseline P "Presentation"** = age, sex, the GCS / FOUR /
RASS nearest to t0 in [-6 h, +1 h] (the cohort's strict-severity rule; the one bounded exception to the t0 mask, a second call of
the gate restricted to the `score` domain, `presentation_score_after_h`), and the first vitals and point-of-care glucose of the
current encounter before t0. No diagnosis codes, no history; not nested in A to C. The parquet also carries two `meta__` columns
(hours from the encounter start to t0; a primary-label-family ICD code recorded at or before t0, any encounter) that define the
subgroup and are listed under "meta" in the sidecar, never in a model set. Streaming keeps the first vital and glucose of the
encounter through the memory prefilter (the earliest older row per person and key) and the +1 h score grace; a test proves the
streamed matrices equal the in-memory build with and without the prefilter.

### 2.2 Analysis (`run_silver_feasibility.py`)

**Cohort and join.** Strict cohort (`in_strict`, D-105) at S0001 and S0002 (D-120) by default (`--cohort-def`, `--sites`).
Each cohort row is joined to (i) its Baseline A-C row, (ii) the extractor's **primary-window row passing QC**, found through
`recording_id = "rec" + sha256(edf_key)[:20]` with `edf_key = data_io.edf_key_for_row(...)` (pinned to the extractor by a
test; `--recording-map` overrides), (iii) its silver-label row. Rows with no assessable analysed label drop out. Feature
parts are read batch by batch with the window and recording filters applied inside Arrow. Only `qeeg.*` and `conn.*` columns
are model inputs; QC fields, onset offset and usable fraction are never inputs.

**Labels.** E1, E2, E4a, E5, E6 are primary. E7 is primary only with >= 100 positives over >= 2 sites (`e7_eligible`),
otherwise exploratory: trained and reported per label but outside the pooled Delta. E3 (partly EEG-defined, D-002) and E4b
(iatrogenic sedation, D-003) are never loaded as labels. A label with < 11 positives or negatives at any site is not analysed
and the report says why. Missing/unassessable silver values are masked, not treated as negatives.

**Models.** The harness ladder: prior, qEEG, connectivity, and `combined` (qEEG + connectivity, the top available rung and
the headline). MORGOTH, CBraMod embeddings and dynamics are placeholders and are skipped. Each rung is fitted twice with the
same shallow head, grid and recalibration: baseline only and baseline + EEG, for **Baseline A** (H1-style) and **Baseline C**
(H2-style).

**Roles.** Silver labels fill both the "silver" and the "gold" slots of `ModelData`. 20% of rows (`--dev-frac`) are
development rows: inside a training fold they choose C and the EEG shrinkage and fit the recalibrator, and they do not train
the head. Everything else is an evaluation row, scored only when it is a test row of the fold. Dev rows are drawn at random
within site (LOSO) or from the training region only (temporal), so no test-region row is ever a dev row.

**Validation (D-120).** (a) Leave-one-site-out across the two sites; (b) late-calendar temporal holdout within each site (latest
20% test, `metrics.splits.late_temporal_holdout`). Both are reported; neither is preferred.

**Delta and intervals.** Delta = masked log loss(baseline + EEG) - masked log loss(baseline) on the same evaluation rows,
primary labels only; negative = EEG helps. Primary interval: **99% within-site stratified bootstrap** (D-095;
`docs/research/evaluation_sample_size.md` section 8 still recommends it) with the **every-site rule**; 95% within-site,
cluster, two-stage and site-level intervals are co-reported. The report prints whether the H1/H2 *pattern* (99% CI upper bound < 0
and Delta < 0 at every site) occurs, labelled descriptive. Default B = 4,000 (D-141's 10,000 is for final analyses).
Also reported: per-site Delta, per-label Delta with 95% CI, per-label AUROC (pooled and per site, suppressed when a class has
< 11), AUROC difference (+EEG minus baseline) with a paired within-site bootstrap interval, and calibration (O/E, CITL, ECE, Brier,
and slope only for labels with >= 100 events, D-094).

**Intended use first (D-145).** See section 2.3; that block is the first thing in `report.md`, `report.json` and stdout, ahead of the
A / C comparisons below.

**Mandatory controls** (headline rung, both schemes, both baselines):

| Control | How |
|---|---|
| Leakage probes | cross-validated AUROC predicting site, recording duration (above median) and usable channel count (10 minus missing/dead minimum-set electrodes) from the EEG features alone; flag > 0.70 (PLACEHOLDER). Also run on the frozen CBraMod embedding and the MORGOTH findings when those rungs ran. An informational **sedation** probe (the t0 sedation flag, pooled and within each site; needs >= 11 per class) is added to each block; it is not a pass/fail control |
| Site concentration | share of the pooled gain from the largest site; a failed control = flagged site probe and > 60% from one site |
| Sedative-excluded subset | ladder rerun without patients with t0 sedative/opioid exposure (Baseline A `on_t0` flags and/or the silver `e4b_sedative_exposure` hint, `--sedation-source`); trains AND scores without sedated patients, so it is badly underpowered when sedation is common (kept unchanged for continuity; see sedation confounding below) |
| Severity strata | Delta in three GCS-equivalent strata fixed here a priori (GCS <= 5, 6-8, >= 9; FOUR mapped as 3 + 0.75 x FOUR where no GCS exists); strata < 50 patients not estimable; patients with no score form an "unknown" group, never the top stratum |
| Shuffled-label negative control | silver and evaluation labels permuted within site x role, `--null-reps` times; Delta ~ 0 expected |
| Permuted-EEG negative control | EEG rows permuted within site; Delta ~ 0 expected |

**Sedation confounding (report section "Sedation confounding").** The main worry is that part of the EEG gain is sedation: EEG sees
sedative effects and sedation correlates with the silver labels. The sedative-excluded rerun is underpowered (it removes the sedated
patients from training and scoring), so three further analyses are reported, none of which refits a model:

1. *Sedation-stratified Delta.* The headline-rung per-patient differences d_i of each scheme x baseline (the very ones behind the
   headline Delta) are split by the sedation flag (`cov["sedated"]`, i.e. `sedation_flag` with `--sedation-source`) and Delta is
   re-estimated in the sedated and non-sedated strata, with 99% and 95% within-site patient bootstrap intervals, per-site Delta, and
   stratum n (< 11 shown as `<11`; when either stratum is < 11 both counts are hidden). A stratum under 50 patients is *not
   estimable* (no Delta, no interval; with LOSO about 20% of rows are development rows and the temporal scheme scores only the late
   20%, so the sedated stratum is often the one that falls under 50). The *difference of Deltas* (sedated minus non-sedated) comes
   from the same bootstrap replicates (patients resampled within site over both strata, both means computed on each resample), so it
   is paired by bootstrap; it is estimable only when both strata are. Reading: if the gain were only a sedation proxy, Delta would
   be about 0 in the non-sedated stratum; the report prints whether its 99% interval lies below 0 ("gain persists"). An interval
   that includes 0 is inconclusive, not exculpatory (smaller stratum). Strata split the *held-out rows* only; the models are the
   ones fitted on all training rows. Code: `sortinghat/metrics/sedation.py`.
2. *Sedation-adjusted Delta.* Baseline A is "severity + sedation" (`sed__<agent>__on_t0`, `__qty_6h/24h`, `sed__<class>__on_t0` and
   `__n_24h`, `sed__n_agents_24h`) and C contains A, so Delta is already the EEG increment over *recorded* t0 sedation exposure; the
   report states this from the actual column sets of the run (column counts only) and flags the case where a set has no sedation
   column. What stays confounded is sedation the recorded features miss (a bolus that ended before t0, no administration record, depth
   of sedation, and patients flagged only by the silver `e4b` hint, whose count is reported). A sedation x EEG *interaction* variant
   is **not run**: it needs a refitted grid for every scheme x baseline x fold.
3. *Sedation leakage probe.* AUROC of predicting the sedation flag from the EEG representation alone (hand-crafted qEEG + connectivity,
   frozen CBraMod, MORGOTH findings when present), pooled and within site, in the leakage-probe table. A high AUROC means the
   representation encodes sedation and that sedation is a live explanation of Delta; a low one makes it less plausible. A pooled AUROC
   can ride on a site shift when sedation prevalence differs by site; the within-site AUROCs cannot.

**Checkpoints.** None of these adds a model fit: they read the stored `d_i` / predictions, so a resumed run reuses every cached fit
(fit units are keyed by data, not code). No new command-line argument was added, so the step's argument digest is unchanged. The step key
does contain the code version (`git HEAD` + hash of the uncommitted tracked diff), `--out` and the other arguments: a commit or edit
of any tracked file, or a different `--out`, starts a fresh checkpoint directory and refits everything once. To reuse the fits of an
earlier run across this change, relaunch with the same arguments and `SORTINGHAT_CODE_VERSION` set to the earlier run's code version string.

**Circularity audit** (`labels.circularity_audit`, plan: "Silver labels: the circularity rules"). For every case, the
leave-one-site-out prediction (the model never trained on that site) is compared, by AUROC, with (i) the EEG-report finding
mapped to the label (E1 <- foc slowing/LPD/LRDA; E2 <- BS/low voltage; E5 <- GPD; the mapping is provisional and needs
EEG-clinician sign-off) and (ii) the silver label. The silver label stands where the plan's audit uses gold. It is run for the
baseline-only and the headline EEG model of each baseline. `reports_findings` is read only here and never produces a label.

### 2.2.1 Label drift screen (D-153)

For every analysed label, in the temporal scheme (`late_temporal_holdout`: within site the latest `--temporal-fraction` of t0 is test), the report
compares prevalence among ASSESSABLE test rows with assessable train rows, per site and pooled, and flags a test/train ratio above `--drift-ratio`
(default 3) or below its inverse. Section "Label drift screen" in `report.md`, key `label_drift` in `report.json`: counts and prevalences per side,
the ratio, the flag (`null` = not estimable). A flagged label moved over the record period (data availability, coding or case mix), so its temporal
O/E, Delta and AUROC describe that drift and not the model; read it under leave-one-site-out. Counts < 11 are `<11`; a prevalence needs both of its
cells (positives and negatives) >= 11 and the ratio needs both prevalences, so a suppressed count cannot be recovered; a pooled count is hidden when
exactly one per-site count of the same kind is hidden. Not estimable is not stable: a label whose test positives fall below 11 is reported as such.
The screen reads labels only; it does not change model fitting. With the E6 availability rule (`docs/labels_spec.md` section 4) the culture-free early
era is not assessable, so E6 should no longer be flagged by the data feed.

### 2.3 Intended-use analyses (D-145)

The intended use is undifferentiated altered mental status or unexplained unconsciousness in the ED, with little or no history:
"given an unknown EEG, what is the cause?". The first reported block, **Intended use: undifferentiated AMS**, asks that question
with baselines the intended user would have. For the headline rung and both validation schemes it reports Delta = masked log loss(set
+ EEG) - masked log loss(set) on the same evaluation rows (99% within-site interval, 95% interval, per-site Delta, every-site rule):

| | Reference set | EEG model |
|---|---|---|
| a | the smoothed training prevalence (no features) | EEG only |
| b | age and sex (A's demographic columns) | EEG + age/sex |
| c | Baseline P (Presentation) | EEG + P |

All three use the same head, grid and recalibration as the A / C comparisons. They are reported twice: for the whole analysed set,
and for the **undifferentiated subgroup**, defined from the cohort and baseline columns and requiring all of:

1. EEG on encounter day 0 to 1: 0 <= t0 - encounter start <= 36 h (`--undiff-max-hours`; the cohort's `encounter_start`, else the
   `meta__hours_since_encounter_start` that `build_baselines.py` resolved for an older cohort table);
2. no ICD diagnosis code of the primary label families recorded at or before t0, in any encounter (acute structural dx, arrest,
   asphyxia: the `code_event_items` with condition codes that are anchor items; an unknown-time code counts as recorded, a date-only
   one counts from the start of its day);
3. no sedative or opioid exposure in the 6 h before t0 from the Baseline A sedation features (running at t0, or a recorded quantity
   attributable to the 6 h window).

A component that cannot be established counts as not met, so missing data never enlarges the subgroup. The report gives the
subgroup size (overall and per site, suppressed below 11) and how many patients meet each component. The subgroup comparisons train
on all training-fold rows and score only subgroup rows (the dev draw is unchanged); a scheme with fewer than 50 scored subgroup rows
or fewer than 11 at a site is "not estimable". Shuffled-label and permuted-EEG negative controls run for the three comparisons on
the whole set. **EEG-derived information is never an input or label evidence:** no baseline, P or meta column is derived from EEG, the
silver labels are EEG-blind (E3 and EEG-report flags excluded), and `reports_findings` feeds only the circularity audit.

Reading it: (a) says whether the EEG carries etiologic signal at all; (b) and (c) say whether it adds to what is known at the door.
The A and C comparisons that follow answer a different question (an EEG recorded after a workup) and stay as before. A favourable
(a) with an unfavourable (c) means the presentation already carries what the EEG carries on these labels.

Limits specific to this block: the subgroup is a restriction on the same silver labels, so structured ascertainment is thinner
there (little coded workup by construction) and the labels that need the workup (E5 labs, E6 cultures) are the least ascertained; a
sedative bolus with no recorded quantity that ended before t0 is invisible to the Baseline A sedation features; ICD codes are timed
at their start date, which can precede the coding time; encounter starts are date-granular because visits are date-only (D-112);
the +1 h score window uses data charted after the EEG began (P only).

## 3. Deviations from the plan

| Plan / earlier decision | This analysis | Why it is acceptable here |
|---|---|---|
| Silver labels train; they never score (plan, SAP) | Silver labels also tune and **score** | No gold exists; the question is feasibility, not accuracy. Scores are agreement with the label source |
| Nothing is trained before Gate 0 (D-039, D-125) | Models are trained pre-Gate-0 | Lead's explicit direction, D-143; no result moves a gate |
| Model selection on gold development cases (D-098) | Selection and recalibration on 20% silver development rows | No gold; the dev rows are the same noisy labels |
| H1-H6 preregistered tests | "H1-style" and "H2-style" Delta only, descriptive | Pre-Gate-0, two sites, silver truth, exploratory banner |
| Three sites (D-024) | Two sites (D-120); the every-site rule is a two-site rule | Only S0001 and S0002 have labelled patients |
| MORGOTH / CBraMod / dynamics rungs | Skipped | Placeholders (no loaders on approved compute) |
| E3 as positive control (SAP 10.8c) | Not run | E3 is not a silver label here (D-002) |
| H3 cut points fixed before unblinding (D-132) | Fixed in the script before any run | Three GCS-equivalent strata, see 2.2 |
| Bootstrap B = 10,000 for final analyses (D-141) | B = 4,000 default | Exploratory |
| Baseline D (D-107), imaging in C and NESI in A (D-122) | Dropped / absent | As decided |
| Baselines A to C may use prior-encounter data (earlier behaviour) | Current encounter only by default; `--with-history` is the labelled variant (D-145) | The intended user has no prior history |
| Baselines are t0-masked | Baseline P reads GCS / FOUR / RASS up to 1 h after t0 (D-145) | The cohort's own severity window (D-105); `presentation_score_after_h = 0` restores strict masking |
| Cohort, baselines and silver at metadata start; features at signal onset (D-124) | Same | Baselines precede the feature window |

## 4. Interpretation limits (read before quoting anything)

1. **Circular by construction.** Silver labels train, tune and score. A model can score well by learning the label
   source's habits (who is coded, who is scanned, who has labs drawn), not physiology.
2. **Baseline-anchor overlap cuts both ways.** Baseline C contains labs, cultures, vitals and history flags that are also
   silver anchors (E2 shock from MAP, E5 labs, E6 cultures/antimicrobials, arrest history). Where the baseline sees an anchor, baseline
   AUROC is very high and any EEG increment is uninterpretable; the report lists labels whose baseline-only AUROC is >= 0.90. Where it
   does not (E1 from diagnosis codes), EEG may pick up what the codes encode (structural lesions are visible on EEG), which is signal
   but is not independent of the label source.
3. **Structured-only silver (D-123).** Toxic, infectious and metabolic etiologies are under-ascertained; E1 rests on codes with
   approximate timing; E4a "antidote given" is a proxy. Label noise is not random: it is correlated with care intensity,
   which also drives EEG availability and quality.
4. **Two sites.** Two LOSO folds; cluster and site-level intervals are degenerate or extremely wide (the site-level interval
   has 1 degree of freedom); between-site variance cannot be estimated. A favourable result at both sites is a weaker statement
   than at three.
5. **The temporal scheme may not be a temporal scheme.** HEEDB dates are shifted. If the shift is per patient (the Phase 0a
   alignment gate checks only within-patient consistency), "late" cases are an arbitrary subset and this scheme is a random split.
   Do not read agreement between schemes as proof of stability over time.
6. **Selection on the same noise.** Development rows share the silver noise, so selected hyperparameters and the
   recalibration are tuned to it. Calibration here describes agreement with silver labels only.
7. **Delta is not the whole story.** A negative pooled Delta can come from one or two labels (see per-label Delta), from a
   proxy for care intensity or sedation (see the controls) or from a recording-property leak (site/duration probes).
8. **Selection of the analysed set.** Only patients with a QC-passing primary window, a baseline row and an assessable label are
   analysed. QC failure is related to sedation, agitation and electrode problems, so the analysed set is not the cohort.
9. **Multiplicity.** Many rungs, labels, baselines and schemes are reported with no correction; a single interval below 0 in
   a table of 100 is expected by chance.

10. **Label availability over time (D-153).** A label can only be ascertained where its source data exist. E6 rests on blood-culture rows, which are
   nearly absent early in each site's record period; those cases are now not assessable (NaN) instead of negative, and the label drift screen flags any
   label whose prevalence moves more than 3-fold between train and test. A silver run made before D-153 codes that era as negative: its temporal E6
   O/E (~14), Delta and AUROC are feed-era artefacts.

### How to read the report

* Look first at the controls. A shuffled-label or permuted-EEG Delta away from 0, a flagged site/duration probe with the gain
  concentrated in one site, or a sedative-excluded Delta that vanishes means the headline Delta is not interpretable. Then "Sedation confounding": a non-sedated-stratum Delta near 0 (99% interval including 0 while the sedated stratum is clearly negative), a difference of Deltas away from 0, or a high sedation-probe AUROC all point to sedation as part of the gain; a non-sedated Delta that stays negative with a 99% interval below 0 argues against sedation as the whole explanation (it cannot rule out sedation the baseline features miss).
* Then the per-label table. A gain on E1/E2/E5 with no gain on E4a (no EEG signature expected) is the pattern the
  planted-signal test shows; a gain on every label equally suggests a recording-property proxy.
* Then baseline-only AUROC (anchor overlap) and the circularity audit. A flag for the **baseline-only** model (which never
  sees EEG) points at contamination of the silver label by EEG-report content; a flag only for the EEG model is expected if it reads
  the waveform findings the report describes and is not a leak by itself.
* A result favourable under both schemes, both baselines, with clean controls and gains on labels the baseline does not anchor, supports
  spending adjudication effort. Anything less does not argue against it either: the analysis cannot
  show EEG is useless.

## 5. Safety

Inputs are record-level and live under `local_only/` (the scripts refuse other paths; `--out` refuses `local_only/`). Stdout, `report.md` and
`report.json` pass `assert_aggregate_only` against every person and recording id; counts < 11 are `"<11"`; sites are `site_1`, `site_2`
in lexicographic order; no per-patient array leaves memory (a test checks no list in the JSON is longer than 50). Nothing reads notes. Failures print the
exception class only. No LLM sees any record.

## 6. Open items for a human

* Confirm the provisional EEG-report-to-label mapping (`configs/anchor_concepts.yaml`) with an EEG clinician before reading the audit.
* Decide whether the 20% dev fraction, the 0.70 leakage threshold and the GCS strata stay.
* Check what share of `reports_findings` rows carry a usable patient id at S0001/S0002 (the audit prints coverage; blank
  `BDSPPatientID` rows cannot be matched).
* If cohort sizes allow, repeat with `--cohort-def broad` as a sensitivity analysis.
* Run `build_baselines.py --with-history` to a second parquet and `run_silver_feasibility.py --baselines <that file> --out <another dir>`
  for the "with history" sensitivity variant; compare it with the default (current encounter only) report.
* Check the `encounter start from {...}` line of the baseline build: 'cohort' or 'visits' for (nearly) everyone; many 'fallback' rows
  mean the covering visit was not found and the current-encounter restriction used t0 minus 3 days.
