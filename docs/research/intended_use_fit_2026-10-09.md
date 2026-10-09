# Is the analysed cohort the right dataset for the intended use? (2026-10-09)

Numbers-free summary (CLAUDE.md rule 6). The suppressed aggregate tables are kept out of git in `out/diag/intended_use_fit.md` and
`.json`; regenerate with

    env -u CLAUDECODE -u SORTINGHAT_AGENT_SESSION scripts/heedb_run.sh python3 scripts/diag_intended_use_fit.py --s3 --max-memory-gb 6

(`scripts/diag_intended_use_fit.py`, tests in `tests/test_diag_intended_use_fit.py`; OMOP-derived proxies are read from the local OMOP
cache, nothing new is streamed; `--s3` is used only for the small patient-merge map).

**Intended use.** An unknown EEG from an ED patient with undifferentiated altered mental status / unexplained impaired consciousness,
little history known: what is the cause?

## Short answer

The analysed cohort is **not** that population, and the strict cohort is the furthest from it. It is, to a first approximation, EEG
monitoring of already-sedated, mostly ventilated, often vasopressor-supported ICU patients who frequently already carry a diagnosis in
the label families. The broad cohort is a mixed acute-care EEG population that contains an ED-like minority; that minority is too
small, and too short of silver-label positives, to estimate EEG increments label by label with the current labels. Results from the
current cohort should be read as "EEG increment over structured baselines in acute-care (largely ICU) EEG", not as evidence about ED
triage.

## What was measured

Populations (pooled and per site, pseudonyms): the cohort table at the study sites before any EEG / baseline / silver requirement (a
contrast, "source"); source rows meeting the strict and the broad definitions; and the rows actually analysed by
`run_silver_feasibility.py` for strict and for broad (baselines present, QC-passing primary EEG window, silver labels).
For each: recording-length class and the continuous-EEG task folder; service class; hours from visit start to t0; covering-visit
length; ventilation, vasopressor and sedative/opioid-infusion proxies from OMOP rows; Baseline A sedation tiers; any label-family
ICD code before t0; and the size of the current undifferentiated subgroup and ten relaxed or stricter variants (U0 to U10 in the
tables), with the silver-label positives each variant could contribute.

The current undifferentiated definition (U0) is reproduced exactly by the diagnostic for both cohorts (checked in the run).

## Findings

1. **Strict cohort: ICU-continuous-EEG, sedated, ventilated.** The large majority of rows come from the continuous-EEG task folder and
   the long-term-monitoring service, and recordings are long-skewed (about half run past 12 h pooled, most at one site). Nearly all have a
   sedative or opioid recorded at t0 (the concern's figure is reproduced); a sedative or opioid infusion in the 6 h before t0 is the typical pattern, not
   PRN doses. A ventilator / intubation record or a vasopressor in the prior 24 h is present in most rows, and most already carry a
   label-family diagnosis code. The undifferentiated subgroup is a small single-digit percentage of the rows (under a hundred).
2. **The strict criterion and the data requirements push toward this phenotype.** Between the source rows and the analysed rows the
   sedated share rises sharply, and strict versus the cohort table roughly doubles the ventilation and pressor proxies. Requiring a
   charted GCS/FOUR at or below the threshold (and the EEG, baseline and silver requirements) selects patients who are charted in an
   ICU, intubated or sedated; ED-style undifferentiated AMS with a low score is rare in it.
3. **Broad cohort: mixed, but ICU-dominated.** A majority is still continuous-EEG / long-term-monitoring service, but the largest
   single recording class is shorter than an hour and only a minority run past 12 h; sedation at t0 is a majority but a continuous infusion is a minority; ventilation and
   pressor proxies are present in a sizeable minority; most rows still carry a label-family code. The current undifferentiated
   subgroup is roughly one row in seven (several hundred).
4. **Timing is the one dimension that largely fits.** Most analysed EEGs fall within the first two calendar days of the visit.
   Caveat: the visit start is date-only (midnight), so "hours since visit start" is over-stated by up to a day; "within 24 h" means
   roughly the same calendar day and "within 6 h" is unmeasurable (almost nobody qualifies by construction).
5. **No direct ED flag exists.** `visit_concept_id` is zero throughout; `ServiceName` takes essentially only a long-term-monitoring
   value and a routine value in this table (no ED or ICU value ever reaches a reportable count); care-site and admitted-from fields are
   not in the local cache; arterial gases are almost never recorded as parsed (uninformative proxy). The cohort build also admits an
   EEG only if the covering visit is longer than a day or the service is acute, so **ED visits treated and released the same day
   are excluded by design** unless their service name is acute.
6. **The undifferentiated subgroup is label-starved.** U0 conditions on the absence of any label-family ICD code before t0 (any
   encounter, ambiguous timing counted as earlier). The silver labels E1/E2 are built from those same code families, so the subgroup
   is almost label-negative: in the strict cohort every label has fewer positives than the disclosure floor in U0 and in all the
   sedation- or time-relaxed variants; in the broad cohort only E1 just reaches the floor in some variants; E5 and E6 reach a few tens
   only in the source rows (before the EEG / QC requirements). Tolerating a known diagnosis (U5) gives about a thousand E1 positives in
   the broad cohort (a couple of hundred in strict), about a hundred for E2 and tens for E5 in broad, everything else below the floor;
   but that is no longer an undifferentiated population.

### Subgroup sizes (qualitative, pooled over the two sites)

| Variant | Strict, analysed | Broad, analysed | Source (cohort table) |
|---|---|---|---|
| U0 current (day 0-1, no label-family code, no sedative/opioid in 6 h) | under a hundred | several hundred | a couple of thousand |
| U3 / U4 (sedation allowed if not an infusion; U4 within 48 h) | about a hundred | about a thousand | a few thousand |
| U8 (within 48 h, no code, no infusion, recording <= 12 h) | about a hundred | about a thousand | a few thousand |
| U6 ED-like proxy (same calendar day, no code, no infusion, not ventilated, no pressor, not cEEG task) | tens at most | a couple of hundred | a few hundred |
| U7 ED-like, sedation-free | tens at most | well over a hundred | a couple of hundred |
| U5 (within 48 h, no infusion, diagnosis tolerated) | a few hundred | a couple of thousand | several thousand |
| U10 (within 6 h of visit start) | below floor | a handful | a few tens |

Label positives inside U0 to U4, U6 to U10: at or below the disclosure floor for nearly every label and cohort (single digits to low
tens). The estimable-subgroup rule of the feasibility script (at least fifty rows and eleven per site) is met by U0 to U5, U8 and U9
in both analysed cohorts (U6 and U7 also in the broad cohort), but not by U6/U7 in the strict cohort and not by U10 anywhere.

## Assessment

* **Distance from the intended use.** Large for strict (spectrum mismatch on setting, sedation, ventilation, severity-selection and
  prior diagnosis); moderate-to-large for broad. The mismatch is mainly in sedation/ventilation/prior diagnosis, less in timing.
* **Best approximating subgroup.** Within the broad cohort, U6 (same calendar day, no label-family code, no continuous sedative or
  opioid infusion, no ventilation or pressor proxy, not a continuous-EEG task) is the closest by construction; U8 / U4 (within
  48 h, no code, no infusion) is the closest subgroup that is large enough to analyse as a group.
* **Large enough?** U6 is a couple of hundred rows pooled (a few tens at the smaller site) and holds almost no
  label positives: usable for describing the case mix, the EEG-derived distribution and sedation-robust performance of an
  all-label or binary "any structural cause" read-out, **not** for label-wise Delta or calibration. U8 / U4 is about a thousand rows but
  still holds few positives per label and still contains ICU-like patients. The strict cohort cannot supply an intended-use subgroup of
  useful size under any variant.

## Recommendations

1. **Make the broad cohort primary and keep strict as a sensitivity set.** Say plainly in the report that the headline estimand is the
   acute-care EEG population (largely ICU), and that the intended-use estimand is a pre-specified secondary stratum.
2. **Pre-specify the intended-use stratum by presentation, not by absence of label codes:** calendar day 0-1 of the visit (the date-only
   proxy for roughly 48 h), no sedative/opioid infusion in the 6 h before t0 (PRN allowed), no ventilation or vasopressor proxy, and
   recording length <= 12 h (or non-continuous task). Report it as U8/U4-style (large) and U6-style (small) tiers with their sizes.
3. **Re-weight rather than only restrict.** Fit a pre-t0 domain model (ED-like stratum versus the rest) on covariates available at
   t0 and independent of the EEG (age, sex, time since visit start, first vitals, GCS/FOUR, service, sedation and
   ventilation/pressor proxies), use stabilised inverse-odds weights with trimming, and report effective sample size; cross-check with
   a stratified (post-stratified) estimate. Report Delta in the stratum and the weighted-to-intended-use Delta next to the all-rows Delta.
4. **Decouple the subgroup from the label source.** The no-prior-diagnosis condition and the code-built silver labels share evidence.
   Either stop requiring "no label-family code" in the stratum and require only that the code postdates t0 (use codes stamped after the
   EEG as the outcome), or add labels whose evidence is independent of ICD codes (the structured anchors that do not use them), and
   run the human check of how many label-family codes have a usable time.
5. **Obtain a real ED indicator.** A human-run, names-and-aggregate probe of `visit_occurrence` for admitted-from / care-site / discharged-to
   (not in the local cache) and ED flowsheet or triage concepts would replace the proxies; and consider whether a cohort variant that
   keeps same-day ED visits with a routine EEG (currently excluded by the acute-care proxy) can be built, since the intended-use
   population is the ED patient who may not be admitted.
6. **Keep the sedation guards.** Sedation-stratified Delta, the sedative-excluded rerun and the sedation leakage probe stay mandatory;
   an EEG signal in this cohort can reflect sedation or ICU monitoring intensity.
7. **Claims.** Until 2 to 5 are done, describe results as exploratory increments over structured baselines in acute-care (largely ICU)
   EEG; make no ED-triage claim.

## Limitations of this diagnostic

* Proxies are free-text and free-vocabulary rules over the local OMOP cache: ventilation is a ventilator / intubation / PEEP / tidal
  volume / ventilation-rate record or an intubation or ventilation procedure code in the prior 24 h; infusion is an infusion-only
  agent, an infusion-like drug text or an administered interval of at least an hour; vasopressors exclude topical and oral forms.
  They are not chart-validated; order-time versus administration-time ambiguity remains.
* Retired patient identifiers are re-keyed through the merge map; OMOP completeness of the cache was checked by part counts only.
* The cohort table is already an adult acute-care proxy cohort (OR/EMU excluded, recording of at least 11 minutes); the "source"
  columns are not all HEEDB EEGs, and routine outpatient EEGs are absent by design.
* The two sites differ (recording length especially); the pooled picture hides site differences that the per-site tables show.
