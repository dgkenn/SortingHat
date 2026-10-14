# Gold adjudication pilot for the ED-like stratum IU-large (D-157)

Two clinician raters adjudicate the etiology of impaired consciousness for 200 sampled IU-large patients, blinded to EEG, EEG reports and
model output. This is the Phase 0b label pilot of `docs/research_plan_v1.txt` ("the handoff's 200-case blinded sample stays, with three added
outputs: measured minutes per case, per-label kappa, and the share of positive silver labels whose only evidence was EEG-derived") run on the
stratum the intended use is about. It produces the first gold labels in the project; none exist before it.

Code: `sortinghat/adjudication/` (sampling, blinding, sources, packets, crf, agreement, aggregates, private_io); scripts
`scripts/adjudication_sample.py`, `adjudication_packets.py`, `adjudication_redcap.py`, `adjudication_analyze.py`; configs
`configs/adjudication.yaml` (design, fixed before the draw) and `configs/adjudication_crf.yaml` (the case report form); tests
`tests/test_adjudication_*.py`. Everything was built and tested on synthetic data only.

## 1. Design

**Population.** IU-large (`docs/silver_feasibility.md` section 2.4, D-155) rows of the BROAD cohort at S0001 / S0002, i.e. the rows the silver
feasibility analysis scored: t0 on encounter calendar day 0 or 1, no sedative / opioid continuous infusion running at t0, no ventilation or
vasopressor proxy in the 24 h before t0 (local OMOP cache), recording <= 12 h or not a cEEG task folder. The rule uses presentation only: no
diagnosis code, silver label or EEG-derived quantity. It approximates, and does not identify, the ED population (HEEDB has no ED flag).

**Sample.** N = 200 (`sampling.n`), seeded (`sampling.seed`), stratified by site and by SILVER-LABEL PATTERN to enrich rarer etiologies:

| Pattern | Rule (first match in this fixed order) | Target share of a site's sample |
|---|---|---|
| E7, E2, E4a, E6, E5 | silver-positive for that label and for no label earlier in the list | 0.10, 0.12, 0.12, 0.14, 0.14 |
| E1 | silver-positive for E1 and none of the above | 0.16 |
| all_negative | no silver-positive label (a label with no assessable value is ignored) | 0.22 |

The order is the rarest-first order fixed a priori (not read from the data). A stratum smaller than its target is taken whole and the shortfall
goes to the other strata of the site in proportion to their shares. Site sizes are proportional to the IU-large population. Rows are drawn
without replacement with `numpy.random.default_rng([seed, crc32(stratum)])`, so the sample is a pure function of (frame, seed, config).

**Weights.** w = N_h / n_h per site x pattern stratum (Horvitz-Thompson). Sum(w) over the sample = the IU-large population, so any gold
prevalence is re-weighted to IU-large. Strata are defined by the SILVER pattern: the enrichment is an artefact of the sampling frame, not
a gold-label property, and every silver-vs-gold number in the analysis is weighted. The Kish effective sample size and design effect are
printed by the sampler.

**Extra packets.** 10% duplicates (20 patients re-issued under a new pseudonymous case id, shown to the rater at least one batch later;
intra-rater reliability) and 5 calibration cases drawn from IU-large patients NOT in the sample (rater training; excluded from every analysis).

**Realised design (real run, 2026-10-14).** Aggregates (cohort and IU-large sizes per site, per-stratum sampled counts, sampling weights, effective sample size and design effect) are in `out/adjudication/sample_summary.json` (gitignored; CLAUDE.md rule 6), with counts below 11 suppressed. In outline: the rare silver strata are small, so every silver-positive E7 / E2 / E4a / E6 / E5 patient was taken whole (weight 1) and the remaining places went to silver-E1 and silver-negative patients; the gold prevalence of the rare etiologies in the sample will therefore be driven by their silver-positive strata plus whatever the raters find among silver-E1 and silver-negative patients; E7 will almost surely stay exploratory (D-004).

**Packet availability and size (real dry run, 200 sampled cases).** Aggregates are in `out/adjudication/` (gitignored). In outline: diagnoses, procedures, medications, flowsheet observations, age / sex and note timestamps are available for nearly every case; neurological / sedation scores, vital signs and labs for most. The packets are large (dozens of notes and over a hundred diagnosis rows per typical case). Reading that in 10-15 minutes is not
realistic without triage, so the packet has a note index and one row per diagnosis code, and `packets.notes_after_days` can end the notes window earlier
(for example 7 days; log it). The calibration batch and the first interim analysis show the measured minutes; if the median is above 25 minutes shorten
the notes window before the main batches. Note text itself was not read or counted: the OMOP cache holds note timestamps only (section 3).

**Relation to SAP Amendment A1 (D-156).** The pilot sample is NOT the evaluation set (its sampling frame is amendment A2) and its gold labels do
not tune any model. All nine labels are still adjudicated, because the original all-label endpoint stays a co-equal sensitivity analysis and the
silver labels of E4a, E6 and E7 are still validated. The analysis reports Gate 0 for the plan's five families and, separately, kappa and re-weighted
gold prevalence for the A1 primary labels {E1, E2, E5} against the same two thresholds; how Gate 0's "at least four families" reads on a three-label
set is open item [OP-A1], so no pass or fail is called for it. The A1 text says the Phase 0b pilot (kappa and prevalence) is the only new information that
could prompt a further label-set change, and that any such change needs a dated amendment before the evaluation set is unblinded.

## 2. Files and who may open them

| File | Content | Who |
|---|---|---|
| `out/local_only/adjudication/sample.csv` | sample ids, person ids, t0, silver pattern and labels, weights, role | study team only; NEVER to raters (the pattern is the silver label); never printed or read by an agent |
| `out/local_only/adjudication/id_map.csv` | pseudonymous case id <-> sample id, role, duplicate_of_case, person id | study team only |
| `out/local_only/adjudication/packets/<case_id>/` | `packet.html`, `structured/*.csv`, `notes/note_NNN.txt` | the rater assigned the case (give a rater the case folders only) |
| `out/local_only/adjudication/manifests/<case_id>.json` | section counts, rows dropped by reason (names the EEG reasons) | study team only; not inside the packet folder on purpose |
| `out/local_only/adjudication/assignments.csv` | per-rater order and batch | study team, coordinator |
| `out/local_only/adjudication/redcap_case_ids.csv` | case ids to pre-load as REDCap records | study team |
| `out/adjudication/*.json` | aggregates only (n < 11 shown `<11`) | anyone on the team |

Directories 0700, files 0600, always under `local_only/` (gitignored). No agent session reads any of the record-level files (CLAUDE.md rules
1, 3, 5): the sampler, the dry run and the tests print counts only.

## 3. Commands

```bash
# 0. CRF -> REDCap data dictionary (no patient data)
python3 scripts/adjudication_redcap.py                     # writes docs/adjudication_redcap_dictionary.csv; --check fails if stale

# 1. Sample (done 2026-10-14; reads only local_only inputs; aggregate-only stdout)
env -u CLAUDECODE -u SORTINGHAT_AGENT_SESSION scripts/heedb_run.sh python3 scripts/adjudication_sample.py --max-memory-gb 10

# 2a. Packet availability dry run: counts only, local OMOP cache, never streams, writes no packet / no note text
env -u CLAUDECODE -u SORTINGHAT_AGENT_SESSION scripts/heedb_run.sh python3 scripts/adjudication_packets.py --dry-run --max-memory-gb 10

# 2b. Full packet build: A HUMAN, in a plain terminal on approved compute. The OMOP cache holds note TIMESTAMPS only, so the note text is
#     streamed for the ~225 sampled people from the access point (large: it reads person_id of every row group of `note`; run it overnight).
scripts/heedb_run.sh python3 scripts/adjudication_packets.py --s3 --notes-source store --max-memory-gb 12
#     Without note text the structured part is still built and every packet says NOTES PENDING (`--notes-source none`, the default for a build).
#     From a local export of notes (parquet / csv with person_id, note_datetime, note_title, note_source_value, note_text):
scripts/heedb_run.sh python3 scripts/adjudication_packets.py --notes-source dir --notes-dir <dir>

# 3. After rating: aggregate analysis (rater exports are record-level: keep them under local_only/)
python3 scripts/adjudication_analyze.py --ratings out/local_only/adjudication/ratings_R1.csv out/local_only/adjudication/ratings_R2.csv \
    [out/local_only/adjudication/ratings_R3.csv]
```

Nothing is summarised and no LLM reads a note (rule 4): the packet builder redacts by rule and a human reads the result. The optional
open-weight summariser of the plan (validated on 50 cases) is NOT part of this pilot.

## 4. Packets and blinding

A packet holds, with every time RELATIVE to t0 (hours; negative = before) and no absolute date, site or patient id: age and sex; diagnoses
(window [-72 h, +30 d]; earlier rows up to 365 days listed as "prior history"); procedures; medications overlapping the window; neurological
and sedation scores (GCS, FOUR, RASS, NIHSS, pupils; every value); vital signs (6 h bins: n, median, min, max); every other measurement; flowsheet
observations; and clinical notes in the window. Window and caps are in `configs/adjudication.yaml`.

Blinding is layered and conservative (`sortinghat/adjudication/blinding.py`; a false positive costs a sentence, a false negative breaks blinding):

1. **Note level.** A note whose title or source value names an EEG, neurophysiology, epilepsy-monitoring, LTM, evoked-potential or sleep-study
   service is never loaded.
2. **Sentence level.** Every sentence of every other note that mentions EEG-derived content is dropped: the silver labels' own filter
   (`sortinghat/labels/eeg_filter.py`: EEG and prefixes, slowing, triphasic, burst suppression, epileptiform, periodic discharges, ...) plus EEG-derived
   monitors (BIS, SedLine, PSI, suppression ratio, spectral edge), neurophysiology / epileptology vocabulary, electrodes / montage, evoked
   potentials / SSEP, and phrases such as "monitoring showed". Identifier-shaped tokens (`sub-...`, `ses-...`, site-id numbers) become `[id]`.
3. **Structured rows.** Rows whose text mentions EEG-derived content are dropped, as are EEG procedure codes (CPT 957xx, 95812-95830, 95950-95962;
   ICD-10-PCS 4A10*; ICD-9-CM 89.14 / 89.15 / 89.19), the abnormal-EEG diagnosis codes (R94.0*, 794.0*) and the nonspecific encephalopathy
   diagnoses (G92*, G93.4*, 348.3x, 349.82; D-006), which can be assigned after an EEG.
4. **Final audit.** The text of every table and note is re-checked; a packet that still mentions EEG-derived content, or contains the patient's own
   person id, is withheld (listed in `id_map.csv` as `packet_status`; the count is printed).

**Residual risk, stated plainly.** Indirect references that name no EEG word survive ("seen by neurology, pattern consistent with ..."), and the
clinical course itself carries EEG consequences (an antiseizure loading dose, sedation started to treat a seizure). The data window runs to
+30 days on purpose: the rater judges the best-supported etiology, and the CRF records when the information made it certain. Raters report
any EEG information they saw (field `eeg_seen`); the analysis counts those forms. Before release a coordinator reads 10 packets as a blinding
check and amends `blinding.py` if anything slips through.

Date-only rows (a timestamp at exactly 00:00:00 is treated as date-only, as elsewhere in the code base) are shown as day offsets and are
accurate to +/-24 h.

## 5. Rater instructions

You are one of two independent raters. Work alone; do not discuss cases with anyone until the study is unblinded. You do not know and must not
try to find out what the EEG showed. If a packet shows EEG information (a report, a finding, a BIS value), set `eeg_seen`, do not use it, and
do not copy the text anywhere.

1. Open the packet (`packet.html`; the CSV files hold the same tables with every value). Hour 0 is t0, the index time of the assessment;
   the patient is in the ED-like, early, pre-ICU presentation of the intended use. Read the notes first, then the tables.
2. `ams_at_t0`: was the patient acutely unconscious or in an altered mental status at t0?
3. For EACH label rate the state and your confidence. States: **absent** (no evidence and nothing suggests it); **possible** (plausible, evidence
   inconclusive, less likely than not); **probable** (more likely than not a cause or contributor of the impaired consciousness at t0);
   **definite** (established by objective evidence or an unambiguous documented diagnosis); **unassessable** (the packet lacks the information needed,
   which is not the same as absent). Labels can co-occur. Confidence 1-5: guess, low, moderate, high, certain (of THIS rating).
4. Pick the single `primary_etiology` that best explains the impaired consciousness at t0, or `unknown`. If you choose a label, that label
   should be probable or definite; if you choose `unknown`, no label should be.
5. Tick the evidence you relied on, give the time of diagnostic certainty (when the packet first made your primary etiology at least probable;
   "never" if it did not), the packet quality, and the minutes you spent (reading plus form; this measures workload).

| Label | Includes | Careful with |
|---|---|---|
| E1 Acute structural | stroke, ICH, SAH, SDH/EDH, TBI, mass effect, hydrocephalus; sub-label focal / diffuse / both | old strokes and chronic lesions are not E1 unless they explain this presentation |
| E2 Global hypoxic-ischemic | post-arrest, asphyxia, profound shock or global hypoperfusion | hypotension that is mild or after the fact |
| E3 Epileptic contributor | seizure, non-convulsive status, causally relevant postictal state | you are EEG-blind: judge from witnessed seizure, postictal course, the team's suspicion. This is a clinical judgement, NOT the EEG-based positive-control adjudication of D-002 |
| E4a Exogenous intoxication | overdose, drug or toxin exposure that brought the patient in or arose unintentionally (alcohol, opioids, sedative-hypnotics, toxic alcohols, CO) | planned medical sedation is E4b |
| E4b Iatrogenic sedation | planned or administered sedatives / opioids sufficient to depress the examination | secondary label; a sedated patient can still have another etiology |
| E5 Metabolic/physiologic | hepatic, uremic, glucose, sodium, hypercapnia, endocrine, acid-base | a lab abnormality that is incidental is not a cause |
| E6 Systemic infection/inflammation | sepsis-associated encephalopathy WITHOUT CNS infection | CNS infection is E7 |
| E7 CNS infection/inflammation | meningitis, encephalitis, autoimmune encephalitis | |
| other | any other cause (name it in the comment, no identifiers) | |
| unknown (primary only) | no etiology can be assigned from the packet | |

Free text: no identifiers, do not paste note text. Time budget: the plan assumes 10-15 minutes per review with a pre-assembled packet; if you
regularly need much longer, tell the coordinator rather than rushing.

## 6. Disagreement resolution

Per label, the gold call is binary: positive = probable or definite. Raters R1 and R2 rate every case independently (separate REDCap projects built from the same
dictionary, so neither can see the other's form). The analysis then builds the consensus:

* R1 and R2 agree (both positive or both negative; `possible` counts as negative): that call.
* They disagree: **a third rater (R3)** rates the case independently, blind to R1 and R2, and decides. The disagreement queue
  (`out/local_only/adjudication/third_rater_queue.csv`, case ids plus the labels in dispute, written by `adjudication_analyze.py`) tells the coordinator which
  cases go to R3; R3 sees the whole packet and fills the whole form.
* One rater `unassessable` for a label: the label's call needs two valid calls; R3 supplies the missing one.
* No third rating yet: the case is `pending` for that label and excluded from the silver-vs-gold numbers (reported as a count).

R3 is a different person from R1 and R2. If a label's kappa is below 0.6 after the first batches, the three raters meet to
sharpen the DEFINITIONS (not to re-rate cases), the change is logged in `DECISION_LOG.md`, and later batches follow the new text.

## 7. Analysis (aggregate-only)

`scripts/adjudication_analyze.py` writes `out/adjudication/agreement_report.{md,json}`:

* **Inter-rater**: Cohen's kappa per label (binary) with a 95% percentile bootstrap CI resampling cases within the sampling strata; prevalence-
  and bias-adjusted kappa (PABAK = 2 po - 1) with the prevalence and bias indices; the same kappa re-weighted to IU-large; linear-weighted kappa
  on the four ordinal states; Fleiss' kappa; a categorical kappa for the primary etiology.
* **Intra-rater**: the same statistics between a rater's two ratings of a duplicated patient, pooled and per rater.
* **Gold**: consensus prevalence per label, in the sample and re-weighted to IU-large, with a bootstrap CI.
* **Silver vs gold**: sensitivity, specificity, PPV and NPV of each silver label (E1, E2, E4a, E5, E6, E7) against the consensus gold, weighted by the
  sampling weights, with stratified bootstrap CIs and the unweighted values alongside.
* **Gate 0** (plan: kappa >= 0.6 in >= 4 primary families and >= 3 families with >= 10% prevalence), computed twice for the plan's five families: on the enriched sample, as
  the plan words it, and with the prevalence re-weighted to IU-large, because the enriched sample over-states rare prevalence by design.
* **Amendment A1 labels** {E1, E2, E5}: kappa with CI and re-weighted gold prevalence per label against the same thresholds (arithmetic only, [OP-A1]).
* Minutes per review (quantiles), forms reporting EEG information, internal-consistency counts, and the third-rater queue size.

Small cells: every count below 11 is `<11`, and a statistic is suppressed unless its cells (pairs, positives either, negatives either; for
silver-vs-gold both cells of each proportion) are at least 11. With 200 cases the rare labels (E7, E2 ...) will often be suppressed: that is the
rule working, and a reason to read the CI and the counts together. `--min-cell` can be lowered only by a person outside an agent session
(for the study team's own working report; never for anything leaving the team).

## 8. Workload estimate

Per rater 225 packets (200 + 20 duplicates + 5 calibration); two main raters = 450 reviews. The third rater sees the cases with at least one
binary disagreement among the 9 labels: with a per-label disagreement rate of 5% to 10% that is 1 - (1 - d)^9 = 37% to 61% of the 200 (75 to 120 cases),
at about 0.6 of a full review (the whole packet is read, the form is partly pre-answered by the dispute).

| Minutes per review | R1 + R2 (450 reviews) | R3 (75-120 cases x 0.6) | Total physician-hours |
|---|---|---|---|
| 10 (plan, D-046 lower bound) | 75 h | 8-12 h | 83-87 h |
| 15 (plan, D-046 upper bound) | 113 h | 11-18 h | 124-131 h |
| 25 (plan before pre-assembled packets) | 188 h | 19-30 h | 206-218 h |

About 37-94 h per main rater (6-8 weeks at 6-12 h a week) and 8-30 h for R3. This pilot measures the real minutes per review (`minutes_spent`),
which replaces the D-046 planning figure for Study 1; if the median is above 20 minutes the evaluation-set budget (570-850 h) must be re-planned
before Gate 0 is called.

## 9. Timeline (10 weeks from the go decision)

| Week | Work |
|---|---|
| 0 | Confirm IRB determination and BDSP DUA coverage (section 10); credential the raters; build the REDCap projects from the dictionary |
| 1 | HUMAN builds the packets (notes streamed); coordinator reads 10 packets for blinding; fix `blinding.py` if needed; pre-load case ids |
| 2 | Calibration: R1 and R2 rate the 5 calibration packets; definitions meeting |
| 3-8 | Independent rating in 11 batches of 20 (duplicates appear only from batch 7 on, at least one batch after their originals); weekly count of forms entered; first interim kappa after batch 3 |
| 9 | Disagreement queue to R3; R3 rates |
| 10 | Analysis, Gate 0 report, decision log entry |

## 10. IRB and BDSP data-use notes (to confirm, not legal advice)

* HEEDB is a restricted, credentialed BDSP dataset. The BDSP terms prohibit sending record-level data into a hosted model's context (CLAUDE.md
  rule 1); packets contain note text, so they stay on approved compute, are read in a browser there, and are never e-mailed, copied to personal
  devices, uploaded to a hosted LLM or pasted into a chat. Every rater needs their own BDSP credentialing and DUA coverage; confirm that the
  project's DUA covers clinicians who only read packets.
* Confirm with the IRB whether chart-style adjudication of de-identified HEEDB records is exempt or not human-subjects research, and add the
  adjudication, the creation of gold labels and the REDCap project to the protocol or determination. Confirm with BDSP that per-patient gold
  labels (derived data) may be stored in the institutional REDCap project and which aggregates may leave the team.
* REDCap: use an institution-hosted instance approved for restricted data. The form contains only pseudonymous case ids, categorical ratings,
  minutes and two short free-text fields (no identifiers, no note text). The id map stays on approved compute. Use three separate projects
  (R1, R2, R3) from the same dictionary so independence is enforced by access, or one project with a repeating instrument and per-rater
  user rights (weaker: instances are visible to project users).
* What leaves the team: aggregate, small-cell-suppressed tables only (CLAUDE.md rule 3). Gold labels, ids and packets are not committed (rule 6).
* Authorship is the plan's incentive for the adjudicator pool (D-047).

## 11. REDCap import

`docs/adjudication_redcap_dictionary.csv` is generated from `configs/adjudication_crf.yaml` (33 fields: 9 labels x state + confidence with
branching, primary etiology, evidence checkboxes, certainty timing, packet quality, EEG-seen check, minutes). Upload it under Project Setup >
Data Dictionary, pre-load the records from `redcap_case_ids.csv`, and export "CSV / raw data" per project (state values as codes 0-3, 9 or names:
the analysis accepts both). Re-run `scripts/adjudication_redcap.py --check` after editing the YAML.

## 12. Limits

The strata are defined by silver labels, so unweighted summaries describe the enriched sample (use the weights). Silver patterns inherit the
structured-data ascertainment limits of D-123 and the E6 availability rule (D-153). The pilot has two sites. IU-large approximates the ED
population (proxies for infusion, ventilation and vasopressors; date-granular visit start). EEG-blind E3 is a clinical judgement and is not the
EEG-based positive control. Redaction is rule-based and conservative but not perfect (section 4). No LLM reads a note or summarises a
packet in this pilot.
