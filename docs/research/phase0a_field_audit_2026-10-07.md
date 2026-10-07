# Phase 0a HEEDB field audit: status 2026-10-07

**Status: audit code ready and tested on synthetic data; NOT YET RUN ON REAL HEEDB DATA.**
**Gate 0 field-audit verdict: NOT DETERMINED (no real-data result exists).** The pilot-kappa part of Gate 0 is
pending humans in any case.

No number in this file comes from real HEEDB. The only real-data evidence used is the names-only dry run
(`heedb_schema_dryrun_2026-10-07.md`), which settled which columns and prefixes exist and nothing about values.

## Why the real run was not executed from the agent session

The requested command wraps the audit in `env -u CLAUDECODE -u SORTINGHAT_AGENT_SESSION ...`. Those two variables
are the repo's agent-session guard: `scripts/heedb_run.sh` exits 77 and `data_io.make_client` raises
`RestrictedDataError` when either is set (CLAUDE.md rule 2: restricted jobs run from a plain terminal or scheduler,
never from inside an agent session). Unsetting them defeats that control rather than satisfying it. The only
project-lead authorisation recorded in the repository (CLAUDE.md, `docs/heedb_schema_real.md` section 0) covers the
names-only probe ("never values, rows, counts, dates or IDs"); this audit reads values and produces counts, which is
a different scope, and the instruction reached the agent as a message from another agent rather than from the
project lead. So the job was left for a human to launch (command below), or for the project lead to give an explicit,
recorded authorisation for this specific value-level run inside an agent session.

## How to run (human, plain terminal)

```
cd SortingHat
HEEDB_AWS_PROFILE=bdsp scripts/heedb_run.sh python3 -m sortinghat.audit.field_audit --s3 --out out/audit
```

Writes `out/audit/field_audit.md` and `field_audit.json` (aggregate-only, n<11 shown as `<11`) and
`out/audit/local_only/handcheck_sample_ids.csv` (20 patient ids, mode 0600, for the human hand-check; never open it in an
agent session). On a failure against `--s3` the CLI prints only the stage and the exception class (messages and
tracebacks can quote values); reproduce on synthetic data to debug.

Memory: EEG CSVs are read whole (one row per session); every OMOP table is streamed part by part, column-pruned, and
filtered inside Arrow to the candidate cohort and, for drugs / measurements / observations, by regex or concept id
before anything reaches pandas. Only these columns are requested: drug (person, start/end datetime, source value,
type concept id), measurement (person, datetime/date, source value, concept id, any result-time alias), observation
(same shape), note (person, note_datetime, note_date; `note_text` is never requested), concept (id, name, domain,
vocabulary; vocabulary rows, not patient data).

## Implementation against each Phase 0a row (real names)

| Plan row | Field names used | Criterion as implemented | Plan fallback if it fails | Stop row |
|---|---|---|---|---|
| EEG start date and time of day | `reports_findings."StartTime(EEG)"`; `eeg_metadata.StartDateTime` (I0008/I0009); eeg_metadata `StartTime` otherwise. Acute-care class derived from `omop_visit_occurrence` (`visit_concept_id`, else `visit_source_value`) because no header has `PatientClass`; adults by age (AgeAtVisit, AgeInDaysAtVisit, or StartTime minus DateOfBirth) | start present for >=95% of adult acute-care EEG sessions (all adult sessions, flagged in the report, if no visit matched) | Stop; no study | yes |
| Consistent within-patient date shift | candidate t0 vs median offset of notes, labs, imaging, sedation drugs; within-record ordering (EEG start<=end, drug start<=end, imaging study<=final) | automated proxy: <=5% candidates with a table median >30 d from t0 and <=1% ordering violations; **plus** the 20-case hand-check list, written only to `local_only/` (drawn from candidates having both labs and notes). The automated part cannot see a shift applied identically to every table, so the row is not closed until a human checks the 20 cases | Stop | yes |
| Medication administration times | `omop_drug_exposure.drug_type_concept_id` resolved to a NAME through `omop_concept` and classified administration / order / other (`classify_drug_type`); start and end datetimes; sedation class by regex on `drug_source_value` (propofol, midazolam, dexmedetomidine/precedex, fentanyl, ketamine, pentobarbital) | among candidates with a sedation-class exposure in [t0-48h, t0+1h], >=80% have an ADMINISTRATION-type record. If no type id resolves to a name (zero-filled ids, rule 6), falls back to the proxy "non-empty `drug_exposure_end_datetime`" and says so in the report. Also reports the share among all candidates and the type-concept names with suppressed counts | use orders; label 1B "approximate" | no |
| Lab result or verification time | `omop_measurement`: no result-time column exists (dry run); aliases `measurement_result_datetime`, `result_datetime`, `ResultDTS` are checked in case a release has one | pass only if an alias column exists and is filled for >=95% of candidate lab rows. Expected from the dry run: column absent, so FAIL | collection time plus assay lag; label 1B "approximate" | no |
| Imaging report finalization time | `Imaging/imaging_metadata/` (placeholder) does not exist; real prefixes are `Imaging/I0001/` and `Imaging/I0004/` only, neither being one of the six HEEDB EEG-metadata sites; columns inside unread | >=95% of candidate imaging studies have a finalization time. Expected from the dry run: no imaging table for the EEG sites, so FAIL. The report prints how many candidate sites have an imaging prefix | drop H5 | no |
| GCS, FOUR or RASS near EEG | score concepts found BY NAME in `omop_concept` (Measurement/Observation domain, name matches glasgow / gcs / eye opening / best motor|verbal response / FOUR score / full outline of unresponsiveness / richmond / rass) matched on `measurement_concept_id` and `observation_concept_id`, plus a source-text match on `measurement_source_value` / `observation_source_value` (concept ids can be zero-filled). Ramsay, arousal, level of consciousness and generic sedation scales are classed OTHER and do not count | >=50% of candidates have a GCS (incl. components), FOUR or RASS value within +-6 h of t0 (`measurement_datetime` / `observation_datetime`); coverage by class and by source table is reported | BDSP GCS-from-EHR tool; broad cohort only | no |
| Site identifier | `SiteID` (header, or `InstituteID`/file name at I0008/I0009) | >=3 adult sites with >=300 candidates each (first qualifying acute-care adult EEG per patient, start time present) | grouped split; weaker claim | no |
| Timestamped notes | `omop_note.note_datetime` | operationalised as >=95% of candidates with >=1 note that has a `note_datetime`. The field-level share of notes with a timestamp is reported beside it. **Decision for the project lead:** the plan says only "Present"; if 95% of candidates is stricter than intended, change the threshold in `run_audit` before the real run | Stop | yes |

Semantics still unread (cannot be settled without a real run): whether `note_datetime` is authoring or service time;
the actual drug-type concept names; the units of `RecordingDuration` at I0008/I0009.

## Per-row results

| Row | Observed aggregate | Verdict | Fallback triggered |
|---|---|---|---|
| EEG start date and time | not run | PENDING | n/a |
| Consistent within-patient date shift (automated) | not run | PENDING | n/a |
| Consistent within-patient date shift (20-case hand-check) | list not generated | PENDING (human) | n/a |
| Medication administration times | not run | PENDING | n/a |
| Lab result or verification time | not run (dry run: no result-time column in `omop_measurement`) | PENDING; FAIL expected | collection time plus assay lag; 1B "approximate" |
| Imaging report finalization time | not run (dry run: no imaging prefix for the six EEG sites) | PENDING; FAIL expected | drop H5 |
| GCS / FOUR / RASS within +-6 h | not run | PENDING | n/a |
| Site identifier | not run | PENDING | n/a |
| Timestamped notes | not run | PENDING | n/a |

## Per-site candidate counts

Not available (requires the real run). The audit writes them suppressed (n<11 as `<11`) in the "Site identifier"
row and in the per-site breakdown of `field_audit.md`.

## Gate 0 field-audit verdict

**NOT DETERMINED.** Gate 0 needs every Stop row to pass (EEG start time, within-patient date shift including the
20-case human check, timestamped notes), then pilot kappa >=0.6 for at least four primary families and >=3 families
with >=10% prevalence (humans, Phase 0b). None of that can be stated until the real run is done by a human or
explicitly authorised.

## Code changes made for this audit (`sortinghat/audit/field_audit.py`, tests in `tests/test_field_audit.py`)

1. Medication row uses `drug_type_concept_id` semantics via `omop_concept` names (was only an end-time proxy);
   the proxy remains as a flagged fallback.
2. Score row finds GCS / FOUR / RASS concepts by name in `omop_concept`, reads `omop_observation` as well as
   `omop_measurement`, and no longer counts OTHER scales.
3. Streaming loader: Arrow-side regex / concept-id filtering and minimal column lists (previously every candidate
   measurement row, including values, was materialised in pandas before filtering, which would not fit in memory on
   the 66 GB measurement table); added observation, concept and note streams.
4. Real-data failures print the stage and exception class only.
5. Per-row aggregate details are rendered in `field_audit.md` (method used, drug-type category counts, score coverage by
   class and source, imaging-prefix sites, note timestamp share).

Verification: `pytest tests/test_field_audit.py` passes (28 tests, synthetic data only, including an on-disk test of
the concept-name path and a check that real-data failures leak no message text).
