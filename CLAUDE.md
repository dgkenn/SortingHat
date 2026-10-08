# SortingHat: agent rules (Phase 0e, from docs/research_plan_v1.txt)

This repo will eventually run against restricted BDSP HEEDB data. The BDSP terms
prohibit sending record-level data into a hosted model's context. Treat every
agent session as a hosted-model context.

## Hard rules

1. **Synthetic data only in agent sessions.** Develop and test against
   `data/synthetic/` (generate with `python -m sortinghat.synthetic --out data/synthetic`).
   Never read, list, head, describe or open real HEEDB tables, EDFs, notes or any
   file under a restricted path.
2. **Restricted jobs run from a plain terminal or scheduler**, never from inside an
   agent session. Agents write and review the scripts; a human runs them.
   Exception (D-118): the project lead has authorised streaming, aggregate-only real-data jobs launched from agent sessions via scripts/heedb_run.sh and scripts/overnight.sh; all other rules still apply.
3. **Aggregate-only output.** Scripts that may touch restricted data print/write only
   aggregates with small-cell suppression (n < 11 shown as "<11"). Use
   `sortinghat.safe_output` (`suppress_count`, `suppress_proportion`, `safe_quantiles`,
   `safe_write_json`, `safe_print`, `assert_aggregate_only`). Never call `df.head()`,
   `df.sample()`, `print(df)`, `df.to_string()`, `display(df)` or log rows/IDs/timestamps.
   Do not emit min/max (record-level); use quantiles with n >= 11.
4. **Open-weight LLMs on approved compute for notes.** Any LLM that reads notes is an
   open-weight model on approved compute. Never send notes to a hosted API (including Claude).
5. Files holding IDs for human use (e.g. the hand-check sample) go only in a
   `local_only/` directory (gitignored, mode 0600) and are never printed or read by an agent.
6. Do not commit data, reports derived from restricted data, or secrets.

## Restricted paths

`.claude/settings.json` denies Read/Edit/Bash on `/restricted/**`, `data/heedb/**`,
`data/restricted/**` and `*.edf`. Additional roots go in env `SORTINGHAT_RESTRICTED_ROOT`
(os.pathsep-separated). Env vars cannot be expanded in settings, so run
`SORTINGHAT_RESTRICTED_ROOT=/path python -m sortinghat.agent_safety --write-local`
once to write matching deny rules to the gitignored `.claude/settings.local.json`.
The audit CLI also refuses restricted paths when `CLAUDECODE=1`
(`sortinghat.agent_safety.assert_not_restricted_in_agent`). Deny rules are
best-effort for Bash; the human-run-only rule is the real control.

## Commands

- Tests: `pytest`
- Synthetic data: `python -m sortinghat.synthetic --out data/synthetic`
- Field audit: `python -m sortinghat.audit.field_audit --data data/synthetic --out out/audit`
- Schema dry run (names only; the first command a human runs on real data, via `scripts/heedb_run.sh ... --s3`):
  `python -m sortinghat.audit.field_audit --data data/synthetic --dry-run-schema`

- Names-only real probe (project-lead authorised only; headers, parquet footers and `Delimiter='/'` prefix names, never values, rows,
  counts, dates or IDs, never inside per-patient folders): `scripts/heedb_run.sh python3 scripts/heedb_dry_run.py`
  (= `field_audit --s3 --dry-run-schema --list-unlisted --probe-prefixes`).

## Schema

Real table/column names and provenance (CONFIRMED / NAMED / ASSUMED) are in `sortinghat/schema.py`,
`docs/heedb_schema_real.md` and `docs/heedb_schema.md`. The two per-site CSV tables have per-site header variants
(`schema.SITE_VARIANTS`; I0008/I0009 have no reports_findings); read them with `data_io.read_site_table`, which maps to canonical names.
Only `PatientClass`, `ReferralIndication` and the `imaging` table remain ASSUMED.
