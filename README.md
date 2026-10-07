# SortingHat

Phase 0 tooling for the SortingHat research plan (`docs/research_plan_v1.txt`):

- `sortinghat/safe_output.py` - aggregate-only reporting, small-cell suppression (n < 11 -> "<11"), row-level guard.
- `sortinghat/synthetic/` - seeded synthetic HEEDB-shaped dataset (fake; generate on demand, gitignored).
- `sortinghat/audit/field_audit.py` - Phase 0a field audit (pass/fail markdown + JSON).
- `CLAUDE.md`, `.claude/settings.json` - Phase 0e agent-safety rules.

Both the synthetic generator and the audit use the REAL HEEDB layout (`docs/heedb_schema_real.md`,
`sortinghat/schema.py`): per-site `EEG/eeg-metadata` and `EEG/HEEDB_Metadata` CSVs and
`OMOP/Merged/<table>/*.parquet`, read through `sortinghat/data_io.py` (a local directory for synthetic data,
S3 for the real run).

Agent or CI session (synthetic data only):

```
pip install -e ".[dev]"
python -m sortinghat.synthetic --out data/synthetic
python -m sortinghat.audit.field_audit --data data/synthetic --out out/audit
python -m sortinghat.audit.field_audit --data data/synthetic --dry-run-schema
pytest
```

Real HEEDB, **from your own terminal, never inside an agent session** (credentials: `docs/credentials_template.md`;
`scripts/heedb_run.sh` refuses to start under an agent and strips a stub `AWS_ACCESS_KEY_ID`):

```
# 1. FIRST: which expected tables/columns exist vs missing (names only; reads CSV headers and parquet footers)
HEEDB_AWS_PROFILE=physionet scripts/heedb_run.sh \
    python -m sortinghat.audit.field_audit --s3 --dry-run-schema --out out/schema_dry_run
# 2. Then remap ASSUMED / missing columns in sortinghat/schema.py and run the audit
HEEDB_AWS_PROFILE=physionet scripts/heedb_run.sh \
    python -m sortinghat.audit.field_audit --s3 --out out/audit
```

`--sites S0001 S0002` limits the sites (default: every site with an eeg-metadata CSV); `--data <dir>` instead
of `--s3` reads a local mirror of the layout. The dry run prints table/column names only. The audit writes a
20-case hand-check ID list only to `out/audit/local_only/` (never printed); everything else is aggregate-only
with n < 11 suppressed. Paste only the dry-run markdown (names, no values) back into an agent session.
