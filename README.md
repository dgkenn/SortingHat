# SortingHat

Phase 0 tooling for the SortingHat research plan (`docs/research_plan_v1.txt`):

- `sortinghat/safe_output.py` - aggregate-only reporting, small-cell suppression (n < 11 -> "<11"), row-level guard.
- `sortinghat/synthetic/` - seeded synthetic HEEDB-shaped dataset (fake; generate on demand, gitignored).
- `sortinghat/audit/field_audit.py` - Phase 0a field audit (pass/fail markdown + JSON).
- `CLAUDE.md`, `.claude/settings.json` - Phase 0e agent-safety rules.

```
pip install -e ".[dev]"
python -m sortinghat.synthetic --out data/synthetic
python -m sortinghat.audit.field_audit --data data/synthetic --out out/audit
pytest
```

The audit writes a 20-case hand-check ID list only to `out/audit/local_only/` (never printed).
Restricted-data runs must be launched from a plain terminal, not an agent session.
