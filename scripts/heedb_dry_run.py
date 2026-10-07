"""Names-only schema dry run against the real access point, with unlisted column names and prefix names.

Run from a plain terminal (or, when the project lead has authorised names-only probing, via the wrapper):

    env -u CLAUDECODE -u SORTINGHAT_AGENT_SESSION HEEDB_AWS_PROFILE=bdsp scripts/heedb_run.sh python3 scripts/heedb_dry_run.py

Prints table / column / prefix NAMES only: CSV header lines, parquet footers (names), and
``Delimiter='/'`` prefix listings that never enter per-patient folders. No row, value, count, date or ID.
"""
import sys

from sortinghat.audit import field_audit

if __name__ == "__main__":
    sys.exit(field_audit.main(["--s3", "--dry-run-schema", "--list-unlisted", "--probe-prefixes"] + sys.argv[1:]))
