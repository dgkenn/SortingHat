"""Aggregate-only diagnostics for the cohort's visit matching and duration unit (human-run on real data).

    python scripts/diag_cohort.py --data data/synthetic_layout_dir          # synthetic / local mirror
    scripts/heedb_run.sh python scripts/diag_cohort.py --s3 --out out/cohort/diag.json   # real access point

Prints (and writes, if --out is given) one JSON document of aggregates: counts and proportions with n < 11 shown as
"<11", quantiles only for n >= 11 (no min/max), text values only with counts >= 11 and digits masked, merge-history
column NAMES only. No identifier, date or row is printed. See sortinghat/cohort/diag.py for what is computed.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # repo root, so `sortinghat` imports

from sortinghat import agent_safety, data_io                       # noqa: E402
from sortinghat.cohort.diag import run_diag                         # noqa: E402
from sortinghat.safe_output import safe_write_json                  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="local directory in the HEEDB layout (synthetic data or a local mirror)")
    src.add_argument("--s3", action="store_true", help="read the real BDSP access point (human-run only)")
    ap.add_argument("--profile", help="AWS profile for --s3")
    ap.add_argument("--sites", nargs="+", help="site codes (default: every site with an eeg-metadata CSV)")
    ap.add_argument("--out", help="also write the JSON here (aggregate only)")
    a = ap.parse_args(argv)
    if a.data:
        agent_safety.assert_not_restricted_in_agent(a.data)
    store = data_io.open_store(a.data, profile=a.profile)            # make_client refuses inside an agent session
    report, known = run_diag(store, a.sites)
    if a.out:
        safe_write_json(a.out, report, known)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
