"""Build the Study 1 cohort (flow report + local-only cohort table + recording key list).

    # synthetic data, or a local mirror of the HEEDB layout
    python scripts/build_cohort.py --data data/synthetic --out out
    # the real access point (HUMAN-RUN ONLY, from your own terminal; refuses inside an agent session)
    scripts/heedb_run.sh python scripts/build_cohort.py --s3 --out out

Writes
    out/local_only/cohort_study1.csv     record-level cohort table      (mode 0600, never printed)
    out/local_only/recording_keys.csv    recording key list for the streaming extractor (mode 0600, never printed)
    out/cohort/flow.md, flow.json        CONSORT-style flow, aggregate only, counts < 11 suppressed

Stdout carries aggregates only (via sortinghat.safe_output). Choices are listed in docs/cohort_spec.md.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # repo root, so `sortinghat` imports

from sortinghat import agent_safety, data_io                       # noqa: E402
from sortinghat.cohort import CohortConfig, StoreSources, build_cohort, write_outputs   # noqa: E402
from sortinghat.cohort.config import ONSET_RULES, SCORE_RULES        # noqa: E402
from sortinghat.cohort.build import MERGE_NOTICE                      # noqa: E402
from sortinghat.cohort.memguard import peak_rss_gb, run_guarded     # noqa: E402
from sortinghat.cohort.output import known_ids                       # noqa: E402
from sortinghat.safe_output import safe_print, suppress_count         # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="local directory in the HEEDB layout (synthetic data or a local mirror)")
    src.add_argument("--s3", action="store_true", help="read the real BDSP access point (human-run only)")
    ap.add_argument("--profile", help="AWS profile for --s3")
    ap.add_argument("--sites", nargs="+", help="site codes (default: every site with an eeg-metadata CSV)")
    ap.add_argument("--out", default="out", help="output root (default: out; gitignored)")
    ap.add_argument("--onset-rule", choices=ONSET_RULES, default="score_then_visit")
    ap.add_argument("--score-rule", choices=SCORE_RULES, default="nearest",
                    help="primary strict rule in [-6 h, +1 h] of t0 (D-105); strict_pm6 is always reported too")
    ap.add_argument("--service-fallback", action="store_true",
                    help="treat an unclassifiable visit as acute care when ServiceName is LTM (decision C-03)")
    ap.add_argument("--debug-flow", action="store_true",
                    help="also write out/cohort/flow_debug.md/json and print the ALL-sites table with EVERY step on its "
                         "own row (only counts < 11 suppressed; not for sharing)")
    ap.add_argument("--visit-slack-h", type=float, default=0.0,
                    help="widen every visit interval by this many hours both sides when matching an EEG (decision C-19)")
    ap.add_argument("--open-visit-days", type=float, default=30.0,
                    help="a visit with no end is treated as open this many days after its start (C-19)")
    ap.add_argument("--merge-cols", nargs=2, metavar=("OLD", "NEW"),
                    help="column names of the retired and surviving id in PatientMergeHistory/ (see diag_cohort.py)")
    ap.add_argument("--max-memory-gb", type=float, default=None,
                    help="guard: set RLIMIT_AS to this many GB (address space, an upper bound on RSS; pick generously) "
                         "and print a clear aggregate error instead of a traceback if exceeded")
    ap.add_argument("--duration-scale", nargs="*", default=[], metavar="SITE=FACTOR",
                    help="unit fix for DurationInSeconds / RecordingDuration after reading the flow's unit check, "
                         "e.g. I0008=60")
    a = ap.parse_args(argv)
    return run_guarded(lambda: _run(a), a.max_memory_gb, "cohort build")


def _run(a) -> int:

    if a.data:
        agent_safety.assert_not_restricted_in_agent(a.data)
    store = data_io.open_store(a.data, profile=a.profile)            # make_client refuses inside an agent session
    scale = tuple((kv.split("=")[0], float(kv.split("=")[1])) for kv in a.duration_scale)
    cfg = CohortConfig(onset_rule=a.onset_rule, score_rule=a.score_rule, use_service_fallback=a.service_fallback,
                       duration_scale_by_site=scale, visit_slack_h=a.visit_slack_h,
                       open_visit_days=a.open_visit_days)
    result = build_cohort(StoreSources(store, a.sites, merge_cols=tuple(a.merge_cols) if a.merge_cols else None), cfg)
    paths = write_outputs(result, a.out, a.debug_flow)

    ids = known_ids(result)
    t = result.table
    safe_print("Study 1 cohort built (aggregate-only output; n<11 suppressed).", known_ids=ids)
    safe_print(f"  table rows (EEG within the widest onset window, strict or phenotype): {suppress_count(len(t))}",
               known_ids=ids)
    safe_print(f"  strict cohort [-6 h, +1 h], primary window: {suppress_count(int(t['in_strict'].sum()))}", known_ids=ids)
    safe_print(f"  strict_pm6 sensitivity cohort (+-6 h), primary window: "
               f"{suppress_count(int(t['in_strict_pm6'].sum()))}", known_ids=ids)
    safe_print(f"  broad cohort (includes strict), primary window: {suppress_count(int(t['in_broad'].sum()))}",
               known_ids=ids)
    if a.debug_flow:
        safe_print("DEBUG FLOW (all sites; every step; only counts < 11 suppressed; do not share):", known_ids=ids)
        for r in result.debug["sites"]["ALL"].get("rows", []):
            safe_print(f"  {r['step']} | excluded {r['n_excluded']} | remaining {r['n_remaining']}", known_ids=ids)
    safe_print(f"  notice: {MERGE_NOTICE[result.merge_status]}", known_ids=ids)
    safe_print(f"  peak RSS: {peak_rss_gb():.2f} GB", known_ids=ids)
    safe_print(f"Flow report: {paths['flow_md']}", known_ids=ids)
    safe_print(f"Record-level files (mode 0600, not printed): {paths['cohort'].parent}/", known_ids=ids)
    return 0


if __name__ == "__main__":
    sys.exit(main())
