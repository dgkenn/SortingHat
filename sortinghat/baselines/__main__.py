"""python -m sortinghat.baselines --data data/synthetic --out out/baselines

Builds the t0-masked Baseline A-D feature matrix from a HEEDB-layout directory (synthetic data in agent
sessions). Writes the provenance table (not record-level) to --out and the record-level matrix only to
``<out>/local_only/features.csv`` (mode 0600, gitignored). Prints aggregates only, n < 11 suppressed.
Whole tables are loaded (``tableio.load_tables``): fine for synthetic scale; a human-run restricted job should
cohort-filter first (see docs/baselines_spec.md)."""

from __future__ import annotations

import argparse
from pathlib import Path

from .. import agent_safety
from ..safe_output import safe_print, safe_write_text, suppress_count, write_local_only
from ..tableio import load_tables
from .config import BaselineConfig
from .features import BASELINES, build_feature_set


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="local HEEDB-layout directory (synthetic or a local mirror)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--drug-time-basis", default="auto", choices=["auto", "order", "admin"])
    ap.add_argument("--lab-time-basis", default="auto", choices=["auto", "collect_plus_lag", "result"])
    ap.add_argument("--imaging-time-basis", default="auto", choices=["auto", "study_plus_lag", "result"])
    a = ap.parse_args(argv)
    agent_safety.assert_not_restricted_in_agent(a.data)
    cfg = BaselineConfig(drug_time_basis=a.drug_time_basis, lab_time_basis=a.lab_time_basis,
                         imaging_time_basis=a.imaging_time_basis)
    fs = build_feature_set(load_tables(a.data), cfg)
    out = Path(a.out)
    safe_write_text(out / "feature_provenance.csv", fs.provenance.to_csv(index=False))
    write_local_only(out / "local_only" / "features.csv", fs.X.reset_index().to_csv(index=False))
    n = len(fs.X)
    safe_print(f"Baseline feature matrix built (aggregate-only; n<11 suppressed): patients={suppress_count(n)}")
    for b in BASELINES:
        safe_print(f"  Baseline {b}: {len(fs.columns(b))} columns")
    for g, feats in fs.provenance[fs.provenance["role"] == "missing_indicator"].groupby("group")["feature"]:
        rate = fs.X[feats.tolist()].to_numpy(float).mean()      # pooled over patients and variables
        safe_print(f"  mean missing-indicator rate, {g}: {round(float(rate), 3) if n >= 11 else "<11"}")
    d = fs.diagnostics
    safe_print(f"  events total={suppress_count(d['n_events_total'])}, kept (available by t0)="
               f"{suppress_count(d['n_events_by_t0'])}, dropped as post-t0={suppress_count(d['n_events_dropped_post_t0'])}")
    safe_print(f"Provenance: {out / 'feature_provenance.csv'}; matrix (local only): {out / 'local_only' / 'features.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
