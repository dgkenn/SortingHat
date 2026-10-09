#!/usr/bin/env python3
"""Subset the recording key list to one cohort definition (strict, broad, ...) for the rung extractors.

    python3 scripts/make_recording_keys.py --cohort-def broad \
        --keys out/local_only/recording_keys.csv --cohort out/local_only/cohort_study1.csv \
        --out out/local_only/recording_keys_broad.csv

Selection is the SAME code the feasibility script uses: ``build_baselines.load_cohort(cohort, sites, cohort_def)`` (sites default
``bb.STUDY_SITES``, the feasibility default), and a key row is kept when its (SiteID, person_id, SessionID) is a row of that cohort.
Record-level: input and output live under local_only/ (mode 0600) and are never printed; stdout is aggregate counts only
(n < 11 shown as "<11").
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import pandas as pd  # noqa: E402

import build_baselines as bb  # noqa: E402
from sortinghat.safe_output import safe_print, suppress_count  # noqa: E402

JOIN = ["SiteID", "person_id", "SessionID"]


def select_keys(keys: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """Rows of ``keys`` whose (SiteID, person_id, SessionID) is in ``cohort`` (already restricted by load_cohort)."""
    k = keys.copy()
    k["_pid"] = pd.to_numeric(k["person_id"], errors="coerce")
    k["_site"] = k["SiteID"].astype(str)
    k["_sess"] = k["SessionID"].astype(str)
    ids = pd.MultiIndex.from_frame(pd.DataFrame({"_site": cohort["SiteID"].astype(str), "_pid": cohort["person_id"].astype("int64"),
                                                 "_sess": cohort["SessionID"].astype(str)}))
    mask = pd.MultiIndex.from_frame(k[["_site", "_pid", "_sess"]]).isin(ids)
    return keys[mask].reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cohort-def", choices=sorted(bb.COHORT_DEFS), required=True)
    ap.add_argument("--keys", default="out/local_only/recording_keys.csv")
    ap.add_argument("--cohort", default="out/local_only/cohort_study1.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sites", nargs="+", default=list(bb.STUDY_SITES), help="default as run_silver_feasibility.py; 'all' for every site")
    a = ap.parse_args(argv)
    for p, what in ((a.keys, "--keys"), (a.cohort, "--cohort"), (a.out, "--out")):
        bb.require_local_only(Path(p), what)
    sites = None if a.sites == ["all"] else a.sites
    cohort = bb.load_cohort(a.cohort, sites, a.cohort_def)
    keys = pd.read_csv(a.keys, dtype={c: str for c in bb.TEXT_COLS}, low_memory=False)
    need = set(JOIN) - set(keys.columns)
    if need:
        raise SystemExit(f"key list lacks columns {sorted(need)}")
    sel = select_keys(keys, cohort)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", newline="") as fh:
        sel.to_csv(fh, index=False)
    os.chmod(tmp, 0o600)
    os.replace(tmp, out)
    safe_print(f"cohort_def={a.cohort_def} cohort rows={suppress_count(len(cohort))} key rows in={suppress_count(len(keys))} "
               f"key rows out={suppress_count(len(sel))}")
    for i, (_s, n) in enumerate(sel["SiteID"].astype(str).value_counts().sort_index().items(), 1):
        safe_print(f"  site #{i} (lexicographic): {suppress_count(n)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
