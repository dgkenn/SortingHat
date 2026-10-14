"""BIND imaging-E1 plan, steps 0a and 0b (docs/bind_e1_analysis_plan.md section 9): linkage and time-base checks.

Aggregate-only. Reads the BIND I0001 demographics table (person, session, DOB, study date) and findings person ids, the Study 1
cohort table and the cached OMOP person table; prints and writes suppressed counts and shares only. No label is computed and no
finding value is read (findings are read for the person column only). Stop rules S1 (overlap < 100) and S2 (DOB agreement
< 95 %) are evaluated and printed.

Run (human or D-118 launcher): scripts/heedb_run.sh python3 scripts/bind_step0.py
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sortinghat.labels import bind  # noqa: E402
from sortinghat.safe_output import safe_print, safe_write_json, suppress_count  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="out/local_only/cohort_study1.csv")
    ap.add_argument("--person-cache", default="out/local_only/omop_cache/person")
    ap.add_argument("--site", default="I0001")
    ap.add_argument("--out", default="out/bind/step0.json")
    a = ap.parse_args(argv)
    cfg = bind.load_config()
    demo = bind.read_bind_table("demographics", a.site, cfg=cfg)
    dcol = cfg.raw["demographics_columns"][a.site]
    fcol = cfg.findings_columns
    findings = bind.read_bind_table("findings", a.site, cfg=cfg)
    img_people = set(demo[dcol["person"]].dropna()) | set(findings[fcol["person"]].dropna())

    coh = pd.read_csv(a.cohort, usecols=["SiteID", "person_id"], dtype=str)
    out: dict = {"bind_site": a.site, "bind_people": suppress_count(len({bind.norm_id(x) for x in img_people} - {None})),
                 "linkage": {}, "stop_rules": {}}
    overlap_total = 0
    for site, g in coh.groupby("SiteID"):
        q = bind.linkage_qc(img_people, g["person_id"])
        out["linkage"][site] = q
        overlap_total += len({bind.norm_id(x) for x in img_people} & {bind.norm_id(x) for x in g["person_id"]} - {None})
    out["linkage"]["all"] = bind.linkage_qc(img_people, coh["person_id"])

    parts = sorted(glob.glob(f"{a.person_cache}/**/*.parquet", recursive=True))
    person = pd.concat([pd.read_parquet(p, columns=["person_id", "birth_datetime"]) for p in parts], ignore_index=True)
    ref = pd.Series(person["birth_datetime"].values, index=person["person_id"].astype(str))
    bd = demo[[dcol["person"], dcol["dob"]]].dropna().drop_duplicates(dcol["person"])
    bdob = pd.Series(bd[dcol["dob"]].values, index=bd[dcol["person"]].astype(str))
    out["time_base"] = bind.shift_agreement(bdob, ref)

    # stop rules are evaluated on unsuppressed values in memory (a suppressed share prints "<11" when it is near 0 OR near 1)
    b = pd.to_datetime(bdob, errors="coerce").dt.normalize(); b.index = [bind.norm_id(i) for i in b.index]
    r = pd.to_datetime(ref, errors="coerce").dt.normalize(); r.index = [bind.norm_id(i) for i in r.index]
    b = b[~b.index.duplicated()].dropna(); r = r[~r.index.duplicated()].dropna()
    both = b.index.intersection(r.index)
    exact = float(((b.loc[both] - r.loc[both]).dt.days == 0).mean()) if len(both) else 0.0
    out["stop_rules"]["S1_overlap_ge_100"] = overlap_total >= 100
    out["stop_rules"]["S2_dob_agreement_ge_0.95"] = exact >= 0.95
    safe_write_json(a.out, out)
    safe_print(out)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as ex:  # class name only (a traceback could carry values)
        print(f"bind_step0 failed: {type(ex).__name__}", file=sys.stderr)
        raise SystemExit(1)
