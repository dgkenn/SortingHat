#!/usr/bin/env python3
"""Baselines A-C feature matrices for the Study 1 cohort (real-data runner; Baseline D dropped, D-107).

    # synthetic data, or a local mirror of the HEEDB layout
    python3 scripts/build_baselines.py --data data/synthetic --cohort out/local_only/cohort_study1.csv \
        --out out/local_only/baselines_AC.parquet
    # the real access point (streaming, aggregate-only; D-118)
    scripts/heedb_run.sh python3 scripts/build_baselines.py --s3 --cohort out/local_only/cohort_study1.csv \
        --out out/local_only/baselines_AC.parquet

What it does
    * reads the cohort table (``--cohort``, under local_only/; one row per patient) and rebuilds the baseline INDEX from
      it (person_id, SiteID, SessionID, t0 = metadata EEG start per D-124, age, sex). Sex comes from the site
      ``reports_findings`` / ``eeg_metadata`` CSVs (``SexDSC``); no referral indication (Baseline D is dropped).
    * streams the OMOP tables (measurement, drug_exposure, condition_occurrence, procedure_occurrence, observation) for the
      cohort's people (and ids merged into them) ROW GROUP BY ROW GROUP through ``data_io.iter_omop_batches`` (filtered to
      the cohort inside Arrow), turns each chunk into baseline events with the package's own event builders and keeps only
      those events (memory scales with the cohort's relevant events, not with the 66 GB measurement table).
    * runs the package's ``build_feature_set`` unchanged, so the single t0 gate ``baselines.asof.as_of`` decides what a
      clinician could have seen at t0. A memory PREFILTER drops events that ``as_of`` could never keep or that no window
      can read (event time after t0; high-volume charted vitals/scores/pupils/POC glucose older than their window plus
      1 h). It only removes rows; it never admits one. ``tests/test_silver_feasibility_baselines.py`` proves the matrix is
      identical with and without it.
    * writes the record-level matrix (person_id + the A-C columns, float64) to ``--out`` (must be under local_only/, mode
      0600) and a names-only sidecar ``<out>.columns.json`` (which column belongs to A, B, C).

Stdout carries aggregates only (``sortinghat.safe_output``; n < 11 shown as "<11"): per baseline the pooled missing rate of
its value-type variables overall, per group and per site, and how many variables are observed in < 11 / few / most patients.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # repo root, so `sortinghat` imports

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sortinghat import agent_safety, data_io, schema  # noqa: E402
from sortinghat.audit import field_audit as fa  # noqa: E402
from sortinghat.baselines import BaselineConfig, features as bl_features  # noqa: E402
from sortinghat.baselines import lexicon as lx  # noqa: E402
from sortinghat.baselines.asof import EVENT_COLUMNS, empty_events  # noqa: E402
from sortinghat.baselines.events import drug_events, history_events, measurement_events  # noqa: E402
from sortinghat.cohort import StoreSources  # noqa: E402
from sortinghat.cohort.memguard import peak_rss_gb, run_guarded  # noqa: E402
from sortinghat.cohort.sources import remap_ids  # noqa: E402
from sortinghat.safe_output import safe_print, safe_write_json, suppress_count, suppress_proportion  # noqa: E402

BASELINES_OUT = ("A", "B", "C")                      # D dropped (D-107)
TEXT_COLS = ("SiteID", "SessionID", "BidsFolder", "EEGFolder")
STUDY_SITES = ("S0001", "S0002")                     # D-120: the only sites with labelled patients
COHORT_DEFS = {"table": None, "strict": "in_strict", "strict_pm6": "in_strict_pm6", "broad": "in_broad"}
CHUNK_ROWS = 250_000
PREFILTER_MARGIN_H = 1.0

MEAS_COLS = ["person_id", "measurement_datetime", "measurement_date", "measurement_source_value", "value_as_number",
             "unit_source_value", *schema.COLUMN_ALIASES["measurement.result_datetime"]]
DRUG_COLS = ["person_id", "drug_exposure_start_datetime", "drug_exposure_end_datetime", "drug_source_value", "quantity",
             "drug_concept_id"]
COND_COLS = ["person_id", "condition_start_datetime", "condition_source_value"]
PROC_COLS = ["person_id", "procedure_datetime", "procedure_date", "procedure_source_value"]
OBS_COLS = ["person_id", "observation_datetime", "observation_date", "observation_source_value", "value_as_string"]
_DRUG_ANY = re.compile("|".join(rx for _n, _c, rx in lx._DRUG_RULES), re.I)          # prefilter: superset of every rule
_DRUG_ANY_RE2 = r"\b(?:" + "|".join(rx for _n, _c, rx in lx._DRUG_RULES) + r")\b"


# --------------------------------------------------------------------------------------------------- inputs
def require_local_only(p: Path, what: str) -> None:
    if "local_only" not in Path(p).resolve().parts:
        raise SystemExit(f"{what} must be under a local_only/ directory (gitignored; CLAUDE.md rule 5)")


def _as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(("true", "1", "1.0", "yes"))


def load_cohort(path: str | Path, sites=None, cohort_def: str = "table") -> pd.DataFrame:
    """Cohort table (record-level; never printed) restricted to ``sites`` and to a cohort definition's flag."""
    path = Path(path)
    require_local_only(path, "--cohort")
    df = pd.read_csv(path, dtype={c: str for c in TEXT_COLS}, low_memory=False)
    need = {"SiteID", "person_id", "SessionID", "t0"}
    if need - set(df.columns):
        raise SystemExit(f"cohort table lacks columns {sorted(need - set(df.columns))}")
    df["person_id"] = pd.to_numeric(df["person_id"], errors="coerce")
    df = df[df["person_id"].notna()].copy()
    df["person_id"] = df["person_id"].astype("int64")
    if "person_id_source" in df:
        src = pd.to_numeric(df["person_id_source"], errors="coerce")
        df["person_id_source"] = src.fillna(df["person_id"]).astype("int64")
    else:
        df["person_id_source"] = df["person_id"]
    df["t0"] = schema.parse_datetimes(df["t0"])
    for c in ("in_strict", "in_strict_pm6", "in_broad"):
        if c in df:
            df[c] = _as_bool(df[c])
    df["SiteID"] = df["SiteID"].astype(str)
    if sites:
        df = df[df["SiteID"].isin([str(s) for s in sites])]
    flag = COHORT_DEFS[cohort_def]
    if flag is not None:
        if flag not in df:
            raise SystemExit(f"cohort table has no column {flag!r}")
        df = df[df[flag]]
    df = df[df["t0"].notna()]
    if df["person_id"].duplicated().any():
        raise SystemExit("cohort table must have one row per person_id")
    return df.sort_values("person_id").reset_index(drop=True)


def sex_lookup(store, cohort: pd.DataFrame) -> pd.Series:
    """sex_male (1.0 / 0.0 / NaN) per cohort row, from ``SexDSC`` of the session's reports_findings, else eeg_metadata
    (the rule of ``baselines.events.build_index``), matched on (source BDSPPatientID, SessionID). Names/ids stay in memory."""
    out = pd.Series(np.nan, index=cohort.index, dtype=float)
    for site in sorted(cohort["SiteID"].unique()):
        sel = cohort["SiteID"] == site
        sub = cohort[sel]
        text = pd.Series(pd.NA, index=sub.index, dtype="string")
        for table in ("reports_findings", "eeg_metadata"):
            try:
                df = data_io.read_site_table(table, site, s3=store)
            except FileNotFoundError:
                continue
            if "SexDSC" not in df.columns or "SessionID" not in df.columns:
                continue
            ref = pd.DataFrame({"pid": fa.person_ids(df, site), "sid": df["SessionID"].astype(str),
                                "sex": df["SexDSC"].astype("string")}).dropna(subset=["pid", "sex"])
            ref = ref[ref["sex"].str.strip() != ""].drop_duplicates(["pid", "sid"])
            lut = pd.Series(ref["sex"].to_numpy(object),
                            index=pd.MultiIndex.from_arrays([ref["pid"].astype("int64").to_numpy(), ref["sid"].to_numpy(object)]))
            q = pd.MultiIndex.from_arrays([sub["person_id_source"].astype("int64").to_numpy(),
                                           sub["SessionID"].astype(str).to_numpy(object)])
            got = pd.Series(lut.reindex(q).to_numpy(object), index=sub.index, dtype="string")
            text = text.where(text.notna(), got)
        s = text.str.strip().str.lower()
        v = pd.Series(np.nan, index=sub.index, dtype=float)
        v[s.str.startswith("m").fillna(False).to_numpy(bool)] = 1.0
        v[s.str.startswith("f").fillna(False).to_numpy(bool)] = 0.0
        out[sel] = v
    return out


def make_index(cohort: pd.DataFrame, sex_male: pd.Series) -> pd.DataFrame:
    """The baseline index (``baselines.events.build_index`` columns) from the cohort table; no referral indication."""
    age = pd.to_numeric(cohort["age_years"], errors="coerce") if "age_years" in cohort else \
        pd.Series(np.nan, index=cohort.index)
    return pd.DataFrame({
        "person_id": cohort["person_id"].astype("int64").to_numpy(), "SiteID": cohort["SiteID"].astype(str).to_numpy(),
        "SessionID": cohort["SessionID"].astype(str).to_numpy(), "t0": cohort["t0"].astype("datetime64[us]").to_numpy(),
        "age_years": age.to_numpy(float), "sex_male": sex_male.reindex(cohort.index).to_numpy(float),
        "indication_raw": None}).sort_values("person_id").reset_index(drop=True)


# ------------------------------------------------------------------------------------------ event streaming
def prune_events(ev: pd.DataFrame, t0: pd.Series, cfg: BaselineConfig, prefilter: bool = True) -> pd.DataFrame:
    """MEMORY PREFILTER (never admits a row; ``as_of`` stays the gate). Drops events whose event time is after the
    person's t0 (``t_avail >= t_event``, so ``as_of`` would drop them anyway) and charted vital/score/pupil/POC-glucose rows
    older than their feature window plus a margin. Also blanks the two text columns no extractor reads."""
    if ev.empty:
        return ev
    ev = ev.assign(raw_name=None, unit=None)
    if not prefilter:
        return ev
    t = ev["person_id"].map(t0)
    keep = (ev["t_event"].isna() | (ev["t_event"] <= t)) & t.notna()      # NaT event time: as_of decides
    win = {"score": cfg.score_window_h, "vital": cfg.vital_window_h, "pupil": cfg.vital_window_h,
           "poc_glucose": cfg.poc_glucose_window_h}
    for dom, w in win.items():
        old = (ev["domain"] == dom) & (ev["t_event"] < t - pd.to_timedelta(w + PREFILTER_MARGIN_H, unit="h"))
        keep &= ~old
    return ev[keep.to_numpy(bool)]


def prune_rows(chunk: pd.DataFrame, t0: pd.Series, dt_col: str, date_col: str | None = None) -> pd.DataFrame:
    """RAW-ROW prefilter, same contract as ``prune_events`` (removes only rows no event could survive ``as_of`` from): drops
    rows timed after the person's t0 by their datetime, else (date-only) by a date after t0's calendar day. Rows with no
    usable time are kept for the event builder to judge. Ids must already be the surviving ids."""
    if chunk.empty or dt_col not in chunk:
        return chunk
    t = chunk["person_id"].map(t0)
    dt = chunk[dt_col]
    late = dt.notna() & (dt > t)
    if date_col is not None and date_col in chunk:
        late |= dt.isna() & chunk[date_col].notna() & (chunk[date_col] > t.dt.normalize())
    return chunk[(~late & t.notna()).to_numpy(bool)]


def drug_concept_names(store, on_error=None) -> pd.DataFrame:
    """``omop_concept`` rows (concept_id, concept_name) whose NAME mentions a lexicon sedative/opioid (brand names
    included): the only concept names ``drug_events`` can use. Streamed; the match runs inside Arrow."""
    import pyarrow.compute as pc
    parts = []
    for b in data_io.iter_omop_batches("concept", columns=["concept_id", "concept_name"], s3=store, on_error=on_error):
        names = b.column("concept_name").cast("string") if "concept_name" in b.schema.names else None
        if names is None:
            continue
        hit = pc.fill_null(pc.match_substring_regex(names, _DRUG_ANY_RE2, ignore_case=True), False)
        if pc.any(hit).as_py():
            parts.append(b.filter(hit).to_pandas())
    if not parts:
        return pd.DataFrame({"concept_id": pd.Series(dtype="Int64"), "concept_name": pd.Series(dtype="string")})
    return schema.coerce_types("omop_concept", pd.concat(parts, ignore_index=True))


def collect_events(src: StoreSources, index: pd.DataFrame, cfg: BaselineConfig, mm: dict | None = None,
                   prefilter: bool = True, chunk_rows: int = CHUNK_ROWS) -> tuple[pd.DataFrame, dict]:
    """Baseline events for the indexed people, streamed table by table, row group by row group."""
    pids = {int(p) for p in index["person_id"]}
    rev: dict[int, list[int]] = {}
    for old, new in (mm or {}).items():
        rev.setdefault(new, []).append(old)
    ids = set(pids) | {o for p in pids for o in rev.get(p, [])}
    t0 = index.set_index("person_id")["t0"]
    diag: dict = {"n_measurement_rows_unmapped": 0}
    parts: list[pd.DataFrame] = []

    def add(ev: pd.DataFrame) -> None:
        ev = prune_events(ev, t0, cfg, prefilter)
        if len(ev):
            parts.append(ev)

    def early(chunk, dt_col, date_col=None):
        chunk = remap_ids(chunk, mm)
        return prune_rows(chunk, t0, dt_col, date_col) if prefilter else chunk

    for chunk in src.iter_rows("omop_measurement", MEAS_COLS, ids, chunk_rows):
        d: dict = {}
        add(measurement_events({"omop_measurement": early(chunk, "measurement_datetime", "measurement_date")}, pids, cfg, d))
        diag["n_measurement_rows_unmapped"] += d.get("n_measurement_rows_unmapped", 0)
        del chunk
    concept = drug_concept_names(src.s3)
    hit_ids = set(concept["concept_id"].dropna().astype("int64"))
    for chunk in src.iter_rows("omop_drug_exposure", DRUG_COLS, ids, chunk_rows):
        txt = chunk["drug_source_value"].astype("string").fillna("")
        hit = txt.str.contains(_DRUG_ANY, na=False).to_numpy(bool)
        if "drug_concept_id" in chunk and hit_ids:
            hit |= pd.to_numeric(chunk["drug_concept_id"], errors="coerce").isin(hit_ids).to_numpy(bool)
        chunk = early(chunk[hit], "drug_exposure_start_datetime")
        if len(chunk):
            add(drug_events({"omop_drug_exposure": chunk, "omop_concept": concept}, pids, cfg))
    for table, cols in (("omop_condition_occurrence", COND_COLS), ("omop_procedure_occurrence", PROC_COLS),
                        ("omop_observation", OBS_COLS)):
        for chunk in src.iter_rows(table, cols, ids, chunk_rows):
            add(history_events({table: remap_ids(chunk, mm)}, pids))      # small tables: events pruned, rows kept
    ev = pd.concat(parts, ignore_index=True) if parts else empty_events()
    return ev[EVENT_COLUMNS], diag


def build_matrices(src: StoreSources, index: pd.DataFrame, cfg: BaselineConfig, mm: dict | None = None,
                   prefilter: bool = True, chunk_rows: int = CHUNK_ROWS):
    """(FeatureSet, number of events fed to the gate) with the package's own ``build_feature_set``: the streamed events
    replace its in-memory ``build_events`` step, everything after it (the ``as_of`` gate, the extractors) is untouched."""
    ev, diag = collect_events(src, index, cfg, mm, prefilter, chunk_rows)
    with mock.patch.object(bl_features, "build_events", lambda tables, idx, c: (ev, diag)):
        fs = bl_features.build_feature_set({}, cfg, index)
    return fs, len(ev)


# ---------------------------------------------------------------------------------------------- aggregates
def baseline_columns(fs, baselines=BASELINES_OUT) -> dict[str, list[str]]:
    """Output columns per baseline: the package's A/B/C subsets minus the imaging group (D-122: no imaging at the Study 1
    sites, and this runner does not read the imaging table, so those columns would be a constant 0 that means nothing)."""
    grp = fs.provenance.set_index("feature")["group"]
    return {b: [c for c in fs.columns(b) if grp[c] != "imaging"] for b in baselines}


def missingness_summary(fs, index: pd.DataFrame, baselines=BASELINES_OUT) -> dict:
    """Aggregate-only missingness of the VALUE-type variables (NaN = not observed by t0) per baseline."""
    prov = fs.provenance.set_index("feature")
    n = len(fs.X)
    site = index.set_index("person_id")["SiteID"].reindex(fs.X.index).to_numpy()
    out = {"n_patients": suppress_count(n), "baselines": {}}
    for b, cols in baseline_columns(fs, baselines).items():
        val = [c for c in cols if prov.at[c, "role"] in ("value", "age_h")]
        X = fs.X[val]
        n_obs = X.notna().sum()
        cells_total, cells_miss = n * len(val), int(X.isna().to_numpy().sum())
        bins = {"observed_in_fewer_than_11": 0, "missing_ge_90pct": 0, "missing_50_to_90pct": 0,
                "missing_10_to_50pct": 0, "missing_lt_10pct": 0}
        for c, k in n_obs.items():
            f = 1.0 - k / n if n else 1.0
            key = ("observed_in_fewer_than_11" if k < 11 else "missing_ge_90pct" if f >= 0.9 else
                   "missing_50_to_90pct" if f >= 0.5 else "missing_10_to_50pct" if f >= 0.1 else "missing_lt_10pct")
            bins[key] += 1
        groups = {}
        for g in sorted({prov.at[c, "group"] for c in val}):
            gc = [c for c in val if prov.at[c, "group"] == g]
            miss = int(X[gc].isna().to_numpy().sum())
            groups[g] = {"n_variables": len(gc), "pooled_missing_rate": suppress_proportion(miss, n * len(gc))}
        per_site = {}
        for i, s in enumerate(sorted({str(x) for x in site})):
            m = site == s
            ns = int(m.sum())
            per_site[f"site_{i + 1}"] = {"n": suppress_count(ns),
                                         "pooled_missing_rate": suppress_proportion(int(X[m].isna().to_numpy().sum()),
                                                                                    ns * len(val)) if ns >= 11 else "<11"}
        out["baselines"][b] = {"n_columns": len(cols), "n_value_variables": len(val),
                               "pooled_missing_rate": suppress_proportion(cells_miss, cells_total),
                               "variables_by_missingness": bins, "by_group": groups, "by_site": per_site}
    return out


def print_summary(summ: dict, n_events: int, rss: float) -> None:
    safe_print("Baselines A-C built (aggregate-only output; n<11 suppressed). Baseline D dropped (D-107). "
               f"patients={summ['n_patients']}; events kept after the as_of gate inputs={suppress_count(n_events)}")
    for b, s in summ["baselines"].items():
        safe_print(f"  Baseline {b}: {s['n_columns']} columns, {s['n_value_variables']} value variables, pooled missing "
                   f"rate {s['pooled_missing_rate']}; variables by missingness {s['variables_by_missingness']}")
        safe_print("    by group: " + "; ".join(f"{g}={v['pooled_missing_rate']}" for g, v in s["by_group"].items()))
        safe_print("    by site: " + "; ".join(f"{k}: n={v['n']} missing={v['pooled_missing_rate']}"
                                              for k, v in s["by_site"].items()))
    safe_print(f"  peak RSS: {rss:.2f} GB")


# -------------------------------------------------------------------------------------------------- driver
def write_outputs(fs, index: pd.DataFrame, cfg: BaselineConfig, out: Path) -> Path:
    cols = baseline_columns(fs)["C"]                             # A subset B subset C
    X = fs.X.reindex(index["person_id"].to_numpy())[cols].astype("float64")
    df = X.reset_index()
    df["person_id"] = df["person_id"].astype("int64")
    out.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(out.parent, 0o700)
    df.to_parquet(out, index=False)
    os.chmod(out, 0o600)
    side = out.with_name(out.stem + ".columns.json")
    side.write_text(json.dumps({"baselines": baseline_columns(fs),
                                "t0_basis": "metadata EEG start (D-124)", "baseline_d": "dropped (D-107)",
                                "imaging": "no imaging inputs (D-122); img__* columns not written",
                                "config": cfg.to_dict()}, indent=1, default=list))
    os.chmod(side, 0o600)
    return side


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="local directory in the HEEDB layout (synthetic data or a local mirror)")
    src.add_argument("--s3", action="store_true", help="read the real BDSP access point")
    ap.add_argument("--profile", help="AWS profile for --s3")
    ap.add_argument("--cohort", default="out/local_only/cohort_study1.csv", help="cohort table (under local_only/)")
    ap.add_argument("--cohort-def", choices=sorted(COHORT_DEFS), default="table",
                    help="rows to build: the whole table (default; a superset of every cohort flag) or one flag")
    ap.add_argument("--sites", nargs="+", default=list(STUDY_SITES), help="default: S0001 S0002 (D-120); 'all' for every site")
    ap.add_argument("--out", default="out/local_only/baselines_AC.parquet", help="record-level parquet (under local_only/)")
    ap.add_argument("--summary", help="optional aggregate JSON (safe_write_json)")
    ap.add_argument("--merge-cols", nargs=2, metavar=("OLD", "NEW"), help="PatientMergeHistory id columns (as build_cohort)")
    ap.add_argument("--chunk-rows", type=int, default=CHUNK_ROWS)
    ap.add_argument("--no-prefilter", action="store_true", help="skip the memory prefilter (tests; needs much more memory)")
    ap.add_argument("--max-memory-gb", type=float, default=None, help="RLIMIT_AS guard (see build_cohort.py)")
    a = ap.parse_args(argv)
    return run_guarded(lambda: _run(a), a.max_memory_gb, "baseline build")


def _run(a) -> int:
    out = Path(a.out)
    require_local_only(out, "--out")
    if a.data:
        agent_safety.assert_not_restricted_in_agent(a.data)
    store = data_io.open_store(a.data, profile=a.profile)
    sites = None if a.sites == ["all"] else a.sites
    cohort = load_cohort(a.cohort, sites, a.cohort_def)
    sex = sex_lookup(store, cohort)
    index = make_index(cohort, sex)
    del cohort
    cfg = BaselineConfig()
    sources = StoreSources(store, merge_cols=tuple(a.merge_cols) if a.merge_cols else None)
    mm, _status = sources.merge_map()
    fs, n_events = build_matrices(sources, index, cfg, mm or None, not a.no_prefilter, a.chunk_rows)
    write_outputs(fs, index, cfg, out)
    summ = missingness_summary(fs, index)
    print_summary(summ, n_events, peak_rss_gb())
    if a.summary:
        safe_write_json(a.summary, summ)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:                       # never print a traceback (it could carry values); class name only
        print(f"build_baselines failed: {type(e).__name__}", file=sys.stderr)
        raise SystemExit(1)
