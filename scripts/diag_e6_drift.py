#!/usr/bin/env python3
"""Diagnose temporal drift of the silver label E6 (infectious/septic) in the silver-feasibility temporal holdout.   (aggregate-only)

Problem: in scripts/run_silver_feasibility.py the temporal holdout (train on earlier, test on later cases within site) shows E6
observed/expected ~ 14 (about 17% of test cases positive against about 4% overall) while leave-one-site-out is ~1.1, so E6 positives
sit in the late period. Hypotheses: (a) ascertainment change over time (culture / antibiotic / lactate data only in later years),
(b) date-shift artefact (the split assumes shifted dates preserve order within site), (c) coding / vocabulary change, (d) real
case-mix change.

What it does. It rebuilds the feasibility script's analysed rows with ITS OWN assembly (``run_silver_feasibility.assemble``) and ITS
OWN temporal split (``late_temporal_holdout``), then tabulates, per site x time bin, counts and proportions of:
  label.<L>.not_assessable / positive   from the silver labels the feasibility script reads (L = every label
                                                                present: E1 E2 E4a E5 E6 E7; E1, E2, E5 are the comparison labels)
  comp.<L>.<node>       every anchor rule node (leaf and composite, E6 incl. its blood-culture, QAD and organ-dysfunction legs) re-evaluated
                        on the anchor event table with the pipeline's own evaluator (``sortinghat.labels.anchors``); E6 also after the E7 exclusion
  e6pos.<node>          the E6 legs among E6 positives only (which route carries the positives)
  avail.*               availability of the underlying data in [-72 h, +24 h] of t0 (and over the whole extraction window for cultures and
                        antimicrobials): matched blood-culture / lactate / creatinine / bilirubin / platelet rows, antimicrobial and vasopressor
                        drug rows, matched diagnosis / procedure rows, and RAW table coverage (any row of measurement, drug_exposure,
                        condition_occurrence, procedure_occurrence, observation, visit_occurrence)
  vocab.*               match route (concept / source_concept / source_code / name) of the culture and antimicrobial rows, and the number of
                        cases whose culture or antimicrobial wording never occurs in the site's training bins (counts only, never the text)
  casemix.*             age, severity, recording duration, hours from visit start to t0 (quantiles, n >= 11), sedation and subgroup shares
  timing.*              signed hours from t0 of the nearest blood culture and of the first antimicrobial start (quantiles, n >= 11)
  consistency.*         the re-derived E6 against the silver_labels.csv E6 (the two should agree)
Time bins: the temporal split's train / test assignment, and ``--n-bins`` (default 5) quantile bins of the within-site t0 rank. Bins are
reported by ORDINAL INDEX only (train/test, Q1..Q5); no date, year, cutoff or rank value is ever emitted.

Disclosure control: every count < 11 is shown "<11"; a cell is also hidden when its complement (denominator - count) < 11; when exactly
one cell of a row is hidden the smallest remaining cell of the row is hidden too (complementary suppression, so it cannot be recovered
from the site total published elsewhere); a pooled (all sites) cell is hidden when exactly one per-site cell of its bin is hidden (it would reveal it by subtraction).
Everything goes through ``sortinghat.safe_output``.

    # synthetic / local mirror (what the tests run)
    python3 scripts/diag_e6_drift.py --data data/synthetic --cohort ... --features ... --silver ... --baselines ...
    # real (aggregate-only; D-118), same input flags as run_silver_feasibility.py; --s3 streams from the OMOP cache / S3
    env -u CLAUDECODE -u SORTINGHAT_AGENT_SESSION scripts/heedb_run.sh python3 scripts/diag_e6_drift.py --s3 --profile bdsp
Without --data / --s3 only the label, case-mix and bin tables are produced (no data-availability block).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))             # repo root, so `sortinghat` imports
sys.path.insert(0, str(HERE))                    # sibling scripts (run_silver_feasibility, build_baselines)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import run_silver_feasibility as rsf  # noqa: E402
from sortinghat import checkpoint, data_io, schema  # noqa: E402
from sortinghat.labels import anchors as an  # noqa: E402
from sortinghat.labels import extract as ex  # noqa: E402
from sortinghat.safe_output import (SUPPRESS_BELOW, SUPPRESSED, safe_print, safe_quantiles,  # noqa: E402
                                    safe_write_json, suppress_count)

BANNER = "EXPLORATORY DIAGNOSTIC — E6 temporal drift in the silver-feasibility holdout; aggregate-only; not a test of any hypothesis"
K = SUPPRESS_BELOW
POOLED = "all_sites"
AVAIL_WINDOW = (-72.0, 24.0)                     # hours from t0: the E6 blood-culture / QAD window
COMPARISON_LABELS = ("E1", "E2", "E5")           # compared with E6
COMPONENT_LABELS = ("E6", "E1", "E2", "E5")
BASES = ("concept", "source_concept", "source_code", "name")
# raw table -> (timestamp column, date-only fallback column); read with person_id only (no values, no text)
RAW_TABLES = {"measurement": ("measurement_datetime", "measurement_date"),
              "drug_exposure": ("drug_exposure_start_datetime", "drug_exposure_start_date"),
              "condition_occurrence": ("condition_start_datetime", "condition_start_date"),
              "procedure_occurrence": ("procedure_datetime", "procedure_date"),
              "observation": ("observation_datetime", "observation_date")}


# ============================================================================================ small types
@dataclass
class Metric:
    name: str
    num: np.ndarray                      # bool (n,)
    den: np.ndarray | None = None        # bool (n,); None = all rows
    note: str = ""


@dataclass
class Quant:
    name: str
    values: np.ndarray                   # float (n,), NaN = unknown
    note: str = ""


@dataclass
class DiagInput:
    sites: np.ndarray                    # object (n,) raw site codes
    times: np.ndarray                    # datetime64 (n,)  (RECORD-LEVEL, memory only; ordering only)
    Y: pd.DataFrame                      # (n, labels) float 1.0 / 0.0 / NaN (= not assessable), positional
    site_labels: dict                    # raw site -> "site_k"
    temporal_fraction: float = 0.2
    metrics: list = field(default_factory=list)       # extra Metric objects (components, availability, vocabulary, consistency, case mix)
    quants: list = field(default_factory=list)
    notes: list = field(default_factory=list)


# ================================================================================================= bins
def quantile_bins(sites: np.ndarray, times: np.ndarray, n_bins: int) -> np.ndarray:
    """Ordinal bin 0..n_bins-1 of the t0 rank within site (ties broken by input order)."""
    s = pd.Series(np.asarray(times).astype("datetime64[us]"))
    sg = pd.Series(np.asarray(sites, dtype=object))
    rank = s.groupby(sg).rank(method="first").to_numpy(float)
    size = sg.map(sg.value_counts()).to_numpy(float)
    return np.minimum(((rank - 1) * n_bins // size).astype(int), n_bins - 1)


def build_groupings(sites, times, n_bins: int, test_fraction: float) -> dict:
    """scheme -> (bin names, int bin index per row). The temporal scheme is the feasibility script's own split."""
    ts = rsf.late_temporal_holdout(sites, times, test_fraction)
    tb = np.full(len(sites), -1, dtype=int)
    tb[ts.train_idx] = 0
    tb[ts.test_idx] = 1
    return {"temporal": (["train", "test"], tb),
            "quantile": ([f"Q{i + 1}" for i in range(n_bins)], quantile_bins(sites, times, n_bins))}


# ================================================================================================ cells
def _small(num: int, den: int) -> bool:
    return den < K or num < K or (den - num) < K


def _flags(nums: list, dens: list, force: list | None = None) -> list:
    """Per cell of one row (a site over the bins of a scheme): None = shown, else why it is hidden: 'low' (count or denominator
    < 11), 'high' (complement < 11) or 'linked' (hidden because another cell would reveal it: a forced cell, or the smallest
    remaining cell of a row with exactly one hidden cell, i.e. complementary suppression)."""
    force = force or [False] * len(nums)
    why = [(("low" if (n < K or d < K) else "high") if _small(n, d) else ("linked" if f else None))
           for n, d, f in zip(nums, dens, force)]
    if sum(w is not None for w in why) == 1:
        shown = [i for i, w in enumerate(why) if w is None]
        if shown:
            why[min(shown, key=lambda i: (nums[i], i))] = "linked"
    return why


def _cells(nums: list, dens: list, nwhy: list, dwhy: list | None = None) -> list:
    out = []
    for i, (n, d, w) in enumerate(zip(nums, dens, nwhy)):
        dh = dwhy[i] if dwhy else None
        c = {"n": SUPPRESSED if w else int(n), "den": SUPPRESSED if (dh or d < K) else int(d),
             "share": SUPPRESSED if w else round(n / d, 4)}
        if w:
            c["why"] = w
        out.append(c)
    return out


def tabulate_metric(m: Metric, di: DiagInput, groupings: dict, site_order: list) -> dict:
    """Cells per scheme x (site | pooled) x bin. A restricted denominator (``m.den``) is itself a count and is protected like one."""
    n = len(di.sites)
    restricted = m.den is not None
    den = np.ones(n, bool) if not restricted else np.asarray(m.den, bool)
    num = np.asarray(m.num, bool) & den
    out = {"definition": m.note}
    for scheme, (bins, bi) in groupings.items():
        nb = len(bins)
        per_site, s_nwhy, s_dwhy = {}, {}, {}
        for s in site_order:
            ms = di.sites == s
            nums = [int((num & ms & (bi == b)).sum()) for b in range(nb)]
            dens = [int((den & ms & (bi == b)).sum()) for b in range(nb)]
            sizes = [int((ms & (bi == b)).sum()) for b in range(nb)]
            dwhy = _flags(dens, sizes) if restricted else [None] * nb
            nwhy = _flags(nums, dens, [w is not None for w in dwhy])
            per_site[di.site_labels[s]] = dict(zip(bins, _cells(nums, dens, nwhy, dwhy)))
            s_nwhy[s], s_dwhy[s] = nwhy, dwhy
        nums = [int((num & (bi == b)).sum()) for b in range(nb)]
        dens = [int((den & (bi == b)).sum()) for b in range(nb)]
        sizes = [int((bi == b).sum()) for b in range(nb)]
        # a pooled cell would let a reader recover a lone hidden site cell by subtraction; with none or >= 2 hidden it cannot
        lone = lambda why: [sum(why[s][b] is not None for s in site_order) == 1 for b in range(nb)]  # noqa: E731
        dwhy = _flags(dens, sizes, lone(s_dwhy)) if restricted else [None] * nb
        force = [a or (w is not None) for a, w in zip(lone(s_nwhy), dwhy)]
        nwhy = _flags(nums, dens, force)
        per_site[POOLED] = dict(zip(bins, _cells(nums, dens, nwhy, dwhy)))
        out[scheme] = per_site
    return out


def tabulate_quant(q: Quant, di: DiagInput, groupings: dict, site_order: list) -> dict:
    out = {"definition": q.note}
    v = np.asarray(q.values, float)
    for scheme, (bins, bi) in groupings.items():
        per_site = {}
        for s in site_order + [None]:
            ms = np.ones(len(v), bool) if s is None else (di.sites == s)
            per_site[POOLED if s is None else di.site_labels[s]] = {
                b: safe_quantiles(v[ms & (bi == k) & ~np.isnan(v)]) for k, b in enumerate(bins)}
        out[scheme] = per_site
    return out


def bin_sizes(di: DiagInput, groupings: dict, site_order: list) -> dict:
    out = {}
    for scheme, (bins, bi) in groupings.items():
        d = {}
        for s in site_order + [None]:
            ms = np.ones(len(di.sites), bool) if s is None else (di.sites == s)
            d[POOLED if s is None else di.site_labels[s]] = {b: suppress_count(int((ms & (bi == k)).sum())) for k, b in enumerate(bins)}
        out[scheme] = d
    return out


def label_metrics(Y: pd.DataFrame) -> list:
    out = []
    for lab in Y.columns:
        a = Y[lab].notna().to_numpy()
        p = (Y[lab] == 1).to_numpy()
        out.append(Metric(f"label.{lab}.not_assessable", ~a, None, f"{lab} not assessable (no value in the silver labels); denominator all analysed rows"))
        out.append(Metric(f"label.{lab}.positive", p & a, None, f"{lab} positive; denominator all analysed rows (not-assessable rows are negatives here)"))
    return out


# ============================================================================== anchor-rule components
def walk(node: dict):
    """Every descendant of a rule node (composites and leaves), depth first."""
    for key in ("any_of", "all_of"):
        for ch in node.get(key, []):
            yield ch
            yield from walk(ch)


def rule_nodes(anchor_cfg: dict, label: str) -> list:
    """[(name, node)] for a label: its whole rule (``<L>_rule``) then every node that has an id (leaf or composite)."""
    root = anchor_cfg["labels"][label]
    return [(f"{label}_rule", root)] + [(n["id"], n) for n in walk(root) if n.get("id")]


def component_flags(events: pd.DataFrame, case_ids: list, anchor_cfg: dict, labels=COMPONENT_LABELS + ("E7",)) -> dict:
    """{(label, node name): bool array over ``case_ids``}: each node evaluated on the case's events with the pipeline's own
    evaluator (``anchors._eval``), independent of whether the label as a whole fired."""
    ev = events[[c for c in events.columns if c not in ex.INTERNAL_EVENT_COLUMNS]]
    cases = an._cases_of(ev, case_ids) if len(ev) else {}
    blank = an._Case({}, "acute" in ev.columns)
    out = {}
    for lab in labels:
        if lab not in anchor_cfg["labels"]:
            continue
        acuity = bool(anchor_cfg["labels"][lab].get("acuity_required"))
        for name, node in rule_nodes(anchor_cfg, lab):
            out[(lab, name)] = np.array([an._eval(node, cases.get(cid, blank), [], acuity) for cid in case_ids], bool)
    return out


def component_metrics(flags: dict, e6_positive: np.ndarray, e6_assessable: np.ndarray) -> list:
    out = []
    for (lab, name), v in flags.items():
        out.append(Metric(f"comp.{lab}.{name}", v, None, f"anchor node {name} satisfied (alone, regardless of the rest of the {lab} rule)"))
    if ("E6", "E6_rule") in flags and ("E7", "E7_rule") in flags:
        after = flags[("E6", "E6_rule")] & ~flags[("E7", "E7_rule")]
        out.append(Metric("comp.E6.E6_rule_after_E7_exclusion", after, None, "full E6 rule satisfied and E7 rule not satisfied (the pipeline's E6)"))
    for (lab, name), v in flags.items():
        if lab == "E6" and name != "E6_rule":
            out.append(Metric(f"e6pos.{name}", v, e6_positive & e6_assessable, f"E6 node {name} satisfied; denominator the E6-positive rows"))
    return out


# ===================================================================================== availability
def _hours(df: pd.DataFrame, tcol: str, t0_by_pid: pd.Series) -> pd.Series:
    t0 = df["person_id"].map(t0_by_pid)
    return (pd.to_datetime(df[tcol]) - t0).dt.total_seconds() / 3600.0


def person_flag(df: pd.DataFrame, tcol: str, pids: np.ndarray, t0_by_pid: pd.Series, window=AVAIL_WINDOW, mask=None) -> np.ndarray:
    """Per analysed row: at least one row of ``df`` (person_id, ``tcol``) for the person, inside the window (None = any time)."""
    if df is None or not len(df):
        return np.zeros(len(pids), bool)
    d = df if mask is None else df[mask]
    if window is not None:
        h = _hours(d, tcol, t0_by_pid)
        d = d[h.between(*window)]
    return np.isin(pids, d["person_id"].unique())


def _cls(df: pd.DataFrame, col: str, value) -> pd.Series:
    return df[col] == value if col in df and len(df) else pd.Series(dtype=bool)


def availability_metrics(cl: ex.Classified, pids: np.ndarray, t0_by_pid: pd.Series, sites: np.ndarray,
                         temporal_bin: np.ndarray) -> list:
    out = []
    M, D = cl.meas, cl.drug

    def meas_item(item):
        return (M["item"] == item) if len(M) and "item" in M else pd.Series(False, index=M.index)

    def drug_group(g):
        return (D["group"] == g) if len(D) and "group" in D else pd.Series(False, index=D.index)

    w = f"[{AVAIL_WINDOW[0]:g}, {AVAIL_WINDOW[1]:g}] h of t0"
    for item, label in (("blood_culture_drawn", "blood culture row (any status)"), ("blood_culture_pos", "blood-culture result row (any status)"),
                        ("lactate", "lactate value"), ("creatinine", "creatinine value"), ("bilirubin", "bilirubin value"),
                        ("platelets", "platelet value")):
        sel = M[meas_item(item)] if len(M) else M
        out.append(Metric(f"avail.{item}.in_window", person_flag(sel, "t_event", pids, t0_by_pid), None, f"matched {label} in {w}"))
        if item in ("blood_culture_drawn", "blood_culture_pos"):
            out.append(Metric(f"avail.{item}.extract_window", person_flag(sel, "t_event", pids, t0_by_pid, window=None), None,
                              f"matched {label} anywhere in the extraction window (t0 - 30 d, t0 + 10 d)"))
    in_w = next(m for m in out if m.name == "avail.blood_culture_drawn.in_window").num
    ext_w = next(m for m in out if m.name == "avail.blood_culture_drawn.extract_window").num
    out.append(Metric("avail.blood_culture_drawn.in_window_among_extract_window", in_w, ext_w,
                      f"matched blood-culture row in {w}; denominator the rows with a culture anywhere in the extraction window (timing, not availability)"))
    pos = M[meas_item("blood_culture_pos") & (M["status"] == "positive")] if len(M) else M
    out.append(Metric("avail.blood_culture_positive_result.in_window", person_flag(pos, "t_event", pids, t0_by_pid), None,
                      f"positive blood-culture result in {w}"))
    for g, label in (("antimicrobial", "antimicrobial (route-accepted) drug row"), ("vasopressor", "vasopressor drug row")):
        sel = D[drug_group(g)] if len(D) else D
        out.append(Metric(f"avail.{g}.in_window", person_flag(sel, "t_start", pids, t0_by_pid), None, f"{label} starting in {w}"))
        if g == "antimicrobial":
            out.append(Metric("avail.antimicrobial.extract_window", person_flag(sel, "t_start", pids, t0_by_pid, window=None), None,
                              "antimicrobial drug row anywhere in the extraction window"))
    for nm, fr in (("matched_diagnosis", cl.cond), ("matched_procedure", cl.proc)):
        out.append(Metric(f"avail.{nm}.in_window", person_flag(fr, "t_event", pids, t0_by_pid), None, f"matched {nm.split('_')[1]} row in {w}"))
    v = cl.visits
    if len(v):
        h0 = (pd.to_datetime(v["v_start"]) - v["person_id"].map(t0_by_pid)).dt.total_seconds() / 3600.0
        h1 = (pd.to_datetime(v["v_end"]) - v["person_id"].map(t0_by_pid)).dt.total_seconds() / 3600.0
        cover = (h0 <= AVAIL_WINDOW[1]) & (h1.isna() | (h1 >= AVAIL_WINDOW[0]))
        out.append(Metric("avail.visit_overlapping_window", np.isin(pids, v.loc[cover, "person_id"].unique()), None, f"visit_occurrence overlapping {w}"))
    # vocabulary: match route of the culture and antimicrobial rows in the window
    bc = M[meas_item("blood_culture_drawn")] if len(M) else M
    for b in BASES:
        out.append(Metric(f"vocab.blood_culture.route_{b}", person_flag(bc, "t_event", pids, t0_by_pid, mask=_cls(bc, "basis", b) if len(bc) else None), None,
                          f"a blood-culture row matched by route '{b}' in {w}"))
    ab = D[drug_group("antimicrobial")] if len(D) else D
    for b in ("concept", "name"):
        out.append(Metric(f"vocab.antimicrobial.route_{b}", person_flag(ab, "t_start", pids, t0_by_pid, mask=_cls(ab, "basis", b) if len(ab) else None), None,
                          f"an antimicrobial row matched by route '{b}' in {w}"))
    if len(bc) and "approx" in bc:
        out.append(Metric("vocab.blood_culture.timing_date_only", person_flag(bc, "t_event", pids, t0_by_pid, mask=bc["approx"].fillna(False).astype(bool)), None,
                          f"a blood-culture row in {w} that carries a date but no time of day (stored at 00:00)"))
    out += novelty_metrics(bc, "t_event", "blood_culture", pids, t0_by_pid, sites, temporal_bin)
    out += novelty_metrics(ab, "t_start", "antimicrobial", pids, t0_by_pid, sites, temporal_bin)
    return out


def nearest_offset(df: pd.DataFrame, tcol: str, pids: np.ndarray, t0_by_pid: pd.Series, first: bool = False) -> np.ndarray:
    """Signed hours from t0 of the person's row nearest to t0 (or the earliest start with ``first``) in the extraction window;
    NaN without a row. Relative time only; used for quantiles with n >= 11."""
    out = np.full(len(pids), np.nan)
    if df is None or not len(df):
        return out
    d = df[["person_id", tcol]].copy()
    d["h"] = _hours(d, tcol, t0_by_pid)
    d = d[d["h"].notna()]
    d = d.assign(k=d["h"] if first else d["h"].abs()).sort_values("k", kind="stable").drop_duplicates("person_id")
    pos = pd.Series(np.arange(len(pids)), index=pids)
    ix = d["person_id"].map(pos)
    ok = ix.notna().to_numpy()
    out[ix[ok].astype(int).to_numpy()] = d["h"].to_numpy()[ok]
    return out


def timing_quants(cl: ex.Classified, pids: np.ndarray, t0_by_pid: pd.Series) -> list:
    M, D = cl.meas, cl.drug
    bc = M[M["item"] == "blood_culture_drawn"] if len(M) else M
    ab = D[D["group"] == "antimicrobial"] if len(D) else D
    return [Quant("timing.blood_culture_nearest_hours", nearest_offset(bc, "t_event", pids, t0_by_pid),
                  "signed hours from t0 of the blood-culture row nearest to t0 (extraction window t0-30 d..t0+10 d); cases with a culture only"),
            Quant("timing.antimicrobial_first_start_hours", nearest_offset(ab, "t_start", pids, t0_by_pid, first=True),
                  "signed hours from t0 of the first antimicrobial start in the extraction window; cases with one only")]


def novelty_metrics(df: pd.DataFrame, tcol: str, nm: str, pids, t0_by_pid, sites, temporal_bin) -> list:
    """Cases (in any bin) that have a row in the window whose source wording never occurs in the SAME SITE's training-bin rows.
    Counts only; the wording itself is never emitted."""
    if df is None or not len(df) or "src_text" not in df:
        return [Metric(f"vocab.{nm}.wording_not_seen_in_train", np.zeros(len(pids), bool), None, "no rows")]
    d = df.copy()
    d["h"] = _hours(d, tcol, t0_by_pid)
    d = d[d["h"].between(*AVAIL_WINDOW)]
    pos = pd.Series(np.arange(len(pids)), index=pids)
    d["row"] = d["person_id"].map(pos)
    d = d[d["row"].notna()]
    d["row"] = d["row"].astype(int)
    d["site"] = sites[d["row"].to_numpy()]
    d["train"] = temporal_bin[d["row"].to_numpy()] == 0
    d["txt"] = d["src_text"].astype(str).str.strip().str.lower()
    seen = {(s, t) for s, t in zip(d.loc[d["train"], "site"], d.loc[d["train"], "txt"])}
    nov = np.array([(s, t) not in seen for s, t in zip(d["site"], d["txt"])], bool)
    flag = np.zeros(len(pids), bool)
    flag[d["row"].to_numpy()[nov]] = True
    return [Metric(f"vocab.{nm}.wording_not_seen_in_train", flag, None,
                   f"{nm} row in the window whose source wording is absent from the same site's temporal-train rows (a train case has none by construction)")]


def raw_coverage(source, pids: np.ndarray, t0_by_pid: pd.Series) -> list:
    """Raw table coverage: any row (matched or not) of each OMOP table within the window, person_id and timestamp columns only."""
    out = []
    ids = [int(p) for p in pids]
    for table, (dt, dd) in RAW_TABLES.items():
        have: set = set()
        if isinstance(source, dict):
            df = source.get(f"omop_{table}")
            frames = [df] if df is not None else []
        else:
            frames = (b.to_pandas() for _u, bs, _t in data_io.iter_omop_units(table, person_ids=ids, columns=["person_id", dt, dd], s3=source)
                      for b in bs)
        for d in frames:
            if d is None or not len(d) or "person_id" not in d:
                continue
            d = d.copy()
            d["person_id"] = pd.to_numeric(d["person_id"], errors="coerce")
            d = d[d["person_id"].notna()]
            d["person_id"] = d["person_id"].astype("int64")
            d = d[d["person_id"].isin(t0_by_pid.index)]
            if not len(d):
                continue
            t = schema.parse_datetimes(d[dt]) if dt in d else pd.Series(pd.NaT, index=d.index)
            if dd in d:
                t = t.where(t.notna(), schema.parse_datetimes(d[dd]))
            d = d.assign(_t=t)
            h = _hours(d, "_t", t0_by_pid)
            have |= set(d.loc[h.between(*AVAIL_WINDOW), "person_id"].unique())
        out.append(Metric(f"avail.raw_{table}.in_window", np.isin(pids, list(have)) if have else np.zeros(len(pids), bool), None,
                          f"ANY row of omop {table} (matched or not) within [{AVAIL_WINDOW[0]:g}, {AVAIL_WINDOW[1]:g}] h of t0"))
    return out


# ======================================================================================= extraction
def run_extraction(source, cohort: pd.DataFrame):
    """The silver extraction (``extract_silver``, unchanged) with the classified tables captured on the way."""
    got = {}
    orig = ex.build_anchor_events

    def spy(cl, cases, ctx, anchor_cfg):
        got["cl"], got["anchor_cfg"] = cl, anchor_cfg
        return orig(cl, cases, ctx, anchor_cfg)
    with mock.patch.object(ex, "build_anchor_events", spy):
        res = ex.extract_silver(source, cohort, config=ex.ExtractConfig())
    return res, got["cl"], got["anchor_cfg"]


def data_metrics(source, frame: pd.DataFrame, sites, temporal_bin, csv_e6: np.ndarray, csv_assessable: np.ndarray,
                 csv_labels: pd.DataFrame) -> tuple[list, list, list]:
    """(metrics, quants, notes): component, availability, vocabulary, timing and consistency tables from the OMOP data behind
    ``frame`` (person_id, t0, SiteID)."""
    cohort = frame[["person_id", "t0", "SiteID"]].copy()
    cohort["t0"] = pd.to_datetime(cohort["t0"])
    cohort["person_id"] = cohort["person_id"].astype("int64")
    if cohort["person_id"].duplicated().any():
        raise SystemExit("several analysed rows per person_id; the diagnostic needs one row per person")
    res, cl, anchor_cfg = run_extraction(source, cohort)
    pids = cohort["person_id"].to_numpy()
    t0_by_pid = pd.Series(cohort["t0"].to_numpy(), index=pids)
    case_ids = [str(p) for p in pids]
    flags = component_flags(res.events, case_ids, anchor_cfg)
    metrics = component_metrics(flags, csv_e6, csv_assessable)
    bc_in = person_flag(cl.meas[cl.meas["item"] == "blood_culture_drawn"] if len(cl.meas) else cl.meas, "t_event", pids, t0_by_pid)
    metrics.append(Metric("label.E6.positive_among_culture_in_window", csv_e6 & csv_assessable, bc_in,
                          "E6 positive; denominator the rows with a matched blood-culture row in the E6 window (conditional prevalence)"))
    metrics += availability_metrics(cl, pids, t0_by_pid, np.asarray(sites, dtype=object), temporal_bin)
    metrics += raw_coverage(source, pids, t0_by_pid)
    quants = timing_quants(cl, pids, t0_by_pid)
    # consistency: the re-derived labels against the silver_labels.csv the feasibility script read
    notes = []
    for lab in (l for l in COMPONENT_LABELS if l in res.labels and l in csv_labels):
        re_v = res.labels[lab].reindex(case_ids).astype("boolean").fillna(False).to_numpy(bool)
        c_v = (csv_labels[lab] == 1).to_numpy()
        metrics.append(Metric(f"consistency.{lab}.csv_positive_not_rederived", c_v & ~re_v, None, f"{lab} positive in silver_labels.csv but not re-derived"))
        metrics.append(Metric(f"consistency.{lab}.rederived_positive_not_csv", re_v & ~c_v, None, f"{lab} re-derived positive but not positive in silver_labels.csv"))
    return metrics, quants, notes


# ==================================================================================== case mix
def casemix(A) -> tuple[list, list]:
    """Case-mix tables from the feasibility assembly: age (baseline demographic column), severity, duration, sedation, subgroup."""
    metrics, quants = [], []
    cov = A.covariates
    if "sedated" in cov:
        metrics.append(Metric("casemix.sedated", np.asarray(cov["sedated"], bool), None, "sedative / opioid exposure flag used by the feasibility script"))
    if getattr(A, "subgroup", None) is not None:
        metrics.append(Metric("casemix.undifferentiated_subgroup", np.asarray(A.subgroup, bool), None, "undifferentiated-AMS subgroup of the feasibility script"))
    for name, key in (("severity", "severity"), ("duration_s", "duration_s")):
        if key in cov:
            quants.append(Quant(f"casemix.{name}", np.asarray(cov[key], float), {"severity": "15 - GCS-equivalent", "duration_s": "recording duration (s)"}[name]))
    for c in A.baseline.columns:
        if str(c).startswith("demo__age"):
            quants.append(Quant("casemix.age_years", pd.to_numeric(A.baseline[c], errors="coerce").to_numpy(float), f"baseline column {c} (age)"))
            break
    return metrics, quants


# ===================================================================================== the report
def diagnose(di: DiagInput, n_bins: int = 5) -> dict:
    site_order = sorted(di.site_labels, key=lambda s: di.site_labels[s])
    groupings = build_groupings(di.sites, di.times, n_bins, di.temporal_fraction)
    metrics = label_metrics(di.Y) + list(di.metrics)
    return {"banner": BANNER,
            "settings": {"n_bins": n_bins, "temporal_test_fraction": di.temporal_fraction, "n_analysed": suppress_count(len(di.sites)),
                         "sites": [di.site_labels[s] for s in site_order], "avail_window_hours_from_t0": list(AVAIL_WINDOW),
                         "bins": {k: v[0] for k, v in groupings.items()},
                         "bin_definition": {"temporal": "late_temporal_holdout (feasibility script): within site the latest fraction of t0 is test, the rest train",
                                            "quantile": "ordinal quantile bin of the within-site t0 rank; Q1 earliest"},
                         "suppression": "count < 11 or complement < 11 -> '<11'; one hidden cell hides the smallest other cell of its row; "
                                        "a pooled cell is hidden when exactly one site cell of its bin is hidden",
                         "notes": list(di.notes)},
            "bin_sizes": bin_sizes(di, groupings, site_order),
            "tables": {m.name: tabulate_metric(m, di, groupings, site_order) for m in metrics},
            "quantile_tables": {q.name: tabulate_quant(q, di, groupings, site_order) for q in di.quants}}


def drift_ratios(report: dict) -> dict:
    """test share / train share (temporal scheme) for each metric, pooled and per site, where both shares are shown."""
    out = {}
    for name, t in report["tables"].items():
        row = {}
        for site, cells in t["temporal"].items():
            a, b = cells["train"]["share"], cells["test"]["share"]
            if isinstance(a, float) and isinstance(b, float) and a > 0:
                row[site] = round(b / a, 2)
        if row:
            out[name] = row
    return out


def render_text(report: dict, names=None) -> list:
    lines = [BANNER, f"analysed n={report['settings']['n_analysed']}; sites={','.join(report['settings']['sites'])}"]
    for scheme in ("temporal", "quantile"):
        lines.append(f"-- bin sizes {scheme}: " + "; ".join(f"{s}: " + " ".join(f"{b}={v}" for b, v in d.items())
                                                            for s, d in report["bin_sizes"][scheme].items()))
    for name, t in report["tables"].items():
        if names and name not in names:
            continue
        for scheme in ("temporal", "quantile"):
            for site, cells in t[scheme].items():
                lines.append(f"{name} [{scheme} {site}] " + " ".join(f"{b}={c['n']}/{c['den']}({c['share']})" for b, c in cells.items()))
    for name, t in report["quantile_tables"].items():
        for scheme in ("temporal", "quantile"):
            for site, cells in t[scheme].items():
                lines.append(f"{name} [{scheme} {site}] " + " ".join(f"{b}:q25/q50/q75={c['q25']}/{c['q50']}/{c['q75']}" for b, c in cells.items()))
    return lines


# ======================================================================================== the driver
def build_parser() -> argparse.ArgumentParser:
    ap = rsf.build_parser()
    ap.description = __doc__
    ap.set_defaults(out="out/diag")
    ap.add_argument("--diag-out", default="out/diag/e6_drift.json", help="aggregate JSON (must not be under local_only/)")
    ap.add_argument("--n-bins", type=int, default=5, help="quantile bins of the within-site t0 rank (default 5)")
    ap.add_argument("--also-n-bins", type=int, nargs="*", default=[],
                    help="also write the same tables with these other numbers of quantile bins (to <diag-out stem>_n<k>.json), e.g. 10")
    ap.add_argument("--print-all", action="store_true", help="print every table (default: the E6 / comparison / availability tables)")
    return ap


PRINT_PREFIXES = ("label.E6.", "label.E1.positive", "label.E2.positive", "label.E5.positive", "comp.E6.", "e6pos.", "avail.", "vocab.", "consistency.E6",
                  "casemix.", "timing.")


def run(a, source=None) -> dict:
    from threadpoolctl import threadpool_limits
    out = Path(a.diag_out)
    if "local_only" in out.resolve().parts:
        raise SystemExit("--diag-out holds aggregates and must NOT be under local_only/")
    cp = None
    if source is not None:
        cp = checkpoint.open_step("diag_e6_drift", a, no_resume=a.no_resume, store=source if not isinstance(source, dict) else None,
                                  inputs=[p for p in (a.cohort, a.silver, a.baselines) if p],
                                  exclude=("diag_out", "n_bins", "also_n_bins", "print_all", "out", "max_memory_gb"))
    with checkpoint.use(cp), threadpool_limits(limits=a.blas_threads or None):
        A = rsf.assemble(a)
        sites = np.asarray(A.sites, dtype=object)
        times = np.asarray(A.times)
        Y, _h = rsf.load_silver(a.silver)
        Ys = Y.reindex(A.frame["person_id"]).reset_index(drop=True)
        di = DiagInput(sites, times, Ys, A.site_labels, a.temporal_fraction)
        m, q = casemix(A)
        try:                                                    # hours from the visit start to t0 (cohort table, else baselines meta)
            blm, _c = rsf.load_baselines(a.baselines)
            hrs, hsrc = rsf.hours_since_encounter(A.frame, blm.set_index("person_id").reindex(A.frame["person_id"]).reset_index(drop=True))
            if np.isfinite(hrs).any():
                q.append(Quant("casemix.hours_from_visit_start", np.asarray(hrs, float), f"hours from the visit start to t0 ({hsrc})"))
        except (OSError, KeyError, ValueError):
            pass
        di.metrics += m
        di.quants += q
        if source is None:
            di.notes.append("no --data / --s3: the OMOP-based availability, component and vocabulary tables were not produced")
        else:
            groupings = build_groupings(sites, times, a.n_bins, a.temporal_fraction)
            e6 = (Ys["E6"] == 1).to_numpy() if "E6" in Ys else np.zeros(len(Ys), bool)
            e6a = Ys["E6"].notna().to_numpy() if "E6" in Ys else np.zeros(len(Ys), bool)
            dm, dq, notes = data_metrics(source, A.frame, sites, groupings["temporal"][1], e6, e6a, Ys)
            di.metrics += dm
            di.quants += dq
            di.notes += notes
    report = diagnose(di, a.n_bins)
    report["drift_ratio_test_over_train"] = drift_ratios(report)
    known = {str(p) for p in A.frame["person_id"]} | set(A.frame["rid"].astype(str))
    safe_write_json(out, report, known)
    for line in render_text(report, None if a.print_all else [n for n in report["tables"] if n.startswith(PRINT_PREFIXES)]):
        safe_print(line, known_ids=known)
    safe_print(f"Report: {out} (aggregate-only)", known_ids=known)
    for k in a.also_n_bins:
        extra = diagnose(di, k)
        extra["drift_ratio_test_over_train"] = drift_ratios(extra)
        path = out.with_name(f"{out.stem}_n{k}{out.suffix}")
        safe_write_json(path, extra, known)
        safe_print(f"Report: {path} (aggregate-only, {k} quantile bins)", known_ids=known)
    return report


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    from sortinghat.cohort.memguard import run_guarded
    source = None
    if a.data:
        from sortinghat import agent_safety
        agent_safety.assert_not_restricted_in_agent(a.data)
        source = data_io.open_store(a.data)
    elif a.s3:
        source = data_io.open_store(None, profile=a.profile)
    checkpoint.log_to_stderr()                  # progress goes to stderr; stdout is aggregate tables only
    try:
        return run_guarded(lambda: (run(a, source), 0)[1], a.max_memory_gb, "E6 drift diagnostic")
    finally:
        checkpoint.log_to_stderr(False)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:                       # never print a traceback (it could carry values); class name only
        print(f"diag_e6_drift failed: {type(e).__name__}", file=sys.stderr)
        raise SystemExit(1)
