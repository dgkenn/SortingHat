"""STRUCTURED silver-label extractor: OMOP tables -> anchor event table -> per-patient silver labels.

Pipeline (spec: ``docs/silver_extraction.md``; concept rules: ``configs/anchor_concepts.yaml``)::

    cohort (person_id, t0) --+
    omop_concept ------------+--> ConceptIndex
    omop_visit_occurrence ---------> encounter start per case
    omop_measurement / drug_exposure / condition_occurrence / procedure_occurrence / observation
        --column-pruned batches via data_io.iter_omop_batches, classified per batch (only matched rows are kept)-->
    flags computed upstream (profound_shock, qad_ge4, *_doubling, platelets_drop, E4a/E4b split, antidote given, ...)
    banned_evidence exclusions --> anchor event table (case_id, item, value, hours_from_t0 + audit columns)
    anchors.silver_anchor_table --> silver labels E1, E2, E4a, E5, E6, E7

Rules this module enforces:

* NO note text and NO EEG report content ever enters the labels. The tables read are ``STRUCTURED_TABLES``;
  ``reports_findings``, ``omop_note`` and ``omop_note_nlp`` are never read here (``FORBIDDEN_TABLES``). EEG-report flags
  feed ONLY the circularity audit's EEG-impression comparator (``eeg_impression_comparator``), never a positive.
* The silver labels are retrospective (the anchor windows bound them): the baselines' ``as_of`` t0 gate is deliberately NOT
  applied. Silver labels are training targets and must never be used as model INPUT features.
* Unit conversion, plausibility limits and result semantics live in ``concepts.py``; an unrecognised unit drops the row.
* All outputs that may leave a restricted job go through ``silver_report`` (n < 11 suppressed, complementary suppression).
  The event table and label table are record-level: they stay in memory or in ``local_only/``.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from .. import checkpoint as ck, data_io, schema
from ..safe_output import (SUPPRESSED, SUPPRESS_BELOW, assert_aggregate_only, safe_write_json, suppress_count,
                           suppress_proportion, write_local_only)
from .anchors import EVENT_COLUMNS, load_anchor_config, silver_anchor_table
from .banned_evidence import EvidenceSource, classify_evidence
from .concepts import (ConceptIndex, ConceptMap, STATUS_UNAVAILABLE, anchor_item_status, anchor_leaf_status,
                       label_status, load_concept_map, normalize_code, normalize_source_code,
                       validate_concept_map)
from .eeg_filter import mentions_eeg_content
from .ontology import SILVER_CIRCULARITY_LABELS

STRUCTURED_TABLES = ("omop_concept", "omop_visit_occurrence", "omop_measurement", "omop_drug_exposure",
                     "omop_condition_occurrence", "omop_procedure_occurrence", "omop_observation")
FORBIDDEN_TABLES = ("reports_findings", "omop_note", "omop_note_nlp", "imaging", "eeg_metadata")
EEG_COMPARATOR_ONLY_TABLES = ("reports_findings",)

# Column-pruned requests (columns absent from a real part are silently skipped by iter_omop_batches).
COLUMNS = {
    "concept": ["concept_id", "concept_name", "domain_id", "vocabulary_id", "concept_code"],
    "visit_occurrence": ["person_id", "visit_occurrence_id", "visit_start_datetime", "visit_end_datetime",
                         "preceding_visit_occurrence_id"],
    "measurement": ["person_id", "measurement_datetime", "measurement_date", "measurement_time",
                    "measurement_source_value", "value_as_number", "unit_source_value", "measurement_concept_id",
                    "measurement_source_concept_id", "unit_concept_id", "value_as_concept_id", "value_source_value"],
    "drug_exposure": ["person_id", "drug_exposure_start_datetime", "drug_exposure_end_datetime",
                      "drug_exposure_start_date", "drug_exposure_end_date", "drug_source_value", "drug_concept_id",
                      "route_source_value"],
    "condition_occurrence": ["person_id", "condition_start_datetime", "condition_start_date", "condition_source_value",
                             "condition_concept_id", "condition_source_concept_id", "visit_occurrence_id"],
    "procedure_occurrence": ["person_id", "procedure_datetime", "procedure_date", "procedure_source_value",
                             "procedure_concept_id", "procedure_source_concept_id"],
    "observation": ["person_id", "observation_datetime", "observation_date", "observation_source_value",
                    "value_as_string", "observation_concept_id"],
}
INTERNAL_EVENT_COLUMNS = ["source", "basis", "timing_approximate", "evidence_source", "evidence_text",
                          "evidence_code"]
EVENT_FRAME_COLUMNS = list(EVENT_COLUMNS) + INTERNAL_EVENT_COLUMNS


@dataclass
class ExtractConfig:
    """Run-time switches (thresholds and windows come from the YAML files, not from here)."""
    allow_name_fallback: bool = True          # name regex over *_source_value when concept ids are zero-filled
    include_unused_items: bool = False        # wbc / temp / HR / RR are used by no rule (SIRS removed): skip them
    exposure_lookback_days: float = 0.0       # drugs given before the encounter start still explain a tox screen
    anchor_config_path: str | None = None
    concept_config_path: str | None = None


# ============================================================================================ context
class _Ctx:
    def __init__(self, cm: ConceptMap, index: ConceptIndex, cases: pd.DataFrame, cfg: ExtractConfig):
        self.cm, self.index, self.cfg = cm, index, cfg
        self.cases = cases
        pf = cm.raw["prefilter"]
        g = cases.groupby("person_id")["t0"]
        self.lo = (g.min() - pd.Timedelta(days=pf["lookback_days"])).astype("datetime64[us]")
        self.hi = (g.max() + pd.Timedelta(days=pf["forward_days"])).astype("datetime64[us]")
        self.pids = set(int(p) for p in self.lo.index)
        self.diag: Counter = Counter()
        self.visit_start: pd.Series = pd.Series(dtype="datetime64[us]", index=pd.Index([], dtype="int64"))   # visit_id -> start

    def count(self, key: str, n: int = 1) -> None:
        if n:
            self.diag[key] += int(n)


def prepare_cohort(cohort: pd.DataFrame) -> pd.DataFrame:
    """Normalise a cohort frame to ``case_id, person_id, t0, SiteID`` (one row per case).

    ``cohort`` needs ``person_id`` and ``t0`` (EEG start); ``case_id`` defaults to ``str(person_id)`` when each person has one
    row; ``SiteID`` defaults to ``"NA"``."""
    need = {"person_id", "t0"} - set(cohort.columns)
    if need:
        raise ValueError(f"cohort needs columns {sorted(need)}")
    c = cohort.copy()
    c["person_id"] = pd.to_numeric(c["person_id"], errors="coerce")
    c["t0"] = schema.parse_datetimes(c["t0"])
    c = c[c["person_id"].notna() & c["t0"].notna()]
    c["person_id"] = c["person_id"].astype("int64")
    if "case_id" not in c.columns:
        if c["person_id"].duplicated().any():
            raise ValueError("several cohort rows per person_id: pass an explicit case_id column")
        c["case_id"] = c["person_id"].astype(str)
    c["case_id"] = c["case_id"].astype(str)
    if c["case_id"].duplicated().any():
        raise ValueError("case_id must be unique")
    if "SiteID" not in c.columns:
        c["SiteID"] = "NA"
    c["SiteID"] = c["SiteID"].astype(str)
    return c[["case_id", "person_id", "t0", "SiteID"]].reset_index(drop=True)


# ============================================================================================ helpers
def _num(d: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(d[col], errors="coerce") if col in d else pd.Series(np.nan, index=d.index, dtype=float)


def _txt(d: pd.DataFrame, col: str) -> pd.Series:
    return d[col].astype(object) if col in d else pd.Series(None, index=d.index, dtype=object)


def _isna(a) -> np.ndarray:
    """Boolean missing-mask of an object array / Series (``None`` and NaN, not lists)."""
    return np.asarray(pd.isna(np.asarray(a, dtype=object)), dtype=bool)


def _gather(per_unique: list, codes: np.ndarray) -> np.ndarray:
    """Object array ``out[i] = per_unique[codes[i]]`` (``None`` where ``codes[i] == -1``); values are stored as they are
    (no numpy broadcasting of tuples / lists)."""
    arr = np.empty(len(per_unique) + 1, dtype=object)
    for j, v in enumerate(per_unique):
        arr[j] = v
    arr[-1] = None
    return arr[codes]


def _id_hits(ids: pd.Series, mapping: Mapping) -> np.ndarray:
    """``ids.map(mapping)`` as an object array (``None`` where the id is missing / not in ``mapping``), one dict lookup per UNIQUE id."""
    codes, uniq = pd.factorize(ids)
    vals = []
    for u in uniq:
        f = float(u)
        vals.append(mapping.get(int(f)) if f.is_integer() and mapping else None)
    return _gather(vals, codes)


def _times(d: pd.DataFrame, dt_col: str, date_col: str | None = None, time_col: str | None = None):
    """Event time and a flag for date-only fallbacks (date at 00:00, plus ``time_col`` when present)."""
    nat = pd.Series(pd.NaT, index=d.index, dtype="datetime64[us]")
    t = schema.parse_datetimes(d[dt_col]) if dt_col in d else nat
    approx = pd.Series(False, index=d.index)
    if date_col and date_col in d:
        dd = schema.parse_datetimes(d[date_col])
        fill = t.isna() & dd.notna()
        base = dd.dt.normalize()
        if time_col and time_col in d:
            td = pd.to_timedelta(d[time_col].astype(object).where(d[time_col].notna(), None).astype(str),
                                 errors="coerce").fillna(pd.Timedelta(0))
            base = base + td
        else:
            approx = fill
        t = t.where(~fill, base).astype("datetime64[us]")
    return t, approx


def _restrict(d: pd.DataFrame, ctx: _Ctx) -> pd.DataFrame:
    if "person_id" not in d.columns:
        return d.iloc[0:0]
    d = d.copy()
    d["person_id"] = pd.to_numeric(d["person_id"], errors="coerce")
    d = d[d["person_id"].notna()]
    d["person_id"] = d["person_id"].astype("int64")
    return d[d["person_id"].isin(ctx.pids)].reset_index(drop=True)


def _in_window(d: pd.DataFrame, t: pd.Series, ctx: _Ctx, unbounded_lookback: pd.Series | None = None) -> pd.Series:
    lo = d["person_id"].map(ctx.lo)
    hi = d["person_id"].map(ctx.hi)
    lo_ok = t >= lo
    if unbounded_lookback is not None:
        lo_ok |= unbounded_lookback
    return t.notna() & lo_ok & (t <= hi)


def _empty(cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=object) for c in cols})


_OBS_NEG = re.compile(r"^\s*(no|n|none|false|0|denied|denies|absent|negative|neg)\b", re.I)
MEAS_COLS = ["person_id", "t_event", "item", "basis", "value", "status", "src_text", "approx"]
COND_COLS = ["person_id", "t_event", "item", "basis", "evidence_code", "approx", "src_text"]
DRUG_COLS = ["person_id", "t_start", "t_end", "group", "ingredient", "route", "src_text", "basis"]
VISIT_COLS = ["person_id", "visit_id", "v_start", "v_end", "preceding_id"]


# ======================================================================================= classifiers
def classify_visits(d: pd.DataFrame, ctx: _Ctx) -> pd.DataFrame:
    d = _restrict(d, ctx)
    if d.empty:
        return _empty(VISIT_COLS)
    return pd.DataFrame({
        "person_id": d["person_id"], "visit_id": _num(d, "visit_occurrence_id"),
        "v_start": schema.parse_datetimes(d["visit_start_datetime"]) if "visit_start_datetime" in d else pd.NaT,
        "v_end": schema.parse_datetimes(d["visit_end_datetime"]) if "visit_end_datetime" in d else pd.NaT,
        "preceding_id": _num(d, "preceding_visit_occurrence_id")})


def classify_measurements(d: pd.DataFrame, ctx: _Ctx) -> pd.DataFrame:
    """Rows of ``omop_measurement`` matched to anchor / support items, with converted values and result status.

    Vectorised: names, concept ids and units are resolved once per UNIQUE value and gathered back to the rows; only the rows
    that matched an item (or a tox agent) reach Python loops."""
    cm, ix, cfg = ctx.cm, ctx.index, ctx.cfg
    d = _restrict(d, ctx)
    if d.empty:
        return _empty(MEAS_COLS)
    t, approx = _times(d, "measurement_datetime", "measurement_date", "measurement_time")
    keep = _in_window(d, t, ctx)
    ctx.count("meas_rows_outside_window_or_untimed", int((~keep).sum()))
    d, t, approx = d[keep].reset_index(drop=True), t[keep].reset_index(drop=True), approx[keep].reset_index(drop=True)
    if d.empty:
        return _empty(MEAS_COLS)
    names = _txt(d, "measurement_source_value")
    ncodes, nuniq = pd.factorize(names)
    r1 = _id_hits(_num(d, "measurement_concept_id"), ix.meas)
    r2 = _id_hits(_num(d, "measurement_source_concept_id"), ix.meas)
    loinc = {c: v for (s, c), v in cm.meas_code.items() if s == "LOINC"}
    r3 = _gather([loinc.get(str(u).strip()) for u in nuniq], ncodes)
    hit = r1.copy()
    basis = np.where(~_isna(r1), "concept", "").astype(object)
    routes = [(r2, "source_concept"), (r3, "source_code")]
    if cfg.allow_name_fallback:
        r4 = _gather([[(k, "") for k in cm.classify_measurement_name(u)] or None for u in nuniq], ncodes)
        routes.append((r4, "name"))
    for r, b in routes:
        m = _isna(hit) & ~_isna(r)
        hit[m] = r[m]
        basis[m] = b
    matched_pos = np.flatnonzero(~_isna(hit))
    rows = [(int(i), item, unit, basis[i]) for i in matched_pos for item, unit in hit[i]]
    out = []
    m = None
    if rows:
        m = pd.DataFrame(rows, columns=["i", "item", "unit_default", "basis"])
        ctx.count("meas_rows_matched", len(m))
        for b, n in m["basis"].value_counts().items():
            ctx.count(f"meas_match_basis_{b}", n)
        val = _num(d, "value_as_number")
        unit = _txt(d, "unit_source_value")
        blank = unit.isna() | (unit.astype(str).str.strip() == "")
        if "unit_concept_id" in d:
            unit = pd.Series(np.where(blank.to_numpy(), _id_hits(_num(d, "unit_concept_id"), ix.unit_code),
                                      unit.to_numpy(dtype=object)), index=d.index, dtype=object)
        vname = _id_hits(_num(d, "value_as_concept_id"), ix.value_name) if "value_as_concept_id" in d else \
            np.full(len(d), None, dtype=object)
        vtext = _txt(d, "value_source_value").to_numpy(dtype=object)
        valn = val.to_numpy(dtype=float)
        status_memo: dict = {}

        def status(item: str, j: int) -> str:
            tx, vn, nu = vtext[j], vname[j], valn[j]
            key = (item, None if tx is None or tx != tx else tx, None if vn is None or vn != vn else vn,
                   None if nu != nu else nu)
            got = status_memo.get(key)
            if got is None:
                got = status_memo[key] = cm.result_status(item, tx, vn, nu)
            return got
        for item, g in m.groupby("item", sort=False):
            sp = cm.specs[item]
            ii = g["i"].to_numpy()
            if sp.kind in ("quant", "support"):
                if sp.unused and not cfg.include_unused_items:
                    continue
                conv, st = cm.convert_series(item, val.iloc[ii].reset_index(drop=True),
                                             unit.iloc[ii].reset_index(drop=True),
                                             g["unit_default"].reset_index(drop=True))
                for s_, n in st.value_counts().items():
                    ctx.count(f"meas_{'rows_converted' if s_ == 'ok' else 'dropped_' + s_}", n)
                ok = (st == "ok").to_numpy()
                if ok.any():
                    out.append(pd.DataFrame({
                        "person_id": d["person_id"].iloc[ii[ok]].to_numpy(), "t_event": t.iloc[ii[ok]].to_numpy(),
                        "item": item, "basis": g["basis"].to_numpy()[ok], "value": conv.to_numpy()[ok],
                        "status": "ok", "src_text": names.iloc[ii[ok]].to_numpy(),
                        "approx": approx.iloc[ii[ok]].to_numpy()}))
            else:                                                       # qualitative
                sts = [status(item, j) for j in ii]
                for s_, n in Counter(sts).items():
                    ctx.count(f"qual_{item}_{s_}", n)
                out.append(pd.DataFrame({
                    "person_id": d["person_id"].iloc[ii].to_numpy(), "t_event": t.iloc[ii].to_numpy(), "item": item,
                    "basis": g["basis"].to_numpy(), "value": np.nan, "status": sts,
                    "src_text": names.iloc[ii].to_numpy(), "approx": approx.iloc[ii].to_numpy()}))
    # tox screens: rows that no item claimed
    tox_u = [cm.classify_tox_name(u) for u in nuniq]
    tox_flag = np.array([bool(x) for x in tox_u] + [False], dtype=bool)      # last slot: missing name (code -1)
    cand = np.flatnonzero(tox_flag[ncodes])
    if rows:
        taken = np.zeros(len(d), dtype=bool)
        taken[m["i"].to_numpy()] = True
        cand = cand[~taken[cand]]
    if len(cand):
        vtext = _txt(d, "value_source_value").to_numpy(dtype=object)
        vname = _id_hits(_num(d, "value_as_concept_id"), ix.value_name) if "value_as_concept_id" in d else \
            np.full(len(d), None, dtype=object)
        valn = _num(d, "value_as_number").to_numpy(dtype=float)
        memo: dict = {}
        trows = []
        for i in cand:
            for ag in tox_u[ncodes[i]]:
                tx, vn, nu = vtext[i], vname[i], valn[i]
                key = (ag, None if tx is None or tx != tx else tx, None if vn is None or vn != vn else vn,
                       None if nu != nu else nu)
                got = memo.get(key)
                if got is None:
                    got = memo[key] = cm.result_status(f"tox:{ag}", tx, vn, nu)
                trows.append((int(i), ag, got))
        if trows:
            tr = pd.DataFrame(trows, columns=["i", "agent", "st"])
            ctx.count("tox_rows_matched", len(tr))
            ii = tr["i"].to_numpy()
            out.append(pd.DataFrame({
                "person_id": d["person_id"].iloc[ii].to_numpy(), "t_event": t.iloc[ii].to_numpy(),
                "item": "tox:" + tr["agent"].to_numpy(), "basis": "name", "value": np.nan, "status": tr["st"].to_numpy(),
                "src_text": names.iloc[ii].to_numpy(), "approx": approx.iloc[ii].to_numpy()}))
    return pd.concat(out, ignore_index=True) if out else _empty(MEAS_COLS)


def _code_event_rows(d: pd.DataFrame, ctx: _Ctx, *, table: str, dt_col: str, date_col: str, src_col: str,
                     cid_col: str, scid_col: str | None, id_map: Mapping, src_hits: Callable, text_col: str | None
                     ) -> pd.DataFrame:
    """Shared logic for conditions and procedures: concept ids, then source concept ids, then code text, then wording.
    Vectorised: code / wording lookups run once per UNIQUE source value; only rows with a hit reach the Python loop."""
    cm = ctx.cm
    d = _restrict(d, ctx)
    if d.empty:
        return _empty(COND_COLS)
    t, approx = _times(d, dt_col, date_col)
    if table == "condition" and "visit_occurrence_id" in d and len(ctx.visit_start):    # diagnosis timing: visit-start fallback
        vid = _num(d, "visit_occurrence_id")
        pos = ctx.visit_start.index.get_indexer(vid.fillna(-1).astype("int64"))
        pos = np.where(vid.notna().to_numpy(), pos, -1)
        fb = pd.Series(ctx.visit_start.to_numpy()[np.where(pos < 0, 0, pos)], index=d.index)
        fb = fb.where(pos >= 0, pd.NaT)
        use = t.isna() & fb.notna()
        t = t.where(~use, pd.to_datetime(fb).astype("datetime64[us]")).astype("datetime64[us]")
        approx = approx | use
        ctx.count("dx_time_from_visit_start", int(use.sum()))
    src = _txt(d, src_col)
    scodes, suniq = pd.factorize(src)
    none_arr = np.full(len(d), None, dtype=object)
    h1 = _id_hits(_num(d, cid_col), id_map) if cid_col in d else none_arr
    h2 = _id_hits(_num(d, scid_col), id_map) if scid_col and scid_col in d else none_arr
    h3 = _gather([src_hits(u) or None for u in suniq], scodes)
    has = ~_isna(h1) | ~_isna(h2) | ~_isna(h3)
    text_rows: dict[int, tuple] = {}
    if text_col:
        th = [cm.text_hits(table, u) for u in suniq]
        flag = np.array([bool(x) for x in th] + [False], dtype=bool)
        for i in np.flatnonzero(flag[scodes] & ~has):
            text_rows[int(i)] = th[scodes[i]]
    rows = []
    cand = np.flatnonzero(has)
    if text_rows:
        cand = np.union1d(cand, np.fromiter(text_rows, dtype="int64"))
    for i in cand:
        if has[i]:
            for h, b in ((h1[i], "concept"), (h2[i], "source_concept"), (h3[i], "source_code")):
                if isinstance(h, (tuple, list)) and len(h):
                    rows += [(int(i), it, b) for it in h]
                    break
        else:
            rows += [(int(i), it, "name") for it in text_rows[int(i)]]
    if not rows:
        return _empty(COND_COLS)
    r = pd.DataFrame(rows, columns=["i", "item", "basis"])
    ii = r["i"].to_numpy()
    tt = t.iloc[ii].reset_index(drop=True)
    p = d["person_id"].iloc[ii].reset_index(drop=True)
    unb = pd.Series([bool(ctx.cm.event_items.get(it, {}).get("lookback_days", 1) is None) for it in r["item"]])
    keep = _in_window(pd.DataFrame({"person_id": p}), tt, ctx, unb).to_numpy()
    ctx.count("code_rows_outside_window_or_untimed", int((~keep).sum()))
    r, ii = r[keep], ii[keep]
    ctx.count("code_rows_matched", len(r))
    for b, n in r["basis"].value_counts().items():
        ctx.count(f"code_match_basis_{b}", n)
    code = src.iloc[ii].map(normalize_source_code).to_numpy()
    return pd.DataFrame({"person_id": d["person_id"].iloc[ii].to_numpy(), "t_event": t.iloc[ii].to_numpy(),
                         "item": r["item"].to_numpy(), "basis": r["basis"].to_numpy(), "evidence_code": code,
                         "approx": approx.iloc[ii].to_numpy(), "src_text": src.iloc[ii].to_numpy()})


def classify_conditions(d: pd.DataFrame, ctx: _Ctx) -> pd.DataFrame:
    return _code_event_rows(d, ctx, table="condition", dt_col="condition_start_datetime",
                            date_col="condition_start_date", src_col="condition_source_value",
                            cid_col="condition_concept_id", scid_col="condition_source_concept_id",
                            id_map=ctx.index.cond, src_hits=ctx.cm.condition_source_hits, text_col=None)


def classify_procedures(d: pd.DataFrame, ctx: _Ctx) -> pd.DataFrame:
    return _code_event_rows(d, ctx, table="procedure", dt_col="procedure_datetime", date_col="procedure_date",
                            src_col="procedure_source_value", cid_col="procedure_concept_id",
                            scid_col="procedure_source_concept_id", id_map=ctx.index.proc,
                            src_hits=ctx.cm.procedure_source_hits, text_col="procedure_source_value")


def classify_observations(d: pd.DataFrame, ctx: _Ctx) -> pd.DataFrame:
    """Observation rows whose source text names an arrest (wording rule; weaker than codes, flagged basis ``name``)."""
    cm = ctx.cm
    d = _restrict(d, ctx)
    if d.empty:
        return _empty(COND_COLS)
    t, approx = _times(d, "observation_datetime", "observation_date")
    src, val = _txt(d, "observation_source_value"), _txt(d, "value_as_string")
    scodes, suniq = pd.factorize(src)
    th = [cm.text_hits("observation", u) for u in suniq]
    flag = np.array([bool(x) for x in th] + [False], dtype=bool)
    valv = val.to_numpy(dtype=object)
    rows = []
    for i in np.flatnonzero(flag[scodes]):
        hits = th[scodes[i]]
        v = valv[i]
        if v is not None and v == v and (_OBS_NEG.search(str(v)) or cm.result_re["negative"].search(str(v).strip().lower())):
            ctx.count("obs_negated")
            continue
        rows += [(int(i), it) for it in hits]
    if not rows:
        return _empty(COND_COLS)
    r = pd.DataFrame(rows, columns=["i", "item"])
    ii = r["i"].to_numpy()
    keep = _in_window(pd.DataFrame({"person_id": d["person_id"].iloc[ii].reset_index(drop=True)}),
                      t.iloc[ii].reset_index(drop=True), ctx).to_numpy()
    r, ii = r[keep], ii[keep]
    ctx.count("obs_rows_matched", len(r))
    return pd.DataFrame({"person_id": d["person_id"].iloc[ii].to_numpy(), "t_event": t.iloc[ii].to_numpy(),
                         "item": r["item"].to_numpy(), "basis": "name", "evidence_code": "",
                         "approx": approx.iloc[ii].to_numpy(),
                         "src_text": (src.iloc[ii] + " " + val.iloc[ii].fillna("").astype(str)).to_numpy()})


def classify_drugs(d: pd.DataFrame, ctx: _Ctx) -> pd.DataFrame:
    cm, ix = ctx.cm, ctx.index
    d = _restrict(d, ctx)
    if d.empty:
        return _empty(DRUG_COLS)
    ts, _ = _times(d, "drug_exposure_start_datetime", "drug_exposure_start_date")
    te, _ = _times(d, "drug_exposure_end_datetime", "drug_exposure_end_date")
    keep = _in_window(d, ts, ctx)
    d, ts, te = d[keep].reset_index(drop=True), ts[keep].reset_index(drop=True), te[keep].reset_index(drop=True)
    if d.empty:
        return _empty(DRUG_COLS)
    src, route = _txt(d, "drug_source_value"), _txt(d, "route_source_value")
    c_hits = _id_hits(_num(d, "drug_concept_id"), ix.drug)
    scodes, suniq = pd.factorize(src)
    th = [cm.drug_text_hits(u) for u in suniq]
    flag = np.array([bool(x) for x in th] + [False], dtype=bool)
    cand = np.flatnonzero(np.array([isinstance(c, tuple) for c in c_hits], dtype=bool) | flag[scodes])
    srcv, routev = src.to_numpy(dtype=object), route.to_numpy(dtype=object)
    rows = []
    for i in cand:
        seen: dict = {}
        ch = c_hits[i]
        if isinstance(ch, tuple):
            for g, ing in ch:
                seen[(g, ing)] = "concept"
        for g, ing in th[scodes[i]] if scodes[i] >= 0 else ():
            seen.setdefault((g, ing), "name")
        for (g, ing), b in seen.items():
            if g == "antimicrobial" and not cm.abx_route_ok(ing, routev[i], srcv[i]):
                ctx.count("abx_rows_route_rejected")
                continue
            rows.append((int(i), g, ing, b))
    if not rows:
        return _empty(DRUG_COLS)
    r = pd.DataFrame(rows, columns=["i", "group", "ingredient", "basis"])
    ii = r["i"].to_numpy()
    ctx.count("drug_rows_matched", len(r))
    return pd.DataFrame({"person_id": d["person_id"].iloc[ii].to_numpy(), "t_start": ts.iloc[ii].to_numpy(),
                         "t_end": te.iloc[ii].to_numpy(), "group": r["group"].to_numpy(),
                         "ingredient": r["ingredient"].to_numpy(), "route": route.iloc[ii].to_numpy(),
                         "src_text": src.iloc[ii].to_numpy(), "basis": r["basis"].to_numpy()})


# ======================================================================================== table loading
def _concat(parts: list[pd.DataFrame], cols: list[str]) -> pd.DataFrame:
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else _empty(cols)


@dataclass
class Classified:
    visits: pd.DataFrame
    meas: pd.DataFrame
    cond: pd.DataFrame
    proc: pd.DataFrame
    obs: pd.DataFrame
    drug: pd.DataFrame


def _guard_forbidden(names: Iterable[str]) -> list[str]:
    return sorted(set(names) & set(FORBIDDEN_TABLES))


def classify_tables(tables: Mapping[str, pd.DataFrame], ctx: _Ctx) -> Classified:
    """In-memory path (tests, synthetic). Forbidden tables in ``tables`` are ignored and counted, never read."""
    ctx.count("forbidden_tables_ignored", len(_guard_forbidden(tables)))
    if "omop_concept" in tables:
        ctx.index.ingest(tables["omop_concept"])
    empty = pd.DataFrame()
    visits = classify_visits(tables.get("omop_visit_occurrence", empty), ctx)
    _set_visit_starts(visits, ctx)
    return Classified(
        visits, classify_measurements(tables.get("omop_measurement", empty), ctx),
        classify_conditions(tables.get("omop_condition_occurrence", empty), ctx),
        classify_procedures(tables.get("omop_procedure_occurrence", empty), ctx),
        classify_observations(tables.get("omop_observation", empty), ctx),
        classify_drugs(tables.get("omop_drug_exposure", empty), ctx))


def _set_visit_starts(visits: pd.DataFrame, ctx: _Ctx) -> None:
    """``ctx.visit_start``: visit_id -> start (a Series with a unique int64 index; the last row wins, like the old dict)."""
    v = visits[visits["visit_id"].notna() & visits["v_start"].notna()]
    s = pd.Series(v["v_start"].astype("datetime64[us]").to_numpy(), index=pd.Index(v["visit_id"].astype("int64").to_numpy()))
    ctx.visit_start = s[~s.index.duplicated(keep="last")]


def classify_store(store, ctx: _Ctx, on_error: Callable | None = None) -> Classified:
    """Streaming path: each table is read in column-pruned, cohort-filtered Arrow batches and classified ROW GROUP by row group,
    so only matched rows are held in memory. ``store`` is a ``data_io.LocalStore`` or the S3 client (human-run only).

    Restart safety (active checkpoint): the classified concept map is stored once (``stage:concept-index``), then every
    (table, row group) result, i.e. its matched anchor-event rows plus the diagnostic counts it added, is stored atomically
    as soon as it is classified. A relaunch skips every stored unit WITHOUT reading it, so it resumes at the first unfinished
    row group. Unit keys carry the cohort, config, concept map, concept index, the column list and (conditions) the visit starts.
    A part that could not be read (``on_error``) is simply not stored and is retried on the next run."""
    pids = sorted(ctx.pids)
    cp = ck.active()

    def units(table: str, fn: Callable, cols: list[str], extra=None, label: str | None = None) -> pd.DataFrame:
        prog = ck.UnitProgress("silver_labels", label or table, enabled=cp is not None)
        key = ck.digest("silver-unit-v2", table, fn.__name__, cols, ctx.cases, ctx.cfg, ctx.cm.raw, ck.digest(ctx.index.state()),
                        extra) if cp is not None else None
        loaded: dict = {}

        def have(uid: str) -> bool:
            if cp is None:
                return False
            got = cp.get(f"silver:{key}:{uid}")
            if got is ck.MISS:
                return False
            loaded[uid] = got
            return True
        frames = []
        for uid, batches, total in data_io.iter_omop_units(table, person_ids=pids, columns=cols, s3=store, on_error=on_error,
                                                           batch_rows=1 << 18, skip=have if cp is not None else None):
            if batches is None:                                   # stored result: replay its frame and diagnostic counts
                res, delta = loaded.pop(uid)
                ctx.diag.update(delta)
                frames.append(res)
                prog.tick(True, total)
                continue
            before = Counter(ctx.diag)
            res = _concat([fn(b.to_pandas(), ctx) for b in batches], [])
            if cp is not None:
                cp.put(f"silver:{key}:{uid}", (res, dict(Counter(ctx.diag) - before)))
            frames.append(res)
            prog.tick(False, total)
        prog.emit(final=True)
        return _concat(frames, [])

    ctx.index = _concept_index(store, ctx, on_error, cp)
    visits = units("visit_occurrence", classify_visits, COLUMNS["visit_occurrence"])
    if visits.empty:
        visits = _empty(VISIT_COLS)
    _set_visit_starts(visits, ctx)
    return Classified(visits,
                      _or_empty(units("measurement", classify_measurements, COLUMNS["measurement"]), MEAS_COLS),
                      _or_empty(units("condition_occurrence", classify_conditions, COLUMNS["condition_occurrence"],
                                      extra=ctx.visit_start), COND_COLS),
                      _or_empty(units("procedure_occurrence", classify_procedures, COLUMNS["procedure_occurrence"]),
                                COND_COLS),
                      _or_empty(units("observation", classify_observations, COLUMNS["observation"]), COND_COLS),
                      _or_empty(units("drug_exposure", classify_drugs, COLUMNS["drug_exposure"]), DRUG_COLS))


def _concept_index(store, ctx: _Ctx, on_error: Callable | None, cp) -> ConceptIndex:
    """The classified concept map, built once: omop_concept is streamed row group by row group into ``ConceptIndex.ingest``
    (vectorised) and the resulting index is stored as one checkpoint unit. A restart loads it in a moment."""
    name = "stage:concept-index:" + ck.digest("v2", ctx.cm.raw) if cp is not None else None
    if cp is not None:
        got = cp.get(name)
        if got is not ck.MISS:
            ctx.index.load_state(got)
            ck.log("silver_labels: concept map loaded from checkpoint")
            return ctx.index
    failed = []
    handler = on_error if on_error is None else (lambda k, e: (failed.append(k), on_error(k, e)))
    prog = ck.UnitProgress("silver_labels", "concept", enabled=cp is not None)
    for _uid, batches, total in data_io.iter_omop_units("concept", columns=COLUMNS["concept"], s3=store, on_error=handler,
                                                       batch_rows=1 << 20):
        for b in batches:
            ctx.index.ingest(b)
        prog.tick(False, total)
    prog.emit(final=True)
    if cp is not None and not failed:
        cp.put(name, ctx.index.state())
        ck.log("silver_labels: concept map done")
    return ctx.index


def _or_empty(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df if len(df.columns) else _empty(cols)


# =============================================================================================== encounter
def encounter_starts(cases: pd.DataFrame, visits: pd.DataFrame, fallback_days: float, chain_hops: int) -> pd.Series:
    """Encounter start per case: the visit containing t0, extended back along ``preceding_visit_occurrence_id``
    (an ED visit before admission belongs to the encounter); ``t0 - fallback_days`` when no visit covers t0."""
    out = {}
    byp = {p: g for p, g in visits.groupby("person_id")} if len(visits) else {}
    for r in cases.itertuples(index=False):
        start = r.t0 - pd.Timedelta(days=fallback_days)
        g = byp.get(r.person_id)
        if g is not None:
            cov = g[(g["v_start"] <= r.t0) & (g["v_end"].isna() | (g["v_end"] >= r.t0))].sort_values("v_start")
            if len(cov):
                row = cov.iloc[-1]
                start = row["v_start"]
                prev = row["preceding_id"]
                ids = g.set_index("visit_id")
                for _ in range(chain_hops):
                    if prev != prev or prev not in ids.index:
                        break
                    pr = ids.loc[prev]
                    pr = pr.iloc[0] if isinstance(pr, pd.DataFrame) else pr
                    if pr["v_start"] == pr["v_start"]:
                        start = min(start, pr["v_start"])
                    prev = pr["preceding_id"]
        out[r.case_id] = start
    return pd.Series(out).astype("datetime64[us]")


# =========================================================================================== event build
@dataclass
class AnchorEvents:
    events: pd.DataFrame
    covariates: pd.DataFrame           # case_id-indexed boolean E4b hints
    diagnostics: dict
    gaps: list[str]


def _join_cases(df: pd.DataFrame, cases: pd.DataFrame, tcol: str = "t_event") -> pd.DataFrame:
    """Attach every case of the person: adds case_id, t0, enc_start, hours_from_t0."""
    if df.empty:
        return df.assign(case_id=pd.Series(dtype=object), t0=pd.NaT, enc_start=pd.NaT, hours_from_t0=np.nan)
    j = df.merge(cases[["case_id", "person_id", "t0", "enc_start"]], on="person_id", how="inner")
    j["hours_from_t0"] = (j[tcol] - j["t0"]).dt.total_seconds() / 3600.0
    return j


def _mk(j: pd.DataFrame, item: str, *, value=1.0, source: str, basis=None, approx=None,
        ev_source=EvidenceSource.OBJECTIVE, text=None, code=None) -> pd.DataFrame:
    """Anchor-event rows for the joined frame ``j`` (needs case_id, hours_from_t0). Audit columns default to the
    frame's own basis / approx / src_text / evidence_code columns when present."""
    def col(v, name, default):
        if v is not None:
            return v
        return j[name].to_numpy() if name in j.columns else default
    return pd.DataFrame({
        "case_id": j["case_id"].to_numpy(), "item": item, "value": value, "hours_from_t0": j["hours_from_t0"].to_numpy(),
        "source": source, "basis": col(basis, "basis", ""), "timing_approximate": col(approx, "approx", False),
        "evidence_source": ev_source, "evidence_text": col(text, "src_text", ""),
        "evidence_code": col(code, "evidence_code", "")})


def _prior_extreme(rows: pd.DataFrame, kind: str) -> pd.Series:
    """Running min/max of ``value`` per case over EARLIER rows (rows sorted by case, time)."""
    g = rows.groupby("case_id")["value"]
    run = g.cummin() if kind == "min" else g.cummax()
    return run.groupby(rows["case_id"]).shift(1)


def _flag_events(rows: pd.DataFrame, item: str, mask: pd.Series) -> pd.DataFrame:
    sel = rows[mask]
    return _mk(sel, item, source=f"derived:{item}", basis="concept_or_name") if len(sel) else _empty(EVENT_FRAME_COLUMNS)


def shock_onsets(series: pd.DataFrame, map_lt: float, min_minutes: float, max_gap_minutes: float) -> pd.DataFrame:
    """Sustained-hypotension runs from a MAP series (columns ``person_id, t_event, map``): a run is consecutive readings
    < ``map_lt`` with no normal reading between them and gaps <= ``max_gap_minutes``; it qualifies when first-to-last low
    reading spans >= ``min_minutes``. Returns one row per qualifying run: ``person_id, t_event`` (= run onset)."""
    if series.empty:
        return pd.DataFrame({"person_id": pd.Series(dtype="int64"), "t_event": pd.Series(dtype="datetime64[us]")})
    s = series.sort_values(["person_id", "t_event"], kind="stable").reset_index(drop=True)
    low = (s["map"] < map_lt).to_numpy()
    same = (s["person_id"] == s["person_id"].shift()).to_numpy()
    dt = (s["t_event"] - s["t_event"].shift()).dt.total_seconds().to_numpy() / 60.0
    prev_low = np.r_[False, low[:-1]]
    cont = low & prev_low & same & (dt <= max_gap_minutes)
    newrun = low & ~cont
    rid = np.where(low, np.cumsum(newrun), 0)
    g = s[low].assign(rid=rid[low]).groupby("rid")["t_event"].agg(["first", "last"])
    pid = s[low].assign(rid=rid[low]).groupby("rid")["person_id"].first()
    dur = (g["last"] - g["first"]).dt.total_seconds() / 60.0
    ok = dur >= min_minutes
    return pd.DataFrame({"person_id": pid[ok].to_numpy(), "t_event": g.loc[ok, "first"].to_numpy()})


def _day(t: pd.Series) -> pd.Series:
    return ((t.dt.normalize() - pd.Timestamp("1970-01-01")).dt.days).astype("int64")


def qad_cultures(case_cultures: dict, case_abx: dict, p: dict) -> list:
    """CDC-ASE-style QAD rule. ``case_cultures``: case -> [(day, Timestamp)]; ``case_abx``: case -> {agent: set(days)}.
    Returns ``[(case_id, Timestamp)]`` for each culture day with >= ``min_days`` consecutive qualifying antimicrobial days
    starting within ``window_days`` of the culture day and beginning with a NEW agent (not given the 2 prior days).
    A single missing day between administrations of the same agent is bridged. No death/transfer shortcut."""
    out = []
    w, need, newlb = int(p["window_days"]), int(p["min_days"]), int(p["new_agent_lookback_days"])
    gap = int(p["same_agent_gap_days"])
    for cid, cults in case_cultures.items():
        agents = case_abx.get(cid)
        if not agents:
            continue
        bridged = {}
        for a, days in agents.items():
            b = set(days)
            if gap >= 1:
                b |= {d + 1 for d in days if d + 2 in days and d + 1 not in days}
            bridged[a] = b
        union = set().union(*bridged.values())
        by_day: dict = {}
        for day, ts in cults:
            by_day[day] = min(ts, by_day.get(day, ts))
        for day, ts in sorted(by_day.items()):
            for s in range(day - w, day + w + 1):
                if s not in union or not all((s + k) in union for k in range(need)):
                    continue
                if any(s in agents[a] and all((s - k) not in agents[a] for k in range(1, newlb + 1)) for a in agents):
                    out.append((cid, ts))
                    break
    return out


def _exposed_before(drugs: pd.DataFrame, ingredients: set, ref: pd.DataFrame, lookback_days: float) -> pd.Series:
    """For each row of ``ref`` (case_id, t_ref, enc_start): was any ``ingredients`` exposure STARTED in the encounter at or
    before t_ref? (Order versus administration is not distinguished: every exposure row counts, which is the conservative
    direction for E4a.)"""
    if ref.empty:
        return pd.Series(dtype=bool)
    d = drugs[drugs["ingredient"].isin(ingredients)][["case_id", "t_start"]]
    if d.empty:
        return pd.Series(False, index=ref.index)
    m = ref[["case_id", "t_ref", "enc_start"]].reset_index().merge(d, on="case_id", how="left")
    lo = m["enc_start"] - pd.Timedelta(days=lookback_days)
    hit = m["t_start"].notna() & (m["t_start"] >= lo) & (m["t_start"] <= m["t_ref"])
    res = hit.groupby(m["index"]).any()
    return res.reindex(ref.index).fillna(False).astype(bool)


def build_anchor_events(cl: Classified, cases: pd.DataFrame, ctx: _Ctx, anchor_cfg: dict) -> AnchorEvents:
    cm, cfg, raw = ctx.cm, ctx.cfg, ctx.cm.raw
    enc = raw["encounter"]
    cases = cases.copy()
    cases["enc_start"] = cases["case_id"].map(encounter_starts(cases, cl.visits, enc["fallback_days"], enc["chain_hops"]))
    ctx.count("visit_fallback_cases", int((cases["enc_start"] == cases["t0"] - pd.Timedelta(days=enc["fallback_days"])).sum()))
    items = set(anchor_cfg["items"])
    parts: list[pd.DataFrame] = []
    M = _join_cases(cl.meas, cases)
    ok = M[(M["status"] == "ok")] if len(M) else M

    # 1. direct quantitative anchor items
    for item, sp in cm.specs.items():
        if sp.kind != "quant" or item not in items:
            continue
        sel = ok[ok["item"] == item]
        if len(sel):
            parts.append(_mk(sel, item, value=sel["value"].to_numpy(), source="measurement"))

    # 2. qualitative results
    qual = M[M["item"].isin([k for k, s in cm.specs.items() if s.kind == "qual"])] if len(M) else M
    for item in ("blood_culture_drawn",):
        sel = qual[qual["item"] == item]
        if len(sel):
            parts.append(_mk(sel, item, source="measurement"))
    for item in ("blood_culture_pos", "csf_culture_pos", "csf_pcr_pos", "csf_autoimmune_ab_pos"):
        sel = qual[(qual["item"] == item) & (qual["status"] == "positive")] if len(qual) else qual
        if len(sel):
            parts.append(_mk(sel, item, source="measurement"))

    # 3. code / wording events (conditions, procedures, observations)
    esrd_cases: set = set()
    for frame, kind in ((cl.cond, "cond"), (cl.proc, "proc"), (cl.obs, "obs")):
        J = _join_cases(frame, cases)
        if J.empty:
            continue
        esrd = J[J["item"] == "esrd"]
        esrd_cases |= set(esrd.loc[esrd["hours_from_t0"] <= 24 * 7, "case_id"])
        sel = J[J["item"].isin(items)]
        if sel.empty:
            continue
        if kind == "cond":
            evs = EvidenceSource.ICD
        elif kind == "proc":
            evs = np.where(sel["basis"].to_numpy() == "name", EvidenceSource.NOTE_SENTENCE, EvidenceSource.OBJECTIVE)
        else:
            evs = EvidenceSource.NOTE_SENTENCE
        for item, g in sel.groupby("item"):
            parts.append(_mk(g, item, source="code" if kind != "obs" else "wording",
                             ev_source=evs if isinstance(evs, str) else evs[sel.index.get_indexer(g.index)]))

    # 4. drugs
    D = _join_cases(cl.drug.assign(t_event=cl.drug["t_start"]), cases) if len(cl.drug) else cl.drug
    if len(D):
        D["hours_from_t0"] = (D["t_start"] - D["t0"]).dt.total_seconds() / 3600.0
    cov = pd.DataFrame(index=cases["case_id"])
    for col in ("e4b_tox_inhospital", "e4b_antidote_reversal", "e4b_sedative_exposure"):
        cov[col] = False
    e4b_cfg = raw["e4b_covariates"]
    lookback = cfg.exposure_lookback_days
    if len(D):
        # vasopressor initiation
        vp = D[D["group"] == "vasopressor"].sort_values(["case_id", "t_start"])
        if len(vp):
            wash = pd.Timedelta(hours=raw["drug_groups"]["vasopressor"]["washout_hours"])
            prev = vp.groupby("case_id")["t_start"].shift()
            init = vp[(prev.isna() | ((vp["t_start"] - prev) >= wash)) & (vp["t_start"] >= vp["enc_start"])]
            if len(init):
                parts.append(_mk(init, "vasopressor_initiation", source="drug", approx=False))
        # antidote given (E4a) vs reversal of an in-hospital agent (E4b)
        ad = D[D["group"] == "antidote"].copy()
        if len(ad):
            pair_of = {ing: d.get("pairs_with") for ing, d in raw["drug_groups"]["antidote"]["ingredients"].items()}
            ad["pair"] = ad["ingredient"].map(pair_of)
            ad["reversal"] = False
            for pair, g in ad[ad["pair"].notna()].groupby("pair"):
                ref = pd.DataFrame({"case_id": g["case_id"], "t_ref": g["t_start"], "enc_start": g["enc_start"]})
                ad.loc[g.index, "reversal"] = _exposed_before(D, set(raw["drug_groups"]["exposure_sets"][pair]), ref,
                                                              lookback).to_numpy()
            given = ad[~ad["reversal"]]
            if len(given):
                parts.append(_mk(given, "antidote_response", source="drug:antidote_given", approx=False))
            rev = ad[ad["reversal"] & ad["hours_from_t0"].between(*e4b_cfg["antidote_reversal_window_hours"])]
            cov.loc[rev["case_id"].unique(), "e4b_antidote_reversal"] = True
        sd = D[(D["group"] == "sedative") & D["hours_from_t0"].between(*e4b_cfg["sedative_exposure_window_hours"])]
        cov.loc[sd["case_id"].unique(), "e4b_sedative_exposure"] = True

    # 5. tox screens: E4a (agent not given in hospital) vs E4b
    if len(M):
        tx = M[M["item"].str.startswith("tox:") & (M["status"] == "positive")].copy()
        if len(tx):
            tx["agent"] = tx["item"].str[4:]
            tx["in_hosp"] = False
            for agent, g in tx.groupby("agent"):
                ref = pd.DataFrame({"case_id": g["case_id"], "t_ref": g["t_event"], "enc_start": g["enc_start"]})
                tx.loc[g.index, "in_hosp"] = _exposed_before(D, set(cm.tox[agent][1]), ref, lookback).to_numpy() \
                    if len(D) else False
            ctx.count("tox_positive_rows", len(tx))
            ctx.count("tox_positive_inhospital_agent", int(tx["in_hosp"].sum()))
            ev = tx[~tx["in_hosp"]]
            if len(ev):
                parts.append(_mk(ev, "tox_screen_positive_nontherapeutic", source="measurement:tox"))
            inh = tx[tx["in_hosp"] & tx["hours_from_t0"].between(*e4b_cfg["tox_inhospital_window_hours"])]
            cov.loc[inh["case_id"].unique(), "e4b_tox_inhospital"] = True

    # 6. profound shock from the MAP series (direct MAP, else computed from SBP/DBP at the same minute)
    sh = raw["derived_items"]["profound_shock"]
    if len(ok):
        mp = ok[ok["item"] == "map"][["person_id", "t_event", "value"]].rename(columns={"value": "map"})
        sb = ok[ok["item"] == "sbp"][["person_id", "t_event", "value"]].rename(columns={"value": "sbp"})
        db = ok[ok["item"] == "dbp"][["person_id", "t_event", "value"]].rename(columns={"value": "dbp"})
        calc = sb.drop_duplicates(["person_id", "t_event"]).merge(db.drop_duplicates(["person_id", "t_event"]),
                                                                  on=["person_id", "t_event"])
        calc["map"] = calc["dbp"] + (calc["sbp"] - calc["dbp"]) / 3.0
        calc = calc[["person_id", "t_event", "map"]]
        if len(mp):
            calc = calc.merge(mp[["person_id", "t_event"]].drop_duplicates(), on=["person_id", "t_event"], how="left",
                              indicator=True)
            calc = calc[calc["_merge"] == "left_only"][["person_id", "t_event", "map"]]
        series = pd.concat([mp.drop_duplicates(["person_id", "t_event"]), calc], ignore_index=True)
        on = shock_onsets(series, sh["map_lt"], sh["min_duration_min"], sh["max_gap_min"])
        ctx.count("shock_runs", len(on))
        S = _join_cases(on, cases)
        if len(S):
            parts.append(_mk(S, "profound_shock", source="derived:map_series", basis="concept_or_name", approx=False,
                             text=""))

    # 7. baseline-relative flags
    def series_rows(item: str) -> pd.DataFrame:
        r = ok[ok["item"] == item] if len(ok) else ok
        if not len(r):
            return r
        r = r[r["t_event"] >= r["enc_start"]]
        return r.sort_values(["case_id", "t_event"], kind="stable").reset_index(drop=True)
    d = raw["derived_items"]
    cr = series_rows("creatinine")
    if len(cr):
        pm = _prior_extreme(cr, "min")
        m = pm.notna() & (pm > 0) & (cr["value"] >= d["creatinine_doubling"]["ratio"] * pm) & ~cr["case_id"].isin(esrd_cases)
        ctx.count("creatinine_esrd_excluded_cases", int(cr.loc[cr["case_id"].isin(esrd_cases), "case_id"].nunique()))
        p = _flag_events(cr, "creatinine_doubling", m)
        if len(p):
            parts.append(p)
    bi = series_rows("bilirubin")
    if len(bi):
        pm = _prior_extreme(bi, "min")
        c = d["bilirubin_doubling_ge2"]
        m = pm.notna() & (pm > 0) & (bi["value"] >= c["min_value_mgdl"]) & (bi["value"] >= c["ratio"] * pm)
        p = _flag_events(bi, "bilirubin_doubling_ge2", m)
        if len(p):
            parts.append(p)
    pl = series_rows("platelets")
    if len(pl):
        pm = _prior_extreme(pl, "max")
        c = d["platelets_drop"]
        m = pm.notna() & (pm >= c["min_baseline"]) & (pl["value"] < c["max_value"]) & \
            (pl["value"] <= (1 - c["min_fractional_decline"]) * pm)
        p = _flag_events(pl, "platelets_drop", m)
        if len(p):
            parts.append(p)

    # 8. QAD (needs the blood cultures and the antimicrobial days)
    cult = M[M["item"] == "blood_culture_drawn"] if len(M) else M
    if len(cult) and len(D):
        ab = D[D["group"] == "antimicrobial"]
        if len(ab):
            qp = raw["qad"]
            case_cults = {cid: [(int(dy), ts) for dy, ts in zip(_day(g["t_event"]), g["t_event"])]
                          for cid, g in cult.groupby("case_id")}
            case_abx: dict = {}
            for r in ab.itertuples(index=False):
                sd = (pd.Timestamp(r.t_start).normalize() - pd.Timestamp("1970-01-01")).days
                ed = sd
                if pd.notna(r.t_end) and r.t_end >= r.t_start:
                    ed = min((pd.Timestamp(r.t_end).normalize() - pd.Timestamp("1970-01-01")).days, sd + qp["max_span_days"])
                case_abx.setdefault(r.case_id, {}).setdefault(r.ingredient, set()).update(range(sd, ed + 1))
            hits = qad_cultures(case_cults, case_abx, qp)
            ctx.count("qad_cultures_qualifying", len(hits))
            if hits:
                h = pd.DataFrame(hits, columns=["case_id", "t_event"]).merge(
                    cases[["case_id", "person_id", "t0", "enc_start"]], on="case_id")
                h["hours_from_t0"] = (h["t_event"] - h["t0"]).dt.total_seconds() / 3600.0
                parts.append(_mk(h, "qad_ge4", source="derived:qad", basis="concept_or_name", approx=False, text=""))

    ev = _concat(parts, EVENT_FRAME_COLUMNS)
    ev = _gate_mass_effect(ev, anchor_cfg, cm)
    ev = _apply_banned(ev, ctx)
    # trim to the largest anchor window
    lo, hi = _window_bounds(anchor_cfg)
    ev["hours_from_t0"] = pd.to_numeric(ev["hours_from_t0"], errors="coerce")
    ev = ev[ev["hours_from_t0"].between(lo, hi)].copy()
    for c in EVENT_FRAME_COLUMNS:
        if c not in ev.columns:
            ev[c] = None
    ev = ev[EVENT_FRAME_COLUMNS].reset_index(drop=True)
    ev["value"] = pd.to_numeric(ev["value"], errors="coerce")
    gaps = [f"{it}: {v['reason']}" for it, v in raw["unavailable_items"].items()]
    return AnchorEvents(ev, cov, dict(ctx.diag), gaps)


def _window_bounds(anchor_cfg: dict) -> tuple[float, float]:
    ws = []

    def walk(n):
        for k in ("any_of", "all_of"):
            for c in n.get(k, []):
                walk(c)
        if "window_hours" in n:
            ws.append(n["window_hours"])
    for lab in anchor_cfg["labels"].values():
        walk(lab)
    return min(w[0] for w in ws), max(w[1] for w in ws)


def _item_windows(anchor_cfg: dict) -> dict[str, list[float]]:
    out: dict = {}

    def walk(n):
        for k in ("any_of", "all_of"):
            for c in n.get(k, []):
                walk(c)
        if "item" in n:
            out[n["item"]] = n["window_hours"]
            if n.get("rbc_correct"):                           # the same-tap RBC row rides on the WBC window
                out[n["rbc_correct"]["item"]] = n["window_hours"]
    for lab in anchor_cfg["labels"].values():
        walk(lab)
    return out


def _gate_mass_effect(ev: pd.DataFrame, anchor_cfg: dict, cm: ConceptMap) -> pd.DataFrame:
    """Items with ``requires_any_of`` (G93.5 / G93.6) count only when a listed structural item is present for the SAME case
    inside the gated item's window."""
    if ev.empty:
        return ev
    wins = _item_windows(anchor_cfg)
    for item, spec in cm.event_items.items():
        req = spec.get("requires_any_of")
        if not req or item not in wins:
            continue
        lo, hi = wins[item]
        support = set(ev.loc[ev["item"].isin(req) & ev["hours_from_t0"].between(lo, hi), "case_id"])
        drop = (ev["item"] == item) & ~ev["case_id"].isin(support)
        ev = ev[~drop]
    return ev


def _apply_banned(ev: pd.DataFrame, ctx: _Ctx) -> pd.DataFrame:
    """banned_evidence exclusions on the audit columns: nonspecific-encephalopathy ICD codes, EEG-mentioning text and
    anchorless encephalopathy wording never count as positive evidence."""
    if ev.empty:
        return ev
    keep = np.ones(len(ev), dtype=bool)
    cache: dict = {}
    for i, (src, txt, code) in enumerate(zip(ev["evidence_source"], ev["evidence_text"], ev["evidence_code"])):
        key = (src, txt if isinstance(txt, str) else "", code if isinstance(code, str) else "")
        v = cache.get(key)
        if v is None:
            allowed, cat = True, "allowed"
            if key[1] and mentions_eeg_content(key[1]):
                allowed, cat = False, "eeg_derived"
            elif src == EvidenceSource.ICD and key[2]:
                vd = classify_evidence(EvidenceSource.ICD, code=key[2])
                allowed, cat = vd.allowed, vd.category
            elif src == EvidenceSource.NOTE_SENTENCE:
                vd = classify_evidence(EvidenceSource.NOTE_SENTENCE, text=key[1], eeg_filtered=False, has_anchor=False)
                allowed, cat = vd.allowed, vd.category
            v = cache[key] = (allowed, cat)
        if not v[0]:
            keep[i] = False
            ctx.count(f"banned_dropped_{v[1]}")
    return ev[keep]


def _anchor_table(ev: pd.DataFrame, case_ids: list, anchor_cfg: dict) -> pd.DataFrame:
    """``anchors.silver_anchor_table`` on the events that can matter. ``anchors.py`` evaluates every rule per case with
    pandas (~0.05 s per case), so cases with no event inside any anchor window skip it: they cannot fire, all labels False."""
    wins = _item_windows(anchor_cfg)
    labs = list(anchor_cfg["labels"])
    if len(ev):
        lo = ev["item"].map(lambda i: wins[i][0] if i in wins else np.inf).astype(float)
        hi = ev["item"].map(lambda i: wins[i][1] if i in wins else -np.inf).astype(float)
        rel = ev[(ev["hours_from_t0"] >= lo) & (ev["hours_from_t0"] <= hi)]
    else:
        rel = ev
    have = set(rel["case_id"]) if len(rel) else set()
    active = [c for c in case_ids if c in have]
    t = silver_anchor_table(rel, case_ids=active, cfg=anchor_cfg) if active else \
        pd.DataFrame(columns=labs, dtype=bool)
    table = t.reindex(case_ids).fillna(False).astype(bool)
    table.index.name = "case_id"
    fired = dict(t.attrs.get("fired", {}))
    for c in case_ids:
        fired.setdefault(c, {lab: [] for lab in labs})
    table.attrs["fired"] = fired
    return table


# ================================================================================================ result
@dataclass
class SilverResult:
    labels: pd.DataFrame                     # case_id x (E1,E2,E4a,E5,E6,E7) nullable boolean + e4b_* covariates (RECORD-LEVEL)
    events: pd.DataFrame                     # anchor event table (RECORD-LEVEL; local only)
    cases: pd.DataFrame                      # case_id, person_id, t0, SiteID
    fired: dict                              # case_id -> label -> anchor ids (RECORD-LEVEL)
    label_status: dict                       # label -> full | partial | unavailable
    anchor_status: dict                      # anchor leaf id -> full | proxy | unavailable
    gaps: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


def extract_silver(source, cohort: pd.DataFrame, *, config: ExtractConfig | None = None,
                   extra_events: pd.DataFrame | None = None, on_error: Callable | None = None) -> SilverResult:
    """Silver labels for a cohort of ``(person_id, t0)`` from STRUCTURED OMOP tables.

    ``source``: a ``data_io.LocalStore`` / S3 client (streamed, column-pruned) or a dict of in-memory tables keyed like
    ``schema`` (``omop_measurement`` ...). ``reports_findings`` and every other FORBIDDEN table are never read.
    ``extra_events`` (anchor-event frame, e.g. future imaging flags) is merged before the anchors run.
    """
    config = config or ExtractConfig()
    cm = ConceptMap(load_concept_map(config.concept_config_path))
    anchor_cfg = load_anchor_config(config.anchor_config_path)
    problems = validate_concept_map(cm, anchor_cfg)
    if problems:
        raise ValueError("anchor_concepts.yaml is inconsistent with silver_anchors.yaml: " + "; ".join(problems))
    cases = prepare_cohort(cohort)
    ctx = _Ctx(cm, ConceptIndex(cm), cases, config)
    cl = classify_tables(source, ctx) if isinstance(source, Mapping) else classify_store(source, ctx, on_error)
    ae = build_anchor_events(cl, cases, ctx, anchor_cfg)
    ev = ae.events
    if extra_events is not None and len(extra_events):
        x = extra_events.copy()
        if "acute" in x.columns and "acute" not in ev.columns:
            ev = ev.assign(acute=True)               # our rows are acute-filtered by code (chronic codes excluded)
        ev = pd.concat([ev, x], ignore_index=True)
    ev_for_anchors = ev[[c for c in ev.columns if c not in INTERNAL_EVENT_COLUMNS]]
    table = _anchor_table(ev_for_anchors, list(cases["case_id"]), anchor_cfg)
    lstat = label_status(cm, anchor_cfg)
    labels = table[[l for l in SILVER_CIRCULARITY_LABELS if l in table.columns]].astype("boolean")
    for lab, st in lstat.items():
        if st == "unavailable" and lab in labels:
            labels[lab] = pd.array([pd.NA] * len(labels), dtype="boolean")
    labels = labels.join(ae.covariates.astype(bool))
    labels.index.name = "case_id"
    return SilverResult(labels=labels, events=ev, cases=cases, fired=table.attrs.get("fired", {}), label_status=lstat,
                        anchor_status=anchor_leaf_status(cm, anchor_cfg), gaps=ae.gaps, diagnostics=ae.diagnostics)


# ============================================================================================== reporting
def _suppress_group(counts: Mapping[str, int], total: int) -> dict:
    """Suppress each count < 11; if exactly one cell is suppressed while the total is shown, also suppress the smallest
    remaining cell (complementary suppression, so the hidden cell cannot be recovered by subtraction)."""
    out = {k: suppress_count(v) for k, v in counts.items()}
    hidden = [k for k, v in out.items() if v == SUPPRESSED]
    if len(hidden) == 1 and total >= SUPPRESS_BELOW:
        shown = sorted((v, k) for k, v in counts.items() if out[k] != SUPPRESSED)
        if shown:
            out[shown[0][1]] = SUPPRESSED
    return out


def silver_report(res: SilverResult, *, site_col: str = "SiteID") -> dict:
    """AGGREGATE-ONLY report: per-site and overall silver prevalence per label, per-anchor firing counts (n < 11 shown as
    ``"<11"``, complementary suppression across sites), computability status and recorded gaps. No IDs, no timestamps."""
    labs = [c for c in SILVER_CIRCULARITY_LABELS if c in res.labels.columns]
    cases = res.cases.set_index("case_id")
    site = cases[site_col].reindex(res.labels.index)
    sites = sorted(site.unique())
    n_by_site = {s: int((site == s).sum()) for s in sites}
    rep: dict = {"n_cases": suppress_count(len(res.labels)), "label_status": dict(res.label_status),
                 "n_cases_per_site": _suppress_group(n_by_site, len(res.labels)), "per_label": {}, "anchor_firing": {},
                 "anchor_status": dict(res.anchor_status), "gaps": list(res.gaps), "diagnostics": {}}
    for lab in labs:
        col = res.labels[lab]
        if res.label_status.get(lab) == "unavailable":
            rep["per_label"][lab] = "unavailable from structured data"
            continue
        pos_by_site = {s: int(col[site == s].fillna(False).sum()) for s in sites}
        total_pos = int(col.fillna(False).sum())
        rep["per_label"][lab] = {
            "n_positive": suppress_count(total_pos), "prevalence": suppress_proportion(total_pos, len(col)),
            "per_site": {s: {"n_positive": None, "prevalence": suppress_proportion(pos_by_site[s], n_by_site[s])}
                         for s in sites}}
        sup = _suppress_group(pos_by_site, total_pos)
        for s in sites:
            rep["per_label"][lab]["per_site"][s]["n_positive"] = sup[s]
    # per-anchor firing counts
    fire: Counter = Counter()
    fire_site: dict = {}
    for cid, per_lab in res.fired.items():
        for lab, ids in per_lab.items():
            for a in ids:
                fire[(lab, a)] += 1
                fire_site.setdefault((lab, a), Counter())[cases.at[cid, site_col]] += 1
    for (lab, a), n in sorted(fire.items()):
        rep["anchor_firing"].setdefault(lab, {})[a] = {
            "n_cases": suppress_count(n),
            "per_site": _suppress_group({s: fire_site[(lab, a)].get(s, 0) for s in sites}, n)}
    for lab in labs:                                  # anchors that never fired are reported, as "<11"
        for aid in sorted(k for k in res.anchor_status if k.startswith(lab + "_")):
            rep["anchor_firing"].setdefault(lab, {}).setdefault(aid, {"n_cases": SUPPRESSED, "per_site": SUPPRESSED})
    rep["diagnostics"] = {k: suppress_count(v) for k, v in sorted(res.diagnostics.items())}
    assert_aggregate_only(rep, known_ids=set(res.labels.index.astype(str)))
    return rep


# ===================================================================== EEG-report comparator (NOT silver)
def eeg_impression_comparator(reports_findings: pd.DataFrame, cohort: pd.DataFrame, *, mapping: Mapping | None = None,
                              tolerance_hours: float = 6.0) -> pd.DataFrame:
    """EEG-report-impression reference for the CIRCULARITY AUDIT ONLY: long frame ``case_id, label, eeg_impr`` (1 when a
    mapped finding flag is asserted on the matching report, 0 when a report exists without one, NaN when the case has no
    matching report). Labels without a mapped flag get no rows. NEVER silver evidence (D-006): it is not an argument of
    ``extract_silver`` and nothing on the label path calls it.

    Matching: ``SessionID`` when both frames carry it, else any report of the same patient whose ``StartTime(EEG)`` is
    within ``tolerance_hours`` of t0. ``reports_findings`` is in canonical names (``data_io.read_site_table``)."""
    mapping = dict(mapping) if mapping is not None else \
        {k: v for k, v in load_concept_map_comparator().items() if k != "status"}
    cases = prepare_cohort(cohort)
    rf = reports_findings.copy()
    rf["person_id"] = pd.to_numeric(rf["BDSPPatientID"], errors="coerce") if "BDSPPatientID" in rf else np.nan
    rf = rf[rf["person_id"].notna()].copy()
    rf["person_id"] = rf["person_id"].astype("int64")
    labs = {lab: [f for f in fl if f in rf.columns] for lab, fl in mapping.items()}
    labs = {lab: fl for lab, fl in labs.items() if fl}
    for lab, fl in labs.items():
        rf[f"_lab_{lab}"] = np.column_stack([data_io.finding_present(rf[f]).to_numpy() for f in fl]).any(axis=1)
    rf["_has"] = True
    use = ["person_id", "_has"] + [f"_lab_{lab}" for lab in labs]
    if "SessionID" in cohort.columns and "SessionID" in rf.columns:
        key = cohort[["person_id", "SessionID"]].copy()
        key["person_id"] = pd.to_numeric(key["person_id"]).astype("int64")
        c = cases.merge(key.drop_duplicates("person_id"), on="person_id", how="left")
        c["_s"] = c["SessionID"].astype(str)
        r = rf.assign(_s=rf["SessionID"].astype(str))[use + ["_s"]]
        j = c.merge(r, on=["person_id", "_s"], how="left")
    else:
        if schema.START_EEG not in rf.columns:
            raise ValueError("reports_findings needs SessionID (with a cohort SessionID) or StartTime(EEG)")
        rf["_start"] = schema.parse_datetimes(rf[schema.START_EEG])
        j = cases.merge(rf[use + ["_start"]], on="person_id", how="left")
        j = j[(j["_start"] - j["t0"]).abs() <= pd.Timedelta(hours=tolerance_hours)].merge(
            cases[["case_id"]], on="case_id", how="right")
    j["_has"] = j["_has"].fillna(False).astype(bool)
    rows = []
    for lab in labs:
        g = j.groupby("case_id")
        has = g["_has"].any().reindex(cases["case_id"]).fillna(False).astype(bool)
        pos = g[f"_lab_{lab}"].apply(lambda s: bool(s.fillna(False).astype(bool).any())).reindex(cases["case_id"])
        val = np.where(has.to_numpy(), pos.fillna(False).astype(float).to_numpy(), np.nan)
        rows.append(pd.DataFrame({"case_id": cases["case_id"].to_numpy(), "label": lab, "eeg_impr": val}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame({"case_id": [], "label": [], "eeg_impr": []})


def load_concept_map_comparator(path: str | Path | None = None) -> dict:
    from .concepts import load_concept_map
    return dict(load_concept_map(path).get("eeg_impression_comparator", {}))


def load_reports_findings(store, sites: Iterable[str] | None = None) -> pd.DataFrame:
    """Read ``reports_findings`` through ``data_io.read_site_table`` (sites without the file are skipped). For the
    circularity-audit comparator ONLY."""
    sites = list(sites) if sites is not None else data_io.discover_sites(store)
    frames = []
    for s in sites:
        try:
            frames.append(data_io.read_site_table("reports_findings", s, s3=store))
        except FileNotFoundError:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ====================================================================================================== CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Structured silver-label extraction (aggregate-only report).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="local HEEDB-layout directory (synthetic data in agent sessions)")
    src.add_argument("--s3", action="store_true", help="BDSP access point (human-run only)")
    ap.add_argument("--cohort", required=True, help="CSV/parquet with person_id, t0 (and SiteID, optional case_id)")
    ap.add_argument("--out", default="out/silver", help="directory for the aggregate report JSON")
    ap.add_argument("--labels-out", help="record-level silver labels CSV; the path must contain a local_only/ directory")
    ap.add_argument("--no-name-fallback", action="store_true")
    ap.add_argument("--profile")
    ck.add_arguments(ap)
    args = ap.parse_args(argv)
    if args.data:
        from ..agent_safety import assert_not_restricted_in_agent
        assert_not_restricted_in_agent(args.data)
    if args.labels_out and "local_only" not in Path(args.labels_out).parts:
        ap.error("--labels-out must be under a local_only/ directory (record-level data)")
    store = data_io.open_store(args.data, profile=args.profile)       # S3 client refuses inside an agent session
    p = Path(args.cohort)
    cohort = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    # Restart safety: the classified concept map (once) and every (table, row group) classification result are stored under
    # out/local_only/checkpoints/silver_labels/<key>/ as they finish; a relaunch with the same arguments skips them.
    cp = ck.open_step("silver_labels", args, inputs=[p], store=store, no_resume=args.no_resume)
    with ck.use(cp):
        res = extract_silver(store, cohort, config=ExtractConfig(allow_name_fallback=not args.no_name_fallback))
    rep = silver_report(res)
    safe_write_json(Path(args.out) / "silver_report.json", rep)
    if args.labels_out:
        write_local_only(args.labels_out, res.labels.to_csv())
    print(f"wrote aggregate report to {Path(args.out) / 'silver_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
