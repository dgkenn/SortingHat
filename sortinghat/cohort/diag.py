"""Aggregate-only diagnostics for the cohort's visit matching (and the duration unit).

Run through ``scripts/diag_cohort.py`` (human-run on real data). Everything returned is an aggregate: counts and
proportions go through ``safe_output`` (n < 11 -> ``"<11"``), quantiles need n >= 11 and never include min/max, and
text values (visit source strings, column names) are listed only with counts >= 11 and with digit runs masked.
Concept ids are vocabulary codes, not patient data, and are listed with their suppressed counts. Lists of
``{"value", "n"}`` entries are used instead of dicts keyed by text, so no free-text ever becomes a key.

What it answers, per site, for ADULT sessions with a start time:

* does the session's person_id have ANY ``visit_occurrence`` row (the join), and which id forms join best
  (BDSPPatientID as a number, the BidsFolder tail, text with leading zeros kept);
* how visit starts/ends sit relative to the EEG start (signed hours, nearest visit by absolute distance), how many
  visit ends are null, how many are date-only (midnight);
* the share of sessions covered by a closed visit interval, and when it is widened by 24 h, when a null end is
  treated as start + 30 days, and both; and the share matched by the cohort's current rule (``rules.match_visits``);
* ``visit_concept_id`` and ``visit_source_value`` value counts;
* the share of sessions whose EEG date is outside the patient's overall visit date range;
* column NAMES of ``PatientMergeHistory/`` files;
* the metadata-duration / clock-duration ratio by ServiceName (LTM vs Routine vs other).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .. import data_io, schema
from ..audit import field_audit as fa
from ..safe_output import SUPPRESS_BELOW, SUPPRESSED, assert_aggregate_only, safe_quantiles, suppress_count
from . import rules
from .sources import MERGE_PREFIX, StoreSources, _VISIT_COLS

QS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
_DIGITS = re.compile(r"\d{3,}")
CHUNK = 20000


def _safe_text(x, width: int = 60) -> str:
    """A text value made safe to list: digit runs masked, id-like strings replaced, truncated."""
    t = str(x).strip()
    if data_io.looks_id_like(t):
        return "<id-like>"
    return _DIGITS.sub("#", t)[:width]


def _share(mask: pd.Series) -> dict:
    """{n, of, share}: each exact when >= 11, else "<11" (the share needs n >= 11 and of >= 11). Diagnostic output:
    a small complement (of - n) is inferable from two exact counts, so do not share this file outside the project."""
    n, den = int(mask.sum()), int(len(mask))
    share = round(n / den, 4) if n >= SUPPRESS_BELOW and den >= SUPPRESS_BELOW else SUPPRESSED
    return {"n": suppress_count(n), "of": suppress_count(den), "share": share}


def _counts(values: pd.Series, top: int | None = None, numeric: bool = False) -> dict:
    """Value counts as a list of {value, n}; values with n < 11 are pooled in ``other_values_n``."""
    vc = values.dropna().astype(str).value_counts()
    shown = vc[vc >= SUPPRESS_BELOW]
    if top is not None:
        shown = shown.iloc[:top]
    other = int(vc.sum() - shown.sum())
    return {"n_distinct": suppress_count(len(vc)),
            "values": [{"value": str(v) if numeric else _safe_text(v), "n": int(n)} for v, n in shown.items()],
            "other_values_n": suppress_count(other), "null_n": suppress_count(int(values.isna().sum()))}


def _hours_q(x: pd.Series) -> dict:
    return safe_quantiles(x.dropna().to_numpy(float), qs=QS)


# ------------------------------------------------------------------------------------------ visit geometry
def _visit_frame(visits: pd.DataFrame) -> pd.DataFrame:
    """Visits with raw (unrepaired) start/end as datetime64[s]: ``_s`` (datetime, else the date), ``_e`` (NaT if
    none), and date-only flags. Only the columns the diagnostic needs are kept."""
    pid = pd.to_numeric(visits["person_id"], errors="coerce")
    s_ = rules._dt(visits, "visit_start_datetime")
    s_ = s_.where(s_.notna(), rules._dt(visits, "visit_start_date"))
    e_ = rules._dt(visits, "visit_end_datetime")
    e_ = e_.where(e_.notna(), rules._dt(visits, "visit_end_date"))
    ok = (pid.notna() & s_.notna()).to_numpy(bool)
    return pd.DataFrame({"person_id": pid[ok].astype("int64").to_numpy(),
                         "_s": s_[ok].astype("datetime64[s]").to_numpy(), "_e": e_[ok].astype("datetime64[s]").to_numpy(),
                         "_s_date_only": (s_[ok] == s_[ok].dt.normalize()).to_numpy(),
                         "_e_date_only": (e_[ok].notna() & (e_[ok] == e_[ok].dt.normalize())).to_numpy()})


class _Counts:
    """Capped value counter: at most ``cap`` distinct values are tracked; the rest are pooled."""

    def __init__(self, cap: int = 20000):
        self.c: dict[str, int] = {}
        self.cap, self.pooled, self.nulls = cap, 0, 0

    def add(self, s: pd.Series) -> None:
        self.nulls += int(s.isna().sum())
        for k, n in s.dropna().astype(str).value_counts().items():
            if k in self.c or len(self.c) < self.cap:
                self.c[k] = self.c.get(k, 0) + int(n)
            else:
                self.pooled += int(n)

    def report(self, top: int | None = None, numeric: bool = False) -> dict:
        vc = pd.Series(self.c, dtype="int64").sort_values(ascending=False)
        shown = vc[vc >= SUPPRESS_BELOW]
        if top is not None:
            shown = shown.iloc[:top]
        other = int(vc.sum() - shown.sum()) + self.pooled
        return {"n_distinct": suppress_count(len(vc)),
                "values": [{"value": str(v) if numeric else _safe_text(v), "n": int(n)} for v, n in shown.items()],
                "other_values_n": suppress_count(other), "null_n": suppress_count(self.nulls)}


class _Acc:
    """Per-session and per-site accumulators, updated one visit chunk at a time (nothing else is kept)."""

    def __init__(self, a: pd.DataFrame, site_list: list[str]):
        self.pid = a["person_id"].to_numpy("int64")
        self.t0 = a["t0"].astype("datetime64[s]").to_numpy()
        self.site_ix = pd.Categorical(a["SiteID"], categories=site_list).codes.astype("int64")
        self.nsite = len(site_list)
        n = len(a)
        self.best_key = np.full(n, np.iinfo("int64").max, "int64")
        self.d_s_abs, self.d_s = np.full(n, np.inf), np.full(n, np.nan)
        self.d_e_abs, self.d_e = np.full(n, np.inf), np.full(n, np.nan)
        self.cover = {k: np.zeros(n, bool) for k in ("closed", "widened_24h", "null_end_30d",
                                                    "widened_and_null_end_30d")}
        self.latest = np.full(n, np.iinfo("int64").min, "int64")
        self.latest_end_known = np.zeros(n, bool)
        self.upids = np.unique(self.pid)
        self.p_site = np.zeros(len(self.upids), "int64")
        self.p_site[np.searchsorted(self.upids, self.pid)] = self.site_ix
        self.pmin = np.full(len(self.upids), np.iinfo("int64").max, "int64")
        self.pmax = np.full(len(self.upids), np.iinfo("int64").min, "int64")
        self.seen: list[np.ndarray] = []
        z = lambda: np.zeros(self.nsite, "int64")          # noqa: E731
        self.rows, self.null_end, self.mid_start, self.known_end, self.mid_end = z(), z(), z(), z(), z()
        self.concept, self.source = _Counts(), _Counts()
        self.site_concept = [_Counts() for _ in range(self.nsite)]
        self.n_rows = self.n_start_null = 0
        self.pid_dtype = None

    def add(self, raw: pd.DataFrame) -> None:
        self.n_rows += len(raw)
        self.pid_dtype = self.pid_dtype or str(raw["person_id"].dtype)
        self.n_start_null += int(rules._dt(raw, "visit_start_datetime").isna().sum()) if "visit_start_datetime" in raw else 0
        v = _visit_frame(raw)
        if not len(v):
            return
        d24, d30 = pd.Timedelta(hours=24), pd.Timedelta(days=30)
        self.seen.append(np.unique(v["person_id"].to_numpy()))
        # ---- per-site visit-row statistics, concept / source counts, person date ranges (adult-session persons only)
        pi = np.searchsorted(self.upids, v["person_id"].to_numpy())
        pi_c = np.minimum(pi, len(self.upids) - 1)
        valid = self.upids[pi_c] == v["person_id"].to_numpy()
        if valid.any():
            si = self.p_site[pi_c[valid]]
            vv = v[valid]
            e_known = vv["_e"].notna().to_numpy()
            bc = lambda m: np.bincount(si[m], minlength=self.nsite)       # noqa: E731
            self.rows += bc(np.ones(len(vv), bool))
            self.null_end += bc(~e_known)
            self.mid_start += bc(vv["_s_date_only"].to_numpy())
            self.known_end += bc(e_known)
            self.mid_end += bc(vv["_e_date_only"].to_numpy())
            self.concept.add(raw["visit_concept_id"] if "visit_concept_id" in raw else pd.Series(dtype=object))
            self.source.add(raw["visit_source_value"] if "visit_source_value" in raw else pd.Series(dtype=object))
            if "visit_concept_id" in raw:
                rp = pd.to_numeric(raw["person_id"], errors="coerce")
                ri = np.minimum(np.searchsorted(self.upids, rp.fillna(-1).astype("int64").to_numpy()), len(self.upids) - 1)
                rv = self.upids[ri] == rp.fillna(-1).astype("int64").to_numpy()
                for k in range(self.nsite):
                    m = rv & (self.p_site[ri] == k)
                    if m.any():
                        self.site_concept[k].add(raw.loc[m, "visit_concept_id"])
            hi = vv["_e"].where(vv["_e"].notna(), vv["_s"])
            g = pd.DataFrame({"p": pi[valid], "lo": vv["_s"].astype("int64").to_numpy(),
                              "hi": hi.astype("int64").to_numpy()}).groupby("p").agg(lo=("lo", "min"), hi=("hi", "max"))
            self.pmin[g.index] = np.minimum(self.pmin[g.index], g["lo"].to_numpy())
            self.pmax[g.index] = np.maximum(self.pmax[g.index], g["hi"].to_numpy())
        # ---- per-session geometry against this chunk's visits
        sidx = np.nonzero(np.isin(self.pid, v["person_id"].unique()))[0]
        if not len(sidx):
            return
        es = pd.DataFrame({"person_id": self.pid[sidx], "t0": self.t0[sidx], "_i": sidx})
        j = es.merge(v[["person_id", "_s", "_e"]], on="person_id")
        t, s_, en = j["t0"], j["_s"], j["_e"]
        en30 = en.where(en.notna(), s_ + d30)
        variants = {"closed": en.notna() & (s_ <= t) & (t <= en),
                    "widened_24h": en.notna() & (s_ - d24 <= t) & (t <= en + d24),
                    "null_end_30d": (s_ <= t) & (t <= en30),
                    "widened_and_null_end_30d": (s_ - d24 <= t) & (t <= en30 + d24)}
        for name, m in variants.items():
            self.cover[name][j.loc[m.to_numpy(), "_i"].unique()] = True
        dh = (t - s_).dt.total_seconds() / 3600.0
        self._nearest(j["_i"], dh, self.d_s_abs, self.d_s)
        k = en.notna().to_numpy()
        self._nearest(j.loc[k, "_i"], ((t - en).dt.total_seconds() / 3600.0)[k], self.d_e_abs, self.d_e)
        bk = (s_ <= t).to_numpy()
        if bk.any():
            jb = j[bk].sort_values(["_i", "_s"]).drop_duplicates("_i", keep="last")
            upd = jb["_s"].astype("int64").to_numpy() > self.latest[jb["_i"].to_numpy()]
            ii = jb["_i"].to_numpy()[upd]
            self.latest[ii] = jb["_s"].astype("int64").to_numpy()[upd]
            self.latest_end_known[ii] = jb["_e"].notna().to_numpy()[upd]
        # ---- the cohort's current matching rule (rules.match_visits semantics, default knobs)
        cv = rules.with_horizon(rules.compact_visits(raw, dates_only=True))
        j2 = es.merge(cv[["person_id", "_start", "_endf", "_cls", "_inpt"]], on="person_id")
        covered, key, _ = rules.visit_keys(j2, 24.0, True)
        if covered.any():
            kk = pd.DataFrame({"_i": j2.loc[covered, "_i"].to_numpy(), "k": key[covered].to_numpy()}).groupby("_i")["k"].min()
            self.best_key[kk.index] = np.minimum(self.best_key[kk.index], kk.to_numpy())

    @staticmethod
    def _nearest(i: pd.Series, dh: pd.Series, best_abs: np.ndarray, best: np.ndarray) -> None:
        d = pd.DataFrame({"i": i.to_numpy(), "dh": dh.to_numpy(), "a": np.abs(dh.to_numpy())})
        d = d.sort_values(["i", "a"]).drop_duplicates("i")
        ii = d["i"].to_numpy()
        upd = d["a"].to_numpy() < best_abs[ii]
        best_abs[ii[upd]] = d["a"].to_numpy()[upd]
        best[ii[upd]] = d["dh"].to_numpy()[upd]


# ------------------------------------------------------------------------------------------ id-join checks
def _id_forms(meta: pd.DataFrame, site: str) -> pd.DataFrame:
    """Per session: the candidate OMOP person_id forms from the raw metadata (names are canonical)."""
    bd = meta["BDSPPatientID"].astype("string") if "BDSPPatientID" in meta else pd.Series(pd.NA, index=meta.index,
                                                                                         dtype="string")
    bf = meta["BidsFolder"].astype("string") if "BidsFolder" in meta else pd.Series(pd.NA, index=meta.index,
                                                                                    dtype="string")
    tail = bf.str.replace(rf"^sub-{site}", "", regex=True)
    return pd.DataFrame({
        "bd_blank": bd.isna() | (bd.str.strip() == ""),
        "bd_num": pd.to_numeric(bd, errors="coerce"),
        "bd_nonnumeric": bd.notna() & (bd.str.strip() != "") & pd.to_numeric(bd, errors="coerce").isna(),
        "bd_leading_zero": bd.notna() & bd.str.fullmatch(r"0\d+").fillna(False),
        "bids_num": pd.to_numeric(tail, errors="coerce"), "SessionID": meta["SessionID"].astype("string")})


def _merge_history_names(s3) -> dict:
    keys = data_io.list_keys(s3, MERGE_PREFIX)
    groups: dict[tuple, int] = {}
    for k in keys:
        low = k.lower()
        try:
            if low.endswith(".parquet"):
                cols = data_io.parquet_column_names(s3, k)
            elif low.endswith((".csv", ".csv.gz", ".tsv", ".txt")):
                cols = data_io.csv_header(s3, k)
            else:
                cols = []
        except Exception:                                    # noqa: BLE001 - reported as unreadable, no detail
            cols = ["<unreadable>"]
        ext = "parquet" if low.endswith(".parquet") else ("csv" if ".csv" in low else "other")
        groups[(ext, tuple(_safe_text(c, 40) for c in cols))] = groups.get((ext, tuple(_safe_text(c, 40) for c in cols)), 0) + 1
    return {"n_files": suppress_count(len(keys)),
            "layouts": [{"format": ext, "column_names": list(cols), "n_files": suppress_count(n)}
                        for (ext, cols), n in groups.items()]}


def _duration_by_service(S: pd.DataFrame) -> dict:
    out = {}
    clock = (S["t_end"] - S["t0"]).dt.total_seconds()
    meta = pd.to_numeric(S["duration_raw_s"], errors="coerce")
    svc = S["ServiceName"].astype("string").str.strip().str.upper()
    cls = svc.map(lambda x: "missing" if pd.isna(x) or x == "" else ("LTM" if x == "LTM" else (
        "Routine" if x == "ROUTINE" else "other")))
    ok = (clock > 0) & (meta > 0)
    for site in sorted(S["SiteID"].astype(str).unique()):
        blk = {}
        for c in ("LTM", "Routine", "other", "missing"):
            m = ok & (S["SiteID"].astype(str) == site) & (cls == c)
            n = int(m.sum())
            if n == 0:
                continue
            ratio = meta[m] / clock[m]
            blk[c] = {"n_sessions": suppress_count(n), "ratio_meta_over_clock": safe_quantiles(ratio.to_numpy(float), qs=QS),
                      "share_ratio_within_10pct": _share((ratio > 0.9) & (ratio < 1.1)),
                      "clock_hours": safe_quantiles((clock[m] / 3600.0).to_numpy(float), qs=QS),
                      "metadata_value_raw_units": safe_quantiles(meta[m].to_numpy(float), qs=QS)}
        if blk:
            out[site] = blk
    return out


# ----------------------------------------------------------------------------------------------- main entry
def run_diag(store, sites: list[str] | None = None, min_rows: int = 500_000) -> tuple[dict, set[str]]:
    """Return ``(aggregate_report, known_ids)``. ``known_ids`` are the record identifiers seen (to prove none leaks).

    Memory: the visit table is read one coalesced row-group chunk at a time and folded into per-session /
    per-site accumulators (``_Acc``); no visit frame is ever kept."""
    src = StoreSources(store, sites)
    S = src.sessions().reset_index(drop=True)
    site_list = sorted(S["SiteID"].astype(str).unique())
    adult = S[S["person_id"].notna() & S["t0"].notna() & (S["age_years"] >= 18)].copy()
    adult["person_id"] = adult["person_id"].astype("int64")
    adult["SiteID"] = adult["SiteID"].astype(str)
    adult = adult.reset_index(drop=True)

    forms = []
    for site in site_list:
        try:
            meta = schema.coerce_types("eeg_metadata", data_io.read_site_table("eeg_metadata", site, s3=store))
        except FileNotFoundError:
            continue
        f = _id_forms(meta, site)
        f["SiteID"] = site
        forms.append(f)
        del meta
    forms = pd.concat(forms, ignore_index=True) if forms else pd.DataFrame()
    cand = set(adult["person_id"].unique())
    if len(forms):
        cand |= set(forms["bd_num"].dropna().astype("int64")) | set(forms["bids_num"].dropna().astype("int64"))
    known = {str(i) for i in cand} | set(S["SessionID"].dropna().astype(str))

    acc = _Acc(adult, site_list)
    columns_seen: set[str] = set()
    for raw in src.iter_rows("omop_visit_occurrence", _VISIT_COLS, sorted(cand), min_rows=min_rows):
        columns_seen |= set(raw.columns)
        acc.add(raw)
        del raw
    vp = np.unique(np.concatenate(acc.seen)) if acc.seen else np.zeros(0, "int64")

    report: dict = {"note": "aggregates only; counts < 11 shown as \"<11\"; quantiles need n >= 11; hours are signed "
                            "(EEG start minus visit time: positive = EEG after the visit time)",
                    "sites": {}, "visit_concept_id": {}, "visit_source_value": {}}
    report["visit_table"] = {
        "n_visit_rows_for_adult_candidates": suppress_count(acc.n_rows),
        "pid_dtype_after_read": acc.pid_dtype or "n/a", "columns_read": sorted(columns_seen),
        "share_start_datetime_null": _count_share(acc.n_start_null, acc.n_rows)}
    parts = data_io.omop_parts(store, "visit_occurrence")
    if parts:
        try:
            pf = data_io.open_parquet(store, data_io.access_point(), parts[0])
            report["visit_table"]["pid_arrow_type"] = str(pf.schema_arrow.field("person_id").type)
            report["visit_table"]["footer_column_names"] = list(pf.schema_arrow.names)
        except Exception:                                                  # noqa: BLE001
            report["visit_table"]["pid_arrow_type"] = "unreadable"

    has_all = np.isin(acc.pid, vp)
    matched = acc.best_key < np.iinfo("int64").max
    hi_part = acc.best_key // (1 << 35)
    rank = hi_part % 8                                            # care-setting class (no-op: all concept ids are 0)
    inpt = ((hi_part // 8) // 2) % 2 == 0                         # key flag: 0 = inpatient-length visit
    cls_name = np.where(matched, np.where(inpt, "inpatient_length_visit", "short_visit"), "no_visit")
    pidx = np.searchsorted(acc.upids, acc.pid)
    for k, site in enumerate(site_list):
        m = acc.site_ix == k
        if not m.any():
            continue
        has = has_all & m
        blk = {"n_adult_sessions_with_start": suppress_count(int(m.sum())),
               "share_pid_has_any_visit_row": _share(pd.Series(has_all[m]))}
        f = forms[forms["SiteID"] == site] if len(forms) else forms
        if len(f):
            blk["id_join"] = {
                "n_metadata_sessions": suppress_count(len(f)),
                "share_bdsp_id_blank": _share(f["bd_blank"]),
                "share_bdsp_id_non_numeric": _share(f["bd_nonnumeric"]),
                "share_bdsp_id_leading_zero": _share(f["bd_leading_zero"]),
                "share_bdsp_and_folder_ids_differ": _share(
                    f["bd_num"].notna() & f["bids_num"].notna() & (f["bd_num"] != f["bids_num"])),
                "share_with_visit_via_bdsp_id_number": _share(f["bd_num"].isin(vp)[f["bd_num"].notna()]),
                "share_with_visit_via_folder_tail": _share(f["bids_num"].isin(vp)[f["bids_num"].notna()])}
        blk["share_without_visit_row_by_pid"] = _share(pd.Series(~has_all[m]))
        if has.any():
            blk["visit_end_null_share_of_visit_rows"] = _count_share(acc.null_end[k], acc.rows[k])
            blk["visit_start_midnight_share"] = _count_share(acc.mid_start[k], acc.rows[k])
            blk["visit_end_midnight_share_of_known_ends"] = _count_share(acc.mid_end[k], acc.known_end[k])
            lat = has & (acc.latest > np.iinfo("int64").min)
            blk["share_latest_start_visit_has_null_end"] = _share(pd.Series(~acc.latest_end_known[lat]))
            blk["hours_eeg_minus_nearest_visit_start"] = _hours_q(pd.Series(acc.d_s[has]))
            blk["hours_eeg_minus_nearest_visit_end"] = _hours_q(pd.Series(acc.d_e[has]))
            t0d = acc.t0[has].astype("datetime64[D]")
            lo = acc.pmin[pidx[has]].astype("datetime64[s]").astype("datetime64[D]")
            hi = acc.pmax[pidx[has]].astype("datetime64[s]").astype("datetime64[D]")
            blk["share_eeg_date_outside_visit_date_range"] = _share(pd.Series((t0d < lo) | (t0d > hi)))
        blk["share_covered_of_all_adult_eegs"] = {n: _share(pd.Series(c[m])) for n, c in acc.cover.items()}
        if has.any():
            blk["share_covered_of_eegs_with_a_visit_row"] = {n: _share(pd.Series(c[has])) for n, c in acc.cover.items()}
        blk["share_matched_by_current_rule"] = _share(pd.Series(matched[m]))
        blk["share_matched_multi_day_visit_by_current_rule"] = _share(pd.Series(matched[m] & inpt[m]))
        blk["current_rule_matched_visit_kind_counts"] = _counts(pd.Series(cls_name[m]))
        report["sites"][site] = blk

    report["visit_concept_id"] = acc.concept.report(numeric=True)
    report["visit_source_value"] = acc.source.report(top=20)
    report["visit_concept_id_by_site"] = {site: acc.site_concept[k].report(numeric=True)
                                          for k, site in enumerate(site_list) if acc.site_concept[k].c}
    report["merge_history_table"] = _merge_history_names(store)
    report["duration_by_service"] = _duration_by_service(S)
    assert_aggregate_only(report, known)
    return report, known


def _count_share(n: int, den: int) -> dict:
    return {"n": suppress_count(int(n)), "of": suppress_count(int(den)),
            "share": round(n / den, 4) if n >= SUPPRESS_BELOW and den >= SUPPRESS_BELOW else SUPPRESSED}
