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
from .sources import MERGE_PREFIX, StoreSources

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
def _closed_variants(e: pd.DataFrame, v: pd.DataFrame, chunk: int = CHUNK) -> pd.DataFrame:
    """Per session (index of ``e``): covered by a visit interval under four variants (see module doc)."""
    cols = {k: np.zeros(len(e), bool) for k in ("closed", "widened_24h", "null_end_30d", "widened_and_null_end_30d")}
    vv = v[["person_id", "_s", "_e"]]
    d24, d30 = pd.Timedelta(hours=24), pd.Timedelta(days=30)
    for a in range(0, len(e), chunk):
        ec = e.iloc[a:a + chunk]
        j = ec[["person_id", "t0"]].reset_index().merge(vv, on="person_id")
        if not len(j):
            continue
        t, s, en = j["t0"], j["_s"], j["_e"]
        en30 = en.where(en.notna(), s + d30)
        variants = {"closed": en.notna() & (s <= t) & (t <= en),
                    "widened_24h": en.notna() & (s - d24 <= t) & (t <= en + d24),
                    "null_end_30d": (s <= t) & (t <= en30),
                    "widened_and_null_end_30d": (s - d24 <= t) & (t <= en30 + d24)}
        key = j.columns[0]
        for name, m in variants.items():
            hit = e.index.get_indexer(j.loc[m.to_numpy(), key].unique())
            cols[name][hit[hit >= 0]] = True
    return pd.DataFrame(cols, index=e.index)


def _visit_frame(visits: pd.DataFrame) -> pd.DataFrame:
    """Visits with raw (unrepaired) start/end: ``_s`` (datetime, else the date), ``_e`` (NaT if none)."""
    v = visits.copy()
    v["person_id"] = pd.to_numeric(v["person_id"], errors="coerce")
    s = rules._dt(v, "visit_start_datetime")
    v["_s"] = s.where(s.notna(), rules._dt(v, "visit_start_date"))
    en = rules._dt(v, "visit_end_datetime")
    v["_e"] = en.where(en.notna(), rules._dt(v, "visit_end_date"))
    v["_s_date_only"] = v["_s"].notna() & (v["_s"] == v["_s"].dt.normalize())
    v["_e_date_only"] = v["_e"].notna() & (v["_e"] == v["_e"].dt.normalize())
    v = v[v["person_id"].notna() & v["_s"].notna()].copy()
    v["person_id"] = v["person_id"].astype("int64")
    return v


def _nearest_diff(e: pd.DataFrame, v: pd.DataFrame, col: str) -> pd.Series:
    """EEG start minus the person's nearest visit ``col`` (hours, signed; positive = EEG after the visit time)."""
    vv = v[["person_id", col]].dropna().sort_values(col)
    ee = e[["person_id", "t0"]].assign(_i=e.index).sort_values("t0")
    if not len(vv) or not len(ee):
        return pd.Series(dtype=float)
    m = pd.merge_asof(ee, vv.rename(columns={col: "_k"}), left_on="t0", right_on="_k", by="person_id",
                      direction="nearest")
    return pd.Series(((m["t0"] - m["_k"]).dt.total_seconds() / 3600.0).to_numpy(), index=m["_i"].to_numpy())


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
def run_diag(store, sites: list[str] | None = None) -> tuple[dict, set[str]]:
    """Return ``(aggregate_report, known_ids)``. ``known_ids`` are the record identifiers seen (to prove none leaks)."""
    src = StoreSources(store, sites)
    S = src.sessions().reset_index(drop=True)
    site_list = sorted(S["SiteID"].astype(str).unique())
    adult = S[S["person_id"].notna() & S["t0"].notna() & (S["age_years"] >= 18)].copy()
    adult["person_id"] = adult["person_id"].astype("int64")
    adult["SiteID"] = adult["SiteID"].astype(str)

    # alternative id forms, from the raw metadata, per site
    forms = []
    for site in site_list:
        try:
            meta = schema.coerce_types("eeg_metadata", data_io.read_site_table("eeg_metadata", site, s3=store))
        except FileNotFoundError:
            continue
        f = _id_forms(meta, site)
        f["SiteID"] = site
        forms.append(f)
    forms = pd.concat(forms, ignore_index=True) if forms else pd.DataFrame()
    cand_ids = set(adult["person_id"])
    if len(forms):
        cand_ids |= set(forms["bd_num"].dropna().astype("int64")) | set(forms["bids_num"].dropna().astype("int64"))

    visits = src.visits(sorted(cand_ids))
    v = _visit_frame(visits)
    vpids = set(v["person_id"])
    known = {str(i) for i in cand_ids} | set(S["SessionID"].dropna().astype(str))

    report: dict = {"note": "aggregates only; counts < 11 shown as \"<11\"; quantiles need n >= 11; hours are signed "
                            "(EEG start minus visit time: positive = EEG after the visit time)",
                    "sites": {}, "visit_concept_id": {}, "visit_source_value": {}}
    report["visit_table"] = {
        "n_visit_rows_for_adult_candidates": suppress_count(len(visits)),
        "pid_dtype_after_read": str(visits["person_id"].dtype) if len(visits) else "n/a",
        "columns_read": sorted(c for c in visits.columns),
        "share_start_datetime_null": _share(rules._dt(visits, "visit_start_datetime").isna()) if len(visits) else SUPPRESSED}
    parts = data_io.omop_parts(store, "visit_occurrence")
    if parts:
        try:
            pf = data_io.open_parquet(store, data_io.access_point(), parts[0])
            report["visit_table"]["pid_arrow_type"] = str(pf.schema_arrow.field("person_id").type)
            report["visit_table"]["footer_column_names"] = list(pf.schema_arrow.names)
        except Exception:                                                  # noqa: BLE001
            report["visit_table"]["pid_arrow_type"] = "unreadable"

    cur = rules.match_visits(adult, visits, ("ICU", "Inpatient", "ED"), 6.0)
    cover = _closed_variants(adult[["person_id", "t0"]], v)
    d_start, d_end = _nearest_diff(adult, v, "_s"), _nearest_diff(adult, v, "_e")
    pmin = v.groupby("person_id")["_s"].min()
    pmax = v.assign(_hi=v["_e"].where(v["_e"].notna(), v["_s"])).groupby("person_id")["_hi"].max()
    back = pd.merge_asof(adult[["person_id", "t0"]].assign(_i=adult.index).sort_values("t0"),
                         v[["person_id", "_s", "_e"]].sort_values("_s"), left_on="t0", right_on="_s",
                         by="person_id", direction="backward").set_index("_i")

    for site in site_list:
        a = adult[adult["SiteID"] == site]
        if a.empty:
            continue
        ix = a.index
        has = a["person_id"].isin(vpids)
        blk = {"n_adult_sessions_with_start": suppress_count(len(a)),
               "share_pid_has_any_visit_row": _share(has)}
        f = forms[forms["SiteID"] == site] if len(forms) else forms
        if len(f):
            keyed = f["bd_num"].dropna().astype("int64"), f["bids_num"].dropna().astype("int64")
            blk["id_join"] = {
                "n_metadata_sessions": suppress_count(len(f)),
                "share_bdsp_id_blank": _share(f["bd_blank"]),
                "share_bdsp_id_non_numeric": _share(f["bd_nonnumeric"]),
                "share_bdsp_id_leading_zero": _share(f["bd_leading_zero"]),
                "share_bdsp_and_folder_ids_differ": _share(
                    f["bd_num"].notna() & f["bids_num"].notna() & (f["bd_num"] != f["bids_num"])),
                "share_with_visit_via_bdsp_id_number": _share(f["bd_num"].isin(vpids)[f["bd_num"].notna()]),
                "share_with_visit_via_folder_tail": _share(f["bids_num"].isin(vpids)[f["bids_num"].notna()])}
        hv = a[has]
        blk["share_without_visit_row_by_pid"] = _share(~has)
        if len(hv):
            hix = hv.index
            blk["visit_end_null_share_of_visit_rows"] = _share(
                v[v["person_id"].isin(set(a["person_id"]))]["_e"].isna())
            vs = v[v["person_id"].isin(set(a["person_id"]))]
            blk["visit_start_midnight_share"] = _share(vs["_s_date_only"])
            blk["visit_end_midnight_share_of_known_ends"] = _share(vs.loc[vs["_e"].notna(), "_e_date_only"])
            blk["share_latest_start_visit_has_null_end"] = _share(back.loc[hix, "_e"].isna() & back.loc[hix, "_s"].notna())
            blk["hours_eeg_minus_nearest_visit_start"] = _hours_q(d_start.reindex(hix))
            blk["hours_eeg_minus_nearest_visit_end"] = _hours_q(d_end.reindex(hix))
            blk["share_eeg_date_outside_visit_date_range"] = _share(
                (a.loc[hix, "t0"].dt.normalize() < a.loc[hix, "person_id"].map(pmin).dt.normalize())
                | (a.loc[hix, "t0"].dt.normalize() > a.loc[hix, "person_id"].map(pmax).dt.normalize()))
        blk["share_covered_of_all_adult_eegs"] = {k: _share(cover.loc[ix, k]) for k in cover.columns}
        if has.any():
            blk["share_covered_of_eegs_with_a_visit_row"] = {k: _share(cover.loc[hv.index, k]) for k in cover.columns}
        blk["share_matched_by_current_rule"] = _share(cur.loc[ix, "visit_start"].notna())
        blk["share_matched_acute_by_current_rule"] = _share(cur.loc[ix, "visit_class"].isin(["ICU", "Inpatient", "ED"]))
        blk["current_rule_visit_class_counts"] = _counts(cur.loc[ix, "visit_class"].fillna("unclassified").where(
            cur.loc[ix, "visit_start"].notna(), "no_visit"))
        report["sites"][site] = blk

    vc = v[v["person_id"].isin(cand_ids)]
    report["visit_concept_id"] = _counts(vc["visit_concept_id"] if "visit_concept_id" in vc else pd.Series(dtype=str),
                                         numeric=True)
    report["visit_source_value"] = _counts(vc["visit_source_value"] if "visit_source_value" in vc else pd.Series(
        dtype=str), top=20)
    report["visit_concept_id_by_site"] = {}
    if "visit_concept_id" in vc:
        site_of = adult.drop_duplicates("person_id").set_index("person_id")["SiteID"]
        for site in site_list:
            m = vc["person_id"].map(site_of) == site
            if m.any():
                report["visit_concept_id_by_site"][site] = _counts(vc.loc[m, "visit_concept_id"], numeric=True)
    report["merge_history_table"] = _merge_history_names(store)
    report["duration_by_service"] = _duration_by_service(S)
    assert_aggregate_only(report, known)
    return report, known
