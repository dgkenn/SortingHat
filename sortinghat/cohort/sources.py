"""Where the cohort builder gets its inputs: an in-memory dict of HEEDB-layout tables, or a store (a local
directory in the access-point layout, or the real S3 client - human-run only).

Both return the same shapes, so the builder never knows which it is running on:

* ``sessions()``           one row per EEG session (canonical names; ``SITE_VARIANTS`` already applied)
* ``visits(person_ids)``   ``omop_visit_occurrence`` rows of those people
* ``scores(person_ids)``   ``omop_measurement`` rows whose source text is a GCS / FOUR item (nothing else is kept)
* ``conditions(person_ids)`` ``omop_condition_occurrence`` rows whose code is in the phenotype list

Stores stream OMOP parts through ``data_io.iter_omop_batches`` filtered to the requested people inside Arrow,
and apply the row filters per batch, so memory scales with the cohort, not with the 66 GB measurement table.
Nothing here prints; callers must not either (CLAUDE.md rule 3).
"""

from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd

from .. import data_io, schema
from ..audit import field_audit as fa
from . import rules

SESSION_COLUMNS = ["SiteID", "person_id", "SessionID", "BidsFolder", "EEGFolder", "ServiceName", "t0",
                   "t_end", "age_years", "duration_raw_s"]
_VISIT_COLS = ["person_id", "visit_occurrence_id", "visit_start_datetime", "visit_end_datetime",
               "visit_concept_id", "visit_source_value",
               "visit_start_date", "visit_end_date"]       # date twins: real header names, used when a datetime is null
_MEAS_COLS = ["person_id", "measurement_datetime", "measurement_date", "measurement_time",
              "measurement_source_value", "value_as_number"]
_COND_COLS = ["person_id", "condition_start_datetime", "condition_source_value"]


# ------------------------------------------------------------------------------------------ merge history
MERGE_PREFIX = "PatientMergeHistory/"
# The layout of PatientMergeHistory/ is UNREAD (the dry run listed three file names at the access-point root, nothing
# inside them). Column names are therefore ASSUMED: one column naming the retired id and one the surviving id.
_ID_COL = re.compile(r"id|patient|person|mrn", re.I)
_OLD_COL = re.compile(r"old|merged|retired|source|from|secondary|duplicate|prior|previous|deprecated", re.I)
_NEW_COL = re.compile(r"new|surviv|target|\bto\b|primary|master|current|final|canonical|kept", re.I)


MERGE_DEFAULT_COLS = ("MergedBDSPPatientID", "BDSPPatientID")        # D-114: seen in the real merge-history CSVs
MERGE_TIME_COL = "BDSPLastModifiedDTS"


def merge_rows(df: pd.DataFrame, cols: tuple[str, str] | None = None) -> pd.DataFrame | None:
    """Merge-history rows as ``old, new, mod`` (int64, int64, datetime), or None when the old/new id columns are not
    identifiable. Column choice, in order: explicit ``cols``; the real layout ``MergedBDSPPatientID`` ->
    ``BDSPPatientID`` (D-114); else exactly one id-like column matching the 'old' words and one matching the 'new'
    words. ``mod`` is ``BDSPLastModifiedDTS`` when present (NaT otherwise)."""
    if cols is None and set(MERGE_DEFAULT_COLS) <= set(df.columns):
        cols = MERGE_DEFAULT_COLS
    if cols is not None:                                    # explicit names from the diagnostic
        old, new = ([cols[0]], [cols[1]]) if cols[0] in df and cols[1] in df else ([], [])
    else:
        idc = [c for c in df.columns if _ID_COL.search(str(c))]
        old = [c for c in idc if _OLD_COL.search(str(c)) and not _NEW_COL.search(str(c))]
        new = [c for c in idc if _NEW_COL.search(str(c)) and not _OLD_COL.search(str(c))]
    if len(old) != 1 or len(new) != 1:
        return None
    o = pd.to_numeric(df[old[0]], errors="coerce")
    n = pd.to_numeric(df[new[0]], errors="coerce")
    ok = (o.notna() & n.notna() & (o != n)).to_numpy(bool)
    mod = pd.to_datetime(df[MERGE_TIME_COL], errors="coerce") if MERGE_TIME_COL in df else pd.Series(pd.NaT, index=df.index)
    return pd.DataFrame({"old": o[ok].astype("int64").to_numpy(), "new": n[ok].astype("int64").to_numpy(),
                         "mod": mod[ok].astype("datetime64[us]").to_numpy()})


def merge_pairs(df: pd.DataFrame, cols: tuple[str, str] | None = None) -> dict[int, int] | None:
    """{retired id: surviving id} from one merge-history frame (see ``merge_rows``); conflicts: latest ``mod`` wins."""
    r = merge_rows(df, cols)
    return None if r is None else resolve_rows(r)


def resolve_rows(rows: pd.DataFrame) -> dict[int, int]:
    """One surviving id per retired id: when a retired id appears with several survivors, the row with the latest
    ``BDSPLastModifiedDTS`` wins (rows without a time lose to rows with one; remaining ties: the later row)."""
    r = rows.reset_index(drop=True).assign(_o=lambda d: d.index).sort_values(["mod", "_o"], na_position="first")
    r = r.drop_duplicates("old", keep="last")
    return {int(a): int(b) for a, b in zip(r["old"], r["new"])}


def resolve_merges(m: dict[int, int], max_hops: int = 20) -> dict[int, int]:
    """Follow merge chains (A -> B -> C gives A -> C); a cycle stops at the last id reached before it repeats."""
    out = {}
    for a in m:
        seen, cur = {a}, m[a]
        for _ in range(max_hops):
            nxt = m.get(cur)
            if nxt is None or nxt in seen:
                break
            seen.add(cur)
            cur = nxt
        out[a] = cur
    return out



def remap_ids(df: pd.DataFrame, mm: dict[int, int] | None) -> pd.DataFrame:
    """Re-key ``person_id`` through the merge map (retired -> surviving id)."""
    if not mm or not len(df):
        return df
    pid = pd.to_numeric(df["person_id"], errors="coerce")
    mapped = pid.map(mm)
    return df.assign(person_id=mapped.where(mapped.notna(), pid).astype("int64"))


def concat_frames(chunks: list[pd.DataFrame], like: pd.DataFrame) -> pd.DataFrame:
    """Concatenate compact chunks COLUMN BY COLUMN, releasing each chunk's column as soon as it is copied, so the peak
    is about the result plus one column (``pd.concat`` would hold the chunks and a full copy together)."""
    if not chunks:
        return like.iloc[0:0]
    out: dict = {}
    for col in like.columns:
        if isinstance(like[col].dtype, pd.CategoricalDtype):
            cats = list(like[col].cat.categories)
            codes = np.concatenate([c[col].cat.codes.to_numpy() for c in chunks])
            out[col] = pd.Categorical.from_codes(codes, categories=cats)
        else:
            out[col] = np.concatenate([c[col].to_numpy() for c in chunks])
        for c in chunks:
            del c[col]
    chunks.clear()
    return pd.DataFrame(out)


def prune_window(df: pd.DataFrame, tcol: str, window: pd.DataFrame | None) -> pd.DataFrame:
    """Keep rows whose time is within the person's [lo, hi] (``window`` indexed by person_id)."""
    if window is None or not len(df):
        return df
    lo, hi = df["person_id"].map(window["lo"]), df["person_id"].map(window["hi"])
    t = df[tcol]
    return df[(lo.notna() & (t >= lo) & (t <= hi)).to_numpy(bool)]



def _text(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].astype("string") if col in df else pd.Series(pd.NA, index=df.index, dtype="string")


def site_sessions(meta: pd.DataFrame, rf: pd.DataFrame | None, site: str) -> pd.DataFrame:
    """One site's session frame from its canonical eeg_metadata and (optional) reports_findings.

    Start/end/age come from ``field_audit.merge_eeg`` (the audit's rule, so the cohort and the audit agree):
    ``StartTime(EEG)`` of the findings table, else the metadata ``StartTime`` (the only source at I0008/I0009,
    whose findings file does not exist). ``ServiceName`` prefers the metadata column, else ``ServiceName(EEG)``."""
    m = fa.merge_eeg(meta, rf, site)
    extra = pd.DataFrame({
        "SessionID": _text(meta, "SessionID"),
        "BidsFolder": _text(meta, "BidsFolder"), "EEGFolder": _text(meta, "EEGFolder"),
        "ServiceName": _text(meta, "ServiceName"),
        "duration_raw_s": pd.to_numeric(meta["DurationInSeconds"], errors="coerce") if "DurationInSeconds" in meta
        else np.nan}).drop_duplicates("SessionID")
    out = m.merge(extra, on="SessionID", how="left")
    if rf is not None and schema.SERVICE_EEG in rf:
        svc = pd.DataFrame({"SessionID": _text(rf, "SessionID"), "_svc_rf": _text(rf, schema.SERVICE_EEG)}
                           ).drop_duplicates("SessionID")
        out = out.merge(svc, on="SessionID", how="left")
        out["ServiceName"] = out["ServiceName"].where(out["ServiceName"].notna(), out["_svc_rf"])
        out = out.drop(columns="_svc_rf")
    out["SiteID"] = site
    out = out.rename(columns={"StartTime": "t0", "EndTime": "t_end", "AgeAtVisit": "age_years"})
    out["t0"] = schema.parse_datetimes(out["t0"])
    out["t_end"] = schema.parse_datetimes(out["t_end"])
    out["age_years"] = pd.to_numeric(out["age_years"], errors="coerce")
    return out[SESSION_COLUMNS].reset_index(drop=True)


def empty_sessions() -> pd.DataFrame:
    dt = {"person_id": "Int64", "t0": "datetime64[us]", "t_end": "datetime64[us]", "age_years": float,
          "duration_raw_s": float}
    return pd.DataFrame({c: pd.Series(dtype=dt.get(c, object)) for c in SESSION_COLUMNS})


def _empty(table: str) -> pd.DataFrame:
    return schema.coerce_types(table, pd.DataFrame({c: pd.Series(dtype=object) for c in schema.columns(table)}))


class FrameSources:
    """In-memory raw tables keyed by ``schema`` table name (what ``sortinghat.synthetic.generate`` returns)."""

    def __init__(self, tables: dict[str, pd.DataFrame]):
        self.t = tables

    def sessions(self) -> pd.DataFrame:
        meta, rf = self.t["eeg_metadata"], self.t.get("reports_findings")
        parts = []
        for site, g in meta.groupby("SiteID", sort=True):
            sess = set(g["SessionID"].astype(str))
            fg = rf[rf["SessionID"].astype(str).isin(sess)] if rf is not None and len(rf) else None
            parts.append(site_sessions(g, fg, str(site)))
        if not parts:
            return empty_sessions()
        return pd.concat(parts, ignore_index=True)

    def merge_map(self) -> tuple[dict[int, int], str]:
        """({retired id: surviving id}, status). Optional table key ``patient_merge_history`` (not a ``schema`` table)."""
        d = self.t.get("patient_merge_history")
        if d is None:
            return {}, "absent"
        m = merge_pairs(d)
        return ({}, "unrecognised") if m is None else (resolve_merges(m), "applied")

    def _rows(self, table: str, person_ids) -> pd.DataFrame:
        d = self.t.get(table)
        if d is None or not len(d):
            return _empty(table)
        return d[d["person_id"].isin(set(int(p) for p in person_ids))]

    def visits(self, person_ids, bounds=None, remap=None, slack_h: float = 0.0) -> pd.DataFrame:
        return remap_ids(self._rows("omop_visit_occurrence", person_ids), remap)

    def scores(self, person_ids, window=None, remap=None) -> pd.DataFrame:
        return remap_ids(rules.filter_score_rows(self._rows("omop_measurement", person_ids)), remap)

    def conditions(self, person_ids, window=None, remap=None) -> pd.DataFrame:
        return remap_ids(rules.filter_phenotype_rows(self._rows("omop_condition_occurrence", person_ids)), remap)


class StoreSources:
    """A store in the access-point layout: ``data_io.LocalStore`` (synthetic data / a local mirror) or the real
    S3 client. Real runs are HUMAN-RUN ONLY (``data_io.make_client`` refuses inside an agent session)."""

    def __init__(self, store, sites: list[str] | None = None, merge_cols: tuple[str, str] | None = None):
        self.s3 = store
        self.sites = sites
        self.merge_cols = merge_cols          # (retired-id column, surviving-id column) once known from the diagnostic

    def sessions(self) -> pd.DataFrame:
        sites = self.sites or data_io.discover_sites(self.s3)
        parts = []
        for site in sites:
            try:
                meta = data_io.read_site_table("eeg_metadata", site, s3=self.s3)
            except FileNotFoundError:
                continue
            try:
                rf = data_io.read_site_table("reports_findings", site, s3=self.s3)    # absent at I0008 / I0009
            except FileNotFoundError:
                rf = None
            meta = schema.coerce_types("eeg_metadata", meta)
            rf = None if rf is None else schema.coerce_types("reports_findings", rf)
            parts.append(site_sessions(meta, rf, site))
        if not parts:
            raise FileNotFoundError("no eeg_metadata CSV found for any requested site")
        return pd.concat(parts, ignore_index=True)

    def merge_map(self) -> tuple[dict[int, int], str]:
        """Read ``PatientMergeHistory/`` if it has files (CSV or parquet); only whole-file reads, ids kept in memory."""
        keys = data_io.list_keys(self.s3, MERGE_PREFIX)
        frames = []
        for k in keys:
            low = k.lower()
            if not low.endswith((".csv", ".csv.gz", ".parquet", ".txt", ".tsv")):
                continue
            body = self.s3.get_object(Bucket=data_io.access_point(), Key=k)["Body"].read()
            df = pd.read_parquet(io.BytesIO(body)) if low.endswith(".parquet") else pd.read_csv(
                io.BytesIO(body), dtype=str, sep="\t" if low.endswith((".tsv", ".txt")) else ",")
            del body
            r = merge_rows(df, self.merge_cols)
            del df
            if r is not None and len(r):
                frames.append(r)
        if not keys:
            return {}, "absent"
        return (resolve_merges(resolve_rows(pd.concat(frames, ignore_index=True))), "applied") if frames else (
            {}, "unrecognised")

    def iter_rows(self, table: str, columns: list[str], person_ids, min_rows: int = 0):
        """Yield coerced pandas chunks of ``table`` for ``person_ids`` (filtered inside Arrow, row group by row group),
        coalesced to at least ``min_rows`` rows. Nothing is accumulated here."""
        pids = sorted({int(p) for p in person_ids})
        spec = table[len("omop_"):]
        buf, n = [], 0
        for batch in data_io.iter_omop_batches(spec, person_ids=pids, columns=columns, s3=self.s3,
                                               batch_rows=max(65536, min_rows)):
            d = schema.coerce_types(table, batch.to_pandas())
            del batch
            buf.append(d)
            n += len(d)
            if n >= min_rows:
                yield buf[0] if len(buf) == 1 else pd.concat(buf, ignore_index=True)
                buf, n = [], 0
        if buf:
            yield buf[0] if len(buf) == 1 else pd.concat(buf, ignore_index=True)

    def visits(self, person_ids, bounds=None, remap=None, slack_h: float = 0.0) -> pd.DataFrame:
        """COMPACT visits (``rules.compact_visits``), compacted, re-keyed and pruned PER CHUNK as they stream in."""
        chunks = []
        for raw in self.iter_rows("omop_visit_occurrence", _VISIT_COLS, person_ids, min_rows=250_000):
            c = rules.compact_visits(raw)
            del raw
            c = rules.prune_visits(remap_ids(c, remap), bounds, slack_h)
            if len(c):
                chunks.append(c)
        return concat_frames(chunks, rules.compact_visits(pd.DataFrame({
            "person_id": [1], "visit_start_datetime": [pd.Timestamp("2000-01-01")]})).iloc[0:0])

    def scores(self, person_ids, window=None, remap=None) -> pd.DataFrame:
        """COMPACT GCS / FOUR rows (``rules.compact_scores``), re-keyed and pruned to ``window`` per chunk."""
        chunks = []
        for raw in self.iter_rows("omop_measurement", _MEAS_COLS, person_ids, min_rows=250_000):
            c = rules.compact_scores(rules.filter_score_rows(raw))
            del raw
            c = prune_window(remap_ids(c, remap), "t", window)
            if len(c):
                chunks.append(c)
        return concat_frames(chunks, rules.compact_scores(pd.DataFrame()))

    def conditions(self, person_ids, window=None, remap=None) -> pd.DataFrame:
        chunks = []
        for raw in self.iter_rows("omop_condition_occurrence", _COND_COLS, person_ids, min_rows=250_000):
            f = rules.filter_phenotype_rows(raw)
            del raw
            if not len(f):
                continue
            c = pd.DataFrame({"person_id": pd.to_numeric(f["person_id"]).to_numpy().astype("int64"),
                              "condition_start_datetime": pd.to_datetime(f["condition_start_datetime"], errors="coerce"
                                                                         ).astype("datetime64[s]").to_numpy()})
            c = prune_window(remap_ids(c, remap), "condition_start_datetime", window)
            if len(c):
                chunks.append(c)
        like = pd.DataFrame({"person_id": np.zeros(0, "int64"),
                             "condition_start_datetime": np.zeros(0, "datetime64[s]")})
        return concat_frames(chunks, like)
