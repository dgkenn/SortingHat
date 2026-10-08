"""Row-integrity checks for the cohort (run on EVERY build; abort with an aggregate message on violation).

Why: ``SessionID`` is a per-patient counter, not a global key, so a join on it alone silently gives thousands of
patients the same ``BidsFolder`` / ``SessionID`` (a real run produced 12,826 patients with 33 distinct BidsFolders).
The checks compare the cohort's rows with an INDEPENDENT view of the source ``eeg_metadata`` rows
(``source_index``: built from the raw columns, with the patient id taken from the ``BidsFolder`` text and the start
time looked up by that id and ``SessionID``, using none of the code that builds the session frame):

* every row's ``(SiteID, BidsFolder, SessionID)`` is one of the source rows;
* the row's patient id (before any merge map) equals the id inside its ``BidsFolder``;
* the row's start time equals one of the source row's start times (metadata StartTime or any StartTime(EEG));
* in the final table, ``(SiteID, BidsFolder, SessionID)`` and ``edf_key`` are unique (one row per patient).

Failures raise ``CohortIntegrityError`` carrying only suppressed counts: no id, key or date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import schema
from ..safe_output import suppress_count


class CohortIntegrityError(RuntimeError):
    """The cohort's rows do not match the source rows (aggregate message only)."""


INDEX_COLUMNS = ["SiteID", "BidsFolder", "SessionID", "pid_folder", "start_src"]


def empty_index() -> pd.DataFrame:
    return pd.DataFrame({"SiteID": pd.Series(dtype=object), "BidsFolder": pd.Series(dtype=object),
                         "SessionID": pd.Series(dtype=object), "pid_folder": pd.Series(dtype="Int64"),
                         "start_src": pd.Series(dtype="datetime64[us]"),
                         "pid_meta": pd.Series(dtype="Int64")})


def source_index(meta: pd.DataFrame, rf: pd.DataFrame | None, site: str) -> pd.DataFrame:
    """Keys of one site's source rows. Independent of ``fa.merge_eeg`` / ``site_sessions``: the patient id comes from
    the ``BidsFolder`` text (``sub-<SITE><id>``), the start from ``StartTime(EEG)`` looked up by (that id,
    ``SessionID``) with the metadata ``StartTime`` as the fallback (the I0008/I0009 source)."""
    bf = meta["BidsFolder"].astype("string") if "BidsFolder" in meta else pd.Series(pd.NA, index=meta.index, dtype="string")
    sid = meta["SessionID"].astype("string")
    pid = pd.to_numeric(bf.str.replace(rf"^sub-{site}", "", regex=True), errors="coerce").astype("Int64")
    start = schema.parse_datetimes(meta["StartTime"]) if "StartTime" in meta else \
        pd.Series(pd.NaT, index=meta.index, dtype="datetime64[us]")
    pid_meta = pd.to_numeric(meta["BDSPPatientID"], errors="coerce").astype("Int64") if "BDSPPatientID" in meta else \
        pd.Series(pd.NA, index=meta.index, dtype="Int64")
    out = pd.DataFrame({"SiteID": site, "BidsFolder": bf.astype(object).to_numpy(), "SessionID": sid.astype(object).to_numpy(),
                        "pid_folder": pid.to_numpy(), "start_src": start.to_numpy(), "pid_meta": pid_meta.to_numpy()})
    if rf is not None and len(rf) and schema.START_EEG in rf:
        # every legitimate start of a key is a candidate: the metadata StartTime AND each StartTime(EEG) of the report
        # rows for (patient id, SessionID); the cohort uses the first report start, the check accepts any of them
        r = pd.DataFrame({"pid_folder": pd.to_numeric(rf["BDSPPatientID"], errors="coerce").astype("Int64"),
                          "SessionID": rf["SessionID"].astype("string"),
                          "start_src": schema.parse_datetimes(rf[schema.START_EEG])}).dropna(subset=["pid_folder"])
        r = r.drop_duplicates().assign(SessionID=lambda d: d["SessionID"].astype(object))
        extra = out.drop(columns="start_src").drop_duplicates().merge(r, on=["pid_folder", "SessionID"], how="inner")
        out = pd.concat([out, extra[out.columns]], ignore_index=True).drop_duplicates()
        # a missing start is a candidate only for keys with no known start at all
        key = ["SiteID", "BidsFolder", "SessionID", "pid_folder"]
        known = out["start_src"].notna().groupby([out[k].astype(str) for k in key]).transform("any")
        out = out[out["start_src"].notna() | ~known.to_numpy(bool)]
    return out[INDEX_COLUMNS + ["pid_meta"]].reset_index(drop=True)


def ambiguous_identity(rows: pd.DataFrame, index: pd.DataFrame, pid_col: str = "person_id_source") -> pd.Series:
    """True for sessions whose SOURCE row has a ``BDSPPatientID`` different from the id inside its ``BidsFolder``
    (``sub-<SITE><id>``) and which carry that source ``BDSPPatientID`` (they faithfully reflect the source conflict).

    A row that merely has the wrong folder (a join error) is NOT flagged: the patient id it carries is not the one
    the source row for that folder holds, so ``verify_rows`` still aborts on it."""
    if "pid_meta" not in index or not len(rows) or not len(index):
        return pd.Series(False, index=rows.index)
    conflict = index[index["pid_meta"].notna() & index["pid_folder"].notna() & (index["pid_meta"] != index["pid_folder"])]
    conflict = conflict[["SiteID", "BidsFolder", "SessionID", "pid_meta"]].drop_duplicates()
    r = rows[["SiteID", "BidsFolder", "SessionID", pid_col]].copy()
    r["_k"] = np.arange(len(r))
    for c in ("SiteID", "BidsFolder", "SessionID"):
        r[c] = r[c].astype(object)
    j = r.merge(conflict, on=["SiteID", "BidsFolder", "SessionID"], how="inner")
    hit = j.loc[(j[pid_col].astype("Int64") == j["pid_meta"]).fillna(False).to_numpy(bool), "_k"].unique()
    return pd.Series(np.isin(np.arange(len(r)), hit), index=rows.index)


def _count(n: int) -> str:
    return str(suppress_count(int(n)))


def verify_rows(rows: pd.DataFrame, index: pd.DataFrame, stage: str, *, pid_col: str = "person_id_source",
                t_col: str = "t0", skip_bids: pd.Series | None = None) -> None:
    """Raise unless every row exists in the source index with the same patient id and start time."""
    if not len(rows):
        return
    r = rows[["SiteID", "BidsFolder", "SessionID", pid_col, t_col]].copy()
    r["_k"] = np.arange(len(r))
    r["SiteID"], r["SessionID"] = r["SiteID"].astype(object), r["SessionID"].astype(object)
    r["BidsFolder"] = r["BidsFolder"].astype(object)
    idx = index.drop_duplicates(INDEX_COLUMNS)
    # rows that cannot be checked: no BidsFolder in the source (the folder is built later from the id) or no patient id
    skip = (r["BidsFolder"].isna() | r[pid_col].isna()).to_numpy(bool)
    if skip_bids is not None:
        skip = skip | skip_bids.to_numpy(bool)
    j = r.merge(idx, on=["SiteID", "BidsFolder", "SessionID"], how="left", indicator=True)
    j = j.sort_values("_k", kind="stable")
    ok_key = (j["_merge"] == "both").to_numpy()
    # a source key can occur on several rows (duplicates): the row is fine if ANY of them agrees
    pid_ok = (j[pid_col].astype("Int64") == j["pid_folder"]).fillna(False).to_numpy()
    t_ok = ((j[t_col] == j["start_src"]) | (j[t_col].isna() & j["start_src"].isna())).fillna(False).to_numpy()
    k = j["_k"].to_numpy()

    def any_per_row(mask):
        return pd.Series(mask).groupby(k).any().reindex(range(len(r))).to_numpy(bool)

    has_key, has_pid, good = any_per_row(ok_key), any_per_row(ok_key & pid_ok), any_per_row(ok_key & pid_ok & t_ok)
    good = good | skip
    bad = int((~good).sum())
    if bad:
        n_key = int((~has_key & ~skip).sum())
        n_pid = int((has_key & ~has_pid & ~skip).sum())
        n_t = int((has_pid & ~good & ~skip).sum())
        raise CohortIntegrityError(
            f"ROW INTEGRITY VIOLATED at stage '{stage}': {_count(bad)} of {_count(len(r))} rows do not match any source "
            f"eeg_metadata row on (SiteID, BidsFolder, SessionID) with the same patient id and start time. "
            f"By cause: key missing (no source row) {_count(n_key)}; patient-id mismatch (key found, id differs from "
            f"the BidsFolder id) {_count(n_pid)}; start-time mismatch (key and id match, no source start equals the "
            f"row's) {_count(n_t)}. Aborting: the output would be misaligned.")


def verify_output(table: pd.DataFrame, keys: pd.DataFrame) -> None:
    """Uniqueness of the final rows: one (SiteID, BidsFolder, SessionID) and one ``edf_key`` per patient."""
    n = len(table)
    if not n:
        return
    k = table[["SiteID", "BidsFolder", "SessionID"]].astype(str).drop_duplicates().shape[0]
    u = keys["edf_key"].nunique()
    if k != n or u != n or table["person_id"].nunique() != n:
        raise CohortIntegrityError(
            f"ROW INTEGRITY VIOLATED in the output: {_count(n)} rows but {_count(k)} distinct (SiteID, BidsFolder, "
            f"SessionID), {_count(u)} distinct edf_key, {_count(table['person_id'].nunique())} distinct patients. Aborting.")
