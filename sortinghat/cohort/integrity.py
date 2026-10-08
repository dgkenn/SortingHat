"""Row-integrity checks for the cohort (run on EVERY build; abort with an aggregate message on violation).

Why: ``SessionID`` is a per-patient counter, not a global key, so a join on it alone silently gives thousands of
patients the same ``BidsFolder`` / ``SessionID`` (a real run produced 12,826 patients with 33 distinct BidsFolders).
The checks compare the cohort's rows with an INDEPENDENT view of the source ``eeg_metadata`` rows
(``source_index``: built from the raw columns, with the patient id taken from the ``BidsFolder`` text and the start
time looked up by that id and ``SessionID``, using none of the code that builds the session frame):

* every row's ``(SiteID, BidsFolder, SessionID)`` is one of the source rows;
* the row's patient id (before any merge map) equals the id inside its ``BidsFolder``;
* the row's start time equals the source row's start time;
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
                         "start_src": pd.Series(dtype="datetime64[us]")})


def source_index(meta: pd.DataFrame, rf: pd.DataFrame | None, site: str) -> pd.DataFrame:
    """Keys of one site's source rows. Independent of ``fa.merge_eeg`` / ``site_sessions``: the patient id comes from
    the ``BidsFolder`` text (``sub-<SITE><id>``), the start from ``StartTime(EEG)`` looked up by (that id,
    ``SessionID``) with the metadata ``StartTime`` as the fallback (the I0008/I0009 source)."""
    bf = meta["BidsFolder"].astype("string") if "BidsFolder" in meta else pd.Series(pd.NA, index=meta.index, dtype="string")
    sid = meta["SessionID"].astype("string")
    pid = pd.to_numeric(bf.str.replace(rf"^sub-{site}", "", regex=True), errors="coerce").astype("Int64")
    start = schema.parse_datetimes(meta["StartTime"]) if "StartTime" in meta else \
        pd.Series(pd.NaT, index=meta.index, dtype="datetime64[us]")
    out = pd.DataFrame({"SiteID": site, "BidsFolder": bf.astype(object).to_numpy(), "SessionID": sid.astype(object).to_numpy(),
                        "pid_folder": pid.to_numpy(), "start_src": start.to_numpy()})
    if rf is not None and len(rf) and schema.START_EEG in rf:
        r = pd.DataFrame({"pid_folder": pd.to_numeric(rf["BDSPPatientID"], errors="coerce").astype("Int64"),
                          "SessionID": rf["SessionID"].astype("string"),
                          "_rf_start": schema.parse_datetimes(rf[schema.START_EEG])}).dropna(subset=["pid_folder"])
        r = r.drop_duplicates(["pid_folder", "SessionID"]).assign(SessionID=lambda d: d["SessionID"].astype(object))
        n = len(out)
        out = out.merge(r, on=["pid_folder", "SessionID"], how="left")
        if len(out) != n:
            raise CohortIntegrityError("source index lost rows")
        out["start_src"] = out["_rf_start"].where(out["_rf_start"].notna(), out["start_src"])
        out = out.drop(columns="_rf_start")
    return out[INDEX_COLUMNS]


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
    good = pd.Series(ok_key & pid_ok & t_ok).groupby(j["_k"].to_numpy()).any().reindex(range(len(r))).to_numpy(bool)
    good = good | skip
    bad = int((~good).sum())
    if bad:
        miss = int((~pd.Series(ok_key).groupby(j["_k"].to_numpy()).any().reindex(range(len(r))).to_numpy(bool) & ~skip).sum())
        raise CohortIntegrityError(
            f"ROW INTEGRITY VIOLATED at stage '{stage}': {_count(bad)} of {_count(len(r))} rows do not match any source "
            f"eeg_metadata row on (SiteID, BidsFolder, SessionID) with the same patient id and start time "
            f"({_count(miss)} have no source row at all). Aborting: the output would be misaligned.")


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
