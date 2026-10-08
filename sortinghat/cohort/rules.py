"""Pure, row-level rules of the cohort builder (no I/O, no printing). Record-level frames stay in memory.

* visit matching and the acute-care proxy         ``match_visits``
* GCS / FOUR extraction                           ``filter_score_rows``, ``extract_scores``
* ACI onset proxy                                 ``onset_times``
* strict severity (GCS <= 11 or FOUR <= 12)       ``severity``
* broad EHR phenotype                             ``filter_phenotype_rows``, ``phenotype``

Score items are identified by ``baselines.lexicon.classify_measurement`` and range-checked against
``lexicon.PLAUSIBLE``, so the cohort and the baselines read the same rows the same way. NOTE: the cohort uses
scores up to +6 h AFTER t0 to define who is in the strict cohort (plan: "within +-6 h of t0"). That is a
selection rule, not a feature; baselines never see these rows because every feature goes through
``baselines.asof.as_of``.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..audit import field_audit as fa
from ..baselines import lexicon as lx

SCORE_KEYS = ("gcs", "gcs_eye", "gcs_motor", "gcs_verbal", "four")
_COMPONENTS = ("gcs_eye", "gcs_motor", "gcs_verbal")

# Broad phenotype: symptom-level codes for impaired consciousness only (no etiology-specific code, so the cohort
# does not select on a label family; G92 / G93.4 are banned as label evidence by D-007 and are not used here either).
# Codes are compared after removing dots and upper-casing.  ICD-10-CM: R40.0 somnolence, R40.1 stupor,
# R40.2x coma (incl. the coma-scale codes), R40.3 vegetative state, R40.4 transient alteration of awareness,
# R41.0 disorientation, R41.82 altered mental status.  ICD-9-CM: 780.01 coma, 780.02 transient alteration of
# awareness, 780.09 other alteration of consciousness, 780.97 altered mental status.
PHENOTYPE_CODES = re.compile(r"^(?:R40[0-4]|R410|R4182|78001|78002|78009|78097)")


# ---------------------------------------------------------------------------------------------- visits
ACUITY_RANK = {"ICU": 0, "ED": 1, "Inpatient": 2, "Outpatient": 3}      # unknown class ranks last
FAR_FUTURE_DAYS = 365.25 * 100


def _dt(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df:
        return pd.to_datetime(df[col], errors="coerce").astype("datetime64[us]")
    return pd.Series(pd.NaT, index=df.index, dtype="datetime64[us]")


def prepare_visits(visits: pd.DataFrame, open_days: float | None = 30.0, date_only_end_of_day: bool = True
                   ) -> pd.DataFrame:
    """Visits with a usable closed interval and a care-setting class (``_cls``). Explicit rules, in order:

    1. **Start**: ``visit_start_datetime``; if null, ``visit_start_date`` (start of that day). No start -> dropped.
    2. **End**: ``visit_end_datetime``; if null, ``visit_end_date``. A date-only end (a date column, or a datetime
       at exactly 00:00:00) is the END of that day when ``date_only_end_of_day`` (a visit ending "2020-03-02" covers
       the whole of 2 March).
    3. **End before start** is a charting error: the visit is zero-length at its start.
    4. **Null end** (``end_known`` False): the visit is treated as open for ``open_days`` days after its start
       (``None`` = unbounded).
    5. ``person_id`` is cast to int64 (OMOP ``person_id`` may arrive as text, with or without leading zeros)."""
    v = visits.copy()
    v["person_id"] = pd.to_numeric(v["person_id"], errors="coerce")
    start = _dt(v, "visit_start_datetime")
    start = start.where(start.notna(), _dt(v, "visit_start_date").dt.normalize())
    end = _dt(v, "visit_end_datetime")
    if date_only_end_of_day:
        midnight = end.notna() & (end == end.dt.normalize())
        end = end.where(~midnight, end + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1))
    d_end = _dt(v, "visit_end_date")
    if date_only_end_of_day:
        d_end = d_end.dt.normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    end = end.where(end.notna(), d_end)
    v = v.assign(_start=start, _end=end)
    v = v[v["person_id"].notna() & v["_start"].notna()].copy()
    v["person_id"] = v["person_id"].astype("int64")
    v["_end"] = v["_end"].where(v["_end"] >= v["_start"], v["_start"]).where(v["_end"].notna(), pd.NaT)
    v["end_known"] = v["_end"].notna()
    horizon = pd.Timedelta(days=open_days if open_days is not None else FAR_FUTURE_DAYS)
    v["_end"] = v["_end"].where(v["end_known"], v["_start"] + horizon)
    cols = [c for c in ("visit_concept_id", "visit_source_value") if c in v]
    if cols:
        pairs = v[cols].drop_duplicates().copy()
        for c in ("visit_concept_id", "visit_source_value"):
            if c not in pairs:
                pairs[c] = pd.NA
        pairs["_cls"] = [fa.visit_class(a, b) for a, b in zip(pairs["visit_concept_id"], pairs["visit_source_value"])]
        v = v.merge(pairs[cols + ["_cls"]], on=cols, how="left")
    else:
        v["_cls"] = None
    return v


def match_visits(sessions: pd.DataFrame, visits: pd.DataFrame, acute: tuple[str, ...], gap_h: float,
                 slack_h: float = 0.0, open_days: float | None = 30.0, date_only_end_of_day: bool = True
                 ) -> pd.DataFrame:
    """Care-setting visit of each session (index = ``sessions`` index).

    **Any-overlap rule** (replaces "the visit with the latest start", which loses a session whenever a short visit
    nested inside the real admission started later): among ALL of the person's prepared visits (see
    ``prepare_visits``) whose interval covers t0 (start <= t0 <= end), the visit is chosen by
      1. covered exactly before covered only through ``slack_h`` (the interval widened by ``slack_h`` hours both
         sides; 0 = off),
      2. then care-setting priority ICU > ED > Inpatient > Outpatient > unclassified,
      3. then the latest start.
    Columns: ``visit_class`` (None when the matched visit says nothing about its setting), ``visit_start``,
    ``encounter_start``, ``visit_match`` ('exact' | 'slack'). ``encounter_start`` is the earliest start among the same
    person's ACUTE visits with a KNOWN end that overlap the matched visit or end within ``gap_h`` before it begins
    (one hop), so an ED visit followed by an admission counts from ED arrival; it is never later than t0 (a visit
    matched only through ``slack_h`` that starts after the EEG gives an encounter start of t0). NaT / None when unmatched."""
    out = pd.DataFrame({"visit_class": pd.Series(None, index=sessions.index, dtype=object),
                        "visit_start": pd.Series(pd.NaT, index=sessions.index, dtype="datetime64[us]"),
                        "encounter_start": pd.Series(pd.NaT, index=sessions.index, dtype="datetime64[us]"),
                        "visit_match": pd.Series(None, index=sessions.index, dtype=object)})
    if not len(sessions) or not len(visits) or not {"person_id", "visit_start_datetime"} & set(visits):
        return out
    v = prepare_visits(visits, open_days, date_only_end_of_day)
    e = pd.DataFrame({"person_id": sessions["person_id"].astype("int64"),
                      "t0": sessions["t0"].astype("datetime64[us]"), "_i": sessions.index}).dropna()
    if not len(e) or not len(v):
        return out
    slack = pd.Timedelta(hours=slack_h)
    j = e.merge(v[["person_id", "_start", "_end", "_cls", "end_known"]], on="person_id")
    exact = (j["_start"] <= j["t0"]) & (j["t0"] <= j["_end"])
    wide = (j["_start"] - slack <= j["t0"]) & (j["t0"] <= j["_end"] + slack)
    j = j.assign(_exact=exact)[wide]
    if not len(j):
        return out
    j["_rank"] = j["_cls"].map(ACUITY_RANK).fillna(4)
    m = j.sort_values(["_i", "_exact", "_rank", "_start"], ascending=[True, False, True, False]
                      ).drop_duplicates("_i", keep="first")
    ix = m["_i"].to_numpy()
    out.loc[ix, "visit_class"] = m["_cls"].to_numpy()
    out.loc[ix, "visit_start"] = m["_start"].to_numpy()
    out.loc[ix, "visit_match"] = np.where(m["_exact"], "exact", "slack")
    # encounter start: one hop back over acute visits (known end) that touch the matched one
    av = v[v["_cls"].isin(acute) & v["end_known"]][["person_id", "_start", "_end"]]
    gap = pd.Timedelta(hours=gap_h)
    k = m[["_i", "person_id", "_start"]].merge(av, on="person_id", suffixes=("", "_w"))
    k = k[(k["_start_w"] <= k["_start"]) & (k["_end"] >= k["_start"] - gap)]
    chain = k.groupby("_i")["_start_w"].min()
    enc = m.set_index("_i")["_start"]
    enc = pd.concat([enc, chain]).groupby(level=0).min()
    enc = enc.where(enc <= e.set_index("_i")["t0"].reindex(enc.index), e.set_index("_i")["t0"].reindex(enc.index))
    out.loc[enc.index.to_numpy(), "encounter_start"] = enc.to_numpy()
    return out


# ---------------------------------------------------------------------------------------------- scores
def filter_score_rows(meas: pd.DataFrame) -> pd.DataFrame:
    """Keep ``omop_measurement`` rows that are a GCS / FOUR item (total or component); add ``key``."""
    if not len(meas) or "measurement_source_value" not in meas:
        return meas.iloc[0:0].assign(key=pd.Series(dtype=object))
    names = meas["measurement_source_value"].astype("string")
    cls = {n: (r.key if (r := lx.classify_measurement(n)) is not None and r.domain == "score" else None)
           for n in names.dropna().unique()}
    key = names.map(cls)
    keep = key.isin(SCORE_KEYS).fillna(False).to_numpy(bool)
    return meas.loc[keep].assign(key=key[keep].astype(object))


def _times(m: pd.DataFrame) -> pd.Series:
    """Measurement time: the datetime; else date + time-of-day; a date alone cannot place a score within +-6 h."""
    t = m["measurement_datetime"].astype("datetime64[us]") if "measurement_datetime" in m else \
        pd.Series(pd.NaT, index=m.index, dtype="datetime64[us]")
    if {"measurement_date", "measurement_time"} <= set(m):
        d = pd.to_datetime(m["measurement_date"], errors="coerce").dt.normalize()
        tod = pd.to_timedelta(m["measurement_time"].astype("string"), errors="coerce")
        t = t.where(t.notna(), (d + tod).astype("datetime64[us]"))
    return t


def extract_scores(meas: pd.DataFrame) -> pd.DataFrame:
    """Long frame ``person_id, t, instrument ('gcs'|'four'), value`` from raw GCS / FOUR rows.

    Implausible values (``lexicon.PLAUSIBLE``) are dropped, never clipped. A GCS total is taken as charted; when
    the eye, motor and verbal components are all charted at the SAME timestamp and no total is, the total is their
    sum (verbal 'T'/intubated text values are non-numeric and simply absent)."""
    cols = ["person_id", "t", "instrument", "value"]
    if not len(meas):
        return pd.DataFrame({c: pd.Series(dtype=object) for c in cols})
    m = meas if "key" in meas else filter_score_rows(meas)
    m = pd.DataFrame({"person_id": m["person_id"].astype("int64").to_numpy(), "t": _times(m).to_numpy(),
                      "key": m["key"].to_numpy(),
                      "value": pd.to_numeric(m["value_as_number"], errors="coerce").to_numpy()})
    lo_hi = m["key"].map(lx.PLAUSIBLE)
    in_range = np.array([lh is not None and lh[0] <= v <= lh[1] for lh, v in zip(lo_hi, m["value"])], dtype=bool)
    m = m[(m["t"].notna() & m["value"].notna()).to_numpy(bool) & in_range]
    comp = m[m["key"].isin(_COMPONENTS)]
    wide = comp.pivot_table(index=["person_id", "t"], columns="key", values="value", aggfunc="min")
    if len(wide) and set(_COMPONENTS) <= set(wide.columns):
        wide = wide.dropna(subset=list(_COMPONENTS))
        summed = wide[list(_COMPONENTS)].sum(axis=1).rename("value").reset_index().assign(key="gcs")
    else:
        summed = pd.DataFrame({"person_id": pd.Series(dtype="int64"), "t": pd.Series(dtype="datetime64[us]"),
                               "value": pd.Series(dtype=float), "key": pd.Series(dtype=object)})
    tot = m[m["key"].isin(("gcs", "four"))][["person_id", "t", "key", "value"]]
    if len(summed):
        have = set(zip(tot.loc[tot["key"] == "gcs", "person_id"], tot.loc[tot["key"] == "gcs", "t"]))
        summed = summed[[(p, t) not in have for p, t in zip(summed["person_id"], summed["t"])]]
        summed = summed[(summed["value"] >= lx.PLAUSIBLE["gcs"][0]) & (summed["value"] <= lx.PLAUSIBLE["gcs"][1])]
    both = pd.concat([tot, summed[["person_id", "t", "key", "value"]]], ignore_index=True)
    both = both.rename(columns={"key": "instrument"})
    both["t"] = both["t"].astype("datetime64[us]")
    return both[cols].reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ onset
def onset_times(index: pd.DataFrame, scores: pd.DataFrame, rule: str, gcs_max: float, four_max: float
                ) -> pd.DataFrame:
    """ACI onset proxy for each index row (index must carry ``person_id``, ``t0``, ``encounter_start``).

    * ``score_then_visit``: the first abnormal score (GCS <= ``gcs_max`` or FOUR <= ``four_max``) charted in
      [encounter start, t0]; else the encounter start (ED arrival / admission);
    * ``visit_start``: the encounter start only;  * ``score_only``: the first abnormal score only.
    Scores before the encounter start belong to an earlier encounter and are ignored. Returns ``onset`` and
    ``onset_basis`` ('abnormal_score' | 'visit_start' | None)."""
    out = pd.DataFrame({"onset": pd.Series(pd.NaT, index=index.index, dtype="datetime64[us]"),
                        "onset_basis": pd.Series(None, index=index.index, dtype=object)})
    first = pd.Series(pd.NaT, index=index.index, dtype="datetime64[us]")
    if rule != "visit_start" and len(scores):
        ab = scores[((scores["instrument"] == "gcs") & (scores["value"] <= gcs_max))
                    | ((scores["instrument"] == "four") & (scores["value"] <= four_max))]
        j = index[["person_id", "t0", "encounter_start"]].reset_index().merge(ab[["person_id", "t"]], on="person_id")
        j = j[(j["t"] >= j["encounter_start"]) & (j["t"] <= j["t0"])]
        f = j.groupby(j.columns[0])["t"].min()                # first column is the preserved index
        first.loc[f.index] = f.to_numpy()
    has_score = first.notna()
    out.loc[has_score, "onset"] = first[has_score]
    out.loc[has_score, "onset_basis"] = "abnormal_score"
    if rule != "score_only":
        use_visit = ~has_score & index["encounter_start"].notna()
        out.loc[use_visit, "onset"] = index.loc[use_visit, "encounter_start"]
        out.loc[use_visit, "onset_basis"] = "visit_start"
    return out


# ------------------------------------------------------------------------------------------- severity
def severity(index: pd.DataFrame, scores: pd.DataFrame, before_h: float, after_h: float, gcs_max: float,
             four_max: float, rule: str = "nearest") -> pd.DataFrame:
    """Strict-severity flag per index row: GCS <= ``gcs_max`` or FOUR <= ``four_max`` charted in
    [t0 - ``before_h``, t0 + ``after_h``].

    ``rule='nearest'`` (primary): per instrument only the score closest to t0 counts, the earlier (pre-t0) one on a
    tie. ``rule='any'`` (the ``strict_pm6`` sensitivity): any score of the instrument in the window qualifies.
    Also returns the lowest and the nearest GCS / FOUR in the window and the number of score observations
    (record-level; local file only)."""
    idx = index.index
    out = pd.DataFrame({"gcs_min": np.nan, "four_min": np.nan, "gcs_nearest": np.nan, "four_nearest": np.nan,
                        "n_score_obs": 0, "strict": False}, index=idx).astype(
        {"gcs_min": float, "four_min": float, "gcs_nearest": float, "four_nearest": float})
    if not len(scores) or not len(index):
        return out
    j = index[["person_id", "t0"]].reset_index().merge(scores, on="person_id")
    key = j.columns[0]
    j["dt_h"] = (j["t"] - j["t0"]).dt.total_seconds() / 3600.0
    j = j[(j["dt_h"] >= -before_h) & (j["dt_h"] <= after_h)]
    out["n_score_obs"] = j.groupby(key).size().reindex(idx).fillna(0).astype(int)
    for ins in ("gcs", "four"):
        out[f"{ins}_min"] = j[j["instrument"] == ins].groupby(key)["value"].min().reindex(idx)
    near = j.assign(_a=j["dt_h"].abs(), _post=(j["dt_h"] > 0).astype(int)).sort_values(
        [key, "instrument", "_a", "_post", "t"]).drop_duplicates([key, "instrument"])
    for ins in ("gcs", "four"):
        out[f"{ins}_nearest"] = near[near["instrument"] == ins].set_index(key)["value"].reindex(idx)
    if rule == "nearest":
        gq, fq = out["gcs_nearest"], out["four_nearest"]
    else:
        gq, fq = out["gcs_min"], out["four_min"]
    out["strict"] = ((gq <= gcs_max) | (fq <= four_max)).fillna(False).to_numpy(bool)
    return out


# ----------------------------------------------------------------------------------------- phenotype
def filter_phenotype_rows(cond: pd.DataFrame) -> pd.DataFrame:
    if not len(cond) or "condition_source_value" not in cond:
        return cond.iloc[0:0]
    code = cond["condition_source_value"].astype("string").str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    return cond.loc[code.str.contains(PHENOTYPE_CODES, na=False).to_numpy(bool)]


def phenotype(index: pd.DataFrame, cond: pd.DataFrame, after_h: float) -> pd.Series:
    """True when a phenotype condition starts in [encounter start, t0 + ``after_h``] (index = ``index.index``)."""
    out = pd.Series(False, index=index.index)
    if not len(cond) or not len(index):
        return out
    c = cond[["person_id", "condition_start_datetime"]].dropna()
    j = index[["person_id", "t0", "encounter_start"]].reset_index().merge(c, on="person_id")
    t = j["condition_start_datetime"].astype("datetime64[us]")
    j = j[(t >= j["encounter_start"]) & (t <= j["t0"] + pd.Timedelta(hours=after_h))]
    out.loc[j[j.columns[0]].unique()] = True
    return out
