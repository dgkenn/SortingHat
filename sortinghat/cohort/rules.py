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
from functools import lru_cache

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
CLASS_CATS = ["ICU", "ED", "Inpatient", "Outpatient"]                    # category order = acuity priority
ACUITY_RANK = {c: i for i, c in enumerate(CLASS_CATS)}                  # unclassified ranks last (4)
FAR_FUTURE_DAYS = 365.25 * 100
_S_OFFSET = 1 << 33                                                      # seconds offset so start fits 34 bits
PRUNE_BEFORE_DAYS = 400.0                                                # see ``prune_visits``


def _dt(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df:
        return pd.to_datetime(df[col], errors="coerce").astype("datetime64[us]")
    return pd.Series(pd.NaT, index=df.index, dtype="datetime64[us]")


def compact_visits(visits: pd.DataFrame, date_only_end_of_day: bool = True, dates_only: bool = False
                   ) -> pd.DataFrame:
    """Raw ``omop_visit_occurrence`` rows -> a COMPACT frame ``person_id`` int64, ``_start`` / ``_end``
    datetime64[s] (``_end`` NaT = unknown) and ``_cls`` (category: ICU / ED / Inpatient / Outpatient, NaN =
    unclassified). Roughly 25 bytes per visit, so tens of millions of visits fit in memory. Explicit rules:

    1. **Start**: ``visit_start_datetime``; if null, ``visit_start_date`` (start of that day). No start -> dropped.
    2. **End**: ``visit_end_datetime``; if null, ``visit_end_date``. A date-only end (a date column, or a datetime
       at exactly 00:00:00) is the END of that day when ``date_only_end_of_day``.
    3. **End before start** is a charting error: the visit is zero-length at its start.
    4. ``person_id`` is cast to int64 (OMOP ``person_id`` may arrive as text, with or without leading zeros).
    5. ``_inpt`` (inpatient-length, D-111): the end is known and its DATE is after the start's DATE.
    6. ``dates_only`` (D-112): start and end are reduced to their dates (midnight), with no end-of-day shift; the
       cover window is then [start date - slack, end date + slack].
    The null-end horizon (``open_days``) is applied at matching time (``with_horizon``)."""
    v = visits
    pid = pd.to_numeric(v["person_id"], errors="coerce")
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
    end = end.where(~(end < start), start)
    inpt = end.notna() & (end.dt.normalize() > start.dt.normalize())
    if dates_only:
        start, end = start.dt.normalize(), end.dt.normalize()
    ok = (pid.notna() & start.notna()).to_numpy(bool)
    cls = pd.Series(pd.Categorical([None] * int(ok.sum()), categories=CLASS_CATS), dtype="category")
    cols = [c for c in ("visit_concept_id", "visit_source_value") if c in v]
    if cols and ok.any():
        sub = v.loc[ok, cols]
        pairs = sub.drop_duplicates().copy()
        for c in ("visit_concept_id", "visit_source_value"):
            if c not in pairs:
                pairs[c] = pd.NA
        pairs["_cls"] = [fa.visit_class(a, b) for a, b in zip(pairs["visit_concept_id"], pairs["visit_source_value"])]
        cls = pd.Series(pd.Categorical(sub.merge(pairs[cols + ["_cls"]], on=cols, how="left")["_cls"].to_numpy(),
                                       categories=CLASS_CATS))
    return pd.DataFrame({"person_id": pid[ok].astype("int64").to_numpy(),
                         "_start": start[ok].astype("datetime64[s]").to_numpy(),
                         "_end": end[ok].astype("datetime64[s]").to_numpy(),
                         "_cls": pd.Categorical(cls.to_numpy(object), categories=CLASS_CATS),
                         "_inpt": inpt[ok].to_numpy(bool)})


def with_horizon(v: pd.DataFrame, open_days: float | None = 30.0) -> pd.DataFrame:
    """Add ``end_known`` and the closed ``_endf``: a visit with no end is open for ``open_days`` days (None = 100 y)."""
    horizon = pd.Timedelta(days=open_days if open_days is not None else FAR_FUTURE_DAYS)
    known = v["_end"].notna()
    return v.assign(end_known=known, _endf=v["_end"].where(known, v["_start"] + horizon))


def prepare_visits(visits: pd.DataFrame, open_days: float | None = 30.0, date_only_end_of_day: bool = True,
                   dates_only: bool = False) -> pd.DataFrame:
    """``compact_visits`` (unless already compact) plus the open-end horizon (``with_horizon``)."""
    v = visits if "_start" in visits else compact_visits(visits, date_only_end_of_day, dates_only)
    if dates_only and "_start" in visits:                    # already compact (e.g. streamed): reduce to dates here too
        v = v.assign(_start=v["_start"].dt.normalize(), _end=v["_end"].dt.normalize())
    return with_horizon(v, open_days)


def prune_visits(v: pd.DataFrame, bounds: pd.DataFrame, slack_h: float = 0.0) -> pd.DataFrame:
    """Drop visits that cannot matter to any of the person's EEGs. ``bounds`` is indexed by person_id with the
    person's earliest (``lo``) and latest (``hi``) EEG start. A visit is kept unless
    * it starts after ``hi`` + slack (it cannot cover an EEG and cannot be part of an earlier encounter), or
    * its KNOWN end is more than ``PRUNE_BEFORE_DAYS`` before ``lo`` (it cannot cover an EEG; it could only extend
      an encounter that began over a year earlier, which fails the 48 h onset window either way).
    Persons without bounds are dropped."""
    if bounds is None or not len(v):
        return v
    hi = v["person_id"].map(bounds["hi"])
    lo = v["person_id"].map(bounds["lo"])
    keep = hi.notna() & (v["_start"] <= hi + pd.Timedelta(hours=slack_h)) & ~(
        v["_end"].notna() & (v["_end"] < lo - pd.Timedelta(days=PRUNE_BEFORE_DAYS)))
    return v[keep.to_numpy(bool)]


def visit_keys(j: pd.DataFrame, slack_h: float, dates_only: bool = False) -> tuple[pd.Series, pd.Series, pd.Series]:
    """For joined (session, visit) rows with columns ``t0``, ``_start``, ``_endf``, ``_cls``, ``_inpt``:
    ``(covered, key, exact)``. With ``dates_only`` (D-112) the EEG is compared by its DATE, so ``covered`` is
    [visit start date - slack, visit end date + slack]. ``exact`` is the same without slack. ``key`` is an integer
    where LOWER is better, in this order: exact before slack-only cover; inpatient-length visit (D-111); visit
    started on or before the EEG date; care-setting class (ICU < ED < Inpatient < Outpatient < unclassified; a no-op
    while every concept id is 0); then the LATEST start (D-112 tie-break)."""
    slack = pd.Timedelta(hours=slack_h)
    t = j["t0"].dt.normalize() if dates_only else j["t0"]
    exact = (j["_start"] <= t) & (t <= j["_endf"])
    covered = (j["_start"] - slack <= t) & (t <= j["_endf"] + slack)
    codes = pd.Series(pd.Categorical(j["_cls"], categories=CLASS_CATS).codes, index=j.index).astype("int64")
    rank = codes.where(codes >= 0, 4)
    inpt = j["_inpt"] if "_inpt" in j else pd.Series(False, index=j.index)
    after = (j["_start"] > t).astype("int64")
    start_s = j["_start"].astype("datetime64[s]").astype("int64") + _S_OFFSET
    flags = ((~exact).astype("int64") * 2 + (~inpt.astype(bool)).astype("int64")) * 2 + after
    key = (flags * 8 + rank) * (1 << 35) + ((1 << 34) - 1 - start_s)
    return covered, key, exact


def match_visits(sessions: pd.DataFrame, visits: pd.DataFrame, acute: tuple[str, ...], gap_h: float,
                 slack_h: float = 0.0, open_days: float | None = 30.0, date_only_end_of_day: bool = True,
                 chunk: int = 20000, dates_only: bool = False) -> pd.DataFrame:
    """Care-setting visit of each session (index = ``sessions`` index). ``visits`` is raw ``omop_visit_occurrence``
    or already compact (``compact_visits``). Sessions are processed ``chunk`` at a time, so memory is bounded by
    ``chunk`` x visits-per-person, not by the whole visit table.

    **Any-overlap rule** (replaces "the visit with the latest start", which loses a session whenever a short visit
    nested inside the real admission started later): among ALL of the person's visits whose interval covers t0
    (``visit_keys``; null ends per ``with_horizon``), the visit is chosen by
      1. covered exactly before covered only through ``slack_h`` (the interval widened by ``slack_h`` hours both
         sides; 0 = off),
      2. then care-setting priority ICU > ED > Inpatient > Outpatient > unclassified,
      3. then the latest start.
    Columns: ``visit_class`` (None when the matched visit says nothing about its setting), ``visit_start``,
    ``encounter_start``, ``visit_match`` ('exact' | 'slack'), ``visit_inpatient_length`` (end date after start date). ``encounter_start`` is the earliest start among the same
    person's ACUTE visits with a KNOWN end that overlap the matched visit or end within ``gap_h`` before it begins
    (one hop), so an ED visit followed by an admission counts from ED arrival; it is never later than t0 (a visit
    matched only through ``slack_h`` that starts after the EEG gives an encounter start of t0). NaT / None when
    unmatched."""
    out = pd.DataFrame({"visit_class": pd.Series(None, index=sessions.index, dtype=object),
                        "visit_start": pd.Series(pd.NaT, index=sessions.index, dtype="datetime64[us]"),
                        "encounter_start": pd.Series(pd.NaT, index=sessions.index, dtype="datetime64[us]"),
                        "visit_match": pd.Series(None, index=sessions.index, dtype=object),
                        "visit_inpatient_length": pd.Series(False, index=sessions.index, dtype=bool)})
    if not len(sessions) or not len(visits) or not ({"person_id"} <= set(visits)) or not (
            {"visit_start_datetime", "visit_start_date", "_start"} & set(visits)):
        return out
    v = prepare_visits(visits, open_days, date_only_end_of_day, dates_only)
    e = pd.DataFrame({"person_id": sessions["person_id"].astype("int64"),
                      "t0": sessions["t0"].astype("datetime64[s]"), "_i": sessions.index}).dropna()
    if not len(e) or not len(v):
        return out
    e = e.sort_values("person_id", kind="stable")
    av = v[(v["_cls"].isin(acute) | v["_inpt"]) & v["end_known"]][["person_id", "_start", "_end"]]
    gap = pd.Timedelta(hours=gap_h)
    vv = v[["person_id", "_start", "_endf", "_cls", "_inpt"]]
    for a in range(0, len(e), chunk):
        ec = e.iloc[a:a + chunk]
        pids = ec["person_id"].unique()
        j = ec.merge(vv[vv["person_id"].isin(pids)], on="person_id")
        if not len(j):
            continue
        covered, key, exact = visit_keys(j, slack_h, dates_only)
        j = j.assign(_key=key, _exact=exact)[covered.to_numpy(bool)]
        if not len(j):
            continue
        m = j.sort_values(["_i", "_key"]).drop_duplicates("_i", keep="first")
        ix = m["_i"].to_numpy()
        out.loc[ix, "visit_class"] = m["_cls"].astype(object).where(m["_cls"].notna(), None).to_numpy()
        out.loc[ix, "visit_start"] = m["_start"].astype("datetime64[us]").to_numpy()
        out.loc[ix, "visit_match"] = np.where(m["_exact"], "exact", "slack")
        out.loc[ix, "visit_inpatient_length"] = m["_inpt"].to_numpy(bool)
        k = m[["_i", "person_id", "_start", "t0"]].merge(av[av["person_id"].isin(m["person_id"].unique())],
                                                         on="person_id", suffixes=("", "_w"))
        k = k[(k["_start_w"] <= k["_start"]) & (k["_end"] >= k["_start"] - gap)]
        chain = k.groupby("_i")["_start_w"].min()
        enc = m.set_index("_i")["_start"]
        enc = pd.concat([enc, chain]).groupby(level=0).min()
        t0 = m.set_index("_i")["t0"].reindex(enc.index)
        enc = enc.where(enc <= t0, t0)
        out.loc[enc.index.to_numpy(), "encounter_start"] = enc.astype("datetime64[us]").to_numpy()
    return out


def acute_parts(visit_class: pd.Series, inpatient_length: pd.Series, service: pd.Series, acute_classes,
                service_acute, use_service_proxy: bool = True) -> tuple[pd.Series, pd.Series, pd.Series]:
    """The acute-care proxy (D-111) on a matched visit: ``(by_class, by_length, by_service)`` boolean Series.
    A visit classified by concept id / text (a no-op while every concept id is 0) decides by its class alone;
    otherwise the covering visit is inpatient-length (end date after start date) OR ``ServiceName`` (upper case)
    contains one of ``service_acute``. Used by the cohort build and by the Phase 0a audit."""
    known = visit_class.notna()
    by_class = known & visit_class.isin(list(acute_classes))
    by_length = ~known & inpatient_length.fillna(False).astype(bool)
    svc = service.fillna("").astype(str).str.upper()
    toks = [t.upper() for t in service_acute]
    hit = pd.Series([any(t in x for t in toks) for x in svc], index=service.index) if use_service_proxy else \
        pd.Series(False, index=service.index)
    return by_class, by_length, ~known & hit


# ---------------------------------------------------------------------------------------------- scores
@lru_cache(maxsize=200_000)
def _score_key(name: str) -> str | None:
    r = lx.classify_measurement(name)
    return r.key if r is not None and r.domain == "score" else None


def filter_score_rows(meas: pd.DataFrame) -> pd.DataFrame:
    """Keep ``omop_measurement`` rows that are a GCS / FOUR item (total or component); add ``key``."""
    if not len(meas) or "measurement_source_value" not in meas:
        return meas.iloc[0:0].assign(key=pd.Series(dtype=object))
    names = meas["measurement_source_value"].astype("string")
    cls = {n: _score_key(n) for n in names.dropna().unique()}
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


def compact_scores(meas: pd.DataFrame) -> pd.DataFrame:
    """Raw GCS / FOUR measurement rows -> COMPACT frame ``person_id`` int64, ``t`` datetime64[s], ``key`` category
    (``SCORE_KEYS``), ``value`` float32. Rows without a time or with an implausible value (``lexicon.PLAUSIBLE``;
    dropped, never clipped) are removed; non-score rows are removed. Safe to call per batch."""
    cols = {"person_id": np.zeros(0, "int64"), "t": np.zeros(0, "datetime64[s]"),
            "key": pd.Categorical([], categories=list(SCORE_KEYS)), "value": np.zeros(0, "float32")}
    if not len(meas) or "measurement_source_value" not in meas:
        return pd.DataFrame(cols)
    m = meas if "key" in meas else filter_score_rows(meas)
    if not len(m):
        return pd.DataFrame(cols)
    t = _times(m)
    val = pd.to_numeric(m["value_as_number"], errors="coerce")
    key = pd.Series(m["key"].to_numpy(object), index=m.index)
    lo = key.map({k: v[0] for k, v in lx.PLAUSIBLE.items()}).astype(float)
    hi = key.map({k: v[1] for k, v in lx.PLAUSIBLE.items()}).astype(float)
    ok = (t.notna() & val.notna() & (val >= lo) & (val <= hi)).to_numpy(bool)
    return pd.DataFrame({"person_id": pd.to_numeric(m["person_id"]).to_numpy()[ok].astype("int64"),
                         "t": t[ok].astype("datetime64[s]").to_numpy(),
                         "key": pd.Categorical(key[ok].to_numpy(object), categories=list(SCORE_KEYS)),
                         "value": val[ok].to_numpy("float32")})


def finish_scores(c: pd.DataFrame) -> pd.DataFrame:
    """Compact scores -> long frame ``person_id, t, instrument ('gcs'|'four'), value``. A GCS total is taken as
    charted; when eye, motor and verbal are all charted at the SAME timestamp and no total is, the total is their sum
    (verbal 'T'/intubated text values are non-numeric and simply absent)."""
    cols = ["person_id", "t", "instrument", "value"]
    if not len(c):
        return pd.DataFrame({k: pd.Series(dtype=object) for k in cols})
    m = pd.DataFrame({"person_id": c["person_id"].to_numpy(), "t": c["t"].to_numpy(),
                      "key": c["key"].astype(object).to_numpy(), "value": c["value"].to_numpy("float64")})
    comp = m[m["key"].isin(_COMPONENTS)]
    wide = comp.pivot_table(index=["person_id", "t"], columns="key", values="value", aggfunc="min")
    if len(wide) and set(_COMPONENTS) <= set(wide.columns):
        wide = wide.dropna(subset=list(_COMPONENTS))
        summed = wide[list(_COMPONENTS)].sum(axis=1).rename("value").reset_index().assign(key="gcs")
    else:
        summed = pd.DataFrame({"person_id": pd.Series(dtype="int64"), "t": pd.Series(dtype="datetime64[s]"),
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


def extract_scores(meas: pd.DataFrame) -> pd.DataFrame:
    """Long frame ``person_id, t, instrument, value`` from raw GCS / FOUR rows OR from ``compact_scores`` output."""
    if {"person_id", "t", "key", "value"} <= set(meas.columns) and "measurement_source_value" not in meas:
        return finish_scores(meas)
    return finish_scores(compact_scores(meas))


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
