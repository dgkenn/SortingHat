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
def _classified_visits(visits: pd.DataFrame) -> pd.DataFrame:
    v = visits.dropna(subset=["person_id", "visit_start_datetime"]).copy()
    if "visit_end_datetime" not in v:
        v["visit_end_datetime"] = pd.NaT
    cols = [c for c in ("visit_concept_id", "visit_source_value") if c in v]
    pairs = v[cols].drop_duplicates() if cols else pd.DataFrame(index=[0])
    for c in ("visit_concept_id", "visit_source_value"):
        if c not in pairs:
            pairs[c] = pd.NA
    pairs["_cls"] = [fa.visit_class(a, b) for a, b in zip(pairs["visit_concept_id"], pairs["visit_source_value"])]
    v = v.merge(pairs, on=[c for c in ("visit_concept_id", "visit_source_value") if c in v], how="left") \
        if cols else v.assign(_cls=None)
    v["person_id"] = v["person_id"].astype("int64")
    v["visit_start_datetime"] = v["visit_start_datetime"].astype("datetime64[us]")
    v["visit_end_datetime"] = v["visit_end_datetime"].astype("datetime64[us]")
    return v


def match_visits(sessions: pd.DataFrame, visits: pd.DataFrame, acute: tuple[str, ...], gap_h: float) -> pd.DataFrame:
    """Visit of each session (index = ``sessions`` index): the same person's visit with the latest start <= t0
    that has not ended before t0 (``field_audit.derive_patient_class``'s rule; an open visit has no end).

    Columns: ``visit_class`` (ICU / Inpatient / ED / Outpatient, or None when the visit exists but its concept id
    and source text say nothing), ``visit_start``, ``encounter_start``. ``encounter_start`` is the earliest start
    among the same person's ACUTE visits that overlap the matched visit or end within ``gap_h`` before it begins
    (one hop), so an ED visit followed by an admission counts from ED arrival. NaT everywhere when unmatched."""
    out = pd.DataFrame({"visit_class": pd.Series(None, index=sessions.index, dtype=object),
                        "visit_start": pd.Series(pd.NaT, index=sessions.index, dtype="datetime64[us]"),
                        "encounter_start": pd.Series(pd.NaT, index=sessions.index, dtype="datetime64[us]")})
    if not len(sessions) or not len(visits) or not {"person_id", "visit_start_datetime"} <= set(visits):
        return out
    v = _classified_visits(visits)
    e = pd.DataFrame({"person_id": sessions["person_id"].astype("int64"),
                      "t0": sessions["t0"].astype("datetime64[us]"), "_i": sessions.index}).dropna()
    if not len(e) or not len(v):
        return out
    m = pd.merge_asof(e.sort_values("t0"), v.sort_values("visit_start_datetime")[
        ["person_id", "visit_start_datetime", "visit_end_datetime", "_cls"]],
        left_on="t0", right_on="visit_start_datetime", by="person_id", direction="backward")
    ok = m["visit_start_datetime"].notna() & (m["visit_end_datetime"].isna() | (m["visit_end_datetime"] >= m["t0"]))
    m = m[ok]
    out.loc[m["_i"].to_numpy(), "visit_class"] = m["_cls"].to_numpy()
    out.loc[m["_i"].to_numpy(), "visit_start"] = m["visit_start_datetime"].to_numpy()
    # encounter start: one hop back over acute visits that touch the matched one
    av = v[v["_cls"].isin(acute) & v["visit_end_datetime"].notna()][["person_id", "visit_start_datetime",
                                                                      "visit_end_datetime"]]
    gap = pd.Timedelta(hours=gap_h)
    j = m[["_i", "person_id", "visit_start_datetime"]].merge(av, on="person_id", suffixes=("", "_w"))
    j = j[(j["visit_start_datetime_w"] <= j["visit_start_datetime"])
          & (j["visit_end_datetime"] >= j["visit_start_datetime"] - gap)]
    chain = j.groupby("_i")["visit_start_datetime_w"].min()
    enc = m.set_index("_i")["visit_start_datetime"]
    enc = pd.concat([enc, chain]).groupby(level=0).min()
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
def severity(index: pd.DataFrame, scores: pd.DataFrame, window_h: float, gcs_max: float, four_max: float,
             rule: str = "any") -> pd.DataFrame:
    """Strict-severity flags per index row: GCS <= ``gcs_max`` or FOUR <= ``four_max`` within +-``window_h`` of t0.

    ``rule='any'``: any score of the instrument in the window qualifies; ``'nearest'``: only the score closest to
    t0 (earlier on a tie). Also returns the lowest GCS / FOUR in the window and the number of score observations
    (record-level; local file only)."""
    idx = index.index
    out = pd.DataFrame({"gcs_min_window": np.nan, "four_min_window": np.nan, "n_score_obs_window": 0,
                        "severity_strict": False}, index=idx)
    out["gcs_min_window"] = out["gcs_min_window"].astype(float)
    out["four_min_window"] = out["four_min_window"].astype(float)
    if not len(scores) or not len(index):
        out["severity_scored"] = False
        return out
    j = index[["person_id", "t0"]].reset_index().merge(scores, on="person_id")
    key = j.columns[0]
    j["dt_h"] = (j["t"] - j["t0"]).dt.total_seconds() / 3600.0
    j = j[j["dt_h"].abs() <= window_h]
    out["n_score_obs_window"] = j.groupby(key).size().reindex(idx).fillna(0).astype(int)
    for ins, col in (("gcs", "gcs_min_window"), ("four", "four_min_window")):
        out[col] = j[j["instrument"] == ins].groupby(key)["value"].min().reindex(idx)
    if rule == "nearest":
        j = j.assign(_a=j["dt_h"].abs()).sort_values([key, "instrument", "_a", "t"])
        near = j.drop_duplicates([key, "instrument"])
        gq = near[near["instrument"] == "gcs"].set_index(key)["value"].reindex(idx)
        fq = near[near["instrument"] == "four"].set_index(key)["value"].reindex(idx)
    else:
        gq, fq = out["gcs_min_window"], out["four_min_window"]
    out["severity_strict"] = ((gq <= gcs_max) | (fq <= four_max)).fillna(False).to_numpy(bool)
    out["severity_scored"] = out["n_score_obs_window"] > 0
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
