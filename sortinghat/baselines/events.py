"""Index table (who, when is t0) and the long event frame built from OMOP-layout tables.

Nothing here compares anything with t0 except ``build_index`` (which *defines* t0 as the first qualifying EEG
start). Availability times are assigned here; the t0 gate itself is ``asof.as_of``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import schema
from ..audit import field_audit as fa
from . import lexicon as lx
from .asof import EVENT_COLUMNS, empty_events
from .config import BaselineConfig
from .encounter import resolve_encounter_start, restrict_to_current_encounter

ONE_DAY = pd.Timedelta(days=1)


# ----------------------------------------------------------------------------------------------- index
def build_index(tables: dict[str, pd.DataFrame], cfg: BaselineConfig | None = None) -> pd.DataFrame:
    """One row per adult patient: first qualifying (acute-care, start-time-present) EEG.

    Columns: person_id, SiteID, SessionID, t0 (= ``StartTime(EEG)``), age_years, sex_male, indication_raw, encounter_start,
    encounter_basis (D-145: the cohort's covering-visit rule; ``t0 - encounter_fallback_days`` when no visit covers t0).
    Selection mirrors ``field_audit.build_candidates`` so the cohort is the audit's cohort; it additionally
    keeps the session so the referral indication of *that* EEG is read.
    """
    meta_all = tables["eeg_metadata"]
    rf_all = tables.get("reports_findings")
    parts = []
    for site, meta in meta_all.groupby("SiteID", sort=True):
        rf = fa.site_findings(rf_all, meta, str(site))
        m = fa.merge_eeg(meta, rf, str(site))
        # merge_eeg is one row per metadata row, in order (left join on the unique (person_id, SessionID)): take the
        # metadata columns POSITIONALLY. SessionID alone is not a key (1..N per patient, repeated across patients).
        if len(m) != len(meta):
            raise RuntimeError("EEG frame lost row alignment with eeg_metadata")
        for c, name in (("ReferralIndication", "ReferralIndication"), ("SexDSC", "sex_meta")):
            if c in meta:
                m[name] = meta[c].to_numpy()
        if rf is not None and "SexDSC" in rf:
            sx = pd.DataFrame({"person_id": fa.person_ids(rf, str(site)), "SessionID": rf["SessionID"].astype("string"),
                               "sex_rf": rf["SexDSC"].to_numpy()}).drop_duplicates(["person_id", "SessionID"])
            n = len(m)
            m = m.merge(sx, on=["person_id", "SessionID"], how="left")
            if len(m) != n:
                raise RuntimeError("EEG frame lost row alignment with reports_findings")
        parts.append(m)
    eeg = pd.concat(parts, ignore_index=True)
    e, _ = fa.acute_adult(eeg)
    e = e[e["StartTime"].notna() & e["person_id"].notna()].sort_values(["person_id", "StartTime"], kind="stable")
    first = e.drop_duplicates("person_id", keep="first").reset_index(drop=True)   # whole row, not per-column first
    sex = first.get("sex_rf", pd.Series(index=first.index, dtype=object)).where(
        lambda s: s.notna(), first.get("sex_meta", pd.Series(index=first.index, dtype=object)))
    s = sex.astype("string").str.strip().str.lower()
    sex_male = pd.Series(np.nan, index=first.index)
    sex_male[s.str.startswith("m").fillna(False).to_numpy(bool)] = 1.0
    sex_male[s.str.startswith("f").fillna(False).to_numpy(bool)] = 0.0
    out = pd.DataFrame({
        "person_id": first["person_id"].astype("int64"), "SiteID": first["SiteID"].astype(str),
        "SessionID": first["SessionID"].astype(str), "t0": first["StartTime"].astype("datetime64[us]"),
        "age_years": first["AgeAtVisit"].astype(float), "sex_male": sex_male.to_numpy(float),
        "indication_raw": first.get("ReferralIndication")})
    out = out.sort_values("person_id").reset_index(drop=True)
    return add_encounter_start(out, tables.get("omop_visit_occurrence"), cfg or BaselineConfig())


def add_encounter_start(index: pd.DataFrame, visits: pd.DataFrame | None, cfg: BaselineConfig,
                        given: "pd.Series | None" = None) -> pd.DataFrame:
    """Index plus ``encounter_start`` / ``encounter_basis`` (``given``: a positionally aligned start, e.g. the cohort column)."""
    idx = index.drop(columns=[c for c in ("encounter_start", "encounter_basis") if c in index]).reset_index(drop=True)
    start, basis = resolve_encounter_start(idx, given, visits, cfg.encounter_fallback_days)
    return idx.assign(encounter_start=start.to_numpy(), encounter_basis=basis.to_numpy())


# ------------------------------------------------------------------------------------------------ utils
def _frame(person_id, domain, key, value, quantity, t_event, t_avail, t_end_raw, basis, approx, unit, raw_name):
    """Assemble an event frame; scalars broadcast to ``len(person_id)``."""
    n = len(person_id)

    def col(x, dtype):
        if isinstance(x, pd.Series):
            x = x.reset_index(drop=True)
        elif np.ndim(x) == 0 or x is pd.NaT:
            x = [x] * n
        return pd.Series(x, dtype=dtype) if not isinstance(x, pd.Series) else x.astype(dtype)

    out = pd.DataFrame({
        "person_id": col(person_id, "int64"), "domain": col(domain, object), "key": col(key, object),
        "value": col(value, float), "quantity": col(quantity, float),
        "t_event": col(t_event, "datetime64[us]"), "t_avail": col(t_avail, "datetime64[us]"),
        "t_end_raw": col(t_end_raw, "datetime64[us]"),
        "time_basis": col(basis, object), "approx": col(approx, bool), "unit": col(unit, object),
        "raw_name": col(raw_name, object)})
    return out[EVENT_COLUMNS]


def _dt(df: pd.DataFrame, dt_col: str, date_col: str | None = None):
    """Event time and a flag for date-only fallback. A date-only value is placed at the END of its day, so
    it can never be assumed available earlier than the day is over (conservative for the t0 gate)."""
    t = schema.parse_datetimes(df[dt_col]) if dt_col in df else pd.Series(pd.NaT, index=df.index, dtype="datetime64[us]")
    date_only = pd.Series(False, index=df.index)
    if date_col and date_col in df:
        d = schema.parse_datetimes(df[date_col])
        fill = t.isna() & d.notna()
        t = t.where(~fill, d.dt.normalize() + ONE_DAY - pd.Timedelta(microseconds=1))
        date_only = fill
    return t.astype("datetime64[us]"), date_only


def _restrict(df: pd.DataFrame, pids: set[int]) -> pd.DataFrame:
    return df[df["person_id"].astype("Int64").isin(pids)]


# -------------------------------------------------------------------------------------------------- drugs
def drug_events(tables: dict, pids: set[int], cfg: BaselineConfig) -> pd.DataFrame:
    d = tables.get("omop_drug_exposure")
    if d is None or d.empty:
        return empty_events()
    d = _restrict(d, pids).copy()
    text = d["drug_source_value"].astype("string").fillna("")
    concept = tables.get("omop_concept")
    if concept is not None and "drug_concept_id" in d and len(concept):
        names = concept.drop_duplicates("concept_id").set_index("concept_id")["concept_name"].astype("string")
        text = text + " " + d["drug_concept_id"].map(names).astype("string").fillna("")
    start = schema.parse_datetimes(d["drug_exposure_start_datetime"])
    end = schema.parse_datetimes(d["drug_exposure_end_datetime"]) if "drug_exposure_end_datetime" in d else \
        pd.Series(pd.NaT, index=d.index, dtype="datetime64[us]")
    end = end.where(end.isna() | (end >= start), start)            # stop before start is a charting error
    if cfg.drug_time_basis == "order":
        end = pd.Series(pd.NaT, index=d.index, dtype="datetime64[us]")
        admin = pd.Series(False, index=d.index)
    elif cfg.drug_time_basis == "admin":
        admin = pd.Series(True, index=d.index)
    else:
        admin = end.notna()
    frames = []
    for ing in lx.DRUG_CLASS:
        hit = text.str.contains(lx._DRUG_RE[ing], na=False)
        if not hit.any():
            continue
        s = d[hit.to_numpy(bool)]
        n = len(s)
        a = admin[hit]
        frames.append(_frame(
            s["person_id"].astype("int64").reset_index(drop=True), "drug", ing,
            np.nan, pd.to_numeric(s.get("quantity"), errors="coerce").reset_index(drop=True),
            start[hit].reset_index(drop=True), start[hit].reset_index(drop=True), end[hit].reset_index(drop=True),
            np.where(a, "admin", "order"), (~a).to_numpy(bool), None, s["drug_source_value"].reset_index(drop=True)))
    return pd.concat(frames, ignore_index=True) if frames else empty_events()


# ------------------------------------------------------------------------------------------ measurements
def measurement_events(tables: dict, pids: set[int], cfg: BaselineConfig, diag: dict) -> pd.DataFrame:
    m = tables.get("omop_measurement")
    if m is None or m.empty:
        return empty_events()
    m = _restrict(m, pids).copy()
    names = m["measurement_source_value"].astype("string")
    uniq = names.dropna().unique()
    rules = {u: lx.classify_measurement(u) for u in uniq}
    extra = set(cfg.extra_labs)
    # one lookup per UNIQUE name, gathered to the rows (the maps below are dict lookups, not a Python call per row)
    rule_dom = names.map({u: (r.domain if r else None) for u, r in rules.items()})
    rule_key = names.map({u: (r.key if r else None) for u, r in rules.items()})
    rule_lag = names.map({u: (r.lag_h if r else np.nan) for u, r in rules.items()}).astype(float)
    norm = names.map({u: lx.norm_name(u) for u in uniq})
    is_extra = rule_dom.isna() & norm.isin(extra)
    rule_dom = rule_dom.where(~is_extra, "lab")
    rule_key = rule_key.where(~is_extra, "x_" + norm.str.replace(" ", "_"))
    rule_lag = rule_lag.where(~is_extra, lx.DEFAULT_LAB_LAG_H)
    diag["n_measurement_rows_unmapped"] = int((rule_dom.isna()).sum())
    keep = rule_dom.notna().to_numpy(bool)
    m, rule_dom, rule_key, rule_lag = m[keep], rule_dom[keep], rule_key[keep], rule_lag[keep]
    t_coll, date_only = _dt(m, "measurement_datetime", "measurement_date")
    val = pd.to_numeric(m["value_as_number"], errors="coerce")
    unit = m["unit_source_value"] if "unit_source_value" in m else pd.Series(None, index=m.index, dtype=object)
    conv = [lx.to_canonical(k, v, u) for k, v, u in zip(rule_key, val, unit)]
    val = pd.Series(conv, index=m.index, dtype=float)
    lo_hi = rule_key.map(lx.PLAUSIBLE)
    bad = pd.Series([(isinstance(b, tuple) and v == v and not (b[0] <= v <= b[1])) for b, v in zip(lo_hi, val)],
                    index=m.index)
    val = val.mask(bad)
    # --- availability. Charted domains: charting time. Result domains: result time if present, else collection + lag.
    charted = rule_dom.isin(["score", "vital", "pupil", "poc_glucose", "hx"])
    res_col = next((c for c in schema.COLUMN_ALIASES["measurement.result_datetime"] if c in m.columns), None)
    t_res = schema.parse_datetimes(m[res_col]) if res_col else pd.Series(pd.NaT, index=m.index, dtype="datetime64[us]")
    t_lag = (t_coll + pd.to_timedelta(rule_lag.fillna(0.0), unit="h")).astype("datetime64[us]")
    if cfg.lab_time_basis == "collect_plus_lag":
        t_res = pd.Series(pd.NaT, index=m.index, dtype="datetime64[us]")
    use_res = t_res.notna() & ~charted
    t_avail = pd.Series(t_coll, index=m.index)
    t_avail = t_avail.where(charted, t_lag)
    t_avail = t_avail.where(~use_res, t_res)
    if cfg.lab_time_basis == "result":
        t_avail = t_avail.where(charted | use_res, pd.NaT)
    t_avail = t_avail.where(t_avail.isna() | (t_avail >= t_coll), t_coll)   # result earlier than collection = error
    approx = (~charted & ~use_res) | date_only
    basis = np.where(charted, "charted", np.where(use_res, "result", "collect+lag"))
    ok = ~(rule_dom == "hx") | (val.isna() | (val > 0))          # a charted 0 for a history item is a negative
    out = _frame(m["person_id"].astype("int64").reset_index(drop=True), rule_dom.reset_index(drop=True).to_numpy(object),
                 rule_key.reset_index(drop=True).to_numpy(object), val.reset_index(drop=True), np.nan,
                 t_coll.reset_index(drop=True), t_avail.reset_index(drop=True), pd.NaT, basis,
                 approx.to_numpy(bool), unit.reset_index(drop=True), names[keep].reset_index(drop=True))
    return out[ok.reset_index(drop=True).to_numpy(bool)].reset_index(drop=True)


# ----------------------------------------------------------------------------------------------- history
def history_events(tables: dict, pids: set[int]) -> pd.DataFrame:
    """Arrest / trauma / convulsion history from conditions, procedures and observations.

    Diagnosis codes have an unknown coding time (often assigned at discharge): they are treated as available
    at ``condition_start_datetime`` and flagged approximate (documented limitation, ``docs/baselines_spec.md``).
    """
    frames = []
    c = tables.get("omop_condition_occurrence")
    if c is not None and len(c):
        c = _restrict(c, pids)
        code = c["condition_source_value"].astype("string").str.replace(".", "", regex=False).str.upper().fillna("")
        t, d_only = _dt(c, "condition_start_datetime")
        for key, rx in (("arrest", lx.ARREST_ICD), ("head_trauma", lx.HEAD_TRAUMA_ICD), ("trauma", lx.TRAUMA_ICD)):
            hit = code.str.contains(rx, na=False).to_numpy(bool)
            if hit.any():
                s = c[hit]
                frames.append(_frame(s["person_id"].astype("int64").reset_index(drop=True), "hx", key, 1.0, np.nan,
                                     t[hit].reset_index(drop=True), t[hit].reset_index(drop=True), pd.NaT, "coded",
                                     True, None, s["condition_source_value"].reset_index(drop=True)))
    p = tables.get("omop_procedure_occurrence")
    if p is not None and len(p):
        p = _restrict(p, pids)
        t, _ = _dt(p, "procedure_datetime", "procedure_date")
        hit = p["procedure_source_value"].astype("string").str.strip().str.contains(lx.CPR_PROCEDURE, na=False).to_numpy(bool)
        if hit.any():
            s = p[hit]
            frames.append(_frame(s["person_id"].astype("int64").reset_index(drop=True), "hx", "arrest", 1.0, np.nan,
                                 t[hit].reset_index(drop=True), t[hit].reset_index(drop=True), pd.NaT, "coded", True,
                                 None, s["procedure_source_value"].reset_index(drop=True)))
    o = tables.get("omop_observation")
    if o is not None and len(o):
        o = _restrict(o, pids)
        t, d_only = _dt(o, "observation_datetime", "observation_date")
        txt = o["observation_source_value"].astype("string").fillna("")
        val = o["value_as_string"].astype("string").fillna("").str.strip()
        hit = (txt.str.contains(lx.CONVULSION_OBS, na=False) & ~val.str.match(lx.NEGATIVE_VALUE, na=False)).to_numpy(bool)
        if hit.any():
            s = o[hit]
            frames.append(_frame(s["person_id"].astype("int64").reset_index(drop=True), "hx", "convulsion", 1.0, np.nan,
                                 t[hit].reset_index(drop=True), t[hit].reset_index(drop=True), pd.NaT, "charted",
                                 d_only[hit].to_numpy(bool), None, s["observation_source_value"].reset_index(drop=True)))
    return pd.concat(frames, ignore_index=True) if frames else empty_events()


# -------------------------------------------------------------------------------- label-family diagnosis codes
UNKNOWN_TIME = pd.Timestamp("1900-01-01")


def label_dx_items() -> tuple[str, ...]:
    """Event items with ICD condition codes that are anchor items of the silver labels E1-E7 (``configs/anchor_concepts.yaml``
    and ``silver_anchors.yaml``: acute structural dx, arrest, asphyxia). Helper items (for example ESRD) are not anchors."""
    from ..labels.anchors import load_anchor_config
    from ..labels.concepts import ConceptMap
    cm = ConceptMap()
    anchors = set(load_anchor_config()["items"])
    return tuple(sorted(it for it, d in cm.event_items.items() if d.get("condition_codes") and it in anchors))


def label_dx_events(tables: dict, pids: set[int]) -> pd.DataFrame:
    """ICD diagnosis codes of the primary label families (domain ``dxlab``), at ANY time up to what ``as_of`` admits.

    Used only to define the undifferentiated subgroup (no such code recorded before t0): never a model input, exempt from the
    current-encounter restriction. Coding time is unknown (often discharge), so the condition start is used, and the
    conservative direction for the subgroup is taken: a date-only start is the START of its day and a code with no time at
    all counts as recorded long ago, so an ambiguous code can only remove a patient from the subgroup."""
    c = tables.get("omop_condition_occurrence")
    if c is None or not len(c):
        return empty_events()
    from ..labels.concepts import ConceptMap
    cm = ConceptMap()
    items = set(label_dx_items())
    c = _restrict(c, pids)
    src = c["condition_source_value"].astype("string")
    uniq = {u: tuple(i for i in cm.condition_source_hits(u) if i in items) for u in src.dropna().unique()}
    hits = src.map(lambda u: uniq.get(u, ()) if isinstance(u, str) else ())
    keep = hits.map(len).gt(0).to_numpy(bool)
    if not keep.any():
        return empty_events()
    c, hits = c[keep], hits[keep]
    t = schema.parse_datetimes(c["condition_start_datetime"]).astype("datetime64[us]")
    unknown = t.isna()
    t = t.fillna(UNKNOWN_TIME)
    rows = [(i, it) for i, hs in zip(range(len(c)), hits) for it in hs]
    ii = np.array([r[0] for r in rows])
    return _frame(c["person_id"].astype("int64").to_numpy()[ii], "dxlab", np.array([r[1] for r in rows], dtype=object), 1.0, np.nan,
                  t.to_numpy()[ii], t.to_numpy()[ii], pd.NaT, np.where(unknown.to_numpy()[ii], "unknown_time", "coded"), True,
                  None, c["condition_source_value"].to_numpy(object)[ii])


# ----------------------------------------------------------------------------------------------- imaging
def imaging_events(tables: dict, pids: set[int], cfg: BaselineConfig) -> pd.DataFrame:
    im = tables.get("imaging")
    if im is None or im.empty:
        return empty_events()
    im = _restrict(im, pids)
    key = im["modality"].map(lx.imaging_key)
    study = schema.parse_datetimes(im["study_datetime"])
    final = schema.parse_datetimes(im["report_final_datetime"]) if "report_final_datetime" in im else \
        pd.Series(pd.NaT, index=im.index, dtype="datetime64[us]")
    lag = pd.to_timedelta(key.map(lx.IMAGING_LAG_H).astype(float), unit="h")
    if cfg.imaging_time_basis == "study_plus_lag":
        final = pd.Series(pd.NaT, index=im.index, dtype="datetime64[us]")
    use_final = final.notna()
    t_avail = (study + lag).astype("datetime64[us]").where(~use_final, final)
    if cfg.imaging_time_basis == "result":
        t_avail = t_avail.where(use_final, pd.NaT)
    t_event = study.where(study.notna(), final)
    t_avail = t_avail.where(t_avail.isna() | (t_avail >= t_event), t_event)
    return _frame(im["person_id"].astype("int64").reset_index(drop=True), "imaging", key.reset_index(drop=True).to_numpy(object),
                  np.nan, np.nan, t_event.reset_index(drop=True), t_avail.reset_index(drop=True), pd.NaT,
                  np.where(use_final, "result", "study+lag"), (~use_final).to_numpy(bool), None,
                  im["modality"].reset_index(drop=True))


def build_events(tables: dict[str, pd.DataFrame], index: pd.DataFrame, cfg: BaselineConfig | None = None
                 ) -> tuple[pd.DataFrame, dict]:
    """Long event frame for the indexed patients (ALL times after the encounter start, including after t0: the t0 gate is
    ``as_of``). With ``cfg.encounter_scope == "current"`` (default, D-145) rows timed before the person's encounter start are
    dropped here, so a prior encounter can never reach a baseline; ``diag['n_events_prior_encounter_dropped']`` counts them.
    ``index`` must carry ``encounter_start`` (``build_index`` and ``add_encounter_start`` add it)."""
    cfg = cfg or BaselineConfig()
    pids = {int(p) for p in index["person_id"]}
    diag: dict = {}
    parts = [drug_events(tables, pids, cfg), measurement_events(tables, pids, cfg, diag),
             history_events(tables, pids), label_dx_events(tables, pids), imaging_events(tables, pids, cfg)]
    parts = [p for p in parts if len(p)]
    ev = pd.concat(parts, ignore_index=True) if parts else empty_events()
    diag["n_events_prior_encounter_dropped"] = 0
    if cfg.encounter_scope == "current":
        if "encounter_start" not in index:
            raise ValueError("encounter_scope='current' needs index['encounter_start'] (see add_encounter_start)")
        ev, diag["n_events_prior_encounter_dropped"] = restrict_to_current_encounter(
            ev, index.set_index("person_id")["encounter_start"])
    return ev, diag
