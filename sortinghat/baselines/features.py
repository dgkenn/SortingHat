"""t0-masked clinical feature builder for Baselines A-D.

One wide matrix (one row per patient, stable column names, raw NaN preserved) plus a provenance table that
says which baseline each column belongs to. Baselines are nested column subsets: A subset B subset C subset D.

    fs = build_feature_set(tables)                 # tables in the HEEDB layout (synthetic or real)
    XA = fs.matrix("A")                            # raw, NaN where unobserved (+ explicit __miss columns)
    imp = MedianImputer().fit(fs.matrix("C").loc[train_ids]); Xc = imp.transform(fs.matrix("C"))

Leak control: all events go through ``asof.as_of`` once; extractors see only its output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import lexicon as lx
from .asof import as_of, assert_masked
from .config import BaselineConfig
from .events import build_events, build_index

BASELINES = ("A", "B", "C", "D")
PROVENANCE_COLUMNS = ["feature", "baseline", "group", "role", "source_table", "source_field", "time_basis",
                      "window", "imputation", "approximate_time_possible", "description"]


# ------------------------------------------------------------------------------------------ registry
class _Registry:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, feature, baseline, group, role, source_table, source_field, time_basis, window, imputation,
            description, approx=False):
        self.rows.append(dict(feature=feature, baseline=baseline, group=group, role=role,
                              source_table=source_table, source_field=source_field, time_basis=time_basis,
                              window=window, imputation=imputation, approximate_time_possible=approx,
                              description=description))

    def frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.rows, columns=PROVENANCE_COLUMNS)
        assert df["feature"].is_unique, "duplicate feature names"
        return df


def _value_block(reg, base, group, key, label, src_table, src_field, time_basis, window, approx, with_age=False):
    f = f"{group}__{key}"
    reg.add(f"{f}__value", base, group, "value", src_table, src_field, time_basis, window, "median(train)",
            f"{label}: latest value in window; NaN if none", approx)
    reg.add(f"{f}__miss", base, group, "missing_indicator", src_table, src_field, time_basis, window, "none",
            f"1 if no {label} value in window", approx)
    if with_age:
        reg.add(f"{f}__age_h", base, group, "age_h", src_table, src_field, time_basis, window, "median(train)",
                f"hours from the {label} measurement to t0", approx)


def build_registry(cfg: BaselineConfig) -> pd.DataFrame:
    r = _Registry()
    # ------------------------------------------------------------------ Baseline A
    r.add("demo__age_years", "A", "demographics", "value", "reports_findings / eeg_metadata", "AgeAtVisit",
          "static", "index EEG", "median(train)", "Age at the index EEG (years)")
    r.add("demo__age_years__miss", "A", "demographics", "missing_indicator", "reports_findings / eeg_metadata",
          "AgeAtVisit", "static", "index EEG", "none", "1 if age missing")
    r.add("demo__sex_male", "A", "demographics", "value", "reports_findings / eeg_metadata", "SexDSC", "static",
          "index EEG", "median(train)", "1 male, 0 female, NaN unknown")
    r.add("demo__sex_male__miss", "A", "demographics", "missing_indicator", "reports_findings / eeg_metadata",
          "SexDSC", "static", "index EEG", "none", "1 if sex missing")
    for k in lx.SCORE_KEYS:
        _value_block(r, "A", "score", k, k.upper(), "omop_measurement", "measurement_source_value, value_as_number",
                     "charted", f"[t0-{cfg.score_window_h:g}h, t0]", False, with_age=True)
    w6, w24 = (f"{w:g}h" for w in cfg.drug_windows_h)
    for ing in lx.NAMED_INGREDIENTS:
        cls = lx.DRUG_CLASS[ing]
        src = ("omop_drug_exposure", "drug_source_value, drug_exposure_start/end_datetime, quantity")
        r.add(f"sed__{ing}__on_t0", "A", "sedation_opioid", "flag", *src, "admin|order", "at t0", "none (0 = no record)",
              f"{ing} ({cls}) running at t0 (administration record: started, no stop by t0; order fallback: "
              f"ordered within {cfg.order_active_h:g}h)", True)
        for w in (w6, w24):
            r.add(f"sed__{ing}__qty_{w}", "A", "sedation_opioid", "dose", *src, "admin|order", f"prior {w}",
                  "none (0 = no record)", f"{ing}: recorded quantity, completed administrations pro-rated into "
                  "the window; running infusions contribute 0 (total unknown at t0). Units not normalized", True)
    for cls in (lx.SEDATIVE, lx.OPIOID):
        r.add(f"sed__{cls}__on_t0", "A", "sedation_opioid", "flag", "omop_drug_exposure", "drug_source_value",
              "admin|order", "at t0", "none (0 = no record)", f"any {cls} ingredient running at t0", True)
        r.add(f"sed__{cls}__n_{w24}", "A", "sedation_opioid", "count", "omop_drug_exposure", "drug_source_value",
              "admin|order", f"prior {w24}", "none (0 = no record)", f"{cls} administrations/orders started in window", True)
    r.add("sed__n_agents_24h", "A", "sedation_opioid", "count", "omop_drug_exposure", "drug_source_value",
          "admin|order", f"prior {w24}", "none (0 = no record)", "distinct sedative/opioid ingredients active or started", True)
    r.add("sed__approx_time", "A", "sedation_opioid", "approx_flag", "omop_drug_exposure",
          "drug_exposure_end_datetime", "admin|order", f"prior {w24}", "none",
          "1 if any contributing drug row used the order-time fallback (no administration record)", True)
    # ------------------------------------------------------------------ Baseline B
    for k in lx.VITAL_KEYS:
        _value_block(r, "B", "vital", k, k.upper(), "omop_measurement", "measurement_source_value, value_as_number",
                     "charted", f"[t0-{cfg.vital_window_h:g}h, t0]", False)
    for k in lx.PUPIL_KEYS:
        _value_block(r, "B", "pupil", k, k, "omop_measurement", "measurement_source_value, value_as_number",
                     "charted", f"[t0-{cfg.vital_window_h:g}h, t0]", False)
    for f, desc in (("pupil__any_nonreactive", "1 if either pupil charted non-reactive (ASSUMED coding 0 = non-reactive)"),
                    ("pupil__size_asymmetry_mm", "|left - right| pupil size (mm)")):
        r.add(f, "B", "pupil", "value", "omop_measurement", "derived from pupil rows", "charted",
              f"[t0-{cfg.vital_window_h:g}h, t0]", "median(train)", desc)
        r.add(f + "__miss", "B", "pupil", "missing_indicator", "omop_measurement", "derived from pupil rows",
              "charted", f"[t0-{cfg.vital_window_h:g}h, t0]", "none", "1 if not computable")
    _value_block(r, "B", "poc_glucose", "poc_glucose", "POC glucose", "omop_measurement",
                 "measurement_source_value, value_as_number", "charted", f"[t0-{cfg.poc_glucose_window_h:g}h, t0]",
                 False, with_age=True)
    hx_src = ("omop_condition_occurrence / procedure_occurrence / observation / measurement",
              "ICD source value, CPR procedure code, witnessed-convulsion observation")
    for k, d in (("arrest", "cardiac arrest"), ("trauma", "trauma"), ("head_trauma", "head trauma")):
        r.add(f"hx__{k}_any", "B", "history", "flag", *hx_src, "coded|charted", "any time <= t0", "none (0 = no record)",
              f"{d} documented at any time up to t0 (ICD coding time unknown: start date used)", True)
        r.add(f"hx__{k}_recent", "B", "history", "flag", *hx_src, "coded|charted", f"[t0-{cfg.hx_recent_h:g}h, t0]",
              "none (0 = no record)", f"{d} documented in the {cfg.hx_recent_h:g}h before t0", True)
    r.add("hx__convulsion_recent", "B", "history", "flag", *hx_src, "charted", f"[t0-{cfg.hx_recent_h:g}h, t0]",
          "none (0 = no record)", "witnessed convulsion/seizure documented before t0", True)
    # ------------------------------------------------------------------ Baseline C
    lab_win = f"[t0-{cfg.lab_lookback_h:g}h, t0]"
    lab_keys = list(lx.LAB_KEYS) + [f"x_{n.replace(' ', '_')}" for n in cfg.extra_labs]
    for k in lab_keys:
        _value_block(r, "C", "lab", k, k, "omop_measurement", "measurement_source_value, value_as_number",
                     "result|collect+lag", lab_win, True)
    for k in lx.TOX_KEYS:
        _value_block(r, "C", "tox", k, k, "omop_measurement", "measurement_source_value, value_as_number",
                     "result|collect+lag", lab_win, True)
    for k in lx.CULTURE_KEYS:
        r.add(f"culture__{k}__resulted", "C", "culture", "flag", "omop_measurement", "measurement_source_value",
              "result|collect+lag", lab_win, "none (0 = no result)", f"{k}: a result is available by t0", True)
        r.add(f"culture__{k}__positive", "C", "culture", "value", "omop_measurement", "value_as_number",
              "result|collect+lag", lab_win, "median(train)", f"{k}: latest result positive (1) or negative (0)", True)
        r.add(f"culture__{k}__miss", "C", "culture", "missing_indicator", "omop_measurement", "value_as_number",
              "result|collect+lag", lab_win, "none", f"1 if no {k} result is available by t0", True)
    for k in lx.IMAGING_KEYS:
        r.add(f"img__{k}__final_by_t0", "C", "imaging", "flag", "imaging", "modality, report_final_datetime",
              "result|study+lag", "any time <= t0", "none (0 = no final report by t0)",
              f"{k}: final report available by t0 (availability = final-read time, else study + lag)", True)
        r.add(f"img__{k}__n_by_t0", "C", "imaging", "count", "imaging", "modality, report_final_datetime",
              "result|study+lag", "any time <= t0", "none (0 = none)", f"{k}: number of reports available by t0", True)
    r.add("lab__n_results_by_t0", "C", "lab", "count", "omop_measurement", "measurement_source_value",
          "result|collect+lag", lab_win, "none", "number of lab/tox/culture results available in window", True)
    r.add("lab__approx_time", "C", "lab", "approx_flag", "omop_measurement", "measurement_datetime",
          "result|collect+lag", lab_win, "none",
          "1 if any lab/tox/culture result used collection time + assay lag instead of a result time", True)
    r.add("img__approx_time", "C", "imaging", "approx_flag", "imaging", "report_final_datetime", "result|study+lag",
          "any time <= t0", "none", "1 if any imaging report used study time + read lag instead of the final time", True)
    # ------------------------------------------------------------------ Baseline D
    for lev in lx.INDICATION_LEVELS:
        r.add(f"ind__{lev}", "D", "indication", "flag", "eeg_metadata", "ReferralIndication", "static (pre-EEG order)",
              "index EEG", "none (explicit 'missing' level)", f"EEG referral indication category = {lev}")
    return r.frame()


# ---------------------------------------------------------------------------------------- extractors
def _latest(df: pd.DataFrame, key: str, window_h: float, order: str = "t_event") -> pd.Series:
    """Per person, the value of the latest row of ``key`` inside [t0-window_h, t0] (NaN rows ignored)."""
    s = df[(df["key"] == key) & df["value"].notna() & (df["hours_since_event"] <= window_h)]
    if s.empty:
        return pd.Series(dtype=float)
    return s.sort_values([order, "t_avail"], kind="stable").groupby("person_id")["value"].last()


def _latest_with_age(df, key, window_h):
    s = df[(df["key"] == key) & df["value"].notna() & (df["hours_since_event"] <= window_h)]
    if s.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    last = s.sort_values(["t_event", "t_avail"], kind="stable").groupby("person_id").last()
    return last["value"], last["hours_since_event"]


def _put(X: pd.DataFrame, col: str, s: pd.Series) -> None:
    if len(s):
        X.loc[s.index.intersection(X.index), col] = s.reindex(s.index.intersection(X.index)).to_numpy()


def _value_cols(X, group, key, value, age=None):
    f = f"{group}__{key}"
    _put(X, f"{f}__value", value)
    X[f"{f}__miss"] = X[f"{f}__value"].isna().astype(float)
    if age is not None:
        _put(X, f"{f}__age_h", age)


def _score_vital_pupil(X, m, cfg):
    sc = m[m["domain"] == "score"]
    for k in lx.SCORE_KEYS:
        v, a = _latest_with_age(sc, k, cfg.score_window_h)
        _value_cols(X, "score", k, v, a)
    vi = m[m["domain"] == "vital"]
    for k in lx.VITAL_KEYS:
        _value_cols(X, "vital", k, _latest(vi, k, cfg.vital_window_h))
    pu = m[m["domain"] == "pupil"]
    for k in lx.PUPIL_KEYS:
        _value_cols(X, "pupil", k, _latest(pu, k, cfg.vital_window_h))
    react = X[["pupil__pupil_react_l__value", "pupil__pupil_react_r__value"]]
    nonreact = (react == 0).any(axis=1).astype(float).where(react.notna().any(axis=1))
    X["pupil__any_nonreactive"] = nonreact
    X["pupil__any_nonreactive__miss"] = nonreact.isna().astype(float)
    asym = (X["pupil__pupil_size_l__value"] - X["pupil__pupil_size_r__value"]).abs()
    X["pupil__size_asymmetry_mm"] = asym
    X["pupil__size_asymmetry_mm__miss"] = asym.isna().astype(float)
    pg = m[m["domain"] == "poc_glucose"]
    v, a = _latest_with_age(pg, "poc_glucose", cfg.poc_glucose_window_h)
    _value_cols(X, "poc_glucose", "poc_glucose", v, a)


def _history(X, m, cfg):
    h = m[m["domain"] == "hx"]
    for k in ("arrest", "trauma", "head_trauma"):
        s = h[h["key"] == k]
        X.loc[X.index.intersection(s["person_id"].unique()), f"hx__{k}_any"] = 1.0
        r = s[s["hours_since_event"] <= cfg.hx_recent_h]
        X.loc[X.index.intersection(r["person_id"].unique()), f"hx__{k}_recent"] = 1.0
    c = h[(h["key"] == "convulsion") & (h["hours_since_event"] <= cfg.hx_recent_h)]
    X.loc[X.index.intersection(c["person_id"].unique()), "hx__convulsion_recent"] = 1.0


def _ingredient_windows(d: pd.DataFrame, w: float) -> pd.Series:
    """Per-row recorded quantity attributable to [t0-w, t0], using only information known at t0."""
    start_h = d["hours_since_event"].to_numpy(float)                 # hours before t0 the row started (>= 0)
    q = d["quantity"].to_numpy(float)
    end_h = ((d["t0"] - d["t_end"]).dt.total_seconds() / 3600.0).to_numpy(float)   # NaN when no ended interval
    out = np.zeros(len(d))
    has_iv = ~np.isnan(end_h) & (end_h < start_h)
    # completed administration with duration: pro-rate by overlap with the window
    lo, hi = np.maximum(end_h, 0.0), np.minimum(start_h, w)
    frac = np.where(has_iv, np.clip(hi - lo, 0, None) / np.where(has_iv, start_h - end_h, 1.0), 0.0)
    out = np.where(has_iv, q * frac, np.where(start_h <= w, q, 0.0))
    return pd.Series(np.nan_to_num(out, nan=0.0), index=d.index)


def _sedation(X, m, cfg):
    d = m[m["domain"] == "drug"].copy()
    if d.empty:
        return
    w6, w24 = cfg.drug_windows_h
    d["cls"] = d["key"].map(lx.DRUG_CLASS)
    d["active"] = ((d["time_basis"] == "admin") & d["ongoing"]) | \
                  ((d["time_basis"] == "order") & (d["hours_since_event"] <= cfg.order_active_h))
    d["q6"], d["q24"] = _ingredient_windows(d, w6), _ingredient_windows(d, w24)
    d["in24"] = d["active"] | (d["hours_since_event"] <= w24)
    for ing in lx.NAMED_INGREDIENTS:
        s = d[d["key"] == ing]
        if s.empty:
            continue
        g = s.groupby("person_id")
        _put(X, f"sed__{ing}__on_t0", g["active"].any().astype(float))
        _put(X, f"sed__{ing}__qty_{w6:g}h", g["q6"].sum())
        _put(X, f"sed__{ing}__qty_{w24:g}h", g["q24"].sum())
    for cls in (lx.SEDATIVE, lx.OPIOID):
        s = d[d["cls"] == cls]
        if s.empty:
            continue
        _put(X, f"sed__{cls}__on_t0", s.groupby("person_id")["active"].any().astype(float))
        _put(X, f"sed__{cls}__n_{w24:g}h", s[s["hours_since_event"] <= w24].groupby("person_id").size().astype(float))
    recent = d[d["in24"]]
    _put(X, "sed__n_agents_24h", recent.groupby("person_id")["key"].nunique().astype(float))
    _put(X, "sed__approx_time", recent[recent["time_basis"] == "order"].groupby("person_id").size().gt(0).astype(float))


def _labs(X, m, cfg, lab_keys):
    lw = cfg.lab_lookback_h
    lab = m[m["domain"] == "lab"]
    for k in lab_keys:
        _value_cols(X, "lab", k, _latest(lab, k, lw, order="t_avail"))
    tox = m[m["domain"] == "tox"]
    for k in lx.TOX_KEYS:
        _value_cols(X, "tox", k, _latest(tox, k, lw, order="t_avail"))
    cu = m[m["domain"] == "culture"]
    for k in lx.CULTURE_KEYS:
        s = cu[(cu["key"] == k) & (cu["hours_since_event"] <= lw)]
        if s.empty:
            X[f"culture__{k}__miss"] = 1.0
            continue
        X.loc[X.index.intersection(s["person_id"].unique()), f"culture__{k}__resulted"] = 1.0
        pos = s[s["value"].notna()].sort_values(["t_avail", "t_event"], kind="stable").groupby("person_id")["value"].last()
        _put(X, f"culture__{k}__positive", (pos > 0).astype(float))
        X[f"culture__{k}__miss"] = X[f"culture__{k}__positive"].isna().astype(float)
    inw = m[m["domain"].isin(["lab", "tox", "culture"]) & (m["hours_since_event"] <= lw)]
    _put(X, "lab__n_results_by_t0", inw.groupby("person_id").size().astype(float))
    _put(X, "lab__approx_time", inw[inw["approx"]].groupby("person_id").size().gt(0).astype(float))


def _imaging(X, m):
    im = m[m["domain"] == "imaging"]
    for k in lx.IMAGING_KEYS:
        s = im[im["key"] == k]
        if s.empty:
            continue
        n = s.groupby("person_id").size().astype(float)
        _put(X, f"img__{k}__n_by_t0", n)
        _put(X, f"img__{k}__final_by_t0", (n > 0).astype(float))
    _put(X, "img__approx_time", im[im["approx"]].groupby("person_id").size().gt(0).astype(float))


def _static(X, index):
    X["demo__age_years"] = index["age_years"].to_numpy(float)
    X["demo__age_years__miss"] = index["age_years"].isna().to_numpy(float)
    X["demo__sex_male"] = index["sex_male"].to_numpy(float)
    X["demo__sex_male__miss"] = index["sex_male"].isna().to_numpy(float)
    cat = index["indication_raw"].map(lx.indication_category)
    for lev in lx.INDICATION_LEVELS:
        X[f"ind__{lev}"] = (cat == lev).to_numpy(float)


# -------------------------------------------------------------------------------------- orchestrator
ZERO_FILL_ROLES = {"flag", "dose", "count", "approx_flag"}


@dataclass
class FeatureSet:
    X: pd.DataFrame                    # index person_id; columns = provenance["feature"], in registry order
    index: pd.DataFrame                # person_id, SiteID, SessionID, t0 (NOT features; never feed to a model)
    provenance: pd.DataFrame
    config: BaselineConfig
    diagnostics: dict = field(default_factory=dict)

    def columns(self, baseline: str) -> list[str]:
        if baseline not in BASELINES:
            raise ValueError(f"baseline must be one of {BASELINES}")
        ok = BASELINES[:BASELINES.index(baseline) + 1]
        return self.provenance.loc[self.provenance["baseline"].isin(ok), "feature"].tolist()

    def matrix(self, baseline: str) -> pd.DataFrame:
        """Raw (un-imputed) feature matrix for a baseline (columns = A, then B, ... in provenance order)."""
        return self.X[self.columns(baseline)]

    def to_long(self, baseline: str = "D") -> pd.DataFrame:
        """Tidy long form: person_id, feature, value (plus the provenance baseline)."""
        long = self.matrix(baseline).rename_axis("person_id").reset_index().melt(
            id_vars="person_id", var_name="feature", value_name="value")
        return long.merge(self.provenance[["feature", "baseline", "group", "role"]], on="feature", how="left")


def build_feature_set(tables: dict[str, pd.DataFrame], cfg: BaselineConfig | None = None,
                      index: pd.DataFrame | None = None) -> FeatureSet:
    cfg = cfg or BaselineConfig()
    index = build_index(tables) if index is None else index
    prov = build_registry(cfg)
    events, diag = build_events(tables, index, cfg)
    masked = assert_masked(as_of(events, index.set_index("person_id")["t0"]))     # THE t0 gate
    X = pd.DataFrame(np.nan, index=pd.Index(index["person_id"].to_numpy(), name="person_id"), columns=prov["feature"])
    zero_cols = prov.loc[prov["role"].isin(ZERO_FILL_ROLES), "feature"]
    X[zero_cols] = 0.0
    X.loc[:, [f"culture__{k}__miss" for k in lx.CULTURE_KEYS]] = 1.0
    lab_keys = [c[len("lab__"):-len("__value")] for c in prov["feature"] if c.startswith("lab__") and c.endswith("__value")]
    _static(X, index)
    _score_vital_pupil(X, masked, cfg)
    _history(X, masked, cfg)
    _sedation(X, masked, cfg)
    _labs(X, masked, cfg, lab_keys)
    _imaging(X, masked)
    X = X[prov["feature"].tolist()].copy()          # defragment
    diag.update(n_patients=int(len(X)), n_events_total=int(len(events)), n_events_by_t0=int(len(masked)),
                n_events_dropped_post_t0=int(len(events) - len(masked)))
    return FeatureSet(X=X, index=index[["person_id", "SiteID", "SessionID", "t0"]].reset_index(drop=True),
                      provenance=prov, config=cfg, diagnostics=diag)
