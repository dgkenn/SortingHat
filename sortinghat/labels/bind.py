"""BIND imaging-report findings -> imaging-anchored structural-injury label (imaging-E1) per EEG case.

BIND = Brain Imaging and Neurophysiology Dataset (BDSP, doi:10.60508/mby8-3a26; Sci Data 13:1176, 2026,
doi:10.1038/s41597-026-07421-x). Plan: ``docs/bind_e1_analysis_plan.md`` (D-158). Config: ``configs/bind_e1.yaml``.

PROVENANCE (CLAUDE.md schema convention)
* Table locations and COLUMN NAMES are CONFIRMED by a names-only header read of the credentialed access point on
  2026-10-14: ``Imaging/{I0001,I0004}/Clinical/{site}_{imaging_findings_De-id,reports,demographics}.csv``.
* The VALUE vocabularies (what ``Acute/Chronic``, ``Label_LLAMA_Level1`` ... contain) are ASSUMED: no value of any BIND column
  has been seen. The regexes in the YAML implement fixed clinical definitions and are re-checked against a values-only
  vocabulary listing (plan step 0c) before the first label is linked to an EEG case.

EEG-BLIND BY CONSTRUCTION. Nothing here reads an EEG feature, EEG report or EEG-derived field. ``Report_txt`` (free text) is
never loaded; ``read_bind_table`` refuses any request that would load it.

TIMING. The imaging date is the date-shifted STUDY DATE only (no time of day is known to exist), so the window is in calendar
days: ``offset = study_date - date(t0)``. ``t0`` is the case's EEG start (D-124), in the same shifted time base IF the BIND and
HEEDB date shifts agree, which ``shift_agreement`` checks (plan step 0b) before any label is built.

AGGREGATE-ONLY. ``build_label_table`` returns a RECORD-LEVEL frame (one row per EEG case): keep it in memory or under
``local_only/``; print only ``summarise`` / ``linkage_qc`` / ``shift_agreement`` / ``concordance`` output.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import yaml

from ..safe_output import suppress_count, suppress_proportion

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "bind_e1.yaml"

REASON_POSITIVE = "positive"
REASON_NEGATIVE = "negative"
REASON_NO_STUDY = "no_brain_study_in_window"
REASON_INDETERMINATE = "indeterminate_acuity"


# ------------------------------------------------------------------------------------------------ config
@dataclass(frozen=True)
class BindConfig:
    raw: dict
    findings_columns: dict[str, str]
    never_load: frozenset[str]
    window_days: tuple[int, int]
    primary_classes: tuple[str, ...]
    secondary_classes: tuple[str, ...]
    classes: dict[str, dict[str, str]]
    truthy: frozenset[str]
    falsy: frozenset[str]
    acute_re: re.Pattern
    chronic_re: re.Pattern
    negation_re: re.Pattern
    match_columns: tuple[str, ...]
    indeterminate_masks: bool
    report_type_re: re.Pattern | None
    from_findings_brain: bool
    status: str = "ASSUMED_VALUES"
    _cls_re: dict = field(default_factory=dict, compare=False, repr=False)

    def class_names(self, include_secondary: bool = True) -> tuple[str, ...]:
        return self.primary_classes + (self.secondary_classes if include_secondary else ())


def load_config(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> BindConfig:
    with open(path or CONFIG_PATH, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw = {**raw, **(overrides or {})}
    classes = raw["classes"]
    prim, sec = tuple(raw["primary_classes"]), tuple(raw["secondary_classes"])
    unknown = [c for c in prim + sec if c not in classes]
    if unknown:
        raise ValueError(f"classes without a definition: {unknown}")
    w = tuple(int(x) for x in raw["window_days"])
    if len(w) != 2 or w[0] > w[1]:
        raise ValueError(f"window_days must be [lo, hi] with lo <= hi, got {raw['window_days']}")
    sa = raw.get("session_assessable") or {}
    cfg = BindConfig(
        raw=raw, findings_columns=dict(raw["findings_columns"]), never_load=frozenset(raw["never_load"]), window_days=w,
        primary_classes=prim, secondary_classes=sec, classes=classes,
        truthy=frozenset(s.lower() for s in raw["truthy"]), falsy=frozenset(s.lower() for s in raw["falsy"]),
        acute_re=re.compile(raw["acute_regex"], re.I), chronic_re=re.compile(raw["chronic_regex"], re.I),
        negation_re=re.compile(raw["negation_regex"], re.I), match_columns=tuple(raw["match_columns"]),
        indeterminate_masks=bool(raw.get("indeterminate_masks_label", True)),
        report_type_re=re.compile(sa["report_type_regex"], re.I) if sa.get("report_type_regex") else None,
        from_findings_brain=bool(sa.get("from_findings_brain", True)), status=str(raw.get("status", "ASSUMED_VALUES")))
    for c, d in classes.items():
        cfg._cls_re[c] = (re.compile(d["include"], re.I), re.compile(d["exclude"], re.I) if d.get("exclude") else None)
    return cfg


# ------------------------------------------------------------------------------------------------ S3 reading (human-run)
def bind_key(kind: str, site: str, cfg: BindConfig) -> str:
    if site not in cfg.raw["sites"]:
        raise ValueError(f"unknown BIND site {site!r}")
    return cfg.raw["prefix"].format(site=site) + cfg.raw["tables"][kind].format(site=site)


def read_bind_table(kind: str, site: str, *, cfg: BindConfig | None = None, s3=None, profile: str | None = None) -> pd.DataFrame:
    """One site-level BIND clinical table as text. ``kind`` is findings | reports | demographics.

    Human-run only (``data_io.make_client`` refuses inside an agent session). Loads only the columns the loader uses; the free
    text ``Report_txt`` is on the never-load list and the reports table is read as (person, session, Type) only.
    """
    from .. import data_io
    cfg = cfg or load_config()
    s3 = s3 or data_io.make_client(profile)
    key = bind_key(kind, site, cfg)
    header = data_io.csv_header(s3, key)
    if kind == "findings":
        want = list(cfg.findings_columns.values())
    elif kind == "reports":
        want = list(cfg.raw["reports_columns"].values())
    else:
        want = list(cfg.raw["demographics_columns"][site].values())
    use = [c for c in header if c in want and c not in cfg.never_load]
    missing = [c for c in want if c not in header]
    if missing:
        raise KeyError(f"{key}: expected columns not in the header: {missing} (names only)")
    assert not (set(use) & cfg.never_load)
    body = data_io.read_object(s3, data_io.access_point(), key)
    import io
    return pd.read_csv(io.BytesIO(body), dtype=str, usecols=use, encoding="utf-8-sig", low_memory=False)


# ------------------------------------------------------------------------------------------------ normalisation
def norm_id(x: Any) -> str | None:
    """Patient/session id as a canonical string: numeric ids lose '.0' and leading zeros; blank -> None."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "none"):
        return None
    try:
        return str(int(float(s)))
    except ValueError:
        return s


def _bmap(series: pd.Series, fn) -> pd.Series:
    """Elementwise bool test as a real bool Series (also for an empty or string-typed input)."""
    return pd.Series([bool(fn(t)) for t in series], index=series.index, dtype=bool)


def _truth(series: pd.Series, cfg: BindConfig) -> pd.Series:
    """True / False / <NA> from a Yes/No-style text column (values ASSUMED)."""
    s = series.fillna("").astype(str).str.strip().str.lower()
    out = pd.Series(pd.NA, index=series.index, dtype="boolean")
    out[s.isin(cfg.truthy)] = True
    out[s.isin(cfg.falsy)] = False
    return out


def normalise_findings(df: pd.DataFrame, cfg: BindConfig) -> pd.DataFrame:
    """Rename the real findings header to canonical names. Free-text report columns must not be present."""
    bad = set(df.columns) & cfg.never_load
    if bad:
        raise ValueError(f"never-load column present: {sorted(bad)}")
    inv = {v: k for k, v in cfg.findings_columns.items()}
    missing = [v for v in ("BDSPPatientID", "Session_id") if v not in df.columns]
    if missing:
        raise KeyError(f"findings table lacks {missing}")
    out = df.rename(columns={c: inv[c] for c in df.columns if c in inv}).copy()
    out["person"] = out["person"].map(norm_id)
    out["session"] = out["session"].map(norm_id)
    if "study_date" in out:
        out["study_date"] = pd.to_datetime(out["study_date"], errors="coerce")
    return out


def normalise_sessions(df: pd.DataFrame, site: str, cfg: BindConfig, *, reports: pd.DataFrame | None = None) -> pd.DataFrame:
    """Session catalogue (person, session, study_date, dob) from the demographics table; optional report Type."""
    m = cfg.raw["demographics_columns"][site]
    inv = {v: k for k, v in m.items()}
    out = df.rename(columns=inv)[list(m)].copy()
    out["person"], out["session"] = out["person"].map(norm_id), out["session"].map(norm_id)
    out["study_date"] = pd.to_datetime(out["study_date"], errors="coerce")
    out["dob"] = pd.to_datetime(out["dob"], errors="coerce")
    out["report_brain"] = False
    if reports is not None and cfg.report_type_re is not None:
        rm = cfg.raw["reports_columns"]
        r = reports.rename(columns={rm["person"]: "person", rm["session"]: "session", rm["type"]: "type"})
        r["person"], r["session"] = r["person"].map(norm_id), r["session"].map(norm_id)
        hit = r[r["type"].fillna("").astype(str).map(lambda t: bool(cfg.report_type_re.search(t)))]
        keys = set(zip(hit["person"], hit["session"]))
        out["report_brain"] = [(p, s) in keys for p, s in zip(out["person"], out["session"])]
    return out.drop_duplicates(["person", "session"])


# ------------------------------------------------------------------------------------------------ finding classification
def classify_findings(f: pd.DataFrame, cfg: BindConfig) -> pd.DataFrame:
    """Per finding row: ``is_brain``, ``acuity`` (acute | chronic | unknown) and, per class, ``pos_<c>`` (class concept AND acute)
    and ``ind_<c>`` (class concept, acuity unknown) and ``chr_<c>`` (class concept, chronic: the negative-control outcome). A chronic qualifier wins over an acute one. Null findings and negated
    mentions never match. Value vocabularies are ASSUMED (see module docstring)."""
    f = f.copy()
    n = len(f)
    brain = pd.Series(pd.NA, index=f.index, dtype="boolean")
    for col in ("brain_related", "brain"):
        if col in f:
            t = _truth(f[col], cfg)
            brain = brain.where(brain.notna(), t)
    f["is_brain"] = brain.fillna(False).astype(bool)
    null = (_truth(f["nullfinding"], cfg).fillna(False).astype(bool) if "nullfinding" in f
            else pd.Series(False, index=f.index, dtype=bool))
    ac = f["acuity"].fillna("").astype(str) if "acuity" in f else pd.Series("", index=f.index)
    chronic = _bmap(ac, cfg.chronic_re.search)
    acute = _bmap(ac, cfg.acute_re.search) & ~chronic
    f["acuity_state"] = pd.Series(np.where(chronic, "chronic", np.where(acute, "acute", "unknown")), index=f.index, dtype=object)
    parts = [f[c].fillna("").astype(str).tolist() for c in cfg.match_columns if c in f]
    text = pd.Series([" | ".join(vals) for vals in zip(*parts)] if parts else [""] * n, index=f.index, dtype=object)
    negated = _bmap(text, cfg.negation_re.search)
    for c in cfg.class_names():
        inc, exc = cfg._cls_re[c]
        m = _bmap(text, lambda t: inc.search(t) and not (exc is not None and exc.search(t)))
        m = m & ~negated & ~null & f["is_brain"]
        f[f"pos_{c}"] = (m & (f["acuity_state"] == "acute")).astype(bool)
        f[f"ind_{c}"] = (m & (f["acuity_state"] == "unknown")).astype(bool)
        f[f"chr_{c}"] = (m & (f["acuity_state"] == "chronic")).astype(bool)      # negative-control outcome (plan, controls)
    return f


# ------------------------------------------------------------------------------------------------ label table
def build_label_table(cases: pd.DataFrame, sessions: pd.DataFrame, findings: pd.DataFrame, cfg: BindConfig | None = None,
                      window_days: tuple[int, int] | None = None, *, include_secondary_in_primary: bool = False) -> pd.DataFrame:
    """One row per EEG case.

    ``cases``: ``case_id``, ``person_id`` (= int(BDSPPatientID)), ``t0`` (datetime). ``sessions``: ``normalise_sessions`` output;
    ``findings``: ``classify_findings(normalise_findings(...))``. Returns ``case_id``, ``E1_img`` (boolean: True positive, False
    assessable negative, <NA> masked), ``ctrl_chronic`` (negative-control outcome), ``reason``, the class flags (``cls_<c>``) and ``n_brain_studies`` (count of brain studies in
    the window). A positive needs a brain-related finding of a primary class with an ACUTE/subacute qualifier in a study dated
    within the window; a negative needs >= 1 brain study in the window and no positive and (with ``indeterminate_masks_label``) no
    class mention of unknown acuity. Window = calendar days, imaging date minus t0 date, inclusive.
    """
    cfg = cfg or load_config()
    lo, hi = window_days or cfg.window_days
    prim = cfg.class_names(include_secondary=False) if not include_secondary_in_primary else cfg.class_names()
    cl = cfg.class_names()
    c = cases[["case_id", "person_id", "t0"]].copy()
    c["person"] = c["person_id"].map(norm_id)
    c["t0_day"] = pd.to_datetime(c["t0"]).dt.normalize()

    # a study is a BRAIN study when any of its findings rows is brain-related (or its report Type says so)
    fb = findings.groupby(["person", "session"], dropna=True).agg(
        brain=("is_brain", "max"), **{f"{p}_{k}": (f"{p}_{k}", "max") for k in cl for p in ("pos", "ind", "chr")})
    fb = fb.reset_index()
    s = sessions[["person", "session", "study_date", "report_brain"]].copy()
    if "study_date" in findings and findings["study_date"].notna().any():
        fd = findings.dropna(subset=["study_date"]).drop_duplicates(["person", "session"])[["person", "session", "study_date"]]
        s = pd.concat([s, fd.assign(report_brain=False)], ignore_index=True)
        s["study_date"] = pd.to_datetime(s["study_date"])
        s = s.sort_values("study_date").drop_duplicates(["person", "session"])
    s = s.merge(fb, on=["person", "session"], how="left")
    for col in ["brain"] + [f"{p}_{k}" for p in ("pos", "ind", "chr") for k in cl]:
        s[col] = s[col].fillna(False).astype(bool)
    s["brain_study"] = (s["brain"] & cfg.from_findings_brain) | s["report_brain"].astype(bool)

    m = c.merge(s, on="person", how="left")
    m["offset"] = (m["study_date"].dt.normalize() - m["t0_day"]).dt.days
    m = m[m["offset"].between(lo, hi) & m["brain_study"]]
    g = m.groupby("case_id")
    agg = g.agg(n_brain_studies=("session", "nunique"),
                **{f"{p}_{k}": (f"{p}_{k}", "max") for k in cl for p in ("pos", "ind", "chr")})
    out = c[["case_id"]].merge(agg, left_on="case_id", right_index=True, how="left")
    out["n_brain_studies"] = out["n_brain_studies"].fillna(0).astype(int)
    for col in [f"{p}_{k}" for p in ("pos", "ind", "chr") for k in cl]:
        out[col] = out[col].fillna(False).astype(bool)
    pos = out[[f"pos_{k}" for k in prim]].any(axis=1)
    ind = out[[f"ind_{k}" for k in prim]].any(axis=1)
    e1 = pd.Series(pd.NA, index=out.index, dtype="boolean")
    reason = pd.Series(REASON_NO_STUDY, index=out.index, dtype=object)
    has = out["n_brain_studies"] > 0
    neg_ok = has & ~pos & (~ind if cfg.indeterminate_masks else True)
    e1[pos] = True
    reason[pos] = REASON_POSITIVE
    e1[neg_ok] = False
    reason[neg_ok] = REASON_NEGATIVE
    reason[has & ~pos & ind & cfg.indeterminate_masks] = REASON_INDETERMINATE
    out["E1_img"], out["reason"] = e1, reason
    # negative-control outcome: a CHRONIC lesion of a primary class on a brain study in the window, with no acute positive
    ctrl = pd.Series(pd.NA, index=out.index, dtype="boolean")
    ctrl[has] = out.loc[has, [f"chr_{k}" for k in prim]].any(axis=1) & ~pos[has]
    out["ctrl_chronic"] = ctrl
    for k in cl:
        out[f"cls_{k}"] = out[f"pos_{k}"]
    return out[["case_id", "E1_img", "ctrl_chronic", "reason", "n_brain_studies"] + [f"cls_{k}" for k in cl]]


# ------------------------------------------------------------------------------------------------ aggregate-only reporting
def summarise(lab: pd.DataFrame, cfg: BindConfig | None = None) -> dict:
    """Suppressed counts: cases, assessable, imaging-E1 positives, per-class positives, reasons. Never record-level."""
    cfg = cfg or load_config()
    n = len(lab)
    assess = int(lab["E1_img"].notna().sum())
    pos = int((lab["E1_img"] == True).sum())            # noqa: E712
    return {
        "cases": suppress_count(n), "assessable": suppress_count(assess), "positive": suppress_count(pos),
        "prevalence_among_assessable": suppress_proportion(pos, assess),
        "classes": {k: suppress_count(int(lab[f"cls_{k}"].sum())) for k in cfg.class_names()},
        "reasons": {r: suppress_count(int((lab["reason"] == r).sum()))
                    for r in (REASON_POSITIVE, REASON_NEGATIVE, REASON_NO_STUDY, REASON_INDETERMINATE)},
    }


def linkage_qc(imaging_people: Iterable[Any], cohort_people: Iterable[Any]) -> dict:
    """Plan step 0a: overlap of the imaging patient ids with the analysed EEG cohort (suppressed counts only)."""
    img = {norm_id(x) for x in imaging_people} - {None}
    coh = {norm_id(x) for x in cohort_people} - {None}
    k = len(img & coh)
    return {"cohort_people": suppress_count(len(coh)), "with_any_imaging": suppress_count(k),
            "share_with_imaging": suppress_proportion(k, len(coh))}


def shift_agreement(bind_dob: pd.Series, ref_dob: pd.Series) -> dict:
    """Plan step 0b: do BIND and HEEDB share one date shift? Both series are indexed by patient id and hold the shifted date of
    birth (HEEDB side: OMOP person birth_datetime). Reports the share of linked patients whose DOB dates agree exactly and
    within one day (suppressed)."""
    b = bind_dob.copy(); b.index = [norm_id(i) for i in b.index]
    r = ref_dob.copy(); r.index = [norm_id(i) for i in r.index]
    b = pd.to_datetime(b, errors="coerce").dt.normalize(); r = pd.to_datetime(r, errors="coerce").dt.normalize()
    b = b[~b.index.duplicated()].dropna(); r = r[~r.index.duplicated()].dropna()
    both = b.index.intersection(r.index)
    d = (b.loc[both] - r.loc[both]).dt.days.abs()
    return {"linked_with_dob": suppress_count(len(both)),
            "exact_agreement": suppress_proportion(int((d == 0).sum()), len(both)),
            "within_1_day": suppress_proportion(int((d <= 1).sum()), len(both))}


def concordance(silver: pd.Series, img: pd.Series) -> dict:
    """Silver E1 vs imaging-E1 on rows assessable for both: suppressed 2x2 cells and (only if every cell and margin is >= 11)
    sensitivity, specificity, PPV, NPV of silver against imaging. Imaging is the reference here, not the truth."""
    ok = silver.notna() & img.notna()
    s, i = silver[ok].astype(bool), img[ok].astype(bool)
    tp, fp, fn, tn = int((s & i).sum()), int((s & ~i).sum()), int((~s & i).sum()), int((~s & ~i).sum())
    return {"n_both_assessable": suppress_count(int(ok.sum())),
            "silver_pos_img_pos": suppress_count(tp), "silver_pos_img_neg": suppress_count(fp),
            "silver_neg_img_pos": suppress_count(fn), "silver_neg_img_neg": suppress_count(tn),
            "sensitivity": suppress_proportion(tp, tp + fn), "specificity": suppress_proportion(tn, tn + fp),
            "ppv": suppress_proportion(tp, tp + fp), "npv": suppress_proportion(tn, tn + fn)}
