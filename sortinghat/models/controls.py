"""Mandatory Study 1 controls as functions (SAP section 10, plan "Splits and analyses").

1. ``leakage_probes``              predict site / recording duration / channel count from EEG features alone.
2. ``sedative_excluded_rerun``     full rerun with sedated patients removed (train and test).
3. ``severity_stratified_delta``   H3 strata from per-patient d_i.
4. ``negative_control_*``          shuffled labels / permuted EEG: Delta ~ 0 expected.
5. ``exposure_accounting``         STUB: which pretrained components saw which sites / datasets.

All return aggregates only (no row-level values).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..metrics.bootstrap import paired_bootstrap_delta
from ..metrics.hypotheses import h3_severity_stratified
from ..safe_output import suppress_count
from .data import ModelData
from .ladder import COMMERCIAL_RUNGS, DEFAULT_RUNGS, LadderConfig, LadderResult, RungSpec, run_ladder

LEAKAGE_AUROC_FLAG = 0.70   # [PLACEHOLDER] "high" threshold; fix before unblinding


# --------------------------------------------------------------------------
# 1. Leakage probes
# --------------------------------------------------------------------------
def _probe_auc(X: pd.DataFrame, target: np.ndarray, multiclass: bool, n_splits: int, seed: int) -> float:
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=1000))
    classes, counts = np.unique(target, return_counts=True)
    if len(classes) < 2 or counts.min() < n_splits:
        return float("nan")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    # imputation / scaling are re-fitted inside every CV training fold by the pipeline
    pr = cross_val_predict(pipe, X.values, target, cv=cv, method="predict_proba")
    if multiclass and len(classes) > 2:
        return float(roc_auc_score(target, pr, multi_class="ovr", average="macro", labels=classes))
    return float(roc_auc_score(target == classes[-1], pr[:, -1]))


def _binarise(x: np.ndarray) -> np.ndarray | None:
    x = np.asarray(x, dtype=float)
    for thr in (np.nanmedian(x), np.nanmin(x)):
        b = (x > thr).astype(int)
        if 0 < b.sum() < len(b):
            return b
    return None


def leakage_probes(eeg: pd.DataFrame, sites, duration_s=None, n_channels=None, threshold: float = LEAKAGE_AUROC_FLAG,
                   n_splits: int = 5, seed: int = 0) -> dict:
    """Cross-validated AUROC for predicting site (macro one-vs-rest), long-vs-short recording (above median) and
    high-vs-low channel count from the EEG feature columns alone. ``flagged`` = AUROC > threshold.

    A probe that is flagged is a warning that the features encode the factor; it fails the control only when
    Delta is also carried by it (see ``delta_concentration_by_site``).
    """
    out = {"threshold": threshold, "probes": {}}
    sites = np.asarray(sites)
    n = len(eeg)
    tasks = {"site": (sites, True)}
    if duration_s is not None:
        b = _binarise(duration_s)
        tasks["duration"] = (b, False)
    if n_channels is not None:
        b = _binarise(n_channels)
        tasks["channel_count"] = (b, False)
    for name, (t, multi) in tasks.items():
        if t is None:
            out["probes"][name] = {"auroc": float("nan"), "flagged": False, "estimable": False, "n": suppress_count(n)}
            continue
        auc = _probe_auc(eeg, np.asarray(t), multi, n_splits, seed)
        ok = not np.isnan(auc)
        out["probes"][name] = {"auroc": auc, "flagged": bool(ok and auc > threshold), "estimable": ok,
                               "n": suppress_count(n)}
    out["any_flagged"] = any(p["flagged"] for p in out["probes"].values())
    return out


def delta_concentration_by_site(rung_result, site_labels: Mapping | None = None, max_share: float = 0.6) -> dict:
    """Share of the pooled improvement contributed by the single largest site (n-weighted site gains).

    ``carried_by_one_site`` = that share exceeds ``max_share`` ([PLACEHOLDER] threshold) with a pooled gain < 0.
    """
    ps = rung_result.per_site
    gains = {s: -v["delta"] * v["n"] for s, v in ps.items() if v["n"] > 0 and not np.isnan(v["delta"])}
    pos = {s: max(g, 0.0) for s, g in gains.items()}
    tot = sum(pos.values())
    if tot <= 0:
        return {"top_share": float("nan"), "carried_by_one_site": False}
    s_top = max(pos, key=pos.get)
    share = pos[s_top] / tot
    return {"top_share": float(share), "carried_by_one_site": bool(share > max_share and rung_result.delta < 0),
            "top_site": (site_labels or {}).get(str(s_top), "site_x")}


def control_failed(probes: dict, rung_result, site_labels: Mapping | None = None) -> bool:
    """SAP 10.6: a site probe that predicts site well while Delta is carried by one site is a failed control."""
    site_flag = probes["probes"].get("site", {}).get("flagged", False)
    return bool(site_flag and delta_concentration_by_site(rung_result, site_labels)["carried_by_one_site"])


def stratified_delta(d, strata, sites, min_n: int = 50, n_boot: int = 500, seed: int = 0) -> dict:
    """Delta within strata of a leakage factor (for example duration tertiles)."""
    r = h3_severity_stratified(d, strata, sites, n_boot=n_boot, seed=seed, min_stratum_n=min_n)
    return {"strata": {k: {**v, "n": suppress_count(v["n"])} for k, v in r.strata.items()},
            "n_favorable": r.n_favorable, "n_strata": r.n_strata}


# --------------------------------------------------------------------------
# 2. Sedative-excluded rerun
# --------------------------------------------------------------------------
def sedative_excluded_rerun(data: ModelData, sedated=None, rungs: Sequence[RungSpec] | None = None,
                            cfg: LadderConfig | None = None, baseline_sets=None, split: str = "loso") -> dict:
    """Rerun the ladder on patients without t0 sedative / opioid exposure (removed from training AND testing).

    ``sedated``: bool (n,) (default ``data.covariates['sedated']``). Returns the LadderResult and counts.
    """
    s = np.asarray(data.covariates["sedated"] if sedated is None else sedated, dtype=bool)
    sub = data.take(np.flatnonzero(~s))
    res = run_ladder(sub, baseline_sets, rungs, cfg, split)
    return {"result": res, "n_excluded": suppress_count(int(s.sum())), "n_kept": suppress_count(int((~s).sum()))}


# --------------------------------------------------------------------------
# 3. Severity-stratified Delta (H3)
# --------------------------------------------------------------------------
def severity_stratified_delta(result: LadderResult, data: ModelData, rung: str, severity=None,
                              baseline: str | None = None, cut_points: Sequence[float] | None = None,
                              n_boot: int = 500, seed: int = 0, min_stratum_n: int = 50) -> dict:
    """H3: Delta within 3 severity strata (point estimate < 0 in >= 2 of 3 = met).

    ``cut_points`` (two values) should be fixed before unblinding; when omitted, tertiles of the evaluated
    patients are used and the output says so (``cut_points_prespecified`` False). Strata are ordered low->high.
    """
    r = result.get(rung, baseline)
    sev = np.asarray(data.covariates["severity"] if severity is None else severity, dtype=float)[result.eval_idx]
    prespec = cut_points is not None
    cuts = np.asarray(cut_points if prespec else np.nanquantile(sev, [1 / 3, 2 / 3]), dtype=float)
    stratum = np.digitize(sev, cuts)
    names = np.array(["low", "mid", "high"])[stratum]
    h3 = h3_severity_stratified(r.d, names, result.eval_sites, n_boot=n_boot, seed=seed, min_stratum_n=min_stratum_n)
    return {"met": h3.met, "n_favorable": h3.n_favorable, "n_strata": h3.n_strata,
            "cut_points_prespecified": prespec,
            "strata": {k: {**v, "n": suppress_count(v["n"])} for k, v in h3.strata.items()}}


# --------------------------------------------------------------------------
# 4. Negative controls
# --------------------------------------------------------------------------
def permute_labels_within_site(data: ModelData, seed: int = 0) -> ModelData:
    """Shuffle label rows (silver and gold, with their masks) among patients within (site, gold-role) groups.

    Roles stay with their rows, so dev/eval structure is unchanged while features and baseline lose any relation
    to the labels. Expected Delta ~ 0 for every rung.
    """
    rng = np.random.default_rng(seed)
    perm = np.arange(data.n)
    keys = pd.Series(list(zip(data.sites.astype(str), data.gold_role)))
    for _, ix in keys.groupby(keys).groups.items():
        ix = np.asarray(list(ix))
        perm[ix] = rng.permutation(ix)
    return data.replace(y_silver=data.y_silver[perm], m_silver=data.m_silver[perm],
                        y_gold=data.y_gold[perm], m_gold=data.m_gold[perm])


def permute_eeg_within_strata(data: ModelData, strata=None, seed: int = 0) -> ModelData:
    """SAP 10.8(a): permute EEG feature rows across patients within site (and optional baseline-score strata)."""
    rng = np.random.default_rng(seed)
    perm = np.arange(data.n)
    s = data.sites.astype(str)
    key = s if strata is None else np.char.add(np.char.add(s, "|"), np.asarray(strata).astype(str))
    for k in np.unique(key):
        ix = np.flatnonzero(key == k)
        perm[ix] = rng.permutation(ix)
    return data.replace(eeg=data.eeg.iloc[perm].reset_index(drop=True))


def negative_control(data: ModelData, kind: str = "labels", rungs: Sequence[RungSpec] | None = None,
                     cfg: LadderConfig | None = None, baseline_sets=None, split: str = "loso", seed: int = 0,
                     strata=None) -> dict:
    """Run the ladder on permuted data. ``spurious_gain`` = 95% within-site CI entirely below 0 (a control failure)."""
    cfg = cfg or LadderConfig()
    pdata = (permute_labels_within_site(data, seed) if kind == "labels"
             else permute_eeg_within_strata(data, strata, seed))
    res = run_ladder(pdata, baseline_sets, rungs, cfg, split)
    out = {"kind": kind, "rungs": {}}
    for (b, rung), r in res.rungs.items():
        if not (r.available and r.has_eeg):
            continue
        ci = r.ci_modes.get("within_site", {})
        out["rungs"][f"{b}/{rung}"] = {"delta": r.delta, "lo": ci.get("lo"), "hi": ci.get("hi"),
                                       "spurious_gain": bool(ci.get("hi", 1.0) < 0)}
    out["any_spurious_gain"] = any(v["spurious_gain"] for v in out["rungs"].values())
    return out


# --------------------------------------------------------------------------
# 5. Pretrained-model exposure accounting (STUB)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PretrainedExposure:
    component: str
    datasets: tuple = ()               # public / institutional datasets known to be in pretraining
    sites_exposed: tuple = ()          # HEEDB site keys known or suspected to be in pretraining
    status: str = "UNVERIFIED"         # "DECLARED" (from the plan / model card) or "UNVERIFIED"
    note: str = ""


def default_exposure_registry() -> list[PretrainedExposure]:
    """STUB. CBraMod: plan states TUEG pretraining (TUSZ/TUAB/TUEV are subsets). MORGOTH: left UNVERIFIED for a human
    to fill from the model card / paper; nothing here asserts its training sources."""
    return [
        PretrainedExposure("cbramod", datasets=("TUEG", "TUSZ", "TUAB", "TUEV"), status="DECLARED",
                           note="research plan: pretrained on TUEG; TUSZ/TUAB/TUEV are subsets (contaminated benchmarks)"),
        PretrainedExposure("morgoth", datasets=(), status="UNVERIFIED",
                           note="fill in training datasets / sites from the model card before the SAP freeze"),
    ]


def exposure_accounting(registry: Sequence[PretrainedExposure], heldout_sites: Sequence[str],
                        site_datasets: Mapping[str, Sequence[str]] | None = None,
                        site_labels: Mapping[str, str] | None = None) -> dict:
    """Per component x held-out site: "exposed" | "not_exposed" | "unknown" (component UNVERIFIED and no overlap found).

    ``site_datasets``: site -> datasets the site's data are part of (usually empty for HEEDB sites).
    """
    site_datasets = site_datasets or {}
    lab = lambda s: (site_labels or {}).get(str(s), str(s))
    table = {}
    for comp in registry:
        row = {}
        for s in heldout_sites:
            overlap = set(comp.datasets) & set(site_datasets.get(s, ()))
            if str(s) in map(str, comp.sites_exposed) or overlap:
                row[lab(s)] = "exposed"
            elif comp.status != "DECLARED":
                row[lab(s)] = "unknown"
            else:
                row[lab(s)] = "not_exposed"
        table[comp.component] = {"status": comp.status, "datasets": list(comp.datasets), "sites": row,
                                 "note": comp.note}
    return table


def exposed_site_delta(rung_result, exposure_row: Mapping[str, str], site_labels: Mapping[str, str]) -> dict:
    """Delta (n-weighted) over held-out sites grouped by exposure status for one component."""
    groups: dict[str, list] = {}
    for s, v in rung_result.per_site.items():
        st = exposure_row.get(site_labels.get(str(s), str(s)), "unknown")
        if v["n"] > 0 and not np.isnan(v["delta"]):
            groups.setdefault(st, []).append((v["delta"], v["n"]))
    out = {}
    for st, items in groups.items():
        n = sum(i[1] for i in items)
        out[st] = {"delta": float(sum(d * k for d, k in items) / n), "n": suppress_count(n), "n_sites": len(items)}
    return out


# --------------------------------------------------------------------------
# Bundle
# --------------------------------------------------------------------------
def run_mandatory_controls(data: ModelData, result: LadderResult, rung: str = "combined",
                           cfg: LadderConfig | None = None, baseline_sets=None, split: str = "loso",
                           severity_cut_points=None, seed: int = 0, registry=None) -> dict:
    """Run every control for one primary rung and return one aggregate dict."""
    cfg = cfg or LadderConfig()
    spec = [r for r in DEFAULT_RUNGS + COMMERCIAL_RUNGS if r.name == rung]
    rr = result.get(rung)
    probes = leakage_probes(data.eeg, data.sites, data.covariates.get("duration_s"), data.covariates.get("n_channels"),
                            seed=seed)
    sed = sedative_excluded_rerun(data, None, spec, cfg, baseline_sets, split)
    sr = sed["result"].get(rung)
    out = {
        "leakage_probes": probes,
        "site_concentration": delta_concentration_by_site(rr, result.site_labels),
        "site_probe_control_failed": control_failed(probes, rr, result.site_labels),
        "sedative_excluded": {"n_excluded": sed["n_excluded"], "n_kept": sed["n_kept"], "delta": sr.delta,
                              "ci_95_within_site": sr.ci_modes.get("within_site"),
                              "all_sites_favorable": sr.all_sites_favorable, "pooled_delta_negative": bool(sr.delta < 0)},
        "severity_stratified_h3": severity_stratified_delta(result, data, rung, cut_points=severity_cut_points, seed=seed),
        "negative_control_labels": negative_control(data, "labels", spec, cfg, baseline_sets, split, seed),
        "negative_control_eeg": negative_control(data, "eeg", spec, cfg, baseline_sets, split, seed),
        "exposure_accounting": exposure_accounting(registry or default_exposure_registry(),
                                                   sorted({str(s) for s in data.sites}), None, result.site_labels),
    }
    return out
