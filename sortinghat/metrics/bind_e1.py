"""Statistics of the imaging-anchored E1 analysis (docs/bind_e1_analysis_plan.md sections 5 to 7). Pure functions on arrays; no I/O.

Everything returned is an AGGREGATE (counts through ``suppress_count`` / ``count_row``-style suppression, n < 11 shown "<11"), except
the private helpers whose names start with an underscore.

Estimands (plan section 5), evaluated on the HELD-OUT per-patient probabilities of the frozen silver-feasibility models (no refit,
never tuned on imaging-E1):

* delta_auroc      AUROC(EEG arm) - AUROC(baseline arm) for the binary imaging label. Calibration-free.
* delta_logloss_shifted
                   mean over scored rows of log loss(EEG arm) - log loss(baseline arm) AFTER an intercept-only logit shift. The shift
                   delta is the maximum-likelihood intercept of  logit P(y = 1) = logit(p_baseline) + delta  on the fitting labels and
                   is applied IDENTICALLY to both arms (the arms differ only by what they learned; a common shift removes the
                   silver-vs-imaging prevalence difference without re-calibrating the EEG arm separately). Cross-fitted by site: the
                   shift for a row of site s is estimated on the rows of the OTHER sites; if any site lacks 11 positives or 11
                   negatives in the stratum (or there is one site) the shift is estimated on the evaluation set itself (all sites
                   pooled) and the block is labelled ``in_sample``.
* delta_logloss_unshifted
                   the same without the shift, reported alongside.

Interval: percentile interval of a within-site patient bootstrap (rows resampled within site, site sizes fixed), 99% primary and 95%
secondary. The shift is RE-ESTIMATED inside every replicate (its uncertainty is part of the interval).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

from ..safe_output import SUPPRESSED, SUPPRESS_BELOW, suppress_count, suppress_proportion
from .loss import DEFAULT_EPS

K = SUPPRESS_BELOW
MIN_POSITIVES = 50                  # stop rule S3: fewer imaging-positive cases in a stratum is "not estimable"
DESCRIPTIVE_BELOW = 200             # estimable but fewer than 200 positives: descriptive only (plan section 5)
MIN_STRATUM_N = 50                  # sedation grade strata (plan section 7 / SAP A1.4 c1): a stratum under 50 patients is not estimable
MIN_VALID_REPS = 0.9                # an interval needs at least this share of valid replicates
MODES = ("cross", "insample", "none")


# ------------------------------------------------------------------------------------------------ primitives
def auroc(y, p) -> float:
    y = np.asarray(y, bool)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(np.asarray(p, float))
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _logit(p, eps=DEFAULT_EPS):
    p = np.clip(np.asarray(p, float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _ll(y, p, eps=DEFAULT_EPS):
    p = np.clip(p, eps, 1.0 - eps)
    return -(y * np.log(p) + (1.0 - y) * np.log1p(-p))


def fit_shift(y, logit0, iters: int = 30, tol: float = 1e-9) -> float:
    """Maximum-likelihood intercept shift ``delta`` of ``logit P(y=1) = logit0 + delta`` (Newton; NaN if one class is empty)."""
    y = np.asarray(y, float)
    n1 = y.sum()
    if n1 == 0 or n1 == len(y):
        return float("nan")
    d = 0.0
    for _ in range(iters):
        p = _sigmoid(logit0 + d)
        h = float((p * (1.0 - p)).sum())
        if h <= 1e-12:
            return float("nan")
        step = float(np.clip((y - p).sum() / h, -2.0, 2.0))
        d += step
        if abs(step) < tol:
            break
    return float(d)


def _ci(reps, alpha: float):
    r = np.asarray(reps, float)
    ok = r[np.isfinite(r)]
    if len(r) == 0 or len(ok) < MIN_VALID_REPS * len(r) or len(ok) < 20:
        return None
    lo, hi = np.percentile(ok, [100 * alpha / 2.0, 100 * (1.0 - alpha / 2.0)])
    return {"lo": round(float(lo), 5), "hi": round(float(hi), 5)}


def _r(x, nd=5):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


# ------------------------------------------------------------------------------------------------ the paired statistic
def shift_mode(y, sites, min_cell: int = K) -> str:
    """'cross' when there are >= 2 sites and every site has >= ``min_cell`` positives and negatives in the stratum, else 'insample'."""
    y = np.asarray(y, bool)
    sites = np.asarray(sites).astype(str)
    u = np.unique(sites)
    if len(u) < 2:
        return "insample"
    ok = all(int((y & (sites == s)).sum()) >= min_cell and int((~y & (sites == s)).sum()) >= min_cell for s in u)
    return "cross" if ok else "insample"


def _row_shift(y, l0, sc, n_sites: int, mode: str) -> np.ndarray:
    if mode == "none":
        return np.zeros(len(y))
    if mode == "insample" or n_sites < 2:
        return np.full(len(y), fit_shift(y, l0))
    out = np.empty(len(y))
    for s in range(n_sites):
        m = sc == s
        out[m] = fit_shift(y[~m], l0[~m])
    return out


def _paired(y, p1, p0, l1, l0, sc, n_sites: int, mode: str, with_ll: bool = True):
    """(auc1, auc0, delta auc, delta log loss shifted, delta log loss unshifted) for one sample."""
    a1, a0 = auroc(y, p1), auroc(y, p0)
    if not with_ll:
        return a1, a0, a1 - a0, float("nan"), float("nan")
    yf = y.astype(float)
    raw = float(np.mean(_ll(yf, _sigmoid(l1)) - _ll(yf, _sigmoid(l0))))
    if mode == "none":
        return a1, a0, a1 - a0, float("nan"), raw
    d = _row_shift(yf, l0, sc, n_sites, mode)
    sh = float(np.mean(_ll(yf, _sigmoid(l1 + d)) - _ll(yf, _sigmoid(l0 + d)))) if np.all(np.isfinite(d)) else float("nan")
    return a1, a0, a1 - a0, sh, raw


def _codes(sites):
    sites = np.asarray(sites).astype(str)
    u = sorted(np.unique(sites))
    return np.searchsorted(u, sites), u


def _boot(y, p1, p0, l1, l0, sc, n_sites: int, mode: str, B: int, seed: int, with_ll: bool) -> np.ndarray:
    groups = [np.flatnonzero(sc == s) for s in range(n_sites)]
    rng = np.random.default_rng(seed)
    out = np.full((B, 5), np.nan)
    for b in range(B):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        out[b] = _paired(y[idx], p1[idx], p0[idx], l1[idx], l0[idx], sc[idx], n_sites, mode, with_ll)
    return out


@dataclass
class Pair:
    """One EEG-arm-versus-baseline-arm comparison on one stratum's scored rows (RECORD-LEVEL arrays; memory only)."""
    y: np.ndarray
    p1: np.ndarray
    p0: np.ndarray
    sites: np.ndarray


def pair_block(pr: Pair, *, B: int, seed: int, site_labels: dict, with_ll: bool = True, mode: str | None = None,
               min_positives: int = MIN_POSITIVES, label: str = "") -> dict:
    """Aggregate block of one comparison: counts, point estimates and 99% / 95% within-site bootstrap intervals of the delta AUROC and
    the delta log loss (shifted and unshifted), per-site point estimates. ``estimable`` is False below ``min_positives`` positives
    (stop rule S3) or with an empty class, and then no estimate is returned."""
    y = np.asarray(pr.y, bool)
    sites = np.asarray(pr.sites).astype(str)
    n, n1 = len(y), int(y.sum())
    out: dict = {"label": label, "n_scored": suppress_count(n), "n_positive": suppress_count(n1),
                 "positives_per_site": _site_counts(y, sites, site_labels),
                 "min_positives": min_positives}
    if n1 < min_positives or (n - n1) < K:
        out.update(estimable=False, why=(f"fewer than {min_positives} imaging-positive cases in the stratum (stop rule S3)"
                                          if n1 < min_positives else "fewer than 11 imaging-negative cases"))
        return out
    sc, u = _codes(sites)
    n_sites = len(u)
    mode = mode or shift_mode(y, sites)
    p1 = np.where(np.isfinite(pr.p1), pr.p1, 0.5).astype(float)
    p0 = np.where(np.isfinite(pr.p0), pr.p0, 0.5).astype(float)
    l1, l0 = _logit(p1), _logit(p0)
    pt = _paired(y, p1, p0, l1, l0, sc, n_sites, mode, with_ll)
    reps = _boot(y, p1, p0, l1, l0, sc, n_sites, mode, B, seed, with_ll)
    out.update(estimable=True, descriptive_only=bool(n1 < DESCRIPTIVE_BELOW), n_boot=B, shift_mode=mode,
               shift_note={"cross": "intercept shift cross-fitted: a row's shift is estimated on the other site(s)' imaging labels",
                           "insample": "intercept shift estimated on the evaluation set itself (in-sample), applied identically to both arms",
                           "none": "no shift"}[mode],
               auroc_eeg_arm={"point": _r(pt[0], 4), "ci_99": _ci(reps[:, 0], 0.01)},
               auroc_baseline_arm={"point": _r(pt[1], 4), "ci_99": _ci(reps[:, 1], 0.01)},
               delta_auroc={"point": _r(pt[2]), "ci_99": _ci(reps[:, 2], 0.01), "ci_95": _ci(reps[:, 2], 0.05)})
    if with_ll:
        out["delta_logloss_shifted"] = {"point": _r(pt[3]), "ci_99": _ci(reps[:, 3], 0.01), "ci_95": _ci(reps[:, 3], 0.05)}
        out["delta_logloss_unshifted"] = {"point": _r(pt[4]), "ci_99": _ci(reps[:, 4], 0.01), "ci_95": _ci(reps[:, 4], 0.05)}
    out["per_site"] = _per_site(y, p1, p0, l1, l0, sites, sc, n_sites, u, mode, site_labels, with_ll)
    return out


def _site_counts(y, sites, site_labels: dict) -> dict:
    """Per-site positive counts, each shown only if the site's positives AND negatives are >= 11; the pooled count is not repeated here."""
    out = {}
    for s in sorted(np.unique(sites)):
        m = sites == s
        n1, n0 = int((y & m).sum()), int((~y & m).sum())
        out[site_labels.get(s, s)] = {"n_positive": SUPPRESSED if min(n1, n0) < K else n1,
                                      "n_negative": SUPPRESSED if min(n1, n0) < K else n0}
    return out


def _per_site(y, p1, p0, l1, l0, sites, sc, n_sites, u, mode, site_labels, with_ll) -> dict:
    d_shift = _row_shift(y.astype(float), l0, sc, n_sites, mode) if with_ll and mode != "none" else None
    out = {}
    for i, s in enumerate(u):
        m = sc == i
        n1, n0 = int((y & m).sum()), int((~y & m).sum())
        if min(n1, n0) < K:
            out[site_labels.get(s, s)] = {"delta_auroc": SUPPRESSED}
            continue
        yy = y[m].astype(float)
        blk = {"n_positive": n1, "delta_auroc": _r(auroc(y[m], p1[m]) - auroc(y[m], p0[m]))}
        if with_ll:
            blk["delta_logloss_unshifted"] = _r(float(np.mean(_ll(yy, _sigmoid(l1[m])) - _ll(yy, _sigmoid(l0[m])))))
            if d_shift is not None and np.isfinite(d_shift[m]).all():
                dd = d_shift[m]
                blk["delta_logloss_shifted"] = _r(float(np.mean(_ll(yy, _sigmoid(l1[m] + dd)) - _ll(yy, _sigmoid(l0[m] + dd)))))
        out[site_labels.get(s, s)] = blk
    return out


# ------------------------------------------------------------------------------------------------ verdict (plan section 5)
def verdict(block: dict, *, label_validated: bool | None = None) -> dict:
    """Apply the pre-specified decision rule to a primary-pair block.

    supportive      delta AUROC 99% interval above 0 AND shifted delta log loss 99% interval below 0, AND the point estimates have the
                    expected sign at every site with at least ``MIN_POSITIVES`` positives.
    harm            a point estimate in the wrong direction with an interval excluding 0 (delta AUROC hi < 0 or shifted delta log loss lo > 0).
    not_supportive  anything else that is estimable.
    not_estimable   below the S3 minimum.
    ``label_validated`` None/False: a not-supportive result is read as 'inconclusive, label unvalidated' (plan section 5)."""
    if not block.get("estimable"):
        return {"verdict": "not_estimable", "why": block.get("why")}
    da, dl = block["delta_auroc"], block.get("delta_logloss_shifted") or {}
    auc_ci, ll_ci = da.get("ci_99"), dl.get("ci_99")
    auc_up = bool(auc_ci and auc_ci["lo"] > 0)
    ll_down = bool(ll_ci and ll_ci["hi"] < 0)
    harm_auc = bool(auc_ci and auc_ci["hi"] < 0 and (da["point"] or 0) < 0)
    harm_ll = bool(ll_ci and ll_ci["lo"] > 0 and (dl.get("point") or 0) > 0)
    harm = harm_auc or harm_ll
    big = [s for s, v in block.get("per_site", {}).items() if isinstance(v.get("n_positive"), int) and v["n_positive"] >= MIN_POSITIVES]
    site_ok = all((block["per_site"][s].get("delta_auroc") or 0) > 0 and (block["per_site"][s].get("delta_logloss_shifted") or 0) < 0
                  for s in big)
    if auc_up and ll_down and site_ok:
        v = "supportive"
    elif harm:
        v = "harm"
    else:
        v = "not_supportive"
    out = {"verdict": v, "harm_on": [m for m, f in (("delta_auroc", harm_auc), ("shifted_delta_logloss", harm_ll)) if f],
           "delta_auroc_ci99_above_0": auc_up, "shifted_delta_logloss_ci99_below_0": ll_down,
           "same_sign_at_every_site_with_ge_50_positives": site_ok, "sites_with_ge_50_positives": len(big),
           "descriptive_only_lt_200_positives": bool(block.get("descriptive_only"))}
    if v != "supportive" and not label_validated:
        out["reading"] = "inconclusive, label unvalidated (step 0d not passed or not run)"
    return out


# ------------------------------------------------------------------------------------------------ light version (delta AUROC only)
def auc_pair_block(y, p1, p0, sites, *, B: int, seed: int, site_labels: dict, min_positives: int = MIN_POSITIVES, label: str = "") -> dict:
    return pair_block(Pair(np.asarray(y, bool), np.asarray(p1, float), np.asarray(p0, float), np.asarray(sites).astype(str)),
                      B=B, seed=seed, site_labels=site_labels, with_ll=False, mode="none", min_positives=min_positives, label=label)


def single_auroc_block(y, p, sites, *, B: int, seed: int, site_labels: dict, min_positives: int = MIN_POSITIVES) -> dict:
    """AUROC of one score (EEG-only arm), 99% within-site bootstrap interval. Not estimable below ``min_positives`` positives."""
    y = np.asarray(y, bool)
    sites = np.asarray(sites).astype(str)
    n1 = int(y.sum())
    out = {"n_scored": suppress_count(len(y)), "n_positive": suppress_count(n1)}
    if n1 < min_positives or len(y) - n1 < K:
        out.update(estimable=False, why=f"fewer than {min_positives} imaging-positive cases (S3)")
        return out
    sc, u = _codes(sites)
    p = np.where(np.isfinite(p), p, 0.5).astype(float)
    groups = [np.flatnonzero(sc == s) for s in range(len(u))]
    rng = np.random.default_rng(seed)
    reps = np.full(B, np.nan)
    for b in range(B):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        reps[b] = auroc(y[idx], p[idx])
    out.update(estimable=True, auroc={"point": _r(auroc(y, p), 4), "ci_99": _ci(reps, 0.01)})
    return out


# ------------------------------------------------------------------------------------------------ sedation-intensity grades (SAP A1.4 c1)
GRADES = (0, 1, 2, 3)


def _grade_stats(y, p1, p0, sc, g, used_g, used_cell, n_sites: int):
    """[dAUC at grades 0..3, light, heavy, heavy - light, pooled slope, site-adjusted slope] for one sample (NaN when a class is empty)."""
    def d(m):
        yy = y[m]
        if yy.sum() == 0 or yy.sum() == len(yy):
            return float("nan")
        return auroc(yy, p1[m]) - auroc(yy, p0[m])
    out = np.full(len(GRADES) + 5, np.nan)
    for gg in GRADES:
        out[gg] = d(g == gg)
    out[4], out[5] = d(g <= 1), d(g >= 2)
    out[6] = out[5] - out[4]
    if used_g.sum() >= 3:
        x = np.array([gg for gg in GRADES if used_g[gg]], float)
        yv = out[:4][used_g]
        if np.isfinite(yv).all():
            xc = x - x.mean()
            out[7] = float((xc * (yv - yv.mean())).sum() / (xc * xc).sum())
    xs, ys, ws = [], [], []
    for s in range(n_sites):
        cells = [gg for gg in GRADES if used_cell[s, gg]]
        if len(cells) < 2:
            continue
        v = np.array([d((sc == s) & (g == gg)) for gg in cells])
        if not np.isfinite(v).all():
            continue
        x = np.array(cells, float)
        xs.append(x - x.mean())
        ys.append(v - v.mean())
    if sum(len(x) for x in xs) >= 3:
        xx, yy = np.concatenate(xs), np.concatenate(ys)
        if (xx * xx).sum() > 0:
            out[8] = float((xx * yy).sum() / (xx * xx).sum())
    return out


def graded_delta_auroc(y, p1, p0, sites, grade, *, B: int, seed: int, site_labels: dict, min_n: int = MIN_STRATUM_N) -> dict:
    """Delta AUROC (EEG arm minus baseline arm) within each sedation-intensity grade 0-3, the light (0-1) and heavy (2-3) strata, their
    difference and the trend over grade. A stratum is estimable with >= ``min_n`` patients and >= 11 positives and 11 negatives; a trend
    needs >= 3 estimable grades. Slopes are in delta-AUROC units per grade step; POSITIVE = the gain GROWS with intensity (the direction
    is opposite to the log-loss version of D-154, where negative is a gain). Pooled slope: OLS of the grade-level delta AUROC on grade over
    the estimable grades; site-adjusted slope (primary): the same with site fixed effects, from (site, grade) cells that are estimable
    on their own. Same resamples for every statistic."""
    y = np.asarray(y, bool)
    sites = np.asarray(sites).astype(str)
    g = np.asarray(grade, int)
    p1 = np.where(np.isfinite(p1), p1, 0.5).astype(float)
    p0 = np.where(np.isfinite(p0), p0, 0.5).astype(float)
    sc, u = _codes(sites)
    S = len(u)

    def ok(m):
        n1 = int((y & m).sum())
        return int(m.sum()) >= min_n and n1 >= K and int(m.sum()) - n1 >= K

    used_g = np.array([ok(g == gg) for gg in GRADES])
    used_cell = np.array([[ok((sc == s) & (g == gg)) for gg in GRADES] for s in range(S)])
    light_ok, heavy_ok = ok(g <= 1), ok(g >= 2)
    n_g = {gg: int((g == gg).sum()) for gg in GRADES}
    shown = _grade_counts_shown(n_g)
    out: dict = {"n_total": suppress_count(len(y)), "min_stratum_n": min_n, "grades": {}, "strata": {}}
    if not (used_g.any() or light_ok or heavy_ok):
        out["estimable"] = False
        out["why"] = f"no grade or light/heavy stratum with >= {min_n} patients and >= 11 positives and negatives"
        for gg in GRADES:
            out["grades"][str(gg)] = {"n": shown[gg], "estimable": False}
        return out
    groups = [np.flatnonzero(sc == s) for s in range(S)]
    rng = np.random.default_rng(seed)
    reps = np.full((B, len(GRADES) + 5), np.nan)
    for b in range(B):
        idx = np.concatenate([gi[rng.integers(0, len(gi), len(gi))] for gi in groups])
        reps[b] = _grade_stats(y[idx], p1[idx], p0[idx], sc[idx], g[idx], used_g, used_cell, S)
    pt = _grade_stats(y, p1, p0, sc, g, used_g, used_cell, S)
    out["estimable"] = True
    for gg in GRADES:
        if used_g[gg]:
            out["grades"][str(gg)] = {"n": shown[gg], "estimable": True, "delta_auroc": _r(pt[gg]), "ci_99": _ci(reps[:, gg], 0.01),
                                      "ci_95": _ci(reps[:, gg], 0.05)}
        else:
            out["grades"][str(gg)] = {"n": shown[gg], "estimable": False,
                                      "why": f"fewer than {min_n} patients or fewer than 11 positives / negatives at this grade"}
    n_light, n_heavy = n_g[0] + n_g[1], n_g[2] + n_g[3]
    hide = min(n_light, n_heavy) < K
    for nm, okk, n_, col in (("light", light_ok, n_light, 4), ("heavy", heavy_ok, n_heavy, 5)):
        sn = SUPPRESSED if (hide or n_ < K) else n_
        out["strata"][nm] = ({"n": sn, "estimable": True, "delta_auroc": _r(pt[col]), "ci_99": _ci(reps[:, col], 0.01),
                              "ci_95": _ci(reps[:, col], 0.05)} if okk else {"n": sn, "estimable": False})
    out["difference_heavy_minus_light"] = ({"estimable": True, "delta_auroc": _r(pt[6]), "ci_99": _ci(reps[:, 6], 0.01),
                                            "ci_95": _ci(reps[:, 6], 0.05)} if light_ok and heavy_ok else {"estimable": False})
    trend = {"slope_sign": "positive = the gain grows with sedation intensity"}
    trend["slope_pooled"] = ({"estimable": True, "grades_used": [gg for gg in GRADES if used_g[gg]], "slope": _r(pt[7]),
                              "ci_99": _ci(reps[:, 7], 0.01)} if used_g.sum() >= 3 and np.isfinite(pt[7]) else {"estimable": False})
    trend["slope_site_adjusted_primary"] = ({"estimable": True, "slope": _r(pt[8]), "ci_99": _ci(reps[:, 8], 0.01),
                                             "n_cells": int(used_cell.sum())} if np.isfinite(pt[8]) else {"estimable": False})
    out["trend"] = trend
    out["reading"] = sedation_reading(out)
    return out


def _grade_counts_shown(n_g: dict) -> dict:
    hide = {g: n_g.get(g, 0) < K for g in GRADES}
    for pair in ((0, 1), (2, 3)):
        if any(hide[g] for g in pair):
            for g in pair:
                hide[g] = True
    return {g: (SUPPRESSED if hide[g] else int(n_g[g])) for g in GRADES}


def sedation_reading(blk: dict) -> str:
    """SAP A1.4(c1) reading rules, mapped to delta AUROC (gain = positive): (a) light-stratum 99% interval above 0 and trend interval
    including 0: against a sedation explanation; (b) light interval including 0 and trend above 0: consistent with a sedation
    contribution; (c) both: a gain at every level that also grows with intensity; (d) anything not estimable: inconclusive."""
    light = blk.get("strata", {}).get("light", {})
    tr = blk.get("trend", {}).get("slope_site_adjusted_primary", {})
    if not light.get("estimable") or not tr.get("estimable") or not light.get("ci_99") or not tr.get("ci_99"):
        return "(d) inconclusive: the light stratum or the trend is not estimable"
    light_pos = light["ci_99"]["lo"] > 0
    tr_lo, tr_hi = tr["ci_99"]["lo"], tr["ci_99"]["hi"]
    trend_pos, trend_incl0 = tr_lo > 0, tr_lo <= 0 <= tr_hi
    if light_pos and trend_incl0:
        return "(a) against a sedation explanation: gain at light grades, no trend with intensity"
    if (not light_pos) and trend_pos:
        return "(b) consistent with a sedation contribution: no gain at light grades, gain grows with intensity"
    if light_pos and trend_pos:
        return "(c) gain at every level that also grows with intensity: a sedation contribution cannot be excluded for the part that grows"
    return "(d) inconclusive: neither reading (a) to (c) applies (for example a light-stratum interval including 0 with no trend)"


# ------------------------------------------------------------------------------------------------ concordance with kappa
def concordance_kappa(silver, img) -> dict:
    """Silver E1 against imaging-E1 on rows assessable for both. 2x2 cells and margins are suppressed (< 11); sensitivity, specificity, PPV,
    NPV and Cohen's kappa are shown only when all four cells are >= 11. Imaging is the reference, not the truth."""
    s = np.asarray(silver, float)
    i = np.asarray(img, float)
    ok = np.isfinite(s) & np.isfinite(i)
    sb, ib = s[ok] > 0.5, i[ok] > 0.5
    tp, fp, fn, tn = int((sb & ib).sum()), int((sb & ~ib).sum()), int((~sb & ib).sum()), int((~sb & ~ib).sum())
    n = tp + fp + fn + tn
    out = {"n_both_assessable": suppress_count(n), "silver_pos_img_pos": suppress_count(tp), "silver_pos_img_neg": suppress_count(fp),
           "silver_neg_img_pos": suppress_count(fn), "silver_neg_img_neg": suppress_count(tn)}
    if min(tp, fp, fn, tn) < K:
        out.update(sensitivity=SUPPRESSED, specificity=SUPPRESSED, ppv=SUPPRESSED, npv=SUPPRESSED, kappa=SUPPRESSED,
                   note="a 2x2 cell is < 11: no rate or kappa shown")
        return out
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
    out.update(sensitivity=round(tp / (tp + fn), 4), specificity=round(tn / (tn + fp), 4), ppv=round(tp / (tp + fp), 4),
               npv=round(tn / (tn + fn), 4), kappa=round((po - pe) / (1 - pe), 4) if pe < 1 else None)
    return out


# ------------------------------------------------------------------------------------------------ drift and selection
def assessable_by_tenth(sites, order, assessable, site_labels: dict) -> dict:
    """Within site, cases ordered by ``order`` (t0) and cut into 10 bins of ~10%: the assessable share per bin (suppressed, with the
    complement rule of ``suppress_proportion``). Imaging coverage can change with the data feed (D-153)."""
    sites = np.asarray(sites).astype(str)
    a = np.asarray(assessable, bool)
    o = np.asarray(order)
    out = {}
    for s in sorted(np.unique(sites)):
        ix = np.flatnonzero(sites == s)
        ix = ix[np.argsort(o[ix], kind="stable")]
        bins = np.array_split(ix, 10)
        out[site_labels.get(s, s)] = [suppress_proportion(int(a[b].sum()), len(b)) for b in bins]
    return out


def prevalence_drift(y, m, train, test, ratio: float = 3.0) -> dict:
    """Test versus train prevalence of a label among ASSESSABLE rows (D-153): ratio, flagged when > ``ratio`` or < 1 / ``ratio``. A cell
    < 11 makes the prevalence (and the ratio) 'not estimable'; never reported as stable."""
    y, m = np.asarray(y, bool), np.asarray(m, bool)

    def cell(sel):
        n = int((m & sel).sum())
        p = int((y & m & sel).sum())
        return n, p

    out = {}
    prev = {}
    for nm, sel in (("train", np.asarray(train, bool)), ("test", np.asarray(test, bool))):
        n, p = cell(sel)
        prev[nm] = suppress_proportion(p, n)
        out[nm] = {"n_assessable": suppress_count(n), "n_positive": suppress_count(p), "prevalence": prev[nm]}
    if isinstance(prev["train"], float) and isinstance(prev["test"], float) and prev["train"] > 0:
        r = prev["test"] / prev["train"]
        out.update(ratio_test_over_train=round(r, 3), flag=bool(r > ratio or r < 1.0 / ratio))
    else:
        out.update(ratio_test_over_train="not estimable (a cell < 11)", flag=None)
    return out


AGE_BANDS = ((0, 18, "<18"), (18, 40, "18-39"), (40, 65, "40-64"), (65, 80, "65-79"), (80, 400, ">=80"))


def window_study_share(has_study, sites, age, site_labels: dict) -> dict:
    """Share of the stratum with a brain study in the window, by age band and site (suppressed): the imaged subpopulation versus the
    stratum (plan section 7, 'Selection'). No weighting is applied in the primary."""
    h = np.asarray(has_study, bool)
    sites = np.asarray(sites).astype(str)
    age = np.asarray(age, float)
    out = {"overall": suppress_proportion(int(h.sum()), len(h)), "by_site": {}, "by_age_band": {}}
    for s in sorted(np.unique(sites)):
        m = sites == s
        out["by_site"][site_labels.get(s, s)] = suppress_proportion(int((h & m).sum()), int(m.sum()))
    for lo, hi, nm in AGE_BANDS:
        m = np.isfinite(age) & (age >= lo) & (age < hi)
        out["by_age_band"][nm] = suppress_proportion(int((h & m).sum()), int(m.sum()))
    out["age_unknown"] = suppress_proportion(int((h & ~np.isfinite(age)).sum()), int((~np.isfinite(age)).sum()))
    return out
