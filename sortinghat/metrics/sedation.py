"""Sedation-stratified Delta: is the EEG gain a sedation proxy?  (aggregate-only; no model refit)

The headline Delta of a ladder run is the mean of per-patient differences d_i = loss_i(baseline + EEG) - loss_i(baseline) over
the held-out rows (negative = EEG helps). ``stratified_delta`` splits those SAME d_i by a boolean flag (here t0 sedative /
opioid exposure) and re-estimates Delta inside each stratum, with the within-site patient bootstrap of the headline. Nothing
is refitted: the models are the ones that scored the held-out rows, so this reuses the stored predictions (and, in the
driver, the checkpointed fits).

Why this and not only the sedative-excluded rerun: excluding sedated patients from training AND scoring removes most of the
data when sedation is common (the real run keeps about 150 of 2,077), so the rerun is underpowered and also changes the
training population. The stratified view keeps every model as fitted and only asks where, among the held-out patients, the
gain lives.

Bootstrap. One replicate resamples patients WITHIN SITE over all stratum members together (site sizes fixed; the stratum sizes
vary from replicate to replicate), then computes both stratum means and their difference on that same resample. The stratum
intervals and the difference interval therefore come from the same replicates (paired by bootstrap), so the difference
interval carries the covariance between strata (zero in expectation, as the strata are disjoint patients) and the variation
in stratum size. Percentile intervals at 99% (primary, D-095) and 95%.

Suppression (SAFE_OUTPUT, n < 11): a stratum count is shown as "<11" when it is under 11, and BOTH counts are hidden whenever
the smaller one is, so a suppressed count cannot be recovered from the other and the total. A stratum under ``min_stratum_n``
(50) patients is "not estimable": no Delta, no interval. The difference is estimable only when both strata are.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ..safe_output import SUPPRESS_BELOW, SUPPRESSED, suppress_count
from .bootstrap import percentile_ci, resample_indices

MIN_STRATUM_N = 50
STRATA = ("sedated", "non_sedated")


def _ci(reps: np.ndarray, alpha: float) -> dict | None:
    lo, hi, nv = percentile_ci(reps, alpha)
    return None if nv == 0 else {"lo": lo, "hi": hi}


def joint_bootstrap(d, flag, sites, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, 3) array of [mean d | flag, mean d | ~flag, difference] over within-site patient resamples of all rows
    (NaN where a stratum is empty in the replicate)."""
    d = np.asarray(d, float)
    flag = np.asarray(flag, bool)
    sites = np.asarray(sites)
    groups = [np.flatnonzero(sites == s) for s in np.unique(sites)]
    rng = np.random.default_rng(seed)
    out = np.full((n_boot, 3), np.nan)
    for b in range(n_boot):
        idx = resample_indices(groups, rng, "within_site")
        db, fb = d[idx], flag[idx]
        a, c = db[fb], db[~fb]
        ma = a.mean() if a.size else np.nan
        mc = c.mean() if c.size else np.nan
        out[b] = (ma, mc, ma - mc)
    return out


def _site_cells(d, flag, sites, site_labels: dict, min_cell: int) -> dict:
    """Per-site Delta in each stratum. A delta is hidden when its own cell is < ``min_cell``; the counts are both hidden when
    either cell is."""
    out = {}
    for s in sorted(np.unique(sites), key=str):
        m = sites == s
        na, nb = int((m & flag).sum()), int((m & ~flag).sum())
        hide_n = min(na, nb) < min_cell
        cell = {}
        for nm, msk, n in (("sedated", m & flag, na), ("non_sedated", m & ~flag, nb)):
            cell[nm] = {"n": SUPPRESSED if hide_n else n,
                        "delta": SUPPRESSED if n < min_cell else float(d[msk].mean())}
        out[site_labels.get(str(s), str(s))] = cell
    return out


def stratified_delta(d, flag, sites, site_labels: dict | None = None, n_boot: int = 2000, seed: int = 0,
                     alpha_primary: float = 0.01, alpha_secondary: float = 0.05, min_stratum_n: int = MIN_STRATUM_N,
                     min_cell: int = SUPPRESS_BELOW) -> dict:
    """Delta within ``flag`` (sedated) and ``~flag`` (non-sedated) from the per-patient differences ``d`` (NaN rows are
    dropped), the difference of Deltas (sedated minus non-sedated) with paired-by-bootstrap intervals, and per-site Deltas.
    Aggregates only."""
    d = np.asarray(d, float)
    flag = np.asarray(flag, bool)
    sites = np.asarray(sites).astype(str)
    if not (d.shape == flag.shape == sites.shape):
        raise ValueError("d, flag and sites must have the same length")
    keep = ~np.isnan(d)
    d, flag, sites = d[keep], flag[keep], sites[keep]
    site_labels = site_labels or {}
    n_a, n_b = int(flag.sum()), int((~flag).sum())
    hide_n = min(n_a, n_b) < min_cell
    est = {"sedated": n_a >= min_stratum_n, "non_sedated": n_b >= min_stratum_n}
    out: dict = {"n_total": suppress_count(len(d)), "overall_delta": float(d.mean()) if d.size else float("nan"),
                 "min_stratum_n": min_stratum_n, "strata": {}}
    reps = joint_bootstrap(d, flag, sites, n_boot, seed) if any(est.values()) else None
    for k, (nm, msk, n) in enumerate((("sedated", flag, n_a), ("non_sedated", ~flag, n_b))):
        shown = SUPPRESSED if hide_n else n
        if not est[nm]:
            out["strata"][nm] = {"n": shown, "estimable": False,
                                 "not_estimable": f"fewer than {min_stratum_n} patients in the stratum"}
            continue
        dm = d[msk]
        cell = {"n": shown, "estimable": True, "delta": float(dm.mean()), "n_boot": n_boot}
        cell["ci_99_within_site"] = _ci(reps[:, k], alpha_primary)
        cell["ci_95_within_site"] = _ci(reps[:, k], alpha_secondary)
        cell["all_sites_favorable"] = bool(all(float(dm[sites[msk] == s].mean()) < 0 for s in np.unique(sites[msk]))) \
            if all(int((msk & (sites == s)).sum()) >= min_cell for s in np.unique(sites)) else None
        out["strata"][nm] = cell
    ns = out["strata"]["non_sedated"]
    out["non_sedated_gain_persists_99"] = bool(ns["estimable"] and ns["ci_99_within_site"] is not None
                                               and ns["ci_99_within_site"]["hi"] < 0 and ns["delta"] < 0) \
        if ns["estimable"] else None
    if all(est.values()):
        diff = float(d[flag].mean() - d[~flag].mean())
        out["difference_sedated_minus_non_sedated"] = {
            "estimable": True, "delta": diff, "n_boot": n_boot,
            "ci_99_within_site": _ci(reps[:, 2], alpha_primary), "ci_95_within_site": _ci(reps[:, 2], alpha_secondary)}
    else:
        out["difference_sedated_minus_non_sedated"] = {"estimable": False,
                                                       "not_estimable": "needs both strata with at least "
                                                                        f"{min_stratum_n} patients"}
    out["per_site"] = _site_cells(d, flag, sites, site_labels, min_cell)
    return out


# =====================================================================================================================
# Sedation INTENSITY (D-154): an ordinal t0 grade from the Baseline A ``sed__*`` columns, and Delta by grade
# =====================================================================================================================
# Why: the binary flag covers about 93% of the real patients, so the non-sedated stratum is ~120 patients and inconclusive.
# An ordinal grade asks the sharper question: does the EEG gain shrink toward 0 as sedation intensity falls, and does it exist
# at grades 0-1 (none / light)?  The grade uses ONLY the baseline ``sed__*`` columns (EEG-blind and outcome-blind: no silver
# label, no silver e4b hint, no EEG). Specified in docs/silver_feasibility.md ("Sedation intensity") and DECISION_LOG D-154
# BEFORE any intensity-stratified result was seen; the cut points below are not to be re-tuned after a real run.
#
# What the columns support (baselines/features.py): per named ingredient (propofol, midazolam, lorazepam, dexmedetomidine,
# ketamine, fentanyl, hydromorphone, morphine) ``on_t0`` / ``qty_6h`` / ``qty_24h``; per class (sedative, opioid) ``on_t0`` and
# ``n_24h``; ``n_agents_24h``; ``approx_time``.  ``on_t0`` = an administration record started with no stop recorded by t0
# (ongoing), or, with no administration record, an order within 2 h of t0 (order fallback).  That is the best available proxy
# for a continuous infusion.  NOT available: infusion rate/dose of a running drug (its quantity is censored to unknown at t0),
# normalised units, the route (bolus vs infusion) and the identity of barbiturates / diazepam / etomidate (inside the class
# aggregate only).  Hence "high dose" cannot be graded; grade 3 is "propofol running" or "two or more named sedatives running".
NAMED_SEDATIVES = ("propofol", "midazolam", "lorazepam", "dexmedetomidine", "ketamine")   # continuous-infusion capable
NAMED_OPIOIDS = ("fentanyl", "hydromorphone", "morphine")
GRADES = (0, 1, 2, 3)
LIGHT_GRADES, HEAVY_GRADES = (0, 1), (2, 3)
GRADE_SPEC = {
    "version": "v1 (D-154)",
    "source_columns": "Baseline A sed__* columns only",
    "window_h": {"on_t0": 0, "recent": 24},
    "grades": {
        0: "none recorded: no sedative / opioid running at t0 and no recorded administration or order in the prior 24 h",
        1: "recent intermittent exposure only: a recorded sedative / opioid administration in the prior 24 h "
           "(sed__<class>__n_24h, sed__<agent>__qty_6h/24h, sed__n_agents_24h) but nothing running at t0",
        2: "a sedative or opioid running at t0 (sed__sedative__on_t0, sed__opioid__on_t0 or any named sed__<agent>__on_t0): "
           "ongoing administration / infusion proxy",
        3: "anaesthetic-depth proxy: propofol running at t0 (sed__propofol__on_t0) OR two or more of the named "
           "continuous-capable sedatives (propofol, midazolam, lorazepam, dexmedetomidine, ketamine) running at t0",
    },
    "precedence": "the highest grade whose rule is met (3 > 2 > 1 > 0); a missing column counts as 0 (0 = no record)",
    "light": "grades 0-1", "heavy": "grades 2-3",
    "not_available": ["infusion rate or dose of a running drug (quantity of a running infusion is censored at t0)",
                      "route (bolus vs infusion): 'running at t0' is the proxy",
                      "barbiturate / diazepam / etomidate identity (only inside the class aggregate)",
                      "normalised dose units"],
}


def _col(bl, name: str) -> np.ndarray:
    return (pd.to_numeric(bl[name], errors="coerce").fillna(0).to_numpy(float) > 0) if name in bl \
        else np.zeros(len(bl), bool)


def sedation_intensity_grade(bl: pd.DataFrame) -> tuple[np.ndarray | None, dict]:
    """Ordinal sedation-intensity grade 0-3 per row of the baseline frame (see ``GRADE_SPEC``), and an aggregate description
    (which columns were found, the suppressed grade distribution is added by the caller). None when the frame carries no
    ``on_t0`` sedation column at all (the grade cannot be established)."""
    run_cols = [c for c in ("sed__sedative__on_t0", "sed__opioid__on_t0")
                + tuple(f"sed__{a}__on_t0" for a in NAMED_SEDATIVES + NAMED_OPIOIDS) if c in bl]
    seen = [c for c in bl.columns if str(c).startswith("sed__")]
    info = {"spec": GRADE_SPEC, "sed_columns_found": len(seen), "on_t0_columns_found": len(run_cols)}
    if not run_cols:
        info["available"] = False
        info["reason"] = "no sed__*__on_t0 column in the baseline frame: the grade cannot be established"
        return None, info
    n = len(bl)
    running_named_sed = np.column_stack([_col(bl, f"sed__{a}__on_t0") for a in NAMED_SEDATIVES])
    running_any = _col(bl, "sed__sedative__on_t0") | _col(bl, "sed__opioid__on_t0") | running_named_sed.any(1) \
        | np.column_stack([_col(bl, f"sed__{a}__on_t0") for a in NAMED_OPIOIDS]).any(1)
    recent = _col(bl, "sed__sedative__n_24h") | _col(bl, "sed__opioid__n_24h") | _col(bl, "sed__n_agents_24h")
    for a in NAMED_SEDATIVES + NAMED_OPIOIDS:
        recent |= _col(bl, f"sed__{a}__qty_6h") | _col(bl, f"sed__{a}__qty_24h")
    g3 = _col(bl, "sed__propofol__on_t0") | (running_named_sed.sum(1) >= 2)
    grade = np.zeros(n, int)
    grade[recent] = 1
    grade[running_any] = 2
    grade[g3] = 3
    info["available"] = True
    info["has_recent_window_columns"] = any(str(c).endswith(("__n_24h", "__qty_6h", "__qty_24h")) or c == "sed__n_agents_24h"
                                            for c in seen)
    info["order_fallback_share_note"] = "sed__approx_time marks patients whose exposure rests on an order time (no administration record)"
    if "sed__approx_time" in bl:
        info["n_order_time_only_fallback"] = suppress_count(int(_col(bl, "sed__approx_time").sum()))
    return grade, info


def grade_counts_shown(counts: dict, min_cell: int = SUPPRESS_BELOW) -> dict:
    """Grade counts for display. A count under ``min_cell`` is '<11'. Grades are shown in the pairs (0,1) and (2,3), the pairs
    of the light / heavy strata: if either member of a pair is hidden both are, so a hidden count cannot be recovered from the
    pooled stratum count or from the total."""
    hide = {g: counts.get(g, 0) < min_cell for g in GRADES}
    for pair in (LIGHT_GRADES, HEAVY_GRADES):
        if any(hide[g] for g in pair):
            for g in pair:
                hide[g] = True
    return {g: (SUPPRESSED if hide[g] else int(counts[g])) for g in GRADES}


def _slope(x, y):
    xc = x - x.mean()
    den = float((xc * xc).sum())
    return float((xc * y).sum() / den) if den > 0 and len(x) > 1 else float("nan")


def _rep_stats(d, g, s, used, site_ids, trend: bool) -> np.ndarray:
    """[mean d at grades 0..3, light, heavy, heavy - light, pooled slope, site-adjusted slope, Spearman rho] for one sample.
    The trend entries use only the patients in ``used`` grades (those with >= min_stratum_n patients in the observed data)."""
    out = np.full(len(GRADES) + 6, np.nan)
    cnt = np.bincount(g, minlength=len(GRADES))
    sm = np.bincount(g, weights=d, minlength=len(GRADES))
    with np.errstate(invalid="ignore", divide="ignore"):
        out[:4] = np.where(cnt > 0, sm / np.maximum(cnt, 1), np.nan)
    if cnt[:2].sum() > 0:
        out[4] = sm[:2].sum() / cnt[:2].sum()
    if cnt[2:].sum() > 0:
        out[5] = sm[2:].sum() / cnt[2:].sum()
    out[6] = out[5] - out[4]
    if trend:
        m = used[g]
        if m.sum() > 2 and len(np.unique(g[m])) >= 2:
            x, y, ss = g[m].astype(float), d[m], s[m]
            out[7] = _slope(x, y)
            sxy = sxx = 0.0
            for sid in site_ids:                                  # site fixed effects: demean grade and d within site
                k = ss == sid
                if k.sum() > 1:
                    xc, yc = x[k] - x[k].mean(), y[k] - y[k].mean()
                    sxy += float((xc * yc).sum())
                    sxx += float((xc * xc).sum())
            out[8] = sxy / sxx if sxx > 0 else float("nan")
            rx, ry = rankdata(x), rankdata(y)
            if rx.std() > 0 and ry.std() > 0:
                out[9] = float(np.corrcoef(rx, ry)[0, 1])
    return out


def graded_bootstrap(d, grade, sites, used, n_boot: int, seed: int, trend: bool) -> np.ndarray:
    """(n_boot, 10) replicates of ``_rep_stats`` over within-site patient resamples of all rows together (the stratum
    Deltas, the light / heavy Deltas, their difference and the trend statistics come from the SAME resamples)."""
    groups = [np.flatnonzero(sites == s) for s in np.unique(sites)]
    site_ids = list(np.unique(sites))
    rng = np.random.default_rng(seed)
    out = np.full((n_boot, len(GRADES) + 6), np.nan)
    for b in range(n_boot):
        idx = resample_indices(groups, rng, "within_site")
        out[b] = _rep_stats(d[idx], grade[idx], sites[idx], used, site_ids, trend)
    return out


def _est(delta, reps, alpha_primary, alpha_secondary, n_boot, extra=None) -> dict:
    o = {"estimable": True, "delta": float(delta), "n_boot": n_boot,
         "ci_99_within_site": _ci(reps, alpha_primary), "ci_95_within_site": _ci(reps, alpha_secondary)}
    return {**o, **(extra or {})}


def graded_delta(d, grade, sites, site_labels: dict | None = None, n_boot: int = 2000, seed: int = 0,
                 alpha_primary: float = 0.01, alpha_secondary: float = 0.05, min_stratum_n: int = MIN_STRATUM_N,
                 min_cell: int = SUPPRESS_BELOW) -> dict:
    """Delta within each sedation-intensity grade (0-3) and within the pooled light (0-1) and heavy (2-3) strata, from the
    per-patient differences ``d`` (NaN rows are dropped); the heavy-minus-light difference; a trend test of Delta over grade;
    per-site Deltas.  Aggregates only.

    Trend (pre-specified): three statistics on the patients of the grades with >= ``min_stratum_n`` patients, needing at least
    3 such grades, all from the same bootstrap replicates as the stratum Deltas.  Slopes are in Delta units per grade step
    (negative = the gain grows with intensity).  PRIMARY = the site-adjusted slope (OLS of d on grade with site fixed effects,
    i.e. grade and d demeaned within site; the pooled slope can ride on a site difference in grade mix and Delta).  Also the
    pooled OLS slope and Spearman's rho between grade and d (rho > 0: d rises = gain shrinks with intensity).  Intervals: 99%
    (primary) and 95% percentile intervals of the within-site patient bootstrap.
    """
    d = np.asarray(d, float)
    grade = np.asarray(grade, int)
    sites = np.asarray(sites).astype(str)
    if not (d.shape == grade.shape == sites.shape):
        raise ValueError("d, grade and sites must have the same length")
    if grade.size and (grade.min() < 0 or grade.max() > 3):
        raise ValueError("grade must be in 0..3")
    keep = ~np.isnan(d)
    d, grade, sites = d[keep], grade[keep], sites[keep]
    site_labels = site_labels or {}
    n_g = {g: int((grade == g).sum()) for g in GRADES}
    n_light, n_heavy = n_g[0] + n_g[1], n_g[2] + n_g[3]
    est = {g: n_g[g] >= min_stratum_n for g in GRADES}
    light_ok, heavy_ok = n_light >= min_stratum_n, n_heavy >= min_stratum_n
    used = np.array([est[g] for g in GRADES])
    trend_ok = int(used.sum()) >= 3
    shown = grade_counts_shown(n_g, min_cell)
    pool_hidden = min(n_light, n_heavy) < min_cell
    out: dict = {"n_total": suppress_count(len(d)), "overall_delta": float(d.mean()) if d.size else float("nan"),
                 "min_stratum_n": min_stratum_n, "grades": {}, "strata": {}}
    reps = graded_bootstrap(d, grade, sites, used, n_boot, seed, trend_ok) if (any(est.values()) or light_ok or heavy_ok) else None
    nb = n_boot
    for g in GRADES:
        if not est[g]:
            out["grades"][str(g)] = {"n": shown[g], "estimable": False,
                                     "not_estimable": f"fewer than {min_stratum_n} patients at this grade"}
        else:
            out["grades"][str(g)] = _est(d[grade == g].mean(), reps[:, g], alpha_primary, alpha_secondary, nb, {"n": shown[g]})
    for nm, msk, n_, ok, col in (("light", grade <= 1, n_light, light_ok, 4), ("heavy", grade >= 2, n_heavy, heavy_ok, 5)):
        shown_n = SUPPRESSED if (pool_hidden or n_ < min_cell) else n_
        if not ok:
            out["strata"][nm] = {"n": shown_n, "estimable": False,
                                 "not_estimable": f"fewer than {min_stratum_n} patients in the stratum"}
        else:
            out["strata"][nm] = _est(d[msk].mean(), reps[:, col], alpha_primary, alpha_secondary, nb, {"n": shown_n})
    ls = out["strata"]["light"]
    out["light_gain_persists_99"] = bool(ls["ci_99_within_site"] is not None and ls["ci_99_within_site"]["hi"] < 0
                                         and ls["delta"] < 0) if ls["estimable"] else None
    if light_ok and heavy_ok:
        diff = float(d[grade >= 2].mean() - d[grade <= 1].mean())
        out["difference_heavy_minus_light"] = _est(diff, reps[:, 6], alpha_primary, alpha_secondary, nb)
    else:
        out["difference_heavy_minus_light"] = {"estimable": False,
                                               "not_estimable": f"needs light and heavy strata with at least {min_stratum_n} patients"}
    if trend_ok:
        o = _rep_stats(d, grade, sites, used, list(np.unique(sites)), True)
        used_g = [g for g in GRADES if used[g]]
        out["trend"] = {
            "estimable": True, "grades_used": used_g, "n_boot": nb,
            "slope_site_adjusted_primary": _est(o[8], reps[:, 8], alpha_primary, alpha_secondary, nb),
            "slope_pooled": _est(o[7], reps[:, 7], alpha_primary, alpha_secondary, nb),
            "spearman_rho": _est(o[9], reps[:, 9], alpha_primary, alpha_secondary, nb)}
        sl = out["trend"]["slope_site_adjusted_primary"]["ci_99_within_site"]
        out["trend"]["gain_grows_with_intensity_99"] = bool(sl is not None and sl["hi"] < 0)
        out["trend"]["slope_ci_includes_0_99"] = bool(sl is not None and sl["lo"] < 0 < sl["hi"])
    else:
        out["trend"] = {"estimable": False, "grades_used": [g for g in GRADES if used[g]],
                        "not_estimable": f"needs at least 3 grades with {min_stratum_n} or more patients"}
    # per-site Deltas by grade; counts in the pairs (0,1)/(2,3), a Delta only when its own cell has >= min_cell patients
    per_site = {}
    for sid in sorted(np.unique(sites), key=str):
        m = sites == sid
        cn = {g: int((m & (grade == g)).sum()) for g in GRADES}
        sh = grade_counts_shown(cn, min_cell)
        per_site[site_labels.get(str(sid), str(sid))] = {
            str(g): {"n": sh[g], "delta": SUPPRESSED if cn[g] < min_cell else float(d[m & (grade == g)].mean())} for g in GRADES}
    out["per_site"] = per_site
    return out
