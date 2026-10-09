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
