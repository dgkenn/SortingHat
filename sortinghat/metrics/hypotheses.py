"""Hypothesis-specific estimators for H3-H6 (and per-label Delta for H4 / Gate G2).

Every estimator operates on the per-patient difference vector d_i (see loss.per_patient_delta)
so that all intervals come from the same paired, site-aware bootstrap (bootstrap.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .bootstrap import (
    BootstrapResult,
    bootstrap_ci,
    bootstrap_replicates,
    one_sided_p_below_zero,
    percentile_ci,
)
from .loss import DEFAULT_EPS, binary_log_loss

NESTED_WINDOWS_S: tuple[int, ...] = (20, 60, 120, 300, 600)  # 20 s, 1, 2, 5, 10 min


# --------------------------------------------------------------------------
# H3: severity-stratified Delta
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class H3Result:
    strata: dict          # stratum -> {"delta", "lo", "hi", "n", "favorable", "estimable"}
    n_favorable: int
    n_strata: int
    met: bool


def h3_severity_stratified(
    d, strata, sites, n_boot: int = 2000, seed: int = 0, mode: str = "within_site",
    min_favorable: int = 2, min_stratum_n: int = 50, alpha: float = 0.05,
) -> H3Result:
    """Delta within each severity stratum (GCS / FOUR / NESI strata, defined a priori).

    A stratum is *favorable* if its point estimate is < 0. H3 is met if at least
    ``min_favorable`` strata (plan: 2 of 3) are favorable. A stratum with fewer than
    ``min_stratum_n`` patients is not estimable and counts as not favorable. CIs are
    reported for every stratum but do not enter the decision rule.
    """
    d = np.asarray(d, dtype=float)
    strata = np.asarray(strata)
    sites = np.asarray(sites)
    keep = ~np.isnan(d)
    d, strata, sites = d[keep], strata[keep], sites[keep]
    res, nfav = {}, 0
    for s in np.unique(strata):
        m = strata == s
        ds, ss = d[m], sites[m]
        est = ok = False
        if ds.size >= min_stratum_n:
            est = True
            r = bootstrap_ci(lambda idx, ds=ds: ds[idx].mean(), ds.mean(), ss, mode, n_boot, seed, alpha)
            lo, hi, delta = r.lo, r.hi, r.estimate
            ok = bool(delta < 0)
        else:
            lo = hi = delta = float("nan")
        nfav += int(ok)
        res[str(s)] = {"delta": float(delta), "lo": float(lo), "hi": float(hi), "n": int(ds.size),
                       "favorable": ok, "estimable": est}
    return H3Result(res, nfav, len(res), bool(nfav >= min_favorable))


# --------------------------------------------------------------------------
# H5: early-EEG interaction
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class H5Result:
    interaction: BootstrapResult  # Delta_early - Delta_late (or adjusted coefficient); negative supports H5
    delta_early: float
    delta_late: float
    n_early: int
    n_late: int
    adjusted: bool
    met: bool                     # estimate < 0 and 95% CI entirely below 0


def _ols_coef_early(d, early, X):
    """Coefficient on ``early`` from OLS d ~ 1 + early + X."""
    A = np.column_stack([np.ones(len(d)), early.astype(float)] + ([X] if X is not None else []))
    coef, *_ = np.linalg.lstsq(A, d, rcond=None)
    return float(coef[1])


def h5_interaction(
    d, early, sites, covariates=None, n_boot: int = 2000, seed: int = 0,
    mode: str = "within_site", alpha: float = 0.05,
) -> H5Result:
    """Interaction: Delta in the early-EEG subgroup (t0 before head-CT result) minus Delta otherwise.

    H5 predicts the gain is larger (Delta more negative) when t0 precedes the CT result, so the
    interaction estimate is expected to be < 0. With ``covariates`` (n, q), the estimate is the OLS
    coefficient on the early indicator in d ~ 1 + early + covariates (include site dummies and
    severity); otherwise the raw difference in subgroup means. Bootstrap draws in which either
    subgroup is empty are discarded (NaN).
    """
    d = np.asarray(d, dtype=float)
    early = np.asarray(early, dtype=bool)
    sites = np.asarray(sites)
    X = None if covariates is None else np.asarray(covariates, dtype=float)
    keep = ~np.isnan(d)
    d, early, sites = d[keep], early[keep], sites[keep]
    if X is not None:
        X = X[keep]
        if X.ndim == 1:
            X = X[:, None]

    def stat(idx):
        e = early[idx]
        if e.all() or (~e).all():
            return float("nan")
        if X is None:
            return float(d[idx][e].mean() - d[idx][~e].mean())
        return _ols_coef_early(d[idx], e, X[idx])

    full = np.arange(len(d))
    est = stat(full)
    r = bootstrap_ci(stat, est, sites, mode, n_boot, seed, alpha)
    return H5Result(
        interaction=r, delta_early=float(d[early].mean()) if early.any() else float("nan"),
        delta_late=float(d[~early].mean()) if (~early).any() else float("nan"),
        n_early=int(early.sum()), n_late=int((~early).sum()), adjusted=X is not None,
        met=bool(r.estimate < 0 and r.hi < 0),
    )


# --------------------------------------------------------------------------
# H6: ratio Delta(2 min) / Delta(10 min)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class H6Result:
    ratio: float
    lo: float
    hi: float
    n_valid: int          # bootstrap replicates with a usable (negative) 10-min Delta
    n_boot: int
    delta_short: float
    delta_full: float
    met: bool             # point estimate >= threshold with a negative full-window Delta
    threshold: float


def h6_ratio(
    d_short, d_full, sites, threshold: float = 0.70, n_boot: int = 2000, seed: int = 0,
    mode: str = "within_site", alpha: float = 0.05,
) -> H6Result:
    """Ratio Delta(short window) / Delta(10 min window) on the same patients.

    Patients need both d_short and d_full (complete cases). The ratio is only meaningful if the
    10-minute Delta is negative; bootstrap draws where it is >= 0 are discarded and counted in
    ``n_valid``. A low ``n_valid / n_boot`` means the ratio is not interpretable.
    """
    ds = np.asarray(d_short, dtype=float)
    df = np.asarray(d_full, dtype=float)
    sites = np.asarray(sites)
    keep = ~(np.isnan(ds) | np.isnan(df))
    ds, df, sites = ds[keep], df[keep], sites[keep]

    def stat(idx):
        den = df[idx].mean()
        if den >= 0:
            return float("nan")
        return float(ds[idx].mean() / den)

    den0 = df.mean()
    est = float(ds.mean() / den0) if den0 != 0 else float("nan")
    reps = bootstrap_replicates(stat, sites, mode, n_boot, seed)
    lo, hi, nv = percentile_ci(reps, alpha)
    return H6Result(est, lo, hi, nv, n_boot, float(ds.mean()), float(den0),
                    bool(den0 < 0 and est >= threshold), threshold)


def first_window_reaching_fraction(deltas_by_window: dict, full_window: int | None = None, fraction: float = 0.70):
    """Smallest prespecified window whose Delta is >= ``fraction`` of the full-window gain.

    ``deltas_by_window`` maps window seconds -> Delta (negative = gain). Returns the window
    in seconds, or None if no window reaches it or the full-window Delta is not negative.
    """
    wins = sorted(deltas_by_window)
    full = deltas_by_window[wins[-1] if full_window is None else full_window]
    if not full < 0:
        return None
    for w in wins:
        if deltas_by_window[w] / full >= fraction:
            return w
    return None


# --------------------------------------------------------------------------
# H4 / Gate G2 per-label helpers
# --------------------------------------------------------------------------
def per_label_delta_ci(
    y, p_model, p_base, mask, label_names: Sequence[str], sites, n_boot: int = 2000, seed: int = 0,
    mode: str = "within_site", alpha: float = 0.05, eps: float = DEFAULT_EPS,
) -> dict:
    """Per-label Delta_k with percentile CI and a one-sided bootstrap p-value (H0: Delta_k >= 0).

    One resample of patients is shared across labels so label-level results are jointly
    coherent. ``improved`` = CI upper bound < 0 (the SAP's reading of "improves individually").
    """
    y = np.asarray(y, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    sites = np.asarray(sites)
    K = y.shape[1]
    yy = np.where(mask, y, 0.0)
    diff = np.where(
        mask,
        binary_log_loss(yy, np.where(mask, p_model, 0.5), eps) - binary_log_loss(yy, np.where(mask, p_base, 0.5), eps),
        0.0,
    )
    cnt = mask.astype(float)

    def delta_k(idx):
        c = cnt[idx].sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(c > 0, diff[idx].sum(axis=0) / c, np.nan)

    est = delta_k(np.arange(len(y)))
    groups = [np.flatnonzero(sites == s) for s in np.unique(sites)]
    from .bootstrap import resample_indices

    rng = np.random.default_rng(seed)
    reps = np.empty((n_boot, K))
    for b in range(n_boot):
        reps[b] = delta_k(resample_indices(groups, rng, mode))
    out = {}
    for k, name in enumerate(label_names):
        lo, hi, nv = percentile_ci(reps[:, k], alpha)
        out[name] = {"delta": float(est[k]), "lo": lo, "hi": hi, "n": int(cnt[:, k].sum()),
                     "p_one_sided": one_sided_p_below_zero(reps[:, k]), "improved": bool(hi < 0)}
    return out


def pairwise_auroc(y, p, mask, a: int, b: int) -> float:
    """AUROC for telling label a from label b among patients with exactly one of them (both assessable).

    Score = logit(p_a) - logit(p_b); target = y_a. Used for H4's "E6 vs E5 near chance".
    """
    from sklearn.metrics import roc_auc_score
    from .calibration import logit

    y = np.asarray(y, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    p = np.asarray(p, dtype=float)
    m = mask[:, a] & mask[:, b] & (y[:, a] != y[:, b])
    if m.sum() < 2 or len(np.unique(y[m, a])) < 2:
        return float("nan")
    s = logit(p[m, a]) - logit(p[m, b])
    return float(roc_auc_score(y[m, a], s))


def kendall_ranking(observed_delta: dict, predicted_order: Sequence[str]) -> float:
    """Kendall tau-b between the observed identifiability ranking and the preregistered order.

    Identifiability = more negative per-label Delta. ``predicted_order`` lists labels from most to
    least identifiable (H4: E3, E2, E1, E4a, E5, E6). Labels missing from ``observed_delta`` are dropped.
    """
    from scipy.stats import kendalltau

    labs = [l for l in predicted_order if l in observed_delta and not np.isnan(observed_delta[l])]
    if len(labs) < 3:
        return float("nan")
    pred_rank = np.arange(len(labs), dtype=float)
    obs = np.array([observed_delta[l] for l in labs])  # ascending Delta == descending identifiability
    tau = kendalltau(pred_rank, obs).statistic
    return float(tau)
