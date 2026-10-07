"""Patient-level paired bootstrap that respects site structure.

Pairing: the unit is the patient. The per-patient difference d_i = loss_i(model) -
loss_i(baseline) carries both predictions for the same patient, so resampling
patients (rows) preserves the model-baseline correlation. Never resample the two
models' predictions independently.

Site structure (three modes, all reported):

``within_site``  Resample patients with replacement *inside* each site, holding every
    site's size fixed (stratified bootstrap). The CI reflects patient sampling
    variance conditional on the observed sites. It is the narrowest interval and
    cannot see between-site variance in Delta.

``cluster``      Resample *sites* with replacement and keep every patient of a drawn
    site. Reflects between-site variance (inference to a population of sites) but
    ignores within-site sampling variance in the sense that a drawn site is
    reproduced exactly. With S sites there are only C(2S-1, S) distinct draws
    (10 for S = 3, and 1/9 of draws for S = 3 contain one site only), so the interval
    is coarse and its percentile coverage is unreliable. Report it, do not lean on it.

``two_stage``    Resample sites, then resample patients within each drawn site.
    Captures both sources of variance; the most conservative of the three.

Why report both: the plan's critique is that "a within-site bootstrap hides between-site
variance". The per-site favorable check (loss.favorable_at_every_site) is the guard
against between-site heterogeneity that the within-site CI cannot see.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import stats

MODES = ("within_site", "cluster", "two_stage")


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    lo: float
    hi: float
    se: float
    n_boot: int
    n_valid: int
    mode: str
    alpha: float = 0.05

    @property
    def excludes_zero_below(self) -> bool:
        return bool(self.hi < 0)

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("estimate", "lo", "hi", "se", "n_boot", "n_valid", "mode", "alpha")}


def _site_groups(sites) -> list[np.ndarray]:
    sites = np.asarray(sites)
    return [np.flatnonzero(sites == s) for s in np.unique(sites)]


def resample_indices(groups: Sequence[np.ndarray], rng: np.random.Generator, mode: str) -> np.ndarray:
    """One bootstrap draw of row indices under the given mode."""
    if mode == "within_site":
        return np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
    S = len(groups)
    drawn = rng.integers(0, S, S)
    if mode == "cluster":
        return np.concatenate([groups[j] for j in drawn])
    if mode == "two_stage":
        parts = []
        for j in drawn:
            g = groups[j]
            parts.append(g[rng.integers(0, len(g), len(g))])
        return np.concatenate(parts)
    raise ValueError(f"mode must be one of {MODES}, got {mode!r}")


def bootstrap_replicates(
    stat_fn: Callable[[np.ndarray], float],
    sites,
    mode: str = "within_site",
    n_boot: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    """Replicates of ``stat_fn(idx)``; ``idx`` indexes the caller's arrays. NaN allowed."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    groups = _site_groups(sites)
    if mode != "within_site" and len(groups) < 2:
        raise ValueError(f"mode {mode!r} needs at least 2 sites")
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        out[b] = stat_fn(resample_indices(groups, rng, mode))
    return out


def percentile_ci(reps, alpha: float = 0.05) -> tuple[float, float, int]:
    r = np.asarray(reps, dtype=float)
    r = r[~np.isnan(r)]
    if r.size == 0:
        return float("nan"), float("nan"), 0
    lo, hi = np.quantile(r, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi), int(r.size)


def bootstrap_ci(
    stat_fn: Callable[[np.ndarray], float],
    estimate: float,
    sites,
    mode: str = "within_site",
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> BootstrapResult:
    reps = bootstrap_replicates(stat_fn, sites, mode, n_boot, seed)
    lo, hi, nv = percentile_ci(reps, alpha)
    se = float(np.nanstd(reps, ddof=1)) if nv > 1 else float("nan")
    return BootstrapResult(float(estimate), lo, hi, se, n_boot, nv, mode, alpha)


def paired_bootstrap_delta(
    d,
    sites,
    mode: str = "within_site",
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> BootstrapResult:
    """CI for Delta = mean_i d_i. Patients with undefined d_i (no assessable label) are dropped first."""
    d = np.asarray(d, dtype=float)
    sites = np.asarray(sites)
    keep = ~np.isnan(d)
    d, sites = d[keep], sites[keep]
    if d.size == 0:
        raise ValueError("no patients with a defined d_i")
    return bootstrap_ci(lambda idx: d[idx].mean(), d.mean(), sites, mode, n_boot, seed, alpha)


def delta_ci_all_modes(d, sites, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Report within-site and cluster (and two-stage) intervals together, plus a site-level t interval."""
    out = {m: paired_bootstrap_delta(d, sites, m, n_boot, seed, alpha) for m in MODES}
    out["site_t"] = site_mean_t_interval(d, sites, alpha)
    return out


def site_mean_t_interval(d, sites, alpha: float = 0.05) -> dict:
    """Equal-weight mean of per-site Deltas with a t_{S-1} interval (a between-site sensitivity)."""
    d = np.asarray(d, dtype=float)
    sites = np.asarray(sites)
    vals = []
    for s in np.unique(sites):
        di = d[(sites == s) & ~np.isnan(d)]
        if di.size:
            vals.append(di.mean())
    vals = np.asarray(vals)
    S = vals.size
    if S < 2:
        return {"estimate": float(vals.mean()) if S else float("nan"), "lo": float("nan"), "hi": float("nan"), "df": 0}
    m = vals.mean()
    se = vals.std(ddof=1) / np.sqrt(S)
    t = stats.t.ppf(1 - alpha / 2, S - 1)
    return {"estimate": float(m), "lo": float(m - t * se), "hi": float(m + t * se), "df": int(S - 1)}


def one_sided_p_below_zero(reps) -> float:
    """Bootstrap p-value for H0: stat >= 0 vs H1: stat < 0, (1 + #{rep >= 0}) / (B + 1)."""
    r = np.asarray(reps, dtype=float)
    r = r[~np.isnan(r)]
    if r.size == 0:
        return float("nan")
    return float((1 + np.sum(r >= 0)) / (r.size + 1))


def holm_adjust(pvals) -> np.ndarray:
    """Holm step-down adjusted p-values (NaN preserved)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    m = pv.size
    order = np.argsort(pv)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pv[i])
        adj[i] = min(1.0, running)
    out[ok] = adj
    return out
