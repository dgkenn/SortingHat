"""Masked multi-label binary log loss and the primary endpoint Delta.

Primary endpoint (research plan, "Primary endpoint"):

    loss(model) = (1/N) * sum_i  (1/|L_i|) * sum_{k in L_i}  l(y_ik, p_ik)

where L_i is patient i's set of *assessable* labels and l is binary log loss.
Patients with no assessable label contribute nothing (they are dropped from the
mean, identically for model and baseline). Delta = loss(with EEG) - loss(baseline);
negative means EEG helps.

Arrays: ``y`` (n, K) in {0, 1} (ignored where unassessable), ``p`` (n, K)
probabilities, ``mask`` (n, K) bool, True where the label is assessable.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

# Probability clipping bound. Preregistered default; sensitivity at 1e-6 and 1e-3.
DEFAULT_EPS = 1e-4


def _validate(y, p, mask):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if not (y.shape == p.shape == mask.shape) or y.ndim != 2:
        raise ValueError(f"y, p, mask must share a 2-D shape; got {y.shape}, {p.shape}, {mask.shape}")
    ya = y[mask]
    if not np.all(np.isin(ya, (0.0, 1.0))):
        raise ValueError("y must be 0/1 wherever mask is True")
    pa = p[mask]
    if not np.all(np.isfinite(pa)) or np.any(pa < 0) or np.any(pa > 1):
        raise ValueError("p must be finite and within [0, 1] wherever mask is True")
    return y, p, mask


def clip_probs(p, eps: float = DEFAULT_EPS):
    if not (0 < eps < 0.5):
        raise ValueError("eps must be in (0, 0.5)")
    return np.clip(p, eps, 1.0 - eps)


def binary_log_loss(y, p, eps: float = DEFAULT_EPS):
    """Elementwise binary log loss with clipping."""
    pc = clip_probs(np.asarray(p, dtype=float), eps)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(pc) + (1.0 - y) * np.log1p(-pc))


def per_patient_loss(y, p, mask, eps: float = DEFAULT_EPS) -> np.ndarray:
    """Mean log loss over each patient's assessable labels; NaN if none assessable."""
    y, p, mask = _validate(y, p, mask)
    pc = np.where(mask, p, 0.5)  # placeholder on masked cells, zeroed below
    loss = binary_log_loss(np.where(mask, y, 0.0), pc, eps)
    loss = np.where(mask, loss, 0.0)
    cnt = mask.sum(axis=1)
    out = np.full(len(cnt), np.nan)
    ok = cnt > 0
    out[ok] = loss[ok].sum(axis=1) / cnt[ok]
    return out


def mean_masked_log_loss(y, p, mask, eps: float = DEFAULT_EPS) -> float:
    """Mean over patients of mean over that patient's assessable labels."""
    pl = per_patient_loss(y, p, mask, eps)
    if np.all(np.isnan(pl)):
        return float("nan")
    return float(np.nanmean(pl))


def per_patient_delta(y, p_model, p_base, mask, eps: float = DEFAULT_EPS) -> np.ndarray:
    """d_i = loss_i(model with EEG) - loss_i(baseline); NaN for patients with no assessable label.

    This vector is the pairing unit for every bootstrap in the SAP.
    """
    return per_patient_loss(y, p_model, mask, eps) - per_patient_loss(y, p_base, mask, eps)


def delta_log_loss(y, p_model, p_base, mask, eps: float = DEFAULT_EPS) -> float:
    """Primary endpoint: loss(model) - loss(baseline). Negative = EEG helps."""
    d = per_patient_delta(y, p_model, p_base, mask, eps)
    if np.all(np.isnan(d)):
        return float("nan")
    return float(np.nanmean(d))


def per_label_delta(y, p_model, p_base, mask, eps: float = DEFAULT_EPS) -> np.ndarray:
    """Per-label Delta_k: mean over patients assessable for label k of the log-loss difference."""
    y, p_model, mask = _validate(y, p_model, mask)
    _validate(y, p_base, mask)
    yy = np.where(mask, y, 0.0)
    lm = binary_log_loss(yy, np.where(mask, p_model, 0.5), eps)
    lb = binary_log_loss(yy, np.where(mask, p_base, 0.5), eps)
    diff = np.where(mask, lm - lb, 0.0)
    cnt = mask.sum(axis=0)
    out = np.full(mask.shape[1], np.nan)
    ok = cnt > 0
    out[ok] = diff.sum(axis=0)[ok] / cnt[ok]
    return out


# --------------------------------------------------------------------------
# Per-site Delta and the "favorable at every held-out site" check
# --------------------------------------------------------------------------
def per_site_delta(d, sites) -> dict[str, dict[str, float]]:
    """Delta and n (patients with a defined d_i) for each site."""
    d = np.asarray(d, dtype=float)
    sites = np.asarray(sites)
    if d.shape[0] != sites.shape[0]:
        raise ValueError("d and sites must have the same length")
    out: dict[str, dict[str, float]] = {}
    for s in np.unique(sites):
        di = d[sites == s]
        di = di[~np.isnan(di)]
        out[str(s)] = {"delta": float(di.mean()) if di.size else float("nan"), "n": int(di.size)}
    return out


def favorable_at_every_site(d, sites, min_sites: int = 1) -> dict:
    """True iff the point estimate of Delta is strictly below 0 at every held-out site.

    Returns {"all_favorable": bool, "n_sites": int, "per_site": {...}, "unfavorable": [...]}.
    A site with undefined Delta (no patients) counts as unfavorable.
    """
    ps = per_site_delta(d, sites)
    bad = [s for s, v in ps.items() if not (v["delta"] < 0)]
    return {
        "all_favorable": bool(len(ps) >= min_sites and not bad),
        "n_sites": len(ps),
        "per_site": ps,
        "unfavorable": bad,
    }


def site_weighted_delta(d, sites) -> float:
    """Unweighted mean of per-site Deltas (sensitivity to patient-weighted pooling)."""
    ps = per_site_delta(d, sites)
    vals = [v["delta"] for v in ps.values() if not np.isnan(v["delta"])]
    return float(np.mean(vals)) if vals else float("nan")
