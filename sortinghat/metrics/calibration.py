"""Per-label calibration, discrimination and selective-prediction metrics.

All functions work on one label at a time: ``y`` (n,) in {0,1}, ``p`` (n,) probabilities,
already restricted to patients for whom the label is assessable. ``per_label_report``
applies the mask for a (n, K) problem.

Calibration slope and intercept use logistic recalibration (Van Calster et al. 2016,
PMID 26772608): fit logit P(y=1) = a + b * logit(p). b is the calibration slope
(ideal 1); a is the intercept of the *joint* model. Calibration-in-the-large (CITL) is
the intercept with the slope fixed at 1 (offset model; ideal 0). O/E is observed events
over expected events (ideal 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

from .loss import DEFAULT_EPS, binary_log_loss, clip_probs


def logit(p, eps: float = DEFAULT_EPS):
    pc = clip_probs(np.asarray(p, dtype=float), eps)
    return np.log(pc) - np.log1p(-pc)


def fit_logistic(X, y, offset=None, max_iter: int = 100, tol: float = 1e-9):
    """Unpenalised logistic regression by Newton-Raphson with step halving.

    Returns (coef, cov, converged). Non-convergence or |coef| > 30 (separation) is reported
    as converged=False and callers return NaN.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    off = np.zeros(n) if offset is None else np.asarray(offset, dtype=float)
    beta = np.zeros(k)

    def nll(b):
        eta = off + X @ b
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta))

    cur = nll(beta)
    converged = False
    for _ in range(max_iter):
        eta = off + X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = mu * (1.0 - mu)
        grad = X.T @ (y - mu)
        H = (X * w[:, None]).T @ X
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            return beta, np.full((k, k), np.nan), False
        t = 1.0
        while t > 1e-8:
            new = nll(beta + t * step)
            if new <= cur + 1e-12:
                break
            t /= 2
        beta = beta + t * step
        if abs(cur - new) < tol and np.max(np.abs(t * step)) < 1e-7:
            cur = new
            converged = True
            break
        cur = new
    eta = off + X @ beta
    mu = 1.0 / (1.0 + np.exp(-eta))
    H = (X * (mu * (1 - mu))[:, None]).T @ X
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.full((k, k), np.nan)
    if np.max(np.abs(beta)) > 30:
        converged = False
    return beta, cov, converged


def calibration_slope_intercept(y, p, eps: float = DEFAULT_EPS, alpha: float = 0.05) -> dict:
    """Logistic recalibration summary for one label. NaN fields if a class is empty or fit fails."""
    from scipy import stats

    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n = int(y.size)
    n1 = int(y.sum())
    nan = float("nan")
    out = {
        "n": n, "n_events": n1, "oe": nan, "slope": nan, "slope_se": nan, "slope_lo": nan, "slope_hi": nan,
        "intercept": nan, "citl": nan, "citl_se": nan, "converged": False,
    }
    if n == 0:
        return out
    exp = float(np.clip(p, eps, 1 - eps).sum())
    out["oe"] = float(n1 / exp) if exp > 0 else nan
    if n1 == 0 or n1 == n:
        return out
    lp = logit(p, eps)
    z = stats.norm.ppf(1 - alpha / 2)
    X = np.column_stack([np.ones(n), lp])
    beta, cov, ok = fit_logistic(X, y)
    c_beta, c_cov, ok_c = fit_logistic(np.ones((n, 1)), y, offset=lp)
    if ok_c:
        out["citl"] = float(c_beta[0])
        out["citl_se"] = float(np.sqrt(c_cov[0, 0]))
    if ok:
        se = float(np.sqrt(cov[1, 1]))
        out.update(
            intercept=float(beta[0]), slope=float(beta[1]), slope_se=se,
            slope_lo=float(beta[1] - z * se), slope_hi=float(beta[1] + z * se), converged=True,
        )
    return out


def ece(y, p, n_bins: int = 10, strategy: str = "uniform") -> float:
    """Expected calibration error for a binary label.

    ``uniform``: equal-width bins on [0, 1]. ``quantile``: equal-count bins. ECE =
    sum_b (n_b / n) * |mean(y in b) - mean(p in b)|. Binned ECE is biased upward in small
    samples; report with bootstrap CI and never use it for model selection.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n = y.size
    if n == 0:
        return float("nan")
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        b = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    elif strategy == "quantile":
        order = np.argsort(p, kind="mergesort")
        b = np.empty(n, dtype=int)
        b[order] = np.minimum((np.arange(n) * n_bins) // n, n_bins - 1)
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile'")
    total = 0.0
    for k in range(n_bins):
        m = b == k
        if m.any():
            total += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return float(total)


def brier(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2)) if y.size else float("nan")


def auroc(y, p) -> float:
    y = np.asarray(y)
    if y.size == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def per_label_report(y, p, mask, label_names=None, eps: float = DEFAULT_EPS, n_bins: int = 10) -> dict:
    """Calibration slope/intercept/CITL/O-E, ECE, Brier and AUROC for every label column."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    K = y.shape[1]
    names = list(label_names) if label_names is not None else [f"L{k}" for k in range(K)]
    rep = {}
    for k in range(K):
        m = mask[:, k]
        yk, pk = y[m, k], p[m, k]
        r = calibration_slope_intercept(yk, pk, eps)
        r.update(ece=ece(yk, pk, n_bins), brier=brier(yk, pk), auroc=auroc(yk, pk))
        rep[names[k]] = r
    return rep


# --------------------------------------------------------------------------
# Risk-coverage (selective prediction)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RiskCoverage:
    coverage: np.ndarray  # fraction of cases retained, ascending, k / n
    risk: np.ndarray      # mean risk over the retained (most confident) cases
    aurc: float           # area under the risk-coverage curve (lower is better)


def risk_coverage(y, p, risk: str = "logloss", confidence=None, eps: float = DEFAULT_EPS) -> RiskCoverage:
    """Per-label risk-coverage curve.

    Cases are ranked by confidence (default max(p, 1-p)); the curve gives the mean risk
    over the top-k most confident cases for k = 1..n. ``risk`` is 'logloss', 'brier' or
    'error' (0/1 at threshold 0.5). Ties are broken by input order (stable sort), so results
    are deterministic. AURC is the mean of the curve over k (equivalently, trapezoid-free
    average risk across coverages 1/n..1).
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n = y.size
    if n == 0:
        raise ValueError("empty input")
    conf = np.maximum(p, 1 - p) if confidence is None else np.asarray(confidence, dtype=float)
    order = np.argsort(-conf, kind="mergesort")
    if risk == "logloss":
        r = binary_log_loss(y, p, eps)
    elif risk == "brier":
        r = (p - y) ** 2
    elif risk == "error":
        r = ((p >= 0.5).astype(float) != y).astype(float)
    else:
        raise ValueError("risk must be 'logloss', 'brier' or 'error'")
    cum = np.cumsum(r[order]) / np.arange(1, n + 1)
    return RiskCoverage(coverage=np.arange(1, n + 1) / n, risk=cum, aurc=float(cum.mean()))


def per_label_risk_coverage(y, p, mask, label_names=None, risk: str = "logloss", eps: float = DEFAULT_EPS) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    names = list(label_names) if label_names is not None else [f"L{k}" for k in range(y.shape[1])]
    return {names[k]: risk_coverage(y[mask[:, k], k], p[mask[:, k], k], risk, eps=eps) for k in range(y.shape[1])}
