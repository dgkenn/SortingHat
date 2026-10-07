"""Small dependency-free agreement statistics (numpy only)."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def cohen_kappa(a: Sequence[int], b: Sequence[int], categories: Sequence[int] | None = None,
                weights: str | None = None) -> float:
    """Cohen's kappa; ``weights`` in {None, 'linear', 'quadratic'} (for ordinal categories).

    Returns nan when undefined (n=0 or expected disagreement is 0).
    """
    a = np.asarray(a); b = np.asarray(b)
    if len(a) != len(b):
        raise ValueError("rating vectors differ in length")
    if len(a) == 0:
        return float("nan")
    cats = np.array(sorted(set(a.tolist()) | set(b.tolist())) if categories is None else list(categories))
    k = len(cats)
    idx = {c: i for i, c in enumerate(cats.tolist())}
    obs = np.zeros((k, k))
    for x, y in zip(a.tolist(), b.tolist()):
        obs[idx[x], idx[y]] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0))
    i, j = np.indices((k, k))
    if weights is None:
        w = (i != j).astype(float)
    elif weights == "linear":
        w = np.abs(i - j) / max(k - 1, 1)
    elif weights == "quadratic":
        w = ((i - j) / max(k - 1, 1)) ** 2
    else:
        raise ValueError(f"unknown weights {weights!r}")
    den = (w * exp).sum()
    if den == 0:
        return float("nan")
    return float(1.0 - (w * obs).sum() / den)


def auroc(y_true: Sequence[int], score: Sequence[float]) -> float:
    """Rank-based AUROC with tie correction; nan if one class is absent."""
    y = np.asarray(y_true).astype(int); s = np.asarray(score, dtype=float)
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
