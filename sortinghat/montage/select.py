"""Data-driven 4-8 electrode selection that can only see training-fold data.

The API enforces the no-leakage rule structurally: the selector receives *already sliced* training
arrays (``X_train``, ``y_train``) with their record indices, never the full dataset.  Callbacks get
read-only views of the training data only.  If the caller also supplies the held-out indices, they
are checked to be disjoint from the training indices.  ``select_within_folds`` does the slicing
itself so held-out rows are never passed to the selector.

Callbacks (channel axis is axis 1 of X: ``(n_records, n_channels, ...)``):
  importance_fn(X_train, y_train, channel_names) -> per-channel importance (higher = better)
  score_fn(X_train_subset, y_train, subset_names) -> scalar score (higher = better); do any inner CV
      on the training data yourself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ElectrodeSelection:
    electrodes: Tuple[str, ...]
    method: str
    n_train: int
    train_idx: Tuple[int, ...]
    history: Tuple[Tuple[str, float], ...] = field(default_factory=tuple)  # (added electrode, score/importance)


def _readonly(a):
    v = np.asarray(a).view()
    v.flags.writeable = False
    return v


def select_electrode_subset(X_train, y_train, train_idx: Sequence[int], channel_names: Sequence[str],
                            k: int, *, importance_fn: Optional[Callable] = None,
                            score_fn: Optional[Callable] = None, heldout_idx: Optional[Sequence[int]] = None,
                            candidates: Optional[Sequence[str]] = None,
                            forced: Sequence[str] = ()) -> ElectrodeSelection:
    """Pick ``k`` (4-8) electrodes using training data only.

    Give exactly one of ``importance_fn`` (top-k by importance) or ``score_fn`` (greedy forward selection).
    """
    if (importance_fn is None) == (score_fn is None):
        raise ValueError("provide exactly one of importance_fn or score_fn")
    if not 4 <= k <= 8:
        raise ValueError("k must be between 4 and 8 (study-plan electrode budget)")
    X = _readonly(X_train)
    y = _readonly(y_train)
    tr = np.asarray(train_idx)
    if tr.ndim != 1 or len(np.unique(tr)) != len(tr):
        raise ValueError("train_idx must be a 1-D list of unique indices")
    if X.shape[0] != len(tr) or len(y) != len(tr):
        raise ValueError("X_train/y_train must contain exactly the rows named by train_idx")
    if heldout_idx is not None and np.intersect1d(tr, np.asarray(heldout_idx)).size:
        raise ValueError("train_idx overlaps heldout_idx: held-out data must never be used for selection")
    names = list(channel_names)
    if X.ndim < 2 or X.shape[1] != len(names) or len(set(names)) != len(names):
        raise ValueError("X_train axis 1 must match unique channel_names")
    cand = list(candidates) if candidates is not None else list(names)
    unknown = [c for c in list(cand) + list(forced) if c not in names]
    if unknown:
        raise ValueError(f"unknown electrodes: {unknown}")
    forced = list(dict.fromkeys(forced))
    if len(forced) > k:
        raise ValueError("more forced electrodes than k")
    cand = [c for c in cand if c not in forced]
    if len(cand) + len(forced) < k:
        raise ValueError("not enough candidate electrodes for k")
    col = {n: i for i, n in enumerate(names)}

    chosen: List[str] = list(forced)
    hist: List[Tuple[str, float]] = []
    if importance_fn is not None:
        imp = np.asarray(importance_fn(X, y, tuple(names)), dtype=float)
        if imp.shape != (len(names),) or not np.all(np.isfinite(imp)):
            raise ValueError("importance_fn must return a finite vector, one value per channel")
        for c in sorted(cand, key=lambda c: (-imp[col[c]], names.index(c)))[: k - len(chosen)]:
            chosen.append(c)
            hist.append((c, float(imp[col[c]])))
        method = "importance_topk"
    else:
        remaining = list(cand)
        while len(chosen) < k:
            best, best_s = None, -np.inf
            for c in remaining:
                sub = chosen + [c]
                s = float(score_fn(_readonly(X[:, [col[n] for n in sub]]), y, tuple(sub)))
                if not np.isfinite(s):
                    raise ValueError("score_fn returned a non-finite score")
                if s > best_s:
                    best, best_s = c, s
            chosen.append(best)
            remaining.remove(best)
            hist.append((best, best_s))
        method = "greedy_forward"
    return ElectrodeSelection(tuple(chosen), method, len(tr), tuple(int(i) for i in tr), tuple(hist))


def select_within_folds(X, y, folds: Sequence[Tuple[Sequence[int], Sequence[int]]],
                        channel_names: Sequence[str], k: int, **kwargs) -> List[ElectrodeSelection]:
    """Run selection independently per outer fold, passing only each fold's training rows.

    ``folds`` is a list of (train_idx, test_idx).  Test rows are never sliced or handed to callbacks.
    """
    y = np.asarray(y)
    out = []
    for tr, te in folds:
        tr = np.asarray(tr)
        out.append(select_electrode_subset(np.asarray(X)[tr], y[tr], tr, channel_names, k,
                                           heldout_idx=te, **kwargs))
    return out
