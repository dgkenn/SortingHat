"""Fold-local preprocessing: imputation, missingness indicators, scaling, clipping, feature selection.

Everything is fitted by ``FoldPreprocessor.fit`` on training rows only; ``transform`` is a pure function of
those stored parameters and of the rows passed in (no test-row statistic is ever computed).
SAP section 9: in-training-fold median imputation plus a missingness indicator per variable.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PreprocConfig:
    missing_indicators: bool = True
    clip_z: float | None = 5.0              # winsorise standardised features at +/- clip_z
    max_eeg_features: int | None = None     # keep top-k non-protected columns by silver-label association
    min_sd: float = 1e-8


class FoldPreprocessor:
    def __init__(self, cfg: PreprocConfig | None = None):
        self.cfg = cfg or PreprocConfig()
        self.fitted_ = False

    def fit(self, X, columns: Sequence[str], y=None, m=None, protected: Sequence[str] = ()):
        X = np.asarray(X, dtype=float)
        cols = list(columns)
        n, d = X.shape
        if n == 0:
            raise ValueError("cannot fit preprocessing on zero rows")
        allnan = np.isnan(X).all(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(X, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        Xi = np.where(np.isnan(X), med, X)
        mu = Xi.mean(axis=0)
        sd = Xi.std(axis=0)
        keep = (~allnan) & (sd > self.cfg.min_sd)
        prot = set(protected)
        k = self.cfg.max_eeg_features
        if k is not None and y is not None and m is not None:
            cand = [j for j in range(d) if keep[j] and cols[j] not in prot]
            if len(cand) > k:
                Z = (Xi - mu) / np.where(sd > 0, sd, 1.0)
                y = np.asarray(y, float)
                m = np.asarray(m, bool)
                score = np.zeros(d)
                for kk in range(y.shape[1]):
                    r = m[:, kk]
                    if r.sum() < 5:
                        continue
                    yk = y[r, kk] - y[r, kk].mean()
                    if yk.std() == 0:
                        continue
                    Zr = Z[r]
                    c = np.abs(Zr.T @ yk) / (r.sum() * yk.std() * np.maximum(Zr.std(axis=0), 1e-12))
                    score = np.maximum(score, c)
                cand_sorted = sorted(cand, key=lambda j: (-score[j], j))
                for j in cand_sorted[k:]:
                    keep[j] = False
        self.keep_ = np.flatnonzero(keep)
        self.median_ = med[self.keep_]
        self.mean_ = mu[self.keep_]
        self.sd_ = sd[self.keep_]
        had_nan = np.isnan(X[:, self.keep_]).any(axis=0)
        self.ind_ = np.flatnonzero(had_nan) if self.cfg.missing_indicators else np.array([], dtype=int)
        self.in_columns_ = tuple(cols)
        self.protected_ = frozenset(prot)
        self.out_columns_ = tuple(cols[j] for j in self.keep_) + tuple(cols[self.keep_[j]] + "__missing" for j in self.ind_)
        self.protected_out_ = np.array([c.replace("__missing", "") in prot for c in self.out_columns_], dtype=bool)
        self.fitted_ = True
        return self

    def transform(self, X) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("FoldPreprocessor not fitted")
        X = np.asarray(X, dtype=float)[:, self.keep_]
        isn = np.isnan(X)
        Z = (np.where(isn, self.median_, X) - self.mean_) / self.sd_
        if self.cfg.clip_z is not None:
            Z = np.clip(Z, -self.cfg.clip_z, self.cfg.clip_z)
        if self.ind_.size:
            Z = np.hstack([Z, isn[:, self.ind_].astype(float)])
        return Z

    def param_arrays(self):
        return [self.keep_, self.median_, self.mean_, self.sd_, self.ind_]
