"""Shallow multi-label head shared by every ladder rung, with in-fold model selection and recalibration.

* Head: one L2-regularised logistic regression per label (optionally a small MLP), trained on SILVER labels with
  per-label masks. A label with too few examples of a class falls back to a smoothed prevalence.
* Model selection (single regularisation strength for all labels) and Platt/isotonic recalibration use ONLY gold
  *development* cases that sit inside the training fold. Gold evaluation cases never reach this module.
* The same code and the same hyperparameter budget fit the baseline-only and the baseline+EEG model (SAP 4.3).
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from ..metrics.loss import DEFAULT_EPS, clip_probs, mean_masked_log_loss
from .preprocess import FoldPreprocessor, PreprocConfig


@dataclass(frozen=True)
class HeadConfig:
    kind: str = "logistic"                  # "logistic" | "mlp"
    c_grid: tuple = (0.01, 0.1, 1.0)        # inverse L2 strength (mlp: alpha = 1 / C)
    default_c: float = 0.1                  # used when no usable dev set exists
    # Extra shrinkage of the non-baseline (EEG) block: columns are multiplied by s <= 1 after standardisation, which
    # is an L2 penalty 1/s^2 times larger on EEG than on baseline columns. Chosen on dev gold with C.
    eeg_scale_grid: tuple = (1.0, 0.3, 0.1)
    default_eeg_scale: float = 0.3
    mlp_hidden: tuple = (16,)
    max_iter: int = 500
    seed: int = 0
    min_class_count: int = 3                # per label, per class, in silver training rows
    calibration: str = "platt"              # "platt" | "isotonic" | "none"
    min_dev_patients: int = 30              # dev rows with an assessable selection label
    min_dev_per_class: int = 8              # per label, to fit a calibrator
    eps: float = DEFAULT_EPS


def _logit(p, eps):
    p = clip_probs(np.asarray(p, float), eps)
    return np.log(p) - np.log1p(-p)


class ShallowHead:
    def __init__(self, cfg: HeadConfig, c: float, col_scale=None):
        self.cfg, self.c = cfg, float(c)
        self.col_scale = None if col_scale is None else np.asarray(col_scale, float)

    def _s(self, Z):
        return Z if self.col_scale is None else Z * self.col_scale

    def fit(self, Z, y, m):
        Z = self._s(Z)
        self.models_ = []
        for k in range(y.shape[1]):
            r = m[:, k]
            yk = y[r, k]
            n1, n0 = int(yk.sum()), int((1 - yk).sum())
            if min(n1, n0) < self.cfg.min_class_count:
                self.models_.append(("const", (n1 + 0.5) / (n1 + n0 + 1.0)))
                continue
            if self.cfg.kind == "logistic":
                est = LogisticRegression(C=self.c, max_iter=self.cfg.max_iter, solver="lbfgs")
            elif self.cfg.kind == "mlp":
                est = MLPClassifier(hidden_layer_sizes=self.cfg.mlp_hidden, alpha=1.0 / self.c,
                                    max_iter=self.cfg.max_iter, random_state=self.cfg.seed)
            else:
                raise ValueError(f"unknown head kind {self.cfg.kind!r}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est.fit(Z[r], yk)
            self.models_.append(("est", est))
        return self

    def predict_proba(self, Z) -> np.ndarray:
        Z = self._s(Z)
        out = np.empty((Z.shape[0], len(self.models_)))
        for k, (kind, obj) in enumerate(self.models_):
            out[:, k] = obj if kind == "const" else obj.predict_proba(Z)[:, 1]
        return out

    def param_arrays(self):
        arrs = [] if self.col_scale is None else [self.col_scale]
        for kind, obj in self.models_:
            if kind == "const":
                arrs.append(np.array([obj]))
            elif hasattr(obj, "coef_"):
                arrs += [obj.coef_, obj.intercept_]
            else:
                arrs += list(obj.coefs_) + list(obj.intercepts_)
        return arrs


class LabelCalibrator:
    """Per-label Platt (logistic on logit p) or isotonic map; identity where dev data are too thin."""

    def __init__(self, method: str, eps: float, min_per_class: int):
        self.method, self.eps, self.min_per_class = method, eps, min_per_class

    def fit(self, P, y, m):
        self.maps_ = []
        for k in range(P.shape[1]):
            r = m[:, k]
            yk = y[r, k]
            if self.method == "none" or min(yk.sum(), (1 - yk).sum()) < self.min_per_class:
                self.maps_.append(None)
                continue
            if self.method == "platt":
                lr = LogisticRegression(C=1e3, max_iter=500)
                lr.fit(_logit(P[r, k], self.eps)[:, None], yk)
                self.maps_.append(("platt", lr))
            elif self.method == "isotonic":
                iso = IsotonicRegression(y_min=self.eps, y_max=1 - self.eps, out_of_bounds="clip")
                iso.fit(P[r, k], yk)
                self.maps_.append(("isotonic", iso))
            else:
                raise ValueError(f"unknown calibration {self.method!r}")
        return self

    def transform(self, P) -> np.ndarray:
        out = np.array(P, dtype=float, copy=True)
        for k, mp in enumerate(self.maps_):
            if mp is None:
                continue
            kind, obj = mp
            out[:, k] = (obj.predict_proba(_logit(P[:, k], self.eps)[:, None])[:, 1]
                         if kind == "platt" else obj.predict(P[:, k]))
        return out

    def param_arrays(self):
        arrs = []
        for mp in self.maps_:
            if mp is None:
                arrs.append(np.zeros(1))
            elif mp[0] == "platt":
                arrs += [mp[1].coef_, mp[1].intercept_]
            else:
                arrs += [mp[1].X_thresholds_, mp[1].y_thresholds_]
        return arrs


@dataclass
class FittedModel:
    pre: FoldPreprocessor
    head: ShallowHead
    calibrator: LabelCalibrator | None
    columns: tuple
    info: dict = field(default_factory=dict)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if tuple(X.columns) != self.columns:
            raise ValueError("prediction columns differ from training columns")
        P = self.head.predict_proba(self.pre.transform(X.values))
        if self.calibrator is not None:
            P = self.calibrator.transform(P)
        return clip_probs(P, self.head.cfg.eps)

    def fingerprint(self) -> str:
        """Hash of every fitted parameter (scaler, imputer, selection, head weights, calibrators)."""
        h = hashlib.sha256()
        arrs = list(self.pre.param_arrays()) + self.head.param_arrays()
        if self.calibrator is not None:
            arrs += self.calibrator.param_arrays()
        for a in arrs:
            a = np.ascontiguousarray(np.asarray(a, dtype=float))
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
        h.update(repr(self.columns).encode())
        return h.hexdigest()


def fit_model(X_train: pd.DataFrame, y_silver, m_silver, X_dev: pd.DataFrame | None, y_dev, m_dev, *,
              protected: Sequence[str] = (), select_cols: Sequence[int] | None = None,
              pre_cfg: PreprocConfig | None = None, head_cfg: HeadConfig | None = None) -> FittedModel:
    """Fit preprocessing + head on silver labels; choose C and recalibrate on gold dev cases (all inside the fold).

    ``select_cols``: label columns scored for model selection (default: all). Nothing here can see a test row:
    the function receives only training features/labels and dev features/gold labels.
    """
    head_cfg = head_cfg or HeadConfig()
    K = y_silver.shape[1]
    sel = np.zeros(K, bool)
    sel[list(range(K)) if select_cols is None else list(select_cols)] = True
    pre = FoldPreprocessor(pre_cfg).fit(X_train.values, list(X_train.columns), y_silver, m_silver, protected)
    Z = pre.transform(X_train.values)

    use_dev = False
    if X_dev is not None and len(X_dev):
        md = np.asarray(m_dev, bool) & sel[None, :]
        use_dev = int((md.any(axis=1)).sum()) >= head_cfg.min_dev_patients
    has_eeg = bool((~pre.protected_out_).any())
    scales = tuple(head_cfg.eeg_scale_grid) if (use_dev and has_eeg) else \
        ((head_cfg.default_eeg_scale,) if has_eeg else (1.0,))
    cs = tuple(head_cfg.c_grid) if use_dev else (head_cfg.default_c,)
    grid = tuple((c, s) for c in cs for s in scales)

    def _cs(s):
        return np.where(pre.protected_out_, 1.0, s)

    heads = {g: ShallowHead(head_cfg, g[0], _cs(g[1])).fit(Z, y_silver, m_silver) for g in grid}
    chosen, dev_scores = grid[0], {}
    Pd = None
    if use_dev:
        Zd = pre.transform(X_dev.values)
        yd = np.where(md, y_dev, 0.0)
        for c, hd in heads.items():
            dev_scores[c] = mean_masked_log_loss(yd, np.where(md, hd.predict_proba(Zd), 0.5), md, head_cfg.eps)
        chosen = min(grid, key=lambda c: (dev_scores[c], grid.index(c)))
        Pd = heads[chosen].predict_proba(Zd)
    cal = None
    if use_dev and head_cfg.calibration != "none":
        cal = LabelCalibrator(head_cfg.calibration, head_cfg.eps, head_cfg.min_dev_per_class)
        cal.fit(Pd, np.asarray(y_dev, float), np.asarray(m_dev, bool))
    info = {"selected_c": float(chosen[0]), "selected_eeg_scale": float(chosen[1]), "used_dev": bool(use_dev), "calibrated": cal is not None,
            "n_train": int(len(X_train)), "n_dev": int(0 if X_dev is None else len(X_dev)),
            "n_features_in": int(X_train.shape[1]), "n_features_out": int(len(pre.out_columns_))}
    return FittedModel(pre, heads[chosen], cal, tuple(X_train.columns), info)


def prevalence_prior(y_silver, m_silver) -> np.ndarray:
    """Smoothed per-label prevalence on silver training rows (the 'prior' rung)."""
    n1 = (y_silver * m_silver).sum(axis=0)
    n = m_silver.sum(axis=0)
    return (n1 + 0.5) / (n + 1.0)
