"""Container for the Study 1 modelling harness.

``ModelData`` holds one row per patient (first qualifying EEG), aligned across:

* ``baseline``  generic numeric DataFrame (n x b) of t0 clinical features. The baseline builder lives elsewhere
  (``sortinghat/baselines/``); this harness consumes it as a plain frame and never looks inside it.
* ``eeg``       numeric DataFrame (n x f) of EEG representation columns, grouped into ladder rungs by column-name
  prefix: ``qeeg.``, ``conn.``, ``morgoth.``, ``emb.<family>.``, ``dyn.`` (see ``ladder.DEFAULT_RUNGS``).
* silver labels ``y_silver`` / ``m_silver``  (training only; never scored).
* gold labels   ``y_gold`` / ``m_gold`` with ``gold_role`` in {"dev", "eval", "none"}. Dev gold cases are the
  only gold labels a fit may see (model selection and recalibration). Eval gold cases are touched only by the
  evaluator. Rows with role "none" carry no usable gold label (mask forced False).
* ``sites`` (n,), optional ``times`` (n,), and ``covariates`` (named (n,) arrays such as ``duration_s``,
  ``n_channels``, ``severity``, ``sedated``) used only by the controls, never as model inputs.

Frames are re-indexed 0..n-1 on construction so no opaque IDs travel with the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np
import pandas as pd

ROLES = ("dev", "eval", "none")


def _as_bool2d(a, shape, name):
    a = np.asarray(a, dtype=bool)
    if a.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {a.shape}")
    return a


@dataclass
class ModelData:
    baseline: pd.DataFrame
    eeg: pd.DataFrame
    y_silver: np.ndarray
    m_silver: np.ndarray
    y_gold: np.ndarray
    m_gold: np.ndarray
    gold_role: np.ndarray
    sites: np.ndarray
    label_names: tuple
    times: np.ndarray | None = None
    covariates: dict = field(default_factory=dict)

    def __post_init__(self):
        self.baseline = pd.DataFrame(self.baseline).reset_index(drop=True)
        self.eeg = pd.DataFrame(self.eeg).reset_index(drop=True)
        n = len(self.baseline)
        if len(self.eeg) != n:
            raise ValueError("baseline and eeg must have the same number of rows")
        self.label_names = tuple(self.label_names)
        K = len(self.label_names)
        self.y_silver = np.asarray(self.y_silver, dtype=float)
        self.y_gold = np.asarray(self.y_gold, dtype=float)
        if self.y_silver.shape != (n, K) or self.y_gold.shape != (n, K):
            raise ValueError(f"label arrays must have shape {(n, K)}")
        self.m_silver = _as_bool2d(self.m_silver, (n, K), "m_silver")
        m_gold = _as_bool2d(self.m_gold, (n, K), "m_gold")
        self.gold_role = np.asarray(self.gold_role).astype(str)
        self.sites = np.asarray(self.sites)
        if self.gold_role.shape != (n,) or self.sites.shape != (n,):
            raise ValueError("gold_role and sites must have shape (n,)")
        bad = set(np.unique(self.gold_role)) - set(ROLES)
        if bad:
            raise ValueError(f"gold_role values must be in {ROLES}; got {sorted(bad)}")
        # gold labels exist only for dev / eval rows
        self.m_gold = m_gold & (self.gold_role != "none")[:, None]
        for nm, y, m in (("silver", self.y_silver, self.m_silver), ("gold", self.y_gold, self.m_gold)):
            if not np.all(np.isin(y[m], (0.0, 1.0))):
                raise ValueError(f"{nm} labels must be 0/1 where assessable")
        for df, nm in ((self.baseline, "baseline"), (self.eeg, "eeg")):
            non_num = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
            if non_num:
                raise ValueError(f"{nm} must be numeric (encode categoricals upstream); non-numeric: {len(non_num)} columns")
        if self.times is not None:
            self.times = np.asarray(self.times)
            if self.times.shape != (n,):
                raise ValueError("times must have shape (n,)")
        self.covariates = {k: np.asarray(v) for k, v in self.covariates.items()}
        for k, v in self.covariates.items():
            if v.shape[0] != n:
                raise ValueError(f"covariate {k!r} must have length n")

    @property
    def n(self) -> int:
        return len(self.baseline)

    @property
    def K(self) -> int:
        return len(self.label_names)

    def take(self, idx) -> "ModelData":
        idx = np.asarray(idx, dtype=int)
        return ModelData(
            baseline=self.baseline.iloc[idx], eeg=self.eeg.iloc[idx],
            y_silver=self.y_silver[idx], m_silver=self.m_silver[idx],
            y_gold=self.y_gold[idx], m_gold=self.m_gold[idx], gold_role=self.gold_role[idx],
            sites=self.sites[idx], label_names=self.label_names,
            times=None if self.times is None else self.times[idx],
            covariates={k: v[idx] for k, v in self.covariates.items()},
        )

    def design(self, idx, baseline_cols: Sequence[str], eeg_cols: Sequence[str]) -> pd.DataFrame:
        """Feature frame for rows ``idx``: baseline columns then EEG columns. Features only, no labels."""
        idx = np.asarray(idx, dtype=int)
        parts = [self.baseline.loc[:, list(baseline_cols)].iloc[idx]]
        if len(eeg_cols):
            parts.append(self.eeg.loc[:, list(eeg_cols)].iloc[idx])
        return pd.concat(parts, axis=1).reset_index(drop=True)

    def replace(self, **kw) -> "ModelData":
        return replace(self, **kw)

    def copy(self) -> "ModelData":
        return self.take(np.arange(self.n))
