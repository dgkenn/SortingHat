"""Missing-data handling for the baselines (SAP section 9, item 3).

Policy
------
* Absence is information at t0 (a lab not yet drawn), so every variable that can be unobserved carries an
  explicit ``<name>__miss`` indicator, created by the builder (deterministic, nothing fitted).
* Drug exposure, history flags, imaging availability and culture "resulted" flags are 0 when there is no record
  (zero-fill is part of the feature definition, not imputation).
* Remaining NaN in ``value`` / ``age_h`` columns are filled with the median of the *training* rows
  (``MedianImputer``). Fit it on training sites only, inside each training fold; apply the frozen medians to the
  held-out site. A column with no observed training value is filled with 0. No multiple imputation: a deployed
  model cannot do it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

IMPUTE_ROLES = ("value", "age_h")


class MedianImputer:
    """Median imputation fitted on training rows only; records the medians for the SAP appendix."""

    def __init__(self, provenance: pd.DataFrame | None = None):
        self.provenance = provenance
        self.medians_: dict[str, float] | None = None

    def _imputable(self, cols) -> list[str]:
        if self.provenance is None:
            return [c for c in cols if not (c.endswith("__miss") or c.startswith(("ind__", "hx__")))]
        ok = set(self.provenance.loc[self.provenance["role"].isin(IMPUTE_ROLES), "feature"])
        return [c for c in cols if c in ok]

    def fit(self, X_train: pd.DataFrame) -> "MedianImputer":
        cols = self._imputable(X_train.columns)
        med = X_train[cols].median(skipna=True)
        self.medians_ = {c: (0.0 if pd.isna(med[c]) else float(med[c])) for c in cols}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.medians_ is None:
            raise RuntimeError("fit the imputer on training rows first")
        out = X.copy()
        for c, m in self.medians_.items():
            if c in out:
                out[c] = out[c].fillna(m)
        rest = out.columns[out.isna().any()]
        if len(rest):                                    # flags / counts must never be NaN after the builder
            raise ValueError(f"unexpected NaN in non-imputable columns: {list(rest)[:5]}")
        return out

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X_train).transform(X_train)
