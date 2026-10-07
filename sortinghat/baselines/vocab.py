"""Human-run, aggregate-only discovery of lab names that the lexicon does not map (feeds
``BaselineConfig.extra_labs`` so Baseline C can cover "every lab"). Run on TRAINING sites only, then freeze."""

from __future__ import annotations

import pandas as pd

from ..safe_output import suppress_count
from . import lexicon as lx


def discover_lab_vocabulary(measurements: pd.DataFrame, min_patients: int = 50) -> pd.DataFrame:
    """Unmapped normalized measurement names seen for >= ``min_patients`` patients.

    Returns name + suppressed patient count (n < 11 shown as "<11"); ``min_patients`` is clamped to >= 11 so the
    returned names are never small-cell. Names are free text: review before adding to ``extra_labs``.
    """
    min_patients = max(int(min_patients), 11)
    names = measurements["measurement_source_value"].astype("string")
    unmapped = names.map(lambda u: lx.classify_measurement(u) is None)
    sub = measurements.loc[unmapped.to_numpy(bool)].assign(norm=names[unmapped].map(lx.norm_name))
    cnt = sub[sub["norm"] != ""].groupby("norm")["person_id"].nunique().sort_values(ascending=False)
    cnt = cnt[cnt >= min_patients]
    return pd.DataFrame({"norm_name": cnt.index, "n_patients": [suppress_count(int(n)) for n in cnt.to_numpy()]})
