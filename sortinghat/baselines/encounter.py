"""Encounter start and the current-encounter restriction (D-145).

By default every baseline (A, B, C and P) may use only events of the CURRENT encounter: those timed at or after the
encounter start. ``as_of`` then keeps only what was available by t0, so the usable window is [encounter start, t0]. An
EEG is often recorded hours or days into an admission, and the intended user ("what caused this altered mental
status?", little or nothing known) has no prior-encounter history; a prior visit's labs, drugs or diagnosis codes would
be information that user lacks.

* ``encounter_start_from_visits`` reuses the cohort's own rule (``cohort.rules.match_visits``: the covering visit, one hop
  back along acute visits ending within the chain gap, never later than t0) so the baselines and the cohort agree.
* ``restrict_to_current_encounter`` drops event rows timed before the encounter start. Rows with no event time are dropped
  (they cannot be shown to belong to the current encounter). It is applied when the event frame is built, so the single
  t0 gate ``as_of`` keeps its contract. Domains in ``EXEMPT_DOMAINS`` are not subject to it: ``dxlab`` holds
  label-family diagnosis codes used ONLY to define the undifferentiated subgroup (any time before t0), never a model input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EXEMPT_DOMAINS = frozenset({"dxlab"})
BASIS_COHORT, BASIS_VISITS, BASIS_FALLBACK = "cohort", "visits", "fallback"


def encounter_start_from_visits(index: pd.DataFrame, visits: pd.DataFrame | None, cohort_cfg=None) -> pd.Series:
    """Encounter start per index row (NaT when no visit covers t0). ``visits`` is raw ``omop_visit_occurrence`` or the cohort's
    compact visit frame. ``cohort_cfg`` is a ``CohortConfig`` (default: the cohort builder's defaults)."""
    out = pd.Series(pd.NaT, index=index.index, dtype="datetime64[us]")
    if visits is None or not len(visits) or not len(index):
        return out
    from ..cohort.config import CohortConfig
    from ..cohort.rules import match_visits
    cfg = cohort_cfg or CohortConfig()
    sess = pd.DataFrame({"person_id": index["person_id"].astype("int64").to_numpy(),
                         "t0": pd.to_datetime(index["t0"]).astype("datetime64[us]").to_numpy()}, index=index.index)
    m = match_visits(sess, visits, cfg.acute_classes, cfg.visit_chain_gap_h, cfg.visit_slack_h, cfg.open_visit_days,
                     cfg.date_only_end_of_day, dates_only=cfg.visit_dates_only)
    return m["encounter_start"].astype("datetime64[us]")


def resolve_encounter_start(index: pd.DataFrame, given: pd.Series | None, visits: pd.DataFrame | None,
                            fallback_days: float, cohort_cfg=None) -> tuple[pd.Series, pd.Series]:
    """(encounter_start, basis) per index row. Order: a value ``given`` (the cohort table's column), else the visit rule, else
    ``t0 - fallback_days``. The start is never after t0."""
    n = len(index)
    start = pd.Series(pd.NaT, index=index.index, dtype="datetime64[us]")
    basis = pd.Series(BASIS_FALLBACK, index=index.index, dtype=object)
    if given is not None:
        g = pd.Series(pd.to_datetime(np.asarray(given), errors="coerce").astype("datetime64[us]"), index=index.index)
        start = start.where(g.isna(), g)
        basis[g.notna().to_numpy()] = BASIS_COHORT
    need = start.isna()
    if need.any():
        v = encounter_start_from_visits(index[need], visits, cohort_cfg)
        got = v.notna()
        start.loc[v.index[got]] = v[got]
        basis.loc[v.index[got]] = BASIS_VISITS
    t0 = pd.to_datetime(index["t0"]).astype("datetime64[us]")
    fb = (t0 - pd.Timedelta(days=fallback_days)).astype("datetime64[us]")
    start = start.where(start.notna(), fb)
    start = start.where(start <= t0, t0)                                  # never after the index EEG
    return start.astype("datetime64[us]"), basis


def restrict_to_current_encounter(ev: pd.DataFrame, encounter_start: "pd.Series | dict") -> tuple[pd.DataFrame, int]:
    """Keep event rows timed at or after the person's encounter start (exempt domains pass). Returns (events, rows dropped).
    A person with no encounter start, or a row with no event time, loses the row."""
    if ev.empty:
        return ev, 0
    start = ev["person_id"].map(pd.Series(encounter_start) if not isinstance(encounter_start, pd.Series) else encounter_start)
    ok = ev["t_event"].notna() & start.notna() & (ev["t_event"] >= start)
    keep = (ok | ev["domain"].isin(EXEMPT_DOMAINS)).to_numpy(bool)
    return ev[keep].reset_index(drop=True), int((~keep).sum())
