"""The single t0 gate: ``as_of``.

Every clinical event enters the baselines as a row of the long *event frame* (``EVENT_COLUMNS``) with an
**availability time** ``t_avail`` (when a clinician could have seen it). ``as_of(events, t0)`` is the ONLY place
in the package that compares an event with t0. It

1. keeps rows with ``t_avail <= t0`` (a row with unknown availability is dropped, never assumed early);
2. censors interval ends: an administration whose recorded end is after t0 is returned as ``ongoing`` with
   ``t_end`` removed and ``quantity`` set to NaN (the total dose and the stop time are future information);
3. attaches ``t0`` and ``hours_since_event`` / ``hours_since_avail`` so extractors never touch t0 again;
4. drops ``t_end_raw`` so no later step can read an uncensored end.

Feature extractors accept only the frame returned here and re-verify the invariant with ``assert_masked``.
Static fields (age, sex, referral indication) are not events and are read from the index table.

One bounded exception (D-145), ``as_of_presentation``: Baseline P reads the GCS / FOUR / RASS nearest to t0 in
[t0 - 6 h, t0 + ``after_h``] (default 1 h), the cohort's own strict-severity window (D-105). It is a second call of the same
gate with t0 shifted by ``after_h``, restricted to the ``score`` domain; every other domain, and every other baseline, stays
masked at t0. Set ``presentation_score_after_h = 0`` for a strictly t0-masked P.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

EVENT_COLUMNS = ["person_id", "domain", "key", "value", "quantity", "t_event", "t_avail", "t_end_raw",
                 "time_basis", "approx", "unit", "raw_name"]
MASKED_COLUMNS = ["person_id", "domain", "key", "value", "quantity", "t_event", "t_avail", "t_end", "ongoing",
                  "time_basis", "approx", "t0", "hours_since_event", "hours_since_avail", "raw_name"]


def empty_events() -> pd.DataFrame:
    return pd.DataFrame({
        "person_id": pd.Series(dtype="int64"), "domain": pd.Series(dtype=object), "key": pd.Series(dtype=object),
        "value": pd.Series(dtype=float), "quantity": pd.Series(dtype=float),
        "t_event": pd.Series(dtype="datetime64[us]"), "t_avail": pd.Series(dtype="datetime64[us]"),
        "t_end_raw": pd.Series(dtype="datetime64[us]"), "time_basis": pd.Series(dtype=object),
        "approx": pd.Series(dtype=bool), "unit": pd.Series(dtype=object), "raw_name": pd.Series(dtype=object)})


def _t0_series(person_id: pd.Series, t0: "pd.Series | Mapping | pd.Timestamp") -> pd.Series:
    if isinstance(t0, pd.Timestamp):
        return pd.Series(t0, index=person_id.index)
    m = t0 if isinstance(t0, pd.Series) else pd.Series(dict(t0))
    return person_id.map(m)


def as_of(events: pd.DataFrame, t0: "pd.Series | Mapping | pd.Timestamp") -> pd.DataFrame:
    """Return the events a model may use for a patient indexed at ``t0`` (per-person Series/mapping or one
    Timestamp). See module docstring for the exact contract. Patients without a t0 get no rows."""
    ev = events
    if ev.empty:
        out = pd.DataFrame({c: pd.Series(dtype=ev[c].dtype if c in ev else object) for c in MASKED_COLUMNS})
        out.attrs["as_of"] = True
        return out
    t0_row = _t0_series(ev["person_id"], t0)
    keep = ev["t_avail"].notna() & t0_row.notna() & (ev["t_avail"] <= t0_row)
    ev = ev.loc[keep].copy()
    t0_row = t0_row.loc[keep]
    ev["t0"] = pd.to_datetime(t0_row).astype("datetime64[us]")
    has_end = ev["t_end_raw"].notna()
    ongoing = has_end & (ev["t_end_raw"] > ev["t0"])                 # stop not yet recorded at t0
    ev["ongoing"] = ongoing
    ev["t_end"] = ev["t_end_raw"].where(has_end & ~ongoing)          # only ends that happened by t0 survive
    ev["quantity"] = ev["quantity"].where(~ongoing, np.nan)          # total dose of a running infusion is unknown
    ev["hours_since_event"] = (ev["t0"] - ev["t_event"]).dt.total_seconds() / 3600.0
    ev["hours_since_avail"] = (ev["t0"] - ev["t_avail"]).dt.total_seconds() / 3600.0
    out = ev[MASKED_COLUMNS].reset_index(drop=True)
    out.attrs["as_of"] = True
    return out


def assert_masked(masked: pd.DataFrame) -> pd.DataFrame:
    """Raise unless ``masked`` came out of ``as_of`` and still satisfies the t0 invariants."""
    if not masked.attrs.get("as_of") and not masked.empty:
        raise ValueError("feature extractors accept only the output of as_of(events, t0)")
    if masked.empty:
        return masked
    if "t_end_raw" in masked.columns:
        raise ValueError("uncensored interval end present; pass the as_of() output")
    if (masked["t_avail"] > masked["t0"]).any():
        raise AssertionError("event available after t0 reached a feature extractor")
    if (masked["t_end"].notna() & (masked["t_end"] > masked["t0"])).any():
        raise AssertionError("interval end after t0 reached a feature extractor")
    return masked


PRESENTATION_DOMAINS = ("score",)


def as_of_presentation(events: pd.DataFrame, t0: "pd.Series | Mapping | pd.Timestamp", after_h: float) -> pd.DataFrame:
    """The ``score``-domain events a presentation exam may use: available by ``t0 + after_h`` (see module docstring).

    Same contract as ``as_of`` with the gate moved to ``t0 + after_h``, then ``t0`` is restored to the true index time so
    ``hours_since_event`` is negative for a score charted after the EEG start. Only ``PRESENTATION_DOMAINS`` rows pass."""
    ev = events[events["domain"].isin(PRESENTATION_DOMAINS)]
    shift = pd.Timedelta(hours=float(after_h))
    shifted = (t0 + shift) if isinstance(t0, pd.Timestamp) else (pd.Series(t0) + shift)
    out = as_of(ev, shifted)
    out.attrs["as_of_presentation_after_h"] = float(after_h)
    if out.empty:
        return out
    out["t0"] = (out["t0"] - shift).astype("datetime64[us]")
    out["hours_since_event"] = (out["t0"] - out["t_event"]).dt.total_seconds() / 3600.0
    out["hours_since_avail"] = (out["t0"] - out["t_avail"]).dt.total_seconds() / 3600.0
    out.attrs["as_of"] = False
    out.attrs["as_of_presentation_after_h"] = float(after_h)
    return out


def assert_presentation_masked(masked: pd.DataFrame, after_h: float) -> pd.DataFrame:
    """Raise unless ``masked`` came out of ``as_of_presentation`` and respects its bounds."""
    if masked.attrs.get("as_of_presentation_after_h") is None and not masked.empty:
        raise ValueError("presentation extractors accept only the output of as_of_presentation")
    if masked.empty:
        return masked
    if "t_end_raw" in masked.columns:
        raise ValueError("uncensored interval end present")
    if not masked["domain"].isin(PRESENTATION_DOMAINS).all():
        raise AssertionError("a non-score event reached the presentation extractor")
    if (masked["t_avail"] > masked["t0"] + pd.Timedelta(hours=float(after_h))).any():
        raise AssertionError("event available after t0 + after_h reached the presentation extractor")
    return masked
