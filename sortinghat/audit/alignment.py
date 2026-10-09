"""Phase 0a date-shift check: nearest-event alignment of the EHR to each candidate's EEG start (D-117).

The earlier proxy (an ancillary table's MEDIAN time more than 30 d from the EEG start) could not detect a date-shift
mismatch: patients have multi-year EHR histories, so the median is naturally far from the EEG. The valid question is
whether SOME event of the same patient sits near the EEG start. For each candidate (first qualifying EEG, ``t0``)
this module keeps, per source table, the signed gap in hours from ``t0`` to the NEAREST row (``event - t0``;
negative = the event is before the EEG), streamed so memory is O(candidates), not O(rows):

* ``visit_start`` / ``visit_end``: omop_visit_occurrence start and end (datetime, else the date twin);
* ``measurement``, ``observation``, ``note``, ``drug_exposure``, ``condition``: the row time (datetime, else the
  date twin; a date-only row sits at 00:00).

It also keeps what the shift-detection test needs (the nearest covering visit: EEG date minus visit start date, and
the visit length in days) and the counts of the secondary ordering checks. Everything is aggregate-only; the
per-candidate frame never leaves the process.

Whole-day gap, used for the constant-offset test: ``trunc(gap_hours / 24)`` (toward zero), so day 0 is "within +-24 h"
and a constant site offset of k days moves the whole distribution to day k.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# source -> (omop table, datetime column, date-twin column)
EVENT_SOURCES = {
    "measurement": ("measurement", "measurement_datetime", "measurement_date"),
    "observation": ("observation", "observation_datetime", "observation_date"),
    "note": ("note", "note_datetime", "note_date"),
    "drug_exposure": ("drug_exposure", "drug_exposure_start_datetime", "drug_exposure_start_date"),
    "condition": ("condition_occurrence", "condition_start_datetime", "condition_start_date"),
}
VISIT_COLS = ["person_id", "visit_start_datetime", "visit_end_datetime", "visit_start_date", "visit_end_date"]
GAP_SOURCES = ("visit_start", "visit_end") + tuple(EVENT_SOURCES)
BIRTH_COLS = ["person_id", "birth_datetime", "year_of_birth"]
DEATH_COLS = ["person_id", "death_datetime", "death_date"]
VISIT_COVER_H = 24.0           # a visit "covers" the EEG when start - 24 h <= t0 <= end + 24 h
DEATH_TOL_H = 24.0             # a row is "after death" when it is more than 24 h after the death time
FIRST_VISIT_TOL_H = 24.0       # EEG start is "before the first visit" when it precedes the earliest visit start by > 24 h
US = np.timedelta64(1, "us")


def _us(x) -> np.ndarray:
    return pd.to_datetime(pd.Series(x), errors="coerce").astype("datetime64[us]").to_numpy()


def event_times(d: pd.DataFrame, dt_col: str, date_col: str) -> np.ndarray:
    """Row times as datetime64[us]: the datetime column, else its date twin; NaT where neither is present."""
    t = pd.to_datetime(d[dt_col], errors="coerce") if dt_col in d else pd.Series(pd.NaT, index=d.index)
    if date_col in d:
        t = t.where(t.notna(), pd.to_datetime(d[date_col], errors="coerce"))
    return t.astype("datetime64[us]").to_numpy()


def life_arrays(person_ids, person: pd.DataFrame | None, death: pd.DataFrame | None):
    """(birth, death_end) as datetime64[us] arrays aligned to ``person_ids``. Birth = ``birth_datetime``, else
    1 January of ``year_of_birth``; death_end = ``death_datetime``, else the END of ``death_date``. NaT = unknown."""
    idx = pd.Index(np.asarray(person_ids, "int64"))
    birth = np.full(len(idx), np.datetime64("NaT", "us"), "datetime64[us]")
    dend = birth.copy()
    if person is not None and len(person) and "person_id" in person:
        p = person.drop_duplicates("person_id").set_index("person_id")
        b = pd.to_datetime(p["birth_datetime"], errors="coerce") if "birth_datetime" in p else \
            pd.Series(pd.NaT, index=p.index)
        if "year_of_birth" in p:
            y = pd.to_numeric(p["year_of_birth"], errors="coerce")
            yb = pd.to_datetime(y.dropna().astype("int64").astype(str) + "-01-01", errors="coerce")
            b = b.where(b.notna(), yb.reindex(b.index))
        birth = b.reindex(idx).astype("datetime64[us]").to_numpy()
    if death is not None and len(death) and "person_id" in death:
        g = death.groupby("person_id")
        dt = pd.to_datetime(g["death_datetime"].max(), errors="coerce") if "death_datetime" in death else None
        dd = (pd.to_datetime(g["death_date"].max(), errors="coerce").dt.normalize() + pd.Timedelta(days=1)
              if "death_date" in death else None)
        e = dt if dt is not None else dd
        if dt is not None and dd is not None:
            e = dt.where(dt.notna(), dd)
        dend = e.reindex(idx).astype("datetime64[us]").to_numpy()
    return birth, dend


def _best_by(pos: np.ndarray, key: np.ndarray):
    """Row indices (into ``pos``) of the row with the smallest ``key`` per distinct position."""
    o = np.lexsort((key, pos))
    p = pos[o]
    first = np.r_[True, p[1:] != p[:-1]] if len(p) else np.zeros(0, bool)
    return o[first]


class NearestGaps:
    """Streaming accumulator of nearest-event gaps for a fixed candidate set. ``add_events`` / ``add_visits`` may be
    called chunk by chunk; different sources may be fed from different threads (each owns its own arrays)."""

    def __init__(self, cands: pd.DataFrame, birth=None, death_end=None):
        self.ids = pd.Index(cands["person_id"].to_numpy("int64"))
        if not self.ids.is_unique:
            raise ValueError("candidates must have one row per person")
        self.t0 = _us(cands["t0"])
        n = len(self.ids)
        self.best = {s: np.full(n, np.nan) for s in GAP_SOURCES}
        self.first_visit_gap = np.full(n, np.nan)       # earliest visit start, signed hours from t0
        self.cov_gap = np.full(n, np.nan)               # |start gap| of the nearest covering visit
        self.cov_off = np.full(n, np.nan)               # EEG date - visit start date, days
        self.cov_len = np.full(n, np.nan)               # visit end date - visit start date, days
        self.birth = np.full(n, np.datetime64("NaT", "us"), "datetime64[us]") if birth is None else birth
        self.death_end = np.full(n, np.datetime64("NaT", "us"), "datetime64[us]") if death_end is None else death_end
        self.order = {s: [0, 0, 0] for s in GAP_SOURCES}   # rows, before birth, after death + tolerance

    # -- internals
    def _gap(self, pid, times):
        pos = self.ids.get_indexer(np.asarray(pid, "int64"))
        t = np.asarray(times, "datetime64[us]")
        ok = (pos >= 0) & ~np.isnat(t)
        pos, t = pos[ok], t[ok]
        return pos, t, (t - self.t0[pos]) / np.timedelta64(1, "h")

    def _take_nearest(self, src: str, pos: np.ndarray, gap: np.ndarray) -> None:
        if not len(pos):
            return
        sel = _best_by(pos, np.abs(gap))
        p, g = pos[sel], gap[sel]
        cur = self.best[src][p]
        upd = np.isnan(cur) | (np.abs(g) < np.abs(cur))
        self.best[src][p[upd]] = g[upd]

    def _count_order(self, src: str, pos: np.ndarray, t: np.ndarray) -> None:
        b, d = self.birth[pos], self.death_end[pos]
        o = self.order[src]
        o[0] += len(pos)
        o[1] += int((~np.isnat(b) & (t < b)).sum())
        o[2] += int((~np.isnat(d) & (t > d + np.timedelta64(int(DEATH_TOL_H * 3600), "s"))).sum())

    # -- public
    def add_events(self, src: str, pid, times) -> None:
        pos, t, gap = self._gap(pid, times)
        self._take_nearest(src, pos, gap)
        self._count_order(src, pos, t)

    def add_visits(self, pid, start, end) -> None:
        pos, ts, gap = self._gap(pid, start)
        self._take_nearest("visit_start", pos, gap)
        self._count_order("visit_start", pos, ts)
        if len(pos):
            sel = _best_by(pos, gap)
            p, g = pos[sel], gap[sel]
            cur = self.first_visit_gap[p]
            upd = np.isnan(cur) | (g < cur)
            self.first_visit_gap[p[upd]] = g[upd]
        pos_e, te, gap_e = self._gap(pid, end)
        self._take_nearest("visit_end", pos_e, gap_e)
        # covering visits (both ends known): start - 24 h <= t0 <= end + 24 h
        s_all = np.asarray(start, "datetime64[us]")
        e_all = np.asarray(end, "datetime64[us]")
        p_all = self.ids.get_indexer(np.asarray(pid, "int64"))
        ok = (p_all >= 0) & ~np.isnat(s_all) & ~np.isnat(e_all)
        p, s, e = p_all[ok], s_all[ok], e_all[ok]
        t0 = self.t0[p]
        h = np.timedelta64(int(VISIT_COVER_H * 3600), "s")
        cov = ~np.isnat(t0) & (s - h <= t0) & (t0 <= e + h)
        p, s, e, t0 = p[cov], s[cov], e[cov], t0[cov]
        if not len(p):
            return
        sgap = np.abs((s - t0) / np.timedelta64(1, "h"))
        sel = _best_by(p, sgap)
        p, s, e, t0, sgap = p[sel], s[sel], e[sel], t0[sel], sgap[sel]
        upd = np.isnan(self.cov_gap[p]) | (sgap < self.cov_gap[p])
        p, s, e, t0, sgap = p[upd], s[upd], e[upd], t0[upd], sgap[upd]
        self.cov_gap[p] = sgap
        self.cov_off[p] = (t0.astype("datetime64[D]") - s.astype("datetime64[D]")).astype("int64")
        self.cov_len[p] = (e.astype("datetime64[D]") - s.astype("datetime64[D]")).astype("int64")

    # -- snapshots (restart safety): the arrays one source owns, as plain copies. ``"visits"`` = visit start / end, the first
    # visit gap and the covering-visit arrays (everything ``add_visits`` writes); any other name is an ``EVENT_SOURCES`` key.
    def get_state(self, src: str) -> dict:
        if src == "visits":
            return {"best": {k: self.best[k].copy() for k in ("visit_start", "visit_end")},
                    "first_visit_gap": self.first_visit_gap.copy(), "cov_gap": self.cov_gap.copy(),
                    "cov_off": self.cov_off.copy(), "cov_len": self.cov_len.copy(), "order": list(self.order["visit_start"])}
        return {"best": {src: self.best[src].copy()}, "order": list(self.order[src])}

    def set_state(self, src: str, st: dict) -> None:
        for k, v in st["best"].items():
            self.best[k][:] = v
        self.order["visit_start" if src == "visits" else src] = list(st["order"])
        if src == "visits":
            for k in ("first_visit_gap", "cov_gap", "cov_off", "cov_len"):
                getattr(self, k)[:] = st[k]

    def frame(self) -> pd.DataFrame:
        """Per-candidate frame indexed by person_id: ``gap_h_<source>`` (signed hours), ``cov_off_days``,
        ``cov_len_days``, ``first_visit_gap_h``. Record-level: used in memory only, never written."""
        d = {f"gap_h_{s}": self.best[s] for s in GAP_SOURCES}
        d.update(cov_off_days=self.cov_off, cov_len_days=self.cov_len, first_visit_gap_h=self.first_visit_gap)
        return pd.DataFrame(d, index=pd.Index(self.ids, name="person_id"))

    def order_frame(self) -> pd.DataFrame:
        return pd.DataFrame({s: v for s, v in self.order.items()},
                            index=["n_rows", "n_before_birth", "n_after_death"]).T


def alignment_from_frames(cands: pd.DataFrame, raw: dict[str, pd.DataFrame]):
    """Run the accumulator over whole in-memory raw OMOP tables (``omop_<table>`` keys). Returns
    ``(gaps frame, ordering-count frame)``."""
    birth, dend = life_arrays(cands["person_id"], raw.get("omop_person"), raw.get("omop_death"))
    acc = NearestGaps(cands, birth, dend)
    for src, (tbl, dt, dd) in EVENT_SOURCES.items():
        d = raw.get("omop_" + tbl)
        if d is not None and len(d) and "person_id" in d:
            acc.add_events(src, d["person_id"], event_times(d, dt, dd))
    v = raw.get("omop_visit_occurrence")
    if v is not None and len(v) and "person_id" in v:
        acc.add_visits(v["person_id"], event_times(v, "visit_start_datetime", "visit_start_date"),
                       event_times(v, "visit_end_datetime", "visit_end_date"))
    return acc.frame(), acc.order_frame()


def nearest_overall(gaps: pd.DataFrame) -> pd.Series:
    """Signed gap (hours) to the nearest event over all sources; NaN where a candidate has no event at all."""
    a = gaps[[f"gap_h_{s}" for s in GAP_SOURCES]].to_numpy(float)
    ab = np.where(np.isnan(a), np.inf, np.abs(a))
    j = ab.argmin(axis=1)
    out = a[np.arange(len(a)), j]
    out[~np.isfinite(ab.min(axis=1))] = np.nan
    return pd.Series(out, index=gaps.index)


def whole_days(gap_h: pd.Series) -> pd.Series:
    """Whole-day gap, truncated toward zero (day 0 = within +-24 h)."""
    return np.trunc(gap_h / 24.0)


def day_peaks(days: pd.Series, n_cands: int, share: float) -> list[int]:
    """Non-zero whole days that are a MODE (a local peak of the day histogram: count at least that of both neighbouring
    days) holding more than ``share`` of the ``n_cands`` candidates. A constant site offset of k days creates exactly
    such a peak at k; the natural tail of day +-1 beside a dominant day 0 is not a peak."""
    vc = days.dropna().astype(int).value_counts()
    out = []
    for d, c in vc.items():
        if d != 0 and c > share * n_cands and c >= vc.get(d - 1, 0) and c >= vc.get(d + 1, 0):
            out.append(int(d))
    return sorted(out)
