"""Index time, analysis windows and usable-data QC (research plan, Study 1).

* t0 = recording start. Primary window = minutes 1-11 (60-660 s).
* Nested windows start at the primary window start: 20 s, 1, 2, 5, 10 min.
* QC runs on a grid of 2-s epochs. Per channel x epoch flags: flat, clipping, extreme amplitude (>500 uV),
  high line noise, disconnected. An epoch is *usable* when enough of the minimum channel set is clean.
  The window passes when >= 60% of its (intended) duration is usable. Epochs outside the recording count as
  unusable, and masks are never compressed (a compressed mask glues time together; docs/heedb_access.md rule 27).

QC runs on the *unfiltered* uV signal (line noise must be visible), before notch/band-pass.
Only aggregate summaries leave the human-run job (``summarize_window_qc`` goes through ``sortinghat.safe_output``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Mapping, Sequence

import numpy as np

from ..safe_output import (assert_aggregate_only, safe_quantiles, suppress_count, suppress_proportion,
                           safe_write_json)
from .io import DEFAULT_MINIMUM_CHANNELS

EPOCH_S = 2.0
PRIMARY_START_S = 60.0
PRIMARY_DURATION_S = 600.0
NESTED_DURATIONS_S = {"20s": 20.0, "1min": 60.0, "2min": 120.0, "5min": 300.0, "10min": 600.0}
USABLE_THRESHOLD = 0.60
FLAG_NAMES = ("flat", "clipping", "extreme", "line_noise", "disconnected")


@dataclass(frozen=True)
class WindowSpec:
    name: str
    start_s: float
    duration_s: float

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s


def primary_window() -> WindowSpec:
    return WindowSpec("primary", PRIMARY_START_S, PRIMARY_DURATION_S)


def nested_windows(primary: WindowSpec | None = None) -> dict[str, WindowSpec]:
    """20 s, 1, 2, 5, 10 min windows, all starting at the primary window start."""
    p = primary or primary_window()
    return {k: WindowSpec(k, p.start_s, d) for k, d in NESTED_DURATIONS_S.items()}


def all_windows() -> dict[str, WindowSpec]:
    return {"primary": primary_window(), **nested_windows()}


@dataclass
class QCConfig:
    epoch_s: float = EPOCH_S
    flat_ptp_uv: float = 0.5            # peak-to-peak below this: flat
    clip_fraction: float = 0.05         # >= this share of samples sitting on the epoch max or min: clipping
    clip_tol_uv: float = 1e-6
    extreme_uv: float = 500.0           # max |x - median| above this: extreme amplitude
    line_hz: float = 60.0
    line_ratio: float = 1.0             # power(line +-1 Hz) / power(1-40 Hz) above this: high line noise
    disconnected_rel_std: float = 0.02  # epoch std < this x median across channels: disconnected
    disconnected_epoch_frac: float = 0.5  # share of a channel's epochs (in the span) with flat/clip/line flags
    epoch_channel_frac: float = 0.9     # share of minimum-set channels that must be clean for a usable epoch
    usable_threshold: float = USABLE_THRESHOLD
    minimum_channels: tuple[str, ...] = DEFAULT_MINIMUM_CHANNELS


@dataclass
class EpochFlags:
    """Per channel x epoch boolean flags over a span starting at ``start_s``."""
    ch_names: list[str]
    start_s: float
    epoch_s: float
    flags: dict[str, np.ndarray]         # name -> (C, E) bool
    n_epochs_expected: int

    @property
    def bad(self) -> np.ndarray:
        out = np.zeros_like(next(iter(self.flags.values())))
        for v in self.flags.values():
            out |= v
        return out

    @property
    def clean(self) -> np.ndarray:       # (C, E) True = clean cell
        return ~self.bad


def epoch_artifact_flags(data: np.ndarray, fs: float, ch_names: Sequence[str], start_s: float = 0.0,
                         cfg: QCConfig | None = None, span_s: float | None = None) -> EpochFlags:
    """Flag artifacts on non-overlapping ``cfg.epoch_s`` epochs of ``data`` (C, N) in uV.

    ``start_s`` is the time of sample 0. ``span_s`` (default: data length) fixes the expected epoch count so
    that a short recording yields trailing all-bad epochs rather than silently fewer epochs.
    """
    cfg = cfg or QCConfig()
    C, N = data.shape
    L = int(round(cfg.epoch_s * fs))
    n_have = N // L
    n_exp = int(math.floor((span_s if span_s is not None else N / fs) / cfg.epoch_s + 1e-9))
    n_exp = max(n_exp, n_have) if span_s is None else n_exp
    flags = {k: np.zeros((C, n_exp), bool) for k in FLAG_NAMES}
    nh = min(n_have, n_exp)
    if nh:
        ep = data[:, : nh * L].reshape(C, nh, L)
        nanrow = np.isnan(ep).any(axis=2)
        ep = np.nan_to_num(ep)
        ptp = ep.max(axis=2) - ep.min(axis=2)
        flat = ptp < cfg.flat_ptp_uv
        mx = ep.max(axis=2, keepdims=True)
        mn = ep.min(axis=2, keepdims=True)
        at_ext = (np.abs(ep - mx) <= cfg.clip_tol_uv) | (np.abs(ep - mn) <= cfg.clip_tol_uv)
        clip = (at_ext.mean(axis=2) >= cfg.clip_fraction) & ~flat
        dev = np.abs(ep - np.median(ep, axis=2, keepdims=True)).max(axis=2)
        extreme = dev > cfg.extreme_uv
        line = np.zeros_like(flat)
        if fs / 2 > cfg.line_hz + 2:
            w = np.hanning(L)
            P = np.abs(np.fft.rfft((ep - ep.mean(axis=2, keepdims=True)) * w, axis=2)) ** 2
            f = np.fft.rfftfreq(L, 1.0 / fs)
            bl = (f >= cfg.line_hz - 1) & (f <= cfg.line_hz + 1)
            bb = (f >= 1) & (f <= 40)
            denom = P[:, :, bb].sum(axis=2)
            line = (P[:, :, bl].sum(axis=2) > cfg.line_ratio * np.maximum(denom, 1e-12)) & ~flat
        sd = ep.std(axis=2)
        med = np.median(sd, axis=0, keepdims=True)
        low = (sd < cfg.disconnected_rel_std * med) & (med > 1.0)
        flags["flat"][:, :nh] = flat | nanrow
        flags["clipping"][:, :nh] = clip
        flags["extreme"][:, :nh] = extreme
        flags["line_noise"][:, :nh] = line
        flags["disconnected"][:, :nh] = low
    # persistent disconnection: a channel mostly flat / clipped / line-dominated over the whole span
    if nh:
        persistent = (flags["flat"] | flags["clipping"] | flags["line_noise"])[:, :nh].mean(axis=1) \
            >= cfg.disconnected_epoch_frac
        flags["disconnected"][persistent, :] = True
    # epochs past the end of the recording: every channel bad (counted as 'flat' = no signal)
    if n_exp > nh:
        flags["flat"][:, nh:] = True
    return EpochFlags(list(ch_names), float(start_s), cfg.epoch_s, flags, n_exp)


def line_noise_ratio(ep: np.ndarray, fs: float, cfg: QCConfig | None = None) -> np.ndarray:
    """(C, E) ratio power(line +-1 Hz) / power(1-40 Hz) per channel x epoch for ``ep`` of shape (C, E, L): the quantity
    ``epoch_artifact_flags`` compares with ``cfg.line_ratio``. Zeros when the sampling rate is too low to see the line."""
    cfg = cfg or QCConfig()
    C, E, L = ep.shape
    if not (fs / 2 > cfg.line_hz + 2) or L == 0:
        return np.zeros((C, E))
    ep = np.nan_to_num(ep)
    P = np.abs(np.fft.rfft((ep - ep.mean(axis=2, keepdims=True)) * np.hanning(L), axis=2)) ** 2
    f = np.fft.rfftfreq(L, 1.0 / fs)
    bl = (f >= cfg.line_hz - 1) & (f <= cfg.line_hz + 1)
    bb = (f >= 1) & (f <= 40)
    return P[:, :, bl].sum(axis=2) / np.maximum(P[:, :, bb].sum(axis=2), 1e-12)


@dataclass
class WindowQC:
    window: str
    n_epochs: int
    usable_fraction: float
    passes: bool
    epoch_usable: np.ndarray                    # (E,) bool, window epoch grid (not compressed)
    flag_fraction: dict[str, float]             # share of minimum-set channel x epoch cells flagged, per flag
    clean_cell_fraction: float
    n_min_present: int
    n_min_required: int
    coverage_fraction: float                    # share of the window inside the recording
    reasons: list[str] = field(default_factory=list)
    disconnected_channels: int = 0              # minimum-set channels disconnected for the whole window
    n_dead_min: int = 0                         # minimum-set channels exactly constant for the whole segment (missing)
    n_invalid_min: int = 0                      # minimum-set channels with a zero calibration range (missing)

    def to_row(self) -> dict:
        """Scalar summary (no per-epoch arrays); safe to aggregate."""
        d = {"window": self.window, "usable_fraction": self.usable_fraction, "passes": self.passes,
             "clean_cell_fraction": self.clean_cell_fraction, "coverage_fraction": self.coverage_fraction,
             "n_disconnected": self.disconnected_channels}
        d.update({f"flag_{k}": v for k, v in self.flag_fraction.items()})
        return d


def window_qc(ef: EpochFlags, spec: WindowSpec, rec_duration_s: float, cfg: QCConfig | None = None,
              channel_notes: Mapping[str, Sequence[str]] | None = None) -> WindowQC:
    """Usable-data fraction for ``spec`` from epoch flags (grid must contain the window).

    ``channel_notes`` (``dead`` / ``invalid_scaling``: canonical names already removed from the data) only explains
    WHY minimum-set channels are missing; they are not in ``ef`` and count as bad cells either way."""
    cfg = cfg or QCConfig()
    e0 = int(round((spec.start_s - ef.start_s) / ef.epoch_s))
    n = int(round(spec.duration_s / ef.epoch_s))
    if e0 < 0 or e0 + n > ef.n_epochs_expected:
        raise ValueError("window is outside the flagged span")
    idx = {c: i for i, c in enumerate(ef.ch_names)}
    present = [c for c in cfg.minimum_channels if c in idx]
    reasons: list[str] = []
    coverage = max(0.0, min(spec.end_s, rec_duration_s) - spec.start_s) / spec.duration_s
    if coverage < 1.0:
        reasons.append("window_exceeds_recording")
    notes = channel_notes or {}
    n_dead = len(set(cfg.minimum_channels) & set(notes.get("dead", ())))
    n_inv = len(set(cfg.minimum_channels) & set(notes.get("invalid_scaling", ())))
    if len(present) < len(cfg.minimum_channels):
        reasons.append("missing_minimum_channels")
    if n_dead:
        reasons.append("dead_minimum_channels")
    if n_inv:
        reasons.append("invalid_scaling_minimum_channels")
    rows = [idx[c] for c in present]
    sl = slice(e0, e0 + n)
    cells = {k: v[rows][:, sl] for k, v in ef.flags.items()}
    bad = np.zeros((len(rows), n), bool)
    for v in cells.values():
        bad |= v
    # absent minimum channels count as bad cells
    n_req = len(cfg.minimum_channels)
    clean_per_epoch = (~bad).sum(axis=0)
    need = math.ceil(cfg.epoch_channel_frac * n_req - 1e-9)
    usable = clean_per_epoch >= need
    uf = float(usable.mean()) if n else 0.0
    # past-the-end epochs already flagged flat by epoch_artifact_flags; enforce for safety
    t_ep = ef.start_s + (np.arange(e0, e0 + n) + 1) * ef.epoch_s
    usable &= t_ep <= rec_duration_s + 1e-6
    uf = float(usable.mean()) if n else 0.0
    ffrac = {k: float(v.mean()) if v.size else float("nan") for k, v in cells.items()}
    disc_ch = int(cells["disconnected"].all(axis=1).sum()) if cells["disconnected"].size else 0
    passes = uf >= cfg.usable_threshold and "missing_minimum_channels" not in reasons
    if uf < cfg.usable_threshold:
        reasons.append("usable_below_threshold")
    return WindowQC(spec.name, n, uf, bool(passes), usable, ffrac, float((~bad).mean()) if bad.size else 0.0,
                    len(present), n_req, coverage, reasons, disc_ch, n_dead, n_inv)


def qc_recording(data: np.ndarray, fs: float, ch_names: Sequence[str], offset_s: float = 0.0,
                 windows: Mapping[str, WindowSpec] | None = None, cfg: QCConfig | None = None,
                 rec_duration_s: float | None = None,
                 channel_notes: Mapping[str, Sequence[str]] | None = None) -> tuple[dict[str, WindowQC], EpochFlags]:
    """Run QC for every window in one pass. ``data`` starts at ``offset_s`` of the recording."""
    cfg = cfg or QCConfig()
    windows = dict(windows or all_windows())
    span0 = min(w.start_s for w in windows.values())
    span1 = max(w.end_s for w in windows.values())
    s0 = int(round((span0 - offset_s) * fs))
    s1 = int(round((span1 - offset_s) * fs))
    if s0 < 0:
        raise ValueError("data starts after the first window")
    seg = data[:, s0:s1]
    ef = epoch_artifact_flags(seg, fs, ch_names, start_s=span0, cfg=cfg, span_s=span1 - span0)
    total = rec_duration_s if rec_duration_s is not None else offset_s + data.shape[1] / fs
    return {k: window_qc(ef, w, total, cfg, channel_notes) for k, w in windows.items()}, ef


def extract_window(data: np.ndarray, fs: float, spec: WindowSpec, offset_s: float = 0.0) -> np.ndarray:
    """Slice ``spec`` from ``data`` (starting at ``offset_s``). Shorter than requested if the recording ends."""
    a = int(round((spec.start_s - offset_s) * fs))
    b = int(round((spec.end_s - offset_s) * fs))
    return data[:, max(0, a): max(0, b)]


# --------------------------------------------------------------------------------------------
# Aggregate QC summary (the only QC output that may leave a restricted-data job)
# --------------------------------------------------------------------------------------------
def summarize_window_qc(qcs: Sequence[Mapping[str, WindowQC]], threshold: float = USABLE_THRESHOLD) -> dict:
    """Aggregate one-``WindowQC``-dict-per-recording into suppressed counts, proportions and quantiles.

    Per window: n recordings, pass proportion, usable-fraction quantiles (n>=11 else suppressed), and the
    share of recordings in which each artifact flag touches >5% of minimum-set cells. Never returns
    per-recording values or min/max.
    """
    out: dict = {"n_recordings": suppress_count(len(qcs)), "usable_threshold": threshold, "windows": {}}
    names = sorted({w for q in qcs for w in q})
    for w in names:
        rows = [q[w] for q in qcs if w in q]
        n = len(rows)
        npass = sum(r.passes for r in rows)
        entry = {"n": suppress_count(n),
                 "pass_proportion": suppress_proportion(npass, n),
                 "usable_fraction_quantiles": safe_quantiles([r.usable_fraction for r in rows]),
                 "flag_prevalence": {},
                 "reason_counts": {r: suppress_count(sum(r in x.reasons for x in rows))
                                   for r in sorted({r for x in rows for r in x.reasons})}}
        for k in FLAG_NAMES:
            entry["flag_prevalence"][k] = suppress_proportion(sum(r.flag_fraction.get(k, 0) > 0.05 for r in rows), n)
        out["windows"][w] = entry
    assert_aggregate_only(out)
    return out


def write_qc_summary(path, qcs: Sequence[Mapping[str, WindowQC]]):
    """Write ``summarize_window_qc`` to JSON through ``safe_write_json``."""
    return safe_write_json(path, summarize_window_qc(qcs))


def window_clean_mask(ef: EpochFlags, spec: WindowSpec, ch_names: Sequence[str]) -> np.ndarray:
    """(C, E_window) per-channel clean-epoch mask for ``spec``, rows ordered as ``ch_names`` (channels absent from
    the flags are all-False). Feed this to ``features.extract_features``."""
    e0 = int(round((spec.start_s - ef.start_s) / ef.epoch_s))
    n = int(round(spec.duration_s / ef.epoch_s))
    idx = {c: i for i, c in enumerate(ef.ch_names)}
    out = np.zeros((len(ch_names), n), bool)
    clean = ef.clean
    for r, c in enumerate(ch_names):
        if c in idx:
            out[r] = clean[idx[c], e0:e0 + n]
    return out
