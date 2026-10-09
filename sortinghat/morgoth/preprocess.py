"""MORGOTH input preparation from a SortingHat ``Recording`` (uV, canonical 10-20 names, native sampling rate).

Mirrors bdsp-core/morgoth ``utils.ContinuousToSnippetDataset`` + ``finetune_classification`` (commit pinned in
``weights.CODE_COMMIT``):

1. channels in MORGOTH order (``heads.MORGOTH_CHANNELS``); an absent electrode is zero-filled after the common
   average is taken over the electrodes that are present (``--allow_missing_channels yes``);
2. resample to 200 Hz, 4th-order Butterworth band-pass 0.5-70 Hz (zero phase, ``filtfilt``), 50 Hz and 60 Hz notch
   (Q = 50 and 60), applied to the whole fetched segment. For the 1-s spike head a recording above 128 Hz is first
   resampled to 128 Hz and filtered there, then taken to 200 Hz (as in the reference);
3. per snippet (10 s, or 1 s for spikes): common average over present electrodes, clip +-500 uV, per-channel min-max
   scale to [-100, 100], divide by 100 (so the model sees [-1, 1]);
4. a snippet is skipped (not fed to the model) when it is all-NaN, entirely below 2 uV or above 3000 uV in magnitude,
   or constant after the common average, as the reference marks invalid snippets.

Differences from the reference, deliberate: resampling is ``scipy.signal.resample_poly`` (the reference uses
``mne.filter.resample``), and the network runs in float32 (the reference uses CPU bfloat16 autocast).
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy import signal

from .heads import MORGOTH_CHANNELS, SORTINGHAT_ORDER

TARGET_FS = 200.0
SPIKE_FS = 128.0


def to_morgoth_order(rec) -> tuple[np.ndarray, list[str]]:
    """(19, T) uV in MORGOTH channel order (NaN rows where an electrode is absent) and the missing MORGOTH names."""
    idx = {n: i for i, n in enumerate(rec.ch_names)}
    out = np.full((len(SORTINGHAT_ORDER), rec.data.shape[1]), np.nan)
    missing = []
    for r, (sh, mo) in enumerate(zip(SORTINGHAT_ORDER, MORGOTH_CHANNELS)):
        if sh in idx:
            out[r] = rec.data[idx[sh]]
        else:
            missing.append(mo)
    return out, missing


def _resample(x: np.ndarray, fs: float, target: float) -> np.ndarray:
    if abs(fs - target) < 1e-9:
        return x
    fr = Fraction(target / fs).limit_denominator(1000)
    return signal.resample_poly(x, fr.numerator, fr.denominator, axis=-1)


def _bandfilter(x: np.ndarray, fs: float, order: int = 4, low: float = 0.5, high: float = 70.0) -> np.ndarray:
    nyq = 0.5 * fs
    if high / nyq > 1:
        b, a = signal.butter(order, low / nyq, btype="high")
    else:
        b, a = signal.butter(order, [low / nyq, high / nyq], btype="band")
    return signal.filtfilt(b, a, x, axis=-1)


def _notch(x: np.ndarray, fs: float) -> np.ndarray:
    for f0 in (50.0, 60.0):
        if f0 < fs / 2:
            b, a = signal.iirnotch(f0, f0 / 1.0, fs)
            x = signal.filtfilt(b, a, x, axis=-1)
    return x


def filter_series(x: np.ndarray, fs: float, *, spikes: bool = False) -> np.ndarray:
    """Resample to 200 Hz and band-pass / notch ``x`` (C, T); NaN rows (absent channels) stay NaN. ``spikes`` applies
    the extra 128 Hz stage the reference uses for the 1-s spike head."""
    ok = ~np.isnan(x).any(axis=1)
    out = np.full((x.shape[0], int(round(x.shape[1] * TARGET_FS / fs))), np.nan)
    if not ok.any():
        return out
    y = x[ok]
    if spikes and fs > SPIKE_FS:
        y = _resample(y, fs, SPIKE_FS)
        y = _notch(_bandfilter(y, SPIKE_FS), SPIKE_FS)
        fs = SPIKE_FS
    y = _resample(y, fs, TARGET_FS)
    y = _notch(_bandfilter(y, TARGET_FS), TARGET_FS)
    n = min(out.shape[1], y.shape[1])
    out[np.flatnonzero(ok), :n] = y[:, :n]
    return out


def snippet_starts(n_samples: int, win: int, step: int) -> np.ndarray:
    return np.arange(0, n_samples - win + 1, step) if n_samples >= win else np.zeros(0, int)


def snippet_valid(sn: np.ndarray) -> bool:
    """Reference validity rules for one (C, win) snippet after the common average (``sn`` still in uV)."""
    live = sn[~np.isnan(sn).any(axis=1)]
    if live.size == 0:
        return False
    if np.all(np.abs(live) < 2) or np.all(np.abs(live) > 3000):
        return False
    c = live - live.mean(axis=0, keepdims=True)
    return not np.all((c.max(axis=1) - c.min(axis=1)) < 1)


def snippets_valid(sn: np.ndarray) -> np.ndarray:
    """Vectorised ``snippet_valid`` over a stack ``sn`` (B, C, win) in uV -> (B,) bool. Same rules and the same arithmetic."""
    present = ~np.isnan(sn).any(axis=2)                                   # (B, C) electrodes with data
    n = present.sum(axis=1)
    a = np.abs(sn)
    p3 = present[:, :, None]
    small = np.all((a < 2) | ~p3, axis=(1, 2))
    big = np.all((a > 3000) | ~p3, axis=(1, 2))
    mean = np.where(p3, sn, 0.0).sum(axis=1, keepdims=True) / np.maximum(n, 1)[:, None, None]
    c = sn - mean
    rng = np.where(p3, c, -np.inf).max(axis=2) - np.where(p3, c, np.inf).min(axis=2)      # (B, C)
    flat = np.all((rng < 1) | ~present, axis=1)
    return (n > 0) & ~small & ~big & ~flat


def normalise_snippets(x: np.ndarray, starts: np.ndarray, win: int, chunk: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Cut ``x`` (C, T) at ``starts`` -> (S, C, win) in model units ([-1, 1]) and a (S,) validity mask.
    Absent (NaN) channels are zero-filled after the common average over the present ones and then scaled like any
    other (a constant row -> -1), as in the reference. Vectorised over ``chunk`` snippets at a time (no per-snippet
    Python loop); element-wise identical to the per-snippet formulation."""
    S, C = len(starts), x.shape[0]
    out = np.zeros((S, C, win), dtype=np.float32)
    valid = np.zeros(S, bool)
    for a in range(0, S, chunk):
        st = np.asarray(starts[a:a + chunk], dtype=np.int64)
        sn = np.stack([x[:, s:s + win] for s in st])                      # (B, C, win)
        v = snippets_valid(sn)
        valid[a:a + len(st)] = v
        if not v.any():
            continue
        sn = sn[v]
        present = ~np.isnan(sn).any(axis=2)[:, :, None]
        n = present.sum(axis=1, keepdims=True)
        mean = np.where(present, sn, 0.0).sum(axis=1, keepdims=True) / n
        y = np.where(present, sn - mean, 0.0)
        y = np.clip(y, -500.0, 500.0)
        lo, hi = y.min(axis=2, keepdims=True), y.max(axis=2, keepdims=True)
        span = np.where(hi - lo > 0, hi - lo, 1.0)
        out[a:a + len(st)][v] = ((y - lo) / span * 200.0 - 100.0) / 100.0       # MinMaxScaler((-100, 100)) then / 100
    return out, valid
