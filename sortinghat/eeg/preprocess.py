"""Preprocessing: resample -> notch -> band-pass, plus common-average and double-banana derivations.

Target rate is 200 Hz, CBraMod's input rate (docs/research/cbramod_provenance.md: 19 channels, 60 Hz notch,
resampled to 200 Hz, band-pass 0.3-75 Hz). Study 1 band-pass defaults to 0.5-45 Hz and is configurable
(``PreprocessConfig.band``). Units stay in uV; there is NO per-channel z-scoring (it destroyed the amplitude scale
in the earlier programme, docs/heedb_access.md section 4). CBraMod's own /100 input scaling belongs to the
embedding step, not here.

Filtering is zero-phase (``sosfiltfilt``), so filter a padded interval and trim (``pipeline`` pads 10 s).
Notch and band-pass are applied to NaN-free data only; NaN rows (missing channels filled by ``select_channels``)
pass through as NaN.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

import numpy as np
from scipy import signal

TARGET_FS = 200.0

# 18-channel longitudinal bipolar ("double banana") chains, anode - cathode.
DOUBLE_BANANA: tuple[tuple[str, str], ...] = (
    ("Fp1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
    ("Fp2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
    ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
    ("Fz", "Cz"), ("Cz", "Pz"),
)


@dataclass
class PreprocessConfig:
    target_fs: float = TARGET_FS
    notch_hz: float | None = 60.0
    notch_q: float = 30.0
    band: tuple[float, float] | None = (0.5, 45.0)
    filter_order: int = 4


def resample(data: np.ndarray, fs: float, target_fs: float = TARGET_FS) -> np.ndarray:
    """Polyphase resample (anti-aliased) along the last axis. Returns a copy when rates already match."""
    if abs(fs - target_fs) < 1e-9:
        return np.array(data, dtype=np.float64, copy=True)
    fr = Fraction(target_fs / fs).limit_denominator(1000)
    return signal.resample_poly(data, fr.numerator, fr.denominator, axis=-1)


def _apply(sos: np.ndarray, data: np.ndarray) -> np.ndarray:
    out = np.array(data, dtype=np.float64, copy=True)
    ok = ~np.isnan(out).any(axis=-1)
    if ok.any():
        out[ok] = signal.sosfiltfilt(sos, out[ok], axis=-1)
    return out


def notch(data: np.ndarray, fs: float, freq: float = 60.0, q: float = 30.0) -> np.ndarray:
    if freq >= fs / 2:
        return np.array(data, dtype=np.float64, copy=True)
    b, a = signal.iirnotch(freq, q, fs=fs)
    return _apply(signal.tf2sos(b, a), data)


def bandpass(data: np.ndarray, fs: float, band: tuple[float, float] = (0.5, 45.0), order: int = 4) -> np.ndarray:
    lo, hi = band
    hi = min(hi, 0.99 * fs / 2)
    sos = signal.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return _apply(sos, data)


def preprocess(data: np.ndarray, fs: float, cfg: PreprocessConfig | None = None) -> tuple[np.ndarray, float]:
    """resample -> notch -> band-pass. Returns (data, fs_out)."""
    cfg = cfg or PreprocessConfig()
    x = resample(data, fs, cfg.target_fs)
    if cfg.notch_hz:
        x = notch(x, cfg.target_fs, cfg.notch_hz, cfg.notch_q)
    if cfg.band:
        x = bandpass(x, cfg.target_fs, cfg.band, cfg.filter_order)
    return x, cfg.target_fs


def common_average(data: np.ndarray, exclude: Sequence[bool] | np.ndarray | None = None) -> np.ndarray:
    """Common-average reference. Channels flagged in ``exclude`` (e.g. disconnected) and NaN rows are left out of
    the reference mean but still re-referenced (so a bad channel cannot contaminate the others)."""
    use = ~np.isnan(data).any(axis=1)
    if exclude is not None:
        use &= ~np.asarray(exclude, bool)
    if not use.any():
        return np.array(data, copy=True)
    return data - data[use].mean(axis=0, keepdims=True)


def bipolar_double_banana(data: np.ndarray, ch_names: Sequence[str],
                          pairs: Sequence[tuple[str, str]] = DOUBLE_BANANA) -> tuple[np.ndarray, list[str]]:
    """Longitudinal bipolar derivations from referential data. Pairs whose electrodes are absent are skipped."""
    idx = {c: i for i, c in enumerate(ch_names)}
    rows, names = [], []
    for a, b in pairs:
        if a in idx and b in idx:
            rows.append(data[idx[a]] - data[idx[b]])
            names.append(f"{a}-{b}")
    return (np.vstack(rows) if rows else np.zeros((0, data.shape[1]))), names
