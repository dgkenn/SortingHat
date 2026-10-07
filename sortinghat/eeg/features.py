"""qEEG rung of the representation ladder (prior -> qEEG -> connectivity -> ...).

``extract_features`` maps one preprocessed window (channels x samples, uV, 200 Hz, common-average referenced,
band-passed) to one flat vector of named features. Names are stable and always present (NaN when a region/pair
has no clean data), so vectors from different recordings and windows line up column-for-column.

Spectra use 4-s Hann segments with 50% overlap (0.25 Hz resolution). A segment contributes to a channel only if
all of its 2-s QC epochs are clean for that channel (``clean_mask``), so artifacts and flat stretches never enter
power, BSR, complexity or connectivity. Naming:

  qeeg.<scope>.<metric>          scope in global, frontal, central, temporal, parietal, occipital (+ ch_<name>)
  qeeg.global.bsr                burst-suppression ratio (median over channels)
  qeeg.asym.<band>_absdiff_log10 mean |log10(L/R)| band power over homologous pairs
  conn.<coh|wpli>.<pair>.<band>  homologous / anteroposterior pairs; conn.<m>.mean_<set>.<band> summaries

Absolute power is log10(uV^2); relative power uses 0.5-30 Hz total; SEF95/SEF50 are over 0.5-40 Hz.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.stats import kurtosis

from .io import CANONICAL_19

BANDS: dict[str, tuple[float, float]] = {"delta": (0.5, 4.0), "theta": (4.0, 8.0),
                                         "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
REGIONS: dict[str, tuple[str, ...]] = {
    "frontal": ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"),
    "central": ("C3", "Cz", "C4"),
    "temporal": ("T3", "T4", "T5", "T6"),
    "parietal": ("P3", "Pz", "P4"),
    "occipital": ("O1", "O2"),
}
HOMOLOGOUS_PAIRS = (("Fp1", "Fp2"), ("F7", "F8"), ("F3", "F4"), ("C3", "C4"),
                    ("T3", "T4"), ("P3", "P4"), ("T5", "T6"), ("O1", "O2"))
ANTEROPOSTERIOR_PAIRS = (("Fp1", "O1"), ("Fp2", "O2"), ("F3", "P3"), ("F4", "P4"), ("F7", "T5"), ("F8", "T6"))
CONN_PAIRS = HOMOLOGOUS_PAIRS + ANTEROPOSTERIOR_PAIRS


@dataclass
class FeatureConfig:
    bands: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(BANDS))
    epoch_s: float = 2.0
    seg_s: float = 4.0                 # Welch / cross-spectrum segment (multiple of epoch_s)
    rel_range: tuple[float, float] = (0.5, 30.0)
    sef_range: tuple[float, float] = (0.5, 40.0)
    bsr_threshold_uv: float = 5.0      # ASSUMPTION: RMS envelope below this = suppressed (calibrate on pilot)
    bsr_env_s: float = 0.25
    bsr_min_s: float = 0.5
    lzc_max_segments: int = 10
    per_channel: bool = False
    connectivity: bool = True
    min_segments: int = 5              # fewer clean segments than this -> NaN


# ---------------------------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------------------------
def _scope_metrics(cfg: FeatureConfig) -> list[str]:
    m = [f"{b}_abs_log10" for b in cfg.bands] + [f"{b}_rel" for b in cfg.bands]
    m += ["sef95", "sef50", "alpha_delta_log10", "slowing_log10",
          "amp_rms", "amp_p95", "amp_kurtosis", "line_length", "lzc"]
    return m


def _pair_name(p: tuple[str, str]) -> str:
    return f"{p[0]}_{p[1]}"


def feature_names(cfg: FeatureConfig | None = None, ch_names: Sequence[str] = CANONICAL_19) -> list[str]:
    cfg = cfg or FeatureConfig()
    scopes = ["global", *REGIONS] + ([f"ch_{c}" for c in ch_names] if cfg.per_channel else [])
    names = [f"qeeg.{s}.{m}" for s in scopes for m in _scope_metrics(cfg)]
    names.append("qeeg.global.bsr")
    names += [f"qeeg.asym.{b}_absdiff_log10" for b in ("delta", "alpha")]
    if cfg.connectivity:
        for meas in ("coh", "wpli"):
            for p in CONN_PAIRS:
                names += [f"conn.{meas}.{_pair_name(p)}.{b}" for b in cfg.bands]
            for grp in ("all", "interhemispheric", "anteroposterior"):
                names += [f"conn.{meas}.mean_{grp}.{b}" for b in cfg.bands]
    return names


# ---------------------------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------------------------
def segment_clean_mask(clean_mask: np.ndarray, n_seg: int, epochs_per_seg: int) -> np.ndarray:
    """(C, S) bool: segment s (covering epochs s..s+m-1) is clean iff all its epochs are clean."""
    C, E = clean_mask.shape
    out = np.ones((C, n_seg), bool)
    for k in range(epochs_per_seg):
        out &= clean_mask[:, k:k + n_seg]
    return out


def _segments(x: np.ndarray, L_ep: int, m: int) -> np.ndarray:
    """(C, S, m*L_ep) overlapping segments with hop = one epoch."""
    C, N = x.shape
    E = N // L_ep
    S = E - m + 1
    if S < 1:
        return np.zeros((C, 0, m * L_ep))
    idx = np.arange(S)[:, None] * L_ep + np.arange(m * L_ep)[None, :]
    return x[:, : E * L_ep][:, idx]


def lempel_ziv(seq: Sequence[int]) -> int:
    """Lempel-Ziv (1976) complexity count, Kaspar-Schuster algorithm, on a 0/1 sequence."""
    n = len(seq)
    if n < 2:
        return n
    i, k, l, c, kmax = 0, 1, 1, 1, 1
    while True:
        if seq[i + k - 1] == seq[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > kmax:
                kmax = k
            i += 1
            if i == l:
                c += 1
                l += kmax
                if l + 1 > n:
                    break
                i, k, kmax = 0, 1, 1
            else:
                k = 1
    return c


def lz_complexity_norm(x: np.ndarray) -> float:
    """Normalised LZ complexity of a signal binarised about its median: c * log2(n) / n (~1 for random)."""
    x = np.asarray(x)
    n = len(x)
    if n < 8:
        return float("nan")
    bits = (x > np.median(x)).astype(np.int8).tolist()
    return lempel_ziv(bits) * np.log2(n) / n


def burst_suppression_ratio(x: np.ndarray, fs: float, clean: np.ndarray | None, threshold_uv: float,
                            env_s: float = 0.25, min_s: float = 0.5) -> float:
    """Fraction of clean time in suppression: RMS envelope < ``threshold_uv`` sustained >= ``min_s``."""
    N = len(x)
    if clean is None:
        clean = np.ones(N, bool)
    if clean.sum() == 0:
        return float("nan")
    w = max(1, int(round(env_s * fs)))
    env = np.sqrt(np.maximum(uniform_filter1d(np.nan_to_num(x) ** 2, size=w, mode="nearest"), 0))
    supp = env < threshold_uv
    # keep runs >= min_s
    d = np.diff(np.concatenate([[0], supp.astype(np.int8), [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    keep = np.zeros(N, bool)
    for a, b in zip(starts, ends):
        if b - a >= min_s * fs:
            keep[a:b] = True
    return float((keep & clean).sum() / clean.sum())


def _crossing(freqs: np.ndarray, psd: np.ndarray, frac: float) -> float:
    c = np.cumsum(psd)
    if c[-1] <= 0:
        return float("nan")
    c = c / c[-1]
    j = int(np.searchsorted(c, frac))
    j = min(j, len(c) - 1)
    if j == 0:
        return float(freqs[0])
    f0, f1, c0, c1 = freqs[j - 1], freqs[j], c[j - 1], c[j]
    return float(f0 + (frac - c0) / (c1 - c0) * (f1 - f0)) if c1 > c0 else float(f1)


def _nanmean(a) -> float:
    a = np.asarray(a, float)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


# ---------------------------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------------------------
def extract_features(data: np.ndarray, fs: float, ch_names: Sequence[str], clean_mask: np.ndarray | None = None,
                     cfg: FeatureConfig | None = None) -> dict[str, float]:
    """One feature vector (ordered ``dict``) for one recording window.

    ``data``: (C, N) uV, preprocessed, common-average referenced. ``ch_names``: canonical 10-20 names.
    ``clean_mask``: (C, N//(epoch_s*fs)) bool from ``window.window_clean_mask`` (None = everything clean).
    """
    cfg = cfg or FeatureConfig()
    ch = list(ch_names)
    L_ep = int(round(cfg.epoch_s * fs))
    m = int(round(cfg.seg_s / cfg.epoch_s))
    L = m * L_ep
    E = data.shape[1] // L_ep
    if clean_mask is None:
        clean_mask = np.ones((len(ch), E), bool)
    clean_mask = np.asarray(clean_mask, bool)[:, :E].copy()
    clean_mask &= ~np.isnan(data[:, : clean_mask.shape[1] * L_ep]).any(axis=1, keepdims=True)
    x = np.nan_to_num(data[:, : E * L_ep])

    X = _segments(x, L_ep, m)                      # (C, S, L)
    S = X.shape[1]
    cs = segment_clean_mask(clean_mask, S, m) if S else np.zeros((len(ch), 0), bool)
    cnt = cs.sum(axis=1)
    enough = cnt >= cfg.min_segments

    # --- spectra
    vals: dict[str, float] = {}
    C = len(ch)
    win = np.hanning(L)
    freqs = np.fft.rfftfreq(L, 1.0 / fs)
    df = freqs[1] - freqs[0]
    if S:
        F = np.fft.rfft((X - X.mean(axis=2, keepdims=True)) * win, axis=2)       # (C,S,F)
        P = (np.abs(F) ** 2) * (2.0 / (fs * (win ** 2).sum()))
        w = cs.astype(float)
        psd = np.einsum("csf,cs->cf", P, w) / np.maximum(cnt, 1)[:, None]
    else:
        F = np.zeros((C, 0, len(freqs)), complex)
        psd = np.zeros((C, len(freqs)))
    psd[~enough] = np.nan

    def bp(lo, hi):
        sel = (freqs >= lo) & (freqs < hi)
        return psd[:, sel].sum(axis=1) * df

    absb = {b: bp(*r) for b, r in cfg.bands.items()}
    total = bp(*cfg.rel_range)
    per: dict[str, np.ndarray] = {}
    for b in cfg.bands:
        per[f"{b}_abs_log10"] = np.log10(np.maximum(absb[b], 1e-12))
        per[f"{b}_rel"] = absb[b] / np.where(total > 0, total, np.nan)
    sef = (freqs >= cfg.sef_range[0]) & (freqs <= cfg.sef_range[1])
    sef95 = np.full(C, np.nan)
    sef50 = np.full(C, np.nan)
    for i in range(C):
        if enough[i]:
            sef95[i] = _crossing(freqs[sef], psd[i, sef], 0.95)
            sef50[i] = _crossing(freqs[sef], psd[i, sef], 0.50)
    per["sef95"], per["sef50"] = sef95, sef50
    d, t, a, b_ = (absb.get(k, np.full(C, np.nan)) for k in ("delta", "theta", "alpha", "beta"))
    with np.errstate(divide="ignore", invalid="ignore"):
        per["alpha_delta_log10"] = np.log10(np.maximum(a, 1e-12) / np.maximum(d, 1e-12))
        per["slowing_log10"] = np.log10(np.maximum(d + t, 1e-12) / np.maximum(a + b_, 1e-12))

    # --- amplitude statistics + complexity on clean samples
    rms = np.full(C, np.nan); p95 = np.full(C, np.nan); kurt = np.full(C, np.nan)
    ll = np.full(C, np.nan); lzc = np.full(C, np.nan); bsr = np.full(C, np.nan)
    for i in range(C):
        samp_clean = np.repeat(clean_mask[i], L_ep)
        if samp_clean.sum() < cfg.min_segments * L_ep:
            continue
        xi = x[i]
        xc = xi[samp_clean]
        rms[i] = np.sqrt(np.mean(xc ** 2))
        p95[i] = np.percentile(np.abs(xc), 95)
        kurt[i] = kurtosis(xc, fisher=True)
        ok = samp_clean[1:] & samp_clean[:-1]
        ll[i] = np.mean(np.abs(np.diff(xi))[ok]) if ok.any() else np.nan
        bsr[i] = burst_suppression_ratio(xi, fs, samp_clean, cfg.bsr_threshold_uv, cfg.bsr_env_s, cfg.bsr_min_s)
        good = np.where(cs[i])[0]
        good = good[:: m] if len(good) else good           # non-overlapping segments
        if len(good) > cfg.lzc_max_segments:
            good = good[np.linspace(0, len(good) - 1, cfg.lzc_max_segments).astype(int)]
        if len(good):
            lzc[i] = _nanmean([lz_complexity_norm(X[i, s]) for s in good])
    per.update({"amp_rms": rms, "amp_p95": p95, "amp_kurtosis": kurt, "line_length": ll, "lzc": lzc})

    idx = {c: i for i, c in enumerate(ch)}
    metrics = _scope_metrics(cfg)

    def put(scope: str, rows: list[int]):
        for mname in metrics:
            vals[f"qeeg.{scope}.{mname}"] = _nanmean(per[mname][rows]) if rows else float("nan")

    put("global", list(range(C)))
    for r, members in REGIONS.items():
        put(r, [idx[c] for c in members if c in idx])
    if cfg.per_channel:
        for c in CANONICAL_19:
            put(f"ch_{c}", [idx[c]] if c in idx else [])
    vals["qeeg.global.bsr"] = float(np.nanmedian(bsr)) if np.isfinite(bsr).any() else float("nan")
    for band in ("delta", "alpha"):
        diffs = []
        for l, r in HOMOLOGOUS_PAIRS:
            if l in idx and r in idx and band in cfg.bands:
                diffs.append(abs(per[f"{band}_abs_log10"][idx[l]] - per[f"{band}_abs_log10"][idx[r]]))
        vals[f"qeeg.asym.{band}_absdiff_log10"] = _nanmean(diffs) if diffs else float("nan")

    # --- connectivity (coherence, wPLI) over fixed electrode pairs
    if cfg.connectivity:
        res: dict[str, float] = {}
        grp: dict[tuple[str, str, str], list[float]] = {}
        sel_bands = {bn: (freqs >= lo) & (freqs < hi) for bn, (lo, hi) in cfg.bands.items()}
        for p in CONN_PAIRS:
            pn = _pair_name(p)
            vec = {(meas, bn): float("nan") for meas in ("coh", "wpli") for bn in cfg.bands}
            if p[0] in idx and p[1] in idx and S:
                i, j = idx[p[0]], idx[p[1]]
                ok = cs[i] & cs[j]
                if ok.sum() >= cfg.min_segments:
                    Fi, Fj = F[i, ok], F[j, ok]
                    Sxy = Fi * np.conj(Fj)
                    Sxx = (np.abs(Fi) ** 2).mean(axis=0)
                    Syy = (np.abs(Fj) ** 2).mean(axis=0)
                    coh = np.abs(Sxy.mean(axis=0)) ** 2 / np.maximum(Sxx * Syy, 1e-24)
                    im = Sxy.imag
                    wpli = np.abs(im.mean(axis=0)) / np.maximum(np.abs(im).mean(axis=0), 1e-24)
                    for bn, sl in sel_bands.items():
                        vec[("coh", bn)] = float(coh[sl].mean())
                        vec[("wpli", bn)] = float(wpli[sl].mean())
            sets = ["all"] + (["interhemispheric"] if p in HOMOLOGOUS_PAIRS else ["anteroposterior"])
            for (meas, bn), v in vec.items():
                res[f"conn.{meas}.{pn}.{bn}"] = v
                for g in sets:
                    grp.setdefault((meas, g, bn), []).append(v)
        for meas in ("coh", "wpli"):
            for g in ("all", "interhemispheric", "anteroposterior"):
                for bn in cfg.bands:
                    res[f"conn.{meas}.mean_{g}.{bn}"] = _nanmean(grp.get((meas, g, bn), [np.nan]))
        vals.update(res)

    names = feature_names(cfg)
    return {n: vals.get(n, float("nan")) for n in names}
