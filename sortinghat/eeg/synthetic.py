"""Synthetic 19-channel EEG and an EDF writer, for tests and dry runs. No real data, ever.

Backgrounds: ``normal`` (posterior alpha + 1/f), ``slowing`` (diffuse delta/theta), ``burst_suppression``
(synchronous bursts over a ~2 uV suppressed floor), ``periodic_discharges`` (1 Hz sharp transients, generalised
or lateralised), ``low_voltage`` (near-isoelectric). Artifacts are injected afterwards with ``Artifact``.
Amplitudes are in microvolts and roughly clinical, not physiologically exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .io import CANONICAL_19

BACKGROUNDS = ("normal", "slowing", "burst_suppression", "periodic_discharges", "low_voltage")
ARTIFACT_KINDS = ("flat", "clipping", "extreme", "line_noise", "disconnected")

_ALPHA_WEIGHT = {c: w for w, members in ((1.0, ("O1", "O2", "P3", "P4", "Pz")),
                                         (0.6, ("C3", "C4", "Cz", "T5", "T6")),
                                         (0.35, ("T3", "T4", "F3", "F4", "Fz", "F7", "F8", "Fp1", "Fp2")))
                 for c in members}
_LEFT = ("Fp1", "F7", "F3", "T3", "C3", "T5", "P3", "O1")


def _bandlimited(n: int, fs: float, lo: float, hi: float, rms: float, rng, exponent: float = 0.0) -> np.ndarray:
    """Gaussian noise confined to [lo, hi] Hz (optional 1/f^exponent shaping), scaled to ``rms`` uV."""
    spec = rng.standard_normal(n // 2 + 1) + 1j * rng.standard_normal(n // 2 + 1)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    shape = np.where((f >= lo) & (f <= hi), np.maximum(f, 0.1) ** (-exponent / 2), 0.0)
    x = np.fft.irfft(spec * shape, n)
    s = x.std()
    return x * (rms / s) if s > 0 else x


def _mix(n_ch: int, make, rho: float, rng) -> np.ndarray:
    """Common + independent mixture: each channel = sqrt(rho) * common + sqrt(1-rho) * independent."""
    common = make()
    return np.vstack([np.sqrt(rho) * common + np.sqrt(1 - rho) * make() for _ in range(n_ch)])


def _discharge_wave(fs: float) -> np.ndarray:
    t = np.arange(-0.25, 0.25, 1 / fs)
    sharp = -t / 0.02 * np.exp(-0.5 * (t / 0.02) ** 2)            # biphasic spike (derivative of Gaussian)
    slow = 0.6 * np.exp(-0.5 * ((t - 0.08) / 0.07) ** 2)           # after-going slow wave
    w = sharp + slow
    return w / np.abs(w).max()


def generate_eeg(duration_s: float, fs: float = 200.0, background: str = "normal", seed: int = 0,
                 artifacts: Sequence["Artifact"] = (), ch_names: Sequence[str] = CANONICAL_19,
                 lateralized: bool = False) -> np.ndarray:
    """(n_channels, n_samples) synthetic EEG in uV."""
    if background not in BACKGROUNDS:
        raise ValueError(f"background must be one of {BACKGROUNDS}")
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fs))
    C = len(ch_names)
    t = np.arange(n) / fs
    aw = np.array([_ALPHA_WEIGHT.get(c, 0.35) for c in ch_names])

    if background == "normal":
        x = _mix(C, lambda: _bandlimited(n, fs, 0.5, 40, 9.0, rng, exponent=1.0), 0.3, rng)
        am = 1 + 0.4 * np.sin(2 * np.pi * 0.2 * t + rng.uniform(0, 6.28))
        for i in range(C):
            phase = rng.uniform(0, 6.28)
            x[i] += aw[i] * 22 * am * np.sin(2 * np.pi * 10.0 * t + phase)
            x[i] += _bandlimited(n, fs, 15, 25, 2.5, rng)
    elif background == "slowing":
        x = _mix(C, lambda: _bandlimited(n, fs, 0.8, 3.5, 40.0, rng), 0.4, rng)
        x += _mix(C, lambda: _bandlimited(n, fs, 4, 7, 12.0, rng), 0.2, rng)
        x += _mix(C, lambda: _bandlimited(n, fs, 8, 25, 3.0, rng, exponent=1.0), 0.1, rng)
    elif background == "low_voltage":
        x = _mix(C, lambda: _bandlimited(n, fs, 0.5, 40, 2.0, rng, exponent=1.0), 0.2, rng)
    elif background == "burst_suppression":
        x = _mix(C, lambda: _bandlimited(n, fs, 0.5, 40, 2.0, rng, exponent=1.0), 0.2, rng)   # suppressed floor
        env = np.zeros(n)
        pos = 0
        while pos < n:
            pos += int(rng.uniform(2.0, 5.0) * fs)                              # suppression
            blen = int(rng.uniform(1.0, 2.5) * fs)
            a, b = pos, min(n, pos + blen)
            if b > a:
                env[a:b] = np.hanning(b - a + 2)[1:-1]
            pos += blen
        burst = _mix(C, lambda: _bandlimited(n, fs, 2, 25, 70.0, rng, exponent=0.5), 0.5, rng)
        x = x + burst * env[None, :] * rng.uniform(0.8, 1.2, size=(C, 1))
    else:  # periodic_discharges
        x = _mix(C, lambda: _bandlimited(n, fs, 0.5, 40, 8.0, rng, exponent=1.0), 0.3, rng)
        x += aw[:, None] * 8 * np.sin(2 * np.pi * 9.0 * t)[None, :]
        w = _discharge_wave(fs)
        k = len(w)
        pos = int(0.5 * fs)
        # non-uniform spatial weights so a common-average reference does not cancel generalised discharges
        spatial = rng.uniform(0.4, 1.6, size=C) * np.array(
            [(1.0 if (not lateralized or c in _LEFT) else 0.1) for c in ch_names])
        while pos + k < n:
            x[:, pos:pos + k] += spatial[:, None] * 130.0 * w[None, :]
            pos += int(fs * 1.0 * rng.uniform(0.95, 1.05))
    out = np.asarray(x, dtype=np.float64)
    for a in artifacts:
        apply_artifact(out, fs, a, ch_names, rng)
    return out


@dataclass(frozen=True)
class Artifact:
    kind: str                    # one of ARTIFACT_KINDS
    channels: tuple[str, ...]    # canonical channel names
    start_s: float = 0.0
    dur_s: float | None = None   # None = until the end
    amp: float | None = None     # kind-specific amplitude (uV)


def apply_artifact(data: np.ndarray, fs: float, art: Artifact, ch_names: Sequence[str], rng=None) -> None:
    """Inject ``art`` into ``data`` in place."""
    if art.kind not in ARTIFACT_KINDS:
        raise ValueError(f"unknown artifact kind {art.kind!r}")
    rng = rng or np.random.default_rng(0)
    a = int(round(art.start_s * fs))
    b = data.shape[1] if art.dur_s is None else min(data.shape[1], a + int(round(art.dur_s * fs)))
    t = np.arange(b - a) / fs
    for c in art.channels:
        i = list(ch_names).index(c)
        seg = data[i, a:b]
        if art.kind == "flat":
            data[i, a:b] = seg.mean() if seg.size else 0.0
        elif art.kind == "clipping":
            lim = art.amp or 40.0
            data[i, a:b] = np.clip(seg * 8.0, -lim, lim)
        elif art.kind == "extreme":
            amp = art.amp or 1500.0
            for s in range(0, len(seg), int(fs)):                    # one big transient per second
                pulse = amp * np.exp(-0.5 * ((np.arange(len(seg)) - s - fs / 2) / (0.05 * fs)) ** 2)
                data[i, a:b] += pulse
        elif art.kind == "line_noise":
            data[i, a:b] += (art.amp or 150.0) * np.sin(2 * np.pi * 60.0 * t)
        elif art.kind == "disconnected":
            data[i, a:b] = (art.amp or 300.0) * np.sin(2 * np.pi * 60.0 * t) + rng.standard_normal(b - a) * 0.3


# ---------------------------------------------------------------------------------------------
# EDF writer
# ---------------------------------------------------------------------------------------------
def _fmt(v: float, width: int) -> str:
    for p in range(8, 0, -1):
        s = f"{v:.{p}g}"
        if len(s) <= width:
            return s.ljust(width)
    raise ValueError(f"cannot fit {v} in {width} chars")


def _asc(s: str, width: int) -> bytes:
    return s.encode("ascii")[:width].ljust(width, b" ")


def write_edf(path, data_uv: np.ndarray, fs: float, ch_names: Sequence[str], label_fmt: str = "EEG {}-REF",
              phys_range_uv: float = 3276.7, record_s: float = 1.0, phys_dim: str = "uV",
              annotation_channel: bool = False, reserved: str = "") -> Path:
    """Write ``data_uv`` (C, N) as 16-bit EDF. Samples beyond +-``phys_range_uv`` saturate (like a real amplifier).
    ``phys_dim`` in {"uV","mV","V"} writes the signal in that unit. Trailing samples that do not fill a record
    are dropped."""
    scale = {"uV": 1.0, "mV": 1e-3, "V": 1e-6}[phys_dim]
    C, N = data_uv.shape
    spr = int(round(fs * record_s))
    n_rec = N // spr
    R = float(_fmt(phys_range_uv * scale, 8))        # use the value as written in the header
    if float(_fmt(-R, 8)) != -R:
        raise ValueError("phys_range_uv does not fit the 8-char EDF header field in this unit; pick a rounder value")
    gain = 2 * R / 65535.0
    dig = np.clip(np.round((data_uv[:, : n_rec * spr] * scale + R) / gain - 32768), -32768, 32767).astype("<i2")
    ns = C + (1 if annotation_channel else 0)
    hdr = bytearray()
    hdr += _asc("0", 8) + _asc("X X X X", 80) + _asc("Startdate X X X X", 80)
    hdr += _asc("01.01.20", 8) + _asc("00.00.00", 8) + _asc(str(256 * (ns + 1)), 8)
    hdr += _asc(reserved, 44) + _asc(str(n_rec), 8) + _asc(_fmt(record_s, 8).strip(), 8) + _asc(str(ns), 4)
    labels = [label_fmt.format(c) for c in ch_names] + (["EDF Annotations"] if annotation_channel else [])

    def field(vals, w):
        return b"".join(_asc(v, w) for v in vals)

    hdr += field(labels, 16) + field(["AgCl"] * ns, 80)
    hdr += field([phys_dim] * C + ([""] if annotation_channel else []), 8)
    hdr += field([_fmt(-R, 8).strip()] * C + (["-1"] if annotation_channel else []), 8)
    hdr += field([_fmt(R, 8).strip()] * C + (["1"] if annotation_channel else []), 8)
    hdr += field(["-32768"] * ns, 8) + field(["32767"] * ns, 8)
    hdr += field([""] * ns, 80)
    hdr += field([str(spr)] * C + (["2"] if annotation_channel else []), 8)
    hdr += field([""] * ns, 32)
    assert len(hdr) == 256 * (ns + 1), len(hdr)
    p = Path(path)
    with open(p, "wb") as fh:
        fh.write(hdr)
        for r in range(n_rec):
            for i in range(C):
                fh.write(dig[i, r * spr:(r + 1) * spr].tobytes())
            if annotation_channel:
                fh.write(b"+0\x14\x14")      # 2 int16 samples = 4 bytes (empty TAL)
    return p
