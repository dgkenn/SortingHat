"""EDF reading and channel-name normalisation (no third-party EDF dependency).

Why a built-in reader: HEEDB ICU cEEG EDFs are ~1 GB (docs/heedb_access.md section 4), and Study 1 only needs
minutes 1-11. EDF has a fixed-layout header, so ``read_edf`` seeks to just the data records covering the
requested interval (works on any binary file-like object with ``seek``/``read``, e.g. a ranged-S3 wrapper).
Output is always **microvolts** (``mne`` returns volts; foundation models expect uV; no per-channel z-scoring).

Privacy: the patient / recording / start-date header fields are never stored on ``Recording``; t0 is "recording
start" so only relative time matters (CLAUDE.md rule 3).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import BinaryIO

import numpy as np

# Standard 10-20 set, canonical order. Classical T3/T4/T5/T6 are canonical (CBraMod/TUEG convention);
# modern T7/T8/P7/P8 are aliases.
CANONICAL_19: tuple[str, ...] = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6", "O1", "O2")

# Minimum channel set for the >=60% usable rule (D-096): the 10 Ceribell-headband hairline electrodes, so every
# included recording supports the headband simulation (configs/montages.yaml, ceribell_headband). Classical
# T3/T4/T5/T6 labels. The other nine 10-20 channels (F3, F4, C3, C4, P3, P4, Fz, Cz, Pz) are optional for QC.
DEFAULT_MINIMUM_CHANNELS: tuple[str, ...] = ("Fp1", "Fp2", "F7", "F8", "T3", "T4", "T5", "T6", "O1", "O2")

_ALIASES = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
_CANON_UPPER = {c.upper(): c for c in CANONICAL_19}
_PREFIX_RE = re.compile(r"^\s*EEG[\s_:\-]+", re.I)
_SUFFIX_RE = re.compile(r"[\s_\-]*(REF|LE|AV|AVG|AR|AVERAGE|LINKED)\s*$", re.I)


class EDFError(ValueError):
    """Malformed or unsupported EDF."""


def normalize_channel_name(name: str) -> str | None:
    """Map a raw EDF label to a canonical 10-20 name, or ``None`` if it is not a referential 10-20 channel.

    Strips an ``EEG `` prefix and ``-REF`` / ``-LE`` / ``-AV`` style suffixes, case-insensitive, and maps
    T7/T8/P7/P8 to T3/T4/T5/T6. Bipolar labels (``Fp1-F7``), ECG, EOG, etc. return ``None``.
    """
    s = str(name).strip()
    s = _PREFIX_RE.sub("", s)
    s = _SUFFIX_RE.sub("", s)
    s = s.strip().replace(" ", "").upper()
    s = _ALIASES.get(s, s)
    return _CANON_UPPER.get(s)


@dataclass
class Recording:
    """Channels x samples EEG in microvolts. ``ch_names`` are canonical when produced by ``select_channels``."""

    data: np.ndarray                      # (n_channels, n_samples), float64, uV
    fs: float
    ch_names: list[str]
    offset_s: float = 0.0                 # time of sample 0 relative to recording start
    meta: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.data.shape[1] / self.fs


@dataclass
class EDFHeader:
    n_signals: int
    n_records: int
    record_duration: float
    labels: list[str]
    phys_dim: list[str]
    phys_min: np.ndarray
    phys_max: np.ndarray
    dig_min: np.ndarray
    dig_max: np.ndarray
    samples_per_record: np.ndarray
    header_bytes: int
    discontinuous: bool
    is_annotation: list[bool]

    @property
    def fs(self) -> np.ndarray:
        return self.samples_per_record / self.record_duration

    @property
    def record_bytes(self) -> int:
        return int(self.samples_per_record.sum()) * 2

    @property
    def duration_s(self) -> float:
        return self.n_records * self.record_duration


def _f(raw: bytes, what: str) -> float:
    try:
        return float(raw.decode("ascii", "replace").strip())
    except ValueError as e:
        raise EDFError(f"bad numeric EDF header field: {what}") from e


def _open(src):
    if isinstance(src, (str, os.PathLike)):
        return open(src, "rb"), True
    return src, False


def _parse_header(fh: BinaryIO, file_size: int | None) -> EDFHeader:
    fh.seek(0)
    fixed = fh.read(256)
    if len(fixed) < 256:
        raise EDFError("file shorter than an EDF header")
    if fixed[:8] != b"0       ":
        raise EDFError("not an EDF file (bad version field)")
    reserved = fixed[192:236].decode("ascii", "replace")
    discontinuous = reserved.startswith("EDF+D")
    header_bytes = int(_f(fixed[184:192], "header bytes"))
    n_records = int(_f(fixed[236:244], "n records"))
    rec_dur = _f(fixed[244:252], "record duration")
    ns = int(_f(fixed[252:256], "n signals"))
    if ns < 1 or rec_dur <= 0:
        raise EDFError("invalid signal count or record duration")
    body = fh.read(ns * 256)
    if len(body) < ns * 256:
        raise EDFError("truncated EDF signal header")
    off = 0

    def take(width: int) -> list[bytes]:
        nonlocal off
        out = [body[off + i * width: off + (i + 1) * width] for i in range(ns)]
        off += ns * width
        return out

    labels = [b.decode("ascii", "replace").strip() for b in take(16)]
    take(80)  # transducer
    dims = [b.decode("ascii", "replace").strip() for b in take(8)]
    pmin = np.array([_f(b, "phys min") for b in take(8)])
    pmax = np.array([_f(b, "phys max") for b in take(8)])
    dmin = np.array([_f(b, "dig min") for b in take(8)])
    dmax = np.array([_f(b, "dig max") for b in take(8)])
    take(80)  # prefilter
    spr = np.array([int(_f(b, "samples/record")) for b in take(8)])
    take(32)  # reserved
    hdr = EDFHeader(ns, n_records, rec_dur, labels, dims, pmin, pmax, dmin, dmax, spr, header_bytes,
                    discontinuous, [lb.startswith("EDF Annotations") for lb in labels])
    if n_records < 0 and file_size is not None:     # unknown record count: infer from size
        hdr.n_records = max(0, (file_size - header_bytes) // hdr.record_bytes)
    return hdr


def read_edf_header(src) -> EDFHeader:
    """Parse only the EDF header from a path or binary file-like."""
    fh, close = _open(src)
    try:
        size = os.path.getsize(src) if close else None
        return _parse_header(fh, size)
    finally:
        if close:
            fh.close()


_UNIT_TO_UV = {"UV": 1.0, "\u00b5V".upper(): 1.0, "\u03bcV".upper(): 1.0, "MV": 1e3, "V": 1e6, "NV": 1e-3}


def _uv_scale(dim: str) -> float:
    d = dim.strip().upper()
    if d in ("", "?"):
        return 1.0           # assume uV when unspecified (common in clinical EDF)
    if d in _UNIT_TO_UV:
        return _UNIT_TO_UV[d]
    raise EDFError(f"unsupported physical dimension {dim!r}")


def read_edf(src, start_s: float = 0.0, duration_s: float | None = None, channels: list[str] | None = None,
             keep_nonstandard: bool = False, allow_discontinuous: bool = False) -> Recording:
    """Read an EDF interval as channels x samples in uV.

    ``channels`` filters by *canonical* name (after ``normalize_channel_name``); by default only recognised
    10-20 channels are kept, in file order (use ``select_channels`` to order/check them). With
    ``keep_nonstandard`` unrecognised signals keep their raw label (annotations are always skipped).
    Signals with differing sampling rates are resampled to the highest selected rate.
    EDF+D (discontinuous) files raise unless ``allow_discontinuous``: a glued time axis is a known failure
    (docs/heedb_access.md, rule 27).
    """
    fh, close = _open(src)
    try:
        size = os.path.getsize(src) if close else None
        h = _parse_header(fh, size)
        if h.discontinuous and not allow_discontinuous:
            raise EDFError("EDF+D (discontinuous) recording: time axis has gaps; pass allow_discontinuous=True")
        names: dict[int, str] = {}
        for i, lb in enumerate(h.labels):
            if h.is_annotation[i]:
                continue
            c = normalize_channel_name(lb)
            if c is None:
                if keep_nonstandard:
                    names[i] = lb
                continue
            if channels is not None and c not in channels:
                continue
            if c in names.values():
                raise EDFError(f"two signals map to channel {c}")
            names[i] = c
        if not names:
            raise EDFError("no EEG channels found")
        total = h.duration_s
        t0 = min(max(0.0, start_s), total)
        t1 = total if duration_s is None else min(total, t0 + duration_s)
        r0 = int(np.floor(t0 / h.record_duration + 1e-9))
        r1 = int(np.ceil(t1 / h.record_duration - 1e-9))
        nrec = max(0, r1 - r0)
        fh.seek(h.header_bytes + r0 * h.record_bytes)
        raw = fh.read(nrec * h.record_bytes)
        nrec = len(raw) // h.record_bytes                     # tolerate truncated tail
        arr = np.frombuffer(raw[: nrec * h.record_bytes], dtype="<i2").reshape(nrec, -1)
        bounds = np.concatenate([[0], np.cumsum(h.samples_per_record)])
        sigs, rates = [], []
        for i in names:
            dig = arr[:, bounds[i]:bounds[i + 1]].astype(np.float64).reshape(-1)
            span = h.dig_max[i] - h.dig_min[i]
            if span == 0:
                raise EDFError("digital min == max")
            gain = (h.phys_max[i] - h.phys_min[i]) / span
            phys = (dig - h.dig_min[i]) * gain + h.phys_min[i]
            sigs.append(phys * _uv_scale(h.phys_dim[i]))
            rates.append(float(h.fs[i]))
        fs = max(rates)
        if len(set(rates)) > 1:
            from scipy.signal import resample_poly
            from fractions import Fraction
            out = []
            for x, r in zip(sigs, rates):
                if r != fs:
                    fr = Fraction(fs / r).limit_denominator(1000)
                    x = resample_poly(x, fr.numerator, fr.denominator)
                out.append(x)
            n = min(len(x) for x in out)
            sigs = [x[:n] for x in out]
        data = np.vstack(sigs) if sigs else np.zeros((0, 0))
        # trim to the exact requested interval
        s0 = int(round((t0 - r0 * h.record_duration) * fs))
        n_want = data.shape[1] - s0 if duration_s is None else int(round((t1 - t0) * fs))
        data = data[:, s0: s0 + max(0, n_want)]
        return Recording(data=data, fs=fs, ch_names=list(names.values()), offset_s=t0,
                         meta={"edf_duration_s": total, "discontinuous": h.discontinuous,
                               "phys_limits_uv": [float(max(abs(h.phys_min[i]), abs(h.phys_max[i]))
                                                        * _uv_scale(h.phys_dim[i])) for i in names]})
    finally:
        if close:
            fh.close()


@dataclass
class ChannelCheck:
    present: list[str]
    missing: list[str]
    extra: list[str]
    has_19: bool
    has_minimum: bool
    missing_minimum: list[str]


def check_channel_set(names, required=CANONICAL_19, minimum=DEFAULT_MINIMUM_CHANNELS) -> ChannelCheck:
    """Compare channel names (raw or canonical) with the 19-channel set and the minimum set."""
    canon = []
    for n in names:
        c = normalize_channel_name(n)
        canon.append(c if c is not None else n)
    cs = set(canon)
    present = [c for c in required if c in cs]
    missing = [c for c in required if c not in cs]
    extra = [c for c in canon if c not in required]
    miss_min = [c for c in minimum if c not in cs]
    return ChannelCheck(present, missing, extra, not missing, not miss_min, miss_min)


def select_channels(rec: Recording, required=CANONICAL_19, fill_missing: bool = False) -> Recording:
    """Return ``rec`` restricted to ``required`` in that order. Missing channels are dropped (default) or
    filled with NaN rows (``fill_missing``) so downstream arrays keep a fixed 19-row layout."""
    idx = {n: i for i, n in enumerate(rec.ch_names)}
    rows, names = [], []
    for c in required:
        if c in idx:
            rows.append(rec.data[idx[c]])
            names.append(c)
        elif fill_missing:
            rows.append(np.full(rec.data.shape[1], np.nan))
            names.append(c)
    data = np.vstack(rows) if rows else np.zeros((0, rec.data.shape[1]))
    meta = dict(rec.meta)
    meta["missing_channels"] = [c for c in required if c not in idx]
    return Recording(data, rec.fs, names, rec.offset_s, meta)
