"""Simulate deployable-hardware EEG geometries from a 19-channel 10-20 recording.

Inputs are plain numpy arrays (channels x samples) plus channel-name lists, so this module has no
dependency on the EEG loaders.  Geometry definitions come from ``configs/montages.yaml``.

Non-10-20-19 sites (Fpz, AFz, A1, A2, ...) are approximated either by the linear combinations in the
YAML ``approximation_notes`` or by spherical-spline interpolation (Perrin et al. 1989) on standard
10-20 coordinates, implemented here with numpy only.

Electrode coordinates use the EEGLAB-style topographic 10-20 convention (azimuth clockwise from the
nose, radius fraction of a 180 degree arc from Cz), projected onto the unit sphere.  Ear sites sit
below the equator and are approximate; spline extrapolation there is unreliable and is only used
when explicitly requested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "montages.yaml"

# Aliases to the source-19 labels used in the YAML (T3/T4/T5/T6).
_ALIASES = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}

# (azimuth deg clockwise from nose, radius as fraction of 180 deg from Cz)
_TOPO = {
    "Fp1": (-18, 0.5111), "Fp2": (18, 0.5111), "Fpz": (0, 0.5111),
    "AFz": (0, 0.3833),
    "F7": (-54, 0.5111), "F8": (54, 0.5111),
    "F3": (-39, 0.3333), "F4": (39, 0.3333), "Fz": (0, 0.2556),
    "T3": (-90, 0.5111), "T4": (90, 0.5111),
    "C3": (-90, 0.2556), "C4": (90, 0.2556), "Cz": (0, 0.0),
    "T5": (-126, 0.5111), "T6": (126, 0.5111),
    "P3": (-141, 0.3333), "P4": (141, 0.3333), "Pz": (180, 0.2556),
    "O1": (-162, 0.5111), "O2": (162, 0.5111), "Oz": (180, 0.5111),
    "FCz": (0, 0.1278),
    "A1": (-90, 0.64), "A2": (90, 0.64),  # earlobes, ~115 deg from Cz (approximate)
}

# Linear approximations from approximation_notes (site -> {source: weight}).
LINEAR_RULES: Dict[str, Dict[str, float]] = {
    "Fpz": {"Fp1": 0.5, "Fp2": 0.5},
    "AFz": {"Fp1": 0.25, "Fp2": 0.25, "Fz": 0.5},
}
_EAR_SITES = ("A1", "A2")
SOURCE_19 = ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4",
             "T6", "O1", "O2")
_REF_TOKENS = {"REF_AFz_APPROX": ("AFz", ["AFz"]), "REF_LINKED_EAR": ("LE", ["A1", "A2"])}


class LowConfidenceGeometryError(ValueError, NotImplementedError):
    """Geometry is low confidence or has no electrode list; pass allow_low_confidence=True (low only)."""


def canonical_name(name: str) -> str:
    """Normalise a channel label to the source-19 convention (e.g. 'EEG Fp1-REF' -> 'Fp1', 'T7' -> 'T3')."""
    s = str(name).strip()
    if s.upper().startswith("EEG "):
        s = s[4:].strip()
    for sep in ("-", " "):
        if sep in s:
            s = s.split(sep)[0]
    low = s.lower()
    for k in list(_TOPO) + list(_ALIASES):
        if k.lower() == low:
            return _ALIASES.get(k, k)
    return s


def electrode_positions(names: Sequence[str]) -> np.ndarray:
    """Unit-sphere (x right, y anterior, z up) coordinates, shape (n, 3)."""
    out = []
    for n in names:
        c = canonical_name(n)
        if c not in _TOPO:
            raise KeyError(f"no 10-20 coordinates for electrode {n!r}")
        az, r = _TOPO[c]
        pol, az = np.deg2rad(r * 180.0), np.deg2rad(az)
        out.append([np.sin(pol) * np.sin(az), np.sin(pol) * np.cos(az), np.cos(pol)])
    return np.asarray(out, dtype=float)


# ---------------------------------------------------------------- spherical spline (Perrin 1989)
def _g(cos_angle: np.ndarray, m: int = 4, n_terms: int = 7) -> np.ndarray:
    x = np.clip(cos_angle, -1.0, 1.0)
    n = np.arange(1, n_terms + 1)
    coef = (2 * n + 1) / (n ** m * (n + 1) ** m)
    # Legendre series sum_n coef_n P_n(x); legval takes coefficients from P_0.
    series = np.polynomial.legendre.legval(x, np.concatenate([[0.0], coef]))
    return series / (4 * np.pi)


def spherical_spline_matrix(known_pos: np.ndarray, target_pos: np.ndarray, m: int = 4,
                            n_terms: int = 7, lam: float = 1e-5) -> np.ndarray:
    """Interpolation operator W (n_target x n_known) so that v_target = W @ v_known.

    Perrin et al. 1989: solve [[G+lam*I, 1], [1', 0]] [c; c0] = [v; 0]; v_new = G_new c + c0.
    """
    k = np.asarray(known_pos, float)
    t = np.asarray(target_pos, float)
    k = k / np.linalg.norm(k, axis=1, keepdims=True)
    t = t / np.linalg.norm(t, axis=1, keepdims=True)
    n = len(k)
    if n < 3:
        raise ValueError("spherical spline needs at least 3 known electrodes")
    G = _g(k @ k.T, m, n_terms) + lam * np.eye(n)
    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = G
    A[:n, n] = 1.0
    A[n, :n] = 1.0
    Ainv = np.linalg.inv(A)[:, :n]                       # maps v -> [c; c0]
    Gt = np.hstack([_g(t @ k.T, m, n_terms), np.ones((len(t), 1))])
    return Gt @ Ainv


def spline_interpolate(data: np.ndarray, known_names: Sequence[str], target_names: Sequence[str],
                       m: int = 4, n_terms: int = 7, lam: float = 1e-5) -> np.ndarray:
    """Interpolate ``target_names`` from ``data`` (known channels x samples) by spherical spline."""
    W = spherical_spline_matrix(electrode_positions(known_names), electrode_positions(target_names),
                                m, n_terms, lam)
    return W @ np.asarray(data, float)


# ---------------------------------------------------------------- geometry config
@dataclass(frozen=True)
class Geometry:
    name: str
    description: str
    confidence: str
    electrodes: Tuple[str, ...]
    derivations: Tuple[Tuple[str, str], ...]
    reference_scheme: str = ""
    sampling_rate_hz: Optional[float] = None
    vendor_documented_derivations: bool = False
    approximation_notes: str = ""
    source_url: object = None


def load_geometries(path: Optional[str | Path] = None) -> Dict[str, Geometry]:
    cfg = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text())
    out = {}
    for key, g in cfg["geometries"].items():
        out[key] = Geometry(
            name=g.get("name", key), description=g.get("description", ""),
            confidence=str(g.get("confidence", "low")).lower(),
            electrodes=tuple(g.get("electrodes") or ()),
            derivations=tuple((str(a), str(b)) for a, b in (g.get("derivations") or ())),
            reference_scheme=str(g.get("reference_scheme", "")),
            sampling_rate_hz=g.get("sampling_rate_hz"),
            vendor_documented_derivations=bool(g.get("vendor_documented_derivations", False)),
            approximation_notes=str(g.get("approximation_notes", "")),
            source_url=g.get("source_url"),
        )
    return out


@dataclass
class MontageRecording:
    geometry: str
    channel_names: List[str]               # derivation labels, e.g. 'Fp1-F7'
    data: np.ndarray                       # derivations x samples
    sfreq: float                           # sampling rate of the input (no resampling is done)
    device_sampling_rate_hz: Optional[float]
    confidence: str
    electrode_names: List[str]             # device electrodes (canonical labels)
    electrode_data: np.ndarray             # electrodes x samples (before re-referencing)
    approximated: Dict[str, str] = field(default_factory=dict)   # site -> method used
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------- simulation
class _Resolver:
    def __init__(self, data, names, mode):
        self.names = names
        self.data = data
        self.index = {n: i for i, n in enumerate(names)}
        self.mode = mode
        self.cache: Dict[str, np.ndarray] = {}
        self.approximated: Dict[str, str] = {}
        self.warnings: List[str] = []
        self.scalp = [n for n in names if n in _TOPO]

    def get(self, site: str) -> np.ndarray:
        if site in self.cache:
            return self.cache[site]
        if site in self.index:
            sig = self.data[self.index[site]]
        else:
            sig = self._approx(site)
        self.cache[site] = sig
        return sig

    def _approx(self, site: str) -> np.ndarray:
        mode = self.mode
        if site in SOURCE_19:
            raise ValueError(f"recording lacks source 10-20 channel {site!r}; it is not approximated")
        if mode == "spline":
            method = "spline"
        elif site in _EAR_SITES:
            method = "average"
        elif site in LINEAR_RULES:
            method = "linear"
        elif mode == "auto":
            method = "spline"
        else:
            raise ValueError(f"site {site!r} is not recorded and has no linear rule (approximation='linear')")
        if method == "linear":
            rule = LINEAR_RULES[site]
            missing = [s for s in rule if s not in self.index]
            if missing:
                raise ValueError(f"cannot approximate {site}: recording lacks {missing}")
            sig = sum(w * self.data[self.index[s]] for s, w in rule.items())
        elif method == "average":
            sig = self.data[[self.index[n] for n in self.scalp if n not in _EAR_SITES]].mean(axis=0)
            self.warnings.append(
                f"{site} not recorded: approximated by average of {len(self.scalp)} scalp channels (sensitivity only)")
        else:
            known = [n for n in self.scalp]
            if site not in _TOPO:
                raise ValueError(f"no coordinates for {site!r}")
            sig = spline_interpolate(self.data[[self.index[n] for n in known]], known, [site])[0]
            if site in _EAR_SITES:
                self.warnings.append(f"{site} by spline is an extrapolation below the 10-20 grid; unreliable")
        self.approximated[site] = method
        return sig


def _check_geometry(g: Geometry, allow_low_confidence: bool) -> None:
    if not g.electrodes or not g.derivations:
        raise LowConfidenceGeometryError(
            f"geometry {g.name!r} has no electrode/derivation list (unverified hardware); cannot simulate")
    if g.confidence == "low" and not allow_low_confidence:
        raise LowConfidenceGeometryError(
            f"geometry {g.name!r} has confidence 'low'; pass allow_low_confidence=True to simulate anyway")


def simulate_geometry(data: np.ndarray, channel_names: Sequence[str], sfreq: float,
                      geometry: "str | Geometry", *, config: "Optional[Mapping[str, Geometry]]" = None,
                      allow_low_confidence: bool = False, approximation: str = "auto") -> MontageRecording:
    """Signals a device geometry would record, from a 10-20 recording.

    approximation: 'auto' (linear rule where the notes give one, average reference for A1/A2, otherwise
    spline), 'linear' (linear rules only), or 'spline' (spherical spline for every missing site).
    """
    if approximation not in ("auto", "linear", "spline"):
        raise ValueError("approximation must be 'auto', 'linear' or 'spline'")
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[0] != len(channel_names):
        raise ValueError("data must be (channels, samples) matching channel_names")
    if not sfreq > 0:
        raise ValueError("sfreq must be positive")
    if isinstance(geometry, str):
        geoms = config if config is not None else load_geometries()
        if geometry not in geoms:
            raise KeyError(f"unknown geometry {geometry!r}; known: {sorted(geoms)}")
        geometry = geoms[geometry]
    _check_geometry(geometry, allow_low_confidence)

    names = [canonical_name(n) for n in channel_names]
    if len(set(names)) != len(names):
        raise ValueError("duplicate channel names after canonicalisation")
    res = _Resolver(data, names, approximation)

    def resolve(token: str) -> Tuple[str, np.ndarray]:
        if token == "REF_LINKED_EAR":
            return "LE", 0.5 * (res.get("A1") + res.get("A2"))
        if token == "REF_AFz_APPROX":
            return "AFz", res.get("AFz")
        return token, res.get(token)

    el_names = list(geometry.electrodes)
    el_data = np.vstack([res.get(e) for e in el_names])
    out_names, out = [], []
    for anode, cathode in geometry.derivations:
        a_lbl, a = resolve(anode)
        c_lbl, c = resolve(cathode)
        out_names.append(f"{a_lbl}-{c_lbl}")
        out.append(a - c)
    return MontageRecording(
        geometry=geometry.name, channel_names=out_names, data=np.vstack(out), sfreq=float(sfreq),
        device_sampling_rate_hz=geometry.sampling_rate_hz, confidence=geometry.confidence,
        electrode_names=el_names, electrode_data=el_data,
        approximated=dict(res.approximated), warnings=list(res.warnings))


def simulate_all(data, channel_names, sfreq, *, config_path=None, allow_low_confidence=False,
                 approximation="auto", skip_unsupported=True) -> Dict[str, MontageRecording]:
    """Simulate every geometry; unsupported (low/empty) ones are skipped unless skip_unsupported=False."""
    geoms = load_geometries(config_path)
    out = {}
    for name, g in geoms.items():
        try:
            out[name] = simulate_geometry(data, channel_names, sfreq, g,
                                          allow_low_confidence=allow_low_confidence,
                                          approximation=approximation)
        except LowConfidenceGeometryError:
            if not skip_unsupported:
                raise
    return out
