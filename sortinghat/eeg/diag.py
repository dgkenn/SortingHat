"""Aggregate-only diagnostics for HEEDB EEG key resolution and EDF signal quality (CLAUDE.md rules 1-3).

HUMAN-RUN ONLY on real data (``scripts/diag_eeg_paths.py``, ``scripts/diag_eeg_signals.py`` call this module with a
credentialed S3 client). Everything here is testable on a fake S3 over synthetic bytes.

* ``sample_adult_recordings`` picks N adult sessions from ``eeg_metadata`` (BidsFolder / SessionID / EEGFolder stay in
  memory inside ``RecordingRef``; nothing about a single recording is ever printed or returned in an aggregate).
* ``inspect_paths`` / ``aggregate_paths``: which key pattern resolved each recording, whether its session folder
  exists, holds no .edf or several, and file-extension counts. Pattern NAMES only, never keys.
* ``inspect_signals`` / ``aggregate_signals``: header statistics (signal counts, label strings, sample rates,
  dmin == dmax, pmin == pmax, EDF / EDF+C / EDF+D) and, from the first ~11 minutes, per required electrode the share
  of 2-s epochs that are EXACTLY constant, split into "digital samples constant" and "scaling makes it constant".

Small cells. Counts of TECHNICAL FILE properties use ``safe_output.technical_count``: exact when the sample has at
least 50 recordings, else n < 11 is shown as "<11". Pooled fractions need >= 11 recordings behind them. Raw channel
label strings are equipment labels (not patient data) and are printed with their counts, after a sanity filter that
replaces anything odd-looking or id-like with ``<other>``.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import data_io
from ..safe_output import SUPPRESSED, assert_aggregate_only, safe_quantiles, suppress_count, technical_count
from .io import (CANONICAL_19, DEFAULT_MINIMUM_CHANNELS, SCALING_OK, EDFHeader, channel_scaling_status,
                 drop_dead_channels, is_bipolar_label, normalize_channel_name, read_edf, select_channels)
from .pipeline import PAD_S
from .stream import FailureReason, RecordingTimeout, _StreamError, fetch_raw_window
from .window import QCConfig, WindowQC, primary_window, qc_recording

REQUIRED = DEFAULT_MINIMUM_CHANNELS
TOP_LABELS = 40
MIN_POOL = 11                                    # recordings behind any pooled fraction / quantile
_LABEL_OK = re.compile(r"^[A-Za-z0-9 .\-_+:/()\[\]#*'&]{1,16}$")


@dataclass(frozen=True)
class RecordingRef:
    """One sampled session. NEVER print or log."""
    site: str
    bids_folder: str
    session_id: str
    eeg_folder: str | None = None


# ---------------------------------------------------------------------------------------------------------
# Sampling (adult sessions of eeg_metadata)
# ---------------------------------------------------------------------------------------------------------
def sample_adult_recordings(s3, site: str, n: int, seed: int = 0) -> tuple[list[RecordingRef], int]:
    """``(n sampled refs, number of adult sessions available)`` for one site, read through ``data_io.read_site_table``.

    Adult = ``AgeAtVisit >= 18`` after ``field_audit.merge_eeg`` (age from reports_findings when present, else the
    sparse metadata value, else DateOfBirth). Rows without BidsFolder / SessionID are dropped. Sampling is
    ``numpy.random.default_rng(seed)`` without replacement."""
    from ..audit.field_audit import ADULT_AGE, merge_eeg
    meta = data_io.read_site_table("eeg_metadata", site, s3=s3).reset_index(drop=True)
    try:
        rf = data_io.read_site_table("reports_findings", site, s3=s3)
    except FileNotFoundError:
        rf = None
    from .. import schema
    merged = merge_eeg(schema.coerce_types("eeg_metadata", meta),
                       None if rf is None else schema.coerce_types("reports_findings", rf), site)
    if len(merged) != len(meta):                 # merge_eeg is a left join on a de-duplicated right side
        raise ValueError("merged frame changed the number of sessions")
    adult = (pd.to_numeric(merged["AgeAtVisit"], errors="coerce") >= ADULT_AGE).to_numpy()
    ok = adult & meta["BidsFolder"].notna().to_numpy() & meta["SessionID"].notna().to_numpy()
    idx = np.flatnonzero(ok)
    pick = np.random.default_rng(seed).choice(idx, size=min(n, len(idx)), replace=False) if len(idx) else []
    eeg = meta["EEGFolder"] if "EEGFolder" in meta else pd.Series([None] * len(meta))
    refs = [RecordingRef(site, str(meta["BidsFolder"].iat[i]).strip(), str(meta["SessionID"].iat[i]).strip(),
                         None if pd.isna(eeg.iat[i]) else str(eeg.iat[i])) for i in sorted(int(i) for i in pick)]
    return refs, int(len(idx))


# ---------------------------------------------------------------------------------------------------------
# B. key resolution
# ---------------------------------------------------------------------------------------------------------
@dataclass
class PathInfo:
    pattern: str                                  # a data_io.EDF_PATTERNS name
    folder_exists: bool
    n_edf: int
    ext_counts: Counter
    task: str                                     # expected task token ("cEEG" / "EEG") from EEGFolder
    error: str | None = None                      # exception class name only


def inspect_paths(s3, ref: RecordingRef, *, parent_fallback: bool = False, bucket: str | None = None,
                  policy=None) -> PathInfo:
    task = "cEEG" if (ref.eeg_folder or "").lower().startswith("ceeg") else "EEG"
    try:
        r = data_io.resolve_edf_key(s3, ref.site, ref.bids_folder, ref.session_id, ref.eeg_folder, bucket=bucket,
                                    always_list=True, parent_fallback=parent_fallback, policy=policy)
    except Exception as exc:  # noqa: BLE001 - class name only
        return PathInfo("not_found", False, 0, Counter(), task, type(exc).__name__)
    return PathInfo(r.pattern, bool(r.folder_exists), int(r.n_edf or 0), r.ext_counts, task)


def aggregate_paths(infos: list[PathInfo], n_available: int | None = None) -> dict:
    n = len(infos)
    tc = lambda x: technical_count(x, n)                                                    # noqa: E731
    pats = Counter(i.pattern for i in infos)
    out: dict = {
        "n_sampled": suppress_count(n),
        "n_adult_sessions_available": None if n_available is None else suppress_count(n_available),
        "count_rule": "exact counts (technical file counts) because the sample has >= 50 recordings"
        if n >= 50 else "counts below 11 are suppressed (sample < 50 recordings)",
        "resolved_by_pattern": {p: tc(pats[p]) for p in data_io.EDF_PATTERNS if pats.get(p)},
        "n_resolved": tc(sum(c for p, c in pats.items() if p != "not_found")),
        "n_not_found": tc(pats.get("not_found", 0)),
        "recording_folder": {
            "exists": tc(sum(i.folder_exists for i in infos)),
            "missing": tc(sum(not i.folder_exists for i in infos)),
            "exists_but_no_edf": tc(sum(i.folder_exists and i.n_edf == 0 for i in infos)),
            "exactly_one_edf": tc(sum(i.n_edf == 1 for i in infos)),
            "multiple_edf": tc(sum(i.n_edf > 1 for i in infos)),
        },
        "folders_containing_extension": {e: tc(c) for e, c in sorted(
            Counter(e for i in infos for e in i.ext_counts).items(), key=lambda kv: (-kv[1], kv[0]))},
        "files_by_extension": {e: tc(c) for e, c in sorted(
            sum((i.ext_counts for i in infos), Counter()).items(), key=lambda kv: (-kv[1], kv[0]))},
        "expected_task_token": {t: tc(sum(i.task == t for i in infos)) for t in ("cEEG", "EEG")},
        "not_found_by_expected_task_token": {t: tc(sum(i.task == t and i.pattern == "not_found" for i in infos))
                                             for t in ("cEEG", "EEG")},
        "errors_by_exception_class": {e: tc(c) for e, c in Counter(i.error for i in infos if i.error).items()},
    }
    assert_aggregate_only(out)
    return out


# ---------------------------------------------------------------------------------------------------------
# C. signals
# ---------------------------------------------------------------------------------------------------------
@dataclass
class ChannelEpochs:
    n_epochs: int = 0
    n_const_digital: int = 0                      # all digital samples equal (scaling fine)
    n_const_scaling: int = 0                      # epoch constant only because the calibration range is zero
    n_zero_valued: int = 0                        # constant digital AND physical value 0 (zero-filled)


@dataclass
class SignalInfo:
    status: str = "ok"                            # ok | unresolved | fetch:<reason> | error:<Class>
    n_signals: int = 0
    n_eeg_signals: int = 0
    has_annotation: bool = False
    edf_type: str = "EDF"
    labels: list[str] = field(default_factory=list)
    n_bipolar: int = 0
    found: dict[str, int] = field(default_factory=dict)          # required electrode -> number of signals mapped to it
    n_duplicate: int = 0
    rates: list[float] = field(default_factory=list)             # sample rates of the required electrodes found
    n_dmin_eq_dmax: int = 0
    n_pmin_eq_pmax: int = 0
    req_dmin_eq_dmax: int = 0
    req_pmin_eq_pmax: int = 0
    epochs: dict[str, ChannelEpochs] = field(default_factory=dict)
    usable_primary: float | None = None
    passes_primary: bool | None = None
    reasons: list[str] = field(default_factory=list)
    cause: str = "unknown"


def _channel_epochs(h: EDFHeader, i: int, raw_records: np.ndarray, r0: int, start_s: float, dur_s: float,
                    epoch_s: float) -> ChannelEpochs:
    fs = float(h.fs[i])
    spr = int(h.samples_per_record[i])
    bounds = np.concatenate([[0], np.cumsum(h.samples_per_record)])
    x = raw_records[:, bounds[i]:bounds[i] + spr].reshape(-1)
    a = int(round((start_s - r0 * h.record_duration) * fs))
    L = int(round(epoch_s * fs))
    n = min(int(round(dur_s / epoch_s)), max(0, (len(x) - a) // L)) if L > 0 else 0
    out = ChannelEpochs()
    if n <= 0 or a < 0:
        return out
    ep = x[a:a + n * L].reshape(n, L)
    const = ep.max(axis=1) == ep.min(axis=1)
    out.n_epochs = n
    if channel_scaling_status(h, i) != SCALING_OK:
        out.n_const_scaling = n
        return out
    out.n_const_digital = int(const.sum())
    gain = (h.phys_max[i] - h.phys_min[i]) / (h.dig_max[i] - h.dig_min[i])
    phys = (ep[:, 0].astype(np.float64) - h.dig_min[i]) * gain + h.phys_min[i]
    # zero-filled: the constant is digital 0 or within one quantisation step of physical 0
    out.n_zero_valued = int((const & ((ep[:, 0] == 0) | (np.abs(phys) <= abs(gain) + 1e-12))).sum())
    return out


def inspect_signals(s3, ref: RecordingRef, *, bucket: str | None = None, policy=None, max_attempts: int = 4,
                    sleep=None, parent_fallback: bool = False, qc_cfg: QCConfig | None = None) -> SignalInfo:
    """Header + primary-window statistics for one recording (one ranged header GET + one ranged data GET)."""
    info = SignalInfo()
    try:
        res = data_io.resolve_edf_key(s3, ref.site, ref.bids_folder, ref.session_id, ref.eeg_folder, bucket=bucket,
                                      list_fallback=True, parent_fallback=parent_fallback, policy=policy)
    except Exception as exc:  # noqa: BLE001
        info.status = "error:" + type(exc).__name__
        return info
    if not res.found:
        info.status = "unresolved"
        return info
    pw = primary_window()
    start = max(0.0, pw.start_s - PAD_S)
    kw = {} if sleep is None else {"sleep": sleep}
    try:
        raw = fetch_raw_window(s3, res.key, start, pw.end_s + PAD_S - start, bucket=bucket,
                               max_attempts=max_attempts, allow_discontinuous=True, **kw)
    except _StreamError as e:
        info.status = "fetch:" + e.reason
        return info
    except RecordingTimeout:
        info.status = "fetch:" + FailureReason.TIMEOUT
        return info
    except Exception as exc:  # noqa: BLE001
        info.status = "error:" + type(exc).__name__
        return info

    h = raw.hdr
    eeg_idx = [i for i in range(h.n_signals) if not h.is_annotation[i]]
    info.n_signals, info.n_eeg_signals = h.n_signals, len(eeg_idx)
    info.has_annotation = any(h.is_annotation)
    info.edf_type = h.edf_type
    info.labels = sorted({h.labels[i] for i in eeg_idx})
    info.n_bipolar = sum(is_bipolar_label(h.labels[i]) for i in eeg_idx)
    info.n_dmin_eq_dmax = sum(h.dig_max[i] == h.dig_min[i] for i in eeg_idx)
    info.n_pmin_eq_pmax = sum(h.phys_max[i] == h.phys_min[i] for i in eeg_idx)
    first: dict[str, int] = {}
    for i in eeg_idx:
        c = normalize_channel_name(h.labels[i])
        if c in REQUIRED:
            if c in first:
                info.n_duplicate += 1
            else:
                first[c] = i
            info.found[c] = info.found.get(c, 0) + 1
    info.rates = sorted({float(h.fs[i]) for i in first.values()})
    info.req_dmin_eq_dmax = sum(h.dig_max[i] == h.dig_min[i] for i in first.values())
    info.req_pmin_eq_pmax = sum(h.phys_max[i] == h.phys_min[i] for i in first.values())
    recs = raw.records()
    r0 = raw.plan.r0
    epoch_s = (qc_cfg or QCConfig()).epoch_s
    for c, i in first.items():
        info.epochs[c] = _channel_epochs(h, i, recs, r0, pw.start_s, pw.duration_s, epoch_s)

    # QC as the pipeline sees it (same decode, label normalisation, dead / invalid channel handling)
    try:
        rec = read_edf(raw.sparse(), start_s=start, duration_s=pw.end_s + PAD_S - start, channels=list(CANONICAL_19),
                       allow_discontinuous=True)
        rec = drop_dead_channels(select_channels(rec))
        notes = {"dead": rec.meta.get("dead_channels", []), "invalid_scaling": rec.meta.get("invalid_scaling_channels", [])}
        qcs, _ = qc_recording(rec.data, rec.fs, rec.ch_names, rec.offset_s, {"primary": pw}, qc_cfg,
                              rec_duration_s=rec.meta.get("edf_duration_s"), channel_notes=notes)
        q: WindowQC = qcs["primary"]
        info.usable_primary, info.passes_primary, info.reasons = float(q.usable_fraction), bool(q.passes), list(q.reasons)
    except Exception as exc:  # noqa: BLE001
        info.reasons = ["read_error:" + type(exc).__name__]
    info.cause = _cause(info)
    return info


def _cause(i: SignalInfo) -> str:
    """One mutually exclusive technical cause per recording, in the order a fix would be applied."""
    if i.edf_type == "EDF+D":
        return "edf_plus_d_discontinuous"
    missing = [c for c in REQUIRED if c not in i.found]
    if len(missing) == len(REQUIRED):
        return "bipolar_labels_only" if i.n_bipolar else "no_required_labels_recognised"
    if missing:
        return "required_label_not_found"
    if any(i.epochs.get(c, ChannelEpochs()).n_const_scaling for c in REQUIRED):
        return "required_channel_zero_calibration_range"
    if any(e.n_epochs and e.n_const_digital == e.n_epochs for c, e in i.epochs.items() if c in REQUIRED):
        return "required_channel_constant_whole_window"
    if any(r.startswith("read_error:") for r in i.reasons):
        return "decode_error"
    if i.usable_primary is not None and i.usable_primary < 0.6:
        return "low_usable_other_artifacts"
    return "no_technical_problem"


def _label_ok(lb: str) -> bool:
    return bool(_LABEL_OK.match(lb)) and not data_io.looks_id_like(lb)


def aggregate_signals(infos: list[SignalInfo], n_available: int | None = None) -> dict:
    n = len(infos)
    tc = lambda x: technical_count(x, n)                                                    # noqa: E731
    ok = [i for i in infos if i.status == "ok"]
    m = len(ok)
    status = Counter(i.status for i in infos)

    def dist(values) -> dict:
        return {str(k): tc(c) for k, c in sorted(Counter(values).items())}

    lab_counts = Counter(lb for i in ok for lb in i.labels)
    top = []
    for lb, c in sorted(lab_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_LABELS]:
        top.append([lb if _label_ok(lb) else "<other>", tc(c)])
    const: dict = {}
    for ch in REQUIRED:
        es = [i.epochs[ch] for i in ok if ch in i.epochs and i.epochs[ch].n_epochs]
        k = len(es)
        tot = sum(e.n_epochs for e in es)
        if k < MIN_POOL or not tot:
            const[ch] = {"n_recordings": tc(k), "pooled_fractions": SUPPRESSED}
            continue
        const[ch] = {
            "n_recordings": tc(k),
            "frac_epochs_const_digital": round(sum(e.n_const_digital for e in es) / tot, 4),
            "frac_epochs_const_by_zero_calibration": round(sum(e.n_const_scaling for e in es) / tot, 4),
            "frac_epochs_zero_valued": round(sum(e.n_zero_valued for e in es) / tot, 4),
            "n_recordings_all_epochs_const": tc(sum((e.n_const_digital + e.n_const_scaling) == e.n_epochs for e in es)),
            "n_recordings_over_half_epochs_const": tc(sum((e.n_const_digital + e.n_const_scaling) * 2 > e.n_epochs
                                                          for e in es)),
        }
    out: dict = {
        "n_sampled": suppress_count(n),
        "n_adult_sessions_available": None if n_available is None else suppress_count(n_available),
        "count_rule": "exact counts (technical file counts) because the sample has >= 50 recordings"
        if n >= 50 else "counts below 11 are suppressed (sample < 50 recordings)",
        "status": {s: tc(c) for s, c in sorted(status.items())},
        "n_analysed": tc(m),
        "edf_type": dist(i.edf_type for i in ok),
        "n_signals_total": dist(i.n_signals for i in ok),
        "n_eeg_signals_non_annotation": dist(i.n_eeg_signals for i in ok),
        "recordings_with_annotation_signal": tc(sum(i.has_annotation for i in ok)),
        "required_electrode_found_after_normalisation": {c: tc(sum(c in i.found for i in ok)) for c in REQUIRED},
        "n_required_electrodes_found": dist(sum(c in i.found for c in REQUIRED) for i in ok),
        "recordings_with_duplicate_mapping": tc(sum(i.n_duplicate > 0 for i in ok)),
        "recordings_with_bipolar_labels": tc(sum(i.n_bipolar > 0 for i in ok)),
        "top_raw_labels_by_recordings": top,
        "sample_rates_hz_of_required_electrodes": dist("|".join(f"{r:g}" for r in i.rates) or "none" for i in ok),
        "scaling": {
            "recordings_with_dmin_eq_dmax": tc(sum(i.n_dmin_eq_dmax > 0 for i in ok)),
            "recordings_with_pmin_eq_pmax": tc(sum(i.n_pmin_eq_pmax > 0 for i in ok)),
            "signals_dmin_eq_dmax": tc(sum(i.n_dmin_eq_dmax for i in ok)),
            "signals_pmin_eq_pmax": tc(sum(i.n_pmin_eq_pmax for i in ok)),
            "required_electrodes_dmin_eq_dmax": tc(sum(i.req_dmin_eq_dmax for i in ok)),
            "required_electrodes_pmin_eq_pmax": tc(sum(i.req_pmin_eq_pmax for i in ok)),
        },
        "primary_window_exactly_constant_epochs_by_electrode": const,
        "technical_cause_per_recording": dist(i.cause for i in ok),
        "primary_usable_fraction_quantiles": safe_quantiles([i.usable_primary for i in ok if i.usable_primary is not None]),
        "primary_pass": tc(sum(bool(i.passes_primary) for i in ok)),
        "primary_reason_counts": {r: tc(c) for r, c in sorted(Counter(r for i in ok for r in i.reasons).items())},
    }
    assert_aggregate_only(out)
    return out


def format_lines(report: dict) -> list[str]:
    """Flat, aggregate-only text lines for ``safe_print``."""
    lines = []
    for k, v in report.items():
        if isinstance(v, (dict, list)):
            lines.append(f"{k}:")
            if isinstance(v, dict):
                lines += [f"  {kk}: {vv}" for kk, vv in v.items()]
            else:
                lines += [f"  {x[0]}\t{x[1]}" if isinstance(x, list) else f"  {x}" for x in v]
        else:
            lines.append(f"{k}: {v}")
    return lines
