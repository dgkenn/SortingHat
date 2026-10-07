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

import os
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import data_io
from ..safe_output import SUPPRESSED, assert_aggregate_only, safe_quantiles, suppress_count, technical_count
from .io import (CANONICAL_19, DEFAULT_MINIMUM_CHANNELS, SCALING_OK, EDFHeader, _uv_scale, channel_scaling_status,
                 drop_dead_channels, is_bipolar_label, normalize_channel_name, read_edf, select_channels)
from .pipeline import PAD_S
from .stream import (ONSET_MAX_SEARCH_S, FailureReason, RecordingTimeout, _StreamError, fetch_raw_window,
                     find_signal_onset)
from .window import QCConfig, WindowQC, WindowSpec, line_noise_ratio, primary_window, qc_recording

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
# C. signals (file-start window AND the window after t0 = first sustained live segment, D-109)
# ---------------------------------------------------------------------------------------------------------
@dataclass
class ChannelEpochs:
    n_epochs: int = 0
    n_const_digital: int = 0                      # all digital samples equal (scaling fine)
    n_const_scaling: int = 0                      # epoch constant only because the calibration range is zero
    n_zero_valued: int = 0                        # constant digital AND physical value 0 (zero-filled)
    n_const_rail: int = 0                         # constant digital value == the signal's header digital min or max
    n_const_other: int = 0                        # constant, neither a rail nor zero-valued


@dataclass
class WindowStats:
    """Everything measured on ONE primary window (file-start or after-onset) of one recording."""
    epochs: dict[str, ChannelEpochs] = field(default_factory=dict)
    usable: float | None = None
    passes: bool | None = None
    reasons: list[str] = field(default_factory=list)
    flag_frac: dict[str, float] = field(default_factory=dict)    # QC rule -> share of minimum-set channel x epoch cells
    amp_std_uv: list[float] = field(default_factory=list)        # per required channel found: std over the window
    amp_p99_uv: list[float] = field(default_factory=list)        # per required channel: 99th pct of |x - median|
    line_ratio: list[float] = field(default_factory=list)        # per required channel: median epoch line-noise ratio
    cause: str = "unknown"
    xcheck_status: str = "not_run"                # ok | rate_mismatch | unavailable | error:<Class> | not_run
    xcheck_max_abs_diff_uv: list[float] = field(default_factory=list)   # per required channel: ours vs pyedflib
    xcheck_corr: list[float] = field(default_factory=list)              # per required channel (non-constant pairs)
    xcheck_n_constant_pairs: int = 0


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
    hetero_spr: bool = False                      # non-annotation signals do not all have the same samples per record
    n_distinct_spr: int = 0
    req_calibration: list[str] = field(default_factory=list)   # per required channel "dim|pmin|pmax|dmin|dmax"
    file_start: WindowStats = field(default_factory=WindowStats)         # primary window at 60-660 s of the FILE
    after_onset: WindowStats | None = None                                # primary window at onset + 60-660 s
    onset_s: float | None = None                                          # seconds from the file start
    onset_status: str = "not_run"                 # found | none_within_limit | no_channels | error | not_run

    # file-start window accessors (names used by the first diagnostics)
    @property
    def epochs(self) -> dict[str, ChannelEpochs]:
        return self.file_start.epochs

    @property
    def usable_primary(self) -> float | None:
        return self.file_start.usable

    @property
    def passes_primary(self) -> bool | None:
        return self.file_start.passes

    @property
    def reasons(self) -> list[str]:
        return self.file_start.reasons

    @property
    def cause(self) -> str:
        return self.file_start.cause


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
    zero = (ep[:, 0] == 0) | (np.abs(phys) <= abs(gain) + 1e-12)
    rail = (ep[:, 0] == int(h.dig_min[i])) | (ep[:, 0] == int(h.dig_max[i]))
    out.n_zero_valued = int((const & zero).sum())
    out.n_const_rail = int((const & rail).sum())
    out.n_const_other = int((const & ~zero & ~rail).sum())
    return out


def _mini_edf(raw) -> bytes:
    """The fetched records as a self-contained plain EDF (header patched to the fetched record count, EDF+ marker
    blanked so pyedflib reads annotations as an ordinary signal). Record 0 of it is record ``raw.plan.r0`` of the file."""
    h = raw.hdr
    n = len(raw.data) // h.record_bytes
    head = bytearray(raw.header_seg[: h.header_bytes])
    head[192:236] = b" " * 44
    head[236:244] = str(n).encode("ascii").ljust(8)
    return bytes(head) + raw.data[: n * h.record_bytes]


def _crosscheck(raw, first: dict[str, int], rec0, ws: WindowStats) -> None:
    """Decode the required channels of ``rec0`` (our ranged reader) AND with pyedflib on the same in-memory records
    (an anonymous ``memfd``, never a disk file) and record max |difference| (uV) and correlation per channel."""
    try:
        import pyedflib
    except Exception:  # noqa: BLE001
        ws.xcheck_status = "unavailable"
        return
    if not hasattr(os, "memfd_create"):
        ws.xcheck_status = "unavailable"
        return
    h = raw.hdr
    fd = None
    try:
        fd = os.memfd_create("edf_xcheck")
        os.write(fd, _mini_edf(raw))
        with pyedflib.EdfReader(f"/proc/self/fd/{fd}") as ref:
            skipped = 0
            for c, i in first.items():
                if c not in rec0.ch_names:
                    continue
                if abs(float(h.fs[i]) - rec0.fs) > 1e-9:
                    skipped += 1
                    continue
                theirs = ref.readSignal(i) * _uv_scale(h.phys_dim[i])
                a = int(round((rec0.offset_s - raw.plan.r0 * h.record_duration) * rec0.fs))
                ours = rec0.data[rec0.ch_names.index(c)]
                t = theirs[a:a + len(ours)]
                if len(t) != len(ours):
                    ws.xcheck_status = "length_mismatch"
                    return
                ws.xcheck_max_abs_diff_uv.append(float(np.max(np.abs(ours - t))) if len(ours) else 0.0)
                if np.std(ours) > 0 and np.std(t) > 0:
                    ws.xcheck_corr.append(float(np.corrcoef(ours, t)[0, 1]))
                else:
                    ws.xcheck_n_constant_pairs += 1
            ws.xcheck_status = "rate_mismatch" if skipped and not ws.xcheck_max_abs_diff_uv else "ok"
    except Exception as exc:  # noqa: BLE001
        ws.xcheck_status = "error:" + type(exc).__name__
    finally:
        if fd is not None:
            os.close(fd)


def _window_stats(raw, first: dict[str, int], pw: WindowSpec, qc_cfg: QCConfig | None) -> WindowStats:
    """Constant-epoch counts, QC outcome (with the fixed normaliser / dead-channel handling), per-rule flag
    fractions, amplitude and line-noise statistics for the window ``pw`` inside the fetched ``raw`` records."""
    cfg = qc_cfg or QCConfig()
    h, ws = raw.hdr, WindowStats()
    recs = raw.records()
    for c, i in first.items():
        ws.epochs[c] = _channel_epochs(h, i, recs, raw.plan.r0, pw.start_s, pw.duration_s, cfg.epoch_s)
    start = max(0.0, pw.start_s - PAD_S)
    try:
        rec = read_edf(raw.sparse(), start_s=start, duration_s=pw.end_s + PAD_S - start, channels=list(CANONICAL_19),
                       allow_discontinuous=True)
        rec0 = rec
        _crosscheck(raw, first, rec0, ws)
        rec = drop_dead_channels(select_channels(rec))
        notes = {"dead": rec.meta.get("dead_channels", []), "invalid_scaling": rec.meta.get("invalid_scaling_channels", [])}
        qcs, _ = qc_recording(rec.data, rec.fs, rec.ch_names, rec.offset_s, {"primary": pw}, cfg,
                              rec_duration_s=rec.meta.get("edf_duration_s"), channel_notes=notes)
        q: WindowQC = qcs["primary"]
        ws.usable, ws.passes, ws.reasons = float(q.usable_fraction), bool(q.passes), list(q.reasons)
        ws.flag_frac = {k: float(v) for k, v in q.flag_fraction.items()}
        a = int(round((pw.start_s - rec.offset_s) * rec.fs))
        L = int(round(cfg.epoch_s * rec.fs))
        n_ep = int(round(pw.duration_s / cfg.epoch_s))
        for c in REQUIRED:
            if c not in rec.ch_names:
                continue
            x = rec.data[rec.ch_names.index(c)][a:a + n_ep * L]
            if len(x) < L:
                continue
            ws.amp_std_uv.append(float(np.std(x)))
            ws.amp_p99_uv.append(float(np.percentile(np.abs(x - np.median(x)), 99)))
            ne = len(x) // L
            ratio = line_noise_ratio(x[: ne * L].reshape(1, ne, L), rec.fs, cfg)[0]
            ws.line_ratio.append(float(np.median(ratio)))
    except Exception as exc:  # noqa: BLE001
        ws.reasons = ["read_error:" + type(exc).__name__]
    return ws


def inspect_signals(s3, ref: RecordingRef, *, bucket: str | None = None, policy=None, max_attempts: int = 4,
                    sleep=None, parent_fallback: bool = False, qc_cfg: QCConfig | None = None,
                    onset_max_search_s: float = ONSET_MAX_SEARCH_S) -> SignalInfo:
    """Header, file-start window and after-onset window statistics for one recording (ranged GETs only)."""
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
    kw = {} if sleep is None else {"sleep": sleep}
    try:
        raw = fetch_raw_window(s3, res.key, max(0.0, pw.start_s - PAD_S), pw.end_s + PAD_S - max(0.0, pw.start_s - PAD_S),
                               bucket=bucket, max_attempts=max_attempts, allow_discontinuous=True, **kw)
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
    spr = {int(h.samples_per_record[i]) for i in eeg_idx}
    info.n_distinct_spr, info.hetero_spr = len(spr), len(spr) > 1
    info.req_calibration = [f"{h.phys_dim[i] or '-'}|{h.phys_min[i]:g}|{h.phys_max[i]:g}|{h.dig_min[i]:g}|{h.dig_max[i]:g}"
                            for i in first.values()]
    info.file_start = _window_stats(raw, first, pw, qc_cfg)
    info.file_start.cause = _cause(info, info.file_start)

    # t0 (D-109) and the primary window placed after it
    try:
        on = find_signal_onset(s3, res.key, bucket=bucket, max_attempts=max_attempts, max_search_s=onset_max_search_s,
                               allow_discontinuous=True, **kw)
        info.onset_s = on.onset_s
        info.onset_status = "found" if on.onset_s is not None else "none_within_limit"
    except _StreamError as e:
        info.onset_status = "no_channels" if e.reason == FailureReason.NO_EEG_CHANNELS else "error"
    except Exception:  # noqa: BLE001
        info.onset_status = "error"
    if info.onset_s is not None:
        if info.onset_s == 0.0:
            info.after_onset = info.file_start
        else:
            pw2 = WindowSpec("primary", info.onset_s + pw.start_s, pw.duration_s)
            s0 = max(0.0, pw2.start_s - PAD_S)
            try:
                raw2 = fetch_raw_window(s3, res.key, s0, pw2.end_s + PAD_S - s0, bucket=bucket,
                                        max_attempts=max_attempts, allow_discontinuous=True, **kw)
                info.after_onset = _window_stats(raw2, first, pw2, qc_cfg)
            except Exception as exc:  # noqa: BLE001 - e.g. recording ends before the window: counted below
                info.after_onset = WindowStats(reasons=["window_unreadable:" + (
                    exc.reason if isinstance(exc, _StreamError) else type(exc).__name__)])
        info.after_onset.cause = _cause(info, info.after_onset)
    return info


def _cause(i: SignalInfo, ws: WindowStats) -> str:
    """One mutually exclusive technical cause per recording and window, in the order a fix would be applied."""
    if i.edf_type == "EDF+D":
        return "edf_plus_d_discontinuous"
    missing = [c for c in REQUIRED if c not in i.found]
    if len(missing) == len(REQUIRED):
        return "bipolar_labels_only" if i.n_bipolar else "no_required_labels_recognised"
    if missing:
        return "required_label_not_found"
    if any(r.startswith("window_unreadable:") for r in ws.reasons):
        return "window_unreadable_after_onset"
    if any(e.n_const_scaling for c, e in ws.epochs.items() if c in REQUIRED):
        return "required_channel_zero_calibration_range"
    if any(e.n_epochs and e.n_const_digital == e.n_epochs for c, e in ws.epochs.items() if c in REQUIRED):
        return "required_channel_constant_whole_window"
    if any(r.startswith("read_error:") for r in ws.reasons):
        return "decode_error"
    if ws.usable is not None and ws.usable < 0.6:
        return "low_usable_other_artifacts"
    return "no_technical_problem"


def _label_ok(lb: str) -> bool:
    return bool(_LABEL_OK.match(lb)) and not data_io.looks_id_like(lb)


def _share(num: int, den: int):
    return round(num / den, 4) if den else None


def _xcheck_block(stats: list[WindowStats], tc) -> dict:
    """Aggregate of our-reader-vs-pyedflib agreement over recordings (needs >= MIN_POOL recordings for quantiles)."""
    done = [w for w in stats if w.xcheck_status == "ok"]
    diff = [w.xcheck_max_abs_diff_uv for w in done]
    corr = [w.xcheck_corr for w in done]
    return {
        "status": {k: tc(c) for k, c in sorted(Counter(w.xcheck_status for w in stats).items())},
        "n_recordings_compared": tc(len(done)),
        "max_abs_difference_uv_quantiles": _pooled_q(diff),
        "n_recordings_with_any_difference_over_1e-6_uv": tc(sum(any(d > 1e-6 for d in r) for r in diff)),
        "correlation_quantiles": _pooled_q(corr),
        "n_recordings_with_any_correlation_below_0.999": tc(sum(any(c < 0.999 for c in r) for r in corr)),
        "n_constant_pairs_not_correlated": tc(sum(w.xcheck_n_constant_pairs for w in done)),
    }


def _const_block(stats: list[WindowStats], tc) -> dict:
    out: dict = {}
    for ch in REQUIRED:
        es = [w.epochs[ch] for w in stats if ch in w.epochs and w.epochs[ch].n_epochs]
        k = len(es)
        tot = sum(e.n_epochs for e in es)
        if k < MIN_POOL or not tot:
            out[ch] = {"n_recordings": tc(k), "pooled_fractions": SUPPRESSED}
            continue
        out[ch] = {
            "n_recordings": tc(k),
            "frac_epochs_const_digital": round(sum(e.n_const_digital for e in es) / tot, 4),
            "frac_epochs_const_by_zero_calibration": round(sum(e.n_const_scaling for e in es) / tot, 4),
            "frac_epochs_zero_valued": round(sum(e.n_zero_valued for e in es) / tot, 4),
            "of_constant_epochs_share_at_header_digital_rail": _share(sum(e.n_const_rail for e in es),
                                                                    sum(e.n_const_digital for e in es)),
            "of_constant_epochs_share_zero_valued": _share(sum(e.n_zero_valued for e in es),
                                                          sum(e.n_const_digital for e in es)),
            "of_constant_epochs_share_other_nonzero_value": _share(sum(e.n_const_other for e in es),
                                                                  sum(e.n_const_digital for e in es)),
            "n_recordings_all_epochs_const": tc(sum((e.n_const_digital + e.n_const_scaling) == e.n_epochs for e in es)),
            "n_recordings_over_half_epochs_const": tc(sum((e.n_const_digital + e.n_const_scaling) * 2 > e.n_epochs
                                                          for e in es)),
        }
    return out


def _pooled_q(per_recording: list[list[float]]) -> dict:
    """Quantiles pooled over recording x channel values; needs >= MIN_POOL recordings behind it."""
    vals = [v for r in per_recording for v in r]
    return safe_quantiles(vals) if len([r for r in per_recording if r]) >= MIN_POOL else \
        {f"q{q}": SUPPRESSED for q in ("10", "25", "50", "75", "90")}


def _after_onset_block(infos: list[SignalInfo], tc) -> dict:
    ok = [i for i in infos if i.status == "ok"]
    st = [(i, i.after_onset) for i in ok if i.after_onset is not None and i.onset_s is not None]
    ws_all = [w for _, w in st]
    qs = [w.usable for w in ws_all if w.usable is not None]
    fail = [w for w in ws_all if w.usable is not None and w.usable < 0.6]
    n_fail = len(fail)

    def breakdown(group: list[WindowStats]) -> dict:
        if len(group) < MIN_POOL:
            return {"n_recordings": tc(len(group)), "pooled_fractions": SUPPRESSED}
        rules = ("flat", "clipping", "extreme", "line_noise", "disconnected")
        return {
            "n_recordings": tc(len(group)),
            "mean_share_of_minimum_set_cells_flagged_by_rule": {
                r: round(float(np.mean([w.flag_frac.get(r, 0.0) for w in group])), 4) for r in rules},
            "n_recordings_rule_flags_over_5pct_of_cells": {
                r: tc(sum(w.flag_frac.get(r, 0.0) > 0.05 for w in group)) for r in rules},
            "mean_unusable_epoch_fraction": round(float(np.mean([1.0 - (w.usable or 0.0) for w in group])), 4),
            "required_channel_std_uv_quantiles": _pooled_q([w.amp_std_uv for w in group]),
            "required_channel_p99_abs_dev_uv_quantiles": _pooled_q([w.amp_p99_uv for w in group]),
            "line_noise_ratio_quantiles": _pooled_q([w.line_ratio for w in group]),
        }

    return {
        "n_with_onset_and_window": tc(len(st)),
        "primary_usable_fraction_quantiles": safe_quantiles(qs),
        "primary_pass": tc(sum(bool(w.passes) for w in ws_all)),
        "n_usable_below_threshold": tc(n_fail),
        "reason_counts": {r: tc(c) for r, c in sorted(Counter(r for w in ws_all for r in w.reasons).items())},
        "exactly_constant_epochs_by_electrode": _const_block(ws_all, tc),
        "technical_cause_per_recording": {k: tc(c) for k, c in sorted(Counter(w.cause for w in ws_all).items())},
        "pyedflib_crosscheck": _xcheck_block(ws_all, tc),
        "qc_rule_breakdown_all_recordings": breakdown(ws_all),
        "qc_rule_breakdown_usable_below_threshold": breakdown(fail),
    }


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
    onsets = [i.onset_s / 60.0 for i in ok if i.onset_s is not None]
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
        "samples_per_record": {
            "recordings_with_heterogeneous_samples_per_record": tc(sum(i.hetero_spr for i in ok)),
            "n_distinct_samples_per_record_among_non_annotation_signals": dist(i.n_distinct_spr for i in ok),
        },
        "required_channel_calibration_dim_pmin_pmax_dmin_dmax": {
            k: tc(c) for k, c in sorted(Counter(k for i in ok for k in set(i.req_calibration)).items(),
                                        key=lambda kv: (-kv[1], kv[0]))[:12]},
        "pyedflib_crosscheck_file_start_window": _xcheck_block([i.file_start for i in ok], tc),
        "primary_window_exactly_constant_epochs_by_electrode": _const_block([i.file_start for i in ok], tc),
        "technical_cause_per_recording": dist(i.cause for i in ok),
        "primary_usable_fraction_quantiles": safe_quantiles([i.usable_primary for i in ok if i.usable_primary is not None]),
        "primary_pass": tc(sum(bool(i.passes_primary) for i in ok)),
        "primary_reason_counts": {r: tc(c) for r, c in sorted(Counter(r for i in ok for r in i.reasons).items())},
        "signal_onset": {
            "definition": "t0 = start of the first 60-s period (10-s grid from file start) in which >= 8 of 10 required "
                          "electrodes are non-constant in >= 90% of 2-s epochs, searched within the first 120 min; "
                          "primary window = t0 + 1 to t0 + 11 min",
            "onset_status": dist(i.onset_status for i in ok),
            "n_no_sustained_signal_within_120_min": tc(sum(i.onset_status == "none_within_limit" for i in ok)),
            "n_onset_after_file_start": tc(sum(1 for i in ok if i.onset_s and i.onset_s > 0)),
            "onset_offset_minutes_from_file_start_quantiles": safe_quantiles(onsets),
        },
        "after_onset": _after_onset_block(infos, tc),
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
