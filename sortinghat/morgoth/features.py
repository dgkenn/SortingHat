"""Per-recording MORGOTH finding-probability features for the D-109 primary window and the nested windows.

Input: a raw ``Recording`` (uV, canonical 10-20 names, native rate; as returned by ``eeg.stream.fetch_window``) and
window specs relative to the recording (``eeg.window.all_windows()`` shifted by t0). One model pass runs over the
snippet grid that covers the windows; every window aggregates the snippets that lie wholly inside it:

    morgoth.<head>.<class>.{mean, p90, burden}     mean probability, 90th percentile, share of snippets with p >= 0.5

plus per-window ``morgoth_valid_fraction`` (valid 10-s snippets / snippets in the window) and ``n_missing_channels``.
A window with no valid snippet gets NaN features. Nested windows are subsets of the primary window, so a single pass
serves all of them. Snippet grid: 10-s snippets every ``step_s`` seconds (the EEG-level heads of the reference use a
1-s step; the default here is coarser because CPU inference of 591 snippets x 6 heads is minutes per recording);
the 1-s spike head every ``spike_step_s``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..eeg.prep import RecordingPrep
from ..eeg.window import QCConfig, WindowSpec, all_windows
from ..timing import stage
from . import preprocess as pp
from .heads import BURDEN_THRESHOLD, HEADS, STATS, default_heads, feature_names


@dataclass
class MorgothConfig:
    heads: tuple = field(default_factory=default_heads)
    step_s: float = 5.0
    spike_step_s: float = 1.0
    compute_failed: bool = False
    qc: QCConfig = field(default_factory=QCConfig)

    def columns(self) -> list[str]:
        return ["recording_id", "window", "qc_pass", "usable_fraction", "onset_offset_s", "n_missing_channels",
                "morgoth_valid_fraction", *feature_names(self.heads)]


def _stats(p: np.ndarray) -> tuple[float, float, float]:
    return float(np.mean(p)), float(np.quantile(p, 0.9)), float(np.mean(p >= BURDEN_THRESHOLD))


def process_morgoth(rec, windows: dict[str, WindowSpec] | None, backend, cfg: MorgothConfig | None = None, *,
                    prep: RecordingPrep | None = None, timers=None) -> tuple[list[dict], dict]:
    """Rows (one per window) and info for one raw Recording. Pure given the backend; raises nothing for QC failures.

    The model is run ONCE per head over the snippet grid that covers the windows that run (the primary window); each nested
    window then summarises the stored per-snippet outputs it contains. Heads that share a snippet grid (all 10-s heads) share
    the filtered series and the normalised snippet tensor. ``prep`` (``eeg.prep.RecordingPrep``) lets the CBraMod rung's
    channel selection and QC be reused; ``timers`` (``timing.StageTimers``) collects per-stage seconds."""
    cfg = cfg or MorgothConfig()
    windows = dict(windows or all_windows())
    if prep is None or not prep.compatible(windows, cfg.qc):
        prep = RecordingPrep(rec, windows, cfg.qc, timers)
    rec = prep.rec
    qcs, _ef = prep.qc()
    run = {k for k, q in qcs.items() if q.passes or cfg.compute_failed}
    x, missing = pp.to_morgoth_order(rec)
    info = {"n_missing_channels": len(missing), "missing": missing, "ran": bool(run)}
    series: dict[str, tuple] = {}          # head -> (start seconds, valid, probs)
    if run:
        t_lo = min(windows[k].start_s for k in run)
        t_hi = max(windows[k].end_s for k in run)
        filt: dict[bool, np.ndarray] = {}
        grids: dict[tuple, tuple] = {}     # (spikes, win, step) -> (starts, snippets, valid)
        for name in cfg.heads:
            h = HEADS[name]
            spikes = h.window_s < 5
            if spikes not in filt:
                with stage(timers, "morgoth.filter"):
                    filt[spikes] = pp.filter_series(x, rec.fs, spikes=spikes)
            xf = filt[spikes]
            win = int(round(h.window_s * pp.TARGET_FS))
            step = cfg.spike_step_s if spikes else cfg.step_s
            a = int(round((t_lo - rec.offset_s) * pp.TARGET_FS))
            b = int(round((t_hi - rec.offset_s) * pp.TARGET_FS))
            a_c = max(a, 0)
            gkey = (spikes, win, step)
            if gkey not in grids:
                with stage(timers, "morgoth.snippets"):
                    seg = xf[:, a_c:b]
                    starts = pp.snippet_starts(seg.shape[1], win, max(1, int(round(step * pp.TARGET_FS))))
                    snip, valid = pp.normalise_snippets(seg, starts, win)
                grids[gkey] = (starts, snip, valid)
            starts, snip, valid = grids[gkey]
            probs = np.full((len(starts), h.n_out), np.nan)
            if valid.any():
                with stage(timers, f"morgoth.model.{name}"):
                    probs[valid] = backend.predict(h, snip if valid.all() else snip[valid])
            t_start = rec.offset_s + (a_c + starts) / pp.TARGET_FS
            series[name] = (t_start, valid, probs)
        del filt, grids
    with stage(timers, "morgoth.aggregate"):
        rows = [_window_row(wname, w, qcs[wname], len(missing), cfg, series if wname in run else None)
                for wname, w in windows.items()]
    return rows, info


def _window_row(wname: str, w: WindowSpec, q, n_missing: int, cfg: MorgothConfig, series: dict | None) -> dict:
    """One output row: summaries of the stored per-snippet outputs of every head that lie wholly inside window ``w``."""
    row = {"window": wname, "qc_pass": bool(q.passes), "usable_fraction": float(q.usable_fraction),
           "n_missing_channels": n_missing, "morgoth_valid_fraction": float("nan")}
    for c in feature_names(cfg.heads):
        row[c] = float("nan")
    if not series:
        return row
    fr = []
    for name in cfg.heads:
        h = HEADS[name]
        t_start, valid, probs = series[name]
        inside = (t_start >= w.start_s - 1e-6) & (t_start + h.window_s <= w.end_s + 1e-6)
        use = inside & valid
        if h.window_s >= 5:
            fr.append(float(use.sum() / inside.sum()) if inside.sum() else float("nan"))
        if not use.any():
            continue
        p = probs[use]
        k0 = 0 if h.binary else 1
        for j, cname in enumerate(h.feature_classes):
            for sname, val in zip(STATS, _stats(p[:, k0 + j] if not h.binary else p[:, 0])):
                row[f"morgoth.{name}.{cname}.{sname}"] = val
    row["morgoth_valid_fraction"] = float(np.nanmean(fr)) if fr else float("nan")
    return row
