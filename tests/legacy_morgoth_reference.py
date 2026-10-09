"""Frozen copy of the PRE-optimisation MORGOTH snippet normalisation and per-recording feature loop (Phase speed-up, 2026-10-09).

Used only by tests/test_rungs_speed.py to show that the optimised extractor reproduces the old outputs within float
tolerance. Do not edit; do not import from production code.
"""
from __future__ import annotations

import numpy as np

from sortinghat.eeg.io import drop_dead_channels, select_channels
from sortinghat.eeg.window import all_windows, qc_recording
from sortinghat.morgoth import preprocess as pp
from sortinghat.morgoth.features import MorgothConfig, _stats
from sortinghat.morgoth.heads import HEADS, STATS, feature_names


def snippet_valid(sn):
    live = sn[~np.isnan(sn).any(axis=1)]
    if live.size == 0:
        return False
    if np.all(np.abs(live) < 2) or np.all(np.abs(live) > 3000):
        return False
    c = live - live.mean(axis=0, keepdims=True)
    return not np.all((c.max(axis=1) - c.min(axis=1)) < 1)


def normalise_snippets(x, starts, win):
    S, C = len(starts), x.shape[0]
    out = np.zeros((S, C, win), dtype=np.float32)
    valid = np.zeros(S, bool)
    for i, s in enumerate(starts):
        sn = x[:, s:s + win]
        valid[i] = snippet_valid(sn)
        if not valid[i]:
            continue
        present = ~np.isnan(sn).any(axis=1)
        y = np.zeros_like(sn)
        y[present] = sn[present] - sn[present].mean(axis=0, keepdims=True)
        y = np.clip(y, -500.0, 500.0)
        lo, hi = y.min(axis=1, keepdims=True), y.max(axis=1, keepdims=True)
        span = np.where(hi - lo > 0, hi - lo, 1.0)
        out[i] = ((y - lo) / span * 200.0 - 100.0) / 100.0
    return out, valid


def process_morgoth(rec, windows, backend, cfg=None):
    cfg = cfg or MorgothConfig()
    windows = dict(windows or all_windows())
    rec = drop_dead_channels(select_channels(rec))
    total = rec.meta.get("edf_duration_s", rec.offset_s + rec.duration_s)
    notes = {"dead": rec.meta.get("dead_channels", []), "invalid_scaling": rec.meta.get("invalid_scaling_channels", [])}
    qcs, _ef = qc_recording(rec.data, rec.fs, rec.ch_names, rec.offset_s, windows, cfg.qc, rec_duration_s=total,
                            channel_notes=notes)
    run = {k for k, q in qcs.items() if q.passes or cfg.compute_failed}
    x, missing = pp.to_morgoth_order(rec)
    info = {"n_missing_channels": len(missing), "missing": missing, "ran": bool(run)}
    series = {}
    if run:
        t_lo = min(windows[k].start_s for k in run)
        t_hi = max(windows[k].end_s for k in run)
        filt = {}
        for name in cfg.heads:
            h = HEADS[name]
            spikes = h.window_s < 5
            if spikes not in filt:
                filt[spikes] = pp.filter_series(x, rec.fs, spikes=spikes)
            xf = filt[spikes]
            win = int(round(h.window_s * pp.TARGET_FS))
            step = cfg.spike_step_s if spikes else cfg.step_s
            a = int(round((t_lo - rec.offset_s) * pp.TARGET_FS))
            b = int(round((t_hi - rec.offset_s) * pp.TARGET_FS))
            a_c = max(a, 0)
            seg = xf[:, a_c:b]
            starts = pp.snippet_starts(seg.shape[1], win, max(1, int(round(step * pp.TARGET_FS))))
            snip, valid = normalise_snippets(seg, starts, win)
            probs = np.full((len(starts), h.n_out), np.nan)
            if valid.any():
                probs[valid] = backend.predict(h, snip[valid])
            t_start = rec.offset_s + (a_c + starts) / pp.TARGET_FS
            series[name] = (t_start, valid, probs)
    rows = []
    for wname, w in windows.items():
        q = qcs[wname]
        row = {"window": wname, "qc_pass": bool(q.passes), "usable_fraction": float(q.usable_fraction),
               "n_missing_channels": len(missing), "morgoth_valid_fraction": float("nan")}
        for c in feature_names(cfg.heads):
            row[c] = float("nan")
        if wname in run and series:
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
        rows.append(row)
    return rows, info
