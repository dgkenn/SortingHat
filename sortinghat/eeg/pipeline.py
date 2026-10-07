"""End-to-end Study 1 EEG front end: EDF -> QC -> preprocessing -> one feature vector per recording x window.

``process_recording`` is pure (path or ``Recording`` in, rows out). ``main`` is the batch entry point a HUMAN runs
from a plain terminal on real data; it writes per-recording feature rows only under ``local_only/`` (mode 0600)
and prints / writes only aggregate QC through ``sortinghat.safe_output`` (CLAUDE.md rules 2, 3, 5).

    python -m sortinghat.eeg.pipeline --manifest local_only/eeg_manifest.csv \
        --out local_only/eeg_features.csv --qc-summary out/eeg_qc_summary.json

Manifest columns: ``recording_id`` (opaque, caller-chosen) and ``edf_path``.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..agent_safety import assert_not_restricted_in_agent
from ..safe_output import safe_print
from .features import FeatureConfig, extract_features
from .io import CANONICAL_19, Recording, drop_dead_channels, read_edf, select_channels
from .preprocess import PreprocessConfig, common_average_masked, preprocess
from .window import (QCConfig, WindowQC, all_windows, extract_window, qc_recording, summarize_window_qc,
                     window_clean_mask, write_qc_summary)

PAD_S = 10.0     # extra signal read either side of the windows so zero-phase filter edges fall outside them


@dataclass
class RecordingResult:
    rows: list[dict]                      # one dict per window: window, qc_pass, usable_fraction, <features...>
    qc: dict[str, WindowQC]
    channel_status: dict = field(default_factory=dict)


def process_recording(src, windows=None, qc_cfg: QCConfig | None = None, pre_cfg: PreprocessConfig | None = None,
                      feat_cfg: FeatureConfig | None = None, compute_failed: bool = False) -> RecordingResult:
    """``src``: EDF path / file-like, or an already-read ``Recording`` (uV, raw, with ``offset_s`` and
    ``meta['edf_duration_s']``). Windows default to primary + nested. Features are computed only for windows
    that pass QC unless ``compute_failed``."""
    windows = dict(windows or all_windows())
    qc_cfg = qc_cfg or QCConfig()
    pre_cfg = pre_cfg or PreprocessConfig()
    feat_cfg = feat_cfg or FeatureConfig()
    if isinstance(src, Recording):
        rec = src
    else:
        if isinstance(src, (str, os.PathLike)):
            assert_not_restricted_in_agent(src)
        t0 = max(0.0, min(w.start_s for w in windows.values()) - PAD_S)
        t1 = max(w.end_s for w in windows.values()) + PAD_S
        rec = read_edf(src, start_s=t0, duration_s=t1 - t0, channels=list(CANONICAL_19))
    rec = drop_dead_channels(select_channels(rec))      # exactly-constant (zero-filled) channels are MISSING, not flat
    total = rec.meta.get("edf_duration_s", rec.offset_s + rec.duration_s)
    notes = {"dead": rec.meta.get("dead_channels", []), "invalid_scaling": rec.meta.get("invalid_scaling_channels", [])}
    qcs, ef = qc_recording(rec.data, rec.fs, rec.ch_names, rec.offset_s, windows, qc_cfg, rec_duration_s=total,
                           channel_notes=notes)

    x, fs = preprocess(rec.data, rec.fs, pre_cfg)
    disc = ef.flags["disconnected"].all(axis=1) if ef.flags["disconnected"].size else None
    ep_samples = int(round(qc_cfg.epoch_s * fs))
    start = int(round((ef.start_s - rec.offset_s) * fs))
    x = common_average_masked(x, ef.clean, ep_samples, start, base_exclude=disc)
    rows = []
    for name, w in windows.items():
        q = qcs[name]
        row = {"window": name, "qc_pass": q.passes, "usable_fraction": q.usable_fraction}
        if q.passes or compute_failed:
            seg = extract_window(x, fs, w, rec.offset_s)
            row.update(extract_features(seg, fs, rec.ch_names, window_clean_mask(ef, w, rec.ch_names), feat_cfg))
        rows.append(row)
    minimum = set((qc_cfg or QCConfig()).minimum_channels)
    miss = [c for c in rec.meta.get("missing_channels", [])]
    status = {"n_channels": len(rec.ch_names), "missing": miss, "dead": notes["dead"],
              "invalid_scaling": notes["invalid_scaling"],
              "n_missing_min": len(minimum & set(miss)) - len(minimum & set(notes["invalid_scaling"])),
              "n_dead_min": len(minimum & set(notes["dead"])),
              "n_invalid_min": len(minimum & set(notes["invalid_scaling"]))}
    return RecordingResult(rows, qcs, status)


def main(argv=None) -> int:
    import pandas as pd
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True, help="per-recording feature rows (CSV); keep under local_only/")
    ap.add_argument("--qc-summary", required=True, help="aggregate QC JSON (safe_output)")
    ap.add_argument("--compute-failed", action="store_true")
    args = ap.parse_args(argv)
    man = pd.read_csv(args.manifest)
    all_rows, all_qc, n_err = [], [], 0
    for rid, path in zip(man["recording_id"], man["edf_path"]):
        try:
            res = process_recording(path, compute_failed=args.compute_failed)
        except Exception:                       # never print the path / id / message (record-level)
            n_err += 1
            continue
        all_qc.append(res.qc)
        for r in res.rows:
            all_rows.append({"recording_id": rid, **r})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out, index=False)
    os.chmod(out, 0o600)
    write_qc_summary(args.qc_summary, all_qc)
    summ = summarize_window_qc(all_qc)
    from ..safe_output import suppress_count
    safe_print("recordings processed:", summ["n_recordings"], "| failed to read:", suppress_count(n_err))
    for w, e in summ["windows"].items():
        safe_print(f"window {w}: pass proportion {e['pass_proportion']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
