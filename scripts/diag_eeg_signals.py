#!/usr/bin/env python3
"""HEEDB EDF signal diagnostics, AGGREGATES ONLY (HUMAN-RUN via a plain terminal, never inside an agent).

    HEEDB_AWS_PROFILE=<profile> python3 scripts/diag_eeg_signals.py --site S0001 --n 60 --seed 0

Why: about a quarter of recordings had exactly constant minimum-set epochs and the median usable fraction was 0.
Samples N adult sessions from ``eeg_metadata`` (identifiers used in code only), resolves each EDF key robustly
(``data_io.resolve_edf_key``), fetches the header and minutes ~0.8-11.2 with ranged GETs (nothing is written to disk)
and reports ONLY:
  * distribution of the number of signals (total, and without the EDF annotation signal);
  * for each required electrode (Fp1 Fp2 F7 F8 T3 T4 T5 T6 O1 O2) how many recordings have it after label
    normalisation (``eeg.io.normalize_channel_name``);
  * the 40 most common RAW label strings with recording counts (equipment labels, not patient data; anything
    odd-looking or id-like is printed as <other>);
  * sample-rate distribution of the required electrodes;
  * counts of signals with dmin == dmax or pmin == pmax (calibration range zero => decodes to a constant);
  * EDF vs EDF+C vs EDF+D counts;
  * per required electrode, the share of primary-window (60-660 s) 2-s epochs that are EXACTLY constant, split into
    constant digital samples / constant because the calibration range is zero / zero-valued (zero-filled);
  * one technical cause per recording, and the primary-window usable-fraction quantiles and QC reason counts using
    the fixed normaliser / dead-channel handling.

Since D-109 (t0 = start of the first sustained live segment: the first 60-s period, 10-s grid, with >= 8 of 10 required
electrodes non-constant in >= 90% of 2-s epochs, searched within 120 min) it also reports, for the window placed AFTER t0:
t0-offset quantiles (minutes from file start), the number with no sustained segment in 120 min, the post-fix usable-fraction
quantiles and pass count, the exactly-constant epoch share per electrode, the technical cause, and for recordings still below
the 0.6 usable threshold the share of minimum-set cells removed by each QC rule (flat / clipping / extreme > 500 uV / line
noise / disconnected) plus required-channel amplitude (uV) and line-noise-ratio quantiles.

Small cells: ``sortinghat.safe_output``. EXCEPTION (``safe_output.technical_count``): counts of technical file
properties print exactly when the sample has at least 50 recordings, else n < 11 prints as "<11". Pooled fractions
and quantiles need at least 11 recordings behind them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # repo root, so `sortinghat` imports

from sortinghat import data_io  # noqa: E402
from sortinghat.eeg import diag  # noqa: E402
from sortinghat.safe_output import safe_print, safe_write_json  # noqa: E402


def main(argv=None, s3=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", required=True, help="site code, e.g. S0001")
    ap.add_argument("--n", type=int, default=60, help="recordings to sample (>= 50 for exact technical counts)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", default=None, help="AWS profile (default: HEEDB_AWS_PROFILE / AWS_PROFILE)")
    ap.add_argument("--read-timeout", type=int, default=60)
    ap.add_argument("--max-attempts", type=int, default=4, help="per ranged GET")
    ap.add_argument("--parent-listing", action="store_true",
                    help="also list the subject's own folder when the EDF key cannot be resolved otherwise")
    ap.add_argument("--out", default=None, help="aggregate JSON (safe_write_json)")
    a = ap.parse_args(argv)
    stage = "client"
    try:
        s3 = s3 or data_io.make_client(a.profile, read_timeout=a.read_timeout)
        stage = "sample"
        refs, n_avail = diag.sample_adult_recordings(s3, a.site, a.n, a.seed)
        stage = "inspect"
        infos = [diag.inspect_signals(s3, r, max_attempts=a.max_attempts, parent_fallback=a.parent_listing)
                 for r in refs]
        stage = "aggregate"
        rep = diag.aggregate_signals(infos, n_avail)
        if a.out:
            safe_write_json(a.out, rep)
        for line in diag.format_lines(rep):
            safe_print(line)
    except Exception as exc:  # noqa: BLE001 - class name only: a message or traceback could quote a key
        safe_print(f"diag_eeg_signals FAILED at stage '{stage}': {type(exc).__name__} (message withheld); "
                   f"S3 read retries: {sum(data_io.RETRY_COUNTS.values())}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
