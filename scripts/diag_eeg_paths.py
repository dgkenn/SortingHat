#!/usr/bin/env python3
"""HEEDB EEG key-resolution diagnostics, AGGREGATES ONLY (HUMAN-RUN via a plain terminal, never inside an agent).

    HEEDB_AWS_PROFILE=<profile> python3 scripts/diag_eeg_paths.py --site S0001 --n 60 --seed 0

Samples N adult sessions from ``eeg_metadata`` (BidsFolder / SessionID / EEGFolder are used in code and never printed)
and for each tries the documented key pattern, the other task token, a task-less name and SessionID spelling
variants (``data_io.bids_edf_key`` / ``bids_edf_candidates``), then lists ONLY that recording's own session folder
(``EEG/bids/<site>/<BidsFolder>/ses-<id>/eeg/``, ``Delimiter='/'``) and takes the ``.edf`` that matches the session.
``--parent-listing`` additionally lists the subject's own folder (names matched in memory) when everything else failed.

Output (stdout and optional ``--out`` JSON; no key, folder name or ID is ever printed):
  * how many recordings resolved via each PATTERN NAME (documented, alt_task, no_task, *+sid_variant, folder_listing,
    parent_listing) and how many did not resolve;
  * how many session folders exist, exist without any .edf, hold exactly one or several .edf;
  * file-extension counts (folders containing, and total files).

Small cells: ``sortinghat.safe_output``. EXCEPTION (documented in ``safe_output.technical_count``): these are
technical FILE-LAYOUT counts, so when the sample has at least 50 recordings they are printed exactly, even below 11
(for example "7 of 60 have no .edf"); with fewer than 50 recordings n < 11 prints as "<11". Never extend this to
clinical or demographic counts.
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
    ap.add_argument("--parent-listing", action="store_true",
                    help="also list the subject's own folder (names matched in memory) if nothing else resolves")
    ap.add_argument("--out", default=None, help="aggregate JSON (safe_write_json)")
    a = ap.parse_args(argv)
    stage = "client"
    try:
        s3 = s3 or data_io.make_client(a.profile, read_timeout=a.read_timeout)
        stage = "sample"
        refs, n_avail = diag.sample_adult_recordings(s3, a.site, a.n, a.seed)
        stage = "inspect"
        infos = [diag.inspect_paths(s3, r, parent_fallback=a.parent_listing) for r in refs]
        stage = "aggregate"
        rep = diag.aggregate_paths(infos, n_avail)
        if a.out:
            safe_write_json(a.out, rep)
        for line in diag.format_lines(rep):
            safe_print(line)
    except Exception as exc:  # noqa: BLE001 - class name only: a message or traceback could quote a key
        safe_print(f"diag_eeg_paths FAILED at stage '{stage}': {type(exc).__name__} (message withheld); "
                   f"S3 read retries: {sum(data_io.RETRY_COUNTS.values())}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
