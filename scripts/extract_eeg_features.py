#!/usr/bin/env python3
"""Streaming EEG feature extraction over a list of HEEDB recordings (HUMAN-RUN via scripts/heedb_run.sh).

    HEEDB_AWS_PROFILE=<profile> scripts/heedb_run.sh python3 scripts/extract_eeg_features.py \
        --input out/local_only/cohort/eeg_keys.csv --out-dir out/local_only/features --shard 0 --of 4

Each recording is read with ranged S3 GETs (header + minutes 1-11 plus filter padding only), decoded in memory and
turned into per-window feature rows by ``sortinghat.eeg.stream`` (nothing raw is written to disk). Only DERIVED
feature rows are stored, as parquet parts under a ``local_only/`` directory (mode 0600). stdout and the optional
``--summary`` JSON carry aggregates only, through ``sortinghat.safe_output`` (counts < 11 print as "<11").

Input (must live under a ``local_only/`` directory; never printed). CSV / TSV / parquet / plain text with
  * a key column ``edf_key`` (or ``key``) holding the S3 key of the EDF, OR
  * ``BidsFolder`` + ``SessionID`` (+ ``SiteID`` or ``site``, optional ``EEGFolder``) from which the key is built
    with ``data_io.bids_edf_key``; plain text = one key per line.
An optional ``recording_id`` column gives the opaque ID stored with the rows (default: salted-free SHA-256 prefix of
the key). Sharding is by hash of that ID (``--shard k --of n``), so it is stable if the list grows.

Resumable: parts (``part-*.parquet``) and ledgers (``ledger-*.csv``) in ``--out-dir`` list finished recordings;
a rerun skips recordings that succeeded or failed permanently (``--retry-permanent`` retries those too).
Run one process per shard for parallelism. A crash between the part write and the ledger write can duplicate a
recording's rows; readers should drop duplicates on (recording_id, window).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # repo root, so `sortinghat` imports

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sortinghat.eeg.features import FeatureConfig, feature_names  # noqa: E402
from sortinghat.eeg.stream import ALL_REASONS, DEFAULT_TIMEOUT_S, FailureReason, stream_features  # noqa: E402
from sortinghat.eeg.window import FLAG_NAMES, USABLE_THRESHOLD, all_windows  # noqa: E402
from sortinghat.safe_output import (safe_print, safe_quantiles, safe_write_json, suppress_count,  # noqa: E402
                                    suppress_proportion)

KEY_COLS = ("edf_key", "key")
LEDGER_FIELDS = ("recording_id", "status", "reason", "bytes_fetched", "elapsed_s")


def _require_local_only(p: Path, what: str) -> None:
    if "local_only" not in p.resolve().parts:
        raise SystemExit(f"{what} must be under a local_only/ directory (gitignored; CLAUDE.md rule 5)")


def opaque_id(key: str) -> str:
    return "rec" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def shard_of(recording_id: str, n: int) -> int:
    return int(hashlib.sha256(recording_id.encode("utf-8")).hexdigest()[:8], 16) % n


def load_recordings(path: Path) -> list[tuple[str, str]]:
    """(recording_id, key) pairs, de-duplicated on recording_id. Never printed."""
    return [(r, k) for r, k, _ in load_recordings_with_parts(path)]


def load_recordings_with_parts(path: Path) -> list[tuple[str, str, tuple | None]]:
    """(recording_id, key, bids parts) triples, de-duplicated on recording_id. ``parts`` is
    ``(site, BidsFolder, SessionID, EEGFolder)`` when the key was BUILT from BidsFolder columns (so a missing key can
    be re-resolved by ``data_io.resolve_edf_key``), else ``None``. Never printed."""
    suf = path.suffix.lower()
    if suf == ".parquet":
        df = pd.read_parquet(path)
    elif suf in (".csv", ".tsv"):
        df = pd.read_csv(path, sep="\t" if suf == ".tsv" else ",", dtype=str, encoding="utf-8-sig")
    else:
        keys = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
        df = pd.DataFrame({"edf_key": keys})
    kc = next((c for c in KEY_COLS if c in df.columns), None)
    parts: list = [None] * len(df)
    if kc is not None:
        keys = df[kc].astype(str)
    elif {"BidsFolder", "SessionID"} <= set(df.columns):
        from sortinghat.data_io import bids_edf_key
        site_col = next((c for c in ("SiteID", "site", "Site") if c in df.columns), None)
        if site_col is None:
            raise SystemExit("input needs a site column (SiteID / site) to build keys from BidsFolder")
        eeg = df["EEGFolder"] if "EEGFolder" in df.columns else pd.Series([None] * len(df))
        parts = [(str(s), str(b), str(i), None if pd.isna(e) else str(e))
                 for s, b, i, e in zip(df[site_col], df["BidsFolder"], df["SessionID"], eeg)]
        keys = pd.Series([bids_edf_key(*p) for p in parts])
    else:
        raise SystemExit("input needs an edf_key/key column, or BidsFolder + SessionID + a site column")
    ids = df["recording_id"].astype(str) if "recording_id" in df.columns else keys.map(opaque_id)
    seen, out = set(), []
    for rid, k, p in zip(ids, keys, parts):
        if rid not in seen and k and k != "nan":
            seen.add(rid)
            out.append((rid, k, p))
    return out


def load_done(out_dir: Path, retry_permanent: bool) -> set[str]:
    """Recording IDs that need no further attempt (succeeded, or failed permanently)."""
    done: set[str] = set()
    for p in out_dir.glob("part-*.parquet"):
        try:
            done.update(pd.read_parquet(p, columns=["recording_id"])["recording_id"].astype(str))
        except Exception:                                   # unreadable part: its rows get recomputed
            continue
    for p in out_dir.glob("ledger-*.csv"):
        with open(p, newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("status") == "ok" or (not retry_permanent and row.get("reason") in FailureReason.PERMANENT):
                    done.add(row["recording_id"])
    return done


def fixed_columns(feat_cfg: FeatureConfig) -> list[str]:
    qc_cols = ["qc_clean_cell_fraction", "qc_coverage_fraction", "qc_n_disconnected",
               *[f"qc_flag_{k}" for k in FLAG_NAMES]]
    return ["recording_id", "window", "qc_pass", "usable_fraction", *qc_cols, *feature_names(feat_cfg)]


def write_part(rows: list[dict], cols: list[str], out_dir: Path, tag: str) -> None:
    df = pd.DataFrame(rows).reindex(columns=cols)
    for c in cols[4:]:
        df[c] = df[c].astype("float64")
    df["qc_pass"] = df["qc_pass"].astype(bool)
    p = out_dir / f"part-{tag}-{uuid.uuid4().hex[:10]}.parquet"
    df.to_parquet(p, index=False)
    os.chmod(p, 0o600)


def summarize(attempted: int, n_ok: int, reasons: Counter, elapsed: list[float], bytes_: list[int],
              fetch_s: list[float], proc_s: list[float], retries: int, usable: dict[str, list[float]],
              passes: dict[str, list[bool]], wall_s: float, skipped: int, chan: Counter | None = None,
              resolved: Counter | None = None) -> dict:
    n_fail = attempted - n_ok
    show_ok = suppress_count(n_ok) if (n_fail == 0 or n_fail >= 11) else "<11"      # no recovery by subtraction
    out = {
        "attempted": suppress_count(attempted),
        "already_done_skipped": suppress_count(skipped),
        "succeeded": show_ok,
        "failed": suppress_count(n_fail),
        "success_rate": suppress_proportion(n_ok, attempted),
        "failure_reasons": {r: suppress_count(reasons.get(r, 0)) for r in ALL_REASONS if reasons.get(r, 0)},
        "throughput": {
            "wall_seconds_per_recording": round(wall_s / attempted, 2) if attempted else None,
            "mean_fetch_seconds": round(float(np.mean(fetch_s)), 2) if fetch_s else None,
            "mean_process_seconds": round(float(np.mean(proc_s)), 2) if proc_s else None,
            "mean_bytes_fetched_per_recording": int(np.mean(bytes_)) if bytes_ else None,
            "mean_s3_retries_per_recording": round(retries / attempted, 3) if attempted else None,
            "seconds_per_recording_quantiles": safe_quantiles(elapsed),
        },
        "windows": {},
    }
    if chan:                          # recordings with >= 1 minimum-set channel missing / dead / zero-calibrated
        out["minimum_set_channel_problems"] = {k: suppress_count(v) for k, v in sorted(chan.items())}
    if resolved:                      # not_found recordings recovered by the fallback key resolution, by pattern NAME
        out["key_fallback_resolved_by_pattern"] = {k: suppress_count(v) for k, v in sorted(resolved.items())}
    for w, vals in usable.items():
        out["windows"][w] = {"n": show_ok if show_ok == "<11" else suppress_count(len(vals)),   # n = succeeded: same rule
                             "usable_fraction_quantiles": safe_quantiles(vals),
                             "pass_proportion": suppress_proportion(sum(passes[w]), len(passes[w]))}
    out["usable_threshold"] = USABLE_THRESHOLD
    return out


def main(argv=None, s3=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="recordings list under local_only/ (never printed)")
    ap.add_argument("--out-dir", default="out/local_only/features", help="derived feature parts (local_only/)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="attempt at most this many (after resume filtering)")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="per-recording seconds (fetch + features)")
    ap.add_argument("--max-attempts", type=int, default=4, help="per ranged GET")
    ap.add_argument("--flush-every", type=int, default=25, help="recordings per parquet part")
    ap.add_argument("--read-timeout", type=int, default=60, help="S3 socket read timeout, seconds")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--retry-permanent", action="store_true")
    ap.add_argument("--windows", default=None, help="comma-separated subset of primary,20s,1min,2min,5min,10min "
                    "(default all; the byte range fetched is the same up to the longest window requested)")
    ap.add_argument("--compute-failed", action="store_true", help="features even for windows failing QC")
    ap.add_argument("--summary", default=None, help="aggregate JSON (safe_write_json)")
    args = ap.parse_args(argv)
    if not (0 <= args.shard < args.of):
        raise SystemExit("need 0 <= --shard < --of")
    inp, out_dir = Path(args.input), Path(args.out_dir)
    _require_local_only(inp, "--input")
    _require_local_only(out_dir, "--out-dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(out_dir, 0o700)

    recs = [(r, k, p) for r, k, p in load_recordings_with_parts(inp) if shard_of(r, args.of) == args.shard]
    done = load_done(out_dir, args.retry_permanent)
    todo = [(r, k, p) for r, k, p in recs if r not in done]
    skipped = len(recs) - len(todo)
    if args.limit is not None:
        todo = todo[: args.limit]

    if s3 is None:
        from sortinghat import data_io
        s3 = data_io.make_client(args.profile, read_timeout=args.read_timeout)
    feat_cfg = FeatureConfig()
    windows = all_windows()
    if args.windows:
        want = [w.strip() for w in args.windows.split(",") if w.strip()]
        if not want or any(w not in windows for w in want):
            raise SystemExit("--windows must be a non-empty subset of " + ",".join(windows))
        windows = {w: windows[w] for w in want}
    cols = fixed_columns(feat_cfg)
    tag = f"s{args.shard}of{args.of}"
    ledger_path = out_dir / f"ledger-{tag}.csv"
    new_ledger = not ledger_path.exists()
    ledger = open(ledger_path, "a", newline="")
    os.chmod(ledger_path, 0o600)
    lw = csv.writer(ledger)
    if new_ledger:
        lw.writerow(LEDGER_FIELDS)

    reasons: Counter = Counter()
    elapsed, bytes_, fetch_s, proc_s = [], [], [], []
    usable: dict[str, list[float]] = {}
    passes: dict[str, list[bool]] = {}
    buf_rows: list[dict] = []
    buf_ledger: list[list] = []
    n_ok = retries = 0
    chan: Counter = Counter()
    resolved: Counter = Counter()
    t_wall = time.monotonic()

    def flush():
        nonlocal buf_rows, buf_ledger
        if buf_rows:
            write_part(buf_rows, cols, out_dir, tag)        # rows first, then the ledger: a crash re-does at most this batch
        for r in buf_ledger:
            lw.writerow(r)
        ledger.flush()
        buf_rows, buf_ledger = [], []

    try:
        from sortinghat import data_io
        for rid, key, parts in todo:
            res = stream_features(s3, key, timeout_s=args.timeout, max_attempts=args.max_attempts,
                                  compute_failed=args.compute_failed, feat_cfg=feat_cfg, windows=windows)
            if res.reason == FailureReason.NOT_FOUND and parts is not None:
                # documented key missing: try the other task token / name variants, then list ONLY this recording's
                # own session folder (data_io.resolve_edf_key). Keys are never printed; only the pattern name counts.
                try:
                    found = data_io.resolve_edf_key(s3, *parts)
                except Exception:                           # noqa: BLE001 - keep the not_found result
                    found = None
                if found is not None and found.found and found.key != key:
                    res = stream_features(s3, found.key, timeout_s=args.timeout, max_attempts=args.max_attempts,
                                          compute_failed=args.compute_failed, feat_cfg=feat_cfg, windows=windows)
                    if res.ok:
                        resolved[found.pattern] += 1
            elapsed.append(res.elapsed_s)
            bytes_.append(res.bytes_fetched)
            retries += res.n_retries
            if res.ok:
                n_ok += 1
                chan["any_missing"] += res.n_missing_min > 0
                chan["any_dead_constant"] += res.n_dead_min > 0
                chan["any_zero_calibration"] += res.n_invalid_min > 0
                fetch_s.append(res.fetch_s)
                proc_s.append(res.process_s)
                buf_rows += [{"recording_id": rid, **r} for r in res.rows]
                for r in res.rows:
                    usable.setdefault(r["window"], []).append(float(r["usable_fraction"]))
                    passes.setdefault(r["window"], []).append(bool(r["qc_pass"]))
            else:
                reasons[res.reason] += 1
            buf_ledger.append([rid, "ok" if res.ok else "fail", res.reason or "", res.bytes_fetched,
                               round(res.elapsed_s, 2)])
            if len(buf_ledger) >= args.flush_every:
                flush()
        flush()
    finally:
        ledger.close()
    wall = time.monotonic() - t_wall

    summ = summarize(len(todo), n_ok, reasons, elapsed, bytes_, fetch_s, proc_s, retries, usable, passes, wall,
                     skipped, +chan, +resolved)
    safe_print(f"shard {args.shard}/{args.of} | attempted: {summ['attempted']} | already done (skipped): "
               f"{summ['already_done_skipped']} | succeeded: {summ['succeeded']} | failed: {summ['failed']} "
               f"| success rate: {summ['success_rate']}")
    for r, c in summ["failure_reasons"].items():
        safe_print(f"  failure {r}: {c}")
    for r, c in summ.get("key_fallback_resolved_by_pattern", {}).items():
        safe_print(f"  key fallback resolved via {r}: {c}")
    for r, c in summ.get("minimum_set_channel_problems", {}).items():
        safe_print(f"  recordings with minimum-set channel problem {r}: {c}")
    t = summ["throughput"]
    safe_print("throughput (not patient data): wall s/recording", t["wall_seconds_per_recording"],
               "| mean fetch s", t["mean_fetch_seconds"], "| mean process s", t["mean_process_seconds"],
               "| mean bytes fetched/recording", t["mean_bytes_fetched_per_recording"],
               "| mean S3 retries/recording", t["mean_s3_retries_per_recording"])
    safe_print("seconds/recording quantiles:", t["seconds_per_recording_quantiles"])
    for w, e in summ["windows"].items():
        safe_print(f"window {w}: n {e['n']} | pass proportion {e['pass_proportion']} | usable-fraction quantiles "
                   f"{e['usable_fraction_quantiles']}")
    if args.summary:
        safe_write_json(args.summary, summ)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:                       # never print a traceback (it could carry values); class name only
        print(f"extract_eeg_features failed: {type(e).__name__}", file=sys.stderr)
        raise SystemExit(1)
