#!/usr/bin/env python3
"""Streaming frozen-CBraMod embedding extraction over a list of HEEDB recordings (HUMAN-RUN via scripts/heedb_run.sh).

    HEEDB_AWS_PROFILE=<profile> scripts/heedb_run.sh python3 scripts/extract_embeddings.py \
        --input out/local_only/cohort/eeg_keys.csv --out-dir out/local_only/embeddings --shard 0 --of 4 --torch-threads 1

Same input file, key resolution, sharding, t0 (D-109 sustained-signal onset), ranged-S3 streaming, resumable parts + ledgers
and local_only conventions as ``scripts/extract_eeg_features.py`` (read its docstring); the per-window product is one
200-d frozen CBraMod embedding (``emb.cbramod.<j>``) instead of qEEG features. CPU inference only; the public checkpoint is
read from ``~/.cache/sortinghat/cbramod`` (``SORTINGHAT_CBRAMOD_DIR`` / ``--weights``), verified by SHA-256 (0792cb80...) and
never fetched during a real-data run unless ``--fetch-weights`` is given. Conformance, missing-channel handling (D-110) and
pooling: ``sortinghat/embed/cbramod.py`` and ``docs/embeddings.md``.

Records go ONLY to ``--out-dir`` (a ``local_only/`` directory, mode 0600): parquet parts keyed by ``recording_id`` x window,
ledgers, and ``embed_meta.json`` (model + config provenance; a directory is never mixed across configs). stdout and
``--summary`` carry aggregates only (counts < 11 print as "<11"), including CPU throughput in s/recording.
Run one process per shard; with N shards on N cores pass ``--torch-threads 1``. A crash between a part write and its ledger
write can duplicate a recording; readers keep one row per (recording_id, window).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))             # repo root, so `sortinghat` imports
sys.path.insert(0, str(HERE))                    # sibling scripts (extract_eeg_features)

import numpy as np  # noqa: E402

import extract_eeg_features as xef  # noqa: E402
from sortinghat.eeg.stream import ALL_REASONS, DEFAULT_TIMEOUT_S, FailureReason  # noqa: E402
from sortinghat.eeg.window import all_windows  # noqa: E402
from sortinghat.embed import weights as W  # noqa: E402
from sortinghat.embed.cbramod import AMP_POLICIES, REFERENCES, Embedder, EmbedConfig  # noqa: E402
from sortinghat.embed.store import HEAD_COLUMNS, require_local_only, write_meta, write_part  # noqa: E402
from sortinghat.embed.stream import stream_embeddings  # noqa: E402
from sortinghat.safe_output import (safe_print, safe_quantiles, safe_write_json, suppress_count,  # noqa: E402
                                    suppress_proportion)

LEDGER_FIELDS = xef.LEDGER_FIELDS


def summarize(attempted: int, n_ok: int, reasons: Counter, elapsed, fetch_s, proc_s, bytes_, retries: int, wall_s: float,
              skipped: int, windows: dict[str, list[dict]], meta: dict) -> dict:
    n_fail = attempted - n_ok
    show_ok = suppress_count(n_ok) if (n_fail == 0 or n_fail >= 11) else "<11"      # no recovery by subtraction
    out = {
        "attempted": suppress_count(attempted), "already_done_skipped": suppress_count(skipped),
        "succeeded": show_ok, "failed": suppress_count(n_fail), "success_rate": suppress_proportion(n_ok, attempted),
        "failure_reasons": {r: suppress_count(reasons.get(r, 0)) for r in ALL_REASONS if reasons.get(r, 0)},
        "model": meta,
        "throughput": {
            "device": "cpu", "torch_threads": meta.get("torch_threads"),
            "wall_seconds_per_recording": round(wall_s / attempted, 2) if attempted else None,
            "mean_fetch_seconds": round(float(np.mean(fetch_s)), 2) if fetch_s else None,
            "mean_process_seconds": round(float(np.mean(proc_s)), 2) if proc_s else None,
            "mean_bytes_fetched_per_recording": int(np.mean(bytes_)) if bytes_ else None,
            "mean_s3_retries_per_recording": round(retries / attempted, 3) if attempted else None,
            "seconds_per_recording_quantiles": safe_quantiles(elapsed),
        },
        "windows": {}}
    for w, rows in windows.items():
        n = len(rows)
        ok = [r for r in rows if r["emb_ok"]]
        out["windows"][w] = {
            "n": show_ok if show_ok == "<11" else suppress_count(n),
            "qc_pass_proportion": suppress_proportion(sum(r["qc_pass"] for r in rows), n),
            "embedded_proportion": suppress_proportion(len(ok), n),
            "n_segments_used_quantiles": safe_quantiles([r["emb_n_used"] for r in ok]),
            "valid_token_fraction_quantiles": safe_quantiles([r["emb_valid_token_frac"] for r in ok]),
            "share_segments_over_amp_quantiles": safe_quantiles([r["emb_frac_over_amp"] for r in ok])}
    return out


def main(argv=None, s3=None, embedder=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="recordings list under local_only/ (never printed)")
    ap.add_argument("--out-dir", default="out/local_only/embeddings", help="embedding parts (local_only/)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="attempt at most this many (after resume filtering)")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="per-recording seconds (fetch + embedding)")
    ap.add_argument("--max-attempts", type=int, default=4, help="per ranged GET")
    ap.add_argument("--flush-every", type=int, default=25, help="recordings per parquet part")
    ap.add_argument("--read-timeout", type=int, default=60, help="S3 socket read timeout, seconds")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--retry-permanent", action="store_true")
    ap.add_argument("--retry-reason", action="append", default=[], metavar="REASON")
    ap.add_argument("--windows", default=None, help="comma-separated subset of primary,20s,1min,2min,5min,10min (default all)")
    ap.add_argument("--no-onset", action="store_true", help="windows from the FILE start instead of the D-109 t0")
    ap.add_argument("--compute-failed", action="store_true", help="embed windows that fail QC too")
    ap.add_argument("--weights", default=None, help="checkpoint path (default: cache dir); SHA-256 is verified")
    ap.add_argument("--fetch-weights", action="store_true", help="download the public checkpoint into the cache if absent")
    ap.add_argument("--torch-threads", type=int, default=None, help="CPU threads for the model (1 per process when sharding)")
    ap.add_argument("--batch-size", type=int, default=30, help="10-s segments per forward pass")
    ap.add_argument("--reference", choices=REFERENCES, default="car")
    ap.add_argument("--amp-policy", choices=AMP_POLICIES, default="keep",
                    help="'drop' removes segments with any |x| > 100 uV from the pool (CBraMod pretraining rule)")
    ap.add_argument("--attention", action="store_true", help="also store the parameter-free attention-pooled embedding")
    ap.add_argument("--summary", default=None, help="aggregate JSON (safe_write_json)")
    args = ap.parse_args(argv)
    if not (0 <= args.shard < args.of):
        raise SystemExit("need 0 <= --shard < --of")
    inp, out_dir = Path(args.input), Path(args.out_dir)
    require_local_only(inp, "--input")
    require_local_only(out_dir, "--out-dir")
    bad = [r for r in args.retry_reason if r not in ALL_REASONS]
    if bad:
        raise SystemExit("--retry-reason must be one of " + ",".join(sorted(ALL_REASONS)))
    windows = all_windows()
    if args.windows:
        want = [w.strip() for w in args.windows.split(",") if w.strip()]
        if not want or any(w not in windows for w in want):
            raise SystemExit("--windows must be a non-empty subset of " + ",".join(windows))
        windows = {w: windows[w] for w in want}
    cfg = EmbedConfig(reference=args.reference, amp_policy=args.amp_policy, attention=args.attention,
                      compute_failed=args.compute_failed)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(out_dir, 0o700)

    if embedder is None:
        try:
            ckpt = W.ensure_checkpoint(args.weights, download=args.fetch_weights)
        except W.WeightsError as e:
            raise SystemExit(f"weights: {e}")
        embedder = Embedder.from_checkpoint(ckpt, batch_size=args.batch_size, num_threads=args.torch_threads)
        weights_id = W.CHECKPOINT_SHA256
    else:
        weights_id = "injected-for-tests"
    meta = {"model": "CBraMod (frozen, proj_out removed)", "upstream_repo": W.UPSTREAM_REPO,
            "upstream_commit": W.UPSTREAM_COMMIT, "weights_sha256": weights_id, "config": cfg.meta(),
            "windows": sorted(windows), "onset": "D-109 t0" if not args.no_onset else "file start",
            "embedding_dim": 200, "torch_threads": args.torch_threads}
    meta_file = {k: v for k, v in meta.items() if k != "torch_threads"}          # threads do not change what is stored
    write_meta(out_dir, meta_file)

    recs = [(r, k, p) for r, k, p in xef.load_recordings_with_parts(inp) if xef.shard_of(r, args.of) == args.shard]
    done = xef.load_done(out_dir, args.retry_permanent, set(args.retry_reason))
    todo = [(r, k, p) for r, k, p in recs if r not in done]
    if args.retry_reason:
        only = xef.load_failed_with(out_dir, set(args.retry_reason))
        todo = [(r, k, p) for r, k, p in todo if r in only]
    skipped = len(recs) - len(todo)
    if args.limit is not None:
        todo = todo[: args.limit]

    if s3 is None:
        from sortinghat import data_io
        s3 = data_io.make_client(args.profile, read_timeout=args.read_timeout)
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
    wrows: dict[str, list[dict]] = {}
    buf_rows: list[dict] = []
    buf_ledger: list[list] = []
    n_ok = retries = 0
    t_wall = time.monotonic()

    def flush():
        nonlocal buf_rows, buf_ledger
        if buf_rows:
            write_part(buf_rows, cfg, out_dir, tag)         # rows first, then the ledger: a crash re-does at most this batch
        for r in buf_ledger:
            lw.writerow(r)
        ledger.flush()
        buf_rows, buf_ledger = [], []

    def run_one(key):
        return stream_embeddings(s3, key, embedder, windows=windows, cfg=cfg, timeout_s=args.timeout,
                                 max_attempts=args.max_attempts, onset_search=not args.no_onset)

    try:
        from sortinghat import data_io
        for rid, key, parts in todo:
            res = run_one(key)
            if res.reason == FailureReason.NOT_FOUND and parts is not None:
                try:
                    found = data_io.resolve_edf_key(s3, *parts)
                except Exception:                           # noqa: BLE001 - keep the not_found result
                    found = None
                if found is not None and found.found and found.key != key:
                    res = run_one(found.key)
            elapsed.append(res.elapsed_s)
            bytes_.append(res.bytes_fetched)
            retries += res.n_retries
            if res.ok:
                n_ok += 1
                fetch_s.append(res.fetch_s)
                proc_s.append(res.process_s)
                buf_rows += [{"recording_id": rid, "onset_offset_s": float(res.onset_s or 0.0), **r} for r in res.rows]
                for r in res.rows:
                    wrows.setdefault(r["window"], []).append(r)
            else:
                reasons[res.reason] += 1
            buf_ledger.append([rid, "ok" if res.ok else "fail", res.reason or "", res.bytes_fetched, round(res.elapsed_s, 2)])
            if len(buf_ledger) >= args.flush_every:
                flush()
        flush()
    finally:
        ledger.close()
    wall = time.monotonic() - t_wall

    summ = summarize(len(todo), n_ok, reasons, elapsed, fetch_s, proc_s, bytes_, retries, wall, skipped, wrows,
                     {**meta_file, "torch_threads": args.torch_threads, "weights_sha256": weights_id[:8]})
    safe_print(f"shard {args.shard}/{args.of} | attempted: {summ['attempted']} | already done (skipped): "
               f"{summ['already_done_skipped']} | succeeded: {summ['succeeded']} | failed: {summ['failed']} "
               f"| success rate: {summ['success_rate']}")
    for r, c in summ["failure_reasons"].items():
        safe_print(f"  failure {r}: {c}")
    t = summ["throughput"]
    safe_print("CPU throughput (not patient data): wall s/recording", t["wall_seconds_per_recording"], "| mean fetch s",
               t["mean_fetch_seconds"], "| mean embed s", t["mean_process_seconds"], "| mean bytes fetched/recording",
               t["mean_bytes_fetched_per_recording"], "| torch threads", t["torch_threads"],
               "| weights sha256 prefix", summ["model"]["weights_sha256"])
    safe_print("seconds/recording quantiles:", t["seconds_per_recording_quantiles"])
    for w, e in summ["windows"].items():
        safe_print(f"window {w}: n {e['n']} | QC pass {e['qc_pass_proportion']} | embedded {e['embedded_proportion']} | "
                   f"segments used quantiles {e['n_segments_used_quantiles']} | valid-token-fraction quantiles "
                   f"{e['valid_token_fraction_quantiles']} | share of segments > 100 uV quantiles "
                   f"{e['share_segments_over_amp_quantiles']}")
    if args.summary:
        safe_write_json(args.summary, summ)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:                       # never print a traceback (it could carry values); class name only
        print(f"extract_embeddings failed: {type(e).__name__}", file=sys.stderr)
        raise SystemExit(1)
