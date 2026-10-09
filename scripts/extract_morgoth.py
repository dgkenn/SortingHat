#!/usr/bin/env python3
"""MORGOTH finding-probability features and exposure accounting (HUMAN-RUN via scripts/heedb_run.sh; research-only lineage).

    # one-off: weights (BDSP projects access point) + pinned code into ~/.cache/sortinghat/morgoth
    scripts/heedb_run.sh python3 -m sortinghat.morgoth.weights --fetch

    # per-recording features (streaming, sharded, resumable; same input list as extract_eeg_features.py)
    HEEDB_AWS_PROFILE=<profile> scripts/heedb_run.sh python3 scripts/extract_morgoth.py features \
        --input out/local_only/recording_keys.csv --out-dir out/local_only/morgoth --shard 0 --of 4

    # exposure accounting: which cohort patients are in MORGOTH's own lists (per-patient flags -> local_only)
    python3 scripts/extract_morgoth.py exposure --cohort out/local_only/cohort_study1.csv \
        --lists out/local_only/morgoth_lists/*.xlsx --out out/local_only/morgoth/exposure.parquet

``features`` reads each EDF with ranged S3 GETs (t0 = first sustained live segment, D-109; primary window t0 + 1 to
t0 + 11 min plus the nested windows), runs the MORGOTH event-level heads on CPU over 10-s snippets, and stores one row per
recording x window as parquet parts under a ``local_only/`` directory (mode 0600). Nothing raw is written. stdout and
``--summary`` carry aggregates only (``sortinghat.safe_output``, counts < 11 print as "<11").

Resumable like the EEG extractor: ``part-*.parquet`` and ``ledger-*.csv`` in ``--out-dir`` list finished recordings; a
rerun skips those that succeeded or failed permanently (``--retry-permanent`` / ``--retry-reason`` as there). Run one
process per shard. ``--backend stub`` exists for tests and dry runs; on real data it needs ``--allow-stub`` and the output
is meaningless (the run info file says so).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # repo root, so `sortinghat` imports
sys.path.insert(0, str(Path(__file__).resolve().parent))         # scripts/, for extract_eeg_features

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import extract_eeg_features as base  # noqa: E402
from sortinghat.eeg.stream import ALL_REASONS, FailureReason  # noqa: E402
from sortinghat.eeg.window import USABLE_THRESHOLD, all_windows  # noqa: E402
from sortinghat.morgoth import exposure as expo  # noqa: E402
from sortinghat.morgoth import weights  # noqa: E402
from sortinghat.morgoth.features import MorgothConfig  # noqa: E402
from sortinghat.morgoth.heads import HEADS, default_heads, feature_names  # noqa: E402
from sortinghat.morgoth.model import make_backend  # noqa: E402
from sortinghat.morgoth.stream import DEFAULT_TIMEOUT_S, stream_morgoth  # noqa: E402
from sortinghat.safe_output import safe_print, safe_quantiles, safe_write_json, suppress_count  # noqa: E402

LEDGER_FIELDS = base.LEDGER_FIELDS
RUN_INFO = "run_info.json"


def write_part(rows: list[dict], cols: list[str], out_dir: Path, tag: str) -> None:
    base.write_part(rows, cols, out_dir, tag)


def run_info(backend_name: str, cfg: MorgothConfig) -> dict:
    """Provenance of a run (no ids, no values): backend, heads, steps, code commit, weight hashes of the heads used."""
    m = weights.load_manifest()
    return {"backend": backend_name, "heads": list(cfg.heads), "step_s": cfg.step_s, "spike_step_s": cfg.spike_step_s,
            "code": m.get("code", {}), "licence_code": weights.CODE_LICENCE, "licence_weights": weights.WEIGHTS_LICENCE,
            "weights_sha256": {HEADS[h].checkpoint: m.get("weights", {}).get(HEADS[h].checkpoint, {}).get("sha256")
                               for h in cfg.heads},
            "stub_output_is_meaningless": backend_name == "stub"}


def cmd_features(a, s3=None, backend=None) -> int:
    if not (0 <= a.shard < a.of):
        raise SystemExit("need 0 <= --shard < --of")
    inp, out_dir = Path(a.input), Path(a.out_dir)
    base._require_local_only(inp, "--input")
    base._require_local_only(out_dir, "--out-dir")
    heads = tuple(h for h in a.heads.split(",") if h)
    bad = [h for h in heads if h not in HEADS]
    if bad:
        raise SystemExit("unknown head(s): " + ",".join(bad))
    bad = [r for r in a.retry_reason if r not in ALL_REASONS]
    if bad:
        raise SystemExit("--retry-reason must be one of " + ",".join(sorted(ALL_REASONS)))
    windows = all_windows()
    if a.windows:
        want = [w.strip() for w in a.windows.split(",") if w.strip()]
        if not want or any(w not in windows for w in want):
            raise SystemExit("--windows must be a non-empty subset of " + ",".join(windows))
        windows = {w: windows[w] for w in want}
    cfg = MorgothConfig(heads=heads, step_s=a.step_s, spike_step_s=a.spike_step_s, compute_failed=a.compute_failed)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(out_dir, 0o700)

    recs = [(r, k, p) for r, k, p in base.load_recordings_with_parts(inp) if base.shard_of(r, a.of) == a.shard]
    done = base.load_done(out_dir, a.retry_permanent, set(a.retry_reason))
    todo = [(r, k, p) for r, k, p in recs if r not in done]
    if a.retry_reason:
        only = base.load_failed_with(out_dir, set(a.retry_reason))
        todo = [(r, k, p) for r, k, p in todo if r in only]
    skipped = len(recs) - len(todo)
    if a.limit is not None:
        todo = todo[: a.limit]

    if backend is None and todo:
        if a.backend == "stub" and not a.allow_stub:
            raise SystemExit("--backend stub on a real run needs --allow-stub (its output is meaningless)")
        try:
            backend = make_backend(a.backend, heads=heads, threads=a.threads)
        except FileNotFoundError as e:
            raise SystemExit(str(e))
    info_path = out_dir / RUN_INFO
    if todo:
        info = run_info(getattr(backend, "name", "custom"), cfg)
        if info_path.exists() and json.loads(info_path.read_text()) != info:
            raise SystemExit(f"{RUN_INFO} in --out-dir differs from this run's settings (backend / heads / steps / "
                             "weights); use a new --out-dir so one directory never mixes configurations")
        info_path.write_text(json.dumps(info, indent=2, sort_keys=True))
        os.chmod(info_path, 0o600)

    if s3 is None and todo:
        from sortinghat import data_io
        s3 = data_io.make_client(a.profile, read_timeout=a.read_timeout)
    cols = cfg.columns()
    tag = f"s{a.shard}of{a.of}"
    ledger_path = out_dir / f"ledger-{tag}.csv"
    new_ledger = not ledger_path.exists()
    ledger = open(ledger_path, "a", newline="")
    os.chmod(ledger_path, 0o600)
    lw = csv.writer(ledger)
    if new_ledger:
        lw.writerow(LEDGER_FIELDS)

    reasons: Counter = Counter()
    elapsed, proc_s, bytes_ = [], [], []
    feats: dict[str, list[float]] = {}
    buf_rows, buf_ledger = [], []
    n_ok = 0
    t_wall = time.monotonic()

    def flush():
        nonlocal buf_rows, buf_ledger
        if buf_rows:
            write_part(buf_rows, cols, out_dir, tag)
        for r in buf_ledger:
            lw.writerow(r)
        ledger.flush()
        buf_rows, buf_ledger = [], []

    try:
        from sortinghat import data_io
        for rid, key, parts in todo:
            kw = dict(cfg=cfg, windows=windows, timeout_s=a.timeout, max_attempts=a.max_attempts,
                      onset_search=not a.no_onset)
            res = stream_morgoth(s3, key, backend, **kw)
            if res.reason == FailureReason.NOT_FOUND and parts is not None:
                try:
                    found = data_io.resolve_edf_key(s3, *parts)
                except Exception:                           # noqa: BLE001
                    found = None
                if found is not None and found.found and found.key != key:
                    res = stream_morgoth(s3, found.key, backend, **kw)
            elapsed.append(res.elapsed_s)
            bytes_.append(res.bytes_fetched)
            if res.ok:
                n_ok += 1
                proc_s.append(res.process_s)
                for r in res.rows:
                    buf_rows.append({"recording_id": rid, "onset_offset_s": float(res.onset_s or 0.0), **r})
                    if r["window"] == "primary" and r["qc_pass"]:
                        for c in feature_names(heads):
                            if r.get(c) == r.get(c):
                                feats.setdefault(c, []).append(float(r[c]))
            else:
                reasons[res.reason] += 1
            buf_ledger.append([rid, "ok" if res.ok else "fail", res.reason or "", res.bytes_fetched,
                               round(res.elapsed_s, 2)])
            if len(buf_ledger) >= a.flush_every:
                flush()
        flush()
    finally:
        ledger.close()
    wall = time.monotonic() - t_wall
    attempted = len(todo)
    n_fail = attempted - n_ok
    summ = {"shard": a.shard, "of": a.of, "attempted": suppress_count(attempted),
            "already_done_skipped": suppress_count(skipped),
            "succeeded": suppress_count(n_ok) if (n_fail == 0 or n_fail >= 11) else "<11",
            "failed": suppress_count(n_fail),
            "failure_reasons": {r: suppress_count(reasons[r]) for r in ALL_REASONS if reasons.get(r)},
            "backend": getattr(backend, "name", None), "heads": list(heads), "step_s": a.step_s,
            "wall_seconds_per_recording": round(wall / attempted, 2) if attempted else None,
            "process_seconds_quantiles": safe_quantiles(proc_s),
            "primary_feature_quantiles": {c: safe_quantiles(v) for c, v in feats.items()},
            "usable_threshold": USABLE_THRESHOLD}
    safe_print(f"shard {a.shard}/{a.of} | attempted: {summ['attempted']} | already done (skipped): "
               f"{summ['already_done_skipped']} | succeeded: {summ['succeeded']} | failed: {summ['failed']}")
    for r, c in summ["failure_reasons"].items():
        safe_print(f"  failure {r}: {c}")
    safe_print("backend:", summ["backend"], "| heads:", ",".join(heads), "| wall s/recording:",
               summ["wall_seconds_per_recording"], "| process s quantiles:", summ["process_seconds_quantiles"])
    for c, q in summ["primary_feature_quantiles"].items():
        safe_print(f"primary {c}: {q}")
    if a.summary:
        safe_write_json(a.summary, summ)
    return 0


def cmd_exposure(a) -> int:
    if a.inspect:                                   # header NAMES only; no values
        for n, p in enumerate(a.lists, 1):
            for sheet, d in expo.inspect_columns(p).items():
                safe_print(f"list #{n} {sheet}: n columns {len(d['columns'])} | id-like columns "
                           f"{len(d['id_columns'])} | split-like columns {len(d['split_columns'])}")
                print("   columns:", d["columns"])
        return 0
    base._require_local_only(Path(a.out), "--out")
    base._require_local_only(Path(a.cohort), "--cohort")
    cohort = pd.read_csv(a.cohort, dtype={"person_id": "int64"}) if a.cohort.endswith((".csv", ".tsv")) \
        else pd.read_parquet(a.cohort)
    idx = expo.scan_lists(a.lists)
    if a.pretrain_names:
        from sortinghat import data_io
        s3 = data_io.make_client(a.profile)
        names = data_io.list_keys(s3, "morgoth1/data/pretrain/", bucket=data_io.ap_arn("projects"))
        n_names = len(names)
        n_added = expo.add_names(idx, names)
        safe_print(f"pretrain key names: {suppress_count(n_names)} listed | ids parsed from names (new to the lists): "
                   f"{suppress_count(len(idx.sources.get('pretrain_key_names', ())))} ({suppress_count(n_added)})")
    flags = expo.flag_cohort(cohort, idx)
    expo.write_flags(flags, a.out)
    summ = expo.summarize(flags, idx)
    summ["by_source"] = expo.source_counts(cohort, idx)
    safe_print("MORGOTH exposure accounting (aggregates; per-patient flags are in local_only):")
    for k, v in summ.items():
        if k not in ("by_site", "by_source"):
            safe_print(f"  {k}: {v}")
    safe_print("  cohort patients present per source (list file[#sheet][:split]):")
    for k, v in summ["by_source"].items():
        safe_print(f"    {k}: {v}")
    for s, e in summ.get("by_site", {}).items():
        safe_print(f"  {s}: n {e['n']} | in lists {e['in_lists_proportion']}")
    if a.summary:
        safe_write_json(a.summary, summ)
    return 0


def main(argv=None, s3=None, backend=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features", help="per-recording MORGOTH finding probabilities")
    f.add_argument("--input", required=True, help="recordings list under local_only/ (never printed)")
    f.add_argument("--out-dir", default="out/local_only/morgoth")
    f.add_argument("--shard", type=int, default=0)
    f.add_argument("--of", type=int, default=1)
    f.add_argument("--limit", type=int, default=None)
    f.add_argument("--backend", choices=("morgoth", "stub"), default="morgoth")
    f.add_argument("--allow-stub", action="store_true")
    f.add_argument("--heads", default=",".join(default_heads()), help="comma-separated: " + ",".join(HEADS))
    f.add_argument("--step-s", type=float, default=5.0, help="seconds between 10-s snippets (1 = the reference's EEG-level step)")
    f.add_argument("--spike-step-s", type=float, default=1.0)
    f.add_argument("--threads", type=int, default=None, help="torch CPU threads (one process per shard: keep x shards <= cores)")
    f.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="per-recording seconds (fetch + model)")
    f.add_argument("--max-attempts", type=int, default=4)
    f.add_argument("--flush-every", type=int, default=10)
    f.add_argument("--read-timeout", type=int, default=60)
    f.add_argument("--profile", default=None)
    f.add_argument("--retry-permanent", action="store_true")
    f.add_argument("--retry-reason", action="append", default=[], metavar="REASON")
    f.add_argument("--windows", default=None, help="subset of primary,20s,1min,2min,5min,10min (default all)")
    f.add_argument("--no-onset", action="store_true")
    f.add_argument("--compute-failed", action="store_true")
    f.add_argument("--summary", default=None)
    e = sub.add_parser("exposure", help="flag cohort patients that appear in MORGOTH's data lists")
    e.add_argument("--cohort", required=True, help="cohort table with person_id (+ SiteID, person_id_source), under local_only/")
    e.add_argument("--lists", nargs="+", required=True, help="MORGOTH list files (xlsx/csv/parquet), downloaded by a human")
    e.add_argument("--out", required=True, help="per-patient flags parquet; must be under local_only/")
    e.add_argument("--pretrain-names", action="store_true", help="also take ids embedded in the key NAMES under morgoth1/data/pretrain/ (BDSP credentials)")
    e.add_argument("--inspect", action="store_true", help="print list header NAMES only and exit (confirm id columns first)")
    e.add_argument("--profile", default=None)
    e.add_argument("--summary", default=None)
    a = ap.parse_args(argv)
    return cmd_features(a, s3, backend) if a.cmd == "features" else cmd_exposure(a)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as ex:                      # class name only (a traceback could carry values)
        print(f"extract_morgoth failed: {type(ex).__name__}", file=sys.stderr)
        raise SystemExit(1)
