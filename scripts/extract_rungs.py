#!/usr/bin/env python3
"""Streaming CBraMod embeddings AND MORGOTH finding features in ONE pass per recording (HUMAN-RUN via scripts/heedb_run.sh).

    HEEDB_AWS_PROFILE=<profile> scripts/heedb_run.sh python3 scripts/extract_rungs.py \
        --input out/local_only/recording_keys.csv --shard 0 --of 4 --torch-threads 1

``extract_embeddings.py`` and ``extract_morgoth.py`` each stream every recording's EDF window from S3 separately. This script
does the D-109 onset search and the ranged fetch of the primary + nested windows ONCE (``sortinghat.rungs.stream_rungs``),
then runs both frozen models on the in-memory signal (models loaded once per process, one recording's window in memory).
It writes to the SAME places, with the SAME parquet schemas, ledgers and provenance files the two extractors use
(``--embed-dir`` = out/local_only/embeddings with ``embed_meta.json``; ``--morgoth-dir`` = out/local_only/morgoth with
``run_info.json``), so the ladder wiring and resume logic are unchanged and either old extractor can resume after (or
before) this one. Resume is per rung: a recording done for one rung only is fetched again and processed for the other.

Same CLI conventions as the two (read their docstrings): ``--input``, ``--shard/--of``, ``--limit``, ``--torch-threads``
(applied to both models), ``--windows``, ``--retry-permanent`` / ``--retry-reason``, ``--no-onset``, ``--compute-failed``,
CBraMod options (``--weights --fetch-weights --batch-size --reference --amp-policy --attention``) and MORGOTH options
(``--backend --allow-stub --heads --step-s --spike-step-s``). ``--rungs cbramod,morgoth`` (default both) restricts the run.
``--timeout`` bounds the shared fetch and, separately, each model's processing. Run one process per shard.

Records go ONLY to the two ``local_only/`` directories (mode 0600). stdout and ``--summary`` carry aggregates only
(counts < 11 print as "<11").
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))             # repo root, so `sortinghat` imports
sys.path.insert(0, str(HERE))                    # sibling scripts

import extract_eeg_features as xef  # noqa: E402
import extract_embeddings as xemb  # noqa: E402
import extract_morgoth as xmor  # noqa: E402
from sortinghat.eeg.stream import ALL_REASONS, FailureReason  # noqa: E402
from sortinghat.eeg.window import USABLE_THRESHOLD, all_windows  # noqa: E402
from sortinghat.embed import weights as W  # noqa: E402
from sortinghat.embed.cbramod import AMP_POLICIES, REFERENCES, Embedder, EmbedConfig  # noqa: E402
from sortinghat.embed.store import require_local_only, write_meta, write_part as write_embed_part  # noqa: E402
from sortinghat.morgoth.features import MorgothConfig  # noqa: E402
from sortinghat.morgoth.heads import HEADS, default_heads, feature_names  # noqa: E402
from sortinghat.morgoth.model import make_backend  # noqa: E402
from sortinghat.rungs import DEFAULT_TIMEOUT_S, RUNGS, stream_rungs  # noqa: E402
from sortinghat.safe_output import safe_print, safe_quantiles, safe_write_json, suppress_count  # noqa: E402

LEDGER_FIELDS = xef.LEDGER_FIELDS


class RungState:
    """Per-rung bookkeeping: output dir, ledger, buffers and aggregate accumulators (aggregates only)."""

    def __init__(self, name: str, out_dir: Path, tag: str, write_rows):
        self.name, self.out_dir, self.write_rows = name, out_dir, write_rows
        path = out_dir / f"ledger-{tag}.csv"
        new = not path.exists()
        self.fh = open(path, "a", newline="")
        os.chmod(path, 0o600)
        self.lw = csv.writer(self.fh)
        if new:
            self.lw.writerow(LEDGER_FIELDS)
        self.reasons: Counter = Counter()
        self.elapsed, self.bytes, self.fetch_s, self.proc_s = [], [], [], []
        self.retries = self.n_ok = self.attempted = 0
        self.skipped = 0
        self.wrows: dict[str, list[dict]] = {}
        self.feats: dict[str, list[float]] = {}
        self.buf_rows: list[dict] = []
        self.buf_ledger: list[list] = []

    def add(self, rid: str, res, feat_names=()):
        self.attempted += 1
        self.elapsed.append(res.elapsed_s)
        self.bytes.append(res.bytes_fetched)
        self.retries += res.n_retries
        if res.ok:
            self.n_ok += 1
            self.fetch_s.append(res.fetch_s)
            self.proc_s.append(res.process_s)
            self.buf_rows += [{"recording_id": rid, "onset_offset_s": float(res.onset_s or 0.0), **r} for r in res.rows]
            for r in res.rows:
                if self.name == "cbramod":
                    self.wrows.setdefault(r["window"], []).append(r)
                elif r["window"] == "primary" and r["qc_pass"]:
                    for c in feat_names:
                        if r.get(c) == r.get(c):
                            self.feats.setdefault(c, []).append(float(r[c]))
        else:
            self.reasons[res.reason] += 1
        self.buf_ledger.append([rid, "ok" if res.ok else "fail", res.reason or "", res.bytes_fetched,
                                round(res.elapsed_s, 2)])

    def flush(self):
        if self.buf_rows:
            self.write_rows(self.buf_rows)               # rows first, then the ledger: a crash re-does at most this batch
        for r in self.buf_ledger:
            self.lw.writerow(r)
        self.fh.flush()
        self.buf_rows, self.buf_ledger = [], []

    def close(self):
        self.fh.close()


def main(argv=None, s3=None, embedder=None, backend=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="recordings list under local_only/ (never printed)")
    ap.add_argument("--embed-dir", default="out/local_only/embeddings", help="CBraMod parts (local_only/)")
    ap.add_argument("--morgoth-dir", default="out/local_only/morgoth", help="MORGOTH parts (local_only/)")
    ap.add_argument("--rungs", default=",".join(RUNGS), help="comma-separated subset of " + ",".join(RUNGS))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="attempt at most this many recordings (after resume filtering)")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                    help="seconds for the shared fetch and, separately, for each model's processing")
    ap.add_argument("--max-attempts", type=int, default=4, help="per ranged GET")
    ap.add_argument("--flush-every", type=int, default=10, help="recordings per parquet part")
    ap.add_argument("--read-timeout", type=int, default=60, help="S3 socket read timeout, seconds")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--retry-permanent", action="store_true")
    ap.add_argument("--retry-reason", action="append", default=[], metavar="REASON")
    ap.add_argument("--windows", default=None, help="comma-separated subset of primary,20s,1min,2min,5min,10min (default all)")
    ap.add_argument("--no-onset", action="store_true", help="windows from the FILE start instead of the D-109 t0")
    ap.add_argument("--compute-failed", action="store_true", help="run the models on windows that fail QC too")
    ap.add_argument("--torch-threads", type=int, default=None, help="CPU threads for each model (1 per process when sharding)")
    # CBraMod
    ap.add_argument("--weights", default=None, help="CBraMod checkpoint path (default: cache dir); SHA-256 is verified")
    ap.add_argument("--fetch-weights", action="store_true", help="download the public CBraMod checkpoint if absent")
    ap.add_argument("--batch-size", type=int, default=30, help="10-s segments per CBraMod forward pass")
    ap.add_argument("--reference", choices=REFERENCES, default="car")
    ap.add_argument("--amp-policy", choices=AMP_POLICIES, default="keep")
    ap.add_argument("--attention", action="store_true")
    # MORGOTH
    ap.add_argument("--backend", choices=("morgoth", "stub"), default="morgoth")
    ap.add_argument("--allow-stub", action="store_true")
    ap.add_argument("--heads", default=",".join(default_heads()), help="comma-separated: " + ",".join(HEADS))
    ap.add_argument("--step-s", type=float, default=5.0)
    ap.add_argument("--spike-step-s", type=float, default=1.0)
    ap.add_argument("--summary", default=None, help="aggregate JSON (safe_write_json)")
    a = ap.parse_args(argv)

    if not (0 <= a.shard < a.of):
        raise SystemExit("need 0 <= --shard < --of")
    rungs = tuple(r.strip() for r in a.rungs.split(",") if r.strip())
    if not rungs or any(r not in RUNGS for r in rungs):
        raise SystemExit("--rungs must be a non-empty subset of " + ",".join(RUNGS))
    inp, embed_dir, mg_dir = Path(a.input), Path(a.embed_dir), Path(a.morgoth_dir)
    require_local_only(inp, "--input")
    if "cbramod" in rungs:
        require_local_only(embed_dir, "--embed-dir")
    if "morgoth" in rungs:
        require_local_only(mg_dir, "--morgoth-dir")
    bad = [r for r in a.retry_reason if r not in ALL_REASONS]
    if bad:
        raise SystemExit("--retry-reason must be one of " + ",".join(sorted(ALL_REASONS)))
    heads = tuple(h for h in a.heads.split(",") if h)
    bad = [h for h in heads if h not in HEADS]
    if bad:
        raise SystemExit("unknown head(s): " + ",".join(bad))
    windows = all_windows()
    if a.windows:
        want = [w.strip() for w in a.windows.split(",") if w.strip()]
        if not want or any(w not in windows for w in want):
            raise SystemExit("--windows must be a non-empty subset of " + ",".join(windows))
        windows = {w: windows[w] for w in want}
    ecfg = EmbedConfig(reference=a.reference, amp_policy=a.amp_policy, attention=a.attention, compute_failed=a.compute_failed)
    mcfg = MorgothConfig(heads=heads, step_s=a.step_s, spike_step_s=a.spike_step_s, compute_failed=a.compute_failed)
    dirs = {"cbramod": embed_dir, "morgoth": mg_dir}
    for r in rungs:
        dirs[r].mkdir(parents=True, exist_ok=True)
        os.chmod(dirs[r], 0o700)

    # --- what each rung still needs (the same resume rules as the two extractors, applied per rung) ---
    recs = [(r, k, p) for r, k, p in xef.load_recordings_with_parts(inp) if xef.shard_of(r, a.of) == a.shard]
    needs: dict[str, set[str]] = {}
    skipped: dict[str, int] = {}
    for r in rungs:
        done = xef.load_done(dirs[r], a.retry_permanent, set(a.retry_reason))
        todo_r = [rid for rid, _, _ in recs if rid not in done]
        if a.retry_reason:
            only = xef.load_failed_with(dirs[r], set(a.retry_reason))
            todo_r = [rid for rid in todo_r if rid in only]
        skipped[r] = len(recs) - len(todo_r)
        for rid in todo_r:
            needs.setdefault(rid, set()).add(r)
    todo = [(rid, k, p) for rid, k, p in recs if rid in needs]
    if a.limit is not None:
        todo = todo[: a.limit]
    active = {r for rid, _, _ in todo for r in needs[rid]}

    # --- models once per process; provenance files are checked before anything is fetched ---
    meta_file = weights_id = None
    if "cbramod" in rungs:
        if embedder is None and "cbramod" in active:
            try:
                ckpt = W.ensure_checkpoint(a.weights, download=a.fetch_weights)
            except W.WeightsError as e:
                raise SystemExit(f"weights: {e}")
            embedder = Embedder.from_checkpoint(ckpt, batch_size=a.batch_size, num_threads=a.torch_threads)
            injected = False
        else:
            injected = embedder is not None
        weights_id = "injected-for-tests" if injected else W.CHECKPOINT_SHA256
        meta = {"model": "CBraMod (frozen, proj_out removed)", "upstream_repo": W.UPSTREAM_REPO,
                "upstream_commit": W.UPSTREAM_COMMIT, "weights_sha256": weights_id, "config": ecfg.meta(),
                "windows": sorted(windows), "onset": "D-109 t0" if not a.no_onset else "file start",
                "embedding_dim": 200}
        meta_file = meta                                  # identical to extract_embeddings' embed_meta.json
        write_meta(embed_dir, meta_file)
    if "morgoth" in rungs and "morgoth" in active:
        if backend is None:
            if a.backend == "stub" and not a.allow_stub:
                raise SystemExit("--backend stub on a real run needs --allow-stub (its output is meaningless)")
            try:
                backend = make_backend(a.backend, heads=heads, threads=a.torch_threads)
            except FileNotFoundError as e:
                raise SystemExit(str(e))
        info = xmor.run_info(getattr(backend, "name", "custom"), mcfg)
        info_path = mg_dir / xmor.RUN_INFO
        if info_path.exists() and json.loads(info_path.read_text()) != info:
            raise SystemExit(f"{xmor.RUN_INFO} in --morgoth-dir differs from this run's settings (backend / heads / steps / "
                             "weights); use a new --morgoth-dir so one directory never mixes configurations")
        info_path.write_text(json.dumps(info, indent=2, sort_keys=True))
        os.chmod(info_path, 0o600)

    if s3 is None and todo:
        from sortinghat import data_io
        s3 = data_io.make_client(a.profile, read_timeout=a.read_timeout)

    tag = f"s{a.shard}of{a.of}"
    states: dict[str, RungState] = {}
    if "cbramod" in rungs:
        states["cbramod"] = RungState("cbramod", embed_dir, tag, lambda rows: write_embed_part(rows, ecfg, embed_dir, tag))
    if "morgoth" in rungs:
        cols = mcfg.columns()
        states["morgoth"] = RungState("morgoth", mg_dir, tag, lambda rows: xmor.write_part(rows, cols, mg_dir, tag))
    for r, st in states.items():
        st.skipped = skipped[r]
    mfeat = feature_names(heads)
    n_attempt_total = 0
    t_wall = time.monotonic()

    def run_one(key, want):
        return stream_rungs(s3, key, embedder=embedder, backend=backend, embed_cfg=ecfg, morgoth_cfg=mcfg, rungs=want,
                            windows=windows, timeout_s=a.timeout, max_attempts=a.max_attempts, onset_search=not a.no_onset)

    try:
        from sortinghat import data_io
        for rid, key, parts in todo:
            want = tuple(r for r in rungs if r in needs[rid])
            res = run_one(key, want)
            if any(x.reason == FailureReason.NOT_FOUND for x in res.values()) and parts is not None:
                try:
                    found = data_io.resolve_edf_key(s3, *parts)
                except Exception:                           # noqa: BLE001 - keep the not_found result
                    found = None
                if found is not None and found.found and found.key != key:
                    res = run_one(found.key, want)
            n_attempt_total += 1
            for r in want:
                states[r].add(rid, res[r], mfeat)
            if any(len(st.buf_ledger) >= a.flush_every for st in states.values()):
                for st in states.values():
                    st.flush()
        for st in states.values():
            st.flush()
    finally:
        for st in states.values():
            st.close()
    wall = time.monotonic() - t_wall

    summ: dict = {"shard": a.shard, "of": a.of, "recordings_streamed": suppress_count(n_attempt_total),
                  "wall_seconds_per_recording": round(wall / n_attempt_total, 2) if n_attempt_total else None}
    safe_print(f"shard {a.shard}/{a.of} | recordings streamed (one fetch each): {summ['recordings_streamed']} | wall "
               f"s/recording: {summ['wall_seconds_per_recording']} | torch threads {a.torch_threads}")
    for r, st in states.items():
        if r == "cbramod":
            s = xemb.summarize(st.attempted, st.n_ok, st.reasons, st.elapsed, st.fetch_s, st.proc_s, st.bytes, st.retries,
                               wall, st.skipped, st.wrows,
                               {**meta_file, "torch_threads": a.torch_threads, "weights_sha256": weights_id[:8]})
            safe_print(f"[cbramod] attempted: {s['attempted']} | already done (skipped): {s['already_done_skipped']} | "
                       f"succeeded: {s['succeeded']} | failed: {s['failed']} | success rate: {s['success_rate']}")
            for rr, c in s["failure_reasons"].items():
                safe_print(f"  [cbramod] failure {rr}: {c}")
            safe_print("[cbramod] mean embed s", s["throughput"]["mean_process_seconds"], "| weights sha256 prefix",
                       s["model"]["weights_sha256"])
            for w, e in s["windows"].items():
                safe_print(f"[cbramod] window {w}: n {e['n']} | QC pass {e['qc_pass_proportion']} | embedded "
                           f"{e['embedded_proportion']} | segments used quantiles {e['n_segments_used_quantiles']} | "
                           f"valid-token-fraction quantiles {e['valid_token_fraction_quantiles']}")
        else:
            n_fail = st.attempted - st.n_ok
            s = {"attempted": suppress_count(st.attempted), "already_done_skipped": suppress_count(st.skipped),
                 "succeeded": suppress_count(st.n_ok) if (n_fail == 0 or n_fail >= 11) else "<11",
                 "failed": suppress_count(n_fail),
                 "failure_reasons": {x: suppress_count(st.reasons[x]) for x in ALL_REASONS if st.reasons.get(x)},
                 "backend": getattr(backend, "name", None), "heads": list(heads), "step_s": a.step_s,
                 "process_seconds_quantiles": safe_quantiles(st.proc_s),
                 "primary_feature_quantiles": {c: safe_quantiles(v) for c, v in st.feats.items()},
                 "usable_threshold": USABLE_THRESHOLD}
            safe_print(f"[morgoth] attempted: {s['attempted']} | already done (skipped): {s['already_done_skipped']} | "
                       f"succeeded: {s['succeeded']} | failed: {s['failed']}")
            for rr, c in s["failure_reasons"].items():
                safe_print(f"  [morgoth] failure {rr}: {c}")
            safe_print("[morgoth] backend:", s["backend"], "| heads:", ",".join(heads), "| process s quantiles:",
                       s["process_seconds_quantiles"])
            for c, q in s["primary_feature_quantiles"].items():
                safe_print(f"[morgoth] primary {c}: {q}")
        summ[r] = s
    if a.summary:
        safe_write_json(a.summary, summ)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:                       # never print a traceback (it could carry values); class name only
        print(f"extract_rungs failed: {type(e).__name__}", file=sys.stderr)
        raise SystemExit(1)
