"""One streamed EDF window, several frozen-representation rungs (CBraMod embeddings + MORGOTH finding probabilities).

``stream_embeddings`` and ``stream_morgoth`` each repeat the D-109 onset search and the ranged fetch of the same window.
``stream_rungs`` does both ONCE per recording, then runs the requested rungs on the in-memory ``Recording``; each rung
yields exactly the ``StreamResult`` its own streamer would have returned (same rows, QC, failure codes), so the parquet
parts and ledgers the two extractors write are unchanged. Nothing is written to disk. Never raises for data / network /
timeout problems.

The two extractors read the same 19 canonical channels (``CBRAMOD_CHANNELS`` and ``CANONICAL_19`` differ only in
order, and ``read_edf`` keeps file order), so one decoded ``Recording`` serves both. Rungs run one after the other and
each makes its own working copies, which are freed before the next starts: peak memory is the recording plus the
larger rung's working set. Timeout: ``timeout_s`` bounds the shared fetch and, separately, each rung's processing.
"""

from __future__ import annotations

import random
import time
from dataclasses import replace

from .eeg.io import EDFError
from .eeg.io import CANONICAL_19
from .eeg.pipeline import PAD_S
from .eeg.stream import (CHUNK_BYTES, DEFAULT_MAX_ATTEMPTS, MAX_FETCH_BYTES, ONSET_MAX_SEARCH_S, ONSET_MIN_ACTIVE,
                         Deadline, FailureReason, FetchStats, RecordingTimeout, StreamResult, _StreamError, _type_name,
                         classify_edf_error, fetch_window, find_signal_onset)
from .eeg.window import WindowSpec, all_windows
from .embed.cbramod import CBRAMOD_CHANNELS, EmbedConfig, Embedder, embed_recording
from .morgoth.features import MorgothConfig, process_morgoth

assert set(CBRAMOD_CHANNELS) == set(CANONICAL_19)          # one decoded Recording must serve both rungs

RUNGS = ("cbramod", "morgoth")
DEFAULT_TIMEOUT_S = 1800.0


def _run_stage(res: StreamResult, fn, timeout_s, clock):
    """Run one rung's processing under its own deadline; fills ``res`` on failure. Returns fn()'s value or None."""
    dl = Deadline(timeout_s, clock=clock)
    t0 = clock()
    try:
        with dl.alarm():
            dl.check()
            return fn()
    except RecordingTimeout:
        res.reason = FailureReason.TIMEOUT
    except _StreamError as e:
        res.reason, res.error_type = e.reason, e.error_type
    except EDFError as e:
        res.reason = classify_edf_error(e)
    except Exception as e:                                      # noqa: BLE001 - never leak the message
        res.reason, res.error_type = FailureReason.PROCESSING_ERROR, _type_name(e)
    finally:
        res.process_s = clock() - t0
    return None


def stream_rungs(s3, key: str, *, embedder: Embedder | None = None, backend=None, embed_cfg: EmbedConfig | None = None,
                 morgoth_cfg: MorgothConfig | None = None, rungs=RUNGS, bucket: str | None = None, windows=None,
                 timeout_s: float | None = DEFAULT_TIMEOUT_S, max_attempts: int = DEFAULT_MAX_ATTEMPTS,
                 backoff_s: float = 1.0, max_backoff_s: float = 30.0, sleep=time.sleep, rand=random.random,
                 chunk_bytes: int = CHUNK_BYTES, max_fetch_bytes: int = MAX_FETCH_BYTES, clock=time.monotonic,
                 onset_search: bool = True, onset_max_search_s: float = ONSET_MAX_SEARCH_S,
                 onset_min_active: int = ONSET_MIN_ACTIVE) -> dict[str, StreamResult]:
    """``{rung: StreamResult}`` for each requested rung, from a single onset search + window fetch."""
    rungs = tuple(rungs)
    if not rungs or any(r not in RUNGS for r in rungs):
        raise ValueError("rungs must be a non-empty subset of " + ",".join(RUNGS))
    embed_cfg = embed_cfg or EmbedConfig()
    morgoth_cfg = morgoth_cfg or MorgothConfig()
    t_start = clock()
    windows = dict(windows or all_windows())
    deadline = Deadline(timeout_s, clock=clock)
    stats = FetchStats()
    shared = StreamResult(ok=False)
    rec = None
    try:
        with deadline.alarm():
            onset = 0.0
            if onset_search:
                on = find_signal_onset(s3, key, bucket=bucket, deadline=deadline, stats=stats, max_attempts=max_attempts,
                                       backoff_s=backoff_s, max_backoff_s=max_backoff_s, sleep=sleep, rand=rand,
                                       chunk_bytes=chunk_bytes, max_search_s=onset_max_search_s, min_active=onset_min_active)
                if on.onset_s is None:
                    raise _StreamError(on.reason or FailureReason.NO_SUSTAINED_SIGNAL)
                onset = float(on.onset_s)
            shared.onset_s = onset
            windows = {k: WindowSpec(w.name, w.start_s + onset, w.duration_s) for k, w in windows.items()}
            start = max(0.0, min(w.start_s for w in windows.values()) - PAD_S)
            end = max(w.end_s for w in windows.values()) + PAD_S
            rec = fetch_window(s3, key, start, end - start, bucket=bucket, deadline=deadline, stats=stats,
                               max_attempts=max_attempts, backoff_s=backoff_s, max_backoff_s=max_backoff_s, sleep=sleep,
                               rand=rand, chunk_bytes=chunk_bytes, max_fetch_bytes=max_fetch_bytes,
                               channels=CBRAMOD_CHANNELS)
            shared.fetch_s = clock() - t_start
    except RecordingTimeout:
        shared.reason = FailureReason.TIMEOUT
    except _StreamError as e:
        shared.reason, shared.error_type = e.reason, e.error_type
    except EDFError as e:
        shared.reason = classify_edf_error(e)
    except Exception as e:                                      # noqa: BLE001 - never leak the message
        shared.reason, shared.error_type = FailureReason.S3_ERROR, _type_name(e)
    shared.bytes_fetched, shared.n_requests, shared.n_retries = stats.bytes_fetched, stats.n_requests, stats.n_retries

    fetch_elapsed = clock() - t_start
    out: dict[str, StreamResult] = {}
    for name in rungs:
        res = replace(shared)                                   # own copy: onset, fetch stats, shared failure (if any)
        if rec is not None:
            if name == "cbramod":
                er = _run_stage(res, lambda: embed_recording(rec, embedder, windows=windows, cfg=embed_cfg),
                                timeout_s, clock)
                if er is not None:
                    res.rows, res.qc = er.rows, er.qc
                    st = er.channel_status
                    res.n_channels = int(st.get("n_channels", 0))
                    res.n_missing_min = int(st.get("n_missing_min", 0))
                    res.n_dead_min = int(st.get("n_dead_min", 0))
                    res.n_invalid_min = int(st.get("n_invalid_min", 0))
                    res.ok = True
            else:
                mr = _run_stage(res, lambda: process_morgoth(rec, windows, backend, morgoth_cfg), timeout_s, clock)
                if mr is not None:
                    rows, info = mr
                    res.rows = rows
                    res.n_channels = int(rec.data.shape[0])
                    res.n_missing_min = int(info["n_missing_channels"])
                    res.ok = True
        res.elapsed_s = fetch_elapsed + res.process_s
        out[name] = res
    rec = None                                                  # release the window before the caller moves on
    return out
