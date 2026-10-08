"""Streaming CBraMod embeddings for one S3 EDF: the same ranged reads, D-109 onset and failure codes as
``sortinghat.eeg.stream.stream_features``, but the per-window product is a frozen-CBraMod embedding row.

Nothing raw is written to disk. Never raises for data / network / timeout problems; the returned ``StreamResult`` carries
a fixed ``FailureReason`` code (no keys, paths or messages). ``StreamResult.rows`` hold embedding rows (see
``sortinghat.embed.cbramod.part_columns``): record-level, local_only only.
"""

from __future__ import annotations

import random
import time

from ..eeg.io import EDFError
from ..eeg.pipeline import PAD_S
from ..eeg.stream import (DEFAULT_MAX_ATTEMPTS, DEFAULT_TIMEOUT_S, CHUNK_BYTES, MAX_FETCH_BYTES, ONSET_MAX_SEARCH_S,
                          ONSET_MIN_ACTIVE, Deadline, FailureReason, FetchStats, RecordingTimeout, StreamResult,
                          _StreamError, _type_name, classify_edf_error, fetch_window, find_signal_onset)
from ..eeg.window import QCConfig, WindowSpec, all_windows
from .cbramod import CBRAMOD_CHANNELS, Embedder, EmbedConfig, embed_recording


def stream_embeddings(s3, key: str, embedder: Embedder, *, bucket: str | None = None, windows=None,
                      qc_cfg: QCConfig | None = None, cfg: EmbedConfig | None = None,
                      timeout_s: float | None = DEFAULT_TIMEOUT_S, max_attempts: int = DEFAULT_MAX_ATTEMPTS,
                      backoff_s: float = 1.0, max_backoff_s: float = 30.0, sleep=time.sleep, rand=random.random,
                      chunk_bytes: int = CHUNK_BYTES, max_fetch_bytes: int = MAX_FETCH_BYTES, clock=time.monotonic,
                      onset_search: bool = True, onset_max_search_s: float = ONSET_MAX_SEARCH_S,
                      onset_min_active: int = ONSET_MIN_ACTIVE) -> StreamResult:
    cfg = cfg or EmbedConfig()
    t_start = clock()
    windows = dict(windows or all_windows())
    deadline = Deadline(timeout_s, clock=clock)
    stats = FetchStats()
    res = StreamResult(ok=False)
    stage = "fetch"
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
            res.onset_s = onset
            windows = {k: WindowSpec(w.name, w.start_s + onset, w.duration_s) for k, w in windows.items()}
            start = max(0.0, min(w.start_s for w in windows.values()) - PAD_S)
            end = max(w.end_s for w in windows.values()) + PAD_S
            rec = fetch_window(s3, key, start, end - start, bucket=bucket, deadline=deadline, stats=stats,
                               max_attempts=max_attempts, backoff_s=backoff_s, max_backoff_s=max_backoff_s, sleep=sleep,
                               rand=rand, chunk_bytes=chunk_bytes, max_fetch_bytes=max_fetch_bytes,
                               channels=CBRAMOD_CHANNELS)
            res.fetch_s = clock() - t_start
            stage = "process"
            t_proc = clock()
            deadline.check()
            out = embed_recording(rec, embedder, windows=windows, qc_cfg=qc_cfg, cfg=cfg)
            res.process_s = clock() - t_proc
        res.rows, res.qc = out.rows, out.qc
        st = out.channel_status
        res.n_channels = int(st.get("n_channels", 0))
        res.n_missing_min = int(st.get("n_missing_min", 0))
        res.n_dead_min = int(st.get("n_dead_min", 0))
        res.n_invalid_min = int(st.get("n_invalid_min", 0))
        res.ok = True
    except RecordingTimeout:
        res.reason = FailureReason.TIMEOUT
    except _StreamError as e:
        res.reason, res.error_type = e.reason, e.error_type
    except EDFError as e:
        res.reason = classify_edf_error(e)
    except Exception as e:                                      # noqa: BLE001 - never leak the message
        res.reason = FailureReason.PROCESSING_ERROR if stage == "process" else FailureReason.S3_ERROR
        res.error_type = _type_name(e)
    res.bytes_fetched, res.n_requests, res.n_retries = stats.bytes_fetched, stats.n_requests, stats.n_retries
    res.elapsed_s = clock() - t_start
    return res
