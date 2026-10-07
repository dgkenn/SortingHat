"""Streaming EEG feature extraction: ranged S3 reads of one EDF, decoded and processed in memory.

HEEDB ICU EDFs are ~1 GB but Study 1 needs minutes 1-11 only (``docs/eeg_pipeline.md``). For one recording this
module (1) reads the EDF header with a ranged GET, (2) computes the byte range of the data records covering the
primary window plus the filter padding (``pipeline.PAD_S`` either side; nothing else), (3) fetches only that range
(in bounded chunks, each retried), (4) decodes it in memory with the existing ``io.read_edf`` and (5) runs the
existing QC / preprocess / feature pipeline. **Nothing is written to disk**: the raw bytes live in a
``bytes`` object and a sparse in-memory file that ``read_edf`` seeks in, then both are garbage.

Privacy (CLAUDE.md rules 1-3). ``stream_features`` never raises for data or network problems; it returns a
``StreamResult`` whose ``reason`` is one of the fixed ``FailureReason`` codes. Exception messages, keys and bucket
names are never copied into the result, so a failure can be counted and printed safely. ``error_type`` is only the
exception *class name* (validated against a strict pattern). Callers must still keep ``rows`` out of stdout.

Credentialed S3 access is the caller's business (``data_io.make_client`` refuses agent sessions); this module
only needs an object with ``get_object(Bucket=, Key=, Range=)``, so tests use a fake over a synthetic EDF.
"""

from __future__ import annotations

import contextlib
import io
import random
import re
import signal
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .features import FeatureConfig
from .io import CANONICAL_19, EDFError, EDFHeader, Recording, read_edf, read_edf_header
from .pipeline import PAD_S, process_recording
from .preprocess import PreprocessConfig
from .window import FLAG_NAMES, QCConfig, WindowQC, all_windows

HEADER_PROBE_BYTES = 64 * 1024          # first GET; an EDF header is 256 * (n_signals + 1) bytes (<= 64 KiB for 255)
CHUNK_BYTES = 8 * 1024 * 1024           # each data GET is at most this big, so a retry re-fetches little
MAX_FETCH_BYTES = 256 * 1024 * 1024     # refuse a plan bigger than this (pathological sampling rate x channels)
DEFAULT_TIMEOUT_S = 300.0               # per recording: fetch + decode + features
DEFAULT_MAX_ATTEMPTS = 4                # per GET


class FailureReason:
    """Fixed, ID-free failure codes. ``PERMANENT`` ones are not worth retrying on a resumed run."""
    NOT_FOUND = "not_found"
    ACCESS_DENIED = "access_denied"
    AUTH_ERROR = "auth_error"
    S3_TRANSIENT = "s3_transient"            # retries exhausted on 5xx / throttling / connection problems
    S3_CLIENT_ERROR = "s3_client_error"      # other 4xx
    S3_ERROR = "s3_error"
    TIMEOUT = "timeout"
    HEADER_INVALID = "edf_header_invalid"
    DISCONTINUOUS = "edf_discontinuous"
    NO_EEG_CHANNELS = "no_eeg_channels"
    UNSUPPORTED_UNITS = "edf_unsupported_units"
    DECODE_ERROR = "edf_decode_error"
    TOO_SHORT = "recording_too_short"        # no data records at or after the window start
    TOO_LARGE = "fetch_too_large"
    PROCESSING_ERROR = "processing_error"

    PERMANENT = frozenset({NOT_FOUND, ACCESS_DENIED, AUTH_ERROR, S3_CLIENT_ERROR, HEADER_INVALID, DISCONTINUOUS,
                           NO_EEG_CHANNELS, UNSUPPORTED_UNITS, DECODE_ERROR, TOO_SHORT, TOO_LARGE})


ALL_REASONS = tuple(v for k, v in vars(FailureReason).items() if k.isupper() and isinstance(v, str))


class RecordingTimeout(BaseException):
    """Raised inside the per-recording deadline (alarm or explicit check)."""


class _StreamError(Exception):
    """Internal: carries a FailureReason code only (never a message with data)."""

    def __init__(self, reason: str, error_type: str | None = None):
        super().__init__(reason)
        self.reason = reason
        self.error_type = error_type


@dataclass
class FetchStats:
    bytes_fetched: int = 0
    n_requests: int = 0
    n_retries: int = 0
    file_bytes: int | None = None            # whole-object size (from Content-Range); a size, not content


@dataclass
class StreamResult:
    ok: bool
    reason: str | None = None                # FailureReason code when not ok
    error_type: str | None = None            # exception class name only (processing_error / s3_error)
    rows: list[dict] = field(default_factory=list)     # one dict per window (pipeline rows + qc_* columns)
    qc: dict[str, WindowQC] = field(default_factory=dict)
    n_channels: int = 0
    bytes_fetched: int = 0
    n_requests: int = 0
    n_retries: int = 0
    fetch_s: float = 0.0
    process_s: float = 0.0
    elapsed_s: float = 0.0

    @property
    def retryable(self) -> bool:
        return (not self.ok) and self.reason not in FailureReason.PERMANENT


# --------------------------------------------------------------------------------------------
# Deadline
# --------------------------------------------------------------------------------------------
class Deadline:
    """Wall-clock budget for one recording. ``check()`` is called between steps; ``alarm()`` additionally
    interrupts a long CPU stage via SIGALRM when running in the main thread of a POSIX process."""

    def __init__(self, seconds: float | None, clock=time.monotonic):
        self._clock = clock
        self.end = None if seconds is None else clock() + float(seconds)

    def remaining(self) -> float | None:
        return None if self.end is None else self.end - self._clock()

    def check(self) -> None:
        r = self.remaining()
        if r is not None and r <= 0:
            raise RecordingTimeout()

    @contextlib.contextmanager
    def alarm(self):
        r = self.remaining()
        usable = (r is not None and hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")
                  and threading.current_thread() is threading.main_thread())
        if not usable:
            yield
            return

        def _handler(signum, frame):
            raise RecordingTimeout()

        old = signal.signal(signal.SIGALRM, _handler)
        prev = signal.setitimer(signal.ITIMER_REAL, max(r, 1e-3))
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)
            if prev[0] > 0:                      # restore an outer alarm, minus the time we used
                signal.setitimer(signal.ITIMER_REAL, prev[0])


# --------------------------------------------------------------------------------------------
# Error classification (no messages are ever kept)
# --------------------------------------------------------------------------------------------
_TRANSIENT_CODES = {"SlowDown", "Throttling", "ThrottlingException", "RequestTimeout", "RequestTimeoutException",
                    "RequestLimitExceeded", "InternalError", "ServiceUnavailable", "TooManyRequestsException",
                    "BandwidthLimitExceeded", "429", "500", "502", "503", "504"}
_NOT_FOUND_CODES = {"NoSuchKey", "NotFound", "404", "NoSuchBucket"}
_DENIED_CODES = {"AccessDenied", "403", "Forbidden", "InvalidAccessKeyId", "SignatureDoesNotMatch", "ExpiredToken",
                 "AllAccessDisabled", "401"}
_AUTH_NAMES = {"NoCredentialsError", "PartialCredentialsError", "CredentialRetrievalError", "UnauthorizedSSOTokenError",
               "TokenRetrievalError", "NoAuthTokenError"}
_NET_NAMES = {"EndpointConnectionError", "ConnectTimeoutError", "ReadTimeoutError", "ConnectionClosedError",
              "ResponseStreamingError", "IncompleteReadError", "IncompleteRead", "ProtocolError", "ProxyError",
              "ProxyConnectionError", "SSLError", "HTTPClientError", "ConnectionError", "ConnectionResetError"}
_TYPE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,39}$")


def _type_name(exc: BaseException) -> str | None:
    n = type(exc).__name__
    return n if _TYPE_RE.match(n) else None


def classify_s3_error(exc: BaseException) -> tuple[str, bool]:
    """(FailureReason code, retryable) for an exception raised by an S3 call."""
    resp = getattr(exc, "response", None)
    if isinstance(resp, dict):
        code = str(resp.get("Error", {}).get("Code", ""))
        status = resp.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in _NOT_FOUND_CODES or status == 404:
            return FailureReason.NOT_FOUND, False
        if code in _DENIED_CODES or status in (401, 403):
            return FailureReason.ACCESS_DENIED, False
        if code in _TRANSIENT_CODES or (isinstance(status, int) and (status >= 500 or status == 429)):
            return FailureReason.S3_TRANSIENT, True
        return FailureReason.S3_CLIENT_ERROR, False
    name = type(exc).__name__
    if name in _AUTH_NAMES:
        return FailureReason.AUTH_ERROR, False
    if isinstance(exc, (OSError, EOFError)) or name in _NET_NAMES:     # ConnectionError / TimeoutError are OSError
        return FailureReason.S3_TRANSIENT, True
    return FailureReason.S3_ERROR, False


def classify_edf_error(exc: EDFError) -> str:
    """Map an ``EDFError`` (constant messages raised in ``io.py``) to a code. Matching is on those fixed messages."""
    m = str(exc).lower()
    if "discontinuous" in m:
        return FailureReason.DISCONTINUOUS
    if "no eeg channels" in m:
        return FailureReason.NO_EEG_CHANNELS
    if "physical dimension" in m:
        return FailureReason.UNSUPPORTED_UNITS
    if any(s in m for s in ("not an edf", "shorter than an edf header", "truncated edf", "bad numeric", "invalid signal")):
        return FailureReason.HEADER_INVALID
    return FailureReason.DECODE_ERROR


# --------------------------------------------------------------------------------------------
# Ranged reads
# --------------------------------------------------------------------------------------------
_CR_RE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+|\*)$")


def _parse_content_range(value) -> int | None:
    m = _CR_RE.match(str(value or "").strip())
    return int(m.group(3)) if m and m.group(3) != "*" else None


class _Ranged:
    """Ranged GETs with retry + exponential backoff (full jitter), counted, bounded by a ``Deadline``."""

    def __init__(self, s3, bucket, key, deadline: Deadline, stats: FetchStats, max_attempts: int, backoff_s: float,
                 max_backoff_s: float, sleep, rand):
        self.s3, self.bucket, self.key = s3, bucket, key
        self.deadline, self.stats = deadline, stats
        self.max_attempts, self.backoff_s, self.max_backoff_s = max(1, int(max_attempts)), backoff_s, max_backoff_s
        self.sleep, self.rand = sleep, rand

    def get(self, start: int, end: int) -> bytes:
        """Bytes ``start..end`` inclusive (shorter only if the object ends first)."""
        want = end - start + 1
        last_reason = FailureReason.S3_TRANSIENT
        last_type = None
        for attempt in range(self.max_attempts):
            self.deadline.check()
            if attempt:
                self.stats.n_retries += 1
                delay = self.rand() * min(self.max_backoff_s, self.backoff_s * (2 ** (attempt - 1)))
                rem = self.deadline.remaining()
                if rem is not None and delay >= rem:
                    raise RecordingTimeout()
                self.sleep(delay)
                self.deadline.check()
            self.stats.n_requests += 1
            try:
                resp = self.s3.get_object(Bucket=self.bucket, Key=self.key, Range=f"bytes={start}-{end}")
                body = resp["Body"].read()
                total = _parse_content_range(resp.get("ContentRange"))
            except RecordingTimeout:
                raise
            except Exception as exc:                          # noqa: BLE001 - classified, message discarded
                reason, retry = classify_s3_error(exc)
                last_reason, last_type = reason, _type_name(exc)
                if not retry:
                    raise _StreamError(reason, last_type) from None
                continue
            if total is not None:
                self.stats.file_bytes = total
            if len(body) > want:
                raise _StreamError(FailureReason.S3_ERROR, "RangeIgnored")
            known = self.stats.file_bytes
            expected = want if known is None else max(0, min(want, known - start))
            if len(body) < expected:                          # short read of a non-final range: retry
                last_reason, last_type = FailureReason.S3_TRANSIENT, "ShortRead"
                continue
            self.stats.bytes_fetched += len(body)
            return body
        raise _StreamError(last_reason, last_type)


class SparseEDF(io.RawIOBase):
    """Read-only, seekable view of an EDF of which only a few byte segments are held in memory.

    ``io.read_edf`` seeks to the header and to ``header_bytes + first_record * record_bytes``; this serves exactly
    those. A read that touches bytes we did not fetch raises (it would mean the plan disagrees with the reader)."""

    def __init__(self, segments: list[tuple[int, bytes]], size: int):
        self._segs = sorted(segments)
        self._size = size
        self._pos = 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self._pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self._size}[whence]
        self._pos = max(0, base + offset)
        return self._pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = max(0, self._size - self._pos)
        if n == 0:
            return b""
        for off, buf in self._segs:
            if off <= self._pos < off + len(buf):
                out = buf[self._pos - off: self._pos - off + n]
                if len(out) < n and self._pos + len(out) < self._size and not self._covers(self._pos + len(out)):
                    raise _StreamError(FailureReason.DECODE_ERROR, "UnfetchedBytes")
                self._pos += len(out)
                return out
        if self._pos >= self._size:
            return b""
        raise _StreamError(FailureReason.DECODE_ERROR, "UnfetchedBytes")

    def _covers(self, pos: int) -> bool:
        return any(off <= pos < off + len(b) for off, b in self._segs)

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


@dataclass
class DataPlan:
    r0: int                  # first record
    n_records: int           # records to fetch
    byte_start: int          # absolute
    byte_end: int            # absolute, inclusive
    start_s: float
    duration_s: float


def plan_window(hdr: EDFHeader, start_s: float, duration_s: float, n_records_available: int | None = None) -> DataPlan | None:
    """Record / byte range for ``[start_s, start_s + duration_s)``; same arithmetic as ``io.read_edf`` so the bytes
    fetched are exactly the bytes it reads. ``None`` when the recording ends before the window starts."""
    n_rec = hdr.n_records if n_records_available is None else min(hdr.n_records, n_records_available)
    total = n_rec * hdr.record_duration
    t0 = min(max(0.0, start_s), total)
    t1 = min(total, t0 + duration_s)
    r0 = int(np.floor(t0 / hdr.record_duration + 1e-9))
    r1 = int(np.ceil(t1 / hdr.record_duration - 1e-9))
    if r1 <= r0:
        return None
    rb = hdr.record_bytes
    return DataPlan(r0, r1 - r0, hdr.header_bytes + r0 * rb, hdr.header_bytes + r1 * rb - 1, start_s, duration_s)


def _patch_n_records(header: bytes, n: int) -> bytes:
    b = bytearray(header)
    b[236:244] = str(int(n)).encode("ascii").ljust(8, b" ")[:8]
    return bytes(b)


def fetch_window(s3, key: str, start_s: float, duration_s: float, *, bucket: str | None = None,
                 deadline: Deadline | None = None, stats: FetchStats | None = None,
                 max_attempts: int = DEFAULT_MAX_ATTEMPTS, backoff_s: float = 1.0, max_backoff_s: float = 30.0,
                 sleep=time.sleep, rand=random.random, chunk_bytes: int = CHUNK_BYTES,
                 max_fetch_bytes: int = MAX_FETCH_BYTES, channels=CANONICAL_19) -> Recording:
    """Ranged-read one EDF window and decode it to a ``Recording`` (uV) in memory. Raises ``_StreamError`` /
    ``RecordingTimeout`` only (the caller turns them into a ``StreamResult``)."""
    if bucket is None:
        from ..data_io import access_point
        bucket = access_point()
    deadline = deadline or Deadline(None)
    stats = stats if stats is not None else FetchStats()
    rg = _Ranged(s3, bucket, key, deadline, stats, max_attempts, backoff_s, max_backoff_s, sleep, rand)

    head = rg.get(0, HEADER_PROBE_BYTES - 1)
    if len(head) < 256:
        raise _StreamError(FailureReason.HEADER_INVALID)
    try:
        nbytes = int(head[184:192].decode("ascii", "replace").strip())
    except ValueError:
        raise _StreamError(FailureReason.HEADER_INVALID) from None
    if not 512 <= nbytes <= 1 << 20:
        raise _StreamError(FailureReason.HEADER_INVALID)
    if nbytes > len(head):                                     # unusually many signals: fetch the rest of the header
        head = head + rg.get(len(head), nbytes - 1)
    try:
        hdr = read_edf_header(io.BytesIO(head))
    except EDFError as exc:
        raise _StreamError(classify_edf_error(exc)) from None
    if hdr.discontinuous:
        raise _StreamError(FailureReason.DISCONTINUOUS)
    if hdr.header_bytes > len(head) or hdr.record_bytes <= 0:
        raise _StreamError(FailureReason.HEADER_INVALID)
    size = stats.file_bytes
    avail = None if size is None else max(0, (size - hdr.header_bytes) // hdr.record_bytes)
    if hdr.n_records < 0:                                       # unknown record count: infer from the object size
        if avail is None:
            raise _StreamError(FailureReason.HEADER_INVALID)
        hdr.n_records = avail
    n_eff = hdr.n_records if avail is None else min(hdr.n_records, avail)
    plan = plan_window(hdr, start_s, duration_s, n_eff)
    if plan is None:
        raise _StreamError(FailureReason.TOO_SHORT)
    if plan.byte_end - plan.byte_start + 1 > max_fetch_bytes:
        raise _StreamError(FailureReason.TOO_LARGE)

    header_seg = _patch_n_records(head[:hdr.header_bytes], n_eff)    # reader sees the true (clamped) record count
    parts, pos = [], plan.byte_start
    while pos <= plan.byte_end:
        end = min(plan.byte_end, pos + chunk_bytes - 1)
        parts.append(rg.get(pos, end))
        pos = end + 1
    data = b"".join(parts)
    del parts
    sparse = SparseEDF([(0, header_seg), (plan.byte_start, data)],
                       size=hdr.header_bytes + n_eff * hdr.record_bytes)
    deadline.check()
    try:
        return read_edf(sparse, start_s=start_s, duration_s=duration_s, channels=list(channels))
    except EDFError as exc:
        raise _StreamError(classify_edf_error(exc)) from None


def _augment_rows(rows: list[dict], qc: dict[str, WindowQC]) -> list[dict]:
    """Add scalar QC flag columns (flag prevalence among minimum-set cells, coverage, disconnected channels)."""
    out = []
    for r in rows:
        q = qc[r["window"]]
        extra = {"qc_clean_cell_fraction": float(q.clean_cell_fraction),
                 "qc_coverage_fraction": float(q.coverage_fraction),
                 "qc_n_disconnected": int(q.disconnected_channels)}
        extra.update({f"qc_flag_{k}": float(q.flag_fraction.get(k, 0.0)) for k in FLAG_NAMES})
        out.append({**r, **extra})
    return out


def stream_features(s3, key: str, *, bucket: str | None = None, windows=None, qc_cfg: QCConfig | None = None,
                    pre_cfg: PreprocessConfig | None = None, feat_cfg: FeatureConfig | None = None,
                    compute_failed: bool = False, timeout_s: float | None = DEFAULT_TIMEOUT_S,
                    max_attempts: int = DEFAULT_MAX_ATTEMPTS, backoff_s: float = 1.0, max_backoff_s: float = 30.0,
                    sleep=time.sleep, rand=random.random, chunk_bytes: int = CHUNK_BYTES,
                    max_fetch_bytes: int = MAX_FETCH_BYTES, clock=time.monotonic) -> StreamResult:
    """Features for one S3 EDF, reading only the primary window (default minutes 1-11) plus filter padding.

    Never raises for data / network / timeout problems; see ``StreamResult.reason``. ``windows`` defaults to the
    primary window plus the nested windows (all inside the primary window)."""
    t_start = clock()
    windows = dict(windows or all_windows())
    start = max(0.0, min(w.start_s for w in windows.values()) - PAD_S)
    end = max(w.end_s for w in windows.values()) + PAD_S
    deadline = Deadline(timeout_s, clock=clock)
    stats = FetchStats()
    res = StreamResult(ok=False)
    stage = "fetch"
    try:
        with deadline.alarm():
            rec = fetch_window(s3, key, start, end - start, bucket=bucket, deadline=deadline, stats=stats,
                               max_attempts=max_attempts, backoff_s=backoff_s, max_backoff_s=max_backoff_s,
                               sleep=sleep, rand=rand, chunk_bytes=chunk_bytes, max_fetch_bytes=max_fetch_bytes)
            res.fetch_s = clock() - t_start
            stage = "process"
            t_proc = clock()
            deadline.check()
            out = process_recording(rec, windows=windows, qc_cfg=qc_cfg, pre_cfg=pre_cfg, feat_cfg=feat_cfg,
                                    compute_failed=compute_failed)
            res.process_s = clock() - t_proc
        res.rows = _augment_rows(out.rows, out.qc)
        res.qc = out.qc
        res.n_channels = int(out.channel_status.get("n_channels", 0))
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
