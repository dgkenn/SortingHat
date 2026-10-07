"""Streaming EEG extraction against a fake S3 over a local SYNTHETIC EDF (no real data, no network)."""
import io

import numpy as np
import pytest

from sortinghat.eeg import stream
from sortinghat.eeg.io import CANONICAL_19, read_edf, read_edf_header
from sortinghat.eeg.pipeline import process_recording
from sortinghat.eeg.stream import FailureReason, stream_features
from sortinghat.eeg.synthetic import generate_eeg, write_edf

BUCKET = "fake-ap"
KEY = "EEG/bids/SYN/sub-SYN1/ses-1/eeg/sub-SYN1_ses-1_task-EEG_eeg.edf"


class FakeClientError(Exception):
    def __init__(self, code, status):
        super().__init__(code)
        self.response = {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}}


class _Body:
    def __init__(self, b):
        self._b = b

    def read(self):
        return self._b


class FakeS3:
    """Implements ``get_object(Bucket, Key, Range)`` over in-memory objects; records every request.
    ``faults``: list of exceptions (or None) consumed one per call before serving normally."""

    def __init__(self, objects: dict[str, bytes], faults=None, short_first=False):
        self.objects = objects
        self.calls: list[tuple[str, int, int]] = []
        self.faults = list(faults or [])
        self.short_first = short_first

    def get_object(self, Bucket, Key, Range=None):
        assert Bucket == BUCKET
        if self.faults:
            f = self.faults.pop(0)
            if f is not None:
                raise f
        if Key not in self.objects:
            raise FakeClientError("NoSuchKey", 404)
        data = self.objects[Key]
        assert Range and Range.startswith("bytes="), "full-object GET is not allowed"
        a, b = Range[6:].split("-")
        a, b = int(a), int(b)
        if a >= len(data):
            raise FakeClientError("InvalidRange", 416)
        b = min(b, len(data) - 1)
        self.calls.append((Key, a, b))
        body = data[a:b + 1]
        if self.short_first:
            self.short_first = False
            body = body[: len(body) // 2]
        return {"Body": _Body(body), "ContentRange": f"bytes {a}-{b}/{len(data)}", "ContentLength": len(body)}

    @property
    def bytes_served(self):
        return sum(b - a + 1 for _, a, b in self.calls)


@pytest.fixture(scope="module")
def edf(tmp_path_factory):
    d = tmp_path_factory.mktemp("stream")
    x = generate_eeg(780, background="normal", seed=11)
    p = write_edf(d / "a.edf", x, 200, CANONICAL_19, label_fmt="EEG {}-REF", record_s=1.0)
    return p


@pytest.fixture(scope="module")
def blob(edf):
    return edf.read_bytes()


def _no_sleep(_):
    pass


def test_byte_range_matches_full_decode(edf, blob):
    s3 = FakeS3({KEY: blob})
    rec = stream.fetch_window(s3, KEY, 50.0, 620.0, bucket=BUCKET)
    full = read_edf(edf, start_s=50.0, duration_s=620.0, channels=list(CANONICAL_19))
    assert rec.ch_names == full.ch_names and rec.fs == full.fs and rec.offset_s == full.offset_s
    assert rec.data.shape == full.data.shape
    assert np.array_equal(rec.data, full.data)           # bit-identical decode, not merely close
    assert rec.meta["edf_duration_s"] == full.meta["edf_duration_s"]


def test_only_needed_bytes_are_fetched(edf, blob):
    s3 = FakeS3({KEY: blob})
    stream.fetch_window(s3, KEY, 50.0, 620.0, bucket=BUCKET)
    h = read_edf_header(edf)
    assert all(k == KEY for k, _, _ in s3.calls)
    header_gets = [(a, b) for _, a, b in s3.calls if a == 0]
    assert len(header_gets) == 1
    data_gets = [(a, b) for _, a, b in s3.calls if a != 0]
    lo = min(a for a, _ in data_gets)
    hi = max(b for _, b in data_gets)
    assert lo == h.header_bytes + 50 * h.record_bytes             # record 50 on
    assert hi == h.header_bytes + 670 * h.record_bytes - 1        # up to record 669 inclusive
    fetched = sum(b - a + 1 for a, b in data_gets)
    assert fetched == 620 * h.record_bytes
    assert fetched < 0.85 * len(blob)                             # 620 of 780 s: nowhere near the whole file
    assert s3.bytes_served <= len(blob)


def test_large_file_fetch_is_a_small_fraction(tmp_path):
    x = generate_eeg(2400, background="normal", seed=5)
    p = write_edf(tmp_path / "long.edf", x, 200, CANONICAL_19)
    blob = p.read_bytes()
    s3 = FakeS3({KEY: blob})
    r = stream_features(s3, KEY, bucket=BUCKET, sleep=_no_sleep)
    assert r.ok
    assert r.bytes_fetched == s3.bytes_served
    assert r.bytes_fetched < 0.30 * len(blob)                     # 620 s of 2400 s


def test_chunking_splits_requests_but_not_content(edf, blob):
    s3 = FakeS3({KEY: blob})
    rec = stream.fetch_window(s3, KEY, 50.0, 620.0, bucket=BUCKET, chunk_bytes=1_000_000)
    full = read_edf(edf, start_s=50.0, duration_s=620.0, channels=list(CANONICAL_19))
    assert np.array_equal(rec.data, full.data)
    assert len([c for c in s3.calls if c[1] != 0]) >= 3


def test_features_match_local_pipeline(edf, blob):
    s3 = FakeS3({KEY: blob})
    r = stream_features(s3, KEY, bucket=BUCKET, sleep=_no_sleep)
    assert r.ok and r.reason is None
    ref = process_recording(edf)
    assert [x["window"] for x in r.rows] == [x["window"] for x in ref.rows]
    for got, want in zip(r.rows, ref.rows):
        assert got["qc_pass"] == want["qc_pass"]
        for k, v in want.items():
            if isinstance(v, float):
                assert got[k] == pytest.approx(v, rel=1e-9, abs=1e-9, nan_ok=True)
    assert "qc_flag_flat" in r.rows[0] and "qc_coverage_fraction" in r.rows[0]
    assert r.n_channels == 19 and r.n_requests >= 2 and r.bytes_fetched > 0


def test_short_recording_is_handled_by_qc_not_failure(tmp_path):
    x = generate_eeg(200, background="normal", seed=2)
    p = write_edf(tmp_path / "s.edf", x, 200, CANONICAL_19)
    s3 = FakeS3({KEY: p.read_bytes()})
    r = stream_features(s3, KEY, bucket=BUCKET, sleep=_no_sleep)
    assert r.ok
    prim = next(row for row in r.rows if row["window"] == "primary")
    assert prim["qc_pass"] is False or prim["qc_pass"] == False  # noqa: E712
    assert prim["usable_fraction"] < 0.6


def test_recording_ending_before_window_is_too_short(tmp_path):
    x = generate_eeg(40, background="normal", seed=2)
    p = write_edf(tmp_path / "t.edf", x, 200, CANONICAL_19)
    s3 = FakeS3({KEY: p.read_bytes()})
    r = stream_features(s3, KEY, bucket=BUCKET, sleep=_no_sleep)
    assert (not r.ok) and r.reason == FailureReason.TOO_SHORT and not r.retryable


def test_truncated_object_is_clamped(tmp_path):
    x = generate_eeg(400, background="normal", seed=3)
    p = write_edf(tmp_path / "x.edf", x, 200, CANONICAL_19)
    blob = p.read_bytes()
    h = read_edf_header(p)
    cut = blob[: h.header_bytes + 300 * h.record_bytes + 123]       # header claims 400 s, object holds 300.x s
    s3 = FakeS3({KEY: cut})
    rec = stream.fetch_window(s3, KEY, 50.0, 620.0, bucket=BUCKET)
    assert rec.meta["edf_duration_s"] == 300.0 and rec.data.shape[1] == 250 * 200


def test_unknown_record_count_inferred_from_size(tmp_path):
    x = generate_eeg(300, background="normal", seed=3)
    p = write_edf(tmp_path / "u.edf", x, 200, CANONICAL_19)
    b = bytearray(p.read_bytes())
    b[236:244] = b"-1".ljust(8)
    s3 = FakeS3({KEY: bytes(b)})
    rec = stream.fetch_window(s3, KEY, 50.0, 200.0, bucket=BUCKET)
    assert rec.meta["edf_duration_s"] == 300.0 and rec.data.shape[1] == 200 * 200


def test_many_signals_header_larger_than_probe(tmp_path, monkeypatch, blob):
    monkeypatch.setattr(stream, "HEADER_PROBE_BYTES", 1024)         # force the second header GET
    s3 = FakeS3({KEY: blob})
    rec = stream.fetch_window(s3, KEY, 50.0, 100.0, bucket=BUCKET)
    assert rec.data.shape == (19, 100 * 200)
    assert sum(1 for c in s3.calls if c[1] == 0) == 1 and any(0 < c[1] < 5000 for c in s3.calls)


def test_retry_with_backoff_then_success(blob):
    delays = []
    faults = [FakeClientError("SlowDown", 503), None, FakeClientError("InternalError", 500)]
    s3 = FakeS3({KEY: blob}, faults=faults)
    r = stream_features(s3, KEY, bucket=BUCKET, sleep=delays.append, rand=lambda: 1.0, backoff_s=2.0)
    assert r.ok and r.n_retries == 2
    assert delays[0] == pytest.approx(2.0) and len(delays) == 2   # attempt-1 delay = backoff * 2**0, full jitter = 1.0


def test_backoff_grows_and_exhausts(blob):
    delays = []
    faults = [FakeClientError("SlowDown", 503)] * 10
    s3 = FakeS3({KEY: blob}, faults=faults)
    r = stream_features(s3, KEY, bucket=BUCKET, sleep=delays.append, rand=lambda: 1.0, backoff_s=1.0,
                        max_attempts=4)
    assert (not r.ok) and r.reason == FailureReason.S3_TRANSIENT and r.retryable
    assert delays == [1.0, 2.0, 4.0]
    assert r.n_requests == 4


def test_connection_errors_are_retried(blob):
    s3 = FakeS3({KEY: blob}, faults=[ConnectionResetError("boom"), TimeoutError("slow")])
    r = stream_features(s3, KEY, bucket=BUCKET, sleep=_no_sleep)
    assert r.ok and r.n_retries == 2


def test_short_read_is_retried(blob):
    s3 = FakeS3({KEY: blob}, short_first=True)
    r = stream_features(s3, KEY, bucket=BUCKET, sleep=_no_sleep)
    assert r.ok and r.n_retries == 1


def test_missing_key_and_denied_are_permanent_and_not_retried():
    s3 = FakeS3({})
    r = stream_features(s3, KEY, bucket=BUCKET, sleep=_no_sleep)
    assert r.reason == FailureReason.NOT_FOUND and not r.retryable and r.n_requests == 1
    s3 = FakeS3({}, faults=[FakeClientError("AccessDenied", 403)])
    r = stream_features(s3, KEY, bucket=BUCKET, sleep=_no_sleep)
    assert r.reason == FailureReason.ACCESS_DENIED and r.n_requests == 1


def test_timeout_via_deadline_during_backoff(blob):
    t = [0.0]
    s3 = FakeS3({KEY: blob}, faults=[FakeClientError("SlowDown", 503)] * 10)

    def sleep(d):
        t[0] += d

    r = stream_features(s3, KEY, bucket=BUCKET, timeout_s=5.0, clock=lambda: t[0], sleep=sleep,
                        rand=lambda: 1.0, backoff_s=2.0, max_attempts=10)
    assert r.reason == FailureReason.TIMEOUT and r.retryable


def test_timeout_interrupts_cpu_stage(blob, monkeypatch):
    import time

    def slow(*a, **k):
        time.sleep(5)

    monkeypatch.setattr(stream, "process_recording", slow)
    s3 = FakeS3({KEY: blob})
    t0 = time.monotonic()
    r = stream_features(s3, KEY, bucket=BUCKET, timeout_s=0.5)
    assert r.reason == FailureReason.TIMEOUT
    assert time.monotonic() - t0 < 3.0


def test_corrupt_inputs_give_structured_reasons(tmp_path, blob):
    r = stream_features(FakeS3({KEY: b"x" * 5000}), KEY, bucket=BUCKET, sleep=_no_sleep)
    assert r.reason == FailureReason.HEADER_INVALID
    r = stream_features(FakeS3({KEY: b"abc"}), KEY, bucket=BUCKET, sleep=_no_sleep)
    assert r.reason == FailureReason.HEADER_INVALID
    x = generate_eeg(100, seed=1)
    p = write_edf(tmp_path / "d.edf", x, 200, CANONICAL_19, reserved="EDF+D")
    r = stream_features(FakeS3({KEY: p.read_bytes()}), KEY, bucket=BUCKET, sleep=_no_sleep)
    assert r.reason == FailureReason.DISCONTINUOUS and not r.retryable
    p = write_edf(tmp_path / "n.edf", x[:2], 200, ["Fp1", "Fp2"], label_fmt="ECG{}X")
    r = stream_features(FakeS3({KEY: p.read_bytes()}), KEY, bucket=BUCKET, sleep=_no_sleep)
    assert r.reason == FailureReason.NO_EEG_CHANNELS


def test_processing_error_keeps_only_class_name(blob, monkeypatch):
    def boom(*a, **k):
        raise ValueError("sub-S0001123456 secret message")

    monkeypatch.setattr(stream, "process_recording", boom)
    r = stream_features(FakeS3({KEY: blob}), KEY, bucket=BUCKET, sleep=_no_sleep)
    assert r.reason == FailureReason.PROCESSING_ERROR and r.error_type == "ValueError"
    for v in vars(r).values():
        assert "S0001123456" not in repr(v)


def test_failure_results_carry_no_key_or_message(blob):
    r = stream_features(FakeS3({}), KEY, bucket=BUCKET, sleep=_no_sleep)
    text = repr(vars(r))
    assert "sub-SYN1" not in text and "ses-1" not in text and BUCKET not in text


def test_nothing_written_to_disk(blob, tmp_path, monkeypatch):
    import builtins
    opened = []
    real_open = builtins.open

    def spy(file, mode="r", *a, **k):
        if any(c in str(mode) for c in "wax+"):
            opened.append(str(file))
        return real_open(file, mode, *a, **k)

    monkeypatch.setattr(builtins, "open", spy)
    monkeypatch.chdir(tmp_path)
    r = stream_features(FakeS3({KEY: blob}), KEY, bucket=BUCKET, sleep=_no_sleep)
    assert r.ok and opened == [] and list(tmp_path.iterdir()) == []


def test_plan_window_matches_read_edf_seek(edf):
    h = read_edf_header(edf)
    for start, dur in [(50.0, 620.0), (0.0, 20.0), (59.5, 10.2), (775.0, 100.0)]:
        plan = stream.plan_window(h, start, dur)
        seen = {}

        class Spy(io.BytesIO):
            def seek(self, off, whence=0):
                seen["off"] = off
                return super().seek(off, whence)

        read_edf(Spy(edf.read_bytes()), start_s=start, duration_s=dur)
        assert plan is not None and seen["off"] == plan.byte_start


def test_classifiers():
    assert stream.classify_s3_error(FakeClientError("NoSuchKey", 404)) == ("not_found", False)
    assert stream.classify_s3_error(FakeClientError("SlowDown", 503)) == ("s3_transient", True)
    assert stream.classify_s3_error(FakeClientError("WhoKnows", 400)) == ("s3_client_error", False)
    assert stream.classify_s3_error(OSError("x")) == ("s3_transient", True)
    assert stream.classify_s3_error(KeyError("x")) == ("s3_error", False)
    assert set(FailureReason.PERMANENT) <= set(stream.ALL_REASONS)
