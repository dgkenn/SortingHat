"""Resilient S3 reads (dropped streams, ranged parquet by row group) and robust EDF key resolution.

Everything runs against an in-memory fake S3 over SYNTHETIC bytes. The fake randomly raises botocore's
``ResponseStreamingError`` (and friends) in the middle of ``Body.read()``, which is exactly the real-data failure
(a dropped stream through the proxy), and the reads must still complete and match the clean result.
"""

import io
import random

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from botocore.exceptions import (ConnectionClosedError, IncompleteReadError, ReadTimeoutError,
                                 ResponseStreamingError)

from sortinghat import data_io as dio

AP = "fake-access-point"
FLAKES = (lambda: ResponseStreamingError(error="IncompleteRead(10 bytes read, 90 more expected)"),
          lambda: IncompleteReadError(actual_bytes=10, expected_bytes=100),
          lambda: ReadTimeoutError(endpoint_url="http://fake"),
          lambda: ConnectionResetError("reset"),
          lambda: ConnectionClosedError(endpoint_url="http://fake"))


class ClientError(Exception):
    def __init__(self, code, status):
        super().__init__(code)
        self.response = {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}}


class _Body:
    def __init__(self, data, fail):
        self._data, self._fail = data, fail

    def read(self, *a):
        if self._fail is not None:
            raise self._fail                      # the stream drops while the body is being consumed
        return self._data


class FlakyS3:
    """get_object / head_object / list_objects_v2 over in-memory objects. With probability ``p`` a GET's body read
    raises one of ``FLAKES`` (seeded, so runs are reproducible). ``truncate_p``: return a silently short body."""

    def __init__(self, objects, p=0.35, seed=0, truncate_p=0.0):
        self.objects, self.p, self.truncate_p = objects, p, truncate_p
        self.rng = random.Random(seed)
        self.gets: list[tuple[str, int, int]] = []
        self.n_dropped = 0
        self.lists: list[tuple[str, str | None]] = []
        self.heads: list[str] = []

    def get_object(self, Bucket, Key, Range=None):
        if Key not in self.objects:
            raise ClientError("NoSuchKey", 404)
        data = self.objects[Key]
        a, b = 0, len(data) - 1
        if Range:
            a, b = (int(x) for x in Range.removeprefix("bytes=").split("-"))
            if a >= len(data):
                raise ClientError("InvalidRange", 416)
            b = min(b, len(data) - 1)
        self.gets.append((Key, a, b))
        body = data[a:b + 1]
        fail = None
        if self.rng.random() < self.p:
            fail = self.rng.choice(FLAKES)()
            self.n_dropped += 1
        elif self.rng.random() < self.truncate_p and len(body) > 1:
            body = body[: len(body) // 2]
            self.n_dropped += 1
        resp = {"Body": _Body(body, fail), "ContentLength": len(data[a:b + 1])}
        if Range:
            resp["ContentRange"] = f"bytes {a}-{b}/{len(data)}"
        return resp

    def head_object(self, Bucket, Key):
        self.heads.append(Key)
        if Key not in self.objects:
            raise ClientError("404", 404)
        return {"ContentLength": len(self.objects[Key])}

    def list_objects_v2(self, Bucket, Prefix, MaxKeys=1000, ContinuationToken=None, Delimiter=None):
        self.lists.append((Prefix, Delimiter))
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        if not Delimiter:
            return {"Contents": [{"Key": k} for k in keys], "IsTruncated": False}
        files, cps = [], set()
        for k in keys:
            rest = k[len(Prefix):]
            if "/" in rest:
                cps.add(Prefix + rest.split("/", 1)[0] + "/")
            else:
                files.append(k)
        return {"Contents": [{"Key": k} for k in files], "CommonPrefixes": [{"Prefix": c} for c in sorted(cps)],
                "IsTruncated": False}


FAST = dio.RetryPolicy(max_attempts=60, backoff_s=0.0, max_backoff_s=0.0, sleep=lambda s: None)


def _parquet_bytes(n_rows=6000, row_group=500, extra_cols=True) -> tuple[bytes, pd.DataFrame]:
    rng = random.Random(1)
    df = pd.DataFrame({
        "person_id": [i // 3 for i in range(n_rows)],                       # sorted: row-group statistics are tight
        "measurement_source_value": [f"v{rng.randint(0, 999)}" for _ in range(n_rows)],
        "value_as_number": [rng.random() for _ in range(n_rows)],
        "not_requested": ["x" * 40 for _ in range(n_rows)],
    })
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), buf, row_group_size=row_group)
    return buf.getvalue(), df


def _table(batches) -> pd.DataFrame:
    frames = [b.to_pandas() for b in batches]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


COLS = ["person_id", "measurement_source_value", "value_as_number"]


@pytest.fixture(scope="module")
def parquet():
    return _parquet_bytes()


def _store(parquet, **kw):
    blob, _ = parquet
    return FlakyS3({"OMOP/Merged/measurement/part-00000.parquet": blob}, **kw)


def test_parquet_read_survives_random_dropped_streams(parquet):
    _, df = parquet
    s3 = _store(parquet, p=0.35, seed=3)
    dio.RETRY_COUNTS.clear()
    got = _table(dio.iter_omop_batches("measurement", columns=COLS + ["not_in_file"], s3=s3, retry=FAST,
                                       max_get_bytes=4096, batch_rows=700))
    assert s3.n_dropped > 5 and sum(dio.RETRY_COUNTS.values()) >= s3.n_dropped     # streams really dropped
    assert list(got.columns) == COLS                                              # pruned; missing column skipped
    pd.testing.assert_frame_equal(got, df[COLS])


def test_cohort_filter_matches_and_skips_row_groups(parquet):
    _, df = parquet
    ids = [5, 6, 1700]                                                   # row groups 0 and 3 only (500 rows = 166 ids)
    clean = _store(parquet, p=0.0)
    flaky = _store(parquet, p=0.3, seed=9)
    want = df[df.person_id.isin(ids)][COLS].reset_index(drop=True)
    got_clean = _table(dio.iter_omop_batches("measurement", person_ids=ids, columns=COLS, s3=clean, retry=FAST,
                                             buffer_size=4096))
    got = _table(dio.iter_omop_batches("measurement", person_ids=ids, columns=COLS, s3=flaky, retry=FAST,
                                       max_get_bytes=2048))
    pd.testing.assert_frame_equal(got_clean, want)
    pd.testing.assert_frame_equal(got, want)
    full = _store(parquet, p=0.0)
    _table(dio.iter_omop_batches("measurement", columns=COLS, s3=full, retry=FAST, buffer_size=4096))
    assert sum(b - a + 1 for _, a, b in clean.gets) < 0.6 * sum(b - a + 1 for _, a, b in full.gets)   # early filter (64 KiB footer probe included)


def test_cohort_filter_with_no_match_reads_no_data_pages(parquet):
    s3 = _store(parquet, p=0.0)
    assert _table(dio.iter_omop_batches("measurement", person_ids=[10 ** 9], columns=COLS, s3=s3, retry=FAST,
                                        buffer_size=4096)).empty
    full = _store(parquet, p=0.0)
    _table(dio.iter_omop_batches("measurement", columns=COLS, s3=full, retry=FAST, buffer_size=4096))
    # statistics prove no match: only the footer is fetched, no row-group data
    assert sum(b - a + 1 for _, a, b in s3.gets) < 0.5 * sum(b - a + 1 for _, a, b in full.gets)


def test_person_filter_without_statistics_pruning_still_exact(parquet):
    _, df = parquet
    ids = [2, 4, 1000, 1999]           # in-range ids: statistics cannot exclude every group, so person_id is read first
    s3 = _store(parquet, p=0.2, seed=4)
    got = _table(dio.iter_omop_batches("measurement", person_ids=ids, columns=COLS, s3=s3, retry=FAST,
                                       max_get_bytes=1024))
    pd.testing.assert_frame_equal(got, df[df.person_id.isin(ids)][COLS].reset_index(drop=True))


@pytest.mark.parametrize("seed", range(4))
def test_many_seeds(parquet, seed):
    _, df = parquet
    s3 = _store(parquet, p=0.4, seed=100 + seed, truncate_p=0.1)
    got = _table(dio.iter_omop_batches("measurement", columns=COLS, s3=s3, retry=FAST, max_get_bytes=3000))
    pd.testing.assert_frame_equal(got, df[COLS])


def test_exhausted_retries_raise_and_on_error_reports(parquet):
    s3 = _store(parquet, p=1.0, seed=1)
    pol = dio.RetryPolicy(max_attempts=3, backoff_s=0, sleep=lambda s: None)
    with pytest.raises(Exception) as ei:
        _table(dio.iter_omop_batches("measurement", columns=COLS, s3=s3, retry=pol))
    assert dio.is_retryable(ei.value) or type(ei.value).__name__ in ("ResponseStreamingError", "OSError")
    seen = []
    s3 = _store(parquet, p=1.0, seed=1)
    out = _table(dio.iter_omop_batches("measurement", columns=COLS, s3=s3, retry=pol,
                                       on_error=lambda k, e: seen.append(type(e).__name__)))
    assert out.empty and len(seen) == 1


def test_backoff_is_jittered_exponential_and_capped():
    delays = []
    pol = dio.RetryPolicy(max_attempts=6, backoff_s=2.0, max_backoff_s=10.0, sleep=delays.append, rand=lambda: 1.0)
    n = []

    def boom():
        n.append(1)
        raise ResponseStreamingError(error="x")

    with pytest.raises(ResponseStreamingError):
        dio.with_retries(boom, pol)
    assert len(n) == 6 and delays == [2.0, 4.0, 8.0, 10.0, 10.0]
    delays.clear()
    pol2 = dio.RetryPolicy(max_attempts=3, backoff_s=2.0, sleep=delays.append, rand=lambda: 0.0)
    with pytest.raises(ResponseStreamingError):
        dio.with_retries(boom, pol2)
    assert delays == [1.0, 2.0]                                         # jitter floor is half the cap


def test_non_retryable_errors_fail_fast():
    calls = []

    def nf():
        calls.append(1)
        raise ClientError("NoSuchKey", 404)

    with pytest.raises(ClientError):
        dio.with_retries(nf, FAST)
    assert len(calls) == 1
    for exc in (FileNotFoundError("x"), PermissionError("x"), KeyError("x"), ValueError("x"),
                ClientError("AccessDenied", 403)):
        assert not dio.is_retryable(exc)
    for exc in (ResponseStreamingError(error="x"), IncompleteReadError(actual_bytes=1, expected_bytes=2),
                ReadTimeoutError(endpoint_url="u"), ConnectionError("x"), TimeoutError("x"),
                ClientError("SlowDown", 503), ClientError("InternalError", 500)):
        assert dio.is_retryable(exc)


def test_csv_reads_survive_dropped_streams_and_match():
    rows = ["BidsFolder,SessionID,EEGFolder"] + [f"sub-S0001{i},{i % 5},cEEG" for i in range(4000)]
    csv = ("\n".join(rows) + "\n").encode()
    key = "EEG/eeg-metadata/S0001_eeg_metadata_2026_04_30.csv"
    s3 = FlakyS3({key: csv}, p=0.4, seed=2, truncate_p=0.1)
    got = dio.read_csv_table("eeg_metadata", "S0001", s3=s3, retry=FAST)
    want = pd.read_csv(io.BytesIO(csv), dtype=str)
    pd.testing.assert_frame_equal(got, want)
    small = dio.read_object(s3, AP, key, chunk=1000, policy=FAST)       # many chunks, each retried alone
    assert small == csv and s3.n_dropped > 0
    assert dio.csv_header(s3, key, bucket=AP, policy=FAST) == ["BidsFolder", "SessionID", "EEGFolder"]


def test_read_object_edge_cases():
    s3 = FlakyS3({"a": b"", "b": b"x" * 1000, "c": b"y" * 999}, p=0.0)
    assert dio.read_object(s3, AP, "a", chunk=100, policy=FAST) == b""
    assert dio.read_object(s3, AP, "b", chunk=100, policy=FAST) == b"x" * 1000      # exact multiple of the chunk
    assert dio.read_object(s3, AP, "c", chunk=100, policy=FAST) == b"y" * 999
    with pytest.raises(ClientError):
        dio.read_object(s3, AP, "missing", policy=FAST)


def test_site_table_read_survives_dropped_streams():
    csv = (b"SiteID,BidsFolder,SessionID,EEGFolder,DeidentifiedName(Reports)\n"
           + b"".join(f"S0001,sub-S0001{i},{i},cEEG,name{i}\n".encode() for i in range(500)))
    s3 = FlakyS3({"EEG/eeg-metadata/S0001_eeg_metadata_2026_04_30.csv": csv}, p=0.4, seed=5)
    df = dio.read_site_table("eeg_metadata", "S0001", s3=s3, retry=FAST)
    assert len(df) == 500 and "BidsFolder" in df


def test_local_store_still_works_with_iter_omop_batches(tmp_path, parquet):
    d = tmp_path / "OMOP" / "Merged" / "measurement"
    d.mkdir(parents=True)
    (d / "part-00000.parquet").write_bytes(parquet[0])
    store = dio.LocalStore(tmp_path)
    got = _table(dio.iter_omop_batches("measurement", person_ids=[3, 4], columns=COLS, s3=store))
    assert sorted(set(got.person_id)) == [3, 4] and len(got) == 6


# ------------------------------------------------------------------------------------------ key resolution
S, BF, SID = "S0001", "sub-S0001123", "7"
FOLDER = f"EEG/bids/{S}/{BF}/ses-{SID}/eeg/"


def _res(objects, **kw):
    return dio.resolve_edf_key(FlakyS3(objects, p=0.0), S, BF, SID, kw.pop("eeg", None), bucket=AP,
                               policy=FAST, **kw)


def test_documented_pattern_resolves_first():
    r = _res({FOLDER + f"{BF}_ses-{SID}_task-EEG_eeg.edf": b"x"})
    assert r.found and r.pattern == "documented" and r.key.endswith("task-EEG_eeg.edf")
    r = _res({FOLDER + f"{BF}_ses-{SID}_task-cEEG_eeg.edf": b"x"}, eeg="ceeg_unit")
    assert r.pattern == "documented"


def test_other_task_token_and_task_less_names():
    assert _res({FOLDER + f"{BF}_ses-{SID}_task-cEEG_eeg.edf": b"x"}).pattern == "alt_task"
    assert _res({FOLDER + f"{BF}_ses-{SID}_task-EEG_eeg.edf": b"x"}, eeg="cEEG1").pattern == "alt_task"
    assert _res({FOLDER + f"{BF}_ses-{SID}_eeg.edf": b"x"}).pattern == "no_task"


def test_session_id_spelling_variants():
    r = dio.resolve_edf_key(FlakyS3({FOLDER + f"{BF}_ses-{SID}_task-EEG_eeg.edf": b"x"}, p=0.0), S, BF, "7.0",
                            bucket=AP, policy=FAST)
    assert r.pattern == "documented+sid_variant"
    r = dio.resolve_edf_key(FlakyS3({"EEG/bids/S0001/sub-S0001123/ses-12/eeg/sub-S0001123_ses-12_task-EEG_eeg.edf": b"x"},
                                    p=0.0), S, BF, "0012", bucket=AP, policy=FAST)
    assert r.pattern == "documented+sid_variant"


def test_folder_listing_fallback_lists_only_the_own_folder():
    objs = {FOLDER + f"{BF}_ses-{SID}_acq-xyz_run-01_eeg.edf": b"x",
            FOLDER + f"{BF}_ses-{SID}_eeg.json": b"{}",
            f"EEG/bids/{S}/sub-S0001999/ses-1/eeg/sub-S0001999_ses-1_task-EEG_eeg.edf": b"other patient"}
    s3 = FlakyS3(objs, p=0.0)
    r = dio.resolve_edf_key(s3, S, BF, SID, bucket=AP, policy=FAST)
    assert r.pattern == "folder_listing" and r.key == FOLDER + f"{BF}_ses-{SID}_acq-xyz_run-01_eeg.edf"
    assert r.folder_exists and r.n_edf == 1 and r.ext_counts == {"edf": 1, "json": 1}
    assert s3.lists and all(p == FOLDER and d == "/" for p, d in s3.lists)        # one folder, one level, nothing else


def test_folder_with_no_edf_and_with_multiple_edf():
    r = _res({FOLDER + f"{BF}_ses-{SID}_eeg.json": b"{}", FOLDER + f"{BF}_ses-{SID}_channels.tsv": b"x"})
    assert not r.found and r.pattern == "not_found" and r.folder_exists and r.n_edf == 0
    assert r.ext_counts == {"json": 1, "tsv": 1}
    two = {FOLDER + f"{BF}_ses-{SID}_run-1_eeg.edf": b"x", FOLDER + f"{BF}_ses-{SID}_run-2_eeg.edf": b"y"}
    r = _res(two)
    assert r.found and r.n_edf == 2 and r.pattern == "folder_listing"
    assert r.key.endswith("run-1_eeg.edf")                                        # deterministic choice
    r = _res({})
    assert not r.found and r.folder_exists is False and r.n_edf == 0


def test_upper_case_extension_and_gz_are_counted_not_matched_as_edf():
    r = _res({FOLDER + f"{BF}_ses-{SID}_eeg.EDF": b"x", FOLDER + f"{BF}_ses-{SID}_eeg.edf.gz": b"y"})
    assert r.found and r.key.endswith(".EDF") and r.ext_counts == {"edf": 1, "edf.gz": 1}


def test_always_list_fills_folder_stats_even_when_resolved_by_pattern():
    objs = {FOLDER + f"{BF}_ses-{SID}_task-EEG_eeg.edf": b"x", FOLDER + "extra.json": b"{}"}
    r = _res(objs, always_list=True)
    assert r.pattern == "documented" and r.folder_exists and r.n_edf == 1 and r.ext_counts == {"edf": 1, "json": 1}


def test_parent_listing_is_opt_in():
    objs = {f"EEG/bids/{S}/{BF}/ses-007/eeg/{BF}_ses-007_task-EEG_eeg.edf": b"x"}
    assert not _res(objs, parent_fallback=False).found
    r = _res(objs, parent_fallback=True)
    assert r.found and r.pattern == "parent_listing"


def test_file_ext():
    assert dio.file_ext("a/b/x_eeg.edf") == "edf" and dio.file_ext("x.edf.gz") == "edf.gz"
    assert dio.file_ext("README") == "other" and dio.file_ext("x.weird_ext!") == "other"
    assert set(dio.EDF_PATTERNS) >= {"documented", "folder_listing", "not_found"}


def test_key_exists_treats_missing_and_denied_as_false_and_retries_transient():
    s3 = FlakyS3({"k": b"x"}, p=0.0)
    assert dio.key_exists(s3, "k", bucket=AP, policy=FAST) and not dio.key_exists(s3, "nope", bucket=AP, policy=FAST)

    class Denied(FlakyS3):
        def head_object(self, Bucket, Key):
            raise ClientError("403", 403)

    assert not dio.key_exists(Denied({}), "k", bucket=AP, policy=FAST)

    class Flaky(FlakyS3):
        n = 0

        def head_object(self, Bucket, Key):
            Flaky.n += 1
            if Flaky.n < 3:
                raise ClientError("SlowDown", 503)
            return {"ContentLength": 1}

    assert dio.key_exists(Flaky({}), "k", bucket=AP, policy=FAST)
