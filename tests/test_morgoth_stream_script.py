"""scripts/extract_morgoth.py end to end against a fake S3 over SYNTHETIC EDFs, with the stub backend."""
import importlib.util
import json
import os
import stat
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sortinghat.eeg.io import CANONICAL_19
from sortinghat.eeg.stream import FailureReason
from sortinghat.eeg.synthetic import generate_eeg, write_edf
from sortinghat.morgoth import outputs
from sortinghat.morgoth.features import MorgothConfig
from sortinghat.morgoth.heads import feature_names
from sortinghat.morgoth.model import StubBackend
from sortinghat.morgoth.stream import stream_morgoth
from test_eeg_stream import BUCKET, FakeS3  # noqa: F401  (tests/ is on sys.path under pytest)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "extract_morgoth.py"
spec = importlib.util.spec_from_file_location("extract_morgoth", SCRIPT)
xm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xm)

HEADS = "normal,slowing"


def key(i):
    return f"EEG/bids/SYN/sub-SYN{i:03d}/ses-1/eeg/sub-SYN{i:03d}_ses-1_task-EEG_eeg.edf"


@pytest.fixture(scope="module")
def objects(tmp_path_factory):
    d = tmp_path_factory.mktemp("objs")
    objs = {}
    for i, bg in enumerate(["normal", "slowing", "normal", "low_voltage", "normal", "slowing"]):
        x = generate_eeg(700, background=bg, seed=i)
        objs[key(i)] = write_edf(d / f"{i}.edf", x, 200, CANONICAL_19).read_bytes()
    return objs


@pytest.fixture
def setup(tmp_path, monkeypatch, objects):
    import sortinghat.data_io as dio
    monkeypatch.setattr(dio, "access_point", lambda name="credentialed": BUCKET)
    lo = tmp_path / "local_only"
    lo.mkdir()
    keys = [key(i) for i in range(6)] + [key(99)]                     # key(99) does not exist
    pd.DataFrame({"edf_key": keys}).to_csv(lo / "keys.csv", index=False)
    return lo, FakeS3(dict(objects))


def run(lo, s3, capsys, *extra):
    rc = xm.main(["features", "--input", str(lo / "keys.csv"), "--out-dir", str(lo / "mg"), "--flush-every", "3",
                  "--backend", "stub", "--allow-stub", "--heads", HEADS, "--step-s", "30", "--windows", "20s,primary",
                  *extra], s3=s3)
    return rc, capsys.readouterr().out


def test_stream_morgoth_returns_rows_and_onset(setup):
    lo, s3 = setup
    res = stream_morgoth(s3, key(0), StubBackend(), cfg=MorgothConfig(heads=("normal",), step_s=30.0))
    assert res.ok and res.onset_s == 0.0
    assert [r["window"] for r in res.rows][0] == "primary" and len(res.rows) == 6
    assert res.bytes_fetched > 0 and res.n_requests > 1                   # ranged reads only (FakeS3 refuses full GETs)


def test_stream_morgoth_failure_codes(setup):
    lo, s3 = setup
    res = stream_morgoth(s3, key(99), StubBackend())
    assert not res.ok and res.reason == FailureReason.NOT_FOUND and not res.rows


def test_extract_writes_local_only_parts_and_aggregates(setup, capsys):
    lo, s3 = setup
    rc, out = run(lo, s3, capsys)
    assert rc == 0
    d = lo / "mg"
    parts = list(d.glob("part-*.parquet"))
    assert parts and all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in parts)
    assert stat.S_IMODE(d.stat().st_mode) == 0o700
    df = outputs.read_findings(d, window=None)
    assert df["recording_id"].nunique() == 6 and set(df["window"]) == {"primary", "20s"}
    assert all(c in df.columns for c in feature_names(HEADS.split(",")))
    info = json.loads((d / "run_info.json").read_text())
    assert info["backend"] == "stub" and info["stub_output_is_meaningless"] is True
    assert "attempted: <11" in out and "failure not_found: <11" in out
    assert "SYN" not in out and "sub-" not in out and "rec" + "0" not in out        # no keys or ids on stdout
    assert "morgoth.normal.abnormal.mean" in out                                    # n=6 -> suppressed quantiles
    assert "<11" in out


def test_resume_skips_done_and_does_not_duplicate(setup, capsys):
    lo, s3 = setup
    run(lo, s3, capsys)
    n_calls = len(s3.calls)
    rc, out = run(lo, s3, capsys)
    assert rc == 0 and "attempted: <11" in out and "already done (skipped): <11" in out
    assert len(s3.calls) == n_calls                                      # nothing refetched (not_found is permanent)
    df = outputs.read_findings(lo / "mg", window=None)
    assert not df.duplicated(["recording_id", "window"]).any()


def test_sharding_partitions_recordings(setup, capsys):
    lo, s3 = setup
    for k in range(3):
        xm.main(["features", "--input", str(lo / "keys.csv"), "--out-dir", str(lo / "mg"), "--backend", "stub",
                 "--allow-stub", "--heads", "normal", "--step-s", "60", "--windows", "20s", "--shard", str(k),
                 "--of", "3"], s3=s3)
    df = outputs.read_findings(lo / "mg", window=None)
    assert df["recording_id"].nunique() == 6


def test_refuses_non_local_only_paths(setup, tmp_path):
    lo, s3 = setup
    with pytest.raises(SystemExit):
        xm.main(["features", "--input", str(lo / "keys.csv"), "--out-dir", str(tmp_path / "plain"), "--backend", "stub",
                 "--allow-stub"], s3=s3)


def test_stub_needs_explicit_allow_and_run_info_mismatch_is_refused(setup, capsys):
    lo, s3 = setup
    with pytest.raises(SystemExit):
        xm.main(["features", "--input", str(lo / "keys.csv"), "--out-dir", str(lo / "mg"), "--backend", "stub"], s3=s3)
    run(lo, s3, capsys, "--limit", "1")
    with pytest.raises(SystemExit):                                    # same dir, different heads
        xm.main(["features", "--input", str(lo / "keys.csv"), "--out-dir", str(lo / "mg"), "--backend", "stub",
                 "--allow-stub", "--heads", "bs"], s3=s3)


def test_missing_weights_fail_cleanly(setup, monkeypatch, tmp_path):
    lo, s3 = setup
    monkeypatch.setenv("SORTINGHAT_MORGOTH_CACHE", str(tmp_path / "empty_cache"))
    with pytest.raises(SystemExit) as e:
        xm.main(["features", "--input", str(lo / "keys.csv"), "--out-dir", str(lo / "mg2")], s3=s3)
    assert "weights" in str(e.value)
