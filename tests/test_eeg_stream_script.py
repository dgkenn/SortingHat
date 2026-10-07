"""scripts/extract_eeg_features.py end to end against a fake S3 over SYNTHETIC EDFs."""
import importlib.util
import re
from pathlib import Path

import pandas as pd
import pytest

from sortinghat.eeg.io import CANONICAL_19
from sortinghat.eeg.stream import FailureReason
from sortinghat.eeg.synthetic import generate_eeg, write_edf
from test_eeg_stream import BUCKET, FakeS3  # noqa: F401  (tests/ is on sys.path under pytest)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "extract_eeg_features.py"
spec = importlib.util.spec_from_file_location("extract_eeg_features", SCRIPT)
xef = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xef)


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
    rc = xef.main(["--input", str(lo / "keys.csv"), "--out-dir", str(lo / "features"), "--flush-every", "3",
                   "--windows", "20s", *extra], s3=s3)
    return rc, capsys.readouterr().out


def test_end_to_end_aggregate_only_output(setup, capsys):
    lo, s3 = setup
    rc, out = run(lo, s3, capsys)
    assert rc == 0
    assert "attempted: <11" in out and "sub-" not in out and "SYN" not in out and "ses-" not in out
    assert "failure not_found: <11" in out
    assert "mean bytes fetched/recording" in out
    parts = list((lo / "features").glob("part-*.parquet"))
    assert parts and all(oct(p.stat().st_mode & 0o777) == "0o600" for p in parts)
    df = pd.concat([pd.read_parquet(p) for p in parts])
    assert df["recording_id"].nunique() == 6 and set(df["window"]) == {"20s"}
    assert df["recording_id"].str.startswith("rec").all()
    assert "qeeg.global.delta_abs_log10" in df.columns and "qc_flag_flat" in df.columns
    assert not any(str(c).lower().startswith("key") for c in df.columns)
    assert df["qc_pass"].all()
    # no raw signal files anywhere
    assert not list(lo.rglob("*.edf"))


def test_resume_skips_done_and_permanent_failures(setup, capsys):
    lo, s3 = setup
    run(lo, s3, capsys)
    n_calls = len(s3.calls)
    rc, out = run(lo, s3, capsys)
    assert rc == 0 and len(s3.calls) == n_calls                       # nothing refetched
    assert "attempted: <11" in out
    rc, out = run(lo, s3, capsys, "--retry-permanent")
    assert len(s3.calls) == n_calls                                   # the missing key fails again, 0 bytes served


def test_transient_failure_is_retried_on_resume(setup, capsys):
    lo, s3 = setup
    s3.faults = [__import__("test_eeg_stream").FakeClientError("SlowDown", 503)] * 100
    xef_kwargs = ("--max-attempts", "1")
    run(lo, s3, capsys, *xef_kwargs)
    assert not list((lo / "features").glob("part-*.parquet"))
    led = pd.concat([pd.read_csv(p) for p in (lo / "features").glob("ledger-*.csv")])
    assert set(led["reason"].dropna()) == {FailureReason.S3_TRANSIENT}
    s3.faults = []
    run(lo, s3, capsys)
    df = pd.concat([pd.read_parquet(p) for p in (lo / "features").glob("part-*.parquet")])
    assert df["recording_id"].nunique() == 6


def test_sharding_partitions_recordings(setup, capsys):
    lo, s3 = setup
    for k in range(3):
        xef.main(["--input", str(lo / "keys.csv"), "--out-dir", str(lo / "features"), "--shard", str(k),
                  "--of", "3", "--windows", "20s"], s3=s3)
    capsys.readouterr()
    df = pd.concat([pd.read_parquet(p) for p in (lo / "features").glob("part-*.parquet")])
    assert df["recording_id"].nunique() == 6
    assert df.groupby("recording_id")["window"].apply(lambda s: s.is_unique).all()   # no duplicate processing
    ids = [xef.opaque_id(key(i)) for i in range(7)]
    assert sorted(sum(([i for i in ids if xef.shard_of(i, 3) == k] for k in range(3)), [])) == sorted(ids)


def test_input_and_output_must_be_local_only(setup, tmp_path):
    lo, s3 = setup
    bad = tmp_path / "keys.csv"
    bad.write_text("edf_key\nx\n")
    with pytest.raises(SystemExit):
        xef.main(["--input", str(bad), "--out-dir", str(lo / "f")], s3=s3)
    with pytest.raises(SystemExit):
        xef.main(["--input", str(lo / "keys.csv"), "--out-dir", str(tmp_path / "f")], s3=s3)


def test_keys_built_from_bids_columns(tmp_path):
    lo = tmp_path / "local_only"
    lo.mkdir()
    pd.DataFrame({"SiteID": ["S1", "S1"], "BidsFolder": ["sub-S11", "sub-S12"], "SessionID": ["7", "8"],
                  "EEGFolder": ["ceeg_x", None]}).to_csv(lo / "c.csv", index=False)
    recs = xef.load_recordings(lo / "c.csv")
    assert recs[0][1].endswith("sub-S11_ses-7_task-cEEG_eeg.edf") and recs[1][1].endswith("task-EEG_eeg.edf")
    assert all(re.fullmatch(r"rec[0-9a-f]{20}", r) for r, _ in recs)


def test_summary_suppression_has_no_complement_leak():
    s = xef.summarize(20, 19, xef.Counter({"not_found": 1}), [1.0] * 20, [10] * 20, [1.0] * 19, [1.0] * 19, 0,
                      {"primary": [0.9] * 19}, {"primary": [True] * 19}, 20.0, 0)
    assert s["succeeded"] == "<11" and s["failed"] == "<11" and s["success_rate"] == "<11"
    assert s["failure_reasons"] == {"not_found": "<11"}
    assert s["windows"]["primary"]["usable_fraction_quantiles"]["q50"] == 0.9
