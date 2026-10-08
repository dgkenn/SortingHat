"""scripts/extract_embeddings.py end to end against a fake S3 over SYNTHETIC EDFs (seeded untrained model injected)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from sortinghat.eeg.io import CANONICAL_19  # noqa: E402
from sortinghat.eeg.stream import FailureReason  # noqa: E402
from sortinghat.eeg.synthetic import generate_eeg, write_edf  # noqa: E402
from sortinghat.embed import cbramod as cb  # noqa: E402
from sortinghat.embed.store import load_primary_embeddings  # noqa: E402
from test_eeg_stream import BUCKET, FakeS3  # noqa: E402,F401  (tests/ is on sys.path under pytest)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "extract_embeddings.py"
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("extract_embeddings", SCRIPT)
xem = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xem)


def key(i):
    return f"EEG/bids/SYN/sub-SYN{i:03d}/ses-1/eeg/sub-SYN{i:03d}_ses-1_task-EEG_eeg.edf"


@pytest.fixture(scope="module")
def objects(tmp_path_factory):
    d = tmp_path_factory.mktemp("objs")
    return {key(i): write_edf(d / f"{i}.edf", generate_eeg(700, background=bg, seed=i), 200, CANONICAL_19).read_bytes()
            for i, bg in enumerate(["normal", "slowing", "normal", "low_voltage", "normal", "slowing"])}


@pytest.fixture(scope="module")
def model():
    return cb.Embedder.random_init(seed=0, n_layer=2, num_threads=2)


@pytest.fixture
def setup(tmp_path, monkeypatch, objects):
    import sortinghat.data_io as dio
    monkeypatch.setattr(dio, "access_point", lambda name="credentialed": BUCKET)
    lo = tmp_path / "local_only"
    lo.mkdir()
    pd.DataFrame({"edf_key": [key(i) for i in range(6)] + [key(99)]}).to_csv(lo / "keys.csv", index=False)   # key(99) missing
    return lo, FakeS3(dict(objects))


def run(lo, s3, model, capsys, *extra):
    rc = xem.main(["--input", str(lo / "keys.csv"), "--out-dir", str(lo / "emb"), "--flush-every", "3",
                   "--windows", "primary,20s", *extra], s3=s3, embedder=model)
    return rc, capsys.readouterr().out


def test_end_to_end_local_only_parts_and_aggregate_stdout(setup, model, capsys):
    lo, s3 = setup
    rc, out = run(lo, s3, model, capsys)
    assert rc == 0
    assert "attempted: <11" in out and "failure not_found: <11" in out
    assert "sub-" not in out and "SYN" not in out and "ses-" not in out and "rec" not in out.replace("recording", "")
    assert "CPU throughput" in out and "wall s/recording" in out and "mean embed s" in out
    parts = list((lo / "emb").glob("part-*.parquet"))
    assert parts and all(oct(p.stat().st_mode & 0o777) == "0o600" for p in parts)
    assert oct((lo / "emb").stat().st_mode & 0o777) == "0o700"
    df = pd.concat([pd.read_parquet(p) for p in parts])
    assert df["recording_id"].nunique() == 6 and set(df["window"]) == {"primary", "20s"} and len(df) == 12
    assert df["recording_id"].str.startswith("rec").all() and df["emb_ok"].all() and df["qc_pass"].all()
    emb = df.filter(like="emb.cbramod.")
    assert emb.shape[1] == 200 and (emb.dtypes == np.float32).all() and np.isfinite(emb.to_numpy()).all()
    assert (df["onset_offset_s"] >= 0).all() and not any(str(c).lower().startswith("key") for c in df.columns)
    meta = json.loads((lo / "emb" / "embed_meta.json").read_text())
    assert meta["upstream_commit"] == "b9e961003214326972c567eff390e75b0287e32a" and meta["windows"] == ["20s", "primary"]
    assert not list(lo.rglob("*.edf"))


def test_embeddings_match_a_direct_in_memory_computation(setup, model, capsys, objects):
    import io
    lo, s3 = setup
    run(lo, s3, model, capsys, "--no-onset")
    got, st = load_primary_embeddings(lo / "emb")
    direct = cb.embed_recording(io.BytesIO(objects[key(0)]), model)
    row = next(r for r in direct.rows if r["window"] == "primary")
    want = np.array([row[c] for c in cb.emb_columns()], np.float32)
    have = got.loc[xem.xef.opaque_id(key(0))].filter(like="emb.cbramod.").to_numpy(np.float32)
    np.testing.assert_allclose(have, want, atol=2e-5)
    assert st["n_emb_ok"] == 6


def test_resume_skips_done_and_permanent_failures(setup, model, capsys):
    lo, s3 = setup
    run(lo, s3, model, capsys)
    n_calls = len(s3.calls)
    rc, out = run(lo, s3, model, capsys)
    assert rc == 0 and len(s3.calls) == n_calls and "attempted: <11" in out          # nothing refetched
    run(lo, s3, model, capsys, "--retry-permanent")
    assert len(s3.calls) == n_calls                                                  # the missing key fails again at 0 bytes


def test_transient_failure_is_retried_on_resume(setup, model, capsys):
    lo, s3 = setup
    s3.faults = [__import__("test_eeg_stream").FakeClientError("SlowDown", 503)] * 100
    run(lo, s3, model, capsys, "--max-attempts", "1")
    assert not list((lo / "emb").glob("part-*.parquet"))
    led = pd.concat([pd.read_csv(p) for p in (lo / "emb").glob("ledger-*.csv")])
    assert set(led["reason"].dropna()) == {FailureReason.S3_TRANSIENT}
    s3.faults = []
    run(lo, s3, model, capsys)
    df = pd.concat([pd.read_parquet(p) for p in (lo / "emb").glob("part-*.parquet")])
    assert df["recording_id"].nunique() == 6


def test_sharding_partitions_the_work(setup, model, capsys):
    lo, s3 = setup
    for k in range(2):
        run(lo, s3, model, capsys, "--shard", str(k), "--of", "2")
    df = pd.concat([pd.read_parquet(p) for p in (lo / "emb").glob("part-*.parquet")])
    assert df["recording_id"].nunique() == 6 and not df.duplicated(["recording_id", "window"]).any()


def test_refuses_to_mix_configs_and_non_local_only_paths(setup, model, capsys, tmp_path):
    lo, s3 = setup
    run(lo, s3, model, capsys)
    with pytest.raises(SystemExit):
        run(lo, s3, model, capsys, "--amp-policy", "drop")                       # different config, same directory
    with pytest.raises(SystemExit):
        xem.main(["--input", str(lo / "keys.csv"), "--out-dir", str(tmp_path / "elsewhere")], s3=s3, embedder=model)
    with pytest.raises(SystemExit):
        xem.main(["--input", str(lo / "keys.csv"), "--out-dir", str(lo / "x"), "--weights", str(tmp_path / "nope.pth")], s3=s3)


def test_summary_json_is_aggregate_only(setup, model, capsys, tmp_path):
    lo, s3 = setup
    run(lo, s3, model, capsys, "--summary", str(tmp_path / "summary.json"))
    txt = (tmp_path / "summary.json").read_text()
    assert "rec" not in json.loads(txt).keys() and "SYN" not in txt and "sub-" not in txt
    assert json.loads(txt)["throughput"]["device"] == "cpu"
