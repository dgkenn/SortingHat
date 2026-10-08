"""scripts/extract_rungs.py (one fetch, CBraMod + MORGOTH) against the two separate extractors, on SYNTHETIC EDFs via a fake S3."""
import importlib.util
import json
import stat
import sys
import tracemalloc
from pathlib import Path

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from sortinghat import rungs as rungs_mod  # noqa: E402
from sortinghat.eeg.io import CANONICAL_19  # noqa: E402
from sortinghat.eeg.stream import FailureReason  # noqa: E402
from sortinghat.eeg.synthetic import generate_eeg, write_edf  # noqa: E402
from sortinghat.embed import cbramod as cb  # noqa: E402
from sortinghat.embed.stream import stream_embeddings  # noqa: E402
from sortinghat.morgoth.features import MorgothConfig  # noqa: E402
from sortinghat.morgoth.model import StubBackend  # noqa: E402
from sortinghat.morgoth.stream import stream_morgoth  # noqa: E402
from test_eeg_stream import BUCKET, FakeS3  # noqa: E402,F401  (tests/ is on sys.path under pytest)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


xr, xem, xmg = _load("extract_rungs"), _load("extract_embeddings"), _load("extract_morgoth")
HEADS = "normal,slowing"
COMMON = ["--windows", "primary,20s", "--flush-every", "3"]


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
    pd.DataFrame({"edf_key": [key(i) for i in range(6)] + [key(99)]}).to_csv(lo / "keys.csv", index=False)  # key(99) missing
    return lo, lambda: FakeS3(dict(objects))


def run_rungs(lo, s3, model, *extra, ed="emb", md="mg"):
    return xr.main(["--input", str(lo / "keys.csv"), "--embed-dir", str(lo / ed), "--morgoth-dir", str(lo / md),
                    "--backend", "stub", "--allow-stub", "--heads", HEADS, "--step-s", "30", *COMMON, *extra],
                   s3=s3, embedder=model, backend=StubBackend())


def run_embed(lo, s3, model, *extra, ed="emb"):
    return xem.main(["--input", str(lo / "keys.csv"), "--out-dir", str(lo / ed), *COMMON, *extra], s3=s3, embedder=model)


def run_morgoth(lo, s3, *extra, md="mg"):
    return xmg.main(["features", "--input", str(lo / "keys.csv"), "--out-dir", str(lo / md), "--backend", "stub",
                     "--allow-stub", "--heads", HEADS, "--step-s", "30", *COMMON, *extra], s3=s3)


def parts(d):
    df = pd.concat([pd.read_parquet(p) for p in sorted(d.glob("part-*.parquet"))], ignore_index=True)
    return df.sort_values(["recording_id", "window"]).reset_index(drop=True)


def ledger(d):
    df = pd.concat([pd.read_csv(p) for p in sorted(d.glob("ledger-*.csv"))], ignore_index=True)
    return df.sort_values("recording_id").reset_index(drop=True)


def test_outputs_equal_the_two_separate_extractors_with_one_fetch_per_recording(setup, model, capsys, monkeypatch):
    lo, mk = setup
    s_sep = mk()
    run_embed(lo, s_sep, model, ed="emb_sep")
    n_embed_calls = len(s_sep.calls)
    run_morgoth(lo, s_sep, md="mg_sep")
    assert len(s_sep.calls) == 2 * n_embed_calls                          # the old pair fetches everything twice

    fetches = []
    real = rungs_mod.fetch_window
    monkeypatch.setattr(rungs_mod, "fetch_window", lambda *a, **k: fetches.append(a[1]) or real(*a, **k))
    s_one = mk()
    capsys.readouterr()
    assert run_rungs(lo, s_one, model) == 0
    out = capsys.readouterr().out
    assert sorted(fetches) == sorted(key(i) for i in range(6))              # exactly one window fetch per recording
    assert len(s_one.calls) == n_embed_calls                                # onset search + window: once, not twice
    for i in range(6):
        assert [c for c in s_one.calls if c[0] == key(i)] == [c for c in s_sep.calls[:n_embed_calls] if c[0] == key(i)]

    pd.testing.assert_frame_equal(parts(lo / "emb"), parts(lo / "emb_sep"), check_exact=False, atol=1e-6, rtol=0)
    pd.testing.assert_frame_equal(parts(lo / "mg"), parts(lo / "mg_sep"), check_exact=False, atol=1e-9, rtol=0)
    for a, b in (("emb", "emb_sep"), ("mg", "mg_sep")):
        la, lb = ledger(lo / a), ledger(lo / b)
        assert la[["recording_id", "status", "reason"]].equals(lb[["recording_id", "status", "reason"]])
    assert list(ledger(lo / "emb")["bytes_fetched"]) == list(ledger(lo / "emb_sep")["bytes_fetched"])
    assert (lo / "emb" / "embed_meta.json").read_text() == (lo / "emb_sep" / "embed_meta.json").read_text()
    assert (lo / "mg" / "run_info.json").read_text() == (lo / "mg_sep" / "run_info.json").read_text()
    assert set(ledger(lo / "emb")["reason"].dropna()) == {FailureReason.NOT_FOUND}
    for d in ("emb", "mg"):
        assert stat.S_IMODE((lo / d).stat().st_mode) == 0o700
        assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in (lo / d).iterdir())
    assert "recordings streamed (one fetch each): <11" in out and "failure not_found: <11" in out
    assert "SYN" not in out and "sub-" not in out and "rec0" not in out and "ses-" not in out


def test_old_extractors_resume_after_rungs_without_refetching(setup, model, capsys):
    lo, mk = setup
    run_rungs(lo, mk(), model)
    s3 = mk()
    run_embed(lo, s3, model)
    run_morgoth(lo, s3)
    assert not s3.calls                                                     # everything (and the permanent failure) is done
    assert not parts(lo / "emb").duplicated(["recording_id", "window"]).any()
    assert not parts(lo / "mg").duplicated(["recording_id", "window"]).any()


def test_rungs_resumes_per_rung_after_an_old_extractor(setup, model, capsys):
    lo, mk = setup
    run_embed(lo, mk(), model)
    n_emb_parts = len(list((lo / "emb").glob("part-*.parquet")))
    s3 = mk()
    run_rungs(lo, s3, model)
    assert len(list((lo / "emb").glob("part-*.parquet"))) == n_emb_parts      # CBraMod rung already done: not redone
    assert parts(lo / "mg")["recording_id"].nunique() == 6
    assert {c[0] for c in s3.calls} == {key(i) for i in range(6)}             # the missing key is permanent: not refetched
    n = len(s3.calls)
    run_rungs(lo, s3, model)
    assert len(s3.calls) == n


def test_rungs_subset_limit_and_sharding(setup, model, capsys):
    lo, mk = setup
    run_rungs(lo, mk(), model, "--rungs", "morgoth", "--limit", "2")
    assert parts(lo / "mg")["recording_id"].nunique() == 2 and not (lo / "emb").exists()
    for k in range(2):
        run_rungs(lo, mk(), model, "--shard", str(k), "--of", "2")
    assert parts(lo / "mg")["recording_id"].nunique() == 6 and parts(lo / "emb")["recording_id"].nunique() == 6
    assert not parts(lo / "mg").duplicated(["recording_id", "window"]).any()


def test_transient_failure_is_retried_on_resume(setup, model, capsys):
    lo, mk = setup
    s3 = mk()
    s3.faults = [__import__("test_eeg_stream").FakeClientError("SlowDown", 503)] * 100
    run_rungs(lo, s3, model, "--max-attempts", "1")
    assert not list((lo / "emb").glob("part-*.parquet")) and not list((lo / "mg").glob("part-*.parquet"))
    for d in ("emb", "mg"):
        assert set(ledger(lo / d)["reason"].dropna()) == {FailureReason.S3_TRANSIENT}
    run_rungs(lo, mk(), model)
    assert parts(lo / "emb")["recording_id"].nunique() == 6 and parts(lo / "mg")["recording_id"].nunique() == 6


def test_refuses_non_local_only_paths_and_mixed_configs(setup, model, tmp_path):
    lo, mk = setup
    with pytest.raises(SystemExit):
        xr.main(["--input", str(lo / "keys.csv"), "--embed-dir", str(tmp_path / "plain"), "--morgoth-dir", str(lo / "mg")],
                s3=mk(), embedder=model, backend=StubBackend())
    with pytest.raises(SystemExit):
        xr.main(["--input", str(tmp_path / "keys.csv"), "--embed-dir", str(lo / "e"), "--morgoth-dir", str(lo / "m")],
                s3=mk(), embedder=model, backend=StubBackend())
    run_rungs(lo, mk(), model)
    with pytest.raises(SystemExit):
        run_rungs(lo, mk(), model, "--amp-policy", "drop")
    with pytest.raises(SystemExit):
        run_rungs(lo, mk(), model, "--step-s", "10", "--retry-permanent")


def test_summary_json_is_aggregate_only(setup, model, tmp_path):
    lo, mk = setup
    run_rungs(lo, mk(), model, "--summary", str(tmp_path / "s.json"))
    txt = (tmp_path / "s.json").read_text()
    assert set(json.loads(txt)) >= {"cbramod", "morgoth"} and "SYN" not in txt and "sub-" not in txt and "rec0" not in txt


def test_peak_memory_not_above_larger_separate_extractor(setup, model):
    lo, mk = setup
    ecfg, mcfg = cb.EmbedConfig(), MorgothConfig(heads=("normal", "slowing"), step_s=30.0)

    def peak(fn):
        s3 = mk()
        tracemalloc.start()
        tracemalloc.reset_peak()
        fn(s3)
        p = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return p

    pe = peak(lambda s3: stream_embeddings(s3, key(0), model, cfg=ecfg))
    pm = peak(lambda s3: stream_morgoth(s3, key(0), StubBackend(), cfg=mcfg))
    pc = peak(lambda s3: rungs_mod.stream_rungs(s3, key(0), embedder=model, backend=StubBackend(), embed_cfg=ecfg,
                                                morgoth_cfg=mcfg))
    assert pc <= 1.2 * max(pe, pm)
