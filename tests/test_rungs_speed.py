"""Speed-up of the single-pass CBraMod + MORGOTH extractor: outputs equal the previous implementation within float tolerance,
the models run once per recording, and a timing benchmark is printed in the test log (SYNTHETIC EEG only)."""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from legacy_morgoth_reference import normalise_snippets as legacy_normalise  # noqa: E402
from legacy_morgoth_reference import process_morgoth as legacy_process  # noqa: E402
from sortinghat.eeg import prep as prep_mod  # noqa: E402
from sortinghat.eeg.io import CANONICAL_19, Recording  # noqa: E402
from sortinghat.eeg.prep import RecordingPrep  # noqa: E402
from sortinghat.eeg.synthetic import generate_eeg  # noqa: E402
from sortinghat.eeg.window import WindowSpec, all_windows  # noqa: E402
from sortinghat.embed import cbramod as cb  # noqa: E402
from sortinghat.morgoth import features as F  # noqa: E402
from sortinghat.morgoth import heads as H  # noqa: E402
from sortinghat.morgoth import model as M  # noqa: E402
from sortinghat.morgoth import preprocess as pp  # noqa: E402
from sortinghat.timing import StageTimers  # noqa: E402

PLAN = ("normal", "bs", "slowing", "iiic")
ALL6 = ("normal", "bs", "spikes", "slowing", "spikeloc", "iiic")


def rec_of(x, fs=200.0):
    return Recording(x, fs, list(CANONICAL_19), 0.0, {"edf_duration_s": x.shape[1] / fs})


def same_rows(a, b, atol):
    assert [r["window"] for r in a] == [r["window"] for r in b]
    for ra, rb in zip(a, b):
        for k in rb:
            va, vb = ra[k], rb[k]
            if isinstance(vb, float) and np.isnan(vb):
                assert isinstance(va, float) and np.isnan(va), k
            elif isinstance(vb, float):
                assert abs(va - vb) <= atol, (k, va, vb)
            else:
                assert va == vb, k


# ---------------------------------------------------------------------------------------------- default heads
def test_default_heads_are_the_plan_heads_and_columns_are_a_subset_in_order():
    assert H.plan_heads() == PLAN
    assert set(H.default_heads()) - set(H.plan_heads()) == {"spikes", "spikeloc"}      # E3 extras stay fetchable / selectable
    full, plan = H.feature_names(ALL6), H.feature_names(PLAN)
    assert [c for c in full if c in set(plan)] == plan                                  # same names, same relative order
    spec = importlib.util.spec_from_file_location("xr", Path(__file__).resolve().parents[1] / "scripts" / "extract_rungs.py")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    xr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(xr)
    src = (Path(__file__).resolve().parents[1] / "scripts" / "extract_rungs.py").read_text()
    assert "--morgoth-heads" in src and 'dest="heads"' in src and "plan_heads()" in src


# ---------------------------------------------------------------------------------------------- vectorised snippets
def test_vectorised_normalise_equals_the_per_snippet_loop_including_edge_cases():
    rng = np.random.default_rng(0)
    x = generate_eeg(120, seed=3) * 2.0
    x = x.copy()
    x[2, 3000:3400] = 900                                   # clip
    x[4] = np.nan                                           # absent electrode
    x[:, 8000:10000] = 0.0                                  # flat snippet(s): invalid
    x[:, 12000:14000] = 5000.0                              # above 3000 uV: invalid
    x[:, 16000:18000] = rng.normal(0, 0.1, (19, 2000))      # below 2 uV: invalid
    starts = pp.snippet_starts(x.shape[1], 2000, 1000)
    a, va = pp.normalise_snippets(x, starts, 2000, chunk=7)
    b, vb = legacy_normalise(x, starts, 2000)
    assert (va == vb).all() and not va.all() and va.any()
    assert np.abs(a - b).max() < 1e-6
    assert (a[~va] == 0).all()
    a1, v1 = pp.normalise_snippets(x, starts[:5], 200, chunk=2)               # also the 1-s spike geometry
    b1, w1 = legacy_normalise(x, starts[:5], 200)
    assert (v1 == w1).all() and np.abs(a1 - b1).max() < 1e-6


# ---------------------------------------------------------------------------------------------- equal to the old code
@pytest.mark.parametrize("heads", [PLAN, ALL6])
@pytest.mark.parametrize("bg", ["normal", "slowing"])
def test_process_morgoth_equals_previous_implementation_stub(heads, bg):
    rec = rec_of(generate_eeg(700, background=bg, seed=11))
    cfg = F.MorgothConfig(heads=heads, step_s=10.0)
    new, ninfo = F.process_morgoth(rec, None, M.StubBackend(), cfg)
    old, oinfo = legacy_process(rec, None, M.StubBackend(), cfg)
    assert ninfo == oinfo
    same_rows(new, old, atol=1e-9)


def test_plan_head_columns_equal_the_six_head_values():
    """Dropping spikes / spikeloc does not change any value of the heads that remain."""
    rec = rec_of(generate_eeg(700, seed=12))
    full, _ = F.process_morgoth(rec, None, M.StubBackend(), F.MorgothConfig(heads=ALL6, step_s=10.0))
    plan, _ = F.process_morgoth(rec, None, M.StubBackend(), F.MorgothConfig(heads=PLAN, step_s=10.0))
    cols = H.feature_names(PLAN)
    for rf, rp in zip(full, plan):
        assert set(rp) >= set(cols) and all(c in rf for c in cols)
        for c in cols + ["qc_pass", "usable_fraction", "n_missing_channels"]:
            assert (np.isnan(rf[c]) and np.isnan(rp[c])) or rf[c] == rp[c]


class Spy(M.StubBackend):
    def __init__(self):
        super().__init__()
        self.calls = []

    def predict(self, head, x):
        self.calls.append((head.name, len(x)))
        return super().predict(head, x)


def test_each_head_runs_once_per_recording_over_the_primary_window_and_nested_windows_reuse_it():
    rec = rec_of(generate_eeg(700, seed=13))
    spy = Spy()
    rows, _ = F.process_morgoth(rec, None, spy, F.MorgothConfig(heads=PLAN, step_s=5.0))
    assert sorted(n for n, _ in spy.calls) == sorted(PLAN)                    # one call per head, none per nested window
    assert {n: c for n, c in spy.calls} == {n: 119 for n in PLAN}              # (600 - 10) / 5 + 1 snippets of the primary window
    assert [r["window"] for r in rows] == list(all_windows())


# ---------------------------------------------------------------------------------------------- shared preprocessing
@pytest.fixture(scope="module")
def tiny_embedder():
    return cb.Embedder.random_init(seed=0, n_layer=2, num_threads=1)


def test_shared_prep_gives_identical_outputs_and_runs_the_qc_once(monkeypatch, tiny_embedder):
    rec = rec_of(generate_eeg(700, background="slowing", seed=14))
    windows = all_windows()
    calls = []
    real = prep_mod.qc_recording
    monkeypatch.setattr(prep_mod, "qc_recording", lambda *a, **k: calls.append(1) or real(*a, **k))
    cfg = F.MorgothConfig(heads=PLAN, step_s=30.0)
    solo_e = cb.embed_recording(rec, tiny_embedder, windows=windows)
    solo_m, _ = F.process_morgoth(rec, windows, M.StubBackend(), cfg)
    assert len(calls) == 2                                                  # separate calls: one QC each
    calls.clear()
    shared = RecordingPrep(rec, windows)
    both_e = cb.embed_recording(rec, tiny_embedder, windows=windows, shared=shared)
    both_m, _ = F.process_morgoth(rec, windows, M.StubBackend(), cfg, prep=shared)
    assert len(calls) == 1                                                  # shared: once for both rungs
    same_rows(both_m, solo_m, atol=0.0)
    same_rows(both_e.rows, solo_e.rows, atol=0.0)
    assert both_e.channel_status == solo_e.channel_status


def test_incompatible_prep_is_ignored_not_misused(tiny_embedder):
    rec = rec_of(generate_eeg(700, seed=15))
    other = {"primary": WindowSpec("primary", 60.0, 300.0)}
    shared = RecordingPrep(rec, other)
    out, _ = F.process_morgoth(rec, all_windows(), M.StubBackend(), F.MorgothConfig(heads=("normal",), step_s=30.0), prep=shared)
    assert [r["window"] for r in out] == list(all_windows())


def test_stage_timers_cover_the_stages_and_are_aggregate_seconds_only(tiny_embedder):
    t = StageTimers()
    rec = rec_of(generate_eeg(700, seed=16))
    shared = RecordingPrep(rec, all_windows(), timers=t)
    cb.embed_recording(rec, tiny_embedder, shared=shared, timers=t)
    F.process_morgoth(rec, None, M.StubBackend(), F.MorgothConfig(heads=PLAN, step_s=30.0), prep=shared, timers=t)
    assert {"qc", "cbramod.filter", "cbramod.segments", "cbramod.forward", "morgoth.filter", "morgoth.snippets",
            "morgoth.aggregate", *(f"morgoth.model.{h}" for h in PLAN)} <= set(t.s)
    assert all(isinstance(v, float) and v >= 0 for v in t.s.values())


# ---------------------------------------------------------------------------------------------- real weights
def _real_available():
    try:
        import einops, timm  # noqa: F401
        from sortinghat.embed import weights as W
        W.verify_checkpoint(W.default_checkpoint())
    except Exception:
        return False
    return M.weights.available(ALL6)


real = pytest.mark.skipif(not _real_available(), reason="torch/timm or cached CBraMod / MORGOTH weights not available")


@real
def test_real_morgoth_is_batch_size_invariant_and_equals_the_old_batch_of_64():
    x = np.random.default_rng(0).uniform(-1, 1, (20, 19, 2000)).astype("float32")
    big, small = M.TorchBackend(threads=1, batch_size=64), M.TorchBackend(threads=1, batch_size=8)
    for name in ("normal", "slowing"):
        h = H.HEADS[name]
        a, b = big.predict(h, x), small.predict(h, x)
        assert a.shape == b.shape == (20, h.n_out) and a.dtype == b.dtype == np.float32
        assert np.abs(a - b).max() < 1e-5


@real
def test_benchmark_old_vs_new_real_weights_equal_outputs(capsys):
    """Old = separate QC per rung, six heads, 64 snippets per pass, per-snippet loops. New = shared QC, plan heads, 8 per pass.
    CPU seconds (process_time, 1 torch thread) so that other jobs on the machine do not distort the ratio."""
    torch.set_num_threads(1)
    from sortinghat.embed import weights as W
    windows = {"primary": WindowSpec("primary", 60.0, 300.0), "20s": WindowSpec("20s", 60.0, 20.0),
               "1min": WindowSpec("1min", 60.0, 60.0), "2min": WindowSpec("2min", 60.0, 120.0)}
    rec = rec_of(generate_eeg(400, background="slowing", seed=21))
    emb = cb.Embedder.from_checkpoint(W.ensure_checkpoint(None), num_threads=1)
    old_be = M.TorchBackend(threads=1, batch_size=64, heads=ALL6)
    new_be = M.TorchBackend(threads=1, batch_size=M.DEFAULT_BATCH, heads=ALL6)

    def run_old():
        t = time.process_time()
        e = cb.embed_recording(rec, emb, windows=windows)
        m, _ = legacy_process(rec, windows, old_be, F.MorgothConfig(heads=ALL6))
        return time.process_time() - t, e.rows, m

    def run_new(heads, timers=None):
        t = time.process_time()
        sh = RecordingPrep(rec, windows, timers=timers)
        e = cb.embed_recording(rec, emb, windows=windows, shared=sh, timers=timers)
        m, _ = F.process_morgoth(rec, windows, new_be, F.MorgothConfig(heads=heads), prep=sh, timers=timers)
        return time.process_time() - t, e.rows, m

    t_old, e_old, m_old = run_old()
    tm = StageTimers(clock=time.process_time)
    t_new6, e_new, m_new6 = run_new(ALL6)
    t_new4, _, m_new4 = run_new(PLAN, tm)
    # outputs: same six-head columns within float tolerance; plan heads' columns identical to the six-head run's
    same_rows(e_new, e_old, atol=1e-5)
    same_rows(m_new6, m_old, atol=1e-5)
    cols = H.feature_names(PLAN)
    for r4, r6 in zip(m_new4, m_new6):
        for c in cols:
            assert (np.isnan(r4[c]) and np.isnan(r6[c])) or abs(r4[c] - r6[c]) < 1e-5
    with capsys.disabled():
        print(f"\n[benchmark] synthetic 400-s recording, 300-s primary window, 1 CPU thread, CPU seconds")
        print(f"[benchmark] old (6 heads, bs 64, QC twice):        {t_old:6.1f} s")
        print(f"[benchmark] new, same 6 heads, same outputs:       {t_new6:6.1f} s  -> {t_old / t_new6:.2f}x")
        print(f"[benchmark] new, default plan heads (4):           {t_new4:6.1f} s  -> {t_old / t_new4:.2f}x")
        print("[benchmark] new default stage seconds: " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(tm.s.items())))
    assert t_new6 < t_old and t_new4 < t_new6


# ---------------------------------------------------------------------------------------------- the script
def test_script_defaults_to_plan_heads_prints_stage_seconds_and_refuses_a_mixed_head_set(tmp_path, monkeypatch, capsys,
                                                                                         tiny_embedder):
    import pandas as pd
    from test_eeg_stream import BUCKET, FakeS3
    import sortinghat.data_io as dio
    from sortinghat.eeg.synthetic import write_edf
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("extract_rungs_speed", scripts / "extract_rungs.py")
    xr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(xr)
    monkeypatch.setattr(dio, "access_point", lambda name="credentialed": BUCKET)
    blob = write_edf(tmp_path / "a.edf", generate_eeg(700, seed=31), 200, CANONICAL_19).read_bytes()
    keys = [f"EEG/bids/SYN/sub-SYN{i:03d}/ses-1/eeg/sub-SYN{i:03d}_ses-1_task-EEG_eeg.edf" for i in range(12)]
    lo = tmp_path / "local_only"
    lo.mkdir()
    pd.DataFrame({"edf_key": keys}).to_csv(lo / "keys.csv", index=False)
    argv = ["--input", str(lo / "keys.csv"), "--embed-dir", str(lo / "e"), "--morgoth-dir", str(lo / "m"), "--backend", "stub",
            "--allow-stub", "--step-s", "30", "--windows", "primary,20s", "--summary", str(tmp_path / "s.json")]
    assert xr.main(argv, s3=FakeS3({k: blob for k in keys}), embedder=tiny_embedder, backend=M.StubBackend()) == 0
    out = capsys.readouterr().out
    assert "stage s/recording" in out and "morgoth.model.normal" in out and "cbramod.forward" in out
    assert "morgoth.model.spikes" not in out and "SYN" not in out and "sub-" not in out
    cols = set(pd.concat([pd.read_parquet(p) for p in (lo / "m").glob("part-*.parquet")]).columns)
    assert set(H.feature_names(PLAN)) <= cols and not any(".spikes." in c or ".spikeloc." in c for c in cols)
    assert "morgoth.slowing.focal.mean" in cols and "morgoth.iiic.lpd.burden" in cols
    import json
    assert json.loads((lo / "m" / "run_info.json").read_text())["heads"] == list(PLAN)
    pd.DataFrame({"edf_key": keys + [keys[0].replace("SYN000", "SYN999")]}).to_csv(lo / "keys.csv", index=False)   # work to do
    with pytest.raises(SystemExit):                      # a directory never mixes head sets
        xr.main(argv + ["--morgoth-heads", ",".join(ALL6)], s3=FakeS3({}), embedder=tiny_embedder, backend=M.StubBackend())
