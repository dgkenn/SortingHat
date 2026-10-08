"""Frozen-CBraMod embedding front end on SYNTHETIC EEG only (no real data, no network).

The model-dependent tests use a seeded, untrained 2-layer CBraMod (``Embedder.random_init``) so they run without the
public checkpoint; the tests that need the real checkpoint / upstream clone skip when those are not cached."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sortinghat.eeg.io import CANONICAL_19  # noqa: E402
from sortinghat.eeg.synthetic import generate_eeg, write_edf  # noqa: E402
from sortinghat.eeg.window import EpochFlags, FLAG_NAMES, WindowSpec, all_windows  # noqa: E402
from sortinghat.embed import cbramod as cb  # noqa: E402
from sortinghat.embed import weights as W  # noqa: E402


@pytest.fixture(scope="module")
def embedder():
    return cb.Embedder.random_init(seed=0, n_layer=2, num_threads=2)


@pytest.fixture(scope="module")
def edf(tmp_path_factory):
    d = tmp_path_factory.mktemp("edf")
    return write_edf(d / "a.edf", generate_eeg(700, background="normal", seed=1), 200, CANONICAL_19)


def by_window(res):
    return {r["window"]: r for r in res.rows}


def vec(row, prefix=cb.MEAN_PREFIX):
    return np.array([row[c] for c in cb.emb_columns(prefix)], np.float32)


# ------------------------------------------------------------------------------------------------ conformance
def test_channel_order_is_the_tueg_order_and_a_permutation_of_canonical_19():
    assert cb.CBRAMOD_CHANNELS[:4] == ("Fp1", "Fp2", "F3", "F4") and cb.CBRAMOD_CHANNELS[-3:] == ("Fz", "Cz", "Pz")
    assert sorted(cb.CBRAMOD_CHANNELS) == sorted(CANONICAL_19) and len(cb.CBRAMOD_CHANNELS) == 19
    assert tuple(cb.CBRAMOD_CHANNELS) != tuple(CANONICAL_19)            # the repo's order differs: the reorder matters


def _flags(names, n_epochs, bad=()):
    flags = {k: np.zeros((len(names), n_epochs), bool) for k in FLAG_NAMES}
    for ch, e in bad:
        flags["flat"][names.index(ch), e] = True
    return EpochFlags(list(names), 0.0, 2.0, flags, n_epochs)


def test_segments_are_reordered_scaled_to_100uV_units_and_zero_filled():
    names = [c for c in CANONICAL_19 if c != "Cz"]                       # Cz absent
    rng = np.random.default_rng(0)
    x = rng.normal(0, 20.0, (len(names), 200 * 40))                      # uV
    wins = {"primary": WindowSpec("primary", 0.0, 40.0), "w20": WindowSpec("w20", 0.0, 20.0)}
    ef = _flags(names, 20, bad=[("Fp1", 3)])                             # epoch 3 = patches 6, 7 = segment 0, patches 6-7
    cfg = cb.EmbedConfig()
    p = cb.prepare_segments(x, 200.0, names, 0.0, ef, wins, cfg)
    assert p.tokens.shape == (4, 19, 10, 200) and p.tokens.dtype == np.float32 and p.valid.shape == (4, 19, 10)
    r = cb.CBRAMOD_CHANNELS.index("F7")                                  # F7: row 10 in CBraMod order, row 2 in ours
    np.testing.assert_allclose(p.tokens[1, r].reshape(-1), x[names.index("F7"), 2000:4000] / 100.0, rtol=1e-6)
    cz = cb.CBRAMOD_CHANNELS.index("Cz")
    assert not p.valid[:, cz].any() and not p.tokens[:, cz].any() and p.n_absent_channels == 1
    fp1 = cb.CBRAMOD_CHANNELS.index("Fp1")
    assert not p.valid[0, fp1, 6:8].any() and p.valid[0, fp1, :6].all() and p.valid[0, fp1, 8:].all()
    assert not p.tokens[0, fp1, 6:8].any()                               # zero-filled where the QC flagged the epoch
    assert p.window_segments["w20"] == slice(0, 2) and p.window_segments["primary"] == slice(0, 4)
    assert p.seg_ok.all() and abs(p.seg_valid_frac[1] - 18 / 19) < 1e-9


def test_amplitude_policy_marks_segments_above_100uV():
    names = list(CANONICAL_19)
    x = np.zeros((19, 200 * 20))
    x[:, 200 * 10:] = 150.0                                              # second segment above 100 uV
    wins = {"primary": WindowSpec("primary", 0.0, 20.0)}
    ef = _flags(names, 10)
    keep = cb.prepare_segments(x, 200.0, names, 0.0, ef, wins, cb.EmbedConfig(amp_policy="keep"))
    drop = cb.prepare_segments(x, 200.0, names, 0.0, ef, wins, cb.EmbedConfig(amp_policy="drop"))
    assert keep.seg_over_amp.tolist() == [False, True] and keep.seg_ok.tolist() == [True, True]
    assert drop.seg_ok.tolist() == [True, False]


def test_window_grid_must_align_with_segments():
    with pytest.raises(ValueError):
        cb.prepare_segments(np.zeros((19, 4000)), 200.0, list(CANONICAL_19), 0.0, _flags(list(CANONICAL_19), 10),
                            {"w": WindowSpec("w", 3.0, 17.0)}, cb.EmbedConfig())


def test_config_validation():
    for bad in ({"reference": "x"}, {"amp_policy": "x"}, {"segment_s": 7.5}, {"min_valid_frac": 0.0}):
        with pytest.raises(ValueError):
            cb.EmbedConfig(**bad)


# ------------------------------------------------------------------------------------------------ embeddings
def test_embeddings_are_deterministic_float32_200d_and_nested_windows_pool_the_same_segments(edf, embedder):
    a, b = by_window(cb.embed_recording(edf, embedder)), by_window(cb.embed_recording(edf, embedder))
    assert set(a) == set(all_windows())
    for w in a:
        v = vec(a[w])
        assert v.shape == (200,) and v.dtype == np.float32 and np.isfinite(v).all()
        np.testing.assert_array_equal(v, vec(b[w]))                      # deterministic
        assert a[w]["emb_ok"] and a[w]["qc_pass"] and a[w]["emb_n_used"] == a[w]["emb_n_segments"]
    assert [a[w]["emb_n_segments"] for w in ("20s", "1min", "2min", "5min", "10min", "primary")] == [2, 6, 12, 30, 60, 60]
    np.testing.assert_array_equal(vec(a["primary"]), vec(a["10min"]))    # primary == nested 10-min window by construction
    assert not np.allclose(vec(a["20s"]), vec(a["10min"]))
    assert a["primary"]["emb_valid_token_frac"] == 1.0 and a["primary"]["emb_n_absent_channels"] == 0
    assert a["primary"]["emb_frac_over_amp"] == 0.0


def test_embedding_is_batch_size_invariant(edf):
    e1 = cb.Embedder.random_init(seed=0, n_layer=2, batch_size=7, num_threads=2)
    e2 = cb.Embedder.random_init(seed=0, n_layer=2, batch_size=60, num_threads=2)
    a = by_window(cb.embed_recording(edf, e1))["primary"]
    b = by_window(cb.embed_recording(edf, e2))["primary"]
    np.testing.assert_allclose(vec(a), vec(b), atol=2e-5)


def test_different_signal_gives_different_embedding_and_reference_matters(tmp_path, edf, embedder):
    other = write_edf(tmp_path / "b.edf", generate_eeg(700, background="slowing", seed=5), 200, CANONICAL_19)
    a = vec(by_window(cb.embed_recording(edf, embedder))["primary"])
    b = vec(by_window(cb.embed_recording(other, embedder))["primary"])
    c = vec(by_window(cb.embed_recording(edf, embedder, cfg=cb.EmbedConfig(reference="as_recorded")))["primary"])
    assert not np.allclose(a, b) and not np.allclose(a, c)


def test_missing_and_dead_channels_are_masked_not_embedded_as_data(tmp_path, embedder):
    names = [c for c in CANONICAL_19 if c not in ("Cz", "Pz")]           # two absent
    x = generate_eeg(700, background="normal", seed=2, ch_names=names)
    x[names.index("O1")] = 7.0                                           # a dead (exactly constant) channel
    p = write_edf(tmp_path / "c.edf", x, 200, names)
    res = cb.embed_recording(p, embedder)
    row = by_window(res)["primary"]
    assert row["emb_ok"] and row["emb_n_absent_channels"] == 3           # Cz, Pz missing; O1 dead -> removed
    assert res.channel_status["dead"] == ["O1"] and res.channel_status["n_dead_min"] == 1
    assert abs(row["emb_valid_token_frac"] - 16 / 19) < 1e-9 and np.isfinite(vec(row)).all()


def test_amp_policy_drop_removes_high_amplitude_segments(tmp_path, embedder):
    x = generate_eeg(700, background="normal", seed=3) * 12.0            # far above 100 uV
    p = write_edf(tmp_path / "d.edf", x, 200, CANONICAL_19, phys_range_uv=30000.0)
    keep = by_window(cb.embed_recording(p, embedder))["primary"]
    drop = by_window(cb.embed_recording(p, embedder, cfg=cb.EmbedConfig(amp_policy="drop")))["primary"]
    assert keep["emb_ok"] and keep["emb_frac_over_amp"] > 0.9
    assert not drop["emb_ok"] and drop["emb_n_used"] == 0 and np.isnan(drop["emb.cbramod.0"])


def test_windows_failing_qc_are_not_embedded_unless_requested(tmp_path, embedder):
    x = generate_eeg(700, background="normal", seed=4)
    for t in range(60, 660, 40):                                         # 18 s of every 40 s flat on all channels: 45 % unusable
        x[:, 200 * t:200 * (t + 18)] = np.random.default_rng(t).normal(0, 0.05, (19, 200 * 18))
    p = write_edf(tmp_path / "e.edf", x, 200, CANONICAL_19)
    res = cb.embed_recording(p, embedder)
    prim = by_window(res)["primary"]
    assert not prim["qc_pass"] and not prim["emb_ok"] and res.n_encoded_segments == 0
    forced = cb.embed_recording(p, embedder, cfg=cb.EmbedConfig(compute_failed=True))
    f = by_window(forced)["primary"]
    assert f["emb_ok"] and not f["qc_pass"] and forced.n_encoded_segments > 0
    assert 0 < f["emb_n_used"] < f["emb_n_segments"]                     # fully masked segments never enter the pool


def test_no_usable_channel_raises(tmp_path, embedder):
    p = write_edf(tmp_path / "f.edf", np.full((19, 200 * 700), 3.0), 200, CANONICAL_19)
    with pytest.raises(ValueError):
        cb.embed_recording(p, embedder)


def test_attention_pooling_columns_and_properties(edf):
    e = cb.Embedder.random_init(seed=0, n_layer=2, num_threads=2)
    row = by_window(cb.embed_recording(edf, e, cfg=cb.EmbedConfig(attention=True)))["primary"]
    m, a = vec(row), vec(row, cb.ATTN_PREFIX)
    assert a.shape == (200,) and np.isfinite(a).all() and not np.allclose(m, a)
    E = np.tile(np.arange(5.0, dtype=np.float32), (4, 1))               # identical segments: attention == mean
    mean, attn = cb.pool_window(E, True, 0.1)
    np.testing.assert_allclose(mean, attn, rtol=1e-6)
    E[0] += 3.0
    mean, attn = cb.pool_window(E, True, 0.1)
    assert np.linalg.norm(attn - E[1]) < np.linalg.norm(mean - E[1])     # attention down-weights the outlier segment


# ------------------------------------------------------------------------------------------------ real checkpoint
needs_ckpt = pytest.mark.skipif(not W.default_checkpoint().is_file(), reason="public CBraMod checkpoint not cached")


@needs_ckpt
def test_checkpoint_hash_and_real_weights_embedding(edf):
    assert W.verify_checkpoint(W.default_checkpoint()).startswith("0792cb80")
    e = cb.Embedder.from_checkpoint(num_threads=2)
    a = vec(by_window(cb.embed_recording(edf, e))["20s"])
    b = vec(by_window(cb.embed_recording(edf, e))["20s"])
    np.testing.assert_array_equal(a, b)
    assert np.isfinite(a).all() and a.std() > 0


def test_tampered_checkpoint_is_refused(tmp_path):
    bad = tmp_path / "pretrained_weights.pth"
    bad.write_bytes(b"not a checkpoint")
    with pytest.raises(W.WeightsError):
        W.verify_checkpoint(bad)
    with pytest.raises(W.WeightsError):
        W.ensure_checkpoint(tmp_path / "missing.pth", download=True)    # an explicit path is never downloaded to


UPSTREAM = W.cache_dir() / "code"


@needs_ckpt
@pytest.mark.skipif(not (UPSTREAM / "models" / "cbramod.py").is_file(), reason="upstream clone not cached")
def test_vendored_model_matches_upstream(monkeypatch):
    import subprocess
    import sys
    code = (
        "import sys, torch; sys.path.insert(0, %r)\n"
        "from models.cbramod import CBraMod as Up\n"
        "from sortinghat.embed.cbramod_model import CBraMod\n"
        "sd = torch.load(%r, map_location='cpu', weights_only=True)\n"
        "u, v = Up(), CBraMod(); u.load_state_dict(sd); v.load_state_dict(sd)\n"
        "assert list(u.state_dict()) == list(v.state_dict())\n"
        "u.eval(); v.eval(); torch.manual_seed(0); x = torch.randn(3, 19, 10, 200)\n"
        "with torch.inference_mode(): d = (u(x) - v(x)).abs().max().item()\n"
        "print(d)\n" % (str(UPSTREAM), str(W.default_checkpoint())))
    root = str(__import__("pathlib").Path(__file__).resolve().parents[1])
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=root,
                         env={**__import__("os").environ, "PYTHONPATH": root}, timeout=300)
    assert out.returncode == 0, out.stderr[-500:]
    assert float(out.stdout.strip().splitlines()[-1]) == 0.0
