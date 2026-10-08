"""MORGOTH input preparation on SYNTHETIC EEG: channel order, filtering, snippet scaling against the reference formula."""
import numpy as np
import pytest

from sortinghat.eeg.io import CANONICAL_19, Recording
from sortinghat.eeg.synthetic import generate_eeg
from sortinghat.morgoth import heads, preprocess as pp


def rec_of(x, fs=200.0, names=CANONICAL_19, offset=0.0):
    return Recording(x, fs, list(names), offset, {})


def test_channel_order_and_missing():
    x = generate_eeg(20, seed=1)
    rec = rec_of(x)
    out, missing = pp.to_morgoth_order(rec)
    assert missing == []
    # MORGOTH order differs from canonical: row 1 is F3, canonical index of F3 is 3
    assert np.allclose(out[1], x[CANONICAL_19.index("F3")])
    assert np.allclose(out[8], x[CANONICAL_19.index("Fz")])
    drop = [i for i, c in enumerate(CANONICAL_19) if c not in ("O1", "Pz")]
    out2, missing2 = pp.to_morgoth_order(rec_of(x[drop], names=[CANONICAL_19[i] for i in drop]))
    assert sorted(missing2) == ["O1", "PZ"]
    assert np.isnan(out2[heads.MORGOTH_CHANNELS.index("O1")]).all()


def test_standard_1020_indices_pinned():
    # the model embeds electrodes by position in utils.standard_1020; pinned values must keep MORGOTH_CHANNELS order
    assert set(heads.STANDARD_1020_INDEX) == set(heads.MORGOTH_CHANNELS)
    assert len(set(heads.STANDARD_1020_INDEX.values())) == 19


def test_filter_removes_line_noise_and_resamples():
    fs = 256.0
    t = np.arange(int(60 * fs)) / fs
    x = np.vstack([np.sin(2 * np.pi * 10 * t) * 20 + np.sin(2 * np.pi * 60 * t) * 50 for _ in range(3)])
    y = pp.filter_series(x, fs)
    assert y.shape[1] == 60 * 200
    f = np.fft.rfftfreq(y.shape[1], 1 / 200)
    p = np.abs(np.fft.rfft(y[0])) ** 2
    assert p[np.argmin(abs(f - 10))] > 1000 * p[np.argmin(abs(f - 60))]


def test_filter_keeps_nan_rows():
    x = generate_eeg(30, seed=2)
    x = x.copy()
    x[3] = np.nan
    y = pp.filter_series(x, 200.0)
    assert np.isnan(y[3]).all() and np.isfinite(y[0]).all()


def test_spike_filter_path_runs_above_128hz():
    x = generate_eeg(20, fs=256.0, seed=3)
    y = pp.filter_series(x, 256.0, spikes=True)
    assert y.shape == (19, 20 * 200) and np.isfinite(y).all()


def test_normalise_matches_reference_formula():
    """Reference: CAR -> clip +-500 -> sklearn MinMaxScaler((-100, 100)) per channel -> / 100."""
    from sklearn.preprocessing import MinMaxScaler
    x = generate_eeg(30, seed=4) * 3
    x[2, 100:200] = 900                                        # exercises the clip
    sn, valid = pp.normalise_snippets(x, np.array([0, 400]), 2000)
    assert valid.all()
    for k, s in enumerate([0, 400]):
        seg = x[:, s:s + 2000]
        ref = np.clip(seg - seg.mean(axis=0, keepdims=True), -500, 500)
        ref = np.vstack([MinMaxScaler((-100, 100)).fit_transform(r.reshape(-1, 1)).ravel() for r in ref]) / 100
        assert np.allclose(sn[k], ref, atol=1e-5)
        assert sn[k].min() >= -1 - 1e-6 and sn[k].max() <= 1 + 1e-6


def test_missing_channel_is_zero_filled_then_scaled_to_minus_one():
    x = generate_eeg(15, seed=5)
    x = x.copy()
    x[4] = np.nan
    sn, valid = pp.normalise_snippets(x, np.array([0]), 2000)
    assert valid[0] and np.allclose(sn[0][4], -1.0)


def test_invalid_snippets():
    flat = np.zeros((19, 2000))
    assert not pp.snippet_valid(flat)
    assert not pp.snippet_valid(np.full((19, 2000), np.nan))
    assert not pp.snippet_valid(np.full((19, 2000), 5000.0))
    assert pp.snippet_valid(generate_eeg(10, seed=6)[:, :2000])
    _, valid = pp.normalise_snippets(np.zeros((19, 2200)), np.array([0, 200]), 2000)
    assert not valid.any()


def test_snippet_starts():
    assert list(pp.snippet_starts(2400, 2000, 200)) == [0, 200, 400]
    assert len(pp.snippet_starts(1999, 2000, 200)) == 0
