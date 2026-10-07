import numpy as np
import pytest

from sortinghat.eeg.io import CANONICAL_19
from sortinghat.eeg.preprocess import (DOUBLE_BANANA, PreprocessConfig, bandpass, bipolar_double_banana,
                                       common_average, common_average_masked, notch, preprocess, resample)
from sortinghat.eeg.synthetic import generate_eeg


def _amp_at(x, fs, f):
    """Amplitude of the sinusoidal component at f (single channel), via DFT on whole cycles."""
    n = len(x)
    t = np.arange(n) / fs
    return 2 * np.abs(np.mean(x * np.exp(-2j * np.pi * f * t)))


def test_resample_preserves_tone_and_length():
    t = np.arange(0, 20, 1 / 500)
    x = 20 * np.sin(2 * np.pi * 10 * t)[None, :]
    y = resample(x, 500, 200)
    assert y.shape[1] == 4000
    assert _amp_at(y[0, 200:-200], 200, 10) == pytest.approx(20, rel=0.01)
    z = resample(np.vstack([x, x]), 256.0, 200.0)
    assert abs(z.shape[1] - x.shape[1] * 200 / 256) <= 1
    assert resample(y, 200, 200) is not y


def test_notch_removes_60hz_keeps_10hz():
    fs = 200
    t = np.arange(0, 30, 1 / fs)
    x = (20 * np.sin(2 * np.pi * 10 * t) + 50 * np.sin(2 * np.pi * 60 * t))[None, :]
    y = notch(x, fs, 60.0)[0, 400:-400]
    assert _amp_at(y, fs, 60) < 0.02 * 50
    assert _amp_at(y, fs, 10) == pytest.approx(20, rel=0.02)


def test_bandpass_default_and_configurable():
    fs = 200
    t = np.arange(0, 60, 1 / fs)
    x = (30 * np.sin(2 * np.pi * 0.1 * t) + 20 * np.sin(2 * np.pi * 10 * t) + 20 * np.sin(2 * np.pi * 80 * t))[None, :]
    y = bandpass(x, fs, (0.5, 45.0))[0, 1000:-1000]
    assert _amp_at(y, fs, 0.1) < 3 and _amp_at(y, fs, 80) < 1 and _amp_at(y, fs, 10) == pytest.approx(20, rel=0.05)
    y2 = bandpass(x, fs, (8.0, 12.0))[0, 1000:-1000]
    assert _amp_at(y2, fs, 10) == pytest.approx(20, rel=0.1) and np.std(y2) < 16


def test_preprocess_pipeline_amplitude_not_normalised():
    x = generate_eeg(60, fs=500.0, background="normal", seed=1)
    y, fs = preprocess(x, 500.0)
    assert fs == 200.0 and y.shape == (19, 12000)
    # uV scale preserved: no per-channel z-scoring
    assert 3 < y[:, 1000:-1000].std() < 30
    assert not np.allclose(y.std(axis=1), 1.0, atol=0.2)
    y2, _ = preprocess(x, 500.0, PreprocessConfig(notch_hz=None, band=None))
    assert y2.shape == y.shape


def test_notch_applied_in_pipeline():
    x = generate_eeg(40, fs=256.0, seed=2)
    x = x + 80 * np.sin(2 * np.pi * 60 * np.arange(x.shape[1]) / 256.0)
    y, fs = preprocess(x, 256.0)
    assert _amp_at(y[0, 800:-800], fs, 60) < 2


def test_common_average_zero_mean_and_exclusion():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((19, 500))
    assert np.allclose(common_average(x).mean(axis=0), 0)
    x[4] += 1000.0                                   # bad channel
    car = common_average(x, exclude=np.arange(19) == 4)
    good = np.delete(car, 4, axis=0)
    assert np.abs(good.mean()) < 1.0                 # bad channel did not contaminate the reference
    assert np.allclose(common_average(x).mean(axis=0), 0)


def test_double_banana():
    assert len(DOUBLE_BANANA) == 18
    rng = np.random.default_rng(1)
    x = rng.standard_normal((19, 100))
    bp, names = bipolar_double_banana(x, CANONICAL_19)
    assert bp.shape == (18, 100) and names[0] == "Fp1-F7" and names[-1] == "Cz-Pz"
    i = {c: k for k, c in enumerate(CANONICAL_19)}
    assert np.allclose(bp[names.index("T3-T5")], x[i["T3"]] - x[i["T5"]])
    # sum around a chain telescopes to first minus last
    chain = [names.index(n) for n in ("Fp1-F7", "F7-T3", "T3-T5", "T5-O1")]
    assert np.allclose(bp[chain].sum(axis=0), x[i["Fp1"]] - x[i["O1"]])
    # missing electrode drops its pairs
    keep = [c for c in CANONICAL_19 if c != "T3"]
    bp2, names2 = bipolar_double_banana(x[[i[c] for c in keep]], keep)
    assert "F7-T3" not in names2 and len(names2) == 16


def test_common_average_masked_is_time_varying():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((19, 400))
    x[4, 100:200] += 1500.0                                  # transient on channel 4 during epoch 1 (100-sample epochs)
    mask = np.ones((19, 4), bool)
    mask[4, 1] = False
    car = common_average_masked(x, mask, 100)
    assert np.abs(np.delete(car, 4, axis=0)[:, 100:200]).max() < 6      # no leakage into the other channels
    naive = common_average(x)
    assert np.abs(np.delete(naive, 4, axis=0)[:, 100:200]).max() > 70   # what plain CAR would do
    assert np.allclose(car[:, :100], naive[:, :100])
