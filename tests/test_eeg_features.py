import numpy as np
import pytest

from sortinghat.eeg.features import (FeatureConfig, burst_suppression_ratio, extract_features, feature_names,
                                     lempel_ziv, lz_complexity_norm)
from sortinghat.eeg.io import CANONICAL_19, Recording
from sortinghat.eeg.pipeline import process_recording
from sortinghat.eeg.synthetic import Artifact, apply_artifact, generate_eeg, write_edf
from sortinghat.eeg.window import WindowSpec

WIN = {"w": WindowSpec("w", 10.0, 60.0)}


def _feats(background, seed=0, artifacts=(), lateralized=False, dur=90.0, cfg=None):
    x = generate_eeg(dur, background=background, seed=seed, artifacts=artifacts, lateralized=lateralized)
    rec = Recording(x, 200.0, list(CANONICAL_19), 0.0, {"edf_duration_s": dur})
    res = process_recording(rec, WIN, compute_failed=True, feat_cfg=cfg)
    return res.rows[0], res


@pytest.fixture(scope="module")
def normal():
    return _feats("normal", seed=1)[0]


@pytest.fixture(scope="module")
def slowing():
    return _feats("slowing", seed=2)[0]


@pytest.fixture(scope="module")
def bs():
    return _feats("burst_suppression", seed=3)[0]


def g(row, key):
    return row[f"qeeg.global.{key}"]


def test_names_stable_unique_and_complete(normal, slowing, bs):
    names = feature_names()
    assert len(names) == len(set(names))
    for row in (normal, slowing, bs):
        assert [k for k in row if k not in ("window", "qc_pass", "usable_fraction")] == names
    assert "qeeg.occipital.alpha_rel" in names and "conn.wpli.mean_all.delta" in names
    assert "qeeg.global.bsr" in names and "qeeg.global.sef95" in names
    assert len(feature_names(FeatureConfig(per_channel=True))) > len(names)
    assert not any(n.startswith("conn.") for n in feature_names(FeatureConfig(connectivity=False)))


def test_normal_has_posterior_alpha(normal):
    assert normal["qeeg.occipital.alpha_rel"] > normal["qeeg.frontal.alpha_rel"]
    assert normal["qeeg.occipital.alpha_abs_log10"] > normal["qeeg.frontal.alpha_abs_log10"]
    assert g(normal, "alpha_rel") > g(normal, "delta_rel")
    assert g(normal, "bsr") < 0.02


def test_slowing_raises_delta_lowers_alpha_and_edge(normal, slowing):
    assert g(slowing, "delta_rel") > g(normal, "delta_rel") + 0.3
    assert g(slowing, "delta_abs_log10") > g(normal, "delta_abs_log10")
    assert g(slowing, "alpha_rel") < g(normal, "alpha_rel")
    assert g(slowing, "alpha_delta_log10") < g(normal, "alpha_delta_log10") - 1
    assert g(slowing, "slowing_log10") > g(normal, "slowing_log10")
    assert g(slowing, "sef95") < g(normal, "sef95") - 5
    assert g(slowing, "sef50") < g(normal, "sef50")
    assert g(slowing, "amp_rms") > g(normal, "amp_rms")
    assert g(slowing, "lzc") < g(normal, "lzc")                       # slower, more regular -> lower complexity


def test_relative_powers_sum_to_one(normal, slowing):
    for row in (normal, slowing):
        assert sum(g(row, f"{b}_rel") for b in ("delta", "theta", "alpha", "beta")) == pytest.approx(1.0, abs=1e-6)


def test_burst_suppression_raises_bsr(normal, slowing, bs):
    assert g(bs, "bsr") > 0.3
    assert g(bs, "bsr") > 10 * max(g(normal, "bsr"), g(slowing, "bsr"), 0.005)
    assert g(bs, "amp_kurtosis") > g(normal, "amp_kurtosis") + 3       # bursty = heavy tailed


def test_bsr_function_on_constructed_signal():
    fs = 200
    x = np.zeros(20 * fs)
    rng = np.random.default_rng(0)
    x[:] = rng.standard_normal(x.size) * 1.0                          # 1 uV floor
    x[5 * fs:10 * fs] += rng.standard_normal(5 * fs) * 50              # 5 s burst
    ratio = burst_suppression_ratio(x, fs, None, 5.0)
    assert ratio == pytest.approx(15 / 20, abs=0.05)
    clean = np.ones(x.size, bool)
    clean[:5 * fs] = False                                            # drop 5 s of suppression
    assert burst_suppression_ratio(x, fs, clean, 5.0) == pytest.approx(10 / 15, abs=0.05)
    # brief dips shorter than 0.5 s are not suppression
    y = rng.standard_normal(20 * fs) * 30
    y[3 * fs:3 * fs + 40] = 0.1
    assert burst_suppression_ratio(y, fs, None, 5.0) == 0.0


def test_periodic_discharges_heavy_tails_and_lateralisation(normal):
    gen, _ = _feats("periodic_discharges", seed=4)
    lat, _ = _feats("periodic_discharges", seed=4, lateralized=True)
    assert g(gen, "amp_kurtosis") > g(normal, "amp_kurtosis") + 1
    assert g(gen, "line_length") > g(normal, "line_length") * 0.5
    assert lat["qeeg.asym.delta_absdiff_log10"] > normal["qeeg.asym.delta_absdiff_log10"] + 0.2


def test_lempel_ziv():
    assert lempel_ziv([0] * 100) == 2
    assert lempel_ziv([int(c) for c in "0001101001000101"]) == 6      # Kaspar-Schuster worked example
    rng = np.random.default_rng(0)
    white = rng.standard_normal(2000)
    sine = np.sin(2 * np.pi * np.arange(2000) / 100)
    assert lz_complexity_norm(white) > 0.9
    assert lz_complexity_norm(sine) < 0.2
    assert np.isnan(lz_complexity_norm(np.ones(3)))


def test_connectivity_wpli_detects_lag_and_coherence_detects_sharing():
    rng = np.random.default_rng(5)
    n = 200 * 70
    x = rng.standard_normal((19, n)) * 4
    t = np.arange(n) / 200
    carrier = 40 * np.sin(2 * np.pi * 10 * t + np.cumsum(rng.standard_normal(n)) * 0.02)
    i1, i2 = CANONICAL_19.index("Fp1"), CANONICAL_19.index("Fp2")
    x[i1] += carrier
    x[i2] += np.roll(carrier, 5)                                      # 25 ms lag = quarter cycle at 10 Hz
    f = extract_features(x, 200.0, CANONICAL_19, cfg=FeatureConfig())
    assert f["conn.wpli.Fp1_Fp2.alpha"] > 0.7
    assert f["conn.coh.Fp1_Fp2.alpha"] > 0.5
    assert f["conn.wpli.F3_F4.alpha"] < 0.3 and f["conn.coh.F3_F4.alpha"] < 0.15
    zero = x.copy()
    zero[i2] = x[i1] + rng.standard_normal(n) * 5                     # zero-lag: coherent but wPLI ignores it
    f0 = extract_features(zero, 200.0, CANONICAL_19, cfg=FeatureConfig())
    assert f0["conn.coh.Fp1_Fp2.alpha"] > 0.5 and f0["conn.wpli.Fp1_Fp2.alpha"] < 0.3
    for k, v in f.items():
        if k.startswith("conn."):
            assert 0 <= v <= 1 or np.isnan(v)


def test_artifacts_excluded_from_features(normal):
    arts = [Artifact("extreme", ("O1",), 20.0, 20.0, 1500.0), Artifact("line_noise", ("F3",), 10.0, 60.0, 200.0)]
    row, res = _feats("normal", seed=1, artifacts=arts)
    assert res.qc["w"].passes
    # same underlying EEG: the clean channels' features are unaffected by the artifacts on other channels
    assert abs(row["qeeg.temporal.delta_rel"] - normal["qeeg.temporal.delta_rel"]) < 0.05
    assert row["qeeg.occipital.amp_rms"] < 60                          # O1 transients (1500 uV) were masked


def test_missing_channels_give_nan_but_stable_names():
    x = generate_eeg(90, seed=6)
    keep = [i for i, c in enumerate(CANONICAL_19) if c not in ("O1", "O2")]
    names = [CANONICAL_19[i] for i in keep]
    f = extract_features(x[keep], 200.0, names)
    assert list(f) == feature_names()
    assert np.isnan(f["qeeg.occipital.delta_rel"]) and np.isfinite(f["qeeg.frontal.delta_rel"])


def test_too_little_clean_data_gives_nan():
    x = generate_eeg(20, seed=7)
    mask = np.zeros((19, 10), bool)
    f = extract_features(x, 200.0, CANONICAL_19, mask)
    assert np.isnan(f["qeeg.global.delta_rel"]) and np.isnan(f["qeeg.global.bsr"])


def test_deterministic():
    a = _feats("normal", seed=9, dur=40.0)[0]
    b = _feats("normal", seed=9, dur=40.0)[0]
    assert a == b or all((a[k] == b[k]) or (np.isnan(a[k]) and np.isnan(b[k])) for k in a)


def test_end_to_end_from_edf_with_failed_window_skipped(tmp_path):
    x = generate_eeg(700, fs=250.0, background="slowing", seed=8)
    apply_artifact(x, 250.0, Artifact("line_noise", tuple(CANONICAL_19), 60.0, 400.0), CANONICAL_19)  # 4 of 10 min... >40%
    p = write_edf(tmp_path / "long.edf", x, 250.0, CANONICAL_19, label_fmt="EEG {}-REF")
    res = process_recording(p, windows={"primary": WindowSpec("primary", 60.0, 600.0),
                                        "20s": WindowSpec("20s", 60.0, 20.0)})
    prim, short = res.rows
    assert not res.qc["primary"].passes and set(prim) == {"window", "qc_pass", "usable_fraction"}
    assert not short["qc_pass"]
    ok = process_recording(write_edf(tmp_path / "ok.edf", generate_eeg(700, fs=250.0, background="slowing", seed=8),
                                     250.0, CANONICAL_19),
                           windows={"primary": WindowSpec("primary", 60.0, 600.0)})
    row = ok.rows[0]
    assert row["qc_pass"] and row["qeeg.global.delta_rel"] > 0.7 and len(row) == 3 + len(feature_names())


def test_batch_cli_writes_local_rows_and_only_aggregates(tmp_path, capsys):
    import json
    import os
    import pandas as pd
    from sortinghat.eeg.pipeline import main
    paths = []
    for k in range(2):
        paths.append(write_edf(tmp_path / f"r{k}.edf", generate_eeg(90, seed=20 + k), 200.0, CANONICAL_19))
    man = tmp_path / "man.csv"
    pd.DataFrame({"recording_id": ["recA", "recB"], "edf_path": [str(p) for p in paths]}).to_csv(man, index=False)
    out, qc = tmp_path / "local_only" / "f.csv", tmp_path / "qc.json"
    assert main(["--manifest", str(man), "--out", str(out), "--qc-summary", str(qc)]) == 0
    rows = pd.read_csv(out)
    assert set(rows["recording_id"]) == {"recA", "recB"} and (rows["window"] == "primary").sum() == 2
    assert (os.stat(out).st_mode & 0o777) == 0o600
    printed = capsys.readouterr().out
    assert "recA" not in printed and "recB" not in printed and str(tmp_path) not in printed
    assert json.loads(qc.read_text())["n_recordings"] == "<11"
