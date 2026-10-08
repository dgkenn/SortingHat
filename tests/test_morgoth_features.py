"""MORGOTH features with the STUB backend on SYNTHETIC EEG: windows, QC gating, missing channels, determinism."""
import numpy as np
import pytest

from sortinghat.eeg.io import CANONICAL_19, Recording
from sortinghat.eeg.synthetic import generate_eeg
from sortinghat.eeg.window import all_windows
from sortinghat.morgoth import features as F
from sortinghat.morgoth import heads
from sortinghat.morgoth.model import StubBackend


def rec_of(x, names=CANONICAL_19, fs=200.0):
    return Recording(x, fs, list(names), 0.0, {"edf_duration_s": x.shape[1] / fs})


@pytest.fixture(scope="module")
def result():
    rec = rec_of(generate_eeg(700, background="normal", seed=1))
    return F.process_morgoth(rec, None, StubBackend(), F.MorgothConfig(step_s=10.0))


def test_one_row_per_window_with_all_feature_columns(result):
    rows, info = result
    assert [r["window"] for r in rows] == list(all_windows())
    cols = heads.feature_names()
    assert len(cols) == len(set(cols)) == 12 * 3
    assert all(c in rows[0] for c in cols)
    assert info["n_missing_channels"] == 0 and info["ran"]


def test_probabilities_and_burden_are_in_range(result):
    rows, _ = result
    for r in rows:
        assert r["qc_pass"]
        for c in heads.feature_names():
            assert 0.0 <= r[c] <= 1.0, c
        assert 0.0 <= r["morgoth_valid_fraction"] <= 1.0


def test_multiclass_probabilities_sum_below_one(result):
    r = result[0][0]
    assert sum(r[f"morgoth.slowing.{c}.mean"] for c in ("focal", "generalized")) <= 1.0 + 1e-9
    assert sum(r[f"morgoth.iiic.{c}.mean"] for c in ("seizure", "lpd", "gpd", "lrda", "grda")) <= 1.0 + 1e-9


def test_nested_windows_use_only_their_snippets():
    class Spy(StubBackend):
        calls = []

        def predict(self, head, x):
            Spy.calls.append((head.name, len(x)))
            return super().predict(head, x)

    rec = rec_of(generate_eeg(700, seed=2))
    rows, _ = F.process_morgoth(rec, None, Spy(), F.MorgothConfig(heads=("normal",), step_s=10.0))
    assert Spy.calls == [("normal", 60)]                                # one pass over the primary window serves all
    by = {r["window"]: r for r in rows}
    only, _ = F.process_morgoth(rec, {"20s": all_windows()["20s"]}, Spy(), F.MorgothConfig(heads=("normal",), step_s=10.0))
    assert Spy.calls[-1] == ("normal", 2)                               # the 20-s window alone: snippets at 60 s and 70 s
    for stat in ("mean", "p90", "burden"):
        c = f"morgoth.normal.abnormal.{stat}"
        assert only[0][c] == pytest.approx(by["20s"][c])                # same snippets, same numbers as in the full pass
    assert by["20s"]["morgoth.normal.abnormal.mean"] != by["primary"]["morgoth.normal.abnormal.mean"]


def test_deterministic():
    rec = rec_of(generate_eeg(700, seed=3))
    a, _ = F.process_morgoth(rec, None, StubBackend(), F.MorgothConfig(heads=("bs", "iiic"), step_s=20.0))
    b, _ = F.process_morgoth(rec, None, StubBackend(), F.MorgothConfig(heads=("bs", "iiic"), step_s=20.0))
    assert a == b


def test_qc_failing_window_gets_nan_and_skips_model():
    class Boom(StubBackend):
        def predict(self, head, x):
            raise AssertionError("model must not run when no window passes QC")

    rec = rec_of(np.zeros((19, 700 * 200)) + np.random.default_rng(0).normal(0, 0.01, (19, 700 * 200)))   # near-flat
    rows, info = F.process_morgoth(rec, None, Boom(), F.MorgothConfig(heads=("normal",)))
    assert not info["ran"]
    assert all(not r["qc_pass"] and np.isnan(r["morgoth.normal.abnormal.mean"]) for r in rows)


def test_missing_channels_counted_and_still_scored():
    drop = {"O1", "Pz", "Cz"}
    keep = [i for i, c in enumerate(CANONICAL_19) if c not in drop]
    x = generate_eeg(700, seed=4)[keep]
    rows, info = F.process_morgoth(rec_of(x, names=[CANONICAL_19[i] for i in keep]), None, StubBackend(),
                                   F.MorgothConfig(heads=("normal",), step_s=30.0))
    assert info["n_missing_channels"] == 3 and sorted(info["missing"]) == ["CZ", "O1", "PZ"]
    assert rows[0]["n_missing_channels"] == 3
    assert np.isfinite(rows[0]["morgoth.normal.abnormal.mean"])


def test_burst_suppression_background_scores_differ_from_normal_for_stub():
    kw = dict(heads=("bs",), step_s=10.0)
    a, _ = F.process_morgoth(rec_of(generate_eeg(700, background="normal", seed=5)), None, StubBackend(), F.MorgothConfig(**kw))
    b, _ = F.process_morgoth(rec_of(generate_eeg(700, background="burst_suppression", seed=5)), None, StubBackend(), F.MorgothConfig(**kw))
    assert a[0]["morgoth.bs.burst_suppression.mean"] != b[0]["morgoth.bs.burst_suppression.mean"]


def test_spike_head_uses_one_second_snippets():
    seen = {}

    class Spy(StubBackend):
        def predict(self, head, x):
            seen[head.name] = x.shape
            return super().predict(head, x)

    F.process_morgoth(rec_of(generate_eeg(700, seed=6)), {"primary": all_windows()["20s"]}, Spy(),
                      F.MorgothConfig(heads=("spikes", "normal"), step_s=10.0, spike_step_s=1.0))
    assert seen["spikes"][1:] == (19, 200) and seen["normal"][1:] == (19, 2000)
