"""Backends: stub contract, checkpoint unpickler allow-list, and (when cached) the real network on noise."""
import io
import pickle

import numpy as np
import pytest

from sortinghat.morgoth import heads, model, weights


def test_stub_outputs_match_head_shapes_and_are_probabilities():
    b = model.StubBackend()
    rng = np.random.default_rng(0)
    for h in heads.HEADS.values():
        x = rng.normal(0, 0.3, (5, 19, int(h.window_s * 200))).astype("float32")
        p = b.predict(h, x)
        assert p.shape == (5, h.n_out) and np.all((p >= 0) & (p <= 1))
        if not h.binary:
            assert np.allclose(p.sum(axis=1), 1.0)


def test_input_chans_follow_standard_1020():
    ic = model.input_chans()
    assert ic[0] == 0 and len(ic) == 20 and ic[1] == 1 and ic[2] == 18        # FP1 -> 0 + 1, F3 -> 17 + 1


class _Evil:
    def __reduce__(self):
        return (print, ("pwned",))


def test_restricted_unpickler_rejects_arbitrary_globals():
    blob = pickle.dumps({"x": _Evil()})
    with pytest.raises(pickle.UnpicklingError):
        model._RestrictedUnpickler(io.BytesIO(blob)).load()


def test_restricted_unpickler_accepts_namespace_and_ordered_dict():
    import argparse
    from collections import OrderedDict
    blob = pickle.dumps({"args": argparse.Namespace(a=1), "m": OrderedDict(b=2)})
    out = model._RestrictedUnpickler(io.BytesIO(blob)).load()
    assert out["args"].a == 1 and out["m"]["b"] == 2


def test_auto_backend_never_degrades_to_stub(tmp_path):
    with pytest.raises(FileNotFoundError):
        model.make_backend("auto", cache=tmp_path)
    assert model.make_backend("stub").name == "stub"
    with pytest.raises(ValueError):
        model.make_backend("nope")



def _real_available():
    try:
        import timm, einops, torch  # noqa: F401
    except Exception:
        return False
    return weights.available(("normal", "slowing"))


@pytest.mark.skipif(not _real_available(), reason="torch/timm or cached MORGOTH weights not available")
def test_real_network_runs_on_noise_cpu():
    b = model.make_backend("morgoth", heads=("normal", "slowing"))
    for name in ("normal", "slowing"):
        h = heads.HEADS[name]
        x = np.random.default_rng(0).normal(0, 0.3, (3, 19, 2000)).astype("float32")
        p = b.predict(h, x)
        assert p.shape == (3, h.n_out) and np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1))
