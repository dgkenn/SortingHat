"""Inference backends: map preprocessed snippets (S, 19, win) to head probabilities.

* ``StubBackend`` - deterministic, torch-free, spectral-power driven. For tests and for exercising the pipeline on
  synthetic EEG when no weights are present. It is NOT MORGOTH and its outputs mean nothing clinically.
* ``TorchBackend`` - the real MORGOTH checkpoints, CPU first. Imports ``backbone.py`` from the pinned code clone in the
  cache (CC BY-NC 4.0; never vendored), builds ``morgoth_backbone_base`` exactly as ``finetune_classification.get_models``
  does, and loads ``checkpoint['model']`` with a restricted unpickler (the checkpoints carry an ``argparse.Namespace``,
  so ``weights_only=True`` alone refuses them; this allow-list admits only the globals the files actually reference).
"""

from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path
from typing import Protocol

import numpy as np

from . import weights
from .heads import HEADS, MORGOTH_CHANNELS, STANDARD_1020_INDEX, HeadSpec

PATCH = 200          # samples per patch (1 s at 200 Hz)


class Backend(Protocol):
    name: str

    def predict(self, head: HeadSpec, x: np.ndarray) -> np.ndarray:
        """(S, 19, win) float32 in [-1, 1] -> (S, head.n_out) probabilities (sigmoid / softmax already applied)."""


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class StubBackend:
    """Deterministic fake: logits are fixed linear functions of band-power log-ratios and line length. Pure numpy."""
    name = "stub"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def _stats(self, x: np.ndarray) -> np.ndarray:
        n = x.shape[-1]
        f = np.fft.rfftfreq(n, 1 / PATCH)
        p = np.abs(np.fft.rfft(x, axis=-1)) ** 2
        bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 70)]
        bp = np.stack([p[..., (f >= a) & (f < b)].sum(-1) for a, b in bands], axis=-1) + 1e-9   # (S, C, 5)
        rel = np.log(bp / bp.sum(-1, keepdims=True)).mean(axis=1)                                # (S, 5)
        ll = np.log1p(np.abs(np.diff(x, axis=-1)).mean(axis=(1, 2)))[:, None]                    # (S, 1)
        return np.concatenate([rel, ll], axis=1)                                                 # (S, 6)

    def predict(self, head: HeadSpec, x: np.ndarray) -> np.ndarray:
        s = self._stats(x)
        rng = np.random.default_rng([self.seed, sum(map(ord, head.name))])
        w = rng.normal(0, 1.0, (s.shape[1], head.n_out))
        b = rng.normal(0, 0.5, head.n_out)
        z = s @ w + b
        return _sigmoid(z) if head.binary else _softmax(z)


class _RestrictedUnpickler(pickle.Unpickler):
    _ALLOWED = {("argparse", "Namespace"), ("collections", "OrderedDict"), ("numpy", "dtype"),
                ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"), ("_codecs", "encode"),
                ("torch", "FloatStorage"), ("torch", "HalfStorage"), ("torch", "LongStorage"),
                ("torch._utils", "_rebuild_tensor_v2"), ("torch._utils", "_rebuild_parameter")}

    def find_class(self, module, name):
        if (module, name) not in self._ALLOWED:
            raise pickle.UnpicklingError(f"global {module}.{name} is not allowed in a MORGOTH checkpoint")
        if module == "numpy.core.multiarray":
            module = "numpy._core.multiarray"
        return super().find_class(module, name)


class _SafePickle:
    """``pickle_module`` for ``torch.load`` that routes every global through the allow-list."""
    Unpickler = _RestrictedUnpickler
    load = staticmethod(lambda f, **kw: _RestrictedUnpickler(f, **kw).load())
    UnpicklingError = pickle.UnpicklingError
    __name__ = "morgoth_safe_pickle"


def load_checkpoint(path: str | Path):
    import torch
    return torch.load(str(path), map_location="cpu", pickle_module=_SafePickle, weights_only=False)


def import_backbone(code: Path):
    """Import the pinned ``backbone.py`` under a private module name (registers ``morgoth_backbone_base`` with timm)."""
    if "morgoth_backbone" in sys.modules:
        return sys.modules["morgoth_backbone"]
    spec = importlib.util.spec_from_file_location("morgoth_backbone", code / "backbone.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["morgoth_backbone"] = mod
    spec.loader.exec_module(mod)
    return mod


def input_chans() -> list[int]:
    """``utils.morgoth_get_input_chans(MORGOTH_CHANNELS)``: 0 for the cls token, then index + 1 in standard_1020."""
    return [0] + [STANDARD_1020_INDEX[c] + 1 for c in MORGOTH_CHANNELS]


class TorchBackend:
    name = "morgoth"

    def __init__(self, cache: Path | None = None, *, threads: int | None = None, batch_size: int = 64):
        import torch
        if threads:
            torch.set_num_threads(int(threads))
        self.cache = cache
        self.batch_size = batch_size
        self._models: dict[str, object] = {}
        import_backbone(weights.code_dir(cache))
        self._chans = input_chans()

    def _model(self, head: HeadSpec):
        if head.name not in self._models:
            import torch  # noqa: F401
            from timm.models import create_model
            ck = load_checkpoint(weights.weights_dir(self.cache) / head.checkpoint)
            m = create_model("morgoth_backbone_base", pretrained=False, num_classes=head.n_out, drop_rate=0.0,
                             drop_path_rate=0.0, attn_drop_rate=0.0, drop_block_rate=None, use_mean_pooling=True,
                             init_scale=0.001, use_rel_pos_bias=False, use_abs_pos_emb=True, init_values=0.1,
                             qkv_bias=False)
            res = m.load_state_dict(ck["model"], strict=False)
            if res.missing_keys or res.unexpected_keys:
                raise RuntimeError(f"checkpoint {head.checkpoint} does not match the model "
                                   f"({len(res.missing_keys)} missing, {len(res.unexpected_keys)} unexpected keys)")
            self._models[head.name] = m.eval()
        return self._models[head.name]

    def predict(self, head: HeadSpec, x: np.ndarray) -> np.ndarray:
        import torch
        m = self._model(head)
        out = []
        with torch.no_grad():
            for i in range(0, len(x), self.batch_size):
                b = torch.from_numpy(np.ascontiguousarray(x[i:i + self.batch_size]))
                b = b.reshape(b.shape[0], b.shape[1], -1, PATCH)            # B N (A T) -> B N A T
                z = m(b, input_chans=self._chans)
                out.append(torch.sigmoid(z) if head.binary else torch.softmax(z, dim=1))
        return torch.cat(out).float().numpy()


def make_backend(kind: str = "auto", *, cache: Path | None = None, heads=None, threads: int | None = None):
    """``stub`` | ``morgoth`` | ``auto`` (morgoth when code + weights are cached and torch imports, else stub is NOT
    chosen silently: ``auto`` raises so a real-data run never degrades to the stub)."""
    if kind == "stub":
        return StubBackend()
    if kind in ("morgoth", "auto"):
        if not weights.available(heads, cache):
            raise FileNotFoundError("MORGOTH weights/code are not in the cache; run "
                                    "`python3 -m sortinghat.morgoth.weights --fetch` (docs/morgoth.md)")
        return TorchBackend(cache, threads=threads)
    raise ValueError(f"unknown backend {kind!r}")
