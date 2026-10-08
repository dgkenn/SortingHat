"""Locate, verify and load the public CBraMod checkpoint (no patient data involved; plain network access is fine).

The checkpoint is NOT in the repository. It lives in a cache directory outside the repo
(``~/.cache/sortinghat/cbramod`` by default, override with ``SORTINGHAT_CBRAMOD_DIR`` or ``--weights``), is verified
against the SHA-256 recorded in ``docs/research/cbramod_provenance.md`` on every load, and is read with
``torch.load(weights_only=True)`` so a tampered pickle cannot execute code.

    python -m sortinghat.embed.weights --fetch      # download if absent, verify, print size and hash prefix
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

CHECKPOINT_NAME = "pretrained_weights.pth"
CHECKPOINT_URL = "https://huggingface.co/weighting666/CBraMod/resolve/main/pretrained_weights.pth"
CHECKPOINT_SHA256 = "0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178"
CHECKPOINT_BYTES = 19_775_842
HF_REVISION = "500543c7e30bda1b22bfd51a49301b238dee21fd"
UPSTREAM_REPO = "https://github.com/wjq-learning/CBraMod"
UPSTREAM_COMMIT = "b9e961003214326972c567eff390e75b0287e32a"
ENV_DIR = "SORTINGHAT_CBRAMOD_DIR"


class WeightsError(RuntimeError):
    pass


def cache_dir() -> Path:
    return Path(os.environ.get(ENV_DIR) or Path.home() / ".cache" / "sortinghat" / "cbramod")


def default_checkpoint() -> Path:
    return cache_dir() / CHECKPOINT_NAME


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_checkpoint(path) -> str:
    """Full SHA-256 of ``path``; raises ``WeightsError`` unless it equals the recorded hash (prefix 0792cb80)."""
    p = Path(path)
    if not p.is_file():
        raise WeightsError("checkpoint file not found (run `python -m sortinghat.embed.weights --fetch`)")
    got = sha256_file(p)
    if got != CHECKPOINT_SHA256:
        raise WeightsError(f"checkpoint SHA-256 prefix {got[:8]} does not match the recorded {CHECKPOINT_SHA256[:8]}")
    return got


def ensure_checkpoint(path=None, download: bool = True) -> Path:
    """Return a verified checkpoint path, downloading it into the cache when absent (``download=True``)."""
    p = Path(path) if path else default_checkpoint()
    if not p.is_file():
        if not download or path:
            raise WeightsError("checkpoint file not found (run `python -m sortinghat.embed.weights --fetch`)")
        p.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=p.parent, delete=False, suffix=".part") as tmp:
            try:
                with urllib.request.urlopen(CHECKPOINT_URL) as resp:      # noqa: S310 - fixed https URL
                    shutil.copyfileobj(resp, tmp)
            except Exception:
                Path(tmp.name).unlink(missing_ok=True)
                raise
        try:
            verify_checkpoint(tmp.name)
        except WeightsError:
            Path(tmp.name).unlink(missing_ok=True)
            raise
        os.replace(tmp.name, p)
    verify_checkpoint(p)
    return p


def load_state_dict(path=None) -> dict:
    import torch
    p = ensure_checkpoint(path, download=False)
    return torch.load(p, map_location="cpu", weights_only=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch", action="store_true", help="download into the cache if absent, then verify")
    ap.add_argument("--path", default=None)
    a = ap.parse_args(argv)
    try:
        p = ensure_checkpoint(a.path, download=a.fetch)
    except WeightsError as e:
        print(f"weights: {e}", file=sys.stderr)
        return 1
    print(f"checkpoint ok: {p.stat().st_size} bytes, sha256 prefix {CHECKPOINT_SHA256[:8]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
