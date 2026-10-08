"""Fetch and pin MORGOTH weights (BDSP projects access point) and code (public GitHub), outside the repo.

* Weights: ``morgoth1/models/<head>.pth`` on the BDSP ``projects`` access point (BDSP Credentialed Health Data License
  1.5.0; the credentialed DUA for the "MORGOTH 1.0" project must be active). They are MODEL files, not patient data.
* Code: ``https://github.com/bdsp-core/morgoth`` (CC BY-NC 4.0, commercial use prohibited), pinned to a commit. It is
  cloned into the cache and imported from there; nothing of it is vendored into this repo.
* Cache: ``$SORTINGHAT_MORGOTH_CACHE`` or ``~/.cache/sortinghat/morgoth`` (``weights/``, ``code/``, ``manifest.json``).
  ``manifest.json`` records size and sha256 of every fetched file; ``verify`` re-hashes against it.

Downloading needs the BDSP credentials, so it goes through ``data_io.make_client`` (human-run or via the project lead's
D-118 wrapper): ``scripts/heedb_run.sh python3 -m sortinghat.morgoth.weights --fetch``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from .. import data_io
from .heads import EEG_LEVEL_CHECKPOINTS, HEADS, default_heads

WEIGHTS_PREFIX = "morgoth1/models/"
ACCESS_POINT = "projects"
CODE_REPO = "https://github.com/bdsp-core/morgoth"
CODE_COMMIT = "17eab93182695dd94e5e4cc3b2a6f3af7e32bd41"        # main, fetched 2026-10-08
CODE_LICENCE = "CC BY-NC 4.0"
WEIGHTS_LICENCE = "BDSP Credentialed Health Data License 1.5.0"
CHUNK = 16 << 20


def cache_dir() -> Path:
    return Path(os.environ.get("SORTINGHAT_MORGOTH_CACHE") or "~/.cache/sortinghat/morgoth").expanduser()


def weights_dir(cache: Path | None = None) -> Path:
    return (cache or cache_dir()) / "weights"


def code_dir(cache: Path | None = None) -> Path:
    return (cache or cache_dir()) / "code"


def manifest_path(cache: Path | None = None) -> Path:
    return (cache or cache_dir()) / "manifest.json"


def load_manifest(cache: Path | None = None) -> dict:
    p = manifest_path(cache)
    return json.loads(p.read_text()) if p.exists() else {"weights": {}, "code": {}}


def _save_manifest(m: dict, cache: Path | None = None) -> None:
    p = manifest_path(cache)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(m, indent=2, sort_keys=True))


def checkpoint_files(heads=None, eeg_level: bool = False) -> list[str]:
    files = [HEADS[h].checkpoint for h in (heads or default_heads())]
    return files + (list(EEG_LEVEL_CHECKPOINTS) if eeg_level else [])


def download_file(s3, key: str, dest: Path, *, bucket: str, chunk: int = CHUNK, policy=None) -> tuple[int, str]:
    """Ranged, retried download to ``dest`` (via ``.part``); returns (size, sha256) computed while streaming."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    h, start, total = hashlib.sha256(), 0, None
    with open(tmp, "wb") as fh:
        while True:
            exp = None if total is None else min(chunk, total - start)
            try:
                body, t = data_io.get_bytes(s3, bucket, key, start, start + chunk - 1, expected=exp, policy=policy)
            except Exception as exc:  # noqa: BLE001
                if data_io._is_invalid_range(exc):
                    break
                raise
            total = t if t is not None else total
            fh.write(body)
            h.update(body)
            start += len(body)
            if not body or (total is not None and start >= total) or (total is None and len(body) < chunk):
                break
    os.replace(tmp, dest)
    return start, h.hexdigest()


def fetch_weights(heads=None, *, eeg_level: bool = False, s3=None, cache: Path | None = None, force: bool = False) -> dict:
    """Download the checkpoints of ``heads`` into the cache and record size + sha256. Skips files already present with
    the recorded hash. Returns {file: {size, sha256}} for the requested files."""
    s3 = s3 or data_io.make_client(None)
    bucket = data_io.ap_arn(ACCESS_POINT)
    m = load_manifest(cache)
    out = {}
    for f in checkpoint_files(heads, eeg_level):
        dest = weights_dir(cache) / f
        rec = m["weights"].get(f)
        if not force and rec and dest.exists() and dest.stat().st_size == rec["size"]:
            out[f] = rec
            continue
        size, sha = download_file(s3, WEIGHTS_PREFIX + f, dest, bucket=bucket)
        os.chmod(dest, 0o600)
        rec = {"size": size, "sha256": sha, "key": WEIGHTS_PREFIX + f}
        m["weights"][f] = rec
        _save_manifest(m, cache)
        out[f] = rec
    return out


def fetch_code(cache: Path | None = None, repo: str = CODE_REPO, commit: str = CODE_COMMIT) -> Path:
    """Clone the public repo at the pinned commit into the cache (idempotent). The code is NOT executed here."""
    dest = code_dir(cache)
    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
        subprocess.run(["git", "clone", "--quiet", repo, str(dest)], check=True, env=env)
    subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", commit], check=True)
    head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True).stdout.strip()
    if head != commit:
        raise RuntimeError("code checkout is not at the pinned commit")
    m = load_manifest(cache)
    m["code"] = {"repo": repo, "commit": head, "licence": CODE_LICENCE}
    m["weights_licence"] = WEIGHTS_LICENCE
    _save_manifest(m, cache)
    return dest


def verify(cache: Path | None = None) -> dict:
    """Re-hash every recorded weight file. Returns {file: 'ok' | 'missing' | 'mismatch'}."""
    m = load_manifest(cache)
    out = {}
    for f, rec in m["weights"].items():
        p = weights_dir(cache) / f
        out[f] = "missing" if not p.exists() else ("ok" if data_io.sha256_file(p) == rec["sha256"] else "mismatch")
    return out


def available(heads=None, cache: Path | None = None) -> bool:
    """True when the code and every requested checkpoint are in the cache."""
    return (code_dir(cache) / "backbone.py").exists() and all(
        (weights_dir(cache) / f).exists() for f in checkpoint_files(heads))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch", action="store_true", help="download weights (BDSP credentials) and clone the code")
    ap.add_argument("--heads", default=",".join(default_heads()), help="comma-separated head names")
    ap.add_argument("--eeg-level", action="store_true", help="also fetch the 12 small EEG-level heads")
    ap.add_argument("--no-code", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args(argv)
    heads = [h for h in a.heads.split(",") if h]
    bad = [h for h in heads if h not in HEADS]
    if bad:
        raise SystemExit("unknown head(s): " + ",".join(bad))
    if a.fetch:
        got = fetch_weights(heads, eeg_level=a.eeg_level)
        for f, r in sorted(got.items()):
            print(f"{f} {r['size']} sha256={r['sha256']}")      # model files: names, sizes and hashes only
        if not a.no_code:
            fetch_code()
            print("code pinned at", CODE_COMMIT)
    if a.verify:
        for f, s in sorted(verify().items()):
            print(f, s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
