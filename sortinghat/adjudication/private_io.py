"""Record-level files for HUMANS only: directories 0700, files 0600, always under a ``local_only/`` directory (CLAUDE.md rule 5).

Nothing written through here is ever printed, returned to a caller's stdout or read by an agent.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def require_local_only(p, what: str) -> Path:
    p = Path(p)
    if "local_only" not in p.resolve().parts:
        raise SystemExit(f"{what} must be under a local_only/ directory (gitignored; CLAUDE.md rule 5)")
    return p


def private_dir(p) -> Path:
    """Create ``p`` (and the missing parents below the first existing one) with mode 0700."""
    p = Path(p)
    chain = [d for d in reversed([p, *p.parents]) if not d.exists()]
    for d in chain:
        d.mkdir(exist_ok=True)
        try:
            os.chmod(d, 0o700)
        except OSError:
            pass
    return p


def write_private(path, content: str | bytes) -> Path:
    """Write ``content`` to ``path`` with mode 0600 (the parent directory is created 0700)."""
    p = Path(path)
    private_dir(p.parent)
    data = content.encode("utf-8") if isinstance(content, str) else content
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    os.chmod(p, 0o600)
    return p


def write_private_csv(path, df: pd.DataFrame) -> Path:
    return write_private(path, df.to_csv(index=False))
