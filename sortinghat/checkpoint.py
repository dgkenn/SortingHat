"""Checkpoint / resume for the long real-data steps (the cloud container restarts about hourly and kills every process).

The EEG extractors resume from ledgers and part files. This module gives every other long step the same property
without changing what it computes: units of work (one parquet ROW GROUP of one OMOP part, one site's session frame, one
classified table, one model fit) are written to a checkpoint directory as they finish, and a relaunched step with the
same arguments finds them and skips the work.

Layout and safety
    <root>/<step>/<key>/                     <root> defaults to ``out/local_only/checkpoints`` (gitignored, never read by an
                                             agent), overridable with env ``SORTINGHAT_CHECKPOINT_ROOT`` or ``root=``
    <key> = sha256 of (step, the step's result-affecting arguments, a fingerprint of its input files / store listing, the
            code version); any change gives another directory, so a stale checkpoint is never reused.
    Directories are mode 0700 and files 0600. The files hold RECORD-LEVEL intermediates (filtered row groups, per-site
    frames, fitted predictions): they are never printed, only counted. Writes are atomic (temporary file in the same
    directory, ``fsync``, ``os.replace``), so a kill mid-write leaves either the old file or none; an unreadable file counts as
    a miss. A checkpoint that holds units is kept after the step ends, so relaunching a finished step is cheap and exact;
    delete the directory, or pass ``--no-resume``, to start over. Directories of other keys of the same step are removed once
    they have been idle for ``prune_idle_h`` hours (default 24).

Key parts
    * arguments: ``digest`` of a dict (``args_for_key`` drops flags that cannot change results: ``--no-resume``, workers ...).
    * inputs: ``file_fingerprint`` (small top-level files by size + content hash, so rewriting identical bytes does not
      invalidate downstream steps; files inside directories and large files by size + mtime) and ``store_fingerprint``
      (key, size, last-modified of every object under the table prefixes of a store).
    * code version: ``git rev-parse --short HEAD`` plus a hash of the uncommitted diff of tracked files (so an edited
      working tree never reuses a checkpoint written by other code); env ``SORTINGHAT_CODE_VERSION`` overrides.

Use
    cp = checkpoint.open_step("cohort", args=vars(a), inputs=[a.cohort], store=store, no_resume=a.no_resume)
    with checkpoint.use(cp):                  # data_io's streaming readers then cache every row group automatically
        ...
    cp.stage("sessions-S0001", key_obj, fn)   # compute-or-load one unit
Library code asks ``checkpoint.active()``; with no active checkpoint every helper is a no-op, so tests and notebooks behave as before.

Progress lines are aggregate-only ("omop_measurement: 37/120 row groups (cached 30), part 3/181").

Test hook: env ``SORTINGHAT_CHECKPOINT_FAIL_AFTER=N`` (or ``open_step(fail_after=N)``) raises ``SimulatedKill`` (a
``BaseException``, like a SIGKILL nothing can catch) right after the N-th unit written by this process, to rehearse a restart.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import os
import pickle
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUBDIR = Path("out") / "local_only" / "checkpoints"
ENV_ROOT = "SORTINGHAT_CHECKPOINT_ROOT"
ENV_VERSION = "SORTINGHAT_CODE_VERSION"
ENV_FAIL_AFTER = "SORTINGHAT_CHECKPOINT_FAIL_AFTER"
CONTENT_HASH_MAX_BYTES = 256 << 20
# Arguments that cannot change a step's results (execution knobs); never part of the key.
KEY_EXCLUDE = frozenset({"no_resume", "checkpoint_dir", "max_memory_gb", "workers", "blas_threads", "profile"})
MISS = object()


class SimulatedKill(BaseException):
    """Raised by the test hook; a BaseException so no ``except Exception`` in the pipeline can swallow it."""


# --------------------------------------------------------------------------------------------------- hashing
def _feed(h, o: Any, depth: int = 0) -> None:
    if depth > 30:
        raise ValueError("object too deep to digest")
    if o is None:
        h.update(b"N;")
    elif isinstance(o, (bool, np.bool_)):
        h.update(b"b1;" if o else b"b0;")
    elif isinstance(o, (int, np.integer)):
        h.update(b"i" + str(int(o)).encode() + b";")
    elif isinstance(o, (float, np.floating)):
        h.update(b"f" + repr(float(o)).encode() + b";")
    elif isinstance(o, str):
        h.update(b"s" + str(len(o)).encode() + b":" + o.encode("utf-8", "surrogatepass") + b";")
    elif isinstance(o, bytes):
        h.update(b"y" + str(len(o)).encode() + b":" + o + b";")
    elif isinstance(o, Path):
        _feed(h, str(o), depth + 1)
    elif isinstance(o, (pd.Timestamp, pd.Timedelta)) or hasattr(o, "isoformat"):
        h.update(b"t" + o.isoformat().encode() + b";")
    elif isinstance(o, np.ndarray):
        h.update(b"a" + str(o.dtype).encode() + str(o.shape).encode() + b":")
        if o.dtype == object:
            for v in o.ravel().tolist():
                _feed(h, v, depth + 1)
        else:
            h.update(np.ascontiguousarray(o).tobytes())
        h.update(b";")
    elif isinstance(o, pd.DataFrame):
        h.update(b"D" + repr([str(c) for c in o.columns]).encode() + repr([str(t) for t in o.dtypes]).encode() + b":")
        try:
            h.update(pd.util.hash_pandas_object(o, index=True).to_numpy().tobytes())
        except TypeError:                                    # unhashable cells (lists ...): fall back to text
            h.update(o.astype(str).to_csv(index=True).encode())
        h.update(b";")
    elif isinstance(o, (pd.Series, pd.Index)):
        _feed(h, o.to_frame(name="v") if isinstance(o, pd.Series) else o.to_frame(index=False, name="v"), depth + 1)
    elif isinstance(o, Mapping):
        h.update(b"M" + str(len(o)).encode() + b":")
        for k in sorted(o, key=repr):
            _feed(h, k, depth + 1)
            _feed(h, o[k], depth + 1)
        h.update(b";")
    elif isinstance(o, (list, tuple)):
        h.update(b"L" + str(len(o)).encode() + b":")
        for v in o:
            _feed(h, v, depth + 1)
        h.update(b";")
    elif isinstance(o, (set, frozenset)):
        h.update(b"S" + str(len(o)).encode() + b":")
        for d in sorted(digest(v) for v in o):
            h.update(d.encode())
        h.update(b";")
    elif dataclasses.is_dataclass(o) and not isinstance(o, type):
        h.update(b"C" + type(o).__name__.encode() + b":")
        for f in dataclasses.fields(o):
            _feed(h, f.name, depth + 1)
            _feed(h, getattr(o, f.name), depth + 1)
        h.update(b";")
    else:
        h.update(b"r" + type(o).__name__.encode() + repr(o).encode() + b";")


def digest(*objs: Any) -> str:
    """Stable sha256 hex digest of arbitrary objects (scalars, containers, numpy / pandas objects, dataclasses)."""
    h = hashlib.sha256()
    for o in objs:
        _feed(h, o)
    return h.hexdigest()


def args_for_key(args: Any, exclude: Iterable[str] = ()) -> dict:
    """Result-affecting arguments of a step: ``vars(namespace)`` (or a dict) minus the execution knobs in ``KEY_EXCLUDE``."""
    d = dict(vars(args)) if not isinstance(args, Mapping) else dict(args)
    drop = KEY_EXCLUDE | set(exclude)
    return {k: v for k, v in d.items() if k not in drop}


# ----------------------------------------------------------------------------------------- input fingerprints
def _stat_entry(p: Path, content: bool) -> list:
    try:
        st = p.stat()
    except OSError:
        return [str(p), "missing"]
    if content and st.st_size <= CONTENT_HASH_MAX_BYTES:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 20), b""):
                h.update(blk)
        return [str(p), st.st_size, h.hexdigest()]
    return [str(p), st.st_size, st.st_mtime_ns]


def file_fingerprint(paths: Iterable[str | Path]) -> list:
    """Fingerprint of input files / directories. A top-level file up to 256 MB is identified by size and content hash (a
    step that rewrites identical bytes must not invalidate the steps after it); other files (large ones, and everything
    found inside a directory) by size and mtime. Missing paths are recorded as missing."""
    out = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            out.append([str(p), "dir"])
            out.extend(_stat_entry(f, False) for f in sorted(p.rglob("*")) if f.is_file())
        else:
            out.append(_stat_entry(p, True))
    return out


DEFAULT_STORE_PREFIXES = ("EEG/eeg-metadata/", "EEG/HEEDB_Metadata/", "OMOP/Merged/", "PatientMergeHistory/",
                          "Imaging/imaging_metadata/")


def store_fingerprint(store, prefixes: Iterable[str] = DEFAULT_STORE_PREFIXES) -> list:
    """(key, size, last-modified) of every object under the table prefixes of a store (S3 client or ``LocalStore``): a
    refreshed table changes the checkpoint key. Never lists per-patient folders."""
    from . import data_io
    out = []
    for pre in prefixes:
        try:
            out.append([pre, data_io.list_objects(store, pre)])
        except Exception as exc:  # noqa: BLE001 - an unreadable prefix is part of the fingerprint, not a failure here
            out.append([pre, f"unlisted:{type(exc).__name__}"])
    return out


_VERSION: str | None = None


def code_version() -> str:
    """``<git HEAD short hash>`` (+ ``+<hash of the uncommitted diff of tracked files>`` when the tree is dirty)."""
    global _VERSION
    if os.environ.get(ENV_VERSION):
        return os.environ[ENV_VERSION]
    if _VERSION is None:
        def git(*a: str) -> str | None:
            try:
                r = subprocess.run(["git", "-C", str(REPO_ROOT), *a], capture_output=True, timeout=30)
            except (OSError, subprocess.SubprocessError):
                return None
            return r.stdout.decode("utf-8", "replace").strip() if r.returncode == 0 else None
        head = git("rev-parse", "--short", "HEAD")
        diff = git("diff", "HEAD")
        _VERSION = (head or "nogit") + (("+" + hashlib.sha256(diff.encode()).hexdigest()[:8]) if diff else "")
    return _VERSION


# --------------------------------------------------------------------------------------------------- logging
_LOG_LOCK = threading.Lock()


def log(msg: str) -> None:
    """One aggregate-only progress line on stdout (guarded by ``safe_print``; progress is never worth a crash)."""
    from .safe_output import safe_print
    with _LOG_LOCK:
        try:
            safe_print(msg)
        except Exception:  # noqa: BLE001
            print("checkpoint: progress line withheld by the aggregate-only guard", flush=True)
        else:
            import sys
            sys.stdout.flush()


class RowGroupProgress:
    """Aggregate progress of one streaming read (``label: done/total row groups (cached c), part i/P``). Lines are throttled
    to one per 25 row groups or 15 s, plus one at the end. The total is shown once every part's row-group count is known
    (always the case on a resume; on a first pass the count is built up as footers are read)."""

    def __init__(self, label: str, n_parts: int, known_total: int | None):
        self.label, self.n_parts, self.total = label, n_parts, known_total
        self.done = self.cached = self.part = 0
        self._last_n, self._last_t = 0, time.monotonic()

    def start_part(self) -> None:
        self.part += 1

    def tick(self, cached: bool) -> None:
        self.done += 1
        self.cached += int(cached)
        if self.done - self._last_n >= 25 or time.monotonic() - self._last_t >= 15:
            self.emit()

    def emit(self, final: bool = False) -> None:
        self._last_n, self._last_t = self.done, time.monotonic()
        tot = f"/{self.total}" if self.total is not None else ""
        log(f"{self.label}: {self.done}{tot} row groups (cached {self.cached}), "
            f"{'done, ' if final else ''}part {min(self.part, self.n_parts)}/{self.n_parts}")


# ---------------------------------------------------------------------------------------------- the checkpoint
class Checkpoint:
    def __init__(self, step: str, key: str, directory: Path, *, resume: bool = True, fail_after: int | None = None,
                 meta: dict | None = None, prune_idle_h: float | None = 24.0, quiet: bool = False):
        self.step, self.key, self.dir = step, key, Path(directory)
        self.resume, self.quiet = resume, quiet
        self.fail_after = fail_after
        self.n_hits = self.n_puts = 0
        self.counts: dict[str, list[int]] = {}
        self._lock = threading.Lock()
        self._seq = itertools.count()
        parent = self.dir.parent
        self._mkdir(parent.parent)
        self._mkdir(parent)
        if prune_idle_h is not None:
            self._prune_siblings(parent, prune_idle_h)
        if not resume and self.dir.exists():
            shutil.rmtree(self.dir)
        existed = self.dir.exists()
        self._mkdir(self.dir)
        self._write_meta(meta or {})
        if not quiet:
            n = sum(1 for f in self.dir.rglob("*.pkl")) if existed else 0
            log(f"checkpoint {step}: key {key[:12]}, "
                f"{'resuming with ' + str(n) + ' stored units' if n else ('fresh run (--no-resume)' if not resume else 'fresh run')}")

    # -- filesystem helpers
    @staticmethod
    def _mkdir(p: Path) -> None:
        p.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(p, 0o700)
        except OSError:
            pass

    def _prune_siblings(self, parent: Path, idle_h: float) -> None:
        now = time.time()
        for d in parent.iterdir():
            if d.is_dir() and d != self.dir:
                try:
                    if now - d.stat().st_mtime >= idle_h * 3600:
                        shutil.rmtree(d, ignore_errors=True)
                except OSError:
                    pass

    def _write_meta(self, meta: dict) -> None:
        body = json.dumps({"step": self.step, "key": self.key, "code_version": code_version(),
                           "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **meta}, indent=1, default=str)
        self._atomic(self.dir / "meta.json", body.encode())

    def _atomic(self, path: Path, data: bytes) -> None:
        tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{threading.get_ident()}.{next(self._seq)}")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    def path(self, name: str) -> Path:
        h = hashlib.sha1(name.encode()).hexdigest()
        return self.dir / h[:2] / f"{h[2:30]}.pkl"

    # -- units
    def get(self, name: str, default: Any = MISS) -> Any:
        """The stored object, or ``default`` (``MISS``) when absent / unreadable / resuming is off."""
        p = self.path(name)                                          # (--no-resume already wiped the directory at open)
        try:
            with open(p, "rb") as fh:
                stored_name, obj = pickle.load(fh)
        except FileNotFoundError:
            return default
        except Exception:  # noqa: BLE001 - truncated / corrupt file: a miss, recomputed and overwritten
            return default
        if stored_name != name:                                      # hash collision guard
            return default
        with self._lock:
            self.n_hits += 1
        return obj

    def has(self, name: str) -> bool:
        return self.path(name).exists()

    def put(self, name: str, obj: Any, *, count: bool = True) -> None:
        """Store ``obj`` atomically. ``count=False`` for bookkeeping units that must not trigger the test kill hook."""
        p = self.path(name)
        self._mkdir(p.parent)
        self._atomic(p, pickle.dumps((name, obj), protocol=pickle.HIGHEST_PROTOCOL))
        if count:
            with self._lock:
                self.n_puts += 1
                kill = self.fail_after is not None and self.n_puts >= self.fail_after
            if kill:
                raise SimulatedKill(f"simulated kill after {self.n_puts} stored units")

    def stage(self, name: str, key_obj: Any, fn: Callable[[], Any], *, count: bool = True) -> Any:
        """Compute-or-load one unit: ``fn()`` runs only when no result for (``name``, ``digest(key_obj)``) is stored."""
        full = f"stage:{name}:{digest(key_obj)}"
        got = self.get(full)
        if got is not MISS:
            log(f"{self.step}: {name} loaded from checkpoint")
            return got
        out = fn()
        self.put(full, out, count=count)
        log(f"{self.step}: {name} done")
        return out

    def note(self, kind: str, cached: bool) -> tuple[int, int]:
        """Count one unit of ``kind`` (done, cached) for progress lines; returns the running totals."""
        with self._lock:
            c = self.counts.setdefault(kind, [0, 0])
            c[0] += 1
            c[1] += int(cached)
            return c[0], c[1]

    def n_units(self) -> int:
        return sum(1 for _ in self.dir.rglob("*.pkl"))


def open_step(step: str, args: Any = None, *, inputs: Iterable[str | Path] = (), store=None,
              store_prefixes: Iterable[str] = DEFAULT_STORE_PREFIXES, extra: Any = None, no_resume: bool = False,
              root: str | Path | None = None, fail_after: int | None = None, prune_idle_h: float | None = 24.0,
              exclude: Iterable[str] = (), quiet: bool = False) -> Checkpoint:
    """Open (or create) the checkpoint of a step. ``args``: its argparse namespace / dict (execution knobs are dropped);
    ``inputs``: input files / directories; ``store``: the data store whose table listing is fingerprinted;
    ``extra``: any other object that changes results. ``no_resume`` starts the run from nothing (and still checkpoints it)."""
    a = args_for_key(args, exclude) if args is not None else {}
    parts = {"step": step, "args": a, "inputs": file_fingerprint(inputs),
             "store": store_fingerprint(store, store_prefixes) if store is not None else None,
             "extra": extra, "code": code_version()}
    key = digest(parts)
    base = Path(root) if root is not None else Path(os.environ[ENV_ROOT]) if os.environ.get(ENV_ROOT) \
        else REPO_ROOT / DEFAULT_SUBDIR
    if fail_after is None and os.environ.get(ENV_FAIL_AFTER):
        fail_after = int(os.environ[ENV_FAIL_AFTER])
    meta = {"args": {k: str(v)[:200] for k, v in a.items()}, "n_input_files": len(parts["inputs"])}
    return Checkpoint(step, key, base / step / key[:24], resume=not no_resume, fail_after=fail_after, meta=meta,
                      prune_idle_h=prune_idle_h, quiet=quiet)


def add_arguments(ap) -> None:
    """``--no-resume`` for a step's argparse parser."""
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore and replace this step's checkpoint: start from nothing (the run is still checkpointed, "
                         "so a restart of THIS run resumes). Default: resume from out/local_only/checkpoints/<step>/")


# ------------------------------------------------------------------------------------------ the active checkpoint
_ACTIVE: Checkpoint | None = None
_TAG = ""


def active() -> Checkpoint | None:
    return _ACTIVE


@contextmanager
def use(cp: Checkpoint | None) -> Iterator[Checkpoint | None]:
    """Make ``cp`` the process-wide active checkpoint (threads included) for the duration of the block."""
    global _ACTIVE
    prev, _ACTIVE = _ACTIVE, cp
    try:
        yield cp
    finally:
        _ACTIVE = prev


@contextmanager
def tag(text: str) -> Iterator[None]:
    """Name the surrounding unit of work for progress lines (e.g. ``"loso fold 2"``)."""
    global _TAG
    prev, _TAG = _TAG, text
    try:
        yield
    finally:
        _TAG = prev


def current_tag() -> str:
    return _TAG
