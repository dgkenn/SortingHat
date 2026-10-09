"""Shared, cross-step local cache of person-filtered, column-pruned OMOP row groups (D-149).

WHY. The cohort build, the Phase 0a field audit, the structured silver labels and the baselines all stream the SAME OMOP tables
from S3, each for (nearly) the same people and (nearly) the same columns, and the container restarts about hourly. The network is
the bottleneck, so the first step to read a row group stores it on the container disk and every later step (and every restart)
reads it locally.

WHAT IS STORED. One parquet file per (table, source part, row group), under ``out/local_only/omop_cache/`` (gitignored, directories
0700, files 0600, never printed):

    <root>/<table>/<profile>/<part>/manifest.json   row-group count and the columns present in the part
    <root>/<table>/<profile>/<part>/rg-000012.parquet   the row group, filtered to the CANDIDATE people, only the SUPERSET columns
    <root>/<table>/<profile>/<part>/rg-000012.empty     marker: nothing in the row group survives the filter
    <root>/candidates-<digest>.npy                    the candidate person ids (sorted int64), its own small file

* ``<part>`` = hash of (S3 key, size, ETag): a changed source object is a different directory, so a stale entry is never read.
* ``<profile>`` = hash of (table, superset columns, candidate-set digest, cache version): a different column list or candidate
  set is a different directory.
* SUPERSET COLUMNS (``superset_columns``): per table, the union of every column any step reads (cohort sources, audit, silver
  extract, baselines, alignment) plus the audit's predicate columns. ``note`` holds ONLY the timestamp columns (never text).
* CANDIDATES (``candidate_ids``): every ADULT person at a Study 1 site (``CohortConfig.study_sites``, ``adult_age_years``) in the
  site eeg_metadata, re-keyed through the patient merge map, plus the ids merged into them and the source ids (about 51k on the
  real data). It is the cohort's pre-first-EEG candidate set before the visit, onset and severity steps, so the cohort, the audit
  candidates, the silver cohort and the baseline cohort are all subsets. A request for a person NOT in the candidate set (e.g.
  ``--all-sites``) or for a column outside the superset is simply not served by the cache and reads S3 directly, exactly as before.

HOW A STEP USES IT. Automatically: ``data_io.iter_omop_batches`` and ``cohort.sources.iter_filtered_batches`` ask ``reader_for``
whenever a checkpoint is active (the long steps open one) and the cache is not switched off (env ``SORTINGHAT_OMOP_CACHE=off``).
A request gets the cached row group, then filters further IN MEMORY (its own person ids, its own predicate, its own columns), so the
rows and columns are the same as an uncached read. Missing row groups are fetched from S3 (superset columns, candidate filter),
stored, then served. ``--no-resume`` ignores stored entries and refreshes them.

BOUNDED PREFETCH. Up to ``SORTINGHAT_FETCH_WORKERS`` (default 4) row groups are fetched/loaded in parallel by a thread pool and
handed to the caller IN ORDER, so peak memory stays about N row groups (plus the one being processed). The retry wrapper
(``data_io.with_retries``) is kept around every S3 read.

WARM-UP. ``python -m sortinghat.omop_cache warm --s3 [--tables measurement visit_occurrence ...]`` fills the cache once for every
table and column, with aggregate-only progress lines; it is resumable (stored row groups are skipped on a rerun).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from . import checkpoint as ck

CACHE_VERSION = "1"
ENV_ROOT = "SORTINGHAT_OMOP_CACHE_ROOT"
ENV_SWITCH = "SORTINGHAT_OMOP_CACHE"                 # off / 0 / false / no disables the shared cache
ENV_WORKERS = "SORTINGHAT_FETCH_WORKERS"
DEFAULT_WORKERS = 4
DEFAULT_SUBDIR = Path("out") / "local_only" / "omop_cache"
NO_PERSON_TABLES = frozenset({"concept"})             # vocabulary: cached with the superset columns, no person filter
SKIPPED = object()                                    # yielded by ``iter_rowgroups`` for a unit the caller asked to skip


# ------------------------------------------------------------------------------------------------ configuration
def cache_root() -> Path:
    return Path(os.environ[ENV_ROOT]) if os.environ.get(ENV_ROOT) else ck.REPO_ROOT / DEFAULT_SUBDIR


def enabled() -> bool:
    return os.environ.get(ENV_SWITCH, "on").strip().lower() not in {"off", "0", "false", "no"}


def fetch_workers() -> int:
    try:
        return max(1, int(os.environ.get(ENV_WORKERS, DEFAULT_WORKERS)))
    except ValueError:
        return DEFAULT_WORKERS


_SUPERSET: dict[str, list[str]] | None = None


def superset_columns() -> dict[str, list[str]]:
    """Per table, the union (stable order) of every column any step reads. Collected from the code, so a new column added to a
    step's list is picked up here (``tests/test_omop_cache.py`` fails if a step asks for a column outside the superset)."""
    global _SUPERSET
    if _SUPERSET is not None:
        return _SUPERSET
    from . import schema
    from .audit import alignment as al
    from .audit import field_audit as fa
    from .baselines import omop_columns as bl
    from .cohort import sources as cs
    from .labels import extract as lx
    need: dict[str, list[str]] = {}

    def add(table: str, cols) -> None:
        cur = need.setdefault(table, [])
        cur.extend(c for c in cols if c not in cur)

    for t, cols in lx.COLUMNS.items():                                       # silver extract (incl. concept)
        add(t, cols)
    add("visit_occurrence", cs._VISIT_COLS)                                   # cohort
    add("measurement", cs._MEAS_COLS)
    add("condition_occurrence", cs._COND_COLS)
    add("drug_exposure", fa.DRUG_COLS)                                        # audit
    add("note", fa.NOTE_COLS)
    add("measurement", fa.MEAS_COLS + schema.COLUMN_ALIASES["measurement.result_datetime"])
    add("observation", fa.OBS_COLS)
    add("concept", ["concept_id", "concept_name", "domain_id", "vocabulary_id"])
    add("visit_occurrence", al.VISIT_COLS)                                    # audit alignment
    add("person", al.BIRTH_COLS)
    add("death", al.DEATH_COLS)
    for tbl, dt, dd in al.EVENT_SOURCES.values():
        add(tbl, ["person_id", dt, dd])
    add("measurement", bl.MEAS_COLS)                                          # baselines
    add("drug_exposure", bl.DRUG_COLS)
    add("condition_occurrence", bl.COND_COLS)
    add("procedure_occurrence", bl.PROC_COLS)
    add("observation", bl.OBS_COLS)
    _SUPERSET = need
    return need


# ----------------------------------------------------------------------------------------------- candidate set
_CANDIDATES: dict[tuple, np.ndarray | None] = {}


def candidate_ids(store) -> np.ndarray | None:
    """Sorted int64 candidate person ids (see the module docstring), computed once and stored in its own small file keyed by the
    eeg-metadata / merge-history listing; ``None`` when the store cannot give them (the cache is then not used)."""
    from . import data_io
    from .cohort.config import CohortConfig
    cfg = CohortConfig()
    memo = (id(store), ck.active().key if ck.active() else None)
    if memo in _CANDIDATES:
        return _CANDIDATES[memo]
    out: np.ndarray | None = None
    try:
        fp = ck.digest(ck.store_fingerprint(store, (data_io.EEG_METADATA_PREFIX, data_io.HEEDB_METADATA_PREFIX,
                                                    "PatientMergeHistory/")),
                       cfg.study_sites, cfg.adult_age_years, CACHE_VERSION)
        f = cache_root() / f"candidates-{fp[:20]}.npy"
        if f.exists():
            out = np.load(f)
        else:
            cp = ck.active()
            if cp is None:
                out = _compute_candidates(store, cfg)
            else:
                with cp.kill_suspended():                       # bookkeeping: not part of the test kill budget
                    out = _compute_candidates(store, cfg)
            _atomic_bytes(f, lambda fh: np.save(fh, out))
    except Exception as exc:  # noqa: BLE001 - never fatal: the steps read S3 directly
        ck.log(f"omop_cache: candidate set unavailable ({type(exc).__name__}); reading S3 directly")
        out = None
    _CANDIDATES[memo] = out
    return out


def _compute_candidates(store, cfg) -> np.ndarray:
    import pandas as pd
    from .cohort.sources import StoreSources
    src = StoreSources(store)
    S = src.sessions()
    mm, _status = src.merge_map()
    if cfg.study_sites is not None:
        S = S[S["SiteID"].astype(str).isin(cfg.study_sites)]
    S = S[S["person_id"].notna() & (pd.to_numeric(S["age_years"], errors="coerce") >= cfg.adult_age_years)]
    src_ids = {int(p) for p in S["person_id"].unique()}
    surv = {int(mm.get(p, p)) for p in src_ids}
    merged_in = {int(o) for o, n in mm.items() if int(n) in surv}
    return np.array(sorted(src_ids | surv | merged_in), dtype="int64")


# --------------------------------------------------------------------------------------------------- file I/O
def _mkdir(p: Path) -> None:
    """Create ``p`` (and every missing level from the cache root down) with mode 0700."""
    root = cache_root()
    chain = [d for d in reversed([p, *p.parents]) if d == root or root in d.parents]
    for d in chain or [p]:
        d.mkdir(exist_ok=True, parents=True)
        try:
            os.chmod(d, 0o700)
        except OSError:
            pass


_SEQ = iter(range(1 << 60))
_SEQ_LOCK = threading.Lock()


def _atomic_bytes(path: Path, write: Callable[[Any], None]) -> None:
    """Write through ``write(fileobj)`` to a temporary file in the same directory, fsync, rename."""
    _mkdir(path.parent)
    with _SEQ_LOCK:
        n = next(_SEQ)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{threading.get_ident()}.{n}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "wb") as fh:
            write(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


# ------------------------------------------------------------------------------------------------- the reader
class SharedReader:
    """The cache seen from one streaming read of one table. ``iter_rowgroups`` yields ``(part key, row group, table | None)`` in
    order, where the table is the cached SUPERSET row group (candidate-filtered; ``None`` = empty)."""

    def __init__(self, cp, s3, bucket: str, table: str, columns: list[str], ids: np.ndarray | None, *, retry, max_get_bytes: int,
                 buffer_size: int, workers: int | None = None, load: bool = True):
        from . import data_io
        self.dio, self.cp, self.s3, self.bucket, self.table = data_io, cp, s3, bucket, table
        self.columns, self.ids = list(columns), ids
        self.retry, self.max_get_bytes, self.buffer_size = retry, max_get_bytes, buffer_size
        self.workers = workers or fetch_workers()
        self.load, self.refresh = load, not cp.resume
        self.parts, self.meta = data_io._checkpoint_parts(s3, table, None, bucket)
        self.profile = ck.digest(table, self.columns, None if ids is None else ids, CACHE_VERSION)[:16]
        self.outer = data_io._outer(retry)
        self.pid_arr, self.ids_list = None, None
        if ids is not None:
            self.ids_list = [int(i) for i in ids]
            import pyarrow as pa
            self.pid_arr = pa.array(ids, type=pa.int64())
        self.max_in_flight = 0
        self._tl = threading.local()
        # --no-resume refreshes every stored unit ONCE per run (not once per read of the table): units refreshed so far
        self._fresh: set[str] = cp.__dict__.setdefault("_fresh_units", set())
        self.progress = ck.RowGroupProgress(f"omop_{table}", len(self.parts), self._known_total())

    # -- paths
    def _part_dir(self, key: str) -> Path:
        size, etag = self.meta.get(key, (None, None))
        h = hashlib.sha1(f"{key}|{size}|{etag}".encode()).hexdigest()[:20]
        return cache_root() / self.table / self.profile / h

    def _manifest(self, key: str) -> dict | None:
        path = self._part_dir(key) / "manifest.json"
        if self.refresh and str(path) not in self._fresh:
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return None

    def _known_total(self) -> int | None:
        if self.refresh:
            return None
        counts = [self._manifest(k) for k in self.parts]
        return sum(c["n"] for c in counts) if all(c is not None for c in counts) else None

    # -- units
    def unit_id(self, key: str, rg: int) -> str:
        """Stable id of one cached row group (part directory name + row group number), for callers that checkpoint per unit."""
        return f"{self._part_dir(key).name}:{rg}"

    def _lookup(self, key: str, rg: int, cols: list[str] | None = None):
        """(found, table | None). ``load=False`` (warm-up) only checks presence. ``cols``: read only these columns (those the
        cached file has) instead of the whole superset row group."""
        d = self._part_dir(key)
        if (d / f"rg-{rg:06d}.empty").exists():
            return True, None
        p = d / f"rg-{rg:06d}.parquet"
        if not p.exists():
            return False, None
        if not self.load:
            return True, None
        try:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(p)
            if cols is None:
                return True, pf.read()
            have = set(pf.schema_arrow.names)
            return True, pf.read(columns=[c for c in cols if c in have])
        except Exception:  # noqa: BLE001 - unreadable (truncated by a kill / disk full): a miss, refetched and replaced
            return False, None

    def _store(self, key: str, rg: int, tbl) -> None:
        import pyarrow.parquet as pq
        d = self._part_dir(key)
        if tbl is None or not tbl.num_rows:
            _atomic_bytes(d / f"rg-{rg:06d}.empty", lambda fh: None)
            (d / f"rg-{rg:06d}.parquet").unlink(missing_ok=True)
        else:
            _atomic_bytes(d / f"rg-{rg:06d}.parquet", lambda fh: pq.write_table(tbl, fh, compression="zstd"))
            (d / f"rg-{rg:06d}.empty").unlink(missing_ok=True)
        self.cp.tick()                                               # test kill hook (counts stored units)

    def _open(self, key: str):
        pf = getattr(self._tl, "pf", None)
        if getattr(self._tl, "key", None) == key and pf is not None:
            return pf
        pf = self.dio.with_retries(lambda: self.dio.open_parquet(self.s3, self.bucket, key, policy=self.retry,
                                                                 max_get_bytes=self.max_get_bytes,
                                                                 buffer_size=self.buffer_size), self.outer)
        self._tl.pf, self._tl.key = pf, key
        return pf

    def _work(self, key: str, rg: int, use: list[str], cols: list[str] | None = None):
        unit = str(self._part_dir(key) / f"rg-{rg:06d}")
        if not self.refresh or unit in self._fresh:
            hit, tbl = self._lookup(key, rg, cols)
            if hit:
                return True, tbl
        pf = self._open(key)
        tbl = self.dio.read_rowgroup(pf, rg, use, self.ids_list, self.pid_arr, self.outer)
        self._store(key, rg, tbl)
        self._fresh.add(unit)
        return False, tbl

    # -- ordered, bounded prefetch
    def _tasks(self, on_error) -> Iterator[tuple[str, int, list[str]]]:
        for key in self.parts:
            self.progress.start_part()
            try:
                man = self._manifest(key)
                if man is None:
                    pf = self._open(key)
                    have = pf.schema_arrow.names
                    use = [c for c in self.columns if c in have]
                    man = {"n": pf.num_row_groups if use else 0, "have": list(have), "use": use}
                    _atomic_bytes(self._part_dir(key) / "manifest.json", lambda fh: fh.write(json.dumps(man).encode()))
                    self._fresh.add(str(self._part_dir(key) / "manifest.json"))
            except Exception as exc:  # noqa: BLE001
                if on_error is None:
                    raise
                on_error(key, exc)
                continue
            for rg in range(man["n"]):
                yield key, rg, man["use"]

    def iter_rowgroups(self, on_error: Callable[[str, Exception], None] | None = None, cols: list[str] | None = None,
                       skip: Callable[[str], bool] | None = None):
        """``cols``: load only these columns of a cached row group. ``skip(unit_id)``: true for a unit the caller already has
        (e.g. a stored result); it is yielded as ``SKIPPED`` without being loaded or fetched."""
        tasks = self._tasks(on_error)
        window: deque = deque()
        failed: set[str] = set()
        pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="omop-fetch")
        exhausted = False
        try:
            while True:
                while not exhausted and len(window) < self.workers:
                    nxt = next(tasks, None)
                    if nxt is None:
                        exhausted = True
                        break
                    key, rg, use = nxt
                    if skip is not None and skip(self.unit_id(key, rg)):
                        window.append((key, rg, None))
                    else:
                        window.append((key, rg, pool.submit(self._work, key, rg, use, cols)))
                    self.max_in_flight = max(self.max_in_flight, len(window))
                if not window:
                    break
                key, rg, fut = window.popleft()
                if fut is None:
                    if key not in failed:
                        self.progress.tick(True)
                        yield key, rg, SKIPPED
                    continue
                try:
                    hit, tbl = fut.result()
                except Exception as exc:  # noqa: BLE001 - a part that still fails after the retries: reported, rest skipped
                    if on_error is None:
                        raise
                    if key not in failed:
                        failed.add(key)
                        on_error(key, exc)
                    continue
                if key in failed:
                    continue
                self.progress.tick(hit)
                yield key, rg, tbl
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        self.progress.emit(final=True)

    # -- serving a request from a cached superset row group
    def select_request(self, cached, want: list[str], ids, pid_arr):
        """What ``data_io.iter_omop_batches`` would have produced for this row group: the request's own person ids (a no-op when
        they are all the candidates), then its own columns in its own order; ``None`` for nothing."""
        import pyarrow as pa
        import pyarrow.compute as pc
        if cached is None:
            return None
        cols = [c for c in want if c in cached.column_names]
        if not cols:
            return None
        t = cached
        if ids is not None and "person_id" in t.column_names and (self.ids is None or len(ids) != len(self.ids)):
            mask = pc.fill_null(pc.is_in(pc.cast(t.column("person_id"), pa.int64(), safe=False), value_set=pid_arr), False)
            t = t.filter(mask)
        t = t.select(cols)
        return t if t.num_rows else None

    def select_filtered(self, cached, columns: list[str], pid_arr, text_col, pattern, id_col, id_arr):
        """What ``cohort.sources.iter_filtered_batches`` would have yielded for this row group (person filter AND the row
        predicate ``text matches pattern OR id_col in ids``), applied to the cached superset row group; ``None`` for nothing."""
        import pyarrow as pa
        import pyarrow.compute as pc
        if cached is None:
            return None
        have = set(cached.column_names)
        use = [c for c in columns if c in have]
        if "person_id" not in use:
            return None
        pred = bool(pattern) or id_arr is not None
        t_ok = bool(pattern) and text_col in have
        i_ok = id_arr is not None and id_col in have
        if pred and not (t_ok or i_ok):
            return None
        mask = pc.fill_null(pc.is_in(pc.cast(cached.column("person_id"), pa.int64(), safe=False), value_set=pid_arr), False)
        if pred:
            pm = []
            if t_ok:
                pm.append(pc.match_substring_regex(pc.cast(cached.column(text_col), pa.string()), pattern, ignore_case=True))
            if i_ok:
                pm.append(pc.is_in(pc.cast(cached.column(id_col), pa.int64(), safe=False), value_set=id_arr))
            m2 = pm[0]
            for x in pm[1:]:
                m2 = pc.or_kleene(m2, x)
            mask = pc.and_kleene(mask, pc.fill_null(m2, False))
        if not pc.any(mask).as_py():
            return None
        return cached.select(use).filter(mask)


def reader_for(cp, s3, bucket: str, table: str, *, columns: list[str], ids, prefix: str | None = None, retry=None,
               max_get_bytes: int, buffer_size: int = 1 << 20, load: bool = True) -> SharedReader | None:
    """A ``SharedReader`` when the shared cache can serve this request, else ``None`` (the caller reads S3 as before).

    Served when: a checkpoint is active, the cache is on, the table is a merged OMOP table with a superset entry, every requested
    column is in the superset, and the request's person ids (if any) are all candidates. Tables without people (``concept``)
    take no ids."""
    if cp is None or not enabled() or prefix:
        return None
    sup = superset_columns().get(table)
    if sup is None or not set(columns) <= set(sup):
        return None
    if table in NO_PERSON_TABLES:
        if ids is not None:
            return None
        cand = None
    else:
        if ids is None or "person_id" not in columns:
            return None
        cands = candidate_ids(s3)
        if cands is None:
            return None
        req = np.asarray(ids, dtype="int64")
        if len(req) and not np.isin(req, cands).all():
            return None
        cand = cands
    return SharedReader(cp, s3, bucket, table, sup, cand, retry=retry, max_get_bytes=max_get_bytes,
                        buffer_size=buffer_size, load=load)


# ------------------------------------------------------------------------------------------------ warm-up CLI
def warm(store, tables: list[str], *, workers: int | None = None, no_resume: bool = False) -> int:
    """Fill the cache for ``tables`` (default: every table with a superset entry). Resumable; aggregate-only progress lines."""
    from . import data_io
    sup = superset_columns()
    bad = [t for t in tables if t not in sup]
    if bad:
        raise SystemExit(f"no cache entry defined for table(s) {bad}; known: {sorted(sup)}")
    cp = ck.open_step("omop_warm", {"tables": sorted(tables)}, store=store, no_resume=no_resume)
    bucket = data_io.access_point()
    with ck.use(cp):
        cands = None
        for t in tables:
            if t not in NO_PERSON_TABLES and cands is None:
                cands = candidate_ids(store)
                if cands is None:
                    raise SystemExit("cannot determine the candidate set from the store's eeg_metadata; nothing cached")
                ck.log(f"omop_cache: candidate set of {len(cands)} people")
            r = reader_for(cp, store, bucket, t, columns=sup[t], ids=None if t in NO_PERSON_TABLES else cands,
                           max_get_bytes=data_io.GET_CHUNK_BYTES, load=False)
            if r is None:
                raise SystemExit(f"table {t!r} cannot be served by the cache (disabled or no candidate set)")
            if workers:
                r.workers = workers
            for _ in r.iter_rowgroups(None):
                pass
    ck.log(f"omop_cache: warm-up finished for {len(tables)} table(s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sortinghat.omop_cache", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("warm", help="fill the shared OMOP cache once for all tables / columns (resumable)")
    src = w.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="local directory in the HEEDB layout (synthetic data or a local mirror)")
    src.add_argument("--s3", action="store_true", help="read the real BDSP access point (human-run; refuses inside an agent session)")
    w.add_argument("--profile", help="AWS profile for --s3")
    w.add_argument("--tables", nargs="+", help="default: every table any step reads (see superset_columns)")
    w.add_argument("--workers", type=int, help=f"parallel row-group fetches (default env {ENV_WORKERS} or {DEFAULT_WORKERS})")
    ck.add_arguments(w)
    a = ap.parse_args(argv)
    from . import agent_safety, data_io
    if a.data:
        agent_safety.assert_not_restricted_in_agent(a.data)
    store = data_io.open_store(a.data, profile=a.profile)
    tables = a.tables or sorted(superset_columns())
    return warm(store, tables, workers=a.workers, no_resume=a.no_resume)


if __name__ == "__main__":
    sys.exit(main())
