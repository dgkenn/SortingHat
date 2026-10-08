"""Memory guard for the human-run cohort scripts: an address-space limit and a clear, aggregate error message.

``apply_limit(gb)`` sets ``RLIMIT_AS`` (virtual address space, an upper bound on RSS). Because Arrow and numpy map
more address space than they touch, pick a limit comfortably above the RSS you want (the scripts report the real
peak RSS at the end). ``run_guarded`` turns ``MemoryError`` (including ``pyarrow.lib.ArrowMemoryError``) into one
aggregate line and exit code 3 instead of a traceback.
"""

from __future__ import annotations

import resource
import sys


def peak_rss_gb() -> float:
    """Peak resident set size of this process in GB (``ru_maxrss`` is KB on Linux, bytes on macOS)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1 << 20) if sys.platform != "darwin" else r / (1 << 30)


def apply_limit(gb: float | None) -> None:
    if not gb:
        return
    try:                                     # the system allocator reserves far less address space than jemalloc/mimalloc
        import pyarrow as pa
        pa.set_memory_pool(pa.system_memory_pool())
    except Exception:                        # noqa: BLE001
        pass
    b = int(gb * (1 << 30))
    resource.setrlimit(resource.RLIMIT_AS, (b, b))


def run_guarded(fn, gb: float | None = None, what: str = "job") -> int:
    """Run ``fn()`` under the limit; return its exit code, or 3 with a one-line aggregate error on MemoryError."""
    apply_limit(gb)
    try:
        return fn()
    except MemoryError:
        msg = (f"ERROR (aggregate only): {what} exceeded its memory limit"
               f"{f' of {gb:g} GB (RLIMIT_AS)' if gb else ''}; peak RSS so far {peak_rss_gb():.2f} GB. "
               "No partial result was written for the step in progress. Re-run with a larger --max-memory-gb, "
               "fewer --sites, or on a machine with more memory.\n")
        try:
            resource.setrlimit(resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
        except Exception:                    # noqa: BLE001
            pass
        sys.stderr.write(msg)
        return 3
