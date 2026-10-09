"""Per-stage wall-clock timers for the extractors. Aggregates only: stage name -> seconds, never row or ID content."""

from __future__ import annotations

import time
from contextlib import contextmanager, nullcontext


class StageTimers:
    """``with timers("name"):`` adds the block's elapsed seconds to ``timers.s["name"]`` (repeat uses accumulate)."""

    def __init__(self, clock=time.perf_counter):
        self.s: dict[str, float] = {}
        self._clock = clock

    @contextmanager
    def __call__(self, name: str):
        t0 = self._clock()
        try:
            yield
        finally:
            self.s[name] = self.s.get(name, 0.0) + (self._clock() - t0)


def stage(timers: StageTimers | None, name: str):
    """``with stage(timers, "x"):`` - a no-op when ``timers`` is None."""
    return nullcontext() if timers is None else timers(name)
