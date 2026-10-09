"""Work that the CBraMod and MORGOTH rungs would otherwise repeat on the same fetched ``Recording``.

Both rungs start with the same three steps on identical inputs: restrict to the 19 canonical channels, drop exactly-constant
(dead) channels, and run the window QC (``qc_recording``) on the unfiltered signal. ``RecordingPrep`` does them once; the QC is
computed lazily and memoised, including a raised exception, so each rung sees exactly what it would have seen alone.
Only used when the QC configuration and windows the rung asks for are the ones the prep was built with (``compatible``).
"""

from __future__ import annotations

from .io import Recording, drop_dead_channels, select_channels
from .stream import RecordingTimeout
from .window import QCConfig, WindowSpec, qc_recording


class RecordingPrep:
    def __init__(self, rec: Recording, windows: dict[str, WindowSpec], qc_cfg: QCConfig | None = None, timers=None):
        self.windows = dict(windows)
        self.qc_cfg = qc_cfg or QCConfig()
        self._timers = timers
        self.rec = drop_dead_channels(select_channels(rec))
        self.total = self.rec.meta.get("edf_duration_s", self.rec.offset_s + self.rec.duration_s)
        self.notes = {"dead": self.rec.meta.get("dead_channels", []),
                      "invalid_scaling": self.rec.meta.get("invalid_scaling_channels", [])}
        self._qc = None                                   # ("ok", (qcs, ef)) | ("err", exception)

    def compatible(self, windows, qc_cfg) -> bool:
        return dict(windows) == self.windows and (qc_cfg or QCConfig()) == self.qc_cfg

    def qc(self):
        """``(qcs, epoch_flags)`` of ``qc_recording`` on the prepared recording (computed once)."""
        if self._qc is None:
            from ..timing import stage
            try:
                with stage(self._timers, "qc"):
                    r = qc_recording(self.rec.data, self.rec.fs, self.rec.ch_names, self.rec.offset_s, self.windows,
                                     self.qc_cfg, rec_duration_s=self.total, channel_notes=self.notes)
                self._qc = ("ok", r)
            except RecordingTimeout:                      # the rung's own deadline: not replayed to the other rung
                raise
            except Exception as e:                       # noqa: BLE001 - replayed to the second rung unchanged
                self._qc = ("err", e)
        kind, val = self._qc
        if kind == "err":
            raise val
        return val
