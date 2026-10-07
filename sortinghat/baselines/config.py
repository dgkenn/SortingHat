"""Tunable choices of the baseline builder. Every value here is an operationalization the SAP must freeze
(``docs/baselines_spec.md``, "Operationalizations"); defaults are the proposed values."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass(frozen=True)
class BaselineConfig:
    # --- windows (hours before t0)
    score_window_h: float = 6.0          # GCS / FOUR / RASS / NESI: nearest value within [t0-6h, t0] (plan, Phase 0a)
    vital_window_h: float = 6.0          # vitals and pupils: latest value in [t0-6h, t0]            [OP]
    poc_glucose_window_h: float = 24.0   # POC glucose: latest value in [t0-24h, t0]                  [OP]
    lab_lookback_h: float = 72.0         # labs / tox / cultures: latest result in [t0-72h, t0]       [OP]
    hx_recent_h: float = 72.0            # "recent" arrest / trauma / convulsion history window       [OP]
    drug_windows_h: tuple[float, float] = (6.0, 24.0)    # cumulative dose windows (plan: prior 6 / 24 h)
    order_active_h: float = 2.0          # order-time fallback: an order <= 2 h before t0 counts as on infusion [OP]
    # --- time-basis switches (SAP sensitivity row 8 "approximate-time versions")
    drug_time_basis: str = "auto"        # auto | order | admin
    lab_time_basis: str = "auto"         # auto (result time if present, else collection + lag) | collect_plus_lag | result
    imaging_time_basis: str = "auto"     # auto (final-read time if present, else study + lag) | study_plus_lag | result
    # --- vocabulary frozen in training (human-run discovery, see baselines.vocab.discover_lab_vocabulary)
    extra_labs: tuple[str, ...] = field(default_factory=tuple)   # normalized extra lab names to include in Baseline C

    def __post_init__(self):
        for name, allowed in (("drug_time_basis", {"auto", "order", "admin"}),
                              ("lab_time_basis", {"auto", "collect_plus_lag", "result"}),
                              ("imaging_time_basis", {"auto", "study_plus_lag", "result"})):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {sorted(allowed)}")

    def to_dict(self) -> dict:
        return asdict(self)
