"""Every tunable choice of the Study 1 cohort builder, in one frozen dataclass.

Each field is an operationalisation the plan does not state; ``docs/cohort_spec.md`` lists them with the
DECISION_LOG entry they need (C-nn). Defaults are the proposed values. Nothing here reads data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..eeg.io import DEFAULT_MINIMUM_CHANNELS
from ..eeg.window import PRIMARY_DURATION_S, PRIMARY_START_S

ONSET_RULES = ("score_then_visit", "visit_start", "score_only")
SCORE_RULES = ("nearest", "any")


@dataclass(frozen=True)
class CohortConfig:
    # --- population
    adult_age_years: float = 18.0                       # plan: adults
    acute_classes: tuple[str, ...] = ("ICU", "Inpatient", "ED")        # plan: ICU, inpatient, ED
    use_service_fallback: bool = False                  # C-03: visit class unknown + ServiceName in service_acute -> acute
    service_acute: tuple[str, ...] = ("LTM",)
    exclude_services: tuple[str, ...] = ("OR", "EMU")   # C-04: intra-operative and epilepsy-unit EEGs are not ACI work-ups
    visit_chain_gap_h: float = 6.0                        # C-05: ED -> inpatient/ICU visits <= this gap apart form one encounter
    # --- recording
    min_duration_s: float = PRIMARY_START_S + PRIMARY_DURATION_S      # minutes 1-11 must exist (660 s)
    duration_scale_by_site: tuple[tuple[str, float], ...] = ()        # C-07: unit fix, e.g. (("I0008", 60.0),) once checked
    # --- ACI onset (time since onset is a covariate; windows are sensitivity analyses)
    onset_rule: str = "score_then_visit"                # C-06
    abnormal_gcs_max: float = 14.0                      # "abnormal" GCS total for the onset proxy
    abnormal_four_max: float = 15.0
    onset_primary_h: float = 24.0                       # plan: EEG within 24 h of onset
    onset_sensitivity_h: tuple[float, ...] = (6.0, 12.0, 48.0)        # plan: <= 6 / 12 / 48 h sensitivity analyses
    # --- strict severity. PRIMARY (D-105): nearest qualifying-instrument score in [t0 - 6 h, t0 + 1 h], pre-t0 on ties.
    score_before_h: float = 6.0
    score_after_h: float = 1.0
    gcs_strict_max: float = 11.0                        # plan: GCS <= 11
    four_strict_max: float = 12.0                       # plan: FOUR <= 12
    score_rule: str = "nearest"                         # primary rule ("any": any qualifying score in the window)
    # SENSITIVITY ``strict_pm6``: any qualifying score within +-6 h of t0 (the plan's wording; uses 6 h of post-t0 data)
    pm6_window_h: float = 6.0
    # --- broad EHR phenotype
    phenotype_after_h: float = 6.0                      # C-10: condition start in [encounter start, t0 + this]
    # --- hand-off to the streaming extractor
    window_start_s: float = PRIMARY_START_S
    window_duration_s: float = PRIMARY_DURATION_S
    minimum_channels: tuple[str, ...] = tuple(DEFAULT_MINIMUM_CHANNELS)
    min_usable_fraction: float = 0.60

    def __post_init__(self):
        if self.onset_rule not in ONSET_RULES:
            raise ValueError(f"onset_rule must be one of {ONSET_RULES}")
        if self.score_rule not in SCORE_RULES:
            raise ValueError(f"score_rule must be one of {SCORE_RULES}")
        if self.onset_max_h < self.onset_primary_h:
            raise ValueError("the widest sensitivity window must cover the primary window")

    @property
    def onset_max_h(self) -> float:
        """Widest onset window: patients up to here are kept in the table so every sensitivity window is a flag."""
        return max((self.onset_primary_h, *self.onset_sensitivity_h))

    @property
    def onset_windows_h(self) -> tuple[float, ...]:
        return tuple(sorted({self.onset_primary_h, *self.onset_sensitivity_h}))

    def to_dict(self) -> dict:
        return asdict(self)
