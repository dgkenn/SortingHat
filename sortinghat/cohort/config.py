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
    study_sites: tuple[str, ...] | None = ("I0002", "I0003", "S0001", "S0002")   # D-113; None = every site
    adult_age_years: float = 18.0                       # plan: adults
    # Acute-care proxy (D-111): the covering visit is inpatient-length (visit_end_date > visit_start_date) OR ServiceName
    # names an acute setting. ``acute_classes`` applies ONLY to a visit whose concept id / source text classify it
    # (all visit_concept_id are 0 in HEEDB, so this is a no-op unless non-zero ids ever appear).
    acute_classes: tuple[str, ...] = ("ICU", "Inpatient", "ED")
    use_service_proxy: bool = True
    service_acute: tuple[str, ...] = ("LTM", "ICU", "ED", "INPATIENT", "EMERGENCY")   # substring match, upper case
    exclude_services: tuple[str, ...] = ("OR", "EMU")   # C-04: intra-operative and epilepsy-unit EEGs are not ACI work-ups
    # Visit cover (D-112): date-only visits, [visit_start_date - 24 h, visit_end_date + 24 h].
    visit_dates_only: bool = True
    visit_slack_h: float = 24.0
    open_visit_days: float | None = 30.0                # a visit with no end is open this long after its start
    date_only_end_of_day: bool = True                   # used only when visit_dates_only is False
    visit_chain_gap_h: float = 6.0                      # C-05: ED -> inpatient/ICU visits <= this gap apart form one encounter
    # --- recording (D-115): clock duration EndTime - StartTime; the metadata duration is not used
    min_duration_s: float = PRIMARY_START_S + PRIMARY_DURATION_S      # minutes 1-11 must exist (660 s)
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
