"""Reference implementation of the Study 1 evaluation metrics (docs/prereg_study1_sap.md).

numpy / scipy / scikit-learn only. No record-level output; every function returns aggregates
or per-patient arrays that stay in the caller's memory.
"""

from .bootstrap import (
    BootstrapResult,
    bootstrap_ci,
    bootstrap_replicates,
    delta_ci_all_modes,
    holm_adjust,
    one_sided_p_below_zero,
    paired_bootstrap_delta,
    percentile_ci,
    site_mean_t_interval,
)
from .calibration import (
    auroc,
    brier,
    calibration_slope_intercept,
    ece,
    per_label_report,
    per_label_risk_coverage,
    risk_coverage,
)
from .hypotheses import (
    NESTED_WINDOWS_S,
    first_window_reaching_fraction,
    h3_severity_stratified,
    h5_interaction,
    h6_ratio,
    kendall_ranking,
    pairwise_auroc,
    per_label_delta_ci,
)
from .labels import ALL_LABELS, PRIMARY_LABELS, e7_eligible, primary_label_indices
from .loss import (
    DEFAULT_EPS,
    delta_log_loss,
    favorable_at_every_site,
    mean_masked_log_loss,
    per_label_delta,
    per_patient_delta,
    per_patient_loss,
    per_site_delta,
    site_weighted_delta,
)
from .splits import check_site_requirements, late_temporal_holdout, leave_one_site_out
