"""Study 1 modelling harness: shallow multi-label head, representation ladder, split orchestration, controls.

See docs/models_spec.md. Aggregate-only output through ``sortinghat.safe_output``.
"""

from .controls import (PretrainedExposure, default_exposure_registry, delta_concentration_by_site, exposed_site_delta,
                       exposure_accounting, leakage_probes, negative_control, permute_eeg_within_strata,
                       permute_labels_within_site, run_mandatory_controls, sedative_excluded_rerun,
                       severity_stratified_delta, stratified_delta)
from .data import ModelData
from .head import FittedModel, HeadConfig, fit_model
from .inputs import (assemble_eeg_frame, dynamics_frame, eeg_frame_from_pipeline_rows, embedding_frame,
                     morgoth_findings_frame)
from .ladder import (COMMERCIAL_RUNGS, DEFAULT_RUNGS, LadderConfig, LadderResult, RungSpec, commercial_clean_gap,
                     fit_predict_fold, run_ladder)
from .preprocess import FoldPreprocessor, PreprocConfig
from .report import write_results
