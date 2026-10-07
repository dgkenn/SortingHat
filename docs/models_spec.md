# Study 1 modelling harness: specification

Code: `sortinghat/models/`. Tests: `tests/test_models_*.py`. Design source: `docs/prereg_study1_sap.md` (SAP) sections 4, 5, 7, 9, 10; plan "Representations and models" and "Splits and analyses". Synthetic data only; outputs aggregate-only through `sortinghat.safe_output`.

## Data contract (`ModelData`)

One row per patient (first qualifying EEG). `baseline` is a generic numeric DataFrame (built elsewhere; the harness never inspects its contents). `eeg` is a numeric DataFrame whose column prefixes define ladder rungs: `qeeg.`, `conn.` (from `eeg/pipeline.py` rows, see `eeg_frame_from_pipeline_rows`), `morgoth.`, `emb.<family>.`, `dyn.`. Silver labels (`y_silver`, `m_silver`) train; gold labels (`y_gold`, `m_gold`) carry a `gold_role` of `dev`, `eval` or `none` (mask forced False for `none`). `covariates` (duration, channel count, severity, sedated) feed controls only, never models.

## Head and fold fitting (`head.py`, `preprocess.py`)

- One L2 logistic regression per label (optional small MLP, `HeadConfig(kind="mlp")`), fitted on silver labels with per-label masks; a label with fewer than 3 examples of a class gets a smoothed prevalence.
- Preprocessing (median imputation, missingness indicator for columns with training NaN, standardisation, clipping at 5 SD, optional top-k selection of non-baseline columns by silver association, constant-column removal) is fitted by `FoldPreprocessor.fit` on training rows only.
- Selection grid, identical for every model: C in (0.01, 0.1, 1) and an extra EEG-block shrinkage s in (1, 0.3, 0.1) applied to non-baseline columns (equivalent to a heavier L2 penalty on EEG than baseline). Needed because one C shared by a few strong baseline variables and many noisy EEG variables makes the EEG model worse than baseline under the null. Baseline-only models have no EEG block, so their grid is C only. Selection criterion: masked log loss on gold **dev** cases inside the training fold (primary labels).
- Recalibration: Platt (default) or isotonic per label, fitted on the dev cases' predictions inside the training fold; identity if a label has fewer than 8 dev cases of either class. With fewer than 30 dev patients, no selection or recalibration is done (defaults used, recorded in `fold_info`).
- Silver training rows exclude dev rows. Gold eval labels are never passed to any fitter.
- Deviation to note: the SAP says hyperparameters use inner folds grouped by site; here selection uses the dev gold set inside the training sites (as specified for this build). Revisit before freeze.

## Strict fold API (`ladder.fit_predict_fold`)

Takes the full `ModelData` and index arrays, slices training rows itself, passes fitters only training features, silver labels and dev gold, and reads **features only** for test rows. Raises if train and test overlap, or (LOSO) if a held-out site is in the training fold. `tests/test_models_cv.py` corrupts every held-out feature, label, mask, role and covariate and asserts identical fitted-parameter fingerprints (scaler, imputer, selection, head weights, calibrators); a companion test shows the fingerprint does change when training rows change, and that eval-role gold labels of training sites do not enter fitting.

## Ladder (`ladder.run_ladder`)

Rungs: prior (prevalence; no EEG so Delta = 0 by identity; its loss is a reference), qeeg, connectivity, morgoth, embeddings (`emb.cbramod.`), dynamics, combined, plus `combined_morgoth` and `combined_cbramod` for the commercial-clean gap (`commercial_clean_gap`, paired bootstrap on d_i differences). A rung with no matching columns is reported `available: False`. Each rung is fitted as baseline-only vs baseline+EEG with the same head, grid and recalibration, per baseline set (`baseline_sets`: name -> baseline columns, so A to D can be run side by side). Delta is computed on gold **eval** patients, primary labels only (`include_e7` optional), with `metrics.loss.per_patient_delta`. Reported per rung: pooled Delta, 99% within-site bootstrap CI (D-095), 95% within_site / cluster / two_stage / site_t, per-site Delta, favourable-at-every-site flag, site-weighted Delta, `h1h2_rule_met` (CI hi < 0 and every site < 0; descriptive for non-primary rungs), per-label Delta with Holm-ready one-sided p. Splits: `loso` or `temporal` (`late_temporal_holdout`).

## Controls (`controls.py`)

| Control | Function | Notes |
|---|---|---|
| Site / duration / channel-count leakage | `leakage_probes` | Cross-validated logistic probe with imputation and scaling refitted per CV fold. Site: macro one-vs-rest AUROC. Duration and channel count: above-median vs not. Flag if AUROC > 0.70 (PLACEHOLDER). Channel count constant -> `estimable: False`. |
| Probe failure rule | `delta_concentration_by_site`, `control_failed` | Failed if the site probe is flagged and one site contributes > 60% (PLACEHOLDER) of the pooled gain. |
| Stratified Delta | `stratified_delta` | Delta by duration/channel strata. |
| Sedative-excluded subset | `sedative_excluded_rerun` | Removes sedated patients from training and testing, reruns the ladder. |
| Severity strata (H3) | `severity_stratified_delta` | Three strata; tertiles if cut points are not prespecified (flagged in output). |
| Negative controls | `negative_control` (`labels`, `eeg`) | Labels permuted within site x gold role; EEG rows permuted within site (and optional strata). `spurious_gain` = 95% CI entirely below 0. |
| Pretrained exposure | `exposure_accounting`, `exposed_site_delta` | STUB. CBraMod DECLARED from the plan (TUEG; TUSZ/TUAB/TUEV subsets). MORGOTH left UNVERIFIED for a human to fill from the model card. |
| Bundle | `run_mandatory_controls` | All of the above for one primary rung. |

Not built: E3 positive-control analysis, early-EEG subgroup (H5), nested windows (H6), IPW sensitivity.

## Placeholders to replace

MORGOTH loader (`morgoth_findings_frame`: accepts a DataFrame only), CBraMod embeddings (`embedding_frame`: array in), dynamics (`dynamics_frame`), leakage and concentration thresholds, MORGOTH exposure facts.

## Output (`report.py`)

`write_results(path, ladder, controls)` writes JSON through `safe_write_json`: pseudonymous sites (`site_1`...), counts < 11 as "<11", site Delta suppressed when n < 11, NaN as null, per-patient arrays refused.

## Synthetic check

`sortinghat.models.synthetic.make_synthetic_study(signal=..., site_shift=...)`. With a planted signal, every EEG rung has Delta < 0 at all three sites and a 99% CI below 0; with `signal=0`, |Delta| < 0.02 and no rung meets the rule.
