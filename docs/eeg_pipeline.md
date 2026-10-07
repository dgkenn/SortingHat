# Study 1 EEG front end (`sortinghat/eeg/`)

EDF in, one qEEG/connectivity feature vector per recording x window out. Built and tested on **synthetic signals
only** (CLAUDE.md rule 1). Real HEEDB EDFs are processed by a human from a plain terminal; only aggregates leave
those jobs. Tests: `pytest tests/test_eeg_*.py`.

## Data layout and lessons applied

HEEDB EDFs live at `EEG/bids/{SITE}/{BidsFolder}/ses-{SessionID}/eeg/{BidsFolder}_ses-{SessionID}_task-{cEEG|EEG}_eeg.edf`
(`docs/heedb_access.md`). Applied from the earlier programme:

| Lesson | Where |
|---|---|
| `mne` returns volts, CBraMod expects uV; per-channel z-scoring destroyed amplitude information | `io.read_edf` always returns uV; no z-scoring anywhere. CBraMod's own input scaling belongs to the embedding step |
| ICU cEEG EDFs are ~1 GB; read a bounded window | `read_edf(start_s, duration_s)` seeks to the needed records; accepts any seekable binary object (e.g. a ranged-S3 wrapper). `pipeline` reads only minutes 1-11 plus 10 s padding |
| A mask that compresses bad samples glues time together (rule 27) | QC masks live on a fixed 2-s epoch grid and are never compressed; uncovered time counts as unusable; EDF+D (gapped) files are refused |
| Parse BIDS/ID fields, do not substring-match (rule 61) | The reader never uses patient/recording header fields or file names; recordings are keyed by a caller-supplied opaque `recording_id` |
| One record per patient, first EEG | Cohort selection is upstream; this package processes the files it is given |

## Modules

* `io.py`: built-in EDF reader (no dependency; cross-checked against `pyedflib` in tests when installed). Channel
  normalisation strips `EEG ` and `-REF/-LE/-AV/-AVG/-AR`, case-insensitive, maps T7/T8/P7/P8 to canonical
  **T3/T4/T5/T6**. Bipolar, ECG, EOG, annotation signals are dropped. `CANONICAL_19`, `check_channel_set`,
  `select_channels` (optionally NaN-fills missing rows). Header patient/start-date fields are not retained.
* `window.py`: windows, artifact QC, usable-data rule, aggregate summary.
* `preprocess.py`: resample, notch, band-pass, common average, double banana.
* `features.py`: the qEEG and connectivity rungs.
* `synthetic.py`: generator + EDF writer for tests (`normal`, `slowing`, `burst_suppression`,
  `periodic_discharges` generalised/lateralised, `low_voltage`; artifacts `flat`, `clipping`, `extreme`,
  `line_noise`, `disconnected`).
* `pipeline.py`: `process_recording` and a batch CLI.

## Windows (plan: t0 = EEG start)

Primary window = minutes 1-11 (60-660 s). Nested windows start at the primary start: `20s`, `1min`, `2min`,
`5min`, `10min` (the 10-min nested window equals the primary window by construction). A window passes when
**>= 60 % of its intended duration is usable** on the minimum channel set.

## QC (2-s epochs, raw uV, before filtering)

| Flag | Rule (default, `QCConfig`) |
|---|---|
| flat | peak-to-peak < 0.5 uV (or NaN) |
| clipping | >= 5 % of samples sit exactly on the epoch max or min |
| extreme | max abs deviation from epoch median > 500 uV |
| line_noise | power at 60 +/- 1 Hz > 1.0 x power 1-40 Hz |
| disconnected | epoch std < 2 % of the cross-channel median, **or** channel is flat/clipped/line-dominated in >= 50 % of epochs over the QC span (then flagged for the whole span) |

An epoch is usable when >= 90 % of the minimum set (ceil, so 9 of 10 channels) is clean in it; one bad channel
does not sink an epoch, two do. Missing minimum channels fail the window. `window_clean_mask` exports the
per-channel clean mask used by the features.

**Decisions to confirm in the SAP** (all are constructor/config fields):
1. *Minimum channel set* (D-096): the 10 Ceribell-headband hairline electrodes Fp1, Fp2, F7, F8, T3, T4, T5, T6,
   O1, O2 (`DEFAULT_MINIMUM_CHANNELS`, defined in `sortinghat/eeg/io.py` and imported by `window.py`); the other nine
   10-20 channels are optional for QC. CBraMod uses all 19.
2. Epoch rule: 90 % of the minimum set clean (`epoch_channel_frac`).
3. Thresholds above, especially line-noise ratio and the 0.5 uV flat limit, are untuned. Calibrate on a pilot
   with counts only, then freeze (D-097; the 5 uV burst-suppression threshold below is in the same category).

**Aggregate QC output**: `summarize_window_qc` / `write_qc_summary` produce, per window, suppressed n, pass
proportion (`suppress_proportion`), usable-fraction quantiles (`safe_quantiles`, n >= 11) and flag prevalence,
checked with `assert_aggregate_only`, written via `safe_write_json`. No per-recording values, no min/max.

## Preprocessing

`preprocess()` = resample to **200 Hz** (CBraMod input rate; `docs/research/cbramod_provenance.md`) -> 60 Hz notch
(IIR, Q = 30, zero-phase) -> Butterworth band-pass **0.5-45 Hz**, order 4, zero-phase (configurable via
`PreprocessConfig.band`; CBraMod's own pipeline used 0.3-75 Hz, so a frozen-embedding run may want its own config
and a different `band`). Derivations: `common_average_masked` (time-varying reference over channels clean in each
epoch, so a one-channel transient cannot leak into the rest) and `bipolar_double_banana` (18 longitudinal
channels, defined locally; it does not import `sortinghat.montage`). Note CBraMod's 100 uV bad-sample rule
differs from the 500 uV QC rule here; the embedding step must apply its own.

## Features (`features.extract_features`)

Input: window of CAR-referenced, filtered uV at 200 Hz plus the clean mask. 4-s Hann segments, 50 % overlap; a
segment counts for a channel only if both its epochs are clean. Fewer than 5 clean segments gives NaN.
Names are fixed (`feature_names()`), so every row has identical columns; regions without data are NaN.

* `qeeg.<scope>.<metric>`, scope = `global`, `frontal`, `central`, `temporal`, `parietal`, `occipital`
  (`FeatureConfig(per_channel=True)` adds `ch_<name>`). Metrics: `{delta,theta,alpha,beta}_abs_log10`
  (log10 uV^2), `{band}_rel` (of 0.5-30 Hz), `sef95`, `sef50` (0.5-40 Hz), `alpha_delta_log10`,
  `slowing_log10` ((delta+theta)/(alpha+beta)), `amp_rms`, `amp_p95`, `amp_kurtosis`, `line_length`, `lzc`.
  Bands: delta 0.5-4, theta 4-8, alpha 8-13, beta 13-30. `lzc` = Lempel-Ziv complexity (Kaspar-Schuster) of
  4-s segments binarised at their median, normalised by n/log2(n), up to 10 segments per channel.
* `qeeg.global.bsr`: burst-suppression ratio, median over channels of the clean-time fraction in which the 0.25-s
  RMS envelope stays < 5 uV for >= 0.5 s. **The 5 uV threshold is an assumption to calibrate**; low-voltage
  traces read as suppressed.
* `qeeg.asym.{delta,alpha}_absdiff_log10`: mean |log10 L/R| power over 8 homologous pairs (lateralisation).
* `conn.{coh,wpli}.<A>_<B>.<band>` over 14 fixed pairs (8 homologous, 6 anteroposterior) from the same
  segments (magnitude-squared coherence; Vinck wPLI, band-averaged), plus `conn.<m>.mean_{all,interhemispheric,
  anteroposterior}.<band>`. CAR reduces but does not remove volume conduction; wPLI is the lag-sensitive measure.

Periodic discharges have no dedicated feature at this rung; they show in kurtosis, line length and the
broadband/delta terms and are left to the MORGOTH rung.

## Running (human, plain terminal)

```
python -m sortinghat.eeg.pipeline --manifest local_only/eeg_manifest.csv \
    --out local_only/eeg_features.csv --qc-summary out/eeg_qc_summary.json
```
Manifest columns: `recording_id` (opaque), `edf_path`. Per-recording feature rows go only under `local_only/`
(mode 0600). Stdout and the QC JSON are aggregate-only; read errors are counted, never described (no paths/IDs).
Feature rows for windows that fail QC contain only `window, qc_pass, usable_fraction` unless `--compute-failed`.
Runtime is dominated by complexity (pure Python), on the order of 10 s per recording for all six windows (sandbox measurement on synthetic data);
an optional `numba` step is a possible later optimisation. Install the optional reader cross-check with
`pip install -e .[eeg]`.

## Real-data hardening (labels, calibration, dead channels, key resolution)

* **Labels** (`io.normalize_channel_name`): `Fp1`, `FP1`, `EEG Fp1-Ref/-REF/-REF1`, `EEG T7`, `POL Fp1`, `Fp1-AVG/-LE/-AV`,
  `C3-A2` / `T5-M1` (ear or mastoid reference), `Fp1-G2`, `EEG Fp1 - Ref` map to the canonical name (T7/T8/P7/P8 -> T3/T4/T5/T6).
  Bipolar labels (`Fp1-F7`, `C3-Cz`) are NOT referential and map to nothing (`io.is_bipolar_label` counts them); a file with
  only bipolar chains fails as `no_eeg_channels`. Two signals mapping to one channel no longer abort the read: the first wins
  (`meta['duplicate_channels']`).
* **Calibration** (`io.channel_scaling_status`): a signal with physical range 0 (pmin == pmax, e.g. 0/0) or digital range 0
  decodes to a CONSTANT whatever the samples are. Such a channel is dropped and listed in `meta['invalid_scaling_channels']`
  (reason `invalid_scaling_minimum_channels`), never counted as a flat electrode; if every EEG channel is like that the stream
  result is `edf_invalid_scaling`.
* **Dead channels** (`io.drop_dead_channels`, run by `pipeline.process_recording`): a channel that is EXACTLY constant (or NaN)
  over the whole fetched segment (zero-filled placeholder, unplugged input) is removed and reported as missing
  (`meta['dead_channels']`, reason `dead_minimum_channels`), not as flat data. Partly flat channels are still flagged by the QC.
* **EDF+**: the `EDF Annotations` signal is skipped; `EDF+C` reads; `EDF+D` is refused (`edf_discontinuous`). `EDFHeader.edf_type`
  and `Recording.meta['edf_type']` record which.
* **Key resolution** (`data_io.resolve_edf_key`, used by `scripts/extract_eeg_features.py` when a documented key is `not_found`):
  documented key, other task token, task-less name, SessionID spellings (`12.0` -> `12`), then a listing of ONLY the recording's own
  `ses-<id>/eeg/` folder (`Delimiter='/'`); `parent_fallback` (off) also lists the subject's folder.
* **Streaming summary** now also prints counts of recordings with a missing / dead / zero-calibration minimum-set channel and the
  number recovered by the key fallback (by pattern name).

### t0 = start of the first sustained live segment (D-109, supersedes D-108)

HEEDB files can start with constant padding or a short live blip followed by a hold, so the file start is not the EEG start
(a pyedflib cross-check on real data showed the reader is exact; the constant epochs are genuine). `stream.find_signal_onset`
returns the first 60-s period (10-s grid from the file start) in which at least 8 of the 10 required electrodes are non-constant
(digital samples) in at least 90% of their 2-s epochs (27 of 30), searched within the first 120 min. The search is exact and cheap:
every window contains exactly one probe block (one 10-s block per minute), and a qualifying channel has at most 3 constant epochs,
so a probe block with fewer than 8 channels having >= 2 non-constant epochs (of 5) cannot belong to a qualifying window. Only a probe
that passes is refined (its 6 candidate windows are fetched once and tested exactly, in order), so long padding costs about a sixth
of its bytes. None -> failure `no_sustained_signal` (permanent). `stream_features` places every window relative to t0 (primary =
t0 + 1 to t0 + 11 min); `StreamResult.onset_s` and the `onset_offset_s` column of the local_only feature parts hold the offset in
seconds from the FILE start (per recording, never printed; stdout shows quantiles). **The cohort / feature join must use
t0 = metadata start + onset offset**; `stream.onset_offset_s(s3, key)` returns the offset alone (None = excluded). `--no-onset`
on the extractor restores file-start windows.

### Diagnostics (human-run, aggregates only)

```
HEEDB_AWS_PROFILE=<profile> python3 scripts/diag_eeg_paths.py   --site S0001 --n 60 --seed 0
HEEDB_AWS_PROFILE=<profile> python3 scripts/diag_eeg_signals.py --site S0001 --n 60 --seed 0
```

`diag_eeg_signals.py` reports both the file-start window and the window after t0: t0-offset quantiles (minutes), the number with
no sustained segment within 120 min, the post-fix usable-fraction quantiles and pass count, the exactly-constant epoch share per electrode after onset, and for recordings whose
usable fraction is still below 0.6 a breakdown by QC rule (flat / clipping / extreme > 500 uV / line noise / disconnected, as the
mean share of minimum-set cells), required-channel amplitude (std, 99th percentile of |x - median|, uV) and line-noise-ratio
quantiles. Everything is aggregate-only.

It also reports reader cross-checks: the number of recordings with heterogeneous samples per record, with an EDF+ annotation
signal, the required channels' calibration strings (dimension, pmin, pmax, dmin, dmax), the share of constant epochs sitting at the
header digital rail vs zero vs another value, and
`pyedflib_crosscheck`: our ranged decode vs pyedflib on the same in-memory records (an anonymous memfd, never a disk file), as
quantiles of max |difference| (uV) and correlation. `tests/test_edf_multirate.py` proves exact agreement on synthetic EDFs with
EEG 256 Hz, ECG 512 Hz, osat 1/record, DC 8 Hz and an annotation signal, record duration 1 s and 0.5 s.

Both sample adult sessions from `eeg_metadata` (identifiers used in code only) and print no key, folder name or ID.
Small cells: counts of TECHNICAL FILE properties (files with a header quirk, folders without a `.edf`, label counts) use
`safe_output.technical_count`: exact when the sample has at least 50 recordings (so "7 of 60" is shown as 7), otherwise n < 11
prints "<11". This exception is for file-layout diagnostics only, never clinical or demographic counts. Pooled epoch fractions
and quantiles still need at least 11 recordings.

## Known limits

* Only plain EDF (and EDF+C) is read; mixed per-signal rates are resampled to the highest rate.
* Artifact rules are heuristics on synthetic signals; real-data prevalence and thresholds need a counts-only pilot.
* The synthetic generator is for direction-of-effect tests, not physiological validation.
