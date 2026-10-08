# Frozen CBraMod embeddings (`sortinghat/embed/`)

The "frozen embeddings" rung of the representation ladder, built for the commercial-clean gap (Delta of the CBraMod-based
rung versus Delta of the MORGOTH-based rung). Pretrained CBraMod is run as a fixed feature extractor on CPU; nothing is
trained or tuned here. Developed and tested on **synthetic EEG only** (CLAUDE.md rule 1); real recordings are embedded by the
human-run `scripts/extract_embeddings.py`. Provenance, licence discrepancy and open counsel questions:
`docs/research/cbramod_provenance.md`.

## Model code and weights

| Item | Decision |
|---|---|
| Code | Official repo `github.com/wjq-learning/CBraMod` (MIT), commit `b9e961003214326972c567eff390e75b0287e32a`. The **minimal inference definition is vendored** in `sortinghat/embed/cbramod_model.py` (attribution + full MIT text in the header; `state_dict` keys unchanged). Vendored rather than imported from a cache so a real-data run needs no clone and no `models` package on `sys.path`. |
| Equivalence | `tests/test_embed_cbramod.py::test_vendored_model_matches_upstream` loads the checkpoint into both the upstream class (clone cached at `~/.cache/sortinghat/cbramod/code`) and the vendored one: identical keys, max abs output difference 0.0. Skips when the clone is absent. |
| Weights | HF `weighting666/CBraMod` `pretrained_weights.pth` (19,775,842 bytes), SHA-256 `0792cb80...7178`, cached **outside the repo** at `~/.cache/sortinghat/cbramod/` (override `SORTINGHAT_CBRAMOD_DIR` or `--weights`). The hash is verified on every load; the file is read with `torch.load(weights_only=True)`. Get it with `python -m sortinghat.embed.weights --fetch` (or copy the file; real-data hosts may be offline, so the extractor never downloads unless `--fetch-weights`). |
| Runtime | CPU PyTorch. `pip install torch --index-url https://download.pytorch.org/whl/cpu` (torch 2.14.1+cpu used here; wheel 196 MB, about 0.7 GB installed). `torch` is optional: importing `sortinghat.embed` and reading embedding parquet (the ladder) need only numpy/pandas/pyarrow. |

## Input conformance (what the checkpoint expects)

From the paper (section 3.1) and the upstream `preprocessing_tueg_for_pretraining.py` / `tuab_dataset.py`:

| Aspect | CBraMod | Here |
|---|---|---|
| Channels | 19, order FP1 FP2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T3 T4 T5 T6 FZ CZ PZ (the positional convolution runs along this axis) | `CBRAMOD_CHANNELS`; our `CANONICAL_19` order differs, so channels are reordered by name |
| Rate / filters | 200 Hz, band-pass 0.3-75 Hz, 60 Hz notch | same (`EmbedConfig.band`); the qEEG rung's 0.5-45 Hz is not used |
| Patch / segment | 200 samples (1 s) per patch; pretraining 30 patches (30 s), downstream TUAB 10 patches | 10-s segments (10 patches), `EmbedConfig.segment_s` |
| Units | uV / 100, no per-channel z-scoring | `x_uV / 100` |
| Reference | TUEG `*-REF` / `*-LE` | HEEDB's is unknown: default **common average over clean channels** (same `common_average_masked` as the qEEG rung); `--reference as_recorded` skips it. SAP item |
| Bad samples | pretraining dropped 30-s samples with any amplitude above 100 uV | policy switch, see below |

## Windows and pooling

t0 and windows are the pipeline's (D-109): primary = t0 + 1 to t0 + 11 min, nested `20s`, `1min`, `2min`, `5min`, `10min` from the
primary start (H6). All windows are multiples of 10 s on one grid, so the 60 segments of the primary window are encoded
**once** and each nested window pools its subset (the `10min` row equals `primary`). Segments are encoded only when a
window that needs them passes QC (or `--compute-failed`).

* Backbone output with the reconstruction head replaced by the identity: 200-d representation per (channel, patch).
* **Segment** embedding = mean over valid channel x patch cells. **Window** embedding = mean over valid segments:
  `emb.cbramod.0 ... emb.cbramod.199` (the prefix the ladder groups into the `cbramod_frozen` rung).
* Optional `--attention`: parameter-free centroid attention over segments, softmax(cosine(segment, window mean) / 0.1), stored as
  `emb.cbramodattn.<j>`. A learned attention pool would be a trained head and belongs inside the ladder's fold-local fit.

## Missing, dead and artefact channels (D-110)

Channels absent from the file or exactly constant over the fetched segment (`io.drop_dead_channels`) are removed and counted as
missing; channel x 2-s epochs failing the QC (`EpochFlags.clean`) are invalid. Every invalid cell is **zero-filled at patch level and
passed through the model's own `mask` argument** (the upstream mask token is all zeros and not trainable, so zero-fill and mask
are the same thing, and match masked pretraining). Invalid cells are excluded from pooling; a segment with under 50 % valid cells
(`min_valid_frac`) is dropped. Stored per window: `emb_valid_token_frac`, `emb_n_absent_channels`, `emb_n_segments`,
`emb_n_used`, `emb_ok`.

**100 uV rule (SAP decision).** `amp_policy="keep"` (default) embeds every valid segment and stores `emb_frac_over_amp`;
`"drop"` removes segments with any |x| > 100 uV (the pretraining rule) and a window left with none gets no embedding. Default is
keep because D-110 aggregates (median std 82 uV in failing windows) say ICU EEG is often above 100 uV: dropping would select
recordings by amplitude, which correlates with the very labels under study. Run `drop` as a sensitivity analysis in a separate
`--out-dir`.

## Running (human, plain terminal or `scripts/heedb_run.sh`)

Synthetic smoke test (any machine):
```
python -m sortinghat.embed.weights --fetch
python -m pytest -o addopts="" -q tests/test_embed_cbramod.py tests/test_embed_script.py tests/test_embed_ladder.py
```
Real data (D-118 aggregate-only streaming; same input list as the feature extractor, one process per shard):
```
HEEDB_AWS_PROFILE=<profile> scripts/heedb_run.sh python3 scripts/extract_embeddings.py \
    --input out/local_only/cohort/eeg_keys.csv --out-dir out/local_only/embeddings --shard 0 --of 4 --torch-threads 1
```
Output (all under `local_only/`, mode 0600): `part-*.parquet` keyed by `recording_id` x `window` (`recording_id`, `onset_offset_s`,
QC and `emb_*` columns, 200 float32 `emb.cbramod.*`), `ledger-*.csv`, and `embed_meta.json` (model commit, weights hash, config; a
directory is never mixed across configs). Resume, `--retry-permanent`, `--retry-reason`, key fallback, `--limit`, `--windows`,
`--no-onset` behave as in `extract_eeg_features.py`. stdout / `--summary` are aggregates with `<11` suppression, including CPU
throughput (wall, fetch and embed seconds per recording, quantiles) and per-window QC-pass, embedded, segments-used,
valid-token and over-100-uV quantiles.

**Throughput (CPU, synthetic 19-channel 200 Hz EDF, 4-core sandbox).** Embedding step (QC + filtering + 12-layer CBraMod on 60
segments) about 2.1 s per recording on 1 thread, 1.5 s on 2-4 threads; the model alone is 1.7 s (1 thread) and 0.96 s (4 threads) for
60 segments. Peak memory about 0.6 GB per process. Real runs add the ranged S3 fetch and the onset search (the dominant cost for
about 1 GB ICU files), so expect roughly 5-10 s wall per recording per process; 4 single-thread shards give about 1,500-3,000
recordings per hour. Real-file sampling rates above 200 Hz add a resampling cost not measured here.

## Ladder wiring

`sortinghat.models.CBRAMOD_FROZEN_RUNG` (`RungSpec("cbramod_frozen", ("emb.cbramod.",), requires=("emb.cbramod.",))`) is unavailable
(never fitted) when no `emb.cbramod.` column exists. `scripts/run_silver_feasibility.py` reads `--embeddings` (default
`<features dir>/../embeddings`), takes the **primary-window** rows with `emb_ok`, and, if the directory has parts, adds two rungs to
both validation schemes and both baselines: `cbramod_frozen` (embedding alone) and `combined_cbramod` (qEEG + connectivity + embedding;
the CBraMod arm of `commercial_clean_gap`). Otherwise both are skipped silently and the report is unchanged. The headline rung
`combined` stays qEEG + connectivity; the leakage probes for it use only those columns, with a separate probe for the embedding
(`leakage_probes_cbramod_frozen`). Rows without an embedding stay NaN (fold-local median imputation plus a missingness indicator, as
for every other column); coverage is reported suppressed under `frozen_cbramod`.

## Limits

* The checkpoint is pretrained on all of TUEG (including TUAB and TUEV), so a rung that beats qEEG may partly reflect a model that
  has seen similar clinical EEG; licence and weights-terms questions for a regulated commercial product are open (provenance doc,
  section 6). This rung says nothing about MORGOTH's weights; the gap is a reported comparison, not a test.
* Reference convention, the 100 uV policy and 10-s segmentation are choices to confirm in the SAP; each is one config field.
* Mean pooling over channels discards spatial layout; a head over per-channel or per-region pooled embeddings is a possible later rung.
* Tests use a seeded, untrained 2-layer model for speed and portability; only `test_checkpoint_hash_and_real_weights_embedding` and the
  upstream-equivalence test use the real checkpoint (skipped when it is not cached).
