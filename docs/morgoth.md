# MORGOTH rung (research-only lineage)

Module `sortinghat/morgoth/`, extractor `scripts/extract_morgoth.py`, tests `tests/test_morgoth*.py`. Status: **weights and code
obtained, inference verified on CPU against the real checkpoints (synthetic EEG only); not yet run on any real recording.**

## 1. Where MORGOTH lives (names-only probe, 2026-10-08)

Probe: `sortinghat/morgoth/probe.py` (`Delimiter='/'` listings, names and sizes of code/weights/config/list files only; labelled
data and recording prefixes not entered; no object body read). Project-lead authorised.

* BDSP "MORGOTH 1.0: A Foundation Model for Clinical EEG - Data and Code", v1.0.0, 2 Jun 2026, Database Credentialed
  (`https://bdsp.io/content/morgoth1/1.0.0/`). Code licence **CC BY-NC 4.0** (commercial use prohibited); data/weights
  under the BDSP Credentialed Health Data License 1.5.0 (per the project page; read it before relying on this).
* Access point: **projects** (`arn:aws:s3:us-east-1:184438910517:accesspoint/bdsp-credentialed-projects-ap`).
  * `morgoth1/models/` : 20 checkpoints (the v1.0 release): event-level `BS, FOCGENSPIKES, IIIC, NORMAL, SLEEP, SLEEPPSG_6class,
    SLOWING, SPIKES` (about 70 MB each) and EEG-level `*_EEGlevel.pth` (BS, FOC_SLOWING, FOC_SPIKES, GEN_SLOWING, GEN_SPIKES,
    GPD, GRDA, LPD, LRDA, NORMAL, SEIZURE, SPIKES; about 2.2 MB each).
  * `morgoth1/data/pretrain/` (SSL recordings, 9,242 `.mat` per the project page; not entered) and
    `morgoth1/data/internal_dataset/<TASK>/` for 30 tasks (ABNORMAL, AWAKE, BETS, BIPD, BIRD, BS, FOCALSLOWING, GENSLOWING, GPD,
    GRDA, IIIC, LPD, LRDA, MoE, Morgoth_test_dataset_10s, N1/N2_19Channel, NORMAL, PDR, POSTS, SEIZURE, SEIZURE_BCH,
    SLEEPSTAGING, SPIKES, SPIKES_BCH, SPIKES_FOCAL_GEN, SPIKES_HM, SPINDLES, VW, WICKETS). Each task folder holds
    `segments_*` folders (not entered) and one or two **event-list workbooks** (`list_events_*.xlsx`, `list_*.csv`; 12 KB to 1.8 MB),
    plus `morgoth1/data/internal_dataset/datasets_deidentified_list.xlsx` (30.8 MB). SLEEPSTAGING has `train/ val/ test/ list/` folders.
  * `morgoth2/models/202605/morgoth/` : a later set (37 `.pth` listed: adds BIPD, BIRD, PD, SPIKE_LOCALIZATION, SPIKE_1CHANNEL,
    VW, SLEEPPSG variants, `*_EEGlevel_2` ...). `morgoth2/data/{external_dataset,internal_dataset}/` (BCH_seizures, vepiset,
    ELROND, Growth_curves, SZ_HM, breach, centaur ...). `baby_morgoth/{model,dataset,elrond_prepared,results}/`. Not used here:
    v1.0 (`morgoth1`) is the documented release, and exposure lists must match the weights used.
  * `morgoth-slowing/` : the earlier programme's derived-analysis outputs (README, LICENSE, `derived/`, `manifest/`); not walked.
    (Its first listing printed a few numeric file names such as `1.csv` before the probe's name filter was tightened; they are
    ordinal indices, not patient ids.)
* **No code in S3.** Code is `https://github.com/bdsp-core/morgoth` (public), pinned to commit
  `17eab93182695dd94e5e4cc3b2a6f3af7e32bd41`; cloned into the cache and imported from there (never vendored into this repo).
* No MORGOTH paper could be located by web search (as in `docs/research/novelty_sweep.md`); the spec below is from the README and code.

## 2. Model spec (from the pinned code)

Input: 19 referential 10-20 channels in the order `FP1 F3 C3 P3 F7 T3 T5 O1 FZ CZ PZ FP2 F4 C4 P4 F8 T4 T6 O2` (classical T3/T4/T5/T6),
200 Hz. Preprocessing (`utils.ContinuousToSnippetDataset`, `finetune_classification.predict`): resample to 200 Hz, 4th-order
Butterworth band-pass 0.5-70 Hz (zero phase), 50 and 60 Hz notch, 10-s snippet, common average, clip +-500 uV, per-channel
min-max to [-100, 100], divide by 100. Absent electrode: zero-filled after the common average (`--allow_missing_channels yes`).
The 1-s spike head resamples to 128 Hz first when the source is faster. Backbone `morgoth_backbone_base` (patch 200, 12 layers, width 200,
about 17 M parameters), electrodes embedded by index in `standard_1020`, so any channel subset is legal.
`preprocess.py` reproduces the filter and snippet scaling exactly at 200 Hz (checked against the reference functions: max difference 0
and 2e-7).

Outputs (README order; binary heads: sigmoid, multi-class: softmax):

| head | file | window | classes |
|---|---|---|---|
| `normal` | NORMAL.pth | 10 s | abnormal (binary) |
| `bs` | BS.pth | 10 s | burst suppression (binary) |
| `spikes` | SPIKES.pth | 1 s | spike (binary) |
| `slowing` | SLOWING.pth | 10 s | none / focal / generalized |
| `spikeloc` | FOCGENSPIKES.pth | 10 s | none / focal / generalized spikes |
| `iiic` | IIIC.pth | 10 s | other / seizure / LPD / GPD / LRDA / GRDA |
| `sleep3` (off by default) | SLEEP.pth | 10 s | awake / N1 / N2 |

There is **no** IIC-pattern-specific burden other than `iiic`; no sedation or encephalopathy-severity head. The 12 EEG-level heads
(17 binary EEG-level findings in the project page) take the per-second event-level sequence through a CNN+transformer
(`EEG_level_head.py`); they are downloaded but **not run in v1** (see section 6).

Features (`features.py`): per recording x window (primary = t0 + 1 to t0 + 11 min, D-109 t0; nested 20 s, 1, 2, 5, 10 min):
`morgoth.<head>.<class>.{mean,p90,burden}` (burden = share of valid snippets with p >= 0.5) for every class except the reference class of a
multi-class head: 12 probabilities x 3 = 36 columns, plus `n_missing_channels`, `morgoth_valid_fraction`, `qc_pass`, `usable_fraction`,
`onset_offset_s`. One model pass over the primary window serves the nested windows. Snippet step defaults to 5 s (`--step-s`; the
reference's EEG-level step is 1 s). Windows failing the project QC (>= 60% usable, >= 8 of 10 minimum electrodes) get NaN features.
CPU cost with the real weights: roughly 7 to 40 s per recording at a 10-s step on 4 threads (60 snippets x 5 heads plus 600 one-second spike snippets).

**Heads run by default, and speed (2026-10-09).** `scripts/extract_rungs.py --morgoth-heads` defaults to the heads the plan needs: `normal`,
`bs`, `slowing`, `iiic` (`heads.plan_heads()`). `spikes` (600 one-second snippets) and `spikeloc` (a fifth 10-s head) are **E3 positive-control
extras**: about 35% of the model time and not used by the ladder; add them with `--morgoth-heads normal,bs,spikes,slowing,spikeloc,iiic`. The columns
and values of the heads that run are unchanged (they are a subset of the six-head columns; tested). A `--morgoth-dir` holds one head set
(`run_info.json`): to resume a directory begun with the old six-head default, pass the six names, or start a new directory.
Measured on one CPU thread, synthetic 700-s recording, all six windows, CPU seconds per recording (CBraMod + MORGOTH, noisy shared machine):
old 52 (six heads, 64 snippets per pass, QC run once per rung, one normalisation per head); same six heads after the changes 44 (1.2x, outputs
equal to float rounding, max abs difference 3e-7); default four heads 33 (1.6x); default four heads with `--step-s 10` 19 (2.8x; this changes the
features, it is a coarser snippet grid, and is not the default). Each model already ran once over the primary window and the nested windows pooled
its per-snippet outputs; what changed is shared QC / channel selection, one normalised tensor shared by the 10-s heads, a vectorised normaliser,
8 snippets per forward pass (small batches stay in cache; 64 was ~20% slower on 1 thread), `torch.inference_mode`, and per-stage timers in the
run summary. About 87% of what remains is the four 10-s transformer forward passes (~7 s each at a 5-s step), which float-exact CPU tricks did
not reduce (tried: `torch.compile` with freezing 8%, hand-written attention 0%, 2 threads per shard 1.1x speed for 1.4x CPU). Keep 4 shards x 1 thread.

## 3. Weights obtained

Downloaded 2026-10-08 with `scripts/heedb_run.sh python3 -m sortinghat.morgoth.weights --fetch --eeg-level --heads normal,bs,spikes,slowing,spikeloc,iiic,sleep3`
into `~/.cache/sortinghat/morgoth/{weights,code}` (outside the repo, mode 0600), manifest `manifest.json`. `weights --verify` re-hashes.

| file | bytes | sha256 |
|---|---|---|
| `BS.pth` | 70,108,410 | `6a7c6e19eef4af4481c5ad9095fbd2ce207bd362cc570a5ae5b2c14de9c16091` |
| `BS_EEGlevel.pth` | 2,222,778 | `fa65acf48c3dd32c249fee78b6be59650eb78cf416c641437d67c6cff7fb89a6` |
| `FOCGENSPIKES.pth` | 70,113,402 | `f278ef996c19541030a6996412c2a6964ba138e89f934645935caf242e2280b7` |
| `FOC_SLOWING_EEGlevel.pth` | 2,223,450 | `02fe53a395e9647663df59ece619b824a0309d724e6472cacb963a4799b5401b` |
| `FOC_SPIKES_EEGlevel.pth` | 2,223,450 | `bd8fa2aee05d3f86a6379ebb3c66b5c13d2b3d56892f1a72d3e03fe02f4c383c` |
| `GEN_SLOWING_EEGlevel.pth` | 2,223,450 | `a6055ba4e0e7bb35fd991742acbe8636e45bc3b524e9f4ea86202f5ca0a14b55` |
| `GEN_SPIKES_EEGlevel.pth` | 2,223,450 | `2b00e98e402950c1e959a3d45935ff516f8fb37058446f1c9049164e0e60fcb6` |
| `GPD_EEGlevel.pth` | 2,225,658 | `feb7f0bde9150fa2bba0aa023adff2137816647b1b27d8d4c89576bd6041f88a` |
| `GRDA_EEGlevel.pth` | 2,225,658 | `a85e7d38014985afb90f489e964461567b4c35dde3e5313f51a7aef6de1df122` |
| `IIIC.pth` | 70,120,570 | `f2dbed82f6dcd971a9a5a8266fbfeca38efc019e30303f90fef8d92a9501965d` |
| `LPD_EEGlevel.pth` | 2,225,658 | `769898b9334514f97dab553e8a5147d5dc56106236db7320e69fc131ac5eae4e` |
| `LRDA_EEGlevel.pth` | 2,225,658 | `e722f6a0f9663117ae4188e96baf897198d402469849d7d30a2fa69328e39396` |
| `NORMAL.pth` | 70,108,410 | `20af6465a7b7a26ef537f30dfe8f062bdd90bda7d8d304b2b86c07b5e0e1508f` |
| `NORMAL_EEGlevel.pth` | 2,222,778 | `54fc4e2970a00accb1a1af3589e12583eee50aef18a96708a11f436d5fac617d` |
| `SEIZURE_EEGlevel.pth` | 2,225,658 | `d0c27ed9c056eab951bc7eb77f356e605a15acdbc86d07e5657a9ab4dfb5a784` |
| `SLEEP.pth` | 70,113,274 | `09cd697471a1d388140b4122ea88193b4dbc62e974f06cf3b707b6bcf179783c` |
| `SLOWING.pth` | 70,113,338 | `45a841d75615b3eb79758adf0850b15f9dc4642ddfae4f388546def883358549` |
| `SPIKES.pth` | 70,108,410 | `e0a24884a3bedc15261ca01f223f7f8af407a64381022e773522a1fd2712df57` |
| `SPIKES_EEGlevel.pth` | 2,222,778 | `3b94e68af32c3310fd4583b93238bf66d9faa25f8eafe1833450d9ef8875dbcd` |

Checkpoints hold an `argparse.Namespace` and numpy scalars, so `torch.load(weights_only=True)` refuses them; `model.load_checkpoint` uses a
restricted unpickler that admits only the globals the files reference (tested to reject anything else). The model is built with the
checkpoint's own training arguments (`qkv_bias=False`, no relative position bias): `load_state_dict` is strict and matches for all 7 heads.

## 4. Running it

Dependencies for real inference (not in `pyproject.toml`, torch is optional; the stub and all tests need none of them):
`pip install torch --index-url https://download.pytorch.org/whl/cpu timm einops`. The README pins torch 2.4.1 / timm 1.0.11; torch 2.14.1 and timm 1.0.30
worked. Weights step needs only BDSP credentials.

```
# once (plain terminal or the D-118 wrapper)
scripts/heedb_run.sh python3 -m sortinghat.morgoth.weights --fetch

# features: same input list as the EEG extractor; one process per shard
HEEDB_AWS_PROFILE=<profile> scripts/heedb_run.sh python3 scripts/extract_morgoth.py features \
    --input out/local_only/recording_keys.csv --out-dir out/local_only/morgoth --shard 0 --of 4 --threads 4

# ladder: add the rung when the parquet exists (None -> rung skipped cleanly)
frame = models.load_morgoth_findings("out/local_only/morgoth", recording_ids)      # None if absent
eeg = models.assemble_eeg_frame(qeeg_frame, frame)                                  # ignores None
run_ladder(data, rungs=DEFAULT_RUNGS + (MORGOTH_FINDINGS_RUNG,) + COMMERCIAL_RUNGS)
```

`--out-dir`/`--input` must be under `local_only/`; parts and ledger are resumable exactly like the EEG extractor (`--retry-permanent`,
`--retry-reason`); `run_info.json` in the out dir records backend, heads, steps, code commit and weight hashes and refuses a mixed
configuration. `--backend stub` (needs `--allow-stub` on a real run) exists for tests only. stdout/`--summary` carry suppressed counts
and quantiles of the primary-window features only.

Ladder wiring: `models.ladder.MORGOTH_FINDINGS_RUNG` (`morgoth_findings`, prefix `morgoth.`, requires a `morgoth.` column) follows the
`cbramod_frozen` pattern: not in `DEFAULT_RUNGS` (whose generic `morgoth` rung and `combined_morgoth` already accept `morgoth.` columns),
unavailable and never fitted when the parquet is absent. `scripts/run_silver_feasibility.py` still lists MORGOTH as skipped (not touched).

## 5. Exposure accounting (plan control: "MORGOTH exposure accounting")

`sortinghat/morgoth/exposure.py` + `scripts/extract_morgoth.py exposure`. The release exposes only the **data lists**
(`internal_dataset/*/list*.xlsx|csv`, `datasets_deidentified_list.xlsx`) and the pretraining file names, not a train/validation/test split of
HEEDB patients per se (the project page lists none). The helper scans whatever lists a human downloads for HEEDB patient ids
(`[SI]dddd` + 6 or more digits in any text cell, `sub-` form included, or a numeric column named patient/person/bdsp id), reduces them to the
integer BDSPPatientID, and flags each cohort patient (`person_id`, or the pre-merge `person_id_source`, optionally through a merge map):
`in_morgoth_lists`, `in_morgoth_train_split` (only when a split column exists), `n_morgoth_lists`. The per-patient table is written only under
`local_only/` (0600); stdout carries suppressed counts, proportions and per-site (pseudonymised) shares.

**Run on the real release (2026-10-09, D-118 wrapper, aggregate output only; per-patient flags in `out/local_only/morgoth/exposure.parquet`, 0600).**
Lists: 29 files downloaded from the projects access point into `out/local_only/morgoth_lists/` (0700/0600): `morgoth1/data/internal_dataset/datasets_deidentified_list.xlsx`
(18 sheets) and the per-task event lists `internal_dataset/<TASK>/list*.xlsx|csv` (BETS, BIPD x2, BIRD, BS, FOCALSLOWING, GENSLOWING, GPD, GRDA, IIIC, LPD, LRDA, N1/N2_19Channel,
PDR x2, POSTS, SEIZURE, SEIZURE_BCH, SPIKES, SPIKES_BCH, SPIKES_FOCAL_GEN, SPIKES_HM, SPINDLES, VW x2, WICKETS x2). NORMAL, ABNORMAL, AWAKE and `MoE/` have no list file at their top level
(label sets only); `SLEEPSTAGING/list/*.csv` (BCH/MGH PSG sets, not the EEG cohort's source) was not downloaded or checked. `--pretrain-names` listed the 9,242 key names under `morgoth1/data/pretrain/`
(names only, in memory): every name embeds one patient id (about 1,565 cohort patients).

Layout (header names plus aggregate value classes; no values read): list columns are `bdsp_mrn` (integer BDSPPatientID, 9 digits, equal to cohort `person_id`),
`file_name` (`sub-<SITE><id>_...` or `<SITE><id>_<n>_...`), `event_time`, label and rater columns. The workbook's sheet 1 is a **master table of 108,666 HEEDB patients** (`BDSPPatientID`,
`SiteID`, `HasEEG` ...) with a `Morgoth` column that is blank for 81,491 patients and `pretrain` (10,000) / `train` (7,081) / `test` (10,094) for the others, plus `SpikeNet` / `SparcNet`
columns (other models, ignored). The other sheets repeat the `Morgoth` column as train or test per task. Consequence, and the fix made in `exposure.py`: presence in the master table is NOT
exposure; a column named `Morgoth` is the membership column (blank rows dropped, value = split). Cohort ids match directly: 13,219 of 14,516 cohort ids appear in the master table, 5,783 of them
as MORGOTH members (the per-task lists add 636 more cohort patients whose master `Morgoth` cell is blank, so the union is broader than the master column alone).

Id mapping: `person_id` (= int BDSPPatientID, 9 digits) or the pre-merge `person_id_source` against the integer id from `bdsp_mrn` / `BDSPPatientID` / the `file_name` regex (site code ignored).

Flag columns (`flag_cohort`): `in_morgoth_lists` (any pretrain / train / test membership or any per-task list or pretrain key name; the conservative flag
`run_silver_feasibility.py` uses), `in_morgoth_train_or_pretrain`, `in_morgoth_pretrain`, `in_morgoth_test_only`, `in_morgoth_train_split` (NaN here: the per-task lists carry no split, so the
narrow flag is withheld and readers fall back to `in_morgoth_lists`), `n_morgoth_lists`.

Cohort result (n = 14,516 patients, `cohort_study1.csv`): in any MORGOTH list **6,425 (44.3%)**; train or pretrain (split-labelled, a lower bound for training exposure) 2,794 (19.3%); pretrain 1,712 (11.8%);
known test-only <11. By pseudonymised site: site 1 (n 1,504) 3.1%, site 2 (n 12) <11, site 3 (n 8,032) 51.1%, site 4 (n 4,968) 45.8%. Per-list counts are printed by the command (largest: the master
sheet's test 3,070, pretrain 1,712, train 1,001; SPIKES_FOCAL_GEN 1,626; GENSLOWING 1,438; IIIC 876).

Caveats still open: (1) "in a list" means available to MORGOTH's development (train, validation or test); only the master/per-sheet `Morgoth` column separates train from test, and the per-task
event lists (the head fine-tuning sets) have no split. (2) The heads this project runs (`normal`, `bs`, `slowing`, `iiic`) are tied to specific task lists; the all-list flag is the safe sensitivity.
(3) A bare numeric `id` column (workbook sheet 13, with `SiteID`) is not read as an id column; it would add about 60 cohort patients not otherwise flagged. (4) Whether the pretraining
recording ids are the same patients as the master `pretrain` rows is not checked beyond the counts above. The models-layer registry (`controls.default_exposure_registry`) still reports MORGOTH
as UNVERIFIED (a test pins it; not edited here); fill it from these flags.

## 6. Limits and open items

* Real-recording behaviour is untested; on synthetic EEG the real heads respond plausibly (burst-suppression head higher on synthetic burst suppression) but
  synthetic EEG is not a validation.
* Binary-head polarity (the second class in "A / B", i.e. abnormal / burst suppression / spike) is read from the README, not checked on labelled data.
* Differences from the reference: `scipy.signal.resample_poly` instead of `mne.filter.resample` for rates other than 200 Hz; float32 instead of CPU bfloat16 autocast;
  snippet step 5 s (the reference uses 1 s with extra smoothing for spikes/IIIC that is not applied).
* EEG-level heads (17 findings, whole-recording) not implemented: they need the 1-s-step event sequences and the repo's `EEG_level_head.py` pipeline.
* Licence: CC BY-NC 4.0 code and credentialed weights make this a research-only rung; it cannot back a commercial build (hence the commercial-clean-gap pairing with CBraMod).
* The morgoth2 (202605) models, baby_morgoth and the NESI release were not used.
