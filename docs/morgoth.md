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

**Not done, needs a human:** the list files have not been opened by an agent (patient-level). Open points to confirm before using the flag:
1. download the lists (`aws s3 cp` via the wrapper) into `out/local_only/morgoth_lists/`, then run `exposure --inspect --lists ...` (prints header
   names only) and confirm the id columns; the file name `datasets_deidentified_list.xlsx` suggests the ids may be re-keyed, in which case
   no match is possible and the output carries a "no patient ids parsed" warning instead of a silent zero;
2. "in a list" means available to MORGOTH's development (train, validation or test), not necessarily trained on; the narrower flag needs a split column;
3. `--pretrain-names` also takes ids embedded in the 9,242 key names under `morgoth1/data/pretrain/` (a listing of names only), if those names carry ids.
The models-layer registry (`controls.default_exposure_registry`) still reports MORGOTH as UNVERIFIED (a test pins it); fill it from the flags once confirmed.

## 6. Limits and open items

* Real-recording behaviour is untested; on synthetic EEG the real heads respond plausibly (burst-suppression head higher on synthetic burst suppression) but
  synthetic EEG is not a validation.
* Binary-head polarity (the second class in "A / B", i.e. abnormal / burst suppression / spike) is read from the README, not checked on labelled data.
* Differences from the reference: `scipy.signal.resample_poly` instead of `mne.filter.resample` for rates other than 200 Hz; float32 instead of CPU bfloat16 autocast;
  snippet step 5 s (the reference uses 1 s with extra smoothing for spikes/IIIC that is not applied).
* EEG-level heads (17 findings, whole-recording) not implemented: they need the 1-s-step event sequences and the repo's `EEG_level_head.py` pipeline.
* Licence: CC BY-NC 4.0 code and credentialed weights make this a research-only rung; it cannot back a commercial build (hence the commercial-clean-gap pairing with CBraMod).
* The morgoth2 (202605) models, baby_morgoth and the NESI release were not used.
