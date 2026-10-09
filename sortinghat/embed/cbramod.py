"""Frozen CBraMod embeddings of the D-109 primary window and the nested H6 windows (representation ladder rung).

One call turns a ``Recording`` (or EDF) into one 200-d embedding per window (``primary``, ``20s``, ``1min``, ``2min``,
``5min``, ``10min``). Everything is CPU inference of the public pretrained checkpoint; nothing is trained or tuned, so no
fold-local fitting is needed here (normalisation / PCA of the embeddings is a fold-local step in the ladder).

CBraMod input conformance (paper section 3.1 and upstream ``preprocessing_tueg_for_pretraining.py`` / ``tuab_dataset.py``):

* 19 channels in the TUEG order FP1 FP2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T3 T4 T5 T6 FZ CZ PZ (``CBRAMOD_CHANNELS``; the
  criss-cross model's positional convolution runs along this axis, so the order matters, our ``CANONICAL_19`` differs);
* 200 Hz, notch 60 Hz and band-pass 0.3-75 Hz (``EmbedConfig``; the front end's default 0.5-45 Hz is NOT used here);
* patch = 200 samples (1 s); a segment is ``segment_s`` patches (default 10 s = 10 patches, the TUAB downstream setting;
  pretraining used 30 s = 30 patches);
* units: microvolts divided by 100 (upstream ``x / 100``); no per-channel z-scoring;
* reference: TUEG ``*-REF`` / ``*-LE`` recordings. HEEDB's reference is unknown, so the default re-references to the
  common average over clean channels (the same ``common_average_masked`` as the qEEG rung); ``reference="as_recorded"``
  skips it.

Missing / dead / artefact channels (D-110): channels absent or exactly constant over the fetched segment, and channel x 2-s
epochs that fail the QC (``EpochFlags.clean``), are ZERO-FILLED at patch level and passed to the model through its own mask
argument (the upstream mask token is all zeros, so zero-fill and mask coincide: exactly what the model saw in masked
pretraining). They are excluded from the pooling (a token average over valid channel x patch cells), and a segment with
less than ``min_valid_frac`` valid cells is dropped. ``emb_valid_token_frac`` and ``emb_n_absent_channels`` are stored.

The 100 uV pretraining bad-sample rule (segments with any |x| > 100 uV were dropped from pretraining) is a POLICY switch:
``amp_policy="keep"`` (default) embeds every valid segment and stores the share of segments above 100 uV
(``emb_frac_over_amp``); ``"drop"`` removes them from the pool (a window with none left gets no embedding). Default is
keep because ICU EEG is often above 100 uV (D-110 aggregates: median std 82 uV in failing windows), so dropping would
select recordings by amplitude. Flagged for the SAP.

Pooling: segment embedding = mean of the 200-d patch representations over valid channel x patch cells; window embedding =
mean over valid segments (``emb.cbramod.<j>``). Optional parameter-free attention pooling over segments
(``emb.cbramodattn.<j>``): softmax of cosine(segment, window mean) / tau. Learned attention would be a trained head and
belongs inside the ladder's fold-local fit, not here.

Cost: all windows start at the primary start and are multiples of the segment length, so the 60 ten-second segments of the
primary window are encoded ONCE and each nested window pools its subset (segments are only encoded when a window that
needs them passes QC, or ``compute_failed``).
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field

import numpy as np

from ..agent_safety import assert_not_restricted_in_agent
from ..eeg.io import Recording, read_edf
from ..eeg.prep import RecordingPrep
from ..eeg.pipeline import PAD_S
from ..eeg.preprocess import PreprocessConfig, common_average_masked, preprocess
from ..eeg.window import QCConfig, WindowQC, WindowSpec, all_windows
from ..timing import stage

CBRAMOD_CHANNELS = ("Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
                    "Fz", "Cz", "Pz")
FS = 200.0
PATCH_SAMPLES = 200                  # 1 s
UV_PER_UNIT = 100.0                  # model units = uV / 100
EMB_DIM = 200
MEAN_PREFIX = "emb.cbramod."
ATTN_PREFIX = "emb.cbramodattn."
REFERENCES = ("car", "as_recorded")
AMP_POLICIES = ("keep", "drop")


@dataclass(frozen=True)
class EmbedConfig:
    segment_s: float = 10.0
    reference: str = "car"
    band: tuple = (0.3, 75.0)
    notch_hz: float = 60.0
    amp_policy: str = "keep"
    amp_uv: float = 100.0
    min_valid_frac: float = 0.5
    attention: bool = False
    attn_tau: float = 0.1
    compute_failed: bool = False

    def __post_init__(self):
        if self.reference not in REFERENCES:
            raise ValueError(f"reference must be one of {REFERENCES}")
        if self.amp_policy not in AMP_POLICIES:
            raise ValueError(f"amp_policy must be one of {AMP_POLICIES}")
        if self.segment_s <= 0 or abs(self.segment_s - round(self.segment_s)) > 1e-9:
            raise ValueError("segment_s must be a positive whole number of seconds (one patch = 1 s)")
        if not 0.0 < self.min_valid_frac <= 1.0:
            raise ValueError("min_valid_frac must be in (0, 1]")

    def meta(self) -> dict:
        d = asdict(self)
        d["band"] = list(self.band)
        return d


def emb_columns(prefix: str = MEAN_PREFIX, dim: int = EMB_DIM) -> list[str]:
    return [f"{prefix}{j}" for j in range(dim)]


ROW_COLUMNS = ("window", "qc_pass", "usable_fraction", "qc_n_missing_or_dead_min", "qc_coverage_fraction", "emb_ok",
               "emb_n_segments", "emb_n_used", "emb_valid_token_frac", "emb_frac_over_amp", "emb_n_absent_channels")


def part_columns(cfg: EmbedConfig) -> list[str]:
    """Fixed parquet columns after ``recording_id`` / ``onset_offset_s``."""
    return list(ROW_COLUMNS) + emb_columns() + (emb_columns(ATTN_PREFIX) if cfg.attention else [])


# ------------------------------------------------------------------------------------------------ encoder
class Embedder:
    """CPU wrapper around a (frozen, eval-mode) CBraMod backbone with the reconstruction head removed."""

    def __init__(self, model, batch_size: int = 30, num_threads: int | None = None):
        import torch
        self.torch = torch
        if num_threads:
            torch.set_num_threads(int(num_threads))
        model.proj_out = torch.nn.Identity()
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        self.batch_size = int(batch_size)

    @classmethod
    def from_checkpoint(cls, path=None, batch_size: int = 30, num_threads: int | None = None) -> "Embedder":
        from .cbramod_model import CBraMod
        from .weights import load_state_dict
        model = CBraMod()
        model.load_state_dict(load_state_dict(path))
        return cls(model, batch_size, num_threads)

    @classmethod
    def random_init(cls, seed: int = 0, n_layer: int = 2, batch_size: int = 30,
                    num_threads: int | None = None) -> "Embedder":
        """Untrained, seeded model for tests and benchmarks (NOT for scientific use)."""
        import torch
        from .cbramod_model import CBraMod
        g = torch.random.fork_rng()
        with g:
            torch.manual_seed(seed)
            model = CBraMod(n_layer=n_layer)
        return cls(model, batch_size, num_threads)

    def encode(self, tokens: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """``tokens`` (n, 19, P, 200) float32 in model units with invalid cells already zeroed; ``valid`` (n, 19, P)
        bool. Returns (n, 200) float32: the mean of the patch representations over valid cells (NaN row if none)."""
        torch = self.torch
        out = np.full((len(tokens), EMB_DIM), np.nan, np.float32)
        with torch.inference_mode():
            for a in range(0, len(tokens), self.batch_size):
                x = torch.from_numpy(np.ascontiguousarray(tokens[a:a + self.batch_size], dtype=np.float32))
                v = torch.from_numpy(valid[a:a + self.batch_size])
                feats = self.model(x, mask=(~v).to(torch.int64))                  # (b, 19, P, 200)
                w = v.to(feats.dtype).unsqueeze(-1)
                n = w.sum(dim=(1, 2)).clamp_min(1.0)
                pooled = (feats * w).sum(dim=(1, 2)) / n
                pooled[(v.sum(dim=(1, 2)) == 0)] = float("nan")
                out[a:a + len(x)] = pooled.numpy()
        return out


# ------------------------------------------------------------------------------------------------ preparation
@dataclass
class Prepared:
    tokens: np.ndarray                  # (n_seg, 19, P, 200) float32, model units, invalid cells zero
    valid: np.ndarray                   # (n_seg, 19, P) bool
    seg_valid_frac: np.ndarray          # (n_seg,) share of valid cells
    seg_over_amp: np.ndarray            # (n_seg,) any valid sample above amp_uv
    seg_ok: np.ndarray                  # (n_seg,) segment may enter a pool
    window_segments: dict               # name -> slice of segment indices
    n_absent_channels: int


@dataclass
class EmbedResult:
    rows: list
    qc: dict
    channel_status: dict = field(default_factory=dict)
    n_encoded_segments: int = 0


def _grid(windows: dict, seg_s: float):
    span0 = min(w.start_s for w in windows.values())
    span1 = max(w.end_s for w in windows.values())
    n_seg = int(round((span1 - span0) / seg_s))
    slices = {}
    for k, w in windows.items():
        a, n = (w.start_s - span0) / seg_s, w.duration_s / seg_s
        if abs(a - round(a)) > 1e-6 or abs(n - round(n)) > 1e-6 or n < 1:
            raise ValueError("every window must start on, and span a whole number of, segments")
        slices[k] = slice(int(round(a)), int(round(a)) + int(round(n)))
    return span0, n_seg, slices


def prepare_segments(x: np.ndarray, fs: float, ch_names, offset_s: float, ef, windows: dict, cfg: EmbedConfig) -> Prepared:
    """Cut preprocessed, referenced ``x`` (C, N) uV into CBraMod-conformant segments on the grid anchored at the earliest
    window start. ``ef`` supplies the clean mask on the 2-s epoch grid (same anchor)."""
    span0, n_seg, slices = _grid(windows, cfg.segment_s)
    spp = int(round(cfg.segment_s * FS / PATCH_SAMPLES))
    n_tok = n_seg * spp
    n_samp = n_tok * PATCH_SAMPLES
    a0 = int(round((span0 - offset_s) * fs))
    full = np.zeros((len(CBRAMOD_CHANNELS), n_samp), np.float64)
    valid_tok = np.zeros((len(CBRAMOD_CHANNELS), n_tok), bool)
    present = 0
    t_epoch = (np.arange(n_tok) * (PATCH_SAMPLES / FS) / ef.epoch_s + 1e-9).astype(int)      # epoch index of every patch
    clean = ef.clean
    idx = {c: i for i, c in enumerate(ch_names)}
    for r, name in enumerate(CBRAMOD_CHANNELS):
        i = idx.get(name)
        if i is None:
            continue
        present += 1
        seg = x[i, max(0, a0):a0 + n_samp]
        lo = max(0, -a0)
        sig = np.full(n_samp, np.nan)
        sig[lo:lo + len(seg)] = seg
        full[r] = sig
        ok = np.zeros(n_tok, bool)
        inside = t_epoch < clean.shape[1]
        ok[inside] = clean[i, t_epoch[inside]]
        ok &= np.isfinite(sig).reshape(n_tok, PATCH_SAMPLES).all(axis=1)
        valid_tok[r] = ok
    full = np.nan_to_num(full, nan=0.0)
    full4 = full.reshape(len(CBRAMOD_CHANNELS), n_tok, PATCH_SAMPLES)
    full4 = full4 * valid_tok[:, :, None]                                           # zero-fill invalid cells
    over = (np.abs(full4) > cfg.amp_uv).any(axis=2)                                 # (19, n_tok), valid cells only
    tokens = (full4 / UV_PER_UNIT).astype(np.float32).reshape(len(CBRAMOD_CHANNELS), n_seg, spp, PATCH_SAMPLES)
    tokens = np.ascontiguousarray(tokens.transpose(1, 0, 2, 3))
    valid = np.ascontiguousarray(valid_tok.reshape(len(CBRAMOD_CHANNELS), n_seg, spp).transpose(1, 0, 2))
    over_seg = over.reshape(len(CBRAMOD_CHANNELS), n_seg, spp).any(axis=(0, 2))
    vfrac = valid.mean(axis=(1, 2))
    ok = vfrac >= cfg.min_valid_frac
    if cfg.amp_policy == "drop":
        ok &= ~over_seg
    return Prepared(tokens, valid, vfrac, over_seg, ok, slices, len(CBRAMOD_CHANNELS) - present)


def pool_window(E: np.ndarray, attention: bool, tau: float):
    """Mean (and optional centroid-attention) pooling of segment embeddings ``E`` (k, 200)."""
    mean = E.mean(axis=0)
    if not attention:
        return mean, None
    m = mean.astype(np.float64)
    s = (E.astype(np.float64) @ m) / (np.linalg.norm(E.astype(np.float64), axis=1) * np.linalg.norm(m) + 1e-12) / tau
    w = np.exp(s - s.max())
    w /= w.sum()
    return mean, (w @ E.astype(np.float64)).astype(np.float32)


def embed_prepared(prep: Prepared, embedder: Embedder, qcs: dict, cfg: EmbedConfig, timers=None) -> EmbedResult:
    wanted = {k: sl for k, sl in prep.window_segments.items() if qcs[k].passes or cfg.compute_failed}
    need = np.zeros(len(prep.tokens), bool)
    for sl in wanted.values():
        need[sl] |= prep.seg_ok[sl]
    E = np.full((len(prep.tokens), EMB_DIM), np.nan, np.float32)
    ix = np.flatnonzero(need)
    if len(ix):
        with stage(timers, "cbramod.forward"):
            E[ix] = embedder.encode(prep.tokens[ix], prep.valid[ix])
    rows = []
    for name, sl in prep.window_segments.items():
        q: WindowQC = qcs[name]
        row = {"window": name, "qc_pass": bool(q.passes), "usable_fraction": float(q.usable_fraction),
               "qc_n_missing_or_dead_min": int(q.n_missing_or_dead_min), "qc_coverage_fraction": float(q.coverage_fraction),
               "emb_ok": False, "emb_n_segments": int(sl.stop - sl.start), "emb_n_used": 0,
               "emb_valid_token_frac": float(prep.seg_valid_frac[sl].mean()),
               "emb_frac_over_amp": float(prep.seg_over_amp[sl].mean()), "emb_n_absent_channels": prep.n_absent_channels,
               **dict.fromkeys(emb_columns(), float("nan")),
               **(dict.fromkeys(emb_columns(ATTN_PREFIX), float("nan")) if cfg.attention else {})}
        if name in wanted:
            use = np.flatnonzero(prep.seg_ok[sl]) + sl.start
            if len(use):
                mean, attn = pool_window(E[use], cfg.attention, cfg.attn_tau)
                row.update(emb_ok=True, emb_n_used=int(len(use)))
                row.update({c: float(v) for c, v in zip(emb_columns(), mean)})
                if attn is not None:
                    row.update({c: float(v) for c, v in zip(emb_columns(ATTN_PREFIX), attn)})
        rows.append(row)
    return EmbedResult(rows, qcs, {}, int(len(ix)))


def embed_recording(src, embedder: Embedder, windows=None, qc_cfg: QCConfig | None = None,
                    cfg: EmbedConfig | None = None, *, shared: RecordingPrep | None = None, timers=None) -> EmbedResult:
    """Embeddings for one recording. ``src``: EDF path / file-like, or a ``Recording`` already read in uV with
    ``offset_s`` and ``meta['edf_duration_s']`` (as ``stream.fetch_window`` returns). Windows default to primary + nested
    relative to the file start; the streaming wrapper shifts them by the D-109 onset. Raises on unreadable input.
    ``shared`` (``eeg.prep.RecordingPrep``, built from the same Recording / windows / QC config) lets the MORGOTH rung reuse
    this rung's channel selection and QC; ``timers`` (``timing.StageTimers``) collects per-stage seconds."""
    cfg = cfg or EmbedConfig()
    qc_cfg = qc_cfg or QCConfig()
    windows = dict(windows or all_windows())
    if isinstance(src, Recording):
        rec = src
    else:
        if isinstance(src, (str, os.PathLike)):
            assert_not_restricted_in_agent(src)
        t0 = max(0.0, min(w.start_s for w in windows.values()) - PAD_S)
        t1 = max(w.end_s for w in windows.values()) + PAD_S
        rec = read_edf(src, start_s=t0, duration_s=t1 - t0, channels=list(CBRAMOD_CHANNELS))
    if shared is None or not shared.compatible(windows, qc_cfg):
        shared = RecordingPrep(rec, windows, qc_cfg, timers)
    rec = shared.rec                                        # exactly-constant channels are MISSING (D-110), not flat data
    if not rec.ch_names:
        raise ValueError("no usable EEG channels")
    notes = shared.notes
    qcs, ef = shared.qc()
    status = {"n_channels": len(rec.ch_names), "missing": list(rec.meta.get("missing_channels", [])),
              "dead": notes["dead"], "invalid_scaling": notes["invalid_scaling"]}
    minimum = set(qc_cfg.minimum_channels)
    status.update(n_missing_min=len(minimum & set(status["missing"])) - len(minimum & set(notes["invalid_scaling"])),
                  n_dead_min=len(minimum & set(notes["dead"])), n_invalid_min=len(minimum & set(notes["invalid_scaling"])))
    if not (cfg.compute_failed or any(q.passes for q in qcs.values())):      # nothing to embed: skip filtering and the model
        nan_row = dict.fromkeys(emb_columns() + (emb_columns(ATTN_PREFIX) if cfg.attention else []), float("nan"))
        rows = [{"window": k, "qc_pass": False, "usable_fraction": float(q.usable_fraction),
                 "qc_n_missing_or_dead_min": int(q.n_missing_or_dead_min), "qc_coverage_fraction": float(q.coverage_fraction),
                 "emb_ok": False, "emb_n_segments": int(round(windows[k].duration_s / cfg.segment_s)), "emb_n_used": 0,
                 "emb_valid_token_frac": float("nan"), "emb_frac_over_amp": float("nan"),
                 "emb_n_absent_channels": len(CBRAMOD_CHANNELS) - len(rec.ch_names), **nan_row} for k, q in qcs.items()]
        return EmbedResult(rows, qcs, status, 0)
    with stage(timers, "cbramod.filter"):
        pre = PreprocessConfig(target_fs=FS, notch_hz=cfg.notch_hz, band=tuple(cfg.band))
        x, fs = preprocess(rec.data, rec.fs, pre)
        if cfg.reference == "car":
            disc = ef.flags["disconnected"].all(axis=1) if ef.flags["disconnected"].size else None
            ep = int(round(qc_cfg.epoch_s * fs))
            start = int(round((ef.start_s - rec.offset_s) * fs))
            x = common_average_masked(x, ef.clean, ep, start, base_exclude=disc)
    with stage(timers, "cbramod.segments"):
        prep = prepare_segments(x, fs, rec.ch_names, rec.offset_s, ef, windows, cfg)
    del x
    res = embed_prepared(prep, embedder, qcs, cfg, timers)
    res.channel_status = status
    return res
