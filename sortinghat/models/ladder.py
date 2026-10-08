"""Representation-ladder runner and split orchestration (Study 1, SAP sections 4, 5, 7).

Rungs: prior -> qEEG -> connectivity -> MORGOTH findings -> frozen embeddings -> dynamics -> combined, each fitted
with the same shallow head as (a) the baseline alone and (b) baseline + the rung's EEG columns. Delta = masked log
loss(baseline + EEG) - masked log loss(baseline) on identical gold *evaluation* patients, primary labels only;
negative = EEG helps. Splits: leave-one-site-out (primary) or a late-calendar temporal holdout.

Strictness: ``fit_predict_fold`` receives the full ``ModelData`` plus index arrays, slices the training rows
itself, hands the fitters nothing but training features / silver labels / dev-gold, and reads only FEATURES for
the test rows. Held-out labels, roles and covariates are never touched before evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ..metrics.bootstrap import delta_ci_all_modes, paired_bootstrap_delta
from ..metrics.hypotheses import per_label_delta_ci
from ..metrics.labels import primary_label_indices
from ..metrics.loss import (DEFAULT_EPS, favorable_at_every_site, mean_masked_log_loss, per_patient_delta,
                            per_site_delta, site_weighted_delta)
from ..metrics.splits import late_temporal_holdout, leave_one_site_out
from ..safe_output import suppress_count
from .data import ModelData
from .head import HeadConfig, fit_model, prevalence_prior
from .preprocess import PreprocConfig

BASELINE_KEY = "__baseline__"


@dataclass(frozen=True)
class RungSpec:
    name: str
    prefixes: tuple = ()
    has_eeg: bool = True
    requires: tuple = ()        # prefixes that must each match >= 1 column for the rung to be available


DEFAULT_RUNGS = (
    RungSpec("prior", (), has_eeg=False),
    RungSpec("qeeg", ("qeeg.",)),
    RungSpec("connectivity", ("conn.",)),
    RungSpec("morgoth", ("morgoth.",)),
    RungSpec("embeddings", ("emb.cbramod.",)),
    RungSpec("dynamics", ("dyn.",)),
    RungSpec("combined", ("qeeg.", "conn.", "morgoth.", "emb.", "dyn.")),
)
# Frozen public-CBraMod embedding rung (scripts/extract_embeddings.py -> emb.cbramod.<j>). Not part of DEFAULT_RUNGS (which already
# holds the generic "embeddings" family rung); it is added by callers that load the embedding parquet, and it is unavailable
# (RungResult.available False, never fitted) when no emb.cbramod. column exists.
CBRAMOD_FROZEN_RUNG = RungSpec("cbramod_frozen", ("emb.cbramod.",), requires=("emb.cbramod.",))
# MORGOTH finding-probability rung (scripts/extract_morgoth.py features -> morgoth.<head>.<class>.<stat>, loaded with
# inputs.load_morgoth_findings). Same pattern: not in DEFAULT_RUNGS (whose generic "morgoth" rung stays for synthetic
# frames); callers that load the parquet add it, and it is unavailable (never fitted) when no morgoth. column exists.
MORGOTH_FINDINGS_RUNG = RungSpec("morgoth_findings", ("morgoth.",), requires=("morgoth.",))
# Commercial-clean gap pair (plan: CBraMod vs MORGOTH alongside each other).
COMMERCIAL_RUNGS = (
    RungSpec("combined_morgoth", ("qeeg.", "conn.", "morgoth.", "dyn."), requires=("morgoth.",)),
    RungSpec("combined_cbramod", ("qeeg.", "conn.", "emb.cbramod.", "dyn."), requires=("emb.cbramod.",)),
)


def rung_columns(eeg_cols: Sequence[str], spec: RungSpec) -> list[str]:
    if not spec.has_eeg:
        return []
    cols = list(eeg_cols)
    if any(not any(c.startswith(r) for c in cols) for r in spec.requires):
        return []
    return [c for c in cols if c.startswith(tuple(spec.prefixes))] if spec.prefixes else []


@dataclass(frozen=True)
class LadderConfig:
    head: HeadConfig = field(default_factory=HeadConfig)
    pre: PreprocConfig = field(default_factory=PreprocConfig)
    include_e7: bool = False
    eps: float = DEFAULT_EPS
    n_boot: int = 2000
    seed: int = 0
    alpha_primary: float = 0.01          # H1/H2 interval (D-095): 99% within-site
    alpha_secondary: float = 0.05
    temporal_test_fraction: float = 0.20
    temporal_embargo: float = 0.0
    per_label: bool = True


# --------------------------------------------------------------------------
# One fold
# --------------------------------------------------------------------------
def fit_predict_fold(data: ModelData, train_idx, test_idx, baseline_cols: Sequence[str],
                     variants: Mapping[str, Sequence[str]], cfg: LadderConfig,
                     require_site_disjoint: bool = False, return_models: bool = False):
    """Fit every variant on the training rows and predict the test rows.

    ``variants``: name -> EEG columns added to the baseline columns (an empty list is the baseline-only model).
    Training = non-dev training-fold rows with SILVER labels; dev-gold rows inside the training fold drive model
    selection and recalibration; eval-role rows contribute nothing but (silver) training rows if they are in the
    training fold. Returns {name: (n_test, K) probabilities} plus per-variant fit info (and models if asked).
    """
    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    if np.intersect1d(train_idx, test_idx).size:
        raise ValueError("train and test rows overlap")
    if require_site_disjoint and set(np.unique(data.sites[train_idx])) & set(np.unique(data.sites[test_idx])):
        raise ValueError("a held-out site also appears in the training fold")
    tr = data.take(train_idx)
    dev_rows = np.flatnonzero(tr.gold_role == "dev")
    fit_rows = np.flatnonzero(tr.gold_role != "dev")
    sel = primary_label_indices(data.label_names, cfg.include_e7)
    preds, info, models = {}, {}, {}
    for name, eeg_cols in variants.items():
        if not len(baseline_cols) and not len(eeg_cols):
            # No baseline columns and no EEG columns: the model IS the smoothed prevalence prior (training silver rows). This is the
            # reference of the "EEG-only vs prevalence prior" comparison (D-145); it has no features to select or recalibrate.
            preds[name] = np.tile(prevalence_prior(tr.y_silver[fit_rows], tr.m_silver[fit_rows]), (len(test_idx), 1))
            info[name] = {"selected_c": float("nan"), "selected_eeg_scale": float("nan"), "used_dev": False,
                          "calibrated": False, "n_train": int(len(fit_rows)), "n_dev": 0, "n_features_in": 0,
                          "n_features_out": 0}
            models[name] = None
            continue
        Xtr = tr.design(fit_rows, baseline_cols, eeg_cols)
        Xdev = tr.design(dev_rows, baseline_cols, eeg_cols)
        mdl = fit_model(Xtr, tr.y_silver[fit_rows], tr.m_silver[fit_rows],
                        Xdev, tr.y_gold[dev_rows], tr.m_gold[dev_rows],
                        protected=list(baseline_cols), select_cols=sel, pre_cfg=cfg.pre, head_cfg=cfg.head)
        preds[name] = mdl.predict(data.design(test_idx, baseline_cols, eeg_cols))   # features only
        info[name] = mdl.info
        models[name] = mdl
    preds["__prior__"] = np.tile(prevalence_prior(tr.y_silver[fit_rows], tr.m_silver[fit_rows]),
                                 (len(test_idx), 1))
    return (preds, info, models) if return_models else (preds, info)


def make_splits(data: ModelData, split: str, cfg: LadderConfig):
    """List of (name, train_idx, test_idx, site_disjoint)."""
    if split == "loso":
        return [(s.held_out, s.train_idx, s.test_idx, True) for s in leave_one_site_out(data.sites)]
    if split == "temporal":
        if data.times is None:
            raise ValueError("temporal holdout needs data.times")
        ts = late_temporal_holdout(data.sites, data.times, cfg.temporal_test_fraction, cfg.temporal_embargo)
        return [("temporal", ts.train_idx, ts.test_idx, False)]
    raise ValueError("split must be 'loso' or 'temporal'")


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
@dataclass
class RungResult:
    rung: str
    baseline: str
    available: bool
    has_eeg: bool = True
    n_eeg_features: int = 0
    delta: float = float("nan")
    loss_model: float = float("nan")
    loss_baseline: float = float("nan")
    ci_primary: dict = field(default_factory=dict)       # 99% within-site (H1/H2 interval)
    ci_modes: dict = field(default_factory=dict)         # 95% within_site / cluster / two_stage / site_t
    per_site: dict = field(default_factory=dict)         # site -> {delta, n}
    all_sites_favorable: bool = False
    site_weighted_delta: float = float("nan")
    rule_met: bool = False                               # primary CI hi < 0 and every site < 0 (descriptive off-primary)
    per_label: dict = field(default_factory=dict)
    d: np.ndarray | None = field(default=None, repr=False)   # per-patient d_i (evaluation rows), memory only


@dataclass
class LadderResult:
    split: str
    label_names: tuple
    primary_labels: tuple
    rungs: dict                       # (baseline, rung) -> RungResult
    site_labels: dict                 # real site -> pseudonym
    eval_idx: np.ndarray = field(repr=False, default=None)
    eval_sites: np.ndarray = field(repr=False, default=None)
    reference: dict = field(default_factory=dict)        # baseline -> {loss_baseline, loss_prior, baseline_information}
    fold_info: list = field(default_factory=list)
    n_eval: int = 0

    def get(self, rung: str, baseline: str | None = None) -> RungResult:
        if baseline is None:
            baseline = next(b for (b, _r) in self.rungs)
        return self.rungs[(baseline, rung)]

    def to_aggregate(self) -> dict:
        from .report import ladder_to_aggregate
        return ladder_to_aggregate(self)


def _evaluate(data: ModelData, P: Mapping[str, np.ndarray], tested: np.ndarray, baseline_name: str,
              rung_cols: Mapping[str, tuple], cfg: LadderConfig, split: str, site_labels: dict):
    K = data.K
    prim = np.zeros(K, bool)
    prim[primary_label_indices(data.label_names, cfg.include_e7)] = True
    ev = (data.gold_role == "eval") & tested
    ev_idx = np.flatnonzero(ev)
    y = np.where(data.m_gold[ev_idx], data.y_gold[ev_idx], 0.0)
    m_all = data.m_gold[ev_idx]
    m_prim = m_all & prim[None, :]
    sites = data.sites[ev_idx]

    def p_of(name):
        return np.where(np.isnan(P[name][ev_idx]), 0.5, P[name][ev_idx])

    pb = p_of(BASELINE_KEY)
    loss_b = mean_masked_log_loss(y, pb, m_prim, cfg.eps)
    loss_prior = mean_masked_log_loss(y, p_of("__prior__"), m_prim, cfg.eps)
    reference = {"loss_baseline": loss_b, "loss_prior": loss_prior, "baseline_information": loss_b - loss_prior}
    out = {}
    for rung, (has_eeg, cols) in rung_cols.items():
        if rung == "prior" or not has_eeg:
            out[rung] = RungResult(rung, baseline_name, True, False, 0, 0.0, loss_b, loss_b,
                                   all_sites_favorable=False, d=np.zeros(len(ev_idx)))
            continue
        if not cols:
            out[rung] = RungResult(rung, baseline_name, False)
            continue
        pm = p_of(rung)
        d = per_patient_delta(y, pm, pb, m_prim, cfg.eps)
        r = RungResult(rung, baseline_name, True, True, len(cols))
        r.d = d
        r.delta = float(np.nanmean(d)) if np.any(~np.isnan(d)) else float("nan")
        r.loss_model = mean_masked_log_loss(y, pm, m_prim, cfg.eps)
        r.loss_baseline = loss_b
        r.ci_primary = paired_bootstrap_delta(d, sites, "within_site", cfg.n_boot, cfg.seed,
                                              cfg.alpha_primary).as_dict()
        try:
            modes = delta_ci_all_modes(d, sites, cfg.n_boot, cfg.seed, cfg.alpha_secondary)
            r.ci_modes = {k: (v.as_dict() if hasattr(v, "as_dict") else dict(v)) for k, v in modes.items()}
        except ValueError:                      # single evaluation site: only within_site exists
            r.ci_modes = {"within_site": paired_bootstrap_delta(d, sites, "within_site", cfg.n_boot, cfg.seed,
                                                                cfg.alpha_secondary).as_dict()}
        r.per_site = per_site_delta(d, sites)
        fav = favorable_at_every_site(d, sites)
        r.all_sites_favorable = bool(fav["all_favorable"])
        r.site_weighted_delta = site_weighted_delta(d, sites)
        r.rule_met = bool(r.ci_primary["hi"] < 0 and r.all_sites_favorable)
        if cfg.per_label:
            r.per_label = per_label_delta_ci(y, pm, pb, m_all, data.label_names, sites, cfg.n_boot, cfg.seed,
                                             "within_site", cfg.alpha_secondary, cfg.eps)
        out[rung] = r
    return out, reference, ev_idx, sites


def run_ladder(data: ModelData, baseline_sets: Mapping[str, Sequence[str]] | None = None,
               rungs: Sequence[RungSpec] | None = None, cfg: LadderConfig | None = None,
               split: str = "loso") -> LadderResult:
    """Run every rung as baseline-only vs baseline+EEG, per baseline set, under LOSO or temporal holdout."""
    cfg = cfg or LadderConfig()
    baseline_sets = dict(baseline_sets) if baseline_sets else {"all": list(data.baseline.columns)}
    rungs = tuple(rungs) if rungs is not None else DEFAULT_RUNGS + COMMERCIAL_RUNGS
    splits = make_splits(data, split, cfg)
    uniq = sorted({str(s) for s in data.sites})
    site_labels = {s: f"site_{i + 1}" for i, s in enumerate(uniq)}
    eeg_cols = list(data.eeg.columns)
    rung_cols = {sp.name: (sp.has_eeg, tuple(rung_columns(eeg_cols, sp))) for sp in rungs}
    variants = {BASELINE_KEY: []}
    variants.update({n: list(c) for n, (h, c) in rung_cols.items() if h and c})
    res = LadderResult(split, data.label_names,
                       tuple(data.label_names[i] for i in primary_label_indices(data.label_names, cfg.include_e7)),
                       {}, site_labels)
    for bname, bcols in baseline_sets.items():
        P = {k: np.full((data.n, data.K), np.nan) for k in list(variants) + ["__prior__"]}
        tested = np.zeros(data.n, bool)
        for fname, tr, te, disjoint in splits:
            preds, info = fit_predict_fold(data, tr, te, bcols, variants, cfg, require_site_disjoint=disjoint)
            for k, v in preds.items():
                P[k][te] = v
            tested[te] = True
            res.fold_info.append({"baseline": bname, "fold": site_labels.get(fname, fname),
                                  "n_train": suppress_count(len(tr)), "n_test": suppress_count(len(te)),
                                  "selected_c": {k: v["selected_c"] for k, v in info.items()},
                                  "selected_eeg_scale": {k: v["selected_eeg_scale"] for k, v in info.items()},
                                  "calibrated": {k: v["calibrated"] for k, v in info.items()}})
        out, ref, ev_idx, ev_sites = _evaluate(data, P, tested, bname, rung_cols, cfg, split, site_labels)
        for rname, rr in out.items():
            res.rungs[(bname, rname)] = rr
        res.reference[bname] = ref
        res.eval_idx, res.eval_sites, res.n_eval = ev_idx, ev_sites, int(len(ev_idx))
    return res


def commercial_clean_gap(result: LadderResult, baseline: str | None = None, n_boot: int = 2000, seed: int = 0,
                         alpha: float = 0.05) -> dict:
    """Delta(CBraMod-based combined) - Delta(MORGOTH-based combined), paired bootstrap on d_i differences.

    Reported, not tested (SAP 4.2). ``NaN``/unavailable when either combined rung is missing.
    """
    a = result.get("combined_cbramod", baseline)
    b = result.get("combined_morgoth", baseline)
    if not (a.available and b.available):
        return {"available": False}
    diff = a.d - b.d
    ci = paired_bootstrap_delta(diff, result.eval_sites, "within_site", n_boot, seed, alpha)
    return {"available": True, "gap": float(np.nanmean(diff)), "lo": ci.lo, "hi": ci.hi,
            "delta_cbramod": a.delta, "delta_morgoth": b.delta}
