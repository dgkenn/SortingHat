#!/usr/bin/env python3
"""EXPLORATORY — imaging-anchored E1 (structural injury) test inside HEEDB using BIND.   (D-158, D-160; docs/bind_e1_analysis_plan.md)

Does the FROZEN EEG representation (the `combined` rung = qEEG + connectivity) carry information about acute structural brain injury
beyond Baseline A when "injury" is defined by RADIOLOGY (BIND findings of a CT / MRI read without the EEG) and not by our ICD-based silver
anchors?  No model is refitted or tuned on imaging-E1: the probabilities scored here are the HELD-OUT per-patient predictions of the silver
feasibility run (label E1, trained on silver E1).  Pre-specified; the plan, `configs/bind_e1.yaml` (hash registered in D-160) and
`sortinghat/metrics/bind_e1.py` define everything; nothing is tuned to a result.

HOW THE PREDICTIONS ARE OBTAINED.  The feasibility pipeline stores every model fit it makes (scheme x fold x rung, held-out predictions and
fit info) as a checkpoint unit under out/local_only/checkpoints/silver_feasibility/<run key>/, keyed by a digest of the actual data, the
train / test indices and the ladder config.  This script rebuilds the SAME assembly (`run_silver_feasibility.assemble`, broad cohort, the
broad run's defaults: seed 0, dev fraction 0.2, temporal fraction 0.2, n_boot 4000) and the SAME per-fold calls
(`ladder.fit_predict_fold`), looking the fits up in those checkpoints through a read-only lookup (`LookupCheckpoint`).  A MISSING unit
aborts the run ("require cache") because it would mean the inputs or the config differ from the feasibility run; only `--allow-refit`
regenerates a missing fit deterministically with the same seeds (hours; written to a separate scratch checkpoint, never over the originals).
The run states which route it took and how many fits were cached.

ORDER (plan section 9, nothing is skipped).
  step 1  label table (windows and masking of the plan) for the analysed broad-cohort patients; suppressed counts of imaging-positive cases
          per stratum are written to <out>/step1_counts.json BEFORE any prediction is loaded or any model-based number is computed; stop rule S3
          (< 50 positives in a stratum = not estimable; both below 50 = stop).  `--counts-only` ends here.
  step 2  primary pair (delta AUROC, shifted and unshifted delta log loss; 99% within-site bootstrap, B = 10,000), secondary arms and
          strata, concordance with silver E1, per-class, negative-control outcome, sedation-intensity grades, drift and selection screens,
          sensitivities -> <out>/report.md and report.json (aggregate-only; n < 11 shown "<11"; out/ is gitignored, rule 6).

LAUNCH (human or D-118 launcher; the exact broad-run settings; `--s3` is only used for the merge map of the IU-large OMOP proxies, as in the feasibility run):
    env -u CLAUDECODE -u SORTINGHAT_AGENT_SESSION scripts/heedb_run.sh python3 scripts/run_bind_e1.py --s3 --check-cache      # cheap: no BIND data
    env -u CLAUDECODE -u SORTINGHAT_AGENT_SESSION scripts/heedb_run.sh python3 scripts/run_bind_e1.py --s3 --counts-only      # step 1 only
    env -u CLAUDECODE -u SORTINGHAT_AGENT_SESSION scripts/heedb_run.sh python3 scripts/run_bind_e1.py --s3                    # steps 1 and 2
    # strict cohort, reported separately (never pooled): add  --cohort-def strict --out out/bind_e1_strict
Tests run it end to end on synthetic inputs (`--bind-data` is a local mirror of the BIND tables).
"""
from __future__ import annotations

import hashlib
import itertools
import pickle
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import run_silver_feasibility as rsf  # noqa: E402
from sortinghat import agent_safety, checkpoint as ck, data_io  # noqa: E402
from sortinghat.labels import bind  # noqa: E402
from sortinghat.metrics import bind_e1 as be  # noqa: E402
from sortinghat.metrics import intended_use as iu  # noqa: E402
from sortinghat.metrics.sedation import sedation_intensity_grade  # noqa: E402
from sortinghat.metrics.splits import late_temporal_holdout  # noqa: E402
from sortinghat.models import COMMERCIAL_RUNGS, CBRAMOD_FROZEN_RUNG, LadderConfig, MORGOTH_FINDINGS_RUNG  # noqa: E402
from sortinghat.models import ladder as ladder_mod  # noqa: E402
from sortinghat.safe_output import SUPPRESSED, safe_print, safe_write_json, safe_write_text, suppress_count, suppress_proportion  # noqa: E402

BANNER = "EXPLORATORY — imaging-anchored E1 (BIND); frozen silver-trained predictions, no refit; not a test of preregistered hypotheses"
REGISTERED_CONFIG_SHA256 = "451f5f30946859a285ca21bda9d6532133d1861d13bff91c5103579c60a4f012"     # D-160
STRATUM_IU, STRATUM_WHOLE = "IU_large", "whole_broad"
PRIMARY_ARM = "A+combined vs A"
HEADLINE = "combined"
LIMITATIONS = (
    "The label is LLM-extracted from radiology reports (Bio-Medical-Llama-3-8B) with no published accuracy; label noise biases delta AUROC toward 0 "
    "and a null result is uninformative without step 0d (clinician validation).",
    "Imaging exists only for patients who were imaged and had an EEG; the estimand is conditional on a brain study in the window (spectrum and "
    "verification bias; an EEG finding may have prompted the imaging). It is not 'all patients with altered mental status'.",
    "Study dates are date-only: day-resolution timing cannot order a same-day CT/MRI and EEG.",
    "BIND covers MGB (I0001) only for EEG linkage; nothing is said beyond the labelled sites.",
    "The frozen model optimises silver E1; imaging-E1 is a different construct (SAP A1.1 caveat). A gain on imaging-E1 is evidence about structural "
    "signal in the representation, not the SAP's gold-standard H-hypotheses.",
    "BIND-derived labels are research evidence only; the commercial-clean path will not use them for training or tuning (D-158).",
    "Not run here: the imaging version of the circularity audit (plan section 1.4) and the per-silver-anchor-family concordance / procedure-only "
    "sensitivity unless --silver-families is given (the stored silver labels carry no anchor-family detail).",
)


# ============================================================================================ parser
def build_parser():
    ap = rsf.build_parser()
    ap.description = __doc__
    ap.set_defaults(cohort_def="broad", out="out/bind_e1")
    ap.add_argument("--bind-config", default=str(bind.CONFIG_PATH))
    ap.add_argument("--bind-site", default="I0001")
    ap.add_argument("--bind-data", default=None, help="local mirror holding Imaging/<site>/Clinical/*.csv (SYNTHETIC tests only); default: BDSP S3")
    ap.add_argument("--counts-only", action="store_true", help="stop after step 1 (suppressed counts of imaging-positive cases; stop rule S3)")
    ap.add_argument("--check-cache", action="store_true", help="only check that the stored held-out predictions can be rebuilt; reads no BIND data")
    ap.add_argument("--allow-refit", action="store_true", help="regenerate a missing fit deterministically (hours); default: abort on a cache miss")
    ap.add_argument("--checkpoint-roots", nargs="+", default=None,
                    help="feasibility checkpoint run directories to look fits up in (default: every run under out/local_only/checkpoints/silver_feasibility)")
    ap.add_argument("--refit-scratch", default="out/local_only/checkpoints/bind_e1_refit", help="where --allow-refit stores regenerated fits")
    ap.add_argument("--n-boot-primary", type=int, default=10000)
    ap.add_argument("--n-boot-secondary", type=int, default=2000)
    ap.add_argument("--min-positives", type=int, default=be.MIN_POSITIVES, help="stop rule S3 (fixed at 50 by the plan; lowering it marks the report)")
    ap.add_argument("--feasibility-report", default="out/silver_feasibility_broad/report.json")
    ap.add_argument("--silver-families", default=None,
                    help="optional local_only CSV (person_id, proc_only) flagging silver-E1 positives that rest only on a procedure code")
    ap.add_argument("--mgb-sites", nargs="+", default=None, help="real HEEDB site ids to treat as MGB for sensitivity 6 (not assumed)")
    ap.add_argument("--ppv-lower", type=float, default=None, help="step 0d lower 95%% bound of PPV (human-supplied aggregate)")
    ap.add_argument("--npv-lower", type=float, default=None, help="step 0d lower 95%% bound of NPV (human-supplied aggregate)")
    ap.add_argument("--allow-config-change", action="store_true", help="run with a YAML whose hash differs from D-160 (marks the report)")
    return ap


# ============================================================================================ stored predictions
class CacheMiss(RuntimeError):
    """A fit that the feasibility run should have stored is not in the checkpoints."""


class LookupCheckpoint(ck.Checkpoint):
    """Read-only view over the feasibility run's checkpoint units. ``get`` searches every root; a miss aborts (``CacheMiss``) unless
    refits are allowed, in which case a regenerated fit is stored in ``scratch`` only. The originals are never written, renamed or pruned."""

    def __init__(self, roots, scratch=None, allow_put: bool = False):
        self.step, self.key, self.quiet, self.resume = "silver_feasibility", "bind_e1_lookup", True, True
        self.roots = [Path(r) for r in roots]
        self.dir = Path(scratch) if scratch else self.roots[0]
        self.allow_put = allow_put
        if allow_put:
            self.roots.append(self.dir)
            self._mkdir(self.dir.parent)
            self._mkdir(self.dir)
        self.n_hits = self.n_puts = self.n_miss = 0
        self.counts: dict = {}
        self._kill_suspended, self.fail_after = 0, None
        self._lock, self._seq = threading.Lock(), itertools.count()

    def get(self, name, default=ck.MISS):
        h = hashlib.sha1(name.encode()).hexdigest()
        for r in self.roots:
            p = r / h[:2] / f"{h[2:30]}.pkl"
            try:
                with open(p, "rb") as fh:
                    stored, obj = pickle.load(fh)
            except (FileNotFoundError, OSError, pickle.UnpicklingError, EOFError):
                continue
            if stored == name:
                with self._lock:
                    self.n_hits += 1
                return obj
        with self._lock:
            self.n_miss += 1
        if not self.allow_put:
            raise CacheMiss("a model fit of the feasibility run is not in the stored checkpoints")
        return default

    def put(self, name, obj, *, count: bool = True) -> None:
        if not self.allow_put:
            raise CacheMiss("refit not allowed")
        super().put(name, obj, count=False)


def default_roots() -> list[Path]:
    base = ck.REPO_ROOT / ck.DEFAULT_SUBDIR / "silver_feasibility"
    return sorted((d for d in base.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime, reverse=True) if base.is_dir() else []


def ladder_cfg(A, a) -> LadderConfig:
    return LadderConfig(n_boot=a.n_boot, seed=a.seed, include_e7=A.include_e7, temporal_test_fraction=a.temporal_fraction)   # as rsf.run_analysis


def ladder_rungs(A) -> list:
    rungs = list(rsf.BASE_RUNGS)                                      # as rsf.run_analysis
    if any(str(c).startswith(rsf.CBRAMOD_PREFIX) for c in A.eeg.columns):
        rungs += [CBRAMOD_FROZEN_RUNG, next(r for r in COMMERCIAL_RUNGS if r.name == "combined_cbramod")]
    if any(str(c).startswith(rsf.MORGOTH_PREFIX) for c in A.eeg.columns):
        rungs += [MORGOTH_FINDINGS_RUNG, next(r for r in COMMERCIAL_RUNGS if r.name == "combined_morgoth")]
    return rungs


def collect(md, baseline_sets: dict, rungs: list, cfg: LadderConfig, split: str) -> dict:
    """{set: {'P': {variant: (n, K) held-out probabilities, NaN where not scored}, 'tested': bool (n,)}} with the fold / variant calls of
    ``ladder.run_ladder`` (same arguments, so the same checkpoint units), without its bootstrap evaluation."""
    folds = ladder_mod.make_splits(md, split, cfg)
    rung_cols = {sp.name: (sp.has_eeg, tuple(ladder_mod.rung_columns(list(md.eeg.columns), sp))) for sp in rungs}
    variants = {ladder_mod.BASELINE_KEY: []}
    variants.update({n: list(c) for n, (h, c) in rung_cols.items() if h and c})
    out = {}
    for bname, bcols in baseline_sets.items():
        P = {k: np.full((md.n, md.K), np.nan) for k in list(variants) + ["__prior__"]}
        tested = np.zeros(md.n, bool)
        for i, (_f, tr, te, disjoint) in enumerate(folds, 1):
            with ck.tag(f"bind_e1 {split} {bname} fold {i}/{len(folds)}"):
                preds, _info = ladder_mod.fit_predict_fold(md, tr, te, bcols, variants, cfg, require_site_disjoint=disjoint)
            for k, v in preds.items():
                P[k][te] = v
            tested[te] = True
        out[bname] = {"P": P, "tested": tested}
    return out


@dataclass
class Scheme:
    name: str
    md: object
    sets: dict                                     # 'A', 'C' (ladder) and 'prior', 'agesex', 'P' (intended-use sets)
    unavailable: dict = field(default_factory=dict)

    @property
    def tested(self) -> np.ndarray:
        return self.sets["A"]["tested"]

    def scored(self, all_heldout: bool = False) -> np.ndarray:
        return self.tested if all_heldout else (self.tested & (self.md.gold_role == "eval"))

    def arm(self, set_name: str, rung: str, k: int):
        s = self.sets.get(set_name)
        if s is None or rung not in s["P"] or ladder_mod.BASELINE_KEY not in s["P"]:
            return None
        return s["P"][rung][:, k], s["P"][ladder_mod.BASELINE_KEY][:, k]


def stored_predictions(A, a, cfg, splits: list, cp) -> dict:
    """Held-out predictions of both schemes from the stored fits (see the module docstring)."""
    rungs = ladder_rungs(A)
    spec = [r for r in rsf.BASE_RUNGS if r.name == HEADLINE]
    iu_sets = {k: A.iu_sets[k] for k, _ in rsf.IU_COMPARISONS if k in A.iu_sets}
    schemes = {}
    with ck.use(cp):
        for split in splits:
            md = rsf.build_model_data(A, split, a.dev_frac, a.temporal_fraction, a.seed)
            sets = collect(md, A.baseline_sets, rungs, cfg, split)                 # primary arms must be stored (CacheMiss aborts)
            unavailable = {}
            try:                                                                    # secondary arms: best effort
                sets.update(collect(md, iu_sets, spec, cfg, split))
            except CacheMiss:
                unavailable["intended_use_sets"] = "not in the stored checkpoints (P-arm and EEG-only AUROC skipped)"
            schemes[split] = Scheme(split, md, sets, unavailable)
    return schemes


# ============================================================================================ BIND tables
def _read_local(root, kind: str, site: str, cfg) -> pd.DataFrame:
    p = Path(root) / bind.bind_key(kind, site, cfg)
    agent_safety.assert_not_restricted_in_agent(p)
    header = list(pd.read_csv(p, nrows=0, encoding="utf-8-sig").columns)
    want = (list(cfg.findings_columns.values()) if kind == "findings" else list(cfg.raw["reports_columns"].values())
            if kind == "reports" else list(cfg.raw["demographics_columns"][site].values()))
    missing = [c for c in want if c not in header]
    if missing:
        raise KeyError("expected columns not in the header")
    use = [c for c in header if c in want and c not in cfg.never_load]
    return pd.read_csv(p, dtype=str, usecols=use, encoding="utf-8-sig", low_memory=False)


def read_bind(a, cfg):
    """(sessions, findings) of the BIND site. The reports table (and its free text) is read only if the YAML uses the report Type."""
    if a.bind_data:
        demo, raw = _read_local(a.bind_data, "demographics", a.bind_site, cfg), _read_local(a.bind_data, "findings", a.bind_site, cfg)
        rep = _read_local(a.bind_data, "reports", a.bind_site, cfg) if cfg.report_type_re is not None else None
    else:
        demo = bind.read_bind_table("demographics", a.bind_site, cfg=cfg, profile=a.profile)
        raw = bind.read_bind_table("findings", a.bind_site, cfg=cfg, profile=a.profile)
        rep = bind.read_bind_table("reports", a.bind_site, cfg=cfg, profile=a.profile) if cfg.report_type_re is not None else None
    sessions = bind.normalise_sessions(demo, a.bind_site, cfg, reports=rep)
    findings = bind.classify_findings(bind.normalise_findings(raw, cfg), cfg)
    return sessions, findings


def first_study_filter(cases, sessions, findings, cfg, window):
    """Keep only, per case, the EARLIEST brain study dated inside the window (sensitivity 4b). One row per patient, so filtering the session
    and findings tables by (person, session) is enough."""
    lo, hi = window
    s = sessions[["person", "session", "study_date", "report_brain"]].copy()
    fd = findings.dropna(subset=["study_date"]).drop_duplicates(["person", "session"])[["person", "session", "study_date"]]
    s = pd.concat([s, fd.assign(report_brain=False)], ignore_index=True)
    s["study_date"] = pd.to_datetime(s["study_date"])
    s = s.sort_values("study_date").drop_duplicates(["person", "session"])
    fb = findings.groupby(["person", "session"])["is_brain"].max().rename("fb").reset_index()
    s = s.merge(fb, on=["person", "session"], how="left")
    s["brain_study"] = (s["fb"].fillna(False).astype(bool) & cfg.from_findings_brain) | s["report_brain"].astype(bool)
    c = cases[["case_id", "person_id", "t0"]].copy()
    c["person"] = c["person_id"].map(bind.norm_id)
    c["t0_day"] = pd.to_datetime(c["t0"]).dt.normalize()
    m = c.merge(s, on="person", how="left")
    m["offset"] = (m["study_date"].dt.normalize() - m["t0_day"]).dt.days
    m = m[m["offset"].between(lo, hi) & m["brain_study"]].sort_values(["case_id", "study_date", "session"]).drop_duplicates("case_id")
    keep = set(zip(m["person"], m["session"]))
    ks = pd.Series(list(zip(sessions["person"], sessions["session"])), index=sessions.index).isin(keep)
    kf = pd.Series(list(zip(findings["person"], findings["session"])), index=findings.index).isin(keep)
    return sessions[ks], findings[kf]


def label_table(cases, sessions, findings, cfg, window=None, **kw) -> pd.DataFrame:
    return bind.build_label_table(cases, sessions, findings, cfg, window, **kw).set_index("case_id").sort_index()


def label_variants(cases, sessions, findings, cfg, base: pd.DataFrame, a) -> dict:
    """Sensitivity label tables (plan section 7). None of them changes the primary."""
    w0 = cfg.window_days
    out = {}
    for nm, w in (("window [0, +1]", (0, 1)), ("window [-3, +1] (silver window)", (-3, 1)), ("window [-1, +7]", (-1, 7))):
        out[nm] = label_table(cases, sessions, findings, cfg, w)
    ind = base["reason"] == bind.REASON_INDETERMINATE
    neg = base.copy()
    neg.loc[ind, "E1_img"] = False
    pos = base.copy()
    pos.loc[ind, "E1_img"] = True
    out["indeterminate mapped to negative"], out["indeterminate mapped to positive"] = neg, pos
    out["hydrocephalus added to the label"] = label_table(cases, sessions, findings, cfg, None, include_secondary_in_primary=True)
    out["imaging on or before the EEG day [-1, 0]"] = label_table(cases, sessions, findings, cfg, (w0[0], 0))
    s2, f2 = first_study_filter(cases, sessions, findings, cfg, w0)
    out["first imaging study only"] = label_table(cases, s2, f2, cfg, None)
    cfg5 = bind.load_config(a.bind_config, overrides={"primary_classes": [c for c in cfg.primary_classes if c != "mass_effect"]})
    out["no mass-effect class"] = label_table(cases, sessions, findings, cfg5, None)
    return out


# ============================================================================================ helpers
def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def provenance(a) -> dict:
    root = HERE.parent
    h = {"configs/bind_e1.yaml": sha256_file(a.bind_config), "sortinghat/labels/bind.py": sha256_file(root / "sortinghat/labels/bind.py"),
         "docs/bind_e1_analysis_plan.md": sha256_file(root / "docs/bind_e1_analysis_plan.md"),
         "sortinghat/metrics/bind_e1.py": sha256_file(root / "sortinghat/metrics/bind_e1.py"), "scripts/run_bind_e1.py": sha256_file(__file__)}
    return {"sha256": h, "registered_config_sha256": REGISTERED_CONFIG_SHA256,
            "config_as_registered": h["configs/bind_e1.yaml"] == REGISTERED_CONFIG_SHA256, "decisions": ["D-158", "D-160"]}


def iu_mask(A) -> tuple[np.ndarray | None, dict]:
    sed = iu.sedation_features(A.baseline) if len(A.baseline.columns) else None
    om = {c: A.omop[c].to_numpy(bool) for c in A.omop.columns} if A.omop is not None and len(A.omop.columns) else None
    res = iu.intended_use_strata(A.hours, sed, A.covariates["duration_s"], iu.ceeg_task_flag(A.frame), om)
    info = {"definition": iu.DEFINITIONS[iu.IU_LARGE], "applied": res["applied"][iu.IU_LARGE], "missing": res["missing"],
            "unavailable": res["unavailable"], "omop": A.omop_info, "visit_start_source": A.hours_src}
    return res["masks"][iu.IU_LARGE], info


def site_counts(mask, sites, site_labels: dict, den=None) -> dict:
    order = sorted(site_labels)
    return iu.mask_counts(mask, np.asarray(sites).astype(str), order, site_labels, den)


def lab_arrays(lab: pd.DataFrame) -> dict:
    e1 = lab["E1_img"]
    assess = e1.notna().to_numpy(bool)
    pos = e1.fillna(False).astype(bool).to_numpy(bool)
    ctrl = lab["ctrl_chronic"]
    return {"assess": assess, "pos": pos & assess, "ctrl_assess": ctrl.notna().to_numpy(bool),
            "ctrl": ctrl.fillna(False).astype(bool).to_numpy(bool), "has_study": (lab["n_brain_studies"] > 0).to_numpy(bool),
            "reason": lab["reason"].to_numpy(object)}


def step1_counts(lab: pd.DataFrame, la: dict, masks: dict, scored: np.ndarray, sites, site_labels: dict, cfg) -> dict:
    out: dict = {"label_table": bind.summarise(lab, cfg), "strata": {}}
    for nm, m in masks.items():
        if m is None:
            out["strata"][nm] = {"available": False}
            continue
        rows = m & scored
        out["strata"][nm] = {
            "n_scored_rows": site_counts(rows, sites, site_labels),
            "n_assessable": site_counts(rows & la["assess"], sites, site_labels, rows),
            "n_imaging_positive": site_counts(rows & la["pos"], sites, site_labels, rows & la["assess"]),
            "n_ctrl_chronic": site_counts(rows & la["ctrl"], sites, site_labels, rows & la["ctrl_assess"]),
            "classes": {k: suppress_count(int((rows & lab[f"cls_{k}"].to_numpy(bool)).sum())) for k in cfg.class_names()},
            "reasons": {r: suppress_count(int((rows & (la["reason"] == r)).sum()))
                        for r in (bind.REASON_POSITIVE, bind.REASON_NEGATIVE, bind.REASON_NO_STUDY, bind.REASON_INDETERMINATE)}}
    return out


def s3_status(la: dict, masks: dict, scored: np.ndarray, min_pos: int) -> dict:
    npos = {nm: int((m & scored & la["pos"]).sum()) for nm, m in masks.items() if m is not None}
    ok = {nm: v >= min_pos for nm, v in npos.items()}
    return {"threshold": min_pos, "estimable": ok, "stop": not any(ok.values())}        # raw counts stay in memory


# ============================================================================================ analysis blocks
def rows_for(scheme: Scheme, mask, la: dict, all_heldout: bool = False) -> np.ndarray:
    return scheme.scored(all_heldout) & mask & la["assess"]


def pair_for(scheme: Scheme, arm_key: tuple, k: int, rows, y, sites, *, B, seed, site_labels, a, label="", with_ll=True):
    arm = scheme.arm(arm_key[0], arm_key[1], k)
    if arm is None:
        return {"available": False}
    p1, p0 = arm
    pr = be.Pair(y[rows], p1[rows], p0[rows], sites[rows])
    if with_ll:
        return be.pair_block(pr, B=B, seed=seed, site_labels=site_labels, min_positives=a.min_positives, label=label)
    return be.auc_pair_block(pr.y, pr.p1, pr.p0, pr.sites, B=B, seed=seed, site_labels=site_labels, min_positives=a.min_positives, label=label)


ARMS = (("A", HEADLINE, PRIMARY_ARM), ("A", "combined_cbramod", "A+combined_cbramod vs A"), ("A", "combined_morgoth", "A+combined_morgoth vs A"),
        ("C", HEADLINE, "C+combined vs C"), ("P", HEADLINE, "P+combined vs P"))


def manual_agreement(findings: pd.DataFrame, cfg) -> dict:
    """Pre-specified (D-160) agreement of Label_manual with Label_LLAMA_Level2: row-level concept flags (class regexes, negation, no acuity)
    on findings rows with a non-empty Label_manual and Brain-related = yes; per class a suppressed 2x2 table and Cohen's kappa."""
    if "label_manual" not in findings or "level2" not in findings:
        return {"available": False}
    lm = findings["label_manual"].fillna("").astype(str).str.strip()
    ok = (lm != "").to_numpy(bool) & findings["is_brain"].to_numpy(bool)
    out: dict = {"available": True, "n_rows_compared": suppress_count(int(ok.sum())), "classes": {}}
    if ok.sum() < 11:
        return out
    lm, l2 = lm[ok], findings["level2"].fillna("").astype(str).str.strip()[ok]
    out["share_identical_text"] = suppress_proportion(int((lm.str.lower() == l2.str.lower()).sum()), len(lm))

    def flags(series, cls):
        inc, exc = cfg._cls_re[cls]
        u = {t: bool(inc.search(t) and not (exc is not None and exc.search(t)) and not cfg.negation_re.search(t)) for t in series.unique()}
        return series.map(u).to_numpy(bool)
    fl = {c: (flags(lm, c), flags(l2, c)) for c in cfg.class_names()}
    fl["any_primary"] = (np.any([fl[c][0] for c in cfg.primary_classes], axis=0), np.any([fl[c][1] for c in cfg.primary_classes], axis=0))
    ren = {"silver_pos_img_pos": "manual_pos_llama_pos", "silver_pos_img_neg": "manual_pos_llama_neg", "silver_neg_img_pos": "manual_neg_llama_pos",
           "silver_neg_img_neg": "manual_neg_llama_neg"}
    for c, (m_, l_) in fl.items():                                         # the 2x2 helper is shared: first argument = manual, second = LLaMA
        out["classes"][c] = {ren.get(k, k): v for k, v in be.concordance_kappa(m_.astype(float), l_.astype(float)).items()}
    ks = [v["kappa"] for k, v in out["classes"].items() if k in cfg.primary_classes + ("any_primary",) and isinstance(v.get("kappa"), float)]
    out["any_primary_class_kappa_below_0.60"] = bool(any(x < 0.60 for x in ks)) if ks else None
    out["reading"] = "kappa < 0.60 for a primary class: every result is 'label-validity limited'; near 1 with identical text: not informative"
    return out


def run_step2(A, a, cfg, lab, la, masks, schemes: dict, sess, fnd, site_labels: dict, prov: dict) -> dict:
    sites = A.sites.astype(str)
    k = list(A.label_names).index("E1")
    silver = np.where(A.m[:, k], A.y[:, k], np.nan)
    Bp, Bs = a.n_boot_primary, a.n_boot_secondary
    y = la["pos"]
    validated = None if (a.ppv_lower is None or a.npv_lower is None) else bool(a.ppv_lower >= 0.70 and a.npv_lower >= 0.90)
    J: dict = {"label_validity": {"step_0d_ppv_lower": a.ppv_lower, "step_0d_npv_lower": a.npv_lower, "validated": validated,
                                  "reading": ("validated (lower bounds meet 0.70 / 0.90)" if validated else
                                              "label unvalidated: a not-supportive imaging result is 'inconclusive, label unvalidated'; every result is "
                                              "'label-validity limited' (S4)" if validated is None else
                                              "step 0d bounds NOT met: every result is 'label-validity limited' (S4)")}}
    J["label_validity"]["manual_vs_llama"] = manual_agreement(fnd, cfg)
    strata = {nm: m for nm, m in masks.items() if m is not None}
    J["primary"], J["secondary_arms"], J["eeg_only"], J["per_class"], J["negative_control"] = {}, {}, {}, {}, {}
    J["sedation_intensity"], J["concordance"], J["sensitivities"] = {}, {}, {}
    for sn, sch in schemes.items():
        J["primary"][sn], J["secondary_arms"][sn], J["eeg_only"][sn] = {}, {}, {}
        J["per_class"][sn], J["negative_control"][sn], J["sedation_intensity"][sn] = {}, {}, {}
        grade, ginfo = sedation_intensity_grade(A.baseline)
        for stn, m in strata.items():
            rows = rows_for(sch, m, la)
            blk = pair_for(sch, ("A", HEADLINE), k, rows, y, sites, B=Bp, seed=a.seed + 11, site_labels=site_labels, a=a, label=PRIMARY_ARM)
            J["primary"][sn][stn] = {"pair": blk, **be.verdict(blk, label_validated=validated)}
            J["secondary_arms"][sn][stn] = {nm: pair_for(sch, (s_, r_), k, rows, y, sites, B=Bs, seed=a.seed + 21 + i, site_labels=site_labels, a=a, label=nm)
                                            for i, (s_, r_, nm) in enumerate(ARMS[1:])}
            eo = sch.sets.get("prior")
            J["eeg_only"][sn][stn] = (be.single_auroc_block(y[rows], eo["P"][HEADLINE][rows, k], sites[rows], B=Bs, seed=a.seed + 31, site_labels=site_labels,
                                                           min_positives=a.min_positives)
                                      if eo is not None and HEADLINE in eo["P"] else {"available": False})
            # per class: positives = class flag, negatives = assessable imaging-E1 negatives
            pc = {}
            for ci, c in enumerate(cfg.class_names()):
                yc = lab[f"cls_{c}"].to_numpy(bool)
                rc = sch.scored() & m & la["assess"] & (yc | ~la["pos"])
                pc[c] = pair_for(sch, ("A", HEADLINE), k, rc, yc, sites, B=Bs, seed=a.seed + 41 + ci, site_labels=site_labels, a=a, label=c, with_ll=False)
            J["per_class"][sn][stn] = pc
            # negative-control outcome (chronic lesion): as defined in the label table, and against assessable E1 negatives only
            r1 = sch.scored() & m & la["ctrl_assess"]
            r2 = sch.scored() & m & la["assess"] & (la["ctrl"] | ~la["pos"])
            J["negative_control"][sn][stn] = {
                "ctrl_chronic_vs_all_brain_study_rows": pair_for(sch, ("A", HEADLINE), k, r1, la["ctrl"], sites, B=Bs, seed=a.seed + 51, site_labels=site_labels,
                                                                a=a, label="ctrl_chronic", with_ll=False),
                "ctrl_chronic_vs_assessable_E1_negatives": pair_for(sch, ("A", HEADLINE), k, r2, la["ctrl"], sites, B=Bs, seed=a.seed + 52,
                                                                   site_labels=site_labels, a=a, label="ctrl_chronic", with_ll=False)}
            if grade is None:
                J["sedation_intensity"][sn][stn] = {"available": False, "reason": ginfo.get("reason")}
            else:
                arm = sch.arm("A", HEADLINE, k)
                J["sedation_intensity"][sn][stn] = be.graded_delta_auroc(y[rows], arm[0][rows], arm[1][rows], sites[rows], grade[rows], B=Bs,
                                                                         seed=a.seed + 61, site_labels=site_labels)
        J["sedation_intensity"][sn]["grade_spec"] = "sortinghat.metrics.sedation.GRADE_SPEC (D-154)"
    J["primary_scheme_note"] = "leave-one-site-out (loso) is the primary scheme; the temporal holdout is reported alongside (plan section 5)"

    # ---- concordance silver E1 vs imaging-E1 (scored LOSO rows, then all analysed rows)
    first = next(iter(schemes.values()))
    for nm_rows, rr in (("scored_rows", first.scored()), ("all_analysed", np.ones(len(A.frame), bool))):
        img = np.where(la["assess"], la["pos"].astype(float), np.nan)
        blk = {"overall": be.concordance_kappa(silver[rr], img[rr])}
        for stn, m in strata.items():
            blk[stn] = be.concordance_kappa(silver[rr & m], img[rr & m])
        blk["per_site"] = {site_labels[s]: be.concordance_kappa(silver[rr & (sites == s)], img[rr & (sites == s)]) for s in sorted(site_labels)}
        blk["per_imaging_class"] = {}
        for c in cfg.class_names():
            yc = lab[f"cls_{c}"].to_numpy(bool)
            imgc = np.where(la["assess"] & (yc | ~la["pos"]), yc.astype(float), np.nan)        # class vs assessable imaging-E1 negatives
            blk["per_imaging_class"][c] = be.concordance_kappa(silver[rr], imgc[rr])
        J["concordance"][nm_rows] = blk

    # ---- drift and selection screens (D-153, plan section 7)
    ts = late_temporal_holdout(A.sites, A.times, a.temporal_fraction)
    tr, te = np.zeros(len(sites), bool), np.zeros(len(sites), bool)
    tr[ts.train_idx], te[ts.test_idx] = True, True
    J["drift"] = {"rule": "test/train prevalence of imaging-E1 among assessable rows: flag > 3 or < 1/3; a flagged label has no temporal interpretation, "
                          "leave-one-site-out governs", "assessable_share_by_calendar_tenth_within_site": {}, "prevalence": {}}
    for stn, m in strata.items():
        J["drift"]["assessable_share_by_calendar_tenth_within_site"][stn] = be.assessable_by_tenth(sites[m], A.times[m], la["assess"][m], site_labels)
        J["drift"]["prevalence"][stn] = {"pooled": be.prevalence_drift(la["pos"], la["assess"], tr & m, te & m),
                                         "per_site": {site_labels[s]: be.prevalence_drift(la["pos"], la["assess"], tr & m & (sites == s), te & m & (sites == s))
                                                      for s in sorted(site_labels)}}
    flagged = [(stn, w) for stn in strata for w, c in [("pooled", J["drift"]["prevalence"][stn]["pooled"])] if c.get("flag")]
    J["drift"]["flagged"] = [{"stratum": s, "where": w} for s, w in flagged]
    age = pd.to_numeric(A.frame["age_years"], errors="coerce").to_numpy(float) if "age_years" in A.frame else np.full(len(sites), np.nan)
    J["selection"] = {stn: be.window_study_share(la["has_study"][m], sites[m], age[m], site_labels) for stn, m in strata.items()}

    # ---- sensitivities (primary arm, LOSO, both strata; none changes the primary)
    sch = schemes.get("loso") or first
    cases = pd.DataFrame({"case_id": np.arange(len(sites)), "person_id": A.frame["person_id"].astype("int64").to_numpy(), "t0": pd.to_datetime(A.frame["t0"]).to_numpy()})
    variants = label_variants(cases, sess, fnd, cfg, lab, a)
    variants_arr = {nm: lab_arrays(v) for nm, v in variants.items()}

    def sens(nm, lv, rows_extra=None, all_heldout=False, seed=0):
        out = {}
        for stn, m in strata.items():
            r = sch.scored(all_heldout) & m & lv["assess"] & (True if rows_extra is None else rows_extra)
            out[stn] = pair_for(sch, ("A", HEADLINE), k, r, lv["pos"], sites, B=Bs, seed=a.seed + 71 + seed, site_labels=site_labels, a=a, label=nm)
        return out
    for i, (nm, lv) in enumerate(variants_arr.items()):
        J["sensitivities"][nm] = sens(nm, lv, seed=i)
    J["sensitivities"]["development-role rows included (all held-out rows)"] = sens("all held-out rows", la, all_heldout=True, seed=40)
    if a.mgb_sites:
        sel = np.isin(sites, [str(s) for s in a.mgb_sites])
        J["sensitivities"]["sites designated MGB: " + ",".join(site_labels.get(str(s), "site") for s in a.mgb_sites)] = sens("MGB sites", la, sel, seed=41)
    else:
        J["sensitivities"]["MGB-only sites"] = {"not_run": "which HEEDB site codes hold MGB is not documented; the per-site results of the primary block "
                                                         "are the site-level view; pass --mgb-sites to define it"}
    if a.silver_families:
        fam = pd.read_csv(a.silver_families, dtype={"person_id": "int64"})
        proc_only = A.frame["person_id"].astype("int64").map(fam.drop_duplicates("person_id").set_index("person_id")["proc_only"]).fillna(False).astype(bool).to_numpy()
        silver_pos_proc_only = proc_only & (silver > 0.5)
        J["sensitivities"]["silver-E1 positives resting only on a procedure code excluded"] = sens("no procedure-only silver E1", la, ~silver_pos_proc_only, seed=42)
    else:
        J["sensitivities"]["procedure-only silver E1 excluded"] = {"not_run": "needs --silver-families (the stored silver labels carry no anchor-family detail)"}
    return J


# ============================================================================================ report text
def _f(x, nd=3):
    return "n/a" if x is None else (x if isinstance(x, str) else f"{x:.{nd}f}")


def _est(d: dict | None, nd=3) -> str:
    if not d:
        return "n/a"
    ci = d.get("ci_99")
    v = next((d[k] for k in ("point", "delta_auroc", "slope") if k in d), None)
    return f"{_f(v, nd)} [{_f(ci['lo'], nd)}, {_f(ci['hi'], nd)}]" if ci else _f(v, nd)


def _pair_row(label: str, b: dict) -> str:
    if not b.get("estimable"):
        return f"| {label} | {b.get('n_positive', 'n/a')} | not estimable | {b.get('why', b.get('available', ''))} | | |"
    ll = b.get("delta_logloss_shifted") or {}
    return (f"| {label} | {b['n_positive']} | {_est(b['delta_auroc'])} | {_est(ll) if ll else 'n/a'} | "
            f"{_est(b.get('delta_logloss_unshifted')) if ll else 'n/a'} | {b.get('shift_mode', '')}{' (descriptive, < 200 positives)' if b.get('descriptive_only') else ''} |")


HDR = "| comparison | n positive | delta AUROC [99%] | shifted delta log loss [99%] | unshifted delta log loss [99%] | shift |\n|---|---|---|---|---|---|"


def render_markdown(J: dict) -> str:
    L = [f"# Imaging-anchored E1 (BIND): report", "", f"**{BANNER}**", "",
         "Reading: delta AUROC > 0 and delta log loss < 0 mean the EEG arm is better. Imaging is the reference, not the truth. n < 11 shown `<11`.", ""]
    L += ["## Status", "", f"- Config as registered (D-160): {J['provenance']['config_as_registered']}; S3 threshold {J['s3']['threshold']} "
          f"({'as fixed by the plan' if J['s3']['threshold'] == be.MIN_POSITIVES else 'CHANGED, not the plan value'})",
          f"- Predictions: {J['predictions']['route']}; fits loaded from the stored checkpoints {J['predictions']['fits_from_cache']}, regenerated {J['predictions']['fits_regenerated']}",
          f"- Label validity: {J['label_validity']['reading']}"] + [f"- NOTE: {x}" for x in J.get("notes", [])] + [""]
    L += ["## Step 1: imaging-positive counts (feasibility count, logged before any model-based number)", ""]
    for stn, v in J["step1"]["strata"].items():
        if not v.get("available", True):
            L.append(f"- {stn}: unavailable")
            continue
        L.append(f"- {stn}: scored rows {v['n_scored_rows']}; assessable {v['n_assessable']}; imaging-positive {v['n_imaging_positive']}; classes {v['classes']}")
    L.append("")
    L += ["## Primary pair (plan section 5): A + combined versus A, imaging-E1", ""]
    for sn, per in J["primary"].items():
        L += [f"### scheme: {sn}{' (primary)' if sn == 'loso' else ''}", "", HDR]
        for stn, v in per.items():
            L.append(_pair_row(f"{stn}", v["pair"]))
        for stn, v in per.items():
            L.append(f"- {stn}: **{v['verdict']}**" + (f" on {v['harm_on']}" if v.get("harm_on") else "") + (f" ({v.get('reading')})" if v.get("reading") else "") +
                     (f"; per-site same sign: {v.get('same_sign_at_every_site_with_ge_50_positives')}" if "same_sign_at_every_site_with_ge_50_positives" in v else "") +
                     (" [descriptive only: fewer than 200 positives]" if v.get("descriptive_only_lt_200_positives") else ""))
        L.append("")
        for stn, v in per.items():
            ps = v["pair"].get("per_site")
            if ps:
                L.append(f"- per site, {stn}: {ps}")
        L.append("")
    L += ["## Secondary arms (exploratory)", ""]
    for sn, per in J["secondary_arms"].items():
        L += [f"### scheme: {sn}", "", HDR]
        for stn, arms in per.items():
            for nm, b in arms.items():
                L.append(_pair_row(f"{stn}: {nm}", b) if b.get("n_positive") is not None else f"| {stn}: {nm} | unavailable | | | | |")
        L.append("")
        for stn, b in J["eeg_only"][sn].items():
            L.append(f"- EEG-only AUROC, {stn}: " + (_est(b.get("auroc")) if b.get("estimable") else "not estimable"))
        L.append("")
    L += ["## Per class (positives = class, negatives = assessable imaging-E1 negatives; delta AUROC only)", ""]
    for sn, per in J["per_class"].items():
        for stn, cl in per.items():
            L.append(f"- {sn} / {stn}: " + "; ".join(f"{c}: {_est(b.get('delta_auroc')) if b.get('estimable') else 'not estimable'}" for c, b in cl.items()))
    L += ["", "## Negative-control outcome (chronic lesion)", ""]
    for sn, per in J["negative_control"].items():
        for stn, d in per.items():
            for nm, b in d.items():
                L.append(f"- {sn} / {stn} / {nm}: positives {b.get('n_positive')}; " + (f"delta AUROC {_est(b.get('delta_auroc'))}" if b.get("estimable") else "not estimable"))
    L += ["", "## Concordance of silver E1 with imaging-E1", ""]
    for nm, blk in J["concordance"].items():
        L += [f"### {nm}", f"- overall: {blk['overall']}"] + [f"- {k}: {v}" for k, v in blk.items() if k in (STRATUM_IU, STRATUM_WHOLE)]
        L += [f"- per site: {blk['per_site']}", f"- per imaging class: {blk['per_imaging_class']}", ""]
    L += ["## Sedation-intensity (delta AUROC by grade; SAP A1.4 c1 mapped to AUROC)", ""]
    for sn, per in J["sedation_intensity"].items():
        for stn, b in per.items():
            if stn == "grade_spec":
                continue
            L.append(f"- {sn} / {stn}: " + (b.get("reading", b.get("why", "not available")) if isinstance(b, dict) else ""))
            if isinstance(b, dict) and b.get("estimable"):
                gs = "; ".join(f"grade {g}: {_est(v) if v.get('estimable') else 'not estimable'}" for g, v in b["grades"].items())
                st = "; ".join(f"{g}: {_est(v) if v.get('estimable') else 'not estimable'}" for g, v in b["strata"].items())
                tr = b["trend"]
                L.append(f"  - {gs}")
                L.append(f"  - {st}; heavy minus light: {_est(b['difference_heavy_minus_light']) if b['difference_heavy_minus_light'].get('estimable') else 'not estimable'}; "
                         f"site-adjusted slope per grade: {_est(tr['slope_site_adjusted_primary'], 4) if tr['slope_site_adjusted_primary'].get('estimable') else 'not estimable'}")
    L.append(f"- EEG-only sedation probe (label-independent, from the silver feasibility report): {J.get('sedation_probe')}")
    L += ["", "## Drift screen and selection", "", f"- drift: {J['drift']}", f"- share with a brain study in the window: {J['selection']}", ""]
    L += ["## Sensitivities (primary arm, LOSO; none changes the primary)", ""]
    for nm, d in J["sensitivities"].items():
        if "not_run" in d:
            L.append(f"- {nm}: not run ({d['not_run']})")
            continue
        for stn, b in d.items():
            L.append(f"- {nm} / {stn}: " + (f"positives {b['n_positive']}; delta AUROC {_est(b['delta_auroc'])}; shifted delta log loss {_est(b.get('delta_logloss_shifted'))}"
                                           if b.get("estimable") else f"positives {b.get('n_positive')}; not estimable"))
    L += ["", "## Label validity: Label_manual versus Label_LLAMA_Level2 (pre-specified, D-160)", "", f"{J['label_validity']['manual_vs_llama']}", "",
          "## Limits (stated in advance)", ""] + [f"- {x}" for x in LIMITATIONS] + [""]
    return "\n".join(L)


# ============================================================================================ driver
def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    from sortinghat.cohort.memguard import run_guarded
    return run_guarded(lambda: _run(a), a.max_memory_gb, "bind imaging-E1")


def _run(a) -> int:
    out = Path(a.out)
    if "local_only" in out.resolve().parts:
        raise SystemExit("--out holds the aggregate report and must NOT be under local_only/")
    prov = provenance(a)
    if not prov["config_as_registered"] and not a.allow_config_change:
        raise SystemExit("configs/bind_e1.yaml differs from the hash registered in D-160 (use --allow-config-change to run anyway; the report is marked)")
    cfg = bind.load_config(a.bind_config)
    from threadpoolctl import threadpool_limits
    ck.log_to_stderr()
    try:
        with threadpool_limits(limits=a.blas_threads or None):
            A = rsf.assemble(a)
            n_sites = len(np.unique(A.sites))
            splits = [s for s in a.splits if (s == "temporal" or n_sites >= 2)]
            site_labels = {s: l for s, l in A.site_labels.items() if s in set(A.sites.astype(str))}
            known = {str(p) for p in A.frame["person_id"]} | set(A.frame["rid"].astype(str))
            roots = [Path(r) for r in a.checkpoint_roots] if a.checkpoint_roots else default_roots()
            if not roots and not a.allow_refit:
                raise SystemExit("no feasibility checkpoints found (out/local_only/checkpoints/silver_feasibility); pass --checkpoint-roots or --allow-refit")
            cp = LookupCheckpoint(roots or [a.refit_scratch], a.refit_scratch if a.allow_refit else None, a.allow_refit)
            fcfg = ladder_cfg(A, a)

            if a.check_cache:                                      # no BIND data is read, no OMOP proxy, no store
                schemes = stored_predictions(A, a, fcfg, splits, cp)
                safe_print(f"check-cache: schemes {list(schemes)}, fits loaded {cp.n_hits}, missing {cp.n_miss}; secondary sets unavailable: "
                           f"{ {k: v.unavailable for k, v in schemes.items() if v.unavailable} }", known_ids=known)
                return 0

            store = None
            if a.data:
                agent_safety.assert_not_restricted_in_agent(a.data)
                store = data_io.open_store(a.data)
            elif a.s3:
                store = data_io.open_store(None, profile=a.profile)
            if A.omop is None and not A.omop_info:
                rsf.attach_omop(A, a, store)
            mask_iu, iu_info = iu_mask(A)
            masks = {STRATUM_IU: mask_iu, STRATUM_WHOLE: np.ones(len(A.frame), bool)}

            # ---------------------------------------------------------------- step 1: labels and counts, before any prediction
            sessions, findings = read_bind(a, cfg)
            cases = pd.DataFrame({"case_id": np.arange(len(A.frame)), "person_id": A.frame["person_id"].astype("int64").to_numpy(),
                                  "t0": pd.to_datetime(A.frame["t0"]).to_numpy()})
            if cases["person_id"].duplicated().any():
                raise SystemExit("the analysed cohort has more than one row per patient; the label table assumes one")
            lab = label_table(cases, sessions, findings, cfg)
            la = lab_arrays(lab)
            md0 = rsf.build_model_data(A, splits[0], a.dev_frac, a.temporal_fraction, a.seed)
            scored0 = (md0.gold_role == "eval")
            s1 = step1_counts(lab, la, masks, scored0, A.sites, site_labels, cfg)
            s3 = s3_status(la, masks, scored0, a.min_positives)
            J1 = {"banner": BANNER, "provenance": prov, "step": "1 (feasibility count; no model-based number computed)", "scheme_for_counts": splits[0],
                  "iu_large": iu_info, "step1": s1, "s3": {k: v for k, v in s3.items()}}
            safe_write_json(out / "step1_counts.json", J1, known)
            safe_print(BANNER, known_ids=known)
            for stn, v in s1["strata"].items():
                safe_print(f"step 1 {stn}: scored {v.get('n_scored_rows')} assessable {v.get('n_assessable')} imaging-positive {v.get('n_imaging_positive')}",
                           known_ids=known)
            safe_print(f"stop rule S3 (>= {a.min_positives} imaging-positive cases): estimable {s3['estimable']}; stop={s3['stop']}", known_ids=known)
            if s3["stop"] or a.counts_only:
                if s3["stop"]:
                    safe_write_text(out / "report.md", f"# Imaging-anchored E1 (BIND)\n\n**{BANNER}**\n\nStopped by rule S3: fewer than {a.min_positives} "
                                                       "imaging-positive cases in every stratum. Suppressed counts are in step1_counts.json.\n", known)
                return 0

            # ---------------------------------------------------------------- step 2
            schemes = stored_predictions(A, a, fcfg, splits, cp)
            J = run_step2(A, a, cfg, lab, la, masks, schemes, sessions, findings, site_labels, prov)
            J["banner"], J["provenance"], J["step1"], J["iu_large"], J["s3"] = BANNER, prov, s1, iu_info, s3
            J["notes"] = ([] if mask_iu is not None else ["IU-large is unavailable (see iu_large.unavailable): the primary stratum cannot be evaluated; only the whole broad cohort is reported"])
            J["predictions"] = {"route": ("regenerated deterministically (--allow-refit) where missing, otherwise loaded" if a.allow_refit else
                                          "loaded from the stored feasibility checkpoints (require-cache)"),
                                "fits_from_cache": cp.n_hits, "fits_regenerated": cp.n_miss if a.allow_refit else 0,
                                "schemes": list(schemes), "unavailable": {k: v.unavailable for k, v in schemes.items() if v.unavailable},
                                "ladder_settings": {"seed": a.seed, "dev_fraction": a.dev_frac, "temporal_fraction": a.temporal_fraction, "n_boot": a.n_boot,
                                                    "cohort_definition": a.cohort_def}}
            J["settings"] = {"n_boot_primary": a.n_boot_primary, "n_boot_secondary": a.n_boot_secondary, "seed": a.seed, "window_days": list(cfg.window_days),
                             "primary_classes": list(cfg.primary_classes), "bind_site": a.bind_site}
            try:
                import json
                J["sedation_probe"] = json.loads(Path(a.feasibility_report).read_text())["sedation_intensity"].get("probe", "not in the report")
            except (OSError, KeyError, ValueError):
                J["sedation_probe"] = "feasibility report not available"
            J["limitations"] = list(LIMITATIONS)
    finally:
        ck.log_to_stderr(False)
    safe_write_json(out / "report.json", J, known)
    safe_write_text(out / "report.md", render_markdown(J), known)
    for sn, per in J["primary"].items():
        for stn, v in per.items():
            b = v["pair"]
            safe_print(f"  {sn} {stn}: " + (f"delta AUROC {_est(b['delta_auroc'])}; shifted delta log loss {_est(b.get('delta_logloss_shifted'))}; verdict {v['verdict']}"
                                          if b.get("estimable") else f"not estimable ({b.get('why')})"), known_ids=known)
    safe_print(f"Report: {out / 'report.md'} and {out / 'report.json'} (aggregate-only)", known_ids=known)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:                       # never print a traceback (it could carry values); class name only
        print(f"run_bind_e1 failed: {type(e).__name__}", file=sys.stderr)
        raise SystemExit(1)
