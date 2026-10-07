"""Circularity audit (plan: "Silver labels: the circularity rules", D-010).

Rule: if a silver-trained model agrees MORE with EEG-report impressions than with gold
labels, the silver labels are leaking and get rebuilt on objective anchors.

Inputs are per-case, per-label aligned vectors on cases that are (i) held out from silver
training and (ii) have both a gold label and an EEG-report-impression label:

    pred          silver-trained model's predicted probability (cross-fitted / held-out)
    eeg_impr      binary label read from the EEG report impression (mapped to the ontology)
    gold          binary gold label (probable/definite = 1)

Agreement metric: AUROC of ``pred`` against each reference (threshold-free, so unaffected by
calibration shift between silver-trained output and the references), or Cohen's kappa at
``threshold``. Decision rule follows the plan literally (point estimate): leak =
agreement(eeg_impr) > agreement(gold). A seeded paired bootstrap of the difference is also
returned (``leak_confident`` = lower 95% bound of the difference > 0) for context; the
rebuild trigger is ``leak``. Outputs are aggregate-only (n < 11 suppressed).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from ..safe_output import SUPPRESSED, SUPPRESS_BELOW
from .stats import auroc, cohen_kappa


class SilverScoringError(ValueError):
    """Raised when silver-training cases overlap the audit cases (silver never scores)."""


@dataclass
class LabelAudit:
    label: str
    n: int | str
    agreement_eeg_impression: float | str
    agreement_gold: float | str
    difference: float | str          # eeg - gold
    diff_ci_low: float | str
    diff_ci_high: float | str
    leak: bool | None                # plan rule (point estimate); None if undefined/suppressed
    leak_confident: bool | None


def _agree(metric: str, ref: np.ndarray, pred: np.ndarray, threshold: float) -> float:
    if metric == "auroc":
        return auroc(ref, pred)
    if metric == "kappa":
        return cohen_kappa(ref, (pred >= threshold).astype(int), categories=[0, 1])
    raise ValueError(f"unknown metric {metric!r}")


def audit_label(label: str, pred, eeg_impr, gold, *, metric: str = "auroc", threshold: float = 0.5,
                n_boot: int = 1000, seed: int = 0, min_n: int = SUPPRESS_BELOW) -> LabelAudit:
    pred = np.asarray(pred, float); eeg = np.asarray(eeg_impr).astype(int); gold = np.asarray(gold).astype(int)
    if not (len(pred) == len(eeg) == len(gold)):
        raise ValueError("pred, eeg_impr and gold must be aligned")
    ok = ~(np.isnan(pred) | np.isnan(eeg.astype(float)) | np.isnan(gold.astype(float)))
    pred, eeg, gold = pred[ok], eeg[ok], gold[ok]
    n = len(pred)
    if n < min_n:
        return LabelAudit(label, SUPPRESSED, SUPPRESSED, SUPPRESSED, SUPPRESSED, SUPPRESSED, SUPPRESSED, None, None)
    a_eeg = _agree(metric, eeg, pred, threshold)
    a_gold = _agree(metric, gold, pred, threshold)
    if np.isnan(a_eeg) or np.isnan(a_gold):
        return LabelAudit(label, n, SUPPRESSED, SUPPRESSED, SUPPRESSED, SUPPRESSED, SUPPRESSED, None, None)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        d = _agree(metric, eeg[ix], pred[ix], threshold) - _agree(metric, gold[ix], pred[ix], threshold)
        if not np.isnan(d):
            diffs.append(d)
    lo, hi = (np.percentile(diffs, [2.5, 97.5]) if diffs else (np.nan, np.nan))
    diff = a_eeg - a_gold
    return LabelAudit(label, n, round(a_eeg, 4), round(a_gold, 4), round(diff, 4),
                      None if np.isnan(lo) else round(float(lo), 4),
                      None if np.isnan(hi) else round(float(hi), 4),
                      bool(a_eeg > a_gold), None if np.isnan(lo) else bool(lo > 0))


def run_circularity_audit(df: pd.DataFrame, labels: Iterable[str], *, silver_train_ids: Iterable[str] | None = None,
                          metric: str = "auroc", threshold: float = 0.5, n_boot: int = 1000,
                          seed: int = 0) -> dict:
    """``df`` long format with columns case_id, label, pred, eeg_impr, gold (one row per case/label).

    ``silver_train_ids``: if given, any overlap with audit cases raises (silver labels train
    models, never score them; the model must also be held out from these cases).
    """
    need = {"case_id", "label", "pred", "eeg_impr", "gold"}
    if not need <= set(df.columns):
        raise ValueError(f"missing columns: {sorted(need - set(df.columns))}")
    if silver_train_ids is not None and set(df["case_id"]) & set(silver_train_ids):
        raise SilverScoringError("audit cases overlap silver-training cases")
    per: dict[str, dict] = {}
    for lab in labels:
        g = df[df["label"] == lab]
        per[lab] = asdict(audit_label(lab, g["pred"].values, g["eeg_impr"].values, g["gold"].values,
                                      metric=metric, threshold=threshold, n_boot=n_boot, seed=seed))
    leaking = sorted(l for l, r in per.items() if r["leak"])
    return {"metric": metric, "rule": "leak if agreement(EEG impression) > agreement(gold)",
            "per_label": per, "leaking_labels": leaking, "rebuild_silver_labels": bool(leaking)}
