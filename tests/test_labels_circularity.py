import numpy as np
import pandas as pd
import pytest

from sortinghat.labels.circularity_audit import SilverScoringError, audit_label, run_circularity_audit
from sortinghat.labels.stats import auroc, cohen_kappa


def synth(n=300, seed=0, leak=False):
    rng = np.random.default_rng(seed)
    gold = rng.integers(0, 2, n)
    eeg = np.where(rng.random(n) < 0.8, gold, 1 - gold)       # EEG impression: 80% agrees with gold
    if leak:
        pred = eeg + rng.normal(0, 0.3, n)                      # model tracks the EEG reader
    else:
        pred = gold + rng.normal(0, 0.5, n)                     # model tracks gold
    return pred, eeg, gold


def test_stats_sanity():
    assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]) == 1.0
    assert auroc([0, 1], [0.5, 0.5]) == 0.5
    assert np.isnan(auroc([1, 1], [0.1, 0.2]))
    assert cohen_kappa([0, 1, 0, 1], [0, 1, 0, 1]) == 1.0
    assert abs(cohen_kappa([0, 0, 1, 1], [0, 1, 0, 1])) < 1e-9
    assert cohen_kappa([0, 1, 2, 3], [0, 1, 2, 3], weights="quadratic") == 1.0
    # weighted penalises near misses less than far misses
    near = cohen_kappa([0, 1, 2, 3] * 5, [1, 2, 3, 3] * 5, weights="linear")
    far = cohen_kappa([0, 1, 2, 3] * 5, [3, 3, 0, 0] * 5, weights="linear")
    assert near > far


def test_no_leak_when_model_tracks_gold():
    r = audit_label("E5", *synth(leak=False), n_boot=200)
    assert r.leak is False and r.agreement_gold > r.agreement_eeg_impression


def test_leak_flagged_when_model_tracks_eeg_reader():
    r = audit_label("E5", *synth(leak=True), n_boot=200)
    assert r.leak is True and r.leak_confident is True


def test_small_n_suppressed():
    r = audit_label("E5", [0.1] * 5, [0, 1, 0, 1, 0], [0, 1, 1, 0, 0])
    assert r.n == "<11" and r.leak is None


def test_run_audit_overall_flag_and_silver_overlap_guard():
    rows = []
    for lab, leak in (("E1", False), ("E2", True)):
        p, e, g = synth(leak=leak, seed=hash(lab) % 100)
        rows.append(pd.DataFrame({"case_id": [f"c{i}" for i in range(len(p))], "label": lab,
                                  "pred": p, "eeg_impr": e, "gold": g}))
    df = pd.concat(rows)
    res = run_circularity_audit(df, ["E1", "E2"], n_boot=100)
    assert res["leaking_labels"] == ["E2"] and res["rebuild_silver_labels"] is True
    with pytest.raises(SilverScoringError):
        run_circularity_audit(df, ["E1"], silver_train_ids=["c3"])
    with pytest.raises(ValueError):
        run_circularity_audit(df.drop(columns="gold"), ["E1"])
