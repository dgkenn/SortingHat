import numpy as np
import pytest

from sortinghat.montage import select_electrode_subset, select_within_folds

CH = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz"]


def make(n=200, seed=0, informative=("F3", "T3", "Cz", "Fp2", "F8")):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    X = rng.standard_normal((n, len(CH), 6))
    for j, c in enumerate(informative):
        X[:, CH.index(c)] += (1.5 - 0.2 * j) * y[:, None]
    return X, y


def class_gap_importance(X, y, names):
    return np.abs(X[y == 1].mean(axis=(0, 2)) - X[y == 0].mean(axis=(0, 2)))


def gap_score(Xs, y, names):
    return float(np.abs(Xs[y == 1].mean(axis=(0, 2)) - Xs[y == 0].mean(axis=(0, 2))).sum())


def test_importance_topk_picks_informative():
    X, y = make()
    tr = np.arange(len(y))
    sel = select_electrode_subset(X, y, tr, CH, 4, importance_fn=class_gap_importance)
    assert set(sel.electrodes) <= {"F3", "T3", "Cz", "Fp2", "F8"} and len(sel.electrodes) == 4
    assert sel.method == "importance_topk"


def test_greedy_forward_picks_informative_and_forced():
    X, y = make()
    sel = select_electrode_subset(X, y, np.arange(len(y)), CH, 5, score_fn=gap_score, forced=["Fz"])
    assert sel.electrodes[0] == "Fz" and len(sel.electrodes) == 5
    assert {"F3", "T3"} <= set(sel.electrodes)


def test_callbacks_only_see_training_rows_and_readonly():
    X, y = make(100)
    tr, te = np.arange(0, 70), np.arange(70, 100)
    X[te] = np.nan                                   # held-out poison: any peek would break finiteness
    seen = []

    def score(Xs, ys, names):
        seen.append(Xs.shape[0])
        assert not np.isnan(Xs).any() and not Xs.flags.writeable
        return float(Xs.mean())

    sels = select_within_folds(X, y, [(tr, te)], CH, 4, score_fn=score)
    assert set(seen) == {70} and sels[0].n_train == 70 and set(sels[0].train_idx).isdisjoint(te)


def test_leak_is_rejected():
    X, y = make(50)
    with pytest.raises(ValueError, match="overlaps"):
        select_electrode_subset(X[:30], y[:30], np.arange(30), CH, 4,
                                importance_fn=class_gap_importance, heldout_idx=np.arange(25, 50))
    with pytest.raises(ValueError, match="exactly the rows"):
        select_electrode_subset(X, y, np.arange(30), CH, 4, importance_fn=class_gap_importance)


def test_argument_validation():
    X, y = make(50)
    tr = np.arange(50)
    with pytest.raises(ValueError):
        select_electrode_subset(X, y, tr, CH, 3, importance_fn=class_gap_importance)
    with pytest.raises(ValueError):
        select_electrode_subset(X, y, tr, CH, 9, importance_fn=class_gap_importance)
    with pytest.raises(ValueError):
        select_electrode_subset(X, y, tr, CH, 4)
    with pytest.raises(ValueError):
        select_electrode_subset(X, y, tr, CH, 4, importance_fn=class_gap_importance, score_fn=gap_score)
    with pytest.raises(ValueError):
        select_electrode_subset(X, y, tr, CH, 4, importance_fn=lambda *a: np.ones(3))
