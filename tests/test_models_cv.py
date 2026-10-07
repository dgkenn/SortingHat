import numpy as np
import pytest

from sortinghat.models import LadderConfig, fit_predict_fold
from sortinghat.models.ladder import make_splits
from sortinghat.models.synthetic import make_synthetic_study

CFG = LadderConfig(n_boot=50)


@pytest.fixture(scope="module")
def data():
    return make_synthetic_study(n_per_site=250, signal=1.0, seed=7)


def _variants(d):
    return {"__baseline__": [], "qeeg": [c for c in d.eeg.columns if c.startswith("qeeg.")],
            "combined": list(d.eeg.columns)}


def _corrupt_heldout(d, test_idx, rng):
    """Change everything about the held-out rows: features, labels, roles, masks, covariates."""
    e = d.copy()
    e.baseline.iloc[test_idx, :] = rng.normal(1e6, 1e6, (len(test_idx), e.baseline.shape[1]))
    e.baseline.iloc[test_idx[::3], 0] = np.nan
    e.eeg.iloc[test_idx, :] = rng.normal(-1e5, 1e5, (len(test_idx), e.eeg.shape[1]))
    e.eeg.iloc[test_idx[::2], 1] = np.nan
    e.y_silver[test_idx] = 1 - e.y_silver[test_idx]
    e.y_gold[test_idx] = 1 - e.y_gold[test_idx]
    e.m_silver[test_idx] = ~e.m_silver[test_idx]
    e.gold_role[test_idx] = "dev"          # even a "dev" role on a held-out row must not matter
    e.m_gold[test_idx] = True
    return e


def test_heldout_rows_never_influence_fitted_parameters(data):
    sp = make_splits(data, "loso", CFG)
    for _name, tr, te, disjoint in sp:
        p0, _i0, m0 = fit_predict_fold(data, tr, te, list(data.baseline.columns), _variants(data), CFG,
                                       require_site_disjoint=disjoint, return_models=True)
        bad = _corrupt_heldout(data, te, np.random.default_rng(0))
        p1, _i1, m1 = fit_predict_fold(bad, tr, te, list(data.baseline.columns), _variants(data), CFG,
                                       require_site_disjoint=disjoint, return_models=True)
        for k in m0:
            assert m0[k].fingerprint() == m1[k].fingerprint(), f"fitted parameters changed for {k}"
            np.testing.assert_array_equal(m0[k].pre.mean_, m1[k].pre.mean_)
            np.testing.assert_array_equal(m0[k].pre.median_, m1[k].pre.median_)
            np.testing.assert_array_equal(m0[k].pre.keep_, m1[k].pre.keep_)
        np.testing.assert_array_equal(p0["__prior__"], p1["__prior__"])


def test_leak_detector_is_sensitive_to_training_rows(data):
    """Sanity check on the test above: corrupting TRAIN rows must change the fitted parameters."""
    _n, tr, te, _d = make_splits(data, "loso", CFG)[0]
    _, _, m0 = fit_predict_fold(data, tr, te, list(data.baseline.columns), _variants(data), CFG, return_models=True)
    bad = _corrupt_heldout(data, tr[:50], np.random.default_rng(1))
    _, _, m1 = fit_predict_fold(bad, tr, te, list(data.baseline.columns), _variants(data), CFG, return_models=True)
    assert any(m0[k].fingerprint() != m1[k].fingerprint() for k in m0)


def test_eval_gold_labels_of_training_sites_do_not_enter_fit(data):
    _n, tr, te, _d = make_splits(data, "loso", CFG)[0]
    _, _, m0 = fit_predict_fold(data, tr, te, list(data.baseline.columns), _variants(data), CFG, return_models=True)
    e = data.copy()
    ev = tr[e.gold_role[tr] == "eval"]
    e.y_gold[ev] = 1 - e.y_gold[ev]                      # gold eval labels in training sites are never read
    _, _, m1 = fit_predict_fold(e, tr, te, list(data.baseline.columns), _variants(data), CFG, return_models=True)
    assert all(m0[k].fingerprint() == m1[k].fingerprint() for k in m0)
    # while dev gold labels DO drive selection / recalibration
    dv = tr[e.gold_role[tr] == "dev"]
    e2 = data.copy()
    e2.y_gold[dv] = 1 - e2.y_gold[dv]
    _, _, m2 = fit_predict_fold(e2, tr, te, list(data.baseline.columns), _variants(data), CFG, return_models=True)
    assert any(m0[k].fingerprint() != m2[k].fingerprint() for k in m0)


def test_split_guards(data):
    n = data.n
    with pytest.raises(ValueError):
        fit_predict_fold(data, np.arange(0, 100), np.arange(50, 150), ["age"], {"__baseline__": []}, CFG)
    s = data.sites
    tr = np.flatnonzero(s != s[0])
    te = np.flatnonzero(s == s[0])
    with pytest.raises(ValueError):                       # site not held out
        fit_predict_fold(data, np.concatenate([tr, te[:10]]), te[10:], ["age"], {"__baseline__": []}, CFG,
                         require_site_disjoint=True)


def test_temporal_holdout_trains_on_earlier_cases_only(data):
    (_n, tr, te, disjoint), = make_splits(data, "temporal", CFG)
    assert not disjoint
    for s in np.unique(data.sites):
        t_tr = data.times[tr][data.sites[tr] == s]
        t_te = data.times[te][data.sites[te] == s]
        assert t_tr.max() < t_te.min()
    with pytest.raises(ValueError):
        make_splits(data.replace(times=None), "temporal", CFG)
