import numpy as np
import pytest

from sortinghat.metrics import check_site_requirements, late_temporal_holdout, leave_one_site_out


def test_loso_partitions_each_site_once():
    sites = np.array(["A"] * 5 + ["B"] * 4 + ["C"] * 3)
    splits = list(leave_one_site_out(sites))
    assert sorted(s.held_out for s in splits) == ["A", "B", "C"]
    for s in splits:
        assert set(s.train_idx).isdisjoint(s.test_idx)
        assert len(s.train_idx) + len(s.test_idx) == len(sites)
        assert set(sites[s.test_idx]) == {s.held_out}
        assert s.held_out not in set(sites[s.train_idx])


def test_loso_needs_two_sites():
    with pytest.raises(ValueError):
        list(leave_one_site_out(np.array(["A", "A"])))


def test_site_requirements():
    sites = np.array(["A"] * 300 + ["B"] * 350 + ["C"] * 299)
    r = check_site_requirements(sites)
    assert not r["passes"] and r["sites_meeting_min"] == ["A", "B"]
    r2 = check_site_requirements(np.array(["A"] * 300 + ["B"] * 300 + ["C"] * 300))
    assert r2["passes"]


def test_late_temporal_holdout_takes_latest_per_site():
    rng = np.random.default_rng(0)
    sites = np.repeat(["A", "B"], 100)
    times = np.concatenate([rng.permutation(100), 1000 + rng.permutation(100)]).astype(float)
    sp = late_temporal_holdout(sites, times, test_fraction=0.2)
    assert set(sp.train_idx).isdisjoint(sp.test_idx)
    for s in ("A", "B"):
        te = sp.test_idx_by_site[s]
        tr = [i for i in sp.train_idx if sites[i] == s]
        assert len(te) == pytest.approx(20, abs=2)
        assert times[te].min() >= sp.cutoffs[s]
        assert times[tr].max() < sp.cutoffs[s]


def test_late_temporal_holdout_embargo_and_datetimes():
    sites = np.repeat(["A"], 100)
    times = np.datetime64("2020-01-01") + np.arange(100).astype("timedelta64[D]")
    no = late_temporal_holdout(sites, times, 0.2)
    sec_per_day = 86400
    em = late_temporal_holdout(sites, times, 0.2, embargo=5 * sec_per_day)
    assert len(em.train_idx) == len(no.train_idx) - 5
    assert np.array_equal(em.test_idx, no.test_idx)


def test_temporal_rejects_nan_time():
    with pytest.raises(ValueError):
        late_temporal_holdout(np.array(["A", "A"]), np.array([1.0, np.nan]))
