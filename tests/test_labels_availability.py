"""Label-level data-source availability (D-153): E6 is NOT ASSESSABLE (NaN, never 0) where the blood-culture source cannot
fire for a case's site-era. SYNTHETIC fixtures only (hand-built OMOP rows)."""
import numpy as np
import pandas as pd
import pytest

from sortinghat.labels import availability as av
from sortinghat.labels import extract as ex
from sortinghat.labels.anchors import load_anchor_config
from sortinghat.safe_output import SUPPRESSED, assert_aggregate_only
from test_labels_extract import Fx, T0  # noqa: F401  (tests/ is on sys.path)

CFG = load_anchor_config()


def test_yaml_rule_is_present_and_resolves_to_the_anchor_window():
    r = av.resolve_rules(CFG)["E6"]
    assert r["source_item"] == "blood_culture_drawn" and r["window_hours"] == [-72.0, 24.0]
    assert r["bin_fraction"] == 0.10 and r["min_share"] == 0.10 and r["reason"] == "culture_source_absent"
    assert av.resolve_rules(CFG, {"E6": {"window_hours": [-10, 5], "min_share": 0.2}})["E6"]["window_hours"] == [-10.0, 5.0]
    with pytest.raises(ValueError):
        av.resolve_rules(CFG, {"E6": {"bin_fraction": 0.9}})
    with pytest.raises(ValueError):
        av.resolve_rules(CFG, {"E9": {"source_item": "x"}})


def test_binning_is_by_rank_with_a_floor_on_bin_size():
    assert av.bin_site(1000, 0.10, 50) == 10 and av.bin_site(200, 0.10, 50) == 4
    assert av.bin_site(99, 0.10, 50) == 0                      # fewer than 2 * min_bin_cases: not evaluated
    t0 = np.array(["2020-01-03", "2020-01-01", "2020-01-02", "2020-01-01"], dtype="datetime64[us]")
    assert av.rank_bins(t0, 2).tolist() == [1, 0, 1, 0]        # ranks 3, 0, 2, 1 (the two 01-01 rows keep input order)


# ---------------------------------------------------------------------------------------------- unit level
def _frame(n, present, positive=(), t0=None):
    ids = [f"c{i:04d}" for i in range(n)]
    labels = pd.DataFrame({"E6": pd.array([i in set(positive) for i in range(n)], dtype="boolean")},
                          index=pd.Index(ids, name="case_id"))
    cases = pd.DataFrame({"case_id": ids, "person_id": range(n), "SiteID": "S",
                          "t0": pd.Timestamp("2024-01-01") + pd.to_timedelta(range(n) if t0 is None else t0, unit="D")})
    ev = pd.DataFrame({"case_id": [ids[i] for i in present], "item": "blood_culture_drawn", "value": 1.0, "hours_from_t0": -5.0})
    return labels, cases, ev


def test_source_absent_region_is_not_assessable_and_positives_survive():
    # 200 cases in one site; cultures only for the last 100 (every second case) and one early positive (case 10 carries a culture)
    present = [10] + list(range(100, 200, 2))
    labels, cases, ev = _frame(200, present, positive=[10, 120])
    res = av.apply_availability(labels, cases, ev, CFG)
    e6 = res.labels["E6"]
    assert e6.iloc[:100].isna().sum() == 99 and e6.iloc[10] == True                  # noqa: E712  every early case but the positive
    assert e6.iloc[100:].notna().all() and e6.iloc[120] == True                      # noqa: E712  late era unchanged
    assert (res.reason.iloc[:100] == "culture_source_absent").sum() == 99
    pos_before = labels["E6"].fillna(False).to_numpy(bool)
    assert (res.labels["E6"].fillna(False).to_numpy(bool) == pos_before).all()        # no positive is ever lost or created
    i = res.info["E6"]["per_site"]["S"]
    assert i["n_bins"] == 4 and i["n_bins_absent"] == 2 and i["n_not_assessable"] == 99 and i["n_positive_kept_in_absent_bins"] == 1


def test_keep_cases_with_source_keeps_negatives_that_have_a_culture():
    present = [10, 11] + list(range(100, 200, 2))
    labels, cases, ev = _frame(200, present, positive=[10])
    res = av.apply_availability(labels, cases, ev, CFG, {"E6": {"keep_cases_with_source": True}})
    e6 = res.labels["E6"]
    assert e6.iloc[11] == False and e6.iloc[10] == True and e6.iloc[:100].isna().sum() == 98       # noqa: E712


def test_uniformly_low_or_high_share_and_small_sites():
    labels, cases, ev = _frame(200, list(range(0, 200, 5)))                            # 20% everywhere: nothing flagged
    assert av.apply_availability(labels, cases, ev, CFG).labels["E6"].notna().all()
    labels, cases, ev = _frame(60, [])                                                 # no cultures at all but < 2 bins: not evaluated
    res = av.apply_availability(labels, cases, ev, CFG)
    assert res.labels["E6"].notna().all() and res.info["E6"]["per_site"]["S"]["evaluated"] is False
    labels, cases, ev = _frame(200, [])                                                # site with no culture feed at all: all NA
    assert av.apply_availability(labels, cases, ev, CFG).labels["E6"].isna().all()


def test_window_matters_cultures_outside_the_anchor_window_do_not_count_as_source():
    labels, cases, ev = _frame(200, list(range(200)))
    ev["hours_from_t0"] = 200.0                                                        # all cultures days after t0, outside [-72, 24]
    assert av.apply_availability(labels, cases, ev, CFG).labels["E6"].isna().all()


def test_rule_depends_only_on_source_presence_and_t0_rank():
    present = list(range(100, 200, 2))

    def na(**kw):
        labels, cases, ev = _frame(200, present, **kw)
        return av.apply_availability(labels, cases, ev, CFG).labels["E6"].isna().to_numpy()
    base = na()
    assert (na(t0=np.arange(200) * 7) == base).all()                       # any monotone spacing of t0 gives the same ranks
    assert (na(positive=[150, 160]) == base).all()                         # outcomes in the assessed era change nothing
    protected, idx3 = na(positive=[3]), np.arange(200) == 3
    assert not protected[3] and base[3] and ((protected | idx3) == (base | idx3)).all()    # an early positive is protected, nothing else moves
    labels, cases, ev = _frame(200, present)
    reversed_t0 = cases.assign(t0=cases["t0"].iloc[::-1].to_numpy())      # calendar reversed: the NA region follows the rank
    out = av.apply_availability(labels, reversed_t0, ev, CFG).labels["E6"].isna().to_numpy()
    assert out[:100].all() and not out[100:].any()                         # cases 100-199 (the cultures) are now the EARLY ranks


# ------------------------------------------------------------------------------------- end to end (OMOP rows)
def build_cohort_fixture(n=200, early_cut=101, early_positive=10):
    """One site, patient i recorded at t0 + 24 i h. Early era (i < early_cut): no culture rows at all (the feed did not exist),
    except one fully qualifying sepsis case. Late era: every second patient has a culture; every fourth qualifies for E6."""
    fx = Fx()
    rows = []
    for i in range(1, n + 1):
        fx.person(i, "S0001")
        o = 24.0 * i
        rows.append(T0 + pd.Timedelta(hours=o))
        qualifies = (i == early_positive) or (i >= early_cut and i % 4 == 0)
        has_culture = qualifies or (i >= early_cut and i % 2 == 0)
        if has_culture:
            fx.lab(i, o - 5, "Blood culture", np.nan, None, vtext="No growth")
        if qualifies or (i < early_cut and i % 3 == 0):   # early era: antibiotics (and a lactate) exist without any culture
            for d in range(4):
                fx.drug(i, o - 4 + 24 * d, "VANCOMYCIN 1 G IV", end_h=o - 2 + 24 * d, route="IV")
            fx.lab(i, o - 3, "LACTATE, WHOLE BLOOD", 3.4, "mmol/L")
        fx.lab(i, o - 2, "GLUCOSE", 100, "mg/dL")
    cohort = fx.cohort().assign(t0=rows)
    return fx, cohort


@pytest.fixture(scope="module")
def runs():
    fx, cohort = build_cohort_fixture()
    new = ex.extract_silver(fx.tables(), cohort)
    legacy = ex.extract_silver(fx.tables(), cohort, config=ex.ExtractConfig(apply_availability_rules=False))
    return new, legacy, cohort


def test_e6_is_not_assessable_in_the_culture_free_era_and_unchanged_later(runs):
    new, legacy, cohort = runs
    e_new, e_old = new.labels["E6"], legacy.labels["E6"]
    ids = [str(i) for i in range(1, 201)]
    early, late = ids[:100], ids[100:]
    assert e_old.notna().all() and e_old.loc[early].sum() == 1                       # legacy: a confident negative for 99 early cases
    assert e_new.loc[early].isna().sum() == 99 and e_new.loc["10"] == True           # noqa: E712  NA, but the early positive survives
    assert e_new.loc[late].notna().all()
    assert (e_new.loc[late].astype(bool) == e_old.loc[late].astype(bool)).all()      # late era identical to the legacy labels
    assert e_new.loc[late].sum() == 25
    for lab in ("E1", "E2", "E4a", "E5", "E7"):                                      # other labels are untouched
        pd.testing.assert_series_equal(new.labels[lab], legacy.labels[lab])
    assert (new.na_reason.loc[early] == "culture_source_absent").sum() == 99 and (new.na_reason.loc[late] == "").all()


def test_no_positive_is_ever_made_not_assessable(runs):
    new, legacy, _ = runs
    was_pos = legacy.labels.drop(columns=[c for c in legacy.labels if c.startswith("e4b_")]).fillna(False).astype(bool)
    still = new.labels[was_pos.columns].fillna(False).astype(bool)
    assert (still == was_pos).all().all()                                            # every legacy positive is still positive


def test_silver_report_counts_are_suppressed_and_prevalence_uses_assessable_cases(runs):
    new, legacy, _ = runs
    rep = ex.silver_report(new)
    assert_aggregate_only(rep, known_ids=set(new.labels.index))
    a = rep["availability"]["E6"]
    assert a["reason"] == "culture_source_absent" and a["n_not_assessable"] == 99
    site = a["per_site"]["S0001"]
    assert site["n_bins"] == 4 and site["n_bins_source_absent"] == 2 and site["n_not_assessable"] == 99
    assert site["n_positive_kept_in_absent_bins"] == SUPPRESSED                      # one positive: < 11
    e6 = rep["per_label"]["E6"]
    assert e6["n_not_assessable"] == 99 and e6["prevalence"] == round(26 / 101, 4)   # 26 positives over 101 assessable cases
    assert rep["per_label"]["E5"].get("n_not_assessable") is None                     # fully assessable labels carry no such field
    assert ex.silver_report(legacy)["availability"] == {}


def test_overrides_and_switch(runs):
    fx, cohort = build_cohort_fixture()
    off = ex.extract_silver(fx.tables(), cohort, config=ex.ExtractConfig(availability_overrides={"E6": {"enabled": False}}))
    assert off.labels["E6"].notna().all()
    strict = ex.extract_silver(fx.tables(), cohort, config=ex.ExtractConfig(availability_overrides={"E6": {"min_share": 0.9}}))
    assert strict.labels["E6"].isna().sum() > 99                                      # a stricter threshold flags the half-covered late era too
    assert strict.labels["E6"].loc[[str(i) for i in range(101, 201)]].fillna(False).astype(bool).sum() == 25   # positives kept
