"""Sampling design for the gold adjudication pilot (pure functions, SYNTHETIC frames only)."""
import numpy as np
import pandas as pd
import pytest

from sortinghat.adjudication import aggregates as agg
from sortinghat.adjudication import sampling as sp
from sortinghat.safe_output import SUPPRESSED


def make_frame(n_per_site=(600, 400), seed=0, rare=(0.03, 0.05, 0.05, 0.08, 0.10, 0.25)):
    """Eligible frame with a silver pattern mix (rare etiologies are rare)."""
    rng = np.random.default_rng(seed)
    rows = []
    pid = 5_000_000
    for si, n in enumerate(n_per_site):
        pats = rng.choice(list(sp.DEFAULT_PRIORITY) + [sp.ALL_NEGATIVE], size=n, p=[*rare, 1 - sum(rare)])
        for p in pats:
            rows.append({"person_id": pid, "site": f"S000{si + 1}", "pattern": p, "t0": "2024-01-01 00:00:00"})
            pid += 1
    return pd.DataFrame(rows)


def test_pattern_uses_the_fixed_priority_and_ignores_unassessable():
    Y = pd.DataFrame({"E1": [1, 1, 0, np.nan, 0], "E2": [0, 1, 0, 1, np.nan], "E7": [0, 0, 0, 0, 1], "E5": [0, 0, 0, np.nan, 0]})
    pat = sp.assign_pattern(Y)
    # E7 beats E2 beats E1; no positive -> all_negative (NaN ignored)
    assert pat.tolist() == ["E1", "E2", sp.ALL_NEGATIVE, "E2", "E7"]


def test_waterfill_caps_and_redistributes():
    a = sp.waterfill({"x": 5, "y": 100, "z": 100}, {"x": 0.5, "y": 0.25, "z": 0.25}, 40)
    assert a["x"] == 5 and a["y"] + a["z"] == 35 and abs(a["y"] - a["z"]) <= 1 and sum(a.values()) == 40
    assert sum(sp.waterfill({"a": 3, "b": 2}, {"a": 1, "b": 1}, 50).values()) == 5          # cannot exceed the population
    assert sp.waterfill({"a": 0, "b": 4}, {"a": 1, "b": 1}, 3) == {"a": 0, "b": 3}
    assert sp.waterfill({"a": 10}, {"a": 0.0}, 5) == {"a": 0}                                  # no share -> never sampled


def test_site_allocation_modes():
    assert sum(sp.allocate_sites({"A": 600, "B": 400}, 200, "proportional").values()) == 200
    assert sp.allocate_sites({"A": 600, "B": 400}, 200, "proportional") == {"A": 120, "B": 80}
    assert sp.allocate_sites({"A": 600, "B": 400}, 200, "equal") == {"A": 100, "B": 100}
    with pytest.raises(ValueError):
        sp.allocate_sites({"A": 1}, 1, "nope")


def test_draw_is_deterministic_and_seed_sensitive():
    f = make_frame()
    a = sp.draw_sample(f, n=200, seed=7)
    b = sp.draw_sample(f.sample(frac=1.0, random_state=3), n=200, seed=7)       # input order must not matter
    c = sp.draw_sample(f, n=200, seed=8)
    pd.testing.assert_frame_equal(a, b)
    assert set(a.loc[a["role"] == "sample", "person_id"]) != set(c.loc[c["role"] == "sample", "person_id"])


def test_sample_size_roles_and_no_overlap():
    f = make_frame()
    s = sp.draw_sample(f, n=200, seed=1, dup_frac=0.10, n_calibration=5)
    prim, dup, cal = (s[s["role"] == r] for r in (sp.ROLE_SAMPLE, sp.ROLE_DUPLICATE, sp.ROLE_CALIBRATION))
    assert (len(prim), len(dup), len(cal)) == (200, 20, 5)
    assert prim["person_id"].is_unique
    assert set(dup["person_id"]) <= set(prim["person_id"])
    assert not set(cal["person_id"]) & set(prim["person_id"])                    # calibration comes from patients NOT sampled
    # a duplicate points at its original's sample id, and shares its person
    orig = prim.set_index("sample_id")["person_id"]
    assert (dup["duplicate_of"].map(orig).to_numpy() == dup["person_id"].to_numpy()).all()
    assert s["sample_id"].is_unique and s["sample_id"].str.fullmatch(r"s\d{4}").all()
    assert (dup["weight"] == 0).all() and (cal["weight"] == 0).all()


def test_rare_patterns_are_enriched_relative_to_the_population():
    f = make_frame()
    s = sp.draw_sample(f, n=200, seed=2)
    prim = s[s["role"] == sp.ROLE_SAMPLE]
    pop_share = f["pattern"].value_counts(normalize=True)
    smp_share = prim["pattern"].value_counts(normalize=True)
    for p in ("E7", "E2", "E4a", "E6"):
        assert smp_share[p] > 1.5 * pop_share[p]
    assert smp_share[sp.ALL_NEGATIVE] < pop_share[sp.ALL_NEGATIVE]                # the dominant stratum is under-sampled
    # E7 is so rare here (3% of 1000 = ~30) that the target share (10% of 200 = 20) is met without taking the whole stratum
    assert (prim["pattern"] == "E7").sum() >= 15


def test_small_stratum_is_taken_whole_and_the_shortfall_is_shared():
    f = make_frame(n_per_site=(300, 300), rare=(0.01, 0.05, 0.05, 0.08, 0.10, 0.25))        # ~3 E7 per site
    s = sp.draw_sample(f, n=100, seed=3)
    prim = s[s["role"] == sp.ROLE_SAMPLE]
    for site, g in f.groupby("site"):
        pop_e7 = int((g["pattern"] == "E7").sum())
        assert int(((prim["site"] == site) & (prim["pattern"] == "E7")).sum()) == pop_e7       # whole stratum
    assert len(prim) == 100                                                                   # shortfall went to other strata
    assert (prim.loc[prim["pattern"] == "E7", "weight"] == 1.0).all()


def test_weights_reweight_the_sample_to_the_eligible_population():
    f = make_frame()
    s = sp.draw_sample(f, n=200, seed=4)
    prim = s[s["role"] == sp.ROLE_SAMPLE]
    chk = sp.check_weights(s, f)
    assert chk == {"weights_sum_to_population": True, "min_weight_ge_1": True}
    assert prim["weight"].sum() == pytest.approx(len(f))
    # per-stratum: weight = population / sampled, inclusion probability its inverse
    g = prim.groupby("stratum").agg(w=("weight", "first"), pop=("stratum_pop", "first"), n=("stratum_n", "first"), n_rows=("weight", "size"))
    assert (g["n"] == g["n_rows"]).all() and np.allclose(g["w"], g["pop"] / g["n"])
    assert np.allclose(prim["inclusion_prob"] * prim["weight"], 1.0)
    # the re-weighted pattern shares recover the population shares exactly (Horvitz-Thompson)
    est = prim.groupby("pattern")["weight"].sum() / prim["weight"].sum()
    assert np.allclose(est.sort_index(), f["pattern"].value_counts(normalize=True).sort_index())


def test_reweighted_prevalence_is_unbiased_for_a_planted_gold_label():
    """A gold label depending on the silver pattern: the weighted estimate recovers the IU-large prevalence, the raw sample mean does not."""
    f = make_frame(n_per_site=(2000, 1500), seed=5)
    rng = np.random.default_rng(11)
    p_gold = f["pattern"].map({"E7": 0.8, "E2": 0.7, "E4a": 0.6, "E6": 0.5, "E5": 0.5, "E1": 0.4, sp.ALL_NEGATIVE: 0.05})
    f["gold"] = (rng.random(len(f)) < p_gold.to_numpy()).astype(float)
    truth = f["gold"].mean()
    ests, raws = [], []
    for seed in range(40):
        s = sp.draw_sample(f, n=300, seed=seed)
        prim = s[s["role"] == sp.ROLE_SAMPLE]              # the carried column "gold" travels with the rows
        ests.append(np.average(prim["gold"], weights=prim["weight"]))
        raws.append(prim["gold"].mean())
    assert abs(np.mean(ests) - truth) < 0.01
    assert np.mean(raws) - truth > 0.1                                           # the enriched sample over-states prevalence


def test_frame_must_have_one_row_per_patient_and_shares_for_every_pattern():
    f = make_frame(n_per_site=(50, 50))
    with pytest.raises(ValueError):
        sp.draw_sample(pd.concat([f, f.iloc[:1]]), n=10)
    with pytest.raises(ValueError):
        sp.draw_sample(f, n=10, shares={"E1": 1.0})


def test_kish_ess():
    assert sp.kish_ess([1, 1, 1, 1]) == pytest.approx(4.0)
    assert sp.kish_ess([1, 1, 1, 9]) < 4.0
    assert sp.kish_ess([]) == 0.0


# ------------------------------------------------------------------------------------------------ aggregates
def test_suppress_row_hides_small_cells_and_their_complement():
    assert agg.suppress_row({"a": 30, "b": 40}) == {"a": 30, "b": 40}
    assert agg.suppress_row({"a": 3, "b": 40, "c": 50}) == {"a": SUPPRESSED, "b": SUPPRESSED, "c": 50}          # one hidden -> smallest shown too
    r = agg.suppress_row({"a": 3, "b": 4, "c": 50})
    assert r == {"a": SUPPRESSED, "b": SUPPRESSED, "c": 50}


def test_pooled_row_is_hidden_when_a_site_cell_is():
    pooled = agg.pooled_row({"s1": {"x": 5, "y": 40}, "s2": {"x": 60, "y": 70}})
    assert pooled["x"] == SUPPRESSED and pooled["y"] == SUPPRESSED                                   # y: the smallest shown is linked to hidden x
    pooled = agg.pooled_row({"s1": {"x": 20, "y": 40}, "s2": {"x": 60, "y": 70}})
    assert pooled == {"x": 80, "y": 110}


def test_share_cell_rules():
    assert agg.share_cell(30, 100)["share"] == 0.3
    assert agg.share_cell(5, 100)["share"] == SUPPRESSED and agg.share_cell(5, 100)["n"] == SUPPRESSED
    assert agg.share_cell(97, 100)["n"] == SUPPRESSED                                                # complement 3 < 11
    assert agg.share_cell(5, 8)["den"] == SUPPRESSED
    cells = agg.pooled_share_cells({"s1": (5, 100), "s2": (50, 100)})
    assert cells["all_sites"]["n"] == SUPPRESSED and cells["s2"]["n"] == 50 and cells["all_sites"]["why"].startswith("linked")
    # a cell hidden because almost everyone has it says "high" with a lower bound; one hidden because almost no one has it says "low"
    hi, lo = agg.share_cell(195, 200), agg.share_cell(4, 200)
    assert hi["why"] == "high" and hi["share_bound"] == ">=0.950" and lo["why"] == "low" and "share_bound" not in lo
    both_high = agg.pooled_share_cells({"s1": (98, 100), "s2": (99, 100)})
    assert both_high["all_sites"]["why"] == "high" and both_high["all_sites"]["share_bound"] == ">=0.950"
