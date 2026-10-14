"""Agreement / silver-vs-gold statistics and the analysis script, on SYNTHETIC rater data only."""
import json
import re
import stat
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sortinghat.adjudication import agreement as ag
from sortinghat.adjudication import crf
from sortinghat.adjudication import packets as pk
from sortinghat.adjudication import sampling as sp
from sortinghat.safe_output import SUPPRESSED
from test_silver_feasibility_inputs import load_script

az = load_script("adjudication_analyze")
CFG = crf.load_crf()
LABELS = crf.label_codes(CFG)
INV = {v: k for k, v in crf.state_names(CFG).items()}
PREV = {"E1": 0.30, "E2": 0.15, "E3": 0.12, "E4a": 0.15, "E4b": 0.20, "E5": 0.30, "E6": 0.25, "E7": 0.10, "other": 0.12}
SILVER = {"E1": (0.75, 0.90), "E2": (0.5, 0.95), "E4a": (0.6, 0.93), "E5": (0.65, 0.88), "E6": (0.7, 0.9), "E7": (0.5, 0.97)}      # (sens, spec)


# ------------------------------------------------------------------------------------------------ synthetic study
def rate(rng, truth, flip):
    """State name per case from a binary truth with flip noise."""
    obs = truth ^ (rng.random(len(truth)) < flip)
    pos = rng.choice(["probable", "definite"], size=len(truth), p=[0.4, 0.6])
    neg = rng.choice(["absent", "possible"], size=len(truth), p=[0.8, 0.2])
    return np.where(obs, pos, neg)


def export_rows(case_ids, truth, rng, rater, flip, unassessable=0.0):
    d = {"case_id": list(case_ids), "rater_id": rater}
    pos_labels = np.zeros(len(case_ids), dtype=object)
    first = np.full(len(case_ids), "unknown", dtype=object)
    for lab in LABELS:
        st = rate(rng, truth[lab], flip)
        if unassessable:
            st = np.where(rng.random(len(st)) < unassessable, "unassessable", st)
        d[f"{lab.lower()}_state"] = [INV[s] for s in st]
        d[f"{lab.lower()}_conf"] = np.where(st == "unassessable", "", rng.integers(2, 6, len(st)).astype(str))
        is_pos = np.isin(st, ["probable", "definite"])
        first = np.where((first == "unknown") & is_pos, lab, first)
    d["primary_etiology"] = first
    d["minutes_spent"] = rng.integers(8, 26, len(case_ids)).astype(str)
    d["eeg_seen"] = np.where(rng.random(len(case_ids)) < 0.05, "1", "0")
    return pd.DataFrame(d)


def make_study(seed=0, N=4000, n=240, flip=0.04, n_dup=24, third=True):
    """A population with gold truth and a noisy silver label, drawn with the REAL sampling design, rated by R1/R2 (+R3 on disputes),
    with duplicates re-rated.  Returns dict(export, id_map, sample, pop, truth)."""
    rng = np.random.default_rng(seed)
    truth_pop = {lab: rng.random(N) < PREV[lab] for lab in LABELS}
    sil = {}
    for lab, (se, sp_) in SILVER.items():
        sil[lab] = np.where(truth_pop[lab], rng.random(N) < se, rng.random(N) < 1 - sp_).astype(float)
    pop = pd.DataFrame({"person_id": 7_000_000 + np.arange(N), "site": np.where(np.arange(N) % 5 < 3, "S0001", "S0002"), "t0": "2024-01-01 00:00:00"})
    for lab in SILVER:
        pop[f"silver_{lab}"] = sil[lab]
    pop["pattern"] = sp.assign_pattern(pop[[f"silver_{l}" for l in SILVER]].rename(columns=lambda c: c[len("silver_"):]))
    for lab in LABELS:
        pop[f"truth_{lab}"] = truth_pop[lab]
    smp = sp.draw_sample(pop, n=n, seed=seed + 1, dup_frac=n_dup / n, n_calibration=5)
    cids = pk.make_case_ids(len(smp), seed + 2)
    smp["case_id"] = cids
    orig_case = dict(zip(smp["sample_id"], smp["case_id"]))
    idm = smp[["case_id", "sample_id", "role", "person_id", "site"]].copy()
    idm["duplicate_of_case"] = smp["duplicate_of"].map(orig_case).fillna("")
    truth = {lab: smp[f"truth_{lab}"].to_numpy(bool) for lab in LABELS}
    rated = smp["role"] != sp.ROLE_CALIBRATION
    cases = smp.loc[rated, "case_id"].to_numpy()
    tr = {lab: truth[lab][rated.to_numpy()] for lab in LABELS}
    r1 = export_rows(cases, tr, rng, "R1", flip)
    r2 = export_rows(cases, tr, rng, "R2", flip)
    parts = [r1, r2]
    if third:
        a = np.column_stack([r1[f"{l.lower()}_state"].map(lambda c: c in ("2", "3")).to_numpy() for l in LABELS])
        b = np.column_stack([r2[f"{l.lower()}_state"].map(lambda c: c in ("2", "3")).to_numpy() for l in LABELS])
        dispute = (a != b).any(axis=1)
        sub = {lab: tr[lab][dispute] for lab in LABELS}
        parts.append(export_rows(cases[dispute], sub, rng, "R3", flip / 2))
    return {"export": pd.concat(parts, ignore_index=True), "id_map": idm, "sample": smp, "pop": pop, "truth": tr, "cases": cases}


@pytest.fixture(scope="module")
def study():
    return make_study(seed=3)


def kappa_ref(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe)


# ------------------------------------------------------------------------------------------------ kernels
def test_binary_kappa_matches_the_textbook_formula_and_pabak():
    rng = np.random.default_rng(0)
    t = rng.random(300) < 0.3
    a = (t ^ (rng.random(300) < 0.1)).astype(float)
    b = (t ^ (rng.random(300) < 0.1)).astype(float)
    r = ag.binary_agreement(a, b, boot=False)
    assert r["kappa"] == pytest.approx(round(kappa_ref(a, b), 3))
    assert r["pabak"] == pytest.approx(round(2 * (a == b).mean() - 1, 3))
    assert r["percent_agreement"] == pytest.approx(round((a == b).mean(), 3))
    # also equals the library implementation
    from sortinghat.labels.stats import cohen_kappa
    assert r["kappa"] == pytest.approx(round(cohen_kappa(a.astype(int), b.astype(int), categories=[0, 1]), 3))


def test_prevalence_and_bias_indices_have_the_documented_sign():
    a = np.array([1.0] * 40 + [0.0] * 60)
    b = np.array([1.0] * 55 + [0.0] * 45)
    r = ag.binary_agreement(a, b, boot=False)
    assert r["bias_index"] > 0 and r["prevalence_rater_b"] > r["prevalence_rater_a"]        # rater b calls more positives
    swapped = ag.binary_agreement(b, a, boot=False)
    assert swapped["bias_index"] == pytest.approx(-r["bias_index"])


def test_pabak_rescues_kappa_at_extreme_prevalence():
    rng = np.random.default_rng(1)
    t = rng.random(1000) < 0.04
    a, b = (t ^ (rng.random(1000) < 0.01)).astype(float), (t ^ (rng.random(1000) < 0.01)).astype(float)
    r = ag.binary_agreement(a, b, B=300, seed=0)
    assert r["percent_agreement"] > 0.97 and r["pabak"] > 0.95 and r["kappa"] < r["pabak"] - 0.1
    assert abs(r["prevalence_index"]) > 0.8


def test_integer_weights_equal_row_replication():
    rng = np.random.default_rng(2)
    a = (rng.random(60) < 0.4).astype(float)
    b = np.where(rng.random(60) < 0.8, a, 1 - a)
    w = rng.integers(1, 4, 60).astype(float)
    rep = np.repeat(np.arange(60), w.astype(int))
    k_w = ag.binary_agreement(a, b, w, boot=False)["kappa"]
    k_rep = ag.binary_agreement(a[rep], b[rep], None, boot=False)["kappa"]
    assert k_w == pytest.approx(k_rep, abs=1e-3)


def test_fleiss_textbook_example_and_edge_cases():
    # Wikipedia's worked example: 10 subjects, 14 raters, 5 categories -> kappa = 0.210
    counts = np.array([[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1], [7, 7, 0, 0, 0],
                       [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]])
    M = np.array([np.repeat(np.arange(5), c) for c in counts])
    assert ag.fleiss_kappa(M, range(5)) == pytest.approx(0.210, abs=5e-4)
    assert ag.fleiss_kappa(np.array([[1, 1], [0, 0], [1, 1], [0, 0]], float), (0, 1)) == pytest.approx(1.0)
    assert ag.fleiss_kappa(np.array([[1, 1], [0, 0], [1, 0], [0, 1]], float), (0, 1)) == pytest.approx(0.0)
    assert np.isnan(ag.fleiss_kappa(np.full((0, 2), np.nan), (0, 1)))
    assert ag.fleiss_kappa(np.array([[1, np.nan], [0, 0], [1, 1], [0, 0]], float), (0, 1)) == pytest.approx(1.0)      # incomplete rows dropped


def test_categorical_and_weighted_kappa():
    a = np.array([0, 1, 2, 2, 1, 0, 2, 1]); b = np.array([0, 1, 2, 1, 1, 0, 2, 2])
    from sklearn.metrics import cohen_kappa_score
    assert ag.categorical_kappa(a, b, 3) == pytest.approx(cohen_kappa_score(a, b))
    T = ag._ord_table(a.astype(float), b.astype(float), np.ones(8), k=3)
    assert ag._wkappa_table(T, "linear") == pytest.approx(cohen_kappa_score(a, b, weights="linear"))
    assert ag._wkappa_table(T, "quadratic") == pytest.approx(cohen_kappa_score(a, b, weights="quadratic"))


# ------------------------------------------------------------------------------------------------ bootstrap
def test_bootstrap_ci_brackets_the_estimate_and_narrows_with_n():
    rng = np.random.default_rng(5)

    def run(n):
        t = rng.random(n) < 0.35
        a, b = (t ^ (rng.random(n) < 0.12)).astype(float), (t ^ (rng.random(n) < 0.12)).astype(float)
        return ag.binary_agreement(a, b, B=800, seed=1)
    small, large = run(120), run(1200)
    for r in (small, large):
        lo, hi = r["kappa_ci95"]
        assert lo < r["kappa"] < hi
    assert large["kappa_ci95"][1] - large["kappa_ci95"][0] < 0.6 * (small["kappa_ci95"][1] - small["kappa_ci95"][0])


def test_bootstrap_coverage_is_near_nominal():
    rng = np.random.default_rng(6)
    true_kappa, hit, reps = None, 0, 120
    # large-sample truth from the same generator
    t = rng.random(200000) < 0.3
    A = t ^ (rng.random(200000) < 0.1); Bm = t ^ (rng.random(200000) < 0.1)
    true_kappa = kappa_ref(A, Bm)
    for r in range(reps):
        t = rng.random(150) < 0.3
        a, b = (t ^ (rng.random(150) < 0.1)).astype(float), (t ^ (rng.random(150) < 0.1)).astype(float)
        lo, hi = ag.binary_agreement(a, b, B=300, seed=r)["kappa_ci95"]
        hit += lo <= true_kappa <= hi
    assert 0.85 <= hit / reps <= 1.0


def test_stratified_bootstrap_keeps_stratum_sizes():
    strata = np.array(["a"] * 30 + ["b"] * 10)
    ix = ag.boot_indices(40, strata, 50, seed=0)
    assert ix.shape == (50, 40)
    assert ((ix < 30).sum(axis=1) == 30).all() and ((ix >= 30).sum(axis=1) == 10).all()
    assert np.array_equal(ix, ag.boot_indices(40, strata, 50, seed=0))


# ------------------------------------------------------------------------------------------------ disclosure
def test_small_cells_are_suppressed():
    a = np.array([1.0] * 5 + [0.0] * 40); b = a.copy()
    r = ag.binary_agreement(a, b)
    assert r["kappa"] == SUPPRESSED and r["kappa_ci95"] == SUPPRESSED and r["n_positive_either"] == SUPPRESSED and r["n_pairs"] == 45
    few = ag.binary_agreement(np.array([1.0, 0, 1, 0]), np.array([1.0, 0, 0, 0]))
    assert few["n_pairs"] == SUPPRESSED and few["kappa"] == SUPPRESSED
    ok = ag.binary_agreement(np.array([1.0] * 20 + [0.0] * 20), np.array([1.0] * 18 + [0.0] * 22), B=100)
    assert isinstance(ok["kappa"], float)
    assert ag.binary_agreement(np.array([1.0] * 20 + [0.0] * 20), np.array([1.0] * 18 + [0.0] * 22), B=100, k=1)["n_pairs"] == 40


# ------------------------------------------------------------------------------------------------ consensus
def panel_from(rows):
    return ag.Panel(pd.DataFrame(rows, columns=["case_id", "rater", "label", "state"]), ["E1"], ["R1", "R2", "R3"])


def test_consensus_rules():
    P = [("a", "R1", "E1", "definite"), ("a", "R2", "E1", "probable"),                       # agree positive
         ("b", "R1", "E1", "absent"), ("b", "R2", "E1", "possible"),                           # agree negative (possible is negative)
         ("c", "R1", "E1", "definite"), ("c", "R2", "E1", "absent"), ("c", "R3", "E1", "probable"),      # dispute -> third positive
         ("d", "R1", "E1", "definite"), ("d", "R2", "E1", "absent"),                           # dispute, no third -> pending
         ("e", "R1", "E1", "unassessable"), ("e", "R2", "E1", "probable"),                     # one valid call only
         ("f", "R1", "E1", "unassessable"), ("f", "R2", "E1", "absent"), ("f", "R3", "E1", "absent"),     # two valid negatives
         ("g", "R1", "E1", "definite"), ("g", "R2", "E1", "unassessable"), ("g", "R3", "E1", "absent")]   # tie -> missing
    pan = panel_from(P)
    gold, src = ag.consensus(pan, ("R1", "R2"), "R3")["E1"]
    got = {c: (None if np.isnan(g) else int(g), s) for c, g, s in zip(pan.cases, gold, src)}
    assert got == {"a": (1, "agree"), "b": (0, "agree"), "c": (1, "third"), "d": (None, "pending"), "e": (None, "missing"),
                   "f": (0, "majority"), "g": (None, "missing")}


def test_disagreement_queue_lists_cases_and_labels():
    P = pd.DataFrame([("a", "R1", "E1", "definite"), ("a", "R2", "E1", "absent"), ("a", "R1", "E2", "absent"), ("a", "R2", "E2", "absent"),
                      ("b", "R1", "E1", "probable"), ("b", "R2", "E1", "definite"), ("b", "R1", "E2", "probable"), ("b", "R2", "E2", "possible")],
                     columns=["case_id", "rater", "label", "state"])
    q = ag.disagreement_queue(ag.Panel(P, ["E1", "E2"], ["R1", "R2"]))
    assert q.set_index("case_id")["labels_in_dispute"].to_dict() == {"a": "E1", "b": "E2"}


# ------------------------------------------------------------------------------------------------ silver vs gold
def test_silver_vs_gold_point_estimates_by_hand():
    gold = np.array([1] * 40 + [1] * 20 + [0] * 30 + [0] * 60, float)
    silver = np.array([1] * 40 + [0] * 20 + [1] * 30 + [0] * 60, float)         # TP 40, FN 20, FP 30, TN 60
    w = np.ones(150)
    r = ag.silver_vs_gold(gold, silver, w, None, B=200, seed=0)
    assert r["sensitivity"]["weighted"] == pytest.approx(40 / 60, abs=1e-3) and r["specificity"]["weighted"] == pytest.approx(60 / 90, abs=1e-3)
    assert r["ppv"]["weighted"] == pytest.approx(40 / 70, abs=1e-3) and r["npv"]["weighted"] == pytest.approx(60 / 80, abs=1e-3)
    # weights: TN rows count 3x -> specificity (and NPV) move, sensitivity does not
    w2 = np.where((gold == 0) & (silver == 0), 3.0, 1.0)
    r2 = ag.silver_vs_gold(gold, silver, w2, None, B=200, seed=0)
    assert r2["sensitivity"]["weighted"] == r["sensitivity"]["weighted"]
    assert r2["specificity"]["weighted"] == pytest.approx(180 / (180 + 30), abs=1e-3) and r2["specificity"]["unweighted"] == r["specificity"]["unweighted"]
    lo, hi = r["sensitivity"]["weighted_ci95"]
    assert lo < r["sensitivity"]["weighted"] < hi


def test_silver_vs_gold_suppresses_small_cells_and_ignores_unassessable_silver():
    gold = np.array([1] * 8 + [0] * 100, float)
    silver = np.array([1] * 5 + [0] * 3 + [0] * 90 + [1] * 10, float)
    r = ag.silver_vs_gold(gold, silver, np.ones(108), None, B=100)
    assert r["sensitivity"]["weighted"] == SUPPRESSED and r["n_gold_positive"] == SUPPRESSED
    assert isinstance(r["specificity"]["weighted"], float) is False                              # FP = 10 < 11
    silver2 = silver.copy(); silver2[:3] = np.nan
    assert ag.silver_vs_gold(gold, silver2, np.ones(108), None, B=100)["n_silver_not_assessable"] == SUPPRESSED


def test_weighted_silver_vs_gold_tracks_the_population_on_the_study(study):
    """With the design weights the sensitivity / specificity / prevalence describe IU-large (single draw: loose tolerances)."""
    smp, pop = study["sample"], study["pop"]
    prim = smp[smp["role"] == sp.ROLE_SAMPLE]
    for lab in ("E1", "E4a", "E5", "E6"):
        gold = prim[f"truth_{lab}"].to_numpy(float)
        silver = prim[f"silver_{lab}"].to_numpy(float)
        w = prim["weight"].to_numpy(float)
        strata = prim["stratum"].to_numpy(object)
        r = ag.silver_vs_gold(gold, silver, w, strata, B=200, seed=0)
        t, s = pop[f"truth_{lab}"].to_numpy(bool), pop[f"silver_{lab}"].to_numpy(bool)
        true_sens, true_spec = (s & t).sum() / t.sum(), (~s & ~t).sum() / (~t).sum()
        assert abs(r["sensitivity"]["weighted"] - true_sens) < 0.12 and abs(r["specificity"]["weighted"] - true_spec) < 0.06
        pv = ag.prevalence_block(gold, w, strata, B=200, seed=0)
        assert abs(pv["prevalence_weighted"] - t.mean()) < 0.06


def test_weighting_removes_the_enrichment_bias_in_sensitivity_for_a_rare_label():
    """Averaged over many draws: the weighted E7 sensitivity is unbiased, the raw sample's is inflated (E7-silver-positive rows are oversampled)."""
    rng = np.random.default_rng(0)
    N = 6000
    t = rng.random(N) < 0.10
    s = np.where(t, rng.random(N) < 0.5, rng.random(N) < 0.03)
    other = rng.random(N) < 0.3
    pop = pd.DataFrame({"person_id": np.arange(N), "site": "S1", "pattern": np.where(s, "E7", np.where(other, "E1", sp.ALL_NEGATIVE)), "t": t, "s": s})
    true = (s & t).sum() / t.sum()
    w_est, u_est = [], []
    for seed in range(40):
        smp = sp.draw_sample(pop, n=200, seed=seed, shares={"E7": 0.4, "E1": 0.2, sp.ALL_NEGATIVE: 0.4}, dup_frac=0, n_calibration=0)
        r = ag.silver_vs_gold(smp["t"].to_numpy(float), smp["s"].to_numpy(float), smp["weight"].to_numpy(float), smp["stratum"].to_numpy(object),
                              B=20, seed=0, k=0)
        w_est.append(r["sensitivity"]["weighted"]); u_est.append(r["sensitivity"]["unweighted"])
    assert abs(np.mean(w_est) - true) < 0.04
    assert np.mean(u_est) - true > 0.08


# ------------------------------------------------------------------------------------------------ inter / intra on the study
def test_interrater_on_the_synthetic_study_matches_the_planted_noise(study):
    rep, queue = az.analyze(study["export"], study["id_map"], study["sample"], CFG, B=300, seed=1)
    assert "error" not in rep
    for lab in ("E1", "E5", "E6"):
        e = rep["labels"][lab]["inter_rater"]
        # two raters each flipping 4% of the truth: expected kappa ~ 0.84 for a ~30% label
        assert 0.7 < e["kappa"] < 0.97 and e["kappa_ci95"][0] < e["kappa"] < e["kappa_ci95"][1]
        assert e["percent_agreement"] > 0.9
        # the probable/definite and absent/possible splits are random in the synthetic ratings, so the 4-state kappa is lower than the binary one
        assert 0.3 < rep["labels"][lab]["inter_rater_ordinal"]["kappa_linear_weighted"] < e["kappa"]
    assert rep["labels"]["E7"]["inter_rater"]["n_pairs"] == 240
    assert len(queue) > 0 and set(queue.columns) == {"case_id", "labels_in_dispute"}
    # the third rater resolved the disputes: no pending among the cases queued
    src = rep["labels"]["E1"]["consensus_sources"]
    assert src["pending"] == SUPPRESSED or src["pending"] == 0 or src["pending"] < 11
    # primary-etiology agreement and Gate 0 blocks exist
    assert rep["primary_etiology"]["kappa"] > 0.6 and set(rep["gate0"]) == {"on_enriched_sample", "on_iu_large_weighted_prevalence", "note"}
    am = rep["amended_primary_labels"]                                    # SAP Amendment A1 labels
    assert list(am["labels"]) == ["E1", "E2", "E5"] and am["n_labels"] == 3
    assert am["labels"]["E1"]["kappa_meets_threshold"] is True and am["labels"]["E1"]["prevalence_meets_threshold"] is True
    assert am["n_kappa_meets"] == sum(v["kappa_meets_threshold"] is True for v in am["labels"].values())


def test_gold_prevalence_is_reweighted_in_the_report(study):
    rep, _ = az.analyze(study["export"], study["id_map"], study["sample"], CFG, B=200, seed=1)
    pop = study["pop"]
    for lab in ("E1", "E5", "E6"):
        g = rep["labels"][lab]["gold"]
        assert abs(g["prevalence_weighted"] - pop[f"truth_{lab}"].mean()) < 0.07
    g7 = rep["labels"]["E7"]["gold"]                                  # the rare, oversampled label
    assert abs(g7["prevalence_weighted"] - pop["truth_E7"].mean()) < 0.07


def test_intrarater_agreement_on_duplicates(study):
    ex, idm = study["export"], study["id_map"]
    long = crf.long_ratings(ex, CFG)
    pairs = ag.duplicate_pairs(long, idm)
    assert len(pairs) == 24
    res = ag.intrarater(long, pairs, ["E1", "E5"], ["R1", "R2"], B=200)
    assert res["pooled"]["E1"]["n_pairs"] == 48                      # 24 duplicates x 2 raters
    k = res["pooled"]["E1"]["kappa"]
    assert isinstance(k, float) and 0.4 < k <= 1.0
    assert set(res["by_rater"]) == {"rater_1", "rater_2"}
    assert res["by_rater"]["rater_1"]["E1"]["n_pairs"] == 24


def test_a_rater_who_is_perfectly_consistent_has_intra_kappa_one():
    rng = np.random.default_rng(0)
    base = rng.random(60) < 0.4
    rows = []
    for i, v in enumerate(base):
        s = "definite" if v else "absent"
        rows += [(f"o{i}", "R1", "E1", s), (f"d{i}", "R1", "E1", s)]
    long = pd.DataFrame(rows, columns=["case_id", "rater", "label", "state"])
    pairs = pd.DataFrame({"orig": [f"o{i}" for i in range(60)], "dup": [f"d{i}" for i in range(60)]})
    assert ag.intrarater(long, pairs, ["E1"], ["R1"], B=100)["pooled"]["E1"]["kappa"] == 1.0


# ------------------------------------------------------------------------------------------------ quality checks
def test_consistency_and_blinding_checks():
    long = pd.DataFrame([("a", "R1", "E1", "definite"), ("a", "R1", "E2", "absent"), ("b", "R1", "E1", "absent"), ("b", "R1", "E2", "probable"),
                         ("c", "R1", "E1", "probable"), ("c", "R1", "E2", "absent")], columns=["case_id", "rater", "label", "state"])
    cf = pd.DataFrame({"case_id": ["a", "b", "c"], "rater": "R1", "primary_etiology": ["E1", "E1", "unknown"], "eeg_seen": ["0", "1", "2"]})
    c = ag.consistency(long, cf, k=1)
    assert c["primary_etiology_not_positive"] == 1 and c["unknown_primary_with_positive_label"] == 1
    b = ag.blinding_check(cf, k=1)
    assert b["minor"] == 1 and b["major"] == 1 and b["n_forms"] == 3


def test_gate0_reuses_the_plans_rule():
    k = {"E1": 0.8, "E2": 0.7, "E4a": 0.65, "E5": 0.62, "E6": 0.4}
    prev = {"E1": 0.3, "E2": 0.12, "E4a": 0.11, "E5": 0.05, "E6": 0.2}
    g = ag.gate0(k, prev)
    assert g["families_with_kappa_ge_0.6"] == 4 and g["families_with_prevalence_ge_10pct"] == 4 and g["pass"]
    assert not ag.gate0({**k, "E5": 0.5}, prev)["pass"]
    assert not ag.gate0(k, {**prev, "E2": 0.01, "E4a": 0.01})["pass"]


# ------------------------------------------------------------------------------------------------ the script
def write_study(root: Path, st, split=True):
    lo = root / "local_only"
    lo.mkdir(parents=True, exist_ok=True)
    smp = st["sample"].drop(columns=[c for c in st["sample"].columns if c.startswith("truth_")])
    smp.to_csv(lo / "sample.csv", index=False)
    st["id_map"].to_csv(lo / "id_map.csv", index=False)
    paths = []
    ex = st["export"]
    groups = [ex[ex["rater_id"] == r] for r in ("R1", "R2", "R3")] if split else [ex]
    for i, g in enumerate(groups):
        p = lo / f"ratings_{i}.csv"
        g.to_csv(p, index=False)
        paths.append(str(p))
    return paths, lo


def test_script_end_to_end_is_aggregate_only_and_writes_a_private_queue(tmp_path, study, capsys):
    paths, lo = write_study(tmp_path, study)
    rc = az.main(["--ratings", *paths, "--id-map", str(lo / "id_map.csv"), "--sample", str(lo / "sample.csv"), "--out", str(tmp_path / "agg"),
                  "--queue-out", str(lo / "queue.csv"), "--boot", "200"])
    out = capsys.readouterr().out
    assert rc == 0
    rep = json.loads((tmp_path / "agg" / "agreement_report.json").read_text())
    md = (tmp_path / "agg" / "agreement_report.md").read_text()
    assert set(rep["labels"]) == set(LABELS) and "Gate 0" in md and "Inter-rater" in md and "Amendment A1" in md
    assert rep["labels"]["E1"]["silver_vs_gold"]["sensitivity"]["weighted"] is not None
    assert "silver_vs_gold" not in rep["labels"]["E3"] and "silver_vs_gold" not in rep["labels"]["other"]
    # no case id, sample id or person id anywhere in the aggregates or stdout
    blob = out + md + json.dumps(rep)
    for cid in study["id_map"]["case_id"].head(50):
        assert cid not in blob
    assert not re.search(r"\bs\d{4}\b", blob)
    for pid in study["id_map"]["person_id"].head(50):
        assert not re.search(rf"(?<!\d){pid}(?!\d)", blob)
    q = lo / "queue.csv"
    assert q.exists() and stat.S_IMODE(q.stat().st_mode) == 0o600 and stat.S_IMODE(q.parent.stat().st_mode) in (0o700, 0o755, 0o775)
    assert set(pd.read_csv(q)["case_id"]) <= set(study["id_map"]["case_id"])


def test_script_works_with_one_combined_export_and_matches_split_exports(tmp_path, study):
    p1, lo1 = write_study(tmp_path / "a", study, split=True)
    p2, lo2 = write_study(tmp_path / "b", study, split=False)
    for p, lo, o in ((p1, lo1, "a"), (p2, lo2, "b")):
        assert az.main(["--ratings", *p, "--id-map", str(lo / "id_map.csv"), "--sample", str(lo / "sample.csv"), "--out", str(tmp_path / o / "agg"),
                        "--queue-out", str(lo / "q.csv"), "--boot", "100"]) == 0
    a = json.loads((tmp_path / "a" / "agg" / "agreement_report.json").read_text())
    b = json.loads((tmp_path / "b" / "agg" / "agreement_report.json").read_text())
    assert a["labels"]["E1"]["inter_rater"]["kappa"] == b["labels"]["E1"]["inter_rater"]["kappa"]


def test_script_refuses_non_local_only_inputs_and_a_lowered_threshold_in_an_agent_session(tmp_path, study, monkeypatch):
    paths, lo = write_study(tmp_path, study)
    plain = tmp_path / "plain.csv"
    pd.read_csv(paths[0]).to_csv(plain, index=False)
    with pytest.raises(SystemExit):
        az.main(["--ratings", str(plain), "--id-map", str(lo / "id_map.csv"), "--sample", str(lo / "sample.csv")])
    monkeypatch.setenv("CLAUDECODE", "1")
    with pytest.raises(SystemExit) as e:
        az.main(["--ratings", *paths, "--id-map", str(lo / "id_map.csv"), "--sample", str(lo / "sample.csv"), "--min-cell", "1",
                 "--out", str(tmp_path / "agg2"), "--queue-out", str(lo / "q.csv")])
    assert "agent session" in str(e.value)


def test_script_reports_missing_rater_instead_of_failing(tmp_path, study, capsys):
    paths, lo = write_study(tmp_path, study)
    rc = az.main(["--ratings", paths[0], "--id-map", str(lo / "id_map.csv"), "--sample", str(lo / "sample.csv"), "--out", str(tmp_path / "agg"),
                  "--queue-out", str(lo / "q.csv")])
    assert rc == 2 and "both main raters" in capsys.readouterr().out


def test_unmatched_case_ids_are_dropped_and_counted(study):
    ex = study["export"].copy()
    ex.loc[ex.index[:30], "case_id"] = "C0000Z"
    rep, _ = az.analyze(ex, study["id_map"], study["sample"], CFG, B=100)
    assert rep["unmatched_rows_dropped"] != 0
    assert rep["labels"]["E1"]["inter_rater"]["n_pairs"] != 240


def test_amended_block_is_arithmetic_only_and_handles_suppressed_numbers():
    per = {"E1": {"inter_rater": {"kappa": 0.7, "kappa_ci95": [0.6, 0.8]}, "gold": {"prevalence_weighted": 0.2, "prevalence_sample": 0.3}},
           "E2": {"inter_rater": {"kappa": 0.55, "kappa_ci95": [0.4, 0.7]}, "gold": {"prevalence_weighted": 0.05, "prevalence_sample": 0.2}},
           "E5": {"inter_rater": {"kappa": SUPPRESSED, "kappa_ci95": SUPPRESSED}, "gold": {"prevalence_weighted": SUPPRESSED, "prevalence_sample": SUPPRESSED}}}
    b = az.amended_primary_block(per, ["E1", "E2", "E5"])
    assert (b["n_kappa_meets"], b["n_prevalence_meets"], b["n_labels"]) == (1, 1, 3)
    assert b["labels"]["E5"]["kappa_meets_threshold"] is None and "pass" not in b and "OP-A1" in b["note"]
