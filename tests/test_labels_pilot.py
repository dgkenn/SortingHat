import numpy as np
import pandas as pd
import pytest

from sortinghat.labels import pilot
from sortinghat.safe_output import assert_aggregate_only


def cohort():
    rows = []
    for site, n in (("S0001", 600), ("S0002", 300), ("S0003", 90), ("S0004", 10)):
        rows += [(f"{site}-{i:05d}", site) for i in range(n)]
    return pd.DataFrame(rows, columns=["case_id", "site"])


def test_allocation_sums_and_respects_sizes():
    a = pilot.allocate_by_site({"A": 600, "B": 300, "C": 90, "D": 10}, 200)
    assert sum(a.values()) == 200 and all(v >= 1 for v in a.values())
    assert a["A"] > a["B"] > a["C"] > a["D"] - 1
    assert pilot.allocate_by_site({"A": 3, "B": 2}, 200) == {"A": 3, "B": 2}


def test_draw_deterministic_stratified_and_aggregate_alloc(tmp_path):
    c = cohort()
    ids1, alloc = pilot.draw_pilot_sample(c, seed=1)
    ids2, _ = pilot.draw_pilot_sample(c, seed=1)
    ids3, _ = pilot.draw_pilot_sample(c, seed=2)
    assert ids1 == ids2 and ids1 != ids3 and len(ids1) == 200 == len(set(ids1))
    assert alloc["S0004"] == "<11"                       # small cells suppressed
    assert_aggregate_only(alloc)
    p = pilot.save_pilot_ids(ids1, tmp_path / "local_only" / "pilot_ids.txt")
    assert oct(p.stat().st_mode & 0o777) == "0o600"


def ratings(n=60, seed=0, agree=0.9):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        for lab in ("E1", "E5"):
            a = int(rng.integers(0, 4))
            b = a if rng.random() < agree else int(rng.integers(0, 4))
            for rev, st in (("R1", a), ("R2", b)):
                rows.append((f"c{i}", lab, rev, ["absent", "possible", "probable", "definite"][st]))
    return pd.DataFrame(rows, columns=["case_id", "label", "reviewer", "state"])


def test_label_kappas_high_agreement_and_unassessable_excluded():
    r = ratings()
    k = pilot.label_kappas(r, ["E1", "E5"])
    assert k["E1"]["kappa_weighted_linear"] > 0.7 and k["E1"]["kappa_binary"] > 0.6
    r2 = r.copy()
    r2.loc[(r2.case_id == "c0") & (r2.reviewer == "R1"), "state"] = "unassessable"
    k2 = pilot.label_kappas(r2, ["E1", "E5"])
    assert k2["E1"]["n_unassessable"] == "<11"


def test_label_kappas_small_n_suppressed():
    k = pilot.label_kappas(ratings(n=5), ["E1"])
    assert k["E1"]["kappa_binary"] == "<11"


def test_minutes_summary_suppresses_and_has_no_extremes():
    m = pd.DataFrame({"case_id": [f"c{i}" for i in range(40)], "reviewer": ["R1"] * 20 + ["R2"] * 20,
                      "minutes": np.linspace(5, 25, 40)})
    s = pilot.minutes_summary(m)
    assert s["overall"]["q50"] == pytest.approx(15, abs=0.5)
    assert not any(k in s["overall"] for k in ("min", "max", "q00", "q100"))
    s2 = pilot.minutes_summary(m.iloc[:8])
    assert s2["overall"]["q50"] == "<11"
    assert_aggregate_only(s)


def test_eeg_only_share():
    rows = []
    # 30 E5 cases: 12 only-EEG evidence (EEG report), 6 only G93.41, 12 with an objective anchor too
    for i in range(12):
        rows.append((f"a{i}", "E5", "eeg_report", "diffuse slowing", "", None, False))
    for i in range(6):
        rows.append((f"b{i}", "E5", "icd", "", "G93.41", None, False))
    for i in range(12):
        rows.append((f"c{i}", "E5", "icd", "", "G93.41", None, False))
        rows.append((f"c{i}", "E5", "objective", "", "", None, False))
    ev = pd.DataFrame(rows, columns=["case_id", "label", "source", "text", "code", "hours_from_t0", "eeg_filtered"])
    r = pilot.eeg_only_share(ev, ["E5", "E1"])
    assert r["E5"]["n_naive_positive"] == 30
    assert r["E5"]["eeg_only_share"] == round(12 / 30, 4)
    assert r["E5"]["banned_only_share"] == round(18 / 30, 4)
    assert r["E1"]["n_naive_positive"] == 0
    assert_aggregate_only(r)


def test_gate0_and_report():
    good = {l: {"kappa_weighted_linear": 0.7, "prevalence": 0.2} for l in ("E1", "E2", "E4a", "E5", "E6")}
    assert pilot.gate0_pilot_check(good)["pass"]
    bad = dict(good); bad["E1"] = {"kappa_weighted_linear": 0.4, "prevalence": "<11"}
    bad["E2"] = {"kappa_weighted_linear": 0.5, "prevalence": 0.05}
    g = pilot.gate0_pilot_check(bad)
    assert not g["kappa_ok"] and not g["pass"]
    rep = pilot.pilot_report(ratings(), pd.DataFrame({"case_id": ["x"] * 30, "reviewer": ["R1"] * 30,
                                                      "minutes": np.arange(30) + 5.0}),
                             pd.DataFrame(columns=["case_id", "label", "source"]), labels=["E1", "E5"])
    assert set(rep) == {"kappas", "minutes_per_case", "eeg_only_evidence", "gate0"}
