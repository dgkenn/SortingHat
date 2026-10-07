import json

import numpy as np
import pytest

from sortinghat.models import LadderConfig, commercial_clean_gap, run_ladder, write_results
from sortinghat.models.synthetic import make_synthetic_study

CFG = LadderConfig(n_boot=300, seed=1)


@pytest.fixture(scope="module")
def planted():
    d = make_synthetic_study(n_per_site=500, signal=1.0, seed=11)
    return d, run_ladder(d, cfg=CFG)


@pytest.fixture(scope="module")
def null():
    d = make_synthetic_study(n_per_site=500, signal=0.0, seed=12)
    return d, run_ladder(d, cfg=CFG)


def test_planted_signal_is_recovered_at_every_site(planted):
    _d, res = planted
    for rung in ("qeeg", "connectivity", "morgoth", "embeddings", "combined"):
        r = res.get(rung)
        assert r.available and r.delta < -0.01, rung
        assert r.all_sites_favorable, rung
        assert r.ci_primary["hi"] < 0 and r.rule_met, rung
        assert r.ci_primary["alpha"] == 0.01
        assert set(r.ci_modes) >= {"within_site", "cluster", "two_stage", "site_t"}
    assert res.get("combined").delta <= res.get("qeeg").delta + 0.02       # top rung is not worse than a single rung
    assert len(res.get("combined").per_site) == 3
    assert all(v["delta"] < 0 and v["n"] > 100 for v in res.get("combined").per_site.values())
    # planted labels: per-label gain appears on signal labels, not on the label with no EEG signature (E7)
    pl = res.get("combined").per_label
    assert pl["E1"]["improved"] and pl["E5"]["improved"] and not pl["E7"]["improved"]


def test_no_signal_gives_delta_near_zero(null):
    _d, res = null
    for rung in ("qeeg", "connectivity", "morgoth", "embeddings", "combined"):
        r = res.get(rung)
        assert abs(r.delta) < 0.02, (rung, r.delta)
        assert not r.rule_met
        assert r.ci_modes["within_site"]["hi"] > -0.002


def test_prior_and_unavailable_rungs(planted):
    _d, res = planted
    assert res.get("prior").delta == 0.0 and not res.get("prior").has_eeg
    assert not res.get("dynamics").available          # no dyn.* columns supplied
    ref = res.reference["all"]
    assert ref["loss_baseline"] < ref["loss_prior"]    # baseline carries information over prevalence


def test_commercial_clean_gap(planted):
    _d, res = planted
    g = commercial_clean_gap(res, n_boot=200)
    assert g["available"] and g["lo"] <= g["gap"] <= g["hi"]
    assert abs(g["gap"] - (g["delta_cbramod"] - g["delta_morgoth"])) < 1e-9


def test_multiple_baselines_share_identical_test_patients():
    d = make_synthetic_study(n_per_site=300, signal=1.0, seed=3)
    base = {"A": ["age", "gcs"], "C": list(d.baseline.columns)}
    from sortinghat.models import DEFAULT_RUNGS
    res = run_ladder(d, base, [s for s in DEFAULT_RUNGS if s.name == "qeeg"], LadderConfig(n_boot=100))
    a, c = res.get("qeeg", "A"), res.get("qeeg", "C")
    assert len(a.d) == len(c.d) and a.delta < 0 and c.delta < 0
    assert res.reference["A"]["loss_baseline"] != res.reference["C"]["loss_baseline"]


def test_temporal_holdout_runs_and_recovers_signal():
    d = make_synthetic_study(n_per_site=600, signal=1.0, seed=5)
    from sortinghat.models import DEFAULT_RUNGS
    res = run_ladder(d, None, [s for s in DEFAULT_RUNGS if s.name in ("qeeg", "combined")], LadderConfig(n_boot=200),
                     split="temporal")
    r = res.get("combined")
    assert r.delta < 0 and r.ci_primary["hi"] < 0


def test_gold_eval_labels_only_define_the_endpoint():
    """Silver labels never score: replacing silver labels by garbage on the eval rows leaves Delta unchanged."""
    d = make_synthetic_study(n_per_site=300, signal=1.0, seed=9)
    from sortinghat.models import DEFAULT_RUNGS
    rungs = [s for s in DEFAULT_RUNGS if s.name == "qeeg"]
    r0 = run_ladder(d, None, rungs, LadderConfig(n_boot=50, per_label=False))
    # n_eval equals the number of gold-eval patients, not the number of rows
    assert r0.n_eval == int((d.gold_role == "eval").sum())
    assert (d.gold_role[r0.eval_idx] == "eval").all()


def test_results_are_aggregate_only_and_suppressed(planted, tmp_path):
    d, res = planted
    path = write_results(tmp_path / "ladder.json", res, controls={"note": "x"})
    txt = path.read_text()
    obj = json.loads(txt)
    assert set(obj["ladder"]["rungs"]["all"]) >= {"prior", "qeeg", "combined", "dynamics"}
    assert obj["ladder"]["rungs"]["all"]["dynamics"] == {"available": False}
    sites = obj["ladder"]["rungs"]["all"]["combined"]["per_site"]
    assert set(sites) == {"site_1", "site_2", "site_3"}
    for real in set(d.sites):
        assert real not in txt                          # pseudonymous sites only
    # a tiny site is suppressed
    from sortinghat.models.report import rung_to_aggregate
    r = res.get("combined")
    r2 = type(r)(**{**r.__dict__, "per_site": {"S": {"delta": -0.5, "n": 5}}})
    agg = rung_to_aggregate(r2, {"S": "site_1"})
    assert agg["per_site"]["site_1"] == {"n": "<11", "delta": "<11"}
    with pytest.raises(ValueError):
        from sortinghat.models.report import clean
        clean({"d": r.d})                               # per-patient vectors refused
