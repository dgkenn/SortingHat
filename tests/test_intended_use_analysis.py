"""D-145 in scripts/run_silver_feasibility.py on SYNTHETIC inputs: the "Intended use: undifferentiated AMS" block comes first, has
the three comparisons under both schemes for the whole set and the undifferentiated subgroup, the subgroup flag follows its
definition, and nothing record-level is emitted."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sortinghat.models import DEFAULT_RUNGS, LadderConfig, run_ladder
from sortinghat.safe_output import assert_aggregate_only
from test_silver_feasibility import run_main  # noqa: F401  (tests/ on sys.path)
from test_silver_feasibility_inputs import argv_for, make_inputs, rsf

HEAD = "combined"
KEYS = ("prior", "agesex", "P")


@pytest.fixture(scope="module")
def sig(tmp_path_factory):
    d = tmp_path_factory.mktemp("iu_sig")
    paths, ms, cohort = make_inputs(d, signal=1.0, seed=5)
    J, md, stdout = run_main(paths, d / "out")
    return SimpleNamespace(d=d, paths=paths, ms=ms, cohort=cohort, J=J, md=md, stdout=stdout)


@pytest.fixture(scope="module")
def null(tmp_path_factory):
    d = tmp_path_factory.mktemp("iu_null")
    paths, ms, cohort = make_inputs(d, signal=0.0, seed=6)
    J, md, stdout = run_main(paths, d / "out")
    return SimpleNamespace(d=d, paths=paths, ms=ms, cohort=cohort, J=J, md=md, stdout=stdout)


def test_intended_use_is_the_first_reported_block_everywhere(sig):
    md, lines = sig.md, sig.stdout.splitlines()
    assert md.index("## Intended use: undifferentiated AMS") < md.index("## Data flow") < md.index("## Headline Delta")
    assert md.index("## Intended use: undifferentiated AMS") < md.index("Baseline-only AUROC") if "Baseline-only AUROC" in md else True
    assert lines[1].startswith("Intended use: undifferentiated AMS (baselines: current encounter only)")
    assert lines.index(next(x for x in lines if x.startswith("analysed="))) > 1               # the existing summary follows it
    assert list(sig.J)[:2] == ["banner", "intended_use"]
    assert md.splitlines()[0].startswith("# EXPLORATORY")                                       # the banner is still the very first line


def test_three_comparisons_two_schemes_two_populations(sig):
    iu = sig.J["intended_use"]
    assert set(iu["comparisons"]) == set(KEYS) and not iu["comparisons_unavailable"]
    for split in ("loso", "temporal"):
        for pop in ("all_analysed", "undifferentiated"):
            blk = iu["populations"][split][pop]
            if "not_estimable" in blk:
                assert pop == "undifferentiated"
                continue
            assert set(blk) == set(KEYS)
            for k in KEYS:
                r = blk[k]
                assert {"delta", "ci_primary_within_site", "per_site", "all_sites_favorable", "n_eval"} <= set(r)
                assert r["ci_primary_within_site"]["lo"] <= r["delta"] <= r["ci_primary_within_site"]["hi"]
                assert set(r["per_site"]) == {"site_1", "site_2"}
    assert iu["eeg_derived_inputs"].startswith("No baseline column and no label is derived from EEG")
    assert "never an input or label evidence" in sig.md


def test_planted_signal_is_recovered_in_every_comparison_of_the_whole_set(sig):
    for split in ("loso", "temporal"):
        for k in KEYS:
            r = sig.J["intended_use"]["populations"][split]["all_analysed"][k]
            assert r["delta"] < -0.02 and r["ci_primary_within_site"]["hi"] < 0 and r["all_sites_favorable"], (split, k)
    # LOSO subgroup: sized well above the estimability floor in this fixture, and the signal is still there
    sub = sig.J["intended_use"]["populations"]["loso"]["undifferentiated"]
    assert set(sub) == set(KEYS) and all(sub[k]["delta"] < 0 for k in KEYS)


def test_null_run_has_no_intended_use_gain(null):
    for split in ("loso", "temporal"):
        for k in KEYS:
            r = null.J["intended_use"]["populations"][split]["all_analysed"][k]
            assert abs(r["delta"]) < 0.03 and not r["h1h2_rule_met"], (split, k)
    nc = null.J["intended_use"]["negative_controls"]
    assert set(nc) == {"loso", "temporal"} and all(set(v) == set(KEYS) for v in nc.values())


def test_the_prior_comparison_uses_the_prevalence_prior_as_reference(sig):
    a = rsf.build_parser().parse_args(argv_for(sig.paths, sig.d / "x"))
    A = rsf.assemble(a)
    md = rsf.build_model_data(A, "loso", 0.2, 0.2, 0)
    spec = [r for r in DEFAULT_RUNGS if r.name == HEAD]
    res = run_ladder(md, {"prior": []}, spec, LadderConfig(n_boot=50, include_e7=A.include_e7, per_label=False), "loso")
    ref = res.reference["prior"]
    assert ref["loss_baseline"] == pytest.approx(ref["loss_prior"], abs=1e-12)               # the reference IS the prevalence prior
    assert res.get(HEAD, "prior").delta < 0
    assert A.iu_sets["prior"] == [] and set(A.iu_sets) == set(KEYS)


def test_baseline_sets_hold_no_eeg_derived_or_meta_column(sig):
    a = rsf.build_parser().parse_args(argv_for(sig.paths, sig.d / "x"))
    A = rsf.assemble(a)
    meta = {"meta__hours_since_encounter_start", "meta__label_dx_before_t0"}
    for name, cols in {**A.iu_sets, **A.baseline_sets}.items():
        assert not [c for c in cols if c.startswith(("qeeg.", "conn.", "morgoth.", "emb.", "dyn."))], name
        assert not meta & set(cols), name
    assert not meta & set(A.baseline.columns)
    assert set(A.iu_sets["P"]) >= set(A.iu_sets["agesex"]) and A.iu_sets["agesex"] == ["demo__age_years", "demo__sex_male"]


# ------------------------------------------------------------------------------------------------ subgroup flag
def test_undifferentiated_flag_follows_its_definition():
    t0 = pd.Timestamp("2024-03-01 12:00")
    sel = pd.DataFrame({"t0": [t0] * 7, "encounter_start": [t0 - pd.Timedelta(hours=h) for h in (2, 36, 36.01, 100, 10, 10, 10)]})
    bl = pd.DataFrame({"meta__label_dx_before_t0": [0, 0, 0, 0, 1, 0, np.nan],
                       "sed__sedative__on_t0": [0, 0, 0, 0, 0, 1, 0.0], "sed__opioid__on_t0": 0.0,
                       "sed__midazolam__qty_6h": [0, 0, 0, 0, 0, 0, 0.0]})
    flag, parts = rsf.undifferentiated_flag(sel, bl)
    assert flag.tolist() == [True, True, False, False, False, False, False]              # 36 h inclusive; dx, sedation, unknown dx fail
    assert parts["early"].tolist() == [True, True, False, False, True, True, True]
    # a 6 h quantity alone is an exposure, and so is nothing in the 24 h columns (the window is 6 h)
    bl2 = bl.assign(**{"sed__sedative__on_t0": 0.0, "sed__midazolam__qty_6h": [0, 0, 0, 0, 0, 3.0, 0], "sed__midazolam__qty_24h": 9.0})
    assert rsf.undifferentiated_flag(sel, bl2)[0].tolist() == [True, True, False, False, False, False, False]
    # components that cannot be established never grow the subgroup
    assert not rsf.undifferentiated_flag(sel, bl.drop(columns=["meta__label_dx_before_t0"]))[0].any()
    assert not rsf.undifferentiated_flag(sel, bl[["meta__label_dx_before_t0"]])[0].any()
    assert not rsf.undifferentiated_flag(sel.drop(columns=["encounter_start"]), bl)[0].any()
    # an older cohort table: the hours come from the baselines' meta column instead
    flag, parts = rsf.undifferentiated_flag(sel.drop(columns=["encounter_start"]),
                                            bl.assign(meta__hours_since_encounter_start=[2, 36, 40, 100, 10, 10, 10.0]))
    assert flag.tolist() == [True, True, False, False, False, False, False] and "baselines meta" in parts["sources"]["visit_start"]


def test_subgroup_size_is_reported_suppressed_and_matches_the_columns(sig):
    a = rsf.build_parser().parse_args(argv_for(sig.paths, sig.d / "x"))
    A = rsf.assemble(a)
    n = int(A.subgroup.sum())
    info = sig.J["intended_use"]["subgroup"]
    assert info["n"] == n and n >= 11 and sum(info["per_site"].values()) == n
    bl = A.frame
    hours = (pd.to_datetime(bl["t0"]) - pd.to_datetime(bl["encounter_start"])).dt.total_seconds() / 3600.0
    raw = pd.read_parquet(sig.paths["baselines"]).set_index("person_id").reindex(bl["person_id"])
    want = (hours.between(0, 36).to_numpy() & (raw["meta__label_dx_before_t0"] == 0).to_numpy()
            & (raw["sed__sedative__on_t0"] == 0).to_numpy() & (raw["sed__opioid__on_t0"] == 0).to_numpy())
    assert np.array_equal(A.subgroup, want)
    assert 0 < n < len(bl)
    assert info["definition"]["max_hours_from_visit_start"] == 36.0 and info["definition"]["sedative_opioid_window_hours"] == 6.0


def test_small_subgroup_is_suppressed_and_not_estimable(tmp_path):
    paths, ms, cohort = make_inputs(tmp_path, signal=1.0, seed=7, n_per_site=300)
    bl = pd.read_parquet(paths["baselines"])
    bl["meta__label_dx_before_t0"] = 1.0                       # nobody is undifferentiated ...
    bl.loc[:5, "meta__label_dx_before_t0"] = 0.0               # ... except a handful
    bl.to_parquet(paths["baselines"], index=False)
    J, md, stdout = run_main(paths, tmp_path / "out")
    iu = J["intended_use"]
    assert iu["subgroup"]["n"] == "<11" and set(iu["subgroup"]["per_site"].values()) <= {"<11"}
    for split in ("loso", "temporal"):
        assert "not_estimable" in iu["populations"][split]["undifferentiated"]
        assert set(iu["populations"][split]["all_analysed"]) == set(KEYS)
    assert "not estimable" in md and "<11" in md


def test_eval_mask_only_changes_who_is_scored(sig):
    a = rsf.build_parser().parse_args(argv_for(sig.paths, sig.d / "x"))
    A = rsf.assemble(a)
    for scheme in ("loso", "temporal"):
        full = rsf.build_model_data(A, scheme, 0.2, 0.2, 0)
        sub = rsf.build_model_data(A, scheme, 0.2, 0.2, 0, eval_mask=A.subgroup)
        assert np.array_equal(full.gold_role == "dev", sub.gold_role == "dev")                  # same development rows
        assert (sub.gold_role[sub.gold_role == "eval"] == "eval").all()
        assert A.subgroup[sub.gold_role == "eval"].all()                                         # only subgroup rows are scored
        assert ((sub.gold_role == "none") == ((full.gold_role == "eval") & ~A.subgroup)).all()
        assert not sub.m_gold[sub.gold_role == "none"].any() and np.array_equal(sub.m_silver, full.m_silver)   # none still train (silver)


# ---------------------------------------------------------------------------------------------- scope labels, safety
def test_with_history_baselines_are_labelled_in_every_surface(tmp_path):
    paths, ms, cohort = make_inputs(tmp_path, signal=1.0, seed=8, n_per_site=300, scope_label="with history")
    J, md, stdout = run_main(paths, tmp_path / "out", "--skip-controls")
    assert J["intended_use"]["baseline_scope"] == "with history" and J["settings"]["baseline_scope"] == "with history"
    assert "**with history**" in md and "(baselines: with history)" in stdout
    assert "skipped" in J["intended_use"]["negative_controls"]["skipped"].lower() or "NOT run" in J["intended_use"]["negative_controls"]["skipped"]


def test_baselines_built_before_d145_still_run_and_say_so(tmp_path):
    paths, ms, cohort = make_inputs(tmp_path, signal=1.0, seed=9, n_per_site=300, with_p=False, scope_label=None)
    cohort_df = pd.read_csv(paths["cohort"])
    cohort_df.drop(columns=["encounter_start"]).to_csv(paths["cohort"], index=False)
    J, md, stdout = run_main(paths, tmp_path / "out", "--skip-controls")
    iu = J["intended_use"]
    assert iu["baseline_scope"] == rsf.UNKNOWN_SCOPE and "prior-encounter data may be included" in md
    assert set(iu["comparisons"]) == {"prior", "agesex"} and any("Baseline P" in k or "P (Presentation)" in k for k in iu["comparisons_unavailable"])
    assert all("not_estimable" in iu["populations"][s]["undifferentiated"] for s in ("loso", "temporal"))
    assert set(J["ladder"]["loso"]["rungs"]) == {"A", "C"}                                      # the A / C comparisons are unchanged


def test_intended_use_outputs_are_aggregate_only(sig):
    ids = {str(p) for p in sig.cohort["person_id"]} | set(rsf.recording_ids(sig.cohort))
    text = json.dumps(sig.J["intended_use"]) + sig.md[:sig.md.index("## Data flow")]
    assert_aggregate_only(text, ids)
    assert_aggregate_only(sig.J["intended_use"], ids)
    assert "S0001" not in text and "S0002" not in text
