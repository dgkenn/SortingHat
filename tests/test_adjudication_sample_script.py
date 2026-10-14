"""scripts/adjudication_sample.py end to end on the SYNTHETIC silver-feasibility inputs (no real data)."""
import json
import re
import stat

import pandas as pd
import pytest

from sortinghat.adjudication import sampling as sp
from sortinghat.metrics import intended_use as iu
from sortinghat.safe_output import SUPPRESSED
from test_silver_feasibility_inputs import load_script, make_inputs, rsf

smp = load_script("adjudication_sample")


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    d = tmp_path_factory.mktemp("adj_inputs")
    paths, ms, cohort = make_inputs(d, signal=1.0, seed=4, n_per_site=450)
    return paths, d


def argv(paths, out, *extra):
    return ["--cohort", str(paths["cohort"]), "--features", str(paths["features"]), "--silver", str(paths["silver"]),
            "--baselines", str(paths["baselines"]), "--omop-source", "none", "--sample-out", str(out / "local_only" / "adj" / "sample.csv"),
            "--summary-out", str(out / "agg" / "sample_summary.json"), *extra]


@pytest.fixture(scope="module")
def run1(inputs, tmp_path_factory):
    paths, d = inputs
    out = tmp_path_factory.mktemp("adj_run1")
    import contextlib, io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = smp.main(argv(paths, out))
    assert rc == 0
    return out, buf.getvalue()


def test_sample_file_is_private_and_holds_only_iu_large_patients(inputs, run1):
    paths, d = inputs
    out, _ = run1
    f = out / "local_only" / "adj" / "sample.csv"
    assert stat.S_IMODE(f.stat().st_mode) == 0o600 and stat.S_IMODE(f.parent.stat().st_mode) == 0o700
    s = pd.read_csv(f)
    # recompute the IU-large rule independently from the same assembly
    a = rsf.build_parser().parse_args(["--cohort", str(paths["cohort"]), "--features", str(paths["features"]), "--silver", str(paths["silver"]),
                                       "--baselines", str(paths["baselines"]), "--cohort-def", "broad", "--omop-source", "none"])
    A = rsf.assemble(a)
    mask = iu.intended_use_strata(A.hours, iu.sedation_features(A.baseline), A.covariates["duration_s"], iu.ceeg_task_flag(A.frame), None)["masks"][iu.IU_LARGE]
    eligible = set(A.frame.loc[mask, "person_id"].astype(int))
    assert set(s["person_id"].astype(int)) <= eligible and len(eligible) > 200
    assert (s["role"].value_counts().to_dict()) == {"sample": 200, "duplicate": 20, "calibration": 5}
    # weights re-weight to the eligible population, per site
    prim = s[s["role"] == "sample"]
    assert prim["weight"].sum() == pytest.approx(len(eligible))
    assert set(prim["site"]) == {"S0001", "S0002"}


def test_pattern_is_the_first_silver_positive_in_priority_order(run1):
    out, _ = run1
    s = pd.read_csv(out / "local_only" / "adj" / "sample.csv")
    for _, r in s.head(60).iterrows():
        expect = next((l for l in sp.DEFAULT_PRIORITY if r[f"silver_{l}"] == 1.0), sp.ALL_NEGATIVE)
        assert r["pattern"] == expect
    # the sample is enriched: all_negative is under-represented relative to its share among the eligible
    assert (s[s["role"] == "sample"]["pattern"] == sp.ALL_NEGATIVE).mean() < 0.3


def test_stdout_and_json_are_aggregate_only(run1):
    out, text = run1
    s = pd.read_csv(out / "local_only" / "adj" / "sample.csv")
    blob = text + (out / "agg" / "sample_summary.json").read_text()
    for pid in s["person_id"].head(100):
        assert not re.search(rf"(?<!\d){pid}(?!\d)", blob)
    assert not re.search(r"\bs\d{4}\b", blob) and "sub-" not in blob
    rep = json.loads((out / "agg" / "sample_summary.json").read_text())
    # every count in the design table is either >= 11 or suppressed
    for site, row in rep["by_site_and_silver_pattern"].items():
        for cell in row.values():
            for k in ("eligible", "sampled"):
                assert cell[k] == SUPPRESSED or cell[k] >= 11
            assert cell["weight"] == SUPPRESSED or cell["weight"] >= 1.0
        shown = [c["sampled"] for c in row.values()]
        assert sum(v == SUPPRESSED for v in shown) != 1                     # complementary suppression
    assert rep["checks"] == {"weights_sum_to_population": True, "min_weight_ge_1": True}
    sh = rep["iu_large_share_of_analysed"]["all_sites"]                     # the IU-large share of the analysed broad cohort (SAP A1 item 3)
    assert sh["den"] == rep["analysed_broad_cohort"]["pooled"] and 0.0 < sh["share"] < 1.0 and sh["n"] == rep["eligible_iu_large"]["pooled"]
    assert rep["stratum"] == "IU_large" and rep["cohort_def"] == "broad"
    assert "ventilation_vasopressor" in rep["iu_large_rule"]["missing_conditions"]       # no OMOP cache in this run: said so
    assert rep["weights"]["effective_sample_size"] != SUPPRESSED and "q50" in rep["weights"]
    assert "min" not in json.dumps(rep["weights"]) and "max" not in json.dumps(rep["weights"])
    assert "IU-large" in text


def test_the_draw_is_deterministic_and_seed_sensitive(inputs, run1, tmp_path):
    paths, d = inputs
    out1, _ = run1
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        assert smp.main(argv(paths, tmp_path / "again")) == 0
        assert smp.main(argv(paths, tmp_path / "other", "--sample-seed", "99", "--sample-n", "150")) == 0
    a = (out1 / "local_only" / "adj" / "sample.csv").read_bytes()
    b = (tmp_path / "again" / "local_only" / "adj" / "sample.csv").read_bytes()
    assert a == b
    c = pd.read_csv(tmp_path / "other" / "local_only" / "adj" / "sample.csv")
    assert (c["role"] == "sample").sum() == 150
    assert set(c.loc[c["role"] == "sample", "person_id"]) != set(pd.read_csv(out1 / "local_only" / "adj" / "sample.csv").query("role == 'sample'")["person_id"])


def test_paths_are_policed(inputs, tmp_path):
    paths, d = inputs
    with pytest.raises(SystemExit):
        smp.main(["--cohort", str(paths["cohort"]), "--features", str(paths["features"]), "--silver", str(paths["silver"]),
                  "--baselines", str(paths["baselines"]), "--omop-source", "none", "--sample-out", str(tmp_path / "sample.csv")])
    with pytest.raises(SystemExit):
        smp.main(argv(paths, tmp_path, "--summary-out", str(tmp_path / "local_only" / "summary.json")))
