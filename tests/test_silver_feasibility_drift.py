"""Label drift screen of scripts/run_silver_feasibility.py (temporal scheme, test vs train prevalence). SYNTHETIC only."""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from sortinghat.safe_output import SUPPRESSED, assert_aggregate_only
from test_silver_feasibility_inputs import rsf  # noqa: F401  (tests/ is on sys.path)


def assembly(n_per_site=600, prevalence=None, assessable=None, sites=("S1", "S2"), seed=0):
    """Rows ordered by t0 within site. ``prevalence[label](frac_of_record_period) -> P(positive)``;
    ``assessable[label](frac) -> P(assessable)`` (default 1)."""
    rng = np.random.default_rng(seed)
    S, T, frac = [], [], []
    for s in sites:
        S += [s] * n_per_site
        T += list(pd.Timestamp("2022-01-01") + pd.to_timedelta(np.arange(n_per_site), unit="D"))
        frac += list(np.arange(n_per_site) / n_per_site)
    frac = np.array(frac)
    names = tuple(prevalence)
    y = np.column_stack([rng.random(len(frac)) < prevalence[l](frac) for l in names]).astype(float)
    m = np.column_stack([rng.random(len(frac)) < (assessable or {}).get(l, lambda f: np.ones_like(f))(frac) for l in names])
    return SimpleNamespace(sites=np.array(S, dtype=object), times=np.array(T, dtype="datetime64[us]"), label_names=names,
                           y=np.where(m, y, 0.0), m=m, site_labels={s: f"site_{i + 1}" for i, s in enumerate(sites)})


PREV = {"E5": lambda f: np.full_like(f, 0.20),                             # stable
        "E6": lambda f: np.where(f < 0.8, 0.03, 0.17),                     # planted drift: ~5.7x, as in the real E6
        "E1": lambda f: np.where(f < 0.8, 0.30, 0.05)}                     # planted drift downwards (ratio ~0.17)


def test_planted_drift_is_flagged_and_a_stable_label_is_not():
    d = rsf.label_drift_screen(assembly(prevalence=PREV, n_per_site=1500), 0.2)
    assert d["labels"]["E5"]["flagged_in"] == [] and d["labels"]["E5"]["pooled"]["flag"] is False
    r = d["labels"]["E6"]["pooled"]["ratio_test_over_train"]
    assert d["labels"]["E6"]["pooled"]["flag"] is True and r > 3
    assert "pooled" in d["labels"]["E6"]["flagged_in"] and {"site_1", "site_2"} <= set(d["labels"]["E6"]["flagged_in"])
    assert d["labels"]["E1"]["pooled"]["flag"] is True and d["labels"]["E1"]["pooled"]["ratio_test_over_train"] < 1 / 3
    assert d["labels_flagged"] == ["E1", "E6"] and d["flagged_total"] == len(d["flagged"])


def test_flag_threshold_is_configurable():
    a = assembly(prevalence=PREV)
    assert rsf.label_drift_screen(a, 0.2, ratio=10.0)["labels"]["E6"]["pooled"]["flag"] is False
    assert rsf.label_drift_screen(a, 0.2, ratio=1.2)["labels"]["E5"]["pooled"]["flag"] in (True, False)       # runs; no error


def test_not_assessable_early_rows_remove_the_apparent_drift():
    """The E6 fix: positives only occur once the culture feed exists (f >= 0.7); before that the legacy label is a confident
    negative. With the early rows not assessable (NaN) the train prevalence is taken over assessable rows only."""
    pos = {"E6": lambda f: np.where(f < 0.7, 0.0, 0.10)}
    d_legacy = rsf.label_drift_screen(assembly(prevalence=pos, n_per_site=1500), 0.2)["labels"]["E6"]["pooled"]
    assert d_legacy["flag"] is True and d_legacy["ratio_test_over_train"] > 3
    fixed = assembly(prevalence=pos, n_per_site=1500, assessable={"E6": lambda f: (f >= 0.7).astype(float)})
    d_fixed = rsf.label_drift_screen(fixed, 0.2)["labels"]["E6"]["pooled"]
    assert d_fixed["flag"] is False and 0.5 < d_fixed["ratio_test_over_train"] < 2


def test_small_cells_are_suppressed_and_never_reported_as_stable():
    a = assembly(n_per_site=200, prevalence={"E6": lambda f: np.where(f < 0.8, 0.0, 0.05)})     # ~2 test positives per site
    d = rsf.label_drift_screen(a, 0.2)
    blk = d["labels"]["E6"]
    for c in (blk["pooled"], *blk["per_site"].values()):
        assert c["flag"] is None and str(c["ratio_test_over_train"]).startswith("not estimable")
        assert c["train"]["prevalence"] == SUPPRESSED
    assert d["flagged"] == []


def test_pooled_count_is_hidden_when_exactly_one_site_count_is_hidden():
    a = assembly(n_per_site=600, prevalence={"E6": lambda f: np.full_like(f, 0.25)})
    s2 = a.sites == "S2"
    a.m[:, 0] = np.where(s2 & (np.arange(len(s2)) % 600 > 100), False, a.m[:, 0])             # site 2: only ~100 assessable rows
    a.y[:, 0] = np.where(a.m[:, 0], a.y[:, 0], 0.0)
    te = np.zeros(len(s2), bool)
    # site 2's test rows are almost all not assessable -> its test count is < 11 while site 1's is not
    d = rsf.label_drift_screen(a, 0.2)["labels"]["E6"]
    assert d["per_site"]["site_2"]["test"]["n_assessable"] == SUPPRESSED
    assert d["pooled"]["test"]["n_assessable"] == SUPPRESSED                                   # would reveal site 2 by subtraction
    assert d["per_site"]["site_1"]["test"]["n_assessable"] != SUPPRESSED and not te.any()
    assert_aggregate_only(d)


def test_markdown_section_lists_flags_and_is_aggregate_only():
    d = rsf.label_drift_screen(assembly(prevalence=PREV), 0.2)
    text = "\n".join(rsf._drift_markdown(d))
    assert "## Label drift screen" in text and "FLAG" in text and "Flagged: " in text and "E6 (pooled" in text
    ok = rsf._drift_markdown(rsf.label_drift_screen(assembly(prevalence={"E5": PREV["E5"]}), 0.2))
    assert any("No estimable label is flagged" in l for l in ok)
