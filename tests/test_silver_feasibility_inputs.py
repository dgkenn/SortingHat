"""Synthetic inputs for the silver-feasibility scripts (cohort table, EEG feature parts, silver labels, baselines) and
tests of the input readers. SYNTHETIC ONLY: every identifier below is fake. The planted signal lives in the EEG columns
(``sortinghat.models.synthetic.make_synthetic_study``); ``signal=0`` gives pure-noise EEG."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sortinghat.models.synthetic import make_synthetic_study

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.path.insert(0, str(SCRIPTS))
    sys.modules[name] = mod                       # dataclasses resolve string annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


rsf = load_script("run_silver_feasibility")

ID_BASE = 71_000_000
T_BASE = pd.Timestamp("2024-03-01 08:00:00")


def make_inputs(root: Path, signal: float = 1.0, n_per_site: int = 450, seed: int = 0, n_noise: int = 6,
                sites=("S0001", "S0002"), unassessable: float = 0.04, site_shift: float = 0.0, with_p: bool = True,
                scope_label: str | None = "current encounter only"):
    """Write cohort/features/silver/baselines under ``root/local_only`` and return (paths dict, ModelData truth)."""
    ms = make_synthetic_study(n_per_site=n_per_site, sites=sites, signal=signal, seed=seed, n_noise=n_noise,
                              embeddings=False, morgoth=False, unassessable=unassessable,
                              site_shift=site_shift)
    n = ms.n
    lo = Path(root) / "local_only"
    (lo / "features").mkdir(parents=True, exist_ok=True)
    (lo / "silver").mkdir(parents=True, exist_ok=True)
    pid = ID_BASE + np.arange(n)
    rng = np.random.default_rng(seed + 99)
    sev = ms.covariates["severity"]
    gcs = np.clip(np.round(7.0 - 2.0 * (sev - sev.mean()) / sev.std()), 3, 15)
    cohort = pd.DataFrame({
        "SiteID": ms.sites.astype(str), "person_id": pid, "person_id_source": pid,
        "SessionID": [str(1 + (i % 3)) for i in range(n)],
        "BidsFolder": [f"sub-{s}{p}" for s, p in zip(ms.sites, pid)],
        "EEGFolder": np.where(rng.random(n) < 0.5, "cEEG", None),
        "t0": [T_BASE + pd.Timedelta(hours=float(t)) for t in ms.times],
        "age_years": 60 + 5 * ms.baseline["age"].to_numpy(), "duration_s": ms.covariates["duration_s"],
        "gcs_nearest_window": gcs, "four_nearest_window": np.nan,
        "in_strict": True, "in_strict_pm6": True, "in_broad": True})
    hours_in = rng.uniform(1.0, 80.0, n)                                        # hours from the encounter start to t0 (D-145)
    cohort["encounter_start"] = [t - pd.Timedelta(hours=float(h)) for t, h in zip(cohort["t0"], hours_in)]
    cohort.loc[: n // 20, ["in_strict", "in_strict_pm6"]] = False             # rows outside the strict cohort
    cohort.to_csv(lo / "cohort_study1.csv", index=False)

    # baselines: A = demographics + score + sedation flags; C adds a vital and a lab (names follow build_baselines)
    sed = ms.covariates["sedated"].astype(float)
    bl = pd.DataFrame({"person_id": pid, "demo__age_years": ms.baseline["age"].to_numpy(),
                       "score__gcs__value": ms.baseline["gcs"].to_numpy(),
                       "sed__sedative__on_t0": sed, "sed__opioid__on_t0": 0.0,
                       "vital__map__value": ms.baseline["map"].to_numpy(),
                       "lab__lactate__value": ms.baseline["lactate"].to_numpy()})
    cols_a = ["demo__age_years", "score__gcs__value", "sed__sedative__on_t0", "sed__opioid__on_t0"]
    side = {"A": cols_a, "B": cols_a, "C": [c for c in bl.columns if c != "person_id"]}
    extra: dict = {}
    if with_p:                                                      # Baseline P columns + the subgroup meta columns (D-145)
        bl["demo__sex_male"] = (rng.random(n) < 0.5).astype(float)
        bl["pres_score__gcs__value"] = np.clip(gcs + rng.normal(0, 1, n), 3, 15)
        bl["pres_first__map__value"] = ms.baseline["map"].to_numpy() + rng.normal(0, 0.3, n)
        bl["sed__midazolam__qty_6h"] = 0.0
        bl["meta__hours_since_encounter_start"] = hours_in
        bl["meta__label_dx_before_t0"] = (rng.random(n) < 0.3).astype(float)
        side = {**side, "A": cols_a + ["demo__sex_male", "sed__midazolam__qty_6h"]}
        side["B"], side["C"] = side["A"], side["A"] + [c for c in side["C"] if c not in side["A"]]
        side["P"] = ["demo__age_years", "demo__sex_male", "pres_score__gcs__value", "pres_first__map__value"]
        extra = {"meta": ["meta__hours_since_encounter_start", "meta__label_dx_before_t0"]}
    if scope_label:
        extra["encounter_scope_label"] = scope_label
    bl.to_parquet(lo / "baselines_AC.parquet", index=False)
    (lo / "baselines_AC.columns.json").write_text(json.dumps({"baselines": side, **extra}))

    # EEG feature parts keyed by the extractor's recording id; primary + nested windows, QC failures, duplicates, strangers
    keys = [rsf.data_io.edf_key_for_row(s, b, sid, e if isinstance(e, str) else None)
            for s, b, sid, e in zip(cohort["SiteID"], cohort["BidsFolder"], cohort["SessionID"], cohort["EEGFolder"])]
    rid = [rsf.opaque_recording_id(k) for k in keys]
    feat = ms.eeg.copy()
    qc_ok = rng.random(n) > 0.08
    base = pd.DataFrame({"recording_id": rid, "window": "primary", "qc_pass": qc_ok,
                         "usable_fraction": np.where(qc_ok, 0.9, 0.3), "onset_offset_s": rng.uniform(0, 600, n),
                         "qc_n_missing_or_dead_min": rng.integers(0, 2, n).astype(float)})
    primary = pd.concat([base, feat], axis=1)
    junk = feat.sample(frac=0.3, random_state=1).copy() * 0 + 123.0                                # must never be used
    nested = pd.concat([base.iloc[junk.index.to_numpy()].assign(window="20s"), junk], axis=1)
    dup = primary.sample(frac=0.05, random_state=2)
    stray = primary.sample(frac=0.1, random_state=3).assign(recording_id=lambda d: [f"rec{i:020d}" for i in range(len(d))])
    allrows = pd.concat([primary, nested, dup, stray], ignore_index=True).sample(frac=1.0, random_state=4)
    for k, ix in enumerate(np.array_split(np.arange(len(allrows)), 3)):
        t = pa.Table.from_pandas(allrows.iloc[ix].reset_index(drop=True), preserve_index=False)
        pq.write_table(t, lo / "features" / f"part-s{k}of3-{k:010d}.parquet", row_group_size=200)
    # silver labels (as sortinghat.labels.extract writes them: case_id index, nullable booleans, e4b hints)
    sil = pd.DataFrame({"case_id": pid})
    for j, lab in enumerate(ms.label_names):
        v = pd.array(np.where(ms.m_silver[:, j], ms.y_silver[:, j] > 0, pd.NA), dtype="boolean")
        sil[lab] = v
    sil["e4b_tox_inhospital"] = False
    sil["e4b_antidote_reversal"] = False
    sil["e4b_sedative_exposure"] = ms.covariates["sedated"]
    sil.to_csv(lo / "silver" / "silver_labels.csv", index=False)
    paths = {"cohort": lo / "cohort_study1.csv", "features": lo / "features",
             "silver": lo / "silver" / "silver_labels.csv", "baselines": lo / "baselines_AC.parquet"}
    return paths, ms, cohort


def argv_for(paths, out, *extra, small=True):
    a = ["--cohort", str(paths["cohort"]), "--features", str(paths["features"]), "--silver", str(paths["silver"]),
         "--baselines", str(paths["baselines"]), "--out", str(out)]
    if small:
        a += ["--n-boot", "300", "--auc-boot", "100", "--null-reps", "1"]
    return a + list(extra)


@pytest.fixture(scope="module")
def inputs_signal(tmp_path_factory):
    d = tmp_path_factory.mktemp("silver_signal")
    return make_inputs(d, signal=1.0, seed=3) + (d,)


# ------------------------------------------------------------------------------------------------ readers
def test_recording_id_matches_the_extractor():
    xef = load_script("extract_eeg_features")
    for k in ("EEG/bids/S0001/sub-S000112/ses-1/eeg/sub-S000112_ses-1_task-cEEG_eeg.edf", "x"):
        assert rsf.opaque_recording_id(k) == xef.opaque_id(k)


def test_recording_ids_follow_edf_key_for_row(inputs_signal):
    paths, _ms, cohort, _d = inputs_signal
    c = rsf.bb.load_cohort(paths["cohort"], None, "table")
    got = rsf.recording_ids(c)
    row = c.iloc[0]
    ef = None if pd.isna(row["EEGFolder"]) else row["EEGFolder"]
    assert got.iloc[0] == rsf.opaque_recording_id(rsf.data_io.edf_key_for_row(row["SiteID"], row["BidsFolder"], row["SessionID"], ef))
    assert got.is_unique


def test_eeg_reader_takes_only_primary_qc_rows_and_dedupes(inputs_signal):
    paths, ms, cohort, _d = inputs_signal
    c = rsf.bb.load_cohort(paths["cohort"], None, "table")
    rids = set(rsf.recording_ids(c))
    eeg, st = rsf.load_eeg_primary(paths["features"], rids, batch_rows=97)
    assert eeg.index.is_unique and set(eeg.index) <= rids
    assert (eeg.filter(like="qeeg.").abs() < 50).all().all()                     # the nested-window junk (123.0) never appears
    assert list(eeg.columns[:0]) == [] and all(c.startswith(("qeeg.", "conn.")) or c in rsf.QC_COLS for c in eeg.columns)
    assert st["n_recordings"] == len(eeg) and st["n_qc_pass"] == int(eeg["qc_pass"].sum())
    assert 0 < st["n_qc_pass"] < st["n_recordings"]
    # a recording that fails QC in the file is reported as failing
    assert not eeg["qc_pass"].all()


def test_eeg_reader_requires_local_only(tmp_path):
    (tmp_path / "features").mkdir()
    with pytest.raises(SystemExit):
        rsf.load_eeg_primary(tmp_path / "features", set())


def test_silver_loader_parses_nullable_booleans_and_excludes_nothing_silently(inputs_signal):
    paths, ms, _c, _d = inputs_signal
    Y, H = rsf.load_silver(paths["silver"])
    assert set(Y.columns) == set(rsf.CANDIDATE_LABELS)                           # E3 and E4b are never loaded
    assert Y.index.is_unique and len(Y) == ms.n
    j = ms.label_names.index("E1")
    got = Y["E1"].to_numpy()
    assert np.array_equal(np.isnan(got), ~ms.m_silver[:, j])
    assert np.array_equal(got[ms.m_silver[:, j]], ms.y_silver[ms.m_silver[:, j], j])
    assert "e4b_sedative_exposure" in H and H["e4b_sedative_exposure"].sum() == ms.covariates["sedated"].sum()


def test_baseline_loader_uses_sidecar(inputs_signal):
    paths, *_ = inputs_signal
    bl, cols = rsf.load_baselines(paths["baselines"])
    assert {"A", "C", "P", "AGE_SEX", "meta"} == set(cols) and set(cols["A"]) < set(cols["C"]) and bl["person_id"].is_unique
    assert set(cols["AGE_SEX"]) == {"demo__age_years", "demo__sex_male"} and set(cols["P"]) >= set(cols["AGE_SEX"])
    assert not set(cols["meta"]) & (set(cols["A"]) | set(cols["C"]) | set(cols["P"]))        # meta columns are never model inputs
    assert rsf.baseline_scope(paths["baselines"]) == "current encounter only"


def test_sedation_flag_sources():
    bl = pd.DataFrame({"sed__sedative__on_t0": [1.0, 0.0, 0.0, np.nan], "sed__opioid__on_t0": [0.0, 1.0, 0.0, 0.0]})
    h = pd.DataFrame({"e4b_sedative_exposure": [False, False, True, False]})
    assert rsf.sedation_flag(bl, h, "baseline").tolist() == [True, True, False, False]
    assert rsf.sedation_flag(bl, h, "e4b").tolist() == [False, False, True, False]
    assert rsf.sedation_flag(bl, h, "either").tolist() == [True, True, True, False]


def test_gcs_equivalent_falls_back_to_four():
    g = rsf.gcs_equivalent(pd.Series([8.0, np.nan, np.nan]), pd.Series([np.nan, 8.0, np.nan]))
    assert g.iloc[0] == 8.0 and g.iloc[1] == 9.0 and np.isnan(g.iloc[2])


def fake_reports_findings(cohort: pd.DataFrame, ms, mode: str = "silver") -> pd.DataFrame:
    """Synthetic reports_findings (canonical names): flags that mirror the SILVER labels exactly (mode 'silver') or the
    TRUE labels (mode 'true'). E1 <- lpd, E2 <- bs, E5 <- gpd, following the provisional comparator mapping."""
    y, m = (ms.y_silver, ms.m_silver) if mode == "silver" else (ms.y_gold, np.ones_like(ms.m_silver))
    rf = pd.DataFrame({"BDSPPatientID": cohort["person_id"].astype(str), "SessionID": cohort["SessionID"].astype(str)})
    for flag in ("foc slowing", "lpd", "lrda", "bs", "low voltage", "gpd"):
        rf[flag] = "None"
    for lab, flag in (("E1", "lpd"), ("E2", "bs"), ("E5", "gpd")):
        j = ms.label_names.index(lab)
        rf[flag] = np.where((y[:, j] > 0) & m[:, j], "present", "None")
    return rf
