"""Regression: the cohort table / key list must be row-consistent with eeg_metadata at realistic scale.

Real HEEDB SessionIDs are small integers (1..N per patient) that REPEAT across patients, and OMOP tables have many row
groups. A join on SessionID alone (``sources.site_sessions``) once copied the first patient's BidsFolder / EEGFolder /
ServiceName / duration onto every patient with the same SessionID (real run: 12,826 patients, 33 BidsFolders).
Synthetic data only: >= 20k patients, 1-4 sessions each with SessionIDs 1..N, shuffled rows, four site header variants,
OMOP parquet with several row groups per part.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sortinghat import data_io
from sortinghat.cohort import CohortConfig, StoreSources, build_cohort, write_outputs
from sortinghat.cohort.build import KEY_LIST_COLUMNS
from sortinghat.cohort.output import read_key_list

N_PATIENTS = 24_000
SITES = {"S0001": 8000, "S0002": 4000, "I0002": 4000, "I0003": 4000, "I0008": 4000}
BASE = pd.Timestamp("2021-03-01 08:00:00")
FMT = "%Y-%m-%d %H:%M:%S"

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "extract_eeg_features.py"
_spec = importlib.util.spec_from_file_location("extract_eeg_features_keys", SCRIPT)
xef = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(xef)


def _fmt(s: pd.Series) -> pd.Series:
    return s.dt.strftime(FMT).astype(object).where(s.notna(), None)


def _write_parquet(path_dir: Path, df: pd.DataFrame, n_parts: int = 2, groups_per_part: int = 3) -> None:
    """Shuffled rows, ``n_parts`` parts, ``groups_per_part`` row groups per part (person ids spread across all groups)."""
    path_dir.mkdir(parents=True, exist_ok=True)
    df = df.sample(frac=1.0, random_state=3).reset_index(drop=True)
    for k, part in enumerate(np.array_split(np.arange(len(df)), n_parts)):
        t = pa.Table.from_pandas(df.iloc[part].reset_index(drop=True), preserve_index=False)
        pq.write_table(t, path_dir / f"part-{k:05d}.parquet", row_group_size=max(1, len(part) // groups_per_part + 1))


def build_store(root: Path):
    """Write a synthetic HEEDB-layout store and return the per-session truth frame (one row per eeg_metadata row)."""
    rng = np.random.default_rng(20260901)
    pids = rng.choice(np.arange(1_111_111_111, 1_111_111_111 + 400_000), N_PATIENTS, replace=False).astype("int64")
    site_of = np.repeat(list(SITES), list(SITES.values()))
    n_sess = rng.integers(1, 5, N_PATIENTS)
    rows = []
    for pid, site, n in zip(pids, site_of, n_sess):
        for sid in range(1, n + 1):
            rows.append((site, int(pid), str(sid)))
    T = pd.DataFrame(rows, columns=["SiteID", "person_id", "SessionID"])
    n = len(T)
    # session start: sessions of a patient are 3-60 days apart, in a SHUFFLED order relative to SessionID
    pat_off = rng.integers(0, 600, N_PATIENTS)
    T["_pat"] = np.repeat(np.arange(N_PATIENTS), n_sess)
    perm_gap = rng.integers(3, 60, n)
    slot = np.zeros(n, "int64")
    for _, idx in T.groupby("_pat").indices.items():            # random permutation of the day-slots within a patient
        slot[idx] = rng.permutation(len(idx))
    T["t0"] = pd.Series(BASE + pd.to_timedelta(pat_off[T["_pat"]] + slot * perm_gap.max() + (perm_gap % 7), unit="D")
               + pd.to_timedelta(rng.integers(0, 12 * 3600, n), unit="s")).dt.floor("s")
    T["dur"] = rng.integers(800, 4000, n)
    short_first = rng.random(N_PATIENTS) < 0.05              # these patients' first selected EEG is too short
    T["bad_age"] = rng.random(n) < 0.15                       # a pediatric session
    T["age"] = np.where(T["bad_age"], rng.uniform(1, 17, n), rng.uniform(19, 90, n)).round(2)
    T["service"] = rng.choice(["LTM", "ICU", "ED", "OR", "EMU", "Routine"], n, p=[.3, .2, .2, .1, .05, .15])
    T["eeg_folder"] = rng.choice(["cEEG", "EEG", "ceeg_inpatient", ""], n)
    T["bids"] = [f"sub-{s}{p}" for s, p in zip(T["SiteID"], T["person_id"])]
    blank_bids = (T["SiteID"] == "I0008").to_numpy() & (rng.random(n) < 0.03)
    T["bids_given"] = T["bids"].where(~blank_bids, None)

    # durations: set the too-short flag on the patient's earliest otherwise-qualifying session later (after oracle); here
    # just make the earliest session of `short_first` patients short at 100 s (qualification does not depend on duration)
    first_idx = T.sort_values(["_pat", "t0"]).drop_duplicates("_pat").index
    T.loc[first_idx[short_first], "dur"] = 100
    T["t_end"] = T["t0"] + pd.to_timedelta(T["dur"], unit="s")

    eeg = root / "EEG/eeg-metadata"
    rfd = root / "EEG/HEEDB_Metadata"
    eeg.mkdir(parents=True)
    rfd.mkdir(parents=True)
    for site in SITES:
        g = T[T["SiteID"] == site].sample(frac=1.0, random_state=5).reset_index(drop=True)      # shuffled file order
        md = pd.DataFrame({"SiteID": site, "BDSPPatientID": g["person_id"].astype(str), "BidsFolder": g["bids_given"],
                           "SessionID": g["SessionID"], "DurationInSeconds": g["dur"].astype(float)})
        if site in ("S0001", "S0002"):
            md["EEGFolder"] = g["eeg_folder"].where(g["eeg_folder"] != "", None)
            md["ServiceName"] = g["service"]
            if site == "S0002":                                 # BDSPPatientID blank in some releases -> derive from BidsFolder
                md.loc[rng.random(len(md)) < 0.1, "BDSPPatientID"] = None
        elif site == "I0002":
            md["AgeAtVisit"], md["StartTime"], md["EndTime"] = g["age"], _fmt(g["t0"]), _fmt(g["t_end"])
        elif site == "I0003":
            md["ServiceName"] = g["service"]
            md["AgeInDaysAtVisit"], md["StartTime"], md["EndTime"] = (g["age"] * 365.25).round(), _fmt(g["t0"]), _fmt(g["t_end"])
        else:                                                   # I0008: InstituteID, DateOfBirth, no age, no reports_findings
            md = md.drop(columns="SiteID")
            md["InstituteID"] = site
            md["StartTime"], md["EndTime"] = _fmt(g["t0"]), _fmt(g["t_end"])
            md["DateOfBirth"] = _fmt((g["t0"].dt.normalize() - pd.to_timedelta((g["age"] * 365.25).round(), unit="D")))
        data_io.denormalise_site_table("eeg_metadata", site, md).astype(object).to_csv(
            eeg / f"{site}_eeg_metadata_2026_04_30.csv", index=False)
        if site != "I0008":
            rf = pd.DataFrame({"BDSPPatientID": g["person_id"].astype(str), "SessionID": g["SessionID"],
                               "StartTime(EEG)": _fmt(g["t0"]), "EndTime(EEG)": _fmt(g["t_end"]), "AgeAtVisit": g["age"]})
            if site in ("S0001", "S0002"):
                rf["ServiceName(EEG)"] = g["service"]
            data_io.denormalise_site_table("reports_findings", site, rf).astype(object).to_csv(
                rfd / f"{site}_EEG__reports_findings.csv", index=False)

    # OMOP: for EVERY session a covering inpatient-length visit (+ a decoy outpatient visit), a GCS 8 at t0 - 30 min,
    # a sodium lab. So any session that passes the session-level steps qualifies all the way to the table.
    v_start = T["t0"] - pd.to_timedelta(rng.integers(2, 20, n), unit="h")
    visits = pd.DataFrame({
        "person_id": T["person_id"], "visit_occurrence_id": np.arange(n),
        "visit_start_datetime": _fmt(v_start), "visit_end_datetime": _fmt(v_start + pd.Timedelta(days=3)),
        "visit_concept_id": 0, "visit_source_value": ""})
    decoy = pd.DataFrame({
        "person_id": T["person_id"], "visit_occurrence_id": np.arange(n) + n,
        "visit_start_datetime": _fmt(T["t0"] - pd.Timedelta(minutes=50)), "visit_end_datetime": _fmt(T["t0"] - pd.Timedelta(minutes=20)),
        "visit_concept_id": 0, "visit_source_value": ""})
    other = pd.DataFrame({"person_id": np.arange(3000) + 5_000_000_000, "visit_occurrence_id": np.arange(3000) + 2 * n,
                          "visit_start_datetime": _fmt(pd.Series(BASE + pd.to_timedelta(np.arange(3000), unit="h"))),
                          "visit_end_datetime": None, "visit_concept_id": 0, "visit_source_value": ""})
    _write_parquet(root / "OMOP/Merged/visit_occurrence", pd.concat([visits, decoy, other], ignore_index=True), 2, 3)
    meas = pd.concat([
        pd.DataFrame({"person_id": T["person_id"], "measurement_datetime": _fmt(T["t0"] - pd.Timedelta(minutes=30)),
                      "measurement_source_value": "Glasgow Coma Scale Score", "value_as_number": 8.0}),
        pd.DataFrame({"person_id": T["person_id"], "measurement_datetime": _fmt(T["t0"] - pd.Timedelta(hours=1)),
                      "measurement_source_value": "SODIUM", "value_as_number": 140.0})], ignore_index=True)
    _write_parquet(root / "OMOP/Merged/measurement", meas, 2, 3)
    cond = pd.DataFrame({"person_id": T["person_id"].iloc[::5], "condition_start_datetime": _fmt(T["t0"].iloc[::5]),
                         "condition_source_value": "R41.82"})
    _write_parquet(root / "OMOP/Merged/condition_occurrence", cond, 2, 3)
    return T


def _oracle(T: pd.DataFrame, cfg: CohortConfig) -> pd.DataFrame:
    """Independent expectation: per patient the EARLIEST session passing the session-level rules (study site, adult age,
    service not OR / EMU), then kept if its clock duration is >= the minimum (every session has a covering inpatient visit,
    a GCS 8 at t0 - 30 min and a visit start 2-20 h before t0, so the other steps pass)."""
    q = T[T["SiteID"].isin(cfg.study_sites or SITES) & ~T["bad_age"] & ~T["service"].where(T["SiteID"].isin(["S0001", "S0002", "I0003"]), "").isin(["OR", "EMU"])]
    first = q.sort_values(["person_id", "t0"]).drop_duplicates("person_id")
    return first[first["dur"] >= cfg.min_duration_s]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("keys_store")
    T = build_store(root)
    cfg = CohortConfig(study_sites=None)                         # all five sites, so all four header variants are exercised
    assert N_PATIENTS >= 20_000 and T["person_id"].nunique() >= 20_000
    assert all(len(list(p.glob("*.parquet"))) >= 2 for p in (root / "OMOP/Merged").iterdir())
    assert all(pq.ParquetFile(f).num_row_groups >= 3 for f in (root / "OMOP/Merged/measurement").glob("*.parquet"))
    res = build_cohort(StoreSources(data_io.open_store(str(root))), cfg)
    return T, cfg, res, root


def _truth_join(rows: pd.DataFrame, T: pd.DataFrame) -> pd.DataFrame:
    """Each output row joined to the source rows of its (person_id, SessionID); asserts exactly one source row each."""
    j = rows.assign(person_id=rows["person_id"].astype("int64"), SessionID=rows["SessionID"].astype(str)).merge(
        T.assign(_b=T["bids"]), on=["person_id", "SessionID"], how="left", suffixes=("", "_src"), validate="m:1")
    assert j["_b"].notna().all(), "output rows with no source eeg_metadata row"
    return j


def test_scale_and_session_ids_repeat(built):
    T, cfg, res, _ = built
    assert T["SessionID"].nunique() <= 4 and T["person_id"].nunique() >= 20_000      # the real-data shape
    assert len(res.table) > 10_000


def test_table_rows_match_one_source_row(built):
    T, cfg, res, _ = built
    t = res.table
    j = _truth_join(t, T)
    assert (j["SiteID"] == j["SiteID_src"]).all()
    assert (j["BidsFolder"] == j["_b"]).all()                          # given, or filled from the patient id when blank
    assert (j["t0"] == j["t0_src"]).all()
    assert (j["person_id_source"].astype("int64") == j["person_id"]).all()
    # what BidsFolder / SessionID uniqueness the real run lacked
    assert t["person_id"].is_unique and t["BidsFolder"].is_unique
    assert not t.duplicated(["BidsFolder", "SessionID"]).any()
    assert t["BidsFolder"].nunique() == len(t) and t["SessionID"].nunique() <= 4
    # ServiceName / EEGFolder / duration come from the SAME source row
    svc_sites = j["SiteID"].isin(["S0001", "S0002", "I0003"])
    assert (j.loc[svc_sites, "ServiceName"].astype(str) == j.loc[svc_sites, "service"].str.upper()).all()
    s = j["SiteID"].isin(["S0001", "S0002"])
    assert (j.loc[s, "EEGFolder"].fillna("") == j.loc[s, "eeg_folder"]).all()
    assert np.allclose(j["duration_s"], j["dur"])


def test_key_list_rows_match_one_source_row(built):
    T, cfg, res, _ = built
    k = res.keys
    assert list(k.columns) == KEY_LIST_COLUMNS and len(k) == len(res.table)
    assert (k["person_id"].to_numpy() == res.table["person_id"].to_numpy()).all()
    j = _truth_join(k, T)
    assert (j["SiteID"] == j["SiteID_src"]).all() and (j["BidsFolder"] == j["_b"]).all()
    assert not k.duplicated(["BidsFolder", "SessionID"]).any()
    assert k["edf_key"].is_unique and k["BidsFolder"].is_unique
    s = j["SiteID"].isin(["S0001", "S0002"])
    assert (j.loc[s, "EEGFolder"].fillna("") == j.loc[s, "eeg_folder"]).all()
    assert (j["clock_duration_s"] == j["dur"]).all()


def test_first_eeg_is_the_earliest_qualifying_one(built):
    T, cfg, res, _ = built
    want = _oracle(T, cfg).set_index("person_id")
    got = res.table.set_index("person_id")
    assert set(got.index) == set(want.index)
    assert (got["SessionID"].astype(str) == want.loc[got.index, "SessionID"]).all()
    assert (got["t0"] == want.loc[got.index, "t0"]).all()
    assert got["SessionID"].astype(str).nunique() >= 3            # not every patient's first EEG is session 1


def test_build_aborts_when_sessions_are_misaligned(built):
    """The runtime check: a source that mis-joins SessionID (the real-run symptom) must abort the build, aggregate message."""
    T, cfg, res, root = built
    from sortinghat.cohort.integrity import CohortIntegrityError, verify_output

    class Misaligned(StoreSources):
        def sessions(self):
            S = super().sessions()
            first = S.groupby(["SiteID", "SessionID"])["BidsFolder"].transform("first")      # the SessionID-only join
            return S.assign(BidsFolder=first)

    with pytest.raises(CohortIntegrityError) as e:
        build_cohort(Misaligned(data_io.open_store(str(root))), cfg)
    msg = str(e.value)
    assert "sub-" not in msg and not any(str(p) in msg for p in T["person_id"].iloc[:2000])
    with pytest.raises(CohortIntegrityError):                     # output uniqueness: duplicate edf_key / BidsFolder
        verify_output(res.table.assign(BidsFolder=res.table["BidsFolder"].iloc[0]), res.keys)
    with pytest.raises(CohortIntegrityError):
        verify_output(res.table, res.keys.assign(edf_key=res.keys["edf_key"].iloc[0]))
    verify_output(res.table, res.keys)                            # clean: no raise


def test_extractor_loads_every_key_both_ways(built, tmp_path):
    T, cfg, res, _ = built
    out = tmp_path / "o"
    paths = write_outputs(res, out)
    k = read_key_list(paths["keys"])
    n = len(res.keys)
    rec = xef.load_recordings_with_parts(paths["keys"])
    assert len(rec) == n and len({r for r, _, _ in rec}) == n and len({kk for _, kk, _ in rec}) == n
    assert {kk for _, kk, _ in rec} == set(res.keys["edf_key"])
    # the extractor's own construction from BidsFolder + SessionID + SiteID + EEGFolder gives the SAME keys
    alt = tmp_path / "alt_keys.csv"
    k.drop(columns=["edf_key"]).to_csv(alt, index=False)
    rec2 = xef.load_recordings_with_parts(alt)
    assert len(rec2) == n and [kk for _, kk, _ in rec2] == list(res.keys["edf_key"])
    for (_, key, parts), (s, b, sid, ef) in zip(rec2[:2000], res.keys[["SiteID", "BidsFolder", "SessionID", "EEGFolder"]].itertuples(index=False)):
        assert key == data_io.bids_edf_key(s, b, str(sid), None if pd.isna(ef) else str(ef))


def test_baseline_index_does_not_mix_patients_with_equal_session_ids():
    """Same class of bug in baselines.events.build_index: referral indication / sex were joined on SessionID alone."""
    from sortinghat import schema
    from sortinghat.baselines.events import build_index
    n = 300
    pid = 2_000_000_000 + np.arange(n)
    t0 = pd.Timestamp("2022-01-01 10:00") + pd.to_timedelta(np.arange(n), unit="h")
    meta = pd.DataFrame({"SiteID": "S0001", "BDSPPatientID": pid.astype(str), "BidsFolder": [f"sub-S0001{p}" for p in pid],
                         "SessionID": "1", "PatientClass": "ICU", "AgeAtVisit": 50.0,
                         "ReferralIndication": [f"ind{p}" for p in pid], "SexDSC": np.where(np.arange(n) % 2, "Male", "Female"),
                         "StartTime": t0, "EndTime": t0 + pd.Timedelta(hours=1)})
    rf = pd.DataFrame({"BDSPPatientID": pid.astype(str), "SessionID": "1", schema.START_EEG: t0,
                       schema.END_EEG: t0 + pd.Timedelta(hours=1), "AgeAtVisit": 50.0,
                       "SexDSC": np.where(np.arange(n) % 2, "Male", "Female")})
    idx = build_index({"eeg_metadata": meta.sample(frac=1.0, random_state=1), "reports_findings": rf.sample(frac=1.0, random_state=2)})
    assert len(idx) == n
    assert (idx["indication_raw"].to_numpy() == [f"ind{p}" for p in idx["person_id"]]).all()
    assert (idx["sex_male"].to_numpy() == (idx["person_id"].to_numpy() - 2_000_000_000) % 2).all()
