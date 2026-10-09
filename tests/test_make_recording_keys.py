"""make_recording_keys.py selects key rows with the feasibility script's own cohort logic (load_cohort). SYNTHETIC ONLY."""
import importlib.util
import stat
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent))
_spec = importlib.util.spec_from_file_location("make_recording_keys", SCRIPTS / "make_recording_keys.py")
mrk = importlib.util.module_from_spec(_spec)
sys.modules["make_recording_keys"] = mrk
_spec.loader.exec_module(mrk)
import build_baselines as bb  # noqa: E402


def _frames(n=60):
    pid = 9_000_000 + pd.Series(range(n))
    site = ["S0001" if i % 2 == 0 else "S0002" for i in range(n)]
    site[0:3] = ["I0002"] * 3                                    # outside the study sites
    sess = [str(1 + i % 3) for i in range(n)]                    # SessionIDs repeat across patients
    cohort = pd.DataFrame({"SiteID": site, "person_id": pid, "person_id_source": pid, "SessionID": sess,
                           "t0": ["2024-03-01 08:00:00"] * n,
                           "in_strict": [i % 4 == 0 for i in range(n)], "in_strict_pm6": [i % 4 == 0 for i in range(n)],
                           "in_broad": [i % 2 == 0 for i in range(n)]})
    keys = cohort[["SiteID", "person_id", "SessionID"]].copy()
    keys["edf_key"] = [f"k{i}" for i in range(n)]
    keys = pd.concat([keys, pd.DataFrame({"SiteID": ["S0001"], "person_id": [1], "SessionID": ["1"], "edf_key": ["x"]})])  # not in cohort
    return cohort, keys.sample(frac=1, random_state=0).reset_index(drop=True)


@pytest.mark.parametrize("cdef", ["strict", "broad", "table"])
def test_selection_matches_load_cohort(tmp_path, cdef, capsys):
    cohort, keys = _frames()
    lo = tmp_path / "local_only"
    lo.mkdir()
    cohort.to_csv(lo / "cohort.csv", index=False)
    keys.to_csv(lo / "keys.csv", index=False)
    out = lo / f"keys_{cdef}.csv"
    assert mrk.main(["--cohort-def", cdef, "--keys", str(lo / "keys.csv"), "--cohort", str(lo / "cohort.csv"), "--out", str(out)]) == 0
    got = pd.read_csv(out, dtype=str)
    want = bb.load_cohort(lo / "cohort.csv", list(bb.STUDY_SITES), cdef)
    assert len(got) == len(want) > 0
    assert set(got["edf_key"]) == {f"k{i - 9_000_000}" for i in want["person_id"]}
    assert "x" not in set(got["edf_key"]) and "I0002" not in set(got["SiteID"])
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    shown = capsys.readouterr().out
    assert "9000" not in shown and "S0001" not in shown          # aggregate counts only, site pseudonyms


def test_refuses_out_outside_local_only(tmp_path):
    cohort, keys = _frames()
    lo = tmp_path / "local_only"
    lo.mkdir()
    cohort.to_csv(lo / "cohort.csv", index=False)
    keys.to_csv(lo / "keys.csv", index=False)
    with pytest.raises(SystemExit):
        mrk.main(["--cohort-def", "broad", "--keys", str(lo / "keys.csv"), "--cohort", str(lo / "cohort.csv"),
                  "--out", str(tmp_path / "keys_broad.csv")])
