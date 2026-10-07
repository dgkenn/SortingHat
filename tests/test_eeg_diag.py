"""Diagnostics scripts (key resolution + EDF signals) on a fake S3 holding SYNTHETIC recordings only.

The fake bucket mixes the real-world layouts and file quirks the diagnostics are meant to count. Outputs must be
aggregates only: no key, folder name or ID, and small-cell rules as documented in ``safe_output.technical_count``.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from sortinghat import data_io as dio
from sortinghat.eeg import diag
from sortinghat.eeg.io import CANONICAL_19
from sortinghat.eeg.synthetic import generate_eeg, write_edf_raw
from sortinghat.safe_output import SUPPRESSED, assert_aggregate_only, technical_count

sys.path.insert(0, str(Path(__file__).parent))
from test_data_io_resilient import FlakyS3  # noqa: E402

FS, DUR, R = 64.0, 130, 3276.7
SITE = "S0001"


def _dig(x):
    g = 2 * R / 65535.0
    return np.clip(np.round((x + R) / g - 32768), -32768, 32767).astype("<i2")


@pytest.fixture(scope="module")
def blobs(tmp_path_factory):
    d = tmp_path_factory.mktemp("diag")
    x = generate_eeg(DUR, fs=FS, background="normal", seed=5)
    lab = [f"EEG {c}-Ref" for c in CANONICAL_19]
    zero = _dig(np.zeros(1))[0]
    out = {}
    out["normal"] = write_edf_raw(d / "n.edf", _dig(x), lab, FS, phys_min=-R, phys_max=R, annotation_channel=True,
                                  reserved="EDF+C").read_bytes()
    dz = _dig(x)
    dz[CANONICAL_19.index("O2")] = zero
    out["zero_o2"] = write_edf_raw(d / "z.edf", dz, lab, FS, phys_min=-R, phys_max=R).read_bytes()
    pmin = [0.0 if c == "T3" else -R for c in CANONICAL_19]
    pmax = [0.0 if c == "T3" else R for c in CANONICAL_19]
    out["pmin_pmax"] = write_edf_raw(d / "p.edf", _dig(x), lab, FS, phys_min=pmin, phys_max=pmax).read_bytes()
    chain = ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1", "Fp2-F8", "F8-T4", "T4-T6", "T6-O2"]
    out["bipolar"] = write_edf_raw(d / "b.edf", _dig(x[:8]), chain, FS, phys_min=-R, phys_max=R).read_bytes()
    out["bare"] = write_edf_raw(d / "u.edf", _dig(x), [c.upper() for c in CANONICAL_19], FS, phys_min=-R,
                                phys_max=R, reserved="EDF+D").read_bytes()
    return out


def build_bucket(blobs, n=64):
    """n sessions cycling through 8 layouts; returns (objects, expected counts)."""
    rows = ["SiteID,BDSPPatientID,BidsFolder,SessionID,EEGFolder,AgeAtVisit"]
    obj: dict[str, bytes] = {}
    for i in range(n):
        bf, sid = f"sub-{SITE}{1000 + i}", str(i + 1)
        age = 40 if i % 16 else 5                                   # i % 16 == 0 are children (excluded)
        eeg = "ceeg_unit" if i % 8 == 1 else ""
        rows.append(f"{SITE},{1000 + i},{bf},{sid},{eeg},{age}")
        f = f"EEG/bids/{SITE}/{bf}/ses-{sid}/eeg/{bf}_ses-{sid}"
        k = i % 8
        if k == 0:
            obj[f + "_task-EEG_eeg.edf"] = blobs["normal"]
        elif k == 1:
            obj[f + "_task-cEEG_eeg.edf"] = blobs["normal"]
        elif k == 2:
            obj[f + "_task-cEEG_eeg.edf"] = blobs["zero_o2"]            # EEGFolder blank -> alt_task
        elif k == 3:
            obj[f + "_task-EEG_run-01_eeg.edf"] = blobs["bare"]         # extra entity -> folder_listing
            obj[f + "_eeg.json"] = b"{}"
        elif k == 4:
            obj[f + "_eeg.json"] = b"{}"                                # folder without any .edf
            obj[f + "_channels.tsv"] = b"x"
        elif k == 5:
            pass                                                        # no folder at all
        elif k == 6:
            obj[f + "_run-1_eeg.edf"] = blobs["pmin_pmax"]              # two edf
            obj[f + "_run-2_eeg.edf"] = blobs["pmin_pmax"]
        else:
            obj[f + "_task-EEG_eeg.edf"] = blobs["bipolar"]
    obj[f"EEG/eeg-metadata/{SITE}_eeg_metadata_2026_04_30.csv"] = ("\n".join(rows) + "\n").encode()
    return obj


@pytest.fixture(scope="module")
def bucket(blobs):
    return build_bucket(blobs)


def _s3(bucket):
    return FlakyS3(dict(bucket), p=0.0)


FAST = dio.RetryPolicy(max_attempts=3, backoff_s=0, sleep=lambda s: None)


def test_technical_count_rule():
    assert technical_count(7, 60) == 7 and technical_count(0, 50) == 0
    assert technical_count(7, 49) == SUPPRESSED and technical_count(11, 49) == 11
    assert technical_count(10, 10 ** 6) == 10


def test_sampling_is_adult_only_seeded_and_never_printed(bucket):
    refs, n_av = diag.sample_adult_recordings(_s3(bucket), SITE, 20, seed=0)
    refs2, _ = diag.sample_adult_recordings(_s3(bucket), SITE, 20, seed=0)
    refs3, _ = diag.sample_adult_recordings(_s3(bucket), SITE, 20, seed=1)
    assert refs == refs2 and refs != refs3 and len(refs) == 20
    assert n_av == 64 - 4                                           # i % 16 == 0 are children
    assert all(r.session_id != "1" for r in refs)                   # session "1" (i=0) is a child
    assert all(r.eeg_folder in (None, "ceeg_unit") for r in refs)


def test_paths_report_counts_every_layout(bucket):
    s3 = _s3(bucket)
    refs, n_av = diag.sample_adult_recordings(s3, SITE, 64, seed=0)
    infos = [diag.inspect_paths(s3, r, bucket="b", policy=FAST) for r in refs]
    rep = diag.aggregate_paths(infos, n_av)
    assert len(refs) == 60                                          # all adults
    kinds = {k: sum(1 for i in range(64) if i % 8 == k and i % 16) for k in range(8)}
    assert rep["resolved_by_pattern"]["documented"] == kinds[0] + kinds[1] + kinds[7]
    assert rep["resolved_by_pattern"]["alt_task"] == kinds[2]
    assert rep["resolved_by_pattern"]["folder_listing"] == kinds[3] + kinds[6]
    assert rep["n_not_found"] == kinds[4] + kinds[5]
    f = rep["recording_folder"]
    assert (f["exists"], f["missing"]) == (60 - kinds[5], kinds[5])
    assert f["exists_but_no_edf"] == kinds[4] and f["multiple_edf"] == kinds[6]
    assert f["exactly_one_edf"] == 60 - kinds[4] - kinds[5] - kinds[6]
    assert rep["files_by_extension"]["json"] == kinds[3] + kinds[4]
    assert rep["files_by_extension"]["tsv"] == kinds[4]
    assert rep["files_by_extension"]["edf"] == 60 - kinds[4] - kinds[5] + kinds[6]
    assert rep["folders_containing_extension"]["edf"] == 60 - kinds[4] - kinds[5]
    assert "exact counts" in rep["count_rule"]
    text = json.dumps(rep)
    assert "sub-" not in text and "ses-" not in text and ".edf" not in text and SITE not in text
    assert_aggregate_only(rep)


def test_paths_small_samples_are_suppressed(bucket):
    s3 = _s3(bucket)
    refs, n_av = diag.sample_adult_recordings(s3, SITE, 30, seed=0)
    rep = diag.aggregate_paths([diag.inspect_paths(s3, r, bucket="b", policy=FAST) for r in refs], n_av)
    assert "suppressed" in rep["count_rule"]
    vals = list(rep["recording_folder"].values()) + list(rep["resolved_by_pattern"].values())
    assert all(v == SUPPRESSED or v >= 11 for v in vals)


def test_signals_report(bucket):
    s3 = _s3(bucket)
    refs, n_av = diag.sample_adult_recordings(s3, SITE, 64, seed=0)
    infos = [diag.inspect_signals(s3, r, bucket="b", policy=FAST, sleep=lambda s: None) for r in refs]
    rep = diag.aggregate_signals(infos, n_av)
    kinds = {k: sum(1 for i in range(64) if i % 8 == k and i % 16) for k in range(8)}
    assert rep["status"]["ok"] == 60 - kinds[4] - kinds[5]
    assert rep["status"]["unresolved"] == kinds[4] + kinds[5]
    n_ok = 60 - kinds[4] - kinds[5]
    assert rep["n_analysed"] == n_ok
    # header facts
    assert rep["edf_type"]["EDF+C"] == kinds[0] + kinds[1]
    assert rep["edf_type"]["EDF+D"] == kinds[3]
    assert rep["recordings_with_annotation_signal"] == kinds[0] + kinds[1]
    assert rep["n_signals_total"]["20"] == kinds[0] + kinds[1] and rep["n_signals_total"]["8"] == kinds[7]
    assert rep["recordings_with_bipolar_labels"] == kinds[7]
    # label normalisation: every required electrode found except in bipolar-only files
    for ch in diag.REQUIRED:
        assert rep["required_electrode_found_after_normalisation"][ch] == n_ok - kinds[7]
    top = dict((lb, c) for lb, c in rep["top_raw_labels_by_recordings"])
    assert top["EEG Fp1-Ref"] == kinds[0] + kinds[1] + kinds[2] + kinds[6]
    assert top["FP1"] == kinds[3] and len(rep["top_raw_labels_by_recordings"]) <= 40
    # calibration
    assert rep["scaling"]["recordings_with_pmin_eq_pmax"] == kinds[6]
    assert rep["scaling"]["signals_pmin_eq_pmax"] == kinds[6]
    assert rep["scaling"]["recordings_with_dmin_eq_dmax"] == 0
    assert rep["sample_rates_hz_of_required_electrodes"]["64"] == n_ok - kinds[7]
    # constant epochs: O2 zero-filled in kinds[2] recordings (all epochs), T3 zero-range in kinds[6]
    o2 = rep["primary_window_exactly_constant_epochs_by_electrode"]["O2"]
    t3 = rep["primary_window_exactly_constant_epochs_by_electrode"]["T3"]
    assert o2["n_recordings_all_epochs_const"] == kinds[2] and o2["frac_epochs_zero_valued"] > 0
    assert t3["n_recordings_all_epochs_const"] == kinds[6] and t3["frac_epochs_const_by_zero_calibration"] > 0
    assert o2["frac_epochs_const_by_zero_calibration"] == 0
    cause = rep["technical_cause_per_recording"]
    assert cause["required_channel_constant_whole_window"] == kinds[2]
    assert cause["required_channel_zero_calibration_range"] == kinds[6]
    assert cause["bipolar_labels_only"] == kinds[7]
    assert cause["edf_plus_d_discontinuous"] == kinds[3]
    assert rep["primary_reason_counts"]["dead_minimum_channels"] == kinds[2]
    text = json.dumps(rep)
    assert "sub-" not in text and "ses-" not in text and SITE not in text
    assert_aggregate_only(rep)


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("name", ["diag_eeg_paths", "diag_eeg_signals"])
def test_scripts_print_aggregates_only(name, bucket, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(dio, "access_point", lambda name="credentialed": "b")
    mod = _load_script(name)
    out = tmp_path / "rep.json"
    rc = mod.main(["--site", SITE, "--n", "60", "--seed", "0", "--out", str(out)], s3=_s3(bucket))
    text = capsys.readouterr().out
    assert rc == 0 and "n_sampled: 60" in text
    for bad in ("sub-S0001", "ses-", "EEG/bids", "1001", ".edf"):
        assert bad not in text
    assert json.loads(out.read_text())["n_sampled"] == 60


def test_scripts_report_class_name_only_on_failure(monkeypatch, capsys):
    mod = _load_script("diag_eeg_signals")

    class Boom(FlakyS3):
        def list_objects_v2(self, *a, **k):
            raise RuntimeError("secret sub-S0001123 message")

    rc = mod.main(["--site", SITE, "--n", "5"], s3=Boom({}))
    out = capsys.readouterr().out
    assert rc == 2 and "RuntimeError" in out and "secret" not in out and "sub-S0001123" not in out


def test_scripts_refuse_real_client_inside_agent_session(monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    mod = _load_script("diag_eeg_paths")
    assert mod.main(["--site", SITE]) == 2                          # make_client refuses; only the class name is printed


# ---- signal onset (D-108) -------------------------------------------------------------------------------
def _padded_blob(segments, seed=0):
    g = 2 * R / 65535.0
    parts = []
    for k, (kind, sec) in enumerate(segments):
        n = int(sec * FS)
        if kind == "pad":
            parts.append(np.full((19, n), 1234, "<i2"))
        else:
            x = np.random.default_rng(seed + k).standard_normal((19, n)) * 20.0
            parts.append(np.clip(np.round((x + R) / g - 32768), -32768, 32767).astype("<i2"))
    return write_edf_raw_bytes(np.concatenate(parts, axis=1))


def write_edf_raw_bytes(dig, _cache={}):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        return write_edf_raw(Path(d) / "x.edf", dig, [f"EEG {c}-Ref" for c in CANONICAL_19], FS, phys_min=-R,
                             phys_max=R).read_bytes()


@pytest.fixture(scope="module")
def onset_bucket():
    blobs = {
        0: _padded_blob([("sig", 760)]),
        1: _padded_blob([("pad", 120), ("sig", 760)]),
        2: _padded_blob([("pad", 300), ("sig", 100), ("pad", 400), ("sig", 200)]),    # signal, then a mid-file gap
        3: _padded_blob([("pad", 400)]),                                              # never any signal
    }
    n = 52
    rows = ["SiteID,BDSPPatientID,BidsFolder,SessionID,EEGFolder,AgeAtVisit"]
    obj = {}
    for i in range(n):
        bf, sid = f"sub-{SITE}{2000 + i}", str(i + 1)
        rows.append(f"{SITE},{2000 + i},{bf},{sid},,40")
        obj[f"EEG/bids/{SITE}/{bf}/ses-{sid}/eeg/{bf}_ses-{sid}_task-EEG_eeg.edf"] = blobs[i % 4]
    obj[f"EEG/eeg-metadata/{SITE}_eeg_metadata_2026_04_30.csv"] = ("\n".join(rows) + "\n").encode()
    return obj, {k: sum(1 for i in range(n) if i % 4 == k) for k in range(4)}


def test_signal_onset_diagnostics(onset_bucket):
    obj, k = onset_bucket
    s3 = FlakyS3(dict(obj), p=0.0)
    refs, n_av = diag.sample_adult_recordings(s3, SITE, 60, seed=0)
    infos = [diag.inspect_signals(s3, r, bucket="b", policy=FAST, sleep=lambda s: None) for r in refs]
    rep = diag.aggregate_signals(infos, n_av)
    on = rep["signal_onset"]
    assert on["n_no_signal_onset_within_120_min"] == k[3]
    assert on["n_onset_after_file_start"] == k[1] + k[2]
    q = on["onset_offset_minutes_from_file_start_quantiles"]
    assert q["q10"] == 0.0 and q["q90"] == 5.0                       # 0, 2 and 5 min offsets
    a = rep["after_onset"]
    assert a["n_with_onset_and_window"] == k[0] + k[1] + k[2]
    assert a["n_usable_below_threshold"] == k[2]                     # the mid-file gap removes ~2/3 of the window
    assert a["primary_pass"] == k[0] + k[1]
    assert a["technical_cause_per_recording"]["no_technical_problem"] == k[0] + k[1]
    assert a["technical_cause_per_recording"].get("required_channel_constant_whole_window", 0) == 0
    assert a["technical_cause_per_recording"]["low_usable_other_artifacts"] == k[2]
    bd = a["qc_rule_breakdown_usable_below_threshold"]
    flags = bd["mean_share_of_minimum_set_cells_flagged_by_rule"]
    assert flags["flat"] > 0.5 and flags["extreme"] == 0.0 and flags["clipping"] == 0.0
    assert set(flags) == {"flat", "clipping", "extreme", "line_noise", "disconnected"}
    assert bd["n_recordings"] == k[2] and 0.5 < bd["mean_unusable_epoch_fraction"] <= 1.0
    assert 5 < bd["required_channel_std_uv_quantiles"]["q50"] < 100
    assert "line_noise_ratio_quantiles" in bd and bd["line_noise_ratio_quantiles"]["q50"] < 1.0
    # before the fix: the file-start window sees padding / the gap instead of the EEG
    fs_cause = rep["technical_cause_per_recording"]
    assert fs_cause.get("required_channel_constant_whole_window", 0) >= k[2]
    # constant epochs after onset are far fewer than at file start
    c_after = a["exactly_constant_epochs_by_electrode"]["Fp1"]["frac_epochs_const_digital"]
    c_start = rep["primary_window_exactly_constant_epochs_by_electrode"]["Fp1"]["frac_epochs_const_digital"]
    assert 0 < c_after < c_start
    text = json.dumps(rep)
    assert "sub-" not in text and "ses-" not in text and SITE not in text
    assert_aggregate_only(rep)
