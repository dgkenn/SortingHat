"""Synthetic fixtures reproducing the real-data patterns behind "median usable fraction 0":

* label styles (``EEG Fp1-Ref``, ``FP1``, ``Fp1-AVG``, ``EEG T7``, ``C3-A2`` ...) must normalise to the 19 channels;
* a zero-filled / constant placeholder channel must be reported MISSING (``dead``), not flat data;
* a zero physical range (pmin == pmax) or zero digital range must be reported as invalid scaling (a calibration
  problem that decodes to a constant), not as a flat electrode;
* EDF+ annotation signals are skipped, EDF+C reads, EDF+D is refused with its own reason.
"""
import numpy as np
import pytest

from sortinghat.eeg.io import (CANONICAL_19, DEFAULT_MINIMUM_CHANNELS, EDFError, channel_scaling_status,
                               drop_dead_channels, is_bipolar_label, normalize_channel_name, read_edf,
                               read_edf_header, Recording)
from sortinghat.eeg.pipeline import process_recording
from sortinghat.eeg.stream import FailureReason, stream_features
from sortinghat.eeg.synthetic import generate_eeg, write_edf_raw
from sortinghat.eeg.window import WindowSpec, summarize_window_qc

FS = 128.0
DUR = 700
R = 3276.7
KEY = "EEG/bids/SYN/sub-SYN1/ses-1/eeg/sub-SYN1_ses-1_task-EEG_eeg.edf"
BUCKET = "fake-ap"


@pytest.fixture(scope="module")
def eeg_uv():
    return generate_eeg(DUR, fs=FS, background="normal", seed=21)


def to_dig(x_uv, rng_uv=R):
    gain = 2 * rng_uv / 65535.0
    return np.clip(np.round((x_uv + rng_uv) / gain - 32768), -32768, 32767).astype("<i2")


def zero_code(rng_uv=R):
    return int(to_dig(np.zeros(1), rng_uv)[0])


def write(tmp_path, name, x_uv, labels, **kw):
    kw.setdefault("phys_min", -R)
    kw.setdefault("phys_max", R)
    return write_edf_raw(tmp_path / name, to_dig(x_uv), labels, FS, **kw)


STYLES = {
    "ref": lambda c: f"EEG {c}-Ref",
    "REF_upper": lambda c: f"EEG {c.upper()}-REF",
    "bare_upper": lambda c: c.upper(),
    "avg": lambda c: f"{c}-AVG",
    "modern": lambda c: f"EEG {({'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}).get(c, c)}",
    "pol": lambda c: f"POL {c}",
    "ear_ref": lambda c: f"{c}-A2" if c in ("Fp2", "F4", "C4", "P4", "O2", "F8", "T4", "T6") else f"{c}-A1",
    "g2": lambda c: f"EEG {c}-G2",
    "spaced": lambda c: f"EEG {c} - Ref",
}


@pytest.mark.parametrize("style", sorted(STYLES))
def test_label_styles_normalise_and_pass_qc(tmp_path, eeg_uv, style):
    labels = [STYLES[style](c) for c in CANONICAL_19]
    assert [normalize_channel_name(lb) for lb in labels] == list(CANONICAL_19)
    p = write(tmp_path, f"{style}.edf", eeg_uv, labels, annotation_channel=True)
    out = process_recording(p)
    assert out.channel_status["n_channels"] == 19 and out.channel_status["n_missing_min"] == 0
    prim = out.qc["primary"]
    assert prim.passes and prim.usable_fraction > 0.95 and prim.reasons == []


def test_bipolar_labels_are_not_referential(tmp_path, eeg_uv):
    chain = ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1", "Fp2-F8", "F8-T4", "T4-T6", "T6-O2"]
    assert all(normalize_channel_name(c) is None and is_bipolar_label(c) for c in chain)
    assert not is_bipolar_label("EEG Fp1-Ref") and not is_bipolar_label("ECG")
    p = write(tmp_path, "bip.edf", eeg_uv[:8], chain)
    with pytest.raises(EDFError):
        read_edf(p)


def test_duplicate_signals_for_one_channel_do_not_abort_the_read(tmp_path, eeg_uv):
    labels = ["Fp1-A1", "Fp1-A2"] + [f"EEG {c}-Ref" for c in CANONICAL_19[1:]]
    x = np.vstack([eeg_uv[0], eeg_uv[0] * 0.5, eeg_uv[1:]])
    rec = read_edf(write(tmp_path, "dup.edf", x, labels))
    assert rec.ch_names == list(CANONICAL_19) and rec.meta["duplicate_channels"] == ["Fp1"]
    assert np.allclose(rec.data[0], eeg_uv[0], atol=0.1)             # first signal wins


# ---- zero-filled / constant placeholder channels ---------------------------------------------------
def _with_const(eeg_uv, names, value_code=None):
    dig = to_dig(eeg_uv)
    for c in names:
        dig[CANONICAL_19.index(c)] = zero_code() if value_code is None else value_code
    return dig


@pytest.mark.parametrize("code", [None, 1234])
def test_zero_filled_channel_is_missing_not_flat(tmp_path, eeg_uv, code):
    dig = _with_const(eeg_uv, ["O2"], code)
    p = write_edf_raw(tmp_path / "z.edf", dig, [f"EEG {c}-Ref" for c in CANONICAL_19], FS, phys_min=-R, phys_max=R)
    out = process_recording(p)
    assert out.channel_status["dead"] == ["O2"] and out.channel_status["n_dead_min"] == 1
    assert out.channel_status["n_missing_min"] == 0 and out.channel_status["n_channels"] == 18
    prim = out.qc["primary"]
    assert prim.n_dead_min == 1 and prim.n_min_present == 9 and not prim.passes
    assert "dead_minimum_channels" in prim.reasons and "missing_minimum_channels" in prim.reasons
    assert prim.flag_fraction["flat"] == 0.0                          # NOT counted as flat data


def test_dead_optional_channel_is_dropped_but_does_not_fail_the_minimum_set(tmp_path, eeg_uv):
    dig = _with_const(eeg_uv, ["Pz", "Cz"])
    p = write_edf_raw(tmp_path / "zo.edf", dig, [f"EEG {c}-Ref" for c in CANONICAL_19], FS, phys_min=-R, phys_max=R)
    out = process_recording(p)
    prim = out.qc["primary"]
    assert sorted(out.channel_status["dead"]) == ["Cz", "Pz"] and out.channel_status["n_dead_min"] == 0
    assert prim.passes and "dead_minimum_channels" not in prim.reasons


def test_quarter_of_recordings_constant_minimum_set_summary(tmp_path, eeg_uv):
    """A cohort where a quarter of the recordings have constant minimum-set channels: the aggregate names the reason."""
    dig_ok = to_dig(eeg_uv)[:, : 90 * int(FS)]
    qcs = []
    for i in range(60):
        dig = dig_ok.copy()
        if i % 4 == 0:
            dig[CANONICAL_19.index("T4")] = zero_code()
            dig[CANONICAL_19.index("T6")] = zero_code()
        p = write_edf_raw(tmp_path / "q.edf", dig, [f"EEG {c}-Ref" for c in CANONICAL_19], FS, phys_min=-R, phys_max=R)
        qcs.append(process_recording(p, windows={"w": WindowSpec("w", 10.0, 60.0)}).qc)
    summ = summarize_window_qc(qcs)["windows"]["w"]
    assert summ["reason_counts"]["dead_minimum_channels"] == 15
    assert summ["pass_proportion"] == pytest.approx(45 / 60, abs=1e-4)


def test_drop_dead_channels_only_exact_constants():
    x = np.vstack([np.random.default_rng(0).standard_normal(500) * 10, np.zeros(500), np.full(500, 7.0),
                   np.r_[np.zeros(499), 1e-3], np.full(500, np.nan)])
    rec = drop_dead_channels(Recording(x, 100.0, ["a", "b", "c", "d", "e"]))
    assert rec.ch_names == ["a", "d"] and rec.meta["dead_channels"] == ["b", "c", "e"]


# ---- scaling ----------------------------------------------------------------------------------------
def test_zero_physical_range_is_invalid_scaling_not_flat(tmp_path, eeg_uv):
    pmin = [0.0 if c == "O1" else -R for c in CANONICAL_19]
    pmax = [0.0 if c == "O1" else R for c in CANONICAL_19]
    p = write_edf_raw(tmp_path / "p.edf", to_dig(eeg_uv), [f"EEG {c}-Ref" for c in CANONICAL_19], FS,
                      phys_min=pmin, phys_max=pmax)
    h = read_edf_header(p)
    assert channel_scaling_status(h, CANONICAL_19.index("O1")) == "physical_range_zero"
    assert channel_scaling_status(h, 0) == "ok"
    rec = read_edf(p)
    assert "O1" not in rec.ch_names and rec.meta["invalid_scaling_channels"] == ["O1"]
    out = process_recording(p)
    prim = out.qc["primary"]
    assert out.channel_status["n_invalid_min"] == 1 and out.channel_status["n_missing_min"] == 0
    assert "invalid_scaling_minimum_channels" in prim.reasons and not prim.passes
    assert prim.flag_fraction["flat"] == 0.0


def test_zero_digital_range_does_not_abort_other_channels(tmp_path, eeg_uv):
    dmin = [5 if c == "Fp1" else -32768 for c in CANONICAL_19]
    dmax = [5 if c == "Fp1" else 32767 for c in CANONICAL_19]
    p = write_edf_raw(tmp_path / "d.edf", to_dig(eeg_uv), [f"EEG {c}-Ref" for c in CANONICAL_19], FS,
                      phys_min=-R, phys_max=R, dig_min=dmin, dig_max=dmax)
    rec = read_edf(p)
    assert "Fp1" not in rec.ch_names and len(rec.ch_names) == 18 and rec.meta["invalid_scaling_channels"] == ["Fp1"]


def test_all_channels_invalid_scaling_is_its_own_failure_reason(tmp_path, eeg_uv):
    p = write_edf_raw(tmp_path / "all0.edf", to_dig(eeg_uv), [f"EEG {c}-Ref" for c in CANONICAL_19], FS,
                      phys_min=0, phys_max=0)
    with pytest.raises(EDFError, match="invalid channel scaling"):
        read_edf(p)
    r = stream_features(FakeS3({KEY: p.read_bytes()}), KEY, bucket=BUCKET, sleep=lambda s: None)
    assert r.reason == FailureReason.INVALID_SCALING and not r.retryable
    assert FailureReason.INVALID_SCALING in FailureReason.PERMANENT


# ---- EDF+ ----------------------------------------------------------------------------------------------
def test_edf_plus_c_with_annotations_reads_and_edf_plus_d_is_refused(tmp_path, eeg_uv):
    labels = [f"EEG {c}-Ref" for c in CANONICAL_19]
    short = eeg_uv[:, : 90 * int(FS)]
    pc_ = write(tmp_path, "c.edf", short, labels, annotation_channel=True, reserved="EDF+C")
    pd_ = write(tmp_path, "d.edf", short, labels, annotation_channel=True, reserved="EDF+D")
    pp = write(tmp_path, "plain.edf", short, labels)
    assert [read_edf_header(x).edf_type for x in (pc_, pd_, pp)] == ["EDF+C", "EDF+D", "EDF"]
    rec = read_edf(pc_)
    assert len(rec.ch_names) == 19 and rec.meta["edf_type"] == "EDF+C"
    assert np.allclose(rec.data, short[:, : rec.data.shape[1]], atol=0.1)
    with pytest.raises(EDFError):
        read_edf(pd_)
    r = stream_features(FakeS3({KEY: pd_.read_bytes()}), KEY, bucket=BUCKET, sleep=lambda s: None)
    assert r.reason == FailureReason.DISCONTINUOUS


def test_stream_reports_dead_and_missing_counts(tmp_path, eeg_uv):
    dig = _with_const(eeg_uv, ["Fp1"])
    labels = [f"EEG {c}-Ref" for c in CANONICAL_19 if c != "O2"]
    p = write_edf_raw(tmp_path / "s.edf", np.delete(dig, CANONICAL_19.index("O2"), axis=0), labels, FS,
                      phys_min=-R, phys_max=R)
    r = stream_features(FakeS3({KEY: p.read_bytes()}), KEY, bucket=BUCKET, sleep=lambda s: None)
    assert r.ok and (r.n_dead_min, r.n_missing_min, r.n_invalid_min) == (1, 1, 0)
    assert DEFAULT_MINIMUM_CHANNELS == ("Fp1", "Fp2", "F7", "F8", "T3", "T4", "T5", "T6", "O1", "O2")
    prim = next(x for x in r.rows if x["window"] == "primary")
    assert prim["qc_pass"] is False or prim["qc_pass"] == False  # noqa: E712


class _Body:
    def __init__(self, b):
        self._b = b

    def read(self):
        return self._b


class FakeS3:
    def __init__(self, objects):
        self.objects = objects

    def get_object(self, Bucket, Key, Range=None):
        data = self.objects[Key]
        a, b = (int(v) for v in Range[6:].split("-"))
        b = min(b, len(data) - 1)
        return {"Body": _Body(data[a:b + 1]), "ContentRange": f"bytes {a}-{b}/{len(data)}"}
