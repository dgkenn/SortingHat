import numpy as np
import pytest

from sortinghat.eeg.io import (CANONICAL_19, DEFAULT_MINIMUM_CHANNELS, EDFError, Recording, check_channel_set,
                               normalize_channel_name, read_edf, read_edf_header, select_channels)
from sortinghat.eeg.synthetic import generate_eeg, write_edf


@pytest.mark.parametrize("raw,canon", [
    ("EEG Fp1-REF", "Fp1"), ("EEG FP1-LE", "Fp1"), ("fp1", "Fp1"), ("EEG T7-REF", "T3"), ("T8", "T4"),
    ("EEG P7-LE", "T5"), ("P8", "T6"), ("EEG Cz-REF", "Cz"), ("EEG FZ-AV", "Fz"), ("O1-Ref", "O1"),
    ("T3", "T3"), ("T4", "T4"), ("T5", "T5"), ("T6", "T6"), (" EEG  Pz - REF ", "Pz"),
])
def test_normalize_channel_name(raw, canon):
    assert normalize_channel_name(raw) == canon


@pytest.mark.parametrize("raw", ["ECG", "EEG EKG-REF", "Fp1-F7", "EDF Annotations", "EOG L", "", "Photic", "A1"])
def test_non_eeg_labels_rejected(raw):
    assert normalize_channel_name(raw) is None


def test_check_channel_set():
    raw = [f"EEG {c}-REF" for c in CANONICAL_19 if c != "Pz"] + ["ECG"]
    chk = check_channel_set(raw)
    assert not chk.has_19 and chk.missing == ["Pz"] and chk.extra == ["ECG"]
    assert chk.has_minimum            # Pz is not in the minimum set
    assert DEFAULT_MINIMUM_CHANNELS == ("Fp1", "Fp2", "F7", "F8", "T3", "T4", "T5", "T6", "O1", "O2")
    chk2 = check_channel_set(CANONICAL_19)
    assert chk2.has_19 and chk2.has_minimum and len(DEFAULT_MINIMUM_CHANNELS) == 10
    chk3 = check_channel_set(["Fp1", "Fp2"])
    assert not chk3.has_minimum and "O2" in chk3.missing_minimum


def test_roundtrip_uv_and_names(tmp_path):
    x = generate_eeg(20, background="normal", seed=3)
    p = write_edf(tmp_path / "a.edf", x, 200, ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4",
                                                "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"])
    rec = read_edf(p)
    assert rec.ch_names == list(CANONICAL_19)          # T7/T8/P7/P8 normalised
    assert rec.fs == 200 and rec.data.shape == x.shape
    assert np.abs(rec.data - x).max() < 0.06           # 0.1 uV quantisation / 2
    h = read_edf_header(p)
    assert h.n_records == 20 and h.record_duration == 1.0


@pytest.mark.parametrize("dim", ["mV", "V"])
def test_units_converted_to_microvolts(tmp_path, dim):
    x = generate_eeg(10, seed=4)
    p = write_edf(tmp_path / f"{dim}.edf", x, 200, CANONICAL_19, phys_dim=dim, phys_range_uv=3000.0)
    rec = read_edf(p)
    assert np.abs(rec.data - x).max() < 0.06
    assert 5 < rec.data.std() < 50                      # uV scale, not volts


def test_annotation_channel_skipped_and_nonstandard_dropped(tmp_path):
    x = generate_eeg(8, seed=5)
    p = write_edf(tmp_path / "b.edf", x, 200, CANONICAL_19, annotation_channel=True)
    rec = read_edf(p)
    assert len(rec.ch_names) == 19
    rec2 = read_edf(p, channels=["O1", "O2"])
    assert rec2.ch_names == ["O1", "O2"] and np.allclose(rec2.data, rec.data[[17, 18]])


def test_partial_read_matches_slice(tmp_path):
    x = generate_eeg(30, seed=6)
    p = write_edf(tmp_path / "c.edf", x, 200, CANONICAL_19)
    full = read_edf(p)
    part = read_edf(p, start_s=7, duration_s=11)
    assert part.offset_s == 7 and part.data.shape[1] == 11 * 200
    assert np.allclose(part.data, full.data[:, 7 * 200: 18 * 200])
    tail = read_edf(p, start_s=25, duration_s=100)         # runs past the end
    assert tail.data.shape[1] == 5 * 200


def test_read_from_file_object(tmp_path):
    x = generate_eeg(10, seed=7)
    p = write_edf(tmp_path / "d.edf", x, 200, CANONICAL_19)
    with open(p, "rb") as fh:
        rec = read_edf(fh, start_s=2, duration_s=3)
    assert rec.data.shape == (19, 600)


def test_discontinuous_edf_refused(tmp_path):
    x = generate_eeg(6, seed=8)
    p = write_edf(tmp_path / "e.edf", x, 200, CANONICAL_19, reserved="EDF+D")
    with pytest.raises(EDFError):
        read_edf(p)
    assert read_edf(p, allow_discontinuous=True).meta["discontinuous"]


def test_bad_file_raises(tmp_path):
    p = tmp_path / "bad.edf"
    p.write_bytes(b"x" * 500)
    with pytest.raises(EDFError):
        read_edf(p)


def test_select_channels_order_and_fill():
    x = np.arange(3 * 10, dtype=float).reshape(3, 10)
    rec = Recording(x, 200.0, ["O1", "Fp1", "ECG"])
    s = select_channels(rec)
    assert s.ch_names == ["Fp1", "O1"] and s.meta["missing_channels"][0] == "Fp2"
    f = select_channels(rec, fill_missing=True)
    assert f.data.shape[0] == 19 and np.isnan(f.data[1]).all() and np.allclose(f.data[0], x[1])


def test_matches_pyedflib(tmp_path):
    pyedflib = pytest.importorskip("pyedflib")
    x = generate_eeg(10, seed=9)
    p = write_edf(tmp_path / "f.edf", x, 200, CANONICAL_19)
    with pyedflib.EdfReader(str(p)) as r:
        ref = np.vstack([r.readSignal(i) for i in range(r.signals_in_file)])
    assert np.allclose(read_edf(p).data, ref, atol=1e-6)
