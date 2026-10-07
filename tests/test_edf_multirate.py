"""Heterogeneous samples-per-record EDFs (EEG 256 Hz next to ECG 512 Hz, osat 1/record, DC 8 Hz, an EDF+ annotation
signal) must decode IDENTICALLY by (1) full-file decode, (2) our ranged-window decode, (3) pyedflib. SYNTHETIC only."""
import numpy as np
import pytest

from sortinghat.eeg import stream
from sortinghat.eeg.io import CANONICAL_19, _uv_scale, decode_signal_uv, read_edf, read_edf_header
from sortinghat.eeg.synthetic import SigSpec, write_edf_multirate

pyedflib = pytest.importorskip("pyedflib")
BUCKET, KEY = "fake-ap", "EEG/bids/SYN/sub-SYN1/ses-1/eeg/sub-SYN1_ses-1_task-EEG_eeg.edf"
SECONDS = 80
R = 3276.7


class _Body:
    def __init__(self, b):
        self._b = b

    def read(self, *a):
        return self._b


class FakeS3:
    def __init__(self, objects):
        self.objects, self.calls = objects, []

    def get_object(self, Bucket, Key, Range=None):
        data = self.objects[Key]
        a, b = (int(v) for v in Range[6:].split("-"))
        b = min(b, len(data) - 1)
        self.calls.append((a, b))
        return {"Body": _Body(data[a:b + 1]), "ContentRange": f"bytes {a}-{b}/{len(data)}"}


def _eeg_dig(n, seed):
    g = 2 * R / 65535.0
    x = np.random.default_rng(seed).standard_normal(n) * 20.0
    return np.clip(np.round((x + R) / g - 32768), -32768, 32767).astype("<i2")


def build(tmp_path, rd, order="mixed", reserved=""):
    n_rec = int(SECONDS / rd)
    rng = np.random.default_rng(1)
    eeg = [SigSpec(f"EEG {c}-Ref", int(256 * rd), _eeg_dig(n_rec * int(256 * rd), i)) for i, c in enumerate(CANONICAL_19)]
    ecg = SigSpec("ECG", int(512 * rd), (np.sin(np.arange(n_rec * int(512 * rd)) / 7.0) * 20000).astype("<i2"),
                  pmin=-5, pmax=5, dim="mV")
    osat = SigSpec("osat", 1, np.full(n_rec, 97, "<i2"), pmin=0, pmax=100, dmin=0, dmax=100)
    dcs = [SigSpec(f"DC{k}", int(8 * rd), rng.integers(-3000, 3000, n_rec * int(8 * rd)).astype("<i2") * 0 + 1000 * k,
                   pmin=-10000, pmax=10000, dim="uV") for k in range(2, 5)]
    ann = SigSpec("EDF Annotations", 30, annotation=True)
    if order == "mixed":                          # slow / fast signals interleaved BEFORE, between and after the EEG
        sigs = [osat, dcs[0], *eeg[:3], ecg, *eeg[3:12], ann, dcs[1], *eeg[12:], dcs[2]]
    else:
        sigs = [*eeg, ecg, osat, *dcs, ann]
    return write_edf_multirate(tmp_path / f"m{rd}{order}{reserved}.edf", sigs, n_rec, rd, reserved), sigs


@pytest.mark.parametrize("rd", [1.0, 0.5])
@pytest.mark.parametrize("order", ["mixed", "eeg_first"])
def test_full_decode_ranged_decode_and_pyedflib_agree_exactly(tmp_path, rd, order):
    p, sigs = build(tmp_path, rd, order)
    h = read_edf_header(p)
    assert list(h.samples_per_record) == [s.spr for s in sigs]
    assert h.record_bytes == 2 * sum(s.spr for s in sigs)                      # stride = sum over ALL signals
    assert p.stat().st_size == h.header_bytes + h.n_records * h.record_bytes
    assert [i for i, a in enumerate(h.is_annotation) if a] == [i for i, s in enumerate(sigs) if s.annotation]
    blob = p.read_bytes()
    recs = np.frombuffer(blob[h.header_bytes:], dtype="<i2").reshape(h.n_records, -1)

    with pyedflib.EdfReader(str(p)) as ref:
        for i, s in enumerate(sigs):
            if s.annotation:
                continue
            ours = decode_signal_uv(h, recs, i)                                 # every signal, incl. ECG / DC / osat
            theirs = ref.readSignal(i) * _uv_scale(s.dim)
            assert ours.shape == theirs.shape and np.allclose(ours, theirs, rtol=0, atol=1e-9), s.label
            assert ref.getSampleFrequency(i) == pytest.approx(s.spr / rd)

        # 19-channel EEG: full file, then a window by our file reader, then by the ranged (streamed) reader
        idx = {s.label: i for i, s in enumerate(sigs)}
        full = read_edf(p, channels=list(CANONICAL_19))
        assert full.ch_names == list(CANONICAL_19) and full.fs == 256
        start, dur = 7.25, 11.5
        part = read_edf(p, start_s=start, duration_s=dur, channels=list(CANONICAL_19))
        streamed = stream.fetch_window(FakeS3({KEY: blob}), KEY, start, dur, bucket=BUCKET)
        a, n = int(round(start * 256)), int(round(dur * 256))
        for r, c in enumerate(CANONICAL_19):
            want = ref.readSignal(idx[f"EEG {c}-Ref"])
            assert np.allclose(full.data[r], want, rtol=0, atol=1e-9)
            assert np.allclose(part.data[r], want[a:a + n], rtol=0, atol=1e-9)
            assert np.array_equal(streamed.data[r], part.data[r])
    assert part.offset_s == start and streamed.offset_s == start


def test_stream_features_style_window_uses_only_the_needed_records(tmp_path):
    p, sigs = build(tmp_path, 0.5)
    h = read_edf_header(p)
    s3 = FakeS3({KEY: p.read_bytes()})
    stream.fetch_window(s3, KEY, 20.0, 30.0, bucket=BUCKET)
    data = [(a, b) for a, b in s3.calls if a != 0]
    assert min(a for a, _ in data) == h.header_bytes + 40 * h.record_bytes             # record 40 = 20 s / 0.5 s
    assert max(b for _, b in data) == h.header_bytes + 100 * h.record_bytes - 1        # up to record 99


def test_edf_plus_c_flag_with_annotations_decodes_same(tmp_path):
    p, _ = build(tmp_path, 1.0, reserved="EDF+C")
    q, _ = build(tmp_path, 1.0, reserved="")
    a = read_edf(p, channels=list(CANONICAL_19))
    b = read_edf(q, channels=list(CANONICAL_19))
    assert a.meta["edf_type"] == "EDF+C" and np.array_equal(a.data, b.data)


def test_scaling_is_per_signal(tmp_path):
    p, sigs = build(tmp_path, 1.0)
    h = read_edf_header(p)
    recs = np.frombuffer(p.read_bytes()[h.header_bytes:], dtype="<i2").reshape(h.n_records, -1)
    i = [s.label for s in sigs].index("ECG")
    ecg = decode_signal_uv(h, recs, i)
    assert np.abs(ecg).max() <= 5000.0 + 1e-6 and np.abs(ecg).max() > 3000            # mV -> uV with its OWN pmin/pmax
    dc = decode_signal_uv(h, recs, [s.label for s in sigs].index("DC3"))
    assert np.allclose(dc, dc[0]) and abs(dc[0]) > 100                               # a constant NON-zero DC level
