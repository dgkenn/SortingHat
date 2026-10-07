"""Signal-onset search (D-108): files with leading constant padding / mid-file gaps, on SYNTHETIC EDFs only."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sortinghat.eeg import stream
from sortinghat.eeg.io import CANONICAL_19, DEFAULT_MINIMUM_CHANNELS
from sortinghat.eeg.stream import FailureReason, find_signal_onset, onset_offset_s, stream_features
from sortinghat.eeg.synthetic import write_edf_raw
from sortinghat.eeg.window import WindowSpec

FS, R = 64.0, 3276.7
KEY = "EEG/bids/SYN/sub-SYN1/ses-1/eeg/sub-SYN1_ses-1_task-EEG_eeg.edf"
BUCKET = "fake-ap"
LABELS = [f"EEG {c}-Ref" for c in CANONICAL_19]
PAD_CODE = 1234                                   # a constant, NON-zero digital pad


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

    @property
    def bytes_served(self):
        return sum(b - a + 1 for a, b in self.calls)


def _signal(n_s, seed=0):
    g = 2 * R / 65535.0
    x = np.random.default_rng(seed).standard_normal((19, int(n_s * FS))) * 20.0
    return np.clip(np.round((x + R) / g - 32768), -32768, 32767).astype("<i2")


def build(tmp_path, segments, name="f.edf", labels=LABELS, active=None):
    """segments: list of ('pad'|'sig', seconds). ``active``: indices of channels that carry signal (others constant)."""
    parts = []
    for i, (kind, sec) in enumerate(segments):
        if kind == "pad":
            parts.append(np.full((19, int(sec * FS)), PAD_CODE, "<i2"))
        else:
            sig = _signal(sec, seed=i)
            if active is not None:
                for ch in range(19):
                    if ch not in active:
                        sig[ch] = PAD_CODE
            parts.append(sig)
    dig = np.concatenate(parts, axis=1)
    return write_edf_raw(tmp_path / name, dig, labels, FS, phys_min=-R, phys_max=R).read_bytes()


def onset(blob, **kw):
    return find_signal_onset(FakeS3({KEY: blob}), KEY, bucket=BUCKET, **kw)


def test_onset_after_leading_padding(tmp_path):
    r = onset(build(tmp_path, [("pad", 300), ("sig", 150)]))
    assert r.onset_s == 300.0 and r.reason is None
    assert onset(build(tmp_path, [("sig", 100)])).onset_s == 0.0
    assert onset(build(tmp_path, [("pad", 305), ("sig", 100)])).onset_s == 300.0     # block containing the transition
    assert onset(build(tmp_path, [("pad", 70), ("sig", 100)])).onset_s == 70.0       # between two coarse probes
    assert onset(build(tmp_path, [("pad", 125), ("sig", 100)])).onset_s == 120.0


def test_no_onset_when_always_constant_or_beyond_limit(tmp_path):
    r = onset(build(tmp_path, [("pad", 200)]))
    assert r.onset_s is None and r.reason == FailureReason.NO_SIGNAL_ONSET
    r = onset(build(tmp_path, [("pad", 400), ("sig", 100)]), max_search_s=300)
    assert r.onset_s is None and r.reason == FailureReason.NO_SIGNAL_ONSET
    assert onset(build(tmp_path, [("pad", 400), ("sig", 100)]), max_search_s=600).onset_s == 400.0
    assert FailureReason.NO_SIGNAL_ONSET in FailureReason.PERMANENT


def test_needs_eight_of_ten_required_electrodes(tmp_path):
    required_idx = [CANONICAL_19.index(c) for c in DEFAULT_MINIMUM_CHANNELS]
    optional_idx = [i for i in range(19) if i not in required_idx]
    # 7 required + all 9 optional active for 100 s: not enough; then 8 required: onset
    seven = optional_idx + required_idx[:7]
    eight = optional_idx + required_idx[:8]
    blob = build(tmp_path, [("sig", 100)], active=seven)
    assert onset(blob).onset_s is None
    p = tmp_path
    blob2 = np.concatenate([_signal(100)[:], _signal(100, 3)], axis=1)
    for ch in range(19):                                              # first 100 s: only 7 required channels active
        if ch not in seven:
            blob2[ch, : int(100 * FS)] = PAD_CODE
        if ch not in eight:
            blob2[ch, int(100 * FS):] = PAD_CODE
    b = write_edf_raw(p / "g.edf", blob2, LABELS, FS, phys_min=-R, phys_max=R).read_bytes()
    assert onset(b).onset_s == 100.0


def test_mid_file_gap_does_not_move_the_onset(tmp_path):
    blob = build(tmp_path, [("pad", 200), ("sig", 100), ("pad", 100), ("sig", 100)])
    assert onset(blob).onset_s == 200.0


def test_few_bytes_are_fetched_for_long_padding(tmp_path):
    blob = build(tmp_path, [("pad", 1800), ("sig", 30)])
    s3 = FakeS3({KEY: blob})
    r = find_signal_onset(s3, KEY, bucket=BUCKET)
    assert r.onset_s == 1800.0
    assert s3.bytes_served < 0.25 * len(blob)                         # ~1/6 probes + header + one scan


def test_onset_offset_helper_never_raises(tmp_path):
    blob = build(tmp_path, [("pad", 100), ("sig", 100)])
    assert onset_offset_s(FakeS3({KEY: blob}), KEY, bucket=BUCKET) == 100.0
    assert onset_offset_s(FakeS3({KEY: blob}), "missing", bucket=BUCKET) is None
    assert onset_offset_s(FakeS3({KEY: b"junk" * 500}), KEY, bucket=BUCKET) is None


def test_bipolar_only_file_is_no_eeg_channels(tmp_path):
    chain = ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1", "Fp2-F8", "F8-T4", "T4-T6", "T6-O2"]
    dig = _signal(100)[:8]
    blob = write_edf_raw(tmp_path / "b.edf", dig, chain, FS, phys_min=-R, phys_max=R).read_bytes()
    with pytest.raises(stream._StreamError) as ei:
        find_signal_onset(FakeS3({KEY: blob}), KEY, bucket=BUCKET)
    assert ei.value.reason == FailureReason.NO_EEG_CHANNELS


SHORT = {"w": WindowSpec("w", 20.0, 60.0)}


def test_stream_places_windows_relative_to_onset(tmp_path):
    blob = build(tmp_path, [("pad", 150), ("sig", 200)])
    s3 = FakeS3({KEY: blob})
    r = stream_features(s3, KEY, bucket=BUCKET, windows=SHORT, sleep=lambda s: None)
    assert r.ok and r.onset_s == 150.0
    assert r.rows[0]["qc_pass"] and r.rows[0]["usable_fraction"] > 0.95
    assert r.bytes_fetched == s3.bytes_served
    old = stream_features(FakeS3({KEY: blob}), KEY, bucket=BUCKET, windows=SHORT, onset_search=False,
                          sleep=lambda s: None)
    assert old.ok and old.onset_s == 0.0 and not old.rows[0]["qc_pass"]      # the file-start window is all padding
    assert old.rows[0]["usable_fraction"] == 0.0


def test_stream_no_onset_is_a_permanent_reason(tmp_path):
    r = stream_features(FakeS3({KEY: build(tmp_path, [("pad", 200)])}), KEY, bucket=BUCKET, windows=SHORT,
                        sleep=lambda s: None)
    assert (not r.ok) and r.reason == FailureReason.NO_SIGNAL_ONSET and not r.retryable


def test_stream_window_past_end_after_onset_is_handled(tmp_path):
    blob = build(tmp_path, [("pad", 100), ("sig", 50)])
    r = stream_features(FakeS3({KEY: blob}), KEY, bucket=BUCKET, windows=SHORT, sleep=lambda s: None)
    assert r.ok and r.onset_s == 100.0 and not r.rows[0]["qc_pass"]          # only 50 s after onset: window exceeds


def test_script_stores_onset_and_prints_quantiles_only(tmp_path, monkeypatch, capsys):
    import sortinghat.data_io as dio
    monkeypatch.setattr(dio, "access_point", lambda name="credentialed": BUCKET)
    spec = importlib.util.spec_from_file_location(
        "xef", Path(__file__).resolve().parents[1] / "scripts" / "extract_eeg_features.py")
    xef = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(xef)
    blob = build(tmp_path, [("pad", 120), ("sig", 120)])
    keys = [f"EEG/bids/SYN/sub-SYN{i:03d}/ses-1/eeg/sub-SYN{i:03d}_ses-1_task-EEG_eeg.edf" for i in range(12)]
    lo = tmp_path / "local_only"
    lo.mkdir()
    pd.DataFrame({"edf_key": keys}).to_csv(lo / "k.csv", index=False)
    rc = xef.main(["--input", str(lo / "k.csv"), "--out-dir", str(lo / "f"), "--windows", "20s"],
                  s3=FakeS3({k: blob for k in keys}))
    out = capsys.readouterr().out
    assert rc == 0 and "signal onset offset" in out and "sub-" not in out
    df = pd.concat([pd.read_parquet(p) for p in (lo / "f").glob("part-*.parquet")])
    assert (df["onset_offset_s"] == 120.0).all()
