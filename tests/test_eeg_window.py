import json

import numpy as np
import pytest

from sortinghat.eeg.io import CANONICAL_19, DEFAULT_MINIMUM_CHANNELS
from sortinghat.eeg.synthetic import Artifact, generate_eeg
from sortinghat.eeg.window import (QCConfig, WindowSpec, all_windows, epoch_artifact_flags, nested_windows,
                                   primary_window, qc_recording, summarize_window_qc, window_clean_mask,
                                   write_qc_summary)
from sortinghat.safe_output import SUPPRESSED, assert_aggregate_only

FS = 200.0
WIN = {"w": WindowSpec("w", 10.0, 60.0)}


def test_window_definitions():
    p = primary_window()
    assert (p.start_s, p.end_s) == (60.0, 660.0)           # minutes 1-11
    n = nested_windows()
    assert {k: v.duration_s for k, v in n.items()} == {"20s": 20, "1min": 60, "2min": 120, "5min": 300, "10min": 600}
    assert all(w.start_s == p.start_s for w in n.values())  # nested from the primary window start
    assert list(all_windows())[0] == "primary"


def _qc(x, cfg=None, windows=WIN, total=None):
    return qc_recording(x, FS, CANONICAL_19, 0.0, windows, cfg, rec_duration_s=total)


@pytest.fixture(scope="module")
def clean():
    return generate_eeg(90, seed=11)


def test_clean_recording_passes(clean):
    qcs, ef = _qc(clean)
    q = qcs["w"]
    assert q.passes and q.usable_fraction == 1.0 and q.coverage_fraction == 1.0
    assert all(v == 0 for v in q.flag_fraction.values())


def _flag_cells(x, kind, channels, start=20.0, dur=10.0, amp=None):
    y = x.copy()
    arts = [Artifact(kind, tuple(channels), start, dur, amp)]
    from sortinghat.eeg.synthetic import apply_artifact
    for a in arts:
        apply_artifact(y, FS, a, CANONICAL_19, np.random.default_rng(0))
    ef = epoch_artifact_flags(y, FS, CANONICAL_19, 0.0)
    return ef


@pytest.mark.parametrize("kind", ["flat", "clipping", "extreme", "line_noise"])
def test_each_artifact_flagged_only_where_injected(clean, kind):
    ef = _flag_cells(clean, kind, ["C3"])
    row = CANONICAL_19.index("C3")
    f = ef.flags[kind]
    assert f[row, 10:15].all()                              # 20-30 s -> epochs 10..14
    assert f[row, :10].sum() == 0 and f[row, 15:].sum() == 0
    other = np.delete(f, row, axis=0)
    assert other.sum() == 0                                 # other channels untouched


def test_flat_channel_flagged_disconnected_for_the_whole_record(clean):
    y = clean.copy()
    from sortinghat.eeg.synthetic import apply_artifact
    apply_artifact(y, FS, Artifact("flat", ("T4",), 5.0, 60.0), CANONICAL_19, np.random.default_rng(0))   # 60 of 90 s
    ef = epoch_artifact_flags(y, FS, CANONICAL_19, 0.0)
    r = CANONICAL_19.index("T4")
    assert ef.flags["disconnected"][r].all()                 # >= 50% flat epochs -> persistently disconnected
    assert ef.flags["disconnected"].sum() == ef.flags["disconnected"].shape[1]


def test_line_noise_alone_never_makes_a_channel_disconnected(clean):
    """D-110: a mains-contaminated channel is still connected; only flat / clipping feed the persistent rule."""
    y = clean.copy()
    from sortinghat.eeg.synthetic import apply_artifact
    apply_artifact(y, FS, Artifact("line_noise", ("T4",), amp=600.0), CANONICAL_19, np.random.default_rng(0))
    ef = epoch_artifact_flags(y, FS, CANONICAL_19, 0.0)
    r = CANONICAL_19.index("T4")
    assert ef.flags["line_noise"][r].all() and not ef.flags["disconnected"][r].any()
    assert ef.flags["line_noise"].sum() == ef.flags["line_noise"].shape[1]       # nothing else flagged


def test_d110_thresholds():
    c = QCConfig()
    assert (c.line_ratio, c.extreme_uv, c.epoch_channel_frac, c.disconnected_epoch_frac) == (10.0, 1000.0, 0.8, 0.5)


def test_moderate_line_noise_is_not_flagged_but_strong_is(clean):
    from sortinghat.eeg.synthetic import apply_artifact
    for amp, expect in ((20.0, False), (600.0, True)):
        y = clean.copy()
        apply_artifact(y, FS, Artifact("line_noise", ("F7",), amp=amp), CANONICAL_19, np.random.default_rng(0))
        f = epoch_artifact_flags(y, FS, CANONICAL_19, 0.0).flags["line_noise"][CANONICAL_19.index("F7")]
        assert bool(f.all()) is expect and bool(f.any()) is expect


def test_extreme_threshold_is_1000uv(clean):
    assert _flag_cells(clean, "extreme", ["Cz"], amp=700.0).flags["extreme"].sum() == 0
    assert _flag_cells(clean, "extreme", ["Cz"], amp=1500.0).flags["extreme"].sum() > 0


def test_line_noisy_but_otherwise_clean_channel_leaves_epochs_usable(clean):
    from sortinghat.eeg.synthetic import apply_artifact
    y = clean.copy()
    apply_artifact(y, FS, Artifact("line_noise", ("O1",), amp=600.0), CANONICAL_19, np.random.default_rng(0))
    q = _qc(y)[0]["w"]
    assert q.passes and q.usable_fraction == 1.0 and q.disconnected_channels == 0
    assert q.flag_fraction["line_noise"] == pytest.approx(0.1) and q.flag_fraction["disconnected"] == 0.0


def test_eight_of_ten_clean_channels_make_an_epoch_usable(clean):
    from sortinghat.eeg.synthetic import apply_artifact
    two = clean.copy()
    for ch in ("F7", "O2"):
        apply_artifact(two, FS, Artifact("flat", (ch,)), CANONICAL_19)
    q2 = _qc(two)[0]["w"]
    assert q2.passes and q2.usable_fraction == 1.0           # 8 of 10 clean = 0.8
    three = two.copy()
    apply_artifact(three, FS, Artifact("flat", ("T3",)), CANONICAL_19)
    q3 = _qc(three)[0]["w"]
    assert not q3.passes and q3.usable_fraction == 0.0       # 7 of 10 clean


def test_single_bad_channel_still_usable_but_many_are_not(clean):
    from sortinghat.eeg.synthetic import apply_artifact
    one = clean.copy()
    apply_artifact(one, FS, Artifact("flat", ("F7",)), CANONICAL_19)
    q1 = _qc(one)[0]["w"]
    assert q1.passes and q1.usable_fraction == 1.0 and q1.flag_fraction["flat"] > 0
    many = clean.copy()
    for ch in ("F7", "F8", "T3", "T4"):
        apply_artifact(many, FS, Artifact("flat", (ch,), 10.0, 28.0), CANONICAL_19)   # 28 of 60 s
    q2 = _qc(many)[0]["w"]
    assert not q2.passes and "usable_below_threshold" in q2.reasons
    assert q2.usable_fraction == pytest.approx(32 / 60, abs=0.04)


def test_sixty_percent_rule_boundary(clean):
    from sortinghat.eeg.synthetic import apply_artifact
    def run(bad_s):
        y = clean.copy()
        apply_artifact(y, FS, Artifact("line_noise", CANONICAL_19, 10.0, bad_s), CANONICAL_19)
        return _qc(y)[0]["w"]
    assert run(20).passes                                    # 67% usable
    assert not run(26).passes                                # 57% usable
    assert run(20).usable_fraction == pytest.approx(40 / 60, abs=0.02)


def test_short_recording_counts_uncovered_time_as_unusable():
    x = generate_eeg(360, seed=12)
    qcs, _ = qc_recording(x, FS, CANONICAL_19, 0.0, all_windows(), rec_duration_s=360.0)
    assert qcs["primary"].coverage_fraction == pytest.approx(0.5)
    assert not qcs["primary"].passes and qcs["primary"].usable_fraction == pytest.approx(0.5, abs=0.01)
    assert not qcs["10min"].passes
    assert qcs["5min"].passes and qcs["20s"].passes
    assert len(qcs["primary"].epoch_usable) == 300            # grid is not compressed


def test_one_or_two_missing_minimum_channels_can_still_pass_but_three_fail(clean):
    """D-110: >= 8 of the 10 minimum-set electrodes present (and live) is enough; the count is a QC field."""
    def run(drop):
        keep = [i for i, c in enumerate(CANONICAL_19) if c not in drop]
        names = [CANONICAL_19[i] for i in keep]
        return qc_recording(clean[keep], FS, names, 0.0, WIN)[0]["w"]
    q0, q1, q2, q3 = run([]), run(["O2"]), run(["O2", "F7"]), run(["O2", "F7", "T4"])
    assert [q.n_missing_or_dead_min for q in (q0, q1, q2, q3)] == [0, 1, 2, 3]
    assert q0.passes and q0.reasons == []
    assert q1.passes and "missing_minimum_channels" in q1.reasons and "too_few_minimum_channels" not in q1.reasons
    assert q2.passes and q2.usable_fraction == 1.0
    assert not q3.passes and "too_few_minimum_channels" in q3.reasons and q3.n_min_present == 7


def test_midline_channels_not_required(clean):
    keep = [i for i, c in enumerate(CANONICAL_19) if c in DEFAULT_MINIMUM_CHANNELS]
    names = [CANONICAL_19[i] for i in keep]
    qcs, _ = qc_recording(clean[keep], FS, names, 0.0, WIN)
    assert qcs["w"].passes


def test_non_hairline_channels_not_required(clean):
    keep = [i for i, c in enumerate(CANONICAL_19) if c not in ("F3", "F4", "C3", "C4", "P3", "P4")]
    names = [CANONICAL_19[i] for i in keep]
    qcs, _ = qc_recording(clean[keep], FS, names, 0.0, WIN)
    assert qcs["w"].passes


def test_clean_mask_follows_flags(clean):
    from sortinghat.eeg.synthetic import apply_artifact
    y = clean.copy()
    apply_artifact(y, FS, Artifact("flat", ("O1",), 20.0, 10.0), CANONICAL_19)
    qcs, ef = _qc(y)
    m = window_clean_mask(ef, WIN["w"], CANONICAL_19)
    assert m.shape == (19, 30)
    r = CANONICAL_19.index("O1")
    assert not m[r, 5:10].any() and m[r, :5].all() and m[r, 10:].all()   # window starts at 10 s
    assert m[np.arange(19) != r].all()


def test_non_aligned_nested_windows_use_primary_start():
    x = generate_eeg(700, seed=13)
    qcs, ef = qc_recording(x, FS, CANONICAL_19, 0.0, all_windows(), rec_duration_s=700.0)
    assert ef.start_s == 60.0 and ef.n_epochs_expected == 300
    assert all(q.passes for q in qcs.values())


# --- aggregate summary --------------------------------------------------------------------------
def _fake_qcs(n, clean_ok=True):
    x = generate_eeg(80, seed=14)
    q, _ = qc_recording(x, FS, CANONICAL_19, 0.0, WIN, rec_duration_s=80.0)
    return [q] * n


def test_summary_suppresses_small_cells_and_is_aggregate_only(tmp_path):
    small = summarize_window_qc(_fake_qcs(5))
    assert small["n_recordings"] == SUPPRESSED
    assert small["windows"]["w"]["pass_proportion"] == SUPPRESSED
    assert set(small["windows"]["w"]["usable_fraction_quantiles"].values()) == {SUPPRESSED}
    big = summarize_window_qc(_fake_qcs(30))
    e = big["windows"]["w"]
    assert big["n_recordings"] == 30 and e["n"] == 30
    assert e["usable_fraction_quantiles"]["q50"] == 1.0
    assert_aggregate_only(big)
    p = write_qc_summary(tmp_path / "qc.json", _fake_qcs(30))
    assert json.loads(p.read_text())["windows"]["w"]["n"] == 30
    assert "min" not in json.dumps(big) and "max" not in json.dumps(big)


def test_qcconfig_is_configurable(clean):
    y = clean.copy()
    y[3, 1000:1400] += 120.0 * np.sin(np.linspace(0, 40, 400))
    ef = epoch_artifact_flags(y, FS, CANONICAL_19, 0.0, QCConfig(extreme_uv=100.0))
    assert ef.flags["extreme"].any()
    assert not epoch_artifact_flags(y, FS, CANONICAL_19, 0.0).flags["extreme"].any()
