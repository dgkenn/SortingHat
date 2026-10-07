import pytest

from sortinghat.labels.banned_evidence import (BANNED_ICD10_CM, EvidenceSource as S, classify_evidence,
                                                is_banned_icd, mentions_nonspecific_encephalopathy)


def test_enumerated_codes_all_banned_and_sensible():
    codes = {c for c, _, _ in BANNED_ICD10_CM}
    assert {"G92", "G92.8", "G92.9", "G93.40", "G93.41", "G93.49"} <= codes
    assert all(c.startswith(("G92", "G93.4")) for c in codes)
    assert all(is_banned_icd(c) for c in codes)
    assert all(d for _, d, _ in BANNED_ICD10_CM)


@pytest.mark.parametrize("c", ["G93.41", "g9341", "G92.8", "G934", "G93.49 ", "34831"])
def test_banned_variants(c):
    assert is_banned_icd(c)


@pytest.mark.parametrize("c", ["G93.1", "G93.5", "G93.6", "G40.901", "I63.9", "G9", "K72.90", "E87.1", ""])
def test_not_banned(c):
    assert not is_banned_icd(c)


def test_free_text_encephalopathy():
    assert mentions_nonspecific_encephalopathy("Toxic-metabolic encephalopathy suspected")
    assert mentions_nonspecific_encephalopathy("metabolic encephalopathy")
    assert not mentions_nonspecific_encephalopathy("Glucose 33, resolved with dextrose")


def test_classify_rules():
    assert not classify_evidence(S.EEG_REPORT, text="diffuse slowing").allowed
    assert classify_evidence(S.OBJECTIVE).allowed
    v = classify_evidence(S.ICD, code="G93.41")
    assert not v.allowed and v.category == "nonspecific_dx"
    assert classify_evidence(S.ICD, code="I61.9").allowed
    assert not classify_evidence(S.NOTE_SENTENCE, text="EEG shows triphasic waves.").allowed
    assert not classify_evidence(S.NOTE_SENTENCE, text="Toxic-metabolic encephalopathy.").allowed
    assert classify_evidence(S.NOTE_SENTENCE, text="Toxic-metabolic encephalopathy.", has_anchor=True).allowed
    assert classify_evidence(S.NOTE_SENTENCE, text="Ammonia 220, hepatic picture.").allowed


def test_post_t0_neuro_impression():
    t = "Impression: likely uremic picture."
    v = classify_evidence(S.NEURO_IMPRESSION, text=t, hours_from_t0=5)
    assert not v.allowed and v.eeg_derived
    assert not classify_evidence(S.NEURO_IMPRESSION, text=t).allowed          # untimed treated as post-t0
    assert classify_evidence(S.NEURO_IMPRESSION, text=t, hours_from_t0=5, eeg_filtered=True).allowed
    assert classify_evidence(S.NEURO_IMPRESSION, text=t, hours_from_t0=-5).allowed
    # EEG content still banned even pre-t0 unless filtered text
    assert not classify_evidence(S.NEURO_IMPRESSION, text="Plan EEG.", hours_from_t0=-5).allowed


def test_unknown_source():
    with pytest.raises(ValueError):
        classify_evidence("tarot")
