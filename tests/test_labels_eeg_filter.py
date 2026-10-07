import pytest

from sortinghat.labels.eeg_filter import filter_notes, filter_text, mentions_eeg_content, split_sentences

TRUE_POS = [
    "EEG showed generalized slowing.", "cEEG was started overnight.", "c-EEG ordered.", "vEEG monitoring ongoing.",
    "Video-EEG shows no seizures.", "LTM initiated on arrival.", "Triphasic waves noted.", "tri-phasic morphology",
    "Burst suppression pattern on propofol.", "burst-suppression", "burst supression", "Findings: LPDs in right hemisphere.",
    "GPDs at 1.5 Hz.", "LRDA present.", "PLEDs noted.", "GRDA frontally.", "Epileptiform discharges seen.",
    "Spike-and-wave complexes.", "spike and wave", "Background attenuation noted.", "Reactivity on EEG was absent.",
    "Electroencephalogram was abnormal.", "electroencephelogram ordered", "Brain waves slow.", "Rule out NCSE.",
    "Low voltage background.", "Posterior dominant rhythm 7 Hz.", "Interictal spikes.", "Alpha coma pattern.",
    "Reactivity to stimulation on the background was preserved.", "qEEG trend reviewed.", "EEGs were reviewed.",
]
# Documented over-removal: err toward removal.
OVER_REMOVED = [
    "Heart rate slowing to 48 after metoprolol.", "Slowing of heart rate noted.", "LTM: long-term memory intact.",
]
FALSE_POS_TRAPS = [   # must be KEPT
    "Patient is free of pain.", "Feeling better, degree of sedation low.", "Eggs and toast for breakfast.",
    "Seizure-like activity witnessed; postictal on arrival.", "Delta-9 THC screen positive.", "BS 38 on fingerstick.",
    "Background of diabetes and CKD.", "Free water deficit 4 L.", "Intubated for airway protection.",
    "Glucose 33 mg/dL, resolved with dextrose.", "Pupils 3 mm reactive.", "Head CT shows right MCA infarct.",
]


@pytest.mark.parametrize("s", TRUE_POS)
def test_true_positives_removed(s):
    assert mentions_eeg_content(s), s


@pytest.mark.parametrize("s", OVER_REMOVED)
def test_documented_over_removal(s):
    assert mentions_eeg_content(s), s


@pytest.mark.parametrize("s", FALSE_POS_TRAPS)
def test_false_positive_traps_kept(s):
    assert not mentions_eeg_content(s), s


def test_filter_text_counts_and_never_returns_removed():
    txt = "Pt found down. EEG showed burst suppression. GCS 6. Dr. Lee saw the patient; LTM started.\nHead CT negative."
    r = filter_text(txt)
    assert r.n_removed == 2
    assert "EEG" not in r.text and "LTM" not in r.text and "suppression" not in r.text
    assert "GCS 6" in r.text and "Head CT negative" in r.text and "Dr. Lee saw the patient;" in r.text
    assert r.n_total == r.n_removed + r.n_kept


def test_abbreviation_and_decimal_do_not_split():
    s = split_sentences("Dr. Kim saw pt. Na 119.5 mmol/L. Seizure ruled out.")
    assert any("Na 119.5" in x for x in s)
    assert not any(x == "Dr." for x in s)


def test_semicolon_clause_isolation():
    r = filter_text("Glucose 40; EEG with slowing. Given dextrose.")
    assert "Glucose 40" in r.text and "Given dextrose" in r.text and r.n_removed == 1


def test_empty_and_none():
    assert filter_text("").n_removed == 0 and filter_text(None).text == ""
    texts, n = filter_notes(["EEG abnormal.", "CT clear."])
    assert n == 1 and texts == ["", "CT clear."]
