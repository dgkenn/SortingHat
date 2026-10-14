"""EEG-blinding rules for the rater packets (SYNTHETIC text only)."""
import pytest

from sortinghat.adjudication import blinding as bl
from sortinghat.labels.eeg_filter import filter_text


@pytest.mark.parametrize("text", [
    "EEG showed generalized slowing.", "cEEG monitoring started overnight.", "Triphasic waves were noted.", "Burst suppression pattern.",
    "Electroencephalogram: normal.", "BIS 45 on propofol.", "SedLine PSI 30.", "Suppression ratio 20%.", "Spectral edge frequency 12 Hz.",
    "Seen by the neurophysiology team.", "Transferred to the EMU.", "Epileptology recommends levetiracetam.", "Scalp electrodes were placed.",
    "The monitoring showed no seizures.", "SSEP absent bilaterally.", "Evoked potentials were obtained.", "Video-EEG ordered.",
])
def test_eeg_derived_sentences_are_detected(text):
    assert bl.mentions_eeg(text)


@pytest.mark.parametrize("text", [
    "Sodium 118 mmol/L on arrival.", "CT head without hemorrhage.", "Feeling better; free of pain.", "Pressure 30 psi in the cuff.",
    "Blood pressure 120/80.", "Patient is a degree-level teacher.", "Received lorazepam for agitation.", "No seizure activity witnessed by staff.",
])
def test_ordinary_clinical_sentences_are_kept(text):
    assert not bl.mentions_eeg(text)


def test_base_filter_is_a_subset_of_the_packet_filter():
    """Everything the silver labels' EEG filter drops, the packet filter drops too (it only adds)."""
    corpus = ["EEG showed slowing.", "Burst suppression.", "Epileptiform discharges.", "Normal sodium.", "LPDs noted.", "Plan: CT."]
    for s in corpus:
        base_removed = filter_text(s).n_removed
        ours_removed = bl.filter_note_text(s)[1]
        assert ours_removed >= base_removed


def test_filter_note_text_keeps_lines_and_returns_only_counts():
    text = "Found down.\nEEG ordered. BIS 40.\n\nPlan: CT head. Check sodium."
    kept, removed = bl.filter_note_text(text)
    assert removed == 2
    assert kept == "Found down.\nPlan: CT head. Check sodium."
    assert "EEG" not in kept and "BIS" not in kept


def test_id_like_tokens_are_scrubbed_from_notes():
    kept, _ = bl.filter_note_text("Seen as sub-S000112 in ses-1 and S0001000123.")
    assert "sub-S000112" not in kept and "S0001000123" not in kept and "[id]" in kept


@pytest.mark.parametrize("title,excluded", [
    ("EEG report", True), ("Neurophysiology note", True), ("EMU admission", True), ("LTM summary", True), ("Epilepsy Monitoring Unit", True),
    ("Sleep study", True), ("Progress Note", False), ("Neurology consult", False), ("Discharge summary", False),
])
def test_note_level_exclusion(title, excluded):
    assert bl.note_excluded(title, "") is excluded


@pytest.mark.parametrize("code,reason", [
    ("95816", bl.REASON_PROC), ("95951", bl.REASON_PROC), ("4A10X4Z", bl.REASON_PROC), ("89.14", bl.REASON_PROC),
    ("R94.01", bl.REASON_ABN_DX), ("794.02", bl.REASON_ABN_DX),
    ("G93.41", bl.REASON_ENCEPH), ("G92.8", bl.REASON_ENCEPH), ("348.31", bl.REASON_ENCEPH), ("349.82", bl.REASON_ENCEPH),
    ("I63.9", None), ("99291", None), ("J18.9", None), ("95905", None), ("", None),
])
def test_code_blocks(code, reason):
    assert bl.blocked_code(code) == reason


def test_blocked_row_checks_text_and_code():
    assert bl.blocked_row("Heart Rate") is None
    assert bl.blocked_row("BIS Index") == bl.REASON_TEXT
    assert bl.blocked_row("anything", code="95816") == bl.REASON_PROC
    assert bl.blocked_row(None, "ok") is None


def test_audit_counts_remaining_hits():
    assert bl.audit_text(["Sodium 120.", "Plan: CT head."]) == 0
    assert bl.audit_text(["Normal. EEG was done.", None, "BIS 40"]) == 2
