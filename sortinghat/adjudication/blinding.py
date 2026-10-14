"""EEG-blinding rules for the rater packets (D-006, D-012, D-157).

The gold labels must be blinded to EEG, EEG reports and model outputs.  Three layers, all conservative (a false positive costs a little
context, a false negative breaks blinding):

1. NOTE-LEVEL: a note whose title / source value names an EEG or neurophysiology service is never loaded.
2. SENTENCE-LEVEL: every sentence of every remaining note that mentions EEG-derived content is dropped.  The base filter is
   ``sortinghat.labels.eeg_filter`` (the one the silver labels use); this module ADDS patterns for EEG-derived monitors (BIS,
   SedLine, suppression ratio, spectral edge), neurophysiology / epilepsy-monitoring vocabulary, electrodes / montage wording and
   evoked potentials, and a scrub of identifier-shaped tokens.
3. STRUCTURED-ROW BLOCKS: measurement / observation / drug / diagnosis / procedure rows whose text mentions EEG-derived content, EEG
   procedure codes, the "abnormal EEG" diagnosis codes and the nonspecific encephalopathy diagnoses (G92*, G93.4*, ICD-9 348.3x,
   349.82; D-006) are never put in a packet.

``audit_text`` is the final check run over EVERYTHING written for a rater: a packet with a remaining hit is withheld.
Residual risk (documented in docs/adjudication_pilot.md): indirect references that name no EEG word ("the monitoring showed ...",
"neurology says the pattern is ...") may survive; raters have a CRF field to report any EEG information they saw.
"""
from __future__ import annotations

import re
from typing import Iterable

from ..labels.banned_evidence import is_banned_icd, normalize_code
from ..labels.eeg_filter import mentions_eeg_content, split_sentences

_CI = re.IGNORECASE
_EXTRA_CI = [
    r"bispectral", r"sed[-\s]?line", r"patient\s+state\s+index", r"density\s+spectral", r"spectral\s+edge", r"suppression\s+ratio",
    r"neuro[-\s]?physiolog\w*", r"neuro[-\s]?diagnostic\w*", r"epilepsy\s+monitoring", r"epileptolog\w*",
    r"electrodes?", r"montages?", r"evoked\s+potentials?", r"cerebral\s+function\s+monitor\w*",
    r"(?:monitoring|recording|tracing)s?\s+(?:showed|shows|revealed|reveals|demonstrated|demonstrates|captured|confirmed|notable)",
    r"captured\s+(?:on|by|during|with)\s+(?:the\s+)?(?:monitoring|recording|video|study)",
]
_EXTRA_CS = [r"BIS", r"PSI", r"SEF\d*", r"EMU", r"SSEPs?", r"ECoG", r"cEEG", r"vEEG"]
_EXTRA_RE = re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(_EXTRA_CI) + r")(?![A-Za-z0-9])", _CI)
_EXTRA_CS_RE = re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(_EXTRA_CS) + r")(?![A-Za-z0-9])")

NOTE_META_EXCLUDE = re.compile(
    r"eeg|electro[-\s]?en[cs]e?ph|neuro[-\s]?physiolog|neuro[-\s]?diagnostic|epilepsy\s+monitor|\bEMU\b|\bLTM\b|evoked\s+potential|"
    r"\bSSEP\b|sleep\s+study|polysomno|cerebral\s+function\s+monitor", _CI)

# ---- diagnosis / procedure codes (matched on the normalised code: upper case, alphanumerics only)
_EEG_PROC = re.compile(r"^(957\d\d|958(12|13|15|16|19|21|22|24|26|27|29|30)|959(50|51|52|53|54|55|56|57|58|61|62)|4A10\w*|89(14|15|19))$")
_ABNORMAL_EEG_DX = re.compile(r"^(R940\w*|7940\d?)$")        # R94.0x abnormal CNS function study (R94.01 abnormal EEG); ICD-9 794.0x

REASON_TEXT = "eeg_text"
REASON_PROC = "eeg_procedure_code"
REASON_ABN_DX = "abnormal_eeg_dx_code"
REASON_ENCEPH = "nonspecific_encephalopathy_dx"
REASON_NOTE = "eeg_or_neurophysiology_note"

_ID_SCRUB = re.compile(r"(\bsub-[A-Za-z0-9]+|\bses-[A-Za-z0-9]+|\b[SI]\d{4}\d{6,}\b|\bSYN[SI]\d{10}\b)")


def mentions_eeg(text: str) -> bool:
    """True if ``text`` mentions EEG-derived content (base filter plus the packet-specific additions)."""
    t = str(text or "")
    return bool(mentions_eeg_content(t) or _EXTRA_RE.search(t) or _EXTRA_CS_RE.search(t))


def note_excluded(*meta: str) -> bool:
    """True if any note metadata string (title, source value, class name) names an EEG / neurophysiology note."""
    return any(NOTE_META_EXCLUDE.search(str(m)) for m in meta if m is not None and str(m) != "nan")


def scrub_ids(text: str) -> str:
    return _ID_SCRUB.sub("[id]", text)


def filter_note_text(text: str) -> tuple[str, int]:
    """(kept text, sentences removed): line structure kept, every sentence that mentions EEG-derived content dropped, id-like tokens
    scrubbed.  The removed text is never returned."""
    out_lines: list[str] = []
    removed = 0
    for line in re.split(r"[\r\n]+", str(text or "")):
        if not line.strip():
            continue
        kept = []
        for s in split_sentences(line):
            if mentions_eeg(s):
                removed += 1
            else:
                kept.append(s)
        if kept:
            out_lines.append(" ".join(kept))
    return scrub_ids("\n".join(out_lines)), removed


def blocked_code(code: str) -> str | None:
    """Reason a diagnosis / procedure code must not appear in a packet, else None."""
    c = normalize_code(code)
    if not c:
        return None
    if _EEG_PROC.match(c):
        return REASON_PROC
    if _ABNORMAL_EEG_DX.match(c):
        return REASON_ABN_DX
    if is_banned_icd(c):
        return REASON_ENCEPH
    return None


def blocked_row(*texts: str, code: str | None = None) -> str | None:
    """Reason a structured row must be dropped (any text field mentions EEG-derived content, or its code is blocked), else None."""
    if code is not None:
        r = blocked_code(code)
        if r:
            return r
    for t in texts:
        if t is not None and mentions_eeg(str(t)):
            return REASON_TEXT
    return None


def audit_text(texts: Iterable[str]) -> int:
    """Number of sentences, over all ``texts``, that still mention EEG-derived content (must be 0 before a packet is released)."""
    n = 0
    for t in texts:
        if t is None:
            continue
        for s in split_sentences(str(t)):
            n += int(mentions_eeg(s))
    return n
