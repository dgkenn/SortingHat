"""Sentence-level EEG-content filter for reviewer packets and silver-label text evidence.

Policy (D-006, D-012): any note sentence mentioning EEG-derived content is removed
before a reviewer sees it or a rule uses it. **We err toward removal**: a false
positive costs a little context, a false negative breaks blinding / leaks the reader.
Deliberate consequences, documented in docs/labels_spec.md:

* ``slowing`` is removed in every sense ("slowing of heart rate" included).
* ``LTM`` is removed in every sense (long-term memory included); matched case-sensitively
  as an upper-case token so the lower-case "ltm" inside other strings is not hit.
* Words that merely contain the letters (free, feeling, degree, "EGG") never match:
  patterns are word-bounded.

``filter_text`` returns the kept text and the number of sentences removed. It never
returns the removed text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_CI = re.IGNORECASE

# Case-insensitive terms. Written with \b so substrings of other words never match.
_CI_TERMS = [
    # EEG itself and prefixed variants: cEEG, c-EEG, vEEG, qEEG, aEEG, cvEEG, video-EEG, ...
    r"(?:c|v|q|a|cv|ltm|video|rapid|scalp|routine|spot|prolonged|continuous)?[-\s.]?e\.?e\.?g\.?s?(?![a-z])",
    r"electro[-\s]?en[cs]e?ph[ae]l[oa]gra(?:m|ms|phy|phic|phically)\b",   # incl. misspellings
    r"electroenc[ea]ph\w*",
    r"video[-\s]?monitor\w*",
    r"long[-\s]?term\s+(?:video\s+)?(?:eeg\s+)?monitor\w*",
    r"brain[-\s]?waves?",
    r"slowing", r"slow(?:ed)?[-\s]+waves?", r"(?:theta|delta)[-\s/]+(?:slowing|activity|waves?|rhythm)",
    r"triphasic", r"tri[-\s]phasic",
    r"burst[-\s]*supp?res\w*", r"supp?res\w*[-\s]*burst\w*",
    r"epileptiform", r"epilepti[-\s]?form",
    r"spike[-\s]*(?:and[-\s]*)?(?:slow[-\s]*)?waves?", r"spike[-\s]*(?:and[-\s]*)?slow", r"sharp[-\s]*(?:and[-\s]*)?slow\w*",
    r"poly[-\s]?spikes?", r"sharp\s+waves?",
    r"periodic\s+(?:lateralized\s+|generalized\s+)?(?:epileptiform\s+)?discharges?",
    r"rhythmic\s+delta\s+activity",
    r"background\s+(?:attenuation|suppression|slowing|activity|rhythm|reactivity|continuity|discontinuity)",
    r"(?:attenuat\w+|suppress\w+|discontinuous|low[-\s]?voltage)\s+background",
    r"low[-\s]?voltage", r"electro[-\s]?cerebral\s+(?:silence|inactivity)", r"isoelectric",
    r"posterior\s+dominant\s+rhythm", r"alpha\s+coma", r"hypsarrhythmi\w*",
    r"reactivity\s+(?:on|to|of|in)\s+(?:the\s+)?(?:eeg|background)",
    r"(?:eeg|background)\s+reactiv\w*", r"reactiv\w*(?:\W+\w+){0,6}?\W+background",
    r"non[-\s]?convulsive\s+status(?:\s+epilepticus)?", 
    r"sleep\s+spindles?", r"k[-\s]?complexes?", r"interictal", r"ictal\s+pattern\w*",
    r"intermittent\s+rhythmic\s+delta", r"generali[sz]ed\s+periodic",
    # abbreviations safe to match case-insensitively (no common lower-case English homographs)
    r"lpds?", r"gpds?", r"lrda", r"grda", r"pleds?", r"gpeds?", r"bipds?", r"sirpids?",
    r"firda", r"tirda", r"eses", r"ncse", r"iic",
]

# Case-sensitive upper-case abbreviations (avoid hitting ordinary lower-case words).
# Bare "BS" is NOT matched (blood sugar / bowel sounds are far more common in notes); burst
# suppression is caught when spelled out. "BIRDs" is matched only in the CI list above.
_CS_TERMS = [r"LTM", r"PDR"]

_CI_RE = re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(_CI_TERMS) + r")(?![A-Za-z0-9])", _CI)
_CS_RE = re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(_CS_TERMS) + r")(?![A-Za-z0-9])")

# Sentence segmentation: split at . ! ? followed by space + capital/digit/bracket, and at newlines.
# Common clinical abbreviations are protected so they do not split sentences.
_ABBREV = ["Dr", "Mr", "Mrs", "Ms", "vs", "e.g", "i.e", "approx", "pt", "Pt", "No", "Fig", "etc", "q.d", "b.i.d",
           "t.i.d", "q.i.d", "s/p", "w/", "c/w", "St"]
_PROT = "\u0001"


def split_sentences(text: str) -> list[str]:
    """Split into sentences, one per list item. Lines are always sentence boundaries."""
    out: list[str] = []
    for line in re.split(r"[\r\n]+", text):
        if not line.strip():
            continue
        s = line
        for a in _ABBREV:
            s = re.sub(rf"\b({re.escape(a)})\.", rf"\1{_PROT}", s)
        s = re.sub(r"(\d)\.(\d)", rf"\1{_PROT}\2", s)           # decimals
        parts = re.split(r"(?<=[.!?;])\s+(?=[A-Z0-9\[\(\"'])", s)
        out.extend(p.replace(_PROT, ".").strip() for p in parts if p.strip())
    return out


def mentions_eeg_content(sentence: str) -> bool:
    return bool(_CI_RE.search(sentence) or _CS_RE.search(sentence))


@dataclass(frozen=True)
class FilterResult:
    text: str
    n_removed: int
    n_kept: int

    @property
    def n_total(self) -> int:
        return self.n_removed + self.n_kept


def filter_text(text: str, joiner: str = " ") -> FilterResult:
    """Drop every sentence mentioning EEG-derived content; return kept text + removed count."""
    kept: list[str] = []
    removed = 0
    for s in split_sentences(text or ""):
        if mentions_eeg_content(s):
            removed += 1
        else:
            kept.append(s)
    return FilterResult(joiner.join(kept), removed, len(kept))


def filter_notes(notes: list[str]) -> tuple[list[str], int]:
    """Filter a list of notes; returns kept texts and the total sentences removed."""
    res = [filter_text(n) for n in notes]
    return [r.text for r in res], sum(r.n_removed for r in res)
