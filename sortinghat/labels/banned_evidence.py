"""Evidence that may never count as POSITIVE silver-label evidence (D-006..D-008).

Applies to labels E1, E2, E4a, E5, E6, E7 (see ``ontology.SILVER_CIRCULARITY_LABELS``).

Banned:
1. EEG reports, and any note sentence mentioning EEG-derived content (``eeg_filter``).
2. Nonspecific encephalopathy diagnoses: ICD-10 G92.* and G93.4* (enumerated below), and
   free-text "toxic-metabolic / metabolic / toxic / unspecified encephalopathy" without an
   objective anchor.
3. Post-t0 neurology impressions, unless the sentence filter has removed EEG content.

ICD-10-CM code list: enumerated from the ICD-10 MCP server (FY2027 code set) on 2026-10-07 with
``search_codes(query=<prefix>, search_by="code", valid_for_hipaa_only=false)``. The family rule
bans the whole prefix, including entries that are not really "nonspecific" (G92.0x ICANS,
G93.42-G93.45 leukoencephalopathies/DEE): a prefix rule cannot be eroded by recoding, and these
codes are not independent anchors for E1/E2/E4a/E5/E6/E7 either. Re-enumerate each October.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .eeg_filter import mentions_eeg_content

ENUMERATED_ON = "2026-10-07"
ENUMERATED_FROM = "ICD-10-CM FY2027 via ICD-10 MCP search_codes (prefix G92, G93.4)"

# (code, description, billable/HIPAA-valid leaf?)
BANNED_ICD10_CM: tuple[tuple[str, str, bool], ...] = (
    ("G92", "Toxic encephalopathy", False),
    ("G92.0", "Immune effector cell-associated neurotoxicity syndrome", False),
    ("G92.00", "Immune effector cell-associated neurotoxicity syndrome, grade unspecified", True),
    ("G92.01", "Immune effector cell-associated neurotoxicity syndrome, grade 1", True),
    ("G92.02", "Immune effector cell-associated neurotoxicity syndrome, grade 2", True),
    ("G92.03", "Immune effector cell-associated neurotoxicity syndrome, grade 3", True),
    ("G92.04", "Immune effector cell-associated neurotoxicity syndrome, grade 4", True),
    ("G92.05", "Immune effector cell-associated neurotoxicity syndrome, grade 5", True),
    ("G92.8", "Other toxic encephalopathy", True),
    ("G92.9", "Unspecified toxic encephalopathy", True),
    ("G93.4", "Other and unspecified encephalopathy", False),
    ("G93.40", "Encephalopathy, unspecified", True),
    ("G93.41", "Metabolic encephalopathy", True),
    ("G93.42", "Megalencephalic leukoencephalopathy with subcortical cysts", True),
    ("G93.43", "Leukoencephalopathy with calcifications and cysts", True),
    ("G93.44", "Adult-onset leukodystrophy with axonal spheroids", True),
    ("G93.45", "Developmental and epileptic encephalopathy", True),
    ("G93.49", "Other encephalopathy", True),
)

# Family prefixes (dots removed, upper case). Matching is by prefix so codes added in later
# fiscal years under G92/G93.4 are banned automatically.
BANNED_ICD10_PREFIXES: tuple[str, ...] = ("G92", "G934")

# Legacy ICD-9-CM equivalents (HEEDB condition_source_value mixes ICD-9 and ICD-10).
# Verified as correct by the project lead 2026-10-07 (348.30, 348.31, 348.39, 349.82); not tool-enumerated.
BANNED_ICD9_PREFIXES: tuple[str, ...] = ("34830", "34831", "34839", "34982")   # 348.30/.31/.39, 349.82
ICD9_NOTE = ("VERIFIED by project lead 2026-10-07: 348.30 encephalopathy NOS, 348.31 metabolic, "
             "348.39 other, 349.82 toxic")


def normalize_code(code: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(code or "")).upper()


def is_banned_icd(code: str) -> bool:
    c = normalize_code(code)
    return c.startswith(BANNED_ICD10_PREFIXES) or c.startswith(BANNED_ICD9_PREFIXES)


# Free-text nonspecific encephalopathy (needs an objective anchor to mean anything).
_NONSPECIFIC_ENCEPH_RE = re.compile(
    r"\b(?:toxic[-\s/]*metabolic|toxic|metabolic|tox[-\s/]*met|unspecified|acute|septic|sepsis[-\s]associated)"
    r"\s+(?:en[ck]ephalopath\w*)|\bTME\b|\bSAE\b|\ben[ck]ephalopath\w*\b", re.IGNORECASE)


def mentions_nonspecific_encephalopathy(text: str) -> bool:
    return bool(_NONSPECIFIC_ENCEPH_RE.search(text or ""))


class EvidenceSource:
    EEG_REPORT = "eeg_report"
    NOTE_SENTENCE = "note_sentence"
    ICD = "icd"
    NEURO_IMPRESSION = "neuro_impression"
    OBJECTIVE = "objective"            # lab / imaging / culture / arrest / tox / antidote


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    category: str          # eeg_derived | nonspecific_dx | neuro_impression_unfiltered | allowed
    reason: str

    @property
    def eeg_derived(self) -> bool:
        return self.category in ("eeg_derived", "neuro_impression_unfiltered")


def classify_evidence(source: str, *, text: str = "", code: str = "",
                      hours_from_t0: Optional[float] = None, eeg_filtered: bool = False,
                      has_anchor: bool = False) -> Verdict:
    """Decide whether one piece of evidence may count as positive evidence.

    ``eeg_filtered`` means the text passed through ``eeg_filter.filter_text`` (so any
    EEG-mentioning sentence is already gone). ``has_anchor`` is whether the same case/label
    carries an objective anchor (``anchors.py``); only the free-text encephalopathy rule uses it.
    """
    if source == EvidenceSource.EEG_REPORT:
        return Verdict(False, "eeg_derived", "EEG report is banned (D-006)")
    if source == EvidenceSource.OBJECTIVE:
        return Verdict(True, "allowed", "objective anchor")
    if source == EvidenceSource.ICD:
        if is_banned_icd(code):
            return Verdict(False, "nonspecific_dx", "G92/G93.4 family banned (D-007)")
        return Verdict(True, "allowed", "ICD code outside banned families (still needs an anchor)")
    if source in (EvidenceSource.NOTE_SENTENCE, EvidenceSource.NEURO_IMPRESSION):
        if not eeg_filtered and mentions_eeg_content(text):
            return Verdict(False, "eeg_derived", "sentence mentions EEG-derived content (D-006)")
        if source == EvidenceSource.NEURO_IMPRESSION and (hours_from_t0 is None or hours_from_t0 > 0) \
                and not eeg_filtered:
            return Verdict(False, "neuro_impression_unfiltered",
                           "post-t0 (or untimed) neurology impression not EEG-filtered (D-008)")
        if mentions_nonspecific_encephalopathy(text) and not has_anchor:
            return Verdict(False, "nonspecific_dx", "free-text encephalopathy without anchor (D-007)")
        return Verdict(True, "allowed", "note text after filtering")
    raise ValueError(f"unknown evidence source {source!r}")
