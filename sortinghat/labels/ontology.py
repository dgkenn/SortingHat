"""Study 1 label ontology (research plan v1, "Revised ontology")."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum


class Role(str, Enum):
    PRIMARY = "primary"
    PRIMARY_CONDITIONAL = "primary_conditional"      # E7: primary only if gold-positive rule is met
    POSITIVE_CONTROL = "positive_control"            # E3
    COVARIATE_SECONDARY = "covariate_secondary"      # E4b


class GoldState(IntEnum):
    """Ordinal gold states. UNASSESSABLE is not on the ordinal scale (value -1)."""
    UNASSESSABLE = -1
    ABSENT = 0
    POSSIBLE = 1
    PROBABLE = 2
    DEFINITE = 3

    @classmethod
    def parse(cls, s: "str | int | GoldState") -> "GoldState":
        if isinstance(s, cls):
            return s
        if isinstance(s, int):
            return cls(s)
        return cls[str(s).strip().upper()]

    @property
    def is_ordinal(self) -> bool:
        return self is not GoldState.UNASSESSABLE

    @property
    def is_positive(self) -> bool:
        """Binary positive = probable or definite (the 'possible' state is not positive)."""
        return self in (GoldState.PROBABLE, GoldState.DEFINITE)


GOLD_STATE_NAMES = ("absent", "possible", "probable", "definite", "unassessable")


@dataclass(frozen=True)
class Label:
    code: str
    name: str
    content: str
    role: Role
    sublabels: tuple[str, ...] = ()
    silver_banned_evidence_applies: bool = True   # circularity rules apply (E1,E2,E4a,E5,E6,E7)
    note: str = ""


LABELS: dict[str, Label] = {l.code: l for l in [
    Label("E1", "Acute structural", "Stroke, ICH, SAH, SDH/EDH, TBI, mass effect",
          Role.PRIMARY, sublabels=("focal", "diffuse")),
    Label("E2", "Global hypoxic-ischemic", "Post-arrest, asphyxia, profound shock", Role.PRIMARY),
    Label("E3", "Epileptic contributor", "Seizure, NCSE, causally relevant postictal state",
          Role.POSITIVE_CONTROL, silver_banned_evidence_applies=False,
          note="Separate EEG-based adjudication; excluded from the primary endpoint (D-002)."),
    Label("E4a", "Exogenous intoxication",
          "Overdose, drug or toxin exposure that brought the patient in or arose unintentionally",
          Role.PRIMARY),
    Label("E4b", "Iatrogenic sedation", "Planned ICU sedatives sufficient to depress the exam",
          Role.COVARIATE_SECONDARY, silver_banned_evidence_applies=False,
          note="Covariate and secondary label; excluded from the primary endpoint (D-003)."),
    Label("E5", "Metabolic/physiologic",
          "Hepatic, uremic, glucose, sodium, hypercapnia, endocrine, acid-base", Role.PRIMARY),
    Label("E6", "Systemic infection/inflammation",
          "Sepsis-associated encephalopathy without CNS infection", Role.PRIMARY),
    Label("E7", "CNS infection/inflammation",
          "Meningitis, encephalitis, autoimmune encephalitis", Role.PRIMARY_CONDITIONAL,
          note="Primary only with >=100 gold positives across >=2 sites; else exploratory (D-004)."),
]}

ALL_LABELS: tuple[str, ...] = tuple(LABELS)
SILVER_CIRCULARITY_LABELS: tuple[str, ...] = tuple(
    c for c, l in LABELS.items() if l.silver_banned_evidence_applies)       # E1,E2,E4a,E5,E6,E7
E7_MIN_GOLD_POSITIVES = 100
E7_MIN_SITES = 2

# Unconditional primary labels (E7 is added only by primary_endpoint_labels()).
CORE_PRIMARY_LABELS: tuple[str, ...] = tuple(
    c for c, l in LABELS.items() if l.role is Role.PRIMARY)                 # E1,E2,E4a,E5,E6


def e7_is_primary(gold_positives: int, n_sites_with_positives: int) -> bool:
    return gold_positives >= E7_MIN_GOLD_POSITIVES and n_sites_with_positives >= E7_MIN_SITES


def primary_endpoint_labels(e7_gold_positives: int | None = None,
                            e7_sites: int | None = None) -> tuple[str, ...]:
    """Primary-endpoint label set. Never contains E3 or E4b; E7 only if its rule is met.

    With no E7 counts supplied (e.g. before the gold set exists), E7 is excluded.
    """
    out = list(CORE_PRIMARY_LABELS)
    if e7_gold_positives is not None and e7_sites is not None \
            and e7_is_primary(e7_gold_positives, e7_sites):
        out.append("E7")
    return tuple(out)


def get_label(code: str) -> Label:
    try:
        return LABELS[code]
    except KeyError as e:
        raise KeyError(f"unknown label {code!r}; valid: {ALL_LABELS}") from e
