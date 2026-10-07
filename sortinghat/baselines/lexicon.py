"""Vocabularies for the clinical baselines: drug ingredients, measurement-name classes, assay-lag table.

Everything here is a *mapping rule*, not data. The real OMOP ``drug_exposure`` / ``measurement`` rows carry free
text (``drug_source_value``, ``measurement_source_value``), so classes are derived by name matching (the same
approach as ``sortinghat.audit.field_audit``). All rules are provisional until a human checks them against the
real vocabularies (``docs/baselines_spec.md``, "Items needing a human").
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# --------------------------------------------------------------------------------------------------- drugs
# RxNorm *ingredient* name -> (class, regex of synonyms / brand names). Matching is on word boundaries over
# ``drug_source_value`` (+ the OMOP concept_name of ``drug_concept_id`` when the concept table resolves it).
SEDATIVE, OPIOID = "sedative", "opioid"
_DRUG_RULES: tuple[tuple[str, str, str], ...] = (
    ("propofol", SEDATIVE, r"propofol|diprivan"),
    ("midazolam", SEDATIVE, r"midazolam|versed"),
    ("lorazepam", SEDATIVE, r"lorazepam|ativan"),
    ("dexmedetomidine", SEDATIVE, r"dexmedetomidine|precedex"),
    ("ketamine", SEDATIVE, r"ketamine|ketalar"),
    ("diazepam", SEDATIVE, r"diazepam|valium"),
    ("clonazepam", SEDATIVE, r"clonazepam|klonopin"),
    ("pentobarbital", SEDATIVE, r"pentobarbital|nembutal"),
    ("phenobarbital", SEDATIVE, r"phenobarbital|phenobarbitone"),
    ("etomidate", SEDATIVE, r"etomidate|amidate"),
    ("fentanyl", OPIOID, r"fentanyl|sublimaze"),
    ("hydromorphone", OPIOID, r"hydromorphone|dilaudid"),
    ("morphine", OPIOID, r"morphine"),
    ("remifentanil", OPIOID, r"remifentanil|ultiva"),
    ("sufentanil", OPIOID, r"sufentanil"),
    ("oxycodone", OPIOID, r"oxycodone|oxycontin|percocet"),
    ("hydrocodone", OPIOID, r"hydrocodone|vicodin|norco"),
    ("methadone", OPIOID, r"methadone"),
    ("buprenorphine", OPIOID, r"buprenorphine"),
    ("meperidine", OPIOID, r"meperidine|demerol"),
    ("tramadol", OPIOID, r"tramadol"),
    ("codeine", OPIOID, r"codeine"),
)
DRUG_CLASS: dict[str, str] = {name: cls for name, cls, _ in _DRUG_RULES}
_DRUG_RE = {name: re.compile(rf"\b(?:{rx})\b", re.I) for name, _, rx in _DRUG_RULES}

# Ingredients that get their own feature block (the plan's list); the rest roll up to class aggregates.
NAMED_INGREDIENTS: tuple[str, ...] = ("propofol", "midazolam", "lorazepam", "dexmedetomidine", "ketamine",
                                      "fentanyl", "hydromorphone", "morphine")


def match_ingredients(text: str | None) -> list[str]:
    """Ingredients named in ``text`` (combination products return several; empty if none)."""
    if not text:
        return []
    return [name for name, rx in _DRUG_RE.items() if rx.search(text)]


# ------------------------------------------------------------------------------------------- measurements
def norm_name(s: str | None) -> str:
    """Lower-case, non-alphanumerics -> single space. Stable key for lexicon matching and extra-lab keys."""
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


@dataclass(frozen=True)
class Rule:
    domain: str          # score | vital | pupil | poc_glucose | hx | tox | culture | lab
    key: str
    pattern: re.Pattern
    lag_h: float = 0.0   # availability lag after collection when no result time exists


def _r(domain: str, key: str, rx: str, lag_h: float = 0.0) -> Rule:
    return Rule(domain, key, re.compile(rx, re.I), lag_h)


# --- assay-lag table (hours from collection to result availability), used ONLY when the measurement table has
#     no result-time column (the OMOP measurement table has none, docs/heedb_schema_real.md B3).
#     [PLACEHOLDER] values are textbook turnaround guesses; replace with the Phase 0a observed medians.
LAG_POC = 0.1           # bedside / charted values: availability == charting time (plus a few minutes)
LAG_STAT_CHEM = 1.0     # basic chemistry, CBC, gases, lactate
LAG_SLOW_CHEM = 2.0     # LFTs, ammonia, troponin, coag, CRP
LAG_TOX = 3.0           # serum tox levels
LAG_UDS = 2.0           # urine drug screens
LAG_CSF = 1.5           # CSF cell count / chemistry
LAG_ENDO = 6.0          # TSH, cortisol, B12 and other send-outs
LAG_CULTURE = 24.0      # first positive / reportable culture result
IMAGING_LAG_H = {"ct_head": 1.0, "cta_head": 1.5, "mri_brain": 4.0, "other_imaging": 3.0}   # study -> final read

# Score / vital / history rules are matched first (charted values); order within the list matters.
RULES: tuple[Rule, ...] = (
    # ---- scores. Components before totals so "GCS eye" is not read as the total.
    _r("score", "gcs_eye", r"gcs.*eye|glasgow.*eye|eye opening|best eye"),
    _r("score", "gcs_motor", r"gcs.*motor|glasgow.*motor|best motor"),
    _r("score", "gcs_verbal", r"gcs.*verbal|glasgow.*verbal|best verbal"),
    _r("score", "four", r"\bfour score\b|\bfour\b.*score|full outline of unresponsiveness"),
    _r("score", "rass", r"\brass\b|richmond"),
    _r("score", "nesi", r"\bnesi\b"),
    _r("score", "gcs", r"glasgow coma|\bgcs\b"),
    # ---- pupils (side then size / reactivity). Coding of reactivity is ASSUMED: 0 = non-reactive.
    _r("pupil", "pupil_size_l", r"pupil.*(size|diam).*(\bl\b|\blt\b|left)|(\bl\b|\blt\b|left).*pupil.*(size|diam)"),
    _r("pupil", "pupil_size_r", r"pupil.*(size|diam).*(\br\b|\brt\b|right)|(\br\b|\brt\b|right).*pupil.*(size|diam)"),
    _r("pupil", "pupil_react_l", r"pupil.*(react|light).*(\bl\b|\blt\b|left)|(\bl\b|\blt\b|left).*pupil.*(react|light)"),
    _r("pupil", "pupil_react_r", r"pupil.*(react|light).*(\br\b|\brt\b|right)|(\br\b|\brt\b|right).*pupil.*(react|light)"),
    # ---- point-of-care glucose (before the serum glucose lab rule)
    _r("poc_glucose", "poc_glucose",
       r"(poc|point of care|fingerstick|finger stick|bedside|capillary|glucometer).*glucose|"
       r"glucose.*(poc|point of care|fingerstick|finger stick|bedside|capillary|glucometer)"),
    # ---- vitals
    _r("vital", "hr", r"heart rate|pulse rate|^hr$|^pulse$"),
    _r("vital", "sbp", r"systolic"),
    _r("vital", "dbp", r"diastolic"),
    _r("vital", "map", r"mean arterial|^map$|\bmap\b"),
    _r("vital", "rr", r"respiratory rate|resp rate|^rr$"),
    _r("vital", "spo2", r"spo2|oxygen saturation|pulse ox"),
    _r("vital", "temp_c", r"temperature|^temp$"),
    # ---- history flags that may be charted as measurements
    _r("hx", "convulsion", r"witnessed.*(seizure|convuls)|(seizure|convuls).*witnessed"),
    _r("hx", "arrest", r"cardiac arrest|\brosc\b|\bcpr\b"),
    _r("hx", "trauma", r"\btrauma|head injury"),
    # ---- toxicology (levels and urine screens)
    _r("tox", "ethanol", r"ethanol|alcohol level|\betoh\b", LAG_TOX),
    _r("tox", "acetaminophen", r"acetaminophen level|acetaminophen,? (serum|plasma)|\bapap\b|tylenol level", LAG_TOX),
    _r("tox", "salicylate", r"salicylate", LAG_TOX),
    _r("tox", "carboxyhemoglobin", r"carboxyhemoglobin|\bcohb\b", LAG_STAT_CHEM),
    _r("tox", "uds_benzodiazepine", r"(urine|drug screen|utox|uds).*benzodiazepine|benzodiazepine.*(urine|screen)", LAG_UDS),
    _r("tox", "uds_opiate", r"(urine|drug screen|utox|uds).*(opiate|opioid)|(opiate|opioid).*(urine|screen)", LAG_UDS),
    _r("tox", "uds_cocaine", r"(urine|drug screen|utox|uds).*cocaine|cocaine.*(urine|screen)", LAG_UDS),
    _r("tox", "uds_amphetamine", r"(urine|drug screen|utox|uds).*amphetamine|amphetamine.*(urine|screen)", LAG_UDS),
    _r("tox", "uds_cannabinoid", r"(urine|drug screen|utox|uds).*(cannabin|thc)|(cannabin|thc).*(urine|screen)", LAG_UDS),
    _r("tox", "uds_barbiturate", r"(urine|drug screen|utox|uds).*barbiturate|barbiturate.*(urine|screen)", LAG_UDS),
    _r("tox", "uds_fentanyl", r"(urine|drug screen|utox|uds).*fentanyl|fentanyl.*(urine|screen)", LAG_UDS),
    # ---- cultures (value: 1 = positive, 0 = no growth)
    _r("culture", "culture_csf", r"(csf|cerebrospinal).*culture|culture.*(csf|cerebrospinal)", LAG_CULTURE),
    _r("culture", "culture_blood", r"blood culture|culture.*blood|\bbcx\b", LAG_CULTURE),
    _r("culture", "culture_urine", r"urine culture|culture.*urine", LAG_CULTURE),
    _r("culture", "culture_resp", r"(sputum|tracheal|respiratory|bal)\b.*culture|culture.*(sputum|tracheal|respiratory)",
       LAG_CULTURE),
    # ---- CSF chemistry / cell counts (before serum rules so "CSF glucose" is not serum glucose)
    _r("lab", "csf_wbc", r"(csf|cerebrospinal).*(wbc|white|nucleated|cell count)|(wbc|nucleated cell).*(csf|cerebrospinal)", LAG_CSF),
    _r("lab", "csf_rbc", r"(csf|cerebrospinal).*(rbc|red)|rbc.*(csf|cerebrospinal)", LAG_CSF),
    _r("lab", "csf_protein", r"(csf|cerebrospinal).*protein|protein.*(csf|cerebrospinal)", LAG_CSF),
    _r("lab", "csf_glucose", r"(csf|cerebrospinal).*glucose|glucose.*(csf|cerebrospinal)", LAG_CSF),
    # ---- serum / blood labs
    _r("lab", "ph_arterial", r"\bph\b.*arter|arter.*\bph\b|^ph$|blood gas.*\bph\b", LAG_STAT_CHEM),
    _r("lab", "paco2", r"paco2|pco2", LAG_STAT_CHEM),
    _r("lab", "pao2", r"pao2|po2", LAG_STAT_CHEM),
    _r("lab", "lactate", r"lactate|lactic", LAG_STAT_CHEM),
    _r("lab", "ammonia", r"ammonia|\bnh3\b", LAG_SLOW_CHEM),
    _r("lab", "sodium", r"sodium|^na$", LAG_STAT_CHEM),
    _r("lab", "potassium", r"potassium|^k$", LAG_STAT_CHEM),
    _r("lab", "chloride", r"chloride", LAG_STAT_CHEM),
    _r("lab", "bicarbonate", r"bicarbonate|\bco2\b.*total|total co2|hco3", LAG_STAT_CHEM),
    _r("lab", "glucose", r"glucose", LAG_STAT_CHEM),
    _r("lab", "creatinine", r"creatinine", LAG_STAT_CHEM),
    _r("lab", "bun", r"\bbun\b|urea nitrogen", LAG_STAT_CHEM),
    _r("lab", "calcium", r"calcium", LAG_STAT_CHEM),
    _r("lab", "magnesium", r"magnesium", LAG_STAT_CHEM),
    _r("lab", "phosphate", r"phosph", LAG_STAT_CHEM),
    _r("lab", "albumin", r"albumin", LAG_SLOW_CHEM),
    _r("lab", "bilirubin", r"bilirubin", LAG_SLOW_CHEM),
    _r("lab", "ast", r"\bast\b|aspartate|sgot", LAG_SLOW_CHEM),
    _r("lab", "alt", r"\balt\b|alanine|sgpt", LAG_SLOW_CHEM),
    _r("lab", "alk_phos", r"alkaline phos|\balp\b", LAG_SLOW_CHEM),
    _r("lab", "wbc", r"\bwbc\b|white blood|leukocyte", LAG_STAT_CHEM),
    _r("lab", "hemoglobin", r"hemoglobin|haemoglobin|^hgb$", LAG_STAT_CHEM),
    _r("lab", "platelets", r"platelet|\bplt\b", LAG_STAT_CHEM),
    _r("lab", "inr", r"\binr\b", LAG_SLOW_CHEM),
    _r("lab", "troponin", r"troponin", LAG_SLOW_CHEM),
    _r("lab", "ck", r"creatine kinase|\bck\b|\bcpk\b", LAG_SLOW_CHEM),
    _r("lab", "crp", r"c reactive|\bcrp\b", LAG_SLOW_CHEM),
    _r("lab", "procalcitonin", r"procalcitonin", LAG_SLOW_CHEM),
    _r("lab", "osmolality", r"osmolality", LAG_STAT_CHEM),
    _r("lab", "tsh", r"\btsh\b|thyroid stimulating", LAG_ENDO),
    _r("lab", "cortisol", r"cortisol", LAG_ENDO),
    _r("lab", "enolase", r"enolase|\bnse\b", LAG_ENDO),
)

SCORE_KEYS = ("gcs", "gcs_eye", "gcs_motor", "gcs_verbal", "four", "rass", "nesi")
VITAL_KEYS = ("hr", "sbp", "dbp", "map", "rr", "spo2", "temp_c")
PUPIL_KEYS = ("pupil_size_l", "pupil_size_r", "pupil_react_l", "pupil_react_r")
LAB_KEYS = tuple(r.key for r in RULES if r.domain == "lab")
TOX_KEYS = tuple(r.key for r in RULES if r.domain == "tox")
CULTURE_KEYS = tuple(r.key for r in RULES if r.domain == "culture")
IMAGING_KEYS = tuple(IMAGING_LAG_H)
DEFAULT_LAB_LAG_H = LAG_SLOW_CHEM          # for an extra (data-discovered) lab name with no lag entry

# Plausible ranges: values outside are set missing (documentation errors, unit mix-ups). Scores / vitals only.
PLAUSIBLE: dict[str, tuple[float, float]] = {
    "gcs": (3, 15), "gcs_eye": (1, 4), "gcs_motor": (1, 6), "gcs_verbal": (1, 5), "four": (0, 16),
    "rass": (-5, 4), "hr": (0, 300), "sbp": (20, 320), "dbp": (10, 220), "map": (10, 250), "rr": (0, 80),
    "spo2": (30, 100), "temp_c": (20, 45), "pupil_size_l": (0.5, 10), "pupil_size_r": (0.5, 10),
}
# NESI has no fixed documented range in the repo; left unconstrained (flagged in the spec).


def classify_measurement(name: str | None) -> Rule | None:
    """First matching rule for a free-text measurement name (None = unmapped)."""
    n = norm_name(name)
    if not n:
        return None
    for rule in RULES:
        if rule.pattern.search(n):
            return rule
    return None


def to_canonical(key: str, value: float, unit: str | None) -> float:
    """Unit canonicalisation for the few keys with common mixed units; unknown units pass through."""
    if value != value:                      # NaN
        return value
    u = unit.lower() if isinstance(unit, str) else ""
    if key == "temp_c" and (u.startswith("f") or "degf" in u or "fahr" in u or (not u and value > 45)):
        return (value - 32.0) / 1.8
    if key in ("glucose", "poc_glucose", "csf_glucose") and "mmol" in u:
        return value * 18.016
    if key == "lactate" and "mg" in u:
        return value / 9.01
    if key == "creatinine" and ("umol" in u or "µmol" in u):
        return value / 88.4
    if key == "calcium" and "mmol" in u:
        return value * 4.008
    return value


# --------------------------------------------------------------------------------- history (ICD) patterns
ARREST_ICD = re.compile(r"^(?:I46|Z8674|4275)", re.I)                    # ICD-10 I46.x, Z86.74; ICD-9 427.5
HEAD_TRAUMA_ICD = re.compile(r"^(?:S06|S02|85[0-4])", re.I)             # intracranial injury, skull fx; ICD-9 850-854
TRAUMA_ICD = re.compile(r"^(?:S\d\d|T07|T14|8[0-9]{2}|9[0-4][0-9])", re.I)   # ICD-10 S, T07, T14; ICD-9 800-949
CPR_PROCEDURE = re.compile(r"^(?:92950|5A12012)$", re.I)                # CPT 92950; ICD-10-PCS 5A12012
CONVULSION_OBS = re.compile(r"witnessed.*(?:seizure|convuls)|(?:seizure|convuls).*witnessed", re.I)
NEGATIVE_VALUE = re.compile(r"^(?:no|none|negative|neg|denied|absent|false|0)$", re.I)

# --------------------------------------------------------------------------------------------- imaging
IMAGING_MODALITY = (
    ("cta_head", re.compile(r"\bcta\b|ct angio", re.I)),
    ("ct_head", re.compile(r"\bct\b.*(head|brain)|head.*\bct\b|noncontrast ct", re.I)),
    ("mri_brain", re.compile(r"\bmri?\b|magnetic", re.I)),
)


def imaging_key(modality: str | None) -> str:
    n = norm_name(modality)
    for key, rx in IMAGING_MODALITY:
        if rx.search(n):
            return key
    return "other_imaging"


# ------------------------------------------------------------------------------------ EEG indication (D)
INDICATION_LEVELS: tuple[str, ...] = ("rule_out_ncse", "post_arrest", "unexplained_ams", "seizure", "spell",
                                      "other", "missing")
_INDICATION_RULES = (
    ("rule_out_ncse", re.compile(r"ncse|non.?convulsive|status epilepticus|\bsre\b", re.I)),
    ("post_arrest", re.compile(r"arrest|rosc|hypox|anoxic|\bhie\b|post.?cpr", re.I)),
    ("unexplained_ams", re.compile(r"\bams\b|altered|mental status|encephalopath|coma|unrespons|obtund|letharg", re.I)),
    ("seizure", re.compile(r"seiz|convuls|epilep", re.I)),
    ("spell", re.compile(r"spell|syncope|event|episode|fainting", re.I)),
)


def indication_category(text: str | None) -> str:
    """Map the EEG referral-indication field to a fixed category set (``INDICATION_LEVELS``)."""
    n = (text or "").strip()
    if not n or n.lower() in {"none", "nan", "null", "unknown"}:
        return "missing"
    n = norm_name(n)
    for level, rx in _INDICATION_RULES:
        if rx.search(n):
            return level
    return "other"
