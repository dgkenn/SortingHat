"""Case report form (``configs/adjudication_crf.yaml``): REDCap data dictionary and rating-export parsing (D-157).

The YAML is the single source of truth.  ``data_dictionary`` expands it into the 18-column REDCap data-dictionary layout;
``long_ratings`` / ``case_fields`` turn a flat REDCap export (one row per case x rater; state values as codes or names) into the tidy
frames ``sortinghat.adjudication.agreement`` uses.  Nothing here reads patient data.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import yaml

from ..labels.ontology import GoldState

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "adjudication_crf.yaml"
REDCAP_COLUMNS = ["Variable / Field Name", "Form Name", "Section Header", "Field Type", "Field Label",
                  "Choices, Calculations, OR Slider Labels", "Field Note", "Text Validation Type OR Show Slider Number",
                  "Text Validation Min", "Text Validation Max", "Identifier?", "Branching Logic (Show field only if...)",
                  "Required Field?", "Custom Alignment", "Question Number (surveys only)", "Matrix Group Name", "Matrix Ranking?",
                  "Field Annotation"]
FIELD_TYPES = {"text", "notes", "dropdown", "radio", "checkbox", "yesno", "truefalse", "descriptive"}
NAME_RX = re.compile(r"^[a-z][a-z0-9_]{0,25}$")


def load_crf(path=None) -> dict:
    return yaml.safe_load(Path(path or DEFAULT_PATH).read_text(encoding="utf-8"))


def state_names(cfg: dict) -> dict:
    """{code str: state name}."""
    return {str(k): str(v) for k, v in cfg["state_scale"].items()}


def label_codes(cfg: dict) -> list:
    return [l["code"] for l in cfg["labels"]]


def _ctx(cfg: dict) -> dict:
    inv = {v: str(k) for k, v in state_names(cfg).items()}
    return {n: inv[n] for n in ("absent", "possible", "probable", "definite", "unassessable")}


def _fmt(s, **kw):
    return s.format(**kw) if isinstance(s, str) else s


def _choices(cfg: dict, spec) -> dict | None:
    if spec is None:
        return None
    if isinstance(spec, str):
        if spec == "state_scale":
            return {str(k): str(v) for k, v in cfg["state_scale"].items()}
        if spec == "confidence_scale":
            return {str(k): str(v) for k, v in cfg["confidence_scale"].items()}
        if spec == "evidence_choices":
            return {str(k): str(v) for k, v in cfg["evidence_choices"].items()}
        raise KeyError(spec)
    return {str(k): str(v) for k, v in spec.items()}


def expand_fields(cfg: dict) -> list:
    """Ordered list of field dicts (name, type, label, choices, section, ...) with the per-label templates expanded."""
    ctx = _ctx(cfg)
    out: list = []
    for sec in cfg["sections"]:
        first = [True]

        def add(spec: dict, **kw):
            f = dict(spec)
            f["section"] = sec["header"] if first[0] else ""
            first[0] = False
            for k in ("name", "label", "note", "branching"):
                if k in f:
                    f[k] = _fmt(f[k], **ctx, **kw)
            if "choicelist" in f:
                f["choices"] = {c["code"]: c["label"] for c in cfg[f.pop("choicelist")]}
            else:
                f["choices"] = _choices(cfg, f.get("choices"))
            out.append(f)

        for f in sec.get("fields", []):
            add(f)
        if "per_label" in sec:
            after = sec.get("after", {})
            for lab in cfg["labels"]:
                kw = {"f": lab["field"], "code": lab["code"], "name": lab["name"], "definition": lab["definition"]}
                add(sec["per_label"]["state"], **kw)
                add(sec["per_label"]["conf"], **kw)
                for extra in after.get(f"{lab['field']}_conf", []):
                    add(extra, **kw)
    return out


def validate_fields(fields: list) -> list:
    """Problems found in the expanded form (empty list = fine): names, types, choices, branching references."""
    errs, names = [], [f["name"] for f in fields]
    if len(set(names)) != len(names):
        errs.append("duplicate field names")
    for f in fields:
        if not NAME_RX.match(f["name"]):
            errs.append(f"bad field name {f['name']}")
        if f["type"] not in FIELD_TYPES:
            errs.append(f"bad type {f['type']} for {f['name']}")
        if f["type"] in ("radio", "dropdown", "checkbox") and not f.get("choices"):
            errs.append(f"{f['name']}: choice field without choices")
        for ref in re.findall(r"\[([a-z0-9_]+)(?:\(\d+\))?\]", f.get("branching") or ""):
            if ref not in names:
                errs.append(f"{f['name']}: branching refers to unknown field {ref}")
    if fields and fields[0]["type"] != "text":
        errs.append("the first field (record id) must be text")
    return errs


def data_dictionary(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_crf()
    fields = expand_fields(cfg)
    errs = validate_fields(fields)
    if errs:
        raise ValueError("invalid CRF: " + "; ".join(errs))
    rows = []
    for f in fields:
        ch = " | ".join(f"{k}, {v}" for k, v in f["choices"].items()) if f.get("choices") else ""
        rows.append([f["name"], cfg["form"], f["section"], f["type"], f["label"], ch, f.get("note", ""), f.get("validation", ""),
                     f.get("min", ""), f.get("max", ""), "", f.get("branching", ""), "y" if f.get("required") else "", "", "", "", "",
                     f.get("annotation", "")])
    return pd.DataFrame(rows, columns=REDCAP_COLUMNS)


def write_dictionary(path, cfg: dict | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data_dictionary(cfg).to_csv(p, index=False, lineterminator="\n")
    return p


# ============================================================================================ export parsing
def _norm_state(v, names: dict):
    """State name from a REDCap code or a name; NaN / unknown -> None."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    if s == "":
        return None
    if s.endswith(".0") and s[:-2].lstrip("-").isdigit():
        s = s[:-2]
    if s in names:
        return names[s]
    low = s.lower()
    return low if low in names.values() else None


def long_ratings(export: pd.DataFrame, cfg: dict | None = None, labels: list | None = None) -> pd.DataFrame:
    """Flat export (columns ``case_id``, ``rater_id`` and the CRF fields) -> long frame case_id, rater, label, state (name), conf.
    Rows with a blank ``case_id`` or ``rater_id`` are dropped; duplicate (case, rater) rows keep the last (a corrected form)."""
    cfg = cfg or load_crf()
    names = state_names(cfg)
    df = export.rename(columns={"record_id": "case_id"}).copy()
    df = df[df["case_id"].astype(str).str.strip().ne("") & df["rater_id"].astype(str).str.strip().ne("") & df["case_id"].notna() & df["rater_id"].notna()]
    df = df.drop_duplicates(["case_id", "rater_id"], keep="last")
    rows = []
    for lab in cfg["labels"]:
        if labels is not None and lab["code"] not in labels:
            continue
        sc, cc = f"{lab['field']}_state", f"{lab['field']}_conf"
        if sc not in df:
            continue
        st = df[sc].map(lambda v: _norm_state(v, names))
        conf = pd.to_numeric(df[cc], errors="coerce") if cc in df else pd.Series(float("nan"), index=df.index)
        rows.append(pd.DataFrame({"case_id": df["case_id"].astype(str).to_numpy(), "rater": df["rater_id"].astype(str).to_numpy(),
                                  "label": lab["code"], "state": st.to_numpy(), "conf": conf.to_numpy()}))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["case_id", "rater", "label", "state", "conf"])
    return out[out["state"].notna()].reset_index(drop=True)


def case_fields(export: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """One row per (case, rater): primary_etiology, primary_conf, minutes_spent, dx_certainty, eeg_seen, packet_quality, ams_at_t0."""
    df = export.rename(columns={"record_id": "case_id"}).copy()
    df = df[df["case_id"].notna() & df["rater_id"].notna() & df["case_id"].astype(str).str.strip().ne("") & df["rater_id"].astype(str).str.strip().ne("")]
    df = df.drop_duplicates(["case_id", "rater_id"], keep="last")
    keep = [c for c in ("case_id", "rater_id", "primary_etiology", "primary_conf", "minutes_spent", "dx_certainty", "eeg_seen",
                        "packet_quality", "ams_at_t0") if c in df]
    out = df[keep].rename(columns={"rater_id": "rater"}).reset_index(drop=True)
    out["case_id"] = out["case_id"].astype(str)
    return out


def positive_states(cfg: dict | None = None) -> set:
    cfg = cfg or load_crf()
    return {s.lower() for s in cfg["positive_states"]}


def gold_state(name) -> GoldState:
    return GoldState.parse(name)
