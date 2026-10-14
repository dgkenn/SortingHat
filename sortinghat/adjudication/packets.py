"""EEG-blind rater packets (D-157; research plan "Reference standard": locally pre-assembled packets).

For each sampled patient a packet holds, with every time RELATIVE to t0 (hours; negative = before) and no absolute date, site name or
patient identifier:

  header          age, sex, hours from the (date-granular) visit start to t0
  diagnoses       condition rows in [-history, +after] (rows before the packet window flagged "prior history")
  procedures      procedure rows in the window
  medications     drug rows overlapping the window
  scores          GCS / FOUR / RASS / NIHSS / pupil style measurements (full series)
  vitals          vital signs binned (6 h): n, median, min, max
  labs            every other measurement, per name
  observations    flowsheet observations (text passed through the EEG filter)
  notes           clinical notes in the window: EEG / neurophysiology notes never loaded, every sentence mentioning EEG-derived content
                  dropped (``blinding``); or "pending" when no text source is configured

Nothing here is printed: the caller writes packets under ``local_only/`` (mode 0600) and prints only counts.  A packet whose final text
still trips the EEG audit, or that contains the patient's own id, is withheld.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

from ..metrics import intended_use as iu
from . import blinding
from .private_io import private_dir, write_private, write_private_csv

SECTION_ORDER = ("diagnoses", "procedures", "medications", "scores", "vitals", "labs", "observations")
VITAL_RX = re.compile(r"heart\s*rate|pulse|\bhr\b|resp(?:iratory)?\s*rate|\brr\b|spo2|o2\s*sat|oxygen\s*sat|\btemp|blood\s*pressure|\bbp\b|"
                      r"\bmap\b|systolic|diastolic|\bsbp\b|\bdbp\b|mean\s*arterial|\bnibp\b|\bibp\b|\babp\b", re.I)
SCORE_RX = re.compile(r"glasgow|\bgcs\b|four\s*score|\bfour\b|\brass\b|richmond|nihss|ramsay|pupil|gaze|motor\s*response|"
                      r"verbal\s*response|eye\s*opening|best\s*(?:motor|verbal|eye)", re.I)
TIME_COLS = {"measurement": ("measurement_datetime", "measurement_date"), "condition_occurrence": ("condition_start_datetime", "condition_start_date"),
             "procedure_occurrence": ("procedure_datetime", "procedure_date"), "observation": ("observation_datetime", "observation_date"),
             "note": ("note_datetime", "note_date")}
DRUG_START, DRUG_END = ("drug_exposure_start_datetime", "drug_exposure_start_date"), ("drug_exposure_end_datetime", "drug_exposure_end_date")


@dataclass(frozen=True)
class PacketConfig:
    before_h: float = 72.0
    after_days: float = 30.0
    history_days: float = 365.0                 # diagnoses only: earlier rows are listed as "prior history"
    vital_bin_h: float = 6.0
    max_values_per_series: int = 60             # per measurement name in the HTML (the CSV keeps all)
    max_notes: int = 400
    max_note_chars: int = 250_000
    notes_after_days: float | None = None       # notes window end (default: after_days); shorten it if packets prove too long to read

    @property
    def notes_hi(self) -> float:
        return float(self.after_days if self.notes_after_days is None else self.notes_after_days) * 24.0

    @property
    def lo(self) -> float:
        return -float(self.before_h)

    @property
    def hi(self) -> float:
        return float(self.after_days) * 24.0

    @property
    def hist_lo(self) -> float:
        return -float(self.history_days) * 24.0

    @classmethod
    def from_dict(cls, d: Mapping | None) -> "PacketConfig":
        d = dict(d or {})
        names = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in names})


# ===================================================================================================== time helpers
def add_rel(df: pd.DataFrame, dt: str, dd: str, t0_map: Mapping, out: str = "rel_h", prefix: str = "") -> pd.DataFrame:
    """Add ``rel_h`` (hours from the patient's t0; day offset x 24 for a date-only value) and ``date_only``.  Rows with no time get NaN."""
    df = df.copy()
    t = iu._times(df, dt, dd)
    t0 = pd.to_datetime(df["person_id"].map(t0_map))
    tod = iu._has_time_of_day(t)
    rel = (t - t0) / pd.Timedelta(hours=1)
    day = (t.dt.normalize() - t0.dt.normalize()) / pd.Timedelta(hours=1)
    df[prefix + out] = rel.where(tod, day).astype(float)
    df[prefix + "date_only"] = (~tod) & t.notna()
    return df


def window_keep(dt: str, dd: str, t0_map: Mapping, lo: float, hi: float):
    def keep(ch: pd.DataFrame) -> pd.DataFrame:
        ch = add_rel(ch, dt, dd, t0_map)
        return ch[ch["rel_h"].between(lo, hi)]
    return keep


def drug_keep(t0_map: Mapping, lo: float, hi: float):
    """A drug row is kept when it starts inside the window or is running at the window start (start before, end at/after it)."""
    def keep(ch: pd.DataFrame) -> pd.DataFrame:
        ch = add_rel(ch, *DRUG_START, t0_map, out="rel_h")
        e = add_rel(ch[["person_id", *[c for c in DRUG_END if c in ch]]], *DRUG_END, t0_map, out="end_h")
        ch["end_h"] = e["end_h"].to_numpy()
        ch["end_h"] = ch["end_h"].where(ch["end_h"].notna() & (ch["end_h"] >= ch["rel_h"]), ch["rel_h"])
        return ch[(ch["rel_h"] <= hi) & (ch["end_h"] >= lo)]
    return keep


# ===================================================================================================== sections
def _reasons(values: pd.Series, code: bool = False) -> dict:
    return {u: blinding.blocked_row(u, code=u if code else None) for u in values.astype("string").fillna("").unique()}


def _drop(df: pd.DataFrame, key: pd.Series, reasons: dict, drops: Counter) -> pd.DataFrame:
    r = key.astype("string").fillna("").map(reasons)
    for v, c in r.dropna().value_counts().items():
        drops[v] += int(c)
    return df[r.isna().to_numpy()]


def sec_coded(raw: pd.DataFrame, col: str, cfg: PacketConfig, drops: Counter, with_history: bool) -> pd.DataFrame:
    """diagnoses / procedures: code, rel_h, date_only, period; one row per (code, day)."""
    if raw is None or not len(raw):
        return pd.DataFrame(columns=["code", "rel_h", "date_only", "period"])
    df = _drop(raw, raw[col], _reasons(raw[col], code=True), drops)
    df = df[df["rel_h"] >= (cfg.hist_lo if with_history else cfg.lo)]
    out = pd.DataFrame({"code": df[col].astype("string").fillna("").str.strip().to_numpy(), "rel_h": df["rel_h"].round(1).to_numpy(),
                        "date_only": df["date_only"].to_numpy(bool)})
    out["period"] = np.where(out["rel_h"] < cfg.lo, "prior history", "window")
    out = out.assign(_d=np.floor(out["rel_h"] / 24.0)).drop_duplicates(["code", "_d"]).drop(columns="_d")
    return out.sort_values(["rel_h", "code"]).reset_index(drop=True)


def sec_meds(raw: pd.DataFrame, cfg: PacketConfig, drops: Counter) -> pd.DataFrame:
    cols = ["drug", "route", "start_h", "end_h", "quantity", "date_only"]
    if raw is None or not len(raw):
        return pd.DataFrame(columns=cols)
    df = _drop(raw, raw["drug_source_value"], _reasons(raw["drug_source_value"]), drops)
    route = df["route_source_value"].astype("string").fillna("").str.strip() if "route_source_value" in df else pd.Series("", index=df.index)
    out = pd.DataFrame({"drug": df["drug_source_value"].astype("string").fillna("").str.strip().to_numpy(),
                        "route": route.to_numpy(),
                        "start_h": df["rel_h"].round(1).to_numpy(), "end_h": df["end_h"].round(1).to_numpy(),
                        "quantity": (pd.to_numeric(df["quantity"], errors="coerce") if "quantity" in df else pd.Series(np.nan, index=df.index)).to_numpy(),
                        "date_only": df["date_only"].to_numpy(bool)})
    out["route"] = out["route"].replace({"nan": "", "None": ""})
    out = out[out["drug"] != ""]
    return out.sort_values(["start_h", "drug"]).reset_index(drop=True)


def _measure_frame(raw: pd.DataFrame, drops: Counter) -> pd.DataFrame:
    name = raw["measurement_source_value"].astype("string").fillna("")
    num = pd.to_numeric(raw["value_as_number"], errors="coerce") if "value_as_number" in raw else pd.Series(np.nan, index=raw.index)
    txt = raw["value_source_value"].astype("string") if "value_source_value" in raw else pd.Series(pd.NA, index=raw.index)
    val_txt = txt.where(num.isna())
    df = raw.assign(_name=name, _num=num, _txt=val_txt)
    df = _drop(df, df["_name"], _reasons(df["_name"]), drops)
    df = _drop(df, df["_txt"], _reasons(df["_txt"]), drops)
    unit = df["unit_source_value"].astype("string").fillna("") if "unit_source_value" in df else pd.Series("", index=df.index)
    value = df["_num"].where(df["_num"].notna(), df["_txt"].astype("object"))
    return pd.DataFrame({"name": df["_name"].str.strip().to_numpy(), "value": value.to_numpy(), "num": df["_num"].to_numpy(float),
                         "unit": unit.to_numpy(), "rel_h": df["rel_h"].round(2).to_numpy(), "date_only": df["date_only"].to_numpy(bool)})


def sec_measurements(raw: pd.DataFrame, cfg: PacketConfig, drops: Counter) -> dict:
    """{'scores', 'vitals', 'labs'}: scores and labs keep every value; vitals are binned."""
    empty = {"scores": pd.DataFrame(columns=["name", "value", "unit", "rel_h", "date_only"]),
             "vitals": pd.DataFrame(columns=["name", "unit", "bin_start_h", "n", "median", "min", "max"]),
             "labs": pd.DataFrame(columns=["name", "value", "unit", "rel_h", "date_only"])}
    if raw is None or not len(raw):
        return empty
    m = _measure_frame(raw, drops)
    m = m[m["rel_h"].between(cfg.lo, cfg.hi)]
    if not len(m):
        return empty
    cat = m["name"].map(lambda s: "score" if SCORE_RX.search(s) else ("vital" if VITAL_RX.search(s) else "lab"))
    cols = ["name", "value", "unit", "rel_h", "date_only"]
    scores = m[cat == "score"][cols].sort_values(["name", "rel_h"])
    labs = m[cat == "lab"][cols].sort_values(["name", "rel_h"])
    v = m[(cat == "vital") & m["num"].notna()].copy()
    if len(v):
        v["bin_start_h"] = cfg.lo + np.floor((v["rel_h"] - cfg.lo) / cfg.vital_bin_h) * cfg.vital_bin_h
        g = v.groupby(["name", "unit", "bin_start_h"])["num"]
        vit = g.agg(n="size", median="median", min="min", max="max").reset_index().round(2)
        vit = vit.sort_values(["name", "bin_start_h"])
    else:
        vit = empty["vitals"]
    nonnum_vitals = m[(cat == "vital") & m["num"].isna()][cols]          # text-valued rows of a vital name stay as observations
    labs = pd.concat([labs, nonnum_vitals]).sort_values(["name", "rel_h"])
    return {"scores": scores.reset_index(drop=True), "vitals": vit.reset_index(drop=True), "labs": labs.reset_index(drop=True)}


def sec_observations(raw: pd.DataFrame, cfg: PacketConfig, drops: Counter) -> pd.DataFrame:
    cols = ["name", "value", "rel_h", "date_only"]
    if raw is None or not len(raw):
        return pd.DataFrame(columns=cols)
    df = raw.assign(_n=raw["observation_source_value"].astype("string").fillna(""),
                    _v=(raw["value_as_string"].astype("string") if "value_as_string" in raw else pd.Series(pd.NA, index=raw.index)))
    df = _drop(df, df["_n"], _reasons(df["_n"]), drops)
    df = _drop(df, df["_v"], _reasons(df["_v"]), drops)
    df = df[df["rel_h"].between(cfg.lo, cfg.hi)]
    out = pd.DataFrame({"name": df["_n"].str.strip().to_numpy(), "value": df["_v"].fillna("").map(blinding.scrub_ids).to_numpy(),
                        "rel_h": df["rel_h"].round(1).to_numpy(), "date_only": df["date_only"].to_numpy(bool)})
    return out.sort_values(["rel_h", "name"]).reset_index(drop=True)


# ===================================================================================================== notes
def prepare_notes(raw: pd.DataFrame | None, cfg: PacketConfig, drops: Counter) -> list[dict]:
    """Window-trimmed (``rel_h`` present) raw notes -> filtered notes [{rel_h, date_only, title, text}] (title / text EEG-filtered)."""
    if raw is None or not len(raw):
        return []
    out = []
    for _, r in raw.sort_values("rel_h", kind="mergesort").iterrows():
        title = str(r.get("note_title", "") or "")
        src = str(r.get("note_source_value", "") or "")
        if blinding.note_excluded(title, src):
            drops[blinding.REASON_NOTE] += 1
            continue
        text, removed = blinding.filter_note_text(str(r.get("note_text", "") or ""))
        drops["eeg_sentences_removed"] += int(removed)
        if not text.strip():
            drops["notes_empty_after_filter"] += 1
            continue
        t = "Note" if (not title or title == "nan" or blinding.mentions_eeg(title)) else blinding.scrub_ids(title)
        out.append({"rel_h": round(float(r["rel_h"]), 1), "date_only": bool(r.get("date_only", False)), "title": t, "text": text})
    return out


def cap_notes(notes: list[dict], cfg: PacketConfig) -> tuple[list[dict], bool]:
    """Keep notes in time order until the count / character cap; (kept, truncated)."""
    kept, chars = [], 0
    for n in notes:
        if len(kept) >= cfg.max_notes or chars + len(n["text"]) > cfg.max_note_chars:
            return kept, True
        kept.append(n)
        chars += len(n["text"])
    return kept, False


# ===================================================================================================== packet
@dataclass
class Packet:
    case_id: str
    header: dict
    sections: dict
    notes: list
    notes_status: str                       # included | pending_no_source | none_in_window | included_truncated
    counts: dict = field(default_factory=dict)
    dropped: dict = field(default_factory=dict)
    withheld: str | None = None


def _t(h, date_only: bool) -> str:
    if h is None or (isinstance(h, float) and np.isnan(h)):
        return "?"
    return f"day {h / 24.0:+.0f} (date only)" if date_only else f"{h:+.1f} h"


def build_packet(case_id: str, person: Mapping, raw: Mapping[str, pd.DataFrame | None], notes: list[dict] | None,
                 cfg: PacketConfig, *, notes_source_configured: bool) -> Packet:
    """``person``: age_years, sex_male (optional), hours_since_visit_start (optional).  ``raw``: per OMOP table the person's rows with
    ``rel_h`` / ``date_only`` already attached (``add_rel``) and, for drugs, ``end_h``.  ``notes``: output of ``prepare_notes`` or None."""
    drops: Counter = Counter()
    s: dict = {}
    s["diagnoses"] = sec_coded(raw.get("condition_occurrence"), "condition_source_value", cfg, drops, with_history=True)
    s["procedures"] = sec_coded(raw.get("procedure_occurrence"), "procedure_source_value", cfg, drops, with_history=False)
    s["medications"] = sec_meds(raw.get("drug_exposure"), cfg, drops)
    s.update(sec_measurements(raw.get("measurement"), cfg, drops))
    s["observations"] = sec_observations(raw.get("observation"), cfg, drops)
    sex = person.get("sex_male")
    header = {"age_years": None if person.get("age_years") is None or pd.isna(person.get("age_years")) else int(round(float(person["age_years"]))),
              "sex": "not recorded" if sex is None or pd.isna(sex) else ("male" if float(sex) >= 0.5 else "female"),
              "hours_since_visit_start": None if person.get("hours_since_visit_start") is None or pd.isna(person.get("hours_since_visit_start"))
              else round(float(person["hours_since_visit_start"]), 0)}
    kept, status = [], "pending_no_source"
    if notes is not None:
        kept, trunc = cap_notes(notes, cfg)
        status = "included_truncated" if trunc else ("included" if kept else "none_in_window")
    elif notes_source_configured:
        status = "none_in_window"
    counts = {k: int(len(v)) for k, v in s.items()}
    counts["notes"] = len(kept)
    return Packet(case_id, header, s, kept, status, counts, dict(drops))


def _table_html(df: pd.DataFrame, cols: list, labels: list | None = None) -> str:
    if not len(df):
        return "<p class='none'>None in the window.</p>"
    head = "".join(f"<th>{html.escape(l)}</th>" for l in (labels or cols))
    rows = []
    for _, r in df.iterrows():
        rows.append("<tr>" + "".join(f"<td>{html.escape('' if pd.isna(r[c]) else str(r[c]))}</td>" for c in cols) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _dx_html(df: pd.DataFrame) -> str:
    """Diagnoses collapsed to one row per code (first / last time and number of distinct days); prior history listed separately."""
    if not len(df):
        return "<p class='none'>None in the window.</p>"
    out = []
    win, prior = df[df["period"] == "window"], df[df["period"] == "prior history"]
    for title, d, cols in (("In the window", win, ("first", "last", "days")), ("Prior history (before the window)", prior, ("first", "last", "days"))):
        if not len(d):
            continue
        g = d.groupby("code").agg(first=("rel_h", "min"), last=("rel_h", "max"), days=("rel_h", lambda x: len({int(v // 24) for v in x}))).reset_index()
        g = g.sort_values(["first", "code"])
        g["first"], g["last"] = g["first"].map(lambda v: f"{v:+.1f} h"), g["last"].map(lambda v: f"{v:+.1f} h")
        out.append(f"<h3>{title}</h3>" + _table_html(g, ["code", "first", "last", "days"], ["Code", "First listed", "Last listed", "Distinct days"]))
    return "".join(out) + "<p class='small'>One row per code; every listing is in structured/diagnoses.csv.</p>"


def _series_html(df: pd.DataFrame, cfg: PacketConfig) -> str:
    if not len(df):
        return "<p class='none'>None in the window.</p>"
    rows = []
    for (name, unit), g in df.groupby(["name", "unit"], sort=True):
        vals = [f"{_t(r.rel_h, r.date_only)}: {'' if pd.isna(r.value) else r.value}" for r in g.itertuples()]
        if len(vals) > cfg.max_values_per_series:
            k = cfg.max_values_per_series // 2
            vals = vals[:k] + [f"... {len(vals) - 2 * k} values omitted (all are in structured CSV) ..."] + vals[-k:]
        rows.append(f"<tr><td>{html.escape(str(name))}</td><td>{html.escape(str(unit))}</td><td>{html.escape('; '.join(vals))}</td></tr>")
    return "<table><thead><tr><th>Measure</th><th>Unit</th><th>Values (time relative to t0)</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def _meds_html(df: pd.DataFrame) -> str:
    if not len(df):
        return "<p class='none'>None in the window.</p>"
    rows = []
    for (drug, route), g in df.groupby(["drug", "route"], sort=False):
        first, last = g.iloc[0], g.iloc[-1]
        rows.append((float(g["start_h"].min()), f"<tr><td>{html.escape(str(drug))}</td><td>{html.escape(str(route))}</td>"
                     f"<td>{html.escape(_t(first['start_h'], bool(first['date_only'])))}</td><td>{html.escape(_t(g['end_h'].max(), bool(last['date_only'])))}</td>"
                     f"<td>{len(g)}</td></tr>"))
    rows.sort(key=lambda x: x[0])
    return ("<table><thead><tr><th>Drug (as recorded)</th><th>Route</th><th>First start</th><th>Last end</th><th>Records</th></tr></thead><tbody>"
            + "".join(r for _, r in rows) + "</tbody></table><p class='small'>Individual administrations are in structured/medications.csv.</p>")


CSS = ("body{font-family:system-ui,sans-serif;max-width:1100px;margin:1.5rem auto;padding:0 1rem;line-height:1.4}"
       "table{border-collapse:collapse;width:100%;margin:.4rem 0 1rem}th,td{border:1px solid #bbb;padding:.25rem .45rem;vertical-align:top;font-size:.9rem}"
       "th{background:#eee;text-align:left}pre{white-space:pre-wrap;background:#f7f7f7;padding:.6rem;border:1px solid #ddd}"
       ".none,.small{color:#555;font-size:.85rem}.banner{background:#fff3cd;border:1px solid #e0c36a;padding:.6rem}")


def render_html(p: Packet, cfg: PacketConfig) -> str:
    h = p.header
    top = [f"<h1>Case {html.escape(p.case_id)}</h1>",
           "<p class='banner'>EEG-blind packet. Every time is relative to t0 (hour 0, the index time of the assessment): negative = before. "
           "No EEG, EEG report or EEG-derived monitor value is included; sentences mentioning them were removed. If you see any, record it in the "
           "form (field: EEG information seen) and do not use it.</p>",
           f"<p><b>Age</b> {html.escape(str(h['age_years']) if h['age_years'] is not None else 'not recorded')} y &nbsp; <b>Sex</b> {html.escape(h['sex'])}"
           + (f" &nbsp; <b>Hours from hospital visit start to t0</b> about {h['hours_since_visit_start']:.0f} (visit start is date-only: +/-24 h)"
              if h["hours_since_visit_start"] is not None else "") + "</p>",
           f"<p class='small'>Window: {cfg.before_h:g} h before to {cfg.after_days:g} days after t0. Diagnoses also list prior history up to "
           f"{cfg.history_days:g} days before.</p>"]
    body = []
    body.append("<h2>Clinical notes</h2>")
    if p.notes_status == "pending_no_source":
        body.append("<p class='banner'>NOTES PENDING: the note text source was not available when this packet was built. Do not rate until the notes are added.</p>")
    elif not p.notes:
        body.append("<p class='none'>No notes available in the window.</p>")
    if len(p.notes) > 1:
        body.append("<p class='small'>Index: " + " &middot; ".join(
            f"<a href='#n{i}'>{i}: {html.escape(_t(n['rel_h'], n['date_only']))} {html.escape(n['title'])}</a>" for i, n in enumerate(p.notes, 1)) + "</p>")
    for i, n in enumerate(p.notes, 1):
        body.append(f"<h3 id='n{i}'>Note {i}: {html.escape(_t(n['rel_h'], n['date_only']))} - {html.escape(n['title'])}</h3><pre>{html.escape(n['text'])}</pre>")
    if p.notes_status == "included_truncated":
        body.append("<p class='banner'>Note list truncated at the size cap; later notes are not shown.</p>")
    body.append("<h2>Diagnoses and problem list codes</h2>" + _dx_html(p.sections["diagnoses"]))
    body.append("<h2>Procedures</h2>" + _table_html(p.sections["procedures"], ["code", "rel_h"], ["Code", "Hours from t0"]))
    body.append("<h2>Medications</h2>" + _meds_html(p.sections["medications"]))
    body.append("<h2>Neurological and sedation scores</h2>" + _series_html(p.sections["scores"], cfg))
    body.append("<h2>Vital signs (6 h bins)</h2>" + _table_html(p.sections["vitals"], ["name", "unit", "bin_start_h", "n", "median", "min", "max"],
                                                                ["Measure", "Unit", "Bin start (h)", "n", "Median", "Min", "Max"]))
    body.append("<h2>Laboratory and other measurements</h2>" + _series_html(p.sections["labs"], cfg))
    body.append("<h2>Flowsheet observations</h2>" + _table_html(p.sections["observations"], ["name", "value", "rel_h"], ["Observation", "Value", "Hours from t0"]))
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>Case {html.escape(p.case_id)}</title><style>{CSS}</style></head><body>"
            + "".join(top + body) + "</body></html>")


def packet_texts(p: Packet) -> list[str]:
    """Every DATA string a rater can see (tables and notes; the fixed page template, which says "EEG-blind", is not data)."""
    out: list = []
    for df in p.sections.values():
        for c in df.columns:
            if not (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])):
                out.extend(str(v) for v in df[c].dropna())
    out.extend(n["text"] for n in p.notes)
    out.extend(n["title"] for n in p.notes)
    return out


def finalize(p: Packet, cfg: PacketConfig, forbidden_ids: list[str]) -> tuple[Packet, str]:
    """Render, run the final EEG audit and the own-identifier check; a failing packet is marked withheld (and renders nothing useful)."""
    page = render_html(p, cfg)
    texts = packet_texts(p)
    if blinding.audit_text(texts) > 0:
        p.withheld = "eeg_audit"
    else:
        blob = "\n".join(texts) + "\n" + re.sub(r"<style>.*?</style>", "", page, flags=re.S)
        for i in forbidden_ids:
            if i and re.search(rf"(?<![0-9]){re.escape(str(i))}(?![0-9])", blob):
                p.withheld = "contains_patient_id"
                break
    return p, page


def write_packet(root, p: Packet, page: str, manifest_root=None) -> None:
    """Write packet.html, structured/*.csv and notes/note_NNN.txt under ``root/<case_id>/`` (0700 / 0600): the folder a rater gets.  The
    manifest (counts, rows dropped by reason: it names EEG reasons) is STUDY-TEAM ONLY and goes to ``manifest_root/<case_id>.json``."""
    d = private_dir(root) / p.case_id
    private_dir(d)
    write_private(d / "packet.html", page)
    for k, df in p.sections.items():
        write_private_csv(d / "structured" / f"{k}.csv", df)
    for i, n in enumerate(p.notes, 1):
        write_private(d / "notes" / f"note_{i:03d}.txt", f"{_t(n['rel_h'], n['date_only'])} | {n['title']}\n\n{n['text']}\n")
    if manifest_root is not None:
        write_private(Path(manifest_root) / f"{p.case_id}.json", json.dumps(
            {"case_id": p.case_id, "notes_status": p.notes_status, "counts": p.counts, "dropped": p.dropped}, indent=2))


# ===================================================================================================== pseudonyms, assignments
def check_char(digits: str) -> str:
    return chr(65 + (int(digits) * 7 + 3) % 26)


def valid_case_id(cid: str) -> bool:
    return bool(re.fullmatch(r"C\d{4}[A-Z]", cid)) and cid[-1] == check_char(cid[1:5])


def make_case_ids(n: int, seed: int) -> list[str]:
    """``n`` unique pseudonymous case ids ``C####X`` (random 4-digit number + check letter), reproducible from ``seed``."""
    if n > 8000:
        raise ValueError("too many cases for the id space")
    nums = np.random.default_rng([int(seed), 7]).permutation(np.arange(1000, 9000))[:n]
    return [f"C{v:04d}{check_char(f'{v:04d}')}" for v in nums]


def make_assignments(id_map: pd.DataFrame, raters=("R1", "R2"), seed: int = 0, batch_size: int = 20) -> pd.DataFrame:
    """Per rater an independent random order: calibration cases first (batch 0); the duplicate packets are shown only after at least
    ``batch_size`` other packets have followed their originals.  Columns: rater, order, batch, case_id."""
    cal = id_map[id_map["role"] == "calibration"]["case_id"].tolist()
    orig = id_map[id_map["role"] == "sample"]["case_id"].tolist()
    dup = id_map[id_map["role"] == "duplicate"]
    has_dup = set(dup["duplicate_of_case"])
    rows = []
    for r in raters:
        rng = np.random.default_rng([int(seed), 11, sum(ord(c) for c in r)])
        d_orig = [c for c in orig if c in has_dup]
        others = [c for c in orig if c not in has_dup]
        first_n = max(len(d_orig), (len(orig) + 1) // 2)
        rng.shuffle(others)
        pool1 = d_orig + others[:max(first_n - len(d_orig), 0)]
        pool2 = others[max(first_n - len(d_orig), 0):]
        rng.shuffle(pool1)
        rng.shuffle(pool2)
        base = pool1 + pool2
        lo = min(len(pool1) + batch_size, len(base))
        pos = sorted(rng.integers(lo, len(base) + 1, size=len(dup)).tolist())
        dcases = dup["case_id"].tolist()
        rng.shuffle(dcases)
        seq, j = [], 0
        for i, c in enumerate(base):
            while j < len(pos) and pos[j] <= i:
                seq.append(dcases[j]); j += 1
            seq.append(c)
        seq.extend(dcases[j:])
        for k, c in enumerate(cal + seq):
            rows.append({"rater": r, "order": k + 1, "batch": 0 if k < len(cal) else 1 + (k - len(cal)) // batch_size, "case_id": c})
    return pd.DataFrame(rows)
