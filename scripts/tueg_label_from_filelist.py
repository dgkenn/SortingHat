#!/usr/bin/env python3
"""TUEG report-keyword labels via the NEDC keyword-script service (frozen-model transport test; docs/tueg_count_query.md).

NEDC no longer distributes the TUEG reports; it runs a user-supplied standard-library Python script over the anonymized reports
and returns the matching file names (https://isip.piconepress.com/projects/nedc/html/tuh_eeg/).  This tool has three parts:

  emit-queries   configs/tueg_keywords.yaml -> standalone stdlib-only scripts in the NEDC form (input: a file listing report
                 filenames, argv[1]; output: matching filenames on stdout), plus one "profile" script that prints COUNTS only
                 (labels, label x setting, overlaps).  Each script is hashed in a manifest.
  build          NEDC-returned file lists (one text file per set, named <SET>.txt) -> a session-keyed label table with
                 mutual-exclusion and conflict handling, and an aggregate summary with the stop-rule verdict.
  profile-check  parse and sanity-check the counts text returned by the profile script (aggregate only).

The label table is record-level (de-identified TUEG IDs) and is written only under a local_only/ directory, mode 0600, and
is never printed (CLAUDE.md rule 5).  Everything printed or written to JSON is aggregate with counts < 11 shown as "<11".
No report text is ever read by this tool, and none by an agent: NEDC returns file names only.

TUEG path convention (verified 2026-10-14 against the NEDC TUEG v2.0.2 AAREADME and the TUH corpus pages):
    edf/<group 000>/<subject, 8 chars>/<session s001_YYYY_MM_DD>/<montage 01_tcp_ar>/<subject>_s001_t000.edf
  session numbers are chronological per subject, the date in the folder name "is often not precise", t000, t001, ... are tokens
  (EDF files) of one session.  The location and name of the report file within a session are NOT documented on the NEDC pages
  (unverified); the parser therefore accepts any path that carries the subject and session, with or without a token.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from sortinghat.adjudication.private_io import require_local_only, write_private_csv  # noqa: E402
from sortinghat.safe_output import SUPPRESSED, safe_print, safe_write_json, suppress_count  # noqa: E402

DEFAULT_CONFIG = HERE.parent / "configs" / "tueg_keywords.yaml"
PRIMARY_DEFAULT = ("E1", "E2", "E5")
ACUTE_SETS = ("SETTING_ICU", "SETTING_ED", "SETTING_ADMIT")
OTHER_SETTING_SETS = {"clinic": "SETTING_CLINIC", "emu": "SETTING_EMU"}


# =============================================================================================================================
# 1. TUEG path parsing
# =============================================================================================================================
_SUBJ = r"[a-z0-9]{8}"
_SESS_DIR = re.compile(r"^s(?P<n>\d{3})(?:_(?P<y>\d{4})(?:_(?P<m>\d{2})_(?P<d>\d{2}))?)?$")
_FNAME = re.compile(rf"^(?P<subj>{_SUBJ})_s(?P<n>\d{{3}})(?:_t(?P<tok>\d{{3}}))?(?:\.[A-Za-z0-9]+)?$")
_SUBJ_DIR = re.compile(rf"^{_SUBJ}$")


def parse_tueg_path(line: str) -> dict | None:
    """Parse one listed path.  Returns subject, session number, year, token, or None when no subject+session can be found.

    Accepts EDF or report paths, full or relative, with or without the group / date / montage folders.  A path whose folder and
    file name disagree on subject or session is rejected (None) rather than guessed.
    """
    s = line.strip().strip('"').strip("'")
    if not s or s.startswith("#"):
        return None
    parts = [p for p in re.split(r"[\\/]+", s) if p]
    subj_f = sess_f = tok = year = None
    last = parts[-1]
    m = _FNAME.match(last)
    if m:
        subj_f, sess_f, tok = m["subj"], int(m["n"]), (int(m["tok"]) if m["tok"] is not None else None)
    subj_d = sess_d = None
    for i, comp in enumerate(parts):
        md = _SESS_DIR.match(comp)
        if md:
            sess_d = int(md["n"])
            year = int(md["y"]) if md["y"] else None
            if i > 0 and _SUBJ_DIR.match(parts[i - 1]):
                subj_d = parts[i - 1]
            break
    if subj_f is None and subj_d is None:
        return None
    if subj_f and subj_d and subj_f != subj_d:
        return None
    if sess_f is not None and sess_d is not None and sess_f != sess_d:
        return None
    subject = subj_f or subj_d
    sess = sess_f if sess_f is not None else sess_d
    if subject is None or sess is None:
        return None
    return {"subject": subject, "session": int(sess), "year": year, "token": tok, "session_id": f"{subject}_s{int(sess):03d}"}


def read_list(path: Path) -> tuple[dict[str, dict], int, int]:
    """Read a NEDC-returned list.  Returns ({session_id: parsed}, n_lines_parsed, n_lines_rejected).  Duplicates collapse."""
    sessions: dict[str, dict] = {}
    n_ok = n_bad = 0
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        p = parse_tueg_path(raw)
        if p is None:
            n_bad += 1
            continue
        n_ok += 1
        cur = sessions.get(p["session_id"])
        if cur is None:
            sessions[p["session_id"]] = {"subject": p["subject"], "session": p["session"], "year": p["year"], "n_files": 1}
        else:
            cur["n_files"] += 1
            if cur["year"] is None:
                cur["year"] = p["year"]
    return sessions, n_ok, n_bad


# =============================================================================================================================
# 2. Query script emission (standard library only, NEDC form)
# =============================================================================================================================
QUERY_TEMPLATE = r'''#!/usr/bin/env python
# Generated from configs/tueg_keywords.yaml by scripts/tueg_label_from_filelist.py emit-queries.   __HEADER__
# Standard library only.  Usage:  python <this file> <file containing a list of report file names>
# Mode "list": prints the names of the files whose CLINICAL HISTORY / MEDICATIONS text matches (NEDC's documented interface).
# Mode "profile": prints COUNTS only (no file names) as tab-separated lines.
# Only the CLINICAL HISTORY and MEDICATIONS sections are read.  IMPRESSION / CLINICAL CORRELATION / DESCRIPTION are never searched.
import json
import re
import sys

CFG = json.loads(__CFG__)
MODE = CFG["mode"]


def term_regex(term):
    toks = [t for t in re.split(r"[\s\-]+", term.strip()) if t]
    body = r"[\s\-]+".join(re.escape(t).replace(r"\*", r"\w*") for t in toks)
    pre = r"(?<![A-Za-z0-9])" if term.strip()[:1].isalnum() else ""
    post = r"(?![A-Za-z0-9])" if (term.strip()[-1:].isalnum() or term.strip()[-1:] == "*") else ""
    return pre + body + post


def compile_cues(items, default_w):
    out = []
    for it in items:
        if isinstance(it, dict):
            out.append((re.compile(term_regex(it["p"]), re.I), int(it.get("w", default_w))))
        else:
            out.append((re.compile(term_regex(it), re.I), default_w))
    return out


def compile_spec(spec):
    if not isinstance(spec, dict):
        spec = {"t": spec}
    rx = re.compile(term_regex(spec["t"]), 0 if spec.get("cs") else re.I)
    req = re.compile(spec["requires"], re.I) if spec.get("requires") else None
    return rx, req


DW = CFG["cues"]["default_window"]
BREAKERS = [b.lower() for b in CFG["cues"]["breakers"]]
NEG = compile_cues(CFG["cues"]["negation"] + CFG["cues"]["family"], DW)
POST = compile_cues(CFG["cues"]["post_negation"], 4)
POST_UNC = compile_cues(CFG["cues"]["post_uncertain"], 4)
POST_REM = compile_cues(CFG["cues"]["post_remote"], 6)
CUE_LISTS = dict((k, compile_cues(CFG["cues"][k], DW)) for k in ("uncertain", "remote", "historical"))

SETS = {}
for name, s in CFG["sets"].items():
    c = dict(s)
    c["specs"] = [compile_spec(t) for t in s.get("terms", [])]
    c["excl"] = [re.compile(x, re.I) for x in s.get("exclude_context", [])]
    SETS[name] = c

USE = dict((k.upper(), v) for k, v in CFG["sections"]["use"].items())
KNOWN = sorted(set(list(USE) + [h.upper() for h in CFG["sections"]["never_search"]]), key=len, reverse=True)
KNOWN_RX = re.compile(r"^\s*(" + "|".join(re.escape(k) for k in KNOWN) + r")(?:\s*:\s*|\s*$)(.*)$", re.I)
GEN_RX = re.compile(r"^\s*([A-Z][A-Z0-9 /&\-\(\)]{5,60}?)\s*:\s*(.*)$")
AGE_RX = [re.compile(r"\b(\d{1,3})[\s\-]*(?:year|yr|y)s?[\s\-]*old\b", re.I),
          re.compile(r"\b(\d{1,3})[\s\-]*(?:yo|y/o|y\.o\.|yrs?\.? old)\b", re.I)]


def sections(text):
    cur = None
    buf = dict((v, []) for v in set(USE.values()))
    for line in text.splitlines():
        m = KNOWN_RX.match(line)
        if m:
            cur = USE.get(m.group(1).upper())
            if cur and m.group(2).strip():
                buf[cur].append(m.group(2))
            continue
        g = GEN_RX.match(line)
        if g and g.group(1).strip() not in USE:
            cur = None
            continue
        if cur is not None:
            buf[cur].append(line)
    return dict((k, "\n".join(v)) for k, v in buf.items()), dict((k, bool(v)) for k, v in buf.items())


def clauses(text):
    out = []
    for para in re.split(r"\n\s*\n", text):
        flat = " ".join(para.split())
        if flat:
            out.extend(x for x in re.split(r"(?<=[.;])\s+", flat) if x)
    return out


def cued(prefix_text, cues):
    toks = prefix_text.split()
    for rx, w in cues:
        if rx.search(" ".join(toks[-w:])):
            return True
    return False


def prefix_after_breaker(clause, start):
    pad = " " + clause[:start]
    low = pad.lower()
    cut = 0
    for b in BREAKERS:
        i = low.rfind(b)
        if i >= 0:
            cut = max(cut, i + len(b))
    return pad[cut:]


def eval_terms(secs, s):
    policy = s.get("policy", {})
    cues = list(NEG)
    for k in ("uncertain", "remote", "historical"):
        if policy.get(k, "ignore") == "exclude":
            cues = cues + CUE_LISTS[k]
    post = POST + (POST_UNC if policy.get("uncertain", "ignore") == "exclude" else [])
    for sec in s.get("scope", CFG["sections"]["default_scope"]):
        for cl in clauses(secs.get(sec, "")):
            low = cl.lower()
            if any(x.search(low) for x in s["excl"]):
                continue
            for rx, req in s["specs"]:
                for m in rx.finditer(cl):
                    if req is not None and not req.search(low):
                        continue
                    if cued(prefix_after_breaker(cl, m.start()), cues):
                        continue
                    if cued(" ".join(cl[m.end():].split()[:4]), post):
                        continue
                    if policy.get("remote", "ignore") == "exclude" and cued(" ".join(cl[m.end():].split()[:6]), POST_REM):
                        continue
                    return True
    return False


def first_age(text):
    for rx in AGE_RX:
        m = rx.search(text)
        if m:
            return int(m.group(1))
    return None


def evaluate(text):
    secs, present = sections(text)
    res = {}
    for name, s in SETS.items():
        if "section" in s:
            res[name] = bool(present.get(s["section"]))
        elif "age" in s:
            a = first_age(secs.get("history", ""))
            res[name] = a is not None and s["age"][0] <= a <= s["age"][1]
        else:
            res[name] = eval_terms(secs, s)
    return res, present


def session_key(fname):
    m = re.search(r"([a-z0-9]{8})_s(\d{3})", fname)
    return m.group(1) + "_s" + m.group(2) if m else fname


def read_text(fname):
    try:
        with open(fname, "r", encoding="utf8", errors="replace") as fh:
            return fh.read()
    except (IOError, OSError):
        return None


def main(argv):
    files = [line.rstrip("\n") for line in open(argv[1], "r", encoding="utf8") if line.strip()]
    if MODE == "list":
        name = next(iter(SETS))
        for fname in files:
            text = read_text(fname)
            if text is not None and evaluate(text)[0][name]:
                print(fname)
        return
    prof = CFG["profile"]
    sess = dict((n, set()) for n in SETS)
    flags = {}
    n_files = n_read = 0
    for fname in files:
        n_files += 1
        text = read_text(fname)
        if text is None:
            continue
        n_read += 1
        res, present = evaluate(text)
        k = session_key(fname)
        agg = flags.setdefault(k, dict((n, False) for n in SETS))
        for n, v in res.items():
            agg[n] = agg[n] or v
    for k, agg in flags.items():
        for n, v in agg.items():
            if v:
                sess[n].add(k)
    print("PROFILE\tmode=counts_only\tvariant=" + CFG["variant"])
    print("N\tfiles_listed\t%d" % n_files)
    print("N\tfiles_read\t%d" % n_read)
    print("N\tsessions\t%d" % len(flags))
    for n in SETS:
        print("SET\t%s\t%d" % (n, len(sess[n])))
    acute = sess["SETTING_ICU"] | sess["SETTING_ED"] | sess["SETTING_ADMIT"]
    anyset = acute | sess["SETTING_CLINIC"] | sess["SETTING_EMU"]
    setting_groups = {"ACUTE": acute, "NONE": set(flags) - anyset}
    for n in ("SETTING_ICU", "SETTING_ED", "SETTING_ADMIT", "SETTING_CLINIC", "SETTING_EMU"):
        setting_groups[n] = sess[n]
    labs = prof["labels"] + prof["controls"]
    for lab in labs:
        for sg in prof["cross_settings"]:
            print("LABEL_X_SETTING\t%s\t%s\t%d" % (lab, sg, len(sess[lab] & setting_groups[sg])))
    prim = prof["primary"]
    for lab in prim:
        others = set()
        for o in prim:
            if o != lab:
                others |= sess[o]
        pure = sess[lab] - others
        print("PURE\t%s\tall\t%d" % (lab, len(pure)))
        print("PURE\t%s\tACUTE\t%d" % (lab, len(pure & acute)))
    for i, a in enumerate(prim):
        for b in prim[i + 1:]:
            print("OVERLAP\t%s\t%s\t%d" % (a, b, len(sess[a] & sess[b])))
    for a in prim:
        print("OVERLAP\t%s\tE3\t%d" % (a, len(sess[a] & sess["E3"])))
    anylabel = set()
    for lab in prof["labels"]:
        anylabel |= sess[lab]
    ctrl = sess[prof["controls"][0]] - anylabel
    print("CONTROL\tclean\tall\t%d" % len(ctrl))
    print("CONTROL\tclean\tACUTE\t%d" % len(ctrl & acute))


if __name__ == "__main__":
    main(sys.argv)
'''


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if cfg.get("schema_version") != 1:
        raise SystemExit("unsupported tueg_keywords.yaml schema_version")
    return cfg


def _all_sets(cfg: dict) -> dict[str, dict]:
    """Flatten labels / controls / settings / covariates / universe into {name: set-spec}."""
    out: dict[str, dict] = {}
    for grp in ("labels", "controls", "settings", "covariates"):
        for name, spec in (cfg.get(grp) or {}).items():
            if name in out:
                raise SystemExit(f"duplicate set name {name}")
            out[name] = dict(spec, group=grp)
    for name, spec in (cfg.get("universe") or {}).items():
        out[name] = dict(spec, group="universe")
    return out


def _runtime_set(spec: dict, variant: str, default_scope: list[str]) -> dict:
    """Spec reduced to what the generated script needs.  LOOSE ignores the uncertain / remote / historical cue lists.
    Settings and covariates use negation / family cues only; settings read the history section only."""
    if "section" in spec:
        return {"section": spec["section"]}
    if "age" in spec:
        return {"age": list(spec["age"])}
    pol = dict(spec.get("policy") or {})
    scope = list(spec.get("scope") or default_scope)
    if spec.get("group") in ("settings", "covariates"):
        pol = {}
    if spec.get("group") == "settings":
        scope = ["history"]
    if variant == "loose":
        pol = {k: "ignore" for k in ("uncertain", "remote", "historical")}
    return {"terms": list(spec.get("terms") or []), "exclude_context": list(spec.get("exclude_context") or []), "policy": pol,
            "scope": scope}


def build_runtime_cfg(cfg: dict, names: list[str], variant: str, mode: str) -> dict:
    sets = _all_sets(cfg)
    unknown = [n for n in names if n not in sets]
    if unknown:
        raise SystemExit(f"unknown set(s) in query plan: {unknown}")
    default_scope = list(cfg["sections"]["default_scope"])
    rt_sets = {}
    for n in names:
        rt_sets[n] = _runtime_set(sets[n], variant, default_scope)
    cues = cfg["cues"]
    out = {
        "mode": mode, "variant": variant,
        "sections": {"use": cfg["sections"]["use"], "never_search": cfg["sections"]["never_search"],
                     "default_scope": default_scope},
        "cues": {"default_window": cues["default_window"], "breakers": cues["breakers"], "negation": cues["negation"],
                 "family": cues["family"], "post_negation": cues["post_negation"], "post_uncertain": cues["post_uncertain"], "post_remote": cues["post_remote"],
                 "uncertain": cues["uncertain"],
                 "remote": cues["remote"], "historical": cues["historical"]},
        "sets": rt_sets,
    }
    return out


def render_script(runtime_cfg: dict, header: str) -> str:
    cfg_json = json.dumps(runtime_cfg, sort_keys=True, ensure_ascii=True)
    return QUERY_TEMPLATE.replace("__HEADER__", header).replace("__CFG__", repr(cfg_json))


def emit_queries(cfg: dict, out_dir: Path, rounds: tuple[str, ...] = ("profile", "round1", "round2_strict", "round2_loose")) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = cfg["query_plan"]
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]
    manifest = {"config_sha256_16": cfg_hash, "as_of": cfg.get("as_of"), "scripts": []}

    def add(fname: str, text: str, kind: str, set_name: str, variant: str, rnd: str):
        (out_dir / fname).write_text(text, encoding="utf-8")
        manifest["scripts"].append({"file": fname, "round": rnd, "kind": kind, "set": set_name, "variant": variant,
                                    "sha256": hashlib.sha256(text.encode()).hexdigest()})

    sets = _all_sets(cfg)
    if "profile" in rounds:
        prof = plan["profile"]
        names = list(prof["labels"]) + list(prof["controls"]) + [n for n in sets if sets[n]["group"] == "settings"]
        names = list(dict.fromkeys(names + ["UNIVERSE_HISTORY"]))
        rt = build_runtime_cfg(cfg, names, "strict", "profile")
        rt["profile"] = {"labels": list(prof["labels"]), "controls": list(prof["controls"]),
                         "cross_settings": list(prof["cross_settings"]), "primary": list(cfg["stop_rule"]["primary_labels"])}
        add("tueg_query_PROFILE_strict.py", render_script(rt, f"cfg={cfg_hash}; round=0 (counts only)"), "profile", "profile", "strict", "profile")
    for rnd, variant in (("round1", "strict"), ("round2_strict", "strict"), ("round2_loose", "loose")):
        if rnd not in rounds:
            continue
        for n in plan[rnd]:
            if sets[n].get("query") is False:
                continue
            rt = build_runtime_cfg(cfg, [n], variant, "list")
            add(f"tueg_query_{n}_{variant}.py", render_script(rt, f"cfg={cfg_hash}; set={n}; variant={variant}"), "list", n, variant, rnd)
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


# =============================================================================================================================
# 3. Label table from NEDC-returned lists
# =============================================================================================================================
def _partition(total: int, parts: dict[str, int]) -> dict:
    """Suppress counts < 11; if any part is suppressed the total is hidden too (no recovery by subtraction)."""
    shown = {k: suppress_count(v) for k, v in parts.items()}
    hide_total = any(v == SUPPRESSED for v in shown.values())
    return {"total": SUPPRESSED if hide_total and total >= 11 else suppress_count(total), **shown}


def build_table(lists: dict[str, dict[str, dict]], cfg: dict, *, primary: tuple[str, ...] = PRIMARY_DEFAULT,
                subject_conflict: str = "drop_control") -> pd.DataFrame:
    """One row per session that appears in any list: flags per set, derived setting, role and conflict reasons.

    Roles (a session gets exactly one):
      positive_pure    exactly one of the primary labels (E1, E2, E5) and nothing else conflicting
      positive_multi   two or more primary labels (kept out of the primary contrast; sensitivity only)
      e3_positive      E3 (strict) with no primary label and no other label hit (the positive control)
      control          CTRL hit and no hit on ANY label set that was obtained (E1, E2, E5, E3, E3_broad, E5_nonspecific, E4a, E6, E7)
      excluded_*       a control dropped because of a label hit or because its subject has a positive / E3 session
      unlabelled       appears only in setting / covariate / universe lists
    """
    if subject_conflict not in ("drop_control", "keep"):
        raise ValueError("subject_conflict must be drop_control or keep")
    sets = _all_sets(cfg)
    rows: dict[str, dict] = {}
    for sess in lists.values():
        for sid, rec in sess.items():
            r = rows.setdefault(sid, {"session_id": sid, "subject_id": rec["subject"], "session_year": rec["year"], "n_files": 0})
            if r["session_year"] is None:
                r["session_year"] = rec["year"]
            r["n_files"] = max(r["n_files"], rec["n_files"])
    cols = ["session_id", "subject_id", "session_year", "n_files", "role", "primary_label", "setting", "conflicts"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(list(rows.values()))
    for name, sess in lists.items():
        df[name] = df["session_id"].isin(set(sess))
    for n in (*primary, "E3", "CTRL", *ACUTE_SETS, *OTHER_SETTING_SETS.values()):
        if n not in df:
            df[n] = False

    acute = df[list(ACUTE_SETS)].any(axis=1).to_numpy()
    outpt, emu = df["SETTING_CLINIC"].to_numpy(), df["SETTING_EMU"].to_numpy()
    n_kinds = acute.astype(int) + outpt.astype(int) + emu.astype(int)
    setting = ["unknown"] * len(df)
    for i in range(len(df)):
        if n_kinds[i] >= 2:
            setting[i] = "mixed"
        elif acute[i]:
            setting[i] = "acute"
        elif outpt[i]:
            setting[i] = "clinic"
        elif emu[i]:
            setting[i] = "emu"

    n_prim = df[list(primary)].sum(axis=1).to_numpy()
    e3 = df["E3"].to_numpy()
    ctrl = df["CTRL"].to_numpy()
    label_cols = [n for n in lists if n in sets and sets[n]["group"] == "labels"]
    other_cols = [n for n in label_cols if n not in primary and n != "E3"]
    other_not_broad = [n for n in other_cols if n != "E3_broad"]
    any_other = df[other_cols].any(axis=1).to_numpy() if other_cols else np.zeros(len(df), bool)
    any_other_not_broad = df[other_not_broad].any(axis=1).to_numpy() if other_not_broad else np.zeros(len(df), bool)
    in_univ = df["UNIVERSE_HISTORY"].to_numpy() if "UNIVERSE_HISTORY" in df else np.ones(len(df), bool)

    role = ["unlabelled"] * len(df)
    prim_label = [""] * len(df)
    conflicts: list[list[str]] = [[] for _ in range(len(df))]
    for i in range(len(df)):
        c = conflicts[i]
        if n_prim[i] >= 2:
            c.append("multi_primary")
        if e3[i] and n_prim[i] >= 1:
            c.append("e3_overlap_primary")
        if ctrl[i] and (n_prim[i] >= 1 or e3[i] or any_other[i]):
            c.append("control_has_label_hit")
        if setting[i] == "mixed":
            c.append("setting_conflict")
        if not in_univ[i]:
            c.append("outside_universe")
        if n_prim[i] == 1:
            role[i] = "positive_pure"
            prim_label[i] = next(p for p in primary if df[p].iloc[i])
        elif n_prim[i] >= 2:
            role[i] = "positive_multi"
        elif e3[i] and not any_other_not_broad[i]:
            role[i] = "e3_positive"
        elif ctrl[i] and not e3[i] and not any_other[i]:
            role[i] = "control"
        elif ctrl[i]:
            role[i] = "excluded_control_label_hit"
    subj = df["subject_id"].to_numpy()
    pos_subj = {subj[i] for i in range(len(df)) if role[i] in ("positive_pure", "positive_multi", "e3_positive")}
    lab_by_subj: dict[str, set] = defaultdict(set)
    for i in range(len(df)):
        if role[i] == "positive_pure":
            lab_by_subj[subj[i]].add(prim_label[i])
    for i in range(len(df)):
        if role[i] == "control" and subj[i] in pos_subj:
            conflicts[i].append("subject_has_positive_session")
            if subject_conflict == "drop_control":
                role[i] = "excluded_subject_conflict"
        if role[i] == "positive_pure" and len(lab_by_subj[subj[i]]) >= 2:
            conflicts[i].append("subject_multi_label")
    df["role"], df["primary_label"], df["setting"] = role, prim_label, setting
    df["conflicts"] = ["|".join(c) for c in conflicts]
    return df[cols + [c for c in df.columns if c not in cols]]


def summarize(df: pd.DataFrame, lists: dict, rejects: dict, cfg: dict, *, min_pos: int | None = None, ratio: int | None = None) -> dict:
    sr = cfg.get("stop_rule", {})
    min_pos = min_pos if min_pos is not None else int(sr.get("min_positives_acute", 50))
    ratio = ratio if ratio is not None else int(sr.get("min_controls_per_positive", 3))
    primary = list(sr.get("primary_labels", PRIMARY_DEFAULT))
    min_labels = int(sr.get("min_labels_passing", 2))
    out: dict = {"n_sessions_total": suppress_count(len(df)), "n_subjects_total": suppress_count(df["subject_id"].nunique() if len(df) else 0),
                 "n_per_list": {k: suppress_count(len(v)) for k, v in lists.items()},
                 "lines_rejected_unparseable": {k: suppress_count(v) for k, v in rejects.items()}}
    ctrl = df[df["role"] == "control"]
    ctrl_acute = int((ctrl["setting"] == "acute").sum())
    per_label = {}
    passing = []
    for lab in primary:
        pure = df[(df["role"] == "positive_pure") & (df["primary_label"] == lab)]
        n_pure = len(pure)
        n_pure_acute = int((pure["setting"] == "acute").sum())
        n_any = int(df[lab].sum()) if lab in df else 0
        by_setting = {s: int((pure["setting"] == s).sum()) for s in ("acute", "clinic", "emu", "mixed", "unknown")}
        ok = n_pure_acute >= min_pos
        ctrl_ok = ctrl_acute >= ratio * n_pure_acute and ctrl_acute > 0
        if ok and ctrl_ok:
            passing.append(lab)
        per_label[lab] = {
            "n_any_hit": suppress_count(n_any),
            "pure_by_setting": _partition(n_pure, by_setting),
            "pure_acute_ge_threshold": bool(ok),
            "acute_controls_ge_ratio_x_positives": bool(ctrl_ok),
            "label_testable": bool(ok and ctrl_ok),
        }
    out["per_label"] = per_label
    out["controls"] = _partition(len(ctrl), {s: int((ctrl["setting"] == s).sum()) for s in ("acute", "clinic", "emu", "mixed", "unknown")})
    out["e3_positive_control"] = _partition(int((df["role"] == "e3_positive").sum()),
                                           {s: int(((df["role"] == "e3_positive") & (df["setting"] == s)).sum()) for s in ("acute", "clinic", "emu", "mixed", "unknown")})
    out["roles"] = {r: suppress_count(int(n)) for r, n in df["role"].value_counts().items()}
    conf: dict[str, int] = defaultdict(int)
    for c in df["conflicts"]:
        for t in filter(None, c.split("|")):
            conf[t] += 1
    out["conflict_flags"] = {k.replace("subject", "subj").replace("session", "sess"): suppress_count(v) for k, v in sorted(conf.items())}
    out["stop_rule"] = {
        "min_positives_acute": min_pos, "min_controls_per_positive": ratio, "min_labels_passing": min_labels,
        "labels_passing": passing, "n_labels_passing": len(passing),
        "route_continues": len(passing) >= min_labels,
        "stop_if_any_primary_below": any(not per_label[l]["label_testable"] for l in primary),
        "reading": ("A label below the threshold is dropped from the primary analysis.  The route stops when fewer than "
                    f"{min_labels} of {len(primary)} primary labels are testable (survey: stop if all are below; the interpretation rule "
                    "needs at least two).  'stop_if_any_primary_below' is the stricter reading of the brief and is reported for the lead."),
    }
    return out


def cmd_emit(a) -> int:
    cfg = load_config(a.config)
    rounds = tuple(x for x in a.rounds.split(",") if x)
    man = emit_queries(cfg, Path(a.out_dir), rounds)
    safe_print(f"wrote {len(man['scripts'])} scripts to {a.out_dir}; config hash {man['config_sha256_16']}")
    return 0


def cmd_build(a) -> int:
    cfg = load_config(a.config)
    primary = tuple(cfg["stop_rule"]["primary_labels"])
    lists: dict[str, dict] = {}
    rejects: dict[str, int] = {}
    ldir = Path(a.lists_dir)
    sets = _all_sets(cfg)
    for name in sets:
        p = ldir / f"{name}.txt"
        if p.exists():
            sess, n_ok, n_bad = read_list(p)
            lists[name] = sess
            rejects[name] = n_bad
    required = list(primary) + ["CTRL"]
    missing = [n for n in required if n not in lists]
    if missing and not a.allow_missing:
        raise SystemExit(f"missing list file(s) for: {missing} (expected {ldir}/<SET>.txt); use --allow-missing for a partial table")
    df = build_table(lists, cfg, primary=primary, subject_conflict=a.subject_conflict)
    summary = summarize(df, lists, rejects, cfg, min_pos=a.min_positives)
    if a.table_out:
        require_local_only(a.table_out, "--table-out")
        write_private_csv(a.table_out, df)
        summary["table_written"] = True
    safe_write_json(a.summary_out, summary)
    safe_print(json.dumps(summary["stop_rule"]["labels_passing"]), "labels testable;", "route_continues =", summary["stop_rule"]["route_continues"])
    return 0


_PROFILE_LINE = re.compile(r"^(SET|PURE|OVERLAP|CONTROL|LABEL_X_SETTING|N)\t")


def parse_profile(text: str) -> dict:
    """Parse the counts text returned by the profile script.  Counts only; anything else is ignored."""
    out: dict[str, dict] = defaultdict(dict)
    for line in text.splitlines():
        if not _PROFILE_LINE.match(line):
            continue
        parts = line.split("\t")
        try:
            out[parts[0]][tuple(parts[1:-1])] = int(parts[-1])
        except ValueError:
            continue
    return dict(out)


def cmd_profile_check(a) -> int:
    prof = parse_profile(Path(a.file).read_text(encoding="utf-8"))
    cfg = load_config(a.config)
    sr = cfg["stop_rule"]
    min_pos = int(sr["min_positives_acute"])
    ratio = int(sr["min_controls_per_positive"])
    pure_acute = {k[0]: v for k, v in prof.get("PURE", {}).items() if k[1] == "ACUTE"}
    ctrl_acute = prof.get("CONTROL", {}).get(("clean", "ACUTE"), 0)
    passing = [l for l in sr["primary_labels"] if pure_acute.get(l, 0) >= min_pos and ctrl_acute >= ratio * pure_acute.get(l, 0) and ctrl_acute > 0]
    res = {"pure_acute": {l: suppress_count(pure_acute.get(l, 0)) for l in sr["primary_labels"]},
           "clean_acute_controls": suppress_count(ctrl_acute), "labels_passing": passing,
           "route_continues": len(passing) >= int(sr["min_labels_passing"])}
    if a.summary_out:
        safe_write_json(a.summary_out, res)
    safe_print(json.dumps(res))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("emit-queries", help="write the NEDC query scripts")
    e.add_argument("--config", default=str(DEFAULT_CONFIG))
    e.add_argument("--out-dir", required=True)
    e.add_argument("--rounds", default="profile,round1,round2_strict,round2_loose")
    e.set_defaults(fn=cmd_emit)
    b = sub.add_parser("build", help="NEDC file lists -> session label table + aggregate summary")
    b.add_argument("--config", default=str(DEFAULT_CONFIG))
    b.add_argument("--lists-dir", required=True, help="directory holding <SET>.txt returned by NEDC")
    b.add_argument("--table-out", help="record-level CSV; must be under a local_only/ directory")
    b.add_argument("--summary-out", required=True, help="aggregate JSON (counts < 11 suppressed)")
    b.add_argument("--subject-conflict", choices=("drop_control", "keep"), default="drop_control")
    b.add_argument("--min-positives", type=int, default=None)
    b.add_argument("--allow-missing", action="store_true")
    b.set_defaults(fn=cmd_build)
    c = sub.add_parser("profile-check", help="evaluate the stop rule on the counts returned by the profile script")
    c.add_argument("--config", default=str(DEFAULT_CONFIG))
    c.add_argument("--file", required=True)
    c.add_argument("--summary-out")
    c.set_defaults(fn=cmd_profile_check)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
