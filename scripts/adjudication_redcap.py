#!/usr/bin/env python3
"""Gold adjudication pilot: generate the REDCap data dictionary from configs/adjudication_crf.yaml (D-157).   No patient data involved.

    python3 scripts/adjudication_redcap.py                         # writes docs/adjudication_redcap_dictionary.csv
    python3 scripts/adjudication_redcap.py --check                 # exit 1 if the committed CSV is stale

Upload in REDCap: Project Setup > Data Dictionary > Upload.  The form is one instance per rater x case (record = case_id; a repeating
instrument keyed by rater_id, or one record per case-rater pair: see docs/adjudication_pilot.md).  The CRF YAML is the single source of truth:
sortinghat/adjudication/crf.py reads the same file to parse the export in scripts/adjudication_analyze.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from sortinghat.adjudication import crf  # noqa: E402

DEFAULT_OUT = HERE.parent / "docs" / "adjudication_redcap_dictionary.csv"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crf", default=str(crf.DEFAULT_PATH))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--check", action="store_true", help="do not write; fail if --out differs from the regenerated dictionary")
    a = ap.parse_args(argv)
    cfg = crf.load_crf(a.crf)
    dd = crf.data_dictionary(cfg)
    text = dd.to_csv(index=False, lineterminator="\n")
    out = Path(a.out)
    if a.check:
        same = out.exists() and out.read_text(encoding="utf-8") == text
        print(f"{out}: {'up to date' if same else 'STALE or missing'} ({len(dd)} fields)")
        return 0 if same else 1
    crf.write_dictionary(out, cfg)
    print(f"wrote {out}: {len(dd)} fields, {int((dd['Required Field?'] == 'y').sum())} required, "
          f"{int((dd['Branching Logic (Show field only if...)'] != '').sum())} with branching logic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
