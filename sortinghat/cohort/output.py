"""Writing the cohort outputs. The ONLY module in the package that writes files.

* ``<out>/local_only/cohort_study1.csv``     record-level cohort table (mode 0600, directory 0700)
* ``<out>/local_only/recording_keys.csv``    recording key list read by the streaming extractor (mode 0600)
* ``<out>/cohort/flow.md`` and ``flow.json`` aggregate flow report (small cells suppressed; checked with
  ``assert_aggregate_only`` against every known identifier before it is written)

``<out>`` defaults to ``out`` (gitignored as a whole; ``**/local_only/`` is also gitignored on its own).
Record-level files go through ``safe_output.write_local_only`` and are never printed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from ..safe_output import safe_write_json, safe_write_text, write_local_only
from .build import CohortResult
from .flow import flow_markdown

LOCAL_DIR = "local_only"
COHORT_FILE = "cohort_study1.csv"
KEYS_FILE = "recording_keys.csv"
PENDING_STEPS = [
    "EEG QC in the streaming extractor: minimum channel set present (Fp1, Fp2, F7, F8, T3, T4, T5, T6, O1, O2) and "
    ">= 60% usable data in minutes 1-11. Append its counts per site with the same suppression rules.",
    "Sensitivity windows (<= 6 / 12 / 48 h) are flags on the cohort table, not extra exclusion steps.",
]


def known_ids(result: CohortResult) -> set[str]:
    """Every record identifier of the build (person ids, session ids, BIDS folders), as strings, to prove none leaks."""
    t = result.table
    ids = set(t["person_id"].astype(str)) | set(t["SessionID"].astype(str)) | set(t["BidsFolder"].astype(str))
    return {i for i in ids if i and i not in {"nan", "None", "<NA>"}}


def _local_csv(df: pd.DataFrame, path: Path) -> Path:
    write_local_only(path, df.to_csv(index=False))
    os.chmod(path, 0o600)                                        # also when the file already existed
    return path


def write_outputs(result: CohortResult, out_root: str | Path = "out", debug_flow: bool = False) -> dict[str, Path]:
    root = Path(out_root)
    local = root / LOCAL_DIR
    local.mkdir(parents=True, exist_ok=True)
    os.chmod(local, 0o700)
    paths = {"cohort": _local_csv(result.table, local / COHORT_FILE),
             "keys": _local_csv(result.keys, local / KEYS_FILE)}
    ids = known_ids(result)
    paths["flow_md"] = safe_write_text(root / "cohort" / "flow.md",
                                       flow_markdown(result.report, PENDING_STEPS), ids)
    paths["flow_json"] = safe_write_json(root / "cohort" / "flow.json", result.report, ids)
    if debug_flow:                                   # unmerged diagnostic flow: for the analyst, not for sharing
        paths["flow_debug_md"] = safe_write_text(root / "cohort" / "flow_debug.md", flow_markdown(result.debug), ids)
        paths["flow_debug_json"] = safe_write_json(root / "cohort" / "flow_debug.json", result.debug, ids)
    return paths


def read_key_list(path: str | Path) -> pd.DataFrame:
    """Load the recording key list (for the streaming extractor; record-level, never print it)."""
    from .build import KEY_LIST_COLUMNS
    k = pd.read_csv(path, dtype={"SessionID": str, "BidsFolder": str, "EEGFolder": str, "SiteID": str})
    missing = [c for c in KEY_LIST_COLUMNS if c not in k]
    if missing:
        raise ValueError(f"key list lacks columns {missing}")
    return k
