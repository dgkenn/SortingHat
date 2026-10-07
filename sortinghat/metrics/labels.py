"""Label-set constants for the Study 1 primary endpoint.

Source: docs/research_plan_v1.txt, "Revised ontology" and "Primary endpoint".
The primary label set excludes E3 (partly EEG-defined, so it is the pipeline's
positive control) and E4b (iatrogenic sedation, known to the team at t0).
E7 enters the primary set only with >= 100 gold positives across >= 2 sites.
"""

from __future__ import annotations

from typing import Sequence

PRIMARY_LABELS: tuple[str, ...] = ("E1", "E2", "E4a", "E5", "E6")
CONDITIONAL_PRIMARY_LABEL = "E7"
POSITIVE_CONTROL_LABEL = "E3"
COVARIATE_LABEL = "E4b"
ALL_LABELS: tuple[str, ...] = ("E1", "E2", "E3", "E4a", "E4b", "E5", "E6", "E7")

E7_MIN_POSITIVES = 100
E7_MIN_SITES = 2


def e7_eligible(positives_by_site: dict[str, int]) -> bool:
    """E7 is primary only with >= 100 gold positives across >= 2 sites.

    "Across >= 2 sites" is read as: positives (summed) >= 100 and at least two
    sites contribute at least one positive. Operationalization recorded in the SAP.
    """
    total = sum(int(v) for v in positives_by_site.values())
    n_sites = sum(1 for v in positives_by_site.values() if int(v) > 0)
    return total >= E7_MIN_POSITIVES and n_sites >= E7_MIN_SITES


def primary_label_indices(label_names: Sequence[str], include_e7: bool = False) -> list[int]:
    """Column indices of the primary-endpoint labels within ``label_names``."""
    wanted = set(PRIMARY_LABELS) | ({CONDITIONAL_PRIMARY_LABEL} if include_e7 else set())
    idx = [i for i, name in enumerate(label_names) if name in wanted]
    if not idx:
        raise ValueError("no primary labels found in label_names")
    return idx
