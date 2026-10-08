"""MORGOTH 1.0 event-level heads: checkpoint file, output classes, and the ladder feature names derived from them.

Spec source: bdsp-core/morgoth ``README.md`` (class order "in the same order as each column in the output file"),
``finetune_classification.py`` (``get_dataset`` channel list, ``predict``: binary heads -> sigmoid, multi-class -> softmax)
and ``utils.py`` (preprocessing). Checkpoint file names and sizes are from the names-only probe of
``morgoth1/models/`` on the BDSP projects access point (docs/morgoth.md).

Binary heads emit P(positive class), the SECOND class in the README's "A / B" order (Normal / **Abnormal**,
No / **Burst suppression**, No / **Spike**). That polarity is read from the README, not verified against labelled
data (UNVERIFIED; docs/morgoth.md). Multi-class heads emit softmax over the README order; the first class (``none``,
``other``) is the reference and gets no feature column (it is 1 - the rest).
"""

from __future__ import annotations

from dataclasses import dataclass

# Channel order the model was trained with (finetune_classification.get_dataset; utils.eeg_channels1). Classical
# T3/T4/T5/T6. The model embeds each electrode by its index in utils.standard_1020 (+1; 0 is the cls token).
MORGOTH_CHANNELS: tuple[str, ...] = ("FP1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "FZ", "CZ", "PZ",
                                     "FP2", "F4", "C4", "P4", "F8", "T4", "T6", "O2")
# SortingHat canonical (mixed-case) names in the same order.
SORTINGHAT_ORDER: tuple[str, ...] = ("Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", "Pz",
                                     "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2")
assert [c.upper() for c in SORTINGHAT_ORDER] == list(MORGOTH_CHANNELS)

# Positions in utils.standard_1020 (0-based) of the 19 electrodes, i.e. input_chans[1:] - 1. Computed there from the
# list; pinned here so the TorchBackend does not need to import MORGOTH's utils (heavy deps). Verified by a test
# against the cached code when available.
STANDARD_1020_INDEX: dict[str, int] = {
    "FP1": 0, "FP2": 2, "F7": 15, "F3": 17, "FZ": 19, "F4": 21, "F8": 23, "C3": 39, "CZ": 41, "C4": 43,
    "P3": 61, "PZ": 63, "P4": 65, "O1": 80, "O2": 82, "T3": 88, "T5": 89, "T4": 90, "T6": 91}


@dataclass(frozen=True)
class HeadSpec:
    name: str                       # short key used in feature names
    checkpoint: str                 # file under morgoth1/models/
    classes: tuple[str, ...]        # README order; binary heads list the single positive class
    binary: bool
    window_s: float = 10.0          # model input length
    size_bytes: int = 0             # from the names-only probe
    default: bool = True            # fetched and run by default

    @property
    def n_out(self) -> int:
        return 1 if self.binary else len(self.classes)

    @property
    def feature_classes(self) -> tuple[str, ...]:
        """Classes that get a feature column (all of a binary head's positive class; multi-class minus the reference)."""
        return self.classes if self.binary else self.classes[1:]


HEADS: dict[str, HeadSpec] = {h.name: h for h in (
    HeadSpec("normal", "NORMAL.pth", ("abnormal",), True, size_bytes=70108410),
    HeadSpec("bs", "BS.pth", ("burst_suppression",), True, size_bytes=70108410),
    HeadSpec("spikes", "SPIKES.pth", ("spike",), True, window_s=1.0, size_bytes=70108410),
    HeadSpec("slowing", "SLOWING.pth", ("none", "focal", "generalized"), False, size_bytes=70113338),
    HeadSpec("spikeloc", "FOCGENSPIKES.pth", ("none", "focal", "generalized"), False, size_bytes=70113402),
    HeadSpec("iiic", "IIIC.pth", ("other", "seizure", "lpd", "gpd", "lrda", "grda"), False, size_bytes=70120570),
    HeadSpec("sleep3", "SLEEP.pth", ("awake", "n1", "n2"), False, size_bytes=70113274, default=False),
)}

# EEG-level heads (2.2 MB CNN+transformer over the per-second event-level sequence; EEG_level_head.py). Fetched for
# completeness, NOT run in v1 (they need the 1-s-step event sequence and the repo's CSV pipeline).
EEG_LEVEL_CHECKPOINTS: tuple[str, ...] = (
    "BS_EEGlevel.pth", "FOC_SLOWING_EEGlevel.pth", "FOC_SPIKES_EEGlevel.pth", "GEN_SLOWING_EEGlevel.pth",
    "GEN_SPIKES_EEGlevel.pth", "GPD_EEGlevel.pth", "GRDA_EEGlevel.pth", "LPD_EEGlevel.pth", "LRDA_EEGlevel.pth",
    "NORMAL_EEGlevel.pth", "SEIZURE_EEGlevel.pth", "SPIKES_EEGlevel.pth")

STATS = ("mean", "p90", "burden")        # per window: mean prob, 90th percentile, share of snippets with p >= 0.5
BURDEN_THRESHOLD = 0.5


def default_heads() -> tuple[str, ...]:
    return tuple(n for n, h in HEADS.items() if h.default)


def feature_names(heads=None, stats=STATS) -> list[str]:
    """Ladder column names (without the window): ``morgoth.<head>.<class>.<stat>``."""
    out = []
    for n in heads or default_heads():
        h = HEADS[n]
        out += [f"morgoth.{n}.{c}.{s}" for c in h.feature_classes for s in stats]
    return out
