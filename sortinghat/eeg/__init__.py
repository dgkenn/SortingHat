"""Study 1 EEG front end: EDF reading, windowing/QC, preprocessing and qEEG features.

All code here is developed and tested on synthetic signals only (CLAUDE.md rule 1). Real HEEDB EDFs
are processed by a human from a plain terminal; anything leaving those jobs is aggregate-only.
"""

from .io import (CANONICAL_19, DEFAULT_MINIMUM_CHANNELS, EDFError, Recording, check_channel_set,
                 normalize_channel_name, read_edf, read_edf_header, select_channels)

__all__ = ["CANONICAL_19", "DEFAULT_MINIMUM_CHANNELS", "EDFError", "Recording", "check_channel_set",
           "normalize_channel_name", "read_edf", "read_edf_header", "select_channels"]
