"""Study 2 deployable-hardware montage simulation (see docs/research/montage_geometries.md)."""

from .select import ElectrodeSelection, select_electrode_subset, select_within_folds
from .simulate import (
    Geometry,
    LowConfidenceGeometryError,
    MontageRecording,
    canonical_name,
    electrode_positions,
    load_geometries,
    simulate_all,
    simulate_geometry,
    spherical_spline_matrix,
    spline_interpolate,
)

__all__ = [
    "ElectrodeSelection", "select_electrode_subset", "select_within_folds",
    "Geometry", "LowConfidenceGeometryError", "MontageRecording", "canonical_name",
    "electrode_positions", "load_geometries", "simulate_all", "simulate_geometry",
    "spherical_spline_matrix", "spline_interpolate",
]
