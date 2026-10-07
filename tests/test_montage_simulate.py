import numpy as np
import pytest

from sortinghat.montage import (
    LowConfidenceGeometryError, canonical_name, electrode_positions, load_geometries,
    simulate_all, simulate_geometry, spline_interpolate,
)

CH19 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4",
        "T6", "O1", "O2"]


def smooth_field(names, n=400, seed=0):
    """Low-order smooth spatial field (degree<=2 polynomial in x,y,z) with random time courses."""
    rng = np.random.default_rng(seed)
    p = electrode_positions(names)
    x, y, z = p.T
    modes = np.stack([x, y, z, x * y, x * z, y * z, x * x - y * y, 3 * z * z - 1])
    return modes.T @ rng.standard_normal((len(modes), n))


@pytest.fixture
def rec():
    rng = np.random.default_rng(1)
    return rng.standard_normal((19, 500))


def test_yaml_loads_and_aliases():
    g = load_geometries()
    assert g["full_19ch_1020"].confidence == "high" and len(g["full_19ch_1020"].derivations) == 18
    assert canonical_name("EEG T7-REF") == "T3" and canonical_name("p8") == "T6"


def test_full_19ch_round_trip(rec):
    out = simulate_geometry(rec, CH19, 200.0, "full_19ch_1020")
    assert out.electrode_names == CH19
    np.testing.assert_allclose(out.electrode_data, rec)
    assert not out.approximated and not out.warnings
    assert out.data.shape == (18, 500)
    idx = {n: i for i, n in enumerate(CH19)}
    for name, row in zip(out.channel_names, out.data):
        a, c = name.split("-")
        np.testing.assert_allclose(row, rec[idx[a]] - rec[idx[c]])
    # reference-independent: adding a common signal changes nothing
    shifted = simulate_geometry(rec + 5 * rec[:1], CH19, 200.0, "full_19ch_1020")
    np.testing.assert_allclose(shifted.data, out.data, atol=1e-9)


def test_channel_order_and_alias_invariance(rec):
    perm = np.random.default_rng(0).permutation(19)
    names = [CH19[i] for i in perm]
    names = [{"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}.get(n, n) for n in names]
    a = simulate_geometry(rec, CH19, 200.0, "full_19ch_1020")
    b = simulate_geometry(rec[perm], names, 200.0, "full_19ch_1020")
    np.testing.assert_allclose(a.data, b.data)


def test_ceribell_headband_8_bipolar(rec):
    out = simulate_geometry(rec, CH19, 250.0, "ceribell_headband_10el")
    assert out.channel_names == ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1", "Fp2-F8", "F8-T4", "T4-T6", "T6-O2"]
    assert out.data.shape[0] == 8 and out.confidence == "medium"
    assert out.electrode_names == ["Fp1", "F7", "T3", "T5", "O1", "Fp2", "F8", "T4", "T6", "O2"]
    np.testing.assert_allclose(out.data[0], rec[0] - rec[2])


def test_sedline_afz_linear_reference(rec):
    out = simulate_geometry(rec, CH19, 200.0, "masimo_sedline_psi_legacy")
    afz = 0.25 * rec[0] + 0.25 * rec[1] + 0.5 * rec[4]
    assert out.channel_names == ["Fp1-AFz", "Fp2-AFz", "F7-AFz", "F8-AFz"]
    np.testing.assert_allclose(out.data[0], rec[0] - afz)
    assert out.approximated == {"AFz": "linear"}


def test_brainscope_approximations_and_linked_ear(rec):
    out = simulate_geometry(rec, CH19, 200.0, "brainscope_ahead")
    assert out.electrode_names == ["Fp1", "Fp2", "Fpz", "AFz", "F7", "F8", "A1", "A2"]
    assert out.approximated["Fpz"] == "linear" and out.approximated["A1"] == "average"
    assert any("A1" in w for w in out.warnings)
    avg = rec.mean(axis=0)
    np.testing.assert_allclose(out.data[0], rec[0] - avg)   # Fp1 - linked ear(=average ref fallback)
    np.testing.assert_allclose(out.electrode_data[2], 0.5 * (rec[0] + rec[1]))


def test_recorded_ears_are_used(rec):
    names = CH19 + ["A1", "A2"]
    data = np.vstack([rec, rec[:1] * 2, rec[1:2] * 4])
    out = simulate_geometry(data, names, 200.0, "brainscope_ahead")
    assert "A1" not in out.approximated
    np.testing.assert_allclose(out.data[0], rec[0] - 0.5 * (data[19] + data[20]))


def test_spline_mode_for_missing_sites(rec):
    out = simulate_geometry(rec, CH19, 200.0, "masimo_sedline_psi_legacy", approximation="spline")
    assert out.approximated == {"AFz": "spline"}
    assert out.data.shape[0] == 4


@pytest.mark.parametrize("held", ["Fz", "Cz", "Pz", "C3", "C4", "F3", "F4", "P3", "P4"])
def test_spline_reconstructs_heldout_electrode(held):
    field = smooth_field(CH19, n=600, seed=3)
    keep = [i for i, n in enumerate(CH19) if n != held]
    rec = spline_interpolate(field[keep], [CH19[i] for i in keep], [held])[0]
    truth = field[CH19.index(held)]
    assert np.corrcoef(rec, truth)[0, 1] > 0.9


def test_spline_exact_at_known_sites():
    field = smooth_field(CH19, seed=5)
    back = spline_interpolate(field, CH19, CH19)
    # lam > 0 smooths slightly, so reproduction is near-exact rather than exact
    assert np.abs(back - field).max() < 0.05 * np.abs(field).max()


def test_spline_fpz_close_to_linear_midpoint():
    field = smooth_field(CH19, n=600, seed=7)
    fpz = spline_interpolate(field, CH19, ["Fpz"])[0]
    lin = 0.5 * (field[0] + field[1])
    assert np.corrcoef(fpz, lin)[0, 1] > 0.9


@pytest.mark.parametrize("geom", ["ceribell_headcap", "masimo_sedline_current"])
def test_low_confidence_raises_unless_allowed(rec, geom):
    with pytest.raises(LowConfidenceGeometryError):
        simulate_geometry(rec, CH19, 200.0, geom)
    with pytest.raises(ValueError):
        simulate_geometry(rec, CH19, 200.0, geom)
    with pytest.raises(NotImplementedError):
        simulate_geometry(rec, CH19, 200.0, geom)
    assert simulate_geometry(rec, CH19, 200.0, geom, allow_low_confidence=True).confidence == "low"


@pytest.mark.parametrize("geom", ["medtronic_bis", "not_verified_systems"])
def test_empty_geometry_always_raises(rec, geom):
    with pytest.raises(ValueError):
        simulate_geometry(rec, CH19, 200.0, geom, allow_low_confidence=True)


def test_simulate_all_skips_unsupported(rec):
    out = simulate_all(rec, CH19, 200.0)
    assert set(out) == {"full_19ch_1020", "ceribell_headband_10el", "zeto_wr19", "brainscope_ahead",
                        "masimo_sedline_psi_legacy"}
    with pytest.raises(ValueError):
        simulate_all(rec, CH19, 200.0, skip_unsupported=False)


def test_input_validation(rec):
    with pytest.raises(ValueError):
        simulate_geometry(rec[:5], CH19, 200.0, "full_19ch_1020")
    with pytest.raises(KeyError):
        simulate_geometry(rec, CH19, 200.0, "nope")
    with pytest.raises(ValueError):
        simulate_geometry(rec[:-1], CH19[:-1], 200.0, "full_19ch_1020")  # O2 missing -> unresolved
