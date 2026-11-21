"""HeH+ in the homogeneous atomic and molecular H/He EOS (opt-in)."""
import numpy as np
import pytest

from wd_spectra import eos
from wd_spectra._mixed_molecules import (
    molecular_hydrogen_helium_enthalpy,
    molecular_hydrogen_helium_lte,
)
from wd_spectra.constants import BOLTZMANN
from wd_spectra.molecules import helium_hydride_ion_formation_constant

T = np.array([5000.0, 6000.0, 7000.0, 8000.0, 10000.0])
P = np.array([1e9, 1e9, 1e9, 3e8, 1e8])


def _check(state, molecular):
    h = state.helium_hydride_ion_density
    hydrogen, helium = state.hydrogen_lte_state, state.helium_lte_state
    bound = hydrogen.neutral_h_density + hydrogen.proton_density + h
    if molecular:
        bound = bound + (2 * hydrogen.molecular_hydrogen_density
                         + 2 * hydrogen.molecular_hydrogen_ion_density
                         + hydrogen.negative_hydrogen_density
                         + 3 * hydrogen.trihydrogen_ion_density)
    np.testing.assert_allclose(bound, state.hydrogen_nuclei_density, rtol=1e-12)
    np.testing.assert_allclose(
        h, helium_hydride_ion_formation_constant(T) * hydrogen.proton_density * helium.neutral_he_density,
        rtol=1e-12)
    positive = (hydrogen.proton_density + h + helium.singly_ionized_he_density
                + 2 * helium.doubly_ionized_he_density)
    if molecular:
        positive = positive + hydrogen.molecular_hydrogen_ion_density + hydrogen.trihydrogen_ion_density
        negative = state.electron_density + hydrogen.negative_hydrogen_density
    else:
        negative = state.electron_density
    np.testing.assert_allclose(positive, negative, rtol=1e-9)


@pytest.mark.parametrize("log_ratio", [-3.0, -1.0])
def test_atomic_mixed_eos_heh_conservation_and_mass_action(log_ratio):
    state = eos.hummer_mihalas_hydrogen_helium_lte(T, P, log_ratio, include_helium_hydride_ion=True)
    _check(state, molecular=False)
    hydride = state.helium_hydride_ion_density
    particles = state.hydrogen_nuclei_density + state.helium_nuclei_density - hydride + state.electron_density
    np.testing.assert_allclose(particles * BOLTZMANN * T, P, rtol=1e-12)


@pytest.mark.parametrize("log_ratio", [-3.0, -1.0])
def test_molecular_mixed_eos_heh_conservation_and_mass_action(log_ratio):
    off = molecular_hydrogen_helium_lte(T, P, log_ratio, include_helium_hydride_ion=False)
    on = molecular_hydrogen_helium_lte(T, P, log_ratio, include_helium_hydride_ion=True)
    _check(on, molecular=True)
    # HeH+ adds an ionization channel: more electrons where He is dense and cool.
    assert np.all(on.electron_density >= off.electron_density)
    assert on.electron_density[0] / off.electron_density[0] > 1.05


def test_heh_default_on_and_switch_restores_previous_chemistry(monkeypatch):
    assert eos.MIXED_HELIUM_HYDRIDE_ION_DEFAULT is True
    for function in (eos.hummer_mihalas_hydrogen_helium_lte, molecular_hydrogen_helium_lte):
        default = function(T, P, -2.0)
        on = function(T, P, -2.0, include_helium_hydride_ion=True)
        off = function(T, P, -2.0, include_helium_hydride_ion=False)
        assert default.helium_hydride_ion_density is not None
        assert off.helium_hydride_ion_density is None
        np.testing.assert_array_equal(default.electron_density, on.electron_density)
        monkeypatch.setattr(eos, "MIXED_HELIUM_HYDRIDE_ION_DEFAULT", False)
        np.testing.assert_array_equal(function(T, P, -2.0).electron_density, off.electron_density)
        monkeypatch.setattr(eos, "MIXED_HELIUM_HYDRIDE_ION_DEFAULT", True)


def test_molecular_enthalpy_includes_heh_binding(monkeypatch):
    monkeypatch.setattr(eos, "MIXED_HELIUM_HYDRIDE_ION_DEFAULT", False)
    off, _ = molecular_hydrogen_helium_enthalpy(T, P, -1.0)
    monkeypatch.setattr(eos, "MIXED_HELIUM_HYDRIDE_ION_DEFAULT", True)
    on, state = molecular_hydrogen_helium_enthalpy(T, P, -1.0)
    assert state.helium_hydride_ion_density is not None
    assert np.all(np.isfinite(on)) and not np.array_equal(on, off)


def test_helium_minus_default_is_continuous_at_john_edge():
    from wd_spectra import helium
    assert helium.HELIUM_MINUS_FREE_FREE_DEFAULT == "john1968-shape"
    for temperature in (4000.0, 6000.0, 9000.0):
        below, at = helium.helium_minus_free_free_coefficient(
            np.array([0.50629, 0.5063]), temperature, prescription=helium.HELIUM_MINUS_FREE_FREE_DEFAULT)
        assert abs(below / at - 1) < 1e-3


def test_hydrogen_rich_dense_gas_cannot_bind_more_helium_than_exists():
    # H/He = 100 at 1e12 dyn cm^-2: a fixed-point HeH+ update overshot the
    # helium budget here; the conservation quadratic keeps every density positive.
    t = np.array([5000.0, 8000.0, 15000.0, 30000.0, 60000.0])
    for function in (eos.hummer_mihalas_hydrogen_helium_lte, molecular_hydrogen_helium_lte):
        state = function(t, 1e12, 2.0)
        helium = state.helium_lte_state
        atomic = helium.neutral_he_density + helium.singly_ionized_he_density + helium.doubly_ionized_he_density
        assert np.all(np.isfinite(state.electron_density)) and np.all(atomic > 0)
        np.testing.assert_allclose(atomic + state.helium_hydride_ion_density,
                                   state.helium_nuclei_density, rtol=1e-12)
