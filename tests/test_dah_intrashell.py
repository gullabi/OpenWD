"""The n=2 -> 2 H2db components (2s0 -> 2p+1, 2p-1 -> 2s0, 2p0 -> 2s0).

They exist only in a field, so they enter with absolute oscillator strengths
(f = E d / 3 per substate pair) and the LTE share of the n=2 shell in their
lower substate, with a thermal Doppler profile.
"""

import dataclasses
import hashlib

import numpy as np
import pytest

from wd_spectra import DAHConfig
from wd_spectra._compat import trapezoid
from wd_spectra.atmosphere import gray_hydrogen_atmosphere
from wd_spectra.constants import ELECTRON_MASS, ELEMENTARY_CHARGE_ESU, LIGHT_SPEED, PI, PLANCK, BOLTZMANN
from wd_spectra.magnetic import (
    MagneticPhysics,
    frequency_hilbert_dispersion,
    h2db_intrashell_manifolds,
    synthesize_magnetic_hydrogen_spectrum,
    uniform_field_surface_cells,
)
from wd_spectra.magnetic_atomic import (
    ELECTRON_PROTON_MASS_RATIO,
    H2DB_REFERENCE_FIELD_MEGAGAUSS,
    read_h2db_energy_database,
    read_h2db_transition_database,
)
from wd_spectra.models.common import ModelData
from wd_spectra.opacity import _atmosphere_level_distribution

H2DB = ModelData.default().h2db_balmer_subset
# Infinite-mass wavelengths 911.267 A / E of the public archive (lambda with
# R_inf), and the package's hydrogen Rydberg wavelength hc/(13.598 eV).
R_INF_WAVELENGTH = 911.267_05
PACKAGE_WAVELENGTH = 1.0e8 * PLANCK * LIGHT_SPEED / (13.598_434_599_702 * 1.602_176_634e-12)


@pytest.fixture(scope="module")
def database():
    return read_h2db_transition_database(H2DB)


@pytest.fixture(scope="module")
def atmosphere():
    return gray_hydrogen_atmosphere(15_000.0, 8.0, n_depth=24, correlated_microfields=True)


def test_subset_adds_exactly_the_intrashell_rows_and_keeps_every_balmer_row(database):
    labels = {(t.lower_magnetic_quantum_number, t.lower_z_parity, t.lower_excitation_index,
               t.upper_magnetic_quantum_number, t.upper_z_parity, t.upper_excitation_index)
              for t in database.intrashell_transitions}
    # m, z-parity, H2db excitation index of the lower and upper substates.
    assert labels == {(-1, 1, 1, 0, 1, 2), (0, 1, 2, 1, 1, 1), (0, -1, 1, 0, 1, 2)}  # 2p-1->2s0, 2s0->2p+1, 2p0->2s0
    # SHA-256 of every Balmer label and row of the subset shipped before this change.
    with np.load(H2DB) as values:
        label, offset = values["transition_labels"], values["transition_offsets"]
        balmer = label[:, 6] != 2
        digest = hashlib.sha256(label[balmer].tobytes())
        for name in ("transition_beta", "transition_energy_rydberg", "transition_dipole_strength",
                     "transition_initial_energy_rydberg"):
            rows = values[name]
            for index in np.flatnonzero(balmer):
                digest.update(rows[offset[index]:offset[index + 1]].tobytes())
    assert digest.hexdigest() == "8e73b220f68a4bae7a0dd876fcdc1e5da32b067588e3896aeaf8d34932ef9f6f"


def test_positions_against_h2db_and_the_finite_mass_form(database):
    temperature = np.array([15_000.0])
    # Infinite-mass wavelengths of the archive energies (issue table).
    for field, expected in ((300.0, (6791.0, 7526.5)), (400.0, (5179.4, 5542.6))):
        components = database.intrashell_components(field, temperature)
        sigma = components.wavelength_angstrom[components.delta_m == 1] * R_INF_WAVELENGTH / PACKAGE_WAVELENGTH
        np.testing.assert_allclose(sigma, expected, atol=0.1)
        # Finite mass: beta/mu^2 and -Delta m hbar omega_cp, as for the Balmer components.
        finite = database.intrashell_components(field, temperature, finite_nuclear_mass=True)
        mass_ratio = ELECTRON_PROTON_MASS_RATIO
        moved = database.intrashell_components(field * (1.0 + mass_ratio) ** 2, temperature)
        cyclotron = 4.0 * field / H2DB_REFERENCE_FIELD_MEGAGAUSS * mass_ratio * (1.0 + mass_ratio)
        np.testing.assert_allclose(
            PACKAGE_WAVELENGTH / finite.wavelength_angstrom,
            PACKAGE_WAVELENGTH / moved.wavelength_angstrom - cyclotron * finite.delta_m, rtol=1e-12)


def test_oscillator_strengths_and_polarization(database):
    # f = E d / 3: the same convention gives f(2s-3p) = 0.4349 from the H2db Halpha rows.
    halpha_2s = sum(t.transition_energy_rydberg[0] * t.dipole_strength[0] / 3.0
                    for t in database.transitions_by_upper_level[3]
                    if (t.lower_magnetic_quantum_number, t.lower_z_parity, t.lower_excitation_index) == (0, 1, 2))
    assert halpha_2s == pytest.approx(0.4349, abs=2e-4)
    components = database.intrashell_components(400.0, np.array([15_000.0]))
    sigma = components.delta_m == 1
    # Archive at 400 MG: E d = 0.564 and 0.527.
    np.testing.assert_allclose(np.sort(components.oscillator_strength[sigma]), [0.1757, 0.1881], atol=1e-3)
    # q = +1 is the blue sigma component: at 2 MG both lie at +mu_B B = +2 beta Ry, as the
    # Delta m = +1 Balmer components do.
    weak = database.intrashell_components(2.0, np.array([15_000.0]))
    energy = PACKAGE_WAVELENGTH / weak.wavelength_angstrom[weak.delta_m == 1]
    np.testing.assert_allclose(energy, 2.0 * 2.0 / H2DB_REFERENCE_FIELD_MEGAGAUSS, rtol=0.02)
    # Below the H2db floor they lie beyond 20 microns and are omitted.
    assert database.intrashell_components(0.5, np.array([15_000.0])).wavelength_angstrom.size == 0


def test_opacity_integrates_to_the_oscillator_strength_and_lower_population(atmosphere, database):
    field = 300.0
    components = database.intrashell_components(field, atmosphere.temperature)
    line = int(np.flatnonzero(components.delta_m == 1)[0])
    center = components.wavelength_angstrom[line]
    wavelength = np.linspace(center - 3.0, center + 3.0, 6001)
    opacity, phase = h2db_intrashell_manifolds(atmosphere, wavelength, field, database, dispersion=True)
    frequency = LIGHT_SPEED / (wavelength * 1.0e-8)
    integral = -trapezoid(opacity.plus, frequency, axis=0)
    n2 = _atmosphere_level_distribution(atmosphere, 2).population_density[:, 1]
    nu = LIGHT_SPEED / (center * 1.0e-8)
    expected = (3.0 * PI * ELEMENTARY_CHARGE_ESU**2 / (ELECTRON_MASS * LIGHT_SPEED)
                * components.oscillator_strength[line] * components.lower_state_fraction[line]
                * n2 / atmosphere.mass_density * -np.expm1(-PLANCK * nu / (BOLTZMANN * atmosphere.temperature)))
    np.testing.assert_allclose(integral, expected, rtol=1e-4)
    assert not np.any(opacity.minus) and not np.any(opacity.pi[(wavelength > 6000.0)])
    # Shares are the LTE populations of the four n=2 substates.
    assert np.all(components.lower_state_fraction > 0.0) and np.all(np.sum(components.lower_state_fraction, axis=0) < 1.0)
    # The analytic dispersion partner is the Kramers-Kronig partner of the profile.
    numeric = frequency_hilbert_dispersion(wavelength, opacity.plus)
    core = np.abs(wavelength - center) < 1.0
    assert np.max(np.abs(numeric[core] - phase.plus[core])) < 2e-3 * np.max(np.abs(phase.plus))


def test_option_off_is_the_default_and_leaves_the_spectrum_unchanged(atmosphere, database):
    assert DAHConfig().include_intrashell_lines is False
    wavelength = np.arange(6780.0, 6810.0, 0.1)
    cells = uniform_field_surface_cells(300.0, field_angle_deg=30.0, n_mu=1)
    energies = read_h2db_energy_database(H2DB)
    old = dataclasses.replace(database, intrashell_transitions=())

    def flux(transitions, include):
        physics = MagneticPhysics(transitions, energies, line_regime="h2db", include_intrashell_lines=include)
        return synthesize_magnetic_hydrogen_spectrum(atmosphere, wavelength, cells, physics).surface_flux_lambda

    off = flux(database, False)
    np.testing.assert_allclose(off, flux(old, False), rtol=1e-12, atol=0.0)


def test_option_on_adds_the_line_in_the_synthesis(database):
    # The fixture's top is isothermal down to where the 6800 A continuum forms, so
    # a saturated line leaves no mark there; a larger gray opacity moves the
    # continuum-forming layer to where T(tau) rises.
    atmosphere = gray_hydrogen_atmosphere(15_000.0, 8.0, n_depth=24, correlated_microfields=True,
                                          rosseland_opacity=10.0)
    wavelength = np.arange(6788.0, 6801.0, 0.05)
    cells = uniform_field_surface_cells(300.0, field_angle_deg=30.0, n_mu=1)
    energies = read_h2db_energy_database(H2DB)
    flux = [synthesize_magnetic_hydrogen_spectrum(
        atmosphere, wavelength, cells,
        MagneticPhysics(database, energies, line_regime="h2db", include_intrashell_lines=include),
        polarized_transfer="full-stokes-iquv").surface_flux_lambda for include in (False, True)]
    ratio = flux[1] / flux[0]
    center = database.intrashell_components(300.0, np.array([15_000.0])).wavelength_angstrom[0]  # 2p-1 -> 2s0
    assert np.min(ratio) < 0.995
    assert wavelength[np.argmin(ratio)] == pytest.approx(center, abs=0.1)
    assert np.all(ratio[np.abs(wavelength - center) > 2.0] > 0.9999)
