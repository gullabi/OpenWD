"""Finite nuclear mass in the H2db Balmer component positions.

For an atom at rest, Delta E = mu Delta E_inf(beta/mu^2) - Delta m hbar omega_cp
with mu = 1/(1 + m_e/m_p) (Wunner et al. 1985, A&A 149, 102, Eq. 1;
Pavlov & Meszaros 1993, ApJ 416, 752, Eq. 2.12a).
"""

import numpy as np
import pytest

from wd_spectra import DAHConfig
from wd_spectra.magnetic_atomic import (
    ELECTRON_PROTON_MASS_RATIO,
    H2DB_REFERENCE_FIELD_MEGAGAUSS,
    read_h2db_transition_database,
)
from wd_spectra.models.common import ModelData

HALPHA = 6564.636
TEMPERATURE = np.array([12_000.0])
# The references below are pure Bohr energies, whose Halpha is 36/(5 R_H) with
# R_H = R_inf/(1 + m_e/m_p) (CODATA 2018 R_inf).  OpenWD's measured Halpha is
# 9.1e-6 shorter (fine structure, Lamb shift), so they are scaled by this ratio.
BOHR_HALPHA = 36.0 / (5.0 * 109_737.315_681_60e-8 / (1.0 + ELECTRON_PROTON_MASS_RATIO))
TO_OPENWD = HALPHA / BOHR_HALPHA


@pytest.fixture(scope="module")
def transitions():
    return read_h2db_transition_database(ModelData.default().h2db_balmer_subset)


def _index(transitions, upper_m, upper_parity):
    """Index of an Halpha branch from 2s0 (m=0, even, nu=2) to an nu=2 upper state."""
    for index, t in enumerate(transitions.transitions_by_upper_level[3]):
        if (t.lower_magnetic_quantum_number, t.upper_magnetic_quantum_number, t.lower_z_parity,
                t.upper_z_parity, t.lower_excitation_index, t.upper_excitation_index) == (
                0, upper_m, 1, upper_parity, 2, 2):
            return index
    raise KeyError("branch not in H2db")


def _wavelength(transitions, index, field, finite):
    components = transitions.balmer_components(
        3, HALPHA, field, TEMPERATURE, minimum_relative_strength=0.0, finite_nuclear_mass=finite)
    return components.wavelength_angstrom[index]


def test_wunner_1985_table_1_halpha_pi(transitions):
    # Table 1: 3p0-2s0 at 5829.88 A for beta = 0.05 in reduced-mass units
    # B0 = 4.6959e9 G. A pi line (Delta m = 0) checks the units and the beta/mu^2
    # argument, not the hbar omega_cp term.
    pi = _index(transitions, 0, -1)
    expected = 5829.88 * TO_OPENWD
    assert _wavelength(transitions, pi, 0.05 * 4_695.9, True) == pytest.approx(expected, abs=0.02)


def test_stationary_components(transitions):
    # Minima of lambda(B) against an independent diagonalization of the
    # finite-mass Hamiltonian: Halpha pi 2s0-3p0 5829.76 A at 231.46 MG and
    # sigma- 2s0-3p-1 6772.69 A at 196.49 MG (Bohr scale). The infinite-mass
    # default puts the sigma- minimum at 6777.2 A. H2db nodes lie 0.75 MG apart.
    for (upper_m, parity), window, wavelength, field in (
        ((0, -1), (220.0, 245.0), 5829.76, 231.46),
        ((-1, 1), (185.0, 210.0), 6772.69, 196.49),
    ):
        index = _index(transitions, upper_m, parity)
        grid = np.arange(*window, 0.1)
        scan = np.array([_wavelength(transitions, index, b, True) for b in grid])
        assert scan.min() == pytest.approx(wavelength * TO_OPENWD, abs=0.03)
        assert grid[np.argmin(scan)] == pytest.approx(field, abs=1.0)


def test_weak_field_orbital_g_factor(transitions):
    # Below the H2db floor the sigma components lie at +-(1 - m_e/m_p) mu_B B
    # (orbital g-factor of a finite-mass atom at rest); mu_B B = 2 beta Ry_inf
    # and the rest wavelength belongs to mu Ry_inf. The default's g = mu differs
    # only by (m_e/m_p)^2 = 3e-7, so this mainly guards the sign of the new term.
    field = 0.01
    components = transitions.balmer_components(3, HALPHA, field, TEMPERATURE, finite_nuclear_mass=True)
    rest_energy = 0.25 - 1.0 / 9.0
    energy = rest_energy * HALPHA / components.wavelength_angstrom  # units of mu Ry_inf
    mu = 1.0 / (1.0 + ELECTRON_PROTON_MASS_RATIO)
    beta = field / H2DB_REFERENCE_FIELD_MEGAGAUSS
    splitting = mu * (energy[components.delta_m == 1] - energy[components.delta_m == -1])[0] / 2.0
    assert splitting == pytest.approx((1.0 - ELECTRON_PROTON_MASS_RATIO) * 2.0 * beta, rel=1e-9)


def test_option_off_is_the_default_infinite_mass_scaling(transitions):
    assert DAHConfig().include_finite_nuclear_mass is False
    branches = transitions.transitions_by_upper_level[3]
    for field in (0.01, 50.0, 196.49, 400.0):
        default = transitions.balmer_components(3, HALPHA, field, TEMPERATURE, minimum_relative_strength=0.0)
        if field < 0.1:
            beta = field / H2DB_REFERENCE_FIELD_MEGAGAUSS
            old = HALPHA * (5 / 36) / (5 / 36 + 2.0 * beta * default.delta_m)
        else:
            old = HALPHA * (5 / 36) / np.array([t.values_at_field(field)[0] for t in branches])
        np.testing.assert_allclose(default.wavelength_angstrom, old, rtol=1e-12)
