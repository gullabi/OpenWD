"""The opt-in n=3, 4 H2db components of the Paschen and Brackett lines.

Each zero-field Paschen or Brackett line is shared among its H2db components
with Delta-m-resolved oscillator strengths E d and the LTE share of the lower
substate (h2db_paschen_brackett_manifolds).
"""

import numpy as np
import pytest

from wd_spectra import DAHConfig
from wd_spectra.atmosphere import gray_hydrogen_atmosphere
from wd_spectra.magnetic import (
    MagneticPhysics,
    _paschen_brackett_templates,
    _translated_profile,
    h2db_paschen_brackett_manifolds,
    synthesize_magnetic_hydrogen_spectrum,
    uniform_field_surface_cells,
)
from wd_spectra.magnetic_atomic import (
    read_h2db_energy_database,
    read_h2db_paschen_brackett_transitions,
    read_h2db_transition_database,
)
from wd_spectra.models.common import ModelData

DATA = ModelData.default()


@pytest.fixture(scope="module")
def transitions():
    return read_h2db_paschen_brackett_transitions(DATA.h2db_paschen_brackett_subset)


@pytest.fixture(scope="module")
def energies():
    return read_h2db_energy_database(DATA.h2db_balmer_subset)


@pytest.fixture(scope="module")
def atmosphere():
    return gray_hydrogen_atmosphere(18_500.0, 8.0, n_depth=24, correlated_microfields=True)


def test_subset_keeps_exactly_the_window_filtered_n3_n4_transitions():
    with np.load(DATA.h2db_paschen_brackett_subset) as values:
        labels, offsets = values["transition_labels"], values["transition_offsets"]
        beta, energy = values["transition_beta"], values["transition_energy_rydberg"].astype(float)
        dipole, initial = values["transition_dipole_strength"], values["transition_initial_energy_rydberg"]
    assert labels.shape == (855, 8)
    lower, upper = labels[:, 7], labels[:, 6]
    assert set(lower) == {3, 4} and np.all(upper > lower) and np.all(upper <= 22)
    for index, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        assert initial[start] == pytest.approx(-1.0 / lower[index] ** 2, abs=3e-3)
        assert beta[stop - 2] <= 0.25 < beta[stop - 1]
        wavelength = 1.0e8 / 109_678.77 / energy[start:stop]
        assert np.any((energy[start:stop] * dipole[start:stop] / 3.0 >= 1e-3)
                      & (wavelength >= 3400.0) & (wavelength <= 8000.0))


def test_b_to_zero_each_manifold_is_the_zero_field_line(atmosphere, transitions, energies):
    # At B -> 0 the components of a complete line in each manifold carry f_nn' x phi_nn'.
    # The window filter keeps every q = +1 component of Paschen beta.
    templates = _paschen_brackett_templates(atmosphere, False)
    wavelength = np.geomspace(4_000.0, 30_000.0, 20_001)
    opacity = h2db_paschen_brackett_manifolds(atmosphere, wavelength, 0.1, {(3, 5): transitions[3, 5]}, energies,
                                              templates=templates)
    frequency = 2.997_924_58e18 / wavelength
    integral = [-np.trapezoid(y, frequency, axis=0)
                for y in (opacity.plus, _translated_profile(templates[3, 5], wavelength, 0.0))]
    np.testing.assert_allclose(integral[0], integral[1], rtol=0.01)


def test_sigma_plus_lies_blue_of_sigma_minus(transitions):
    # Delta m = +1 is the blue (q = +1) component, as for the Balmer components.
    for key in [(3, 5), (4, 6)]:
        mean = {}
        for q in (-1, 1):
            rows = [t.values_at_field(20.0) for t in transitions[key] if np.sign(t.delta_m) == q]
            energy = np.array([r[0] for r in rows]); weight = np.array([r[0] * r[1] for r in rows])
            mean[q] = np.sum(weight * energy) / np.sum(weight)
        assert mean[1] > mean[-1]


def test_option_is_off_by_default_and_on_adds_absorption(energies):
    assert DAHConfig().include_paschen_brackett_lines is False
    # As in test_dah_intrashell: a larger gray opacity moves the continuum-forming layer
    # below the isothermal top of the fixture, where T(tau) rises.
    atmosphere = gray_hydrogen_atmosphere(18_500.0, 8.0, n_depth=24, correlated_microfields=True,
                                          rosseland_opacity=10.0)
    balmer = read_h2db_transition_database(DATA.h2db_balmer_subset)
    wavelength = np.arange(3400.0, 8000.0, 5.0)
    cells = uniform_field_surface_cells(282.0, field_angle_deg=0.0, n_mu=1)
    flux = [synthesize_magnetic_hydrogen_spectrum(
        atmosphere, wavelength, cells,
        MagneticPhysics(balmer, energies, line_regime="h2db", paschen_brackett_transitions=extra),
    ).surface_flux_lambda for extra in (None, read_h2db_paschen_brackett_transitions(DATA.h2db_paschen_brackett_subset))]
    ratio = flux[1] / flux[0]
    assert np.min(ratio) < 0.995  # 0.993 at 4525 A
