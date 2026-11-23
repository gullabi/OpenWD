"""The opt-in n=3, 4 H2db components of the Paschen and Brackett lines.

Each zero-field Paschen or Brackett line is shared among its H2db components
with Delta-m-resolved oscillator strengths E d and the LTE share of the lower
substate (h2db_paschen_brackett_manifolds).
"""

from dataclasses import replace

import numpy as np
import pytest

from wd_spectra import DAHConfig, compute_dah
from wd_spectra._compat import trapezoid
from wd_spectra.atmosphere import gray_hydrogen_atmosphere
from wd_spectra.constants import LIGHT_SPEED, PLANCK
from wd_spectra.magnetic import (
    MagneticPhysics,
    _paschen_brackett_templates,
    _translated_profile,
    h2db_paschen_brackett_manifolds,
    synthesize_magnetic_hydrogen_spectrum,
    uniform_field_surface_cells,
)
from wd_spectra.magnetic_atomic import (
    _RYDBERG_ENERGY_ERG,
    read_h2db_energy_database,
    read_h2db_paschen_brackett_transitions,
    read_h2db_transition_database,
)
from wd_spectra.models.common import AtmosphereConvergenceWarning, ModelData
from wd_spectra.opacity import PASCHEN_LINES

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


def test_weak_field_paschen_beta_sigma_plus_recovers_zero_field_strength(atmosphere, transitions, energies):
    # At B -> 0 the components of a complete line in each manifold carry f_nn' x phi_nn'.
    # The window filter keeps every q = +1 component of Paschen beta.
    templates = _paschen_brackett_templates(atmosphere, False)
    wavelength = np.geomspace(4_000.0, 30_000.0, 20_001)
    opacity = h2db_paschen_brackett_manifolds(atmosphere, wavelength, 0.1, {(3, 5): transitions[3, 5]}, energies,
                                              templates=templates)
    frequency = 2.997_924_58e18 / wavelength
    integral = [-trapezoid(y, frequency, axis=0)
                for y in (opacity.plus, _translated_profile(templates[3, 5], wavelength, 0.0))]
    np.testing.assert_allclose(integral[0], integral[1], rtol=0.01)


def test_shifted_paschen_wing_is_continuous_at_template_edges(atmosphere, transitions, energies):
    # This Paschen alpha component previously put a hard cutoff at 4687.56 A.
    field = 282.0
    line = PASCHEN_LINES[0]
    key = (line.lower_level, line.upper_level)
    branch = next(
        track for track in transitions[key]
        if track.source_path.as_posix().endswith("m_+2_to_+1/pi_z_+1_to_+1/nu_1_to_4")
    )
    templates = _paschen_brackett_templates(atmosphere, True)
    template = templates[key]
    # The absorption joins zero at both edges. Its dispersion continues outside.
    np.testing.assert_array_equal(template.mass_absorption_coefficient[[0, -1]], 0.0)
    energy, _, _ = branch.values_at_field(field)
    light_angstrom_hz = LIGHT_SPEED * 1.0e8
    shift = energy * _RYDBERG_ENERGY_ERG / PLANCK - light_angstrom_hz / line.wavelength_vacuum_angstrom
    edges = light_angstrom_hz / (light_angstrom_hz / template.wavelength_angstrom[[0, -1]] + shift)
    wavelength = (edges[:, np.newaxis] + np.array([-1.0e-5, 1.0e-5])).ravel()
    _, dispersion = h2db_paschen_brackett_manifolds(
        atmosphere, wavelength, field, {key: (branch,)}, energies, templates=templates, dispersion=True,
    )
    assert np.all(dispersion.plus != 0.0)
    np.testing.assert_allclose(dispersion.plus[::2], dispersion.plus[1::2], rtol=1.0e-6)


def test_paschen_brackett_model_checks_the_corrected_field_limit(atmosphere):
    config = DAHConfig(
        effective_temperature=18_500.0, logg=8.0, field_geometry="uniform", field_angle_deg=0.0,
        magnetic_field_megagauss=1175.0, include_paschen_brackett_lines=True,
        include_rwa_photoionization=False, include_magnetic_eos=False,
    )
    wavelength = np.linspace(5000.0, 5010.0, 3)
    fixed = dict(initial_atmosphere=atmosphere, relax_atmosphere=False)
    with pytest.warns(AtmosphereConvergenceWarning):
        spectrum = compute_dah(config, wavelength, **fixed).spectrum
    assert np.all(np.isfinite(spectrum.surface_flux_lambda))
    with pytest.raises(ValueError, match=r"Paschen/Brackett field limit.*1174\.008 MG.*finite-nuclear-mass"):
        compute_dah(replace(config, include_finite_nuclear_mass=True), wavelength, **fixed)
    with pytest.warns(AtmosphereConvergenceWarning):
        spectrum = compute_dah(
            replace(config, magnetic_field_megagauss=1174.0, include_finite_nuclear_mass=True),
            wavelength, **fixed,
        ).spectrum
    assert np.all(np.isfinite(spectrum.surface_flux_lambda))


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
