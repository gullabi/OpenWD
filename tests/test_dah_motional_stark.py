"""Motional Stark kernel of the H-alpha Zeeman components (balmer_line_broadening="motional")."""

import numpy as np
import pytest

from wd_spectra._compat import trapezoid
from wd_spectra.atmosphere import gray_hydrogen_atmosphere
from wd_spectra.constants import BOLTZMANN, HYDROGEN_MASS, LIGHT_SPEED
from wd_spectra.magnetic import (
    _kurucz_griem_templates, _motional_stark_template, _translated_profile,
    h2db_balmer_manifolds, motional_stark_profile, motional_stark_shift_hz,
)
from wd_spectra.magnetic_atomic import (
    MotionalStarkTable, read_h2db_transition_database, read_motional_stark_table,
)
from wd_spectra.models.common import ModelData
from wd_spectra.opacity import BALMER_LINES


def _data():
    data = ModelData.default()
    return read_h2db_transition_database(data.h2db_balmer_subset), read_motional_stark_table(data.motional_stark_table)


def test_profile_area_mean_and_doppler_limit():
    sigma = 1.0
    x = np.linspace(-40., 800., 840001)
    for shift in (3., -3., 30.):
        p = motional_stark_profile(x if shift > 0 else -x, sigma, shift)
        assert trapezoid(p, x) == pytest.approx(1., rel=1e-6)
        assert trapezoid(x * p, x) == pytest.approx(abs(shift), rel=1e-6)
    # s -> 0 (B -> 0 or motional_scale = 0): the Doppler profile.
    gauss = np.exp(-.5 * x**2) / np.sqrt(2 * np.pi)
    np.testing.assert_allclose(motional_stark_profile(x, sigma, 0.), gauss, rtol=1e-9, atol=1e-300)
    np.testing.assert_allclose(motional_stark_profile(x, sigma, 1e-6), gauss, rtol=0, atol=1e-6)


def test_table_wavelengths_match_h2db_components():
    db, table = _data()
    rest = BALMER_LINES[0].wavelength_vacuum_angstrom
    with np.load(ModelData.default().motional_stark_table) as values:
        grid, labels, wavelength = values["field_megagauss"], values["labels"], values["wavelength_infinite_mass_angstrom"]
    for field in (190., 280., 450.):
        components = db.balmer_components(3, rest, field, np.array([18000.]), minimum_relative_strength=0.)
        h2db = {tuple(label): w for label, w in zip(components.labels, components.wavelength_angstrom)}
        j = int(np.argmin(np.abs(grid - field)))
        for label, w in zip(labels, wavelength[:, j]):
            if 0. < w < 2e4:  # optical/IR components (others vanish or turn negative)
                assert h2db[tuple(label)] == pytest.approx(w, rel=1e-3)


def test_stationary_pi_component_shift():
    # Halpha pi 2s0-3p0 at its stationary field: mean shift +7.0 A to the red at
    # 17.7 kK (s = Delta alpha kT beta^2 / M, Pavlov & Meszaros 1993, Eq. 3.28).
    _, table = _data()
    shift = motional_stark_shift_hz(table.delta_alpha_at((0, 1, 2, 0, -1, 2), 231.9), 231.9, 17700.)
    assert 5829.7**2 * shift / LIGHT_SPEED * 1e-8 == pytest.approx(7.0, abs=.05)


def test_motional_lookup_keeps_finite_knots_beside_flagged_nodes():
    label = (0, 1, 2, 0, -1, 2)
    table = MotionalStarkTable(
        np.array([100., 105., 110., 115.]), {label: np.array([4., np.nan, 8., 10.])},
    )
    assert table.delta_alpha_at(label, 100.) == 4.
    assert np.isnan(table.delta_alpha_at(label, 102.5))
    assert np.isnan(table.delta_alpha_at(label, 105.))
    assert table.delta_alpha_at(label, 112.5) == 9.
    assert table.delta_alpha_at(label, 115.) == 10.


def test_motional_table_keeps_weak_h2db_cell_on_analytic_triplet():
    db, table = _data()
    atmosphere = gray_hydrogen_atmosphere(18_000., 8., n_depth=4)
    wave = np.linspace(6560., 6568., 65)
    templates = _kurucz_griem_templates(atmosphere, 3, True)
    # A positive field below the table floor can occur in a strong-field map.
    baseline = h2db_balmer_manifolds(
        atmosphere, wave, .01, db, templates=templates, maximum_upper_level=3, dispersion=True,
    )
    motional = h2db_balmer_manifolds(
        atmosphere, wave, .01, db, templates=templates, maximum_upper_level=3, dispersion=True,
        motional_stark_table=table,
    )
    for plain, actual in zip(baseline, motional):
        for q in ("minus", "pi", "plus"):
            np.testing.assert_array_equal(getattr(actual, q), getattr(plain, q))


@pytest.mark.parametrize("scale", [1., 2.])
def test_broad_red_motional_component_survives_in_physical_frequency(scale):
    db, table = _data()
    atmosphere = gray_hydrogen_atmosphere(18_000., 8., n_depth=12)
    field = 450.
    components = db.balmer_components(3, BALMER_LINES[0].wavelength_vacuum_angstrom,
                                     field, atmosphere.temperature)
    label = (0, 1, 2, -1, 1, 2)
    i = next(i for i, row in enumerate(components.labels) if tuple(row) == label)
    center = LIGHT_SPEED / (components.wavelength_angstrom[i] * 1e-8)
    shift = scale * motional_stark_shift_hz(table.delta_alpha_at(label, field),
                                           field, atmosphere.temperature)
    template = _kurucz_griem_templates(atmosphere, 3, False)[3]
    kernel = _motional_stark_template(template, center, shift, atmosphere.temperature, True)
    assert np.all(np.isfinite(kernel.wavelength_angstrom))
    assert kernel.wavelength_angstrom[0] > 0.
    assert np.all(np.diff(kernel.wavelength_angstrom) > 0.)
    assert np.all(np.isfinite(kernel.dispersion))
    sigma = center * np.sqrt(BOLTZMANN * atmosphere.temperature / HYDROGEN_MASS) / LIGHT_SPEED
    frequency = np.unique(np.concatenate((
        center * np.geomspace(1.e-6, 4., 16_001),
        center + np.linspace(-12. * sigma.max(), 12. * sigma.max(), 3001),
    )))
    sample = _translated_profile(kernel, LIGHT_SPEED / frequency[::-1] / 1.e-8, 0.)[::-1]
    assert np.all(sample[np.searchsorted(frequency, center)] > 0.)
    np.testing.assert_allclose(trapezoid(sample, frequency, axis=0), template.strength, rtol=1.e-3)


@pytest.mark.parametrize("scale", [1., 0.])
def test_motional_kernel_conserves_line_strength(scale):
    db, table = _data()
    atm = gray_hydrogen_atmosphere(18000., 8., n_depth=8)
    templates = _kurucz_griem_templates(atm, 3, False)
    nu = np.linspace(LIGHT_SPEED / 1.2e-4, LIGHT_SPEED / 2.5e-5, 400001)  # 2500-12000 A, finer than Doppler
    wave = LIGHT_SPEED / nu[::-1] / 1e-8
    # Only Halpha components the table lists at 280 MG take the motional kernel;
    # with a template of zero-field Stark area, the summed manifold area is the line strength.
    opacity = h2db_balmer_manifolds(atm, wave, 280., db, templates=templates, maximum_upper_level=3,
                                    motional_stark_table=table, motional_scale=scale).isotropic()
    reference = h2db_balmer_manifolds(atm, wave, 280., db, templates=templates, maximum_upper_level=3).isotropic()
    area = trapezoid(opacity[::-1], nu, axis=0)
    thin = atm.rosseland_optical_depth < .1  # zero-field Stark wings fit inside the grid
    np.testing.assert_allclose(area[thin], trapezoid(reference[::-1], nu, axis=0)[thin], rtol=1e-3)
    assert np.all(area[~thin] > 0)


def test_motional_dispersion_is_the_hilbert_partner_of_its_absorption():
    # On a disk the V/I of narrow lines comes from their dispersion wing, so the
    # kernel's partner must hold far from the line, not only in its core.
    from wd_spectra.magnetic import frequency_hilbert_dispersion

    _, table = _data()
    atm = gray_hydrogen_atmosphere(18000., 8., n_depth=4)
    template = _kurucz_griem_templates(atm, 3, False)[3]
    rest = LIGHT_SPEED / (BALMER_LINES[0].wavelength_vacuum_angstrom * 1e-8)
    # sigma- 2s0-3d-1 and pi 2s0-3p0 at 280 MG; the sign flip checks a blue (Delta alpha < 0) wing.
    for label, sign in (((0, 1, 2, -1, 1, 2), 1.), ((0, 1, 2, 0, -1, 2), -1.)):
        shift = sign * motional_stark_shift_hz(table.delta_alpha_at(label, 280.), 280., atm.temperature)
        kernel = _motional_stark_template(template, rest, shift, atm.temperature, True)
        wave = np.unique(np.concatenate((kernel.wavelength_angstrom, np.geomspace(3000., 12000., 2000))))
        numeric = frequency_hilbert_dispersion(wave, _translated_profile(kernel, wave, 0.))
        phase = _translated_profile(kernel, wave, 0., kernel.dispersion)
        np.testing.assert_allclose(phase, numeric, rtol=0, atol=2e-3 * np.abs(numeric).max())
        far = np.abs(wave - BALMER_LINES[0].wavelength_vacuum_angstrom) > 500.  # the Faraday wing
        np.testing.assert_allclose(phase[far], numeric[far], rtol=2e-3)
