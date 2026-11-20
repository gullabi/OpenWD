"""Motional Stark kernel of the H-alpha Zeeman components (balmer_line_broadening="motional")."""

import numpy as np
import pytest

from wd_spectra._compat import trapezoid
from wd_spectra.atmosphere import gray_hydrogen_atmosphere
from wd_spectra.constants import LIGHT_SPEED
from wd_spectra.magnetic import (
    _kurucz_griem_templates, h2db_balmer_manifolds, motional_stark_profile, motional_stark_shift_hz,
)
from wd_spectra.magnetic_atomic import read_h2db_transition_database, read_motional_stark_table
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
    from wd_spectra.magnetic import _motional_stark_template, _translated_profile, frequency_hilbert_dispersion

    _, table = _data()
    atm = gray_hydrogen_atmosphere(18000., 8., n_depth=4)
    template = _kurucz_griem_templates(atm, 3, False)[3]
    rest = LIGHT_SPEED / (BALMER_LINES[0].wavelength_vacuum_angstrom * 1e-8)
    # sigma- 2s0-3d-1 and pi 2s0-3p0 at 280 MG; the sign flip checks a blue (Delta alpha < 0) wing.
    for label, sign in (((0, 1, 2, -1, 1, 2), 1.), ((0, 1, 2, 0, -1, 2), -1.)):
        shift = sign * motional_stark_shift_hz(table.delta_alpha_at(label, 280.), 280., atm.temperature)
        kernel = _motional_stark_template(template, rest, rest, shift, atm.temperature, True)
        wave = np.unique(np.concatenate((kernel.wavelength_angstrom, np.geomspace(3000., 12000., 2000))))
        numeric = frequency_hilbert_dispersion(wave, _translated_profile(kernel, wave, 0.))
        phase = _translated_profile(kernel, wave, 0., kernel.dispersion)
        np.testing.assert_allclose(phase, numeric, rtol=0, atol=2e-3 * np.abs(numeric).max())
        far = np.abs(wave - BALMER_LINES[0].wavelength_vacuum_angstrom) > 500.  # the Faraday wing
        np.testing.assert_allclose(phase[far], numeric[far], rtol=2e-3)
