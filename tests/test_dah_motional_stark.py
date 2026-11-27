"""Motional Stark kernel of the H-alpha and H-beta Zeeman components (balmer_line_broadening="motional")."""

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
        h2db = {}
        for upper, line in ((3, rest), (4, BALMER_LINES[1].wavelength_vacuum_angstrom)):
            components = db.balmer_components(upper, line, field, np.array([18000.]), minimum_relative_strength=0.)
            h2db.update({tuple(label): w for label, w in zip(components.labels, components.wavelength_angstrom)})
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
    # rtol 2e-3: the exact profiles hold the strength of their windows, which end 0.12 lambda0 from the line
    # (at 280 MG the H-alpha pi rows keep 0.995-1.000 of it; -0.06 % of all H-alpha strength).
    np.testing.assert_allclose(area[thin], trapezoid(reference[::-1], nu, axis=0)[thin], rtol=2e-3)
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


def _profile_moments(table, label, field, temperature, edge):
    """Blue and red shares and red mean within +-edge A of the exact profile at one temperature."""
    knots, cumulative = (v[0] for v in table.profile_at(label, field, np.array([temperature])))
    x = np.linspace(0., edge, 100001)
    c = np.interp(x, knots, cumulative)
    red = c[-1] - c[0]
    return dict(blue=c[0] - np.interp(-edge, knots, cumulative), red=red,
                red_mean=edge - trapezoid(c - c[0], x) / red)  # int x dC by parts


def test_exact_hbeta_sigma_profile_at_366_megagauss():
    # sigma- 2s0-4f-1 at 366 MG, 17.7 kK, inside +-296.6 A: the solver's lines give 0.2345 blue, 0.7408 red with
    # mean 48.37 A; smoothed by 2 A they give 0.243, 0.733, 48.9 A, the numbers of preprint check run
    # 20261125T213414Z-ulgen-2419b6 (0.243, 0.732, 49.0 A). The table's quantiles, interpolated between 360 and
    # 370 MG and 15 and 20 kK, put the red mean ~1 A lower.
    _, table = _data()
    label = (0, 1, 2, -1, 1, 3)
    assert np.isnan(table.delta_alpha_at(label, 366.))
    m = _profile_moments(table, label, 366., 17700., 296.6)
    assert m['blue'] == pytest.approx(0.2345, abs=0.005)
    assert m['red'] == pytest.approx(0.7408, abs=0.005)
    assert m['red_mean'] == pytest.approx(48.37, abs=1.5)


def test_exact_windows_keep_the_sigma_strength():
    # The windows of sigma- 2s0-4f-1 and 2s0-4p-1 tile the spectrum, so the strength they hold together is the
    # F = 0 strength of the two lines (sum rule 1.597 against 1.600 in 3648-4800 A at 366 MG, run
    # 20261125T222728Z-ulgen-d8711d); their windows reach 3648-5845 A, where ~1.4 % more comes in from outside
    # the pair (the next F = 0 line is 3p-1 at 7280 A). Same lower level, so H2db's strengths weigh the shares.
    db, table = _data()
    components = db.balmer_components(4, BALMER_LINES[1].wavelength_vacuum_angstrom, 366., np.array([17700.]))
    s = {tuple(label): w[0] for label, w in zip(components.labels, components.normalized_strength)}
    labels = [(0, 1, 2, -1, 1, 3), (0, 1, 2, -1, 1, 4)]
    share = [table.profile_at(label, 366., np.array([17700.]))[1][0, -1] for label in labels]
    assert sum(s[label] * w for label, w in zip(labels, share)) == pytest.approx(sum(s[label] for label in labels), rel=2e-2)


def test_profile_on_a_node_next_to_a_field_without_one():
    # On a field node only that node counts: a node whose upper neighbour has no exact profile still gives its own.
    _, table = _data()
    for label, (_, shares) in table.profiles.items():
        ok = np.isfinite(shares[:, 0, 0])
        k = np.flatnonzero(ok[:-1] & ~ok[1:])
        if k.size:
            break
    field = table.profile_field_megagauss[k[0]]
    assert table.profile_at(label, field, np.array([17000.])) is not None
    assert table.profile_at(label, field + 1.0, np.array([17000.])) is None
