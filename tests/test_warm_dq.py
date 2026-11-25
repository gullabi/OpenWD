"""Warm/hot DQ: hydrogen and helium in the bulk mixture and Paschen-Back line splitting."""

from __future__ import annotations

import numpy as np
import pytest

from wd_spectra import WarmDQConfig, d6
from wd_spectra import magnetic_metals as zeeman
from wd_spectra._compat import trapezoid
from wd_spectra.constants import BOLTZMANN
from wd_spectra.atmosphere import Atmosphere
from wd_spectra.eos import hummer_mihalas_helium_lte, hummer_mihalas_helium_thermodynamics
from wd_spectra.models.common import ModelData
from wd_spectra.models.selection import select_physics
from wd_spectra.models.warm_dq import (
    effective_mixing_length_alpha,
    validate_config,
    warm_dq_abundances,
    warm_dq_atomic_inputs,
    warm_dq_surface_cells,
)


@pytest.fixture(scope="module")
def carbon():
    database, photoionization, topbase = warm_dq_atomic_inputs(ModelData.default(), ("C",))
    return database, photoionization, topbase


def _stage(database, charge):
    return next(
        ion for ion in database.ions.values()
        if ion.element == "C" and ion.charge == charge
    )


def _components(ion, parent, low, high, field_megagauss, **options):
    split = zeeman.zeeman_split_ion(
        ion, field_megagauss * 1.0e6, parent_ion=parent,
        wavelength_range_angstrom=(low, high), minimum_oscillator_strength=1.0e-3,
        **options,
    )
    weight = {level.index: level.statistical_weight for level in ion.levels}
    return {
        q: (
            np.asarray([1.0e8 / line.wavelength_vacuum_angstrom for line in lines]),
            np.asarray([
                line.absorption_oscillator_strength * weight[line.lower_index] / 3.0
                for line in lines
            ]),
        )
        for q, lines in split.items()
    }


def test_clebsch_gordan_and_lande_factors():
    assert zeeman.clebsch_gordan(1, 0, 1, 0, 2, 0) == pytest.approx(np.sqrt(2.0 / 3.0))
    assert zeeman.clebsch_gordan(0.5, 0.5, 0.5, -0.5, 0, 0) == pytest.approx(np.sqrt(0.5))
    assert zeeman.clebsch_gordan(1, 1, 0.5, -0.5, 1.5, 0.5) == pytest.approx(np.sqrt(1.0 / 3.0))
    assert zeeman.clebsch_gordan(1, 1, 1, 1, 1, 2) == 0.0
    g_s = zeeman.ELECTRON_SPIN_G_FACTOR
    assert zeeman.lande_g_factor(0.5, 0, 0.5) == pytest.approx(g_s)
    assert zeeman.lande_g_factor(0.5, 1, 1.5) == pytest.approx(1.0 + (g_s - 1.0) / 3.0)
    assert zeeman.lande_g_factor(0.5, 1, 0.5) == pytest.approx(1.0 - (g_s - 1.0) / 3.0)


def test_level_label_parser():
    label = zeeman.parse_ls_level_label("2s2.2p.(2Po).4p.(1D<2>)")
    assert (label.spin, label.orbital, label.total) == (0.0, 2, 2.0)
    assert (label.outer_principal, label.outer_orbital, label.core_orbital) == (4, 1, 1)
    label = zeeman.parse_ls_level_label("2s2.4f.(2Fo<7/2>)")
    assert (label.spin, label.orbital, label.total, label.core_orbital) == (0.5, 3, 3.5, 0)
    assert zeeman.parse_ls_level_label("not a term") is None


def test_zero_field_returns_the_ls_multiplet(carbon):
    database, _, _ = carbon
    ion, parent = _stage(database, 1), _stage(database, 2)
    weight = {level.index: level.statistical_weight for level in ion.levels}
    total = sum(
        weight[line.lower_index] * line.absorption_oscillator_strength
        for line in ion.transitions
        if line.transition_type == "E1"
        and 4266.0 <= line.wavelength_vacuum_angstrom <= 4270.0
        and line.absorption_oscillator_strength >= 1.0e-3
    )
    components = _components(ion, parent, 4266.0, 4270.0, 0.0)
    for q in (-1, 0, 1):
        assert components[q][1].sum() == pytest.approx(total / 3.0, rel=1.0e-9)
    wavenumber = np.concatenate([components[q][0] for q in (-1, 0, 1)])
    strength = np.concatenate([components[q][1] for q in (-1, 0, 1)])
    lines = np.unique(np.round(wavenumber, 2))
    assert lines.size == 3
    # 3d 2D - 4f 2Fo in LS coupling: 14 : 20 : 1.
    ratios = sorted(
        strength[np.abs(wavenumber - line) < 0.01].sum() / total for line in lines
    )
    assert ratios == pytest.approx([1.0 / 35.0, 14.0 / 35.0, 20.0 / 35.0], rel=1.0e-6)


def test_weak_field_matches_the_anomalous_zeeman_pattern(carbon):
    database, _, _ = carbon
    ion, parent = _stage(database, 1), _stage(database, 2)
    field = 1.0e-3  # MG
    larmor = zeeman.BOHR_MAGNETON_WAVENUMBER_PER_GAUSS * field * 1.0e6
    zero = _components(ion, parent, 6575.0, 6590.0, 0.0, include_diamagnetic=False)
    split = _components(ion, parent, 6575.0, 6590.0, field, include_diamagnetic=False)
    parents = np.unique(np.round(zero[0][0], 3))
    assert parents.size == 2
    doublet_p32, doublet_p12 = parents.max(), parents.min()  # 2S1/2 - 2Po3/2 is bluer
    g_s = zeeman.ELECTRON_SPIN_G_FACTOR
    g_s12 = g_s
    g_p32 = 1.0 + (g_s - 1.0) / 3.0
    g_p12 = 1.0 - (g_s - 1.0) / 3.0
    expected = {
        (doublet_p32, 1): sorted([1.5 * g_p32 - 0.5 * g_s12, 0.5 * g_p32 + 0.5 * g_s12]),
        (doublet_p32, 0): sorted([0.5 * (g_p32 - g_s12), -0.5 * (g_p32 - g_s12)]),
        (doublet_p12, 1): [0.5 * g_p12 + 0.5 * g_s12],
        (doublet_p12, 0): sorted([0.5 * (g_p12 - g_s12), -0.5 * (g_p12 - g_s12)]),
    }
    for (parent_wavenumber, q), shifts in expected.items():
        wavenumber, _ = split[q]
        near = np.abs(wavenumber - parent_wavenumber) < 3.0 * larmor
        measured = np.sort((wavenumber[near] - parent_wavenumber) / larmor)
        assert measured == pytest.approx(shifts, abs=2.0e-3)


def test_strong_field_is_a_normal_triplet_and_conserves_strength(carbon):
    database, _, _ = carbon
    ion, parent = _stage(database, 1), _stage(database, 2)
    field = 2.0
    larmor = zeeman.BOHR_MAGNETON_WAVENUMBER_PER_GAUSS * field * 1.0e6
    zero = _components(ion, parent, 4266.0, 4270.0, 0.0, include_diamagnetic=False)
    split = _components(ion, parent, 4266.0, 4270.0, field, include_diamagnetic=False)
    centre = np.sum(zero[0][0] * zero[0][1]) / zero[0][1].sum()
    for q in (-1, 0, 1):
        wavenumber, strength = split[q]
        # Components below 1e-6 of the multiplet are dropped.
        assert strength.sum() == pytest.approx(zero[q][1].sum(), rel=1.0e-4)
        centroid = np.sum(wavenumber * strength) / strength.sum()
        assert centroid - centre == pytest.approx(q * larmor, abs=0.01 * larmor)
        # The fine structure (0.1-1 cm^-1) is all that remains of the spread.
        assert np.sqrt(np.sum(strength * (wavenumber - centroid) ** 2) / strength.sum()) < 1.5
    quadratic = _components(ion, parent, 4266.0, 4270.0, field)
    shift = (
        np.sum(quadratic[0][0] * quadratic[0][1]) / quadratic[0][1].sum()
        - np.sum(split[0][0] * split[0][1]) / split[0][1].sum()
    )
    # Diamagnetic shift of 4f (Z = 2): blueward and well below the Larmor shift.
    assert 0.0 < shift < 0.05 * larmor


def test_hydrogen_free_mixture_is_unchanged(carbon):
    database, photoionization, topbase = carbon
    atmosphere = d6.gray_d6_atmosphere(20_000.0, 8.8, database, {"C": 0.0}, n_depth=12)
    state = d6.bulk_metal_lte_state(atmosphere, database, {"C": 0.0})
    assert state.trace_hydrogen_state is None and atmosphere.hydrogen_lte_state is None
    assert np.all(atmosphere.neutral_h_density == 0.0)


def test_hydrogen_joins_the_pressure_and_charge_closure(carbon):
    database, _, _ = carbon
    abundances = {"C": 0.0, "H": 0.3}
    atmosphere = d6.gray_d6_atmosphere(16_000.0, 9.0, database, abundances, n_depth=16)
    state = d6.bulk_metal_lte_state(atmosphere, database, abundances)
    hydrogen = state.trace_hydrogen_state
    assert hydrogen is not None and atmosphere.hydrogen_lte_state is not None
    carbon_density = state.element_number_density["C"]
    assert hydrogen.hydrogen_nuclei_density / carbon_density == pytest.approx(10.0**0.3)
    particles = state.electron_density + carbon_density + hydrogen.hydrogen_nuclei_density
    assert particles * BOLTZMANN * atmosphere.temperature == pytest.approx(
        atmosphere.gas_pressure, rel=1.0e-9
    )
    charge = hydrogen.proton_density + sum(
        np.sum(np.arange(population.shape[0])[:, np.newaxis] * population, axis=0)
        for population in state.ion_number_density.values()
    )
    assert charge == pytest.approx(state.electron_density, rel=1.0e-9)
    assert hydrogen.neutral_h_density + hydrogen.proton_density == pytest.approx(
        hydrogen.hydrogen_nuclei_density
    )
    assert np.sum(hydrogen.level_population_density, axis=-1) == pytest.approx(
        hydrogen.neutral_h_density
    )
    assert sum(state.mass_fraction.values()) == pytest.approx(1.0)
    thermodynamics = d6.bulk_metal_thermodynamics(atmosphere, database, abundances)
    assert np.all(thermodynamics.adiabatic_temperature_gradient > 0.0)
    assert np.all(thermodynamics.adiabatic_temperature_gradient < 0.4)


def test_hydrogen_lines_appear_in_the_mixture_opacity(carbon):
    database, photoionization, topbase = carbon
    wavelength = np.asarray([6400.0, 6564.6, 6700.0])
    opacity = {}
    for name, abundances in (("C", {"C": 0.0}), ("CH", {"C": 0.0, "H": -1.0})):
        atmosphere = d6.continuum_d6_atmosphere(
            20_000.0, 8.8, database, photoionization, abundances, n_depth=16,
            topbase_photoionization_database=topbase,
        )
        state = d6.bulk_metal_lte_state(atmosphere, database, abundances)
        opacity[name], _ = d6._metal_opacity(
            atmosphere, wavelength, database, state, photoionization,
            minimum_oscillator_strength=1.0, maximum_lines=0, include_lines=True,
            topbase_photoionization_database=topbase,
        )
    photosphere = int(np.argmin(np.abs(atmosphere.rosseland_optical_depth - 2.0 / 3.0)))
    assert opacity["CH"][1, photosphere] > 3.0 * opacity["C"][1, photosphere]
    assert opacity["CH"][0, photosphere] < 1.5 * opacity["C"][0, photosphere]


def test_config_abundances_and_convection_policy():
    config = WarmDQConfig(log_carbon_to_hydrogen=1.2, trace_abundances={"O": -1.0})
    assert warm_dq_abundances(config) == {"C": 0.0, "O": -1.0, "H": -1.2}
    assert warm_dq_abundances(WarmDQConfig(log_carbon_to_hydrogen=None)) == {"C": 0.0}
    with pytest.raises(ValueError):
        validate_config(WarmDQConfig(trace_abundances={"He": 0.0}))
    assert warm_dq_abundances(
        WarmDQConfig(log_carbon_to_hydrogen=2.5, log_carbon_to_helium=-1.5)
    ) == {"C": 0.0, "H": -2.5, "He": 1.5}
    with pytest.raises(ValueError):
        validate_config(WarmDQConfig(log_carbon_to_helium=float("nan")))
    helium_selection = select_physics(WarmDQConfig(log_carbon_to_helium=0.0))
    assert helium_selection.diagnostics["helium_in_mixture"] is True
    with pytest.raises(ValueError):
        validate_config(WarmDQConfig(magnetic_field_megagauss=-1.0))
    assert effective_mixing_length_alpha(config) == 1.25
    magnetic = WarmDQConfig(magnetic_field_megagauss=2.0)
    assert effective_mixing_length_alpha(magnetic) is None
    assert effective_mixing_length_alpha(
        WarmDQConfig(magnetic_field_megagauss=2.0, magnetic_convection_suppression=False)
    ) == 1.25
    cells = warm_dq_surface_cells(magnetic)
    assert cells.projected_weight.sum() == pytest.approx(1.0)
    assert cells.maximum_field_megagauss <= 2.0 + 1.0e-9
    selection = select_physics(magnetic)
    assert selection.workflow == "warm-dq" and selection.experimental


def test_magnetic_synthesis_recovers_the_zero_field_metal_lines(carbon):
    from wd_spectra.magnetic import uniform_field_surface_cells
    from wd_spectra.warm_dq import synthesize_magnetic_bulk_spectrum

    database, photoionization, topbase = carbon
    abundances = {"C": 0.0}
    atmosphere = d6.continuum_d6_atmosphere(
        20_000.0, 8.8, database, photoionization, abundances, n_depth=16,
        topbase_photoionization_database=topbase,
    )
    wavelength = np.arange(4225.0, 4310.01, 0.5)
    options = dict(
        topbase_photoionization_database=topbase,
        metal_series_pseudocontinuum_elements=("C",),
        include_linear_stark_quasistatic=True,
        profile_edge_optical_depth=1.0e-3,
    )
    reference = d6.synthesize_d6_spectrum(
        atmosphere, wavelength, database, photoionization, abundances,
        maximum_metal_lines=25_000, n_angle=4, **options,
    )
    zero = synthesize_magnetic_bulk_spectrum(
        atmosphere, wavelength, database, photoionization, abundances,
        uniform_field_surface_cells(1.0e-6), line_wavelength_margin_angstrom=100.0,
        **options,
    )
    assert zero.surface_flux_lambda == pytest.approx(reference.surface_flux_lambda, rel=5.0e-3)
    field = 2.0
    split = synthesize_magnetic_bulk_spectrum(
        atmosphere, wavelength, database, photoionization, abundances,
        uniform_field_surface_cells(field), line_wavelength_margin_angstrom=100.0,
        **options,
    )
    depth = 1.0 - split.surface_flux_lambda / split.continuum_flux_lambda
    # Each C II 4267 component (normal-triplet spacing 4.67e-13 lambda^2 B)
    # is a local absorption maximum (the broad C I 4270 line underlies the
    # whole window).
    larmor = 4.6686e-13 * 4268.3**2 * field * 1.0e6
    for position in (4268.3, 4268.3 - larmor, 4268.3 + larmor):
        index = int(np.argmin(np.abs(wavelength - position)))
        peak = depth[index - 2:index + 3].max()
        assert peak > depth[index - 8] and peak > depth[index + 8]
    # C II 4267 is saturated at this gravity: separating its components
    # desaturates the line and increases the equivalent width.
    reference_depth = 1.0 - reference.surface_flux_lambda / reference.continuum_flux_lambda
    assert trapezoid(depth, wavelength) > 1.2 * trapezoid(reference_depth, wavelength)


def _pressure_grid_atmosphere():
    temperature = np.repeat([10_000.0, 14_000.0, 20_000.0, 30_000.0], 3)
    pressure = np.tile([1.0e4, 1.0e6, 3.0e7], 4)
    n = temperature.size
    return Atmosphere(
        16_000.0, 8.8, np.geomspace(1.0e-6, 10.0, n), np.geomspace(1.0e-3, 1.0e3, n),
        temperature, pressure, np.ones(n), np.zeros(n), np.zeros(n), np.ones(n), {},
    )


def test_helium_dominated_mixture_reproduces_the_db_equation_of_state(carbon):
    database, _, _ = carbon
    atmosphere = _pressure_grid_atmosphere()
    abundances = {"C": 0.0, "He": 9.0}
    state = d6.bulk_metal_lte_state(atmosphere, database, abundances)
    helium = state.trace_helium_state
    reference = hummer_mihalas_helium_lte(
        atmosphere.temperature, atmosphere.gas_pressure, correlated_microfields=True
    )
    # Carbon at C/He = 1e-9 adds at most a few 1e-3 of the electrons where
    # helium is nearly neutral.
    assert helium.electron_density == pytest.approx(reference.electron_density, rel=3.0e-3)
    assert helium.neutral_he_density == pytest.approx(reference.neutral_he_density, rel=1.0e-6)
    assert helium.neutral_level_population_density == pytest.approx(
        reference.neutral_level_population_density, rel=1.0e-6
    )
    assert helium.neutral_partition_function == pytest.approx(
        reference.neutral_partition_function, rel=1.0e-8
    )
    assert state.total_mass_density == pytest.approx(reference.mass_density, rel=1.0e-6)
    thermodynamics = d6.bulk_metal_thermodynamics(
        d6.atmosphere_with_bulk_metal_state(atmosphere, state), database, abundances
    )
    expected = hummer_mihalas_helium_thermodynamics(
        atmosphere.temperature, atmosphere.gas_pressure, correlated_microfields=True
    )
    for name in (
        "specific_heat_constant_pressure",
        "density_temperature_derivative",
        "adiabatic_temperature_gradient",
    ):
        assert getattr(thermodynamics, name) == pytest.approx(
            getattr(expected, name), rel=1.0e-3
        )


def test_helium_and_hydrogen_join_the_pressure_and_charge_closure(carbon):
    database, _, _ = carbon
    abundances = {"C": 0.0, "He": 1.5, "H": -2.5}
    atmosphere = d6.gray_d6_atmosphere(14_000.0, 8.8, database, abundances, n_depth=16)
    state = d6.bulk_metal_lte_state(atmosphere, database, abundances)
    helium, hydrogen = state.trace_helium_state, state.trace_hydrogen_state
    assert atmosphere.helium_lte_state is not None
    carbon_density = state.element_number_density["C"]
    assert helium.helium_nuclei_density / carbon_density == pytest.approx(10.0**1.5)
    particles = (
        state.electron_density + carbon_density
        + helium.helium_nuclei_density + hydrogen.hydrogen_nuclei_density
    )
    assert particles * BOLTZMANN * atmosphere.temperature == pytest.approx(
        atmosphere.gas_pressure, rel=1.0e-9
    )
    charge = (
        hydrogen.proton_density
        + helium.singly_ionized_he_density + 2.0 * helium.doubly_ionized_he_density
        + sum(
            np.sum(np.arange(population.shape[0])[:, np.newaxis] * population, axis=0)
            for population in state.ion_number_density.values()
        )
    )
    assert charge == pytest.approx(state.electron_density, rel=1.0e-9)
    assert sum(state.mass_fraction.values()) == pytest.approx(1.0)
    thermodynamics = d6.bulk_metal_thermodynamics(atmosphere, database, abundances)
    assert np.all(thermodynamics.adiabatic_temperature_gradient > 0.0)
    assert np.all(thermodynamics.adiabatic_temperature_gradient < 0.41)


def test_helium_lines_enter_the_mixture_opacity(carbon):
    from wd_spectra.helium_ii_stark import read_helium_ii_stark_table
    from wd_spectra.helium_stark import read_helium_stark_table

    database, photoionization, topbase = carbon
    data = ModelData.default()
    tables = (
        read_helium_stark_table(data.helium_i_stark),
        read_helium_ii_stark_table(data.helium_ii_stark),
    )
    abundances = {"C": 0.0, "He": 1.0}
    atmosphere = d6.continuum_d6_atmosphere(
        16_000.0, 8.8, database, photoionization, abundances, n_depth=16,
        topbase_photoionization_database=topbase,
    )
    state = d6.bulk_metal_lte_state(atmosphere, database, abundances)
    wavelength = np.asarray([5800.0, 5877.3])
    options = dict(
        minimum_oscillator_strength=1.0, maximum_lines=0,
        topbase_photoionization_database=topbase,
    )
    with pytest.raises(ValueError, match="helium_line_tables"):
        d6._metal_opacity(
            atmosphere, wavelength, database, state, photoionization,
            include_lines=True, **options,
        )
    lines, _ = d6._metal_opacity(
        atmosphere, wavelength, database, state, photoionization,
        include_lines=True, helium_line_tables=tables, **options,
    )
    continuum, _ = d6._metal_opacity(
        atmosphere, wavelength, database, state, photoionization,
        include_lines=False, **options,
    )
    photosphere = int(np.argmin(np.abs(atmosphere.rosseland_optical_depth - 2.0 / 3.0)))
    assert lines[1, photosphere] > 3.0 * continuum[1, photosphere]
    assert lines[0, photosphere] < 1.5 * continuum[0, photosphere]
    grid = d6._structure_wavelength_grid(atmosphere, database, abundances, 100, 1.0, 0)
    assert np.any(np.abs(grid - 5877.3) < 0.2)
    assert np.any((grid > 504.26) & (grid < 504.4)) and np.any((grid < 504.26) & (grid > 504.1))


def test_c_ii_takes_nist_transition_probabilities_at_every_grade(carbon):
    database, _, _ = carbon
    ion = _stage(database, 1)
    # 3d 4Do3/2 - 4f 4F5/2 (4074.54 A air): Stout f = 2.45, NIST grade E 0.36.
    line = next(
        t for t in ion.transitions
        if abs(t.wavelength_vacuum_angstrom - 4075.69) < 0.03 and t.transition_type == "E1"
    )
    assert 0.25 < line.absorption_oscillator_strength < 0.5
    optical = [
        t.absorption_oscillator_strength for t in ion.transitions
        if t.transition_type == "E1" and 3000.0 < t.wavelength_vacuum_angstrom < 9000.0
    ]
    assert max(optical) < 1.5


def test_c_i_takes_li2021_oscillator_strengths(carbon):
    database, _, _ = carbon
    neutral = _stage(database, 0)
    levels = {level.index: level for level in neutral.levels}

    def log_gf(wavelength):
        line = next(
            t for t in neutral.transitions
            if abs(t.wavelength_vacuum_angstrom - wavelength) < 0.05 and t.transition_type == "E1"
        )
        return np.log10(levels[line.lower_index].statistical_weight * line.absorption_oscillator_strength)

    # Li et al. (2021) Table A2: 5052 A -1.365, 4932 A -1.705, 4771.7 A -1.730.
    assert log_gf(5053.58) == pytest.approx(-1.365, abs=0.002)
    assert log_gf(4933.43) == pytest.approx(-1.705, abs=0.002)
    assert log_gf(4773.08) == pytest.approx(-1.730, abs=0.002)
    # The singlet-to-triplet ratio is 0.2 dex below the NIST compilation's.
    nist_difference = -1.303 - (-1.866)
    assert (log_gf(5053.58) - log_gf(4773.08)) - nist_difference == pytest.approx(-0.198, abs=0.01)
    # Lines outside the calculation keep their NIST values (2s2p3 3Do - 4f, grade C).
    assert log_gf(5040.46) == pytest.approx(np.log10(7 * 2.3e-3), abs=0.01)


def test_carbon_stark_data_replace_kurucz_constants(carbon):
    from wd_spectra.constants import LIGHT_SPEED, PI
    from wd_spectra.warm_dq import carbon_stark_rate_coefficients

    database, _, _ = carbon
    rates = carbon_stark_rate_coefficients()
    # C I 2s2p3 3Do - 4p 3P: one measurement, FWHM 1.03 A at 11,400 K and 6.1e16 cm^-3.
    expected = (
        2.0 * PI * LIGHT_SPEED * 1.03e-8 / 5798.0e-8**2 / 0.610e17
        * (1.0e4 / 11_400.0) ** (1.0 / 6.0)
    )
    width, exponent, shift, _ = rates[(0, "2s.2p3.(3Do)", "2s2.2p.(2Po).4p.(3P)")]
    assert width == pytest.approx(expected)
    assert shift is None and exponent == pytest.approx(1.0 / 6.0)
    # Measured shifts are red and about half the width for 4932 and 5052 A,
    # small for 5380 A (Konjevic et al. 2002).
    singlet = "2s2.2p.(2Po).3s.(1Po)"
    for upper, low, high in (("1S", 0.5, 0.65), ("1D", 0.5, 0.56), ("1P", 0.08, 0.2)):
        width, _, shift, _ = rates[(0, singlet, f"2s2.2p.(2Po).4p.({upper})")]
        assert low < shift / width < high
    # C II 6578/6583 A: STARK-B FWHM 1.05 A at 10 kK and 1e17 cm^-3 (code
    # wavelength 6741.6 A), falling about as T^-0.26 to 0.79 A at 30 kK.
    width, exponent, shift, _ = rates[(1, "2s2.3s.(2S)", "2s2.3p.(2Po)")]
    at_10kK = 2.0 * PI * LIGHT_SPEED * 1.05e-8 * (6580.0 / 6741.6) ** 2 / 6580.0e-8**2 / 1.0e17
    assert width == pytest.approx(at_10kK, rel=0.03)
    assert -0.3 < exponent < -0.2 and shift is None
    # C II 3920 A is shifted to the red, the 5145 A quartet to the blue.
    assert rates[(1, "2s2.3p.(2Po)", "2s2.4s.(2S)")][2] > 0.0
    assert rates[(1, "2s.2p.(3Po).3s.(4Po)", "2s.2p.(3Po).3p.(4P)")][2] < 0.0
    # The default warm-DQ atomic inputs carry these values.
    neutral, ion = _stage(database, 0), _stage(database, 1)
    line_5053 = next(t for t in neutral.transitions if abs(t.wavelength_vacuum_angstrom - 5053.58) < 0.05)
    key = (0, singlet, "2s2.2p.(2Po).4p.(1D)")
    assert line_5053.electron_stark_rate_coefficient_cm3_s == pytest.approx(rates[key][0])
    assert line_5053.electron_stark_shift_rate_coefficient_cm3_s == pytest.approx(rates[key][2])
    assert line_5053.electron_stark_temperature_exponent == pytest.approx(1.0 / 6.0)
    line_4267 = next(
        t for t in ion.transitions
        if abs(t.wavelength_vacuum_angstrom - 4268.2) < 0.05 and t.transition_type == "E1"
    )
    assert line_4267.electron_stark_temperature_exponent < 0.0
    assert line_4267.electron_stark_shift_rate_coefficient_cm3_s is None
    # Every component of a multiplet shares its value.
    multiplet_4772 = {
        t.electron_stark_rate_coefficient_cm3_s for t in neutral.transitions
        if 4766.0 < t.wavelength_vacuum_angstrom < 4778.5 and t.transition_type == "E1"
        and t.electron_stark_temperature_exponent is not None
    }
    assert len(multiplet_4772) == 1


def test_c_i_lines_take_nearest_level_widths(carbon):
    from wd_spectra.constants import LIGHT_SPEED, PI
    from wd_spectra.neutral_stark import ion_terms, nearest_level_stark_width

    database, _, _ = carbon
    neutral = _stage(database, 0)
    terms = ion_terms(neutral)
    # FC78 for C I 5052 A (3s 1Po - 4p 1D) against the measured FWHM of
    # 3.08 A at 11,600 K and 7.93e16 cm^-3 (Jones & Wiese 1984): FC78 quote
    # agreement within 25 per cent for this line.
    width = nearest_level_stark_width(terms, terms["2s2.2p.(2Po).4p.(1D)"], 11_600.0)
    fwhm = (5052.17e-8) ** 2 * width * 0.793e17 / (2.0 * PI * LIGHT_SPEED) * 1.0e8
    assert fwhm == pytest.approx(3.08, rel=0.3)

    def line(wavelength):
        return next(
            t for t in neutral.transitions
            if abs(t.wavelength_vacuum_angstrom - wavelength) < 0.05 and t.transition_type == "E1"
        )

    # 3s 3Po - 5p 3D (4066 A): Kurucz has 6e-6 cm^3 s^-1; the n* = 4.3 upper
    # level is about twenty times broader.
    assert line(4066.39).electron_stark_rate_coefficient_cm3_s > 5.0e-5
    assert line(4066.39).electron_stark_temperature_exponent is not None
    # Measured multiplets keep their laboratory values, and near-hydrogenic
    # Rydberg lines (3p 3D - 4d 3Fo, 7115 A) keep their tabulated constants.
    assert line(5053.58).electron_stark_shift_rate_coefficient_cm3_s is not None
    assert line(7115.14).electron_stark_temperature_exponent is None


def test_c_i_f_series_is_completed_from_the_4f_lines(carbon):
    from wd_spectra.metals import linear_stark_rydberg_level
    from wd_spectra.models.d6 import _atomic_inputs

    database, _, _ = carbon
    neutral = _stage(database, 0)
    assert neutral.rydberg_jk_levels_join_ls_manifold
    levels = {level.index: level for level in neutral.levels}

    def series(n):
        return [
            (t, levels[t.lower_index], levels[t.upper_index]) for t in neutral.transitions
            if levels[t.lower_index].label.startswith("2s.2p3.(3Do")
            and f".{n}f." in levels[t.upper_index].label
        ]

    four, five = series(4), series(5)
    assert len(five) == len(four)
    assert all(4460.0 < t.wavelength_vacuum_angstrom < 4482.0 for t, *_ in five)
    gf = {
        n: sum(lower.statistical_weight * t.absorption_oscillator_strength for t, lower, _ in rows)
        for n, rows in ((4, four), (5, five))
    }
    # n*^-3 scaling (n* = 3.99 -> 4.99) times the transition-energy ratio.
    assert gf[5] / gf[4] == pytest.approx((3.99 / 4.99) ** 3 * 22_350.0 / 19_850.0, rel=0.05)
    # Existing Stout 5f levels keep their energies; the quantum-defect energy
    # of an added level is consistent with them.
    stout = next(l for l in neutral.levels if l.label.endswith(".5f.(3/2[7/2]<4>)"))
    assert stout.energy_wavenumber == pytest.approx(86_469.7, abs=0.2)
    added = next(l for l in neutral.levels if l.label.endswith(".5f.(1/2[7/2]<4>)"))
    assert added.energy_wavenumber == pytest.approx(86_414.7, abs=6.0)  # Kurucz energy
    # The near-hydrogenic nf levels take the quasi-static Rydberg treatment.
    for row in (four[0], five[0]):
        assert linear_stark_rydberg_level(database, neutral, row[2]) is not None
    # A jK level mixes with LS levels of its parent term and with jK levels of
    # its own parent J, not with those of the other parent J.
    from wd_spectra.metals import _same_rydberg_core

    half, three_half, ls = "2s2.2p.(2Po<1/2>)", "2s2.2p.(2Po<3/2>)", "2s2.2p.(2Po)"
    assert _same_rydberg_core(neutral, half, ls) and _same_rydberg_core(neutral, ls, half)
    assert _same_rydberg_core(neutral, half, half)
    assert not _same_rydberg_core(neutral, half, three_half)
    # D6 uses the same rule for every ion. Without it a jK level stays
    # outside the LS manifold.
    from dataclasses import replace

    plain, _, _ = _atomic_inputs(ModelData.default(), ("C",))
    assert _stage(plain, 0).rydberg_jk_levels_join_ls_manifold
    plain_neutral = replace(_stage(plain, 0), rydberg_jk_levels_join_ls_manifold=False)
    level_4f = next(l for l in plain_neutral.levels if l.label.endswith(".4f.(1/2[7/2]<4>)"))
    assert linear_stark_rydberg_level(plain, plain_neutral, level_4f) is None


def test_stark_shift_moves_the_line_centre_with_electron_density(carbon):
    from dataclasses import replace
    from types import MappingProxyType

    from wd_spectra.constants import LIGHT_SPEED, PI
    from wd_spectra.metals import AtomicDatabase, metal_line_mass_absorption_coefficient

    database, photoionization, topbase = carbon
    abundances = {"C": 0.0}
    atmosphere = d6.continuum_d6_atmosphere(
        14_000.0, 8.8, database, photoionization, abundances, n_depth=12,
        topbase_photoionization_database=topbase,
    )
    state = d6.bulk_metal_lte_state(atmosphere, database, abundances)
    neutral = _stage(database, 0)
    line = next(t for t in neutral.transitions if abs(t.wavelength_vacuum_angstrom - 5053.58) < 0.05)
    keys = [("C", 0, line.lower_index, line.upper_index)]
    shift_rate = 1.0e-5

    def with_shift(rate):
        changed = replace(line, electron_stark_shift_rate_coefficient_cm3_s=rate)
        return AtomicDatabase(
            MappingProxyType({
                **database.ions,
                ("C", 0): replace(neutral, transitions=tuple(
                    changed if t is line else t for t in neutral.transitions
                )),
            }),
            source=database.source,
        )

    unshifted, shifted = with_shift(None), with_shift(shift_rate)
    wavelength = np.arange(line.wavelength_vacuum_angstrom - 40.0, line.wavelength_vacuum_angstrom + 40.0, 0.01)
    options = dict(transition_keys=keys, minimum_oscillator_strength=1.0e-8)
    rest = metal_line_mass_absorption_coefficient(atmosphere, wavelength, unshifted, state, **options)
    moved = metal_line_mass_absorption_coefficient(atmosphere, wavelength, shifted, state, **options)
    depth = int(np.argmin(np.abs(atmosphere.rosseland_optical_depth - 0.1)))
    expected = (
        (line.wavelength_vacuum_angstrom * 1.0e-8) ** 2 * shift_rate
        * state.electron_density[depth]
        * (atmosphere.temperature[depth] / 1.0e4) ** (1.0 / 6.0)
        / (2.0 * PI * LIGHT_SPEED) * 1.0e8
    )
    assert expected > 0.3
    peak_shift = wavelength[np.argmax(moved[:, depth])] - wavelength[np.argmax(rest[:, depth])]
    assert peak_shift == pytest.approx(expected, abs=0.02)
    # The shift moves, but does not change, the line strength.
    assert trapezoid(moved[:, depth], wavelength) == pytest.approx(
        trapezoid(rest[:, depth], wavelength), rel=1.0e-3
    )


def test_radiative_cold_start_continues_from_the_convective_structure(monkeypatch):
    import wd_spectra.models.warm_dq as module
    from wd_spectra.spectrum import Spectrum

    calls = []

    def fake_structure(effective_temperature, logg, *args, mixing_length_alpha=None,
                       initial_temperature=None, initial_column_mass=None, **kwargs):
        calls.append((mixing_length_alpha, initial_temperature is not None))
        n = 6
        temperature = (
            np.linspace(9.0e3, 2.0e4, n) if initial_temperature is None
            else np.asarray(initial_temperature) * 1.01
        )
        return Atmosphere(
            effective_temperature, logg, np.geomspace(1.0e-4, 1.0e2, n),
            np.geomspace(1.0e-4, 1.0, n), temperature, np.geomspace(1.0e4, 1.0e8, n),
            np.ones(n), np.zeros(n), np.zeros(n), np.ones(n),
            {"radiative_equilibrium_converged": True},
        )

    def fake_synthesis(atmosphere, wavelength, *args, **kwargs):
        return Spectrum(np.asarray(wavelength), np.ones(len(wavelength)), {})

    monkeypatch.setattr(module, "radiative_equilibrium_d6_atmosphere", fake_structure)
    monkeypatch.setattr(module, "synthesize_d6_spectrum", fake_synthesis)
    monkeypatch.setattr(module, "warn_if_atmosphere_not_converged", lambda *a: "converged")
    wavelength = np.linspace(4000.0, 4010.0, 5)
    result = module.compute_warm_dq(WarmDQConfig(mixing_length_alpha=None), wavelength)
    assert calls == [(module.RADIATIVE_CONTINUATION_MIXING_LENGTH_ALPHA, False), (None, True)]
    assert "radiative_cold_start" in result.atmosphere.metadata
    calls.clear()
    module.compute_warm_dq(WarmDQConfig(), wavelength)
    assert calls == [(1.25, False)]


def _deep_line_support(scale_factor, maximum_rosseland_optical_depth):
    """Support of a line that is thin in the photosphere and strong below it."""

    from wd_spectra.metals import optically_thick_line_minimum_half_window_angstrom

    n = 12
    tau = np.geomspace(1.0e-3, 1.0e2, n)
    atmosphere = Atmosphere(
        20_000.0, 9.0, tau, np.geomspace(1.0e-6, 1.0e-1, n),
        np.geomspace(1.2e4, 5.5e4, n), np.geomspace(1.0e2, 1.0e7, n),
        np.ones(n), np.zeros(n), np.zeros(n), np.ones(n), {},
    )
    # Lower-level number per gram: rises by twelve orders of magnitude inward.
    population = scale_factor * 1.0e20 * (tau / tau[-1]) ** 2.5
    # Integrated strength (pi e^2 / m c) f of a line with f = 0.1.
    return optically_thick_line_minimum_half_window_angstrom(
        atmosphere, 1175.7, 2.654e-3, np.full(n, 0.01), np.full(n, 0.02), population,
        maximum_rosseland_optical_depth=maximum_rosseland_optical_depth,
        edge_optical_depth=1.0e-3, maximum_half_window_angstrom=100.0,
    )


def test_whole_column_line_support_is_wide_and_does_not_jump():
    factors = np.geomspace(1.0, 1.0e4, 41)
    photospheric = np.array([_deep_line_support(factor, 2.0) for factor in factors])
    whole_column = np.array([_deep_line_support(factor, np.inf) for factor in factors])
    # With the spectrum-forming limit the support doubles somewhere in a
    # range of line strengths: a discontinuous opacity for the deep layers.
    assert np.any(photospheric[1:] >= 1.99 * photospheric[:-1])
    # Through the whole column the deep wing is never thin: every strength
    # takes the same, maximum support.
    assert np.all(whole_column == 100.0)
    assert np.all(whole_column >= photospheric)
    with pytest.raises(ValueError):
        _deep_line_support(1.0, np.nan)


def test_structure_requires_thin_line_wings_through_the_whole_column():
    import inspect

    from wd_spectra.d6 import d6_structure_opacity_function, radiative_equilibrium_d6_atmosphere
    from wd_spectra.metals import metal_line_mass_absorption_coefficient

    # The structure (D6 and WarmDQ) requires thin wings in every layer; the
    # emergent spectrum keeps the photospheric limit.
    for function in (radiative_equilibrium_d6_atmosphere, d6_structure_opacity_function):
        parameter = inspect.signature(function).parameters[
            "profile_support_maximum_rosseland_optical_depth"
        ]
        assert parameter.default == np.inf
    spectrum = inspect.signature(metal_line_mass_absorption_coefficient).parameters[
        "profile_support_maximum_rosseland_optical_depth"
    ]
    assert spectrum.default == 2.0
