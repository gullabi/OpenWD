"""Continuation must satisfy the original map within one shared work budget."""
from dataclasses import replace
import numpy as np
import pytest
from wd_spectra import helium_nlte as he, multilevel_nlte as h
from wd_spectra._hot_rates import PreparedHeliumRates
from wd_spectra._sdb_population_continuation import solve_sdb_populations
from wd_spectra.hot_nlte import population_arrays
from test_hot_mali import sdb, _population_grid_and_fields  # noqa: F401
from test_hot_nlte import atom  # noqa: F401


def test_line_stages_match_direct_rate_assembly(sdb):
    model, atmosphere = sdb
    groups, wave, _, mean, fields = _population_grid_and_fields(
        model, atmosphere, model._rate_state(atmosphere))
    prepared = PreparedHeliumRates(model, atmosphere, wave, groups, prepare_line_stages=True)
    for strength in (0.0, 0.25, 1.0):
        direct = he.solve_coupled_helium_statistical_equilibrium(
            atmosphere, model.collision_data, **prepared.kwargs,
            neutral_line_mean_intensity_nu=fields[0], helium_ii_line_mean_intensity_nu=fields[1],
            neutral_continuum_wavelength_angstrom=wave, neutral_continuum_mean_intensity_lambda=mean,
            helium_ii_continuum_wavelength_angstrom=wave, helium_ii_continuum_mean_intensity_lambda=mean,
            _line_strength=strength, _return_rate_matrix=True)
        actual = prepared.rate_matrix(mean, *fields[:2], line_strength=strength)
        np.testing.assert_allclose(actual, direct, rtol=2e-13, atol=0)

    def hydrogen_rates(strength):
        return h.solve_multilevel_hydrogen_statistical_equilibrium(
            atmosphere, model.collision_data, maximum_level=model.maximum_hydrogen_level,
            line_mean_intensity_nu=fields[2], continuum_wavelength_angstrom=wave,
            continuum_mean_intensity_lambda=mean, _line_strength=strength, _return_rate_matrix=True)

    continuum, full = hydrogen_rates(0.0), hydrogen_rates(1.0)
    assert np.any(full > continuum)
    np.testing.assert_array_equal(continuum[:, -1, :], full[:, -1, :])
    np.testing.assert_array_equal(continuum[:, :, -1], full[:, :, -1])
    np.testing.assert_allclose(hydrogen_rates(0.25), 0.75 * continuum + 0.25 * full, rtol=2e-13)


def test_converged_initialization_is_not_full_nlte_convergence(sdb, monkeypatch):
    from wd_spectra import _sdb_population_continuation as continuation
    model, atmosphere = sdb
    model = replace(model, population_tolerance=0.05)
    monkeypatch.setattr(continuation, 'LINE_STRENGTHS', (0.0,))
    result = solve_sdb_populations(
        replace(model, population_maximum_iterations=200), atmosphere)
    assert result.helium.metadata['population_continuation_stages'][0]['stage_converged']
    assert not result.converged

    # The last available evaluation measures the full-strength residual.
    monkeypatch.setattr(continuation, 'LINE_STRENGTHS', (0.0, 1.0))
    budget = result.iterations + 1
    bounded = solve_sdb_populations(
        replace(model, population_maximum_iterations=budget), atmosphere)
    assert not bounded.converged
    assert bounded.iterations == budget
    stages = bounded.helium.metadata['population_continuation_stages']
    assert stages[:-1] == result.helium.metadata['population_continuation_stages']
    assert stages[-1]['line_strength'] == 1 and stages[-1]['iterations'] == 1


@pytest.mark.parametrize('budget', [1, 3])
def test_incomplete_initialization_does_not_certify_populations(sdb, budget):
    model, atmosphere = sdb
    model = replace(model, population_maximum_iterations=budget)
    result = solve_sdb_populations(model, atmosphere)
    assert not result.converged
    assert result.iterations == budget
    stages = result.helium.metadata['population_continuation_stages']
    assert sum(stage['iterations'] for stage in stages) == budget
    assert stages[-1]['line_strength'] == 1
    assert len(stages) == (1 if budget == 1 else 2)


def test_continuation_solution_passes_original_population_solver(sdb):
    model, atmosphere = sdb
    model = replace(model, population_maximum_iterations=600, population_tolerance=1e-4)
    result = solve_sdb_populations(model, atmosphere)
    assert result.converged
    stages = result.helium.metadata['population_continuation_stages']
    assert stages[-1]['line_strength'] == 1
    assert result.iterations == sum(stage['iterations'] for stage in stages) <= 600
    checked = replace(model, population_maximum_iterations=1).solve_populations(
        atmosphere, result, accelerated_lambda=True)
    assert checked.converged
    _, wave, _, mean, fields = _population_grid_and_fields(model, atmosphere, result)
    assert len(fields[1]) > 0
    assert result.helium.metadata['helium_ii_radiation_coupled_transition_count'] == len(fields[1])
    plain = model._rate_state(atmosphere, wave, mean, *fields)
    old, reference = population_arrays(result)
    new, _ = population_arrays(plain)
    assert np.max(abs(new-old)/np.maximum(new, 1e-12*reference.sum(axis=1)[:, None])) < 1e-4
