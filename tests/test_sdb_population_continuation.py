"""Continuation must satisfy the original map within one shared work budget."""
from dataclasses import replace
import numpy as np
import pytest
from wd_spectra._sdb_population_continuation import solve_sdb_populations
from wd_spectra.hot_nlte import population_arrays
from test_hot_mali import sdb, _population_grid_and_fields  # noqa: F401
from test_hot_nlte import atom  # noqa: F401


def test_converged_initialization_is_not_full_nlte_convergence(sdb, monkeypatch):
    from wd_spectra import _sdb_population_continuation as continuation
    model, atmosphere = sdb
    monkeypatch.setattr(continuation, 'FRACTIONS', (0.5,))
    result = solve_sdb_populations(
        replace(model, population_maximum_iterations=200), atmosphere)
    assert result.helium.metadata['population_continuation_stages'][0]['stage_converged']
    assert not result.converged


@pytest.mark.parametrize('budget', [1, 3])
def test_incomplete_initialization_does_not_certify_populations(sdb, budget):
    model, atmosphere = sdb
    model = replace(model, population_maximum_iterations=budget)
    result = solve_sdb_populations(model, atmosphere)
    assert not result.converged
    assert result.iterations == budget
    stages = result.helium.metadata['population_continuation_stages']
    assert sum(stage['iterations'] for stage in stages) == budget
    assert stages[-1]['fraction'] < 1


def test_continuation_solution_passes_original_population_solver(sdb):
    model, atmosphere = sdb
    model = replace(model, population_maximum_iterations=600, population_tolerance=1e-4)
    result = solve_sdb_populations(model, atmosphere)
    assert result.converged
    stages = result.helium.metadata['population_continuation_stages']
    assert stages[-1]['fraction'] == 1
    assert result.iterations == sum(stage['iterations'] for stage in stages) <= 600
    checked = replace(model, population_maximum_iterations=1).solve_populations(
        atmosphere, result, accelerated_lambda=True)
    assert checked.converged
    _, wave, _, mean, fields = _population_grid_and_fields(model, atmosphere, result)
    plain = model._rate_state(atmosphere, wave, mean, *fields)
    old, reference = population_arrays(result)
    new, _ = population_arrays(plain)
    assert np.max(abs(new-old)/np.maximum(new, 1e-12*reference.sum(axis=1)[:, None])) < 1e-4
