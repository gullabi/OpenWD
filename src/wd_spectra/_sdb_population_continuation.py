"""Initialize the sdB statistical equilibrium with progressively stronger lines."""
import logging
from dataclasses import replace

import numpy as np

from . import helium_nlte as he
from . import hot_nlte as hn
from . import multilevel_nlte as h
from ._hot_mali import HotMALI
from ._hot_rates import PreparedLineAverages
from .nlte import _profile_averaged_mean_intensity_nu
from .nlte_core import _FixedTransferCache

LOGGER = logging.getLogger(__name__)
LINE_STRENGTHS = (0.0, 0.05, 0.25, 0.65, 1.0)
_INITIALIZATION_STAGE_LIMIT = 75


def _stage_coefficients(model, atmosphere, wave, state, cache, strength):
    if strength == 1.0:
        return model.transfer_coefficients(atmosphere, wave, state, _cache=cache)
    continuum = model.transfer_coefficients(atmosphere, wave, state, lines=False, _cache=cache)
    if strength == 0.0:
        return continuum
    full = model.transfer_coefficients(atmosphere, wave, state, _cache=cache)
    # Electron scattering is independent of the line contribution.
    return replace(
        continuum,
        true_absorption=(1.0 - strength) * continuum.true_absorption + strength * full.true_absorption,
        thermal_emissivity=(1.0 - strength) * continuum.thermal_emissivity + strength * full.thermal_emissivity,
    )


def _native_line_fields(wave, mean, groups):
    return [{key: np.average([
        _profile_averaged_mean_intensity_nu(
            p.continuum.wavelength_angstrom, p.lte_line_opacity,
            mean[np.searchsorted(wave, p.continuum.wavelength_angstrom)])
        for p in problems], axis=0,
        weights=[p.line.absorption_oscillator_strength for p in problems])
        for key, problems in group.items()} for group in groups]


def solve_sdb_populations(model, atmosphere):
    """Follow the ionization balance from continuum transfer to full NLTE."""
    groups = model._line_problems(atmosphere)
    grids = [he.default_neutral_helium_continuum_wavelength(model.helium_i_atom),
             he.default_helium_ii_continuum_wavelength(model.maximum_helium_ii_level)]
    if model.log_hydrogen_to_helium is not None:
        grids.append(h._default_continuum_wavelength(model.maximum_hydrogen_level))
    grids.extend(p.continuum.wavelength_angstrom for group in groups
                 for problems in group.values() for p in problems)
    wave = np.unique(np.concatenate(grids))
    base = model._rate_state(atmosphere)
    _, reference = hn.population_arrays(base)
    current = hn.with_departures(base, np.ones_like(reference))
    floor = 1e-12 * reference.sum(axis=1)[:, None]
    cache = _FixedTransferCache(atmosphere, wave)
    accelerator = HotMALI(model, atmosphere, wave, groups, prepare_line_stages=True)
    profiles = PreparedLineAverages(wave, groups, atmosphere.n_depth)
    zero = np.zeros(atmosphere.n_depth)
    continuum_fields = [{key: zero for key in group} for group in groups]
    total_iterations = 0
    stages = []
    for strength in LINE_STRENGTHS:
        history = []
        stage_limit = model.population_maximum_iterations - total_iterations
        if strength < 1.0:
            # Reserve a full-strength evaluation, even on a short budget.
            stage_limit = min(stage_limit - 1, _INITIALIZATION_STAGE_LIMIT)
            if stage_limit == 0:
                continue
        best_change = float('inf')
        best_state = current
        best_metadata = current.helium.metadata
        for iteration in range(1, stage_limit + 1):
            coefficients = _stage_coefficients(model, atmosphere, wave, current, cache, strength)
            _, field, _ = hn.transfer_field(
                atmosphere, coefficients, n_angle=model.n_angle, check_source=False)
            mean = field.mean_intensity
            if strength == 1.0:
                fields = _native_line_fields(wave, mean, groups)
            elif strength == 0.0:
                fields = continuum_fields
            else:
                fields = profiles.fields(mean)
            candidate = accelerator.state(current, coefficients, mean, fields, line_strength=strength)
            old, _ = hn.population_arrays(current)
            new, _ = hn.population_arrays(candidate)
            change = float(np.max(abs(new - old) / np.maximum(new, floor)))
            total_iterations += 1
            LOGGER.info('Population continuation: line strength %.6g iteration %d stage %d residual %.9g',
                        strength, total_iterations, iteration, change)
            if change < best_change:
                best_change = change
                best_state = current
                best_metadata = candidate.helium.metadata
            if change < model.population_tolerance or iteration == stage_limit:
                break
            log_old = np.log(old / reference)
            log_new = np.log(new / reference)
            weights = np.minimum(1.0, np.maximum(old, new) / floor)
            proposal, _ = h._anderson_log_population_update(
                log_old, log_new, history, depth=12, mixing=0.65, maximum_step=0.65,
                residual_weights=weights)
            if proposal is None:
                proposal = log_old + 0.3 * (log_new - log_old)
            current = hn.with_departures(base, np.exp(proposal))
        current = best_state
        stages.append({
            'line_strength': strength,
            'iterations': iteration,
            'stage_converged': best_change < model.population_tolerance,
            'population_defect': best_change,
        })
        if total_iterations == model.population_maximum_iterations:
            break
    # Only the full-strength map certifies the state that generated its field.
    converged = strength == 1.0 and best_change < model.population_tolerance
    current = hn.population_status(current, converged, total_iterations, best_change)
    metadata = {**best_metadata, 'population_continuation_stages': stages}
    return replace(current, helium=replace(current.helium, metadata=metadata))
