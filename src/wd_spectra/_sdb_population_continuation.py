"""Initialize hybrid sdB populations by LTE-to-NLTE continuation.

Intermediate stages regularize the nonlinear population map toward LTE. Only
fraction one solves the requested equations and may certify convergence. All
stages share the configured population evaluation budget and final tolerance.
"""
import logging
from dataclasses import replace

import numpy as np

from wd_spectra import hot_nlte as hn
from wd_spectra import helium_nlte as he
from wd_spectra import multilevel_nlte as h
from wd_spectra.nlte import _profile_averaged_mean_intensity_nu
from wd_spectra.nlte_core import _FixedTransferCache
from wd_spectra._hot_mali import HotMALI
LOGGER = logging.getLogger(__name__)
# Small final increments keep the outer helium ionization balance on the
# convergent path. Fractions below one are initialization, never certification.
FRACTIONS = (0.5, 0.75, 0.9, 0.95, 0.975, 1.0)


def solve_sdb_populations(model, atmosphere):
    """Solve cold sdB populations, sharing one work budget across all stages."""
    groups = model._line_problems(atmosphere)
    grids = [he.default_neutral_helium_continuum_wavelength(model.helium_i_atom), he.default_helium_ii_continuum_wavelength(model.maximum_helium_ii_level)]
    if model.log_hydrogen_to_helium is not None:
        grids.append(h._default_continuum_wavelength(model.maximum_hydrogen_level))
    grids.extend((p.continuum.wavelength_angstrom for group in groups for problems in group.values() for p in problems))
    wave = np.unique(np.concatenate(grids))
    base = model._rate_state(atmosphere)
    _, reference = hn.population_arrays(base)
    current = hn.with_departures(base, np.ones_like(reference))
    cache = _FixedTransferCache(atmosphere, wave)
    accelerator = HotMALI(model, atmosphere, wave, groups)
    iteration = 0
    stages = []
    converged = False
    change = float('inf')
    for fraction in FRACTIONS:
        history = []
        stage_iterations = 0
        stage_converged = False
        while iteration < model.population_maximum_iterations:
            iteration += 1
            stage_iterations += 1
            c = model.transfer_coefficients(atmosphere, wave, current, _cache=cache)
            _, field, _ = hn.transfer_field(atmosphere, c, n_angle=model.n_angle, check_source=False)
            fields = []
            for group in groups:
                fields.append({key: np.average([
                    _profile_averaged_mean_intensity_nu(
                        p.continuum.wavelength_angstrom, p.lte_line_opacity,
                        field.mean_intensity[np.searchsorted(
                            wave, p.continuum.wavelength_angstrom)])
                    for p in problems], axis=0,
                    weights=[p.line.absorption_oscillator_strength for p in problems])
                    for key, problems in group.items()})
            candidate = accelerator.state(current, c, field.mean_intensity, fields)
            old, _ = hn.population_arrays(current)
            new, _ = hn.population_arrays(candidate)
            floor = 1e-12 * reference.sum(axis=1)[:, None]
            change = float(np.max(abs(new - old) / np.maximum(new, floor)))
            log_old = np.log(old / reference)
            mapped = hn.with_departures(base, np.exp(fraction * np.log(new / reference)))
            mapped_n, _ = hn.population_arrays(mapped)
            homotopy = float(np.max(abs(mapped_n - old) / np.maximum(mapped_n, floor)))
            LOGGER.info('Population continuation: fraction %.6g iteration %d stage %d original %.9g homotopy %.9g', fraction, iteration, stage_iterations, change, homotopy)
            if fraction == 1.0 and change < model.population_tolerance:
                # Certify the state that generated this radiation field,
                # not an unchecked final population update.
                stage_converged = True
                converged = True
                break
            if fraction < 1.0 and homotopy < min(1e-05, 0.1 * model.population_tolerance):
                stage_converged = True
                break
            if iteration == model.population_maximum_iterations:
                break
            log_new = np.log(mapped_n / reference)
            weights = np.minimum(1.0, np.maximum(old, mapped_n) / floor)
            proposal, _ = h._anderson_log_population_update(log_old, log_new, history, depth=12, mixing=0.65, maximum_step=0.5, residual_weights=weights)
            if proposal is None:
                proposal = log_old + 0.3 * (log_new - log_old)
            current = hn.with_departures(base, np.exp(proposal))
        stages.append({'fraction': fraction, 'iterations': stage_iterations, 'stage_converged': stage_converged, 'original_population_defect': change, 'homotopy_defect': homotopy})
        if not stage_converged or converged:
            break
    current = hn.population_status(current, converged, iteration, change)
    current = replace(current, helium=replace(current.helium, metadata={**current.helium.metadata, 'population_continuation_stages': stages}))
    return current
