"""Fixed-state regression from cold DZ 10000/g8 Ca=-9 H/He=-5.

The 48-depth boundary-expanded state exposed cancellation, not a grid change.
"""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest
from wd_spectra.atmosphere import Atmosphere
import wd_spectra.adaptive_structure as adaptive

def test_thin_dz_source_closure_and_cached_response(monkeypatch):
    d={k:np.asarray(v) for k,v in json.loads((Path(__file__).parent/'data/dz_closure_17034.json').read_text()).items()}
    m=d['column_mass'];n=len(m)
    seed=Atmosphere(effective_temperature=10000.,logg=8.,column_mass=m,
        rosseland_optical_depth=m,gas_pressure=1e8*m,temperature=d['temperature'],
        mass_density=np.ones(n),neutral_h_density=np.zeros(n),proton_density=np.zeros(n),electron_density=np.ones(n),metadata={})
    class Checked(Exception):pass
    def inspect(initial,evaluate,**options):
        value=evaluate(initial,False)
        assert value.payload['scattering_source_maximum_relative_residual'] < 1e-12
        cached=evaluate(initial,True)
        np.testing.assert_array_equal(cached.residual,value.residual)
        assert np.all(np.isfinite(cached.jacobian))
        raise Checked
    monkeypatch.setattr(adaptive,'solve_trust_region_newton',inspect)
    with pytest.raises(Checked):
        adaptive.solve_adaptive_lte_structure(seed,d['wavelength'],
            with_temperature=lambda t:replace(seed,temperature=t),
            true_absorption=lambda a:d['absorption'],scattering_opacity=lambda a:d['scattering'],
            rosseland_opacity=lambda a:np.ones(n),thermodynamics=lambda a:None,
            mixing_length_alpha=None,max_iterations=1,temperature_tolerance=3e-4,
            flux_tolerance=3e-3,n_angle=3,initial_temperature_was_supplied=True,
            resume_supplied_structure_in_formal_flux_phase=True,
            project_initial_convective_gradient=False,use_initial_bolometric_rescaling=False)
