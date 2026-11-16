"""Custom He I atoms retain diagnostics without serializing runtime callables."""
from dataclasses import replace
from functools import partial
import json
import numpy as np
import pytest
from wd_spectra.atmosphere import gray_helium_atmosphere
from wd_spectra.helium_i_atom import read_tlusty_helium_i_atom
from wd_spectra.helium_nlte import CoupledHeliumNLTEState, _coupled_state_views
from wd_spectra.hot_nlte import HotPopulationState
from wd_spectra.models.common import ModelData, ModelResult, _jsonable, save_model_result
from wd_spectra.models.sdb import SdBConfig
from wd_spectra.spectrum import Spectrum

@pytest.mark.parametrize('converged', [False, True])
def test_he24_population_export_preserves_arrays_and_failure_diagnostics(tmp_path, converged):
    atom=read_tlusty_helium_i_atom(ModelData.default().tlusty_atoms/'he1.dat')
    assert isinstance(atom.photoionization, partial)
    neutral=np.arange(1.,97.).reshape(4,24);ion=np.ones((4,32));continuum=np.ones(4)
    defect=0.0273809 if not converged else 1e-5
    helium=CoupledHeliumNLTEState(
        neutral_population_density=neutral,singly_ionized_population_density=ion,
        doubly_ionized_he_density=continuum,lte_neutral_population_density=neutral,
        lte_singly_ionized_population_density=ion,lte_doubly_ionized_he_density=continuum,
        neutral_departure_coefficient=np.ones_like(neutral),
        singly_ionized_departure_coefficient=ion,doubly_ionized_departure_coefficient=continuum,
        iterations=600,converged=converged,maximum_relative_population_change=defect,
        metadata={'model_atom':atom.name,'diagnostic':'retained'},helium_i_atom=atom)
    population=HotPopulationState(helium,None,converged,600,defect)
    atmosphere=gray_helium_atmosphere(35000.,5.8,n_depth=4)
    wave=np.array([4000.,5000.]);status='converged' if converged else 'unconverged'
    result=ModelResult('sdB',atmosphere,Spectrum(wave,np.ones(2),{}),
        SdBConfig(effective_temperature=35000.,logg=5.8,log_helium_to_hydrogen=-1,helium_i_atom='24'),
        {'atmosphere_convergence_status':status,'nlte_populations_converged':converged},population)
    save_model_result(result,tmp_path)
    with np.load(tmp_path/'populations.npz',allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['helium.neutral_population_density'],neutral)
        metadata=json.loads(str(saved['metadata_json']))
        assert metadata['converged'] is converged
        assert metadata['iterations']==600
        assert metadata['maximum_relative_population_change']==defect
        assert metadata['helium']['metadata']['diagnostic']=='retained'
        identity=metadata['helium']['metadata']['helium_i_atom_identity']
        assert identity['name']==atom.name
        assert identity['source']==atom.source
        assert identity['term_label']==list(atom.label)
        assert 'helium_i_atom' not in metadata['helium']
    record=json.loads((tmp_path/'metadata.json').read_text())
    assert record['model_metadata']['atmosphere_convergence_status']==status
    assert 'helium_i_atom' not in _jsonable(helium)
    assert callable(atom.photoionization)
    assert result.population_state.helium.helium_i_atom is atom

    # The neutral view carries the same live callback-bearing atom.
    neutral_view,_=_coupled_state_views(atmosphere,helium)
    save_model_result(replace(result,population_state=neutral_view),tmp_path/'neutral')
    with np.load(tmp_path/'neutral/populations.npz',allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['population_density'],neutral)
        record=json.loads(str(saved['metadata_json']))
        assert record['term_label']==list(atom.label)
        assert record['converged'] is converged
        assert record['maximum_relative_population_change']==defect
        assert 'helium_i_atom' not in record
        assert record['metadata']['helium_i_atom_identity']['source']==atom.source
    # Unknown callables are not silently stringified or ignored globally.
    with pytest.raises(TypeError,match='partial'):
        json.dumps(_jsonable(atom))
