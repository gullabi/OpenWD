from __future__ import annotations

import numpy as np

from wd_spectra import gray_helium_atmosphere
from wd_spectra.metals import metal_lte_state, read_stout_atomic_database
from wd_spectra.models.common import ModelData
from wd_spectra.molecules import helium_hydride_ion_formation_constant


def _states():
    atmosphere = gray_helium_atmosphere(6_000.0, 8.0, n_depth=40)
    database = read_stout_atomic_database(
        ModelData.default().stout, elements=("Mg",), maximum_charge=2
    )
    kwargs = dict(reference_species="He", log_hydrogen_abundance=-4.0)
    without = metal_lte_state(
        atmosphere, database, {"Mg": -9.0}, include_helium_hydride_ion=False, **kwargs
    )
    with_ion = metal_lte_state(
        atmosphere, database, {"Mg": -9.0}, include_helium_hydride_ion=True, **kwargs
    )
    return atmosphere, without, with_ion


def test_helium_hydride_ion_conserves_hydrogen_and_follows_mass_action():
    atmosphere, without, with_ion = _states()
    assert without.helium_hydride_ion_density is None
    hydrogen = with_ion.trace_hydrogen_state
    hydride = with_ion.helium_hydride_ion_density
    np.testing.assert_allclose(
        hydrogen.neutral_h_density + hydrogen.proton_density + hydride,
        hydrogen.hydrogen_nuclei_density, rtol=1e-12,
    )
    helium = atmosphere.helium_lte_state
    neutral_helium = helium.neutral_he_density
    expected = (
        helium_hydride_ion_formation_constant(atmosphere.temperature)
        * neutral_helium * hydrogen.proton_density
    )
    hot = hydrogen.proton_density > 1.0e-30 * hydrogen.hydrogen_nuclei_density
    np.testing.assert_allclose(hydride[hot], expected[hot], rtol=2e-3)


def test_helium_hydride_ion_raises_hydrogen_electron_supply():
    _, without, with_ion = _states()
    assert np.all(with_ion.electron_density >= without.electron_density * (1.0 - 1e-9))
    assert np.max(with_ion.electron_density / without.electron_density) > 1.01


def test_dz_and_daz_defaults_include_heh_plus_and_kirchhoff_wings():
    from wd_spectra import DAZConfig, DZConfig

    assert DZConfig().helium_hydride_ion
    assert DZConfig().ca_ii_crd_frequency_dependent_source
    assert DAZConfig().ca_ii_crd_frequency_dependent_source
