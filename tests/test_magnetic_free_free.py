"""Magnetic Coulomb logarithm option of the cyclotron term."""

from types import SimpleNamespace

import numpy as np

from wd_spectra.magnetic_continuum import (
    magnetic_free_free_coulomb_log_ratio,
    magnetized_free_electron_manifolds,
)

HYDROGEN_MASS = 1.6735575e-24


def _one_point(temperature=18840.0, density=4.1e16):
    return SimpleNamespace(
        temperature=np.array([temperature]),
        electron_density=np.array([density]),
        proton_density=np.array([density]),
        mass_density=np.array([density * HYDROGEN_MASS]),
    )


def test_weak_field_is_unchanged():
    # beta = hbar w_c / kT < 0.05 at 1 MG and 18.8 kK: the option is the identity.
    atmosphere = _one_point()
    wavelength = np.array([3800.0, 5000.0, 7000.0])
    off = magnetized_free_electron_manifolds(atmosphere, wavelength, 1.0, 0.5)
    on = magnetized_free_electron_manifolds(
        atmosphere, wavelength, 1.0, 0.5, magnetic_coulomb_logarithm=True
    )
    for name in ("absorption", "scattering", "dispersion"):
        for a, b in zip(getattr(off, name), getattr(on, name)):
            np.testing.assert_array_equal(a, b)


def test_ratio_at_j2247_photosphere():
    # 18 840 K, 250 MG (beta 1.78): Lambda_perp / Lambda_cl from a direct
    # evaluation of Potekhin & Chabrier (2003) Eq. 41 (1.47 at 4000 A,
    # 0.57 at 6000 A) and Lambda_par / Lambda_cl (1.03, 0.89).
    beta = 1.054571817e-27 * 4.80320471e-10 * 250e6 / (9.1093837e-28 * 2.99792458e10) / (
        1.380649e-16 * 18840.0
    )
    x = 4284.4 / np.array([4000.0, 6000.0])
    np.testing.assert_allclose(
        magnetic_free_free_coulomb_log_ratio(beta, x, transverse=True), [1.47, 0.57], rtol=0.02
    )
    np.testing.assert_allclose(
        magnetic_free_free_coulomb_log_ratio(beta, x, transverse=False), [1.03, 0.89], rtol=0.02
    )


def test_option_scales_mode_free_free():
    atmosphere = _one_point()
    wavelength = np.array([6000.0])
    off = magnetized_free_electron_manifolds(atmosphere, wavelength, 250.0, 0.5)
    on = magnetized_free_electron_manifolds(
        atmosphere, wavelength, 250.0, 0.5, magnetic_coulomb_logarithm=True
    )
    # Far from resonance the q = -1 and q = 0 absorption scale with the ratio.
    ratio = on.absorption[0][0, 0] / off.absorption[0][0, 0]
    np.testing.assert_allclose(ratio, 0.57, rtol=0.03)
    np.testing.assert_allclose(on.scattering[0], off.scattering[0], rtol=1e-6)
