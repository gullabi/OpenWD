"""Coulomb-approximation integrals and simplified semiclassical neutral Stark widths."""

from __future__ import annotations

import numpy as np
import pytest

from wd_spectra.neutral_stark import (
    _Term,
    coulomb_dipole_radial_integral,
    simplified_semiclassical_stark,
    wigner_6j,
)

LIGHT_SPEED = 2.99792458e10


@pytest.mark.parametrize(
    "n1, l1, n2, l2, exact",
    [(1, 0, 2, 1, 1.2903), (2, 1, 3, 2, 4.7476), (2, 0, 3, 1, 3.0648), (3, 2, 4, 3, 10.2303)],
)
def test_coulomb_integrals_reproduce_hydrogen(n1, l1, n2, l2, exact):
    assert abs(coulomb_dipole_radial_integral(n1, l1, n2, l2)) == pytest.approx(exact, rel=1.0e-3)


def test_nearly_integer_effective_quantum_numbers_stay_hydrogenic():
    # He I 3d 1D - 4f 1F (n* = 2.998, 3.998): the divergent inner part of
    # the non-integer Coulomb functions must not dominate the integral.
    assert abs(coulomb_dipole_radial_integral(2.9982, 2, 3.9981, 3)) == pytest.approx(10.23, rel=0.01)


def test_wigner_6j_value_and_sum_rule():
    assert wigner_6j(1, 1, 1, 1, 1, 1) == pytest.approx(1.0 / 6.0)
    # Summed over L', (2L'+1){l L Lc; L' l' 1}^2 = 1/(2l+1).
    for l, big_l, core_l, l_prime in ((1, 0, 1, 2), (1, 2, 1, 2), (2, 3, 1, 1), (1, 1, 1, 0)):
        total = sum(
            (2 * other + 1) * wigner_6j(l, big_l, core_l, other, l_prime, 1) ** 2
            for other in range(abs(core_l - l_prime), core_l + l_prime + 1)
        )
        assert total == pytest.approx(1.0 / (2 * l + 1))


def _helium_singlets():
    # NIST He I singlet term energies (cm^-1); ionization limit 198310.67.
    limit = 198_310.67
    energies = {
        "2s": 166_277.44, "2p": 171_134.90, "3s": 184_864.83, "3p": 186_209.36, "3d": 186_104.71,
        "4s": 190_940.23, "4p": 191_492.71, "4d": 191_444.48, "4f": 191_446.45,
        "5s": 193_346.99, "5p": 193_626.61, "5d": 193_601.62, "5f": 193_602.58,
        "6s": 194_584.68, "6p": 194_745.00, "6d": 194_729.97, "6f": 194_730.5,
        "7s": 195_324.0, "7p": 195_425.0, "7d": 195_415.3, "7f": 195_415.6,
    }
    terms = {}
    for name, energy in energies.items():
        n, l = int(name[0]), "spdf".index(name[1])
        terms[name] = _Term(
            name, "1s.(2S)", 0, n, l, 1, l, bool(l % 2), energy,
            float(np.sqrt(109_722.27 / (limit - energy))),
        )
    return terms


@pytest.mark.parametrize(
    "upper, wavelength, temperature, half_width, shift",
    [
        # Dimitrijevic & Konjevic (1986) Table 1, N_e = 1e16 cm^-3 (HWHM, shift in A).
        ("5s", 4437.55, 10_000.0, 1.77, 1.49),
        ("5s", 4437.55, 20_000.0, 1.85, 1.12),
        ("3d", 6678.15, 10_000.0, 0.428, 0.181),
        ("3d", 6678.15, 20_000.0, 0.389, 0.135),
    ],
)
def test_simplified_semiclassical_reproduces_published_helium_values(
    upper, wavelength, temperature, half_width, shift
):
    terms = _helium_singlets()
    width, displacement = simplified_semiclassical_stark(terms, terms[upper], terms["2p"], temperature)
    to_angstrom = (wavelength * 1.0e-8) ** 2 * 1.0e16 / (2.0 * np.pi * LIGHT_SPEED) * 1.0e8
    assert 0.5 * width * to_angstrom == pytest.approx(half_width, rel=0.12)
    assert displacement * to_angstrom == pytest.approx(shift, rel=0.4)
    assert displacement > 0.0
