"""The sdB metal driver photoionizes C, N, O, Si and S with Opacity Project data.

The hydrogenic (Kramers) excited-level cross sections underestimate the low
excited continua that ionize the lower stages (Si II 3d, 4s, 4p and 3s3p2 by
factors of 3-15 near threshold, where the Opacity Project data have
autoionizing resonances).  Si II was then underionized: Si II 4128-31 too
strong against Si III 4552-75.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research'))
import sdb_trace_metals as driver  # noqa: E402

from wd_spectra.constants import LIGHT_SPEED, PLANCK
from wd_spectra.light_metal_nlte import (
    hydrogenic_kramers_threshold_cross_section, read_tlusty_photoionization_threshold_data)
from wd_spectra.metals import EV_TO_ERG, read_pg1159_atomic_database
from wd_spectra.models.common import ModelData

# Unmodified TLUSTY model atoms (THIRD_PARTY_NOTICES.md).
ATOM_CHECKSUMS = {
    'c2.dat': '74513a8eb5e652c23006db39d32ddaab2b2d3a0561aab6c198cdf367b27e736d',
    'c3.dat': '6e8e9517c3d2f5d5e25a2200d53a302f43451d8cc2a6f947113d8a6863f1395e',
    'c4_35+2lev.dat': '024b8335b3d416f8d079afa15d6935597e9b786ae4738f2d0a2991c9a1554979',
    'n2_32+10lev.dat': '7fd12008bedaf88a3aa793199b3825bc6614f9bbee8ae210ddc887cf89112553',
    'n3.dat': '4a7b8924f1c7e348953ae55928aab9919ab83f3bf7a18cda855d5d9391f1fafb',
    'n4_34+14lev.dat': '4805de9fa439e3d8923402dc9498796ebefb787c0b621130ef554892c8c202ef',
    'o2_36+12lev.dat': '509ea2d1b0bb8c5db99c25b04baf8b97d17dff753a24ab025977cdd4f96e58e2',
    'o3_28+13lev.dat': '2cbd1ffb57e7afd9ed14d3a7d1206dac6e48f456486c2bd12ab43fa4c9564a6e',
    'o4.dat': 'c2b6304ffafc1ddf0a736b47d27928d20121be3a3f5021d406998d69e6cda616',
    'si2_36+4lev.dat': 'a194209c8c876ed7b3f80cbf9d24b1299516977a071aa53dc58f8ec9abdbaf8e',
    'si3.dat': '7b842c26472135465df3c63d3ce28e56f586eefd41fc206093fc166a3307a25e',
    'si4.dat': '0fa7ff0604c0a75f14c324a90e2507403466c9103104470eaccaa204ceca34ea',
    's2_23+10lev.dat': '79d19fca41737511f56a9545aab6dff2ee3bef102525c8b3eede4eadd2786d9d',
    's3_29+12lev.dat': '67138a1d68aada071371738849970f4f83ae75d53ad2a586ec8083a1ba7ff465',
    's4_33+5lev.dat': '7e667a81161d94a9c15b726e7b1113bdecfe2aeead1eddbcf6d98a16d75e24c2',
}
STAGES = [(element, charge) for element, tables in driver.TLUSTY_PHOTOIONIZATION.items()
          for charge in sorted(tables)]


@pytest.fixture(scope='module')
def data():
    return ModelData.default()


def test_tables_are_the_pinned_tlusty_atoms(data):
    names = {name for tables in driver.TLUSTY_PHOTOIONIZATION.values() for name in tables.values()}
    assert names == set(ATOM_CHECKSUMS)
    for name, checksum in ATOM_CHECKSUMS.items():
        assert hashlib.sha256((data.tlusty_atoms / name).read_bytes()).hexdigest() == checksum
    # Every photoionized (non-top) stage of every NLTE element has a table.
    for element, counts in driver.ELEMENT_LEVELS.items():
        assert set(driver.TLUSTY_PHOTOIONIZATION[element]) == set(sorted(counts)[:-1])


def stout_term(label):
    spin, orbital, odd = re.findall(r'\((\d)([SPDFGHIKLMN])(o?)(?:<[^>]*>)?\)', label)[-1]
    return spin + orbital + ('o' if odd else 'e')


def tlusty_term(label):
    found = re.search(r'(\d)([SPDFGHIKLMN])([eo])\s+\d+', label)
    return None if found is None else ''.join(found.groups())


def matched_levels(data, element, charge, count, table):
    ion = read_pg1159_atomic_database(data.stout, elements=(element,)).ions[element, charge]
    matched, missing = {}, []
    for level in sorted(ion.levels, key=lambda x: x.energy_wavenumber)[:count]:
        threshold = ion.ionization_energy_ev - level.energy_wavenumber * PLANCK * LIGHT_SPEED / EV_TO_ERG
        frequency = threshold * EV_TO_ERG / PLANCK
        index = table._matching_index(frequency, 0.02, level_label=level.label)
        if index is None:
            missing.append(level.label)
        else:
            matched[level.label] = (threshold, frequency, table.level_label[index])
    return matched, missing


@pytest.mark.parametrize('element,charge', STAGES)
def test_driver_terms_match_opacity_project_terms(data, element, charge):
    """Matched levels agree in spin, L and parity; the lowest levels never fall back to Kramers."""
    table = read_tlusty_photoionization_threshold_data(
        data.tlusty_atoms / driver.TLUSTY_PHOTOIONIZATION[element][charge])
    count = driver.ELEMENT_LEVELS[element][charge]
    matched, missing = matched_levels(data, element, charge, count, table)
    for label, (_, _, tlusty_label) in matched.items():
        term = tlusty_term(tlusty_label)
        assert term is None or term == stout_term(label), (label, tlusty_label)  # None: superlevel
    ion = read_pg1159_atomic_database(data.stout, elements=(element,)).ions[element, charge]
    lowest = [level.label for level in sorted(ion.levels, key=lambda x: x.energy_wavenumber)[:min(count, 10)]]
    assert not set(lowest) & set(missing)
    # Large atoms (C II 200, C III 80 levels) reach Rydberg terms beyond the
    # compact TLUSTY atoms; those keep the hydrogenic fallback.
    low = [level.label for level in sorted(ion.levels, key=lambda x: x.energy_wavenumber)[:min(count, 30)]]
    assert len(set(low) & set(missing)) <= 0.25 * len(low)


def test_opacity_project_si_ii_3d_continuum_exceeds_kramers(data):
    """The lower level of Si II 4128-31 photoionizes far more strongly than hydrogenic."""
    table = read_tlusty_photoionization_threshold_data(data.tlusty_atoms / 'si2_36+4lev.dat')
    matched, _ = matched_levels(data, 'Si', 1, 11, table)
    threshold, frequency, label = matched['3s2.3d.(2D<5/2>)']
    assert label.startswith('SiII 2De')
    grid = frequency * np.geomspace(1.0, 2.0, 400)
    opacity_project = table.cross_section(grid, frequency, relative_tolerance=0.02,
                                          level_label='3s2.3d.(2D<5/2>)')
    kramers = hydrogenic_kramers_threshold_cross_section(threshold, 1) * (frequency / grid) ** 3
    # Rate-like weighting (sigma / nu) over the first factor of two in frequency.
    integrate = getattr(np, 'trapezoid', None) or np.trapz
    ratio = integrate(opacity_project / grid, grid) / integrate(kramers / grid, grid)
    assert ratio > 5.0
