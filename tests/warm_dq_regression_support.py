"""Fixed-structure WarmDQ synthesis shared by the frozen-spectrum test and its offline capture.

Each control is a converged, certified cold-start structure of a documented
star (docs/models/WarmDQ.md) at its published parameters. The test rebuilds
the bulk LTE atmosphere on the frozen column-mass, optical-depth and
temperature arrays and synthesizes the spectrum without iterating the
structure. The magnetic control reuses a nonmagnetic structure, as the
public model does.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from wd_spectra import WarmDQConfig
from wd_spectra.d6 import _bulk_atmosphere
from wd_spectra.models.common import ModelData
from wd_spectra.models.warm_dq import (
    compute_warm_dq,
    warm_dq_abundances,
    warm_dq_atomic_inputs,
)

STEP = 0.2  # A
# Vacuum windows: C II 4267, the C I 5f band, C I 4772/4932 with Hbeta, the
# C I 3s-4p and 4f lines near 5052, C I 5380, Halpha with C II 6580, C I 7115.
_CARBON_WINDOWS = ((4230.0, 4300.0), (4400.0, 4560.0), (4740.0, 4960.0),
                   (5000.0, 5080.0), (5360.0, 5400.0), (6520.0, 6620.0),
                   (7080.0, 7150.0))
# case: WarmDQConfig keywords, the nonmagnetic source structure, wavelength windows.
CASES = {
    "warm-dq-j1215": dict(
        config=dict(effective_temperature=13574.0, logg=8.86, log_carbon_to_hydrogen=1.11),
        structure="warm-dq-j1215",
        windows=_CARBON_WINDOWS,
    ),
    "warm-dq-j1215-uniform-10mg": dict(
        config=dict(effective_temperature=13574.0, logg=8.86, log_carbon_to_hydrogen=1.11,
                    magnetic_field_megagauss=10.0, field_geometry="uniform"),
        structure="warm-dq-j1215",
        windows=((4900.0, 5080.0), (6520.0, 6620.0)),
    ),
    "warm-dq-j0236-che": dict(
        config=dict(effective_temperature=14611.0, logg=8.777, log_carbon_to_hydrogen=2.677,
                    log_carbon_to_helium=-1.823),
        structure="warm-dq-j0236-che",
        windows=((4230.0, 4300.0), (4440.0, 4560.0), (4900.0, 4960.0),
                 (5860.0, 5890.0), (6520.0, 6620.0), (6660.0, 6700.0)),
    ),
}


def wavelength_grid(windows) -> np.ndarray:
    return np.concatenate([np.arange(lo, hi + 0.5 * STEP, STEP) for lo, hi in windows])


def structure_config(case: str) -> WarmDQConfig:
    """The nonmagnetic configuration whose cold start supplies the case's structure."""

    return WarmDQConfig(**CASES[CASES[case]["structure"]]["config"])


@lru_cache(maxsize=2)
def _line_database(elements: tuple[str, ...]):
    return warm_dq_atomic_inputs(ModelData.default(), elements)[0]


def fixed_spectrum(config: WarmDQConfig, column_mass, optical_depth, temperature, wavelength):
    """Synthesize ``config`` on a frozen structure; the structure is never iterated."""

    abundances = warm_dq_abundances(config)
    elements = tuple(element for element in abundances if element not in ("H", "He"))
    atmosphere, _ = _bulk_atmosphere(
        config.effective_temperature, config.logg, optical_depth, column_mass,
        temperature, _line_database(elements), abundances, reference_element="C",
        metadata={"model": "frozen-warm-dq-regression-input",
                  "radiative_equilibrium_converged": False},
    )
    return compute_warm_dq(config, wavelength, initial_atmosphere=atmosphere,
                           relax_atmosphere=False).spectrum
