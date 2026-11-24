"""Warm and hot DQ models: carbon atmospheres with hydrogen and helium."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Literal, Mapping

import numpy as np
from numpy.typing import ArrayLike

from ..atmosphere import Atmosphere
from ..d6 import radiative_equilibrium_d6_atmosphere, synthesize_d6_spectrum
from ..metals import (
    NIST_ASD_STRONG_ION_FILES,
    _canonical_element,
    atomic_database_with_kurucz_electron_stark,
    read_nist_asd_strong_atomic_database,
)
from ..warm_dq import (
    atomic_database_with_carbon_i_f_series,
    atomic_database_with_li2021_c_i_oscillator_strengths,
    atomic_database_with_carbon_i_nearest_level_stark,
    atomic_database_with_carbon_stark,
    atomic_database_with_multiplet_electron_stark,
    synthesize_magnetic_bulk_spectrum,
)
from .common import (
    ModelData,
    ModelResult,
    Quality,
    atmosphere_matches_model_request,
    atmosphere_with_model_request_fingerprint,
    fixed_synthesis_atmosphere,
    model_request_fingerprint,
    numerical_resolution,
    validate_wavelength,
    warn_if_atmosphere_not_converged,
)
from .d6 import _atomic_inputs

# Fields above about 50 kG inhibit photospheric convection (Tremblay et al.
# 2015; Gentile Fusillo et al. 2018); the DAH class uses the same threshold.
CONVECTION_SUPPRESSION_FIELD_MEGAGAUSS = 0.05
# Mixing length of the convective solve that seeds a radiative cold start.
RADIATIVE_CONTINUATION_MIXING_LENGTH_ALPHA = 1.25


@dataclass(frozen=True)
class WarmDQConfig:
    """Carbon-rich LTE atmosphere of a warm DQ, DAQ or hot DQ white dwarf.

    ``log_carbon_to_hydrogen`` and ``log_carbon_to_helium`` are log10
    N(C)/N(H) and log10 N(C)/N(He); ``None`` leaves that element out.
    ``trace_abundances`` are log10 N(X)/N(C) of additional elements (for
    example ``{"O": -1.0}``).  Carbon, hydrogen, helium and every listed
    element contribute to the pressure, charge, mass and opacity.

    ``magnetic_field_megagauss > 0`` splits the lines in the emergent
    spectrum: LS Paschen--Back components for the metals and the DAH Balmer
    components for hydrogen, summed over the visible disk of a uniform field
    or an offset dipole (the geometry options of :class:`DAHConfig`).  The
    structure, the level populations and the continuum stay nonmagnetic.

    ``mixing_length_alpha=None`` gives a radiative atmosphere.  With
    ``magnetic_convection_suppression`` (the default) the atmosphere is also
    radiative when the field is 0.05 MG or stronger.

    The default parameters are those of WD J1215+4700 (Kilic et al. 2026),
    the cold-start canary of this class.
    """

    effective_temperature: float = 13_574.0
    logg: float = 8.86
    log_carbon_to_hydrogen: float | None = 1.11
    log_carbon_to_helium: float | None = None
    trace_abundances: Mapping[str, float] = field(default_factory=dict)
    quality: Quality = "standard"
    mixing_length_alpha: float | None = 1.25
    structure_maximum_metal_lines: int | None = None
    formal_maximum_metal_lines: int | None = None
    microturbulent_velocity_kms: float = 0.0
    magnetic_field_megagauss: float = 0.0
    field_geometry: Literal["uniform", "dipole"] = "dipole"
    field_angle_deg: float | None = None
    field_strength_definition: Literal["dipole-polar", "visible-mean"] = "dipole-polar"
    dipole_inclination_deg: float = 60.0
    dipole_offset_radius: tuple[float, float, float] = (0.0, 0.0, 0.0)
    disk_field_bins: int | None = 21
    include_diamagnetic_shift: bool = True
    magnetic_convection_suppression: bool = True


_LINE_PROFILE_EDGE_OPTICAL_DEPTH = 1.0e-3
# Ions whose NIST transition probabilities replace the Stout ones at every
# accuracy grade (see warm_dq_atomic_inputs).
_NIST_ANY_GRADE_IONS = frozenset({("C", 1)})
_STRUCTURE_LINES = {"quick": 2_000, "standard": 25_000, "production": 25_000}
_FORMAL_LINES = {"quick": 5_000, "standard": 25_000, "production": 25_000}
_N_DEPTH = {"quick": 24, "standard": 48, "production": 72}
_PSEUDOCONTINUUM_ELEMENTS = ("C", "O", "Mg")


def default_warm_dq_wavelength_grid(quality: Quality = "standard") -> np.ndarray:
    """Optical grid covering the SDSS and DESI spectral ranges."""

    step = {"quick": 0.5, "standard": 0.2, "production": 0.1}[quality]
    return np.arange(3500.0, 9900.0 + 0.5 * step, step)


def warm_dq_abundances(config: WarmDQConfig) -> dict[str, float]:
    """Return the bulk abundance mapping, log10 N(X)/N(C), with optional H, He."""

    abundances: dict[str, float] = {"C": 0.0}
    for element, value in config.trace_abundances.items():
        symbol = str(element).strip().capitalize()
        if symbol in ("H", "He", "C"):
            raise ValueError(
                "trace_abundances lists elements other than C, H and He; "
                "set hydrogen and helium with log_carbon_to_hydrogen and "
                "log_carbon_to_helium"
            )
        symbol = _canonical_element(symbol)
        if not np.isfinite(float(value)):
            raise ValueError("abundances must be finite")
        abundances[symbol] = float(value)
    if config.log_carbon_to_hydrogen is not None:
        if not np.isfinite(float(config.log_carbon_to_hydrogen)):
            raise ValueError("log_carbon_to_hydrogen must be finite or None")
        abundances["H"] = -float(config.log_carbon_to_hydrogen)
    if config.log_carbon_to_helium is not None:
        if not np.isfinite(float(config.log_carbon_to_helium)):
            raise ValueError("log_carbon_to_helium must be finite or None")
        abundances["He"] = -float(config.log_carbon_to_helium)
    return abundances


def validate_config(config: WarmDQConfig) -> None:
    if not isinstance(config, WarmDQConfig):
        raise TypeError("expected WarmDQConfig")
    if not np.isfinite(config.effective_temperature) or config.effective_temperature <= 0.0:
        raise ValueError("effective_temperature must be finite and positive")
    if not np.isfinite(config.logg):
        raise ValueError("logg must be finite")
    if (
        not np.isfinite(config.microturbulent_velocity_kms)
        or config.microturbulent_velocity_kms < 0.0
    ):
        raise ValueError("microturbulent_velocity_kms must be finite and non-negative")
    if (
        not np.isfinite(config.magnetic_field_megagauss)
        or config.magnetic_field_megagauss < 0.0
    ):
        raise ValueError("magnetic_field_megagauss must be finite and non-negative")
    if config.field_geometry not in ("uniform", "dipole"):
        raise ValueError("field_geometry must be 'uniform' or 'dipole'")
    if config.mixing_length_alpha is not None and (
        not np.isfinite(config.mixing_length_alpha) or config.mixing_length_alpha <= 0.0
    ):
        raise ValueError("mixing_length_alpha must be positive or None")
    warm_dq_abundances(config)


def warm_dq_surface_cells(config: WarmDQConfig):
    """Return the visible-disk quadrature of a magnetic configuration."""

    from ..magnetic import dipole_surface_cells, uniform_field_surface_cells

    if config.magnetic_field_megagauss <= 0.0:
        raise ValueError("surface cells require magnetic_field_megagauss > 0")
    if config.field_geometry == "uniform":
        if tuple(config.dipole_offset_radius) != (0.0, 0.0, 0.0):
            raise ValueError("dipole_offset_radius applies to field_geometry='dipole'")
        return uniform_field_surface_cells(
            config.magnetic_field_megagauss, field_angle_deg=config.field_angle_deg
        )
    if config.field_angle_deg is not None:
        raise ValueError("field_angle_deg applies to field_geometry='uniform'")
    return dipole_surface_cells(
        config.magnetic_field_megagauss,
        field_strength_definition=config.field_strength_definition,
        inclination_deg=config.dipole_inclination_deg,
        offset_vector_radius=tuple(config.dipole_offset_radius),  # type: ignore[arg-type]
        n_field_bins=config.disk_field_bins,
    )


def effective_mixing_length_alpha(config: WarmDQConfig) -> float | None:
    """Return the ML2 parameter used by the structure, ``None`` if radiative."""

    if config.mixing_length_alpha is None:
        return None
    if (
        config.magnetic_convection_suppression
        and config.magnetic_field_megagauss > 0.0
    ):
        cells = warm_dq_surface_cells(config)
        mean_field = float(np.sum(
            cells.projected_weight * cells.field_strength_megagauss
        ))
        if mean_field >= CONVECTION_SUPPRESSION_FIELD_MEGAGAUSS:
            return None
    return float(config.mixing_length_alpha)


def warm_dq_atomic_inputs(data: ModelData, elements: tuple[str, ...]):
    """Return the line database and photoionization data of the metals.

    The D6 inputs (Stout lines of charges 0-2, level-resolved TOPbase and
    Verner bound-free) are modified in two ways.

    * Transition probabilities with a NIST ASD value (1150-9000 A) take
      that value: grade D or better for C I and the other ions, every grade
      for C II.  The Stout C II data of the 2s2p(3Po)nl system are 4-20
      times larger than the NIST values (for example f = 7.1 for
      3d 4Fo9/2 - 4f 4G11/2 at 3876 A, against 0.71; f = 2.45 for
      3d 4Do3/2 - 4f 4F5/2 at 4075 A, against a grade-E 0.36), which exceeds
      the one-electron sum rule and produces strong spurious lines near
      3877 and 4076 A.  A grade-E value (uncertainty above 50 percent) is
      far closer than such a Stout value.  For C I the Stout data are not
      known to be wrong and grade-E values are not used: for C I 4269 A the
      grade-E value is 4.7 times the Stout one and literature values spread
      by a factor of four (Koester & Kepler 2019).
    * C I lines between states of the Li et al. (2021) MCDHF calculation with
      a length-velocity difference up to 20 percent then take its oscillator
      strengths (:func:`wd_spectra.warm_dq.
      atomic_database_with_li2021_c_i_oscillator_strengths`).
    * Lines gain the Kurucz electron Stark constants; multiplet members
      without a tabulated constant receive their multiplet's value.  The
      C I 3s-4p multiplets then take their measured widths and red shifts,
      and eight optical/UV C II multiplets their semiclassical widths and
      shifts with temperature dependence (see
      :func:`wd_spectra.warm_dq.carbon_stark_rate_coefficients`).  The
      Kurucz constants of these lines are 0.1-0.9 times the measured or
      semiclassical widths, and the measured shifts (about half the width
      for C I 4932 and 5052 A) bring the C I and Balmer line velocities of
      observed warm DQs into agreement.
    """

    database, photoionization, topbase = _atomic_inputs(data, elements)
    # Maximum NIST uncertainty (percent) accepted per ion; C II takes every grade.
    for any_grade in (False, True):
        nist_paths = [
            data.nist_asd_strong / filename
            for filename, (_, _, element, charge) in NIST_ASD_STRONG_ION_FILES.items()
            if element in elements and charge <= 2
            and ((element, charge) in _NIST_ANY_GRADE_IONS) == any_grade
        ]
        if not nist_paths:
            continue
        data.require(*nist_paths)
        database = read_nist_asd_strong_atomic_database(
            nist_paths, database,
            maximum_accuracy_percent=np.inf if any_grade else 50.0,
            minimum_oscillator_strength=1.0e-6,
            oscillator_strength_from_einstein_a=True,
        )
    database = atomic_database_with_kurucz_electron_stark(database)
    database = atomic_database_with_multiplet_electron_stark(database)
    if "C" in elements:
        database = atomic_database_with_li2021_c_i_oscillator_strengths(database)
        database = atomic_database_with_carbon_i_f_series(database)
        database = atomic_database_with_carbon_i_nearest_level_stark(database)
        database = atomic_database_with_carbon_stark(database)
    return database, photoionization, topbase


def compute_warm_dq(
    config: WarmDQConfig = WarmDQConfig(),
    wavelength: ArrayLike | None = None,
    *,
    data: ModelData | None = None,
    initial_atmosphere: Atmosphere | None = None,
    relax_atmosphere: bool = True,
    iteration_callback: Callable[
        [int, Atmosphere, Mapping[str, object]], None
    ] | None = None,
) -> ModelResult:
    """Calculate one carbon-rich warm/hot DQ atmosphere and its spectrum.

    The structure uses the bulk-mixture closure of the D6 class, with atomic
    hydrogen and helium in the same pressure and charge balance, and the shared adaptive
    trust-region Newton/ML2 solver.  ``relax_atmosphere=False`` performs
    formal synthesis on a supplied structure.
    """

    validate_config(config)
    abundances = warm_dq_abundances(config)
    data = ModelData.default() if data is None else data
    resolution = numerical_resolution(config.quality)
    request_fingerprint = model_request_fingerprint("WarmDQ", config, data)
    checkpoint_matches_request = atmosphere_matches_model_request(
        initial_atmosphere, request_fingerprint
    )
    wave = (
        default_warm_dq_wavelength_grid(config.quality)
        if wavelength is None
        else validate_wavelength(wavelength)
    )
    structure_lines = (
        _STRUCTURE_LINES[config.quality]
        if config.structure_maximum_metal_lines is None
        else int(config.structure_maximum_metal_lines)
    )
    formal_lines = (
        _FORMAL_LINES[config.quality]
        if config.formal_maximum_metal_lines is None
        else int(config.formal_maximum_metal_lines)
    )
    if structure_lines < 0 or formal_lines < 0:
        raise ValueError("metal line limits must be non-negative")
    elements = tuple(element for element in abundances if element not in ("H", "He"))
    helium_tables = None
    if config.log_carbon_to_helium is not None:
        from ..helium_ii_stark import read_helium_ii_stark_table
        from ..helium_stark import read_helium_stark_table

        data.require(data.helium_i_stark, data.helium_ii_stark)
        helium_tables = (
            read_helium_stark_table(data.helium_i_stark),
            read_helium_ii_stark_table(data.helium_ii_stark),
        )
    database, photoionization, topbase = warm_dq_atomic_inputs(data, elements)
    pseudocontinuum = tuple(
        element for element in _PSEUDOCONTINUUM_ELEMENTS if element in elements
    )
    if not relax_atmosphere and initial_atmosphere is None:
        raise ValueError("relax_atmosphere=False requires initial_atmosphere")
    mixing_length_alpha = effective_mixing_length_alpha(config)
    magnetic = config.magnetic_field_megagauss > 0.0
    atmosphere = initial_atmosphere
    if relax_atmosphere:
        structure_options = dict(
            reference_element="C",
            n_depth=_N_DEPTH[config.quality],
            max_iterations=resolution.maximum_iterations,
            maximum_metal_lines=structure_lines,
            minimum_metal_oscillator_strength=1.0e-4,
            metal_series_pseudocontinuum_elements=pseudocontinuum,
            topbase_photoionization_database=topbase,
            include_linear_stark_quasistatic=True,
            microturbulent_velocity_kms=config.microturbulent_velocity_kms,
            profile_edge_optical_depth=_LINE_PROFILE_EDGE_OPTICAL_DEPTH,
            helium_line_tables=helium_tables,
            n_angle=min(resolution.n_angle, 3),
            iteration_callback=iteration_callback,
        )
        seed = initial_atmosphere
        if seed is None and mixing_length_alpha is None:
            # A radiative atmosphere does not converge from the gray seed:
            # its line-blanketed surface is far cooler than gray, and the
            # local-energy completion stalls.  The convective solution is a
            # radiative-equilibrium structure above Rosseland depth ~5, so
            # it is the continuation start for the radiative solve.
            seed = radiative_equilibrium_d6_atmosphere(
                config.effective_temperature,
                config.logg,
                database,
                photoionization,
                abundances,
                mixing_length_alpha=RADIATIVE_CONTINUATION_MIXING_LENGTH_ALPHA,
                **structure_options,
            )
        atmosphere = radiative_equilibrium_d6_atmosphere(
            config.effective_temperature,
            config.logg,
            database,
            photoionization,
            abundances,
            mixing_length_alpha=mixing_length_alpha,
            initial_temperature=None if seed is None else seed.temperature,
            initial_column_mass=None if seed is None else seed.column_mass,
            initial_rosseland_optical_depth=(
                None if seed is None else seed.rosseland_optical_depth
            ),
            resume_supplied_structure_in_formal_flux_phase=checkpoint_matches_request,
            **structure_options,
        )
        if seed is not None and initial_atmosphere is None:
            atmosphere = replace(
                atmosphere,
                metadata={
                    **atmosphere.metadata,
                    "radiative_cold_start": (
                        "continued from the converged ML2 "
                        f"(alpha={RADIATIVE_CONTINUATION_MIXING_LENGTH_ALPHA:g}) structure"
                    ),
                },
            )
    assert atmosphere is not None
    if relax_atmosphere:
        atmosphere = atmosphere_with_model_request_fingerprint(
            atmosphere, request_fingerprint
        )
    else:
        atmosphere = fixed_synthesis_atmosphere(atmosphere, request_fingerprint)
    convergence_status = warn_if_atmosphere_not_converged(atmosphere, "WarmDQ")
    if magnetic:
        from ..magnetic import WEAK_FIELD_MAXIMUM_MEGAGAUSS

        cells = warm_dq_surface_cells(config)
        transitions = None
        if (
            config.log_carbon_to_hydrogen is not None
            and cells.maximum_field_megagauss > WEAK_FIELD_MAXIMUM_MEGAGAUSS
        ):
            from ..magnetic_atomic import read_h2db_transition_database

            data.require(data.h2db_balmer_subset)
            transitions = read_h2db_transition_database(data.h2db_balmer_subset)
        spectrum = synthesize_magnetic_bulk_spectrum(
            atmosphere,
            wave,
            database,
            photoionization,
            abundances,
            cells,
            reference_element="C",
            topbase_photoionization_database=topbase,
            h2db_transitions=transitions,
            minimum_metal_oscillator_strength=1.0e-6,
            maximum_metal_lines=formal_lines,
            metal_series_pseudocontinuum_elements=pseudocontinuum,
            include_linear_stark_quasistatic=True,
            include_diamagnetic=config.include_diamagnetic_shift,
            microturbulent_velocity_kms=config.microturbulent_velocity_kms,
            profile_edge_optical_depth=_LINE_PROFILE_EDGE_OPTICAL_DEPTH,
            n_angle=resolution.n_angle,
            helium_line_tables=helium_tables,
        )
    else:
        spectrum = synthesize_d6_spectrum(
            atmosphere,
            wave,
            database,
            photoionization,
            abundances,
            reference_element="C",
            minimum_metal_oscillator_strength=1.0e-6,
            maximum_metal_lines=formal_lines,
            topbase_photoionization_database=topbase,
            metal_series_pseudocontinuum_elements=pseudocontinuum,
            include_linear_stark_quasistatic=True,
            microturbulent_velocity_kms=config.microturbulent_velocity_kms,
            profile_edge_optical_depth=_LINE_PROFILE_EDGE_OPTICAL_DEPTH,
            n_angle=resolution.n_angle,
            helium_line_tables=helium_tables,
        )
    return ModelResult(
        "WarmDQ",
        atmosphere,
        spectrum,
        config,
        {
            "preset": "WarmDQ-shared-solver-v2",
            "log_number_abundance_relative_to_carbon": abundances,
            "atomic_lines": database.source,
            "bound_free": (
                (topbase.source + "; " if topbase is not None else "")
                + "Verner ground-state fits for the remaining stages"
            ),
            "hydrogen": (
                "absent"
                if config.log_carbon_to_hydrogen is None
                else "bulk-mixture H I/H II with Hummer-Mihalas Q-MHD levels; "
                "Tremblay-Bergeron Stark lines"
            ),
            "helium": (
                "absent"
                if config.log_carbon_to_helium is None
                else "bulk-mixture He I/He II/He III with Hummer-Mihalas Q-MHD "
                "levels; DB continuum, Beauchamp He I and He II Stark lines"
            ),
            "structure_maximum_metal_lines": structure_lines,
            "formal_maximum_metal_lines": formal_lines,
            "microturbulent_velocity_kms": float(config.microturbulent_velocity_kms),
            "convection": (
                "disabled"
                if mixing_length_alpha is None
                else f"ML2/alpha={mixing_length_alpha:g}"
            ),
            "magnetic_field": (
                "none" if not magnetic
                else f"{config.field_geometry}, {config.magnetic_field_megagauss:g} MG; "
                "Zeeman/Paschen-Back line splitting on the nonmagnetic structure"
            ),
            "atmosphere_solver": "adaptive-newton (shared)",
            "atmosphere_mode": (
                "relaxed" if relax_atmosphere else "checkpoint formal synthesis"
            ),
            "atmosphere_convergence_status": convergence_status,
            "checkpoint_matches_model_request": checkpoint_matches_request,
            "model_request_fingerprint": request_fingerprint,
        },
    )
