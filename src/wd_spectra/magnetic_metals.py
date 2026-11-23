"""Zeeman and Paschen--Back splitting of LS-coupled metal lines.

White-dwarf fields of 0.1--10 MG shift magnetic sublevels by
``mu_B B = 4.7--470 cm^-1``, comparable to or larger than the fine-structure
intervals of light elements (C I ``3s 3Po``: 20 and 40 cm^-1; C II ``3p 2Po``:
11 cm^-1).  The anomalous-Zeeman pattern of each fine-structure line is then
wrong: ``J`` is not conserved and the components of a multiplet regroup
toward a normal triplet (the Paschen--Back effect).

Each LS multiplet is therefore treated as a whole.  For the lower and the
upper term the Hamiltonian

``H = sum_J E_J |J M><J M| + mu_B B (L_z + g_s S_z) + H_dia``

is diagonalized for every ``M`` in the uncoupled basis ``|L M_L S M_S>``.
``E_J`` are the tabulated level energies, so the zero-field limit returns the
tabulated fine structure, and the strong-field limit the Paschen--Back
pattern.  The electric-dipole operator acts on the orbital part only,

``<L' M_L' S M_S'| r_q |L M_L S M_S> ~ delta(M_S, M_S') <L M_L 1 q|L' M_L'>``,

which gives the relative strength of every component.  The summed ``g f`` of
the tabulated lines of the multiplet is distributed over the components, so
the total line strength is conserved at every field.

``H_dia`` is the first-order diamagnetic (quadratic Zeeman) energy of the
outer electron, ``(e^2 B^2 / 8 m_e c^2) <r^2> <sin^2 theta>``, with the
hydrogenic ``<r^2>`` of the level's effective quantum number and the angular
factor of the orbital projection.  It is exact for one electron outside closed
shells in the limit where the quadratic term is small against the separation
of the term from its neighbours of the same parity.

Levels without a parseable LS label, and intercombination lines, fall back to
the linear pattern of their own ``J`` levels (LS Lande factors when the term
is known, otherwise a normal triplet).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from math import factorial, sqrt
import re
from types import MappingProxyType
from typing import Iterable, Mapping

import numpy as np
from numpy.typing import NDArray

from .constants import ELECTRON_MASS, ELEMENTARY_CHARGE_ESU, LIGHT_SPEED, PI, PLANCK
from .metals import (
    AtomicDatabase,
    AtomicIon,
    AtomicLevel,
    AtomicTransition,
    EV_TO_WAVENUMBER,
    _level_term_label,
    _parent_limit_excitation_ev,
)

FloatArray = NDArray[np.float64]

# mu_B / (h c) in cm^-1 per gauss.
BOHR_MAGNETON_WAVENUMBER_PER_GAUSS = ELEMENTARY_CHARGE_ESU / (
    4.0 * PI * ELECTRON_MASS * LIGHT_SPEED**2
)
_BOHR_RADIUS_CM = 5.291_772_109_03e-9
# e^2 a_0^2 / (8 m_e c^2 h c) in cm^-1 per gauss^2: the diamagnetic energy of
# <r^2 sin^2 theta> = a_0^2.
DIAMAGNETIC_WAVENUMBER_PER_GAUSS2 = (
    ELEMENTARY_CHARGE_ESU**2
    * _BOHR_RADIUS_CM**2
    / (8.0 * ELECTRON_MASS * LIGHT_SPEED**2)
    / (PLANCK * LIGHT_SPEED)
)
_RYDBERG_WAVENUMBER = 109_737.315_68
ELECTRON_SPIN_G_FACTOR = 2.002_319_304_36
_ORBITAL_LETTERS = "spdfghiklmnoqrtuv"
_TERM_LETTERS = "SPDFGHIKLMNOQRTUV"
_FINAL_TERM = re.compile(r"\((\d+)([A-Z])(o?)<(\d+)(?:/(\d+))?>\)$")
_PARENT_TERM = re.compile(r"^\((\d+)([A-Z])o?\)$")
_ORBITAL = re.compile(r"^(\d+)([a-z])(\d*)$")


def _twice(value: float) -> int:
    doubled = int(round(2.0 * value))
    if abs(doubled - 2.0 * value) > 1.0e-9:
        raise ValueError("angular momenta must be integer or half-integer")
    return doubled


@lru_cache(maxsize=None)
def _clebsch_gordan_twice(j1: int, m1: int, j2: int, m2: int, j: int, m: int) -> float:
    """Clebsch--Gordan coefficient with every argument doubled."""

    if m1 + m2 != m or abs(m1) > j1 or abs(m2) > j2 or abs(m) > j:
        return 0.0
    if j < abs(j1 - j2) or j > j1 + j2 or (j1 + j2 + j) % 2:
        return 0.0
    if (j1 - m1) % 2 or (j2 - m2) % 2 or (j - m) % 2:
        return 0.0

    def half(value: int) -> int:
        return value // 2

    prefactor = (
        (j + 1)
        * factorial(half(j1 + j2 - j))
        * factorial(half(j1 - j2 + j))
        * factorial(half(-j1 + j2 + j))
        / factorial(half(j1 + j2 + j) + 1)
    )
    prefactor *= (
        factorial(half(j1 + m1)) * factorial(half(j1 - m1))
        * factorial(half(j2 + m2)) * factorial(half(j2 - m2))
        * factorial(half(j + m)) * factorial(half(j - m))
    )
    total = 0.0
    for k in range(0, half(j1 + j2 - j) + 1):
        terms = (
            k,
            half(j1 + j2 - j) - k,
            half(j1 - m1) - k,
            half(j2 + m2) - k,
            half(j - j2 + m1) + k,
            half(j - j1 - m2) + k,
        )
        if min(terms) < 0:
            continue
        denominator = 1.0
        for term in terms:
            denominator *= factorial(term)
        total += (-1.0) ** k / denominator
    return sqrt(prefactor) * total


def clebsch_gordan(
    j1: float, m1: float, j2: float, m2: float, j: float, m: float
) -> float:
    """Return ``<j1 m1 j2 m2 | j m>`` (Condon--Shortley phases)."""

    return _clebsch_gordan_twice(
        _twice(j1), _twice(m1), _twice(j2), _twice(m2), _twice(j), _twice(m)
    )


@dataclass(frozen=True)
class LSLevelLabel:
    """LS quantum numbers parsed from a Stout level label."""

    spin: float
    orbital: int
    total: float
    # Outer-electron and parent-core data for the diamagnetic term.
    outer_principal: int | None
    outer_orbital: int | None
    outer_occupancy: int
    core_orbital: int | None


def parse_ls_level_label(label: str) -> LSLevelLabel | None:
    """Parse ``...(2Po).4p.(1D<2>)`` style labels; ``None`` if not LS."""

    match = _FINAL_TERM.search(label)
    if match is None:
        return None
    multiplicity = int(match.group(1))
    orbital = _TERM_LETTERS.find(match.group(2))
    if orbital < 0 or multiplicity < 1:
        return None
    total = int(match.group(4)) / int(match.group(5) or 1)
    spin = 0.5 * (multiplicity - 1)
    if total < abs(orbital - spin) - 1.0e-9 or total > orbital + spin + 1.0e-9:
        return None
    tokens = label[: match.start()].rstrip(".").split(".")
    outer_n = outer_l = None
    occupancy = 1
    core = None
    orbital_positions = [
        index for index, token in enumerate(tokens) if _ORBITAL.match(token)
    ]
    if orbital_positions:
        position = orbital_positions[-1]
        outer = _ORBITAL.match(tokens[position])
        outer_n = int(outer.group(1))
        outer_l = _ORBITAL_LETTERS.find(outer.group(2))
        occupancy = int(outer.group(3) or 1)
        if outer_l < 0:
            outer_n = outer_l = None
        elif position > 0:
            parent = _PARENT_TERM.match(tokens[position - 1])
            if parent is not None:
                core = _TERM_LETTERS.find(parent.group(2))
                core = core if core >= 0 else None
        if outer_l is not None and core is None and occupancy == 1:
            # Closed shells (or s^2) below a single outer electron: L = l.
            core = 0
    return LSLevelLabel(spin, orbital, total, outer_n, outer_l, occupancy, core)


def lande_g_factor(spin: float, orbital: float, total: float) -> float:
    """LS-coupling Lande factor with the anomalous spin g-factor."""

    if total == 0:
        return 0.0
    spin_projection = (
        total * (total + 1.0) + spin * (spin + 1.0) - orbital * (orbital + 1.0)
    ) / (2.0 * total * (total + 1.0))
    return 1.0 + (ELECTRON_SPIN_G_FACTOR - 1.0) * spin_projection


def _orbital_sin2_theta(orbital: int, projection: int) -> float:
    """``<l m| sin^2 theta |l m>``."""

    return 2.0 * (orbital * orbital + orbital - 1.0 + projection * projection) / (
        (2.0 * orbital - 1.0) * (2.0 * orbital + 3.0)
    )


def hydrogenic_mean_square_radius(
    effective_principal: float, orbital: int, core_charge: int
) -> float:
    """``<r^2>/a_0^2`` of a hydrogenic electron with effective ``n``."""

    n2 = effective_principal * effective_principal
    return n2 / (2.0 * core_charge * core_charge) * (
        5.0 * n2 + 1.0 - 3.0 * orbital * (orbital + 1.0)
    )


def term_diamagnetic_coefficients(
    label: LSLevelLabel,
    binding_wavenumber: float | None,
    core_charge: int,
) -> dict[int, float] | None:
    """Return ``<r^2 sin^2 theta>/a_0^2`` for each ``M_L`` of a term.

    The outer electron ``(n* l)`` is coupled to a parent core of orbital
    momentum ``L_c``; the angular factor is the average of
    ``<l m|sin^2 theta|l m>`` over the orbital projections contained in
    ``|L M_L>``.  Equivalent outer electrons, or an unidentified parent, use
    the isotropic value 2/3.  ``None`` when the binding energy is unknown or
    not positive (the quadratic term is then omitted).
    """

    if (
        binding_wavenumber is None
        or not np.isfinite(binding_wavenumber)
        or binding_wavenumber <= 0.0
        or label.outer_orbital is None
    ):
        return None
    effective = core_charge * sqrt(_RYDBERG_WAVENUMBER / binding_wavenumber)
    effective = max(effective, label.outer_orbital + 1.0)
    radius = hydrogenic_mean_square_radius(effective, label.outer_orbital, core_charge)
    if radius <= 0.0:
        return None
    orbital = label.orbital
    result = {}
    for projection in range(-orbital, orbital + 1):
        if label.outer_occupancy != 1 or label.core_orbital is None:
            angular = 2.0 / 3.0
        else:
            angular = 0.0
            weight = 0.0
            for m in range(-label.outer_orbital, label.outer_orbital + 1):
                coefficient = clebsch_gordan(
                    label.core_orbital, projection - m,
                    label.outer_orbital, m, orbital, projection,
                ) ** 2
                angular += coefficient * _orbital_sin2_theta(label.outer_orbital, m)
                weight += coefficient
            angular = angular / weight if weight > 0.0 else 2.0 / 3.0
        result[projection] = radius * angular
    return result


@dataclass(frozen=True)
class TermZeemanStates:
    """Eigenstates of one LS term in a field, by doubled ``M``."""

    spin: float
    orbital: int
    # doubled M -> (basis [(M_L, doubled M_S)], energies [k], vectors [basis, k],
    #               dominant J of each eigenstate)
    states: Mapping[int, tuple[tuple[tuple[int, int], ...], FloatArray, FloatArray, tuple[float, ...]]]


def term_zeeman_states(
    spin: float,
    orbital: int,
    level_energy_wavenumber: Mapping[float, float],
    field_gauss: float,
    diamagnetic: Mapping[int, float] | None = None,
) -> TermZeemanStates:
    """Diagonalize fine structure plus Zeeman terms of one LS term.

    ``level_energy_wavenumber`` maps every ``J`` of the term to its energy.
    """

    spin2 = _twice(spin)
    totals = [0.5 * value for value in range(abs(2 * orbital - spin2), 2 * orbital + spin2 + 1, 2)]
    missing = [total for total in totals if total not in level_energy_wavenumber]
    if missing:
        raise ValueError("every J level of the term is required")
    zeeman = BOHR_MAGNETON_WAVENUMBER_PER_GAUSS * field_gauss
    quadratic = DIAMAGNETIC_WAVENUMBER_PER_GAUSS2 * field_gauss * field_gauss
    states = {}
    for m2 in range(-(2 * orbital + spin2), 2 * orbital + spin2 + 1, 2):
        basis = tuple(
            (ml, ms2)
            for ml in range(-orbital, orbital + 1)
            for ms2 in range(-spin2, spin2 + 1, 2)
            if 2 * ml + ms2 == m2
        )
        size = len(basis)
        hamiltonian = np.zeros((size, size))
        projections = {}
        for total in totals:
            if abs(m2) > _twice(total):
                continue
            vector = np.asarray([
                _clebsch_gordan_twice(2 * orbital, 2 * ml, spin2, ms2, _twice(total), m2)
                for ml, ms2 in basis
            ])
            projections[total] = vector
            hamiltonian += level_energy_wavenumber[total] * np.outer(vector, vector)
        for index, (ml, ms2) in enumerate(basis):
            hamiltonian[index, index] += zeeman * (
                ml + ELECTRON_SPIN_G_FACTOR * 0.5 * ms2
            )
            if diamagnetic is not None:
                hamiltonian[index, index] += quadratic * diamagnetic[ml]
        energies, vectors = np.linalg.eigh(hamiltonian)
        dominant = tuple(
            max(projections, key=lambda total: abs(float(projections[total] @ vectors[:, k])))
            for k in range(size)
        )
        states[m2] = (basis, energies, vectors, dominant)
    return TermZeemanStates(spin, orbital, MappingProxyType(states))


@dataclass(frozen=True)
class ZeemanComponent:
    """One magnetic component of a multiplet."""

    wavenumber: float
    # Fraction of the multiplet's total g f; the fractions of the three
    # polarizations each sum to 1/3.
    strength_fraction: float
    delta_m: int
    lower_total: float
    upper_total: float


def multiplet_zeeman_components(
    lower: TermZeemanStates,
    upper: TermZeemanStates,
    *,
    minimum_fraction: float = 1.0e-6,
) -> tuple[ZeemanComponent, ...]:
    """Return the E1 components between two diagonalized LS terms."""

    if abs(lower.spin - upper.spin) > 1.0e-9:
        raise ValueError("LS multiplets conserve the spin")
    if abs(lower.orbital - upper.orbital) > 1 or lower.orbital + upper.orbital < 1:
        raise ValueError("no electric-dipole coupling between these terms")
    spin2 = _twice(lower.spin)
    normalization = (spin2 + 1.0) * (2.0 * upper.orbital + 1.0)
    components = []
    for m2, (lower_basis, lower_energy, lower_vector, lower_j) in lower.states.items():
        for q in (-1, 0, 1):
            target = upper.states.get(m2 + 2 * q)
            if target is None:
                continue
            upper_basis, upper_energy, upper_vector, upper_j = target
            position = {state: index for index, state in enumerate(upper_basis)}
            # Dipole matrix in the uncoupled basis: [upper basis, lower basis].
            dipole = np.zeros((len(upper_basis), len(lower_basis)))
            for column, (ml, ms2) in enumerate(lower_basis):
                row = position.get((ml + q, ms2))
                if row is None:
                    continue
                dipole[row, column] = _clebsch_gordan_twice(
                    2 * lower.orbital, 2 * ml, 2, 2 * q,
                    2 * upper.orbital, 2 * (ml + q),
                )
            amplitude = upper_vector.T @ dipole @ lower_vector
            for b in range(amplitude.shape[0]):
                for a in range(amplitude.shape[1]):
                    fraction = amplitude[b, a] ** 2 / normalization
                    if fraction < minimum_fraction:
                        continue
                    components.append(ZeemanComponent(
                        float(upper_energy[b] - lower_energy[a]),
                        float(fraction), q, lower_j[a], upper_j[b],
                    ))
    return tuple(components)


def linear_zeeman_components(
    lower_total: float,
    upper_total: float,
    lower_g: float,
    upper_g: float,
    field_gauss: float,
) -> tuple[tuple[float, float, int], ...]:
    """Linear pattern of one ``J -> J'`` line.

    Returns ``(wavenumber shift, strength fraction, delta m)``; the fractions
    of each polarization sum to 1/3.
    """

    zeeman = BOHR_MAGNETON_WAVENUMBER_PER_GAUSS * field_gauss
    lower2, upper2 = _twice(lower_total), _twice(upper_total)
    components = []
    for m2 in range(-lower2, lower2 + 1, 2):
        for q in (-1, 0, 1):
            if abs(m2 + 2 * q) > upper2:
                continue
            strength = _clebsch_gordan_twice(
                lower2, m2, 2, 2 * q, upper2, m2 + 2 * q
            ) ** 2 / (upper2 + 1.0)
            if strength <= 0.0:
                continue
            shift = zeeman * (upper_g * 0.5 * (m2 + 2 * q) - lower_g * 0.5 * m2)
            components.append((float(shift), float(strength), q))
    return tuple(components)


def _term_binding_wavenumber(
    ion: AtomicIon, level: AtomicLevel, parent_ion: AtomicIon | None
) -> float | None:
    if ion.ionization_energy_ev is None:
        return None
    limit = ion.ionization_energy_ev
    if parent_ion is not None:
        limit += _parent_limit_excitation_ev(level, parent_ion)
    return limit * EV_TO_WAVENUMBER - level.energy_wavenumber


def zeeman_split_ion(
    ion: AtomicIon,
    field_gauss: float,
    *,
    parent_ion: AtomicIon | None = None,
    wavelength_range_angstrom: tuple[float, float] | None = None,
    minimum_oscillator_strength: float = 1.0e-6,
    transition_keys: frozenset[tuple[int, int]] | None = None,
    include_diamagnetic: bool = True,
) -> dict[int, tuple[AtomicTransition, ...]]:
    """Return the ion's E1 lines split into magnetic components.

    The result maps ``delta m`` to component transitions.  Each keeps the
    level indices, damping and Stark constants of a parent line of its
    multiplet and carries its own wavelength and an oscillator strength equal
    to *three times* its share, so that each ``delta m`` set alone has the
    full strength of the multiplet (the manifold convention of
    :class:`wd_spectra.magnetic.PolarizedOpacity`).
    """

    if not np.isfinite(field_gauss) or field_gauss < 0.0:
        raise ValueError("field_gauss must be finite and non-negative")
    levels = {level.index: level for level in ion.levels}
    parsed = {index: parse_ls_level_label(level.label) for index, level in levels.items()}
    term_of = {index: _level_term_label(level.label) for index, level in levels.items()}
    term_levels: dict[str, dict[float, AtomicLevel]] = {}
    for index, level in levels.items():
        label = parsed[index]
        if label is not None:
            term_levels.setdefault(term_of[index], {})[label.total] = level

    selected = []
    for transition in ion.transitions:
        if (
            transition.transition_type != "E1"
            or transition.absorption_oscillator_strength < minimum_oscillator_strength
            or transition.lower_index not in levels
            or transition.upper_index not in levels
        ):
            continue
        if transition_keys is not None and (
            transition.lower_index, transition.upper_index
        ) not in transition_keys:
            continue
        if wavelength_range_angstrom is not None and not (
            wavelength_range_angstrom[0]
            <= transition.wavelength_vacuum_angstrom
            <= wavelength_range_angstrom[1]
        ):
            continue
        selected.append(transition)

    core_charge = ion.charge + 1
    result: dict[int, list[AtomicTransition]] = {-1: [], 0: [], 1: []}

    def add(parent: AtomicTransition, wavenumber: float, gf: float, q: int) -> None:
        if wavenumber <= 0.0 or gf <= 0.0:
            return
        lower_weight = levels[parent.lower_index].statistical_weight
        result[q].append(replace(
            parent,
            wavelength_vacuum_angstrom=1.0e8 / wavenumber,
            absorption_oscillator_strength=3.0 * gf / lower_weight,
        ))

    def term_states(term: str) -> TermZeemanStates | None:
        members = term_levels.get(term)
        if not members:
            return None
        sample = parsed[next(iter(members.values())).index]
        spin2 = _twice(sample.spin)
        totals = [
            0.5 * value
            for value in range(abs(2 * sample.orbital - spin2), 2 * sample.orbital + spin2 + 1, 2)
        ]
        if any(total not in members for total in totals):
            return None
        diamagnetic = None
        if include_diamagnetic:
            # The term's centre of gravity sets the effective quantum number.
            weights = np.asarray([2.0 * total + 1.0 for total in totals])
            centroid_level = replace(
                members[totals[0]],
                energy_wavenumber=float(np.sum(
                    weights * np.asarray([members[t].energy_wavenumber for t in totals])
                ) / np.sum(weights)),
            )
            diamagnetic = term_diamagnetic_coefficients(
                sample,
                _term_binding_wavenumber(ion, centroid_level, parent_ion),
                core_charge,
            )
        return term_zeeman_states(
            sample.spin, sample.orbital,
            {total: members[total].energy_wavenumber for total in totals},
            field_gauss, diamagnetic,
        )

    state_cache: dict[str, TermZeemanStates | None] = {}
    multiplets: dict[tuple[str, str], list[AtomicTransition]] = {}
    for transition in selected:
        multiplets.setdefault(
            (term_of[transition.lower_index], term_of[transition.upper_index]), []
        ).append(transition)

    for (lower_term, upper_term), lines in multiplets.items():
        lower_label = parsed[lines[0].lower_index]
        upper_label = parsed[lines[0].upper_index]
        paschen_back = (
            lower_label is not None
            and upper_label is not None
            and abs(lower_label.spin - upper_label.spin) < 1.0e-9
            and abs(lower_label.orbital - upper_label.orbital) <= 1
            and lower_label.orbital + upper_label.orbital >= 1
        )
        lower_states = upper_states = None
        if paschen_back:
            for term in (lower_term, upper_term):
                if term not in state_cache:
                    state_cache[term] = term_states(term)
            lower_states, upper_states = state_cache[lower_term], state_cache[upper_term]
        if lower_states is not None and upper_states is not None:
            total_gf = sum(
                levels[line.lower_index].statistical_weight
                * line.absorption_oscillator_strength
                for line in lines
            )
            by_totals = {
                (parsed[line.lower_index].total, parsed[line.upper_index].total): line
                for line in lines
            }
            strongest = max(lines, key=lambda line: line.absorption_oscillator_strength)
            for component in multiplet_zeeman_components(lower_states, upper_states):
                parent = by_totals.get(
                    (component.lower_total, component.upper_total), strongest
                )
                add(
                    parent, component.wavenumber,
                    total_gf * component.strength_fraction, component.delta_m,
                )
            continue
        # Linear pattern of each line on its own J levels.
        for line in lines:
            lower_level, upper_level = levels[line.lower_index], levels[line.upper_index]
            lower_parsed, upper_parsed = parsed[line.lower_index], parsed[line.upper_index]
            wavenumber = 1.0e8 / line.wavelength_vacuum_angstrom
            gf = lower_level.statistical_weight * line.absorption_oscillator_strength
            lower_total = 0.5 * (lower_level.statistical_weight - 1.0)
            upper_total = 0.5 * (upper_level.statistical_weight - 1.0)
            if (
                lower_parsed is not None and upper_parsed is not None
                and abs(lower_total - upper_total) <= 1.0 and lower_total + upper_total >= 1.0
            ):
                pattern = linear_zeeman_components(
                    lower_total, upper_total,
                    lande_g_factor(lower_parsed.spin, lower_parsed.orbital, lower_total),
                    lande_g_factor(upper_parsed.spin, upper_parsed.orbital, upper_total),
                    field_gauss,
                )
            else:
                zeeman = BOHR_MAGNETON_WAVENUMBER_PER_GAUSS * field_gauss
                pattern = tuple((q * zeeman, 1.0 / 3.0, q) for q in (-1, 0, 1))
            for shift, fraction, q in pattern:
                add(line, wavenumber + shift, gf * fraction, q)
    return {q: tuple(lines) for q, lines in result.items()}


def zeeman_split_database(
    database: AtomicDatabase,
    field_megagauss: float,
    *,
    wavelength_range_angstrom: tuple[float, float] | None = None,
    minimum_oscillator_strength: float = 1.0e-6,
    line_transition_keys: Iterable[tuple[str, int, int, int]] | None = None,
    include_diamagnetic: bool = True,
) -> dict[int, AtomicDatabase]:
    """Return one line database per polarization (``delta m = -1, 0, +1``).

    The levels (and so the partition functions and populations) are those of
    ``database``; only the E1 transitions are replaced by their magnetic
    components.  Evaluating the ordinary line opacity with database ``q``
    gives the ``delta m = q`` manifold opacity.
    """

    field_gauss = float(field_megagauss) * 1.0e6
    keys_by_ion: dict[tuple[str, int], set[tuple[int, int]]] | None = None
    if line_transition_keys is not None:
        keys_by_ion = {}
        for element, charge, lower, upper in line_transition_keys:
            keys_by_ion.setdefault((element, int(charge)), set()).add((int(lower), int(upper)))
    split: dict[int, dict] = {-1: {}, 0: {}, 1: {}}
    by_stage = {(ion.element, ion.charge): ion for ion in database.ions.values()}
    for key, ion in database.ions.items():
        keys = None
        if keys_by_ion is not None:
            keys = frozenset(keys_by_ion.get((ion.element, ion.charge), ()))
        components = zeeman_split_ion(
            ion,
            field_gauss,
            parent_ion=by_stage.get((ion.element, ion.charge + 1)),
            wavelength_range_angstrom=wavelength_range_angstrom,
            minimum_oscillator_strength=minimum_oscillator_strength,
            transition_keys=keys,
            include_diamagnetic=include_diamagnetic,
        )
        for q in (-1, 0, 1):
            split[q][key] = replace(ion, transitions=components[q])
    return {
        q: AtomicDatabase(
            MappingProxyType(split[q]),
            source=database.source
            + f"; LS Paschen-Back components at {float(field_megagauss):g} MG, delta m = {q:+d}",
        )
        for q in (-1, 0, 1)
    }
