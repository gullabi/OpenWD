"""Simplified semiclassical Stark widths and shifts of neutral-atom lines.

Dimitrijevic & Konjevic (1986, A&A 163, 297; DK86), extending Freudenstein &
Cooper (1978, ApJ 224, 1079; FC78), give the electron-impact half width
``w`` and shift ``d`` (angular frequency) of an isolated neutral line as

    w + i d = (32/27)^(1/2) pi (hbar a0/m) N_e (E_H/kT)^(1/2)
              { sum_i' R_ii'^2 [f_w(x_ii') - i eps_ii' f_d(x_ii')]
              + sum_f' R_ff'^2 [f_w(x_ff') + i eps_ff' f_d(x_ff')] },

with ``i``/``f`` the upper/lower level, ``i'``/``f'`` their dipole
perturbers, ``eps_jj' = sign(E_j - E_j')``, ``x_jj' = R_jj' |E_j - E_j'| / 3kT``
and ``R_jj'^2`` (a0^2) the squared coordinate matrix element summed over
``M'`` and averaged over ``M``.  ``f_w`` and ``f_d`` are the fitted forms of
DK86 eqs. (8) and (9).  A positive ``d`` is a red shift.

Matrix elements are those of one optical electron outside a fixed core in LS
coupling: ``R_jj'^2 = (2L'+1) {l L Lc; L' l' 1}^2 max(l, l') <nl|r|n'l'>^2``
(the Shore & Menzel multiplet factor), which sums over ``L'`` to the
one-electron value ``max(l, l') <nl|r|n'l'>^2 / (2l+1)``.  The radial
integrals are evaluated numerically in the Coulomb approximation of Bates &
Damgaard (1949) from the effective quantum numbers of the tabulated term
energies.  Perturbing configurations listed only in jK coupling (``nf``) are
treated as one level at their weighted mean energy with the one-electron
factor.

FC78 and DK86 quote an accuracy of about 40-50 per cent; the electron widths
do not include quasi-static ion broadening.

This implementation reproduces the DK86 He I 4438 and 6678 A widths to
within 7 per cent.  For the measured C I 3s-4p multiplets (Konjevic & Wiese
1990; Konjevic et al. 2002) the DK86 sum gives twice the laboratory widths,
while the nearest-level form of FC78 (:func:`nearest_level_stark_width`)
gives a median of 1.34 times and a scatter of 0.17 dex, the 25 per cent
overestimate FC78 report for it; the Kurucz constants of these lines give
0.72 times with 0.39 dex.  The warm-DQ class therefore uses the FC78 form
for C I lines without measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import factorial
import re

import numpy as np
from scipy.special import hyperu

from ._compat import trapezoid
from .constants import PI
from .metals import AtomicIon, _level_term_label, _rydberg_orbital

_HBAR = 1.054571817e-27
_ELECTRON_MASS = 9.1093837015e-28
_BOHR_RADIUS_CM = 5.29177210903e-9
_RYDBERG_WAVENUMBER = 109_737.31568
_HYDROGEN_IONIZATION_KELVIN = 157_887.5
_WAVENUMBER_TO_KELVIN = 1.438776877
# (32/27)^(1/2) pi hbar a0 / m, cm^3 s^-1 (DK86 eq. 4).
_DK86_PREFACTOR = np.sqrt(32.0 / 27.0) * PI * _HBAR * _BOHR_RADIUS_CM / _ELECTRON_MASS
_L_LETTERS = "SPDFGHIKLMN"
_TERM_PATTERN = re.compile(r"\((\d+)([SPDFGHIKLMN])(o?)(?:<[^>]*>)?\)$")
_CORE_TERM_PATTERN = re.compile(r"\((\d+)([SPDFGHIKLMN])(o?)(?:<[^>]*>)?\)$")


def dk86_width_function(x: np.ndarray | float) -> np.ndarray:
    """DK86 eq. (8): ``f_w(x)``, accurate to a few per cent (FC78)."""

    x = np.maximum(np.asarray(x, dtype=np.float64), 1.0e-12)
    return (
        np.exp(-1.33 * x) * np.log1p(2.27 / x)
        + 0.487 * x / (0.153 + x ** (5.0 / 3.0))
        + x / (7.93 + x**3)
    )


def dk86_shift_function(x: np.ndarray | float) -> np.ndarray:
    """DK86 eq. (9): ``f_d(x)``."""

    x = np.maximum(np.asarray(x, dtype=np.float64), 0.0)
    return (
        1.571 * np.exp(-2.482 * x)
        + 1.295 * x / (0.415 + x ** (5.0 / 3.0))
        + 0.713 * x / (8.139 + x**3)
    )


def wigner_6j(a: int, b: int, c: int, d: int, e: int, f: int) -> float:
    """Wigner 6j symbol ``{a b c; d e f}`` for integer arguments (Racah formula)."""

    def triangle(x, y, z):
        if x + y < z or x + z < y or y + z < x:
            return None
        return factorial(x + y - z) * factorial(x - y + z) * factorial(-x + y + z) / factorial(x + y + z + 1)

    triads = ((a, b, c), (a, e, f), (d, b, f), (d, e, c))
    deltas = [triangle(*t) for t in triads]
    if any(value is None for value in deltas):
        return 0.0
    prefactor = np.sqrt(np.prod(deltas))
    sums = [sum(t) for t in triads]
    quads = (a + b + d + e, a + c + d + f, b + c + e + f)
    total = 0.0
    for t in range(max(sums), min(quads) + 1):
        denominator = factorial(quads[0] - t) * factorial(quads[1] - t) * factorial(quads[2] - t)
        for s in sums:
            denominator *= factorial(t - s)
        total += (-1) ** t * factorial(t + 1) / denominator
    return float(prefactor * total)


# One radial grid (a0) for every function: square-root spacing resolves the
# inner region and reaches 6 n*^2 + 40 for n* up to 16.
_RADIAL_GRID = np.linspace(np.sqrt(1.0e-3), np.sqrt(6.0 * 16.0**2 + 40.0), 60_000) ** 2


def coulomb_radial_function(n_eff: float, l: int, radius: np.ndarray) -> np.ndarray:
    """Normalized Coulomb-approximation radial function ``P(r)`` (Z = 1, r in a0).

    ``P = W_{n*, l+1/2}(2r/n*)``.  For non-integer ``n*`` the function
    diverges at the origin, where the asymptotic (Coulomb) form does not
    describe the atom.  The divergent part is removed at the minimum of
    ``|P|`` inside the classical inner turning point
    ``n*^2 (1 - sqrt(1 - l(l+1)/n*^2))``, where the bound function is
    negligible (Bates & Damgaard 1949).
    """

    z = 2.0 * radius / n_eff
    with np.errstate(all="ignore"):
        value = np.exp(-z / 2.0) * z ** (l + 1.0) * hyperu(l + 1.0 - n_eff, 2.0 * l + 2.0, z)
    value = np.where(np.isfinite(value), value, 0.0)
    magnitude = np.abs(value)
    if l > 0:
        turning = n_eff**2 * (1.0 - np.sqrt(max(1.0 - l * (l + 1.0) / n_eff**2, 0.0)))
        inner = np.flatnonzero(radius <= turning)
        if inner.size > 1:
            value[: inner[np.argmin(magnitude[inner])]] = 0.0
    elif magnitude[1] < magnitude[0]:
        index = 1
        while index < magnitude.size - 1 and magnitude[index + 1] < magnitude[index]:
            index += 1
        value[:index] = 0.0
    norm = np.sqrt(trapezoid(value**2, radius))
    return value / norm


@lru_cache(maxsize=4096)
def _cached_radial_function(n_eff: float, l: int) -> np.ndarray:
    return coulomb_radial_function(n_eff, l, _RADIAL_GRID)


def coulomb_dipole_radial_integral(n_eff_1: float, l_1: int, n_eff_2: float, l_2: int) -> float:
    """``<n1 l1 | r | n2 l2>`` in a0 in the Coulomb approximation (Z = 1)."""

    product = (
        _cached_radial_function(round(n_eff_1, 4), l_1)
        * _cached_radial_function(round(n_eff_2, 4), l_2)
        * _RADIAL_GRID
    )
    return float(trapezoid(product, _RADIAL_GRID))


@dataclass(frozen=True)
class _Term:
    label: str
    core: str
    core_L: int
    n: int
    l: int
    S: int | None  # None for jK-coupled levels
    L: int | None
    odd: bool
    energy: float  # cm^-1, (2J+1)-weighted
    n_eff: float


def _core_term(core: str) -> tuple[int, bool] | None:
    """``(L, odd parity)`` of the core term at the end of a core label."""

    match = _CORE_TERM_PATTERN.search(core)
    return None if match is None else (_L_LETTERS.index(match.group(2)), match.group(3) == "o")


def ion_terms(ion: AtomicIon) -> dict[str, _Term]:
    """LS terms (and jK configurations) of single-electron excitations of an ion."""

    if ion.ionization_energy_ev is None:
        return {}
    limit = ion.ionization_energy_ev * 8065.543937
    groups: dict[str, list] = {}
    for level in ion.levels:
        parsed = _rydberg_orbital(level.label)
        if parsed is None:
            continue
        n, l, core = parsed
        core = re.sub(r"<[^>]*>", "", core)
        term = _level_term_label(level.label)
        match = _TERM_PATTERN.search(term)
        key = term if match is not None else f"{core}.{n}{'spdfghiklmn'[l]}.(jK)"
        groups.setdefault(key, []).append((level, n, l, core, match))
    terms = {}
    for key, members in groups.items():
        weight = sum(level.statistical_weight for level, *_ in members)
        energy = sum(level.statistical_weight * level.energy_wavenumber for level, *_ in members) / weight
        _, n, l, core, match = members[0]
        core_term = _core_term(core)
        if core_term is None or energy >= limit:
            continue
        core_l, core_odd = core_term
        n_eff = float(np.sqrt(_RYDBERG_WAVENUMBER * (ion.charge + 1.0) ** 2 / (limit - energy)))
        odd = (match.group(3) == "o") if match is not None else core_odd != bool(l % 2)
        terms[key] = _Term(
            key, core, core_l, n, l,
            int(match.group(1)) if match is not None else None,
            _L_LETTERS.index(match.group(2)) if match is not None else None,
            odd, energy, n_eff,
        )
    return terms


def _squared_matrix_element(perturbed: _Term, perturber: _Term) -> float:
    radial = coulomb_dipole_radial_integral(perturbed.n_eff, perturbed.l, perturber.n_eff, perturber.l)
    l_max = max(perturbed.l, perturber.l)
    if perturbed.L is None or perturber.L is None:
        return l_max * radial**2 / (2 * perturbed.l + 1)
    angular = (2 * perturber.L + 1) * wigner_6j(
        perturbed.l, perturbed.L, perturbed.core_L, perturber.L, perturber.l, 1
    ) ** 2
    return angular * l_max * radial**2


def perturbers(terms: dict[str, _Term], term: _Term) -> list[tuple[_Term, float]]:
    """Dipole perturbers of ``term`` with their ``R^2`` (a0^2)."""

    result = []
    for other in terms.values():
        if (
            other.core != term.core or abs(other.l - term.l) != 1 or other.odd == term.odd
            or (other.S is not None and term.S is not None and other.S != term.S)
            or (other.L is not None and term.L is not None and abs(other.L - term.L) > 1)
        ):
            continue
        r_squared = _squared_matrix_element(term, other)
        if r_squared > 0.0:
            result.append((other, r_squared))
    return result


def _level_sums(terms, term, temperature):
    """``(sum R^2 f_w, sum eps R^2 f_d)`` for one level at ``temperature``."""

    width = shift = 0.0
    for other, r_squared in perturbers(terms, term):
        gap = (term.energy - other.energy) * _WAVENUMBER_TO_KELVIN
        x = np.sqrt(r_squared) * abs(gap) / (3.0 * temperature)
        width += r_squared * float(dk86_width_function(x))
        shift += np.sign(gap) * r_squared * float(dk86_shift_function(x))
    return width, shift


def simplified_semiclassical_stark(
    terms: dict[str, _Term], upper: _Term, lower: _Term | None, temperature: float,
) -> tuple[float, float]:
    """Return ``(Gamma/N_e, d/N_e)`` in cm^3 s^-1 at ``temperature``.

    ``Gamma`` is the angular-frequency FWHM (twice the DK86 half width) and
    ``d`` the angular-frequency shift, positive to the red.  ``lower=None``
    omits the lower level (a level outside the single-electron term set).
    """

    upper_width, upper_shift = _level_sums(terms, upper, temperature)
    lower_width, lower_shift = (0.0, 0.0) if lower is None else _level_sums(terms, lower, temperature)
    scale = _DK86_PREFACTOR * np.sqrt(_HYDROGEN_IONIZATION_KELVIN / temperature)
    return 2.0 * scale * (upper_width + lower_width), scale * (-upper_shift + lower_shift)


def completeness(terms: dict[str, _Term], term: _Term) -> float:
    """``sum R^2 / <r^2>`` (DK86 p. 299); near 1 when the perturber set is complete."""

    mean_square = 0.5 * term.n_eff**2 * (5.0 * term.n_eff**2 + 1.0 - 3.0 * term.l * (term.l + 1))
    return sum(value for _, value in perturbers(terms, term)) / mean_square


def nearest_level_stark_width(
    terms: dict[str, _Term], upper: _Term, temperature: float,
) -> float:
    """Freudenstein & Cooper (1978) width ``Gamma/N_e`` (FWHM, cm^3 s^-1).

    FC78 eq. (12): the whole upper-level strength ``R^2 = <r^2> =
    (n*^2/2)(5 n*^2 + 1 - 3 l(l+1))`` is placed at the nearest dipole-coupled
    level (spacing ``Delta E``), ``w = (32/27)^(1/2) pi (hbar a0/m) N_e
    (E_H/kT)^(1/2) R^2 f_w(R Delta E / 3kT)``; the lower level is neglected.
    FC78 quote agreement within 25 per cent for C I 5052 A over
    5000-40,000 K and about 40-50 per cent in general.
    """

    candidates = perturbers(terms, upper)
    if not candidates:
        return 0.0
    gap = min(abs(other.energy - upper.energy) for other, _ in candidates) * _WAVENUMBER_TO_KELVIN
    mean_square = 0.5 * upper.n_eff**2 * (5.0 * upper.n_eff**2 + 1.0 - 3.0 * upper.l * (upper.l + 1))
    x = np.sqrt(mean_square) * gap / (3.0 * temperature)
    scale = _DK86_PREFACTOR * np.sqrt(_HYDROGEN_IONIZATION_KELVIN / temperature)
    return float(2.0 * scale * mean_square * dk86_width_function(x))
