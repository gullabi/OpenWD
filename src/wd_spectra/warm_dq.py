"""Carbon-rich warm/hot DQ atmospheres: shared pieces beyond the D6 mixture.

Warm DQ, DAQ and hot DQ white dwarfs (Teff ~ 11,000-25,000 K) have atmospheres
of carbon with hydrogen (and possibly helium).  Their thermodynamics and
opacity use the bulk-mixture closure of :mod:`wd_spectra.d6`, which accepts
hydrogen as a full member of the mixture.  This module holds the atomic-data
preparation that is specific to these stars.
"""

from __future__ import annotations

from dataclasses import replace
import re
from types import MappingProxyType

import numpy as np

from .metals import AtomicDatabase, AtomicLevel, AtomicTransition, _level_term_label


def atomic_database_with_multiplet_electron_stark(
    database: AtomicDatabase,
) -> AtomicDatabase:
    """Give lines without an electron Stark constant their multiplet's value.

    The Kurucz table matches Stout fine-structure lines one by one, and some
    members of a multiplet have no entry (for example C II 4267.26 A,
    3d 2D5/2 - 4f 2Fo7/2, next to its tabulated 2D3/2 - 2Fo5/2 partner).
    Electron-impact widths are set by the radial wave functions of the two
    LS terms and are the same for every fine-structure component to within
    the accuracy of the data, so a missing member receives the median
    constant of the tabulated members of the same lower and upper terms.
    Lines whose multiplet has no tabulated member are unchanged.
    """

    ions = dict(database.ions)
    filled = 0
    for key, ion in database.ions.items():
        term = {level.index: _level_term_label(level.label) for level in ion.levels}
        tabulated: dict[tuple[str, str], list[float]] = {}
        for transition in ion.transitions:
            value = transition.electron_stark_rate_coefficient_cm3_s
            if transition.transition_type == "E1" and value is not None:
                tabulated.setdefault(
                    (term[transition.lower_index], term[transition.upper_index]), []
                ).append(float(value))
        if not tabulated:
            continue
        changed = []
        any_changed = False
        for transition in ion.transitions:
            if (
                transition.transition_type == "E1"
                and transition.electron_stark_rate_coefficient_cm3_s is None
            ):
                values = tabulated.get(
                    (term[transition.lower_index], term[transition.upper_index])
                )
                if values:
                    transition = replace(
                        transition,
                        electron_stark_rate_coefficient_cm3_s=float(np.median(values)),
                    )
                    filled += 1
                    any_changed = True
            changed.append(transition)
        if any_changed:
            ions[key] = replace(ion, transitions=tuple(changed))
    if filled == 0:
        return database
    return AtomicDatabase(
        MappingProxyType(ions),
        source=database.source
        + f"; multiplet-median electron Stark constants for {filled} lines",
    )


# Measured Stark widths (FWHM) and shifts of optical C I lines, from the NIST
# critical reviews of Konjevic & Wiese (1990, J. Phys. Chem. Ref. Data 19,
# 1307; KW90) and Konjevic, Lesage, Fuhr & Wiese (2002, JPCRD 31, 819;
# KLFW02).  Sources of the individual measurements:
#   M95: Mijatovic et al. 1995, Phys. Rev. E 51, 613 (KLFW02, accuracy A);
#   G83: Goly et al. 1983, JQSRT 30, 417 (KW90, widths A-B, shifts B-C+);
#   JW84: Jones & Wiese 1984, Phys. Rev. A 30, 2602 (KW90, widths A-B+);
#   GW87: Goly & Weniger 1987, JQSRT 38, 225 (KW90).
# Every measured shift of these lines is to the red.  The measurements
# (n_e = 0.1-0.8e17 cm^-3, T = 9000-11600 K) include the quasi-static ion
# contribution and span the densities and temperatures at which the lines
# form in warm DQ atmospheres; they are scaled linearly with n_e and with
# (T/10^4 K)^(1/6).
# Entries: (lower LS term, upper LS term, measurements), each measurement
# (air wavelength A, FWHM A, shift A or None, T K, n_e cm^-3).  All
# measurements of a multiplet are averaged in Gamma/n_e and d/n_e.
_C_I_3S = "2s2.2p.(2Po).3s"
_C_I_4P = "2s2.2p.(2Po).4p"
MEASURED_CARBON_I_STARK = (
    # JW84
    (f"{_C_I_3S}.(3Po)", f"{_C_I_4P}.(3S)",
     ((4826.80, 1.88, None, 11_600.0, 0.793e17),
      (4817.37, 1.92, None, 11_600.0, 0.793e17),
      (4812.92, 1.68, None, 11_600.0, 0.793e17))),
    # G83 (width and shift), JW84 (widths)
    (f"{_C_I_3S}.(3Po)", f"{_C_I_4P}.(3P)",
     ((4771.75, 1.52, 0.57, 11_600.0, 0.596e17),
      (4775.91, 0.35, 0.15, 10_000.0, 0.165e17),
      (4771.75, 2.08, None, 11_600.0, 0.793e17),
      (4766.68, 2.04, None, 11_600.0, 0.793e17),
      (4775.91, 2.09, None, 11_600.0, 0.793e17),
      (4770.03, 1.75, None, 11_600.0, 0.793e17))),
    # M95 (width and shift), JW84 (width)
    (f"{_C_I_3S}.(1Po)", f"{_C_I_4P}.(1P)",
     ((5380.34, 0.58, 0.092, 9_900.0, 0.220e17),
      (5380.34, 0.77, 0.081, 10_300.0, 0.285e17),
      (5380.34, 2.29, None, 11_600.0, 0.739e17))),
    # M95 and G83 (widths and shifts), JW84 (width)
    (f"{_C_I_3S}.(1Po)", f"{_C_I_4P}.(1D)",
     ((5052.17, 0.67, 0.37, 9_670.0, 0.185e17),
      (5052.17, 0.79, 0.43, 9_870.0, 0.215e17),
      (5052.17, 0.81, 0.43, 9_890.0, 0.220e17),
      (5052.17, 0.93, 0.49, 10_110.0, 0.255e17),
      (5052.17, 1.06, 0.55, 10_280.0, 0.285e17),
      (5052.17, 1.06, 0.56, 10_310.0, 0.290e17),
      (5052.17, 1.94, 1.01, 11_600.0, 0.596e17),
      (5052.17, 3.08, None, 11_600.0, 0.793e17))),
    (f"{_C_I_3S}.(1Po)", f"{_C_I_4P}.(1S)",
     ((4932.05, 1.05, 0.59, 9_890.0, 0.220e17),
      (4932.05, 1.35, 0.72, 10_280.0, 0.285e17),
      (4932.05, 2.34, 1.56, 11_600.0, 0.596e17),
      (4932.05, 3.90, None, 11_600.0, 0.793e17))),
    # GW87
    ("2s.2p3.(3Do)", f"{_C_I_4P}.(3P)", ((5798.0, 1.03, None, 11_400.0, 0.610e17),)),
)
_C_I_STARK_TEMPERATURE_EXPONENT = 1.0 / 6.0

# Semiclassical-perturbation electron-impact widths and shifts of C II lines
# (STARK-B dataset 126: Mahmoudi et al. 2004, Phys. Scr. 70, 142; 2005, 71,
# 190; Larbi-Terzi et al. 2012, MNRAS 423, 766), FWHM w and shift d in A at
# n_e = 1e17 cm^-3 and 10, 20 and 30 kK, for the code wavelength lambda_c;
# they scale as lambda^2 to the observed wavelength.  Stated accuracy about
# 20 per cent; the measured widths of these multiplets (Konjevic & Wiese
# 1990; Larbi-Terzi et al. 2012, Table 7) agree to within 0.8-1.4 except
# for the resonance-like 2512 A multiplet.  Shifts below a tenth of the
# width are omitted.  Entries: (lower term, upper term, lambda_c A,
# observed air wavelength A, ((T, w, d or None), ...)).
CARBON_II_SCP_STARK = (
    ("2s2.3s.(2S)", "2s2.3p.(2Po)", 6741.6, 6580.0,
     ((10_000.0, 1.05, None), (20_000.0, 0.848, None), (30_000.0, 0.791, None))),
    ("2s2.3p.(2Po)", "2s2.4s.(2S)", 3935.7, 3920.0,
     ((10_000.0, 0.764, 0.453), (20_000.0, 0.683, 0.371), (30_000.0, 0.660, 0.324))),
    ("2s2.3p.(2Po)", "2s2.3d.(2D)", 7226.4, 7234.0,
     ((10_000.0, 1.34, None), (20_000.0, 1.10, None), (30_000.0, 1.03, None))),
    ("2s2.3d.(2D)", "2s2.4f.(2Fo)", 4299.9, 4267.3,
     ((10_000.0, 1.75, None), (20_000.0, 1.42, None), (30_000.0, 1.30, None))),
    ("2s.2p.(3Po).3s.(4Po)", "2s.2p.(3Po).3p.(4P)", 5148.7, 5144.0,
     ((10_000.0, 0.714, -0.270), (20_000.0, 0.513, -0.208), (30_000.0, 0.441, -0.181))),
    ("2s.2p.(3Po).3s.(4Po)", "2s.2p.(3Po).3p.(4S)", 5612.2, 5650.0,
     ((10_000.0, 0.813, -0.309), (20_000.0, 0.583, -0.237), (30_000.0, 0.503, -0.206))),
    ("2s.2p2.(2S)", "2s2.3p.(2Po)", 2998.8, 2837.0,
     ((10_000.0, 0.168, 0.0357), (20_000.0, 0.135, 0.0276), (30_000.0, 0.124, 0.0249))),
    ("2s.2p2.(2P)", "2p3.(2Do)", 2560.7, 2511.0,
     ((10_000.0, 0.0374, None), (20_000.0, 0.0290, None), (30_000.0, 0.0249, None))),
)
_SCP_ELECTRON_DENSITY = 1.0e17


def _angular_rate_per_electron(wavelength_angstrom, value_angstrom, electron_density):
    from .constants import LIGHT_SPEED, PI

    return (
        2.0 * PI * LIGHT_SPEED * (value_angstrom * 1.0e-8)
        / (wavelength_angstrom * 1.0e-8) ** 2 / electron_density
    )


def carbon_stark_rate_coefficients() -> dict[
    tuple[int, str, str], tuple[float, float, float | None, float | None]
]:
    """Return the measured C I and semiclassical C II Stark data per multiplet.

    ``{(charge, lower term, upper term): (Gamma/n_e, width exponent, d/n_e,
    shift exponent)}``: ``Gamma`` is the angular-frequency FWHM and ``d`` the
    angular-frequency shift (positive to the red), both per electron at
    10^4 K, and the exponents of ``(T/10^4 K)`` that scale them.  C I values
    average the measurements in :data:`MEASURED_CARBON_I_STARK`; C II values
    are least-squares power laws through the 10-30 kK points of
    :data:`CARBON_II_SCP_STARK`.  ``d/n_e`` is None without a shift.
    """

    result = {}
    exponent = _C_I_STARK_TEMPERATURE_EXPONENT
    for lower, upper, measurements in MEASURED_CARBON_I_STARK:
        scale = [(1.0e4 / temperature) ** exponent for *_, temperature, _ in measurements]
        widths = [
            _angular_rate_per_electron(wavelength, width, density) * factor
            for (wavelength, width, _, _, density), factor in zip(measurements, scale)
        ]
        shifts = [
            _angular_rate_per_electron(wavelength, shift, density) * factor
            for (wavelength, _, shift, _, density), factor in zip(measurements, scale)
            if shift is not None
        ]
        result[(0, lower, upper)] = (
            float(np.mean(widths)), exponent,
            float(np.mean(shifts)) if shifts else None,
            exponent if shifts else None,
        )
    for lower, upper, code_wavelength, wavelength, table in CARBON_II_SCP_STARK:
        log_temperature = np.log([row[0] / 1.0e4 for row in table])
        conversion = (wavelength / code_wavelength) ** 2

        def power_law(values):
            slope, intercept = np.polyfit(log_temperature, np.log(values), 1)
            return (
                _angular_rate_per_electron(
                    wavelength, conversion * float(np.exp(intercept)), _SCP_ELECTRON_DENSITY
                ),
                float(slope),
            )

        width, width_exponent = power_law([row[1] for row in table])
        shift = shift_exponent = None
        if table[0][2] is not None:
            sign = np.sign(table[0][2])
            magnitude, shift_exponent = power_law([abs(row[2]) for row in table])
            shift = float(sign * magnitude)
        result[(1, lower, upper)] = (width, width_exponent, shift, shift_exponent)
    return result


def atomic_database_with_carbon_stark(database: AtomicDatabase) -> AtomicDatabase:
    """Give the C I and C II multiplets with Stark data their widths and shifts.

    Every fine-structure E1 component of a multiplet in
    :func:`carbon_stark_rate_coefficients` receives the multiplet's
    Gamma/n_e (replacing a Kurucz or multiplet-median constant), its shift
    where one exists, and their temperature exponents.  The Kurucz constants
    of these lines are 0.1 to 0.9 times the measured C I widths (C I 5380 A:
    2.2e-6 against 1.8e-5 cm^3 s^-1) and 0.2-0.6 times the C II ones.
    """

    data = carbon_stark_rate_coefficients()
    ions = dict(database.ions)
    replaced = 0
    for key, ion in database.ions.items():
        if ion.element != "C" or ion.charge not in (0, 1):
            continue
        term = {level.index: _level_term_label(level.label) for level in ion.levels}
        changed = []
        any_changed = False
        for transition in ion.transitions:
            value = data.get(
                (ion.charge, term[transition.lower_index], term[transition.upper_index])
            )
            if transition.transition_type == "E1" and value is not None:
                transition = replace(
                    transition,
                    electron_stark_rate_coefficient_cm3_s=value[0],
                    electron_stark_temperature_exponent=value[1],
                    electron_stark_shift_rate_coefficient_cm3_s=value[2],
                    electron_stark_shift_temperature_exponent=value[3],
                )
                replaced += 1
                any_changed = True
            changed.append(transition)
        if any_changed:
            ions[key] = replace(ion, transitions=tuple(changed))
    if replaced == 0:
        return database
    return AtomicDatabase(
        MappingProxyType(ions),
        source=database.source
        + f"; measured C I and semiclassical C II (STARK-B) Stark widths and shifts"
        f" for {replaced} lines",
    )


LI2021_C_I_TABLE_RESOURCE = "data/atomic/li2021_c_i_e1.csv.xz"
_LI2021_C_I_COLUMNS = [
    "upper_label", "lower_label", "vacuum_wavelength_A", "predicted_wavelength", "gf",
    "einstein_a_s", "gauge_difference",
]


def read_li2021_c_i_table() -> dict[tuple[str, str], tuple[float, float]]:
    """Return ``{(lower label, upper label): (gf, dT)}`` of Li et al. (2021).

    MCDHF/RCI E1 data of C I (Li, Amarsi, Papoulia, Ekman & Jonsson 2021,
    MNRAS 502, 3780), with each state written as the Stout C I level label
    (see ``scripts/build_li2021_c_i_table.py``).  ``dT`` is the authors'
    relative difference between the length- and velocity-gauge values.
    """

    import csv
    import lzma
    from importlib.resources import files

    resource = files("wd_spectra").joinpath(LI2021_C_I_TABLE_RESOURCE)
    table = {}
    with resource.open("rb") as raw, lzma.open(raw, "rt", encoding="ascii") as stream:
        reader = csv.reader(line for line in stream if line.strip() and not line.startswith("#"))
        if next(reader) != _LI2021_C_I_COLUMNS:
            raise ValueError("unexpected Li et al. (2021) C I table header")
        for row in reader:
            table[(row[1], row[0])] = (float(row[4]), float(row[6]))
    return table


def atomic_database_with_li2021_c_i_oscillator_strengths(
    database: AtomicDatabase, maximum_gauge_difference: float = 0.2,
) -> AtomicDatabase:
    """Give C I lines the Li et al. (2021) MCDHF oscillator strengths.

    Every C I E1 transition whose two levels are states of the Li et al.
    calculation, with a length-velocity difference ``dT`` up to
    ``maximum_gauge_difference``, takes their ``gf`` (the Einstein A is
    rescaled with it).  This replaces the NIST ASD values, which for the
    2p3s-2p4p array combine Hibbert et al. (1993) singlet values with
    triplet values renormalized by 0.78 (3Po-3P) and 0.71 (3Po-3S).  The
    singlet-to-triplet strength ratio of Li et al. is 0.17-0.24 dex smaller,
    in line with the arc measurements of Jones & Wiese (1984) and with the
    solar C I 4775.9 A line; the warm DQ spectra show the same difference.
    """

    key = ("C", 0)
    ion = database.ions.get(key)
    if ion is None:
        return database
    table = read_li2021_c_i_table()
    levels = {level.index: level for level in ion.levels}
    changed = []
    replaced = 0
    for transition in ion.transitions:
        lower, upper = levels.get(transition.lower_index), levels.get(transition.upper_index)
        entry = (
            None if lower is None or upper is None or transition.transition_type != "E1"
            else table.get((lower.label, upper.label))
        )
        if entry is None or entry[1] > maximum_gauge_difference or entry[0] <= 0.0:
            changed.append(transition)
            continue
        oscillator = entry[0] / lower.statistical_weight
        scale = (
            oscillator / transition.absorption_oscillator_strength
            if transition.absorption_oscillator_strength > 0.0 else 0.0
        )
        einstein_a = (
            transition.einstein_a * scale if scale > 0.0
            else 6.6702e15 * lower.statistical_weight / upper.statistical_weight
            * oscillator / transition.wavelength_vacuum_angstrom**2
        )
        changed.append(replace(
            transition, absorption_oscillator_strength=oscillator, einstein_a=einstein_a,
        ))
        replaced += 1
    if replaced == 0:
        return database
    ions = dict(database.ions)
    ions[key] = replace(ion, transitions=tuple(changed))
    return AtomicDatabase(
        MappingProxyType(ions),
        source=database.source
        + f"; Li et al. (2021) MCDHF C I oscillator strengths for {replaced} lines"
        f" (dT <= {maximum_gauge_difference})",
    )


_C_I_F_LEVEL = re.compile(
    r"^2s2\.2p\.\(2Po<(?P<jc>\d/2)>\)\.(?P<n>\d+)f\.\((?P=jc)\[(?P<K>\d+/2)\]<(?P<J>\d+)>\)$"
)
# Rydberg constant of 12C, cm^-1.
_RYDBERG_CARBON_WAVENUMBER = 109_732.30


def atomic_database_with_carbon_i_f_series(
    database: AtomicDatabase, maximum_principal: int = 8,
) -> AtomicDatabase:
    """Complete the C I 2s2p3 3Do - 2s2 2p nf (n = 5-8) series.

    The Stout list has the 2s2p3 3Do - 4f lines (5013-5043 A, NIST grade C)
    but no line to the higher nf levels, and lacks the nf levels built on
    the C II 2Po1/2 parent, which carry the strongest components.  The
    series is completed with quantum-defect theory and no free parameter:

    * every missing ``2s2 2p(2Po_jc) nf jc[K]_J`` level is placed at
      ``E = I(jc) - R / (n - delta)^2`` with the quantum defect ``delta`` of
      its 4f member and ``I(jc)`` the limit of its parent (C II 2Po1/2 or
      2Po3/2); for the levels Stout lists, this reproduces their energies
      to within 6 cm^-1;
    * each 4f line gives its nf counterparts the oscillator strength
      ``f_n = f_4f (n*_4f / n*_n)^3 (Delta E_n / Delta E_4f)``, the
      ``n*^-3`` normalization of a Rydberg series in the dipole-length form;
    * their electron-impact widths are the median 4f constant of the
      multiplet scaled by ``(n*_n / n*_4f)^4`` (Wiese & Konjevic 1982);
    * the C I jK-coupled levels join the ``n``-manifold of the LS levels of
      the same parent term and of the jK levels of their own parent ``J``,
      so the near-hydrogenic nf levels (and 4f) take the quasi-static
      linear-Stark profile of D6.

    The 2s2p3 3Do - 5f lines (4462-4480 A) produce the broad absorption
    observed at 4430-4540 A in warm and hot DQ white dwarfs.
    """

    key = ("C", 0)
    ion = database.ions.get(key)
    parent_ion = database.ions.get(("C", 1))
    if ion is None or ion.ionization_energy_ev is None or parent_ion is None:
        return database
    limit = ion.ionization_energy_ev * 8065.543937
    parents = sorted(parent_ion.levels, key=lambda level: level.energy_wavenumber)
    parent_limit = {"1/2": limit, "3/2": limit + parents[1].energy_wavenumber}
    f_levels: dict[tuple[int, str, str, str], AtomicLevel] = {}
    for level in ion.levels:
        match = _C_I_F_LEVEL.match(level.label)
        if match is not None:
            f_levels[(int(match["n"]), match["jc"], match["K"], match["J"])] = level
    levels = {level.index: level for level in ion.levels}
    reference = [
        transition for transition in ion.transitions
        if transition.transition_type == "E1"
        and levels[transition.lower_index].label.startswith("2s.2p3.(3Do")
        and _C_I_F_LEVEL.match(levels[transition.upper_index].label) is not None
        and int(_C_I_F_LEVEL.match(levels[transition.upper_index].label)["n"]) == 4
    ]
    if not reference:
        return database
    widths = [
        t.electron_stark_rate_coefficient_cm3_s for t in reference
        if t.electron_stark_rate_coefficient_cm3_s is not None
    ]
    median_width = float(np.median(widths)) if widths else None
    existing = {(t.lower_index, t.upper_index) for t in ion.transitions}
    new_levels: list[AtomicLevel] = []
    new_transitions: list[AtomicTransition] = []
    next_index = max(levels) + 1

    def effective_n(level, jc):
        return float(np.sqrt(_RYDBERG_CARBON_WAVENUMBER / (parent_limit[jc] - level.energy_wavenumber)))

    for transition in reference:
        lower = levels[transition.lower_index]
        match = _C_I_F_LEVEL.match(levels[transition.upper_index].label)
        jc, k_label, j_label = match["jc"], match["K"], match["J"]
        upper_4f = levels[transition.upper_index]
        n_eff_4f = effective_n(upper_4f, jc)
        defect = 4.0 - n_eff_4f
        gap_4f = upper_4f.energy_wavenumber - lower.energy_wavenumber
        for principal in range(5, maximum_principal + 1):
            level = f_levels.get((principal, jc, k_label, j_label))
            if level is None:
                level = AtomicLevel(
                    next_index,
                    parent_limit[jc] - _RYDBERG_CARBON_WAVENUMBER / (principal - defect) ** 2,
                    upper_4f.statistical_weight,
                    upper_4f.label.replace(".4f.", f".{principal}f."),
                )
                next_index += 1
                f_levels[(principal, jc, k_label, j_label)] = level
                new_levels.append(level)
            if (lower.index, level.index) in existing:
                continue
            gap = level.energy_wavenumber - lower.energy_wavenumber
            n_eff = effective_n(level, jc)
            oscillator = (
                transition.absorption_oscillator_strength
                * (n_eff_4f / n_eff) ** 3 * gap / gap_4f
            )
            wavelength = 1.0e8 / gap
            new_transitions.append(AtomicTransition(
                lower.index, level.index,
                6.6702e15 * lower.statistical_weight / level.statistical_weight
                * oscillator / wavelength**2,
                "E1", wavelength, oscillator,
                electron_stark_rate_coefficient_cm3_s=(
                    None if median_width is None else median_width * (n_eff / n_eff_4f) ** 4
                ),
            ))
            existing.add((lower.index, level.index))
    ions = dict(database.ions)
    ions[key] = replace(
        ion,
        levels=ion.levels + tuple(new_levels),
        transitions=ion.transitions + tuple(new_transitions),
        rydberg_jk_levels_join_ls_manifold=True,
    )
    return AtomicDatabase(
        MappingProxyType(ions),
        source=database.source
        + f"; C I 2s2p3 3Do - nf (n = 5-{maximum_principal}) series from quantum-defect"
        f" scaling of NIST 4f lines ({len(new_transitions)} lines, {len(new_levels)} levels)",
    )


def atomic_database_with_carbon_i_nearest_level_stark(
    database: AtomicDatabase,
) -> AtomicDatabase:
    """Give C I lines Freudenstein & Cooper (1978) electron Stark widths.

    Every E1 line of C I whose upper level is a single-electron LS term
    (``2s2 2p nl``) and is not a near-hydrogenic Rydberg level (which keeps
    its quasi-static manifold profile) receives the nearest-level width of
    :func:`wd_spectra.neutral_stark.nearest_level_stark_width` at 10^4 K,
    with the power-law temperature exponent between 10^4 and 2 x 10^4 K.
    The tabulated Kurucz constants of the measured C I multiplets are 0.1-0.9
    times the laboratory widths (median 0.72, scatter 0.39 dex); this
    estimate gives a median of 1.34 and a scatter of 0.17 dex, in line with
    the 25 per cent overestimate FC78 report for the nearest-level
    approximation.  Lines to high upper levels change most: the 3s-5p 3D
    multiplet at 4065 A has a Kurucz constant of 6e-6 cm^3 s^-1 against
    about 1e-4 here.  Measured multiplets are replaced afterwards by
    :func:`atomic_database_with_carbon_stark`.
    """

    from .metals import linear_stark_rydberg_level
    from .neutral_stark import ion_terms, nearest_level_stark_width

    key = ("C", 0)
    ion = database.ions.get(key)
    if ion is None:
        return database
    terms = ion_terms(ion)
    levels = {level.index: level for level in ion.levels}
    cache: dict[str, tuple[float, float]] = {}
    changed = []
    replaced = 0
    for transition in ion.transitions:
        upper_level = levels.get(transition.upper_index)
        term = (
            None if upper_level is None or transition.transition_type != "E1"
            else terms.get(_level_term_label(upper_level.label))
        )
        if (
            term is None or term.L is None
            or linear_stark_rydberg_level(database, ion, upper_level) is not None
        ):
            changed.append(transition)
            continue
        if term.label not in cache:
            cold = nearest_level_stark_width(terms, term, 1.0e4)
            warm = nearest_level_stark_width(terms, term, 2.0e4)
            cache[term.label] = (
                cold, float(np.log(warm / cold) / np.log(2.0)) if cold > 0.0 else 0.0
            )
        width, exponent = cache[term.label]
        if width <= 0.0:
            changed.append(transition)
            continue
        changed.append(replace(
            transition,
            electron_stark_rate_coefficient_cm3_s=width,
            electron_stark_temperature_exponent=exponent,
        ))
        replaced += 1
    if replaced == 0:
        return database
    ions = dict(database.ions)
    ions[key] = replace(ion, transitions=tuple(changed))
    return AtomicDatabase(
        MappingProxyType(ions),
        source=database.source
        + f"; Freudenstein & Cooper (1978) C I Stark widths for {replaced} lines",
    )


class MagneticBulkSynthesizer:
    """Magnetic Stokes-I synthesis of one bulk-mixture structure.

    Construction evaluates everything that does not depend on the field: the
    populations, the continuum, the parent line list and the zero-field
    profiles of the Rydberg lines.  :meth:`spectrum` then integrates any
    surface map; the line manifolds and scattering source of each field
    strength are computed once and kept in ``field_cache`` (keyed by the
    field in MG), so that many maps of the same star are cheap.

    Every distinct field strength gets its own line opacity:

    * metal lines are split by :func:`wd_spectra.magnetic_metals.
      zeeman_split_database` (LS Paschen--Back with the first-order
      diamagnetic term) and evaluated with the ordinary line-opacity routine,
      so each component keeps the Stark and Doppler profile of its multiplet;
    * Rydberg lines that carry a quasi-static ionic Stark profile keep that
      zero-field profile, translated to the three components of a normal
      triplet (their fine structure is far below ``mu_B B``), as the DAH
      Balmer lines are; no simultaneous Stark--Zeeman problem is solved;
    * Balmer lines use the DAH manifolds (normal triplets at B <= 1 MG, the
      H2db components above);
    * the continuum, the other hydrogen series, the helium lines and the
      level populations are those of the nonmagnetic atmosphere.

    Each cell contributes one ray with the scalar transfer equation for
    ``eta_I = kappa_pi sin^2(psi)/2 + (kappa_- + kappa_+)(1 + cos^2 psi)/4``;
    coherent scattering uses the angle-averaged opacity of the cell.  The
    magneto-optical terms and the polarized Stokes components are not solved.
    """

    def __init__(
        self,
        atmosphere,
        wavelength_angstrom,
        atomic_database: AtomicDatabase,
        photoionization_database,
        log_number_abundance,
        *,
        reference_element: str = "C",
        topbase_photoionization_database=None,
        h2db_transitions=None,
        maximum_field_megagauss: float | None = None,
        minimum_metal_oscillator_strength: float = 1.0e-6,
        maximum_metal_lines: int | None = 25_000,
        metal_series_pseudocontinuum_elements=("C",),
        include_linear_stark_quasistatic: bool = True,
        include_diamagnetic: bool = True,
        microturbulent_velocity_kms: float = 0.0,
        profile_edge_optical_depth: float | None = 1.0e-3,
        line_wavelength_margin_angstrom: float = 400.0,
        n_angle: int = 4,
        field_cache: dict | None = None,
        helium_line_tables=None,
    ) -> None:
        from ._spectrum_source import solve_spectrum_source
        from .d6 import (
            _metal_opacity,
            atmosphere_with_bulk_metal_state,
            bulk_metal_lte_state,
            _bulk_helium_line_opacity,
            split_bulk_light_abundances,
        )
        from .magnetic import BalmerLineTemplate, MagneticPhysics, _validate_grid
        from .metals import (
            linear_stark_mixing_fraction,
            linear_stark_rydberg_level,
            metal_ionic_microfield_perturber_density,
            selected_metal_lines,
        )
        from .opacity import optical_depth_from_mass_opacity
        from .spectrum import planck_lambda_angstrom

        wavelength = _validate_grid(wavelength_angstrom)
        elements = tuple(metal_series_pseudocontinuum_elements)
        state = bulk_metal_lte_state(
            atmosphere, atomic_database, log_number_abundance,
            reference_element=reference_element,
        )
        atmosphere = atmosphere_with_bulk_metal_state(atmosphere, state)
        self.atmosphere = atmosphere
        self.state = state
        self.wavelength = wavelength
        self.atomic_database = atomic_database
        self.photoionization_database = photoionization_database
        self.hydrogen = state.trace_hydrogen_state is not None
        self.helium = state.trace_helium_state is not None
        self.include_diamagnetic = bool(include_diamagnetic)
        self.minimum_metal_oscillator_strength = float(minimum_metal_oscillator_strength)
        self.n_angle = int(n_angle)
        self.field_cache = {} if field_cache is None else field_cache
        options = dict(
            microturbulent_velocity_kms=microturbulent_velocity_kms,
            include_rydberg_dissolution=True,
            include_metal_series_pseudocontinuum=True,
            metal_series_pseudocontinuum_elements=elements,
            topbase_photoionization_database=topbase_photoionization_database,
            include_linear_stark_quasistatic=include_linear_stark_quasistatic,
            profile_edge_optical_depth=profile_edge_optical_depth,
        )
        self._ordinary_options = {**options, "include_linear_stark_quasistatic": False}
        continuum, scattering = _metal_opacity(
            atmosphere, wavelength, atomic_database, state, photoionization_database,
            minimum_oscillator_strength=minimum_metal_oscillator_strength,
            maximum_lines=0, include_lines=False, **options,
        )
        self.continuum = continuum
        self.scattering = np.ascontiguousarray(
            np.broadcast_to(scattering, continuum.shape)
        )
        if self.hydrogen:
            # Hydrogen series without magnetic data stay unsplit.
            from .opacity import (
                brackett_mass_absorption_coefficient,
                lyman_mass_absorption_coefficient,
                paschen_mass_absorption_coefficient,
            )

            self._unsplit_hydrogen = (
                lyman_mass_absorption_coefficient(atmosphere, wavelength)
                + paschen_mass_absorption_coefficient(atmosphere, wavelength)
                + brackett_mass_absorption_coefficient(atmosphere, wavelength)
            )
        else:
            self._unsplit_hydrogen = None
        if self.helium:
            # Helium lines without magnetic data stay unsplit.
            self._unsplit_helium = _bulk_helium_line_opacity(
                atmosphere, wavelength, helium_line_tables
            )
        else:
            self._unsplit_helium = None

        # One fixed parent line list for every field.
        metal_abundance = split_bulk_light_abundances(log_number_abundance)[0]
        margin = float(line_wavelength_margin_angstrom)
        self._window = (max(wavelength[0] - margin, 1.0), wavelength[-1] + margin)
        line_forming = (atmosphere.rosseland_optical_depth >= 1.0e-4) & (
            atmosphere.rosseland_optical_depth <= 30.0
        )
        if not np.any(line_forming):
            line_forming = np.ones(atmosphere.n_depth, dtype=bool)
        ion_stage_weight = {
            (element, charge): float(np.max(
                populations[charge, line_forming]
                / np.maximum(
                    state.element_number_density[element][line_forming],
                    np.finfo(np.float64).tiny,
                )
            ))
            for element, populations in state.ion_number_density.items()
            for charge in range(populations.shape[0])
        }
        parents = selected_metal_lines(
            atomic_database, metal_abundance, self._window[0], self._window[1],
            minimum_metal_oscillator_strength, maximum_metal_lines,
            atmosphere.effective_temperature, ion_stage_weight, flux_weighted=True,
        )
        # Rydberg lines with a quasi-static Stark profile (the criterion of
        # the line-opacity routine) are translated; all others diagonalized.
        microfield_scale = np.maximum(
            metal_ionic_microfield_perturber_density(state), 0.0
        ) ** (2.0 / 3.0)
        ordinary_keys = []
        rydberg_keys = []
        for ion, line in parents:
            key = (ion.element, ion.charge, line.lower_index, line.upper_index)
            quasistatic = False
            if include_linear_stark_quasistatic:
                upper = next(
                    level for level in ion.levels if level.index == line.upper_index
                )
                rydberg = linear_stark_rydberg_level(atomic_database, ion, upper)
                quasistatic = rydberg is not None and float(np.max(
                    linear_stark_mixing_fraction(
                        rydberg[2], rydberg[3], ion.charge + 1.0, microfield_scale
                    )
                )) > 1.0e-3
            (rydberg_keys if quasistatic else ordinary_keys).append(key)
        self.parent_keys = tuple(ordinary_keys)
        self.rydberg_keys = tuple(rydberg_keys)
        self._rydberg_template = None
        if rydberg_keys:
            step = float(np.median(np.diff(wavelength)))
            pad = np.arange(step, 0.5 * margin + step, step)
            extended = np.concatenate((
                wavelength[0] - pad[::-1], wavelength, wavelength[-1] + pad
            ))
            extended = extended[extended > 0.0]
            rydberg_opacity, _ = _metal_opacity(
                atmosphere, extended, atomic_database, state, photoionization_database,
                minimum_oscillator_strength=minimum_metal_oscillator_strength,
                maximum_lines=len(rydberg_keys), include_lines=True,
                line_transition_keys=self.rydberg_keys, metal_lines_only=True,
                **options,
            )
            self._rydberg_template = BalmerLineTemplate(extended, rydberg_opacity)

        self._physics = MagneticPhysics(
            transitions=h2db_transitions,
            energies=None,
            include_rwa_photoionization=False,
            include_magnetic_eos=False,
            include_centered_motion=False,
        )
        # A star's Balmer regime follows its maximum visible field, as in DAH.
        self._maximum_field = maximum_field_megagauss
        self._template_cache: dict = {}
        self.planck = np.ascontiguousarray(planck_lambda_angstrom(
            wavelength[:, np.newaxis], atmosphere.temperature[np.newaxis, :]
        ))
        depth = optical_depth_from_mass_opacity(
            atmosphere.column_mass, self.continuum + self.scattering
        )
        _, radiation, _ = solve_spectrum_source(
            depth, self.planck, self.continuum, self.scattering,
            wavelength=wavelength, n_angle=self.n_angle, discretization="formal-linear",
        )
        self._continuum_mean_intensity = radiation.mean_intensity

    def _regime(self, cells) -> str:
        maximum = (
            float(cells.maximum_field_megagauss)
            if self._maximum_field is None else float(self._maximum_field)
        )
        return self._physics.regime(maximum)

    def manifolds(self, field_megagauss: float, regime: str):
        """Line opacity of the three polarizations at one field strength."""

        from .d6 import _metal_opacity
        from .magnetic import (
            LINEAR_ZEEMAN_FREQUENCY_HZ_PER_GAUSS,
            PolarizedOpacity,
            _line_manifolds,
            _translated_profile,
        )
        from .magnetic_metals import zeeman_split_database

        field = float(field_megagauss)
        split = zeeman_split_database(
            self.atomic_database, field,
            wavelength_range_angstrom=self._window,
            minimum_oscillator_strength=self.minimum_metal_oscillator_strength,
            line_transition_keys=self.parent_keys,
            include_diamagnetic=self.include_diamagnetic,
        )
        metal = {}
        larmor = LINEAR_ZEEMAN_FREQUENCY_HZ_PER_GAUSS * field * 1.0e6
        for q in (-1, 0, 1):
            lines, _ = _metal_opacity(
                self.atmosphere, self.wavelength, split[q], self.state,
                self.photoionization_database,
                minimum_oscillator_strength=1.0e-12,
                maximum_lines=None, include_lines=True, metal_lines_only=True,
                **self._ordinary_options,
            )
            if self._rydberg_template is not None:
                lines = lines + _translated_profile(
                    self._rydberg_template, self.wavelength, q * larmor
                )
            metal[q] = lines
        result = PolarizedOpacity(metal[-1], metal[0], metal[1])
        if self.hydrogen:
            balmer, _ = _line_manifolds(
                self.atmosphere, self.wavelength, field, self._physics, regime,
                self._template_cache, False,
            )
            result = result + balmer + PolarizedOpacity.unpolarized(
                self._unsplit_hydrogen
            )
        if self.helium:
            result = result + PolarizedOpacity.unpolarized(self._unsplit_helium)
        return result

    def field_opacity(self, field_megagauss: float, regime: str):
        """Return (manifolds, scattering mean intensity) of one field, cached."""

        from ._spectrum_source import solve_spectrum_source
        from .opacity import optical_depth_from_mass_opacity

        key = (regime, round(float(field_megagauss), 9))
        if key not in self.field_cache:
            lines = self.manifolds(key[1], regime)
            thermal = self.continuum + lines.isotropic()
            depth = optical_depth_from_mass_opacity(
                self.atmosphere.column_mass, thermal + self.scattering
            )
            _, radiation, _ = solve_spectrum_source(
                depth, self.planck, thermal, self.scattering,
                wavelength=self.wavelength, n_angle=self.n_angle,
                discretization="formal-linear",
            )
            self.field_cache[key] = (lines, radiation.mean_intensity)
        return self.field_cache[key]

    def spectrum(self, cells, *, progress=None):
        """Disk-integrated flux and line-free continuum flux of a surface map."""

        from .constants import PI
        from .magnetic import MAXIMUM_H2DB_BALMER_UPPER_LEVEL, _validate_surface_cells
        from .opacity import optical_depth_from_mass_opacity
        from .radiative_transfer import emergent_specific_intensity
        from .spectrum import Spectrum

        _validate_surface_cells(cells)
        regime = self._regime(cells)
        column_mass = self.atmosphere.column_mass
        continuum, scattering, planck = self.continuum, self.scattering, self.planck
        flux = np.zeros(self.wavelength.size)
        continuum_flux = np.zeros(self.wavelength.size)
        continuum_extinction = continuum + scattering
        continuum_depth = optical_depth_from_mass_opacity(column_mass, continuum_extinction)
        continuum_source = (
            continuum * planck + scattering * self._continuum_mean_intensity
        ) / continuum_extinction
        continuum_intensity: dict[float, np.ndarray] = {}
        n_cells = cells.projected_weight.size
        for cell in range(n_cells):
            field = float(cells.field_strength_megagauss[cell])
            cosine = float(cells.field_ray_cosine[cell])
            mu = float(cells.ray_mu[cell])
            weight = float(cells.projected_weight[cell])
            lines, mean_intensity = self.field_opacity(field, regime)
            eta_i = continuum + lines.stokes(cosine)[0]
            extinction = eta_i + scattering
            source = (eta_i * planck + scattering * mean_intensity) / extinction
            flux += weight * PI * emergent_specific_intensity(
                optical_depth_from_mass_opacity(column_mass, extinction), source, mu
            )
            if mu not in continuum_intensity:
                continuum_intensity[mu] = emergent_specific_intensity(
                    continuum_depth, continuum_source, mu
                )
            continuum_flux += weight * PI * continuum_intensity[mu]
            if callable(progress):
                progress(cell + 1, n_cells)
        return Spectrum(
            self.wavelength,
            flux,
            {
                "wavelength_medium": "vacuum",
                "flux_convention": "surface F_lambda (projected-area sum of Stokes I)",
                "flux_unit": "erg s^-1 cm^-2 Angstrom^-1",
                "metal_lines": self.atomic_database.source,
                "metal_line_splitting": (
                    "LS Paschen-Back diagonalization per multiplet"
                    + (
                        " with first-order diamagnetic term"
                        if self.include_diamagnetic else ""
                    )
                    + "; linear J-level patterns for lines outside LS coupling"
                ),
                "metal_parent_lines": len(self.parent_keys),
                "metal_rydberg_lines_translated": len(self.rydberg_keys),
                "balmer_lines": (
                    "absent" if not self.hydrogen
                    else "normal linear-Zeeman triplets" if regime == "linear-zeeman"
                    else f"H2db components through H{MAXIMUM_H2DB_BALMER_UPPER_LEVEL}"
                ),
                "other_hydrogen_series": "unsplit" if self.hydrogen else "absent",
                "helium_lines": "unsplit" if self.helium else "absent",
                "continuum": "zero-field bound-free, free-free and scattering",
                "populations": "zero-field LTE on the supplied structure",
                "polarized_transfer": "scalar Stokes I with ray-specific pi/sigma opacity",
                "surface_cells": int(n_cells),
                "surface_field_megagauss": cells.field_strength_megagauss.tolist(),
                "surface_projected_weight": cells.projected_weight.tolist(),
                "continuum_flux": "same cells without bound-bound opacity",
            },
            continuum_flux,
        )


def synthesize_magnetic_bulk_spectrum(
    atmosphere,
    wavelength_angstrom,
    atomic_database: AtomicDatabase,
    photoionization_database,
    log_number_abundance,
    cells,
    *,
    progress=None,
    **options,
):
    """Disk-integrated Stokes-I flux of a magnetic carbon-rich atmosphere.

    One-shot form of :class:`MagneticBulkSynthesizer`; ``options`` are its
    keyword arguments.  Without a caller's ``field_cache`` only the opacity
    of the current field strength is kept in memory.
    """

    persistent = options.get("field_cache") is not None
    synthesizer = MagneticBulkSynthesizer(
        atmosphere, wavelength_angstrom, atomic_database, photoionization_database,
        log_number_abundance, **options,
    )
    if persistent:
        return synthesizer.spectrum(cells, progress=progress)

    # Cells arrive grouped by field; drop each field's opacity after use.
    def release(done: int, total: int) -> None:
        if done < total:
            following = round(float(cells.field_strength_megagauss[done]), 9)
            for key in [key for key in synthesizer.field_cache if key[1] != following]:
                del synthesizer.field_cache[key]
        if callable(progress):
            progress(done, total)

    return synthesizer.spectrum(cells, progress=release)
