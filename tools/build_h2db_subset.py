#!/usr/bin/env python3
"""Build the bundled H2db subset used by the DAH model.

Input is the extracted public H2db archive of Schimeczek & Wunner (DaRUS,
doi:10.18419/DARUS-2118, CC BY 4.0): ``transitions.7z`` and ``energies.7z``
unpacked into ``ROOT/transitions`` and ``ROOT/energies``.  The output keeps

* every Balmer transition (lower level n=2, upper n=3..22 identified from the
  near-zero-field row),
* the intra-shell n=2 -> 2 transitions (2s0 -> 2p+1, 2p-1 -> 2s0, 2p0 -> 2s0;
  near-zero transition energy), stored with upper level 2, and
* every stationary-state energy track,

for beta = B/B0 <= ``--maximum-beta`` (B0 = 4.70103e9 G), plus the first row
beyond it so linear interpolation reaches the limit.  Values are copied
without modification; labels are the archive's own directory labels.  The
reader in ``wd_spectra.magnetic_atomic`` applies the physical conventions
(sign of m, mirrored Delta m = 0 partners) and documents them.

``--paschen-brackett`` writes the separate file of the opt-in Paschen and
Brackett components instead: lower level n=3, 4 and upper n'=n+1..22 (same
rule; same-shell n -> n transitions are not kept), beta <= 0.25 (B <= 1175
MG), only transitions whose isotropic oscillator strength f = E d / 3 reaches
1e-3 at a centre inside the DAH window 3400-8000 A at some kept row, after
rows are thinned so that linear interpolation in beta reproduces every
dropped row to 1e-5 (relative) in E and 1e-3 in d.  Labels carry the lower
level as an eighth column; the energy tracks stay in the Balmer file.

Usage::

    python tools/build_h2db_subset.py /path/to/h2db \
        src/wd_spectra/data/runtime/cache/h2db/h2db_balmer_subset.npz
    python tools/build_h2db_subset.py --paschen-brackett --maximum-beta 0.25 /path/to/h2db \
        src/wd_spectra/data/runtime/cache/h2db/h2db_paschen_brackett_subset.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np

_TRANSITION = re.compile(
    r"m_([+-]?\d+)_to_([+-]?\d+)/pi_z_([+-]1)_to_([+-]1)/nu_(\d+)_to_(\d+)\.tsv$"
)
_ENERGY = re.compile(r"m_(-?\d+)/pi_z_([+-]1)/nu_(\d+)\.tsv$")


def _table(path: Path) -> np.ndarray:
    table = np.genfromtxt(
        path, comments="#", dtype=np.float64, missing_values="n/a",
        filling_values=np.nan,
    )
    return table[np.newaxis, :] if table.ndim == 1 else table


def _truncate(table: np.ndarray, maximum_beta: float) -> np.ndarray:
    beyond = np.flatnonzero(table[:, 0] > maximum_beta)
    return table if beyond.size == 0 else table[: beyond[0] + 1]


_WINDOW_ANGSTROM = (3_400.0, 8_000.0)
_MINIMUM_OSCILLATOR_STRENGTH = 1.0e-3
_THINNING_TOLERANCE = (1.0e-5, 1.0e-3)  # relative, in E and in d
# Infinite-mass archive energies to vacuum wavelength with the hydrogen
# Rydberg 109678.77 cm^-1 (the package's line centres).
_RYDBERG_WAVELENGTH_ANGSTROM = 1.0e8 / 109_678.77


def _balmer_upper_level(
    transition_energy: float, initial_energy: float, lower_level: int = 2
) -> int | None:
    # Upper level 2 is the intra-shell n=2 -> 2 group (zero field-free energy).
    if abs(initial_energy + 1.0 / lower_level**2) > 3.0e-3:
        return None
    levels = np.arange(2 if lower_level == 2 else lower_level + 1, 23)
    expected = 1.0 / lower_level**2 - 1.0 / levels.astype(float) ** 2
    index = int(np.argmin(abs(expected - transition_energy)))
    if abs(expected[index] - transition_energy) >= 2.5e-3:
        return None
    return int(levels[index])


def _in_window(table: np.ndarray) -> bool:
    energy, dipole = table[:, 1], table[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        wavelength = _RYDBERG_WAVELENGTH_ANGSTROM / energy
    return bool(np.any(
        (energy * dipole / 3.0 >= _MINIMUM_OSCILLATOR_STRENGTH)
        & (wavelength >= _WINDOW_ANGSTROM[0]) & (wavelength <= _WINDOW_ANGSTROM[1])
    ))


def _thin(table: np.ndarray) -> np.ndarray:
    """Greedily drop rows that linear interpolation in beta reproduces."""

    beta, energy, dipole = table[:, 0], table[:, 1], table[:, 2]
    kept, start = [0], 0
    while start < len(table) - 1:
        end = start + 1
        while end + 1 < len(table):
            inner = np.arange(start + 1, end + 1)
            t = (beta[inner] - beta[start]) / (beta[end + 1] - beta[start])
            if not (
                np.all(abs(energy[start] + t * (energy[end + 1] - energy[start]) - energy[inner])
                       <= _THINNING_TOLERANCE[0] * abs(energy[inner]))
                and np.all(abs(dipole[start] + t * (dipole[end + 1] - dipole[start]) - dipole[inner])
                           <= _THINNING_TOLERANCE[1] * np.maximum(abs(dipole[inner]), 1.0e-6))
            ):
                break
            end += 1
        kept.append(end)
        start = end
    return table[kept]


def build(root: Path, output: Path, maximum_beta: float, paschen_brackett: bool = False) -> None:
    labels, offsets, columns = [], [0], []
    for path in sorted((root / "transitions").rglob("*.tsv")):
        match = _TRANSITION.search(path.as_posix())
        if match is None:
            continue
        table = _table(path)
        finite = np.isfinite(table[:, 0]) & np.isfinite(table[:, 1]) & np.isfinite(table[:, 3])
        table = table[finite]
        if table.shape[0] < 2:
            continue
        for lower in (3, 4) if paschen_brackett else (2,):
            upper = _balmer_upper_level(table[0, 1], table[0, 3], lower)
            if upper is not None:
                break
        if upper is None:
            continue
        table = _truncate(table, maximum_beta)
        if paschen_brackett:
            table = _thin(table)
            if not _in_window(table):
                continue
        labels.append([int(value) for value in match.groups()] + [upper] + ([lower] if paschen_brackett else []))
        columns.append(table[:, :4])
        offsets.append(offsets[-1] + table.shape[0])
    transitions = np.concatenate(columns)
    if paschen_brackett:
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            transition_labels=np.asarray(labels, dtype=np.int16),
            transition_offsets=np.asarray(offsets, dtype=np.int64),
            transition_beta=transitions[:, 0],
            transition_energy_rydberg=transitions[:, 1].astype(np.float32),
            transition_dipole_strength=transitions[:, 2].astype(np.float32),
            transition_initial_energy_rydberg=transitions[:, 3].astype(np.float32),
            maximum_beta=np.float64(maximum_beta),
            reference_field_gauss=np.float64(4.70103e9),
            window_angstrom=np.asarray(_WINDOW_ANGSTROM),
            minimum_oscillator_strength=np.float64(_MINIMUM_OSCILLATOR_STRENGTH),
            thinning_tolerance_energy_dipole=np.asarray(_THINNING_TOLERANCE),
            source=np.str_(
                "Schimeczek & Wunner H2db, DaRUS doi:10.18419/DARUS-2118, CC BY 4.0"
            ),
        )
        print(
            f"{len(labels)} Paschen and Brackett transitions ({transitions.shape[0]} rows) -> "
            f"{output} ({output.stat().st_size / 1e6:.1f} MB)"
        )
        return

    energy_labels, energy_offsets, energy_columns = [], [0], []
    for path in sorted((root / "energies").rglob("*.tsv")):
        match = _ENERGY.search(path.as_posix())
        if match is None:
            continue
        table = _table(path)
        table = table[np.isfinite(table[:, 0]) & np.isfinite(table[:, 1])]
        table = _truncate(table, maximum_beta)
        energy_labels.append([int(value) for value in match.groups()])
        energy_columns.append(table[:, :2])
        energy_offsets.append(energy_offsets[-1] + table.shape[0])
    energies = np.concatenate(energy_columns)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        transition_labels=np.asarray(labels, dtype=np.int16),
        transition_offsets=np.asarray(offsets, dtype=np.int64),
        transition_beta=transitions[:, 0],
        transition_energy_rydberg=transitions[:, 1].astype(np.float32),
        transition_dipole_strength=transitions[:, 2].astype(np.float32),
        transition_initial_energy_rydberg=transitions[:, 3].astype(np.float32),
        energy_labels=np.asarray(energy_labels, dtype=np.int16),
        energy_offsets=np.asarray(energy_offsets, dtype=np.int64),
        energy_beta=energies[:, 0],
        energy_rydberg=energies[:, 1],
        maximum_beta=np.float64(maximum_beta),
        reference_field_gauss=np.float64(4.70103e9),
        source=np.str_(
            "Schimeczek & Wunner H2db, DaRUS doi:10.18419/DARUS-2118, CC BY 4.0"
        ),
    )
    print(
        f"{len(labels)} Balmer and n=2->2 transitions ({transitions.shape[0]} rows), "
        f"{len(energy_labels)} energy tracks ({energies.shape[0]} rows) -> {output} "
        f"({output.stat().st_size / 1e6:.1f} MB)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--maximum-beta", type=float, default=3.0)
    parser.add_argument("--paschen-brackett", action="store_true")
    args = parser.parse_args()
    build(args.root, args.output, args.maximum_beta, args.paschen_brackett)


if __name__ == "__main__":
    main()
