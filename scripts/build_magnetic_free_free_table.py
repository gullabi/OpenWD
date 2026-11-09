"""Build the magnetic free-free Coulomb-logarithm table used by the cyclotron term.

Writes ``src/wd_spectra/data/gaunt/magnetic_free_free_coulomb_log.npz``: the ratio
``Lambda_alpha(beta, u) / Lambda_cl(u)`` of the first-Born magnetic Coulomb
logarithm of electron-proton free-free absorption to its zero-field Born value,
for the longitudinal (``alpha = 0``) and transverse (``alpha = +-1``) modes, on
a grid of ``log10 beta`` (``beta = hbar omega_c / kT``) and
``log10 (omega / omega_c)``.

Lambda_alpha is Eq. (41) of Potekhin & Chabrier (2003, ApJ 585, 955;
arXiv:astro-ph/0212062), which corrects a misprint in Eq. (27) of Pavlov &
Panov (1976, Sov. Phys. JETP 44, 300): nonrelativistic electrons in Boltzmann
equilibrium over the Landau levels (both spins), a fixed Coulomb centre, first
Born approximation.  ``Lambda_cl = exp(u/2) K0(u/2)`` is its B -> 0 limit
(PC03 Eq. 40).  It depends on ``beta`` and ``u = hbar omega / kT`` only.
Checked against Fig. 1 of Potekhin (2010, A&A 518, A24) at beta = 5 (median
ratio 1.0008, worst 0.009 dex) and against the B -> 0 limit (1e-3 at beta =
0.004).  About 10 minutes on one core.

usage: python scripts/build_magnetic_free_free_table.py [OUT.npz]
"""

import sys
from pathlib import Path

import numpy as np
from scipy.special import k0e, k1e

LOG10_BETA = np.linspace(np.log10(0.05), 1.0, 26)
# No grid point falls on an integer harmonic, where Lambda_perp has an
# integrable logarithmic peak.
LOG10_FREQUENCY_RATIO = np.linspace(-1.0, 1.0, 240)


def zero_field_born_coulomb_log(u: float) -> float:
    """PC03 Eq. (40): exp(u/2) K0(u/2)."""

    return float(k0e(0.5 * u))


def magnetic_coulomb_log(u: float, beta: float, transverse: bool, samples: int = 3000) -> float:
    """PC03 Eq. (41), summed over Landau-level changes n with |n| <~ 60/beta."""

    nmax = int(np.ceil(60.0 / beta + 2.0 * u / beta)) + 5
    n = np.arange(-nmax, nmax + 1)[:, np.newaxis]
    an = np.abs(n)
    s = np.linspace(-40.0, np.log(1e6 * max(1.0, beta) / min(1.0, beta)), samples)[np.newaxis, :]
    y = np.exp(s)
    theta = 1.0 / np.tanh(0.5 * beta)
    zeta = np.sqrt(1.0 + 2.0 * theta * y + y * y)
    x = np.maximum(np.abs(u - n * beta) * np.sqrt(0.25 + y / beta), 1e-300)
    log_sinh = 0.5 * beta + np.log1p(-np.exp(-beta)) - np.log(2.0)
    log_power = -an * (np.log(y + theta + zeta) + log_sinh)
    if transverse:
        a = (y + theta + an * zeta) / zeta**2 * k0e(x)
    else:
        a = x * k1e(x) / (y + 0.25 * beta)
    q = (y / zeta) * np.exp(np.log(np.maximum(a, 1e-300)) - x + log_power + 0.5 * u)
    return 0.75 * float(np.trapezoid(q * y, s, axis=1).sum())


def build(path: Path) -> None:
    ratio = np.zeros((2, LOG10_BETA.size, LOG10_FREQUENCY_RATIO.size))
    for i, log_beta in enumerate(LOG10_BETA):
        beta = 10.0**log_beta
        for j, log_x in enumerate(LOG10_FREQUENCY_RATIO):
            u = beta * 10.0**log_x
            classical = zero_field_born_coulomb_log(u)
            ratio[0, i, j] = magnetic_coulomb_log(u, beta, False) / classical
            ratio[1, i, j] = magnetic_coulomb_log(u, beta, True) / classical
        print(f"{i + 1}/{LOG10_BETA.size}", flush=True)
    np.savez(
        path,
        log10_beta=LOG10_BETA,
        log10_frequency_ratio=LOG10_FREQUENCY_RATIO,
        ratio=ratio,
    )


if __name__ == "__main__":
    default = (
        Path(__file__).resolve().parents[1]
        / "src/wd_spectra/data/gaunt/magnetic_free_free_coulomb_log.npz"
    )
    build(Path(sys.argv[1]) if len(sys.argv) > 1 else default)
