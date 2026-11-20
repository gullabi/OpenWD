#!/usr/bin/env python3
"""Build the motional Stark table of the H-alpha Zeeman components (n = 2 -> 3).

An atom moving with v_perp across B feels the electric field F = v_perp B / c
in its rest frame. In a strong field the levels are non-degenerate, so the
shift is quadratic, dE = -alpha_perp F^2 / 2, and a thermal (two-dimensional
Maxwellian) v_perp gives each component a one-sided exponential profile with
mean shift s = Delta alpha kT beta^2 / M in atomic units, Delta alpha =
alpha_up - alpha_low, beta = B / 2.350518e9 G, M = m_p + m_e (Pavlov &
Meszaros 1993, ApJ 416, 752, Eqs. 2.12b, 2.23, 3.27-3.31; Rosato 2023, JQSRT
306, 108628, Eq. 3). This script tabulates Delta alpha for the fifteen H2db
H-alpha components on 50-650 MG every 5 MG with the clean-room solver
``hydrogen_magnetic.py`` (textbook Hamiltonian, infinite nuclear mass).

Delta alpha is NaN, and the zero-field Stark kernel is kept, where
  * a level lies above the lowest Landau threshold of an m +- 1 block: the
    motional field couples it to a continuum, there is no second-order shift
    (``flag_continuum``);
  * second order fails for a level of the pair: exact diagonalisation of
    H_B + F x (150 lowest states per block, m = -6..5) at F^2 = 3 <F^2> and
    20000 K moves the level by more than 20 % from second order or keeps less
    than 0.8 of it in one eigenstate; evaluated every 25 MG and taken from the
    nearest such field (``flag_second_order``);
  * the upper level lies below the lower one (``flag_negative``).

Labels are H2db's (lower m, z-parity, nu, upper m, z-parity, nu) with the
physical sign of m, as stored by ``read_h2db_transition_database``.

Usage::

    python tools/build_motional_stark_table.py \
        src/wd_spectra/data/runtime/cache/motional-stark/halpha_motional_stark.npz

About 25 core-minutes.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hydrogen_magnetic as hm  # noqa: E402

FIELDS = np.arange(50.0, 651.0, 5.0)
SECOND_ORDER_FIELDS, SECOND_ORDER_TEMPERATURE, TOLERANCE = np.arange(50.0, 651.0, 25.0), 20000.0, 0.2
MESH = dict(N=60, h=0.6, lmax=60)
M_ATOM = hm.M_P + 1.0


def states(nmax):
    """{(n, l, m): (m, z-parity, k)} of every state with n <= nmax (k = H2db nu - 1)."""
    out = {}
    for m in range(-nmax + 1, nmax):
        for par in (1, -1):
            for k, (n, l) in enumerate(hm.labels(m, par, nmax)):
                out[n, l, m] = (m, par, k)
    return out


def components():
    """H-alpha (lower, upper) state pairs allowed in electric-dipole order."""
    st = states(3)
    low = [v for (n, _, _), v in st.items() if n == 2]
    up = [v for (n, _, _), v in st.items() if n == 3]
    return [(a, b) for a in low for b in up
            if abs(b[0] - a[0]) == 1 and b[1] == a[1] or b[0] == a[0] and b[1] == -a[1]]


def second_order_failures(beta, levels, temperature):
    """Levels whose shift at F^2 = 3 <F^2> departs from second order (see the module docstring)."""
    H = hm.Hydrogen(beta, **MESH)
    frms = np.sqrt(2 * temperature / hm.K_HARTREE * beta ** 2 / M_ATOM)
    bad = set()
    for par in (1, -1):
        ms = list(range(-6, 6))
        E0 = np.concatenate([H.block(m, par)[0][:150] for m in ms])
        offs = np.cumsum([0] + [150] * len(ms))
        X = np.zeros((E0.size, E0.size))
        for i, m in enumerate(ms[:-1]):
            Xm = H.xmat(m, par)[:150, :150]
            X[offs[i + 1]:offs[i + 2], offs[i]:offs[i + 1]] = Xm
            X[offs[i]:offs[i + 1], offs[i + 1]:offs[i + 2]] = Xm.T
        f = np.sqrt(3.0) * frms
        w, V = np.linalg.eigh(np.diag(E0) + f * X)
        for m, p, k in levels:
            if p != par:
                continue
            j = offs[ms.index(m)] + k
            dE = E0 - E0[j]
            alpha = 2 * np.sum(X[dE != 0, j] ** 2 / dE[dE != 0])
            overlap = V[j, :] ** 2
            kk = int(np.argmax(overlap))
            pert = -0.5 * alpha * f * f
            if pert == 0 or abs((w[kk] - E0[j]) / pert - 1) > TOLERANCE or overlap[kk] < 0.8:
                bad.add((m, p, k))
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('output')
    out = ap.parse_args().output
    pairs = components()
    shape = (len(pairs), FIELDS.size)
    lam, dalpha = np.full(shape, np.nan), np.full(shape, np.nan)
    cont, second = np.zeros(shape, bool), np.zeros(shape, bool)
    failures = {}
    for B in SECOND_ORDER_FIELDS:
        levels = {v for pair in pairs for v in pair}
        failures[B] = second_order_failures(B * 1e6 / hm.B0_G, levels, SECOND_ORDER_TEMPERATURE)
    for j, B in enumerate(FIELDS):
        beta = B * 1e6 / hm.B0_G
        H = hm.Hydrogen(beta, **MESH)
        threshold = lambda mm: 0.5 * beta * (abs(mm) + mm + 1)  # lowest Landau level of block mm [Hartree]  # noqa: E731
        near = failures[SECOND_ORDER_FIELDS[int(np.argmin(np.abs(SECOND_ORDER_FIELDS - B)))]]
        for i, (a, b) in enumerate(pairs):
            Ea, Eb = H.block(*a[:2])[0][a[2]], H.block(*b[:2])[0][b[2]]
            lam[i, j] = hm.HC_EV_A / ((Eb - Ea) * hm.HARTREE_EV)
            dalpha[i, j] = H.alpha(*b) - H.alpha(*a)
            cont[i, j] = any(E > min(threshold(s[0] - 1), threshold(s[0] + 1)) for s, E in ((a, Ea), (b, Eb)))
            second[i, j] = a in near or b in near
        print('B = %g MG' % B, flush=True)
    negative = ~(lam > 0)
    labels = np.array([(a[0], a[1], a[2] + 1, b[0], b[1], b[2] + 1) for a, b in pairs], dtype=np.int64)
    np.savez_compressed(out, field_megagauss=FIELDS, labels=labels, wavelength_infinite_mass_angstrom=lam,
                        delta_alpha=np.where(cont | second | negative, np.nan, dalpha), delta_alpha_unflagged=dalpha,
                        flag_continuum=cont, flag_second_order=second, flag_negative=negative)


if __name__ == '__main__':
    main()
