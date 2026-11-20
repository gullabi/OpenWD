"""Hydrogen in a uniform magnetic field: eigenstates and transverse polarizabilities.

Used by ``build_motional_stark_table.py``. Textbook Hamiltonian in atomic units
(Hartree), infinite nuclear mass, spin omitted (the spin Zeeman term is the
same for both levels of an electric-dipole transition and for the states an
in-plane electric field couples, so it cancels in every quantity here)::

    H_m = -1/2 d2/dr2 + l(l+1)/(2 r^2) - 1/r + (beta/2) m + (beta^2/8) r^2 sin^2(theta)

with beta = B / 2.350518e9 G (Ruder et al. 1994, Atoms in Strong Magnetic
Fields, Springer). This beta is twice the H2db beta (unit 4.70103e9 G).

Radial: regularized Lagrange-Laguerre mesh (Baye 2015, Phys. Rep. 565, 1); u(r)
= r R(r) is expanded in Lagrange functions and local operators are diagonal.
Angular: Y_l^m, l = |m|..lmax, one z-parity (-1)^(l+m) per block.

A motional electric field F is perpendicular to B; its dipole operator x
couples m to m +- 1 at fixed z-parity. Second-order shift of state a:
dE_a = -alpha_a F^2 / 2, alpha_a = 2 sum_b |<b|x|a>|^2 / (E_b - E_a), summed
over the box spectrum of the m +- 1 blocks (bound states and discretized
continuum).
"""

import numpy as np
from scipy.special import roots_genlaguerre, roots_legendre, sph_harm_y

B0_G = 2.350518e9  # atomic unit of magnetic field [G]: hbar omega_c = beta Hartree
HARTREE_EV = 27.211386
HC_EV_A = 12398.4198  # h c [eV A]
M_P = 1836.15267  # proton mass / electron mass
K_HARTREE = 3.1577502e5  # K per Hartree


def mesh(N, h):
    """Regularized Lagrange-Laguerre mesh: points r_i = h x_i, kinetic matrix of -d2/dr2."""
    x, _ = roots_genlaguerre(N, 0)
    T = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                T[i, i] = (4.0 + (4 * N + 2) * x[i] - x[i] ** 2) / (12.0 * x[i] ** 2)
            else:
                T[i, j] = (-1.0) ** (i - j) * (x[i] + x[j]) / (np.sqrt(x[i] * x[j]) * (x[i] - x[j]) ** 2)
    return h * x, T / h ** 2


class Angular:
    """theta parts of Y_l^m on a Gauss-Legendre grid in u = cos(theta), normalised to
    int Theta^2 du = 1 (phi part e^{i m phi}/sqrt(2 pi) implicit)."""

    def __init__(self, lmax, nq=None):
        nq = nq or (lmax + 8)
        u, w = roots_legendre(2 * nq)
        self.u, self.w, self.lmax = u, w, lmax
        self.th = np.arccos(u)
        self._c = {}

    def theta(self, l, m):
        k = (l, m)
        if k not in self._c:
            self._c[k] = np.real(sph_harm_y(l, m, self.th, 0.0)) * np.sqrt(2 * np.pi)
        return self._c[k]

    def ls(self, m, par):
        """l values of the block (m, z-parity par = (-1)^(l+m))."""
        return [l for l in range(abs(m), self.lmax + 1) if (-1) ** (l + m) == par]

    def mat(self, m1, l1s, m2, l2s, f):
        """< l1 m1 | f(u) | l2 m2 > over theta only (phi integral done by the caller)."""
        return np.array([[np.sum(self.w * self.theta(a, m1) * f(self.u) * self.theta(b, m2)) for b in l2s]
                         for a in l1s])


class Hydrogen:
    def __init__(self, beta, N=60, h=0.6, lmax=40):
        self.beta, self.N, self.lmax = beta, N, lmax
        self.r, self.T = mesh(N, h)
        self.ang = Angular(lmax)
        self.blocks = {}

    def block(self, m, par):
        """Eigenvalues and eigenvectors (columns, l-major: l block of N radial points) of block (m, par)."""
        key = (m, par)
        if key in self.blocks:
            return self.blocks[key]
        b, r, N = self.beta, self.r, self.N
        ls = self.ang.ls(m, par)
        S = self.ang.mat(m, ls, m, ls, lambda u: 1.0 - u ** 2)  # sin^2 theta
        nl = len(ls)
        H = np.zeros((nl * N, nl * N))
        for i, l in enumerate(ls):
            s = slice(i * N, (i + 1) * N)
            H[s, s] = 0.5 * self.T + np.diag(l * (l + 1) / (2 * r ** 2) - 1.0 / r + 0.5 * b * m)
            for j in range(nl):
                if abs(S[i, j]) > 1e-14:
                    t = slice(j * N, (j + 1) * N)
                    H[s, t] += np.diag(b * b / 8.0 * r ** 2 * S[i, j])
        E, V = np.linalg.eigh(H)
        self.blocks[key] = (E, V, ls)
        return self.blocks[key]

    def xmat(self, m, par):
        """<b|x|a> between block (m+1, par) [rows] and (m, par) [cols], in the eigenbases."""
        _, V1, l1 = self.block(m + 1, par)
        _, V0, l0 = self.block(m, par)
        # phi part: int e^{-i(m+1)phi} cos(phi) e^{i m phi} dphi / (2 pi) = 1/2
        A = 0.5 * self.ang.mat(m + 1, l1, m, l0, lambda u: np.sqrt(1 - u ** 2))
        return V1.T @ np.kron(A, np.diag(self.r)) @ V0

    def alpha(self, m, par, k):
        """Transverse polarizability of the k-th state of block (m, par)."""
        E0 = self.block(m, par)[0][k]
        a = 0.0
        for mm, X in ((m + 1, self.xmat(m, par)[:, k]), (m - 1, self.xmat(m - 1, par)[k, :])):
            a += 2 * np.sum(X ** 2 / (self.block(mm, par)[0] - E0))
        return a


def labels(m, par, nmax):
    """Zero-field labels (n, l) of block (m, par) in energy order: the nu-th state
    counts n upward and l downward within a shell (H2db's excitation index nu =
    position + 1; checked state by state against the public H2db energies)."""
    return [(n, l) for n in range(1, nmax + 1) for l in range(n - 1, abs(m) - 1, -1) if (-1) ** (l + m) == par]
