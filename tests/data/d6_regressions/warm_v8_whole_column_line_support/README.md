# J1637 whole-column structure line-support reference

This version records the 2026-11-24 change to the profile support of the
structure line opacity. Its seed is the v7 seed file byte for byte: Teff,
log g, composition, column mass, optical-depth grid, base temperature,
48 layers, both 25,000-line budgets and the prescribed 1.5% perturbation are
unchanged.

Change: a structure line's profile now extends until its omitted wing is
optically thin through the whole column (`profile_support_maximum_rosseland_optical_depth`
defaults to `inf` in `radiative_equilibrium_d6_atmosphere`). Before, the
wing had to be thin only down to Rosseland depth 2, as the emergent spectrum
requires. A line of a high ion that is weak in the photosphere and strong in
the hot interior then kept a short support ending at up to several times the
deep Rosseland opacity, and the support could double within the solver's
centred temperature probe (ln T +- 2e-4). At the certified J1637 structure
nine supports change by a factor two within that probe (the largest deep
effect: 3087.1 A, whose edge opacity falls from 0.98 to 0.25 of the deep
Rosseland opacity), and 167 of the 25,000 lines end above the deep Rosseland
opacity. In a WarmDQ carbon atmosphere the same mechanism (C III 1175.7 A)
gave the deep rows of the analytic Jacobian the wrong sign and stalled the
solve.

The fixed spectrum is unchanged: it is computed on the unperturbed seed
atmosphere and the emergent spectrum keeps the photospheric support. The two
measured warm updates change the temperature by at most 0.87% and the density
by 1.6%, both in the deepest layer; above it the changes are smaller. The
updates reach all-depth flux residuals of 1.0e-3 and 1.1e-6 (v7: 1.3e-3 and
3.0e-6). They do not establish cold convergence. Source-convergence,
transfer, profile and spectral tolerances are unchanged, including the
fixed-spectrum relative tolerance of 2e-6.
