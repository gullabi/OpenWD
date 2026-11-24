# J1637 jK Rydberg-manifold reference

This version records the 2026-11-23 change to the quasi-static Stark
manifolds of D6. Its seed is the v6 seed file byte for byte: Teff, log g,
composition, column mass, optical-depth grid, base temperature, 48 layers,
both 25,000-line budgets and the prescribed 1.5% perturbation are unchanged.

Physics change: D6 now sets `AtomicIon.rydberg_jk_levels_join_ls_manifold`
for every ion, as WarmDQ does for C I. A Stout level in jK coupling (parent J
in the core label, for example C I 2p(2Po<1/2>)4f) joins the Stark
n-manifold of the LS levels of its parent term and of the jK levels with the
same parent J, and is measured from that parent's ionization limit. Before,
a jK level could mix only with levels of the identical core label, so the
C I 4f levels stayed apart from the LS 4d levels of the same n. The change
acts on C I (the 2s2p3 3Do-4f blend at 5015-5045 A, and the 3p-nd lines it
now mixes with); Ne I and Fe II change negligibly.

The fixed spectrum changes by at most 12.9% (5040.2 A, in the C I 4f blend)
and by -2.2e-5 in integrated flux over the stored grid. Against the GTC/OSIRIS
spectrum (R = 2500, a cubic continuum per paper-figure panel) chi2 goes from
11965.8 to 11927.0, mostly at 5000-5060 A and 5660-5680 A; C I 3p-5d near
6000-6020 A worsens by 13.5. The Hollands et al. (2025) matched-panel RMS
against the Koester model goes from 0.0326 to 0.0324 (observed: 0.0543,
unchanged). The two measured warm updates do not establish cold convergence.
Source-convergence, transfer, profile and spectral tolerances are unchanged,
including the fixed-spectrum relative tolerance of 2e-6.
