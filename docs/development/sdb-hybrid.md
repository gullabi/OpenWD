# Hot-subdwarf (sdB) H/He models

[Contributor guide](README.md) · [sdB model guide](../models/sdB.md)

The [sdB model guide](../models/sdB.md) tells users how to compute a model
and shows the validation figure. This note describes the solver options and
the research drivers for developers.

## Method

The sdB models use the hybrid LTE/NLTE method for B stars and sdBs
(Przybilla et al. 2006; Irrgang et al. 2018):

1. **LTE H/He structure.** A public DAB cold start (`compute_dab`, LTE, no
   metals) at the requested Teff, log g and He/H.
2. **NLTE H/He populations.** The restricted hot H/He atom (`HotNLTEModel`)
   solves the statistical equilibrium on that fixed temperature and
   gas-pressure structure, with the formal-solution radiation field.
3. **Spectrum.** The emergent spectrum uses the NLTE populations.

`SdBConfig` and `compute_sdb` (in `src/wd_spectra/models/sdb.py`) implement
this method. `run_model` accepts `SdBConfig`. The research drivers
`research/sdb_lte_hhe.py` and `research/sdb_hybrid_nlte.py` contain the same
recipe. `sdb_hybrid_nlte.py` also writes the same synthesis with LTE
populations (`spectrum-lte.npz`) to isolate the NLTE effect.

The self-consistent DO/DAO solver (`solve_hot_nlte_atmosphere`) is not used.
At sdB densities, its full-NLTE stage is nearly singular in the outer layers
and does not converge.

## Solver options for sdBs

`compute_sdb` sets these options. Their defaults elsewhere do not change.

- `solve_sdb_populations` starts with the continuum transfer and then scales
  the explicitly coupled bound-bound radiative rates and the line transfer
  coefficients together. Collisions, bound-free processes, and local Planck
  rates for transitions without explicit line transfer are active from the
  first stage. Each stage with lines uses Rybicki & Hummer (1991) multilevel
  accelerated lambda iteration (`_hot_mali.py`). At full line strength the
  original statistical equilibrium and population convergence test apply.
  Each initialization stage hands off its lowest-residual evaluated state
  after meeting the tolerance or reaching its stage limit. The shared budget
  reserves an evaluation at full line strength.
- `HotNLTEModel.helium_conservation_row="dominant"` (default `"last"`):
  particle conservation replaces the equation of the most populous helium
  state, not the He III equation. In cool sdB surface layers
  (He III/He ~ 10⁻¹⁸), the He III equation makes the system singular to
  roundoff.
- `SdBConfig.helium_i_atom`: `"14"` (default) is the TLUSTY 14-term He I atom.
  `"24"` loads the TLUSTY 24-term atom with `read_tlusty_helium_i_atom`. For
  sdB optical He I lines, the 24-term atom changes the core depths by at most
  1.5%.

The coupled helium solver enforces particle conservation by a common
rescaling. It does not compute He III as the remainder of the total.

## Trace metals (research driver)

`research/sdb_trace_metals.py` adds metals to a finished hybrid model with
the fixed-host solver `solve_hot_trace_metals` (see
[hot DA/DAO trace metals](hot-daz-trace-metals.md)). The H/He structure and
populations stay fixed.

- NLTE elements: C, N, O, Si and S. The Stout atoms reach the standard sdB
  optical lines. C II and O III use bundled CHIANTI collision data.
- Photoionization: every photoionized stage of C, N, O, Si and S uses the
  Opacity Project cross sections of the TLUSTY model atoms
  (`TLUSTY_PHOTOIONIZATION`) for each level whose term matches in threshold
  (2%), spin and L. Unmatched levels use a Verner fit (ground) or the
  hydrogenic Kramers form (excited). Until 2026-11 only C III–IV had OP data.
  `--photoionization-tables carbon-only` restores that for comparisons.
- The low stages are ionized mainly from excited levels. Near threshold, the
  OP cross sections of those levels exceed the hydrogenic values (on average
  by 2–9 times for N II, O II, S II and S III, and by 3–15 times for the Si II
  3d, 4s, 4p and 3s3p² levels) because of autoionizing resonances. With the
  hydrogenic values, these stages were underionized.
- Si II has 60 levels.
- LTE line opacity in the fixed background: Fe, Mg and Al (`--lte-abundance`).
  The optical Fe III 4s–4p lines need a 410-level atom, which is too slow
  for NLTE.
- The driver uses `mali_overlap_velocity=15` (km/s) and
  `mali_overlap_mode="subordinate"`. Without these settings, overlapping lines
  of different elements (S III 702.78/702.82 Å on O III 702.84 Å) make the
  iteration oscillate.

At fixed published parameters, without abundance fits, the metal lines of
HD 4539 (C II) and Feige 38 (N II, O II, Si III 4813–4829 Å, Fe III 4164 Å)
match the observed depths within 0.01–0.02. Si III 4552–4575 Å, S III and
Si IV are too weak in Feige 38.

### Ionization balance

The equivalent-width ratio of Si II 4128+4131 Å to Si III 4552+4568+4575 Å
measures the silicon ionization. Below are the ratios with the full metal
mixture against X-shooter observations, before and after the OP cross
sections (with the 60-level Si II atom):

| Star | Before (C III–IV OP only, 30 Si II levels) | OP for all stages | Observed |
| --- | --- | --- | --- |
| 25 kK, log g 5.2 (grid) | 0.33 | 0.21 | — |
| HD 4539 | 0.62 | 0.44 | 0.18 |
| Feige 38 | 0.27 | 0.17 | 0.08 |

The Si II lines move about halfway to the observed depths. Si III 4552–4575 Å
barely changes and stays too weak. The OP data for the other elements weaken
the low stages. In Feige 38, C II 4267 Å drops from 0.21 to 0.16 against
0.19 observed, while C III stays too weak, so the C II/C III balance
improves. N II changes by at most 0.015 (N II 5679 Å now matches). O II and
S III change by less than 0.01. Si IV becomes weaker and was already too weak
in Feige 38. Feige 38 needs 80 metal iterations instead of 67.

TLUSTY/SYNSPEC with the BSTAR2006 model at 20 kK, log g 4.0 and solar Si
(no microturbulence) gives a ratio of 1.09 in LTE and 0.73 in NLTE. With the
same structure, OpenWD gives 1.07 in LTE, so the line formation agrees. In
NLTE, OpenWD gives 0.64 with OP and 0.76 with the hydrogenic cross sections.
The remaining difference comes from the structure. The OpenWD metal-free LTE
H/He structure at the same parameters is 300–400 K cooler where the lines
form (column mass 0.03–0.4 g cm⁻²), and it raises the ratio by a factor of
1.27 in both LTE and NLTE.

In an experiment (`research/sdb_blanketed_host.py`), the T(m) of an LTE
DAZ at the HD 4539 parameters and abundances was put into the H/He host. The
DAZ has metal opacity in the structure but no He. It converged in 119 s and is
1.1–1.5% hotter at column mass 0.01–0.3 g cm⁻². It lowers the HD 4539 ratio
from 0.44 to 0.36. Si III becomes 9% stronger and Si II about 10% weaker.
Other metal lines change by at most 0.01.

## Research scripts

| Script | Purpose |
| --- | --- |
| `research/sdb_grid_check.sh`, `sdb_grid_report.py`, `sdb_grid_plot_spectra.py` | 12-point H/He grid at 20–40 kK, log g 5.0–6.2, log He/H -4 to -1 |
| `research/sdb_metal_grid.sh` | The same grid with typical metal abundances |
| `research/sdb_fixed_point_check.py` | One unpreconditioned update on a converged model |
| `research/sdb_blanketing_check.py` | Effect of metal opacity on the structure (LTE DA against DAZ) |
| `research/sdb_blanketed_host.py` | H/He host with the T(m) of an LTE DAZ (blanketing experiment) |
| `research/sdb_silicon_ew.py` | Si II/Si III equivalent widths and ratios of metal runs |
| `research/sdb_coadd_xshooter.py`, `sdb_compare_hd4539.py` | HD 4539 X-shooter comparison |

## Limitations

- The structure contains no metals. At Feige 38, metal opacity heats the
  structure by 1.6–2.4% at fixed column mass. The metal-line depths change by
  at most 0.01, except Si IV.
- In both He I atoms, n = 5 is a superlevel coupled to n = 4 by
  Van Regemorter-type rates.
- The hydrostatic equation does not include radiation pressure. This is
  necessary for hot sdOs (above about 40 kK), which are out of scope.
