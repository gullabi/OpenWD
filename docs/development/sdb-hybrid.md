# Hot-subdwarf (sdB) H/He models

[Contributor guide](README.md) · [sdB model guide](../models/sdB.md)

The [sdB model guide](../models/sdB.md) tells users how to compute a model
and shows the validation figure. This note describes the solver options and
the research drivers for developers.

## Method

The sdB models use the hybrid LTE/NLTE method for B stars and sdBs
(Przybilla et al. 2006; Irrgang et al. 2018):

1. **LTE H/He structure.** A public DAB cold start (`compute_dab`, LTE) at the
   requested Teff, log g and He/H. With metals, the research driver gets the
   same abundances (`--abundance`), so the structure is metal-blanketed (see
   [Ionization balance](#ionization-balance)).
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
  hydrogenic Kramers form (excited).
- The low stages are ionized mainly from excited levels. Near threshold, the
  OP cross sections of those levels exceed the hydrogenic values (on average
  by 2–9 times for N II, O II, S II and S III, and by 3–15 times for the Si II
  3d, 4s, 4p and 3s3p² levels) because of autoionizing resonances. Hydrogenic
  cross sections would underionize these stages.
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
match the observed depths within 0.01–0.02. Si III 4552–4575 Å and S III are
too weak in Feige 38. Si IV matches only with the blanketed structure (below).

### Ionization balance

The equivalent-width ratio of Si II 4128+4131 Å to Si III 4552+4568+4575 Å
measures the silicon ionization. Below are the ratios with the full metal
mixture against the X-shooter co-adds, with the metal-free and the blanketed
structure (below). `research/sdb_silicon_ew_observed.py` integrates ±1.6 Å after
division by the convolved H/He host and a linear continuum through the
window edges, in the same way for the model and the observation. The grid
point has no observation and uses `research/sdb_silicon_ew.py`.

| Star | Metal-free structure | Blanketed structure | Observed |
| --- | --- | --- | --- |
| 25 kK, log g 5.2 (grid) | 0.21 | — | — |
| HD 4539 | 0.44 | 0.36 | 0.11 |
| Feige 38 | 0.17 | 0.13 | 0.08 |

TLUSTY/SYNSPEC with the BSTAR2006 model at 20 kK, log g 4.0 and solar Si
(no microturbulence) gives a ratio of 1.09 in LTE and 0.73 in NLTE. With the
same structure, OpenWD gives 1.07 in LTE, so the line formation agrees. In
NLTE, OpenWD gives 0.64.
Line by line, OpenWD NLTE agrees with SYNSPEC NLTE to within 10%
(Si II 4128 Å: 49.7 against 51.2 mÅ; Si III 4552 Å: 67.1 against 65.0 mÅ).
The remaining difference comes from the structure. The OpenWD metal-free LTE
H/He structure at the same parameters is 300–400 K cooler where the lines
form (column mass 0.03–0.4 g cm⁻²), and it raises the ratio by a factor of
1.27 in both LTE and NLTE.

**Blanketed structure.** `research/sdb_lte_hhe.py --abundance X=value` adds
metals to the LTE structure (`DABConfig.abundances`: opacity-sampled lines,
bound-free edges and metal electrons). This is the default recipe:
`sdb_trace_metals.py` stops for a metal-free host unless it gets
`--metal-free-structure` (used by `sdb_metal_grid.sh` on the H/He grid), warns
if the structure abundances differ, and records them as
`structure_abundances`. The NLTE H/He and metal stages do not change. With the published abundances, the structure is 1.1–1.5% (HD 4539)
and 1.5–2.1% (Feige 38) hotter at column mass 0.01–0.3 g cm⁻². It is not
slower: the LTE structure takes 387 s and 400 s (353 s and 627 s without
metals), and the NLTE H/He stage converges in fewer iterations (687 s and
676 s instead of 1110 s and 1172 s). The silicon ratio drops by factors of 1.21
(HD 4539) and 1.34 (Feige 38). In Feige 38, Si IV 4089 and 4116 Å then match
the observed equivalent widths (25 and 20 mÅ against 26 and 20 mÅ). Other
metal-line depths change by at most 0.01.

The ratio is still 3 (HD 4539) and 1.6 (Feige 38) times the observed value.
Si II is too strong (Si II 5041/5056 Å are absent in Feige 38) and
Si III 4552–4575 Å is too weak, by factors of 2 (HD 4539) and 1.4
(Feige 38). The Si III triplet is close to LTE: at HD 4539, NLTE
strengthens Si III 4552 Å by 8%. Geier (2013) measured the silicon
abundances from Si III and Si IV with LTE models. For Feige 38 Geier used
Teff = 30,600 K (here 28,200 K), and Si III and Si IV give abundances that
differ by 0.65 dex. In HD 4539, nearly all metal lines (C II, N II, Si III,
S III and the LTE Fe III lines) are about half the observed depth at the
abundances of Geier (2013). This affects the line strengths but not the
ratio.

TLUSTY 208 reproduces OpenWD at the HD 4539 parameters. The TLUSTY model has
H, He and Si in NLTE (BSTAR2006 Si II–IV atoms, log Si/H = −5.30) and no other
metals. Its NLTE H/He structure agrees with the OpenWD metal-free LTE
structure to 0.2% at column mass 0.01–0.3 g cm⁻². SYNSPEC gives a silicon
ratio of 0.375 on the TLUSTY model, and OpenWD (silicon only, on its own
structure) gives 0.372. The individual Si II and Si III lines agree to within
8%: Si III 4552 Å is 30 mÅ in both, against 65 mÅ observed. In LTE on the
blanketed OpenWD structure, SYNSPEC reproduces the OpenWD equivalent widths
of Si II, Si III, C II, N II, S III and Fe III to within 4% (Fe III 4164 Å:
9%), and the continuum to within 1%. The remaining discrepancy is therefore
not specific to OpenWD. At the published parameters and abundances, hybrid
and full NLTE models both give Si II too strong and Si III too weak.

**Teff sensitivity** (a diagnostic, not a fit). These models are 500–3000 K
hotter than the published Teff, with log g, He/H and abundances fixed and
the blanketed structure.

- HD 4539 at 25,200 K (+2000 K) gives a silicon ratio of 0.116 (observed
  0.11).
  - The Si II 4128/4131 Å and Si III 4568/4575 Å profiles then match.
    Si III 4552 Å is 25% too shallow.
  - The single-line ratio Si II 4128/Si III 4552 matches at about 26,000 K.
  - The OpenWD Balmer lines fit the X-shooter spectrum best at 24,200 K. At
    25,200 K they fit as well as at 23,200 K, and at 26,200 K they fit
    worse. He I barely changes.
- Feige 38: Si II 4128/Si III 4552 matches at 29,200 K (+1000 K). The
  Balmer lines fit best at 30,200–31,200 K; Geier (2013) used 30,600 K.
  - Si IV 4089/4116 Å, which matches at 28,200 K, becomes 1.4 (29,200 K) to
    2 (30,200 K) times too strong.
  - At every Teff, Si III 4552–4575 Å is 1.45 times too weak and
    Si III 4813–4829 Å is 1.3–1.5 times too strong. A Teff change does not
    remove this disagreement between the Si III multiplets.

## Research scripts

| Script | Purpose |
| --- | --- |
| `research/sdb_grid_check.sh`, `sdb_grid_report.py`, `sdb_grid_plot_spectra.py` | 12-point H/He grid at 20–40 kK, log g 5.0–6.2, log He/H -4 to -1 |
| `research/sdb_metal_grid.sh` | The same grid with typical metal abundances |
| `research/sdb_fixed_point_check.py` | One unpreconditioned update on a converged model |
| `research/sdb_blanketing_check.py` | Effect of metal opacity on the structure (LTE DA against DAZ) |
| `research/sdb_silicon_ew.py` | Si II/Si III equivalent widths and ratios of metal runs |
| `research/sdb_silicon_ew_observed.py` | The same for X-shooter co-adds and models, measured identically |
| `research/sdb_coadd_xshooter.py`, `sdb_compare_hd4539.py` | HD 4539 X-shooter comparison |

## Limitations

- Metal blanketing uses the model's own abundances and the Stout line list.
  At Feige 38 it heats the structure by 1.5–2.1% where the lines form.
- In both He I atoms, n = 5 is a superlevel coupled to n = 4 by
  Van Regemorter-type rates.
- The hydrostatic equation does not include radiation pressure. This is
  necessary for hot sdOs (above about 40 kK), which are out of scope.
