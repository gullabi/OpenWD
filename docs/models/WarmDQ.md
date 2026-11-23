# Warm and hot DQ: carbon atmospheres with hydrogen and helium (in development)

[Model guide](README.md) · [Getting started](../getting-started.md)

`WarmDQConfig` computes LTE atmospheres made of carbon with hydrogen, helium
or both. Such atmospheres occur in warm DQ, DAQ and hot DQ white dwarfs (Teff
of about 11,000–25,000 K, masses near 1 M☉ and above). Carbon, hydrogen,
helium and any added element all contribute to the pressure, the charge
balance, the mass and the opacity.
A magnetic field can be added to the emergent spectrum. OpenWD does not fit
observations.

The class is in development. It has been run at the points listed under
[Tested range](#tested-range) only.

## Run a model

```python
from wd_spectra import WarmDQConfig, run_model

run = run_model(
    WarmDQConfig(
        effective_temperature=15_000, logg=8.9,
        log_carbon_to_hydrogen=1.0,       # log10 N(C)/N(H)
        log_carbon_to_helium=None,        # log10 N(C)/N(He); None: no helium
    ),
    "results/warm-dq",
    require_convergence=True,
)
spectrum = run.spectrum  # vacuum wavelength (Å), surface F_lambda
```

The default spectrum covers 3500–9900 Å in 0.2 Å steps. Give `wavelength=`
to `run_model` for a different grid.

## Options

| Option | Default | Effect |
| --- | --- | --- |
| `effective_temperature`, `logg` | 16000 K, 8.8 | Stellar parameters. |
| `log_carbon_to_hydrogen` | 1.5 | log10 N(C)/N(H). `None` removes hydrogen. |
| `log_carbon_to_helium` | `None` | log10 N(C)/N(He). `None` removes helium. |
| `trace_abundances` | `{}` | log10 N(X)/N(C) of other elements, for example `{"O": -1.0}`. |
| `quality` | `"standard"` | `"quick"` (24 depths), `"standard"` (48 depths), `"production"` (72 depths). |
| `mixing_length_alpha` | 1.25 | ML2 convection parameter. `None` gives a radiative atmosphere. |
| `structure_maximum_metal_lines`, `formal_maximum_metal_lines` | `None` (25,000) | Line budgets of the structure and the spectrum. |
| `magnetic_field_megagauss` | 0 | Field strength. Zero gives the nonmagnetic spectrum. |
| `field_geometry` | `"dipole"` | `"dipole"` (offset dipole) or `"uniform"`. |
| `field_strength_definition` | `"dipole-polar"` | `"dipole-polar"` or `"visible-mean"`, as in `DAHConfig`. |
| `dipole_inclination_deg`, `dipole_offset_radius` | 60, (0, 0, 0) | Dipole axis inclination and displacement in stellar radii. |
| `field_angle_deg` | `None` | Angle between a uniform field and the line of sight. `None` is isotropic. |
| `disk_field_bins` | 21 | Number of field-strength bins over the visible disk. |
| `include_diamagnetic_shift` | `True` | First-order quadratic Zeeman shift of the metal lines. |
| `magnetic_convection_suppression` | `True` | Use a radiative atmosphere when the mean visible field is 0.05 MG or stronger. |

## Physics

- **Equation of state.** The bulk-mixture closure of the [D6](D6.md) class:
  multi-element Saha balance with charge neutrality. Hydrogen and helium
  join the same closure. Their partition functions are the Hummer & Mihalas
  (1988) occupation-probability sums of the [DA](DA.md) and [DB](DB.md)
  classes, with the Q-MHD microfield distribution (Nayfonov et al. 1999) and
  neutral H and He as hard-sphere perturbers. In the helium-dominated limit
  the populations, c_P and ∇ad agree with the DB equation of state to
  better than 10⁻³. ML2 convection uses the derivatives of the same EOS.
  The fluid is ideal; the dense-helium corrections of the cool DB and DQ
  classes (He-REOS, dense He⁻ free–free) are not applied.
- **Carbon continuum and lines.** As in D6: level-resolved Opacity Project
  cross sections for C I and C II, Verner et al. (1996) fits for higher
  stages, free–free absorption, electron scattering, and the Atomic Line
  List v3 lines of C I–III with Q-MHD dissolution of the Rydberg lines.
- **Transition probabilities.** Lines with a NIST ASD value (1150–9000 Å;
  Kramida et al. 2024) use that value: accuracy grade D or better for C I,
  every grade for C II. The Atomic Line List values of the C II
  2s2p(³P°)nl system are 4–20 times larger than the NIST values (f = 2.45
  against 0.36 for 3d ⁴D°₃/₂–4f ⁴F₅/₂ at 4075 Å), exceed the sum rule and
  give strong spurious lines near 3877 and 4076 Å. A grade-E NIST value
  (uncertainty above 50%) is far closer than such a value. C I lines between
  states of the MCDHF calculation of Li et al. (2021) with a length–velocity
  difference up to 20% (741 lines) then take its oscillator strengths. For
  the 2p3s–2p4p array, NIST combines the Hibbert et al. (1993) singlet values
  with triplet values renormalized by 0.71–0.78; Li et al., the arc
  measurements of Jones & Wiese (1984) and the solar C I 4775.9 Å line all
  put the singlets 0.17–0.24 dex weaker relative to the triplets, and so do
  the warm DQ spectra (C I 4932, 5052 and 5380 against 4771).
- **Carbon line broadening.** Voigt profiles. The C I 3s–4p multiplets
  (4771, 4817, 4932, 5052 and 5380 Å) take measured Stark widths and red
  shifts (Konjević & Wiese 1990; Konjević et al. 2002; measured at
  n_e = 0.1–0.8 × 10¹⁷ cm⁻³ and 9000–11,600 K, scaled linearly with n_e).
  Eight C II multiplets (6580, 3920, 7234, 4267, 5145, 5650, 2837 and
  2511 Å) take semiclassical widths and shifts from STARK-B (Larbi-Terzi
  et al. 2012), with their temperature dependence. The shift moves each
  depth's profile by its own amount. Other C I lines take the
  nearest-level semiclassical width of Freudenstein & Cooper (1978), from
  the tabulated term energies and Coulomb-approximation radial integrals.
  For the measured C I multiplets it gives 1.34 times the laboratory widths
  (scatter 0.17 dex), against 0.72 times (0.39 dex) for the Kurucz
  constants, which for lines to high levels are up to twenty times too
  small. Lines of other ions use the Kurucz electron Stark constants; a
  multiplet member without one takes the median of its multiplet, and a
  line without any uses the classical estimate. Nearly hydrogenic Rydberg
  lines use the quasi-static ionic Stark profile of D6. The C I nf levels,
  which Stout writes in jK coupling, join the n-manifold of the LS levels
  of their parent term and of the jK levels of their own parent J (the
  ionic field does not change the core state), so the 2s2p³ ³D°–nf lines
  are Stark-mixed with nd as well (other classes keep the previous
  grouping).
- **C I 2s2p³ ³D°–nf series.** Stout has the 4f members (5013–5043 Å,
  NIST grade C) but no line to 5f–8f and no nf level built on the C II
  ²P°₁/₂ parent. The missing levels are placed with the quantum defect of
  their 4f member (this reproduces the tabulated nf levels to 6 cm⁻¹), and
  the lines take the 4f oscillator strengths scaled by n*⁻³ and the
  transition energy. Their Stark-broadened 5f members (4462–4480 Å) produce
  the broad absorption seen at 4430–4540 Å in warm and hot DQs.
- **Hydrogen opacity.** As in DA: H I bound–free with the dissolved-level
  pseudo-continuum, free–free, H⁻, and the Lyman, Balmer, Paschen and
  Brackett lines with the Tremblay & Bergeron (2009) Stark profiles. With
  helium present the Balmer lines also include neutral-helium broadening.
- **Helium opacity.** As in DB: He I bound–free (Opacity Project terms,
  Rydberg shells and the dissolved-level pseudo-continuum), He II bound–free,
  He⁺ and He²⁺ free–free, He⁻ free–free, the optical He I lines with the
  Beauchamp et al. Stark profiles, the He I resonance series and the He II
  lines. Neutral helium also broadens the carbon lines (van der Waals).
- **Structure.** The shared adaptive Newton solver relaxes the atmosphere and
  checks all-depth flux and local energy balance.
- **Magnetic field.** The field changes only the line opacity of the emergent
  spectrum. The structure, the level populations and the continuum are
  nonmagnetic.
  - *Metal lines.* Each LS multiplet is diagonalized as a whole in the field
    (fine structure from the tabulated level energies, plus the linear Zeeman
    term), which gives the Paschen–Back pattern at any field strength. The
    summed gf of the multiplet is conserved. The first-order diamagnetic
    energy of the outer electron is added with a hydrogenic ⟨r²⟩.
    Intercombination lines and levels without an LS label use the linear
    pattern of their own J levels.
  - *Rydberg lines.* Lines with a quasi-static Stark profile keep it and are
    translated to the three components of a normal triplet.
  - *Balmer lines.* The [DAH](DAH.md) components: normal triplets up to 1 MG
    and the H2db components above. Other hydrogen series and the helium
    lines are not split.
  - *Transfer.* One ray per surface cell with the Stokes-I opacity of its
    field strength and field–ray angle, summed over the visible disk.

## Validation

Models were computed at published parameters and compared with public
spectra. No stellar parameter was fitted: only a radial velocity and, per
spectrum, a degree-5 multiplicative Chebyshev polynomial (flux calibration,
reddening and solid angle) were adjusted. The model was averaged over each
pixel and multiplied by the DESI resolution matrix, or convolved with the
SDSS `wdisp` dispersion. Standard quality, 3000 structure lines and
convection (ML2, α = 1.25), as in the published fits.

- **DESI DR1, C+H.** Kilic et al. (2026) fitted 70 DESI DQs with
  helium-free C+H models (spectroscopic Teff and log C/H; log g from the
  Gaia parallax).
- **SDSS, C/He.** Koester & Kepler (2019) fitted warm DQs with C/He models
  containing trace hydrogen ([H/He] = −4.5) and hot DQs with [C/He] = 2
  (photometric Teff and log g).

χ² per pixel is given for the model and for the same model without lines.
The ratios are model/observed pseudo-equivalent widths, relative to the
model continuum.

| Spectrum | Parameters (reference) | S/N | χ²/px | without lines | C I 4933 | C I 7115 | C II 4268 | Hα |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DESI WD J0146−0426 | 12603 K, 8.71, C/H 1.51 (K26) | 14 | 1.34 | 2.03 | 1.54 | 1.22 | 1.28 | 0.97 |
| DESI WD J0918−0525 | 12830 K, 8.80, C/H 1.42 (K26) | 6 | 1.06 | 1.14 | 0.99 | 1.29 | – | 1.10 |
| DESI WD J0936+0607 | 12641 K, 8.88, C/H 3.00 (K26) | 6 | 1.03 | 1.10 | 1.64 | 1.21 | – | 1.07 |
| DESI WD J1215+4700 | 13574 K, 8.86, C/H 1.11 (K26) | 8 | 1.01 | 1.22 | 1.32 | 1.31 | 1.05 | 1.01 |
| DESI G35-26 | 16338 K, 9.05, C/H 0.95 (K26) | 26 | 1.56 | 4.84 | 1.22 | 1.14 | 1.22 | 1.02 |
| DESI WD J2302+0045 | 21851 K, 8.84, C/H 2.39 (K26) | 17 | 1.47 | 2.96 | 1.12 | 1.18 | 1.09 | 0.96 |
| SDSS J1622+3004 | 15193 K, 9.00, C/H 1.35 (K26) | 38 | 1.46 | 5.88 | 1.40 | 1.18 | 1.26 | 1.03 |
| SDSS J1622+3004 | 16131 K, 8.93, [C/He] −0.10 (KK19) | 38 | 3.13 | 5.89 | 1.32 | 1.18 | 1.64 | 0.45 |
| SDSS J1728+5558 | 14655 K, 8.91, C/H 1.76 (K26) | 48 | 2.64 | 9.22 | 1.51 | 1.11 | 1.65 | 0.95 |
| SDSS J1728+5558 | 14772 K, 8.87, [C/He] +0.30 (KK19) | 48 | 3.82 | 9.20 | 1.53 | 1.13 | 1.86 | 0.54 |
| SDSS J0236+2503 | 14611 K, 8.78, [C/He] −1.82 (KK19) | 16 | 1.29 | 1.68 | 1.58 | 1.08 | 0.72 | 0.68 |
| SDSS J0818+0102 | 24483 K, 8.33, [C/He] 2 (KK19) | 23 | 2.49 | 3.28 | 1.00 | 1.28 | 1.32 | 1.06 |

K26: Kilic et al. (2026); KK19: Koester & Kepler (2019).

- Every model reproduces the spectrum far better than the same model
  without lines. Hα and Hβ, including their Stark wings, agree for the
  C+H solutions.
- Where a star has both solutions, the C+H one fits better: the fixed
  trace hydrogen of the C/He grid leaves Hα and Hβ two to four times too
  weak.
- At the C/He parameters the models show He I 4472, 5876 and 6678 Å lines
  (5876 Å: 2.6–4.2 Å equivalent width) that are absent from the spectra.
  Kilic et al. (2024) reach the same conclusion with independent models.
  The [DZ](DZ.md) class at the same parameters agrees with this class to
  2% in temperature, 15% in density and 0.2 Å in the He I equivalent widths.
- For the C+H solutions the carbon lines are systematically too strong:
  C I 4933 Å by up to 65%, C I 5053 and 7115 Å by 10–30%, C II 4268 Å by
  5–65% and C II 3920 Å at 14,000–17,000 K.
  C I 5381 Å is 30% too weak; its Kurucz Stark constant is nine times
  smaller than the measured width (Jones & Wiese 1984). Weak C I lines
  without NIST data (3s–5p at 4065 Å, 3s–4p ³D at 4892 Å) are much stronger
  than observed. Koester & Kepler (2019) and Kilic et al. (2024) report
  similar carbon-line problems with their models.
- A radiative atmosphere of WD J2302+0045 (cold start continued from the
  convective one) gives the same χ² as the convective model; the normalized
  spectra differ by 0.4% rms.

## Limitations

- LTE. No molecules or negative carbon ions. Below about 12,000 K the C₂
  Swan bands appear in observed warm DQs and are not modelled.
- Ideal fluid. Helium-dominated compositions at high density (cool, high
  gravity) need the non-ideal corrections of the DQ class.
- Helium lines are not Zeeman split in magnetic models.
- The magnetic field does not enter the structure, the ionization balance or
  the continuum. Polarization and magneto-optical effects are not computed.
- The diamagnetic shift is first order. It is not valid when the shift
  approaches the separation of neighbouring terms (high Rydberg levels at
  several MG).
- Zeeman and Stark broadening are combined by translating zero-field
  profiles.
- A radiative atmosphere (`mixing_length_alpha=None`, or a field that
  suppresses convection) does not converge from the gray seed, whose
  surface is far hotter than the line-blanketed one. A radiative cold start
  therefore first converges the ML2 model (α = 1.25) and continues from it;
  `atmosphere.metadata["radiative_cold_start"]` records this. Convection
  carries flux only below Rosseland depth ~5, so the convective structure
  is already close.
- At the published parameters of 13,000–16,000 K stars the strong C I
  lines (4771, 4932, 5052 and 5380 Å) are 10–40% stronger than observed in
  equivalent width. The excess is common to singlet and triplet lines and
  is changed alike for all of them by the temperature gradient of the
  photosphere, so it is degenerate with the published effective
  temperatures, which come from different model atmospheres.
- The C I 3s–4p ³D lines at 4888–4898 Å are predicted about 10% deep but
  are not seen in the observed spectra. NIST gives no transition
  probability for them; the Stout value agrees with laboratory intensities
  and the Kurucz value is twenty times larger.
- The observed 4430–4540 Å band extends about 30 Å further to the red
  than the modelled 2s2p³ ³D°–5f band.
- The C I 4269 Å oscillator strength is uncertain by a factor of four
  (Koester & Kepler 2019). The NIST value has grade E and is not used.

See [limitations](../limitations.md) for the shared caveats.

## Tested range

See [tested points](../tested-temperature-ranges.md).
