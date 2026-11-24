# WarmDQ fixed-structure spectra

These are synthetic OpenWD outputs, not observations. No stellar parameter or
tolerance was fitted.

Each `<case>.npz` holds a frozen 48-layer structure (column mass, Rosseland
optical depth, temperature), a vacuum wavelength grid (0.2 Å steps in the
case's windows) and the approved surface flux. The cases and their windows
are defined in `tests/warm_dq_regression_support.py`:

| Case | Configuration | Windows |
| --- | --- | --- |
| `warm-dq-j1215` | WD J1215+4700, 13574 K, log g 8.86, log C/H 1.11 (Kilic et al. 2026) | C II 4267, C I 5f band, C I 4772/4932 and Hβ, C I 5052, C I 5380, Hα and C II 6580, C I 7115 |
| `warm-dq-j1215-uniform-10mg` | the same, with a uniform 10 MG field of unknown direction | C I 4932–5052, Hα |
| `warm-dq-j0236-che` | SDSS J0236+2503, 14611 K, log g 8.777, log C/H 2.677, log C/He −1.823 (Koester & Kepler 2019) | C II 4267, C I 5f band, C I 4932, He I 5876, Hα, He I 6678 |

The structures are the converged cold starts of the
[warm DQ documentation](../../../docs/models/WarmDQ.md) models
(`research/warm_dq_doc_models.py`). The magnetic case uses the nonmagnetic
structure, as the public model does. `manifest.json` records each source run,
its equilibrium certificate, the hashes of its atmosphere and of the frozen
temperatures, the hash of every control file and the capture environment.

`tests/test_warm_dq_regression.py` checks the manifest and, in the `spectral`
tier, rebuilds the bulk LTE atmosphere on each frozen structure, forbids any
structure iteration, and compares the spectrum with the control at
rtol = 2e-6. The cold-start convergence of WD J1215+4700 is the separate
released-family canary `warm-dq-j1215`.

`research/capture_warm_dq_controls.py` wrote these files. It refuses an
existing output directory; tests and CI never run it. Replace a control only
after a physics review, as described in the
[development guide](../../../docs/development/README.md#regression-controls).
