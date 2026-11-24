#!/usr/bin/env python3
"""WarmDQ models for docs/models/WarmDQ.md, at published parameters (nothing fitted).

    PYTHONPATH=src python research/warm_dq_doc_models.py OUTPUT_DIR [TAG ...]

Each model is one cold start of ``run_model(WarmDQConfig(...))`` at standard
quality with the class defaults (structure and formal line budgets, ML2
alpha = 1.25), on a 3500-9900 A vacuum grid with 0.2 A steps, written to
OUTPUT_DIR/<tag>. Existing outputs are skipped. Run one model per thread;
the models are independent.

Parameters:

* ``ch_*``: Kilic et al. (2026, arXiv:2602.07246), DESI DR1 hot white dwarfs,
  helium-free C+H fits (spectroscopic Teff and log C/H; log g from the Gaia
  parallax), from their Zenodo record 10.5281/zenodo.18332706
  (``allfits_DQ.pdf``). log C/H = 3.00 is their grid edge (no hydrogen
  detected).
* ``che_*``: Koester & Kepler (2019, A&A 628, A102), Tables 4 and 5, C/He fits
  of SDSS spectra with trace hydrogen [H/He] = -4.5 (warm DQs) or no hydrogen
  and [C/He] = 2 (hot DQ 0818+0102).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# tag: (name, Teff, log g, log C/H, log C/He, spectrum)
MODELS = {
    "ch_j0146": ("WD J0146-0426", 12603, 8.71, 1.51, None, "DESI"),
    "ch_j1215": ("WD J1215+4700", 13574, 8.86, 1.11, None, "DESI"),
    "ch_j0324": ("WD J0324-0135", 14323, 8.43, 0.17, None, "DESI"),
    "ch_j1728": ("SDSS J1728+5558", 14655, 8.91, 1.76, None, "SDSS"),
    "ch_j2300": ("WD J2300+0848", 14964, 8.92, -0.01, None, "DESI"),
    "ch_j1448": ("WD J1448+0519", 15080, 8.98, 1.04, None, "DESI"),
    "ch_j1622": ("SDSS J1622+3004", 15193, 9.00, 1.35, None, "SDSS"),
    "ch_j0117": ("WD J0117+1602", 15682, 8.92, 0.93, None, "DESI"),
    "ch_j0205": ("G35-26", 16338, 9.05, 0.95, None, "DESI"),
    "ch_j1803": ("WD J1803+2607", 18315, 8.65, 0.94, None, "DESI"),
    "ch_j0907": ("WD J0907+1749", 20721, 8.82, 3.00, None, "DESI"),
    "ch_j2302": ("WD J2302+0045", 21851, 8.84, 2.39, None, "DESI"),
    "ch_j0706": ("WD J0706+4833", 22353, 8.88, 3.00, None, "DESI"),
    "ch_j1104": ("WD J1104+2035", 22567, 8.81, 3.00, None, "DESI"),
    "che_j0236": ("SDSS J0236+2503", 14611, 8.777, 2.677, -1.823, "SDSS"),
    "che_j1728": ("SDSS J1728+5558", 14772, 8.869, 4.800, 0.300, "SDSS"),
    "che_j1622": ("SDSS J1622+3004", 16131, 8.934, 4.395, -0.105, "SDSS"),
    "che_j0818": ("SDSS J0818+0102", 24483, 8.326, None, 2.000, "SDSS"),
}


def main() -> None:
    from wd_spectra import WarmDQConfig, run_model

    output = Path(sys.argv[1])
    selected = sys.argv[2:] or list(MODELS)
    wavelength = np.arange(3500.0, 9900.0 + 0.1, 0.2)
    for tag in selected:
        name, teff, logg, log_c_h, log_c_he, _ = MODELS[tag]
        directory = output / tag
        if (directory / "spectrum.txt").exists():
            continue
        config = WarmDQConfig(
            effective_temperature=float(teff), logg=float(logg),
            log_carbon_to_hydrogen=log_c_h, log_carbon_to_helium=log_c_he,
        )
        run = run_model(config, directory, wavelength=wavelength)
        print(f"=== {tag} ({name}) converged={run.convergence_verified}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OPENWD_NUM_THREADS", "1")
    main()
