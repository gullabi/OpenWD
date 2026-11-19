"""Capture the DZ and DAZ fixed-spectrum controls for the HeH+ and Kirchhoff Ca II defaults.

Since 2026-11-18, DZConfig defaults to helium_hydride_ion=True (H+ + He <-> HeH+
in the trace-hydrogen ionization balance) and DZConfig and DAZConfig default to
ca_ii_crd_frequency_dependent_source=True (Ca II H&K complete-redistribution
emission kappa_L S_L B_nu(nu)/B_nu(nu0), i.e. Kirchhoff's law across the
broadened profile).  The frozen structures, configurations and wavelength grids
are unchanged.  For each case the script first proves that the stored
configuration with both options off reproduces the current approved control at
the test tolerances, then synthesizes the control with the current defaults.
No atmosphere is re-relaxed; these are not cold-start qualifications.

The script refuses an existing output directory and is never run by tests or
CI. Copy the written files into tests/data/approved_regressions/current/ and
record manifest.json in the pull request.

usage: python research/capture_heh_kirchhoff_default_controls.py OUTDIR
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import warnings
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
             "MKL_NUM_THREADS", "OPENWD_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from wd_spectra._compat import trapezoid  # noqa: E402
from wd_spectra.models import (  # noqa: E402
    AtmosphereConvergenceWarning,
    DAZConfig,
    DZConfig,
    compute_daz,
    compute_dz,
    load_atmosphere_checkpoint,
)
from wd_spectra.models.common import config_from_stored  # noqa: E402

DATA = ROOT / "tests/data"
CURRENT = DATA / "approved_regressions/current"
CASES = {
    "dz-pg1225": ("DZ", DATA / "spectral_regressions/dz-pg1225.npz"),
    "dz-j0738": ("DZ", DATA / "spectral_regressions/dz-j0738.npz"),
    "daz-g149_28": ("DAZ", DATA / "daz_regressions/g149_28.npz"),
    "daz-galex1931": ("DAZ", DATA / "daz_regressions/galex1931.npz"),
}
PREVIOUS = {
    "DZ": {"helium_hydride_ion": False, "ca_ii_crd_frequency_dependent_source": False},
    "DAZ": {"ca_ii_crd_frequency_dependent_source": False},
}
SOURCES = (
    "src/wd_spectra/molecules.py",
    "src/wd_spectra/metals.py",
    "src/wd_spectra/atmosphere.py",
    "src/wd_spectra/spectrum.py",
    "src/wd_spectra/cool_metal_nlte.py",
    "src/wd_spectra/models/stellar.py",
    "src/wd_spectra/models/daz.py",
    "src/wd_spectra/models/common.py",
    "research/capture_heh_kirchhoff_default_controls.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthesize(kind, path, config, wave):
    if kind == "DZ":
        atmosphere = load_atmosphere_checkpoint(
            path, config.effective_temperature, config.logg, "helium",
            include_molecules=False, include_negative_hydrogen=False,
            log_hydrogen_to_helium=getattr(config, "log_hydrogen_to_helium", None),
            trihydrogen_ion_partition_model=None,
        )
        compute = compute_dz
    else:
        molecules = config.effective_temperature <= 12000
        atmosphere = load_atmosphere_checkpoint(
            path, config.effective_temperature, config.logg, "hydrogen",
            include_molecules=molecules, include_negative_hydrogen=molecules,
            trihydrogen_ion_partition_model="neale-tennyson-1995",
        )
        compute = compute_daz
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AtmosphereConvergenceWarning)
        result = compute(config, wave, initial_atmosphere=atmosphere, relax_atmosphere=False)
    flux = result.spectrum.surface_flux_lambda
    assert np.all(np.isfinite(flux)) and np.all(flux > 0)
    assert result.spectrum.metadata["source_converged"]
    return flux


def main() -> None:
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    assert DZConfig().helium_hydride_ion and DZConfig().ca_ii_crd_frequency_dependent_source
    assert DAZConfig().ca_ii_crd_frequency_dependent_source
    records = {}
    for case, (kind, path) in CASES.items():
        config_type = DZConfig if kind == "DZ" else DAZConfig
        with np.load(path) as saved:
            stored = json.loads(str(saved["config_json"]))
            wave = saved["wavelength"]
        # As in the regression tests: the structures store the earlier "stout"
        # strong-line data; the controls use the current default.
        stored.pop("strong_line_atomic_data", None)
        assert not set(PREVIOUS[kind]) & set(stored), case
        with np.load(CURRENT / (case + ".npz")) as approved:
            np.testing.assert_array_equal(wave, approved["wavelength"])
            old = approved["surface_flux"]
        significant = wave * old > 0.01 * np.max(wave * old)
        # 1. Both options off reproduce the current control at the test tolerance.
        previous = synthesize(kind, path, config_from_stored(config_type, {**stored, **PREVIOUS[kind]}), wave)
        if kind == "DZ":
            np.testing.assert_allclose(previous, old, rtol=2e-6, atol=1e-12 * np.max(old))
        else:
            np.testing.assert_allclose(previous[significant], old[significant], rtol=1e-3, atol=0)
        # 2. The same structure, configuration and grid with the current defaults.
        new = synthesize(kind, path, config_from_stored(config_type, stored), wave)
        target = out / (case + ".npz")
        np.savez_compressed(target, wavelength=wave, surface_flux=new)
        ratio = new[significant] / old[significant]
        bands = {}
        for lo, hi in ((1150, 3000), (3000, 4000), (4000, 7000), (7000, 300000)):
            take = (wave >= lo) & (wave <= hi)
            if np.count_nonzero(take) >= 2:
                bands[f"{lo}-{hi}"] = float(
                    trapezoid(new[take] - old[take], wave[take]) / trapezoid(old[take], wave[take])
                )
        worst = int(np.argmax(np.abs(np.log(ratio))))
        records[case] = dict(
            family=kind,
            frozen_structure=str(path.relative_to(ROOT)),
            frozen_structure_sha256=sha256(path),
            previous_control_sha256=sha256(CURRENT / (case + ".npz")),
            new_control_sha256=sha256(target),
            wavelength_sha256=hashlib.sha256(wave.tobytes()).hexdigest(),
            previous_settings=PREVIOUS[kind],
            previous_reproduced_at_test_tolerance=True,
            atmosphere_reconverged=False,
            significant_pixels=int(np.count_nonzero(significant)),
            maximum_relative_change=float(ratio[worst] - 1.0),
            maximum_change_wavelength_angstrom=float(wave[significant][worst]),
            pixels_changed_over_0p1_percent=int(np.count_nonzero(np.abs(ratio - 1.0) > 1e-3)),
            integrated_relative_change=bands,
        )
        print(case, json.dumps(records[case]["integrated_relative_change"]),
              f"max {records[case]['maximum_relative_change']:+.4f} at "
              f"{records[case]['maximum_change_wavelength_angstrom']:.2f} A", flush=True)
    manifest = dict(
        reason=("DZConfig defaults to helium_hydride_ion=True; DZConfig and DAZConfig default to "
                "ca_ii_crd_frequency_dependent_source=True"),
        git_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                  text=True).stdout.strip(),
        git_dirty=bool(subprocess.run(["git", "status", "--porcelain", "--", "src"], cwd=ROOT,
                                      capture_output=True, text=True).stdout.strip()),
        source_sha256={name: sha256(ROOT / name) for name in SOURCES},
        python=platform.python_version(),
        numpy=np.__version__,
        cases=records,
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
