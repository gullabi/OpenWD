"""Capture the controls moved by mixed-EOS HeH+ and the continuous He- free-free default.

Since 2026-11-19 two shared defaults changed:

* ``eos.MIXED_HELIUM_HYDRIDE_ION_DEFAULT = True``: H+ + He <-> HeH+ in the
  homogeneous atomic and molecular H/He EOS (DAB/DBA, and the LTE state of
  the sdB adapter).
* ``helium.HELIUM_MINUS_FREE_FREE_DEFAULT = "john1968-shape"``: John (1994)
  He- free-free where tabulated, and below his 0.5063-micron edge John's edge
  value times the Carbon et al. (1969)/John (1968) wavelength shape.  The
  previous "automatic" switched to the Carbon fit itself at the edge, leaving a
  flux step at 5063 A in cool helium-rich models.  DZ and DQ choose
  "partial-wave" explicitly and are unchanged.

For each DAB case the script first proves that the previous defaults reproduce
the approved control at the test tolerance on the unchanged frozen structure
and wavelength grid, then synthesizes it with the current defaults.  No
atmosphere is re-relaxed.

For the sdB fixed state (23,200 K) HeH+ moves the LTE populations by ~3e-9.
The script proves the previous defaults reproduce the stored LTE populations
at the test tolerance, recomputes them with the current defaults, keeps the
stored NLTE departure coefficients (population / lte_population), and checks
that the emergent flux still matches ``hd4539-hhe.npz`` at the test's 1e-7.
Only ``hd4539-fixed-state.npz`` is rewritten; the flux reference is retained.

The script refuses an existing output directory and is never run by tests or
CI.  Copy the written files into tests/data and record manifest.json.

usage: python research/capture_heh_mixed_heminus_controls.py OUTDIR
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import warnings
from dataclasses import replace
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
             "MKL_NUM_THREADS", "OPENWD_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

import wd_spectra.eos as eos  # noqa: E402
import wd_spectra.helium as helium  # noqa: E402
from wd_spectra import ModelData, SdBConfig  # noqa: E402
from wd_spectra._compat import trapezoid  # noqa: E402
from wd_spectra.atmosphere import gray_hydrogen_helium_atmosphere  # noqa: E402
from wd_spectra.hot_nlte import population_arrays, with_departures  # noqa: E402
from wd_spectra.models import DABConfig, compute_dab, load_atmosphere_checkpoint, sdb  # noqa: E402
from wd_spectra.models.common import config_from_stored  # noqa: E402

DATA = ROOT / "tests/data"
CURRENT = DATA / "approved_regressions/current"
SDB = DATA / "sdb_regressions"
DAB_CASES = ("dab-9000", "dab-20000")
PREVIOUS = {"MIXED_HELIUM_HYDRIDE_ION_DEFAULT": False, "HELIUM_MINUS_FREE_FREE_DEFAULT": "automatic"}
CURRENT_DEFAULTS = {"MIXED_HELIUM_HYDRIDE_ION_DEFAULT": True, "HELIUM_MINUS_FREE_FREE_DEFAULT": "john1968-shape"}
SOURCES = (
    "src/wd_spectra/eos.py",
    "src/wd_spectra/_mixed_molecules.py",
    "src/wd_spectra/helium.py",
    "src/wd_spectra/atmosphere.py",
    "src/wd_spectra/spectrum.py",
    "src/wd_spectra/molecules.py",
    "src/wd_spectra/models/common.py",
    "research/capture_heh_mixed_heminus_controls.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_defaults(values) -> None:
    eos.MIXED_HELIUM_HYDRIDE_ION_DEFAULT = values["MIXED_HELIUM_HYDRIDE_ION_DEFAULT"]
    helium.HELIUM_MINUS_FREE_FREE_DEFAULT = values["HELIUM_MINUS_FREE_FREE_DEFAULT"]


def synthesize_dab(path, config, wave):
    atmosphere = load_atmosphere_checkpoint(
        path, config.effective_temperature, config.logg, "mixed",
        include_molecules=False, include_negative_hydrogen=False,
        log_hydrogen_to_helium=config.log_hydrogen_to_helium,
        trihydrogen_ion_partition_model=None,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = compute_dab(config, wave, initial_atmosphere=atmosphere, relax_atmosphere=False)
    flux = result.spectrum.surface_flux_lambda
    assert np.all(np.isfinite(flux)) and np.all(flux > 0)
    assert result.spectrum.metadata["source_converged"]
    return flux


def capture_dab(case, out):
    path = DATA / "spectral_regressions" / (case + ".npz")
    with np.load(path) as saved:
        assert str(saved["spectral_type"]) == "DAB"
        config = config_from_stored(DABConfig, json.loads(str(saved["config_json"])))
        wave = saved["wavelength"]
    with np.load(CURRENT / (case + ".npz")) as approved:
        np.testing.assert_array_equal(wave, approved["wavelength"])
        old = approved["surface_flux"]
    set_defaults(PREVIOUS)
    previous = synthesize_dab(path, config, wave)
    np.testing.assert_allclose(previous, old, rtol=2e-6, atol=1e-12 * np.max(old))
    set_defaults(CURRENT_DEFAULTS)
    new = synthesize_dab(path, config, wave)
    target = out / (case + ".npz")
    np.savez_compressed(target, wavelength=wave, surface_flux=new)
    significant = wave * old > 0.01 * np.max(wave * old)
    ratio = new[significant] / old[significant]
    worst = int(np.argmax(np.abs(np.log(ratio))))
    bands = {}
    for lo, hi in ((1150, 3000), (3000, 5063), (5063, 7000), (7000, 300000)):
        take = (wave >= lo) & (wave <= hi)
        bands[f"{lo}-{hi}"] = float(trapezoid(new[take] - old[take], wave[take]) / trapezoid(old[take], wave[take]))
    i = int(np.searchsorted(wave, 5063.0))
    record = dict(
        frozen_structure=str(path.relative_to(ROOT)), frozen_structure_sha256=sha256(path),
        previous_control_sha256=sha256(CURRENT / (case + ".npz")), new_control_sha256=sha256(target),
        previous_reproduced_at_test_tolerance=True, atmosphere_reconverged=False,
        significant_pixels=int(np.count_nonzero(significant)),
        maximum_relative_change=float(ratio[worst] - 1.0),
        maximum_change_wavelength_angstrom=float(wave[significant][worst]),
        pixels_changed_over_0p1_percent=int(np.count_nonzero(np.abs(ratio - 1.0) > 1e-3)),
        integrated_relative_change=bands,
        log_flux_jump_across_5063_angstrom={
            "previous": float(np.log(old[i] / old[i - 1])), "new": float(np.log(new[i] / new[i - 1]))},
    )
    print(case, f"max {record['maximum_relative_change']:+.5f} at {record['maximum_change_wavelength_angstrom']:.1f} A",
          json.dumps(bands), flush=True)
    return record


def sdb_lte(state):
    config = SdBConfig()
    model = sdb._nlte_model(config, ModelData.default())
    template = gray_hydrogen_helium_atmosphere(
        config.effective_temperature, config.logg, -config.log_helium_to_hydrogen,
        n_depth=state["temperature"].size)
    template = replace(template, column_mass=state["column_mass"], gas_pressure=state["gas_pressure"],
                       rosseland_optical_depth=state["rosseland_optical_depth"])
    atmosphere = model.rebuild_atmosphere(template, state["temperature"])
    return model, atmosphere, model._rate_state(atmosphere)


def capture_sdb(out):
    source = SDB / "hd4539-fixed-state.npz"
    with np.load(source, allow_pickle=False) as saved:
        state = dict(saved)
    set_defaults(PREVIOUS)
    _, _, lte = sdb_lte(state)
    np.testing.assert_allclose(population_arrays(lte)[1], state["lte_population"], rtol=1e-10)
    set_defaults(CURRENT_DEFAULTS)
    model, atmosphere, lte = sdb_lte(state)
    new_lte = np.asarray(population_arrays(lte)[1])
    departures = state["population"] / state["lte_population"]
    new_state = dict(state, lte_population=new_lte, population=departures * new_lte)
    target = out / "hd4539-fixed-state.npz"
    np.savez_compressed(target, **new_state)
    with np.load(target, allow_pickle=False) as check:
        populations = with_departures(lte, check["population"] / check["lte_population"])
    with np.load(SDB / "hd4539-hhe.npz", allow_pickle=False) as reference:
        flux = sdb._emergent_flux(model, atmosphere, populations, reference["wavelength"], lines=True)
        flux_change = float(np.max(np.abs(flux / reference["flux"] - 1.0)))
        np.testing.assert_allclose(flux, reference["flux"], rtol=1e-7, atol=0)
    record = dict(
        previous_state_sha256=sha256(source), new_state_sha256=sha256(target),
        flux_reference_sha256_retained=sha256(SDB / "hd4539-hhe.npz"),
        previous_lte_populations_reproduced_at_test_tolerance=True,
        maximum_lte_population_relative_change=float(np.max(np.abs(new_lte / state["lte_population"] - 1.0))),
        departure_coefficients_retained=True,
        maximum_emergent_flux_relative_change=flux_change,
        flux_reference_retained_byte_for_byte=True,
    )
    print("sdb-hd4539", json.dumps(record), flush=True)
    return record


def main() -> None:
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    assert eos.MIXED_HELIUM_HYDRIDE_ION_DEFAULT is True
    assert helium.HELIUM_MINUS_FREE_FREE_DEFAULT == "john1968-shape"
    records = {case: capture_dab(case, out) for case in DAB_CASES}
    records["sdb-hd4539"] = capture_sdb(out)
    set_defaults(CURRENT_DEFAULTS)
    manifest = dict(
        reason=("Mixed H/He EOS includes HeH+ by default; He- free-free default is john1968-shape "
                "(continuous at John's 0.5063-micron edge)"),
        previous_defaults=PREVIOUS, current_defaults=CURRENT_DEFAULTS,
        git_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        git_dirty=bool(subprocess.run(["git", "status", "--porcelain", "--", "src"], cwd=ROOT,
                                      capture_output=True, text=True).stdout.strip()),
        source_sha256={name: sha256(ROOT / name) for name in SOURCES},
        python=platform.python_version(), numpy=np.__version__,
        cases=records,
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
