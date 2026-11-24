#!/usr/bin/env python3
"""Offline capture of the frozen WarmDQ fixed-spectrum controls. Tests and CI never run it.

    python research/capture_warm_dq_controls.py OUTPUT_DIR \\
        --structure warm-dq-j1215=RUN_DIR --structure warm-dq-j0236-che=RUN_DIR

Each RUN_DIR is a completed cold start written by ``run_model`` (for example
by research/warm_dq_doc_models.py) whose configuration is the structure's
configuration in tests/warm_dq_regression_support.py and whose equilibrium
certificate passed. Its column mass, Rosseland optical depth and temperature
become the frozen structure; every case spectrum is synthesized on it with
the current source tree. OUTPUT_DIR must not exist: controls are replaced
only after review, never overwritten.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def _source(name: str, directory: Path, config) -> dict:
    run = json.loads((directory / "model-run.json").read_text())
    record = json.loads((directory / "metadata.json").read_text())
    expected = json.loads(json.dumps(asdict(config)))
    if run["effective_config"] != expected or record["config"] != expected:
        raise SystemExit(f"{directory.name} is not the {name} configuration")
    certificate = record["atmosphere_metadata"]["equilibrium_certificate"]
    if not (run["cold_start"] and run["status"] == "completed" and run["convergence_verified"]
            and certificate["verified"] and not certificate["failures"]):
        raise SystemExit(f"{directory.name} is not a certified cold start")
    with np.load(directory / "atmosphere.npz", allow_pickle=False) as saved:
        arrays = {key: np.array(saved[key], dtype=np.float64)
                  for key in ("column_mass", "rosseland_optical_depth", "temperature")}
    return dict(
        arrays=arrays,
        record=dict(
            config={key: value for key, value in run["request"].items()
                    if key in ("effective_temperature", "logg", "log_carbon_to_hydrogen",
                               "log_carbon_to_helium") and value is not None},
            run=directory.name,
            cold_start=run["cold_start"],
            status=run["status"],
            atmosphere_convergence_status=record["model_metadata"]["atmosphere_convergence_status"],
            certificate=certificate,
            atmosphere_sha256=_digest((directory / "atmosphere.npz").read_bytes()),
            temperature_sha256=_digest(np.ascontiguousarray(arrays["temperature"]).tobytes()),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path)
    parser.add_argument("--structure", action="append", default=[], metavar="NAME=RUN_DIR")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; controls are never overwritten")

    import wd_spectra
    from wd_spectra import WarmDQConfig, compiled_backend_available
    from wd_spectra.models.common import AtmosphereConvergenceWarning
    from warm_dq_regression_support import CASES, fixed_spectrum, wavelength_grid

    if not compiled_backend_available():
        raise SystemExit("the compiled transfer kernel is required")
    given = dict(item.split("=", 1) for item in args.structure)
    needed = {spec["structure"] for spec in CASES.values()}
    if set(given) != needed:
        raise SystemExit(f"give --structure for exactly {sorted(needed)}")
    sources = {name: _source(name, Path(path), WarmDQConfig(**CASES[name]["config"]))
               for name, path in given.items()}
    for name, source in sources.items():
        if source["record"]["config"] != CASES[name]["config"]:
            raise SystemExit(f"{name}: recorded request differs from the case configuration")

    args.output.mkdir(parents=True)
    cases = {}
    for case, spec in CASES.items():
        arrays = sources[spec["structure"]]["arrays"]
        wavelength = wavelength_grid(spec["windows"])
        start = time.monotonic()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AtmosphereConvergenceWarning)
            spectrum = fixed_spectrum(WarmDQConfig(**spec["config"]), arrays["column_mass"],
                                      arrays["rosseland_optical_depth"], arrays["temperature"],
                                      wavelength)
        path = args.output / f"{case}.npz"
        np.savez(path, column_mass=arrays["column_mass"],
                 optical_depth=arrays["rosseland_optical_depth"],
                 temperature=arrays["temperature"], wavelength=wavelength,
                 surface_flux=np.asarray(spectrum.surface_flux_lambda, dtype=np.float64))
        cases[case] = dict(config=spec["config"], structure=spec["structure"],
                           pixels=int(wavelength.size), sha256=_digest(path.read_bytes()),
                           synthesis_seconds=round(time.monotonic() - start, 1))
        print(case, cases[case]["synthesis_seconds"], "s", flush=True)

    manifest = dict(
        schema=1,
        scope="Fixed-structure WarmDQ spectra (tests/test_warm_dq_regression.py); "
              "cold convergence is the warm-dq released-family canary.",
        capture=dict(
            script="research/capture_warm_dq_controls.py",
            commit=_git("rev-parse", "HEAD"),
            uncommitted_source_changes=bool(_git("status", "--porcelain", "--", "src")),
            package_version=wd_spectra.__version__,
            python=platform.python_version(),
            numpy=np.__version__,
            platform=platform.platform(),
        ),
        structures={name: source["record"] for name, source in sources.items()},
        cases=cases,
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
