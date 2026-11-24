"""Frozen-structure WarmDQ spectra; cold convergence is the released-family canary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from wd_spectra import WarmDQConfig
from wd_spectra.models.common import AtmosphereConvergenceWarning
from warm_dq_regression_support import CASES, fixed_spectrum, structure_config, wavelength_grid

DIRECTORY = Path(__file__).parent / "data/warm_dq_regressions"


def test_frozen_warm_dq_controls_come_from_certified_cold_starts():
    manifest = json.loads((DIRECTORY / "manifest.json").read_text())
    assert manifest["schema"] == 1
    assert set(manifest["cases"]) == set(CASES)
    for case, record in manifest["cases"].items():
        path = DIRECTORY / f"{case}.npz"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
        assert record["config"] == CASES[case]["config"]
        source = manifest["structures"][CASES[case]["structure"]]
        assert source["config"] == CASES[CASES[case]["structure"]]["config"]
        assert source["cold_start"] is True and source["status"] == "completed"
        certificate = source["certificate"]
        assert certificate["verified"] and not certificate["failures"]
        for name in certificate["required_checks"]:
            check = certificate["checks"][name]
            assert check["measured"] and check["passed"]
            assert 0 <= check["value"] < check["tolerance"]
        with np.load(path, allow_pickle=False) as saved:
            assert hashlib.sha256(
                np.ascontiguousarray(saved["temperature"]).tobytes()
            ).hexdigest() == source["temperature_sha256"]
            for key in ("column_mass", "optical_depth", "temperature"):
                assert saved[key].shape == (48,)
                assert np.all(np.isfinite(saved[key])) and np.all(saved[key] > 0)
            np.testing.assert_array_equal(saved["wavelength"], wavelength_grid(CASES[case]["windows"]))
            assert saved["surface_flux"].shape == saved["wavelength"].shape
    # A magnetic control reuses the nonmagnetic structure of the same star.
    assert structure_config("warm-dq-j1215-uniform-10mg") == WarmDQConfig(
        effective_temperature=13574.0, logg=8.86, log_carbon_to_hydrogen=1.11)


@pytest.mark.spectral
@pytest.mark.parametrize("case", sorted(CASES))
def test_fixed_atmosphere_spectrum(case, monkeypatch):
    import wd_spectra.models.warm_dq as public

    def forbidden_solve(*args, **kwargs):
        pytest.fail("fixed-atmosphere WarmDQ spectrum started an atmosphere solve")

    monkeypatch.setattr(public, "radiative_equilibrium_d6_atmosphere", forbidden_solve)
    config = WarmDQConfig(**CASES[case]["config"])
    with np.load(DIRECTORY / f"{case}.npz", allow_pickle=False) as saved:
        with pytest.warns(AtmosphereConvergenceWarning):
            spectrum = fixed_spectrum(config, saved["column_mass"], saved["optical_depth"],
                                      saved["temperature"], saved["wavelength"])
        flux = spectrum.surface_flux_lambda
        assert np.all(np.isfinite(flux)) and np.all(flux > 0)
        np.testing.assert_allclose(flux, saved["surface_flux"], rtol=2e-6,
                                   atol=1e-12 * np.max(saved["surface_flux"]))
