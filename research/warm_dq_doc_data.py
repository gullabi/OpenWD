#!/usr/bin/env python3
"""Fetch the public spectra used in docs/models/WarmDQ.md.

    python research/warm_dq_doc_data.py OUTPUT_DIR

DESI DR1 (iron) healpix coadds are read by HTTP range requests and the row of
each star (matched by Gaia DR3 source_id or within 2 arcsec) is saved with
its resolution matrix as OUTPUT_DIR/desi/<name>_<survey>-<program>.npz. SDSS
DR17 "lite" spectra are saved by plate-MJD-fiber as
OUTPUT_DIR/sdss/spec-PPPP-MMMMM-FFFF.npz. Existing files are skipped. Needs
astropy (with fsspec) and healpy, which are not OpenWD dependencies.
"""

from __future__ import annotations

import io
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

DESI_BASE = "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/healpix"
DESI_COMBINATIONS = [("main", program) for program in ("bright", "dark", "backup")]
# DESI name: Gaia DR3 source_id
DESI_STARS = {
    "WDJ014656.90-042613.50": 2480243386083359360,
    "WDJ121510.65+470011.25": 1545134482103819136,
    "WDJ032436.97-013506.16": 3262453756428156544,
    "WDJ230025.55+084838.01": 2713837689421703552,
    "WDJ144854.90+051903.80": 1159147901516125440,
    "WDJ011731.77+160239.28": 2591704384484374528,
    "WDJ020549.70+205707.96": 94276941624384000,
    "WDJ180300.29+260738.14": 4582662626607120256,
    "WDJ090748.93+174903.20": 635307507975401472,
    "WDJ230245.05+004506.00": 2652157079967798912,
    "WDJ070625.00+483339.26": 978257588544163200,
    "WDJ110406.70+203528.73": 3985258509955943808,
}
# SDSS plate-MJD-fiber of 1622+3004, 1728+5558, 0236+2503 and 0818+0102
SDSS_SPECTRA = ("4953-55749-0483", "0358-51818-0296", "2399-53764-0059", "2077-53846-0575")
SDSS_URLS = (
    "https://data.sdss.org/sas/dr17/eboss/spectro/redux/v5_13_2/spectra/lite/{p:04d}/spec-{p:04d}-{m}-{f:04d}.fits",
    "https://data.sdss.org/sas/dr17/sdss/spectro/redux/26/spectra/lite/{p:04d}/spec-{p:04d}-{m}-{f:04d}.fits",
)


def _radec(name: str) -> tuple[float, float]:
    h, mi, s, sign, d, dm, ds = re.match(
        r"WDJ(\d\d)(\d\d)(\d\d\.\d+)([+-])(\d\d)(\d\d)(\d\d\.\d+)", name
    ).groups()
    ra = 15.0 * (int(h) + int(mi) / 60 + float(s) / 3600)
    dec = (1 if sign == "+" else -1) * (int(d) + int(dm) / 60 + float(ds) / 3600)
    return ra, dec


def _exists(url: str) -> bool:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            return r.status == 200
    except urllib.error.HTTPError:
        return False


def fetch_desi(output: Path) -> None:
    import healpy as hp
    from astropy.io import fits

    output.mkdir(parents=True, exist_ok=True)
    for name, gaia in DESI_STARS.items():
        if list(output.glob(f"{name}_*.npz")):
            continue
        ra, dec = _radec(name)
        pixel = int(hp.ang2pix(64, ra, dec, lonlat=True, nest=True))
        for survey, program in DESI_COMBINATIONS:
            url = f"{DESI_BASE}/{survey}/{program}/{pixel // 100}/{pixel}/coadd-{survey}-{program}-{pixel}.fits"
            if not _exists(url):
                continue
            with fits.open(url, use_fsspec=True, lazy_load_hdus=True) as hdul:
                fibermap = hdul["FIBERMAP"].data
                separation = np.hypot(
                    (fibermap["TARGET_RA"] - ra) * np.cos(np.radians(dec)),
                    fibermap["TARGET_DEC"] - dec,
                ) * 3600.0
                rows = np.flatnonzero((fibermap["REF_ID"] == gaia) | (separation < 2.0))
                if rows.size == 0:
                    continue
                row = int(rows[0])
                record = {}
                for arm in "BRZ":
                    record[f"{arm}_w"] = np.asarray(hdul[f"{arm}_WAVELENGTH"].data, dtype=float)
                    record[f"{arm}_f"] = np.asarray(hdul[f"{arm}_FLUX"].section[row, :], dtype=float)
                    record[f"{arm}_iv"] = np.asarray(hdul[f"{arm}_IVAR"].section[row, :], dtype=float)
                    record[f"{arm}_mask"] = np.asarray(hdul[f"{arm}_MASK"].section[row, :])
                    record[f"{arm}_res"] = np.asarray(
                        hdul[f"{arm}_RESOLUTION"].section[row, :, :], dtype=float
                    )
                np.savez_compressed(output / f"{name}_{survey}-{program}.npz", **record)
                print("saved", name, survey, program, flush=True)


def fetch_sdss(output: Path) -> None:
    from astropy.io import fits

    output.mkdir(parents=True, exist_ok=True)
    for pmf in SDSS_SPECTRA:
        plate, mjd, fiber = (int(x) for x in pmf.split("-"))
        target = output / f"spec-{plate:04d}-{mjd}-{fiber:04d}.npz"
        if target.exists():
            continue
        for template in SDSS_URLS:
            try:
                with urllib.request.urlopen(template.format(p=plate, m=mjd, f=fiber), timeout=120) as r:
                    raw = r.read()
            except Exception:
                continue
            with fits.open(io.BytesIO(raw)) as hdul:
                table = hdul[1].data
                np.savez_compressed(
                    target, wave_vac=10 ** table["loglam"], flux=table["flux"],
                    ivar=table["ivar"], and_mask=table["and_mask"], wdisp=table["wdisp"],
                )
            print("saved", target.name, flush=True)
            break


if __name__ == "__main__":
    root = Path(sys.argv[1])
    fetch_desi(root / "desi")
    fetch_sdss(root / "sdss")
