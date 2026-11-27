#!/usr/bin/env python3
"""Observed vs model Si II / Si III equivalent widths for a star with an X-shooter co-add.

    python research/sdb_silicon_ew_observed.py OBS_DIR RUN [RUN ...]

OBS_DIR holds coadd-uvb.fits (research/sdb_coadd_xshooter.py); each RUN is an
sdb_trace_metals.py output directory.

The observation is divided by the run's convolved H/He host spectrum (which carries the
Hdelta wing), renormalized by a linear fit to the window edges, then integrated.
The model is treated identically (convolved metal+host / convolved host).
RV: fitted to the Si III 4552-75 cores of the first run.
"""
import sys
from pathlib import Path
import numpy as np
from astropy.io import fits
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sdb_compare_hd4539 import vacuum_to_air, gaussian_convolve
C = 299792.458
R = 9861.0
LINES = {'Si II 4128': 4128.054, 'Si II 4131': 4130.894, 'Si III 4552': 4552.622, 'Si III 4568': 4567.840,
         'Si III 4575': 4574.757, 'Si II 5041': 5041.024, 'Si II 5056': 5055.98, 'Si III 4813': 4813.33,
         'Si III 4829': 4828.96, 'Si IV 4089': 4088.862, 'Si IV 4116': 4116.104}
HALF, EDGE = 1.6, 0.5   # integrate +-HALF; continuum from HALF..HALF+EDGE on both sides


def load(obs):
    d = fits.getdata(Path(obs) / 'coadd-uvb.fits', 1)
    g = np.isfinite(d['FLUX']) & np.isfinite(d['ERR']) & (d['ERR'] > 0)
    return d['WAVE'][g], d['FLUX'][g], d['ERR'][g]


def model(run):
    s = np.load(Path(run) / 'spectrum.npz')
    air = vacuum_to_air(s['wavelength_vacuum'])
    out = {}
    for key in ('flux', 'host_flux', 'lte_flux'):
        if key in s:
            # windows are disjoint: convolve each contiguous block separately
            breaks = np.flatnonzero(np.diff(air) > 1.0) + 1
            ws, fs = [], []
            for a, f in zip(np.split(air, breaks), np.split(s[key], breaks)):
                w, c = gaussian_convolve(a, f, R); ws.append(w); fs.append(c)
            out[key] = (np.concatenate(ws), np.concatenate(fs))
    return out


def ew(wave, norm, centre):
    m = np.abs(wave - centre) <= HALF
    integrate = getattr(np, 'trapezoid', None) or np.trapz
    return 1e3 * integrate(1 - norm[m], wave[m])


def normalized(wave, flux, host, centre):
    ratio = flux / host
    edge = (np.abs(wave - centre) > HALF) & (np.abs(wave - centre) <= HALF + EDGE)
    p = np.polyfit(wave[edge] - centre, ratio[edge], 1)
    return ratio / np.polyval(p, wave - centre)


def measure(obsw, obsf, mod, rv, key):
    res = {}
    hw, hf = mod['host_flux']
    mw, mf = mod[key]
    for name, c in LINES.items():
        sel = np.abs(obsw - c) <= HALF + EDGE + 0.2
        w = obsw[sel]
        host = np.interp(w, hw * (1 + rv / C), hf)
        if obsf is not None:
            res[name] = ew(w, normalized(w, obsf[sel], host, c), c)
        else:
            res[name] = ew(w, normalized(w, np.interp(w, mw * (1 + rv / C), mf), host, c), c)
    return res


def fit_rv(obsw, obsf, mod):
    best = None
    for rv in np.arange(-80, 80.1, 1.0):
        chi = 0
        hw, hf = mod['host_flux']; mw, mf = mod['flux']
        for name in ('Si III 4552', 'Si III 4568', 'Si III 4575'):
            c = LINES[name]; sel = np.abs(obsw - c) <= HALF + EDGE
            w = obsw[sel]; host = np.interp(w, hw * (1 + rv / C), hf)
            o = normalized(w, obsf[sel], host, c); m = normalized(w, np.interp(w, mw * (1 + rv / C), mf), host, c)
            chi += np.sum((o - m) ** 2)
        if best is None or chi < best[0]:
            best = (chi, rv)
    return best[1]


def summary(label, e):
    s2 = e['Si II 4128'] + e['Si II 4131']; s3 = e['Si III 4552'] + e['Si III 4568'] + e['Si III 4575']
    print(f'{label:34s} ' + ' '.join(f'{e[k]:6.1f}' for k in LINES) + f'   {s2 / s3:6.3f} {e["Si II 4128"] / e["Si III 4552"]:6.3f}')


def main():
    obsw, obsf, obse = load(sys.argv[1])
    runs = sys.argv[2:]
    mods = [model(r) for r in runs]
    rv = fit_rv(obsw, obsf, mods[0])
    print(f'RV {rv:+.0f} km/s; EW (mA), +-{HALF} A; last two columns: SiII(4128+31)/SiIII(4552+68+75), 4128/4552')
    print(' ' * 35 + ' '.join(f'{k.split()[-1]:>6s}' for k in LINES))
    summary('observed', measure(obsw, obsf, mods[0], rv, 'flux'))
    for r, m in zip(runs, mods):
        summary(Path(r).name + ' nlte', measure(obsw, None, m, rv, 'flux'))
        if 'lte_flux' in m:
            summary(Path(r).name + ' lte', measure(obsw, None, m, rv, 'lte_flux'))


if __name__ == '__main__':
    main()
