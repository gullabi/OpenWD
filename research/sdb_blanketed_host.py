#!/usr/bin/env python3
"""Small experiment: metal-blanketed LTE T(m) transplanted into an sdB H/He host.

    python research/sdb_blanketed_host.py HOST_LTE_DIR OUT_DIR --abundance C=-4.06 ...

Solves an LTE DAZ (H + metals as structure opacity, no He) at the host Teff/log g,
then writes an sdB-driver 'LTE dir' with the host column mass and gas pressure and
the DAZ temperature at the same column mass. The hybrid step rebuilds n_e with the
H/He EOS. Records the DAZ run time.
"""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import wd_spectra, wd_spectra.metals as _m
assert _m._rt is not None and Path(wd_spectra.__file__).resolve().is_relative_to(ROOT / 'src'), 'run with this tree on the path and the C kernel built'
from wd_spectra.models.common import ModelData
from wd_spectra.models.daz import DAZConfig, compute_daz

p = argparse.ArgumentParser()
p.add_argument('host', type=Path); p.add_argument('out', type=Path)
p.add_argument('--abundance', action='append', required=True)
a = p.parse_args()
cfg = json.loads((a.host / 'run-summary.json').read_text())['config']
abund = {k: float(v) for k, v in (x.split('=') for x in a.abundance)}
start = time.monotonic()
res = compute_daz(DAZConfig(effective_temperature=cfg['teff'], logg=cfg['logg'], abundances=abund),
                  np.array([4000.0, 5000.0]), data=ModelData.default(),
                  iteration_callback=lambda i, atm, d: print(i, f'{time.monotonic() - start:.0f}s', flush=True))
elapsed = time.monotonic() - start
z = res.atmosphere
host = np.load(a.host / 'atmosphere.npz')
m = host['column_mass']
T = np.interp(np.log(m), np.log(z.column_mass), z.temperature)
if m[0] < z.column_mass[0] or m[-1] > z.column_mass[-1]:
    print('warning: host column-mass range exceeds the DAZ model; edge temperatures are held constant')
a.out.mkdir(parents=True)
np.savez(a.out / 'atmosphere.npz', temperature=T, column_mass=m, gas_pressure=host['gas_pressure'],
         rosseland_optical_depth=host['rosseland_optical_depth'], electron_density=host['electron_density'])
summary = json.loads((a.host / 'run-summary.json').read_text())
summary.update(note='T(m) of an LTE DAZ (metals as structure opacity, no He) on the host column mass and P_gas',
               daz_abundances=abund, daz_elapsed_seconds=elapsed,
               daz_convergence=res.metadata.get('atmosphere_convergence_status'))
(a.out / 'run-summary.json').write_text(json.dumps(summary, indent=1, default=str))
for x in (1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0):
    t0 = np.interp(np.log(x), np.log(m), host['temperature']); t1 = np.interp(np.log(x), np.log(m), T)
    print(f'm={x:7.0e}  T(H/He LTE)={t0:7.0f}  T(blanketed)={t1:7.0f}  {100 * (t1 / t0 - 1):+5.2f}%')
print(f'DAZ structure: {elapsed:.0f} s, {res.metadata.get("atmosphere_convergence_status")}')
