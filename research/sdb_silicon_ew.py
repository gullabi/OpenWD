#!/usr/bin/env python3
"""Equivalent widths of the Si II 4128/4131 and Si III 4552-75 lines in sdB metal runs.

    python research/sdb_silicon_ew.py RUN_DIR [RUN_DIR ...]

EW = integral of (1 - flux/host_flux) over +-HALF A around each vacuum centre,
so the H/He host (Hdelta wing) is divided out. Run on silicon-only runs: other
metals blend (O II 4132.8 next to Si II 4131).
"""
import sys
from pathlib import Path
import numpy as np

AIR = {'Si II 4128': 4128.054, 'Si II 4131': 4130.894, 'Si III 4552': 4552.622,
       'Si III 4568': 4567.840, 'Si III 4575': 4574.757, 'Si III 4813': 4813.33,
       'Si III 4829': 4828.96, 'Si IV 4089': 4088.862, 'Si IV 4116': 4116.104,
       'Si II 6347': 6347.10, 'Si II 6371': 6371.36, 'Si II 3856': 3856.02, 'Si II 5041': 5041.024,
       'Si II 5056': 5055.98}
HALF = 1.2


def vacuum(air):
    s2 = (1e4 / air) ** 2
    return air * (1 + 0.0000834254 + 0.02406147 / (130 - s2) + 0.00015998 / (38.9 - s2))


def widths(run, key="flux"):
    d = np.load(Path(run) / 'spectrum.npz')
    w, f, h = d["wavelength_vacuum"], d[key], d["host_flux"]
    out = {}
    for name, air in AIR.items():
        c = vacuum(air)
        m = (w > c - HALF) & (w < c + HALF)
        if m.sum() < 5:
            continue
        out[name] = 1e3 * np.trapezoid(1 - f[m] / h[m], w[m]) if hasattr(np, 'trapezoid') else 1e3 * np.trapz(1 - f[m] / h[m], w[m])
    return out


def main():
    for run in sys.argv[1:]:
      keys = ['flux'] + (['lte_flux'] if 'lte_flux' in np.load(Path(run) / 'spectrum.npz') else [])
      for key in keys:
        ew = widths(run, key)
        si2 = ew['Si II 4128'] + ew['Si II 4131']
        si3 = ew['Si III 4552'] + ew['Si III 4568'] + ew['Si III 4575']
        print(f'== {run} [{key}]')
        print('  ' + '  '.join(f'{k} {v:.1f}' for k, v in ew.items()) + '  (mA)')
        print(f'  sum SiII(4128+4131)/SiIII(4552+68+75) = {si2 / si3:.3f};  4128/4552 = '
              f'{ew["Si II 4128"] / ew["Si III 4552"]:.3f}')


if __name__ == '__main__':
    main()
