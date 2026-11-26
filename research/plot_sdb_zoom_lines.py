#!/usr/bin/env python3
"""sdB guide figure: two ~100 A windows of HD 4539 and LS IV-14 116 with labelled metal lines.

    python research/plot_sdb_zoom_lines.py --models results/sdb/paper --output docs/assets

Uses the models, observations and normalization of the validation figure
(research/plot_sdb_paper_comparison.py; models from research/sdb_paper_models.sh).

Line labels are identified, not typed in: every absorption feature of the model
deeper than a threshold is matched to the lines within 0.3 A (Stout lines of the
model's metals, the heavy-element lines of sdb_heavy_lines.py, and a short He list).
Candidates are ranked by an LTE line-opacity estimate, n_ion g f exp(-E_l/kT)/U, at
Rosseland optical depth 0.3 of the model's own structure.  A second line within
30% of the strongest joins the label.  Labels give air wavelengths.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import percentile_filter
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_sdb_paper_comparison as paper  # noqa: E402
from sdb_heavy_lines import HeavyLines  # noqa: E402
from sdb_trace_metals import NIST_IONIZATION_EV, load_host  # noqa: E402
from wd_spectra.hot_trace_metals import fixed_electron_metal_reference  # noqa: E402
from wd_spectra.metals import read_pg1159_atomic_database  # noqa: E402
from wd_spectra.models.common import ModelData  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OBSERVED, MODEL = '.30', '#d95f02'
TICK_SIZE, LABEL_SIZE, LINE_LABEL_SIZE = 8, 9, 6.3
WINDOWS = ((4192.0, 4300.0), (4995.0, 5100.0))   # air A, covered by both spectrographs
STARS = ('hd4539', 'lsiv14116')
MINIMUM_DEPTH = {'hd4539': 0.01, 'lsiv14116': 0.015}
MATCH = 0.3          # A
BLEND_FRACTION = 0.3
TAU = 0.3
BOLTZMANN, WAVENUMBER_TO_ERG = 1.380649e-16, 6.62607015e-27 * 2.99792458e10
# He lines in the windows (air A, lower-level energy cm-1, log gf; NIST).
HELIUM = ((4199.83, 1, 410006.8, 0.07), (5015.68, 0, 166277.4, -0.82), (5047.74, 0, 171134.9, -1.60))


def roman(spectrum):
    numerals = ((10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I'))
    out = ''
    for value, symbol in numerals:
        while spectrum >= value:
            out, spectrum = out + symbol, spectrum - value
    return out


def vacuum_to_air(vacuum):
    s2 = (1e4 / vacuum) ** 2
    return vacuum / (1 + 0.0000834254 + 0.02406147 / (130 - s2) + 0.00015998 / (38.9 - s2))


def candidate_lines(key, summary, data):
    """(air A, species, strength) for every Stout, heavy and He line in the windows."""
    host = summary['host']
    _, atmosphere, _, _ = load_host(ROOT / host['lte'], ROOT / host['hybrid'], data)
    depth = int(np.argmin(np.abs(np.log(np.asarray(atmosphere.rosseland_optical_depth) / TAU))))
    temperature = float(np.asarray(atmosphere.temperature)[depth])
    metals = {**summary['abundances'], **summary.get('lte_abundances', {})}
    database = read_pg1159_atomic_database(data.stout, elements=tuple(metals))
    database = replace(database, ions={k: (replace(v, ionization_energy_ev=NIST_IONIZATION_EV[k])
                                           if v.ionization_energy_ev is None and k in NIST_IONIZATION_EV else v)
                                       for k, v in database.ions.items()},
                       _line_selection_cache={}, _unsold_hydrogen_coefficient_cache={})
    top = {e: min((i.charge for i in database.ion_stages(e) if i.ionization_energy_ev is None), default=None)
           for e in metals}
    database = replace(database, ions={k: v for k, v in database.ions.items() if top[k[0]] is None or k[1] <= top[k[0]]},
                       _line_selection_cache={}, _unsold_hydrogen_coefficient_cache={})
    reference = fixed_electron_metal_reference(atmosphere, database, metals)
    lines = []
    lo, hi = min(w[0] for w in WINDOWS) - 1, max(w[1] for w in WINDOWS) + 1
    for element in metals:
        for index, ion in enumerate(database.ion_stages(element)):
            density = reference.ion_number_density[element][index][depth]
            if density <= 1e-12 * reference.element_number_density[element][depth]:
                continue
            partition = float(reference.partition_function[element, ion.charge][depth])
            levels = {level.index: level for level in ion.levels}
            for transition in ion.transitions:
                if transition.transition_type != 'E1':
                    continue
                air = float(vacuum_to_air(transition.wavelength_vacuum_angstrom))
                if not lo <= air <= hi:
                    continue
                lower = levels[transition.lower_index]
                gf = lower.statistical_weight * transition.absorption_oscillator_strength
                strength = density / partition * gf * np.exp(-lower.energy_wavenumber * WAVENUMBER_TO_ERG
                                                             / (BOLTZMANN * temperature))
                lines.append((air, f'{element} {roman(ion.charge + 1)}', strength))
    heavy_abundances = summary.get('heavy_abundances', {})
    if heavy_abundances:
        heavy = HeavyLines(atmosphere, heavy_abundances)
        for line in heavy.lines:
            air = float(vacuum_to_air(line['wavelength']))
            if not lo <= air <= hi:
                continue
            density, partition = heavy.populations[(line['element'], line['charge'])]
            strength = (density[depth] / partition[depth] * 10 ** line['log_gf']
                        * np.exp(-line['lower_energy'] * WAVENUMBER_TO_ERG / (BOLTZMANN * temperature)))
            lines.append((air, f'{line["element"]} {roman(line["charge"] + 1)}', strength))
    return lines, temperature


def components(summary, target):
    """Observed-grid (air) metal-line ratio (metal model / its NLTE H/He host) and the host itself."""
    air, _ = paper.observation(target)
    metal = paper.predicted(ROOT / summary['host']['lte'].replace('-lte', '-metals') / 'spectrum.npz', target, air)
    host = paper.predicted(ROOT / summary['host']['hybrid'] / 'spectrum.npz', target, air)
    return air, metal / host, host


def identify(air, ratio, host, lo, hi, lines, velocity, minimum_depth):
    """Labels (air position, text): metal features of the ratio matched to metal lines, He features of the host."""
    inside = (air >= lo) & (air <= hi) & np.isfinite(ratio)
    wave, ratio, host = air[inside], ratio[inside], host[inside]
    step = np.median(np.diff(wave))
    shift = 1 + velocity / paper.LIGHT_SPEED_KMS
    labels = []
    for air_he, charge, _, _ in HELIUM:
        position = air_he * shift
        local = np.abs(wave - position) <= 6.0
        core = np.abs(wave - position) <= MATCH
        if core.any() and 1 - host[core].min() / np.percentile(host[local], 95) >= minimum_depth:
            labels.append((position, f'He {roman(charge + 1)} {air_he:.0f}'))
    # The metal run and the H/He run differ slightly in continuum level: normalize the ratio locally.
    ratio = ratio / percentile_filter(ratio, 90, size=max(3, int(10.0 / step)), mode='nearest')
    peaks, _ = find_peaks(1 - ratio, height=minimum_depth, prominence=minimum_depth / 2, distance=max(1, int(0.6 / step)))
    for peak in peaks:
        centre = wave[peak]
        near = [(air, species, s) for air, species, s in lines if abs(air * shift - centre) <= MATCH]
        if not near:
            continue
        near.sort(key=lambda item: -item[2])
        best = near[0]
        species = [best[1]]
        for air, name, s in near[1:]:
            if s >= BLEND_FRACTION * best[2] and name not in species:
                species.append(name)
        labels.append((best[0] * shift, ' + '.join(species) + f' {best[0]:.0f}'))
    return sorted(labels)


def spread(positions, separation, lo, hi):
    """Label x positions at least `separation` apart: overlapping labels form clusters centred on their lines."""
    clusters = [[p] for p in sorted(positions)]
    def place(cluster):
        centre = np.mean(cluster)
        offsets = (np.arange(len(cluster)) - (len(cluster) - 1) / 2) * separation
        x = centre + offsets
        shift = max(0.0, lo + separation / 2 - x[0]) - max(0.0, x[-1] - (hi - separation / 2))
        return x + shift
    merged = True
    while merged:
        merged = False
        for i in range(len(clusters) - 1):
            if place(clusters[i])[-1] + separation > place(clusters[i + 1])[0]:
                clusters[i] = clusters[i] + clusters.pop(i + 1)
                merged = True
                break
    return np.concatenate([place(c) for c in clusters]) if clusters else np.array([])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models', type=Path, default=ROOT / 'results/sdb/paper')
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/assets')
    args = parser.parse_args()
    normalize, _ = paper.paper_normalizer()
    data = ModelData.default()
    plt.rcParams.update({'font.family': 'sans-serif', 'mathtext.fontset': 'dejavusans', 'font.size': 10,
                         'axes.linewidth': .6, 'xtick.major.width': .6, 'ytick.major.width': .6,
                         'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white'})
    rows = [(key, window) for key in STARS for window in WINDOWS]
    fig, axes = plt.subplots(len(rows), 1, figsize=(8.0, 2.45 * len(rows) + .5))
    record, cache = {}, {}
    for ax, (key, (lo, hi)) in zip(axes, rows):
        if key not in cache:
            prepared = paper.prepare(key, args.models, normalize)
            lines, temperature = candidate_lines(key, prepared['summary'], data)
            cache[key] = (prepared, lines, components(prepared['summary'], prepared['target']))
            record[key] = dict(line_forming_temperature=temperature, labels={})
        prepared, lines, (air, ratio, host_flux) = cache[key]
        target, summary = prepared['target'], prepared['summary']
        wave_air = vacuum_to_air(prepared['wave'])
        inside = (wave_air >= lo) & (wave_air <= hi)
        w, obs, mod = wave_air[inside], prepared['observed'][inside], prepared['model'][inside]
        ax.plot(w, obs, color=OBSERVED, lw=.6)
        ax.plot(w, mod, color=MODEL, lw=.8)
        labels = identify(air, ratio, host_flux, lo, hi, lines, target['velocity'], MINIMUM_DEPTH[key])
        record[key]['labels'][f'{lo:.0f}-{hi:.0f}'] = labels
        # Data in the lower half (continuum at mid-height), labels above.
        continuum = np.nanpercentile(mod, 95)
        low = min(np.nanmin(mod), np.nanpercentile(obs, 0.5))
        bottom = low - .06 * (continuum - low)
        top = continuum + 1.4 * (continuum - bottom)
        ax.set_ylim(bottom, top)
        ax.set_xlim(lo, hi)
        to_axes = lambda y: (y - bottom) / (top - bottom)
        positions = [p for p, _ in labels]
        placed = spread(positions, 0.0165 * (hi - lo), lo, hi)
        start = to_axes(continuum) + .02
        for (position, text), x in zip(labels, placed):
            ax.plot([position, position, x], [start, start + .035, start + .07], color='.5', lw=.45,
                    transform=ax.get_xaxis_transform(), clip_on=False)
            ax.text(x, start + .08, text, rotation=90, ha='center', va='bottom', fontsize=LINE_LABEL_SIZE, color='.2',
                    transform=ax.get_xaxis_transform())
        ax.tick_params(labelsize=TICK_SIZE)
        host = summary['host']
        ax.set_title(f'{target["name"]} ({target["instrument"]}; {host["teff"]:,.0f} K, log g {host["logg"]:.2f})',
                     fontsize=LABEL_SIZE, loc='left', pad=3)
        ax.set_ylabel('Normalized flux', fontsize=TICK_SIZE)
    axes[-1].set_xlabel(r'Air wavelength ($\mathrm{\AA}$)', fontsize=LABEL_SIZE)
    handles = [Line2D([], [], color=OBSERVED, lw=.9, label='Observed'), Line2D([], [], color=MODEL, lw=1.0, label='OpenWD')]
    fig.legend(handles=handles, loc='upper right', ncol=2, frameon=False, fontsize=TICK_SIZE, bbox_to_anchor=(.985, 1.0))
    fig.tight_layout(rect=(0, 0, 1, .985), h_pad=.9)
    fig.savefig(args.output / 'sdb-zoom-lines.pdf')
    fig.savefig(args.output / 'sdb-zoom-lines.png', dpi=220)
    print(json.dumps(record, indent=1))


if __name__ == '__main__':
    main()
