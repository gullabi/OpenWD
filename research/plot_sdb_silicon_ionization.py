#!/usr/bin/env python3
"""Figures for the sdB guide: silicon ionization in HD 4539 and Feige 38, metal lines, TLUSTY benchmark.

    python research/plot_sdb_silicon_ionization.py --observations results/sdb/obs --output docs/assets

Inputs (research outputs, not part of the repository): sdb_trace_metals.py runs at the published
parameters with the blanketed structure, the Teff-offset runs, and the HD 4539 silicon-only OpenWD and
TLUSTY/SYNSPEC spectra.  Writes sdb-silicon.{pdf,png}, sdb-metals-hd4539.{pdf,png},
sdb-metals-feige38.{pdf,png} and sdb-silicon-tlusty.{pdf,png}.

Profiles are convolved to the X-shooter resolution of the arm, shifted by one radial velocity per star,
divided by each model's own NLTE H/He spectrum (sdb_hybrid_nlte.py; observation: the published-Teff one)
and normalized by a linear continuum through the window edges.  The metal driver's host_flux is not used
there: it contains the LTE background metals (Fe III), which the division would cancel.  Equivalent
widths use research/sdb_silicon_ew_observed.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sdb_compare_hd4539 import RESOLVING_POWER, gaussian_convolve, load_observation, vacuum_to_air  # noqa: E402
from sdb_silicon_ew_observed import C, R, fit_rv, load, measure, model  # noqa: E402

OBSERVED = '.30'
PUBLISHED = '#d95f02'   # OpenWD in the sdB validation figure
HOTTER = '#2a78d6'
TICK_SIZE, LABEL_SIZE = 8, 9
WINDOWS = [(4126.6, 4132.4, 'Si II 4128, 4131'), (4551.0, 4554.2, 'Si III 4552'),
           (4566.6, 4575.9, 'Si III 4568, 4575')]
EDGE = 0.5          # Angstrom at each window end for the linear continuum (silicon figures)
METAL_EDGE = 1.0    # wider: 5 X-shooter UVB pixels per side
# Metal-line windows (air, Angstrom; clean edges), chosen per star for lines measurable in its spectrum.
HD4539_WINDOWS = [(3916.0, 3923.5, 'C II 3919, 3921'), (4263.0, 4271.5, 'C II 4267'),
                  (3992.0, 3998.0, 'N II 3995'), (5663.0, 5689.5, 'N II 5667–86'),
                  (4626.5, 4654.5, 'N II, N III, O II, C III 4630–51'), (6344.0, 6350.5, 'Si II 6347'),
                  (4477.0, 4485.5, 'Mg II 4481'), (4810.0, 4832.0, 'Si III 4813–29'),
                  (4250.5, 4257.0, 'S III 4254'), (4282.0, 4288.0, 'S III 4285'),
                  (4161.5, 4169.5, 'Fe III 4165, 4167'), (5124.0, 5131.0, 'Fe III 5127')]
FEIGE38_WINDOWS = [(4263.0, 4271.5, 'C II 4267'), (4631.5, 4654.5, 'N III, O II, C III 4634–51'),
                   (3992.0, 3998.0, 'N II 3995'), (5663.0, 5689.5, 'N II 5667–86'),
                   (4411.9, 4418.7, 'O II 4415, 4417'), (4588.0, 4599.0, 'O II 4591, 4596'),
                   (4810.0, 4832.0, 'Si III 4813–29'), (4085.5, 4092.0, 'Si IV 4089, O II'),
                   (4112.5, 4118.6, 'Si IV 4116'), (4250.5, 4257.0, 'S III 4254'),
                   (4282.0, 4288.0, 'S III 4285'), (4161.5, 4169.5, 'Fe III 4165, 4167')]
RUNS = 'results/si-ionization'
STARS = [
    dict(name='HD 4539', obs='hd4539', file='hd4539', teff=23200, hotter=25200, windows=HD4539_WINDOWS,
         published=f'{RUNS}/full-hd4539z-allop', hybrid={23200: f'{RUNS}/hd4539z-hybrid',
                                                         25200: f'{RUNS}/teff-scan/hd4539-t25200-hybrid'},
         scan={23700: f'{RUNS}/teff-scan/hd4539-t23700-metals', 24200: f'{RUNS}/teff-scan/hd4539-t24200-metals',
               25200: f'{RUNS}/teff-scan/hd4539-t25200-metals', 26200: f'{RUNS}/teff-scan/hd4539-t26200-metals'}),
    dict(name='Feige 38', obs='feige38', file='feige38', teff=28200, hotter=29200, windows=FEIGE38_WINDOWS,
         published=f'{RUNS}/full-f38z-allop', hybrid={28200: f'{RUNS}/f38z-hybrid',
                                                      29200: f'{RUNS}/teff-scan/f38-t29200-hybrid'},
         scan={28700: f'{RUNS}/teff-scan/f38-t28700-metals', 29200: f'{RUNS}/teff-scan/f38-t29200-metals',
               30200: f'{RUNS}/teff-scan/f38-t30200-metals', 31200: f'{RUNS}/teff-scan/f38-t31200-metals'}),
]


def style():
    plt.rcParams.update({'font.family': 'sans-serif', 'mathtext.fontset': 'dejavusans', 'font.size': 10,
                         'axes.linewidth': .6, 'xtick.major.width': .6, 'ytick.major.width': .6,
                         'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white'})


def edge_normalized(wave, ratio, lo, hi, width=EDGE):
    edge = (wave < lo + width) | (wave > hi - width)
    coefficients = np.polyfit(wave[edge] - lo, ratio[edge], 1)
    return ratio / np.polyval(coefficients, wave - lo)


def convolved(path, power, key='flux'):
    saved = np.load(Path(path) / 'spectrum.npz')
    return gaussian_convolve(vacuum_to_air(saved['wavelength_vacuum']), saved[key], power)


def star_models(star):
    """Metal flux and NLTE H/He spectrum, convolved per X-shooter arm, at the published and Si II/III Teff."""
    runs = {star['teff']: star['published'], star['hotter']: star['scan'][star['hotter']]}
    return {teff: {arm: dict(flux=convolved(runs[teff], power), host=convolved(star['hybrid'][teff], power))
                   for arm, power in RESOLVING_POWER.items()} for teff in runs}


def model_ratio(wave, arm_model, shift):
    (fw, ff), (hw, hf) = arm_model['flux'], arm_model['host']
    return np.interp(wave, fw * shift, ff) / np.interp(wave, hw * shift, hf)


def window_ylim(ax, curves):
    """Fit the model and observed absorption; isolated upward spikes in the data are clipped."""
    low = min(np.min(c) for c in curves)
    span = max(1 - low, .004)
    ax.set_ylim(low - .1 * span, 1 + .2 * span)


def ratio_4128_4552(widths):
    return widths['Si II 4128'] / widths['Si III 4552']


def model_legend(star):
    return [Line2D([], [], color=OBSERVED, lw=.9, label='X-shooter'),
            Line2D([], [], color=PUBLISHED, lw=1.0, label=f'Published $T_\\mathrm{{eff}}$ ({star["teff"]:,} K)'),
            Line2D([], [], color=HOTTER, lw=1.0, label=f'Si II/Si III $T_\\mathrm{{eff}}$ ({star["hotter"]:,} K)')]


def figure_ionization(observations, output):
    fig, axes = plt.subplots(2, 4, figsize=(8.0, 4.4), gridspec_kw=dict(width_ratios=[1.25, .75, 1.45, 1.3]))
    for index, (row, star) in enumerate(zip(axes, STARS)):
        ow, of, _ = load(observations / star['obs'])
        published = model(star['published'])
        rv = fit_rv(ow, of, published)
        shift = 1 + rv / C
        models = star_models(star)
        for ax, (lo, hi, title) in zip(row[:3], WINDOWS):
            inside = (ow > lo) & (ow < hi)
            wave = ow[inside]
            hw, hf = models[star['teff']]['uvb']['host']
            observed = edge_normalized(wave, of[inside] / np.interp(wave, hw * shift, hf), lo, hi)
            ax.plot(wave, observed, color=OBSERVED, lw=.8)
            fine = np.linspace(lo, hi, 600)
            for teff, color in ((star['teff'], PUBLISHED), (star['hotter'], HOTTER)):
                ax.plot(fine, edge_normalized(fine, model_ratio(fine, models[teff]['uvb'], shift), lo, hi),
                        color=color, lw=1.0)
            ax.set_xlim(lo, hi)
            ax.tick_params(labelsize=TICK_SIZE)
            if index == 0:
                ax.set_title(title, fontsize=LABEL_SIZE, pad=3)
        row[0].text(.06, .06, star['name'], transform=row[0].transAxes, fontsize=LABEL_SIZE, fontweight='bold')

        # Equivalent-width ratio against Teff (the observation measured identically).
        observed_widths = measure(ow, of, published, rv, 'flux')
        temperatures = [star['teff']] + sorted(star['scan'])
        runs = [published] + [model(star['scan'][t]) for t in sorted(star['scan'])]
        values = [ratio_4128_4552(measure(ow, None, run, rv, 'flux')) for run in runs]
        ax = row[3]
        ax.axhline(ratio_4128_4552(observed_widths), color=OBSERVED, lw=.8)
        ax.plot(np.array(temperatures) / 1e3, values, color=PUBLISHED, lw=1.0, marker='o', ms=4)
        ax.plot([star['hotter'] / 1e3], [values[temperatures.index(star['hotter'])]], ls='none', marker='o', ms=6,
                color=HOTTER)
        ax.axvline(star['teff'] / 1e3, color='.75', lw=.6, ls=(0, (1, 2)), zorder=0)
        ax.text(star['teff'] / 1e3, .03, ' published', transform=ax.get_xaxis_transform(), fontsize=TICK_SIZE - 1,
                color='.45', va='bottom')
        if index == 0:
            ax.set_title('Si II 4128 / Si III 4552', fontsize=LABEL_SIZE, pad=3)
        ax.set_ylim(0, max(values) * 1.15)
        ax.tick_params(labelsize=TICK_SIZE)
        ax.set_ylabel(r'$W_\lambda$(4128) / $W_\lambda$(4552)', fontsize=TICK_SIZE)
        ax.yaxis.set_label_position('right')
    axes[1][3].set_xlabel(r'$T_\mathrm{eff}$ (kK)', fontsize=LABEL_SIZE)
    for ax in axes[1][:3]:
        ax.set_xlabel(r'Air wavelength ($\mathrm{\AA}$)', fontsize=TICK_SIZE)
    axes[0][0].set_ylabel('Normalized flux', fontsize=LABEL_SIZE)
    axes[1][0].set_ylabel('Normalized flux', fontsize=LABEL_SIZE)
    handles = [Line2D([], [], color=OBSERVED, lw=.9, label='X-shooter'),
               Line2D([], [], color=PUBLISHED, lw=1.0, marker='o', ms=4, label='Published $T_\\mathrm{eff}$'),
               Line2D([], [], color=HOTTER, lw=1.0, marker='o', ms=4,
                      label='+2000 K (HD 4539), +1000 K (Feige 38)')]
    fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False, fontsize=TICK_SIZE, bbox_to_anchor=(.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, .93), w_pad=.6, h_pad=.8)
    fig.savefig(output / 'sdb-silicon.pdf')
    fig.savefig(output / 'sdb-silicon.png', dpi=220)


def figure_metals(observations, output, star):
    arms = load_observation(observations / star['obs'])
    ow, of, _ = load(observations / star['obs'])
    rv = fit_rv(ow, of, model(star['published']))
    shift = 1 + rv / C
    models = star_models(star)
    fig, axes = plt.subplots(3, 4, figsize=(8.0, 6.0))
    for ax, (lo, hi, title) in zip(axes.ravel(), star['windows']):
        arm = 'uvb' if hi < 5500 else 'vis'
        wave, flux, _ = arms[arm]
        inside = (wave > lo) & (wave < hi)
        wave = wave[inside]
        hw, hf = models[star['teff']][arm]['host']
        observed = edge_normalized(wave, flux[inside] / np.interp(wave, hw * shift, hf), lo, hi, METAL_EDGE)
        ax.plot(wave, observed, color=OBSERVED, lw=.8)
        fine = np.linspace(lo, hi, 800)
        curves = [np.percentile(observed, 2)]
        for teff, color, width in ((star['teff'], PUBLISHED, .9), (star['hotter'], HOTTER, 1.1)):
            profile = edge_normalized(fine, model_ratio(fine, models[teff][arm], shift), lo, hi, METAL_EDGE)
            ax.plot(fine, profile, color=color, lw=width)
            curves.append(profile)
        window_ylim(ax, curves + [observed[observed < 1]] if np.any(observed < 1) else curves)
        ax.set_xlim(lo, hi)
        ax.set_title(title, fontsize=TICK_SIZE + .5, pad=2.5)
        ax.tick_params(labelsize=TICK_SIZE - 1)
    for ax in axes[-1]:
        ax.set_xlabel(r'Air wavelength ($\mathrm{\AA}$)', fontsize=TICK_SIZE)
    for ax in axes[:, 0]:
        ax.set_ylabel('Normalized flux', fontsize=TICK_SIZE)
    fig.legend(handles=model_legend(star), loc='upper center', ncol=3, frameon=False, fontsize=TICK_SIZE,
               bbox_to_anchor=(.5, 1.0), title=star['name'], title_fontsize=LABEL_SIZE)
    fig.tight_layout(rect=(0, 0, 1, .92), w_pad=.5, h_pad=.7)
    fig.savefig(output / f'sdb-metals-{star["file"]}.pdf')
    fig.savefig(output / f'sdb-metals-{star["file"]}.png', dpi=220)


def synspec_profile(path, lo, hi, rv):
    data = np.loadtxt(path)
    order = np.argsort(data[:, 0])
    wave, flux = gaussian_convolve(data[order, 0], data[order, 1], R)
    fine = np.linspace(lo, hi, 600)
    return fine, edge_normalized(fine, np.interp(fine, wave * (1 + rv / C), flux), lo, hi)


def openwd_profile(run, key, lo, hi, rv):
    saved = np.load(Path(run) / 'spectrum.npz')
    air = vacuum_to_air(saved['wavelength_vacuum'])
    inside = (air > lo - 5) & (air < hi + 5)
    wave, flux = gaussian_convolve(air[inside], saved[key][inside], R)
    fine = np.linspace(lo, hi, 600)
    return fine, edge_normalized(fine, np.interp(fine, wave * (1 + rv / C), flux), lo, hi)


def figure_tlusty(observations, output):
    synspec = f'{RUNS}/tlusty/hd4539/syn-hd4539nl-{{mode}}-{{window}}/fort.7'
    openwd = f'{RUNS}/hd4539-op60'
    ow, of, _ = load(observations / 'hd4539')
    rv = fit_rv(ow, of, model(STARS[0]['published']))
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.5), gridspec_kw=dict(width_ratios=[1.25, .75, 1.45]))
    for ax, (lo, hi, title) in zip(axes, WINDOWS):
        window = 4100 if lo < 4300 else 4540
        inside = (ow > lo) & (ow < hi)
        ax.plot(ow[inside], edge_normalized(ow[inside], of[inside], lo, hi), color=OBSERVED, lw=.8)
        for mode, key, dash in (('lte', 'lte_flux', (0, (3, 1.5))), ('nlte', 'flux', '-')):
            ax.plot(*synspec_profile(synspec.format(mode=mode, window=window), lo, hi, rv), color=HOTTER, lw=1.0, ls=dash)
            ax.plot(*openwd_profile(openwd, key, lo, hi, rv), color=PUBLISHED, lw=1.0, ls=dash)
        ax.set_xlim(lo, hi)
        ax.set_title(title, fontsize=LABEL_SIZE, pad=3)
        if lo > 4560:
            ax.text(.57, .08, 'HD 4539 parameters: 23,200 K, log g 5.20\nlog Si/H = −5.30, silicon only',
                    transform=ax.transAxes, fontsize=TICK_SIZE - 1, color='.35', ha='center', va='bottom')
        ax.tick_params(labelsize=TICK_SIZE)
        ax.set_xlabel(r'Air wavelength ($\mathrm{\AA}$)', fontsize=TICK_SIZE)
    axes[0].set_ylabel('Normalized flux', fontsize=LABEL_SIZE)
    handles = [Line2D([], [], color=OBSERVED, lw=.9, label='X-shooter'),
               Line2D([], [], color=PUBLISHED, lw=1.0, label='OpenWD NLTE'),
               Line2D([], [], color=HOTTER, lw=1.0, label='TLUSTY + SYNSPEC NLTE'),
               Line2D([], [], color='.45', lw=1.0, ls=(0, (3, 1.5)), label='LTE (same colours)')]
    fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False, fontsize=TICK_SIZE, bbox_to_anchor=(.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, .86), w_pad=.6)
    fig.savefig(output / 'sdb-silicon-tlusty.pdf')
    fig.savefig(output / 'sdb-silicon-tlusty.png', dpi=220)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--observations', type=Path, default=Path('results/sdb/obs'))
    parser.add_argument('--output', type=Path, default=Path('docs/assets'))
    args = parser.parse_args()
    style()
    figure_ionization(args.observations, args.output)
    for star in STARS:
        figure_metals(args.observations, args.output, star)
    figure_tlusty(args.observations, args.output)
    print(f'wrote {args.output}/sdb-silicon.*, sdb-metals-*.* and sdb-silicon-tlusty.*')


if __name__ == '__main__':
    main()
