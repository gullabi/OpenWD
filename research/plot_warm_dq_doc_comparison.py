#!/usr/bin/env python3
"""WarmDQ documentation figures: DESI and SDSS spectra against OpenWD models.

    python research/plot_warm_dq_doc_comparison.py --models MODELS --data DATA --output docs/assets

MODELS/<tag>/spectrum.txt are the models of research/warm_dq_doc_models.py
(vacuum wavelength, surface F_lambda, continuum). DATA holds the spectra
written by research/warm_dq_doc_data.py (``desi/*.npz``, ``sdss/*.npz``);
they are not installed with OpenWD.

Nothing stellar is fitted. The model is shifted by one radial velocity (a
chi^2 grid over -200 to 500 km/s), averaged over each pixel and multiplied by
the DESI resolution matrix of that pixel (SDSS: a Gaussian of the pipeline
``wdisp``). One multiplicative degree-5 Chebyshev polynomial per spectrum
absorbs flux calibration, reddening and the solid angle. The same polynomial
times the model's line-free continuum normalizes both data and model in the
figures, so they show flux relative to the model continuum. Telluric bands
are excluded from the fit and not drawn. DESI spectra are drawn in 2 A
inverse-variance bins.

Three figures are written, as PDF and PNG (pdftoppm, 1800 px):

* ``warm-dq-paper``: the 14 C+H stars, sorted by Teff.
* ``warm-dq-lines``: carbon lines and H-alpha of six of them.
* ``warm-dq-che``: SDSS spectra with the helium-free C+H solution and the
  C/He solution of Koester & Kepler (2019).

``warm-dq-scores.json`` and a Markdown table (stdout) give chi^2 per pixel of
the model and of its line-free continuum, the velocity, and the model/observed
equivalent widths of five lines against a local linear continuum (in the
normalized spectra; omitted where the observed line is below 3 sigma).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from warm_dq_doc_models import MODELS  # noqa: E402

C_KMS = 299792.458
PRINT_SCALE = 8.0 / 6.77
AXIS_LABEL_SIZE, TICK_SIZE, SECONDARY_SIZE = 8.5 * PRINT_SCALE, 7.5 * PRINT_SCALE, 6.5 * PRINT_SCALE
OBSERVED, OPENWD, CHE = ".30", "#d95f02", "#1b9e77"
DEGREE = 5
DESI_ARMS = {"B": (3600.0, 5790.0), "R": (5790.0, 7590.0), "Z": (7590.0, 9800.0)}
TELLURIC = [(6860, 6960), (7160, 7340), (7590, 7700), (8120, 8350), (8940, 9800)]
SDSS_MASK = TELLURIC[:-1] + [(8940, 9250), (5570, 5586), (6296, 6306)]
# tag: (DESI name or SDSS plate-MJD-fiber)
SPECTRA = {
    "ch_j0146": "WDJ014656.90-042613.50", "ch_j1215": "WDJ121510.65+470011.25",
    "ch_j0324": "WDJ032436.97-013506.16", "ch_j1728": "0358-51818-0296",
    "ch_j2300": "WDJ230025.55+084838.01", "ch_j1448": "WDJ144854.90+051903.80",
    "ch_j1622": "4953-55749-0483", "ch_j0117": "WDJ011731.77+160239.28",
    "ch_j0205": "WDJ020549.70+205707.96", "ch_j1803": "WDJ180300.29+260738.14",
    "ch_j0907": "WDJ090748.93+174903.20", "ch_j2302": "WDJ230245.05+004506.00",
    "ch_j0706": "WDJ070625.00+483339.26", "ch_j1104": "WDJ110406.70+203528.73",
    "che_j0236": "2399-53764-0059", "che_j1728": "0358-51818-0296",
    "che_j1622": "4953-55749-0483", "che_j0818": "2077-53846-0575",
}
LINE_STARS = ("ch_j0146", "ch_j1728", "ch_j2300", "ch_j0205", "ch_j0907", "ch_j2302")
LINE_WINDOWS = (
    ("C II 4267", 4245, 4292), ("C I 5f band", 4400, 4570), ("C I 4772", 4745, 4798),
    ("C I 4932", 4905, 4960), ("C I 5052", 5020, 5078), ("C I 5380", 5358, 5405),
    (r"H$\alpha$, C II 6580", 6520, 6615), ("C I 7115", 7080, 7150),
)
# Equivalent widths against a local linear continuum: (line, blue band, red band), vacuum A.
EW_LINES = {
    "C I 4933": ((4926, 4946), (4908, 4918), (4952, 4960)),
    "C I 5053": ((5046, 5066), (5028, 5033), (5070, 5078)),
    "C I 7115": ((7098, 7135), (7078, 7086), (7142, 7152)),
    "C II 4268": ((4256, 4284), (4243, 4251), (4290, 4298)),
    "Halpha": ((6520, 6610), (6470, 6490), (6640, 6660)),
}


def _resolution_matrix(res: np.ndarray) -> sp.csr_matrix:
    half = res.shape[0] // 2
    return sp.dia_matrix((res, np.arange(half, -half - 1, -1)), shape=(res.shape[1],) * 2).tocsr()


def _sdss_resolution(w: np.ndarray, wdisp: np.ndarray) -> sp.csr_matrix:
    n = w.size
    rows, cols, values = [], [], []
    for offset in range(-8, 9):
        i = np.arange(max(0, -offset), min(n, n - offset))
        rows.append(i); cols.append(i + offset)
        values.append(np.exp(-0.5 * (offset / np.maximum(wdisp[i], 0.3)) ** 2))
    matrix = sp.csr_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return sp.diags(1.0 / np.asarray(matrix.sum(axis=1)).ravel()) @ matrix


def _pixel_average(wave: np.ndarray, flux: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    edges = np.concatenate(([pixels[0] - 0.5 * (pixels[1] - pixels[0])], 0.5 * (pixels[1:] + pixels[:-1]),
                            [pixels[-1] + 0.5 * (pixels[-1] - pixels[-2])]))
    cumulative = np.concatenate(([0.0], np.cumsum(0.5 * (flux[1:] + flux[:-1]) * np.diff(wave))))
    return np.diff(np.interp(edges, wave, cumulative)) / np.diff(edges)


def load_segments(data: Path, key: str) -> list[dict]:
    segments = []
    if key.startswith("WDJ"):
        for coadd, path in enumerate(sorted((data / "desi").glob(f"{key}_*.npz"))):
            z = np.load(path)
            for arm, (lo, hi) in DESI_ARMS.items():
                w, f, iv = z[f"{arm}_w"], z[f"{arm}_f"], z[f"{arm}_iv"]
                good = (iv > 0) & (z[f"{arm}_mask"] == 0) & (w >= lo) & (w < hi) & np.isfinite(f)
                for a, b in TELLURIC:
                    good &= ~((w > a) & (w < b))
                if good.sum() > 50:
                    segments.append(dict(coadd=coadd, w=w, f=f, iv=iv, good=good,
                                         R=_resolution_matrix(z[f"{arm}_res"]), desi=True))
    else:
        z = np.load(data / "sdss" / f"spec-{key}.npz")
        w, f, iv = z["wave_vac"], z["flux"], z["ivar"]
        good = (iv > 0) & (z["and_mask"] == 0) & (w >= 3700.0) & (w < 9250.0) & np.isfinite(f)
        for a, b in SDSS_MASK:
            good &= ~((w > a) & (w < b))
        segments.append(dict(coadd=0, w=w, f=f, iv=iv, good=good, R=_sdss_resolution(w, z["wdisp"]), desi=False))
    return segments


def _on_pixels(segments, wave, flux, velocity):
    shifted = wave * (1.0 + velocity / C_KMS)
    return [s["R"].dot(_pixel_average(shifted, flux, s["w"])) for s in segments]


def _polynomials(segments, models):
    chi2, scales = 0.0, [None] * len(segments)
    for coadd in sorted({s["coadd"] for s in segments}):
        idx = [i for i, s in enumerate(segments) if s["coadd"] == coadd]
        w = np.concatenate([segments[i]["w"][segments[i]["good"]] for i in idx])
        V = np.polynomial.chebyshev.chebvander(2 * (w - 3600.0) / 6200.0 - 1, DEGREE)
        m = np.concatenate([models[i][segments[i]["good"]] for i in idx])
        f = np.concatenate([segments[i]["f"][segments[i]["good"]] for i in idx])
        e = 1 / np.sqrt(np.concatenate([segments[i]["iv"][segments[i]["good"]] for i in idx]))
        coef, *_ = np.linalg.lstsq(V * (m / e)[:, None], f / e, rcond=None)
        chi2 += float(np.sum(((f - (V @ coef) * m) / e) ** 2))
        for i in idx:
            x = 2 * (segments[i]["w"] - 3600.0) / 6200.0 - 1
            scales[i] = np.polynomial.chebyshev.chebvander(x, DEGREE) @ coef
    return chi2, scales


def compare(segments, spectrum):
    """Return velocity, chi^2, line-free chi^2, pixel count and per-segment (w, data, model, error) normalized."""
    wave, flux, continuum = spectrum[:, 0], spectrum[:, 1], spectrum[:, 2]
    grid = np.arange(-200.0, 500.1, 10.0)
    chis = [_polynomials(segments, _on_pixels(segments, wave, flux, v))[0] for v in grid]
    j = int(np.argmin(chis)); velocity = grid[j]
    if 0 < j < grid.size - 1:
        a, b, c = chis[j - 1], chis[j], chis[j + 1]
        velocity = grid[j] + 5.0 * (a - c) / (a - 2 * b + c)
    models = _on_pixels(segments, wave, flux, velocity)
    chi2, scales = _polynomials(segments, models)
    chi2_continuum, _ = _polynomials(segments, _on_pixels(segments, wave, continuum, velocity))
    continua = _on_pixels(segments, wave, continuum, velocity)
    normalized = []
    for s, m, scale, cont in zip(segments, models, scales, continua):
        reference = scale * cont
        g = s["good"]
        normalized.append((s["w"][g], s["f"][g] / reference[g], m[g] / cont[g], 1 / np.sqrt(s["iv"][g]) / reference[g], s["desi"]))
    npix = int(sum(s["good"].sum() for s in segments))
    return velocity, chi2, chi2_continuum, npix, normalized


def _local_ew(w, y, e, line, blue, red):
    """Equivalent width (A) and its error against a straight line through the two side bands."""

    inside = lambda band: (w > band[0]) & (w < band[1])
    b, r, l = inside(blue), inside(red), inside(line)
    if b.sum() < 2 or r.sum() < 2 or l.sum() < 4:
        return None
    xb, yb, xr, yr = w[b].mean(), np.median(y[b]), w[r].mean(), np.median(y[r])
    continuum = yb + (yr - yb) * (w[l] - xb) / (xr - xb)
    dw = np.gradient(w)[l]
    return float(np.sum((1 - y[l] / continuum) * dw)), float(np.sqrt(np.sum((dw * e[l] / continuum) ** 2)))


def equivalent_width_ratios(normalized) -> dict[str, float | None]:
    """Model/observed equivalent width of each EW_LINES line (None: not detected at 3 sigma)."""

    ratios = {}
    for name, bands in EW_LINES.items():
        ratios[name] = None
        for w, f, m, e, _ in normalized:
            observed, model = _local_ew(w, f, e, *bands), _local_ew(w, m, 0.0 * e, *bands)
            if observed and model:
                if observed[0] > 3.0 * observed[1]:
                    ratios[name] = model[0] / observed[0]
                break
    return ratios


def _bin(w, f, e, width=2.0):
    edges = np.arange(w.min(), w.max() + width, width)
    index = np.digitize(w, edges)
    weight = 1 / e**2
    count = np.bincount(index, minlength=edges.size + 1)
    keep = count > 0
    wb = np.bincount(index, w * weight, edges.size + 1)[keep] / np.bincount(index, weight, edges.size + 1)[keep]
    fb = np.bincount(index, f * weight, edges.size + 1)[keep] / np.bincount(index, weight, edges.size + 1)[keep]
    return wb, fb


def _gaps(x, y, step=4.0):
    """Break a curve at masked bands (sampling gaps wider than ``step`` A)."""

    cut = np.flatnonzero(np.diff(x) > step) + 1
    return np.insert(x, cut, np.nan), np.insert(y, cut, np.nan)


def _draw(ax, segments, lo, hi, model_color=OPENWD, extra=None, data=True):
    """Draw data and model(s) in [lo, hi]; return (all values, minimum model value)."""

    observed, modelled = [], []
    for w, f, m, e, desi in segments:
        s = (w > lo) & (w < hi)
        if s.sum() < 3:
            continue
        if data:
            x, y = _bin(w[s], f[s], e[s]) if desi else (w[s], f[s])
            ax.plot(*_gaps(x, y), color=OBSERVED, lw=0.6)
            observed.append(y)
        ax.plot(*_gaps(w[s], m[s]), color=model_color, lw=0.9)
        modelled.append(m[s])
    for w, f, m, e, desi in extra or ():
        s = (w > lo) & (w < hi)
        if s.sum() >= 3:
            ax.plot(*_gaps(w[s], m[s]), color=CHE, lw=0.9)
            modelled.append(m[s])
    values = observed + modelled
    if not values:
        return np.array([1.0]), 1.0
    return np.concatenate(values), float(min(np.nanmin(v) for v in modelled))


def _headroom(ax, drawn, fraction):
    """Robust y-limits that keep the model line cores, with room for a label."""

    values, model_minimum = drawn
    low, high = np.nanpercentile(values, [0.3, 99.7])
    low = min(low, model_minimum)
    span = high - low
    ax.set_ylim(low - 0.05 * span, high + fraction * span)


def _composition(tag):
    _, teff, logg, ch, che, _ = MODELS[tag]
    parts = []
    if ch is not None:
        parts.append(rf"$\log(N_\mathrm{{C}}/N_\mathrm{{H}}) = {ch:.2f}$")
    if che is not None:
        parts.append(rf"$\log(N_\mathrm{{C}}/N_\mathrm{{He}}) = {che:.2f}$")
    return rf"$T_\mathrm{{eff}} = {teff}\,$K, $\log g = {logg:.2f}$", ", ".join(parts)


def _save(fig, output: Path, stem: str):
    pdf = output / f"{stem}.pdf"
    fig.savefig(pdf)
    subprocess.run(["pdftoppm", "-singlefile", "-scale-to", "1800", "-png", str(pdf), str(output / stem)], check=True)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results, scores = {}, {}
    for tag in MODELS:
        path = args.models / tag / "spectrum.txt"
        if not path.exists():
            print("missing", path, file=sys.stderr)
            continue
        velocity, chi2, chi2c, npix, normalized = compare(load_segments(args.data, SPECTRA[tag]), np.loadtxt(path))
        results[tag] = normalized
        scores[tag] = dict(name=MODELS[tag][0], spectrum=MODELS[tag][5], chi2_per_pixel=chi2 / npix,
                           line_free_chi2_per_pixel=chi2c / npix, pixels=npix, velocity_kms=velocity,
                           equivalent_width_ratio=equivalent_width_ratios(normalized))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "warm-dq-scores.json").write_text(json.dumps(scores, indent=1) + "\n")

    # Figure 1: the C+H stars.
    stars = [t for t in MODELS if t.startswith("ch_") and t in results]
    rows = (len(stars) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(13.5, 1.75 * rows + 0.8), sharex=True)
    for ax, tag in zip(axes.T.ravel(), stars):
        values = _draw(ax, results[tag], 3780, 7250)
        _headroom(ax, values, 0.45)
        top, bottom = _composition(tag)
        ax.text(0.99, 0.96, f"{MODELS[tag][0]}   {top}\n{bottom}", transform=ax.transAxes, ha="right",
                va="top", fontsize=SECONDARY_SIZE)
        ax.tick_params(labelsize=TICK_SIZE)
    for ax in axes.T.ravel()[len(stars):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Vacuum wavelength (Å)", fontsize=AXIS_LABEL_SIZE)
    fig.supylabel("Flux / model continuum", fontsize=AXIS_LABEL_SIZE)
    axes[0, 0].legend([Line2D([], [], color=OBSERVED), Line2D([], [], color=OPENWD)], ["Observed", "OpenWD"],
                      fontsize=SECONDARY_SIZE, loc="upper left", frameon=False, ncol=2)
    fig.tight_layout(h_pad=0.2, w_pad=1.0)
    _save(fig, args.output, "warm-dq-paper")

    # Figure 2: carbon lines and H-alpha.
    line_stars = [t for t in LINE_STARS if t in results]
    fig, axes = plt.subplots(len(line_stars), len(LINE_WINDOWS),
                             figsize=(2.0 * len(LINE_WINDOWS), 1.55 * len(line_stars) + 0.6), squeeze=False)
    for r, tag in enumerate(line_stars):
        for c, (label, lo, hi) in enumerate(LINE_WINDOWS):
            ax = axes[r, c]
            _draw(ax, results[tag], lo, hi)
            ax.set_xlim(lo, hi)
            ax.tick_params(labelsize=TICK_SIZE * 0.8)
            ax.locator_params(axis="x", nbins=3)
            if r == 0:
                ax.set_title(label, fontsize=SECONDARY_SIZE)
        axes[r, 0].set_ylabel(f"{MODELS[tag][0]}\n{MODELS[tag][1]} K", fontsize=SECONDARY_SIZE)
    fig.supxlabel("Vacuum wavelength (Å)", fontsize=AXIS_LABEL_SIZE)
    fig.tight_layout(h_pad=0.3, w_pad=0.3)
    _save(fig, args.output, "warm-dq-lines")

    # Figure 3: C+H against C/He.
    panels = [("ch_j1622", "che_j1622"), ("ch_j1728", "che_j1728"), (None, "che_j0236"), (None, "che_j0818")]
    panels = [(a, b) for a, b in panels if b in results and (a is None or a in results)]
    if not panels:
        return _table(scores)
    fig, axes = plt.subplots(len(panels), 1, figsize=(13.5, 2.1 * len(panels) + 0.6), sharex=True, squeeze=False)
    for ax, (ch, che) in zip(axes[:, 0], panels):
        if ch is not None:
            values = _draw(ax, results[ch], 3780, 7250, extra=results[che])
            text = (f"{MODELS[ch][0]}   C+H: {_composition(ch)[0]}, {_composition(ch)[1]}\n"
                    f"C/He: {_composition(che)[0]}, {_composition(che)[1]}")
        else:
            values = _draw(ax, results[che], 3780, 7250, model_color=CHE)
            text = f"{MODELS[che][0]}   C/He: {_composition(che)[0]}, {_composition(che)[1]}"
        _headroom(ax, values, 0.4)
        ax.text(0.99, 0.96, text, transform=ax.transAxes, ha="right", va="top", fontsize=SECONDARY_SIZE)
        for line in (4472.7, 5877.2, 6679.9):
            ax.axvline(line, color="0.75", lw=0.6, ls=":", zorder=0)
        ax.tick_params(labelsize=TICK_SIZE)
    axes[0, 0].legend([Line2D([], [], color=OBSERVED), Line2D([], [], color=OPENWD), Line2D([], [], color=CHE)],
                      ["Observed", "OpenWD, C+H (Kilic et al. 2026)", "OpenWD, C/He (Koester & Kepler 2019)"],
                      fontsize=SECONDARY_SIZE, loc="upper left", frameon=False, ncol=3)
    axes[-1, 0].set_xlabel("Vacuum wavelength (Å); dotted: He I 4472, 5877, 6680", fontsize=AXIS_LABEL_SIZE)
    fig.supylabel("Flux / model continuum", fontsize=AXIS_LABEL_SIZE)
    fig.tight_layout(h_pad=0.3)
    _save(fig, args.output, "warm-dq-che")
    _table(scores)


def _table(scores) -> None:
    lines = " | ".join(EW_LINES).replace("Halpha", "Hα")
    print(f"| Star | Spectrum | Teff (K) | log g | Composition | χ²/px | without lines | {lines} |")
    print("| --- " * (7 + len(EW_LINES)) + "|")
    for tag, s in scores.items():
        _, teff, logg, ch, che, _ = MODELS[tag]
        comp = ", ".join(x for x in (f"C/H {ch:.2f}" if ch is not None else "", f"C/He {che:.2f}" if che is not None else "") if x)
        ratios = " | ".join("–" if r is None else f"{r:.2f}" for r in s["equivalent_width_ratio"].values())
        print(f"| {s['name']} | {s['spectrum']} | {teff} | {logg:.2f} | {comp} | "
              f"{s['chi2_per_pixel']:.2f} | {s['line_free_chi2_per_pixel']:.2f} | {ratios} |")


if __name__ == "__main__":
    main()
