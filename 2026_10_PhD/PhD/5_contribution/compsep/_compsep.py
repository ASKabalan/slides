# ENV: skip
# (a module imported by the figure scripts, not a generator of its own)
"""Helpers shared by the component-separation generators (one folder per slide).

Every sky map of the section goes through `moll_png`: a transparent Mollweide PNG, cropped to
the map, with masked pixels (UNSEEN) drawn in a light neutral grey. Patch maps colour each patch
by a random value, so neighbouring patches stand apart.
"""

import io
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parent / ".cache"
NSIDE = 64
MASKED = "#D9D4CC"
# LiteBIRD's three analysed 20 % regions, named by Galactic latitude
REGION_MASKS = ("GAL020", "GAL040", "GAL060")


def moll_png(m, path, cmap="viridis", vmin=None, vmax=None, figsize=(6.4, 3.4), dpi=170):
    import healpy as hp
    import matplotlib
    import matplotlib.pyplot as plt
    from PIL import Image

    cm = matplotlib.colormaps[cmap].copy() if isinstance(cmap, str) else cmap
    fig = plt.figure(figsize=figsize, dpi=dpi)
    hp.mollview(m, fig=fig.number, title="", cbar=False, cmap=cm, min=vmin, max=vmax,
                notext=True, margins=(0, 0, 0, 0), bgcolor=(0.0,) * 4, badcolor=MASKED)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    im = Image.open(buf).convert("RGBA")
    im.crop(im.getchannel("A").getbbox()).save(path, format="PNG")


def shuffled(labels, seed=0, unseen=-1.6375e30):
    """Map patch labels to random values in [0, 1], one per patch; masked pixels stay UNSEEN."""
    labels = np.asarray(labels)
    out = np.full(labels.shape, unseen, dtype=float)
    ok = labels > unseen / 2
    uniq, inv = np.unique(labels[ok], return_inverse=True)
    vals = np.random.default_rng(seed).permutation(len(uniq)) / max(len(uniq) - 1, 1)
    out[ok] = vals[inv]
    return out


def regions_by_latitude():
    """The three region masks ordered low, mid, high Galactic latitude (by mean |b|)."""
    import healpy as hp
    from furax_cs.data import get_mask

    _, lat = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)), lonlat=True)
    masks = {name: np.asarray(get_mask(name, nside=NSIDE)).astype(bool) for name in REGION_MASKS}
    order = sorted(masks, key=lambda n: np.abs(lat[masks[n]]).mean())
    return {"low": masks[order[0]], "mid": masks[order[1]], "high": masks[order[2]]}, order


def r_formatters():
    """furax-cs's r-hat formatters, without the global matplotlib style its plotting module sets
    on import (the `science` style and usetex), so figures keep the deck's look."""
    import matplotlib as mpl

    saved = mpl.rcParams.copy()
    from furax_cs.r_analysis.plotting import format_r_with_errors, resolve_r_exponents

    mpl.rcParams.update(saved)
    return format_r_with_errors, resolve_r_exponents


def plot_r_likelihoods(ax, entries, xlim=(-0.002, 0.005), exponent=None, legend_loc="upper right",
                       ytop=1.08):
    """Likelihoods on r in the notation of the thesis and the paper (furax-cs r_analysis).

    `entries` is a list of (label, row, colour) with row holding r_grid, L_vals, r_best,
    sigma_r_pos and sigma_r_neg. Each curve is normalised to its peak, the 68 % interval is
    shaded under it, r-hat is a dashed line, and every label quotes r-hat with the shared power
    of ten chosen by furax-cs.
    """
    format_r_with_errors, resolve_r_exponents = r_formatters()
    trip = [(float(r["r_best"]), float(r["sigma_r_pos"]), float(r["sigma_r_neg"])) for _, r, _ in entries]
    exps = [exponent] * len(trip) if exponent is not None else resolve_r_exponents(trip)
    for (label, row, colour), (rb, sp, sn), e in zip(entries, trip, exps):
        r = np.asarray(row["r_grid"], dtype=float)
        like = np.asarray(row["L_vals"], dtype=float)
        like = like / like.max()
        ax.plot(r, like, color=colour, lw=2.2,
                label=rf"{label}  $\hat{{r}} = {format_r_with_errors(rb, sp, sn, e)}$")
        ax.fill_between(r, 0, like, where=(r > rb - sn) & (r < rb + sp), color=colour, alpha=0.2)
        ax.axvline(rb, color=colour, ls="--", lw=1.4, alpha=0.8)
    ax.set_xlim(*xlim)
    ax.set_ylim(0, ytop)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 5, 10]))
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$L_{\mathrm{cosmo}}$")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend(loc=legend_loc, frameon=True, framealpha=0.95)
