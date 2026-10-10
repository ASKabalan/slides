#!/usr/bin/env python3
# ENV: furax-cs
"""
The two PySM foreground models of the analysis, and how much their spectra vary.

For each sky, c1d0s0 and c1d1s1, the three spectral-parameter maps (synchrotron index beta_s,
dust index beta_d, dust temperature T_d) on colour scales shared between the two skies, and the
dust and synchrotron SEDs of 300 random pixels. In d0s0 every pixel has the same spectrum; in
d1s1 each pixel has its own.

SEDs in Rayleigh-Jeans brightness, from the same laws as the spectral-model slide: a power law
normalised at 23 GHz, and a modified blackbody normalised at 353 GHz (the PySM reference
frequencies).

Outputs (this directory):
  {sky}_{beta_pl,beta_dust,temp_dust}.png   transparent Mollweide, with a small colour bar
  {sky}_sed.svg                             the SED fan
"""

import io
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import INK, skip_if_built, slide_style

SKIES = ("c1d0s0", "c1d1s1")
PARAMS = (("beta_pl", r"$\beta_s$"), ("beta_dust", r"$\beta_d$"), ("temp_dust", r"$T_d$ [K]"))
OUTS = [f"{s}_{p}.png" for s in SKIES for p, _ in PARAMS] + [f"{s}_sed.svg" for s in SKIES]
skip_if_built(HERE, *OUTS)

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib import patheffects
from PIL import Image

CACHE = HERE.parent / ".cache"
NSIDE = 64
DUST, SYNC = "#D68910", "#3B6FB6"


def parameters(tag):
    npz = CACHE / f"pysm_params_{tag}_{NSIDE}.npz"
    if npz.exists():
        return dict(np.load(npz))
    from furax._instruments.sky import get_sky

    sky = get_sky(NSIDE, tag)
    npix = hp.nside2npix(NSIDE)
    full = lambda q: np.broadcast_to(np.asarray(q.value, dtype=float), (npix,)).copy()
    out = {
        "beta_dust": full(sky.components[1].mbb_index),
        "temp_dust": full(sky.components[1].mbb_temperature),
        "beta_pl": full(sky.components[2].pl_index),
    }
    CACHE.mkdir(exist_ok=True)
    np.savez(npz, **out)
    return out


P = {tag: parameters(tag) for tag in SKIES}
for tag in SKIES:
    print(tag, {k: (float(v.min()), float(v.max())) for k, v in P[tag].items()})

# colour scales set by the varying sky, shared by both
RANGE = {k: tuple(np.percentile(P["c1d1s1"][k], [1, 99])) for k, _ in PARAMS}

slide_style(scale=1.3)
for tag in SKIES:
    for key, label in PARAMS:
        lo, hi = RANGE[key]
        fig = plt.figure(figsize=(4.6, 3.1), dpi=170)
        hp.mollview(np.clip(P[tag][key], lo, hi), fig=fig.number, title="", cbar=False,
                    cmap="viridis", min=lo, max=hi, notext=True, margins=(0.02, 0.2, 0.02, 0.02),
                    bgcolor=(0.0,) * 4, sub=(1, 1, 1))
        if tag == "c1d0s0":
            # one value over the whole sky: print it on the map, white with a dark outline so it
            # reads on any viridis colour
            v = float(P[tag][key][0])
            text = {"beta_pl": f"{v:.1f}", "beta_dust": f"{v:.2f}", "temp_dust": f"{v:.0f} K"}[key]
            fig.axes[-1].text(0.5, 0.5, text.replace("-", "−"), transform=fig.axes[-1].transAxes,
                              ha="center", va="center", fontsize=34, fontweight="bold", color="white",
                              path_effects=[patheffects.withStroke(linewidth=4, foreground=INK)])
        cax = fig.add_axes([0.2, 0.1, 0.6, 0.045])
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap="viridis",
                                                norm=matplotlib.colors.Normalize(lo, hi)),
                          cax=cax, orientation="horizontal")
        cb.set_ticks([lo, hi])
        cb.set_ticklabels([f"{lo:.2f}" if key != "temp_dust" else f"{lo:.1f}",
                           f"{hi:.2f}" if key != "temp_dust" else f"{hi:.1f}"])
        cb.ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        cb.ax.tick_params(labelsize=12, colors=INK, length=0)
        cb.outline.set_visible(False)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", transparent=True)
        plt.close(fig)
        im = Image.open(buf).convert("RGBA")
        im.crop(im.getchannel("A").getbbox()).save(HERE / f"{tag}_{key}.png")
    print(f"wrote {tag} maps")

# ---------------------------------------------------------------- SED fans
h_over_k = 0.0479924                          # h / k_B in K / GHz
nu = np.geomspace(20, 450, 200)
rng = np.random.default_rng(1)
pix = rng.choice(hp.nside2npix(NSIDE), size=300, replace=False)


def mbb(nu, beta, temp, nu0=353.0):
    x, x0 = h_over_k * nu / temp, h_over_k * nu0 / temp
    return (nu / nu0) ** (beta + 1) * np.expm1(x0) / np.expm1(x)


for tag in SKIES:
    p = P[tag]
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    for i in pix:
        ax.loglog(nu, (nu / 23.0) ** p["beta_pl"][i], color=SYNC, lw=0.8, alpha=0.25)
        ax.loglog(nu, mbb(nu, p["beta_dust"][i], p["temp_dust"][i]), color=DUST, lw=0.8, alpha=0.25)
    ax.text(28, 3e-3, "synchrotron", color=SYNC, fontsize=15, va="center")
    ax.text(110, 0.6, "dust", color=DUST, fontsize=15, va="center", ha="center")
    ax.set_xlim(nu[0], nu[-1])
    ax.set_ylim(1e-4, 3)
    ax.set_xlabel(r"frequency $\nu$  [GHz]")
    ax.set_ylabel("SED  (normalised)")
    ax.set_xticks([30, 100, 300])
    ax.set_xticklabels(["30", "100", "300"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.tick_params(which="minor", top=False, right=False)
    fig.savefig(HERE / f"{tag}_sed.svg", transparent=True)
    plt.close(fig)
    print(f"wrote {tag}_sed.svg")
