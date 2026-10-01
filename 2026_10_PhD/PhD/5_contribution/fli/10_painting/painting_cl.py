#!/usr/bin/env python3
# ENV: jax-fli
"""
How the three spherical deposit schemes smooth one lightcone shell, for the painting slide.

The data and the binning of the thesis figure chap6/spherical_painting.pdf (its script is
These_wassim/figures/chap6/spherical_painting.py): experiment 03 of ASKabalan/jax-fli-experiments,
one shell painted at N_side = 2048 (shell 8, chi ~ 822 Mpc/h, z ~ 0.295) by nearest grid point,
bilinear, and a Gaussian radial basis function of 0.8 and 1.5 pixel FWHM, binned in bands of 32
multipoles. Top: l(l+1) C_l / 2 pi. Bottom: each scheme over nearest grid point, minus one.

Outputs (this directory), same figure size and axes box, so the slide can swap one for the other:
  painting_cl.svg        l up to 1500
  painting_cl_zoom.svg   l from 200 to 1000
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from _common import INK, skip_if_built, slide_style

OUTS = ["painting_cl.svg", "painting_cl_zoom.svg"]
skip_if_built(HERE, *OUTS)

import jax

jax.config.update("jax_enable_x64", True)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from huggingface_hub import snapshot_download
from jax_fli.io import Catalog
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter

REPO = "ASKabalan/jax-fli-experiments"
SHELL, LMAX, NLB = 8, 1500, 32
SPEC = "03-spherical-painting/spectra/spectra_exp3_{tag}_native2048.parquet"
SCHEMES = [                                          # tag, label, colour, style, width
    ("ngp", "nearest grid point", "#555555", ":", 2.6),
    ("bilinear", "bilinear (4 pixels)", "#521463", "-.", 2.2),
    ("rbf08", "RBF, 0.8 pixel FWHM", "#C2560A", "-", 2.4),
    ("rbf15", "RBF, 1.5 pixel FWHM", "#3B6FB6", "--", 2.2),
]

root = snapshot_download(REPO, repo_type="dataset",
                         allow_patterns=[SPEC.format(tag=t) for t, *_ in SCHEMES])
cl = {}
for tag, *_ in SCHEMES:
    cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{SPEC.format(tag=tag)}",
                                            split="train"))
    binned = cat.field[0].bin(nlb=NLB, lmin=2)
    ell = np.asarray(binned.wavenumber)
    cl[tag] = np.asarray(binned.array)[SHELL]
dl = ell * (ell + 1) / (2 * np.pi)
for l0 in (300, 1000, 1480):
    k = int(np.argmin(np.abs(ell - l0)))
    print(f"l = {ell[k]:.0f}: " + ", ".join(f"{t} {cl[t][k] / cl['ngp'][k] - 1:+.3f}" for t, *_ in SCHEMES[1:]))

slide_style(scale=1.3)
plt.rcParams["savefig.bbox"] = "standard"      # the same page for both files, so they swap in place
PLAIN = FuncFormatter(lambda v, _p: f"{v:g}")


def figure(lmin, lmax, out):
    fig = plt.figure(figsize=(7.6, 5.6))
    top = fig.add_axes([0.15, 0.40, 0.79, 0.57])
    bot = fig.add_axes([0.15, 0.12, 0.79, 0.25], sharex=top)
    keep = (ell >= lmin * 0.97) & (ell <= lmax * 1.03)
    for tag, label, colour, ls, lw in SCHEMES:
        top.plot(ell, dl * cl[tag], color=colour, ls=ls, lw=lw, label=label,
                 zorder=4 if tag == "ngp" else 3)
        if tag != "ngp":
            bot.plot(ell, cl[tag] / cl["ngp"] - 1, color=colour, ls=ls, lw=lw)
    bot.axhline(0, color="#555555", ls=":", lw=1.6)
    top.set_xscale("log")
    top.set_yscale("log")
    top.set_xlim(lmin, lmax)
    y = np.concatenate([dl[keep] * cl[t][keep] for t, *_ in SCHEMES])
    top.set_ylim(y.min() / 1.12, y.max() * 1.12)
    r = np.concatenate([cl[t][keep] / cl["ngp"][keep] - 1 for t, *_ in SCHEMES[1:]] + [[0.0]])
    pad = 0.12 * (r.max() - r.min())
    bot.set_ylim(r.min() - pad, r.max() + pad)
    top.set_ylabel(r"$\ell(\ell+1)\,C_\ell/2\pi$")
    bot.set_ylabel(r"$C_\ell/C_\ell^{\rm NGP}-1$", fontsize=14)
    bot.set_xlabel(r"multipole $\ell$")
    ticks = [30, 100, 300, 1000] if lmin < 100 else [200, 300, 500, 700, 1000]
    for ax in (top, bot):
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(PLAIN)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="both", ls=":", alpha=0.4)
    top.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=12))
    top.yaxis.set_major_formatter(PLAIN)
    top.yaxis.set_minor_formatter(NullFormatter())
    top.tick_params(which="both", labelbottom=False)
    top.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=13)
    fig.savefig(HERE / out, transparent=True, bbox_inches=None)
    plt.close(fig)
    print(f"wrote {out}")


figure(20, LMAX, "painting_cl.svg")
figure(200, 1000, "painting_cl_zoom.svg")
