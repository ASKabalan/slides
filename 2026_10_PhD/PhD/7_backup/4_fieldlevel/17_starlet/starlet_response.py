#!/usr/bin/env python3
# ENV: jax-fli
"""
What a starlet decomposition does, for the starlet slide: the slide version of the thesis figure
chap6/starlet_response.pdf (its script is These_wassim/figures/chap6/starlet_response.py).

Top, the convergence of tomographic bin 3 of the thesis run (jax-fli experiment 05c, equal volume,
20 drifted shells, born_gl_drift_20), brought to nside 512 and split by the pycs CMRStarlet transform
into five scales, shown as a 15 degree gnomonic cut-out per scale, coarse to fine, each on its own
symmetric stretch (99th percentile). Bottom, the squared multipole response of each scale, measured on
a white-noise map as the cross power of the coefficient map with its input over the input power,
peak-normalised; the five responses add up to one.

Output (this directory): starlet_response.svg
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import BLUE, INK, KW, KW2, RED, TEAL, skip_if_built, slide_style

OUT = "starlet_response.svg"
skip_if_built(HERE, OUT)

CACHE = (ROOT / "5_contribution/fli") / ".cache"
KAPPA = Path("/home/wassim/Projects/NBody/jax-fli-experiments/05-spacing-n-stepping/05c-equal-volume/"
             "kappa_gauss_legendre/born_gl_drift_20/BORN_kappa_gl_drift_3bin_20.parquet")
NSIDE, NSCALES, BIN, SEED = 512, 5, 2, 0
CUT_PIX, CUT_RESO = 256, 3.5                 # a 14.9 degree window


def load():
    npz = CACHE / "starlet_response.npz"
    if npz.exists():
        d = np.load(npz)
        return d["response"], d["cutouts"], d["ell"]
    import healpy as hp
    from jax_fli.io import Catalog
    from pycs.sparsity.mrs.mrs_starlet import CMRStarlet

    tr = CMRStarlet()
    tr.init_starlet(NSIDE, nscale=NSCALES)
    lmax = int(tr.lmax)
    np.random.seed(SEED)
    white = hp.alm2map(hp.synalm(np.ones(lmax + 1), lmax=lmax), NSIDE)
    tr.transform(white)
    coef = np.stack([np.asarray(tr.coef[j]) for j in range(NSCALES)])
    alm_in = hp.map2alm(white, lmax=lmax, iter=3)
    cl_in = hp.alm2cl(alm_in, lmax=lmax)
    response = np.stack([hp.alm2cl(alm_in, hp.map2alm(coef[j], lmax=lmax, iter=3), lmax=lmax) / cl_in
                         for j in range(NSCALES)])
    assert np.allclose(response.sum(0)[2:], 1.0, atol=1e-6)
    kappa = Catalog.from_parquet(str(KAPPA)).field[0][BIN]
    tr.transform(hp.ud_grade(np.asarray(kappa.array, dtype=np.float64), nside_out=NSIDE))
    proj = hp.projector.GnomonicProj(rot=(0.0, 0.0), xsize=CUT_PIX, reso=CUT_RESO)
    cutouts = np.stack([proj.projmap(np.asarray(tr.coef[j]), lambda x, y, z: hp.vec2pix(NSIDE, x, y, z))
                        for j in range(NSCALES)])
    CACHE.mkdir(exist_ok=True)
    np.savez(npz, response=response, cutouts=cutouts, ell=np.arange(lmax + 1))
    return response, cutouts, np.arange(lmax + 1)


RESP, CUT, ELL = load()

slide_style()
import matplotlib.pyplot as plt

plt.rcParams["savefig.bbox"] = None
COLOURS = (BLUE, KW, TEAL, RED, KW2)          # scales 0 (finest) .. 4 (coarsest)

fig = plt.figure(figsize=(10.4, 5.6))
for col, j in enumerate(range(NSCALES - 1, -1, -1)):     # coarse to fine, left to right
    ax = fig.add_axes([0.07 + col * 0.183, 0.655, 0.165, 0.277])
    span = float(np.percentile(np.abs(CUT[j]), 99))
    ax.imshow(CUT[j], cmap="RdBu_r", origin="lower", vmin=-span, vmax=span)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(COLOURS[j])
        sp.set_linewidth(2.4)
    ax.set_title(f"scale {j}", color=COLOURS[j], fontsize=14, pad=4)

ax = fig.add_axes([0.07, 0.09, 0.9, 0.48])
for j in range(NSCALES):
    sq = RESP[j][2:] ** 2
    sq = sq / sq.max()
    ax.plot(ELL[2:], sq, color=COLOURS[j], lw=2.4)
    x = max(ELL[2:][int(np.argmax(sq >= 0.99))], 3)
    ax.text(x, 1.04, f"{j}", color=COLOURS[j], fontsize=13, ha="center", va="bottom")
ax.set_xscale("log")
ax.set_xlim(2, ELL[-1])
ax.set_ylim(-0.03, 1.22)
ax.set_yticks([0, 0.5, 1])
ax.set_xlabel(r"$\ell$", labelpad=0)
ax.set_ylabel(r"response $\psi_j(\ell)^2$", fontsize=13)
fig.savefig(HERE / OUT)
plt.close(fig)
for j in range(NSCALES):
    sq = RESP[j][2:] ** 2
    print(f"scale {j}: peak at l = {ELL[2:][int(np.argmax(sq))]}")
print(f"wrote {OUT}")
