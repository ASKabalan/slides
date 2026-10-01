#!/usr/bin/env python3
# ENV: shared
"""
Two skies with the same power spectrum.

A full-sky density shell from my forward model (shell 5, z = 0.07, of the CosmoGrid-matched run:
5000 Mpc/h box, 2048^3 particle mesh), and a Gaussian random field drawn from that shell's own
angular power spectrum. A two-point analysis cannot tell them apart; the eye can at once.

The shell comes from the HuggingFace dataset ASKabalan/jax-fli-experiments
(06-cosmogrid-shells/density/cosmogrid_3bin_fullsky_slab/shell_0005.parquet), downgraded and
kept in .cache/. The Gaussian draw is seeded.

Outputs (this directory):
  cosmic_web.gif, gaussian.gif   rotating, looping, transparent (+ first-frame .png stills)
  cl_overlay.svg                 the two spectra, superposed
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from _common import KW, KW2, ortho_frame, skip_if_built, slide_style, write_alpha_gif

OUTS = ["cosmic_web.gif", "gaussian.gif", "cl_overlay.svg"]
skip_if_built(HERE, *OUTS)

import healpy as hp

CACHE = HERE / ".cache"
NSIDE_SRC, NSIDE, SHELL = 512, 256, 5
N_FRAMES, LAT, PX = 40, 25.0, 300
REPO_PATH = ("06-cosmogrid-shells/density/cosmogrid_3bin_fullsky_slab/"
             f"shell_{SHELL:04d}.parquet")


def shell_map() -> np.ndarray:
    npz = CACHE / f"shell{SHELL}_nside{NSIDE_SRC}.npz"
    if not npz.exists():
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        src = hf_hub_download("ASKabalan/jax-fli-experiments", REPO_PATH, repo_type="dataset")
        col = pq.read_table(src, columns=["array"]).column("array").combine_chunks()
        rho = hp.ud_grade(np.asarray(col.values.values, dtype=np.float64), NSIDE_SRC)
        CACHE.mkdir(exist_ok=True)
        np.savez(npz, delta=rho / rho.mean() - 1.0)
    return hp.ud_grade(np.load(npz)["delta"], NSIDE)


delta = shell_map()
lmax = 3 * NSIDE - 1
cl = hp.anafast(delta, lmax=lmax)
np.random.seed(20261012)
gauss = hp.synfast(cl, NSIDE, lmax=lmax)
cl_g = hp.anafast(gauss, lmax=lmax)

# ---------------------------------------------------------------- rotating maps
vmin, vmax = -1.0, float(np.percentile(delta, 99.5))    # one scale, set by the shell
lons = np.arange(N_FRAMES) * 360.0 / N_FRAMES
for m, out in ((delta, "cosmic_web.gif"), (gauss, "gaussian.gif")):
    m = np.clip(m, vmin, vmax)
    frames = [ortho_frame(m, lon, LAT, PX, cmap="magma", min=vmin, max=vmax) for lon in lons]
    write_alpha_gif(frames, HERE / out)
    print(f"wrote {out}")

# ---------------------------------------------------------------- spectra
slide_style(scale=1.3)
import matplotlib.pyplot as plt

# average in log-spaced multipole bins so the overlap reads at a glance
edges = np.unique(np.geomspace(2, lmax + 1, 28).astype(int))
ells = np.arange(lmax + 1)


def binned(c):
    d = ells * (ells + 1) * c / (2 * np.pi)
    return np.array([d[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])


ell = np.sqrt(edges[:-1] * edges[1:])
fig, ax = plt.subplots(figsize=(7.2, 4.0))
ax.loglog(ell, binned(cl), color=KW, lw=3.4, label="cosmic web")
ax.loglog(ell, binned(cl_g), color=KW2, lw=2.2, ls="--", label=r"Gaussian field, same $C_\ell$")
ax.set_xlabel(r"multipole $\ell$")
ax.set_ylabel(r"$\ell(\ell+1)\,C_\ell/2\pi$")
ax.set_xlim(ell[0], ell[-1])
ax.legend(loc="lower right")
fig.savefig(HERE / "cl_overlay.svg")
print("wrote cl_overlay.svg")
