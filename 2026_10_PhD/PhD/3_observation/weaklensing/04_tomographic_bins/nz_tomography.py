#!/usr/bin/env python3
# ENV: jax-fli
"""
The DES Y3 tomographic bins, for the slide "From galaxy shapes to shear": the source redshift
distribution of each bin (the lensing kernels are drawn on the tomography slide, by
05_tomography/tomography.py).

As 5_contribution/fli/06_observable/depth.py: the DES Y3 source n(z) shipped with jax_fli, the CosmoGridV1 cosmology
read from one cached CosmoGrid shell, a secondary comoving-distance axis on top, and the YlOrRd bin
colours of the deck.

Output (this directory): nz.svg, n_i(z) with the mean redshift of each bin in the legend
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import INK, skip_if_built, slide_style

OUTS = ["nz.svg"]
skip_if_built(HERE, *OUTS)

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax_cosmo as jc
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from jax_fli.data import get_des_y3_nz_shear
from jax_fli.io import Catalog

REPO = "ASKabalan/jax-fli-experiments"
SHELL0 = "00-cosmogrid/cosmo_000001/density/cosmogrid_density_nside2048_shell_000.parquet"
LOCAL = Path.home() / "Projects/NBody/jax-fli-experiments"
ZGRID = np.linspace(0.005, 2.995, 600)
XMAX = 2.0

local = LOCAL / SHELL0
fp = local if local.exists() else hf_hub_download(REPO, SHELL0, repo_type="dataset")
cosmo = Catalog.from_dataset(
    load_dataset("parquet", data_files=str(fp), split="train").with_format("numpy")).cosmology[0]

nz_list = get_des_y3_nz_shear()
nz = np.array([np.asarray(f(jnp.asarray(ZGRID))) for f in nz_list])
z_mean = [float(np.trapezoid(ZGRID * v, ZGRID) / np.trapezoid(v, ZGRID)) for v in nz]

z2chi = lambda z: np.asarray(jc.background.radial_comoving_distance(cosmo, jc.utils.z2a(np.atleast_1d(z))))
chi2z = lambda c: np.asarray(jc.utils.a2z(jc.background.a_of_chi(cosmo, np.atleast_1d(c))))

slide_style(scale=1.15)
cmap = plt.get_cmap("YlOrRd")
colors = [cmap(x) for x in np.linspace(0.35, 0.95, len(nz))]


def panel(curves, ylabel, out, legend):
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    fig.subplots_adjust(left=0.085, right=0.98, top=0.84, bottom=0.135)
    for i, v in enumerate(curves):
        ax.plot(ZGRID, v, color=colors[i], lw=2.0,
                label=rf"Bin {i + 1} ($\bar{{z}} = {z_mean[i]:.2f}$)")
    ax.set_xlim(0.0, XMAX)
    ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.set_ylim(0.0, 1.08 * float(np.max(curves)))
    ax.set_xlabel(r"redshift $z$")
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(loc="upper right", labelspacing=0.35, handlelength=1.6, fontsize=13.5, frameon=False)
    sec = ax.secondary_xaxis("top", functions=(z2chi, chi2z))
    sec.set_xticks([0, 1000, 2000, 3000, 4000])
    sec.set_xlabel(r"comoving distance $\chi$  [Mpc/$h$]", labelpad=6)
    sec.tick_params(colors=INK)
    fig.savefig(HERE / out, transparent=True)
    plt.close(fig)
    print(f"wrote {out}")


panel(nz, r"$n_i(z)$", "nz.svg", legend=True)
