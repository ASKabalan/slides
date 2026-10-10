#!/usr/bin/env python3
# ENV: jax-fli
"""
How deep the box must reach to lens the DES Y3 source bins, for the slide on the volume a DES Y3
analysis needs.

The top panel of These_wassim/figures/chap6/geometry.py, set for the slide: the DES Y3 source
n(z) per tomographic bin with a secondary comoving-distance axis. The lensing kernels are shown on
the weak-lensing slide (3_observation/weaklensing/04_tomographic_bins), so the q(z) panel, the
CosmoGrid shell bar and the depth legend are left out here. The effective end of each bin is the thesis rule: the last z where n(z) is
at least 10% of its peak. The cosmology is the CosmoGridV1 one, read from a single cached
nside-2048 density shell of ASKabalan/jax-fli-experiments.

Outputs (this directory), at the same size so they stack exactly:
  depth.svg         the n(z) panel
  depth_lines.svg   the dotted comoving depth of bins 2, 3 and 4 alone, on a transparent page
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import INK, skip_if_built, slide_style

OUTS = ["depth.svg", "depth_lines.svg"]
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
THRESH_FRAC = 0.10
ZGRID = np.linspace(0.005, 2.995, 600)

slide_style(scale=1.2)
# the two pages must keep the same frame to stack, so no tight crop
plt.rcParams["savefig.bbox"] = "standard"


def cosmogrid_cosmology():
    """The cosmology stored with the first CosmoGrid shell: the local experiments clone when
    present, else that one file from the HuggingFace dataset."""
    local = LOCAL / SHELL0
    fp = local if local.exists() else hf_hub_download(REPO, SHELL0, repo_type="dataset")
    cat = Catalog.from_dataset(
        load_dataset("parquet", data_files=str(fp), split="train").with_format("numpy"))
    return cat.cosmology[0]


def z_end(v):
    idx = np.where(v >= THRESH_FRAC * float(v.max()))[0]
    return float(ZGRID[idx[-1]])


cosmo = cosmogrid_cosmology()
nz_list = get_des_y3_nz_shear()
nz_vals = np.array([np.asarray(nz(jnp.asarray(ZGRID))) for nz in nz_list])
z_mean = [float(np.trapezoid(ZGRID * v, ZGRID) / np.trapezoid(v, ZGRID)) for v in nz_vals]
z_ends = [z_end(v) for v in nz_vals]
chi = lambda z: float(jc.background.radial_comoving_distance(cosmo, jc.utils.z2a(z)).squeeze())
chi_ends = [chi(z) for z in z_ends]
for i, (z, r) in enumerate(zip(z_ends, chi_ends)):
    print(f"bin {i + 1}: z_end = {z:.3f}  chi = {r:.0f} Mpc/h  full-sky box 2chi = {2 * r:.0f}")

cmap = plt.get_cmap("YlOrRd")
colors = [cmap(x) for x in np.linspace(0.35, 0.95, len(nz_list))]
XMAX = 2.4

fig, ax1 = plt.subplots(figsize=(8.0, 4.6))
fig.subplots_adjust(left=0.09, right=0.985, top=0.84, bottom=0.135)

for i, v in enumerate(nz_vals):
    ax1.plot(ZGRID, v, color=colors[i], lw=2.0, label=rf"Bin {i + 1} ($\bar{{z}} = {z_mean[i]:.2f}$)")
ax1.set_ylabel(r"$n(z)$")
ax1.set_ylim(0.0, 1.12 * float(nz_vals.max()))
ax1.set_xlim(0.0, XMAX)
ax1.set_xlabel(r"Redshift $z$")
ax1.legend(loc="upper right", labelspacing=0.3, handlelength=1.6)

z2chi = lambda z: np.asarray(jc.background.radial_comoving_distance(cosmo, jc.utils.z2a(np.atleast_1d(z))))
chi2z = lambda c: np.asarray(jc.utils.a2z(jc.background.a_of_chi(cosmo, np.atleast_1d(c))))
secax = ax1.secondary_xaxis("top", functions=(z2chi, chi2z))
secax.set_xlabel(r"Comoving distance $\chi$  [Mpc/$h$]", labelpad=6)
secax.tick_params(colors=INK)

fig.savefig(HERE / "depth.svg")

# the overlay: the same page with everything hidden but the depth lines
for art in ax1.get_children():
    art.set_visible(False)
secax.set_visible(False)
ax1.patch.set_visible(False)
for i in (1, 2, 3):
    ax1.axvline(z_ends[i], color=colors[i], lw=2.2, ls=(0, (1.2, 1.6)), visible=True)
    ax1.text(z_ends[i] + 0.025, 0.96, rf"$\chi = {chi_ends[i]:.0f}$ Mpc/$h$",
             transform=ax1.get_xaxis_transform(), color=colors[i], fontsize=14,
             rotation=90, ha="left", va="top",
             bbox=dict(facecolor="#faf7f0", edgecolor="none", alpha=0.9, pad=1.5))
fig.savefig(HERE / "depth_lines.svg")
print("wrote", ", ".join(OUTS))
