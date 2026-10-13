#!/usr/bin/env python3
# ENV: jax-fli
"""
From density shells to convergence, for the Born slide: the lensing weight of each shell and the
convergence maps it produces.

The run is the thesis lightcone that the CosmoGrid slide compares next: jax-fli experiment 05c
(ASKabalan/jax-fli-experiments, 05-spacing-n-stepping/05c-equal-volume), 2560^3 mesh in a 5000 Mpc/h
box, 25 equal-volume shells (outer shells floored at 60 Mpc/h) drifted on the lightcone, Born
convergence with Gauss-Legendre quadrature across each shell (born_gl_drift_25), for the three stage-3 source bins (jax_fli.data
get_stage3_nz_shear()[:3]).

  born_weights.svg  the lensing kernel of each bin, averaged over its n(z),
                      K_i(chi) = int dz n_i(z) chi (1 + z(chi)) (1 - chi / chi_s(z)),  chi < chi_s(z),
                    as a dashed curve, and over each shell j the weight it receives in the Born sum,
                    W_ij / Delta r_j (the kernel integrated across the shell, per unit width), as steps;
                    arbitrary units, one scale for the three bins.
  born_kappa.png    the convergence of the three bins, nside 2048 brought to 256, mean removed,
                    Mollweide, each bin on its own colour scale (1st to 99.5th percentile).
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import INK, skip_if_built, slide_style

OUTS = ["born_weights.svg", "born_kappa.png"]
skip_if_built(HERE, *OUTS)

CACHE = HERE.parent / ".cache"
REPO = "ASKabalan/jax-fli-experiments"
GEOM = "05-spacing-n-stepping/05c-equal-volume/density_spectra/spectra_exp5c_drift_25.parquet"
KAPPA = ("05-spacing-n-stepping/05c-equal-volume/kappa_gauss_legendre/born_gl_drift_25/"
         "BORN_kappa_gl_drift_3bin_25.parquet")
NSIDE_SHOW = 256


def load():
    npz = CACHE / "born_05c_25.npz"
    if npz.exists():
        d = np.load(npz)
        return {k: d[k] for k in d.files}
    import jax

    jax.config.update("jax_enable_x64", True)
    import healpy as hp
    import jax.numpy as jnp
    import jax_cosmo as jc
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from jax_fli.data import get_stage3_nz_shear
    from jax_fli.io import Catalog

    root = snapshot_download(REPO, repo_type="dataset", allow_patterns=[GEOM, KAPPA])
    cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{GEOM}", split="train"))
    cosmo, spec = cat.cosmology[0], cat.field[0]
    chi_c, w = np.asarray(spec.comoving_centers), np.asarray(spec.density_width)
    o = np.argsort(chi_c)
    edges = np.r_[0.0, np.cumsum(w[o])]

    # the n(z)-averaged kernel on a fine grid in chi
    chi = np.linspace(1.0, edges[-1], 2500)
    z_of_chi = 1 / np.asarray(jc.background.a_of_chi(cosmo, jnp.asarray(chi))) - 1
    zs = np.linspace(0.005, 3.0, 1200)
    chi_s = np.asarray(jc.background.radial_comoving_distance(cosmo, jc.utils.z2a(jnp.asarray(zs))))
    kern = []
    for nz in get_stage3_nz_shear()[:3]:
        n = np.asarray(nz(jnp.asarray(zs)))
        n = n / np.trapezoid(n, zs)
        g = np.clip(1 - chi[:, None] / chi_s[None, :], 0, None)        # (chi, z_s)
        kern.append(chi * (1 + z_of_chi) * np.trapezoid(g * n[None, :], zs, axis=1))
    kern = np.array(kern)
    steps = np.array([[np.trapezoid(k[(chi >= a) & (chi <= b)], chi[(chi >= a) & (chi <= b)]) / (b - a)
                       for a, b in zip(edges[:-1], edges[1:])] for k in kern])

    field = Catalog.from_parquet(f"{root}/{KAPPA}").field[0]
    maps = np.array([hp.ud_grade(np.asarray(field.array[b], dtype=np.float64), NSIDE_SHOW) for b in range(3)])
    z_eff = [float(np.trapezoid(zs * nz(jnp.asarray(zs)), zs) / np.trapezoid(nz(jnp.asarray(zs)), zs))
             for nz in get_stage3_nz_shear()[:3]]
    CACHE.mkdir(exist_ok=True)
    np.savez(npz, chi=chi, kern=kern, edges=edges, steps=steps, maps=maps, z_eff=np.array(z_eff))
    return load()


D = load()
print("shell edges [Mpc/h]:", D["edges"].round(0))
print("mean source redshift per bin:", D["z_eff"].round(2))

slide_style()
import healpy as hp
import matplotlib.pyplot as plt

plt.rcParams["savefig.bbox"] = None
BINS = ["#f2b58a", "#d9772e", "#8a3510"]     # bins 1-3, light to dark (the spacing slide's palette)

# --- the weights
fig = plt.figure(figsize=(6.0, 3.6))
ax = fig.add_axes([0.04, 0.16, 0.93, 0.78])
top = 1.42 * D["kern"].max()                  # headroom for the legend above the curves
for j in range(len(D["edges"]) - 1):
    ax.axvspan(D["edges"][j], D["edges"][j + 1], color="#d8dde6" if j % 2 else "#eef1f5", lw=0, zorder=0)
for i, c in enumerate(BINS):
    ax.stairs(D["steps"][i], D["edges"], color=c, lw=2.4, zorder=3, label=f"bin {i + 1}")
    ax.plot(D["chi"], D["kern"][i], color=c, ls="--", lw=1.4, zorder=2)
ax.set_xlim(0, D["edges"][-1])
ax.set_ylim(0, top)
ax.set_yticks([])
ax.set_xticks([0, 500, 1000, 1500, 2000, 2500])
ax.set_xlabel(r"$\chi$  [Mpc/$h$]", labelpad=2)
ax.set_ylabel("lensing weight", fontsize=12.5)
ax.plot([], [], color=INK, lw=2.0, label="weight per shell")
ax.plot([], [], color=INK, ls="--", lw=1.4, label="continuous kernel")
ax.legend(loc="upper right", ncol=2, fontsize=10.5, handlelength=1.6, borderaxespad=0.3, labelspacing=0.2, columnspacing=1.0)
fig.savefig(HERE / "born_weights.svg")
plt.close(fig)
print("wrote born_weights.svg")

# --- the maps
moll = hp.projector.MollweideProj(xsize=600)
fig = plt.figure(figsize=(9.6, 1.95), dpi=200, facecolor="#faf7f0")
for i in range(3):
    m = D["maps"][i] - D["maps"][i].mean()
    img = moll.projmap(m, lambda x, y, z: hp.vec2pix(NSIDE_SHOW, x, y, z))
    img = np.ma.masked_invalid(np.where(np.isfinite(img), img, np.nan))
    lo, hi = np.percentile(m, [1, 99.5])
    ax = fig.add_axes([0.005 + i * 0.335, 0.02, 0.32, 0.8])
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(alpha=0)
    ax.imshow(img, origin="lower", cmap=cmap, vmin=lo, vmax=hi, interpolation="bilinear")
    h, w = img.shape
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(w / 2 + (w / 2 - 1) * np.cos(t), h / 2 + (h / 2 - 1) * np.sin(t), color="#3b3b3b", lw=0.8)
    ax.axis("off")
    ax.set_title(f"bin {i + 1}", fontsize=15,
                 color=INK, pad=3)
fig.savefig(HERE / "born_kappa.png", facecolor="#faf7f0")
plt.close(fig)
print("wrote born_kappa.png")
