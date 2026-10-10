#!/usr/bin/env python3
# ENV: jax-fli
"""
Starlet l1 norm of the convergence, this forward model against CosmoGrid, for the starlet slide.
The data and the statistic of the thesis figure chap6/starlet_l1.pdf (its script is
These_wassim/figures/chap6/starlet_l1.py):
  model      05c equal-volume lightcone, 20 shells drifted on the lightcone, Born with Gauss-Legendre
             quadrature (05c-equal-volume/kappa_gauss_legendre/born_gl_drift_20), 2560^3 mesh;
  reference  CosmoGrid cosmo_172798 pkdgrav3 lightcone through the same Born integral
             (00-cosmogrid/cosmo_172798/kappa/kappa_born_s3.parquet);
tomographic bin 3, degraded to nside 512, five starlet scales (jax_fli starlet_coefficients). At
each scale j the coefficients are put in signal to noise nu = w_j / sigma_j with sigma_j the
reference standard deviation, used for both runs, and the l1 norm of a bin in nu is the sum of
|nu| over its pixels (40 bins over [-8, 8]).

Output (this directory): starlet_l1.svg, the norm per scale above the ratio to the reference
"""

import gc
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import BLUE, GREY, INK, KW, skip_if_built, slide_style

OUT = "starlet_l1.svg"
skip_if_built(HERE, OUT)

CACHE = (ROOT / "5_contribution/fli") / ".cache"
EXP = Path("/home/wassim/Projects/NBody/jax-fli-experiments")
MODEL = EXP / ("05-spacing-n-stepping/05c-equal-volume/kappa_gauss_legendre/"
               "born_gl_drift_20/BORN_kappa_gl_drift_3bin_20.parquet")
REF = EXP / "00-cosmogrid/cosmo_172798/kappa/kappa_born_s3.parquet"
NSIDE, NSCALES, BIN, NBINS, SNR = 512, 5, 2, 40, (-8.0, 8.0)


def load():
    npz = CACHE / "starlet_l1_bin3.npz"
    if npz.exists():
        d = np.load(npz)
        return d["model"], d["ref"], d["centres"]
    import jax

    jax.config.update("jax_enable_x64", True)
    import healpy as hp
    import jax.numpy as jnp
    from jax_fli.io import Catalog

    def coefficients(path):
        field = Catalog.from_parquet(str(path)).field[0]
        single = field[BIN]
        arr = hp.ud_grade(np.asarray(single.array, dtype=np.float64), nside_out=NSIDE)
        single = single.replace(array=jnp.asarray(arr), nside=NSIDE)
        out = np.asarray(single.starlet_coefficients(nscales=NSCALES).array, dtype=np.float64)
        del field, single, arr
        gc.collect()
        return out

    coef = {"ref": coefficients(REF), "model": coefficients(MODEL)}
    sigma = coef["ref"].std(axis=1)
    edges = np.linspace(*SNR, NBINS + 1)
    l1 = {}
    for key, c in coef.items():
        l1[key] = np.zeros((NSCALES, NBINS))
        for j in range(NSCALES):
            nu = c[j] / sigma[j]
            i = np.digitize(nu, edges) - 1
            keep = (i >= 0) & (i < NBINS)
            l1[key][j] = np.bincount(i[keep], weights=np.abs(nu[keep]), minlength=NBINS)
    centres = 0.5 * (edges[:-1] + edges[1:])
    CACHE.mkdir(exist_ok=True)
    np.savez(npz, model=l1["model"], ref=l1["ref"], centres=centres)
    return l1["model"], l1["ref"], centres


L1_MODEL, L1_REF, NU = load()
for j in range(NSCALES):
    neg, pos = NU < 0, NU > 0
    print(f"scale {j}: tails model/reference {L1_MODEL[j][neg].sum() / L1_REF[j][neg].sum():.3f} "
          f"{L1_MODEL[j][pos].sum() / L1_REF[j][pos].sum():.3f}")

slide_style()
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

plt.rcParams["savefig.bbox"] = None
SUPPORT = L1_REF > 0.01 * L1_REF.max(axis=1, keepdims=True)
RATIO = np.where(SUPPORT, L1_MODEL / np.where(SUPPORT, L1_REF, 1) - 1, np.nan)
# thesis fig. starlet_response: scale 0 a high-pass above l ~ 700, scale 1 peaks near l = 228, each
# band-pass near half the multipole of the one before, scale 4 the low-pass residual
BANDS = ("ℓ ≳ 700", "ℓ ≈ 230", "ℓ ≈ 115", "ℓ ≈ 60", "low-pass")

fig = plt.figure(figsize=(10.4, 5.6))
top_max = max(L1_MODEL.max(), L1_REF.max()) / 1e5 * 1.08
for j in range(NSCALES):
    x0 = 0.075 + j * 0.186
    ax = fig.add_axes([x0, 0.369, 0.165, 0.551])
    ar = fig.add_axes([x0, 0.09, 0.165, 0.25])
    ax.plot(NU, L1_REF[j] / 1e5, color=KW, ls="--", lw=2.0, label="CosmoGrid")
    ax.plot(NU, L1_MODEL[j] / 1e5, color=BLUE, lw=2.0, label="this work")
    ar.axhspan(-0.1, 0.1, color="#d8dde6", lw=0)
    ar.axhline(0, color=INK, lw=0.8)
    ar.plot(NU, RATIO[j], color=INK, lw=1.8)
    for a in (ax, ar):
        a.set_xlim(-4.5, 6.5)
        a.xaxis.set_major_locator(FixedLocator([-4, 0, 4]))
    ax.set_ylim(0, top_max)
    ar.set_ylim(-0.4, 0.4)
    ar.set_yticks([-0.3, 0, 0.3])
    ax.tick_params(labelbottom=False)
    if j:
        ax.tick_params(labelleft=False)
        ar.tick_params(labelleft=False)
    ax.set_title(f"scale {j}\n{BANDS[j]}", fontsize=12.5, color=INK, pad=4, linespacing=1.15)
    ar.set_xlabel(r"$\nu = w_j/\sigma_j$", labelpad=1, fontsize=12.5)
fig.axes[0].set_ylabel(r"$\ell_1$ norm  [$10^5$]", fontsize=12.5)
fig.axes[1].set_ylabel("ratio − 1", fontsize=12.5)
fig.axes[0].legend(loc="upper right", fontsize=10.5, handlelength=1.6, borderaxespad=0.2,
                   labelspacing=0.2)
fig.savefig(HERE / OUT)
print(f"wrote {OUT}")
