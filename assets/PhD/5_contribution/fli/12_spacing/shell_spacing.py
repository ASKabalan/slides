#!/usr/bin/env python3
# ENV: jax-fli
"""
Uniform scale-factor against equal-volume shell spacing, for the slide on choosing the spacing.

The two phase-matched 20-shell lightcones of jax-fli experiment 05f at 2048^3 (5000 Mpc/h box,
observer at the centre, BullFrog, drift on the lightcone, nside 2048): exp5f_m2048_bf_a and
exp5f_m2048_bf_equal_vol, the latter with its outer shells floored at 50 Mpc/h. One row per
spacing, three panels:
  1. the four DES Y3 lensing kernels q(chi) over the 20 shells, drawn as alternating top-hat bands;
  2. the particles each shell receives, nbar * 4/3 pi (r_far^3 - r_near^3), nbar = 2048^3 / 5000^3,
     against one particle per HEALPix pixel at nside 2048;
  3. the angular power spectrum of a near, a middle and a far shell (the first, the one closest
     to chi = 1300 Mpc/h, the last), binned in bands of 32 multipoles, against the Limber
     prediction for the shell's number counts times the squared pixel window.
Shell geometry and spectra are read from the local copy of ASKabalan/jax-fli-experiments.

Outputs (this directory), same size and axes boxes so the slide stacks them:
  spacing_a.svg          uniform scale factor
  spacing_equal_vol.svg  equal volume
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from _common import BLUE, GREY, INK, KW, KW2, skip_if_built, slide_style

OUTS = ["spacing_a.svg", "spacing_equal_vol.svg"]
skip_if_built(HERE, *OUTS)

CACHE = HERE.parent / ".cache"
EXP = Path("/home/wassim/Projects/NBody/jax-fli-experiments/05-spacing-n-stepping/05f-mesh")
RUNS = {"a": "exp5f_m2048_bf_a", "equal_vol": "exp5f_m2048_bf_equal_vol"}
BOX, MESH, NSIDE, LMAX, NLB = 5000.0, 2048, 2048, 1500, 32
CHI_MID = 1300.0


def load():
    npz = CACHE / "shell_spacing_05f.npz"
    if npz.exists():
        d = np.load(npz)
        return {k: d[k] for k in d.files}
    import jax

    jax.config.update("jax_enable_x64", True)
    import healpy as hp
    import jax.numpy as jnp
    import jax_cosmo as jc
    from datasets import load_dataset
    from jax_fli import compute_theory_cl_for_density
    from jax_fli.data import get_des_y3_nz_shear
    from jax_fli.io import Catalog

    out = {}
    for tag, run in RUNS.items():
        cat = Catalog.from_dataset(load_dataset(
            "parquet", data_files=str(EXP / "density_spectra" / f"spectra_{run}.parquet"),
            split="train"))
        spec, cosmo = cat.field[0], cat.cosmology[0]
        theory = (compute_theory_cl_for_density(cosmo, spec, jnp.arange(LMAX + 1))
                  * hp.pixwin(NSIDE, lmax=LMAX) ** 2).bin(nlb=NLB, lmin=2)
        binned = spec.bin(nlb=NLB, lmin=2)
        out["ell"] = np.asarray(binned.wavenumber)
        out[f"{tag}_cl"] = np.asarray(binned.array)
        out[f"{tag}_th"] = np.asarray(theory.array)
        out[f"{tag}_chi"] = np.asarray(spec.comoving_centers)
        out[f"{tag}_w"] = np.asarray(spec.density_width)
        print(tag, "chi", out[f"{tag}_chi"].round(0), "w", out[f"{tag}_w"].round(0))
    z = np.linspace(0.005, 2.995, 600)
    out["q_chi"] = np.asarray(jc.background.radial_comoving_distance(cosmo, jc.utils.z2a(z)))
    out["q"] = np.asarray(jc.probes.WeakLensing(get_des_y3_nz_shear()).kernel(
        cosmo, jnp.asarray(z), 1000.0))
    CACHE.mkdir(exist_ok=True)
    np.savez(npz, **out)
    return out


D = load()

slide_style()
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter

plt.rcParams["savefig.bbox"] = None          # fixed canvas: the two rows line up on the slide
NBAR = MESH**3 / BOX**3
NPIX = 12 * NSIDE**2
BINS = ["#f2b58a", "#d9772e", "#a3450f", "#5c1f04"]   # DES Y3 bins 1-4, light to dark
SHOWN = [("near", KW), ("middle", KW2), ("far", BLUE)]
ELL = D["ell"]
DL = ELL * (ELL + 1) / (2 * np.pi)

# the spectra panel shares one vertical range across both rows
lims = []
for tag in RUNS:
    chi = D[f"{tag}_chi"]
    for i in (0, int(np.argmin(abs(chi - CHI_MID))), len(chi) - 1):
        lims += [DL * D[f"{tag}_cl"][i], DL * D[f"{tag}_th"][i]]
lims = np.concatenate(lims)
lims = lims[np.isfinite(lims) & (lims > 0)]
DL_LIM = (lims.min() / 1.6, lims.max() * 1.6)
COUNT_LIM = (5e4, 2e9)

for tag, out in zip(RUNS, OUTS):
    chi, w = D[f"{tag}_chi"], D[f"{tag}_w"]
    order = np.argsort(chi)
    chi, w = chi[order], w[order]
    cl, th = D[f"{tag}_cl"][order], D[f"{tag}_th"][order]
    near, far = chi - w / 2, chi + w / 2
    count = NBAR * 4 / 3 * np.pi * (far**3 - near**3)
    print(f"{tag}: particles/shell {count.min():.2e} .. {count.max():.2e}, "
          f"{(count < NPIX).sum()} shells under one particle per pixel")

    fig = plt.figure(figsize=(10.4, 2.9))
    boxes = [(0.02, 0.24, 0.26, 0.64), (0.385, 0.24, 0.24, 0.64), (0.745, 0.24, 0.235, 0.64)]
    ax_k, ax_n, ax_c = (fig.add_axes(b) for b in boxes)

    # 1. lensing kernels over the shells
    for j in range(len(chi)):
        ax_k.axvspan(near[j], far[j], color="#d8dde6" if j % 2 else "#eef1f5", lw=0, zorder=0)
    q = D["q"] / D["q"].max()
    for i, c in enumerate(BINS):
        ax_k.plot(D["q_chi"], q[i], color=c, lw=2.0, zorder=2)
    ax_k.set_xlim(0, 2500)
    ax_k.set_ylim(0, 1.08)
    ax_k.set_yticks([])
    ax_k.set_xticks([0, 1000, 2000])
    ax_k.set_xlabel(r"$\chi$  [Mpc/$h$]", labelpad=2)
    ax_k.set_title("lensing kernels and shells", fontsize=13, color=INK, pad=6)

    # 2. particles per shell
    x = np.arange(1, len(chi) + 1)
    ax_n.bar(x, count, width=0.78, color=[KW if c < NPIX else "#9aa3b2" for c in count], lw=0)
    ax_n.axhline(NPIX, color=INK, ls="--", lw=1.4, label="1 particle per pixel")
    ax_n.legend(loc="upper left", fontsize=11, handlelength=1.8, borderaxespad=0.3)
    ax_n.set_yscale("log")
    ax_n.set_ylim(*COUNT_LIM)
    ax_n.set_xlim(0.3, len(chi) + 0.7)
    ax_n.set_xticks([1, 10, 20])
    ax_n.set_xlabel("shell (observer outwards)", labelpad=2)
    ax_n.set_title("particles per shell", fontsize=13, color=INK, pad=6)
    ax_n.yaxis.set_major_locator(LogLocator(numticks=10))
    ax_n.yaxis.set_minor_formatter(NullFormatter())

    # 3. near, middle and far shells against Limber
    shown = (0, int(np.argmin(abs(chi - CHI_MID))), len(chi) - 1)
    for i, (name, c) in zip(shown, SHOWN):
        ax_c.loglog(ELL, DL * th[i], color=c, ls="--", lw=1.5, alpha=0.9)
        ax_c.loglog(ELL, DL * cl[i], color=c, lw=2.2, label=rf"$\chi = {chi[i]:.0f}$")
    ax_c.set_xlim(ELL[0], LMAX)
    ax_c.set_ylim(*DL_LIM)
    ax_c.set_xlabel(r"$\ell$", labelpad=0)
    ax_c.set_title("shell spectra, Limber dashed", fontsize=13, color=INK, pad=6)
    ax_c.xaxis.set_major_locator(FixedLocator([10, 100, 1000]))
    ax_c.yaxis.set_major_locator(LogLocator(numticks=10))
    ax_c.yaxis.set_minor_formatter(NullFormatter())
    ax_c.legend(loc="lower right", fontsize=10.5, handlelength=1.2, borderaxespad=0.2,
                labelspacing=0.2)

    # the arrows from one panel to the next
    for xa in (0.315, 0.662):
        fig.text(xa, 0.56, r"$\Rightarrow$", fontsize=26, color=GREY, ha="center", va="center")
    fig.savefig(HERE / out)
    plt.close(fig)
    print(f"wrote {out}")
