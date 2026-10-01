#!/usr/bin/env python3
# ENV: jax-fli
"""
What the number of shells does to the density shells, for the slide on choosing it.

jax-fli experiment 05b (ASKabalan/jax-fli-experiments, 05-spacing-n-stepping/05b-3bins): 2560^3
mesh in a 5000 Mpc/h box, observer at the centre, BullFrog with 50 steps, shells uniform in the
scale factor, no drift on the lightcone, nside 2048, run at 10, 20 and 30 shells from one seed.
One image per shell count:
  top     the lightcone as a wedge from the observer, each shell a band at its true comoving edges,
          the near, middle and far shells filled in the colours of their spectra;
  bottom  the angular power spectrum of those three shells (the first, the middle one, the last),
          bands of 32 multipoles, against the Limber prediction for the shell's number counts times
          the squared pixel window (dashed), with the particle count of the near shell,
          nbar * 4/3 pi r^3 for nbar = 2560^3 / 5000^3.

Outputs (this directory), same canvas and axes boxes: nshells_10.svg, nshells_20.svg, nshells_30.svg
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from _common import BLUE, GREY, INK, KW, KW2, skip_if_built, slide_style

COUNTS = (10, 20, 30)
OUTS = [f"nshells_{n}.svg" for n in COUNTS]
skip_if_built(HERE, *OUTS)

CACHE = HERE.parent / ".cache"
REPO = "ASKabalan/jax-fli-experiments"
DENSITY = "05-spacing-n-stepping/05b-3bins/density_spectra/spectra_exp5b_nodrift_{n}.parquet"
BOX, MESH, LMAX, NSIDE, NLB, R_MAX = 5000.0, 2560, 1500, 2048, 32, 2500.0


def load():
    npz = CACHE / "nshells_05b_nodrift.npz"
    if npz.exists():
        d = np.load(npz)
        return {k: d[k] for k in d.files}
    import jax

    jax.config.update("jax_enable_x64", True)
    import healpy as hp
    import jax.numpy as jnp
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from jax_fli import compute_theory_cl_for_density
    from jax_fli.io import Catalog

    root = snapshot_download(REPO, repo_type="dataset",
                             allow_patterns=[DENSITY.format(n=n) for n in COUNTS])
    out = {}
    for n in COUNTS:
        cat = Catalog.from_dataset(load_dataset(
            "parquet", data_files=f"{root}/{DENSITY.format(n=n)}", split="train"))
        spec = cat.field[0]
        theory = (compute_theory_cl_for_density(cat.cosmology[0], spec, jnp.arange(LMAX + 1))
                  * hp.pixwin(NSIDE, lmax=LMAX) ** 2).bin(nlb=NLB, lmin=2)
        binned = spec.bin(nlb=NLB, lmin=2)
        o = np.argsort(np.asarray(spec.comoving_centers))
        out["ell"] = np.asarray(binned.wavenumber)
        out[f"cl_{n}"] = np.asarray(binned.array)[o]
        out[f"th_{n}"] = np.asarray(theory.array)[o]
        out[f"edges_{n}"] = np.r_[0.0, np.cumsum(np.asarray(spec.density_width)[o])]
    CACHE.mkdir(exist_ok=True)
    np.savez(npz, **out)
    return out


D = load()
ELL = D["ell"]
DL = ELL * (ELL + 1) / (2 * np.pi)
NBAR = MESH**3 / BOX**3


def shown(n):
    return (0, n // 2, n - 1)


lims = np.concatenate([DL * D[f"{k}_{n}"][i] for n in COUNTS for i in shown(n) for k in ("cl", "th")])
lims = lims[np.isfinite(lims) & (lims > 0)]
DL_LIM = (lims.min() / 1.5, lims.max() * 1.5)
for n in COUNTS:
    e = D[f"edges_{n}"]
    i1000 = np.argmin(abs(ELL - 1000))
    print(f"{n} shells: near shell [0, {e[1]:.0f}] Mpc/h, {NBAR * 4 / 3 * np.pi * e[1] ** 3:.2e} particles,"
          f" C_l/Limber at l = {ELL[i1000]:.0f}: {D[f'cl_{n}'][0, i1000] / D[f'th_{n}'][0, i1000]:.2f}")

slide_style()
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter

plt.rcParams["savefig.bbox"] = None
COLOURS = (KW, KW2, BLUE)
HALF = 11                                   # half-opening of the wedge [deg]

for n, out in zip(COUNTS, OUTS):
    fig = plt.figure(figsize=(3.5, 4.0))
    ax_w = fig.add_axes([0.03, 0.645, 0.94, 0.29])
    ax_c = fig.add_axes([0.235, 0.12, 0.7, 0.45])

    # the lightcone, shell by shell, from the observer at the left
    edges = D[f"edges_{n}"]
    fill = {i: c for i, c in zip(shown(n), COLOURS)}
    for j in range(n):
        ax_w.add_patch(Wedge((0, 0), edges[j + 1], -HALF, HALF, width=edges[j + 1] - edges[j],
                             fc=fill.get(j, "#d8dde6" if j % 2 else "#eef1f5"), ec="#9aa3b2", lw=0.5))
    for i, c, name in zip(shown(n), COLOURS, ("near", "middle", "far")):
        r = 0.5 * (edges[i] + edges[i + 1])
        ax_w.text(r * np.cos(np.radians(HALF)) + (60 if i == 0 else 0),
                  r * np.sin(np.radians(HALF)) + 45, name, color=c, fontsize=10.5,
                  ha="center", va="bottom")
    ax_w.plot(0, 0, "o", color=INK, ms=4)
    ax_w.set_xlim(-60, R_MAX + 40)
    ax_w.set_ylim(-R_MAX * np.sin(np.radians(HALF)) - 30, R_MAX * np.sin(np.radians(HALF)) + 200)
    ax_w.set_aspect("equal")
    ax_w.axis("off")
    ax_w.set_title(f"{n} shells", fontsize=15, color=INK, pad=4)

    # near, middle and far shells against Limber
    for i, c in zip(shown(n), COLOURS):
        ax_c.loglog(ELL, DL * D[f"th_{n}"][i], color=c, ls="--", lw=1.3, alpha=0.9)
        ax_c.loglog(ELL, DL * D[f"cl_{n}"][i], color=c, lw=2.0)
    ax_c.set_xlim(ELL[0], LMAX)
    ax_c.set_ylim(*DL_LIM)
    ax_c.xaxis.set_major_locator(FixedLocator([10, 100, 1000]))
    ax_c.xaxis.set_minor_formatter(NullFormatter())
    ax_c.yaxis.set_major_locator(LogLocator(numticks=10))
    ax_c.yaxis.set_minor_formatter(NullFormatter())
    ax_c.set_xlabel(r"$\ell$", labelpad=0)
    ax_c.set_title(r"$\ell(\ell+1)\,C_\ell / 2\pi$", fontsize=13, color=INK, pad=5)
    if n == COUNTS[0]:
        ax_c.plot([], [], color=GREY, ls="--", lw=1.3, label="Limber")
        ax_c.legend(loc="lower right", fontsize=10.5, handlelength=1.6, borderaxespad=0.2)
    count = NBAR * 4 / 3 * np.pi * edges[1] ** 3
    mant, expo = f"{count:.1e}".split("e")
    ax_c.text(0.04, 0.96, rf"near shell: ${mant}\times10^{{{int(expo)}}}$" + "\nparticles",
              transform=ax_c.transAxes, ha="left", va="top", fontsize=10.5, color=KW)
    fig.savefig(HERE / out)
    plt.close(fig)
    print(f"wrote {out}")
