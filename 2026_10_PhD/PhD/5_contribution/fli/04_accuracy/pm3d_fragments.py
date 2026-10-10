#!/usr/bin/env python3
# ENV: jax-fli
"""
Particle-mesh accuracy, one fragment per mesh resolution.

The thesis figure (figures/chap6/pm3d_accuracy.py) split into a build: each
fragment adds the next mesh to the power-spectrum panel on the left and shows
that mesh's density slice on the right, so the audience sees the deficit
retreat to smaller scales and the cosmic web sharpen at the same click.

Configuration is the thesis one exactly: a 100 Mpc/h box, BullFrog with 40
steps in D, CIC without deconvolution, five realisations per mesh, the ladder
64^3 -> 96^3 -> 128^3 -> 192^3 -> 256^3, compared to halofit at a = 1. The
JAXHACK talk used fewer seeds on the fine meshes, which let sample variance
put 64^3 closest to zero at large scales; five everywhere removes that.

Simulations are cached per mesh in ../.cache/ (the shared fli cache), so re-rendering after a layout
change never reruns the solver. The whole ladder takes of order half an hour
on CPU.

Outputs (this directory): pm3d_frag_1.png .. pm3d_frag_5.png
"""

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import GREY, INK, KW, skip_if_built

RESOLUTIONS = (64, 96, 128, 192, 256)
OUTS = [f"pm3d_frag_{i}.png" for i in range(1, len(RESOLUTIONS) + 1)]
CACHE = HERE.parent / ".cache"          # the shared fli/.cache
skip_if_built(HERE, *OUTS)
CACHE.mkdir(exist_ok=True)

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax_cosmo as jc
import jax_fli as jfli
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (FixedLocator, LogLocator, NullFormatter,
                               ScalarFormatter)

BOX_SIZE = 100.0
N_STEPS = 40
N_SEEDS = 5
T0, T1 = 0.001, 1.0
cosmo = jc.Planck18()


def run_mesh(resolution: int) -> dict:
    npz = CACHE / f"pm3d_{resolution}.npz"
    if npz.exists():
        print(f"  {resolution}^3 cached", flush=True)
        return dict(np.load(npz))

    mesh_size, box = (resolution,) * 3, (BOX_SIZE,) * 3
    nbar = float(np.prod(mesh_size) / np.prod(box))
    solver = jfli.BullFrog(
        interp_kernel=jfli.NoInterp(painting=jfli.PaintingOptions(
            target="density", order="cic", deconvolution=False)),
        time_stepping="D", n_steps=N_STEPS, t0=T0, t1=T1,
        order="cic", deconvolution=False,
    )

    t0 = time.time()
    pk_stack, k, slab = [], None, None
    for seed in range(N_SEEDS):
        ic = jfli.gaussian_initial_conditions(jax.random.PRNGKey(seed), mesh_size,
                                              box, cosmo=cosmo)
        dx, p = jfli.lpt(cosmo, ic, ts=T0, order=1,
                         painting=jfli.PaintingOptions(target="particles"))
        density = jfli.nbody(cosmo, dx, p, ts=jnp.array([T1]), solver=solver)
        pk_obj = density.power(dk=2 * np.pi / BOX_SIZE, compensate_order="cic",
                               shotnoise=("cic", nbar))
        k = np.asarray(pk_obj.wavenumber)
        pk = np.asarray(pk_obj.spectra)
        pk_stack.append(pk[0] if pk.ndim == 2 else pk)
        if seed == 0:
            arr = np.asarray(density.array, dtype=np.float64).reshape(mesh_size)
            delta = arr / arr.mean() - 1.0
            # A slab a few Mpc/h thick reads better than one cell.
            depth = max(1, round(resolution * 4.0 / BOX_SIZE))
            mid = resolution // 2
            slab = delta[:, :, mid - depth // 2: mid - depth // 2 + depth].mean(axis=2)
        print(f"    {resolution}^3 seed {seed} done ({time.time() - t0:.0f} s)",
              flush=True)

    theory = np.asarray(jax.jit(jc.power.nonlinear_matter_power,
                                static_argnames=["nonlinear_fn"])(
        cosmo, jnp.asarray(k), a=T1, nonlinear_fn=jc.power.halofit))
    out = {"k": k, "pk_mean": np.mean(pk_stack, axis=0), "theory": theory,
           "slab": slab}
    np.savez(npz, **out)
    return out


data = {}
for r in RESOLUTIONS:
    print(f"mesh {r}^3", flush=True)
    data[r] = run_mesh(r)

ratio = {r: np.sqrt(np.clip(d["pk_mean"], 0, None) / d["theory"]) - 1.0
         for r, d in data.items()}
k_nyq = {r: np.pi * r / BOX_SIZE for r in RESOLUTIONS}
y_bottom = 1.08 * min(float(v.min()) for v in ratio.values())

plt.rcParams.update({
    "font.family": "DejaVu Sans", "svg.fonttype": "path",
    "font.size": 15, "axes.labelsize": 16, "xtick.labelsize": 13,
    "ytick.labelsize": 13, "legend.fontsize": 13,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "axes.edgecolor": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
})

*ladder, ref = RESOLUTIONS
colours = plt.cm.viridis(np.linspace(0.12, 0.88, len(ladder)))
styles = ("-", (0, (5, 1.6)), (0, (1, 1.1)), (0, (5, 1.3, 1, 1.3)))

# One colour scale for every fragment, from the finest map, so the sharpening
# is the only thing that changes between clicks.
vmax = max(4.0, float(np.percentile(data[ref]["slab"], 99.6)))
norm = mcolors.SymLogNorm(linthresh=1.0, vmin=-1.0, vmax=vmax, base=10.0)

for i, current in enumerate(RESOLUTIONS, start=1):
    shown = RESOLUTIONS[:i]
    fig = plt.figure(figsize=(9.6, 3.9), dpi=220)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 0.88], wspace=0.14,
                          left=0.085, right=0.99, top=0.97, bottom=0.19)
    ax, axm = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    for r in shown:
        if r == ref:
            ax.plot(data[r]["k"], ratio[r], color=KW, ls=(0, (7, 2.5)), lw=2.8,
                    label=rf"${r}^3$")
        else:
            j = ladder.index(r)
            ax.plot(data[r]["k"], ratio[r], color=colours[j], ls=styles[j],
                    lw=2.3, label=rf"${r}^3$")
    ax.axhline(0.0, color=GREY, lw=1.0)
    ax.axvline(k_nyq[current], color=GREY, ls="--", lw=1.3)
    ax.text(k_nyq[current] / 1.04, 0.06, r"$k_\mathrm{Nyq}$", color=GREY, fontsize=14,
            ha="right", va="top")
    ax.set_xscale("log")
    ax.set_xlim(0.3, k_nyq[ref] * 1.05)
    ax.set_ylim(y_bottom, 0.08)
    ax.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5]))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(LogLocator(subs="all"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"$k$  [$h\,\mathrm{Mpc}^{-1}$]")
    ax.set_ylabel(r"$\sqrt{P/P_\mathrm{halofit}}-1$")
    ax.legend(loc="lower left", title="mesh", frameon=False, handlelength=2.4,
              labelcolor=INK)

    axm.imshow(data[current]["slab"].T, origin="lower", cmap="magma", norm=norm,
               extent=(0, BOX_SIZE, 0, BOX_SIZE), interpolation="nearest")
    axm.set_xticks([])
    axm.set_yticks([])
    for sp in axm.spines.values():
        sp.set_color(INK)
        sp.set_linewidth(1.2)
    axm.text(0.04, 0.94, rf"${current}^3$", transform=axm.transAxes,
             color="white", fontsize=17, fontweight="bold", va="top")
    # box size, so the mesh reads as a physical cell size
    axm.text(0.04, 0.05, rf"$L = {BOX_SIZE:.0f}\,h^{{-1}}$Mpc" "\n"
             rf"cell $= {BOX_SIZE / current:.2g}\,h^{{-1}}$Mpc", transform=axm.transAxes,
             color="white", fontsize=13, va="bottom", linespacing=1.3)

    fig.patch.set_alpha(0.0)
    fig.savefig(HERE / OUTS[i - 1], transparent=True)
    plt.close(fig)
    print(f"wrote {OUTS[i - 1]}", flush=True)
