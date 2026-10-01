#!/usr/bin/env python3
# ENV: jax-fli
"""
What linear theory misses in the lensing signal a survey measures.

The Limber convergence power spectrum for the DES Y3 source bins, computed once
with the linear matter power spectrum and once with halofit. Above a few
hundred in ell the two separate by factors that grow with multipole: the
non-linear collapse of structure contains most of the lensing power on the
scales the surveys resolve, and no amount of linear theory recovers it. That is
the case for a gravity solver that follows the collapse, which is where the
talk goes next.

Uses jax_fli.compute_theory_cl with the n(z) shipped in jax_fli.io, so the
curves are the model's own theory prediction.

Output (this directory): limber_lin_vs_halofit.svg
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from _common import GREY, INK, skip_if_built, slide_style

OUT = "limber_lin_vs_halofit.svg"
skip_if_built(HERE, OUT)

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
from jax_fli import compute_theory_cl
from jax_fli.io import get_des_y3_nz_shear

cosmo = jc.Planck18()
ell = jnp.geomspace(10, 3000, 90)
nz = get_des_y3_nz_shear()

lin = np.asarray(compute_theory_cl(cosmo, ell, z_source=nz,
                                   nonlinear_fn="linear").array)
nl = np.asarray(compute_theory_cl(cosmo, ell, z_source=nz).array)
ell = np.asarray(ell)
d = ell * (ell + 1) / (2 * np.pi)

slide_style(scale=1.2)
import matplotlib.pyplot as plt

fig, (ax, axr) = plt.subplots(2, 1, figsize=(6.8, 4.9), sharex=True,
                              gridspec_kw={"height_ratios": [2.3, 1.0],
                                           "hspace": 0.06})
cmap = plt.get_cmap("YlOrRd")
colours = [cmap(x) for x in np.linspace(0.40, 0.95, lin.shape[0])]
for i, c in enumerate(colours):
    ax.loglog(ell, d * nl[i], color=c, lw=2.5)
    ax.loglog(ell, d * lin[i], color=c, lw=2.0, ls="--")
    axr.semilogx(ell, nl[i] / lin[i], color=c, lw=2.3,
                 label=f"bin {i + 1}")

ax.plot([], [], color=INK, lw=2.5, label="halofit")
ax.plot([], [], color=INK, lw=2.0, ls="--", label="linear")
ax.legend(loc="upper left", labelcolor=INK)
ax.set_ylabel(r"$\ell(\ell+1)\,C_\ell^{\kappa\kappa}/2\pi$")

axr.axhline(1.0, color=GREY, lw=1.0)
axr.set_ylabel("ratio")
axr.set_xlabel(r"multipole $\ell$")
axr.set_xlim(10, 3000)
axr.legend(loc="upper left", ncol=4, fontsize=11, labelcolor=INK,
           handlelength=1.2, columnspacing=0.9)

r = nl / lin

fig.savefig(HERE / OUT, transparent=True)
print(f"wrote {OUT}  (largest ratio {r.max():.2f} at ell ~ {ell[np.argmax(r.max(0))]:.0f})")
