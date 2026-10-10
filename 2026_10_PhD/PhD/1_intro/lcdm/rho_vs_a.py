#!/usr/bin/env python3
# ENV: shared
"""
What dominates the Universe, and when.

A slide version of figures/chap2/rho_vs_a.py from the thesis: the three energy
densities against the scale factor, with the era each one rules shaded, so the
audience sees in one look that the component driving the expansion today is the
one we understand least. Log-log in the scale factor a, Planck 2018 values, each
curve labelled with its term of H^2(a) = H0^2 (Omega_r a^-4 + Omega_m a^-3 +
Omega_Lambda), which the slide shows below; time runs left to right (early -> late).

Output (this directory): rho_vs_a.svg
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import GREY, KW, skip_if_built, slide_style

OUT = "rho_vs_a.svg"
skip_if_built(HERE, OUT)
slide_style(scale=1.5)
import matplotlib.pyplot as plt

# Planck 2018, normalised to the present critical density.
OM_R, OM_M, OM_L = 9.2e-5, 0.315, 0.685
RAD, MAT = "#3B6FB6", "#2AA198"

a = np.logspace(-7, 0.9, 1200)
a_eq = OM_R / OM_M
a_lam = (OM_M / OM_L) ** (1 / 3)

fig, ax = plt.subplots(figsize=(8.6, 5.0))

ax.axvspan(1e-7, a_eq, color=RAD, alpha=0.055, lw=0)
ax.axvspan(a_eq, a_lam, color=MAT, alpha=0.065, lw=0)
ax.axvspan(a_lam, 8, color=KW, alpha=0.085, lw=0)

ax.loglog(a, OM_R * a**-4, color=RAD)
ax.loglog(a, OM_M * a**-3, color=MAT)
ax.loglog(a, OM_L * np.ones_like(a), color=KW, lw=3.2)

for x in (a_eq, a_lam):
    ax.axvline(x, color=GREY, ls=":", lw=1.1)

# One label per component, inside its own shaded era, clear of every curve:
# the band colour matches the curve, so nothing needs a legend. Each curve is
# labelled with its scaling, the terms of H^2(a) on the slide; the dotted era
# boundaries (equality at a = Omega_r/Omega_m, Lambda at (Omega_m/Omega_L)^(1/3))
# are left unlabelled.
ax.text(3e-6, 3e23, "radiation", color=RAD, fontweight="bold", fontsize=15, va="center")
ax.text(3e-6, 1e20, r"$\Omega_r\,a^{-4}$", color=RAD, fontsize=15, va="center")
ax.text(1.5e-3, 3e23, "matter", color=MAT, fontweight="bold", fontsize=15, va="center")
ax.text(1.5e-3, 1e20, r"$\Omega_m\,a^{-3}$", color=MAT, fontsize=15, va="center")
ax.text(2.4, 3e23, r"$\Lambda$", color=KW, fontweight="bold", fontsize=20,
        ha="center", va="center")
ax.text(2.4, 1e20, r"$\Omega_\Lambda$", color=KW, fontsize=15, ha="center", va="center")

ax.set_xlabel(r"scale factor $a$")
ax.set_ylabel(r"$\rho_i\,/\,\rho_{\mathrm{crit}}$")
ax.set_xlim(1e-7, 8)
ax.set_ylim(1e-4, 1e26)
ax.set_xticks([1e-6, 1e-4, 1e-2, 1])
ax.set_yticks([1e0, 1e8, 1e16, 1e24])
ax.minorticks_off()
ax.tick_params(top=False, which="both")

# time runs left to right: early -> late, under the axis label
ax.annotate("", xy=(0.70, -0.30), xytext=(0.30, -0.30), xycoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.8, mutation_scale=16))
ax.text(0.27, -0.30, "early", transform=ax.transAxes, color=GREY, fontsize=15,
        ha="right", va="center")
ax.text(0.73, -0.30, "late", transform=ax.transAxes, color=GREY, fontsize=15,
        ha="left", va="center")

fig.savefig(HERE / OUT, transparent=True)
print(f"wrote {OUT}")
