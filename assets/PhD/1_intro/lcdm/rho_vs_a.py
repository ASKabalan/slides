#!/usr/bin/env python3
# ENV: shared
"""
What dominates the Universe, and when.

A slide version of figures/chap2/rho_vs_a.py from the thesis: the three energy
densities against the scale factor, with the era each one rules shaded, so the
audience sees in one look that the component driving the expansion today is the
one we understand least.

Output (this directory): rho_vs_a.svg
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
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

fig, ax = plt.subplots(figsize=(6.4, 4.0))

ax.axvspan(1e-7, a_eq, color=RAD, alpha=0.055, lw=0)
ax.axvspan(a_eq, a_lam, color=MAT, alpha=0.065, lw=0)
ax.axvspan(a_lam, 8, color=KW, alpha=0.085, lw=0)

ax.loglog(a, OM_R * a**-4, color=RAD)
ax.loglog(a, OM_M * a**-3, color=MAT)
ax.loglog(a, OM_L * np.ones_like(a), color=KW, lw=3.2)

for x in (a_eq, a_lam):
    ax.axvline(x, color=GREY, ls=":", lw=1.1)

# One label per component, inside its own shaded era, clear of every curve:
# the band colour matches the curve, so nothing needs a legend.
ax.text(3e-6, 3e23, "radiation", color=RAD, fontweight="bold", fontsize=15, va="center")
ax.text(3e-6, 1e20, r"$\propto a^{-4}$", color=RAD, fontsize=14, va="center")
ax.text(1.5e-3, 3e23, "matter", color=MAT, fontweight="bold", fontsize=15, va="center")
ax.text(1.5e-3, 1e20, r"$\propto a^{-3}$", color=MAT, fontsize=14, va="center")
ax.text(2.4, 3e23, r"$\Lambda$", color=KW, fontweight="bold", fontsize=20,
        ha="center", va="center")
ax.text(2.4, 1e20, r"$\propto a^{0}$", color=KW, fontsize=14, ha="center", va="center")

# Epoch markers below the Λ line, to the right of their dotted lines (no curve there).
ax.text(a_eq * 1.35, 4e-3, r"$a_\mathrm{eq}$", color=GREY, fontsize=15, va="center")
ax.text(a_lam * 1.25, 4e-3, r"$a_\Lambda$", color=GREY, fontsize=15, va="center")

ax.set_xlabel(r"scale factor $a$")
ax.set_ylabel(r"$\rho_i\,/\,\rho_{\mathrm{crit},0}$")
ax.set_xlim(1e-7, 8)
ax.set_ylim(1e-4, 1e26)
ax.set_xticks([1e-6, 1e-4, 1e-2, 1])
ax.set_yticks([1e0, 1e8, 1e16, 1e24])
ax.minorticks_off()

sec = ax.twiny()
sec.set_xscale("log")
sec.set_xlim(ax.get_xlim())
sec.set_xticks([1 / (1 + z) for z in (1e6, 1e4, 1e2, 1)])
sec.set_xticklabels([r"$10^{6}$", r"$10^{4}$", r"$10^{2}$", r"$1$"])
sec.minorticks_off()
sec.set_xlabel(r"redshift $z$")

fig.savefig(HERE / OUT, transparent=True)
print(f"wrote {OUT}")
