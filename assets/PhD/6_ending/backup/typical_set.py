#!/usr/bin/env python3
# ENV: shared
"""
Where the posterior mass lies, for the backup slide on the maximum a posteriori: the slide version of
the thesis figure chap4/typical_set.pdf (These_wassim/figures/chap4/typical_set.py).

A 2D standard Gaussian posterior. Left, its density with three radial zones: the mode (r < 0.45),
the typical set (0.45 < r < 1.85) and the tail (1.85 < r < 3.5). Right, against the radius r, the
density p(r) (dashed) and the mass per unit radius p(r) 2 pi r, filled by zone with the mass each
zone holds, P(r < R) = 1 - exp(-R^2 / 2). No data: the figure is analytic.

Output (this directory): typical_set.svg
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from _common import GREY, INK, skip_if_built, slide_style

OUT = "typical_set.svg"
skip_if_built(HERE, OUT)

slide_style()
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

plt.rcParams["savefig.bbox"] = None
LIM, R1, R2, R3 = 3.5, 0.45, 1.85, 3.5
ZONES = [("mode", 0.0, R1, "#7d2818"), ("typical set", R1, R2, "#d85a30"), ("tail", R2, R3, "#f0997b")]
mass = lambda a, b: np.exp(-a**2 / 2) - np.exp(-b**2 / 2)

g = np.linspace(-LIM, LIM, 600)
GX, GY = np.meshgrid(g, g)
P = np.exp(-0.5 * (GX**2 + GY**2)) / (2 * np.pi)

fig = plt.figure(figsize=(10.4, 4.9))
ax = fig.add_axes([0.06, 0.13, 0.36, 0.8])
ax.contourf(GX, GY, P, levels=20, cmap="Blues", alpha=0.85)
for name, a, b, c in ZONES[::-1]:
    ax.add_patch(Circle((0, 0), b, fill=True, fc=c, alpha=0.15, ec="none", zorder=3))
    ax.add_patch(Circle((0, 0), b, fill=False, ec=c, lw=2.4, ls="--", zorder=4))
ax.plot(0, 0, "x", color=INK, ms=11, mew=2.2, zorder=10)
for name, y in (("mode", -0.75), ("typical set", 1.15), ("tail", 2.6)):
    ax.text(0, y, name, ha="center", va="center", fontsize=15, fontweight="bold", color=INK, zorder=11)
ax.set_xlim(-LIM, LIM)
ax.set_ylim(-LIM, LIM)
ax.set_aspect("equal")
ax.set_xlabel(r"$\theta_1$")
ax.set_ylabel(r"$\theta_2$")

ax = fig.add_axes([0.53, 0.13, 0.45, 0.8])
r = np.linspace(0, LIM, 400)
dens = np.exp(-0.5 * r**2) / (2 * np.pi)
m = dens * 2 * np.pi * r
ax.plot(r, dens, color=GREY, ls="--", lw=2.4, label=r"density $p(r)$")
ax.plot(r, m, color=INK, lw=2.2, label=r"mass $p(r)\,2\pi r$")
for name, a, b, c in ZONES:
    sel = (r >= a) & (r <= b)
    ax.fill_between(r[sel], 0, m[sel], color=c, alpha=0.9)
ax.annotate(f"mode\n{100 * mass(0, R1):.0f} %", xy=(0.3, 0.2), xytext=(0.05, 0.5), fontsize=14,
            fontweight="bold", color=INK, arrowprops=dict(arrowstyle="-", color=ZONES[0][3], lw=1.2))
ax.text((R1 + R2) / 2, 0.16, f"typical set\n{100 * mass(R1, R2):.0f} %", ha="center", va="center",
        fontsize=14, fontweight="bold", color="white")
ax.text((R2 + R3) / 2, 0.1, f"tail\n{100 * mass(R2, R3):.0f} %", ha="center", va="bottom",
        fontsize=14, fontweight="bold", color=INK)
ax.set_xlim(0, LIM)
ax.set_ylim(0, m.max() * 1.3)
ax.set_xlabel(r"radius $r = |\theta|$")
ax.legend(loc="upper right", fontsize=13)
fig.savefig(HERE / OUT)
print(f"wrote {OUT}: masses {mass(0, R1):.3f} {mass(R1, R2):.3f} {mass(R2, R3):.3f}")
