#!/usr/bin/env python3
# ENV: jax-fli
"""
Equal-volume against uniform scale-factor shell spacing, for the backup slide "Uniform scale factor
against equal volume" (data and the shell and particle panels in _spacing.py, beside this file).

Left, one row per spacing (equal volume on top, uniform in a below):
  1. the four DES Y3 lensing kernels q(chi) over the 30 shells, drawn as alternating bands;
  2. the particles each shell receives, nbar * 4/3 pi (r_far^3 - r_near^3), nbar = 2560^3 / 5000^3,
     against one particle per HEALPix pixel at nside 2048.
Right, across both rows: the innermost shell of each spacing, C_ell / Limber - 1 in bandpowers of 4
multipoles (Limber for its number counts times the squared pixel window), with the uniform-a
shot-noise level 4 pi / N_particles in the same units.

Output (this directory): spacing_compare.svg
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import GREY, INK, KW, KW2, skip_if_built, slide_style

OUT = "spacing_compare.svg"
skip_if_built(HERE, OUT)

from _spacing import LMAX, draw_counts, draw_kernels, load, shells

NLB = 4   # bandpowers fine enough for the low multipoles to scatter around zero
ROWS = (("equal_vol", "equal volume", KW), ("a", "uniform in $a$", KW2))

D = load(NLB)

slide_style()
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

plt.rcParams["savefig.bbox"] = None
fig = plt.figure(figsize=(10.4, 6.0))

for i, (tag, label, colour) in enumerate(ROWS):
    y0 = 0.565 - 0.465 * i
    near, far, count, _, _ = shells(D, tag)
    ax_k = fig.add_axes((0.07, y0, 0.25, 0.355))
    ax_n = fig.add_axes((0.40, y0, 0.22, 0.355))
    draw_kernels(ax_k, D, near, far)
    draw_counts(ax_n, count)
    if i:                                  # the panel titles once, over the top row
        ax_k.set_title("")
        ax_n.set_title("")
    fig.text(0.018, y0 + 0.1775, label, rotation=90, ha="center", va="center", fontsize=14,
             color=colour, fontweight="bold")

ax = fig.add_axes((0.715, 0.10, 0.275, 0.82))
ELL = D["ell"]
ax.axhline(0.0, color=GREY, lw=1.0)
for tag, label, colour in ROWS[::-1]:
    near, far, count, cl, th = shells(D, tag)
    r = cl[0] / th[0] - 1
    ax.plot(ELL, r, color=colour, lw=2.2, label=rf"{label}, $\chi < {far[0]:.0f}$ Mpc/$h$")
    if tag == "a":
        ax.plot(ELL, 4 * np.pi / count[0] / th[0], color=colour, lw=1.8, ls=":",
                label="its shot-noise level")
    print(f"{tag}: innermost shell C_ell/Limber - 1 from {r[0]:+.2f} to {r[-1]:+.2f}")
ax.set_xscale("log")
ax.set_yscale("symlog", linthresh=1.0, linscale=1.6)
ax.set_xlim(2, LMAX)
ax.set_ylim(-1, 80)
ax.set_yticks([-1, -0.5, 0, 0.5, 1, 10], ["−1", "−0.5", "0", "0.5", "1", "10"])
ax.xaxis.set_major_locator(FixedLocator([3, 10, 30, 100, 300, 1000]))
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel(r"$\ell$", labelpad=0)
ax.set_ylabel(r"$C_\ell / C_\ell^{\mathrm{Limber}} - 1$", labelpad=0)
ax.set_title("innermost shell against Limber", fontsize=13, color=INK, pad=6)
ax.legend(loc="upper left", fontsize=9.6, handlelength=1.5, borderaxespad=0.25, labelspacing=0.2)
fig.savefig(HERE / OUT)
plt.close(fig)
print(f"wrote {OUT}")
