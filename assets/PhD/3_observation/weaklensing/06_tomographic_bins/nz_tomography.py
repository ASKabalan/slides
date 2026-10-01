#!/usr/bin/env python3
# ENV: jax-fli
"""
Tomographic bins: what a survey actually measures, and where each bin is
sensitive to matter along the line of sight.

Top panel, the DES Y3 source redshift distributions. Bottom panel, the lensing
efficiency of each bin, which peaks roughly midway to the sources and is why a
bin constrains structure well in front of its own galaxies.

Uses jax_fli.data.nz.plot_nz with the DES Y3 n(z) shipped in jax_fli.io, so the
figure is the model's own input rather than a redrawing of it.

Output (this directory): nz_tomography.svg
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from _common import INK, skip_if_built, slide_style

OUT = "nz_tomography.svg"
skip_if_built(HERE, OUT)

slide_style(scale=1.15)
import matplotlib.pyplot as plt
from jax_fli.data.nz import plot_nz
from jax_fli.io import get_des_y3_nz_shear

nz = get_des_y3_nz_shear()
labels = [f"bin {i + 1}" for i in range(len(nz))]
fig, (ax_top, ax_bot) = plot_nz(nz, labels=labels, cmap="YlOrRd")

fig.set_size_inches(7.2, 4.8)
for ax in (ax_top, ax_bot):
    ax.set_facecolor("none")
    ax.tick_params(colors=INK)
    for sp in ax.spines.values():
        sp.set_color(INK)
fig.patch.set_alpha(0.0)
ax_bot.set_xlim(0, 1.8)

fig.savefig(HERE / OUT, transparent=True, bbox_inches="tight")
print(f"wrote {OUT}")
