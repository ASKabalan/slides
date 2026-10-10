#!/usr/bin/env python3
# ENV: shared
"""
The simulation box cut into pencils, one per GPU, for the slide "Distributing the simulation": the
picture a reader needs before the word "distributed" means anything. Four pencils, drawn as solid
prisms in an oblique projection and pulled slightly apart, one deck colour each, labelled GPU 1 to 4
on their front faces. The transpositions, the distributed FFT and the halo exchange themselves are on
backup slides.

Output (this directory): pencils.svg
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import BLUE, KW, KW2, TEAL, skip_if_built, slide_style

OUT = "pencils.svg"
skip_if_built(HERE, OUT)

slide_style(scale=1.2)
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Polygon

S, GAP = 1.0, 0.42                  # side of a pencil's square end, gap between pencils
DEPTH = np.array([1.75, 1.15])      # the long axis of the pencils, in the oblique projection
# (column, row) of each pencil's front face, drawn so that nearer faces cover farther ones
PENCILS = [((0, 0), "GPU 3", TEAL), ((1, 0), "GPU 4", BLUE),
           ((0, 1), "GPU 1", KW), ((1, 1), "GPU 2", KW2)]


def shade(c, f):
    c = np.array(to_rgb(c))
    return tuple(c + (1 - c) * f) if f > 0 else tuple(c * (1 + f))


def prism(ax, x0, y0, c):
    front = [(x0, y0), (x0 + S, y0), (x0 + S, y0 + S), (x0, y0 + S)]
    top = [(x0, y0 + S), (x0 + S, y0 + S), tuple(np.r_[x0 + S, y0 + S] + DEPTH),
           tuple(np.r_[x0, y0 + S] + DEPTH)]
    side = [(x0 + S, y0), tuple(np.r_[x0 + S, y0] + DEPTH), tuple(np.r_[x0 + S, y0 + S] + DEPTH),
            (x0 + S, y0 + S)]
    for pts, f in ((side, -0.28), (top, 0.35), (front, 0.0)):
        ax.add_patch(Polygon(pts, closed=True, fc=shade(c, f), ec="white", lw=1.6, joinstyle="round"))


fig, ax = plt.subplots(figsize=(6.2, 4.6))
ax.set_aspect("equal")
ax.axis("off")
for (col, row), lab, c in PENCILS:
    x0, y0 = col * (S + GAP), row * (S + GAP)
    prism(ax, x0, y0, c)
    ax.text(x0 + S / 2, y0 + S / 2, lab, color="white", fontsize=17, fontweight="bold", ha="center",
            va="center")

ax.set_xlim(-0.15, 2 * S + GAP + DEPTH[0] + 0.15)
ax.set_ylim(-0.15, 2 * S + GAP + DEPTH[1] + 0.15)
fig.savefig(HERE / OUT, transparent=True, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print(f"wrote {OUT}")
