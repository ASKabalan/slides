#!/usr/bin/env python3
# ENV: shared
"""
Hubble's 1929 velocity-distance relation (Hubble 1929, PNAS 15, 168, Fig. 1),
redrawn point for point for the "Modern observational cosmology" slide.

Everything is read off the published figure, not recomputed:
  - filled discs: the 24 nebulae taken individually. Distances are those of
    Table 1 (the digitised x positions land on them to 0.005 Mpc); velocities
    are digitised from the figure, which plots them corrected for the solar
    motion (so they differ from the raw Table 1 column).
  - open circles: the nebulae combined into groups, digitised.
  - cross: the mean of the 22 nebulae without individual distances, digitised.
  - solid and dashed lines: slopes measured on the figure, 513 and 465 km/s/Mpc,
    Hubble's two published solutions.
Axes, grid (0, 1, 2 x 10^6 pc; 0, 500, 1000 km/s) and labels follow the
original; the palette (indigo / gold, Lato, grey hairlines) follows the
cosmology_timeline and flrw_curv figures that share the slide.

Output (this directory):
  hubble_diagram.svg
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from _common import skip_if_built, slide_style

OUT = "hubble_diagram.svg"
skip_if_built(HERE, OUT)

import matplotlib.pyplot as plt

INDIGO = "#4a3fb0"
GOLD = "#b8791d"
TEXT = "#2b2b33"
MUTED = "#8a8a96"
HAIR = "#d6d3de"

# Individual nebulae: distance [Mpc] (Table 1), velocity [km/s] (Fig. 1)
D_IND = np.array([0.032, 0.034, 0.214, 0.263, 0.275, 0.275, 0.45, 0.5, 0.5, 0.63,
                  0.8, 0.9, 0.9, 0.9, 0.9, 1.0, 1.1, 1.1, 1.4, 1.7, 2.0, 2.0, 2.0, 2.0])
V_IND = np.array([-18, 29, 38, 2, -38, -82, 398, 441, 411, 318,
                  374, 594, 427, 211, 95, 827, 724, 562, 610, 1046, 1115, 842, 806, 525])

# Groups (open circles) and the 22-nebula mean (cross), digitised from Fig. 1
D_GRP = np.array([0.036, 0.272, 0.613, 0.674, 0.901, 1.052, 1.398, 1.630])
V_GRP = np.array([57, -124, 384, 198, 513, 737, 758, 726])
CROSS = (1.40, 740)

K_SOLID, K_DASHED = 513.0, 465.0

slide_style(1.25)
plt.rcParams.update({"font.family": "Lato", "xtick.top": False, "ytick.right": False,
                     "xtick.major.size": 0, "ytick.major.size": 0})

fig, ax = plt.subplots(figsize=(6.6, 4.6))
d = np.linspace(0, 2.2, 50)

for x in (0, 1, 2):
    ax.axvline(x, color=HAIR, lw=1.1, zorder=0)
for y in (0, 500, 1000):
    ax.axhline(y, color=HAIR, lw=1.1, zorder=0)

ax.plot(d, K_DASHED * d, color=GOLD, lw=2.0, ls=(0, (5, 3)), zorder=1)
ax.plot(d, K_SOLID * d, color=INDIGO, lw=2.4, zorder=2)
ax.scatter(D_GRP, V_GRP, s=62, facecolor="white", edgecolor=GOLD, linewidth=2.0, zorder=3)
ax.scatter(D_IND, V_IND, s=46, color=INDIGO, edgecolor="white", linewidth=1.0, zorder=4)
ax.scatter(*CROSS, s=150, marker="+", color=TEXT, linewidth=2.0, zorder=5)

ax.set_xlim(-0.33, 2.33)
ax.set_ylim(-340, 1330)
ax.set_xticks([0, 1, 2], ["0", r"$10^6$ parsecs", r"$2\times10^6$ parsecs"])
ax.set_yticks([0, 500, 1000], ["0", "500 km", "+1000 km"])
ax.set_xlabel("distance", color=TEXT, labelpad=6)
ax.set_ylabel("velocity", color=TEXT, labelpad=6)
for side in ax.spines.values():
    side.set_color(MUTED)
    side.set_linewidth(1.0)
ax.tick_params(labelcolor=TEXT, pad=6)

fig.text(0.59, -0.03, "Velocity-Distance Relation among Extra-Galactic Nebulae.",
         ha="center", va="top", color=MUTED, style="italic", fontsize=13)

fig.savefig(HERE / OUT)
print(f"    wrote {OUT}")
