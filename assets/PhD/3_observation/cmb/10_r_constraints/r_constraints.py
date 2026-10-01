#!/usr/bin/env python3
# ENV: shared
"""
A decade of upper limits on the tensor-to-scalar ratio, and where the next
experiments aim.

Numbers as compiled by Tristram et al. (2022), arXiv:2112.07961, Table 3 and
Figure 6; the forecast sensitivities are the published targets of the Simons
Observatory (Ade et al. 2019, arXiv:1808.07445) and LiteBIRD (LiteBIRD
Collaboration 2023, arXiv:2202.02773).

The bar chart is the slide's argument: every factor of two costs a generation
of experiments, and the next factor of ten has to come from the analysis as
much as from the instrument.

Output (this directory): r_constraints.svg
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from _common import GREY, INK, KW, skip_if_built, slide_style

OUT = "r_constraints.svg"
skip_if_built(HERE, OUT)

# label, 95 % upper limit on r
LIMITS = [
    ("Planck PR1 (2013)", 0.11),
    ("PR1 + BK (2015)", 0.12),
    ("PR2 + BK (2016)", 0.09),
    ("BK15 (2018)", 0.07),
    ("PR3 + BK15 (2019)", 0.065),
    ("Planck PR4 (2021)", 0.056),
    ("PR4 + BK15 (2021)", 0.044),
    ("BK18 (2021)", 0.036),
    ("PR4 + BK18 (2022)", 0.032),
]
# name, target sigma(r)
TARGETS = [("Simons Observatory", 0.003), ("LiteBIRD", 0.001)]

slide_style(scale=1.2)
import matplotlib.pyplot as plt

labels = [n for n, _ in LIMITS]
values = np.array([v for _, v in LIMITS])
y = np.arange(len(values))[::-1]

fig, ax = plt.subplots(figsize=(7.4, 4.4))

colors = ["#9FB4CE"] * len(values)
colors[-1] = KW  # the tightest published limit
ax.barh(y, values, height=0.68, color=colors, edgecolor="none")
for yi, v, c in zip(y, values, colors):
    ax.text(v + 0.0025, yi, f"$r<{v:g}$", va="center", fontsize=12.5,
            color=KW if c == KW else INK,
            fontweight="bold" if c == KW else "normal")

# The two forecast sensitivities: one row each in the headroom above the bars,
# each label joined to its own line by a short horizontal leader.
for i, (name, sig) in enumerate(TARGETS):
    ax.axvline(sig, color=GREY, ls=":", lw=1.6)
    ty = len(values) + 1.25 - 0.8 * i
    ax.annotate(f"{name}, $\\sigma(r)\\sim{sig:g}$",
                xy=(sig, ty), xytext=(0.010, ty),
                color=GREY, fontsize=12, va="center",
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.9,
                                shrinkA=3, shrinkB=0))

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=12.5)
ax.set_xlim(0, 0.148)
ax.set_ylim(-0.75, len(values) + 1.9)
ax.set_xlabel("95 % upper limit on the tensor-to-scalar ratio $r$")
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.tick_params(top=False, right=False)

fig.savefig(HERE / OUT, transparent=True)
print(f"wrote {OUT}")
