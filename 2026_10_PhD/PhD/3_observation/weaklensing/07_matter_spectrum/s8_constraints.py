#!/usr/bin/env python3
# ENV: shared
"""
Cosmic-shear measurements of S_8 = sigma_8 sqrt(Omega_m / 0.3), against the CMB.

Published 68 % intervals: Planck 2018 TT,TE,EE+lowE+lensing (Planck Collaboration 2020,
arXiv:1807.06209); KiDS-1000 COSEBIs (Asgari et al. 2021, arXiv:2007.15633); DES Y3 cosmic
shear, real space (Amon et al. 2022, Secco et al. 2022, arXiv:2105.13543); HSC Y3 harmonic
space (Dalal et al. 2023, arXiv:2304.00701); DES Y3 + KiDS-1000 (DES & KiDS Collaboration 2023,
arXiv:2305.17173); KiDS-Legacy (Wright et al. 2025, A&A 703, A158, arXiv:2503.19441); DES Y6
cosmic shear, NLA fiducial (DES Collaboration 2026, arXiv:2602.10065).

The slide's argument: lensing has sat below the CMB (the S_8 tension), and the newest data
close part of the gap.

Output (this directory): s8_constraints.svg
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import GREY, INK, KW, KW2, skip_if_built, slide_style

OUT = "s8_constraints.svg"
skip_if_built(HERE, OUT)

# label, S_8, +err, -err (68 %); sorted by S_8 below (lowest at the top), Planck last
LENSING = [
    ("KiDS-1000 (2021)", 0.759, 0.024, 0.021),
    ("DES Y3 (2022)", 0.759, 0.025, 0.023),
    ("HSC Y3 (2023)", 0.776, 0.032, 0.033),
    ("DES Y3 + KiDS-1000 (2023)", 0.790, 0.018, 0.014),
    ("KiDS-Legacy (2025)", 0.815, 0.016, 0.021),
    ("DES Y6 (2026)", 0.798, 0.014, 0.015),
]
LENSING = sorted(LENSING, key=lambda r: r[1])      # stable: equal values keep the year order
PLANCK = ("Planck 2018 (TT,TE,EE+lowE+lensing)", 0.832, 0.013, 0.013)

slide_style(scale=1.2)
import matplotlib.pyplot as plt

rows = LENSING + [PLANCK]
y = np.arange(len(rows))[::-1]

fig, ax = plt.subplots(figsize=(9.6, 4.5))

_, s, up, lo = PLANCK
ax.axvspan(s - lo, s + up, color=KW2, alpha=0.12, lw=0)
for yi, (name, s, up, lo) in zip(y, rows):
    col = KW2 if name == PLANCK[0] else KW
    ax.errorbar(s, yi, xerr=[[lo], [up]], fmt="o", color=col, ms=8, lw=2.4, capsize=4)

ax.axhline(y[-1] + 0.5, color=GREY, lw=0.8, ls=":")
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], fontsize=12.5)
for lab, (name, *_) in zip(ax.get_yticklabels(), rows):
    lab.set_color(KW2 if name == PLANCK[0] else INK)
ax.set_xlim(0.70, 0.87)
ax.set_ylim(-0.7, len(rows) - 0.3)
ax.set_xlabel(r"$S_8 = \sigma_8\,\sqrt{\Omega_\mathrm{m}/0.3}$")
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.tick_params(top=False, right=False)

fig.tight_layout()
fig.savefig(HERE / OUT, transparent=True)
print(f"wrote {OUT}")
