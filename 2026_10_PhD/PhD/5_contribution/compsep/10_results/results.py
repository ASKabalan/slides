#!/usr/bin/env python3
# ENV: furax-cs
"""
The result of the contribution: multi-resolution patches (LiteBIRD PTEP) against the spherical
K-means partition chosen by the selection criterion, c1d1s1, r = 0, all three regions combined.

r_likelihood.svg                   the two likelihoods on r, in the thesis notation (paper Fig. 11)
patches_{multires,kmeans}_{p}.png  their patch maps for beta_d, T_d and beta_s (transparent
                                   Mollweide, one random colour per patch, masked sky in grey)

Data: raw runs of ASKabalan/furax-cs-results (MULTIRES/ptep1_*, KMEANS_BEST_BEST) through
furax-cs `r_analysis snap`, as runners/paper/section_43.sh.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))
from _common import skip_if_built, slide_style
from _compsep import moll_png, plot_r_likelihoods, shuffled
from _data import snapshot

PARAMS = ("beta_dust", "temp_dust", "beta_pl")
OUTS = ["r_likelihood.svg"] + [f"patches_{m}_{p}.png" for m in ("multires", "kmeans") for p in PARAMS]
skip_if_built(HERE, *OUTS)

multires = snapshot("multires", ["ptep1_GAL020", "ptep1_GAL040", "ptep1_GAL060"], ["MULTIRES"],
                    fetch=["MULTIRES/ptep1_*"], combine=True, names=["MULTIRES"])[0]
kmeans = snapshot("kmeans_best", ["GAL020", "GAL040", "GAL060"], ["KMEANS_BEST_BEST"],
                  fetch=["KMEANS_BEST_BEST/*"], combine=True, names=["KMEANS"])[0]

slide_style(scale=1.3)
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5.4, 5.0))
plot_r_likelihoods(ax, [("Multi-res", multires, "#6A0DAD"), ("This work", kmeans, "#2E8B3E")],
                   xlim=(-0.001, 0.0045), ytop=1.3)
ax.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=12.5)
fig.savefig(HERE / "r_likelihood.svg", transparent=True)
plt.close(fig)
print("wrote r_likelihood.svg")

for name, row in (("multires", multires), ("kmeans", kmeans)):
    for p in PARAMS:
        lab = np.asarray(row[f"patches_{p}"], dtype=float)
        n = np.unique(lab[lab > -1e30]).size
        moll_png(shuffled(lab, seed=len(p)), HERE / f"patches_{name}_{p}.png", vmin=0, vmax=1,
                 figsize=(4.6, 2.5), dpi=180)
        print(f"wrote patches_{name}_{p}.png ({n} patches)")
