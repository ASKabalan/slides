#!/usr/bin/env python3
# ENV: furax-cs
"""
The bias-variance trade-off of sky partitioning, as in Fig. 1 of the paper and the thesis
(chap5/illustrations/r_likelihood_high_low): the likelihood on r for few patches
(K_bd = 4000, K_Td = 10, K_bs = 50) and for many (10000, 3500, 10000), c1d1s1, r = 0,
the three Galactic regions combined.

Data: the raw runs of ASKabalan/furax-cs-results through furax-cs `r_analysis snap`
(_data.snapshot, same arguments as runners/paper/section_42.sh); labels in the thesis notation.

Output (this directory): r_likelihood_high_low.svg
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))
from _common import skip_if_built, slide_style
from _compsep import plot_r_likelihoods
from _data import snapshot

OUT = "r_likelihood_high_low.svg"
skip_if_built(HERE, OUT)

rows = snapshot(
    "lowhigh", ["BD4000_TD10_BS50", "BD10000_TD3500_BS10000"],
    ["KMEANS_C1D1S1/BD4000_TD10_BSXXX", "KMEANS_C1D1S1/BD10000_TD3500_BSXXX"],
    fetch=["KMEANS_C1D1S1/BD4000_TD10_BSXXX/kmeans_c1d1s1_BD4000_TD10_BS50_GAL*",
           "KMEANS_C1D1S1/BD10000_TD3500_BSXXX/kmeans_c1d1s1_BD10000_TD3500_BS10000_GAL*"])
by = {r["kw"]: r for r in rows}

slide_style(scale=1.3)
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5.4, 4.4))
plot_r_likelihoods(ax, [("Low patches", by["BD4000_TD10_BS50"], "#911EB4"),
                        ("High patches", by["BD10000_TD3500_BS10000"], "#4363D8")],
                   xlim=(-0.002, 0.005), ytop=1.32)
ax.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=13)
fig.savefig(HERE / OUT, transparent=True)
print(f"wrote {OUT}")
