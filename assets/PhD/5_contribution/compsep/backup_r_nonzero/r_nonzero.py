#!/usr/bin/env python3
# ENV: furax-cs
"""
Does choosing the partition by r-hat + sigma(r) suppress a true r? The likelihood on r for few
and many patches, on a sky without tensors (c1d0s0, r = 0) and with them (cr4d0s0, r = 3e-3),
as Fig. 14 of the paper (runners/paper/section_44.sh).

Data: raw runs of ASKabalan/furax-cs-results (TENSOR_TO_SCALAR_3_D0S0) through furax-cs
`r_analysis snap`, with the runner's FURAX_CS_ALLOW_FULLSKY settings; thesis notation.

Output (this directory): r_likelihood_r_nonzero.svg
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))
from _common import skip_if_built, slide_style
from _compsep import plot_r_likelihoods
from _data import snapshot

OUT = "r_likelihood_r_nonzero.svg"
skip_if_built(HERE, OUT)

T = "TENSOR_TO_SCALAR_3_D0S0"
r0 = {r["name"]: r for r in snapshot(
    "r0", ["kmeans_c1d0s0_BD1", "kmeans_c1d0s0_BD30000"], [T],
    fetch=[f"{T}/kmeans_c1d0s0_BD1_TD1_BS1_ALL", f"{T}/kmeans_c1d0s0_BD30000_TD1500_BS1500_ALL"],
    names=["C1 BD1", "C1 BD30000"], sky="c1d0s0", env={"FURAX_CS_ALLOW_FULLSKY": "0"})}
r3 = {r["name"]: r for r in snapshot(
    "r3e3", ["kmeans_cr4d0s0_BD1", "kmeans_cr4d0s0_BD30000"], [T],
    fetch=[f"{T}/kmeans_cr4d0s0_BD1_TD1_BS1_ALL", f"{T}/kmeans_cr4d0s0_BD30000_TD1500_BS1500_ALL"],
    names=["CR3 BD1", "CR3 BD30000"], sky="c1d0s0", env={"FURAX_CS_ALLOW_FULLSKY": "1"})}

slide_style(scale=1.25)
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7.6, 4.8))
plot_r_likelihoods(ax, [("Low patches ($r=0$)", r0["C1 BD1"], "#D4A017"),
                        ("High patches ($r=0$)", r0["C1 BD30000"], "#4363D8"),
                        (r"Low patches ($r=3\times10^{-3}$)", r3["CR3 BD1"], "#F58231"),
                        (r"High patches ($r=3\times10^{-3}$)", r3["CR3 BD30000"], "#911EB4")],
                   xlim=(-0.0005, 0.006), ytop=1.7)
for truth in (0, 3e-3):
    ax.axvline(truth, color="black", ls="--", lw=1.2, alpha=0.7)
ax.legend(loc="upper right", fontsize=11, frameon=True, framealpha=0.95)
fig.savefig(HERE / OUT, transparent=True)
print(f"wrote {OUT}")
