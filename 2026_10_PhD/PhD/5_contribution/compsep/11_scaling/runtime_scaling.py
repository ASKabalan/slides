#!/usr/bin/env python3
# ENV: furax-cs
"""
Time to fit the spectral likelihood against the number of sky patches: AdaTopK on one GPU
against the truncated Newton solver (scipy TNC) that FGBuster uses.

Timings: raw/PROFILING/runs/CLUSTERS_FURAX.csv and CLUSTERS_FGBUSTER.csv of the HuggingFace
dataset ASKabalan/furax-cs-results (jax-hpc-profiler layout, headerless; `x` is the patch count,
`min_time` the best of the repeats in ms). FGBuster above 4000 patches comes from
CLUSTERS_FGBUSTER_recovered.csv in this folder (see its header).

Output (this directory): runtime_scaling.svg
"""

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import GREY, INK, KW, skip_if_built, slide_style

OUT = "runtime_scaling.svg"
skip_if_built(HERE, OUT)

from huggingface_hub import hf_hub_download

CACHE = HERE.parent / ".cache"
COLUMNS = ["function", "precision", "x", "y", "z", "px", "py", "backend", "nodes", "jit_time",
           "min_time", "max_time", "mean_time", "std_div", "last_time", "generated_code",
           "argument_size", "output_size", "temp_size"]


def runs(csv, fn):
    p = hf_hub_download("ASKabalan/furax-cs-results", f"raw/PROFILING/runs/{csv}.csv",
                        repo_type="dataset", cache_dir=str(CACHE / "hf"))
    df = pd.read_csv(p, header=None, names=COLUMNS)
    return df[df["function"] == fn][["x", "min_time"]]


furax = runs("CLUSTERS_FURAX", "Furax-ADABK0 n=64").sort_values("x")
fgb = pd.concat([runs("CLUSTERS_FGBUSTER", "FGBuster-TNC n=64"),
                 pd.read_csv(HERE / "CLUSTERS_FGBUSTER_recovered.csv", comment="#")]).sort_values("x")

slide_style(scale=1.3)
import matplotlib.pyplot as plt

NAVY = "#25406B"
fig, ax = plt.subplots(figsize=(7.4, 4.2))
ax.plot(fgb["x"], fgb["min_time"] / 1e3, color=NAVY, marker="o", ms=7, lw=2.6,
        label="Truncated Newton solver (reference)")
ax.plot(furax["x"], furax["min_time"] / 1e3, color=KW, marker="s", ms=7, lw=2.6,
        label="AdaTopK (this work)")
ax.set_yscale("log")
ax.set_ylim(5, 3e3)
ax.set_xlim(500, 11600)
ax.set_xlabel("number of sky patches")
ax.set_ylabel("time to fit  [s]")
ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.46), labelcolor=INK)
ax.grid(True, which="major", ls=":", alpha=0.5)

x = 10000
hi = float(fgb.loc[fgb["x"] == x, "min_time"].iloc[0]) / 1e3
lo = float(furax.loc[furax["x"] == x, "min_time"].iloc[0]) / 1e3
ax.annotate("", xy=(x, lo * 1.15), xytext=(x, hi / 1.15),
            arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.6))
ax.text(x + 220, (lo * hi) ** 0.5, rf"$\times${hi / lo:.0f}", color=INK, fontsize=17,
        ha="left", va="center", fontweight="bold")
fig.savefig(HERE / OUT, transparent=True)
print(f"wrote {OUT}  (x{hi / lo:.1f} at {x} patches)")
