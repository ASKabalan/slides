#!/usr/bin/env python3
# ENV: shared
"""
Particle mesh against moving mesh: the cost of the gravity solve. The FFT Poisson solve of the
particle mesh (jaxdecomp pfft) against the real-space halo multigrid that the moving mesh needs
(B. Horowitz, JaxPM branch bhorowitz/MG), cold (u0 = 0, two V-cycles) and warm-started from a
converged potential (one V-cycle, the steady-state cost inside a simulation). Each run paints the
same displacements (CIC, slab decomposition along x) and reads the forces back; time is the
minimum over five repeats, memory the per-device temporary memory, float32, Jean Zay H100.

Data: the jax_hpc_profiler CSVs of JaxPM bench/bench_forces.py (branch bhorowitz/MG), local in
  ~/Projects/NBody/JaxPM/bench/results/{strong,weak}_scaling/forces.csv
(the replot of bench/results/plot.ipynb). Strong scaling holds a 1024^3 or 2048^3 mesh and adds
GPUs; weak scaling keeps 256^3 cells per GPU.

Output (this directory): moving_mesh_scaling.svg, a 2 x 3 grid, rows time / memory, columns
strong 1024^3 / strong 2048^3 / weak.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import BLUE, INK, KW, KW2, skip_if_built, slide_style

OUT = "moving_mesh_scaling.svg"
skip_if_built(HERE, OUT)

import pandas as pd

RESULTS = Path.home() / "Projects/NBody/JaxPM/bench/results"
COLS = ["function", "precision", "x", "y", "z", "px", "py", "backend", "nodes", "jit_time",
        "min_time", "max_time", "mean_time", "std_div", "last_time", "generated_code",
        "argument_size", "output_size", "temp_size"]


def load(kind):
    d = pd.read_csv(RESULTS / f"{kind}_scaling/forces.csv", header=None)
    d.columns = COLS
    d["time"], d["memory"] = d["min_time"], d["temp_size"] / 2**30   # ms, GiB per GPU
    return d


strong, weak = load("strong"), load("weak")
weak = weak[weak.x // weak.px == 256]
PANELS = [(strong[strong.x == 1024], r"strong scaling, $1024^3$ mesh"),
          (strong[strong.x == 2048], r"strong scaling, $2048^3$ mesh"),
          (weak, r"weak scaling, $256^3$ per GPU")]
SERIES = [("fft", "particle mesh (FFT)", BLUE, "-", "o"),
          ("mg", "moving mesh (multigrid, cold)", KW2, "--", "s"),
          ("mgwarm", "moving mesh (multigrid, warm)", KW, "-.", "^")]

slide_style()
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter, NullLocator

plt.rcParams["savefig.bbox"] = None
plt.rcParams.update({"xtick.labelsize": 13, "ytick.labelsize": 13})
# shown 1000 px wide on the slide: three columns, the column headers and the legend above them
fig = plt.figure(figsize=(10.4, 5.9))
LEFTS, PANEL_W = (0.088, 0.41, 0.732), 0.24
ROWS = (0.505, 0.095)   # bottoms of the time and memory rows
PH = 0.33

for c, (sub, title) in enumerate(PANELS):
    gpus = sorted(sub["px"].unique())
    for r, (col, ylabel) in enumerate((("time", "time  [ms]"), ("memory", "GiB per GPU"))):
        ax = fig.add_axes((LEFTS[c], ROWS[r], PANEL_W, PH))
        for key, label, colour, ls, m in SERIES:
            d = sub[sub.backend == key].sort_values("px")
            ax.plot(d["px"], d[col], color=colour, ls=ls, marker=m, ms=7, lw=2.4, label=label)
            print(f"{title:36s} {key:7s} {col:6s}", dict(zip(d["px"], d[col].round(2))))
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(gpus))
        # up to eight GPU counts: every other one is labelled
        ax.set_xticklabels([str(g) if (len(gpus) <= 4 or k % 2 == 0) else "" for k, g in enumerate(gpus)])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.margins(y=0.12)
        ax.grid(True, axis="y", which="both", ls=":", alpha=0.4)
        if r == 1:
            ax.set_xlabel("GPUs", fontsize=14, labelpad=1)
        else:
            ax.tick_params(labelbottom=False)
            ax.set_title(title, fontsize=15, color=INK, pad=6)
        if c == 0:
            ax.set_ylabel(ylabel, fontsize=14, labelpad=3)
        if c == 0 and r == 0:
            handles, labels = ax.get_legend_handles_labels()

fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=13.5, frameon=False,
           bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=1.6)
fig.savefig(HERE / OUT)
plt.close(fig)
print(f"wrote {OUT}")
