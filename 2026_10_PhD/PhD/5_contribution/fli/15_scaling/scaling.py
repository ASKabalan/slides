#!/usr/bin/env python3
# ENV: jax-fli
"""
Strong and weak scaling of the forward model and of its gradient on Jean Zay (NVIDIA H100), for
the scaling slide. The data of the thesis figures chap6/scaling_*.pdf and chap6/scaling_gradient.pdf:
the per-run profiler CSVs of ASKabalan/jax-fli-scaling,
  11-scaling/perf/perf_pm.csv            the particle-mesh forward model with its lightcone painting
  12-gradient-scaling/perf/perf_pm.csv   its gradient with respect to the initial conditions
in the 5000 Mpc/h box under a slab decomposition. Strong scaling holds a 1024^3 mesh and adds
GPUs; weak scaling keeps 256^3 cells per GPU. Time is the minimum over repeats, memory the peak
per-device temporary memory.

One row of four panels per figure, [strong time | strong memory | weak time | weak memory], the
two figures stacked on the slide with the strong / weak header on the forward one only:
  scaling_forward.svg    float32 against float64
  scaling_gradient.svg   float64, the reverse adjoint against the checkpointed adjoint with 30
                         checkpoints (at least one per shell of the twenty-shell lightcone, the
                         setting the thesis adopts)
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from _common import BLUE, INK, KW, KW2, TEAL, skip_if_built, slide_style

OUTS = ["scaling_forward.svg", "scaling_gradient.svg"]
skip_if_built(HERE, *OUTS)

import pandas as pd
from huggingface_hub import snapshot_download

REPO = "ASKabalan/jax-fli-scaling"
CSVS = {"forward": "11-scaling/perf/perf_pm.csv", "gradient": "12-gradient-scaling/perf/perf_pm.csv"}
COLS = ["function", "precision", "x", "y", "z", "px", "py", "backend", "nodes", "jit_time",
        "min_time", "max_time", "mean_time", "std_div", "last_time", "generated_code",
        "argument_size", "output_size", "temp_size"]

root = snapshot_download(REPO, repo_type="dataset", allow_patterns=list(CSVS.values()))


def load(which):
    d = pd.read_csv(f"{root}/{CSVS[which]}", header=None)
    d.columns = COLS
    tok = d["function"].str.split("_")
    d["kind"], d["series"] = tok.str[2], tok.str[-2]
    d["time"], d["memory"] = d["min_time"] / 1e3, d["temp_size"] / 2**30
    strong = d[(d.kind == "strong") & (d.x == 1024) & (d.y == 1024) & (d.z == 1024)]
    weak = d[(d.kind == "weak") & (d.x // d.px == 256)]
    return strong, weak


SERIES = {
    "forward": [("f32", "float32", BLUE, "-", "o"), ("f64", "float64", KW, "--", "s")],
    "gradient": [("rev", "reverse", TEAL, "-", "o"),
                 ("ckpt30", "checkpointed", KW2, "--", "s")],
}

slide_style()
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter, NullLocator, ScalarFormatter

plt.rcParams["savefig.bbox"] = None
plt.rcParams.update({"xtick.labelsize": 13, "ytick.labelsize": 13})
# shown at 1000 px wide on the slide. The forward figure carries the strong / weak header above its
# panels; the gradient figure sits under it with the same columns, so it drops the header and gives
# that height to its panels.
WIDTH = 10.4
LAYOUT = {"forward": (2.45, 0.22, 0.585), "gradient": (2.15, 0.25, 0.715)}   # height, bottom, panel height
LEFTS, PANEL_W = (0.062, 0.31, 0.558, 0.806), 0.172

for which, out in zip(("forward", "gradient"), OUTS):
    strong, weak = load(which)
    height, bottom, ph = LAYOUT[which]
    fig = plt.figure(figsize=(WIDTH, height))
    axes = [fig.add_axes((x, bottom, PANEL_W, ph)) for x in LEFTS]
    for ax, sub, col, ylabel in zip(axes, (strong, strong, weak, weak),
                                    ("time", "memory", "time", "memory"),
                                    ("time  [s]", "GiB per GPU") * 2):
        gpus = sorted(sub["px"].unique())
        for key, label, c, ls, m in SERIES[which]:
            d = sub[sub.series == key].sort_values("px")
            ax.plot(d["px"], d[col], color=c, ls=ls, marker=m, ms=7, lw=2.6, label=label)
            print(f"{which:8s} {sub.kind.iloc[0]:6s} {label:18s} {col:6s}",
                  dict(zip(d["px"], d[col].round(2))))
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(gpus))
        # seven weak-scaling counts do not fit at this size: every other one is labelled
        ax.set_xticklabels([str(g) if (len(gpus) <= 4 or k % 2 == 0) else "" for k, g in enumerate(gpus)])
        ax.xaxis.set_minor_locator(NullLocator())
        if col == "time":
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.margins(y=0.15)
        else:
            # the weak-scaling memory panel keeps headroom above its flat curves for the legend
            ax.set_ylim(0, (1.9 if sub is weak else 1.25) * sub[col].max())
        ax.set_xlabel("GPUs", fontsize=14, labelpad=1)
        ax.set_ylabel(ylabel, fontsize=14, labelpad=3)
        ax.grid(True, axis="y", which="both", ls=":", alpha=0.4)
    # the weak-scaling memory curves are flat, so the legend goes in the headroom above them
    axes[3].legend(loc="upper center", fontsize=12, handlelength=1.8, borderaxespad=0.15,
                   labelspacing=0.2, frameon=False)
    if which == "forward":
        fig.text(0.28, 0.9, r"strong scaling, $1024^3$ mesh", ha="center", fontsize=15, color=INK)
        fig.text(0.78, 0.9, r"weak scaling, $256^3$ cells per GPU", ha="center", fontsize=15, color=INK)
    fig.savefig(HERE / out)
    plt.close(fig)
    print(f"wrote {out}")
