#!/usr/bin/env python3
# ENV: shared
"""
Memory and accuracy of the two adjoints, for the backup slide on adjoints: the slide version of the
thesis figure chap6/adjoint_memory.pdf (These_wassim/figures/chap6/adjoint_memory.py).

jax-fli experiment 09b (docs/5-experiments/09-gradient-validation/data_f64/degradation.npz, made on GPU
and not published): the gradient of a 50-step particle-mesh run, float64, with the reverse adjoint and
with the checkpointed adjoint storing 1 to 50 step checkpoints, for DoubleKickDrift and BullFrog.
Bars (right axis): the XLA temporary buffer in MB. Markers (left axis): the median per-voxel relative
difference to central finite differences, |g_i - FD_i| / |FD_i|.

Output (this directory): adjoint_memory.svg
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import BLUE, INK, KW, KW2, TEAL, skip_if_built, slide_style

OUT = "adjoint_memory.svg"
skip_if_built(HERE, OUT)

NPZ = Path("/home/wassim/Projects/NBody/jax-fli/docs/5-experiments/09-gradient-validation/data_f64/degradation.npz")
d = np.load(NPZ)
CKPTS = [1, 2, 5, 10, 20, 30, 50]
SERIES = [("kdk", "rev", "kick-drift-kick, reverse", KW), ("bf", "rev", "BullFrog, reverse", KW2),
          ("kdk", "chk", "kick-drift-kick, checkpointed", TEAL), ("bf", "chk", "BullFrog, checkpointed", BLUE)]
mb = lambda k: float(d[k + "__mem"][0]) / 1e6
acc = lambda k: float(d[k + "__med"])

slide_style()
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams["savefig.bbox"] = None
pos = np.concatenate([[0.0], np.arange(len(CKPTS)) + 1.8])
w = 0.8 / len(SERIES)
fig = plt.figure(figsize=(6.0, 5.0))
ax = fig.add_axes([0.17, 0.12, 0.66, 0.72])
axr = ax.twinx()
mems = []
for s, (sv, adj, lab, c) in enumerate(SERIES):
    off = (s - (len(SERIES) - 1) / 2) * w
    if adj == "rev":
        x, m, a = pos[:1], [mb(f"B__{sv}__rev")], [acc(f"B__{sv}__rev")]
    else:
        x = pos[1:]
        m = [mb(f"B__{sv}__c{k}__chk") for k in CKPTS]
        a = [acc(f"B__{sv}__c{k}__chk") for k in CKPTS]
    mems += m
    axr.bar(x + off, m, width=w * 0.9, color=c, alpha=0.4, lw=0, zorder=1)
    ax.plot(x + off, np.clip(a, 1e-9, 1e-5), color=c, lw=1.4, marker="o", ms=6, zorder=5)
ax.set_yscale("log")
ax.set_ylim(1e-9, 1e-5)
ax.set_xticks(pos)
ax.set_xticklabels(["reverse", *[str(k) for k in CKPTS]], fontsize=11)
ax.set_xlabel("checkpoints stored", labelpad=2)
ax.set_ylabel(r"median $|g - \mathrm{FD}| / |\mathrm{FD}|$", fontsize=13)
axr.set_ylabel("memory  [MB]", fontsize=13)
axr.set_ylim(0, 1.12 * max(mems))
ax.set_zorder(axr.get_zorder() + 1)
ax.patch.set_visible(False)
ax.grid(True, axis="y", ls=":", alpha=0.35)
handles = [(Patch(fc=c, alpha=0.4), Line2D([], [], color=c, marker="o", ms=6, ls="none")) for *_, c in SERIES]
fig.legend(handles=handles, labels=[lab for _, _, lab, _ in SERIES], handler_map={tuple: HandlerTuple(ndivide=None)},
           loc="upper center", ncol=2, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 1.0),
           handlelength=1.6, columnspacing=1.0, labelspacing=0.3)
fig.savefig(HERE / OUT)
print("reverse MB:", {sv: round(mb(f'B__{sv}__rev'), 1) for sv in ('kdk', 'bf')},
      "checkpointed 50 MB:", {sv: round(mb(f'B__{sv}__c50__chk'), 1) for sv in ('kdk', 'bf')})
print(f"wrote {OUT}")
