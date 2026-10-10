#!/usr/bin/env python3
# ENV: jax-fli
"""
The forward model against CosmoGrid as the particle-mesh resolution grows, for the slide "Valid
scales for real survey analysis": one frame per mesh, in bandpowers of 16 multipoles (data and
drawing in _ladder.py, shared with the finer-bandpower backup in 7_backup/4_fieldlevel/09_ladder_nlb4).

Outputs (this directory), one per mesh on the same canvas, the meshes before it greyed:
  ladder_512.svg ... ladder_4096.svg
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import skip_if_built, slide_style

from _ladder import MESHES, frame, kernel_peaks, load

OUTS = [f"ladder_{m}.svg" for m in MESHES]
skip_if_built(HERE, *OUTS)

NLB = 16

D = load(NLB)
print("kernel peaks chi* [Mpc/h]:", kernel_peaks().round(0))
ELL = D["ell"]
for m in MESHES:
    r = D[f"m{m}"] / D["ref"] - 1
    at = [np.argmin(abs(ELL - l)) for l in (100, 200, 300)]
    print(f"{m:5d}^3: C_l/CosmoGrid - 1 at l = {ELL[at].round(0)}: bin 2 {r[1, at].round(2)}, "
          f"bin 3 {r[2, at].round(2)}")

slide_style()
for k, out in enumerate(OUTS):
    frame(D, k, HERE / out)
