#!/usr/bin/env python3
# ENV: jax-fli
"""
The last frame of the mesh ladder against CosmoGrid (all six meshes, 4096^3 in orange) in finer
bandpowers of 4 multipoles, for the backup slide "Valid scales: finer bandpowers" (data and drawing
in 5_contribution/fli/10_cosmogrid/_ladder.py, shared with the slide frames).

Output (this directory): ladder_4096_nlb4.svg
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import skip_if_built, slide_style

OUT = "ladder_4096_nlb4.svg"
skip_if_built(HERE, OUT)

sys.path.insert(0, str(ROOT / "5_contribution/fli/10_cosmogrid"))
from _ladder import MESHES, frame, load

slide_style()
frame(load(4), len(MESHES) - 1, HERE / OUT)
