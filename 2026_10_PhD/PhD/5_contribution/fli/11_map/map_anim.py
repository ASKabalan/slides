#!/usr/bin/env python3
# ENV: jax-fli
"""
The MAP reconstruction of the initial conditions animated for the MAP slide (frames in _map_frames.py).

Outputs (this directory): map_run.mp4, map_first.png (the slide's entry state), map_last.png.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import skip_if_built

skip_if_built(HERE, "map_run.mp4", "map_first.png", "map_last.png")

import _map_frames as M

M.render(M.map_frame, M.MAP_SCHED, "map", HERE)
