#!/usr/bin/env python3
# ENV: jax-fli
"""
The starlet video of the MAP reconstruction, for the backup slide "MAP: starlet l1 norm of the reconstruction" (autoplay; frames in
5_contribution/fli/11_map/_map_frames.py, on the timeline shared with the other MAP backup).

Outputs (this directory): starlet_run.mp4, starlet_last.png.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import skip_if_built

skip_if_built(HERE, "starlet_run.mp4", "starlet_last.png")
sys.path.insert(0, str(ROOT / "5_contribution/fli/11_map"))
import _map_frames as M

M.render(M.starlet_frame, M.SCHED, "starlet", HERE, first=False)
