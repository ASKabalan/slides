#!/usr/bin/env python3
# ENV: shared
"""
The flat (k = 0) plane of ../scientists/flrw_curv.svg, for the flatness
problem on the limits slide.

A copy of the three-panel figure with its viewBox cropped to the middle plane
(labels left out: the slide carries them), so the drawing stays vector and
identical to the one on the FLRW slide.

Output (this directory): flrw_flat.svg
"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import skip_if_built

SRC = HERE.parent / "scientists" / "flrw_curv.svg"
OUT = "flrw_flat.svg"
X0, Y0, W, H = 438.0, 106.0, 276.0, 140.0   # the flat plane only, in the source viewBox

skip_if_built(HERE, OUT)

svg = SRC.read_text()
svg, n = re.subn(r'width="[^"]*" height="[^"]*" viewBox="[^"]*"',
                 f'width="{W}pt" height="{H}pt" viewBox="{X0} {Y0} {W} {H}"', svg, count=1)
if n != 1:
    raise RuntimeError("could not find the root <svg> size attributes")
(HERE / OUT).write_text(svg)
print(f"wrote {OUT}")
