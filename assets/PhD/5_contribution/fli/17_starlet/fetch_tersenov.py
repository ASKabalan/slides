#!/usr/bin/env python3
# ENV: shared
"""
Posteriors from the starlet l1 norm against a CNN, from Andreas Tersenov's talks
(https://andreastersenov.github.io/talks, figure posteriors/p2_corner_cnn_vs_l1/step4.png): the l1
norm of each tomographic bin alone, with the cross-bin products, the joint l1 norm, and a CNN, with
their figures of merit (2448, 3045, 3371, 3326). Shown on the starlet slide with his credit.

Output (this directory): tersenov_l1_cnn.png
"""

import io
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from _common import cached_fetch, guard_not_blank, skip_if_built

OUT = "tersenov_l1_cnn.png"
skip_if_built(HERE, OUT)

from PIL import Image

URL = "https://andreastersenov.github.io/talks/assets/figures/posteriors/p2_corner_cnn_vs_l1/step4.png"
im = Image.open(io.BytesIO(cached_fetch(HERE.parent / ".cache", "tersenov_step4", URL))).convert("RGBA")
guard_not_blank(im, "tersenov_step4")
im.save(HERE / OUT)
print(f"wrote {OUT} {im.size}")
