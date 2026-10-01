#!/usr/bin/env python3
# ENV: shared
"""
The Jean Zay supercomputer mark (IDRIS), for the scaling slide.

Source: Wikimedia Commons, File:Jean-Zay-logo.jpg. The white JPEG background is turned
transparent so the mark sits on the creamy slide.

Output (this directory): jean_zay_logo.png
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from _common import commons_image, skip_if_built

OUT = "jean_zay_logo.png"
skip_if_built(HERE, OUT)

from PIL import Image

im = np.asarray(commons_image(HERE.parent / ".cache", "jean_zay_logo", "File:Jean-Zay-logo.jpg",
                              max_side=400)).astype(float)
# alpha from the distance to white, so the JPEG fringe fades out instead of leaving a halo
alpha = np.clip((255 - im.min(axis=2)) / 40, 0, 1)
rgba = np.dstack([im, 255 * alpha]).astype(np.uint8)
Image.fromarray(rgba, "RGBA").save(HERE / OUT)
print(f"wrote {OUT}")
