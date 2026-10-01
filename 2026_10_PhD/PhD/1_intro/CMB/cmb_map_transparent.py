#!/usr/bin/env python3
# ENV: shared
"""
The Planck temperature map (fetched once from Wikimedia Commons and kept in
.cache/) with its black surround made transparent, for the intro CMB slide
where the map sits on the cream slide.

The Mollweide ellipse is measured from the image (the extent of the non-black
pixels) and turned into an anti-aliased alpha mask (4x supersampled), inset by
a pixel so no dark fringe survives.

Output (this directory):
  cmb_map_transparent.png   Credit: ESA / Planck Collaboration
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from _common import commons_image, skip_if_built

from PIL import Image

OUT = "cmb_map_transparent.png"
skip_if_built(HERE, OUT)

im = commons_image(HERE / ".cache", "planck_cmb",
                  "File:Cosmic Microwave Background (CMB).jpeg", 2400).convert("RGB")
rgb = np.asarray(im)
lit = rgb.astype(int).sum(axis=2) > 60
ys, xs = np.nonzero(lit)
cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
a, b = (xs.max() - xs.min()) / 2 - 1.0, (ys.max() - ys.min()) / 2 - 1.0

SS = 4
h, w = lit.shape
yy, xx = np.mgrid[0:h * SS, 0:w * SS]
inside = ((xx + 0.5) / SS - 0.5 - cx) ** 2 / a**2 + ((yy + 0.5) / SS - 0.5 - cy) ** 2 / b**2 <= 1
alpha = inside.reshape(h, SS, w, SS).mean(axis=(1, 3))

rgba = np.dstack([rgb, np.round(255 * alpha).astype(np.uint8)])
out = Image.fromarray(rgba, "RGBA").crop((int(cx - a) - 2, int(cy - b) - 2,
                                           int(cx + a) + 3, int(cy + b) + 3))
out.save(HERE / OUT)
print(f"    wrote {OUT} {out.size}")
