#!/usr/bin/env python3
# ENV: shared
"""
Full-slide backgrounds for the conclusion slides, from the two ends of the cover video
(cmb_to_cosmicweb.mp4), handed over as stills in source/ (2560 x 1440):

  bg_cmb.jpg           the CMB field (Contribution 1 conclusion)
  bg_cosmic_web.jpg    the cosmic web (Contribution 2 conclusion)
  bg_cmb_lss.jpg       the CMB on the left fading into the cosmic web on the right, a smoothstep
                       mask over the middle third of the width, as in the cover (final conclusion)

All 1920 x 1080 JPEG, so the deck does not carry the 10 MB of PNG originals.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _common import skip_if_built

OUTS = ["bg_cmb.jpg", "bg_cosmic_web.jpg", "bg_cmb_lss.jpg"]
skip_if_built(HERE, *OUTS)

SIZE, QUALITY = (1920, 1080), 88
load = lambda name: np.asarray(Image.open(HERE / "source" / name).convert("RGB").resize(SIZE, Image.LANCZOS),
                               dtype=float)
cmb, web = load("cmb_field.png"), load("cosmic_web.png")

x = np.linspace(0, 1, SIZE[0])
t = np.clip((x - 1 / 3) * 3, 0, 1)
w = (t * t * (3 - 2 * t))[None, :, None]          # smoothstep, 0 on the left third, 1 on the right third
blend = (1 - w) * cmb + w * web

for img, out in zip((cmb, web, blend), OUTS):
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(HERE / out, quality=QUALITY)
    print(f"wrote {out}")
