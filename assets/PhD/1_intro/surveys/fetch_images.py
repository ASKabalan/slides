#!/usr/bin/env python3
# ENV: shared
"""
External images used by the introduction slides.

Bytes are cached in .cache/ under a readable key, so a rebuild costs no network.
File URLs are resolved through the Wikimedia Commons API rather than built by
hand, because the hashed upload paths change. Each image is credited on the
slide that shows it.

Outputs (this directory):
  cobe_wmap_planck.png   the same patch of microwave sky seen by COBE, WMAP and
                         Planck.  Credit: NASA / JPL-Caltech / ESA (PIA16874)
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from _common import commons_image, skip_if_built

IMAGES = {
    "cobe_wmap_planck.png": (
        "cobe_wmap_planck",
        "File:PIA16874-CobeWmapPlanckComparison-20130321.jpg",
        2600,
    ),
}

skip_if_built(HERE, *IMAGES)

for out, (key, title, max_side) in IMAGES.items():
    if (HERE / out).exists():
        continue
    im = commons_image(HERE / ".cache", key, title, max_side)
    im.save(HERE / out)
    print(f"    wrote {out} {im.size}")
