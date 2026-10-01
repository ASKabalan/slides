#!/usr/bin/env python3
# ENV: shared
"""
SciPol illustrations used by the CMB slides, downscaled from the originals in
FOR_USAGE/ (10-50 MB each) to a size a slide needs.

Illustrations designed and drawn by Eve Barlier; SciPol is an ERC project held
by Josquin Errard, grant No. 101044073. Both are credited on the slides.

Output (this directory): scipol_cmb_layers.png (read by 2_outline/rail.tex)
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from _common import skip_if_built

from PIL import Image

Image.MAX_IMAGE_PIXELS = None
SRC = HERE.parents[2] / "FOR_USAGE"
ASSETS = {
    "scipol_cmb_layers.png": SRC / "Individual assets/PNG/Poster_CMBLayers_Scipol_v4.png",
}
skip_if_built(HERE, *ASSETS)
for out, src in ASSETS.items():
    if not src.exists():
        print(f"MISSING {src}")
        continue
    im = Image.open(src).convert("RGBA")
    im.thumbnail((1600, 1600), Image.LANCZOS)
    im.save(HERE / out)
    print(f"wrote {out} {im.size}")
