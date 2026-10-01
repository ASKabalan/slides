#!/usr/bin/env python3
# ENV: shared
"""
SciPol illustrations used by the CMB slides, downscaled from the originals in
FOR_USAGE/ (10-50 MB each) to a size a slide needs.

Illustrations designed and drawn by Eve Barlier; SciPol is an ERC project held
by Josquin Errard, grant No. 101044073. Both are credited on the slides.

Outputs (this directory): scipol_polarisation.png, scipol_emodes.png,
scipol_bmodes.png, scipol_cmb_layers.png
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
    "scipol_polarisation.png": SRC / "Individual Assets/PNG/PNG/Polarisation_Transparent_v1.png",
    "scipol_emodes.png": SRC / "Individual assets/PNG/Poster_EModes_Scipol_v4.png",
    "scipol_bmodes.png": SRC / "Individual assets/PNG/Poster_BModes_Scipol_v4.png",
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
