#!/usr/bin/env python3
# ENV: shared
"""
The SciPol CMB data-analysis pipeline, for the last slide of the CMB section.

Source: the pipeline graphic in FOR_USAGE/, on its light-background transparent version,
downscaled from 8000 x 4500 pixels. The slide draws highlight boxes over it (the CMB layers,
then the foregrounds, then component separation) at percentages of this image, so the
aspect ratio must stay 16:9.

Illustration designed and drawn by Ève Barlier; the graphic carries its own credit line.

Output (this directory): scipol_pipeline.png
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from _common import skip_if_built

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

SRC = (HERE.parents[2] / "FOR_USAGE" / "Full Pipeline Graphic" / "Light Background" /
       "Transparent PNG" / "Slides_Process_Scipol_v4_1920x1080_BlackTransparent.png")
OUT = "scipol_pipeline.png"

skip_if_built(HERE, OUT)
if not SRC.exists():
    sys.exit(f"missing {SRC}")

im = Image.open(SRC).convert("RGBA")
im.thumbnail((2400, 2400), Image.LANCZOS)
im.save(HERE / OUT)
print(f"wrote {OUT} {im.size}")
