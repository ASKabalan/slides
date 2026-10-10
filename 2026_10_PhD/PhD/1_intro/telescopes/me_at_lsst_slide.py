#!/usr/bin/env python3
# ENV: shared
"""
A slide-sized copy of me_at_lsst.jpg (2160 x 3840 phone photo) for the
telescope quadrant slide.

Output (this directory): me_at_lsst_slide.jpg (700 px wide)
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import skip_if_built

from PIL import Image, ImageOps

OUT = "me_at_lsst_slide.jpg"
skip_if_built(HERE, OUT)

im = ImageOps.exif_transpose(Image.open(HERE / "me_at_lsst.jpg")).convert("RGB")
im.thumbnail((700, 2000), Image.LANCZOS)
im.save(HERE / OUT, quality=88)
print(f"wrote {OUT} {im.size}")
