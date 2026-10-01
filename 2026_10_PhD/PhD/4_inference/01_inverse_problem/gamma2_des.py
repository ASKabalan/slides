#!/usr/bin/env python3
# ENV: shared
"""
The second shear component over the DES Y3 footprint, from my own forward model.

This is figure 11 of the masked-shear experiment in jax-fli. The experiment is
the generator; regenerate the PDF with

    cd /home/wassim/Projects/NBody/jax-fli/docs/5-experiments/08-masked-shear
    JAX_PLATFORMS=cpu uv run --no-sync python build.py

and this script converts it for the slide and cuts the thumbnail the
inverse-problem diagram embeds.

Outputs (this directory):
  gamma2_des.svg          the full-sky map with the footprint
  .cache/shear_thumb.png  a crop of the footprint, for inverse_problem.tex
"""

import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from _common import skip_if_built

SRC = Path("/home/wassim/Projects/NBody/jax-fli/docs/5-experiments/"
           "08-masked-shear/assets/fig11-gamma2-des.pdf")
OUT = "gamma2_des.svg"
THUMB = HERE / ".cache" / "shear_thumb.png"

skip_if_built(HERE, OUT)
if not SRC.exists():
    sys.exit(f"missing {SRC}; run the masked-shear experiment first")
if shutil.which("pdftocairo") is None:
    sys.exit("pdftocairo not found")

subprocess.run(["pdftocairo", "-svg", str(SRC), str(HERE / OUT)], check=True)
print(f"wrote {OUT}")

# A tight crop of the footprint for the inverse-problem diagram.
THUMB.parent.mkdir(parents=True, exist_ok=True)
subprocess.run(["pdftocairo", "-png", "-r", "400", "-singlefile",
                str(SRC), str(THUMB.with_suffix(""))], check=True)

from PIL import Image

im = Image.open(THUMB).convert("RGBA")
w, h = im.size
im = im.crop((round(0.22 * w), round(0.32 * h), round(0.68 * w), round(0.84 * h)))
# The Mollweide grey outside the footprint, and the white page outside the
# ellipse, become transparent, so the crop drops onto the diagram cleanly.
px = im.load()
for y in range(im.height):
    for x in range(im.width):
        r, g, b, a = px[x, y]
        if abs(r - g) < 6 and abs(g - b) < 6 and (110 < r < 190 or r > 235):
            px[x, y] = (r, g, b, 0)
im = im.crop(im.getchannel("A").getbbox())
im.thumbnail((900, 900), Image.LANCZOS)
im.save(THUMB)
print(f"wrote {THUMB.relative_to(HERE)} {im.size}")
