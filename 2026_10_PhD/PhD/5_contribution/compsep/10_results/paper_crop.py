#!/usr/bin/env python3
# ENV: shared
"""
Title, authors and affiliations of the paper behind this contribution (Kabalan et al. 2026, RASTI),
cut from page one of the local manuscript, for the last fragment of the results slide.

Output (this directory): paper_crop.png
"""

import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from _common import skip_if_built

OUT = "paper_crop.png"
skip_if_built(HERE, OUT)

from PIL import Image

PDF = Path("/home/wassim/Projects/CMB/furax-pub/furax-cs-RASTI/FURAX_COMPONENT_SEPARATION.pdf")
if not PDF.exists():
    sys.exit(f"missing {PDF}")
with tempfile.TemporaryDirectory() as tmp:
    subprocess.run(["pdftoppm", "-r", "220", "-f", "1", "-l", "1", "-png", "-singlefile", str(PDF),
                    f"{tmp}/page"], check=True)
    page = Image.open(f"{tmp}/page.png").convert("RGB")
w, h = page.size
page.crop((round(0.045 * w), round(0.075 * h), round(0.955 * w), round(0.233 * h))).save(HERE / OUT)
print(f"wrote {OUT}")
