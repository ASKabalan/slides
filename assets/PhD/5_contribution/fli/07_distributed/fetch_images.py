#!/usr/bin/env python3
# ENV: shared
"""
External material for the slide on distributing the simulation.

Outputs (this directory):
  joss_header.png   title block of the jaxDecomp software paper,
                    Kabalan, Lanusse, Boucaud & Aubourg, JOSS (2026),
                    doi:10.21105/joss.08852
  decomp2d.gif      the pencil-decomposition animation from the jaxDecomp
                    documentation (https://jaxdecomp.readthedocs.io)
"""

import io
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from _common import cached_fetch, force_regen

CACHE = HERE.parent / ".cache"
JOSS = "https://joss.theoj.org/papers/10.21105/joss.08852.pdf"
GIF = "https://jaxdecomp.readthedocs.io/en/latest/_images/decomp2d.gif"
LOCAL_GIF = HERE.parents[3] / "HPC" / "decomp2d.gif"

# --- JOSS title block ---------------------------------------------------------
header = HERE / "joss_header.png"
if force_regen() or not header.exists():
    pdf = CACHE / "joss_08852.pdf"
    pdf.write_bytes(cached_fetch(CACHE, "joss_08852", JOSS))
    stem = CACHE / "joss_page1"
    subprocess.run(["pdftocairo", "-png", "-r", "260", "-f", "1", "-l", "1",
                    "-singlefile", str(pdf), str(stem)], check=True)
    from PIL import Image

    im = Image.open(stem.with_suffix(".png")).convert("RGB")
    w, h = im.size
    # Title and authors: right of the JOSS sidebar, stopping above the affiliations.
    im.crop((round(0.265 * w), round(0.150 * h), round(0.975 * w), round(0.259 * h))
            ).save(header)
    print(f"wrote {header.name}")

# --- decomposition animation --------------------------------------------------
gif = HERE / "decomp2d.gif"
if force_regen() or not gif.exists():
    if LOCAL_GIF.exists():
        gif.write_bytes(LOCAL_GIF.read_bytes())
        print(f"copied {gif.name} from assets/HPC")
    else:
        gif.write_bytes(cached_fetch(CACHE, "decomp2d", GIF))
        print(f"wrote {gif.name}")
