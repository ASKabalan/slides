#!/usr/bin/env python3
# ENV: shared
"""
Sketch of a full-sky lightcone built by replicating the simulation box, Fig. 1 of Sgier et al.
(2021), "Fast lightcones for combined cosmological probes", arXiv:2007.05735 (UFalcon): the
snapshots fill concentric shells around the observer at the centre, and each shell rotates the
replicated volume differently (dashed lines) so that the repeats do not line up. Taken from the
arXiv source (figures/sketch_lightcone.pdf) and converted to SVG. The white page rectangles are
dropped and the embedded particle image is un-composited from white (alpha = largest drop below
white over the channels), so the figure sits on the slide background unchanged.

Output (this directory): box_replication.svg
"""

import base64
import io
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import cached_fetch, skip_if_built

OUT = "box_replication.svg"
skip_if_built(HERE, OUT)

URL = "https://arxiv.org/e-print/2007.05735"
src = cached_fetch(ROOT / "5_contribution/fli/.cache", "sgier2021_src", URL)
with tarfile.open(fileobj=io.BytesIO(src), mode="r:gz") as tar, tempfile.TemporaryDirectory() as tmp:
    pdf = Path(tmp) / "sketch_lightcone.pdf"
    pdf.write_bytes(tar.extractfile("figures/sketch_lightcone.pdf").read())
    subprocess.run(["pdftocairo", "-svg", str(pdf), str(HERE / OUT)], check=True)

import numpy as np
from PIL import Image


def unwhite(m):
    """The embedded PNG, un-composited from white: c = a * c' + (1 - a) * 255."""
    rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(m.group(1)))).convert("RGB"), float)
    a = (255 - rgb).max(axis=-1, keepdims=True) / 255
    c = np.where(a > 0, (rgb - 255 * (1 - a)) / np.maximum(a, 1e-6), 0)
    rgba = np.concatenate([c, 255 * a], axis=-1).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode() + '"'


svg = (HERE / OUT).read_text()
svg, n_rect = re.subn(r'<rect [^>]*fill="rgb\(100%, 100%, 100%\)"[^>]*/>', "", svg)
svg, n_img = re.subn(r'data:image/png;base64,([^"]+)"', unwhite, svg)
assert n_rect >= 1 and n_img == 1, (n_rect, n_img)
(HERE / OUT).write_text(svg)
print(f"wrote {OUT} ({n_rect} white rectangles dropped, particle image made transparent)")
