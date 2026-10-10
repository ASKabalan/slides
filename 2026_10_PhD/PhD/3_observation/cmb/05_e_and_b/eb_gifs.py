#!/usr/bin/env python3
# ENV: shared
"""
The two everyday pictures of E and B modes on the "E and B modes" slide:
a drop falling into water (rings: the E-like, curl-free pattern) and a vortex
(the B-like, swirling one). Downloaded as-is into .cache/ and copied here.

Outputs (this directory):
  eb_drop.gif     tenor.com/8x5b-Zwr7IQ (slow-motion drop; only the 220 px
                  version is served)
  eb_vortex.gif   giphy.com/gifs/x5fBY4VnAR36IB45wQ
  eb_drop_first.png, eb_vortex_first.png   first frames, for the PDF
"""

import io
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import cached_fetch, skip_if_built

from PIL import Image

GIFS = {
    "eb_drop": "https://media.tenor.com/8x5b-Zwr7IQAAAAM/slow-motion.gif",
    "eb_vortex": "https://media3.giphy.com/media/x5fBY4VnAR36IB45wQ/giphy.gif",
}

skip_if_built(HERE, *(f"{k}.gif" for k in GIFS))

for stem, url in GIFS.items():
    raw = cached_fetch(HERE / ".cache", stem, url)
    (HERE / f"{stem}.gif").write_bytes(raw)
    Image.open(io.BytesIO(raw)).convert("RGB").save(HERE / f"{stem}_first.png")
    print(f"wrote {stem}.gif and {stem}_first.png")
