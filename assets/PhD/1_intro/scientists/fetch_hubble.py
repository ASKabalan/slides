#!/usr/bin/env python3
# ENV: shared
"""
Edwin Hubble at the 100-inch Hooker telescope, Mount Wilson, for the
"Modern observational cosmology" slide.

Bytes are cached in .cache/ so a rebuild costs no network.

Output (this directory):
  hubble_portrait.jpg   Credit: NASA (science.nasa.gov)
"""

import io
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from _common import cached_fetch, guard_not_blank, skip_if_built

from PIL import Image

URL = "https://science.nasa.gov/wp-content/uploads/2023/04/default_1-jpg.webp"
OUT = "hubble_portrait.jpg"

skip_if_built(HERE, OUT)

im = Image.open(io.BytesIO(cached_fetch(HERE / ".cache", "nasa_hubble", URL))).convert("RGB")
guard_not_blank(im, "nasa_hubble")
w, h = im.size
im = im.crop((16, 16, w - 16, h - 16))  # drop the black print border of the scan
im.save(HERE / OUT, quality=92)
print(f"    wrote {OUT} {im.size}")
