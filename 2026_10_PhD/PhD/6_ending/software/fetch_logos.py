#!/usr/bin/env python3
# ENV: shared
"""
Logos for the two software slides.

Project logos come from each project's own repository, mission logos from
Wikimedia Commons or the shared assets/Logos pool. Everything is cached in
.cache/ and written here as PNG (SVG sources are rasterised with rsvg-convert,
so the slide never depends on a remote host at talk time).

Outputs (this directory): logo_<name>.png for every entry below.
"""

import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import cached_fetch, commons_url, force_regen

CACHE = HERE / ".cache"
POOL = ROOT.parent / "assets" / "Logos"   # the shared slides assets, through the deck's assets link
RAW = "https://raw.githubusercontent.com/"

REMOTE = {  # name -> (url or commons title, kind)
    "megatop": (RAW + "CMBSciPol/Megatop/HEAD/docs/megatop_logo.png", "png"),
}
LOCAL = {
    "jax": POOL / "JaxLogo.png",
    "so": POOL / "so.webp",
    "litebird": POOL / "litebird.png",
    "scipol": POOL / "scipol.png",
    "furax": Path("/home/wassim/Projects/Perso/These_wassim/figures/chap5/furax_logo.png"),
}


def out_path(name: str) -> Path:
    return HERE / f"logo_{name}.png"


def svg_to_png(svg: bytes, out: Path, height: int = 360) -> None:
    src = CACHE / (out.stem + ".svg")
    src.write_bytes(svg)
    subprocess.run(["rsvg-convert", "-h", str(height), "-o", str(out), str(src)],
                   check=True)


CACHE.mkdir(exist_ok=True)
for name, (where, kind) in REMOTE.items():
    out = out_path(name)
    if out.exists() and not force_regen():
        continue
    url = commons_url(where) if kind.startswith("commons") else where
    blob = cached_fetch(CACHE, f"logo_{name}", url)
    if kind.endswith("svg"):
        svg_to_png(blob, out)
    else:
        out.write_bytes(blob)
    print(f"wrote {out.name}")

from PIL import Image

for name, src in LOCAL.items():
    out = out_path(name)
    if out.exists() and not force_regen():
        continue
    if not src.exists():
        print(f"MISSING {src}")
        continue
    if src.suffix in (".png",):
        shutil.copy(src, out)
    else:  # webp, jpeg: re-encode as PNG
        Image.open(src).convert("RGBA").save(out)
    print(f"wrote {out.name}")
