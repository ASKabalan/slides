#!/usr/bin/env python3
# ENV: shared
"""
Stand-ins for the private photos of the deck, for the published site only.

The photos listed in PRIVATE stay on Wassim's machine (they are in .gitignore). This script writes,
under PhD/_public/ at the same relative paths, a plain grey box of each photo's exact pixel size
labelled "photo", so the slide layout is unchanged. The GitHub workflow copies them into place before
rendering, without overwriting (cp -n), so a local render always shows the real photos.

Run it locally, after adding or resizing a private photo:  uv run --project .. python make_standins.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PRIVATE = [
    "1_intro/telescopes/me_at_lsst_slide.jpg",
    "6_ending/photos/moriond.jpg",
    "6_ending/photos/ski.jpg",
    "6_ending/photos/sixty.jpg",
    "6_ending/photos/chili.jpg",
]
FILL, TEXT = (216, 221, 230), (110, 110, 110)

for rel in PRIVATE:
    w, h = Image.open(ROOT / rel).size
    im = Image.new("RGB", (w, h), FILL)
    d = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=max(18, min(w, h) // 9))
    d.text((w / 2, h / 2), "photo", fill=TEXT, font=font, anchor="mm")
    out = HERE / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out, quality=85)
    print(f"wrote _public/{rel}  ({w} x {h})")
