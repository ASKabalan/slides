#!/usr/bin/env python3
# ENV: shared
"""
Highlighted variants of the clean cosmic-history figure lcdm_model.svg.

Each variant is a copy of lcdm_model.svg with one rounded <rect> appended, in
the figure's own viewBox (4096 x 2288), so it stays vector and pixel-aligned:

  lcdm_model_cmb.svg   box around the CMB ellipse, dark keyword orange
  lcdm_model_lss.svg   box around the present-day matter distribution, dark blue
  lcdm_model_both.svg  both boxes, each with a curved arrow leaving it downwards (to the
                       left for the CMB, to the right for the matter distribution), on a
                       canvas extended below the figure: the two-column conclusion slide

With lcdm_model.svg and lcdm_model_arrow.svg (both hand-made) these are the
ΛCDM figures of the deck. The limits slide animates the same two boxes as
a CSS overlay: keep .lcdm-box in 2026_10_PhD/phd.scss in step with REGIONS.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import skip_if_built

SRC = HERE / "lcdm_model.svg"
W, H = 4096, 2288

# name: (x, y, width, height) in viewBox units, colour
REGIONS = {
    "cmb": ((1040, 800, 460, 820), "#C2560A"),
    "lss": ((3290, 580, 590, 1240), "#1F3A8A"),
}
STROKE, RADIUS = 22, 60

EXTRA_H = 620          # canvas added below the figure for the arrows
# (path d, colour): cubic curves from each box towards its column
ARROWS = [
    ("M 1090 1640 C 1000 2250, 800 2600, 420 2800", "#C2560A"),
    ("M 3340 1830 C 3340 2380, 3560 2640, 3960 2800", "#1F3A8A"),
]
ARROW_W, HEAD = 26, 110

skip_if_built(HERE, *(f"lcdm_model_{k}.svg" for k in (*REGIONS, "both")))

src = SRC.read_text()
close = src.rindex("</svg>")
for name, ((x, y, w, h), colour) in REGIONS.items():
    rect = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{RADIUS}" '
            f'fill="none" stroke="{colour}" stroke-width="{STROKE}"/>')
    (HERE / f"lcdm_model_{name}.svg").write_text(src[:close] + rect + src[close:])
    print(f"wrote lcdm_model_{name}.svg   CSS: left {100 * x / W:.2f}%  top {100 * y / H:.2f}%  "
          f"width {100 * w / W:.2f}%  height {100 * h / H:.2f}%")

# ---------------------------------------------------------------- both + arrows
rects = "".join(
    f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{RADIUS}" fill="none" '
    f'stroke="{colour}" stroke-width="{STROKE}"/>' for (x, y, w, h), colour in REGIONS.values())
markers = "".join(
    f'<marker id="head{i}" viewBox="0 0 10 10" refX="7" refY="5" markerUnits="userSpaceOnUse" '
    f'markerWidth="{HEAD}" markerHeight="{HEAD}" orient="auto">'
    f'<path d="M0 0 L10 5 L0 10 z" fill="{colour}"/></marker>' for i, (_, colour) in enumerate(ARROWS))
arrows = "".join(
    f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="{ARROW_W}" '
    f'stroke-linecap="round" marker-end="url(#head{i})"/>' for i, (d, colour) in enumerate(ARROWS))
both = src.replace(f'height="{H}" viewBox="0 0 {W} {H}"', f'height="{H + EXTRA_H}" viewBox="0 0 {W} {H + EXTRA_H}"', 1)
assert both != src
close = both.rindex("</svg>")
both = both[:close] + f"<defs>{markers}</defs>" + rects + arrows + both[close:]
(HERE / "lcdm_model_both.svg").write_text(both)
print("wrote lcdm_model_both.svg")
