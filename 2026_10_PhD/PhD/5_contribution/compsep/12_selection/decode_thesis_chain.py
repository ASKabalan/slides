#!/usr/bin/env python3
# ENV: shared
"""
The data of the thesis figure chap5/section_42/variance_vs_residual_r_total, read off its vector
paths, so the selection animation reproduces that exact figure.

The figure's own script is not in furax-cs, and its lower panel (the systematic bias r_sys) is
not a column of the public rows; the chain also holds a few runs the processed rows lack. The
PDF, however, stores every point as a vector: one marker per run and panel, (total patches,
r + sigma(r)) above and (total patches, r_sys) below, filled with the variance colour of that
run; the two grey polylines tell the data markers from the legend and colour-bar glyphs. The
axes are mapped from the tick marks (x: 0 to 20000 patches; upper y: 2 to 5 x 1e-3; lower y:
1e-7 to 1e-3, log). The colour bar runs from 0.9807 to 1.0949 uK^2.

Output (this directory): thesis_chain.csv (panel, total_patches, value, colour, style)
"""

import csv
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import skip_if_built

OUT = "thesis_chain.csv"
skip_if_built(HERE, OUT)

PDF = Path("/home/wassim/Projects/Perso/These_wassim/figures/chap5/section_42/"
           "variance_vs_residual_r_total.pdf")
GREY = "rgb(59.999084%, 59.999084%, 59.999084%)"

with tempfile.TemporaryDirectory() as tmp:
    subprocess.run(["pdftocairo", "-svg", str(PDF), f"{tmp}/f.svg"], check=True)
    svg = Path(f"{tmp}/f.svg").read_text()
body = svg[svg.index("</defs>"):]
paths = re.findall(r"<path ([^>]*?)/>", body)


def attr(e, k):
    m = re.search(k + r'="([^"]*)"', e)
    return m.group(1) if m else None


def points(d):
    return [tuple(map(float, m)) for m in re.findall(r"[ML] ([-\d.]+) ([-\d.]+)", d)]


def rgb(s):
    return "#%02x%02x%02x" % tuple(round(float(v) * 2.55) for v in re.findall(r"([\d.]+)%", s))


# axis maps (coordinates before the page's y flip, as the polylines are stored)
X0, DX = 72.85, (159.68 - 72.85) / 5000             # x ticks at 0 and 5000 patches
top = lambda y: 2 + (y - 295.49) / 51.63             # (r + sigma) x 1e3, ticks 2..5
bottom = lambda y: 10 ** (-3 + (y - 235.09) / 46.255)   # r_sys, ticks 1e-7, 1e-5, 1e-3
SCALE, FLIP = 0.998848, 468.931482                  # page transform of every path

lines = [points(attr(p, "d")) for p in paths if attr(p, "stroke") == GREY]
upper, lower = lines[0], lines[1]


def near_line(x, y, line):
    """Distance from (x, y) to a polyline; collinear vertices (the capped points at 5e-3) are
    merged in the PDF, so markers are matched to its segments, not its vertices."""
    best = float("inf")
    for (x0, y0), (x1, y1) in zip(line[:-1], line[1:]):
        dx, dy = x1 - x0, y1 - y0
        t = max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / (dx * dx + dy * dy or 1)))
        best = min(best, ((x - x0 - t * dx) ** 2 + (y - y0 - t * dy) ** 2) ** 0.5)
    return best


# markers: small circle paths, drawn either around the origin and placed by a transform, or
# directly in page coordinates; a filled marker is a fill path plus a separate outline path
found = {}
for p in paths:
    d = attr(p, "d") or ""
    if " C " not in d:
        continue
    xy = [tuple(map(float, m)) for m in re.findall(r"(-?[\d.]+) (-?[\d.]+)", d)]
    xs, ys = [q[0] for q in xy], [q[1] for q in xy]
    if max(xs) - min(xs) > 12 or max(ys) - min(ys) > 12:
        continue
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    t = attr(p, "transform")
    if t:
        m = list(map(float, re.findall(r"[-\d.]+", t)))
        cx, cy = m[0] * cx + m[2] * cy + m[4], m[1] * cx + m[3] * cy + m[5]
    x, y = cx / SCALE, (FLIP - cy) / SCALE                 # page -> line coordinates
    panel = "upper" if y > 260 else "lower"
    ref = upper if panel == "upper" else lower
    if near_line(x, y, ref) > 1.5:
        continue                                          # legend or colour-bar glyphs
    key = (panel, round(x, 1), round(y, 1))
    fill, stroke = attr(p, "fill"), attr(p, "stroke")
    entry = found.setdefault(key, {"fill": None, "stroke": None, "x": x, "y": y})
    if fill and fill != "none":
        entry["fill"] = rgb(fill)
    if stroke and stroke != "none":
        entry["stroke"] = rgb(stroke)
markers = []
for (panel, _, _), e in found.items():
    total = round((e["x"] - X0) / DX)
    value = top(e["y"]) if panel == "upper" else bottom(e["y"])
    style = "filled" if e["fill"] else "open"
    colour = e["fill"] or e["stroke"]
    markers.append((panel, total, f"{value:.6e}", colour, style))
with open(HERE / OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["panel", "total_patches", "value", "colour", "style"])
    w.writerows(sorted(markers, key=lambda m: (m[0], m[1])))
print(f"wrote {OUT}: {len(markers)} points")
