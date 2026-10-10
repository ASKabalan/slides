#!/usr/bin/env python3
# ENV: shared
"""
Label edits of the cosmic-history figure, applied to the two hand-made originals in source/:

  source/lcdm_model_orig.svg        -> lcdm_model.svg
  source/lcdm_model_arrow_orig.svg  -> lcdm_model_arrow.svg   (same figure + the HOT -> COLD arrow)

Both are a transparent PNG (4096 x 2288) under one traced vector <path> that holds every label
and leader line. The path is split into its subpaths (each starts with M; every command is
absolute: M, L, C, Z), and:

  - "Quantum Fluctuations" and its leader line are removed (and the leader's short stub painted
    inside the inflation bulge is filled from the neighbouring pixels of the PNG);
  - "Big Bang Expansion" becomes "Big Bang", moved down onto the tip of the funnel;
  - the second "1st STARS (EPOCH OF REIONIZATION) ABOUT 400 MILLION YEARS" label, under the
    first-stars ellipse, is removed with its leader line;
  - "TODAY / PRESENT" becomes "TODAY";
  - "RECOMBINATION" is added above "380,000 YEARS", built from glyphs of the removed label (same
    font and size as the figure), and the CMB leader line is shortened to make room.

Run lcdm_variants.py afterwards (FORCE_REGEN=1): the boxed variants are copies of lcdm_model.svg.
"""

import base64
import io
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import skip_if_built

SRC = {"lcdm_model.svg": HERE / "source" / "lcdm_model_orig.svg",
       "lcdm_model_arrow.svg": HERE / "source" / "lcdm_model_arrow_orig.svg"}
skip_if_built(HERE, *SRC)

from PIL import Image

PAIR = re.compile(r"(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)")


def points(sp):
    return np.array([(float(x), float(y)) for x, y in PAIR.findall(sp)])


def bbox(sp):
    p = points(sp)
    return p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()


def transform(sp, f):
    return PAIR.sub(lambda m: "{:.1f},{:.1f}".format(*f(float(m.group(1)), float(m.group(2)))), sp)


def inside(b, box):
    """Subpath bbox b entirely within box (x0, y0, x1, y1)."""
    return b[0] >= box[0] and b[1] >= box[1] and b[2] <= box[2] and b[3] <= box[3]


# ---------------------------------------------------------------- regions (viewBox units)
QUANTUM = (380, 800, 800, 1085)          # "Quantum Fluctuations" + its leader line
EXPANSION = (250, 1206, 525, 1280)       # "Expansion"
BIG_BANG = (250, 1125, 525, 1206)        # "Big Bang", moved down by BIG_BANG_DY
BIG_BANG_DY = 36
STARS_LOW = (1840, 1570, 2520, 1945)     # the lower first-stars label + its leader line
PRESENT = (3440, 1940, 3740, 2012)       # "/ PRESENT"
CMB_LEADER = (1236, 1555, 1318, 1725)    # leader from the CMB ellipse to "380,000 YEARS"
STUB = ((787.0, 1068.0), (843.0, 1123.0))  # the leader's stub inside the bulge (PNG pixels)

# source lines of the lower first-stars label (glyph rows), in viewBox units
LINE_EPOCH = (1860, 1815, 2500, 1890)    # (EPOCH OF REIONIZATION)
LINE_ABOUT = (1850, 1878, 2512, 1945)    # ABOUT 400 MILLION YEARS
# RECOMBINATION from those glyphs: (line, glyph index)
WORD = [("e", 8), ("e", 9), ("e", 4), ("e", 11), ("a", 8), ("a", 1), ("e", 10),
        ("e", 12), ("e", 15), ("e", 16), ("e", 17), ("e", 18), ("e", 19)]
KERN = {1: -4.0}                         # after the E: its open right side reads as a wider gap
EXPECT = {"e": 21, "a": 20}              # glyphs per source line: ( E P O C H O F R E I O N I Z A T I O N )


def glyphs(subs, boxes, region):
    """Subpaths of one text row grouped into glyphs: a subpath starting inside the previous glyph
    (a hole, a dot) joins it; letters that only touch (A and T in REIONIZATION) stay apart."""
    ids = sorted((i for i, b in enumerate(boxes) if inside(b, region)), key=lambda i: boxes[i][0])
    out = []
    for i in ids:
        b = boxes[i]
        if out and b[0] < out[-1]["x1"] - 1.0:
            out[-1]["ids"].append(i)
            out[-1]["x1"] = max(out[-1]["x1"], b[2])
            out[-1]["y1"] = max(out[-1]["y1"], b[3])
        else:
            out.append({"ids": [i], "x0": b[0], "x1": b[2], "y1": b[3]})
    return out


def edit_labels(d):
    subs = re.findall(r"M[^M]*", d)
    boxes = [bbox(sp) for sp in subs]

    rows = {"e": glyphs(subs, boxes, LINE_EPOCH), "a": glyphs(subs, boxes, LINE_ABOUT)}
    for k, g in rows.items():
        if len(g) != EXPECT[k]:
            sys.exit(f"source row {k}: {len(g)} glyphs, expected {EXPECT[k]}; check the regions")
    # letter spacing of the source: median gap between consecutive glyphs of REIONIZATION
    reion = rows["e"][8:20]
    gap = float(np.median([b["x0"] - a["x1"] for a, b in zip(reion, reion[1:])]))
    base = {k: float(np.median([g["y1"] for g in rows[k][1:-1]])) for k in rows}

    # the CMB column: "380,000" sits on the line below the new word
    digits = [b for b in boxes if inside(b, (1160, 1750, 1390, 1795))]
    top_380 = min(b[1] for b in digits)
    pitch = 64.0                                     # "380,000" -> "YEARS" line pitch
    cap = base["e"] - min(boxes[i][1] for i in rows["e"][8]["ids"])
    word_base = top_380 - pitch + cap
    width = sum(rows[k][j]["x1"] - rows[k][j]["x0"] for k, j in WORD) + sum(
        rows[k][j + 1]["x0"] - rows[k][j]["x1"] if (k2, j2) == (k, j + 1) else gap + KERN.get(n, 0.0)
        for n, ((k, j), (k2, j2)) in enumerate(zip(WORD, WORD[1:])))
    x_cmb = 0.5 * (CMB_LEADER[0] + CMB_LEADER[2])
    cursor = x_cmb - width / 2

    added = []
    for n, (k, j) in enumerate(WORD):
        g = rows[k][j]
        dx, dy = cursor - g["x0"], word_base - base[k]
        added += [transform(subs[i], lambda x, y, dx=dx, dy=dy: (x + dx, y + dy)) for i in g["ids"]]
        # letters that were neighbours in the source keep their own spacing (A and T touch)
        nxt = WORD[n + 1] if n + 1 < len(WORD) else None
        step = rows[k][j + 1]["x0"] - g["x1"] if nxt == (k, j + 1) else gap + KERN.get(n, 0.0)
        cursor += g["x1"] - g["x0"] + step

    keep = []
    for sp, b in zip(subs, boxes):
        if any(inside(b, r) for r in (QUANTUM, EXPANSION, STARS_LOW, PRESENT)):
            continue
        if inside(b, BIG_BANG):
            sp = transform(sp, lambda x, y: (x, y + BIG_BANG_DY))
        elif inside(b, CMB_LEADER) and b[3] - b[1] > 100:
            # shorten the leader: its foot moves up to clear the new word, its top stays on the ellipse
            y0, y1 = b[1], b[3]
            new_y1 = word_base - cap - 18
            sp = transform(sp, lambda x, y: (x, y0 + (y - y0) * (new_y1 - y0) / (y1 - y0)))
        keep.append(sp)
    print(f"label path: {len(subs)} subpaths -> {len(keep) + len(added)} "
          f"({len(subs) - len(keep)} removed, {len(added)} added for RECOMBINATION, "
          f"{width:.0f} units wide at x = {x_cmb:.0f})")
    return "".join(keep + added)


def fill_stub(png_bytes):
    """Paint out the leader stub inside the inflation bulge. Each pixel within HALF of the segment
    is replaced by the linear blend, across the stub, of the pixels SIDE away on either side of the
    segment (each the mean over a few offsets, premultiplied alpha). The stub crosses the bulge outline almost at a right
    angle, so the blend also rebuilds the outline where they cross."""
    im = np.array(Image.open(io.BytesIO(png_bytes)).convert("RGBA")).astype(float)
    pre = np.concatenate([im[..., :3] * im[..., 3:4] / 255, im[..., 3:4]], -1)
    p0, p1 = np.array(STUB[0]), np.array(STUB[1])
    u = (p1 - p0) / np.linalg.norm(p1 - p0)
    n = np.array([-u[1], u[0]])
    length = np.linalg.norm(p1 - p0)
    HALF, FEATHER, SIDE = 6.0, 2.5, (9.0, 10.0, 11.0, 12.0, 13.0)
    x0, y0 = np.floor(np.minimum(p0, p1) - max(SIDE) - 12).astype(int)
    x1, y1 = np.ceil(np.maximum(p0, p1) + max(SIDE) + 12).astype(int)
    ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    rel = np.stack([xs - p0[0], ys - p0[1]], -1)
    t, s = rel @ u, rel @ n
    mask = (np.abs(s) < HALF + FEATHER) & (t > -10) & (t < length + 5)

    def sample(pts):
        """Bilinear sample of the premultiplied image."""
        x, y = pts[..., 0], pts[..., 1]
        xf, yf = np.floor(x).astype(int), np.floor(y).astype(int)
        fx, fy = (x - xf)[..., None], (y - yf)[..., None]
        at = lambda yy, xx: pre[np.clip(yy, 0, im.shape[0] - 1), np.clip(xx, 0, im.shape[1] - 1)]  # noqa: E731
        return ((1 - fx) * (1 - fy) * at(yf, xf) + fx * (1 - fy) * at(yf, xf + 1)
                + (1 - fx) * fy * at(yf + 1, xf) + fx * fy * at(yf + 1, xf + 1))

    foot = p0 + t[..., None] * u
    a = np.mean([sample(foot - d * n) for d in SIDE], 0)
    b = np.mean([sample(foot + d * n) for d in SIDE], 0)
    mid = float(np.mean(SIDE))
    w = np.clip((s + mid) / (2 * mid), 0, 1)[..., None]
    fill = (1 - w) * a + w * b
    keep = np.clip((np.abs(s) - HALF) / FEATHER, 0, 1)[..., None]   # feathered edge, no seam
    patch = pre[ys, xs]
    patch[mask] = (keep * patch + (1 - keep) * fill)[mask]
    pre[ys, xs] = patch
    alpha = pre[..., 3:4]
    out = np.concatenate([np.where(alpha > 0, pre[..., :3] * 255 / np.maximum(alpha, 1e-9), 0), alpha], -1)
    buf = io.BytesIO()
    Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8), "RGBA").save(buf, "PNG", optimize=True)
    return buf.getvalue()


def edit_svg(text):
    m = re.search(r'xlink:href="data:image/png;base64,([^"]+)"', text)
    png = base64.b64encode(fill_stub(base64.b64decode(m.group(1)))).decode()
    text = text[:m.start(1)] + png + text[m.end(1):]
    m = re.search(r'<path fill="#000" fill-rule="evenodd" d="([^"]+)"', text)
    return text[:m.start(1)] + edit_labels(m.group(1)) + text[m.end(1):]


for out, src in SRC.items():
    (HERE / out).write_text(edit_svg(src.read_text()))
    print(f"wrote {out}")
