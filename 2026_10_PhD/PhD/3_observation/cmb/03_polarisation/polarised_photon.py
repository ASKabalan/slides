#!/usr/bin/env python3
# ENV: manim
# ─────────────────────────────────────────────────────────────
# Render with:
#   manim -qh -t --format=gif --frame_rate 30 -r 1920,480 polarised_photon.py PolarisedPhoton
# or run this file (it renders and copies polarised_photon.gif beside itself, and
# writes a mid-flight frame as a vector polarised_photon_last.svg for the PDF):
#   uv run --project ../.. python polarised_photon.py
# Manim Community v0.21.x
# ─────────────────────────────────────────────────────────────
"""
A linearly polarised photon travelling into the CMB, for the "Polarisation of the
CMB" slide.

A continuous light wave fills the whole propagation axis and travels with the
photon (orange, with a halo): the field in the vertical plane in red, the
perpendicular one in blue, drawn in an oblique "depth" direction as in the
textbook picture, both as a curve with arrows from the axis. No labels. The
photon runs into the Planck map on the right and the GIF loops with it back at
the left; the flight covers a whole number of wavelengths, so the wave pattern
is identical at both ends of the loop and has no start or finish.

The map is NOT in the GIF (re-encoding its noise in every frame made a 30 MB
file): the slide lays 1_intro/CMB/cmb_map_transparent.png over the GIF at the box
printed by this script, so the wave runs behind it. The vector frame for the PDF
does include it.

Transparent GIF, so alpha is 1-bit: the halo is drawn as opaque rings pre-blended
onto the slide colour (#faf7f0).
"""

import base64
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
from manim import (Arrow, Dot, Line, Scene, ValueTracker, VGroup, VMobject,
                   always_redraw, config, linear)

config.background_opacity = 0.0
config.frame_height = 4.0
config.frame_width = 16.0

HERE = Path(__file__).resolve().parent
MAP = HERE.parents[2] / "1_intro" / "CMB" / "cmb_map_transparent.png"

# ------------------------------------------------------------------ tunables
X_LEFT = -8.2                     # the axis starts just off the left edge
MAP_X, MAP_H = 5.2, 2.6           # map centre and height (2:1 Mollweide)
LAMBDA = 1.6                      # wavelength
AMP = 1.0                         # field amplitude
DEPTH = np.array([-0.55, -0.42])  # screen direction of the perpendicular field
N_ARROWS_PER_LAMBDA = 8
N_LAMBDA_FLIGHT = 7               # flight length in wavelengths (seamless loop)
X_START = -7.2                    # photon position at the start of the loop
T_FLIGHT = 4.0

RED, BLUE, ORANGE, GREY = "#C0392B", "#3B6FB6", "#C2560A", "#9A9A9A"
SLIDE_BG = "#faf7f0"


def tint(colour: str, alpha: float) -> str:
    c = np.array([int(colour[i:i + 2], 16) for i in (1, 3, 5)], float)
    b = np.array([int(SLIDE_BG[i:i + 2], 16) for i in (1, 3, 5)], float)
    return "#" + "".join(f"{v:02x}" for v in np.round(alpha * c + (1 - alpha) * b).astype(int))


def phase(x: np.ndarray, xp: float) -> np.ndarray:
    return np.sin(2 * np.pi * (x - xp) / LAMBDA)


def arrow_xs(xp: float) -> np.ndarray:
    """Arrow feet ride with the wave: fixed phases relative to the photon."""
    step = LAMBDA / N_ARROWS_PER_LAMBDA
    n = np.arange(np.floor((X_LEFT - xp) / step), np.ceil((MAP_X - xp) / step) + 1)
    xs = xp + step * n
    return xs[(xs >= X_LEFT) & (xs <= MAP_X)]


def field_points(xp: float):
    """Curves and arrow segments of both fields: [(colour, curve_xy, [(foot, tip)])]."""
    x = np.linspace(X_LEFT, MAP_X, 500)
    f = AMP * phase(x, xp)
    xs = arrow_xs(xp)
    fa = AMP * phase(xs, xp)
    red = (RED, np.column_stack([x, f]),
           [((xa, 0.0), (xa, ya)) for xa, ya in zip(xs, fa) if abs(ya) > 0.12])
    blue = (BLUE, np.column_stack([x + f * DEPTH[0], f * DEPTH[1]]),
            [((xa, 0.0), (xa + ya * DEPTH[0], ya * DEPTH[1]))
             for xa, ya in zip(xs, fa) if abs(ya) > 0.18])
    return [blue, red]


X_END = X_START + N_LAMBDA_FLIGHT * LAMBDA


class PolarisedPhoton(Scene):
    def construct(self):
        xp = ValueTracker(X_START)
        axis = Line([X_LEFT, 0, 0], [MAP_X, 0, 0], color=GREY, stroke_width=2)

        def waves():
            g = VGroup()
            for colour, curve, arrows in field_points(xp.get_value()):
                c = VMobject(stroke_color=colour, stroke_width=4)
                c.set_points_smoothly([[px, py, 0] for px, py in curve])
                g.add(c)
                for (x0, y0), (x1, y1) in arrows:
                    g.add(Arrow([x0, y0, 0], [x1, y1, 0], buff=0, color=colour,
                                stroke_width=3, tip_length=0.14,
                                max_tip_length_to_length_ratio=0.4,
                                max_stroke_width_to_length_ratio=20))
            return g

        def photon():
            x = xp.get_value()
            return VGroup(
                Dot([x, 0, 0], radius=0.34, color=tint(ORANGE, 0.18)),
                Dot([x, 0, 0], radius=0.24, color=tint(ORANGE, 0.40)),
                Dot([x, 0, 0], radius=0.15, color=ORANGE),
            )

        self.add(axis, always_redraw(waves), always_redraw(photon))
        # a whole number of wavelengths: the last frame runs into the first
        self.play(xp.animate.set_value(X_END), run_time=T_FLIGHT, rate_func=linear)


def write_last_svg(path: Path) -> None:
    """Mid-flight frame, vector, transparent, with the map (for the PDF)."""
    W, H = config.frame_width, config.frame_height
    xp = 0.0
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
             f'viewBox="{-W / 2} {-H / 2} {W} {H}" width="1920" height="480">']
    parts.append(f'<line x1="{X_LEFT}" y1="0" x2="{MAP_X}" y2="0" stroke="{GREY}" stroke-width="0.03"/>')
    for colour, curve, arrows in field_points(xp):
        pts = " ".join(f"{px:.3f},{-py:.3f}" for px, py in curve)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="0.06"/>')
        for (x0, y0), (x1, y1) in arrows:
            d = np.array([x1 - x0, y1 - y0])
            u = d / np.linalg.norm(d)
            base = np.array([x1, y1]) - 0.14 * u
            n = np.array([-u[1], u[0]]) * 0.07
            parts.append(f'<line x1="{x0:.3f}" y1="{-y0:.3f}" x2="{base[0]:.3f}" y2="{-base[1]:.3f}" '
                         f'stroke="{colour}" stroke-width="0.045"/>')
            parts.append(f'<polygon points="{base[0] + n[0]:.3f},{-(base[1] + n[1]):.3f} '
                         f'{base[0] - n[0]:.3f},{-(base[1] - n[1]):.3f} {x1:.3f},{-y1:.3f}" '
                         f'fill="{colour}"/>')
    for r, c in ((0.34, tint(ORANGE, 0.18)), (0.24, tint(ORANGE, 0.40)), (0.15, ORANGE)):
        parts.append(f'<circle cx="{xp}" cy="0" r="{r}" fill="{c}"/>')
    from PIL import Image
    im = Image.open(MAP)
    im.thumbnail((800, 800))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    mw = MAP_H * im.size[0] / im.size[1]
    parts.append(f'<image x="{MAP_X - mw / 2}" y="{-MAP_H / 2}" width="{mw}" height="{MAP_H}" '
                 f'xlink:href="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def render(here: Path, file: str, scene: str, out_stem: str) -> None:
    media = here / ".cache" / "manim"
    subprocess.run([sys.executable, "-m", "manim", "-qh", "-t", "--format=gif",
                    "--frame_rate", "30", "-r", "1920,480",
                    "--media_dir", str(media), file, scene],
                   cwd=here, check=True)
    gif = max(media.rglob(f"{scene}*.gif"), key=lambda p: p.stat().st_mtime)
    (here / f"{out_stem}.gif").write_bytes(gif.read_bytes())
    write_last_svg(here / f"{out_stem}_last.svg")
    print(f"wrote {out_stem}.gif ({gif.stat().st_size / 1e6:.1f} MB) and {out_stem}_last.svg")
    W, H = config.frame_width, config.frame_height
    print(f"map box for the slide CSS: left {100 * (MAP_X - MAP_H + W / 2) / W:.2f}%  "
          f"top {100 * (H / 2 - MAP_H / 2) / H:.2f}%  width {100 * 2 * MAP_H / W:.2f}%  "
          f"height {100 * MAP_H / H:.2f}%")


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parents[2]))
    from _common import skip_if_built

    skip_if_built(HERE, "polarised_photon.gif")
    render(HERE, Path(__file__).name, "PolarisedPhoton", "polarised_photon")
