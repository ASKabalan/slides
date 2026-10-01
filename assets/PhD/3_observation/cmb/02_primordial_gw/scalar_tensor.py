#!/usr/bin/env python3
# ENV: manim
# ─────────────────────────────────────────────────────────────
# Render with:
#   manim -qh -t --format=gif --frame_rate 30 -r 1920,1080 scalar_tensor.py ScalarTensor
# or run this file (it renders and copies scalar_tensor.gif beside itself, and
# writes the maximally deformed state as a vector scalar_tensor_last.svg):
#   uv run --project ../.. python scalar_tensor.py
# Manim Community v0.21.x
# ─────────────────────────────────────────────────────────────
"""
Scalar against tensor metric perturbations, on a ring of 14 test particles.

Left, scalar (gold): the wave travels in the plane (arrow); the ring breathes and
is pushed back and forth along the propagation direction, so the enclosed area
pulses. Right, tensor (indigo): the wave travels towards the viewer (circled
dot); the plus polarisation stretches one axis and squeezes the other, area
preserved (x' -> x'(1+e), y' -> y'/(1+e)).

Two full periods of both at constant speed; the phase ends where it began, so
the GIF loops without a seam. (A plus-to-cross turn and a spin-2 rotation beat
were tried and dropped: the tensor ring then seemed to move on its own.)

Transparent GIF, so alpha is 1-bit: the area fills (12 % of the dot colour) are
pre-blended onto the slide colour (#faf7f0) as opaque tints.
No text: the slide carries it.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
from manim import (Arrow, Circle, Dot, Polygon, Scene, ValueTracker, VGroup,
                   always_redraw, config, linear)

config.background_opacity = 0.0

# ------------------------------------------------------------------ tunables
A = 0.22            # scalar breathing amplitude
H = 0.28            # tensor strain amplitude
N_DOTS = 14
DOT_R = 0.07
R0 = 1.9            # ring radius
LONG = 0.35         # longitudinal push of the scalar ring (in units of R0 * A)
T1 = 5.0            # duration (s) of the two periods

GOLD, INDIGO, GREY = "#B8791D", "#4A3FB0", "#6B7280"
SLIDE_BG = "#faf7f0"
X_SCALAR, X_TENSOR, Y_RING, Y_MARK = -3.5, 3.5, 0.35, -3.05


def tint(colour: str, alpha: float) -> str:
    """colour at `alpha` over the slide background, as an opaque hex."""
    c = np.array([int(colour[i:i + 2], 16) for i in (1, 3, 5)], float)
    b = np.array([int(SLIDE_BG[i:i + 2], 16) for i in (1, 3, 5)], float)
    m = np.round(alpha * c + (1 - alpha) * b).astype(int)
    return "#" + "".join(f"{v:02x}" for v in m)


THETA = 2 * np.pi * np.arange(N_DOTS) / N_DOTS


def scalar_ring(s: float) -> np.ndarray:
    """Radial breathing plus a push along +x (the propagation direction)."""
    r = R0 * (1 + s)
    x = r * np.cos(THETA) + LONG * R0 * s
    y = r * np.sin(THETA)
    return np.column_stack([x, y])


def tensor_ring(e: float, axis: float) -> np.ndarray:
    """Area-preserving plus pattern whose stretch axis is at angle `axis`."""
    p = R0 * np.column_stack([np.cos(THETA), np.sin(THETA)])
    c, s = np.cos(axis), np.sin(axis)
    u = p[:, 0] * c + p[:, 1] * s          # along the stretch axis
    v = -p[:, 0] * s + p[:, 1] * c
    u, v = u * (1 + e), v / (1 + e)
    return np.column_stack([u * c - v * s, u * s + v * c])


def to3(xy: np.ndarray, x0: float) -> np.ndarray:
    return np.column_stack([xy[:, 0] + x0, xy[:, 1] + Y_RING, np.zeros(len(xy))])


class ScalarTensor(Scene):
    def construct(self):
        omega = 2 * (2 * np.pi) / T1            # two periods in T1
        ph = ValueTracker(0.0)                  # wave phase

        def s_now():
            return A * np.sin(ph.get_value())

        def e_now():
            return H * np.sin(ph.get_value())

        def scalar_pts():
            return to3(scalar_ring(s_now()), X_SCALAR)

        def tensor_pts():
            return to3(tensor_ring(e_now(), 0.0), X_TENSOR)

        def area(pts_fn, colour):
            return always_redraw(lambda: Polygon(
                *pts_fn(), stroke_width=0, fill_color=tint(colour, 0.12), fill_opacity=1))

        def dots(pts_fn, colour):
            return always_redraw(lambda: VGroup(
                *[Dot(p, radius=DOT_R, color=colour) for p in pts_fn()]))

        # propagation markers, visible from the first frame
        arrow = Arrow([X_SCALAR - 0.7, Y_MARK, 0], [X_SCALAR + 0.7, Y_MARK, 0],
                      color=GREY, stroke_width=5, buff=0,
                      max_tip_length_to_length_ratio=0.25)
        toward = VGroup(Circle(radius=0.2, color=GREY, stroke_width=5).move_to([X_TENSOR, Y_MARK, 0]),
                        Dot([X_TENSOR, Y_MARK, 0], radius=0.06, color=GREY))

        self.add(area(scalar_pts, GOLD), area(tensor_pts, INDIGO), arrow, toward,
                 dots(scalar_pts, GOLD), dots(tensor_pts, INDIGO))

        # two periods at constant speed; ends at phase 4π = start, so it loops
        self.play(ph.animate.set_value(omega * T1), run_time=T1, rate_func=linear)


def write_last_svg(path: Path) -> None:
    """Maximal deformation (quarter period), vector, transparent: the PDF frame."""
    W, Hh = config.frame_width, config.frame_height
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{-W / 2} {-Hh / 2} {W} {Hh}" '
             f'width="1920" height="1080">']

    def poly(xy, x0, fill):
        pts = " ".join(f"{x + x0:.4f},{-(y + Y_RING):.4f}" for x, y in xy)
        parts.append(f'<polygon points="{pts}" fill="{fill}"/>')

    def dots(xy, x0, colour):
        for x, y in xy:
            parts.append(f'<circle cx="{x + x0:.4f}" cy="{-(y + Y_RING):.4f}" r="{DOT_R}" '
                         f'fill="{colour}"/>')

    s, t = scalar_ring(A), tensor_ring(H, 0.0)
    poly(s, X_SCALAR, tint(GOLD, 0.12))
    poly(t, X_TENSOR, tint(INDIGO, 0.12))
    y = -Y_MARK
    parts.append(f'<line x1="{X_SCALAR - 0.7}" y1="{y}" x2="{X_SCALAR + 0.5}" y2="{y}" '
                 f'stroke="{GREY}" stroke-width="0.07"/>')
    parts.append(f'<polygon points="{X_SCALAR + 0.7},{y} {X_SCALAR + 0.45},{y - 0.14} '
                 f'{X_SCALAR + 0.45},{y + 0.14}" fill="{GREY}"/>')
    parts.append(f'<circle cx="{X_TENSOR}" cy="{y}" r="0.2" fill="none" stroke="{GREY}" '
                 f'stroke-width="0.07"/>')
    parts.append(f'<circle cx="{X_TENSOR}" cy="{y}" r="0.06" fill="{GREY}"/>')
    dots(s, X_SCALAR, GOLD)
    dots(t, X_TENSOR, INDIGO)
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def render(here: Path, file: str, scene: str, out_stem: str) -> None:
    media = here / ".cache" / "manim"
    subprocess.run([sys.executable, "-m", "manim", "-qh", "-t", "--format=gif",
                    "--frame_rate", "30", "-r", "1920,1080",
                    "--media_dir", str(media), file, scene],
                   cwd=here, check=True)
    gif = max(media.rglob(f"{scene}*.gif"), key=lambda p: p.stat().st_mtime)
    (here / f"{out_stem}.gif").write_bytes(gif.read_bytes())
    write_last_svg(here / f"{out_stem}_last.svg")
    print(f"wrote {out_stem}.gif ({gif.stat().st_size / 1e6:.1f} MB) and {out_stem}_last.svg")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parents[2]))
    from _common import skip_if_built

    skip_if_built(here, "scalar_tensor.gif")
    render(here, Path(__file__).name, "ScalarTensor", "scalar_tensor")
