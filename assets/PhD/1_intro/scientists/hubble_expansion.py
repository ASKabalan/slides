#!/usr/bin/env python3
# ENV: manim
# ─────────────────────────────────────────────────────────────
# Render with:
#   manim -qh -t --format=gif --frame_rate 30 -r 1080,1080 hubble_expansion.py HubbleExpansion
# or run this file (it renders and copies hubble_expansion.gif beside itself,
# and writes the last frame as a vector hubble_expansion_last.svg):
#   uv run --project ../.. python hubble_expansion.py
# Manim Community v0.21.x
# ─────────────────────────────────────────────────────────────
"""
Hubble expansion seen from one galaxy, inside a circular window.

Galaxies sit at fixed comoving positions drawn from a clustered point process
(Thomas process: Gaussian clumps on a Poisson background), with a spread of
sizes. Screen position = a(t) x comoving position about the observer, the red
galaxy at the centre, which stays put: every other galaxy recedes from it with
a speed proportional to its distance (v = H d). Sizes never change; only the
separations grow.

The field is seen through a circle: a galaxy is shown while its centre is
inside the window. a(t) runs from 1.0 to 2.2 on an accelerating ease, holds,
and the GIF restarts from a = 1.

Transparent GIF: its alpha is 1-bit, so there is no fade at the rim and no
cross-fade at the loop (either would dither). The last frame is also written
as a transparent SVG straight from the same positions, for the PDF export.
No text, axes or title: the slide carries them.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
from manim import Circle, Dot, Scene, ValueTracker, VGroup, config, linear

config.background_opacity = 0.0
config.frame_height = 8.0
config.frame_width = 8.0

INDIGO = "#4a3fb0"
RED = "#C0392B"
RING = "#d6d3de"
HALO = "#e3aaa3"

A0, A1 = 1.0, 2.2
GROW, HOLD = 3.5, 0.5          # seconds: expansion, then hold before the loop
R_WIN = 3.7                    # window radius (frame units)
SEED = 1929


def scale_factor(u: float) -> float:
    """Accelerating ease: starts slow, speeds up."""
    return A0 + (A1 - A0) * u ** 1.6


def clustered_field(rng: np.random.Generator, radius: float):
    """Thomas process in a disc: clumps of galaxies plus a sparse background."""
    pts = []
    n_clumps = 30
    for _ in range(n_clumps):
        r, t = radius * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
        c = np.array([r * np.cos(t), r * np.sin(t)])
        k = rng.poisson(12)
        pts.append(c + rng.normal(0, 0.28, (k, 2)))
    n_bg = 160
    r, t = radius * np.sqrt(rng.uniform(size=n_bg)), rng.uniform(0, 2 * np.pi, n_bg)
    pts.append(np.column_stack([r * np.cos(t), r * np.sin(t)]))
    pts = np.vstack(pts)
    d = np.hypot(*pts.T)
    pts = pts[(d < radius) & (d > 0.35)]        # keep clear of the observer
    sizes = np.clip(rng.lognormal(np.log(0.05), 0.35, len(pts)), 0.028, 0.11)
    return np.column_stack([pts, np.zeros(len(pts))]), sizes


def inside(r: np.ndarray) -> np.ndarray:
    return (r < R_WIN - 0.05).astype(float)


class HubbleExpansion(Scene):
    def construct(self):
        rng = np.random.default_rng(SEED)
        comoving, sizes = clustered_field(rng, R_WIN / A0 + 0.2)

        a = ValueTracker(A0)

        def place(group, scale):
            pos = scale * comoving
            op = inside(np.hypot(pos[:, 0], pos[:, 1]))
            for dot, p, o in zip(group, pos, op):
                dot.move_to(p)
                dot.set_fill(opacity=o)

        moving = VGroup(*[Dot(radius=s, color=INDIGO) for s in sizes])
        moving.add_updater(lambda g: place(g, a.get_value()))

        ring = Circle(radius=R_WIN, color=RING, stroke_width=3)
        observer = Dot(radius=0.11, color=RED)
        halo = Circle(radius=0.2, color=HALO, stroke_width=2.5)  # solid: GIF alpha is 1-bit

        place(moving, A0)
        self.add(ring, moving, halo, observer)

        u = ValueTracker(0.0)
        a.add_updater(lambda m: m.set_value(scale_factor(u.get_value())))
        self.add(a)
        self.play(u.animate.set_value(1.0), run_time=GROW, rate_func=linear)
        self.wait(HOLD)


def write_last_svg(path: Path) -> None:
    """The final frame (a = A1) as a transparent vector SVG, same geometry."""
    rng = np.random.default_rng(SEED)
    comoving, sizes = clustered_field(rng, R_WIN / A0 + 0.2)
    pos = A1 * comoving
    keep = inside(np.hypot(pos[:, 0], pos[:, 1])) > 0
    h = config.frame_height / 2
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{-h} {-h} {2 * h} {2 * h}" '
             f'width="1080" height="1080">',
             f'<circle cx="0" cy="0" r="{R_WIN}" fill="none" stroke="{RING}" stroke-width="0.022"/>']
    for (x, y, _), s in zip(pos[keep], sizes[keep]):
        parts.append(f'<circle cx="{x:.4f}" cy="{-y:.4f}" r="{s:.4f}" fill="{INDIGO}"/>')
    parts.append(f'<circle cx="0" cy="0" r="0.2" fill="none" stroke="{HALO}" '
                 f'stroke-width="0.019"/>')
    parts.append(f'<circle cx="0" cy="0" r="0.11" fill="{RED}"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def render(here: Path, file: str, scene: str, out_stem: str) -> None:
    """Render one scene to a transparent GIF; write its last frame as SVG."""
    media = here / ".cache" / "manim"
    subprocess.run([sys.executable, "-m", "manim", "-qh", "-t", "--format=gif",
                    "--frame_rate", "30", "-r", "1080,1080",
                    "--media_dir", str(media), file, scene],
                   cwd=here, check=True)
    gif = max(media.rglob(f"{scene}*.gif"), key=lambda p: p.stat().st_mtime)
    (here / f"{out_stem}.gif").write_bytes(gif.read_bytes())
    write_last_svg(here / f"{out_stem}_last.svg")
    print(f"wrote {out_stem}.gif ({gif.stat().st_size / 1e6:.1f} MB) and {out_stem}_last.svg")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parents[1]))
    from _common import skip_if_built

    skip_if_built(here, "hubble_expansion.gif")
    render(here, Path(__file__).name, "HubbleExpansion", "hubble_expansion")
