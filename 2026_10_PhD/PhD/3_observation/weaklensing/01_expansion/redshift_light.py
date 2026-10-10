#!/usr/bin/env python3
# ENV: manim
# ─────────────────────────────────────────────────────────────
# Render with:
#   manim -qh -t --format=gif --frame_rate 30 -r 1440,1080 redshift_light.py RedshiftLight
# or run this file (it renders and copies redshift_light.gif beside itself, and
# writes a late frame as a vector redshift_light_last.svg for the PDF):
#   uv run --project ../../.. python redshift_light.py
# Manim Community v0.21.x
# ─────────────────────────────────────────────────────────────
"""
Light stretched by the expansion, for "The expanding Universe and dark energy".

A faint comoving grid, an emitting galaxy (left) and an observer (right) sit at
fixed comoving positions; everything is drawn at a(t) x comoving position. The
light between them is a wave with a fixed number of periods, so its wavelength
grows with a(t) as the grid expands, and its colour runs from blue to red:
1 + z = a_obs / a_emit. It does not travel; it only stretches. Plays once. No text.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
from manim import (Circle, Dot, Line, Scene, ValueTracker, VGroup, VMobject,
                   always_redraw, config, linear, interpolate_color, ManimColor)

config.background_opacity = 0.0
config.frame_width = 10.0
config.frame_height = 7.5

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
A0, A1 = 1.0, 2.2
X_EMIT, X_OBS = -1.8, 1.8          # comoving positions
N_PERIODS, AMP = 7, 0.42
T = 4.0
BLUE, RED, GRID, INK = "#3B6FB6", "#C0392B", "#D9D4C9", "#2E2E2E"


def a_of(u):
    return A0 + (A1 - A0) * u


def wave(u, n=600):
    """The whole wave from emitter to observer: N_PERIODS periods, stretched by a."""
    a = a_of(u)
    xe, xo = X_EMIT * a + 0.3, X_OBS * a - 0.3
    x = np.linspace(xe, xo, n)
    y = AMP * np.sin(2 * np.pi * N_PERIODS * (x - xe) / (xo - xe))
    col = interpolate_color(ManimColor(BLUE), ManimColor(RED), u)
    return x, y, xo, col


class RedshiftLight(Scene):
    def construct(self):
        u = ValueTracker(0.0)

        def grid():
            a = a_of(u.get_value())
            g = VGroup()
            for i in range(-6, 7):
                x = 0.8 * i * a
                if abs(x) < 5.2:
                    g.add(Line([x, -3.2, 0], [x, 3.2, 0], color=GRID, stroke_width=2))
            for j in range(-3, 4):
                g.add(Line([-5, 0.9 * j * a, 0], [5, 0.9 * j * a, 0], color=GRID, stroke_width=2)
                      if abs(0.9 * j * a) < 3.4 else VGroup())
            return g

        def light():
            x, y, xp, col = wave(u.get_value())
            if len(x) < 2 or x[-1] - x[0] < 1e-3:
                return VGroup()
            c = VMobject(stroke_color=col, stroke_width=7)
            c.set_points_smoothly([[a, b, 0] for a, b in zip(x, y)])
            return c

        def ends():
            a = a_of(u.get_value())
            gal = VGroup(Dot([X_EMIT * a, 0, 0], radius=0.26, color="#E8A33D"),
                         Dot([X_EMIT * a, 0, 0], radius=0.14, color="#FFF3D6"))
            obs = VGroup(Circle(radius=0.26, color=INK, stroke_width=5).move_to([X_OBS * a, 0, 0]),
                         Dot([X_OBS * a, 0, 0], radius=0.1, color=INK))
            return VGroup(gal, obs)

        self.add(always_redraw(grid), always_redraw(light), always_redraw(ends))
        self.wait(0.4)
        self.play(u.animate.set_value(1.0), run_time=T, rate_func=linear)
        self.wait(0.8)


def write_last_svg(path: Path, u: float = 1.0) -> None:
    W, H = config.frame_width, config.frame_height
    a = a_of(u)
    f = lambda x, y: f"{x:.3f},{-y:.3f}"  # noqa: E731
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{-W / 2} {-H / 2} {W} {H}" width="1440" height="1080">']
    for i in range(-6, 7):
        x = 0.8 * i * a
        if abs(x) < 5.2:
            p.append(f'<line x1="{x:.3f}" y1="-3.2" x2="{x:.3f}" y2="3.2" stroke="{GRID}" stroke-width="0.02"/>')
    for j in range(-3, 4):
        y = 0.9 * j * a
        if abs(y) < 3.4:
            p.append(f'<line x1="-5" y1="{-y:.3f}" x2="5" y2="{-y:.3f}" stroke="{GRID}" stroke-width="0.02"/>')
    x, y, xp, col = wave(u)
    p.append(f'<polyline points="{" ".join(f(a_, b) for a_, b in zip(x, y))}" fill="none" '
             f'stroke="{col.to_hex()}" stroke-width="0.07"/>')
    p.append(f'<circle cx="{X_EMIT * a:.3f}" cy="0" r="0.26" fill="#E8A33D"/>'
             f'<circle cx="{X_EMIT * a:.3f}" cy="0" r="0.14" fill="#FFF3D6"/>')
    p.append(f'<circle cx="{X_OBS * a:.3f}" cy="0" r="0.26" fill="none" stroke="{INK}" stroke-width="0.05"/>'
             f'<circle cx="{X_OBS * a:.3f}" cy="0" r="0.1" fill="{INK}"/>')
    p.append("</svg>")
    path.write_text("\n".join(p))


def play_once(src: Path, dst: Path) -> None:
    """Re-save the GIF without the looping extension: it plays once."""
    from PIL import Image, ImageSequence

    im = Image.open(src)
    frames = [f.copy() for f in ImageSequence.Iterator(im)]
    # The renderer leaves the chroma-key green of the first frame in a palette slot of its own, which
    # the browser draws opaque: move those pixels to the transparent slot, or the GIF flashes green
    # each time it restarts.
    f0 = frames[0]
    rgb = f0.getpalette()[:768]
    green = [k for k in range(len(rgb) // 3) if rgb[3 * k:3 * k + 3] == [0, 255, 0]]
    if green and f0.mode == "P":
        tidx = im.info.get("transparency", 0)
        f0.putdata([tidx if v in green else v for v in f0.getdata()])
    for f in frames:
        f.info.pop("loop", None)       # Pillow would copy the loop extension back
    # each frame keeps its own duration: im.info holds only the last frame's (the final hold), and
    # passing that one value to every frame slows the whole GIF to the hold
    frames[0].save(dst, save_all=True, append_images=frames[1:],
                   duration=[f.info.get("duration", 33) for f in frames], disposal=2,
                   transparency=im.info.get("transparency", 0), optimize=False)


def render(here: Path) -> None:
    media = here / ".cache" / "manim"
    subprocess.run([sys.executable, "-m", "manim", "-qh", "-t", "--format=gif", "--frame_rate", "30",
                    "-r", "1440,1080", "--media_dir", str(media), Path(__file__).name, "RedshiftLight"],
                   cwd=here, check=True)
    gif = max(media.rglob("RedshiftLight*.gif"), key=lambda q: q.stat().st_mtime)
    play_once(gif, here / "redshift_light.gif")
    write_last_svg(here / "redshift_light_last.svg")
    print(f"wrote redshift_light.gif ({gif.stat().st_size / 1e6:.1f} MB) and redshift_light_last.svg")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    from _common import skip_if_built

    skip_if_built(HERE, "redshift_light.gif")
    render(HERE)
