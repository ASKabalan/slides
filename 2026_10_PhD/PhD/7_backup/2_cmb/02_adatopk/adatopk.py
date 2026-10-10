#!/usr/bin/env python3
# ENV: manim
# ─────────────────────────────────────────────────────────────
# Render with:
#   manim -qh --frame_rate 30 --format=mp4 adatopk.py AdaTopK
# or run this file (it renders and copies adatopk.mp4 and its last frame,
# adatopk_last.png, beside itself):
#   uv run --project .. python adatopk.py
# Manim Community v0.20.x
# ─────────────────────────────────────────────────────────────
"""
How AdaTopK handles bounds: parameters freeze on a bound, and are released
one at a time when the gradient points back into the allowed range.

Eight parameters in the internal space [0, 1] (the bounds mapped to 0 and 1,
as in AdaTopK), each drawn as a vertical slider, minimising a separable
quadratic whose optimum lies inside the bounds for five of them and outside for
three. A first, overshooting step pins five on a bound. Each pinned parameter
then shows its release score: an arrow along the descent direction -g,
orange when it points back into [0, 1] (positive score) and grey when it points
out. With k = 1 (ADABK0, the setting of the paper) the single strongest
positive score is released per iteration; the three whose optimum is truly out
of bounds never are.

The rule matches cadre/active_set.py: score_i = p_i * g_i with p = -1 at the
lower bound and +1 at the upper bound, released when positive.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, LEFT, ORIGIN, RIGHT, UP, Arrow, Circumscribe, Create,
                   Dot, FadeIn, FadeOut, Line, MathTex, Scene, VGroup, config,
                   smooth)

config.background_color = "#FAF7F0"     # the slide background

INK = "#2E2E2E"
GREY = "#A0A0A0"
FREE = "#521463"      # deck purple: a free parameter
PINNED = "#8A8A8A"    # frozen on a bound
INWARD = "#C2560A"    # keyword orange: a positive release score

# The problem: f(y) = sum_i a_i (y_i - c_i)^2 / 2 on [0, 1]^8.
C = np.array([0.62, 1.30, 0.38, -0.25, 0.78, 0.52, 1.15, 0.18])
A = np.array([1.0, 1.0, 1.0, 1.0, 1.6, 1.0, 1.0, 1.25])
Y0 = np.array([0.20, 0.72, 0.86, 0.42, 0.06, 0.93, 0.47, 0.62])
LR_FIRST = np.array([0.9, 1.9, 0.9, 1.9, 1.35, 0.9, 1.9, 1.45])  # the overshoot

H = 5.2               # slider height on screen
X0, DX = -4.9, 1.25   # slider spacing


def sy(v: float) -> float:
    """Internal value in [0, 1] to screen y."""
    return -H / 2 + H * float(np.clip(v, 0.0, 1.0))


def sx(i: int) -> float:
    return X0 + DX * i


class AdaTopK(Scene):
    def construct(self):
        n = len(C)
        rails = VGroup()
        for i in range(n):
            rails.add(Line([sx(i), sy(0), 0], [sx(i), sy(1), 0], color=GREY,
                           stroke_width=3))
            for v in (0.0, 1.0):
                rails.add(Line([sx(i) - 0.22, sy(v), 0], [sx(i) + 0.22, sy(v), 0],
                               color=INK, stroke_width=5))
        lab_up = MathTex(r"\text{upper bound}", color=INK, font_size=30)
        lab_lo = MathTex(r"\text{lower bound}", color=INK, font_size=30)
        lab_up.next_to([sx(0) - 0.3, sy(1), 0], LEFT, buff=0.25)
        lab_lo.next_to([sx(0) - 0.3, sy(0), 0], LEFT, buff=0.25)
        names = VGroup(*[MathTex(rf"y_{{{i + 1}}}", color=INK, font_size=30)
                         .move_to([sx(i) - 0.12, sy(0) - 0.62, 0]) for i in range(n)])
        # One translation centres the whole figure; the dots use the same one,
        # so they sit exactly on the rails and bound markers.
        grp = VGroup(rails, lab_up, lab_lo, names)
        offset = ORIGIN + DOWN * 0.1 - grp.get_center()
        grp.shift(offset)

        def pos(i, v):
            return np.array([sx(i), sy(v), 0.0]) + offset

        self.play(Create(rails), FadeIn(lab_up), FadeIn(lab_lo), FadeIn(names),
                  run_time=1.2)

        y = Y0.copy()
        dots = [Dot(pos(i, y[i]), radius=0.15, color=FREE) for i in range(n)]
        self.play(*[FadeIn(d, scale=0.5) for d in dots], run_time=0.7)
        self.wait(0.4)

        # One overshooting step, clamped to the box: five parameters pin.
        y1 = np.clip(y - LR_FIRST * A * (y - C), 0.0, 1.0)
        pinned = (y1 <= 0.0) | (y1 >= 1.0)
        pivot = np.where(y1 <= 0.0, -1, np.where(y1 >= 1.0, 1, 0))
        self.play(*[dots[i].animate.move_to(pos(i, y1[i])) for i in range(n)],
                  run_time=1.3, rate_func=smooth)
        self.play(*[dots[i].animate.set_color(PINNED) for i in range(n) if pinned[i]],
                  run_time=0.5)
        y = y1
        self.wait(0.5)

        # Free parameters settle; pinned ones stay put.
        self.play(*[dots[i].animate.move_to(pos(i, C[i])) for i in range(n)
                    if not pinned[i]], run_time=1.0)
        y = np.where(pinned, y, np.clip(C, 0, 1))
        self.wait(0.3)

        # Release loop: score = p * g, top-1 positive score is released.
        while True:
            g = A * (y - C)
            score = np.where(pinned, pivot * g, -np.inf)
            arrows = VGroup()
            for i in range(n):
                if not pinned[i]:
                    continue
                direction = -np.sign(g[i])              # descent direction
                length = 0.35 + 1.1 * min(abs(g[i]) / 0.4, 1.0)
                inward = score[i] > 0
                start = pos(i, y[i]) + np.array([0.33, 0.0, 0.0])
                arrows.add(Arrow(start, start + np.array([0, direction * length, 0]),
                                 buff=0, color=INWARD if inward else GREY,
                                 stroke_width=6, max_tip_length_to_length_ratio=0.3))
            self.play(FadeIn(arrows), run_time=0.6)
            if not np.any(score > 0):
                self.wait(1.2)
                break
            k = int(np.argmax(score))
            self.play(Circumscribe(dots[k], color=INWARD, time_width=1.2), run_time=1.0)
            pinned[k] = False
            pivot[k] = 0
            y[k] = C[k]
            self.play(FadeOut(arrows), dots[k].animate.set_color(FREE), run_time=0.5)
            self.play(dots[k].animate.move_to(pos(k, y[k])), run_time=1.0)
            self.wait(0.3)
        self.wait(1.5)


def render(here: Path, file: str, scene: str, out_stem: str) -> None:
    """Render one scene to MP4 (small and sharp, unlike a 1080p GIF) and keep
    its last frame as a PNG for the printed or PDF version of the deck."""
    media = here / ".cache" / "manim"
    subprocess.run([sys.executable, "-m", "manim", "-qh", "--frame_rate", "30",
                    "--format=mp4", "--media_dir", str(media), file, scene],
                   cwd=here, check=True)
    mp4 = max(media.rglob(f"{scene}.mp4"), key=lambda p: p.stat().st_mtime)
    (here / f"{out_stem}.mp4").write_bytes(mp4.read_bytes())
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-sseof", "-0.1",
                    "-i", str(here / f"{out_stem}.mp4"), "-frames:v", "1",
                    str(here / f"{out_stem}_last.png")], check=True)
    print(f"wrote {out_stem}.mp4 and {out_stem}_last.png")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    ROOT = next(p for p in here.parents if (p / "_common.py").exists())
    sys.path.insert(0, str(ROOT))
    from _common import skip_if_built

    skip_if_built(here, "adatopk.mp4")
    render(here, Path(__file__).name, "AdaTopK", "adatopk")
