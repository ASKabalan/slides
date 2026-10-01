#!/usr/bin/env python3
# ENV: manim
# ─────────────────────────────────────────────────────────────
# Render with:
#   manim -qh --frame_rate 30 --format=mp4 halo_exchange.py HaloExchange
# or run this file (it renders and copies halo_exchange.mp4 and its last frame,
# halo_exchange_last.png, beside itself):
#   uv run --project ../../.. python halo_exchange.py
# Manim Community v0.21.x
# ─────────────────────────────────────────────────────────────
"""
The halo exchange of jaxDecomp, at cell level, for a slab split between two devices: the three
stages of the thesis figure (chap6/halo_exchange_tikz.pdf) in motion.

A particle near the split paints a cloud one cell wide. Without an exchange the share of the
cloud that falls on the neighbouring device is lost (it flashes red and fades). With the halo,
each device is padded with a ghost cell past its boundary, the particle paints its whole cloud
into the padded grid, and the ghost cells are sent to the neighbour and added into its interior,
so the deposit matches a single device. No text on the frames; the slide holds it.
"""

import subprocess
import sys
from pathlib import Path

from manim import (DOWN, LEFT, RIGHT, UP, CurvedArrow, DashedVMobject, Dot, FadeIn, FadeOut,
                   Line, Rectangle, Scene, Text, VGroup, config, smooth)

config.background_color = "#faf7f0"
config.pixel_width, config.pixel_height = 1600, 680
config.frame_width, config.frame_height = 13.6, 13.6 * 680 / 1600

INK = "#2E2E2E"
DEV0 = "#DCE6F7"       # device 0 cells
DEV1 = "#FBE1C4"       # device 1 cells
MASS = "#521463"       # the deposited mass, deck purple
LOST = "#C0392B"
GHOST = "#EDEDED"
W, H = 1.2, 1.2        # cell size
N = 4                  # cells per device


def cell(x, fill):
    return Rectangle(width=W, height=H, stroke_color="#7A7A7A", stroke_width=2,
                     fill_color=fill, fill_opacity=1).move_to([x, 0, 0])


class HaloExchange(Scene):
    @staticmethod
    def ghost(at):
        box = Rectangle(width=W, height=H, stroke_width=0, fill_color=GHOST, fill_opacity=1)
        edge = DashedVMobject(Rectangle(width=W, height=H, stroke_color="#7A7A7A", stroke_width=2),
                              num_dashes=24)
        return VGroup(box, edge).move_to(at)

    def construct(self):
        xs0 = [(-N + i + 0.5) * W for i in range(N)]      # device 0, left of the split
        xs1 = [(i + 0.5) * W for i in range(N)]           # device 1, right of the split
        dev0 = VGroup(*[cell(x, DEV0) for x in xs0])
        dev1 = VGroup(*[cell(x, DEV1) for x in xs1])
        split = Line([0, -H / 2 - 0.15, 0], [0, H / 2 + 0.15, 0], color=INK, stroke_width=7)
        lab0 = Text("device 0", color=INK, font="DejaVu Sans", font_size=30).next_to(dev0, UP, buff=0.35)
        lab1 = Text("device 1", color=INK, font="DejaVu Sans", font_size=30).next_to(dev1, UP, buff=0.35)
        self.add(dev0, dev1, split, lab0, lab1)

        # the particle, a quarter cell left of the split, and its one-cell cloud
        px = -0.25 * W
        dot = Dot([px, 0, 0], radius=0.09, color=MASS)
        cloud = DashedVMobject(Rectangle(width=W, height=0.8 * H, stroke_color=MASS,
                                         stroke_width=4).move_to([px, 0, 0]), num_dashes=28)
        self.play(FadeIn(dot, scale=2), FadeIn(cloud), run_time=0.9)
        self.wait(0.4)

        # (a) no exchange: the share on device 1 is lost
        share_in = Rectangle(width=0.75 * W, height=0.8 * H, stroke_width=0, fill_color=MASS,
                             fill_opacity=0.55).move_to([px - 0.125 * W, 0, 0])
        share_out = Rectangle(width=0.25 * W, height=0.8 * H, stroke_width=0, fill_color=LOST,
                              fill_opacity=0.9).move_to([0.125 * W, 0, 0])
        self.play(FadeIn(share_in), FadeIn(share_out), run_time=0.8)
        self.play(share_out.animate.set_opacity(0.15).shift(0.35 * DOWN), run_time=1.1)
        self.play(FadeOut(share_out), FadeOut(share_in), run_time=0.5)
        self.wait(0.3)

        # (b) pad: the devices part and each gains a ghost cell past its boundary
        gap = 2 * W + 0.5         # room for both ghost cells, with a clear gap between them
        self.play(VGroup(dev0, lab0, dot, cloud).animate.shift(gap / 2 * LEFT),
                  VGroup(dev1, lab1).animate.shift(gap / 2 * RIGHT),
                  FadeOut(split), run_time=1.2, rate_func=smooth)
        # the ghost cells sit just past each boundary
        ghost0 = self.ghost([-gap / 2 + 0.5 * W, 0, 0])
        ghost1 = self.ghost([gap / 2 - 0.5 * W, 0, 0])
        self.play(FadeIn(ghost0, shift=0.2 * RIGHT), FadeIn(ghost1, shift=0.2 * LEFT), run_time=0.8)

        # paint: the whole cloud lands in device 0, a quarter of it in its ghost cell
        pxn = px - gap / 2
        paint_in = Rectangle(width=0.75 * W, height=0.8 * H, stroke_width=0, fill_color=MASS,
                             fill_opacity=0.55).move_to([pxn - 0.125 * W, 0, 0])
        paint_ghost = Rectangle(width=0.25 * W, height=0.8 * H, stroke_width=0, fill_color=MASS,
                                fill_opacity=0.55).move_to([-gap / 2 + 0.125 * W, 0, 0])
        self.play(FadeIn(paint_in), FadeIn(paint_ghost), run_time=1.0)
        self.wait(0.4)

        # (c) exchange: the ghost content travels to device 1 and is added into its first cell
        target = [gap / 2 + 0.125 * W, 0, 0]
        arrow = CurvedArrow([-gap / 2 + 0.5 * W, H / 2 + 0.1, 0], [gap / 2 + 0.2 * W, H / 2 + 0.1, 0],
                            angle=-1.2, color=INK, stroke_width=5)
        self.play(FadeIn(arrow), run_time=0.5)
        self.play(paint_ghost.animate.move_to(target), run_time=1.3, rate_func=smooth)
        self.play(FadeOut(arrow), FadeOut(ghost0), FadeOut(ghost1), run_time=0.6)

        # reduce: the devices rejoin, and the deposit matches a single device
        self.play(VGroup(dev0, lab0, dot, cloud, paint_in).animate.shift(gap / 2 * RIGHT),
                  VGroup(dev1, lab1, paint_ghost).animate.shift(gap / 2 * LEFT),
                  run_time=1.2, rate_func=smooth)
        self.play(FadeIn(split), run_time=0.4)
        self.wait(2.0)


def render(here: Path, file: str, scene: str, out_stem: str) -> None:
    """Render one scene to MP4 and keep its last frame as a PNG for print."""
    media = here.parent / ".cache" / "manim"
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
    sys.path.insert(0, str(here.parents[2]))
    from _common import skip_if_built

    skip_if_built(here, "halo_exchange.mp4")
    render(here, Path(__file__).name, "HaloExchange", "halo_exchange")
