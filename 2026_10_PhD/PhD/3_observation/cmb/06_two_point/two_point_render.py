#!/usr/bin/env python3
# ENV: manim
# ─────────────────────────────────────────────────────────────
# Render with:
#   manim -qh -t --format=gif --frame_rate 30 -r 1920,1080 two_point_render.py TwoPoint
# or run this file (it renders, re-saves the GIF to play once, copies it here as
# two_point.gif, and writes the final state of the plot as two_point_last.svg):
#   uv run --project ../.. python two_point_render.py
# Needs two_point_data.py first (.cache/two_point.npz, two_point_sky.png).
# Manim Community v0.21.x
# ─────────────────────────────────────────────────────────────
"""
From pairs of points to the two-point correlation function.

Left: the sky (two_point_sky.png, NOT in the GIF: the slide lays it under the
transparent GIF at the box this script prints). Anchor by anchor, lines join a
point to partners at separations uniform in theta, orange when the two
temperatures have the same sign and blue otherwise. Right: each bar counts the
same-sign minus opposite-sign pairs in its 3-degree bin (orange adds one, blue
removes one; each drawn line also brings ~15 undrawn pairs of the same sky, so
the bars settle), so it grows from zero, high where the sky is correlated (small
theta) and near zero where it is not. This sky's C(theta), scaled to the bars
by one amplitude, is then drawn over them, and the bars fade. Plays once, retimed to
PLAY_S seconds on the slide.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
from manim import (Dot, Line, Polygon, Rectangle, Scene, Text, ValueTracker, VGroup,
                   VMobject, always_redraw, config, linear, smooth, Create, FadeIn, FadeOut)

config.background_opacity = 0.0
HERE = Path(__file__).resolve().parent
PLAY_S = 7.0          # the GIF's length on the slide: the 10.5 s scene is resampled to it at 30 fps
D = dict(np.load(HERE / ".cache" / "two_point.npz"))

# ------------------------------------------------------------------ layout
SKY_C, SKY_S = np.array([-3.45, 0.2]), 1.6       # map centre, Mollweide scale (x in +-2)
PX0, PX1, PY0, PY1 = 1.55, 6.6, -2.4, 2.7        # plot box
TMAX, YMIN, YMAX = 36.0, 0.0, 4500.0
T_PAIRS, T_CONVERGE, T_CURVE = 7.0, 1.6, 1.6

INK, GREY = "#2E2E2E", "#8A8A96"
# Illustrative scale axes (the bars are not re-binned): ell on top, the matching
# angle 180 deg / ell below, log-spaced across the plot, large scales on the left.
ELL_TICKS = (5, 10, 100, 1000)
DEG_LABELS = ("36°", "18°", "1.8°", "0.18°")


def ell_x(ell):
    f = (np.log10(ell) - np.log10(ELL_TICKS[0])) / (np.log10(ELL_TICKS[-1]) - np.log10(ELL_TICKS[0]))
    return PX0 + 0.04 + (PX1 - PX0 - 0.08) * f
POS, NEG, BAR, CURVE = "#C2560A", "#3B6FB6", "#521463", "#C2560A"
FONT = "Lato"

BINS = D["bins"]
NB = len(BINS) - 1
NA, NP = D["prod"].shape
PROD = D["prod"].ravel()
BIN_OF = np.clip(np.digitize(D["sep_deg"].ravel(), BINS) - 1, 0, NB - 1)


def sky(xy):
    return np.array([SKY_C[0] + SKY_S * xy[0], SKY_C[1] + SKY_S * xy[1], 0.0])


def px(theta):
    return PX0 + (PX1 - PX0) * theta / TMAX


def py(v):
    v = np.clip(v, YMIN, YMAX)
    return PY0 + (PY1 - PY0) * (v - YMIN) / (YMAX - YMIN)


SIGN = np.sign(PROD)


H_BIN = np.clip(np.digitize(D["hidden_sep_deg"], BINS) - 1, 0, NB - 1)
H_SIGN = np.sign(D["hidden_prod"])


def net_counts(n):
    """Same-sign minus opposite-sign pairs per bin: the first n drawn pairs plus
    the matching share of the pairs behind them (~15 per drawn line)."""
    m = int(len(H_SIGN) * n / len(PROD))
    return (np.bincount(BIN_OF[:n], weights=SIGN[:n], minlength=NB)
            + np.bincount(H_BIN[:m], weights=H_SIGN[:m], minlength=NB))


FINAL = net_counts(len(PROD))
COUNT_SCALE = 0.9 * (PY1 - PY0) / FINAL.max()
# C(theta) drawn in the same units: one amplitude, least squares on the bin centres
_centres = 0.5 * (BINS[1:] + BINS[:-1])
_c_at = np.interp(_centres, D["theta"], D["exact"])
AMP = float(FINAL @ _c_at / (_c_at @ _c_at))


def x_axis():
    g = VGroup(Line([PX0, PY0, 0], [PX1 + 0.1, PY0, 0], color=INK, stroke_width=3),
               Line([PX0, PY0, 0], [PX0, PY1, 0], color=INK, stroke_width=3),
               Line([PX0, PY1, 0], [PX1 + 0.1, PY1, 0], color=GREY, stroke_width=2))
    for ell, lab in zip(ELL_TICKS, DEG_LABELS):
        x = ell_x(ell)
        g.add(Line([x, PY0, 0], [x, PY0 - 0.1, 0], color=INK, stroke_width=3))
        g.add(Text(lab, font=FONT, color=INK).scale(0.34).move_to([x, PY0 - 0.35, 0]))
        g.add(Line([x, PY1, 0], [x, PY1 + 0.1, 0], color=GREY, stroke_width=2))
        g.add(Text(f"{ell}", font=FONT, color=GREY).scale(0.32).move_to([x, PY1 + 0.32, 0]))
    g.add(Text("separation θ", font=FONT, color=INK).scale(0.4)
          .move_to([(PX0 + PX1) / 2, PY0 - 0.85, 0]))
    g.add(Text("multipole ℓ", font=FONT, color=GREY).scale(0.38)
          .move_to([(PX0 + PX1) / 2, PY1 + 0.75, 0]))
    return g


def y_label(text):
    return Text(text, font=FONT, color=INK).scale(0.38).rotate(np.pi / 2) \
        .move_to([PX0 - 0.45, (PY0 + PY1) / 2, 0])


class TwoPoint(Scene):
    def construct(self):
        k = ValueTracker(0.0)        # pairs drawn so far
        bar_op = ValueTracker(1.0)
        line_op = ValueTracker(1.0)

        def lines():
            n = k.get_value()
            a = min(int(n // NP), NA - 1)
            m = int(np.clip(n - a * NP, 0, NP))
            g = VGroup()
            if n <= 0:
                return g
            p0 = sky(D["anchors_xy"][a])
            for j in range(m):
                q = sky(D["partners_xy"][a, j])
                col = POS if D["prod"][a, j] > 0 else NEG
                g.add(Line(p0, q, color=col, stroke_width=2))
            g.add(Dot(p0, radius=0.07, color=INK))
            return g.set_opacity(line_op.get_value())

        def bars():
            c = np.maximum(net_counts(int(k.get_value())), 0)
            g = VGroup()
            for i in range(NB):
                if c[i] <= 0:
                    continue
                y1 = PY0 + COUNT_SCALE * c[i]
                x0, x1 = px(BINS[i]) + 0.03, px(BINS[i + 1]) - 0.03
                g.add(Polygon([x0, PY0, 0], [x1, PY0, 0], [x1, y1, 0], [x0, y1, 0],
                              stroke_width=0, fill_color=BAR, fill_opacity=0.85 * bar_op.get_value()))
            return g

        count_lbl = y_label("same-sign − opposite-sign pairs")
        self.add(x_axis(), count_lbl, always_redraw(bars), always_redraw(lines))
        # each orange pair adds one to its separation bin, each blue pair removes one
        self.play(k.animate.set_value(NA * NP), run_time=T_PAIRS, rate_func=linear)

        th = D["theta"][D["theta"] >= 0.5]
        cv = np.interp(th, D["theta"], D["exact"])
        curve = VMobject(stroke_color=CURVE, stroke_width=7)
        curve.set_points_smoothly([[px(t), min(PY0 + COUNT_SCALE * AMP * v, PY1), 0]
                                   for t, v in zip(th, cv)])
        # the curve is drawn over the finished bars, then the bars go
        self.play(line_op.animate.set_value(0.0), Create(curve), run_time=T_CURVE)
        self.play(bar_op.animate.set_value(0.0),
                  count_lbl.animate.become(y_label("C(θ) = ⟨ΔT ΔT⟩")), run_time=0.9)
        self.wait(1.0)


def write_last_svg(path: Path) -> None:
    """Final state (axes + curve), vector; the sky is laid by the slide."""
    W, H = config.frame_width, config.frame_height
    f = lambda x, y: f"{x:.3f},{-y:.3f}"  # noqa: E731
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{-W / 2} {-H / 2} {W} {H}" '
             f'width="1920" height="1080" font-family="{FONT}">',
             f'<polyline points="{f(PX1 + 0.1, PY0)} {f(PX0, PY0)} {f(PX0, PY1)}" fill="none" '
             f'stroke="{INK}" stroke-width="0.04"/>']
    parts.append(f'<line x1="{PX0}" y1="{-PY1}" x2="{PX1 + 0.1}" y2="{-PY1}" stroke="{GREY}" '
                 f'stroke-width="0.03"/>')
    for ell, lab in zip(ELL_TICKS, DEG_LABELS):
        x = ell_x(ell)
        parts.append(f'<text x="{x:.3f}" y="{-(PY0 - 0.45):.3f}" font-size="0.3" '
                     f'text-anchor="middle" fill="{INK}">{lab}</text>')
        parts.append(f'<text x="{x:.3f}" y="{-(PY1 + 0.22):.3f}" font-size="0.28" '
                     f'text-anchor="middle" fill="{GREY}">{ell}</text>')
    parts.append(f'<text x="{(PX0 + PX1) / 2:.3f}" y="{-(PY0 - 0.95):.3f}" font-size="0.34" '
                 f'text-anchor="middle" fill="{INK}">separation θ</text>')
    parts.append(f'<text x="{(PX0 + PX1) / 2:.3f}" y="{-(PY1 + 0.65):.3f}" font-size="0.32" '
                 f'text-anchor="middle" fill="{GREY}">multipole ℓ</text>')
    parts.append(f'<text transform="translate({PX0 - 0.45:.3f},{-(PY0 + PY1) / 2:.3f}) rotate(-90)" '
                 f'font-size="0.32" text-anchor="middle" fill="{INK}">C(θ) = ⟨ΔT ΔT⟩</text>')
    th = D["theta"][D["theta"] >= 0.5]
    cv = np.interp(th, D["theta"], D["exact"])
    pts = " ".join(f(px(t), min(PY0 + COUNT_SCALE * AMP * v, PY1)) for t, v in zip(th, cv))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{CURVE}" stroke-width="0.08"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def play_once(src: Path, dst: Path, play_s: float = PLAY_S) -> None:
    """Re-save the GIF without the looping extension (it plays once), resampled to play_s seconds."""
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
    durations = [f.info.get("duration", 33) for f in frames]
    # retime: sample the scene at 30 fps over play_s seconds, frame k showing the source frame on
    # screen at k / n of the scene; durations land on the GIF's 10 ms grid and add up to play_s
    start = np.cumsum([0] + durations[:-1])
    n = int(round(30 * play_s))
    frames = [frames[int(np.searchsorted(start, k * sum(durations) / n, side="right")) - 1] for k in range(n)]
    durations = np.diff(np.round(np.arange(n + 1) * play_s * 100 / n) * 10).astype(int).tolist()
    frames[0].save(dst, save_all=True, append_images=frames[1:],
                   duration=durations, disposal=2,
                   transparency=im.info.get("transparency", 0), optimize=False)


def render(here: Path) -> None:
    media = here / ".cache" / "manim"
    subprocess.run([sys.executable, "-m", "manim", "-qh", "-t", "--format=gif",
                    "--frame_rate", "30", "-r", "1920,1080", "--media_dir", str(media),
                    Path(__file__).name, "TwoPoint"], cwd=here, check=True)
    gif = max(media.rglob("TwoPoint*.gif"), key=lambda p: p.stat().st_mtime)
    play_once(gif, here / "two_point.gif")
    write_last_svg(here / "two_point_last.svg")
    W, H = config.frame_width, config.frame_height
    print(f"wrote two_point.gif ({(here / 'two_point.gif').stat().st_size / 1e6:.1f} MB) "
          f"and two_point_last.svg")
    print(f"sky box for the slide CSS: left {100 * (SKY_C[0] - 2 * SKY_S + W / 2) / W:.2f}%  "
          f"top {100 * (H / 2 - SKY_C[1] - SKY_S) / H:.2f}%  width {100 * 4 * SKY_S / W:.2f}%  "
          f"height {100 * 2 * SKY_S / H:.2f}%")


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parents[2]))
    from _common import skip_if_built

    skip_if_built(HERE, "two_point.gif")
    render(HERE)
