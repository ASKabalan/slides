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
From pairs of points to the two-point correlation function, with the real
estimator: every number on screen comes from one map (two_point_data.py: Planck
2018 SMICA, Stokes I, Nside 256, 1 deg FWHM).

Left: the sky (two_point_sky.png, NOT in the GIF: the slide lays it under the
transparent GIF at the box this script prints); under it, an inset: the
observer at the centre of the sky, two lines of sight n1 and n2, and the angle
theta between them. Right: C(theta), one bar per 5-degree bin (centred on multiples of
5 deg, half bins at 0 and 90), each bar the mean
of dT1 dT2 over all pairs of the sky in its bin (the estimator itself).

  0 - 6 s   three rounds, 1.5 s each plus a 0.5 s hold: around one point n1,
            the ring of every n2 at angle theta (5, 10 and 45 deg, three bin
            centres) is drawn, orange where dT1 dT2 > 0 and blue where it is
            negative, with a spoke n1 -> n2 marked theta; the inset opens to the
            same theta, a marker moves to the bin, and the bar of that bin rises.
  6 - 12 s  the rings go, the other bars rise quickly from small to large theta,
            the map's own C(theta) = sum_l (2l+1)/4pi C_l P_l(cos theta) is drawn
            over them (no fitted amplitude), and the bars fade.

Linear theta axis, 0 to 90 deg, with the matching multipole l ~ 180 deg / theta
along the top (a guide, not a re-binning). Plays once, PLAY_S seconds.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, UP, AnimationGroup, Arc, Circle, Create, Dot, FadeIn, FadeOut,
                   GrowFromEdge, LaggedStart, Line, MathTex, Polygon, Scene, Text,
                   ValueTracker, VGroup, VMobject, always_redraw, config, linear, smooth)

config.background_opacity = 0.0
HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
PLAY_S = 12.0         # the GIF's length on the slide (the scene is built to last exactly this)
D = dict(np.load(HERE / ".cache" / "two_point.npz"))

# ------------------------------------------------------------------ layout
SKY_C, SKY_S = np.array([-3.45, 0.2]), 1.6       # map centre, Mollweide scale (x in +-2)
PX0, PX1, PY0, PY1 = 1.75, 6.6, -2.4, 2.15       # plot box (top kept clear of the corner badge)
T_RING, T_BAR, T_HOLD = 1.0, 0.5, 0.5             # one round: ring + bar = 1.5 s, then a hold
T_CLEAR, T_FILL, T_CURVE, T_FADE, T_END = 0.3, 3.0, 1.5, 0.5, 0.7
INSET_C, INSET_R, N1_ANGLE = np.array([-3.45, -2.75, 0.0]), 0.95, 115.0   # inset under the sky

INK, GREY = "#2E2E2E", "#8A8A96"
POS, NEG, BAR, CURVE = "#C2560A", "#3B6FB6", "#521463", "#C2560A"
FONT = "Lato"

BINS = D["bins"]
NB = len(BINS) - 1
TMAX = float(BINS[-1])
# the bars only ever rise to their final values: the y-range covers those and the curve, 10 % margin
_lo = min(D["final"].min(), D["exact"].min(), 0.0)
_hi = max(D["final"].max(), D["exact"].max(), 0.0)
YMIN, YMAX = _lo - 0.1 * (_hi - _lo), _hi + 0.1 * (_hi - _lo)
X_TICKS = np.arange(0, TMAX + 1, 15)
Y_TICKS = np.arange(np.ceil(YMIN / 1000) * 1000, YMAX, 1000)
Y_LABEL = "C(θ) = ⟨ΔT ΔT⟩  [µK²]"
# top axis: the multipole matching each angle, l ~ 180 deg / theta (large l on the left)
ELL_TICKS = (180, 18, 6, 3, 2)
ELL_LABEL = "multipole ℓ ≈ 180° / θ"


def sky(xy):
    return np.array([SKY_C[0] + SKY_S * xy[0], SKY_C[1] + SKY_S * xy[1], 0.0])


def px(theta):
    return PX0 + (PX1 - PX0) * theta / TMAX


def py(v):
    return PY0 + (PY1 - PY0) * (v - YMIN) / (YMAX - YMIN)


def axes():
    g = VGroup(Line([PX0, PY0, 0], [PX1 + 0.1, PY0, 0], color=INK, stroke_width=3),
               Line([PX0, PY0, 0], [PX0, PY1, 0], color=INK, stroke_width=3),
               Line([PX0, py(0), 0], [PX1 + 0.1, py(0), 0], color=GREY, stroke_width=1.5))
    for t in X_TICKS:
        x = px(t)
        g.add(Line([x, PY0, 0], [x, PY0 - 0.1, 0], color=INK, stroke_width=3))
        g.add(Text(f"{t:.0f}°", font=FONT, color=INK).scale(0.32).move_to([x, PY0 - 0.33, 0]))
    for v in Y_TICKS:
        y = py(v)
        g.add(Line([PX0, y, 0], [PX0 - 0.1, y, 0], color=INK, stroke_width=3))
        g.add(Text(f"{v + 0:.0f}", font=FONT, color=INK).scale(0.28)
              .next_to([PX0 - 0.12, y, 0], direction=[-1, 0, 0], buff=0.06))
    g.add(Text("separation θ", font=FONT, color=INK).scale(0.4)
          .move_to([(PX0 + PX1) / 2, PY0 - 0.8, 0]))
    g.add(Line([PX0, PY1, 0], [PX1 + 0.1, PY1, 0], color=GREY, stroke_width=2))
    for ell in ELL_TICKS:
        x = px(180.0 / ell)
        g.add(Line([x, PY1, 0], [x, PY1 + 0.1, 0], color=GREY, stroke_width=2))
        g.add(Text(f"{ell}", font=FONT, color=GREY).scale(0.3).move_to([x, PY1 + 0.32, 0]))
    g.add(Text(ELL_LABEL, font=FONT, color=GREY).scale(0.36)
          .move_to([(PX0 + PX1) / 2, PY1 + 0.75, 0]))
    g.add(Text(Y_LABEL, font=FONT, color=INK).scale(0.36).rotate(np.pi / 2)
          .move_to([PX0 - 1.05, (PY0 + PY1) / 2, 0]))
    return g


def polyline_pieces(xy, colours):
    """Mollweide polyline split where it leaves the map (NaN) or changes colour."""
    out, cur, col = [], [], None
    for p, c in zip(xy, colours):
        if not np.all(np.isfinite(p)):
            if len(cur) > 1:
                out.append((cur, col))
            cur, col = [], None
            continue
        if col is not None and c != col:
            if len(cur) > 1:
                out.append((cur + [p], col))
            cur = [cur[-1]] if cur else []
        cur.append(p)
        col = c
    if len(cur) > 1:
        out.append((cur, col))
    return out


def polyline(pts, colour, width):
    v = VMobject(stroke_color=colour, stroke_width=width, fill_opacity=0)
    v.set_points_as_corners([sky(p) for p in pts])
    return v


def direction(deg):
    a = np.deg2rad(deg)
    return np.array([np.cos(a), np.sin(a), 0.0])


class TwoPoint(Scene):
    def construct(self):
        final = D["final"]
        bars = []
        for i in range(NB):
            x0, x1 = px(BINS[i]) + 0.02, px(BINS[i + 1]) - 0.02
            y0, y1 = py(0), py(final[i])
            bars.append(Polygon([x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0],
                                stroke_width=0, fill_color=BAR, fill_opacity=0.85))

        def grow(i):
            return GrowFromEdge(bars[i], DOWN if final[i] >= 0 else UP)

        # ---- the inset: observer, two lines of sight, the angle between them
        th = ValueTracker(0.0)
        n1_tip = INSET_C + INSET_R * direction(N1_ANGLE)
        inset_static = VGroup(
            Circle(radius=INSET_R, color=GREY, stroke_width=2).move_to(INSET_C),
            Line(INSET_C, n1_tip, color=INK, stroke_width=4),
            Dot(n1_tip, radius=0.06, color=INK),
            MathTex(r"\hat n_1", color=INK).scale(0.7).move_to(INSET_C + 1.3 * INSET_R * direction(N1_ANGLE + 9)),
            Dot(INSET_C, radius=0.07, color=INK),
            Text("observer", font=FONT, color=GREY).scale(0.28).move_to(INSET_C + [0, -0.22, 0]))

        def inset_moving():
            t = th.get_value()
            a2 = N1_ANGLE - t
            tip = INSET_C + INSET_R * direction(a2)
            g = VGroup(Line(INSET_C, tip, color=INK, stroke_width=4), Dot(tip, radius=0.06, color=INK),
                       MathTex(r"\hat n_2", color=INK).scale(0.7).move_to(INSET_C + 1.3 * INSET_R * direction(a2 - 9)))
            if t > 0.5:
                g.add(Arc(radius=0.42, start_angle=np.deg2rad(a2), angle=np.deg2rad(t),
                          arc_center=INSET_C, color=POS, stroke_width=5))
                if t > 20:                       # room for the symbol inside the arc
                    g.add(MathTex(r"\theta", color=POS).scale(0.8)
                          .move_to(INSET_C + 0.66 * direction(N1_ANGLE - t / 2)))
                g.add(Text(f"θ = {t:.0f}°", font=FONT, color=POS, weight="BOLD").scale(0.42)
                      .move_to(INSET_C + [INSET_R + 1.0, -0.05, 0]))
            return g

        # ---- the marker under the theta axis
        def marker():
            x = px(th.get_value())
            return Polygon([x, PY0 + 0.02, 0], [x - 0.1, PY0 - 0.15, 0], [x + 0.1, PY0 - 0.15, 0],
                           stroke_width=0, fill_color=POS, fill_opacity=1 if th.get_value() > 0.5 else 0)

        anchor_xy = D["ring_anchor_xy"]
        anchor = VGroup(Dot(sky(anchor_xy), radius=0.08, color=INK),
                        MathTex(r"\hat n_1", color=INK).scale(0.6).next_to(sky(anchor_xy), UP + 0.4 * UP, buff=0.05))
        moving, mark = always_redraw(inset_moving), always_redraw(marker)
        self.add(axes(), inset_static, moving, mark, anchor)

        # ---- 0 - 6 s: three rings, three bars
        previous = None
        for k, theta in enumerate(D["ring_theta"]):
            i = int(np.clip(np.digitize(theta, BINS) - 1, 0, NB - 1))
            roll = int(np.argmin(np.abs(np.linalg.norm(D["ring_xy"][k] - D["spoke_xy"][k, -1], axis=1))))
            ring_xy = np.roll(D["ring_xy"][k], -roll, axis=0)
            ring_col = np.where(np.roll(D["ring_sign"][k], -roll) > 0, POS, NEG)
            frac = ValueTracker(0.0)

            def ring(frac=frac, ring_xy=ring_xy, ring_col=ring_col):
                n = int(round(frac.get_value() * len(ring_xy)))
                return VGroup(*[polyline(pts, c, 5) for pts, c in polyline_pieces(ring_xy[:n], ring_col[:n])])

            spoke_xy = D["spoke_xy"][k][np.all(np.isfinite(D["spoke_xy"][k]), axis=1)]
            spoke = polyline(spoke_xy, INK, 4)
            mid = sky(spoke_xy[len(spoke_xy) // 2])
            label = MathTex(r"\theta", color=INK).scale(0.6).move_to(mid + [0.0, -0.22, 0])
            if theta < 8:                    # no room on a small ring: the inset carries theta
                label.set_opacity(0)
            end = Dot(sky(spoke_xy[-1]), radius=0.06, color=INK)
            ring_m = always_redraw(ring)
            self.add(ring_m)
            outgoing = [FadeOut(m) for m in previous] if previous else []
            self.play(frac.animate.set_value(1.0), Create(spoke), FadeIn(label), FadeIn(end),
                      th.animate.set_value(float(theta)), *outgoing, run_time=T_RING, rate_func=smooth)
            self.play(grow(i), run_time=T_BAR)
            self.wait(T_HOLD)
            ring_m.clear_updaters()
            previous = [ring_m, spoke, label, end]
            bars[i].done = True

        # ---- 6 - 12 s: the other bars, then the curve
        moving.clear_updaters()
        mark.clear_updaters()
        self.play(*[FadeOut(m) for m in previous], FadeOut(anchor), FadeOut(inset_static),
                  FadeOut(moving), FadeOut(mark), run_time=T_CLEAR)
        rest = [grow(i) for i in range(NB) if not getattr(bars[i], "done", False)]
        self.play(LaggedStart(*rest, lag_ratio=0.15), run_time=T_FILL)
        curve = VMobject(stroke_color=CURVE, stroke_width=7)
        curve.set_points_smoothly([[px(t), py(v), 0] for t, v in zip(D["theta"], D["exact"])])
        self.play(Create(curve), run_time=T_CURVE, rate_func=linear)
        self.play(*[b.animate.set_fill(opacity=0) for b in bars], run_time=T_FADE)
        self.wait(T_END)


def write_last_svg(path: Path) -> None:
    """Final state (axes + curve), vector; the sky is laid by the slide."""
    W, H = config.frame_width, config.frame_height
    f = lambda x, y: f"{x:.3f},{-y:.3f}"  # noqa: E731
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{-W / 2} {-H / 2} {W} {H}" '
             f'width="1920" height="1080" font-family="{FONT}">',
             f'<polyline points="{f(PX1 + 0.1, PY0)} {f(PX0, PY0)} {f(PX0, PY1)}" fill="none" '
             f'stroke="{INK}" stroke-width="0.04"/>',
             f'<line x1="{PX0}" y1="{-py(0):.3f}" x2="{PX1 + 0.1}" y2="{-py(0):.3f}" '
             f'stroke="{GREY}" stroke-width="0.02"/>']
    for t in X_TICKS:
        x = px(t)
        parts.append(f'<line x1="{x:.3f}" y1="{-PY0:.3f}" x2="{x:.3f}" y2="{-(PY0 - 0.1):.3f}" '
                     f'stroke="{INK}" stroke-width="0.04"/>')
        parts.append(f'<text x="{x:.3f}" y="{-(PY0 - 0.43):.3f}" font-size="0.28" '
                     f'text-anchor="middle" fill="{INK}">{t:.0f}°</text>')
    for v in Y_TICKS:
        y = py(v)
        parts.append(f'<line x1="{PX0:.3f}" y1="{-y:.3f}" x2="{PX0 - 0.1:.3f}" y2="{-y:.3f}" '
                     f'stroke="{INK}" stroke-width="0.04"/>')
        parts.append(f'<text x="{PX0 - 0.18:.3f}" y="{-y + 0.09:.3f}" font-size="0.25" '
                     f'text-anchor="end" fill="{INK}">{v + 0:.0f}</text>')
    parts.append(f'<text x="{(PX0 + PX1) / 2:.3f}" y="{-(PY0 - 0.9):.3f}" font-size="0.34" '
                 f'text-anchor="middle" fill="{INK}">separation θ</text>')
    parts.append(f'<line x1="{PX0}" y1="{-PY1}" x2="{PX1 + 0.1}" y2="{-PY1}" stroke="{GREY}" '
                 f'stroke-width="0.03"/>')
    for ell in ELL_TICKS:
        x = px(180.0 / ell)
        parts.append(f'<line x1="{x:.3f}" y1="{-PY1:.3f}" x2="{x:.3f}" y2="{-(PY1 + 0.1):.3f}" '
                     f'stroke="{GREY}" stroke-width="0.03"/>')
        parts.append(f'<text x="{x:.3f}" y="{-(PY1 + 0.22):.3f}" font-size="0.26" '
                     f'text-anchor="middle" fill="{GREY}">{ell}</text>')
    parts.append(f'<text x="{(PX0 + PX1) / 2:.3f}" y="{-(PY1 + 0.65):.3f}" font-size="0.3" '
                 f'text-anchor="middle" fill="{GREY}">{ELL_LABEL}</text>')
    parts.append(f'<text transform="translate({PX0 - 1.05:.3f},{-(PY0 + PY1) / 2:.3f}) rotate(-90)" '
                 f'font-size="0.3" text-anchor="middle" fill="{INK}">{Y_LABEL}</text>')
    pts = " ".join(f(px(t), py(v)) for t, v in zip(D["theta"], D["exact"]))
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
    sys.path.insert(0, str(ROOT))
    from _common import skip_if_built

    skip_if_built(HERE, "two_point.gif")
    render(HERE)
