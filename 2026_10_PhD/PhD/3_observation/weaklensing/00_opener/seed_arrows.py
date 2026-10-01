#!/usr/bin/env python3
# ENV: shared
"""
Arrows from the CMB to today's large-scale structure, laid over the ΛCDM figure
on the lensing-section opener: the anisotropies seed the structure we see today.

Transparent SVG with the viewBox of 1_intro/lcdm/lcdm_model.svg (4096 x 2288), so
it overlays that figure exactly in an r-stack. Each arrow is a baseline plus a
sine whose amplitude and frequency grow along the way (small ripples near the
CMB, large tight wiggles near today: perturbations growing non-linearly). The
animation lives in the SVG itself (SMIL): the arrows draw in, their heads appear,
then the wiggles keep moving.

Output (this directory): seed_arrows.svg
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from _common import RED, skip_if_built

OUT = "seed_arrows.svg"
skip_if_built(HERE, OUT)

W, H = 4096, 2288
X0, X1 = 1540.0, 3270.0                               # just right of the CMB -> the LSS ellipse
ROWS = [(1020, 1000), (1210, 1210), (1400, 1420)]     # (y at the CMB, y at the LSS)
STROKE, HEAD = 16, 70
N, PHASES, T_DRAW, T_WIGGLE = 240, 12, 1.6, 1.8


def path(y0, y1, phase):
    u = np.linspace(0, 1, N)
    x = X0 + (X1 - X0) * u
    base = y0 + (y1 - y0) * u
    amp = 85 * u ** 2.2 * np.clip((1 - u) / 0.05, 0, 1)   # grows, then settles into the head
    cycles = 2 * np.pi * (2.0 * u + 5.0 * u ** 2)          # wiggles tighten along the way
    y = base + amp * np.sin(cycles - phase)
    return "M " + " L ".join(f"{a:.1f} {b:.1f}" for a, b in zip(x, y))


parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">']
for y0, y1 in ROWS:
    phases = np.linspace(0, 2 * np.pi, PHASES + 1)
    d0 = path(y0, y1, 0.0)
    cycle = ";".join(path(y0, y1, p) for p in phases)
    parts.append(
        f'<path d="{d0}" fill="none" stroke="{RED}" stroke-width="{STROKE}" stroke-linecap="round" '
        f'stroke-linejoin="round" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1">'
        f'<animate attributeName="stroke-dashoffset" from="1" to="0" dur="{T_DRAW}s" fill="freeze"/>'
        f'<animate attributeName="d" values="{cycle}" dur="{T_WIGGLE}s" begin="{T_DRAW}s" '
        f'repeatCount="indefinite"/></path>')
    ang = np.arctan2(y1 - y0, X1 - X0)
    tip = np.array([X1 + 20, y1])
    back = tip - HEAD * np.array([np.cos(ang), np.sin(ang)])
    n = 0.55 * HEAD * np.array([-np.sin(ang), np.cos(ang)])
    pts = " ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in (tip, back + n, back - n))
    parts.append(f'<polygon points="{pts}" fill="{RED}" opacity="0">'
                 f'<animate attributeName="opacity" from="0" to="1" begin="{T_DRAW - 0.2}s" '
                 f'dur="0.3s" fill="freeze"/></polygon>')
parts.append("</svg>")
(HERE / OUT).write_text("\n".join(parts))
print(f"wrote {OUT} ({(HERE / OUT).stat().st_size / 1e3:.0f} kB)")
