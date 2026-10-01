#!/usr/bin/env python3
# ENV: jax-fli
"""
The MAP reconstruction of experiment 13 (DES Y3), animated for the MAP slides, from the cache that
map_data.py writes (fli/.cache/map_des.npz; see its docstring for the run).

map_run.mp4 (8.2 s): the truth on the left, the reconstruction on the right, for the IC projected
with the lensing efficiency (drawn on the faces of the box, the observer at its centre) and the
convergence of source bins 2 and 3 (Mollweide). The reconstruction moves through the eleven saved
Adam steps (0 to 300), cross-faded from one to the next for display; the step counter and the
correlation with the truth are those of the saved step. While it runs, the box is drawn as the four
slabs of a slab decomposition along x, one per GPU, which join again at the end.

compare_run.mp4 (6 s): over the same saved steps, the coherence r(l) and the transfer
T(l) = sqrt(C_l^MAP / C_l^truth) of kappa for both bins, against the coherence of the joint Wiener
filter (dotted), and the starlet l1 norm of kappa in bin 3 at the five scales against the truth.

Outputs (this directory): map_run.mp4, map_first.png, map_last.png, compare_run.mp4,
compare_first.png, compare_last.png. Each PNG is a frame of its video, drawn by the same code.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from _common import BLUE, GREY, INK, KW, skip_if_built, slide_style

OUTS = ["map_run.mp4", "map_first.png", "map_last.png", "compare_run.mp4", "compare_first.png",
        "compare_last.png"]
skip_if_built(HERE, *OUTS)

D = np.load(HERE.parent / ".cache" / "map_des.npz")
STEPS = D["steps"]
NK = len(STEPS)
FPS = 20
BG = "#faf7f0"

import healpy as hp

slide_style()
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.transforms import Affine2D

plt.rcParams["savefig.bbox"] = None
plt.rcParams["savefig.transparent"] = False


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0, 1))


def timeline(hold0, split, per, gather, hold1):
    """(keyframe position, slab gap in [0, 1]) per video frame."""
    out = [(0.0, 0.0)] * hold0
    out += [(0.0, ease(i / (split - 1))) for i in range(split)]
    for k in range(NK - 1):
        out += [(k + ease((i + 1) / per), 1.0) for i in range(per)]
    out += [(NK - 1.0, 1 - ease(i / (gather - 1))) for i in range(gather)]
    out += [(NK - 1.0, 0.0)] * hold1
    return out


def at(stack, pos):
    """Keyframe stack blended at a fractional position."""
    k = min(int(np.floor(pos)), NK - 2)
    w = pos - k
    return (1 - w) * stack[k] + w * stack[k + 1]


def shown(pos):
    return int(np.clip(np.floor(pos + 0.5), 0, NK - 1))


def encode(frames_dir, out):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i",
                    str(frames_dir / "f_%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
                    "-movflags", "+faststart", str(HERE / out)], check=True)


# ------------------------------------------------------------------ the IC on the faces of the box
NSIDE = hp.npix2nside(D["proj_truth"].size)
N = 128
G = (np.arange(N) + 0.5) / N - 0.5
SLAB_X = [-0.5 + 0.25 * (k + 1) for k in range(4)]     # right face of each slab


def faces(sky):
    """Sky map -> values on the top face, the front face and the right face of each slab."""
    u, v = np.meshgrid(G, G)                             # u along columns, v along rows
    look = lambda x, y, z: sky[hp.vec2pix(NSIDE, x.ravel(), y.ravel(), z.ravel())].reshape(N, N)
    half = np.full_like(u, 0.5)
    out = {"top": look(u, v, half), "front": look(u, -half, v)}
    for k, xk in enumerate(SLAB_X):
        out[f"right{k}"] = look(np.full_like(u, xk), u, v)
    return out


F_TRUTH = faces(D["proj_truth"])
F_GUESS = [faces(p) for p in D["proj"]]
LIM_P = float(np.percentile(np.abs(D["proj_truth"]), 99.8))
NORM_P = Normalize(-LIM_P, LIM_P)
DEPTH = 0.5 * np.array([np.cos(np.pi / 4), np.sin(np.pi / 4)])


def P(x, y, z):
    """Cabinet projection: x to the right, z up, depth y along the diagonal at half length."""
    return np.array([x + DEPTH[0] * (y + 0.5), z + DEPTH[1] * (y + 0.5)])


def draw_face(ax, img, o, e1, e2, z, edge=1.0):
    """One face of a slab: the image mapped affinely onto its parallelogram, or a flat cut face.
    z is the painter's order (matplotlib would otherwise put every patch above every image)."""
    po, p1, p2 = P(*o), P(*(np.add(o, e1))) - P(*o), P(*(np.add(o, e2))) - P(*o)
    corners = np.array([po, po + p1, po + p1 + p2, po + p2, po])
    if img is None:
        ax.fill(corners[:, 0], corners[:, 1], color="#b9bfca", lw=0, zorder=z)
    else:
        tr = Affine2D(np.array([[p1[0], p2[0], po[0]], [p1[1], p2[1], po[1]], [0, 0, 1]]))
        ax.imshow(img, origin="lower", extent=(0, 1, 0, 1), cmap="magma", norm=NORM_P,
                  interpolation="bilinear", transform=tr + ax.transData, zorder=z)
    if edge > 0:
        ax.plot(corners[:, 0], corners[:, 1], color="#3b3b3b", lw=0.8, alpha=edge, zorder=z + 0.5)


def draw_box(ax, fc, gap, label_alpha):
    """The box as four slabs along x, each shifted by gap * 0.2 * (k - 1.5). A cut face inside the
    box is drawn flat: the ray-traced projection has no meaning on a plane through the observer."""
    for k in range(4):
        x0, x1 = -0.5 + 0.25 * k, -0.5 + 0.25 * (k + 1)
        s = gap * 0.2 * (k - 1.5)
        cols = slice(k * N // 4, (k + 1) * N // 4)
        draw_face(ax, fc["front"][:, cols], (x0 + s, -0.5, -0.5), (0.25, 0, 0), (0, 0, 1), 3 * k + 1, gap)
        draw_face(ax, fc["top"][:, cols], (x0 + s, -0.5, 0.5), (0.25, 0, 0), (0, 1, 0), 3 * k + 2, gap)
        draw_face(ax, fc["right3"] if k == 3 else None, (x1 + s, -0.5, -0.5), (0, 1, 0), (0, 0, 1),
                  3 * k + 3, gap)
        if label_alpha > 0:
            c = P(0.5 * (x0 + x1) + s, 0.5, 0.5)
            ax.text(c[0], c[1] + 0.05, f"GPU {k + 1}", ha="center", va="bottom", fontsize=10,
                    color=INK, alpha=label_alpha, zorder=20)
    if gap < 1:                                          # the outline of the whole box
        for o, e1, e2 in (((-0.5, -0.5, -0.5), (1, 0, 0), (0, 0, 1)), ((-0.5, -0.5, 0.5), (1, 0, 0), (0, 1, 0)),
                          ((0.5, -0.5, -0.5), (0, 1, 0), (0, 0, 1))):
            po, p1, p2 = P(*o), P(*np.add(o, e1)) - P(*o), P(*np.add(o, e2)) - P(*o)
            c = np.array([po, po + p1, po + p1 + p2, po + p2, po])
            ax.plot(c[:, 0], c[:, 1], color="#3b3b3b", lw=0.8, alpha=1 - gap, zorder=20)
    ax.set_xlim(-0.9, 1.15)
    ax.set_ylim(-0.56, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")


# ------------------------------------------------------------------ kappa in Mollweide
MOLL = hp.projector.MollweideProj(xsize=420)
f2p = lambda m: np.ma.masked_invalid(np.where(np.isfinite(
    p := MOLL.projmap(np.asarray(m, dtype=np.float64), lambda x, y, z: hp.vec2pix(hp.npix2nside(m.size), x, y, z))),
    p, np.nan))
K_TRUTH = [f2p(D["kappa_truth"][b]) for b in range(2)]
K_GUESS = np.array([[np.ma.filled(f2p(k[b]), np.nan) for b in range(2)] for k in D["kappa"]])
KLIM = np.percentile(D["kappa_truth"], [0.1, 99.9])
NORM_K = Normalize(*KLIM)
MAGMA = plt.get_cmap("magma").copy()
MAGMA.set_bad(alpha=0)


def draw_moll(ax, img):
    ax.imshow(img, origin="lower", cmap=MAGMA, norm=NORM_K, interpolation="bilinear")
    t = np.linspace(0, 2 * np.pi, 400)
    h, w = img.shape
    ax.plot(w / 2 + (w / 2 - 1) * np.cos(t), h / 2 + (h / 2 - 1) * np.sin(t), color="#3b3b3b", lw=0.8)
    ax.axis("off")


def map_frame(pos, gap, label_alpha):
    fig = plt.figure(figsize=(10.4, 5.4), dpi=150, facecolor=BG)
    k = shown(pos)
    fig.text(0.37, 0.955, "truth", ha="center", fontsize=15, color=INK)
    fig.text(0.72, 0.955, "reconstruction", ha="center", fontsize=15, color=INK)
    fig.text(0.015, 0.955, f"Adam step {STEPS[k]:d} / {STEPS[-1]:d}", ha="left", fontsize=13, color=KW)
    fig.text(0.935, 0.955, "correlation", ha="center", fontsize=12.5, color=GREY)
    rows = [("initial conditions\nprojected with\nthe lensing kernel", 0.59, 0.34),
            (r"$\kappa$, DES Y3 bin 2", 0.31, 0.25), (r"$\kappa$, DES Y3 bin 3", 0.03, 0.25)]
    for label, y, h in rows:
        fig.text(0.015, y + h / 2, label, ha="left", va="center", fontsize=12.5, color=INK, linespacing=1.25)
    # row 1: the box
    draw_box(fig.add_axes([0.235, 0.59, 0.27, 0.34]), F_TRUTH, 0.0, 0.0)
    guess = {key: at(np.array([f[key] for f in F_GUESS]), pos) for key in F_TRUTH}
    draw_box(fig.add_axes([0.585, 0.59, 0.27, 0.34]), guess, gap, label_alpha)
    fig.text(0.9, 0.745, rf"$r = {D['r_proj'][k]:.2f}$", fontsize=13, color=INK, va="center")
    # rows 2 and 3: kappa
    for b, (_, y, h) in enumerate(rows[1:]):
        draw_moll(fig.add_axes([0.25, y, 0.24, h]), K_TRUTH[b])
        draw_moll(fig.add_axes([0.6, y, 0.24, h]), np.ma.masked_invalid(at(K_GUESS[:, b], pos)))
        fig.text(0.9, y + h / 2, rf"$r = {D['r_kappa'][k, b]:.2f}$", fontsize=13, color=INK, va="center")
    return fig


# ------------------------------------------------------------------ coherence, transfer, starlet l1
ELL = D["ell"]
NU = D["nu"]
EDGE = np.arange(0, len(ELL) + 1, 6)                 # bands of 6 multipoles, for display
ELL_B = np.array([ELL[a:b].mean() for a, b in zip(EDGE[:-1], EDGE[1:])])
band = lambda y: np.stack([np.asarray(y)[..., a:b].mean(-1) for a, b in zip(EDGE[:-1], EDGE[1:])], -1)
SCALES = ["ℓ ≈ 96–192", "ℓ ≈ 48–96", "ℓ ≈ 24–48", "ℓ ≈ 12–24", "ℓ ≤ 12"]   # ST_NSIDE = 64
BINS = [(KW, "bin 2"), (BLUE, "bin 3")]
L1T = D["l1_truth"][1] / 1e3
L1_TOP = 1.12 * max(L1T.max(), (D["l1"][:, 1] / 1e3).max())


def compare_frame(pos):
    fig = plt.figure(figsize=(10.4, 5.0), dpi=150, facecolor=BG)
    k = shown(pos)
    fig.text(0.07, 0.96, f"Adam step {STEPS[k]:d} / {STEPS[-1]:d}", ha="left", fontsize=13, color=KW)
    coh, trans = band(at(D["coh"], pos)), band(at(D["trans"], pos))
    ax_c = fig.add_axes([0.07, 0.6, 0.4, 0.33])
    ax_t = fig.add_axes([0.575, 0.6, 0.4, 0.33])
    for b, (c, lab) in enumerate(BINS):
        ax_c.plot(ELL_B, band(D["wiener"])[b], color=c, ls=":", lw=2.0)
        ax_c.plot(ELL_B, coh[b], color=c, lw=2.0, label=lab)
        ax_t.plot(ELL_B, trans[b], color=c, lw=2.0, label=lab)
    for ax, ylab in ((ax_c, r"coherence $r(\ell)$"), (ax_t, r"transfer $T(\ell)$")):
        ax.axhline(1, color=GREY, ls="--", lw=1)
        ax.set_xlim(2, 188)
        ax.set_ylim(-0.1, 1.15)
        ax.set_xlabel(r"$\ell$", labelpad=0)
        ax.set_ylabel(ylab)
    ax_c.legend(loc="lower left", fontsize=11, handlelength=1.4, borderaxespad=0.2)
    ax_c.text(0.99, 0.97, "dotted: joint Wiener filter", transform=ax_c.transAxes, ha="right", va="top",
              fontsize=10.5, color=GREY)
    l1 = at(D["l1"][:, 1], pos) / 1e3
    for j in range(5):
        ax = fig.add_axes([0.07 + j * 0.184, 0.1, 0.16, 0.28])
        ax.plot(NU, L1T[j], color=INK, ls="--", lw=1.6, label="truth")
        ax.plot(NU, l1[j], color=BLUE, lw=2.0, label="reconstruction")
        ax.set_xlim(-5, 5)
        ax.set_ylim(0, L1_TOP)
        ax.set_xticks([-4, 0, 4])
        ax.set_title(f"scale {j + 1}, {SCALES[j]}", fontsize=11, color=INK, pad=3)
        ax.set_xlabel(r"$\nu$", labelpad=0)
        if j:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel(r"$\ell_1$ norm [$10^3$]", fontsize=12)
            ax.legend(loc="upper left", fontsize=9.5, handlelength=1.3, borderaxespad=0.1,
                      labelspacing=0.15)
    fig.text(0.07, 0.445, r"starlet $\ell_1$ norm of $\kappa$, bin 3", fontsize=13, color=INK)
    return fig


def render(make, schedule, stem):
    tmp = Path(tempfile.mkdtemp(prefix=f"{stem}_"))
    try:
        for i, args in enumerate(schedule):
            fig = make(*args)
            fig.savefig(tmp / f"f_{i:04d}.png", facecolor=BG)
            plt.close(fig)
        shutil.copy(tmp / "f_0000.png", HERE / f"{stem}_first.png")
        shutil.copy(tmp / f"f_{len(schedule) - 1:04d}.png", HERE / f"{stem}_last.png")
        encode(tmp, f"{stem}_run.mp4")
    finally:
        shutil.rmtree(tmp)
    print(f"wrote {stem}_run.mp4 ({len(schedule)} frames, {len(schedule) / FPS:.1f} s), {stem}_first/last.png")


LABEL = lambda gap: gap
render(lambda pos, gap: map_frame(pos, gap, gap), timeline(12, 16, 10, 16, 20), "map")
sched = [(0.0,)] * 10 + [(k + ease((i + 1) / 8),) for k in range(NK - 1) for i in range(8)] + [(NK - 1.0,)] * 30
render(compare_frame, sched, "compare")
