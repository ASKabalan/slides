"""
The 1200^3 MAP reconstruction of experiment 13 (DES Y3): data and frame drawing for three videos, from the
cache that map_data.py writes (fli/.cache/map_des.npz; see its docstring for the run). The code is that of
jax-fli docs/5-experiments/13-map-lpt2-mass-mapping/animation/map_anim.py, in the deck's style, without the GIFs.

map_frame: the truth on the left, the reconstruction on the right, for the IC projected with the lensing
efficiency (drawn flat: the front face of the box, y = -0.5, seen straight on) and the convergence of source bins 2
and 3 (Mollweide). The reconstruction moves through the 21 saved Adam steps (0 to 400), cross-faded from one to the
next for display; the step counter and the correlation with the truth are those of the saved step. Rendered by
map_anim.py (beside this file) for the MAP slide.

starlet_frame: over the same saved steps, the starlet l1 norm of kappa in bin 3 at the five scales (0, finest,
to 4, low-pass), reconstruction against truth. Rendered by 7_backup/4_fieldlevel/19_map_starlet/map_starlet.py.

spectra_frame: over the same saved steps as the starlet video, the cross-correlation coefficient
C_l^{k^ k} / sqrt(C_l^{k^ k^} C_l^{kk}) and the power ratio C_l^{k^ k^} / C_l^{kk} of the reconstructed kappa k^
against the truth k, for both bins, each against its joint Wiener-filter value (dotted: r_W and r_W^2), up to
l = 1000 with the scale cut of the likelihood (l_max = 700, above which the forward model has no kappa) marked.
Rendered by 7_backup/4_fieldlevel/18_map_spectra/map_spectra.py.

The importing script puts the deck's PhD folder on sys.path first (for _common).
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from _common import BLUE, GREY, INK, KW, slide_style

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "_common.py").exists())
D = np.load(ROOT / "5_contribution/fli/.cache/map_des.npz")
STEPS = D["steps"]
NK = len(STEPS)
FPS = 20
BG = "#faf7f0"

import healpy as hp

slide_style()
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

plt.rcParams["savefig.bbox"] = None
plt.rcParams["savefig.transparent"] = False


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0, 1))


def timeline(hold0, per, hold1):
    """Keyframe position per video frame: a hold on step 0, an eased cross-fade between saved steps, a final hold."""
    out = [(0.0,)] * hold0
    for k in range(NK - 1):
        out += [(k + ease((i + 1) / per),) for i in range(per)]
    return out + [(NK - 1.0,)] * hold1


def at(stack, pos):
    """Keyframe stack blended at a fractional position."""
    k = min(int(np.floor(pos)), NK - 2)
    w = pos - k
    return (1 - w) * stack[k] + w * stack[k + 1]


def shown(pos):
    return int(np.clip(np.floor(pos + 0.5), 0, NK - 1))


def encode(frames_dir, stem, outdir):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(FPS),
            "-i",
            str(frames_dir / "f_%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "16",
            "-movflags",
            "+faststart",
            str(outdir / f"{stem}.mp4"),
        ],
        check=True,
    )


# ------------------------------------------------------------------ the IC, flat: the front face of the box
NSIDE = hp.npix2nside(D["proj_truth"].size)
N = 256
G = (np.arange(N) + 0.5) / N - 0.5


def front(sky):
    """Sky map -> values on the front face of the box (y = -0.5), seen straight on."""
    u, v = np.meshgrid(G, G)  # u along columns (x), v along rows (z)
    x, y, z = u, np.full_like(u, -0.5), v
    return sky[hp.vec2pix(NSIDE, x.ravel(), y.ravel(), z.ravel())].reshape(N, N)


F_TRUTH = front(D["proj_truth"])
F_GUESS = np.array([front(p) for p in D["proj"]])
LIM_P = float(np.percentile(np.abs(D["proj_truth"]), 99.8))
NORM_P = Normalize(-LIM_P, LIM_P)


def draw_square(ax, img):
    ax.imshow(img, origin="lower", cmap="magma", norm=NORM_P, interpolation="bilinear")
    for sp in ax.spines.values():
        sp.set_color("#3b3b3b")
        sp.set_linewidth(0.8)
    ax.set_xticks([])
    ax.set_yticks([])


# ------------------------------------------------------------------ kappa in Mollweide
MOLL = hp.projector.MollweideProj(xsize=420)


def mollweide(m):
    p = MOLL.projmap(np.asarray(m, dtype=np.float64), lambda x, y, z: hp.vec2pix(hp.npix2nside(np.size(m)), x, y, z))
    return np.ma.masked_invalid(np.where(np.isfinite(p), p, np.nan))


K_TRUTH = [mollweide(D["kappa_truth"][b]) for b in range(2)]
K_GUESS = np.array([[np.ma.filled(mollweide(k[b]), np.nan) for b in range(2)] for k in D["kappa"]])
NORM_K = Normalize(*np.percentile(D["kappa_truth"], [0.1, 99.9]))
MAGMA = plt.get_cmap("magma").copy()
MAGMA.set_bad(alpha=0)


def draw_moll(ax, img):
    ax.imshow(img, origin="lower", cmap=MAGMA, norm=NORM_K, interpolation="bilinear")
    t = np.linspace(0, 2 * np.pi, 400)
    h, w = img.shape
    ax.plot(w / 2 + (w / 2 - 1) * np.cos(t), h / 2 + (h / 2 - 1) * np.sin(t), color="#3b3b3b", lw=0.8)
    ax.axis("off")


def map_frame(pos):
    fig = plt.figure(figsize=(10.4, 5.4), dpi=150, facecolor=BG)
    k = shown(pos)
    fig.text(0.37, 0.955, "truth", ha="center", fontsize=15, color=INK)
    fig.text(0.72, 0.955, "reconstruction", ha="center", fontsize=15, color=INK)
    fig.text(0.015, 0.955, f"Adam step {STEPS[k]:d} / {STEPS[-1]:d}", ha="left", fontsize=13, color=KW)
    fig.text(0.935, 0.955, "correlation", ha="center", fontsize=12.5, color=GREY)
    rows = [
        ("initial conditions\nprojected with\nthe lensing kernel", 0.59, 0.34),
        (r"$\kappa$, DES Y3 bin 2", 0.31, 0.25),
        (r"$\kappa$, DES Y3 bin 3", 0.03, 0.25),
    ]
    for label, y, h in rows:
        fig.text(0.015, y + h / 2, label, ha="left", va="center", fontsize=12.5, color=INK, linespacing=1.25)
    # row 1: the projected IC, flat, the same height as the box it replaces, centred over the kappa maps
    draw_square(fig.add_axes([0.235, 0.59, 0.27, 0.34]), F_TRUTH)
    draw_square(fig.add_axes([0.585, 0.59, 0.27, 0.34]), at(F_GUESS, pos))
    fig.text(0.9, 0.745, rf"$r = {D['r_proj'][k]:.2f}$", fontsize=13, color=INK, va="center")
    # rows 2 and 3: kappa
    for b, (_, y, h) in enumerate(rows[1:]):
        draw_moll(fig.add_axes([0.25, y, 0.24, h]), K_TRUTH[b])
        draw_moll(fig.add_axes([0.6, y, 0.24, h]), np.ma.masked_invalid(at(K_GUESS[:, b], pos)))
        fig.text(0.9, y + h / 2, rf"$r = {D['r_kappa'][k, b]:.2f}$", fontsize=13, color=INK, va="center")
    return fig


# ------------------------------------------------------------------ cross-correlation, power ratio, starlet l1
ELL = D["ell"]
ELL_MAX, ELL_SHOW = int(D["ell_max"]), 1000  # the scale cut of the likelihood, and the l range drawn
NU = D["nu"]
EDGE = np.arange(0, len(ELL) + 1, 20)  # bands of 20 multipoles, for display
ELL_B = np.array([ELL[a:b].mean() for a, b in zip(EDGE[:-1], EDGE[1:], strict=True)])
# scales numbered 0 (finest) to 4 (low-pass) as on the starlet slide, named the same way: that slide's starlet runs
# at nside 512 (scale 0 above l ~ 700, scale 1 peaking near l = 230, each next one at half the multipole); here it
# runs at st_nside, so every multipole scales by st_nside / 512
_S = int(D["st_nside"]) / 512
SCALES = [f"ℓ ≳ {700 * _S:.0f}"] + [f"ℓ ≈ {230 * _S / 2 ** j:.0f}" for j in range(3)] + ["low-pass"]
BINS = [(KW, "bin 2"), (BLUE, "bin 3")]
L1T = D["l1_truth"][1] / 1e3
L1_TOP = 1.12 * max(L1T.max(), (D["l1"][:, 1] / 1e3).max())


def band(y):
    return np.stack([np.asarray(y)[..., a:b].mean(-1) for a, b in zip(EDGE[:-1], EDGE[1:], strict=True)], -1)


# a Wiener estimate has C^{k^ k} = C^{k^ k^} = [C (C + N)^-1 C]: its power ratio is r_W^2
WIENER_B = band(D["wiener"])
WIENER2_B = band(D["wiener"] ** 2)
RATIO = D["trans"] ** 2  # C_l^{k^ k^} / C_l^{kk}


def spectra_axes(fig, pos, rect_c, rect_t):
    """The cross-correlation and power-ratio panels of both bins, against the joint Wiener filter."""
    coh, ratio = band(at(D["coh"], pos)), band(at(RATIO, pos))
    ax_c = fig.add_axes(rect_c)
    ax_t = fig.add_axes(rect_t)
    for b, (c, lab) in enumerate(BINS):
        ax_c.plot(ELL_B, WIENER_B[b], color=c, ls=":", lw=2.0)
        ax_c.plot(ELL_B, coh[b], color=c, lw=2.0, label=lab)
        ax_t.plot(ELL_B, WIENER2_B[b], color=c, ls=":", lw=2.0)
        ax_t.plot(ELL_B, ratio[b], color=c, lw=2.0, label=lab)
    for ax, ylab in (
        (ax_c, r"$\dfrac{C_\ell^{\hat\kappa\kappa}}{\sqrt{C_\ell^{\hat\kappa\hat\kappa}\,C_\ell^{\kappa\kappa}}}$"),
        (ax_t, r"$\dfrac{C_\ell^{\hat\kappa\hat\kappa}}{C_\ell^{\kappa\kappa}}$"),
    ):
        ax.axhline(1, color=GREY, ls="--", lw=1)
        ax.axvspan(ELL_MAX, ELL_SHOW, color="#e4e0d6", lw=0)
        ax.axvline(ELL_MAX, color=INK, ls="--", lw=1.2)
        ax.text(0.5 * (ELL_MAX + ELL_SHOW), 0.5, f"scale cut\n$\\ell_{{\\max}} = {ELL_MAX}$",
                ha="center", va="center", fontsize=10.5, color=GREY)
        ax.set_xlim(2, ELL_SHOW)
        ax.set_ylim(-0.1, 1.15)
        ax.set_xlabel(r"$\ell$", labelpad=0)
        ax.set_ylabel(ylab, fontsize=15, labelpad=2)
    handles = ax_c.get_legend_handles_labels()[0] + [Line2D([], [], color=GREY, ls=":", lw=2.0)]
    ax_c.legend(handles, ["bin 2", "bin 3", "joint Wiener filter"], loc="upper right",
                bbox_to_anchor=(0.69, 0.86), fontsize=10.5, handlelength=1.6, borderaxespad=0.2,
                labelspacing=0.2)


def starlet_frame(pos):
    fig = plt.figure(figsize=(10.4, 3.9), dpi=150, facecolor=BG)
    k = shown(pos)
    fig.text(0.07, 0.93, f"Adam step {STEPS[k]:d} / {STEPS[-1]:d}", ha="left", fontsize=13, color=KW)
    fig.text(0.975, 0.93, r"starlet $\ell_1$ norm of $\kappa$, bin 3", ha="right", fontsize=13, color=INK)
    l1 = at(D["l1"][:, 1], pos) / 1e3
    for j in range(5):
        ax = fig.add_axes([0.07 + j * 0.184, 0.15, 0.16, 0.67])
        ax.plot(NU, L1T[j], color=INK, ls="--", lw=1.6, label="truth")
        ax.plot(NU, l1[j], color=BLUE, lw=2.0, label="reconstruction")
        ax.set_xlim(-5, 5)
        ax.set_ylim(0, L1_TOP)
        ax.set_xticks([-4, 0, 4])
        ax.set_title(f"scale {j}, {SCALES[j]}", fontsize=11, color=INK, pad=3)
        ax.set_xlabel(r"$\nu$", labelpad=0)
        if j:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel(r"$\ell_1$ norm [$10^3$]", fontsize=12)
            ax.legend(loc="upper left", fontsize=9.5, handlelength=1.3, borderaxespad=0.1, labelspacing=0.15)
    return fig


def spectra_frame(pos):
    fig = plt.figure(figsize=(10.4, 5.6), dpi=150, facecolor=BG)
    k = shown(pos)
    fig.text(0.12, 0.95, f"Adam step {STEPS[k]:d} / {STEPS[-1]:d}", ha="left", fontsize=13, color=KW)
    fig.text(0.975, 0.95, r"$\hat\kappa$: reconstruction, $\kappa$: truth", ha="right", fontsize=12, color=INK)
    spectra_axes(fig, pos, [0.12, 0.11, 0.35, 0.79], [0.625, 0.11, 0.35, 0.79])
    return fig


def render(make, schedule, stem, outdir, first=True):
    """Render the frames of make over schedule into outdir/<stem>_run.mp4, its last frame as <stem>_last.png
    and, when first, its first frame as <stem>_first.png (the still a click starts the video from)."""
    tmp = Path(tempfile.mkdtemp(prefix=f"{stem}_"))
    try:
        for i, args in enumerate(schedule):
            fig = make(*args)
            fig.savefig(tmp / f"f_{i:04d}.png", facecolor=BG)
            plt.close(fig)
        if first:
            shutil.copy(tmp / "f_0000.png", outdir / f"{stem}_first.png")
        shutil.copy(tmp / f"f_{len(schedule) - 1:04d}.png", outdir / f"{stem}_last.png")
        encode(tmp, f"{stem}_run", outdir)
    finally:
        shutil.rmtree(tmp)
    print(f"wrote {stem}_run.mp4 ({len(schedule)} frames, {len(schedule) / FPS:.1f} s) and its still frames")



# the map video's timeline, and the one the starlet and spectra videos share (backup slides, autoplay)
MAP_SCHED = timeline(12, 6, 36)
SCHED = [(0.0,)] * 10 + [(k + ease((i + 1) / 5),) for k in range(NK - 1) for i in range(5)] + [(NK - 1.0,)] * 30
