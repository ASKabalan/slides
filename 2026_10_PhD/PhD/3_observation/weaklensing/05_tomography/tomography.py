#!/usr/bin/env python3
# ENV: jax-fli
"""
The pieces of the "Tomographic projection to convergence maps" slide: a long strip of matter
overdensity along the line of sight with the DES Y3 lensing kernels under it, and a cutaway sphere
of lightcone shells, in the manner of the GLASS figure (Tessore et al. 2023), that grows one source
bin per click.

Strip: a 1LPT density field on a long thin box (3200 x 500 x 50 Mpc/h, 1024 x 160 x 16 cells, the
CosmoGridV1 cosmology), each particle displaced by D(a(chi)) / D(1) of its Zel'dovich displacement,
chi being its position along the box, so structure grows towards the observer at chi = 0; projected
over the thin axis and drawn as log(1 + delta) in magma.

Kernels: the DES Y3 source n(z) of jax_fli and the jax_cosmo WeakLensing kernel q_i, on the
CosmoGridV1 cosmology read from one cached CosmoGrid shell, as 5_contribution/fli/06_observable/depth.py. The depth of
bin i, chi_i, is the thesis rule: the last z where n_i(z) is at least 10 % of its peak.

Sphere: one shell per source bin, at the depth chi_i of that bin (to scale), one octant cut away,
ray traced in orthographic projection. Each texture is the lightcone shell of the 3072^3 run of
jax-fli experiment 05e (20 equal-volume shells, nside 2048, read at nside 256) nearest to chi_i;
that run stops at 2500 Mpc/h, so bins 3 and 4 take its two outermost shells.

Outputs (this directory, transparent):
  tomo_0.png          the strip and the empty kernel panel
  tomo_{1..4}.png     bin i: the matter in front of chi_i highlighted, the rest dimmed, q_i drawn
  onion_{1..4}.png    the sphere with the shells of bins 1 to i
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import GREY, INK, skip_if_built, slide_style

OUTS = [f"tomo_{i}.png" for i in range(5)] + [f"onion_{i}.png" for i in range(1, 5)]
skip_if_built(HERE, *OUTS)

CACHE = HERE.parent / ".cache"
EXP = Path("/home/wassim/Projects/NBody/jax-fli-experiments")
SHELL0 = "00-cosmogrid/cosmo_000001/density/cosmogrid_density_nside2048_shell_000.parquet"
RUN = EXP / "05-spacing-n-stepping/05e-mesh/density/exp5e_m3072"
REPO = "ASKabalan/jax-fli-experiments"
ZGRID = np.linspace(0.005, 2.995, 600)
THRESH_FRAC = 0.10
NSIDE = 256
MESH, BOX = (1024, 160, 16), (3200.0, 500.0, 50.0)
BG = "#faf7f0"                       # the slide background


# ------------------------------------------------------------------ geometry and kernels
def geometry():
    npz = CACHE / "tomo_geometry.npz"
    if npz.exists():
        d = np.load(npz)
        return {k: d[k] for k in d.files}
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import jax_cosmo as jc
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from jax_fli.data import get_des_y3_nz_shear
    from jax_fli.io import Catalog

    local = EXP / SHELL0
    fp = local if local.exists() else hf_hub_download(REPO, SHELL0, repo_type="dataset")
    cosmo = Catalog.from_dataset(
        load_dataset("parquet", data_files=str(fp), split="train").with_format("numpy")).cosmology[0]
    nz_list = get_des_y3_nz_shear()
    nz = np.array([np.asarray(f(jnp.asarray(ZGRID))) for f in nz_list])
    q = np.asarray(jc.probes.WeakLensing(nz_list).kernel(cosmo, jnp.asarray(ZGRID), 1000.0))
    a = jc.utils.z2a(jnp.asarray(ZGRID))
    chi = np.asarray(jc.background.radial_comoving_distance(cosmo, a))
    z_end = np.array([ZGRID[np.where(v >= THRESH_FRAC * v.max())[0][-1]] for v in nz])
    chi_end = np.interp(z_end, ZGRID, chi)
    # growth along the strip
    xg = np.linspace(1.0, BOX[0], 400)
    ag = np.asarray(jc.background.a_of_chi(cosmo, jnp.asarray(xg)))
    dg = np.asarray(jc.background.growth_factor(cosmo, jnp.asarray(ag)))
    d1 = float(np.asarray(jc.background.growth_factor(cosmo, jnp.atleast_1d(1.0)))[0])
    kk = np.logspace(-4, 1.5, 600)
    pk = np.asarray(jc.power.linear_matter_power(cosmo, jnp.asarray(kk), a=1.0))
    out = dict(chi=chi, nz=nz, q=q, z_end=z_end, chi_end=chi_end, xg=xg, growth=dg / d1, kk=kk, pk=pk)
    CACHE.mkdir(exist_ok=True)
    np.savez(npz, **out)
    return out


# ------------------------------------------------------------------ the strip
def strip(G):
    npz = CACHE / "tomo_strip.npz"
    if npz.exists():
        return np.load(npz)["delta"]
    nx, ny, nz = MESH
    lx, ly, lz = BOX
    rng = np.random.default_rng(11)
    kx = 2 * np.pi * np.fft.fftfreq(nx, lx / nx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, ly / ny)
    kz = 2 * np.pi * np.fft.rfftfreq(nz, lz / nz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    k2 = KX ** 2 + KY ** 2 + KZ ** 2
    k = np.sqrt(k2)
    pk = np.exp(np.interp(np.log(np.where(k > 0, k, 1e-4)), np.log(G["kk"]), np.log(G["pk"])))
    vcell = lx * ly * lz / (nx * ny * nz)
    dk = np.fft.rfftn(rng.standard_normal(MESH)) * np.sqrt(pk / vcell)
    dk[0, 0, 0] = 0.0
    k2[0, 0, 0] = 1.0
    psi = [np.fft.irfftn(1j * K / k2 * dk, s=MESH) for K in (KX, KY, KZ)]
    q = np.meshgrid(*[(np.arange(n) + 0.5) * l / n for n, l in zip(MESH, BOX)], indexing="ij")
    grow = np.interp(q[0], G["xg"], G["growth"])
    pos = [(qi + grow * p) for qi, p in zip(q, psi)]
    # cloud-in-cell on the periodic mesh
    rho = np.zeros(MESH)
    cell = [p / (l / n) - 0.5 for p, l, n in zip(pos, BOX, MESH)]
    i0 = [np.floor(c).astype(int) for c in cell]
    fr = [c - i for c, i in zip(cell, i0)]
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = ((fr[0] if dx else 1 - fr[0]) * (fr[1] if dy else 1 - fr[1])
                     * (fr[2] if dz else 1 - fr[2]))
                idx = tuple(((i + d) % n).ravel() for i, d, n in zip(i0, (dx, dy, dz), MESH))
                np.add.at(rho, idx, w.ravel())
    proj = rho.sum(axis=2)
    delta = proj / proj.mean() - 1.0
    CACHE.mkdir(exist_ok=True)
    np.savez(npz, delta=delta)
    return delta


# ------------------------------------------------------------------ the shells
def shells():
    npz = CACHE / f"onion_shells_{NSIDE}.npz"
    if npz.exists():
        d = np.load(npz)
        return d["maps"], d["chi"]
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax_fli as jfli
    from datasets import load_dataset
    from jax_fli.io import Catalog

    maps, chis = [], []
    for f in sorted(RUN.glob("shell_*.parquet")):
        s = Catalog.from_dataset(load_dataset("parquet", data_files=str(f), split="train")).field[0]
        chis.append(float(np.asarray(s.comoving_centers)))
        s = s.ud_sample(NSIDE).to(jfli.DensityUnit.OVERDENSITY)
        maps.append(np.asarray(s.array, dtype=np.float32).ravel())
        print(f"{f.name}: chi = {chis[-1]:.0f} Mpc/h")
    o = np.argsort(chis)
    CACHE.mkdir(exist_ok=True)
    np.savez(npz, maps=np.stack(maps)[o], chi=np.array(chis)[o])
    return np.stack(maps)[o], np.array(chis)[o]


G = geometry()
print("bin depths chi_i [Mpc/h]:", G["chi_end"].round(0), " z:", G["z_end"].round(2))
DELTA = strip(G)
MAPS, SHELL_CHI = shells()

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle
from PIL import Image

slide_style(scale=1.2)
# the five states must keep the same frame to stack, so no tight crop
plt.rcParams["savefig.bbox"] = "standard"
cmap = plt.get_cmap("YlOrRd")
COLORS = [cmap(x) for x in np.linspace(0.35, 0.95, 4)]
XMAX = BOX[0]


# ------------------------------------------------------------------ strip + kernel states
def eye(fig, x, y, w):
    """The observer's eye of the lightcone slide, in figure coordinates."""
    ax = fig.add_axes([x - w / 2, y - w / 2, w, w])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    t = np.linspace(-1, 1, 200)
    lid = 0.47 * np.cos(t * np.pi / 2) ** 1.5
    ax.plot(0.9 * t, 0.9 * lid, color="#3B6FB6", lw=2.4, solid_capstyle="round")
    ax.plot(0.9 * t, -0.9 * lid, color="#3B6FB6", lw=2.4, solid_capstyle="round")
    ax.add_patch(plt.Circle((0, 0), 0.22, fc="#3B6FB6", ec="none"))


v = np.log1p(np.clip(DELTA, -0.99, None))
lo, hi = np.percentile(v, [1, 99.6])
qmax = float(G["q"].max())

for state in range(5):
    fig = plt.figure(figsize=(9.6, 6.4))
    ax_s = fig.add_axes([0.115, 0.615, 0.865, 0.30])
    ax_q = fig.add_axes([0.115, 0.11, 0.865, 0.445], sharex=ax_s)
    ax_s.imshow(v.T, origin="lower", extent=(0, BOX[0], -BOX[1] / 2, BOX[1] / 2), cmap="magma",
                vmin=lo, vmax=hi, aspect="auto", interpolation="bilinear")
    ax_s.set_yticks([])
    ax_s.tick_params(axis="x", labelbottom=False, length=0)
    for sp in ax_s.spines.values():
        sp.set_color("#2b3342")
    ax_s.set_title("matter overdensity $\\delta$ along the line of sight", fontsize=15, color=INK,
                   loc="left", pad=6)
    eye(fig, 0.052, 0.78, 0.07)
    fig.text(0.052, 0.725, "observer", ha="center", va="top", fontsize=12, color=INK)

    for i in range(4):
        x = G["chi"]
        if state == 0:
            continue
        if i < state - 1:
            ax_q.plot(x, G["q"][i], color=COLORS[i], lw=1.6, alpha=0.45)
        elif i == state - 1:
            ax_q.fill_between(x, 0, G["q"][i], facecolor=to_rgba(COLORS[i], 0.22), edgecolor="none")
            ax_q.plot(x, G["q"][i], color=COLORS[i], lw=3.0)
            k = int(np.argmax(G["q"][i]))
            ax_q.text(x[k], G["q"][i][k] + 0.04 * qmax, f"$q_{i + 1}$", color=COLORS[i],
                      fontsize=17, ha="center", va="bottom", fontweight="bold")
    if state:
        c = G["chi_end"][state - 1]
        col = COLORS[state - 1]
        ax_s.axvspan(c, XMAX, color=BG, alpha=0.8, lw=0)
        ax_s.add_patch(Rectangle((4, -BOX[1] / 2 + 4), c - 8, BOX[1] - 8, fill=False, ec=col, lw=3.4))
        ax_q.axvline(c, color=col, lw=2.0, ls=(0, (1.2, 1.6)))
        right = c > 0.85 * XMAX               # near the edge, the label goes left of the line
        ax_q.text(c - 25 if right else c + 25, 0.97, f"bin {state}", transform=ax_q.get_xaxis_transform(),
                  color=col, fontsize=14, ha="right" if right else "left", va="top")
    ax_q.set_xlim(0, XMAX)
    ax_q.set_ylim(0, qmax * 1.22)
    ax_q.set_yticks([])
    ax_q.set_ylabel("lensing kernel $q_i(\\chi)$", fontsize=14)
    ax_q.set_xlabel("comoving distance $\\chi$  [Mpc/$h$]")
    ax_q.spines[["top", "right"]].set_visible(False)
    ax_q.tick_params(axis="x", top=False)
    fig.savefig(HERE / f"tomo_{state}.png", dpi=170, transparent=True)
    plt.close(fig)
    print(f"wrote tomo_{state}.png")


# ------------------------------------------------------------------ the cutaway sphere
RADII = np.asarray(G["chi_end"], float)          # one shell per bin, at its depth
R_OUT = float(RADII[-1])
THICK = 0.035 * R_OUT
# the run shell nearest to each depth, each used once (bin 4 lies beyond the run's 2500 Mpc/h)
TEX = []
for r in RADII:
    order = np.argsort(np.abs(SHELL_CHI - r))
    TEX.append(int(next(j for j in order if j not in TEX)))
print("textures: run shells", TEX, "at chi =", SHELL_CHI[TEX].round(0))
PX = 1400


def normalise(m):
    v = hp.smoothing(np.log1p(np.clip(m, -0.99, None)).astype(float), fwhm=np.radians(0.7))
    a, b = np.percentile(v, [2, 99.5])
    return np.clip((v - a) / (b - a), 0, 1)


TEXTURES = [normalise(MAPS[j]) for j in TEX]
MAGMA = plt.get_cmap("magma")
RIM = np.array(MAGMA(0.86)[:3])        # the cut faces, one light tone as in the GLASS figure

az, el = np.radians(40.0), np.radians(24.0)
cam = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
fwd = -cam
right = np.cross(fwd, [0, 0, 1.0])
right /= np.linalg.norm(right)
up = np.cross(right, fwd)
light = cam + 0.55 * up - 0.35 * right
light /= np.linalg.norm(light)


def in_cut(p):
    return (p[..., 0] > 0) & (p[..., 1] > 0) & (p[..., 2] > 0)


def render(n_shells, out):
    ext = 1.04 * R_OUT
    u = np.linspace(-ext, ext, PX)
    U, V = np.meshgrid(u, -u)
    o = U[..., None] * right + V[..., None] * up + 3 * R_OUT * cam
    best_t = np.full(U.shape, np.inf)
    rgb = np.zeros(U.shape + (3,))
    for k in range(n_shells):
        r_out, r_in = RADII[k], RADII[k] - THICK
        tex = TEXTURES[k]
        cands = []
        # the two spherical faces of the shell, each hit on its near and far side
        b = (o * fwd).sum(-1)
        cc = (o * o).sum(-1)
        for r, sign in ((r_out, 1.0), (r_in, -1.0)):
            disc = b * b - (cc - r * r)
            ok = disc > 0
            sq = np.sqrt(np.where(ok, disc, 0))
            for t, face in ((-b - sq, 1.0), (-b + sq, -1.0)):
                p = o + t[..., None] * fwd
                valid = ok & ~in_cut(p)
                n = sign * p / r
                # outward-facing (lit) on the near side of the outer sphere, shadowed inside
                cands.append((t, valid, p, n, 1.0 if sign * face > 0 else 0.5, None))
        # the three faces left by the cut, annuli between r_in and r_out
        for ax_i in range(3):
            denom = fwd[ax_i]
            t = -o[..., ax_i] / denom
            p = o + t[..., None] * fwd
            rr = np.linalg.norm(p, axis=-1)
            others = [j for j in range(3) if j != ax_i]
            valid = (p[..., others[0]] >= 0) & (p[..., others[1]] >= 0) & (rr >= r_in) & (rr <= r_out)
            n = np.zeros(3)
            n[ax_i] = 1.0
            cands.append((t, valid, p, np.broadcast_to(n, p.shape), 1.0, RIM))
        for t, valid, p, n, boost, flat in cands:
            take = valid & (t < best_t) & (t > 0)
            if not take.any():
                continue
            pv = p[take]
            pix = hp.vec2pix(NSIDE, pv[:, 0], pv[:, 1], pv[:, 2])
            col = MAGMA(0.08 + 0.9 * tex[pix])[:, :3] if flat is None else np.broadcast_to(flat, (len(pv), 3))
            lam = np.abs((n[take] * light).sum(-1))
            shade = np.clip((0.42 + 0.58 * lam) * boost, 0, 1.25)
            rgb[take] = np.clip(col * shade[:, None], 0, 1)
            best_t[take] = t[take]
    alpha = np.isfinite(best_t)
    img = np.zeros(U.shape + (4,), dtype=np.uint8)
    img[..., :3] = (rgb * 255).astype(np.uint8)
    img[..., 3] = alpha * 255
    im = Image.fromarray(img).resize((PX // 2, PX // 2), Image.LANCZOS)
    im.save(HERE / out)
    print(f"wrote {out}")
    return im


for i in range(4):
    render(i + 1, f"onion_{i + 1}.png")
