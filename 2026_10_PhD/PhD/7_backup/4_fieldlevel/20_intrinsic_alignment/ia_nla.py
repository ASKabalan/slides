#!/usr/bin/env python3
# ENV: jax-fli
"""
Intrinsic alignment in the forward model, for the backup slide: the non-linear alignment (NLA) model
applied shell by shell on the sphere, next to the Born lensing sum, both from the same density
shells, then one Kaiser-Squires step to the shear.

  kappa_lens  = born(cosmo, lightcone, n(z))                        (jax_fli.born)
  kappa_IA    = sum_j  p_j F(z_j) delta_j ,  p_j = int_shell j n(z) dz  (the bin's sources in shell j)
  F(z)        = -A_IA C1 rho_crit Omega_m / D(z) ((1 + z) / (1 + z0))^eta_IA
  (gamma1, gamma2) = (kappa_lens + kappa_IA).get_shear()            (jax_fli Kaiser-Squires)

The thin-shell (delta-NLA) form: each shell's transverse tidal field is the spin-2 derivative of its
projected potential, the operator that takes kappa to gamma, so IA enters as a kappa-like term
weighted by n(z) instead of the lensing kernel. Illustrative values A_IA = 1, eta_IA = 0, z0 = 0.62,
C1 rho_crit = 0.0134.

Data: the twenty equal-volume lightcone shells of jax-fli experiment 05e, 3072^3 mesh (local copy,
the run of the lightcone slide), each brought to nside 256 with SphericalDensity.ud_sample; DES Y3
source bin 3 (jax_fli.data.get_des_y3_nz_shear). Cache: fli/.cache/ia_nla_256.npz.

Output (this directory): ia_nla.png, the diagram on the slide background.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import GREY, INK, KW, KW2, skip_if_built, slide_style

OUT = "ia_nla.png"
skip_if_built(HERE, OUT)

CACHE = ROOT / "5_contribution/fli/.cache"
RUN = Path("/home/wassim/Projects/NBody/jax-fli-experiments/05-spacing-n-stepping/05e-mesh/"
           "density/exp5e_m3072")
NSHELL, NSIDE, BIN = 20, 256, 2          # bin index 2 = DES Y3 bin 3
A_IA, ETA_IA, Z0, C1RHO = 1.0, 0.0, 0.62, 0.0134
SHOW = (2, 7, 13)                        # the three shells drawn, near to far


def compute():
    npz = CACHE / f"ia_nla_{NSIDE}.npz"
    if npz.exists():
        d = np.load(npz)
        return {k: d[k] for k in d.files}
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import jax_cosmo as jc
    import jax_fli as jfli
    from datasets import load_dataset
    from jax_fli.data import get_des_y3_nz_shear
    from jax_fli.fields import SphericalDensity, SphericalKappaField
    from jax_fli.io import Catalog

    shells, cosmo = [], None
    for i in range(NSHELL):
        cat = Catalog.from_dataset(load_dataset("parquet", data_files=str(RUN / f"shell_{i:04d}.parquet"),
                                                split="train"))
        cosmo = cat.cosmology[0]
        shells.append(cat.field[0].ud_sample(NSIDE).replace(name="exp5e_m3072"))   # stack needs one name
        print(f"shell {i}: z = {float(np.asarray(shells[-1].z_sources).ravel()[0]):.3f}")
    order = np.argsort([float(np.asarray(s.comoving_centers)) for s in shells])
    lc = SphericalDensity.stack([shells[k] for k in order])

    nz = get_des_y3_nz_shear()[BIN]
    kappa_lens = jfli.born(cosmo, lc, [nz])

    # NLA: the bin's source fraction in each shell times F(z) at the shell centre
    delta = np.asarray(lc.to(jfli.DensityUnit.OVERDENSITY).array)            # (S, npix)
    chi_c, width = np.asarray(lc.comoving_centers), np.asarray(lc.density_width)
    edges = np.r_[0.0, np.cumsum(width)]
    a_edges = np.asarray(jc.background.a_of_chi(cosmo, jnp.asarray(np.maximum(edges, 1e-3))))
    z_edges = 1 / a_edges - 1
    a_c = np.asarray(jc.background.a_of_chi(cosmo, jnp.asarray(chi_c)))
    z_c = 1 / a_c - 1
    zz = np.linspace(0.0, 3.0, 3000)
    n = np.asarray(nz(jnp.asarray(zz)))
    n = n / np.trapezoid(n, zz)
    p = np.array([np.trapezoid(n[(zz >= a) & (zz <= b)], zz[(zz >= a) & (zz <= b)])
                  for a, b in zip(z_edges[:-1], z_edges[1:])])
    D = np.asarray(jc.background.growth_factor(cosmo, jnp.asarray(a_c)))
    om = float(cosmo.Omega_m)
    F = -A_IA * C1RHO * om / D * ((1 + z_c) / (1 + Z0)) ** ETA_IA
    kappa_ia = np.tensordot(p * F, delta, axes=1)

    k_lens = np.asarray(kappa_lens.array).reshape(-1)
    k_eff = k_lens + kappa_ia
    eff = SphericalKappaField.FromDensityMetadata(array=jnp.asarray(k_eff[None]), field=kappa_lens)
    gamma = np.asarray(eff.get_shear().array).reshape(2, -1)

    # the two weights as continuous curves in z, for the icons: lensing kernel averaged over n(z)
    # (as on the Born slide) and the IA weight n(z) |F(z)|
    zc = np.linspace(0.01, z_edges[-1], 400)
    chi = np.asarray(jc.background.radial_comoving_distance(cosmo, jc.utils.z2a(jnp.asarray(zc))))
    chi_s = np.asarray(jc.background.radial_comoving_distance(cosmo, jc.utils.z2a(jnp.asarray(zz[1:]))))
    g = np.clip(1 - chi[:, None] / chi_s[None, :], 0, None)
    kern = chi * (1 + zc) * np.trapezoid(g * n[None, 1:], zz[1:], axis=1)
    Dc = np.asarray(jc.background.growth_factor(cosmo, jc.utils.z2a(jnp.asarray(zc))))
    w_ia = np.interp(zc, zz, n) * C1RHO * om / Dc * A_IA

    out = dict(delta=delta[list(SHOW)], z_show=z_c[list(SHOW)], k_lens=k_lens, k_ia=kappa_ia, gamma=gamma,
               zc=zc, kern=kern, w_ia=w_ia, z_c=z_c, p=p, F=F)
    CACHE.mkdir(exist_ok=True)
    np.savez(npz, **out)
    return compute()


R = compute()
print("shell z:", R["z_c"].round(2))
print("source fraction per shell:", R["p"].round(3), "sum", R["p"].sum().round(3))
print("F(z):", R["F"].round(4))
print("rms kappa_lens %.2e  kappa_IA %.2e  gamma %.2e" % (R["k_lens"].std(), R["k_ia"].std(), R["gamma"].std()))
print("corr(kappa_lens, kappa_IA) = %.2f" % np.corrcoef(R["k_lens"], R["k_ia"])[0, 1])

# --------------------------------------------------------------------------- the diagram
slide_style()
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

plt.rcParams["savefig.bbox"] = None
BG = "#faf7f0"
W, H = 10.4, 5.6
fig = plt.figure(figsize=(W, H), dpi=200, facecolor=BG)
moll = hp.projector.MollweideProj(xsize=500)
MAPW = 0.185                      # map width in figure units; Mollweide is 2:1
MAPH = MAPW * W / H / 2


def mollmap(x, y, m, cmap, sym=False, title=None, tcolor=INK, pct=(1, 99.5)):
    """A Mollweide map with its lower-left corner at (x, y), outlined, title above."""
    img = moll.projmap(m, lambda a, b, c: hp.vec2pix(NSIDE, a, b, c))
    img = np.ma.masked_invalid(np.where(np.isfinite(img), img, np.nan))
    if sym:
        v = np.percentile(np.abs(m), 99.5)
        lo, hi = -v, v
    else:
        lo, hi = np.percentile(m, pct)
    ax = fig.add_axes([x, y, MAPW, MAPH])
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(alpha=0)
    ax.imshow(img, origin="lower", cmap=cm, vmin=lo, vmax=hi, interpolation="bilinear")
    h, w = img.shape
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(w / 2 + (w / 2 - 1) * np.cos(t), h / 2 + (h / 2 - 1) * np.sin(t), color="#3b3b3b", lw=0.7)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=13, color=tcolor, pad=2)
    return ax


def arrow(x0, y0, x1, y1, label=None, dy=0.02, color=GREY):
    fig.patches.append(FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=14, lw=1.6, color=color, shrinkA=0, shrinkB=0))
    if label:
        fig.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, label, ha="center", va="bottom", fontsize=12, color=INK)


def header(x, text):
    fig.text(x, 0.955, text, ha="center", va="center", fontsize=14.5, color=INK, weight="bold")


TOP, BOT = 0.60, 0.17             # bottoms of the lensing and IA rows
# 1. three of the twenty density shells, near to far
header(0.105, "density shells")
for k, (d, z) in enumerate(zip(R["delta"], R["z_show"])):
    mollmap(0.012, 0.66 - k * 0.27, d, "magma", title=rf"$\delta_j$,  $z = {z:.2f}$")
fig.text(0.105, 0.035, "20 shells, $3072^3$ mesh", ha="center", fontsize=11, color=GREY)

# 2. the two weights per shell
header(0.355, "weight of each shell")
for y0, curve, title, col in ((TOP, R["kern"], "lensing kernel", KW2),
                              (BOT, R["w_ia"], r"sources $n(z)\,\times\,|F(z)|$", KW)):
    ax = fig.add_axes([0.265, y0 + 0.02, 0.18, 0.17])
    ax.fill_between(R["zc"], curve / curve.max(), color=col, alpha=0.25, lw=0)
    ax.plot(R["zc"], curve / curve.max(), color=col, lw=2.2)
    ax.set_xlim(0, R["zc"][-1])
    ax.set_ylim(0, 1.12)
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1.0])
    ax.tick_params(labelsize=10.5, length=3)
    ax.set_xlabel("$z$", fontsize=11.5, labelpad=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, fontsize=13, color=col, pad=3)
fig.text(0.355, BOT - 0.115, r"$F(z) = -\,A_\mathrm{IA}\,C_1\rho_\mathrm{crit}\,\Omega_m\,/\,D(z)$",
         ha="center", fontsize=12.5, color=KW)

# 3. the two sums over shells
header(0.6, "sum over shells")
mollmap(0.507, TOP + 0.02, R["k_lens"], "magma", title=r"$\kappa_\mathrm{lens}$", tcolor=KW2)
# kappa_IA ~ -delta: the reversed map shows dense regions bright, as in the shells
mollmap(0.507, BOT + 0.02, R["k_ia"], "magma_r", title=r"$\kappa_\mathrm{IA}$", tcolor=KW, pct=(0.5, 99))
fig.text(0.6, BOT - 0.045, r"$\propto -\delta$ (colours reversed), $\sim$10 % of $\kappa_\mathrm{lens}$",
         ha="center", fontsize=10.5, color=GREY)

# 4. the shear, after one Kaiser-Squires step on the sum
header(0.895, "shear")
mollmap(0.805, 0.53, R["gamma"][0], "RdBu_r", sym=True, title=r"$\gamma_1$")
mollmap(0.805, 0.24, R["gamma"][1], "RdBu_r", sym=True, title=r"$\gamma_2$")

# arrows: shells -> weights -> maps -> sum -> shear
yt, yb = TOP + 0.02 + MAPH / 2, BOT + 0.02 + MAPH / 2
arrow(0.2, 0.5, 0.262, yt)
arrow(0.2, 0.5, 0.262, yb)
arrow(0.452, yt, 0.503, yt)
arrow(0.452, yb, 0.503, yb)
fig.text(0.735, 0.5, "+", ha="center", va="center", fontsize=22, color=INK)
fig.patches.append(plt.Circle((0.735, 0.5), 0.022, transform=fig.transFigure, fill=False, lw=1.4, color=INK))
arrow(0.695, yt, 0.722, 0.522)
arrow(0.695, yb, 0.722, 0.478)
arrow(0.758, 0.5, 0.80, 0.5)
fig.text(0.735, 0.455, "Kaiser–Squires\nget_shear()", ha="center", va="top", fontsize=10, color=INK,
         linespacing=1.15)

fig.savefig(HERE / OUT, facecolor=BG)
plt.close(fig)
print(f"wrote {OUT}")
