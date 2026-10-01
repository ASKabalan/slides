#!/usr/bin/env python3
# ENV: jax-fli
"""
What a map-level scale cut does, for the backup slide on scale cuts (thesis section 6.6.4, figure
chap6/scalecut_spectra.pdf, script These_wassim/figures/chap6/scalecut_spectra.py).

The convergence of the thesis run (jax-fli experiment 05c, equal volume, 20 drifted shells,
born_gl_drift_20, stage-3 bins) and CosmoGrid through the same Born integral (cosmo_172798,
kappa_born_s3), each bin brought to nside 512. The cut transforms a map to spherical harmonics,
multiplies a_lm by the cosine taper T_l (1 up to l_cut - l_width, 0 from l_cut, l_width = 50) and
transforms back: bin 2 is cut at l_cut = 200, bin 3 at 250, as in the thesis.

  scalecut_map.png       bin 3 before the cut, a 15 degree gnomonic cut-out (magma)
  scalecut_alm.png       its |a_lm| in the s2fft layout (rows l, columns m), l < 400, log scale
  scalecut_alm_cut.png   the same after the taper; the removed multipoles are greyed
  scalecut_map_cut.png   the band-limited map, same cut-out and colour scale as the first
  scalecut_spectra.svg   bins 2 and 3 after the cut: band powers of this work (solid) and CosmoGrid
                         (dashed), bands of 10 multipoles from l = 20, and their ratio minus one
                         over a +-10 % band; the grey band is each bin's roll-off
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from _common import BLUE, GREY, INK, KW, skip_if_built, slide_style

OUTS = ["scalecut_map.png", "scalecut_alm.png", "scalecut_alm_cut.png", "scalecut_map_cut.png",
        "scalecut_spectra.svg"]
skip_if_built(HERE, *OUTS)

CACHE = HERE.parent / ".cache"
EXP = Path("/home/wassim/Projects/NBody/jax-fli-experiments")
MAPS = {"this work": EXP / ("05-spacing-n-stepping/05c-equal-volume/kappa_gauss_legendre/born_gl_drift_20/"
                            "BORN_kappa_gl_drift_3bin_20.parquet"),
        "CosmoGrid": EXP / "00-cosmogrid/cosmo_172798/kappa/kappa_born_s3.parquet"}
NSIDE, LMAX, L_WIDTH = 512, 3 * 512 - 1, 50
L_CUT = {1: 200, 2: 250}                      # bin index -> l_cut (bins 2 and 3)
L_SHOW, CUT_PIX, CUT_RESO = 400, 300, 3.0     # a_lm rows shown; a 15 degree cut-out


def taper(lmax, l_cut, l_width=L_WIDTH):
    ell = np.arange(lmax + 1, dtype=float)
    x = (ell - (l_cut - l_width)) / l_width
    return np.where(ell <= l_cut - l_width, 1.0, np.where(ell >= l_cut, 0.0, 0.5 * (1 + np.cos(np.pi * x))))


def load():
    npz = CACHE / "scalecut_05c.npz"
    if npz.exists():
        d = np.load(npz)
        return {k: d[k] for k in d.files}
    import healpy as hp
    from jax_fli.io import Catalog

    out = {}
    proj = hp.projector.GnomonicProj(rot=(0.0, 0.0), xsize=CUT_PIX, reso=CUT_RESO)
    for name, path in MAPS.items():
        field = Catalog.from_parquet(str(path)).field[0]
        for b, l_cut in L_CUT.items():
            m = hp.ud_grade(np.asarray(field.array[b], dtype=np.float64), NSIDE)
            alm = hp.map2alm(m, lmax=LMAX, iter=3)
            alm_cut = hp.almxfl(alm, taper(LMAX, l_cut))
            m_cut = hp.alm2map(alm_cut, NSIDE, lmax=LMAX)
            out[f"cl_{name}_{b}"] = hp.alm2cl(alm_cut)
            if name == "this work" and b == 2:
                look = lambda mm: proj.projmap(mm, lambda x, y, z: hp.vec2pix(NSIDE, x, y, z))
                out["map"], out["map_cut"] = look(m), look(m_cut)
                L = L_SHOW
                el, em = hp.Alm.getlm(LMAX)
                keep = el < L
                for key, a in (("alm", alm), ("alm_cut", alm_cut)):
                    flm = np.full((L, 2 * L - 1), np.nan)
                    flm[el[keep], L - 1 + em[keep]] = np.abs(a[keep])
                    flm[el[keep], L - 1 - em[keep]] = np.abs(a[keep])
                    out[key] = flm
        del field
    CACHE.mkdir(exist_ok=True)
    np.savez(npz, **out)
    return out


D = load()

slide_style()
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter, ScalarFormatter

plt.rcParams["savefig.bbox"] = None

# --- the two cut-outs, one colour scale
lo, hi = np.nanpercentile(D["map"], [1, 99.5])
for key, out in (("map", OUTS[0]), ("map_cut", OUTS[3])):
    fig = plt.figure(figsize=(3, 3), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(D[key], cmap="magma", origin="lower", vmin=lo, vmax=hi, interpolation="bilinear")
    ax.axis("off")
    fig.savefig(HERE / out, transparent=True)
    plt.close(fig)

# --- |a_lm| before and after the taper, one colour scale; the removed part is greyed
la = np.log10(D["alm"])
vmin, vmax = np.nanpercentile(la, [5, 99.5])
L = L_SHOW
for key, out in (("alm", OUTS[1]), ("alm_cut", OUTS[2])):
    fig = plt.figure(figsize=(3.6, 3.0), dpi=150)
    ax = fig.add_axes([0.16, 0.14, 0.8, 0.82])
    a = np.log10(np.where(D[key] > 0, D[key], np.nan))
    if key == "alm_cut":
        removed = np.isfinite(la) & ~(D[key] > np.nanmax(D["alm"]) * 1e-12)
        grey = np.where(removed, 1.0, np.nan)
        ax.imshow(grey, cmap="Greys", vmin=0, vmax=4, origin="upper", aspect="auto", interpolation="nearest",
                  extent=(-(L - 0.5), L - 0.5, L - 0.5, -0.5))
        ax.axhline(L_CUT[2], color=KW, lw=1.6, ls="--")
    ax.imshow(a, cmap="magma", origin="upper", aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax,
              extent=(-(L - 0.5), L - 0.5, L - 0.5, -0.5))
    ax.set_xlabel("m", fontsize=12, labelpad=1)
    ax.set_ylabel("ℓ", fontsize=12, labelpad=1)
    ax.set_xticks([-(L - 1), 0, L - 1])
    ax.set_yticks([0, 200, L - 1] if key == "alm" else [0, L_CUT[2], L - 1])
    ax.tick_params(labelsize=10, colors=GREY, length=2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.savefig(HERE / out, transparent=True)
    plt.close(fig)

# --- the spectra after the cut
ell = np.arange(LMAX + 1)


def band(y):
    keep = ell >= 20
    e, v = ell[keep], y[keep]
    n = (len(e) // 10) * 10
    return e[:n].reshape(-1, 10).mean(1), v[:n].reshape(-1, 10).mean(1)


fig = plt.figure(figsize=(6.4, 3.6))
for c, (b, l_cut) in enumerate(L_CUT.items()):
    x0 = 0.12 + c * 0.45
    ax = fig.add_axes([x0, 0.42, 0.38, 0.48])
    ar = fig.add_axes([x0, 0.14, 0.38, 0.24])
    dl = {}
    for name, style in (("CosmoGrid", dict(color=KW, ls="--")), ("this work", dict(color=BLUE, ls="-"))):
        cen, dl[name] = band(ell * (ell + 1) / (2 * np.pi) * D[f"cl_{name}_{b}"])
        ax.loglog(cen, dl[name], lw=2.0, label=name, **style)
    below = cen < l_cut
    ar.semilogx(cen[below], dl["this work"][below] / dl["CosmoGrid"][below] - 1, color=BLUE, lw=1.8)
    ar.axhspan(-0.1, 0.1, color="#d8dde6", lw=0)
    ar.axhline(0, color=INK, lw=0.8)
    ar.set_ylim(-0.3, 0.3)
    for a in (ax, ar):
        a.axvspan(l_cut - L_WIDTH, l_cut, color=GREY, alpha=0.18, lw=0)
        a.set_xlim(20, 310)
        a.xaxis.set_major_locator(FixedLocator([30, 100, 300]))
        a.xaxis.set_major_formatter(ScalarFormatter())
        a.xaxis.set_minor_formatter(NullFormatter())
    keep = cen <= l_cut - L_WIDTH
    ax.set_ylim(dl["this work"][keep].min() / 3, dl["this work"][keep].max() * 2.5)
    ax.yaxis.set_major_locator(LogLocator(numticks=3))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(labelbottom=False, labelsize=10)
    ar.tick_params(labelsize=10)
    ar.set_xlabel(r"$\ell$", labelpad=0)
    ax.set_title(rf"bin {b + 1}, $\ell_\mathrm{{cut}} = {l_cut}$", fontsize=13, color=INK, pad=4)
    r = dl["this work"][keep] / dl["CosmoGrid"][keep] - 1
    print(f"bin {b + 1}: residual median {np.median(r):+.3f}, range {r.min():+.3f} to {r.max():+.3f}")
    if c == 0:
        ax.legend(loc="lower right", fontsize=10.5, handlelength=1.6, borderaxespad=0.2)
        ax.set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$", fontsize=12)
        ar.set_ylabel("ratio − 1", fontsize=12)
fig.savefig(HERE / OUTS[4])
print("wrote", ", ".join(OUTS))
