#!/usr/bin/env python3
# ENV: shared
"""
The LCDM predictions for the CMB spectra, and where r sits among them.

TT, TE (|TE|: solid where positive, dashed where negative), EE, the B modes made
by lensing of E, and the primordial B modes for r = 0.003 (thick) and r = 0.01
(thin), with the reionisation and recombination bumps marked. The point: r only
enters C_l^BB, four to seven orders of magnitude below TT and under the lensing
B modes over most of the range.

CAMB, Planck 2018 cosmology, lensed scalar spectra plus tensors at r = 0.01
(rescaled for other r), cached in .cache/. Three build steps share the same axes
so they overlay in an r-stack:

  cl_step1.svg   TT, TE, EE
  cl_step2.svg   + lensing B modes
  cl_step3.svg   + primordial B modes and the two bumps (the full figure)
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from _common import GREY, INK, RED, skip_if_built, slide_style

OUTS = ("cl_step1.svg", "cl_step2.svg", "cl_step3.svg")
CACHE = HERE / ".cache"
skip_if_built(HERE, *OUTS)

R_REF = 0.01


def spectra():
    npz = CACHE / "camb_cl_all.npz"
    if npz.exists():
        d = np.load(npz)
        return {k: d[k] for k in d.files}
    import camb
    pars = camb.set_params(H0=67.36, ombh2=0.02237, omch2=0.1200,
                           ns=0.9649, As=2.1e-9, tau=0.0544, r=R_REF)
    pars.WantTensors = True
    pars.set_for_lmax(2500, lens_potential_accuracy=2)
    pars.max_l_tensor = 2500          # CAMB truncates tensors near l ~ 600 by default
    pars.max_eta_k_tensor = 6000.0
    cls = camb.get_results(pars).get_cmb_power_spectra(pars, CMB_unit="muK", raw_cl=False)
    ls = cls["lensed_scalar"]
    out = {"ell": np.arange(ls.shape[0]), "TT": ls[:, 0], "EE": ls[:, 1],
           "TE": ls[:, 3], "BB_lens": ls[:, 2], "BB_tensor": cls["tensor"][:, 2]}
    CACHE.mkdir(exist_ok=True)
    np.savez(npz, **out)
    return out


s = spectra()
ell = s["ell"]
sl = slice(2, 2501)
L = ell[sl]

slide_style(scale=1.2)
import matplotlib.pyplot as plt

C_TT, C_TE, C_EE = "#F2C48D", "#A9ACD6", "#C9C9C9"


def draw(step):
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    ax.loglog(L, s["TT"][sl], color=C_TT, lw=2.6, label=r"$TT$")
    te = s["TE"][sl]
    pos = np.where(te > 0, np.abs(te), np.nan)
    neg = np.where(te < 0, np.abs(te), np.nan)
    ax.loglog(L, pos, color=C_TE, lw=2.2, label=r"$TE$")
    ax.loglog(L, neg, color=C_TE, lw=2.2, ls=(0, (4, 2.5)))
    ax.loglog(L, s["EE"][sl], color=C_EE, lw=2.4, label=r"$EE$")
    if step >= 2:
        ax.loglog(L, s["BB_lens"][sl], color=GREY, lw=2.4, label=r"lensing $B$ modes")
    if step >= 3:
        for r, lw in ((0.003, 4.0), (0.01, 1.6)):
            ax.loglog(L, s["BB_tensor"][sl] * r / R_REF, color=RED, lw=lw,
                      label=rf"primordial $B$ modes, $r={r:g}$")
        ax.annotate("reionisation\nbump", xy=(3.3, 2.4e-5), xytext=(5.2, 1.2e-6), color=RED,
                    fontsize=13, fontweight="bold", ha="center", va="top", linespacing=1.2,
                    arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.2, mutation_scale=18))
        ax.annotate("recombination\nbump", xy=(85, 1.3e-4), xytext=(85, 4e-6), color=RED,
                    fontsize=13, fontweight="bold", ha="center", va="top", linespacing=1.2,
                    arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.2, mutation_scale=18))
    # invisible stand-ins keep the legend (and so the bounding box) identical in every step
    if step < 2:
        ax.loglog([], [], color=GREY, lw=2.4, label=r"lensing $B$ modes", alpha=0)
    if step < 3:
        for r, lw in ((0.003, 4.0), (0.01, 1.6)):
            ax.loglog([], [], color=RED, lw=lw, alpha=0, label=rf"primordial $B$ modes, $r={r:g}$")
    ax.grid(True, which="major", color="#DDDDDD", lw=0.8)
    ax.set_xlabel(r"multipole $\ell$")
    ax.set_ylabel(r"$D_\ell\ [\mu\mathrm{K}^2]$")
    ax.set_xlim(2, 2500)
    ax.set_ylim(1e-7, 2e4)
    ax.tick_params(colors=INK)
    leg = ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), labelcolor=INK,
                    fontsize=13, handlelength=2.0, labelspacing=0.55)
    for h, t in zip(leg.legend_handles, leg.get_texts()):
        if h.get_alpha() == 0:
            t.set_alpha(0)
    fig.savefig(HERE / f"cl_step{step}.svg", transparent=True, bbox_inches=None)
    plt.close(fig)


for step in (1, 2, 3):
    draw(step)
print("wrote", ", ".join(OUTS))
