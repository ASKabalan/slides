#!/usr/bin/env python3
# ENV: jax-fli
"""
How many particle-mesh steps the two integrators need, for the backup slide on integrators: the slide
version of the thesis figure chap6/step_convergence.pdf (These_wassim/figures/chap6/step_convergence.py).

jax-fli experiment 04 (ASKabalan/jax-fli-experiments, 04-step-size/density_spectra): a 10-shell
lightcone, 2048^3 mesh in a 2000 Mpc/h box, nside 2048, CIC painting, run with BullFrog stepped in
the growth factor and with DoubleKickDrift stepped in the scale factor, at 10, 20, 30 and 50 steps.
Top, the spectrum of the near, middle and far shells (1, 5, 9) at 10, 20 and 30 steps; bottom, each
over the same solver's 50-step run, minus one, over a grey band at +-2 %. Bands of 32 multipoles.

Output (this directory): step_convergence.svg
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import BLUE, INK, KW, skip_if_built, slide_style

OUT = "step_convergence.svg"
skip_if_built(HERE, OUT)

CACHE = (ROOT / "5_contribution/fli") / ".cache"
REPO = "ASKabalan/jax-fli-experiments"
SOLVERS = {"bfd": ("BullFrog, steps in $D$", BLUE), "kdk": ("kick-drift-kick, steps in $a$", KW)}
STEPS, REF = (10, 20, 30), 50
STYLES = {10: ":", 20: "-", 30: "--"}
SHELLS = (("near", 1), ("middle", 5), ("far", 9))


def load():
    npz = CACHE / "step_convergence_04.npz"
    if npz.exists():
        d = np.load(npz)
        return {k: d[k] for k in d.files}
    import jax

    jax.config.update("jax_enable_x64", True)
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from jax_fli.io import Catalog

    pat = "04-step-size/density_spectra/spectra_{sol}_s{st}.parquet"
    root = snapshot_download(REPO, repo_type="dataset",
                             allow_patterns=[pat.format(sol=s, st=t) for s in SOLVERS for t in (*STEPS, REF)])
    out = {}
    for sol in SOLVERS:
        for st in (*STEPS, REF):
            f = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{pat.format(sol=sol, st=st)}",
                                                  split="train")).field[0]
            b = f.bin(nlb=32, lmin=2)
            out["ell"] = np.asarray(b.wavenumber)
            out[f"{sol}_{st}"] = np.asarray(b.array)
            out["z"], out["chi"] = np.asarray(f.z_sources), np.asarray(f.comoving_centers)
    CACHE.mkdir(exist_ok=True)
    np.savez(npz, **out)
    return out


D = load()
ELL = D["ell"]
DL = ELL * (ELL + 1) / (2 * np.pi)

slide_style()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter

plt.rcParams["savefig.bbox"] = None
fig = plt.figure(figsize=(10.4, 5.6))
for c, (name, sh) in enumerate(SHELLS):
    x0 = 0.075 + c * 0.31
    ax = fig.add_axes([x0, 0.43, 0.27, 0.45])
    ar = fig.add_axes([x0, 0.1, 0.27, 0.27])
    for sol, (lab, col) in SOLVERS.items():
        for st in STEPS:
            ax.loglog(ELL, DL * D[f"{sol}_{st}"][sh], color=col, ls=STYLES[st], lw=1.8)
            ar.semilogx(ELL, D[f"{sol}_{st}"][sh] / D[f"{sol}_{REF}"][sh] - 1, color=col, ls=STYLES[st], lw=1.6)
    ar.axhspan(-0.02, 0.02, color="#d8dde6", lw=0)
    ar.axhline(0, color=INK, lw=0.7)
    for a in (ax, ar):
        a.set_xlim(max(2.0, ELL.min() * 0.8), ELL.max())
    ax.tick_params(labelbottom=False)
    ax.yaxis.set_major_locator(LogLocator(numticks=3))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ar.set_ylim(-0.12, 0.12)
    ar.set_yticks([-0.1, 0, 0.1])
    ar.set_xlabel(r"$\ell$", labelpad=0)
    ax.set_title(rf"{name} shell, $z = {D['z'][sh]:.2f}$", fontsize=14, color=INK, pad=5)
    if c == 0:
        ax.set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$")
        ar.set_ylabel(r"$C_\ell / C_\ell^{50} - 1$", fontsize=12)
handles = [Line2D([], [], color=col, lw=2.4, label=lab) for lab, col in SOLVERS.values()]
handles += [Line2D([], [], color=INK, ls=STYLES[st], lw=1.8, label=f"{st} steps") for st in STEPS]
fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=12, frameon=False,
           bbox_to_anchor=(0.5, 1.0), handlelength=2.0, columnspacing=1.2)
fig.savefig(HERE / OUT)
for sol in SOLVERS:
    print(sol, "max |C/C50 - 1| near shell:", [round(float(np.abs(D[f"{sol}_{st}"][1] / D[f"{sol}_{REF}"][1] - 1).max()), 3) for st in STEPS])
print(f"wrote {OUT}")
