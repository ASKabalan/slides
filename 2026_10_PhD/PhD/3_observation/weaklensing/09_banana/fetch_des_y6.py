#!/usr/bin/env python3
# ENV: shared
"""
The two-point banana: DES Y6 cosmic shear constraints on Omega_m and sigma_8, against the CMB.

The Omega_m-sigma_8 panel of Fig. 3 (fig:results_fiducial) of DES Collaboration 2026, "Dark
Energy Survey Year 6 results: cosmological constraints from cosmic shear", arXiv:2602.10065.
The figure loads two files; figures_final/y6_cmb_omsig8.pdf is the Omega_m-sigma_8 panel on its
own (the S_8 panel is y6_cmb.pdf), so it is taken whole, unchanged and vector, from the arXiv
source. Contours: DES Y6 NLA (orange) and TATT (purple), and the CMB (green), which the caption
describes as Planck 2018, ACT DR6 and SPT-3G, TT+TE+EE+lowE, no lensing. The e-print is kept
in .cache/.

Output (this directory): des_y6_om_s8.svg
"""

import io
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import cached_fetch, skip_if_built

OUT = "des_y6_om_s8.svg"
skip_if_built(HERE, OUT)

raw = cached_fetch(HERE / ".cache", "desy6_shear_eprint", "https://arxiv.org/e-print/2602.10065")
with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as tar:
    pdf = tar.extractfile("figures_final/y6_cmb_omsig8.pdf").read()

with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
    tmp.write(pdf)
    tmp.flush()
    subprocess.run(["pdftocairo", "-svg", tmp.name, str(HERE / OUT)], check=True)
print(f"wrote {OUT}")
