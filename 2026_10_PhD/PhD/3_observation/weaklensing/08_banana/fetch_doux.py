#!/usr/bin/env python3
# ENV: shared
"""
The two-point banana: DES Y3 cosmic shear C_ell constraints on Omega_m, sigma_8 and S_8,
against Planck 2018.

The figure is the paper's own (fig:cont_lcdm_cl_planck, figs/cont_lcdm_cl_planck.pdf) from
Doux et al. 2022, "Dark Energy Survey Year 3 results: cosmological constraints from the analysis
of cosmic shear in harmonic space", arXiv:2203.07128. It is taken unchanged from the arXiv source;
the e-print is kept in .cache/.

Output (this directory): doux_banana.svg
"""

import io
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from _common import cached_fetch, skip_if_built

OUT = "doux_banana.svg"
skip_if_built(HERE, OUT)

raw = cached_fetch(HERE / ".cache", "doux2022_eprint", "https://arxiv.org/e-print/2203.07128")
with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as tar:
    pdf = tar.extractfile("figs/cont_lcdm_cl_planck.pdf").read()

with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
    tmp.write(pdf)
    tmp.flush()
    subprocess.run(["pdftocairo", "-svg", tmp.name, str(HERE / OUT)], check=True)
print(f"wrote {OUT}")
