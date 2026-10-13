#!/usr/bin/env python3
# ENV: shared
"""
Samples of the DES Y3 source redshift distributions from HYPERRANK (Cordero et al. 2022), Fig. 5
of Jeffrey et al. (2024), arXiv:2403.02314: each faint line is one realisation of the sky-averaged
n(z) of a tomographic bin given the photometric data, drawn anew for every simulation in their
simulation-based inference. Taken from the arXiv source (nz_plot.pdf) and converted to SVG.

Output (this directory): hyperrank_nz.svg
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

OUT = "hyperrank_nz.svg"
skip_if_built(HERE, OUT)

URL = "https://arxiv.org/e-print/2403.02314"
src = cached_fetch(ROOT / "5_contribution/fli/.cache", "jeffrey2024_src", URL)
with tarfile.open(fileobj=io.BytesIO(src), mode="r:gz") as tar, tempfile.TemporaryDirectory() as tmp:
    pdf = Path(tmp) / "nz_plot.pdf"
    pdf.write_bytes(tar.extractfile("nz_plot.pdf").read())
    subprocess.run(["pdftocairo", "-svg", str(pdf), str(HERE / OUT)], check=True)
print(f"wrote {OUT}")
