#!/usr/bin/env python3
# ENV: shared
"""
Chapter 4 figures of the thesis, converted for the backup slides.

Each is produced by the script beside it in These_wassim/figures/chap4/; this
only converts the committed PDF, so the thesis stays the single source.

Outputs (this directory): one SVG per entry in FIGURES.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from _common import force_regen

CHAP4 = Path("/home/wassim/Projects/Perso/These_wassim/figures/chap4")
FIGURES = {
    "typical_set.svg": CHAP4 / "typical_set.pdf",
    "trajectories.svg": CHAP4 / "trajectories.pdf",
    "vi_vs_mcmc.svg": CHAP4 / "vi_vs_mcmc.pdf",
    "forward_model_pgm.svg": CHAP4 / "forward_model_pgm.pdf",
    "noisy_likelihood.svg": CHAP4 / "noisy_likelihood.pdf",
    "confidence_bias.svg": CHAP4 / "confidence_bias.pdf",
}
for out, src in FIGURES.items():
    if (HERE / out).exists() and not force_regen():
        continue
    if not src.exists():
        print(f"MISSING {src}")
        continue
    subprocess.run(["pdftocairo", "-svg", str(src), str(HERE / out)], check=True)
    print(f"wrote {out}")
