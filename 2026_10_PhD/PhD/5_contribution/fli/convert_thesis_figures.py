#!/usr/bin/env python3
# ENV: shared
"""
Figures from Chapter 6 of the thesis, converted for the slides.

Each of these is produced by a script that lives beside it in
These_wassim/figures/chap6/, several of which need a GPU, the Jean Zay run
outputs, or the HuggingFace experiment archive. The thesis script stays the
single source of truth: regenerate a figure there with

    FORCE_REGEN=1 uv run --group chap6 python figures/chap6/<name>.py

and rerun this script to bring the new version across. Nothing is recomputed
here, only converted, so a number can never drift between the thesis and the
talk.

Outputs (this directory, or the slide subfolder named in the key): one SVG per entry in
FIGURES below.
"""

import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import force_regen

CHAP6 = Path("/home/wassim/Projects/Perso/These_wassim/figures/chap6")
CHAP3 = Path("/home/wassim/Projects/Perso/These_wassim/figures/chap3")

# output name -> source PDF
# 06_observable/geometry.svg moved out: 06_observable/depth.py replots its top two panels.
FIGURES = {}

if shutil.which("pdftocairo") is None:
    sys.exit("pdftocairo not found")

made, missing = 0, []
for out, src in FIGURES.items():
    if (HERE / out).exists() and not force_regen():
        continue
    if not src.exists():
        missing.append(f"{out}  <-  {src}")
        continue
    (HERE / out).parent.mkdir(exist_ok=True)
    subprocess.run(["pdftocairo", "-svg", str(src), str(HERE / out)], check=True)
    made += 1
    print(f"wrote {out}")

# Already SVG in the thesis: copied as is.
for out, src in (("7_backup/4_fieldlevel/14_distributed/jaxdecomp_fft.svg", CHAP6 / "jaxdecomp_fft.svg"),):
    if src.exists() and (force_regen() or not (ROOT / out).exists()):
        shutil.copy(src, ROOT / out)
        print(f"copied {out}")

print(f"\n{made} figures converted")
if missing:
    print("MISSING:")
    for m in missing:
        print("   ", m)
