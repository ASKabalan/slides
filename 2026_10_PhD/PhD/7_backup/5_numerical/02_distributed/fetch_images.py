#!/usr/bin/env python3
# ENV: shared
"""
External material for the slide on distributing the simulation.

Outputs (this directory):
  decomp2d.gif      the pencil-decomposition animation from the jaxDecomp
                    documentation (https://jaxdecomp.readthedocs.io)
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import cached_fetch, force_regen

CACHE = (ROOT / "5_contribution/fli") / ".cache"
GIF = "https://jaxdecomp.readthedocs.io/en/latest/_images/decomp2d.gif"
LOCAL_GIF = ROOT.parent / "assets" / "HPC" / "decomp2d.gif"   # the shared slides assets

# --- decomposition animation --------------------------------------------------
gif = HERE / "decomp2d.gif"
if force_regen() or not gif.exists():
    if LOCAL_GIF.exists():
        gif.write_bytes(LOCAL_GIF.read_bytes())
        print(f"copied {gif.name} from assets/HPC")
    else:
        gif.write_bytes(cached_fetch(CACHE, "decomp2d", GIF))
        print(f"wrote {gif.name}")
