#!/usr/bin/env python3
# ENV: shared
"""
The moving-mesh particle-mesh animation shown with the perspectives of Contribution 2, converted from
the GIF it was delivered as (~/Downloads/moving_anim_z0.gif: CV0 moving mesh, 25 Mpc/h box, 128^3
mesh, 1125 x 612 px, 77 frames at 10 fps, white background) to H.264, which plays and loops in the
deck at a tenth of the size.

Outputs (this directory): moving_mesh.mp4, moving_mesh_last.png (the last frame, for print)
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from _common import skip_if_built

skip_if_built(HERE, "moving_mesh.mp4", "moving_mesh_last.png")
SRC = Path.home() / "Downloads" / "moving_anim_z0.gif"

subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(SRC), "-vf",
                "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p", "-c:v", "libx264", "-crf", "20",
                "-movflags", "+faststart", str(HERE / "moving_mesh.mp4")], check=True)

from PIL import Image

with Image.open(SRC) as g:
    g.seek(g.n_frames - 1)
    g.convert("RGB").save(HERE / "moving_mesh_last.png")
print(f"wrote moving_mesh.mp4 ({(HERE / 'moving_mesh.mp4').stat().st_size / 1e6:.1f} MB) and moving_mesh_last.png")
