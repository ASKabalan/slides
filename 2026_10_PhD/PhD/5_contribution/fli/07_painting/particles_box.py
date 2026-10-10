#!/usr/bin/env python3
# ENV: jax-fli
"""
The evolved particles that the painting step projects onto a shell, for the painting slide.

A small jax-fli run kept as particles end to end (PaintingOptions(target="particles")): 128^3
particles in a 256 Mpc/h box, first-order LPT to a = 0.1, then BullFrog in 10 steps to a = 1.
The particles are drawn with jax_fli's own ParticleField.plot, coloured by the log density read
back at each particle so the filaments show. Every sixth particle along each axis is drawn
(about 21^3), so single particles show at slide size.

Output (this directory): particles_box.png
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import skip_if_built, slide_style

OUTS = ["particles_box.png"]
skip_if_built(HERE, *OUTS)

import jax

jax.config.update("jax_enable_x64", True)
import equinox as eqx
import jax.numpy as jnp
import jax_cosmo as jc
import jax_fli as jfli
import itertools

import matplotlib.pyplot as plt
import numpy as np
from jax_fli.fields.units import PositionUnit
from jaxpm.painting import paint, readout

MESH, BOX, A0 = 128, 256.0, 0.1

slide_style()

cosmo = jc.Planck18()
mesh, box = (MESH,) * 3, (BOX,) * 3
ic = jfli.gaussian_initial_conditions(jax.random.PRNGKey(7), mesh, box, cosmo=cosmo)
particles = jfli.PaintingOptions(target="particles")
dx, p = jfli.lpt(cosmo, ic, ts=A0, order=1, painting=particles)
solver = jfli.BullFrog(interp_kernel=jfli.NoInterp(painting=particles), time_stepping="a",
                       n_steps=10, t0=A0, t1=1.0)
field = jfli.nbody(cosmo, dx, p, ts=jnp.asarray([1.0]), solver=solver).to(PositionUnit.GRID_ABSOLUTE)
flat = jnp.asarray(np.asarray(field.array).reshape(-1, 3) % MESH)
rho = paint(flat, jnp.zeros(mesh))
logrho = np.log10(np.asarray(readout(rho, flat)).reshape(mesh) + 0.1)
# periodic box: particles that drifted across a face are wrapped back in, so the box reads as a cube
field = eqx.tree_at(lambda f: f.array, field, flat.reshape(field.array.shape))

fig = plt.figure(figsize=(5, 5))
ax = fig.add_axes([0, 0, 1, 1], projection="3d")
field.plot(ax=ax, weights=logrho, cmap="magma", vmin=np.percentile(logrho, 5),
           vmax=np.percentile(logrho, 99.5), colorbar=False, thinning=6, point_size=10,
           alpha=0.9, labels=("", "", ""), ticks=([], [], []), titles="", elev=25, azim=-50,
           zoom=1.1)
for a, b in itertools.combinations(list(itertools.product([0, MESH], repeat=3)), 2):
    if sum(ai != bi for ai, bi in zip(a, b)) == 1:      # the twelve cube edges
        ax.plot3D(*zip(a, b), color="0.45", lw=1.2)
ax.set(xlim=(0, MESH), ylim=(0, MESH), zlim=(0, MESH))
ax.set_axis_off()
fig.savefig(HERE / OUTS[0], dpi=300, transparent=True)
print("wrote", ", ".join(OUTS))
