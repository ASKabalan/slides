#!/usr/bin/env python3
# ENV: shared
"""
How the gradient of the log posterior guides optimisation and sampling, as two static vector fields,
for the slide "Differentiable methods for high-dimensional inference".

The posterior is the correlated 2D Gaussian of posterior_moves.py (sigma_1 = 1, sigma_2 = 1.4,
rho = 0.65), on the same square panel.

grad_field.svg  the 68.3, 95.4 and 99.7 % regions in the deck purple and, on a grid, the gradient of
                ln p as dark arrows (length growing with |grad ln p|, capped): every arrow points to
                the maximum a posteriori, an orange dot.
hmc_field.svg   after Betancourt (2017, arXiv:1701.02434), Fig. 12: the same Gaussian drawn faintly,
                its typical set as a glowing band, and a vector field aligned with it (the gradient
                turned by 90 degrees, the direction Hamiltonian dynamics follows once momentum is
                added), on concentric ellipses; one trajectory follows the band. Schematic: in d
                dimensions the band sits at a Mahalanobis radius near sqrt(d).
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import INK, KW, KW2, skip_if_built

OUTS = ["grad_field.svg", "hmc_field.svg"]
skip_if_built(HERE, *OUTS)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

plt.rcParams.update({"svg.fonttype": "path", "font.family": "DejaVu Sans"})

LIM = 5.3
S1, S2, RHO = 1.0, 1.4, 0.65
COV = np.array([[S1 ** 2, RHO * S1 * S2], [RHO * S1 * S2, S2 ** 2]])
PREC = np.linalg.inv(COV)
CHOL = np.linalg.cholesky(COV)

X, Y = np.meshgrid(np.linspace(-LIM, LIM, 500), np.linspace(-LIM, LIM, 500))
XY = np.stack([X, Y], axis=-1)
R2 = np.einsum("...i,ij,...j->...", XY, PREC, XY)          # squared Mahalanobis radius
P = np.exp(-0.5 * R2)
LEVELS = [np.exp(-0.5 * k ** 2) for k in (3, 2, 1)] + [1.0]
FILLS = ["#E9DDF0", "#CDB5DC", "#A57DBE"]


def panel(alpha=0.95):
    fig = plt.figure(figsize=(4.2, 4.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.contourf(X, Y, P, levels=LEVELS, colors=FILLS, alpha=alpha)
    ax.contour(X, Y, P, levels=LEVELS[:-1], colors=[KW2], linewidths=1.0, alpha=0.6 * alpha)
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return fig, ax


def save(fig, name):
    fig.savefig(HERE / name, transparent=True)
    plt.close(fig)
    print(f"wrote {name}")


# ---------------------------------------------------------------- optimisation: the gradient field
fig, ax = panel()
g = np.linspace(-4.7, 4.7, 10)                    # no grid point on the mode
GX, GY = np.meshgrid(g, g)
pts = np.stack([GX.ravel(), GY.ravel()], axis=-1)
grad = -pts @ PREC.T                               # grad ln p = -P theta
norm = np.linalg.norm(grad, axis=1)
length = 0.85 * np.clip(norm / 3.0, 0.3, 1.0)      # longer far from the mode, capped
U, V = (grad / norm[:, None] * length[:, None]).T
ax.quiver(pts[:, 0], pts[:, 1], U, V, angles="xy", scale_units="xy", scale=1, color=INK,
          width=0.0065, headwidth=4.2, headlength=4.5, headaxislength=4.0, alpha=0.5, zorder=3)
ax.plot(0, 0, "o", color=KW, ms=12, mec="white", mew=2.0, zorder=5)
save(fig, "grad_field.svg")

# ---------------------------------------------------------------- sampling: the Hamiltonian flow
fig, ax = panel(alpha=0.35)
RING = 1.8                                         # Mahalanobis radius of the drawn typical set
band = np.exp(-((np.sqrt(R2) - RING) / 0.38) ** 2)
glow = np.zeros(X.shape + (4,))
glow[..., :3] = to_rgb(KW)
glow[..., 3] = 0.6 * band
ax.imshow(glow, extent=(-LIM, LIM, -LIM, LIM), origin="lower", zorder=2, interpolation="bilinear")


def ring(r, a):
    """Point and unit tangent on the ellipse of Mahalanobis radius r, at parameter angle a."""
    p = r * CHOL @ np.array([np.cos(a), np.sin(a)])
    t = CHOL @ np.array([-np.sin(a), np.cos(a)])
    return p, t / np.linalg.norm(t)


for r in (0.6, 1.2, 1.8, 2.4, 3.0):
    n = int(round(7 * r)) + 2
    for a in np.linspace(0, 2 * np.pi, n, endpoint=False) + 0.3 * r:
        p, t = ring(r, a)
        d = 0.42 * t
        ax.annotate("", xy=p + d / 2, xytext=p - d / 2, zorder=3,
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4, mutation_scale=11, alpha=0.5,
                                    shrinkA=0, shrinkB=0))
a = np.linspace(np.radians(200), np.radians(395), 120)
traj = np.array([RING * CHOL @ np.array([np.cos(x), np.sin(x)]) for x in a])
ax.plot(traj[:-3, 0], traj[:-3, 1], color=KW, lw=4.0, solid_capstyle="round", zorder=4)
ax.annotate("", xy=traj[-1], xytext=traj[-4], zorder=5,
            arrowprops=dict(arrowstyle="-|>", color=KW, lw=4.0, mutation_scale=26, shrinkA=0,
                            shrinkB=0))
ax.plot(*traj[0], "o", color=KW, ms=9, mec="white", mew=1.6, zorder=6)
save(fig, "hmc_field.svg")
