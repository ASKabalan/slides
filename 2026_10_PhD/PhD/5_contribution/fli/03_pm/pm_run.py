#!/usr/bin/env python3
# ENV: jax-fli
"""
One particle-mesh step taken click by click, then the rest of the run, for the slide on the
gravity solver.

A 64^3 run in a 100 Mpc/h box: first-order LPT to a = 0.001, then BullFrog with 10 steps uniform in
a, to a = 1, with the particles saved at every step (a = 0.001, 0.101, ..., 1). The panel follows
the particles that start in one thin slab (four grid layers, 16 384 particles), projected on the
slab plane, under a slider that shows the scale factor. Beside it the loop of pm_cycle.tex lights
the operation the panel shows.

The first step is taken on the slide's clicks, one frame per operation, with the fields of that
step computed by jaxpm's own functions on the full 64^3 mesh (paint, fft3d, invlaplace_kernel,
gradient_kernel, ifft3d, readout, as jaxpm.pm.pm_forces does) and averaged over the slab:

  pm_0.png  paint        the particles over the density they paint on the mesh
  pm_1.png  FFT          |delta(k)| on the k_z = 0 plane
  pm_2.png  inverse FFT  the potential phi
  pm_3.png  read forces  the force field -grad(phi) on the mesh, as arrows over phi
  pm_4.png  interpolate  the forces read at the particles (a subset, as arrows)
  pm_5.png  kick, drift  the particles at a = 0.101, the slider moved one step

pm_run.gif continues from there: for each of the other 9 steps the six operations light in turn
(100 ms each, 200 ms for the advance) and the particles and the slider advance, 8 s in all. Its first frame is pm_5.png (written from the GIF itself, so
the two are the same pixels); it plays once and holds on a = 1 for 1.5 s. pm_run_last.png is its last frame,
for print. No text on the frames beyond the slider value; the slide holds the rest.
The cycle renders pm_cycle_{0..6}.png come from pm_cycle.tex (../build.sh).
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROOT = next(p for p in HERE.parents if (p / "_common.py").exists())
sys.path.insert(0, str(ROOT))
from _common import INK, KW, skip_if_built

OUTS = [f"pm_{k}.png" for k in range(6)] + ["pm_run.gif", "pm_run_last.png"]
CACHE = HERE.parent / ".cache"
MESH, BOX = 64, 100.0
A0, A1, N_STEPS = 0.001, 1.0, 10
TS = np.linspace(A0, A1, N_STEPS + 1)
SLAB = (30, 34)                     # Lagrangian grid layers followed
BG = (250, 247, 240)                # the slide background, #faf7f0
H, P = 740, 620                     # frame height, panel size (the slider takes H - P above it)
PARTICLE = "#25406B"

skip_if_built(HERE, *OUTS)


def run() -> dict:
    """Slab positions at every step, and the fields of the first step (slab averages)."""
    npz = CACHE / f"pm_steps_{N_STEPS}.npz"
    if npz.exists():
        return dict(np.load(npz))
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import jax_cosmo as jc
    import jax_fli as jfli
    from jax_fli.fields.units import PositionUnit
    from jaxpm.distributed import fft3d, ifft3d
    from jaxpm.kernels import fftk, gradient_kernel, invlaplace_kernel
    from jaxpm.painting import paint, readout

    cosmo = jc.Planck18()
    mesh, box = (MESH,) * 3, (BOX,) * 3
    ic = jfli.gaussian_initial_conditions(jax.random.PRNGKey(3), mesh, box, cosmo=cosmo)
    dx, p = jfli.lpt(cosmo, ic, ts=A0, order=1, painting=jfli.PaintingOptions(target="particles"))
    solver = jfli.BullFrog(
        interp_kernel=jfli.NoInterp(painting=jfli.PaintingOptions(target="particles")),
        time_stepping="a", n_steps=N_STEPS, t0=A0, t1=A1)
    out = jfli.nbody(cosmo, dx, p, ts=jnp.asarray(TS), solver=solver)
    out = out.to(PositionUnit.GRID_ABSOLUTE)
    pos = np.asarray(out.array).reshape(len(TS), MESH, MESH, MESH, 3) % MESH
    a = np.asarray(out.scale_factors).ravel()
    pos = pos[np.argsort(a)]
    print("scale factors", np.sort(a).round(4))

    # the first step's fields, as jaxpm.pm.pm_forces computes them
    x0 = jnp.asarray(pos[0])
    rho = paint(x0)
    delta = rho / rho.mean() - 1.0
    delta_k = fft3d(delta)
    kvec = fftk(delta_k)
    pot_k = delta_k * invlaplace_kernel(kvec)
    phi = np.asarray(ifft3d(pot_k).real)
    fmesh = [ifft3d(-gradient_kernel(kvec, i, order=1) * pot_k).real for i in range(3)]
    fpart = np.stack([np.asarray(readout(f, x0)) for f in fmesh], axis=-1)
    z = slice(*SLAB)
    out = {
        "slab": pos[:, :, :, z, :].reshape(len(TS), -1, 3),
        "delta": np.asarray(delta)[:, :, z].mean(-1),
        "delta_k": np.abs(np.fft.fftshift(np.fft.fftn(np.asarray(delta))[:, :, 0])),
        "phi": phi[:, :, z].mean(-1),
        "fx": np.asarray(fmesh[0])[:, :, z].mean(-1),
        "fy": np.asarray(fmesh[1])[:, :, z].mean(-1),
        "part_xy": pos[0, ::4, ::4, 31, :2].reshape(-1, 2),       # a 16 x 16 subset, one layer
        "part_f": fpart[::4, ::4, 31, :2].reshape(-1, 2),
    }
    CACHE.mkdir(exist_ok=True)
    np.savez(npz, **out)
    return out


R = run()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

bg = tuple(c / 255 for c in BG)


def right(a, draw):
    """The slider (scale factor a) above the panel drawn by `draw`, as a P x H image."""
    fig = plt.figure(figsize=(P / 100, H / 100), dpi=100, facecolor=bg)
    ax = fig.add_axes([0, 0, 1, P / H])
    draw(ax)
    ax.set_xlim(0, MESH)
    ax.set_ylim(0, MESH)
    ax.set_aspect("equal")
    ax.set_axis_off()
    s = fig.add_axes([0.07, P / H + 0.01, 0.86, (H - P) / H - 0.02], facecolor=bg)
    s.plot([A0, A1], [0, 0], color="#BDBDBD", lw=7, solid_capstyle="round")
    s.plot([A0, a], [0, 0], color=KW, lw=7, solid_capstyle="round")
    s.plot(TS, np.zeros_like(TS), "|", color="#8A8A8A", ms=14, mew=1.6)
    s.plot([a], [0], "o", color=KW, ms=19, mec="white", mew=2.5, zorder=5)
    s.text(min(max(a, 0.09), 0.91), 0.9, f"$a = {a:.3f}$" if a < 0.9995 else "$a = 1$",
           ha="center", va="bottom", fontsize=24, color=INK)
    s.text(A0, -0.9, "0.001", ha="center", va="top", fontsize=16, color="#6E6E6E")
    s.text(A1, -0.9, "1", ha="center", va="top", fontsize=16, color="#6E6E6E")
    s.set_xlim(A0 - 0.03, A1 + 0.03)
    s.set_ylim(-2.1, 2.6)
    s.set_axis_off()
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return Image.fromarray(rgb)


def particles(xy, **kw):
    style = dict(s=1.8, c=PARTICLE, alpha=0.65, linewidths=0)
    style.update(kw)
    return lambda ax: ax.scatter(xy[:, 0], xy[:, 1], **style)


def field(f, cmap):
    lo, hi = np.percentile(f, [1, 99.5])
    return lambda ax: ax.imshow(f.T, origin="lower", cmap=cmap, vmin=lo, vmax=hi,
                                interpolation="bicubic", extent=(0, MESH, 0, MESH))


def both(*draws):
    def d(ax):
        for f in draws:
            f(ax)
    return d


cycle = []
for lit in range(7):
    im = Image.open(HERE / f"pm_cycle_{lit}.png").convert("RGBA")
    im = im.resize((round(im.width * H / im.height), H), Image.LANCZOS)
    card = Image.new("RGBA", im.size, BG + (255,))
    cycle.append(Image.alpha_composite(card, im).convert("RGB"))
W = cycle[0].width + 40 + P
W += W % 2


def frame(lit, panel):
    f = Image.new("RGB", (W, H), BG)
    f.paste(cycle[lit], (0, 0))
    f.paste(panel, (cycle[0].width + 40, 0))
    return f


slab = R["slab"]
k = np.arange(MESH) - MESH // 2
step = 4                                               # mesh arrows every 4 cells
g = np.arange(step // 2, MESH, step)
fx, fy = R["fx"][g][:, g], R["fy"][g][:, g]
fscale = 1.4 * max(np.abs(R["fx"]).max(), np.abs(R["fy"]).max()) / step
quiver_mesh = lambda ax: ax.quiver(*np.meshgrid(g + 0.5, g + 0.5, indexing="ij"), fx, fy,
                                   angles="xy", scale_units="xy", scale=fscale, color=INK,
                                   width=0.004, headwidth=4)
pf = R["part_f"]
quiver_part = lambda ax: ax.quiver(R["part_xy"][:, 0], R["part_xy"][:, 1], pf[:, 0], pf[:, 1],
                                   angles="xy", scale_units="xy", scale=fscale, color=KW,
                                   width=0.005, headwidth=4)

states = [
    frame(1, right(TS[0], both(field(R["delta"], "magma"),
                               particles(slab[0, :, :2], c="white", alpha=0.35)))),
    frame(2, right(TS[0], field(np.log10(R["delta_k"] + 1e-12), "viridis"))),
    frame(3, right(TS[0], field(R["phi"], "RdBu_r"))),
    frame(4, right(TS[0], both(field(R["phi"], "RdBu_r"), quiver_mesh))),
    frame(5, right(TS[0], both(particles(slab[0, :, :2], alpha=0.25), quiver_part))),
]
for i, im in enumerate(states):
    im.save(HERE / f"pm_{i}.png")

# the GIF: step 1 onwards, six operations per step, then a hold on a = 1
panels = {s: right(TS[s], particles(slab[s, :, :2])) for s in range(1, N_STEPS + 1)}
seq, dur = [frame(6, panels[1])], [120]
for s in range(1, N_STEPS):
    for lit in range(1, 6):
        seq.append(frame(lit, panels[s]))
        dur.append(100)
    seq.append(frame(6, panels[s + 1]))
    dur.append(200)
seq.append(frame(0, panels[N_STEPS]))
dur.append(1500)

mosaic = Image.new("RGB", (W, H * 3))
for i, im in enumerate((seq[0], seq[len(seq) // 2], seq[-1])):
    mosaic.paste(im, (0, H * i))
# the background and the diagram's flat colours are pinned exactly in the palette, so the GIF's
# first frame is indistinguishable from the PNG states that precede it on the slide
EXACT = [BG, (0x2E, 0x2E, 0x2E), (0xC2, 0x56, 0x0A), (0xFB, 0xE7, 0xD6), (0xF4, 0xF6, 0xFA),
         (0x9A, 0x9A, 0x9A), (0xBD, 0xBD, 0xBD), (0x8A, 0x8A, 0x8A), (0x25, 0x40, 0x6B), (255, 255, 255)]
base = mosaic.quantize(colors=256 - len(EXACT), method=Image.Quantize.MEDIANCUT).getpalette()
base = base[:3 * (256 - len(EXACT))]
pal = Image.new("P", (1, 1))
pal.putpalette(base + [c for rgb in EXACT for c in rgb])
PAL = np.array(pal.getpalette()[:768]).reshape(256, 3)


def to_palette(im):
    """Exact nearest-colour mapping onto PAL (PIL's own mapping rounds through a coarse cache and
    shifted the background by two levels)."""
    a = np.asarray(im, dtype=np.int32)
    code = (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]
    uniq, inv = np.unique(code.ravel(), return_inverse=True)
    rgb = np.stack([uniq >> 16, (uniq >> 8) & 255, uniq & 255], axis=1)
    idx = np.argmin(((rgb[:, None, :] - PAL[None]) ** 2).sum(-1), axis=1).astype(np.uint8)
    out = Image.fromarray(idx[inv].reshape(code.shape), mode="P")
    out.putpalette(PAL.ravel().tolist())
    return out


gif = [to_palette(im) for im in seq]
gif[0].save(HERE / "pm_run.gif", save_all=True, append_images=gif[1:], duration=dur,
            optimize=True)                               # no loop entry: the GIF plays once
with Image.open(HERE / "pm_run.gif") as g0:
    g0.seek(0)
    g0.convert("RGB").save(HERE / "pm_5.png")            # the same pixels as the GIF's first frame
    g0.seek(g0.n_frames - 1)
    g0.convert("RGB").save(HERE / "pm_run_last.png")
print(f"wrote pm_0..5.png and pm_run.gif ({len(seq)} frames, {sum(dur) / 1000:.2f} s, "
      f"{(HERE / 'pm_run.gif').stat().st_size / 1e6:.1f} MB)")
