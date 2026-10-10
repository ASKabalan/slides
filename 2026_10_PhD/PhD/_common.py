#!/usr/bin/env python3
"""
Helpers shared by the figure generators of the defense deck.

Each generator lives beside its figure; what they all need — a disk cache for
downloads, the Wikimedia Commons lookup, the skip guard, and one matplotlib
style so thirty plots look like one deck — lives here, beside pyproject.toml.

Import it with (adjust the parents index to the section's depth: one step
for top-level sections, two for the nested 3_observation/5_contribution/6_ending ones):

    import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _common import slide_style, outputs_exist, commons_image
"""

from __future__ import annotations

import io
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# The deck palette. Keep these in step with 2026_10_PhD/phd.scss.
KW = "#C2560A"        # keyword orange
KW2 = "#521463"       # deck purple
INK = "#2E2E2E"
GREY = "#6E6E6E"
BLUE = "#3B6FB6"
TEAL = "#2AA198"
RED = "#C0392B"
CYCLE = [KW2, KW, BLUE, TEAL, RED, "#8E44AD", "#16A085"]

UA = {
    "User-Agent": "phd-defense-slides/1.0 "
    "(figure generation; contact: wastondev@gmail.com)"
}


def force_regen() -> bool:
    return os.getenv("FORCE_REGEN", "0").lower() in ("1", "true", "t", "yes")


def outputs_exist(here: Path, *names: str) -> bool:
    """True when every output is present and no rebuild was asked for."""
    if force_regen():
        return False
    return all((here / n).exists() for n in names)


def skip_if_built(here: Path, *names: str) -> None:
    """Exit quietly when the figures are already there (the thesis convention)."""
    import sys

    if outputs_exist(here, *names):
        print(f"{', '.join(names)} already present. Set FORCE_REGEN=1 to rebuild.")
        sys.exit(0)


# --------------------------------------------------------------------------- #
# Downloads
# --------------------------------------------------------------------------- #

def cached_fetch(cache: Path, key: str, url: str, retries: int = 3) -> bytes:
    """Fetch a URL once and keep the bytes under <cache>/<key>.img."""
    import requests

    cache.mkdir(parents=True, exist_ok=True)
    blob = cache / f"{key}.img"
    if blob.exists():
        print(f"[*] {key} (cached)")
        return blob.read_bytes()
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=UA, timeout=120)
            resp.raise_for_status()
            blob.write_bytes(resp.content)
            print(f"[+] {key} ({len(resp.content) / 1e6:.1f} MB)")
            return resp.content
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"    attempt {attempt + 1} failed: {exc}")
    raise RuntimeError(f"could not fetch {key}: {last}")


def cached_file(cache: Path, name: str, url: str) -> Path:
    """Stream a large download to <cache>/<name> once and return its path."""
    import requests

    cache.mkdir(parents=True, exist_ok=True)
    out = cache / name
    if out.exists():
        print(f"[*] {name} (cached)")
        return out
    part = out.with_suffix(out.suffix + ".part")
    with requests.get(url, headers=UA, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with part.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 22):
                fh.write(chunk)
    part.rename(out)
    print(f"[+] {name} ({out.stat().st_size / 1e6:.0f} MB)")
    return out


# Planck 2018 SMICA CMB map (I, Q, U at nside 2048, K_CMB, ~2 GB), from the
# Planck Legacy Archive. Shared by the CMB-section generators.
PLANCK_SMICA = "COM_CMB_IQU-smica_2048_R3.00_full.fits"
PLANCK_SMICA_URL = f"https://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID={PLANCK_SMICA}"


def commons_url(title: str) -> str:
    """Resolve 'File:X.jpg' to its current upload URL via the Commons API.

    Hand-built upload.wikimedia.org paths embed a hash and break; the API does
    not.
    """
    import requests

    resp = requests.get(
        "https://commons.wikimedia.org/w/api.php",
        params={
            "action": "query",
            "titles": title if title.startswith("File:") else f"File:{title}",
            "prop": "imageinfo",
            "iiprop": "url|size",
            "format": "json",
        },
        headers=UA,
        timeout=60,
    )
    resp.raise_for_status()
    pages = resp.json()["query"]["pages"]
    info = next(iter(pages.values())).get("imageinfo")
    if not info:
        raise RuntimeError(f"Commons has no file named {title!r}")
    return info[0]["url"]


def commons_image(cache: Path, key: str, title: str, max_side: int | None = None):
    """Download a Commons file and return it as an RGB PIL image."""
    from PIL import Image

    im = Image.open(io.BytesIO(cached_fetch(cache, key, commons_url(title))))
    im = im.convert("RGB")
    guard_not_blank(im, key)
    if max_side:
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    return im


def guard_not_blank(im, key: str) -> None:
    """A service that returns a placeholder is otherwise silent about it."""
    from PIL import ImageStat

    sd = ImageStat.Stat(im.convert("L")).stddev[0]
    if sd < 2.0:
        raise RuntimeError(f"{key}: image looks blank (stddev {sd:.2f})")


# --------------------------------------------------------------------------- #
# Plot style
# --------------------------------------------------------------------------- #

def slide_style(scale: float = 1.0) -> None:
    """One matplotlib look for every plotted figure in the deck.

    Transparent background so the white slide shows through, no figure title
    (the slide heading holds it), and type large enough to read from the back
    of the room.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "none",
            "savefig.facecolor": "none",
            "savefig.transparent": True,
            "axes.facecolor": "none",
            "font.family": "DejaVu Sans",
            "font.size": 13 * scale,
            "axes.labelsize": 14 * scale,
            "axes.titlesize": 14 * scale,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "axes.linewidth": 1.0,
            "axes.grid": False,
            "axes.prop_cycle": matplotlib.cycler(color=CYCLE),
            "xtick.labelsize": 12 * scale,
            "ytick.labelsize": 12 * scale,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": False,
            "legend.fontsize": 12 * scale,
            "lines.linewidth": 2.2,
            # Text is emitted as paths, not <text>: an SVG that keeps live
            # text renders with whatever font the viewer happens to have, and
            # mathtext accents (a hat over a symbol) come out wrong.
            "svg.fonttype": "path",
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


# --------------------------------------------------------------------------- #
# Rotating sky maps
# --------------------------------------------------------------------------- #

def ortho_frame(m, lon: float, lat: float, px: int, **kw):
    """One transparent orthographic view of a HEALPix map, as a PIL RGBA image.

    ``kw`` goes to ``healpy.orthview`` (cmap, min, max, ...).
    """
    import healpy as hp
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    fig = plt.figure(figsize=(3.2, 3.2), dpi=px / 3.2)
    hp.orthview(m, fig=fig.number, rot=(lon, lat, 0), half_sky=True, cbar=False, title="",
                notext=True, bgcolor=(0.0,) * 4, margins=(0, 0, 0, 0), **kw)
    fig.patch.set_alpha(0.0)
    fig.canvas.draw()
    im = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return im


def write_alpha_gif(frames, path: Path, duration: int = 110) -> None:
    """Looping GIF with 1-bit transparency, cropped to the disc; the first frame is also
    saved beside it as a PNG still (for print)."""
    from PIL import Image

    box = frames[0].getchannel("A").getbbox()
    frames = [f.crop(box) for f in frames]
    frames[0].save(path.with_suffix(".png"))
    pal = frames[0].convert("RGB").quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    gif = []
    for f in frames:
        q = f.convert("RGB").quantize(palette=pal, dither=Image.Dither.NONE)
        q.paste(255, mask=f.getchannel("A").point(lambda a: 255 if a < 128 else 0))
        gif.append(q)
    gif[0].save(path, save_all=True, append_images=gif[1:], duration=duration, loop=0,
                transparency=255, disposal=2, optimize=True)
