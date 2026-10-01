#!/usr/bin/env python3
# ENV: shared
"""
Survey-stage comparison figure for the PhD defense intro.

Stage 2: SDSS DR18  (Apache Point 2.5m, ~1.4" seeing, 0.396"/px)
Stage 4: ESA Euclid ERO (1.2m space, <0.18" diffraction, 0.101"/px)

NOTE: the HSC PDR3 stage-3 leg is not possible on Abell 2390 — Dec +17.7
lies outside the HSC public footprint (the 0h Wide band stops at Dec +10).

Output (assets/PhD/<section>/ convention):
  - stage2_4_abell2390_comparison.svg  matched-FOV pair with vector text
                                       titles, transparent background (no
                                       <rect>, the slide shows through the
                                       title band and the panel gap).
  - stage2_4_abell2390_bg.jpg          the same matched pair, no titles, a
                                       thin dark divider: full-bleed slide
                                       background (data-background-size: cover).
  - abell2390_stage2_sdss.png / abell2390_stage4_euclid.png  native frames
    (kept both as reuse-ready assets and as the download cache fallback).

Composites are SVG: cutouts embedded as JPEG data URIs at native resolution
(SVG being vector, the 600-dpi rule applies to rendering, not to the
embedded survey pixels).
"""

import base64
import html
import io
from pathlib import Path

import requests
from PIL import Image, ImageStat

# --- Output: this directory (the generator lives beside its figure) ---
OUT_DIR = Path(__file__).resolve().parent
CACHE_DIR = OUT_DIR / ".cache"

# --- Target coordinates: Abell 2390 cluster core ---
RA = 328.4033
DEC = 17.6956

SDSS_URL = (
    "https://skyserver.sdss.org/dr18/SkyServerWS/ImgCutout/getjpeg"
    f"?ra={RA}&dec={DEC}&scale=0.396&width=1024&height=1024"
)
# Official ESA ERO Abell 2390 high-resolution frame (ID 497233)
EUCLID_URL = (
    "https://www.esa.int/var/esa/storage/images/esa_multimedia/images/2024/05/"
    "closer_euclid_view_of_abell_2390/26078563-1-eng-GB/"
    "Closer_Euclid_view_of_Abell_2390.jpg"
)

HEADERS = {"User-Agent": "Mozilla/5.0 (Astrophysics Defense Prep)"}


# ---------------------------------------------------------------------------
# Download machinery (cache + retries + blank-guard)
# ---------------------------------------------------------------------------

def fetch_bytes(url: str, key: str, name: str, retries: int = 2) -> bytes:
    """Download raw bytes with a disk cache; key names the cached frame.

    Falls back to the saved native frame <key>.png beside the script when the
    cache is empty (both frames are square crops, so re-cropping is a no-op).
    """
    cache_file = CACHE_DIR / f"{key}.img"
    if cache_file.exists():
        print(f"[*] Loading {key} (cached)")
        return cache_file.read_bytes()
    saved_frame = OUT_DIR / f"{key}.png"
    if saved_frame.exists():
        print(f"[*] Loading {key} (saved native frame)")
        return saved_frame.read_bytes()
    for attempt in range(retries + 1):
        try:
            print(f"[*] Downloading {name}...")
            resp = requests.get(url, headers=HEADERS, timeout=90)
            resp.raise_for_status()
            CACHE_DIR.mkdir(exist_ok=True)
            cache_file.write_bytes(resp.content)
            return resp.content
        except requests.RequestException as exc:
            if attempt == retries:
                raise
            print(f"    -> attempt {attempt + 1} failed ({exc.__class__.__name__}); retrying...")
    raise RuntimeError("unreachable")


def guard_not_blank(img: Image.Image, name: str) -> Image.Image:
    """Raise loudly if a cutout service returned its blank placeholder."""
    stat = ImageStat.Stat(img.convert("L"))
    if stat.stddev[0] < 2.0:
        raise RuntimeError(
            f"{name}: cutout is blank (stddev={stat.stddev[0]:.2f}) — "
            "the survey does not cover this field or the service failed"
        )
    return img


def center_square_crop(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def crop_to_fov(img: Image.Image, pixscale: float, fov_arcsec: float) -> Image.Image:
    """Center-crop to a common field of view, then upscale to 1024 display px."""
    w, h = img.size
    side = min(round(fov_arcsec / pixscale), w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side)).resize(
        (1024, 1024), Image.LANCZOS
    )


# ---------------------------------------------------------------------------
# Survey frames
# ---------------------------------------------------------------------------

def fetch_sdss() -> Image.Image:
    raw = fetch_bytes(SDSS_URL, "abell2390_stage2_sdss", "SDSS DR18")
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return guard_not_blank(center_square_crop(img), "SDSS")


def fetch_euclid_ero() -> Image.Image:
    """ESA ERO closer frame of Abell 2390 (2493x1659), cropped to the core."""
    raw = fetch_bytes(EUCLID_URL, "abell2390_stage4_euclid", "Euclid ERO (Abell 2390)")
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return center_square_crop(img)


# ---------------------------------------------------------------------------
# Transparent SVG composition
# ---------------------------------------------------------------------------

def compose_svg(panels: list, out_path: Path, size: int = 1024, gap: int = 8) -> Path:
    """Side-by-side cutouts with vector titles, transparent background.

    panels: list of (PIL.Image, [title line 1, subtitle line 2]).
    Text is dark to sit on the deck's light slide background.
    """
    title_band = 110
    width = len(panels) * size + (len(panels) - 1) * gap
    height = size + title_band
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
    ]
    for i, (img, lines) in enumerate(panels):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        b64 = base64.b64encode(buf.getvalue()).decode()
        x = i * (size + gap)
        parts.append(
            f'<image x="{x}" y="{title_band}" width="{size}" height="{size}" '
            f'preserveAspectRatio="none" xlink:href="data:image/jpeg;base64,{b64}"/>'
        )
        cx = x + size / 2
        l1, l2 = html.escape(lines[0]), html.escape(lines[1])
        parts.append(
            f'<text x="{cx}" y="46" text-anchor="middle" fill="#1a1a1a" '
            f'font-family="Carlito, Calibri, Helvetica, Arial, sans-serif" '
            f'font-size="34" font-weight="bold">{l1}</text>'
        )
        parts.append(
            f'<text x="{cx}" y="88" text-anchor="middle" fill="#767676" '
            f'font-family="Carlito, Calibri, Helvetica, Arial, sans-serif" '
            f'font-size="26">{l2}</text>'
        )
    parts.append("</svg>")

    out_path.write_text("\n".join(parts))
    print(f"[+] Saved {out_path.name} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


def compose_background(images: list, out_path: Path, gap: int = 6) -> Path:
    """Matched cutouts side by side, no titles, for a full-bleed background."""
    size = images[0].size[0]
    width = len(images) * size + (len(images) - 1) * gap
    canvas = Image.new("RGB", (width, size), (8, 8, 10))
    for i, img in enumerate(images):
        canvas.paste(img, (i * (size + gap), 0))
    canvas.save(out_path, format="JPEG", quality=90)
    print(f"[+] Saved {out_path.name} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Fetch frames ---
    sdss_2390 = fetch_sdss()
    euclid_2390 = fetch_euclid_ero()

    # --- Native frames (reuse-ready assets + cache fallback) ---
    sdss_2390.save(OUT_DIR / "abell2390_stage2_sdss.png")
    euclid_2390.save(OUT_DIR / "abell2390_stage4_euclid.png")
    print(f"[+] Saved individual frames in {OUT_DIR}")

    # --- Matched field of view: the Euclid frame sets the smallest FOV ---
    SDSS_PIX, EU_PIX = 0.396, 0.101
    matched_fov = euclid_2390.size[0] * EU_PIX  # ~167"
    left = crop_to_fov(sdss_2390, SDSS_PIX, matched_fov)
    right = crop_to_fov(euclid_2390, EU_PIX, matched_fov)

    compose_svg(
        [
            (left, ["Stage 2 — SDSS DR18", "Abell 2390 | 2.5m Apache Point | ~1.4'' seeing"]),
            (right, ["Stage 4 — Euclid VIS/NISP", "Abell 2390 | 1.2m Space L2 | <0.18'' diffraction"]),
        ],
        OUT_DIR / "stage2_4_abell2390_comparison.svg",
    )
    compose_background([left, right], OUT_DIR / "stage2_4_abell2390_bg.jpg")


if __name__ == "__main__":
    main()
