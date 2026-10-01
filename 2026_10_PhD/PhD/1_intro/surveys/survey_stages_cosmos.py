#!/usr/bin/env python3
"""
Three-stage comparison on the COSMOS field for the PhD defense intro.

Stage 2: SDSS DR18        (Apache Point 2.5m, ~1.4" seeing, 0.396"/px)
Stage 3: Subaru HSC PDR3  (8.2m Maunakea, ~0.6" seeing, 0.168"/px)
Stage 4: Rubin/LSST EDP2  (8.4m Cerro Pachon, LSSTCam gri deep co-add, 0.201"/px)

The Rubin frame is cut from the official public HiPS of the EDP2 deep co-add
(published with the July 2026 COSMOS press release, noirlab2618):
    https://images.rubinobservatory.org/hips/oceancosmosm18/color_gri/
The CDS hips2fits service cannot process it (webp-only tiles, private flag),
so the cutout is built locally: fetch the Norder-11 tiles overlapping the
target window, map tile pixels to the sky through cdshealpix fractional
(dx, dy) coordinates and resample onto a TAN grid.

Output (2026_10_PhD/PhD/<section>/ convention), vector titles on transparent
background (no <rect>, the slide shows through the title band and gaps):
  - stage2_3_4_cosmos_comparison.svg  matched-FOV triplet with titles
  - cosmos_stage2_sdss.png / cosmos_stage3_hsc.png / cosmos_stage4_rubin.png
    native-resolution frames (reuse-ready assets + cache fallback)

Note: SDSS, HSC and Rubin are all north-up TAN renders. The Rubin HiPS
tile-pixel mapping (cdshealpix fractional dx -> tile row, dy -> tile
column) was validated per-tile against the HSC frame (corr 0.78-0.84),
and the assembled cutout correlates at 0.83 with HSC at matched FOV.
"""

import base64
import html
import io
from pathlib import Path

import numpy as np
import requests
from astropy.io import fits as astropy_fits
from astropy.wcs import WCS
from cdshealpix.nested import healpix_to_lonlat, lonlat_to_healpix
from PIL import Image, ImageStat
from scipy.ndimage import map_coordinates

# --- Output: this directory (the generator lives beside its figure) ---
OUT_DIR = Path(__file__).resolve().parent
CACHE_DIR = OUT_DIR / ".cache"

# --- COSMOS field centre ---
RA, DEC = 150.1, 2.2

SDSS_URL = (
    "https://skyserver.sdss.org/dr18/SkyServerWS/ImgCutout/getjpeg"
    f"?ra={RA}&dec={DEC}&scale=0.396&width=1024&height=1024"
)
HSC_FITS_URL = (
    "https://www.legacysurvey.org/viewer/cutout.fits"
    f"?ra={RA}&dec={DEC}&layer=hsc-dr3&pixscale=0.168&size=1024"
)
# Official Rubin EDP2 deep co-add (LSSTCam gri), public HiPS (July 2026)
RUBIN_HIPS = "https://images.rubinobservatory.org/hips/oceancosmosm18/color_gri"
RUBIN_HIPS_DEPTH = 11  # 512-px tiles at 0.201"/px, native LSSTCam resolution
RUBIN_TILE_PX = 512

HEADERS = {"User-Agent": "Mozilla/5.0 (Astrophysics Defense Prep)"}


# ---------------------------------------------------------------------------
# Download machinery (cache + retries + blank-guard)
# ---------------------------------------------------------------------------

def fetch_bytes(url: str, key: str, name: str, retries: int = 2) -> bytes:
    """Download raw bytes with a disk cache; key names the cached frame."""
    cache_file = CACHE_DIR / f"{key}.img"
    if cache_file.exists():
        print(f"[*] Loading {key} (cached)")
        return cache_file.read_bytes()
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

def fetch_sdss(key: str) -> Image.Image:
    raw = fetch_bytes(SDSS_URL, key, "SDSS DR18")
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return guard_not_blank(center_square_crop(img), "SDSS")


def lupton_rgb(data: np.ndarray, q: float = 8.0) -> Image.Image:
    """Asinh (Lupton-style) RGB stretch of a 3-band FITS cube.

    Legacy Survey HSC planes are g, r, z (header BANDS='grz'). The red channel
    blends z with r (60/40) to suppress z-band coadd-edge artifacts.
    """
    stretched = []
    for band in data:
        med = np.nanmedian(band)
        span = np.nanpercentile(band, 99.5) - med
        x = np.arcsinh(q * np.clip(band - med, 0, None) / max(span, 1e-6)) / np.arcsinh(q)
        stretched.append(np.clip(x, 0, 1))
    g, r, z = stretched
    red = 0.6 * z + 0.4 * r
    rgb = np.stack([red, r, g], axis=-1)
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))


def fetch_hsc(key: str) -> Image.Image:
    """HSC PDR3 3-band cutout via the Legacy Survey FITS service."""
    raw = fetch_bytes(HSC_FITS_URL, key, "HSC PDR3 (FITS)")
    hdul = astropy_fits.open(io.BytesIO(raw))
    data = np.array(hdul[0].data, dtype=float)
    print(f"[*] Stretching HSC FITS ({data.shape[1]}x{data.shape[2]}, {data.shape[0]} bands)")
    return guard_not_blank(center_square_crop(lupton_rgb(data)), "HSC")


# ---------------------------------------------------------------------------
# Rubin HiPS -> TAN cutout
# ---------------------------------------------------------------------------

def tan_wcs(pixscale_arcsec: float, size: int = 1024) -> WCS:
    """North-up TAN projection centered on (RA, DEC) at the given pixel scale."""
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crpix = [size / 2 + 0.5, size / 2 + 0.5]
    w.wcs.crval = [RA, DEC]
    w.wcs.cd = [[-pixscale_arcsec / 3600, 0], [0, pixscale_arcsec / 3600]]
    return w


def fetch_rubin_cutout(fov_arcsec: float, size: int, key: str) -> Image.Image:
    """Local hips2fits-style cutout: fetch HiPS tiles, resample onto TAN grid.

    cdshealpix maps tile pixels (dx, dy) to sky positions; astropy WCS maps
    target pixels to sky positions; scipy interpolates the right tile per
    target pixel.
    """
    depth = RUBIN_HIPS_DEPTH
    nside = 2**depth
    pixscale_arcsec = 0.201  # HiPS native scale (hips_pixel_scale property)

    # --- WCS of the target TAN frame ---
    w = tan_wcs(pixscale_arcsec, size)

    # --- Sky position of every target pixel ---
    yy, xx = np.mgrid[0:size, 0:size]
    ra_g, dec_g = w.all_pix2world(xx.ravel(), yy.ravel(), 0)
    from astropy.coordinates import SkyCoord, Latitude, Longitude
    import astropy.units as u
    sky = SkyCoord(ra_g * u.deg, dec_g * u.deg, frame="icrs")
    lon = Longitude(ra_g * u.deg)
    lat = Latitude(dec_g * u.deg)

    # --- Tiles needed: probe a subgrid (8x-coarser) of the frame ---
    ipix_probe, _, _ = lonlat_to_healpix(
        lon[::8], lat[::8], depth, return_offsets=True
    )
    needed_tiles = sorted(set(ipix_probe.tolist()))
    print(f"[*] Rubin HiPS: {len(needed_tiles)} tiles at depth {depth}")

    # --- Fetch tiles ---
    tiles = {}
    for idx in needed_tiles:
        d = (idx // 10000) * 10000
        tile_key = f"rubin_hips_n{depth}_i{idx}"
        url = f"{RUBIN_HIPS}/Norder{depth}/Dir{d}/Npix{idx}.webp"
        raw = fetch_bytes(url, tile_key, f"HiPS tile {idx}")
        tiles[idx] = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))

    # --- Map target pixels onto tile pixel coordinates ---
    ipix_t, dx_t, dy_t = lonlat_to_healpix(
        lon, lat, depth, return_offsets=True
    )
    ipix_t = ipix_t.astype(np.int64)

    # cdshealpix convention (validated per-tile vs HSC, corr 0.78-0.84):
    # dx -> tile ROW, dy -> tile COLUMN
    src_row = dx_t * RUBIN_TILE_PX - 0.5
    src_col = dy_t * RUBIN_TILE_PX - 0.5

    out = np.zeros((size * size, 3), dtype=float)
    covered = np.zeros(size * size, dtype=bool)
    for idx, img in tiles.items():
        mask = ipix_t == idx
        if not mask.any():
            continue
        for c in range(3):
            vals = map_coordinates(
                img[..., c].astype(float),
                [src_row[mask], src_col[mask]],
                order=1,
                mode="nearest",
            )
            out[mask, c] = vals
        covered |= mask

    if not covered.all():
        n_missing = int((~covered).sum())
        raise RuntimeError(f"Rubin cutout: {n_missing} target pixels fell outside fetched tiles")

    rgb = out.reshape(size, size, 3).astype(np.uint8)
    return Image.fromarray(rgb)


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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Fetch frames ---
    sdss = fetch_sdss(key="cosmos_stage2_sdss")
    hsc = fetch_hsc(key="cosmos_stage3_hsc_fits")
    rubin = fetch_rubin_cutout(fov_arcsec=205.0, size=1024, key="cosmos_stage4_rubin")

    # --- Native frames (reuse-ready assets + cache fallback) ---
    sdss.save(OUT_DIR / "cosmos_stage2_sdss.png")
    hsc.save(OUT_DIR / "cosmos_stage3_hsc.png")
    rubin.save(OUT_DIR / "cosmos_stage4_rubin.png")
    print(f"[+] Saved individual frames in {OUT_DIR}")

    # --- Matched field of view ---
    SDSS_PIX, HSC_PIX, RUBIN_PIX = 0.396, 0.168, 0.201
    sdss_fov = sdss.size[0] * SDSS_PIX
    hsc_fov = hsc.size[0] * HSC_PIX
    rubin_fov = rubin.size[0] * RUBIN_PIX
    matched_fov = min(hsc_fov, rubin_fov)  # ~167"

    panels = [
        crop_to_fov(sdss, SDSS_PIX, matched_fov),
        crop_to_fov(hsc, HSC_PIX, matched_fov),
        crop_to_fov(rubin, RUBIN_PIX, matched_fov),
    ]
    compose_svg(
        [
            (panels[0], ["Stage 2 — SDSS DR18", "COSMOS | 2.5m Apache Point | ~1.4'' seeing"]),
            (panels[1], ["Stage 3 — Subaru HSC PDR3", "COSMOS | 8.2m Maunakea | ~0.6'' seeing"]),
            (panels[2], ["Stage 4 — Rubin LSSTCam", "COSMOS | 8.4m Cerro Pachón | EDP2 co-add"]),
        ],
        OUT_DIR / "stage2_3_4_cosmos_comparison.svg",
    )


if __name__ == "__main__":
    main()
