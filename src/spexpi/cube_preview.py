"""Preview the surface brightness of one SPHEREx DR1 All-Sky Cube pixel."""
from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urlsplit

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import requests


IRSA_SIA = "https://irsa.ipac.caltech.edu/SIA"


class PositionOutsideTile(ValueError):
    """The requested position does not fall within this tile's pixel grid."""


@dataclass
class CubePreview:
    spectrum: Table
    field: np.ndarray
    field_wcs: WCS
    marker: tuple[float, float]
    field_wavelength: float


def find_cube_tiles(ra: float, dec: float) -> list[str]:
    """Return candidate DR1 tile URLs from IRSA's SIA2 cube collection."""
    response = requests.get(
        IRSA_SIA,
        params={
            "COLLECTION": "spherex_dr1_cube",
            "POS": f"CIRCLE {ra} {dec} 0.0001",
            "RESPONSEFORMAT": "CSV",
        },
        timeout=45,
    )
    response.raise_for_status()
    records = csv.DictReader(response.text.splitlines())
    if "access_url" not in (records.fieldnames or []):
        raise ValueError("IRSA did not return a cube discovery table with access URLs.")
    urls = list(dict.fromkeys(row["access_url"] for row in records if row["access_url"]))
    if not urls:
        raise ValueError("IRSA returned no DR1 cube tiles for this position.")
    return urls


def cache_cube_tile(url: str, cache_dir: Path) -> Path:
    """Download a tile into a versioned cache and publish the file atomically."""
    parsed = urlsplit(url)
    parts = Path(parsed.path).parts
    if (parsed.scheme, parsed.hostname) != ("https", "irsa.ipac.caltech.edu") or (
        len(parts) < 5 or parts[-5] != "hips5_cube" or not parts[-1].endswith("_cube.fits")
        or any(part in (".", "..") for part in parts[-5:])
    ):
        raise ValueError(f"Unexpected IRSA cube tile URL: {url}")
    path = cache_dir.joinpath(*parts[-4:])
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as temp:
        temporary_path = Path(temp.name)
    try:
        with requests.get(url, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            with temporary_path.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    output.write(chunk)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def read_cube_preview(
    tile: Path, ra: float, dec: float, wavelength: float, radius_pix: int = 20,
) -> CubePreview:
    """Read one spatial pixel across all channels and a field in the closest plane."""
    if not np.isfinite(wavelength) or wavelength <= 0 or radius_pix < 1:
        raise ValueError("Wavelength and field radius must be positive.")
    target = SkyCoord(ra, dec, unit="deg", frame="icrs")
    with fits.open(tile, memmap=True) as hdul:
        image = hdul["IMAGE"]
        channels, height, width = image.shape
        wcs = WCS(image.header, fobj=hdul, naxis=[1, 2])
        x, y = (float(value) for value in wcs.world_to_pixel(target))
        if not (-0.5 <= x < width - 0.5 and -0.5 <= y < height - 0.5):
            raise PositionOutsideTile(f"The position is outside cube tile {tile.name}.")
        ix, iy = int(np.floor(x + 0.5)), int(np.floor(y + 0.5))
        spectral = hdul["WCS-WAVE"]
        wave = u.Quantity(
            spectral.data["WAVELENGTHS"].reshape(-1),
            spectral.columns["WAVELENGTHS"].unit or image.header["CUNIT3"],
        ).to_value(u.um)
        if len(wave) != channels or not np.isfinite(wave).all():
            raise ValueError("Invalid wavelength lookup for the cube image.")
        if not wave.min() <= wavelength <= wave.max():
            raise ValueError(f"Field wavelength must be within {wave.min():.3f}–{wave.max():.3f} µm.")
        plane = int(np.argmin(np.abs(wave - wavelength)))
        brightness = np.array(image.section[:, iy, ix])
        hits = np.array(hdul["NHITS"].section[:, iy, ix])
        flags = np.array(hdul["FLAGS"].section[:, iy, ix], dtype=np.uint32)
        spectrum = Table({
            "wavelength": wave * u.um,
            "surface_brightness": brightness * u.Unit(image.header["BUNIT"]),
            "nhits": hits,
            "flags": flags,
            "valid": np.isfinite(brightness) & (hits > 0),
        })
        spectrum.meta.update(tile=tile.name, ra_deg=ra, dec_deg=dec, pixel_x=ix, pixel_y=iy)
        if "VARIANCE" in hdul:
            variance = np.array(hdul["VARIANCE"].section[:, iy, ix], dtype=float)
            spectrum["uncertainty"] = np.sqrt(np.where(variance >= 0, variance, np.nan)) * spectrum["surface_brightness"].unit
        x0, x1 = max(0, ix - radius_pix), min(width, ix + radius_pix + 1)
        y0, y1 = max(0, iy - radius_pix), min(height, iy + radius_pix + 1)
        field = np.array(image.section[plane, y0:y1, x0:x1])
        return CubePreview(spectrum, field, wcs[y0:y1, x0:x1], (x - x0, y - y0), float(wave[plane]))


def write_preview(preview: CubePreview, output: Path) -> None:
    """Save the channel samples and a figure showing spectrum, coverage, and field."""
    table = preview.spectrum
    output.parent.mkdir(parents=True, exist_ok=True)
    table.write(output.with_suffix(".ecsv"), overwrite=True)
    font = "Arial"
    try:
        font_manager.findfont(font, fallback_to_default=False)
    except ValueError:
        font = "DejaVu Sans"
        logging.warning("Arial is unavailable. Using DejaVu Sans for this PDF.")

    with plt.rc_context({
        "font.family": font,
        "mathtext.fontset": "custom",
        "mathtext.rm": font,
        "mathtext.it": f"{font}:italic",
        "mathtext.bf": f"{font}:bold",
        "mathtext.sf": font,
        "mathtext.cal": font,
        "pdf.fonttype": 42,
    }):
        figure = plt.figure(figsize=(11, 5), layout="constrained")
        try:
            grid = figure.add_gridspec(2, 2, height_ratios=(3, 1))
            ax = figure.add_subplot(grid[0, 0])
            coverage = figure.add_subplot(grid[1, 0], sharex=ax)
            field = figure.add_subplot(grid[:, 1], projection=preview.field_wcs)
            wave = table["wavelength"]
            brightness = table["surface_brightness"]
            valid_brightness = np.where(table["valid"], brightness, np.nan)
            ax.plot(wave, valid_brightness, marker=".", markersize=3)
            if "uncertainty" in table.colnames:
                ax.errorbar(wave, valid_brightness, yerr=table["uncertainty"], fmt="none", alpha=0.3)
            ax.axvline(preview.field_wavelength, color="0.5", linestyle=":")
            ax.set_ylabel(f"Surface brightness ({brightness.unit})")
            ax.tick_params(labelbottom=False)
            coverage.step(wave, table["nhits"], where="mid")
            coverage.set(xlabel="Wavelength (µm)", ylabel="Hits")
            finite = preview.field[np.isfinite(preview.field)]
            limits = dict(zip(("vmin", "vmax"), np.percentile(finite, [2, 98]))) if finite.size else {}
            rendered = field.imshow(preview.field, origin="lower", cmap="gray", **limits)
            field.plot(*preview.marker, marker="+", color="tab:red", markersize=12)
            field.set_title(f"{preview.field_wavelength:.3f} µm")
            field.coords[0].set_axislabel("RA (ICRS)")
            field.coords[1].set_axislabel("Dec (ICRS)")
            figure.colorbar(rendered, ax=field, label=str(brightness.unit))
            figure.suptitle(f"SPHEREx DR1 cube at ({table.meta['ra_deg']:.6f}°, {table.meta['dec_deg']:.6f}°)")
            figure.savefig(output.with_suffix(".pdf"), dpi=160)
        finally:
            plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ra", type=float, help="ICRS right ascension in degrees")
    parser.add_argument("dec", type=float, help="ICRS declination in degrees")
    parser.add_argument("--wavelength", type=float, default=3.5, help="Field wavelength in microns")
    parser.add_argument("--radius-pix", type=int, default=20, help="Field radius in cube pixels")
    parser.add_argument("--tile", type=Path, help="Use a local FITS tile without querying IRSA")
    parser.add_argument("--cache-dir", type=Path, default=Path("spherex_cube_cache"))
    parser.add_argument("--output", type=Path, default=Path("cube_preview"), help="Output filename stem")
    args = parser.parse_args(argv)
    if not (0 <= args.ra < 360 and -90 <= args.dec <= 90):
        parser.error("Use RA in [0, 360) and Dec in [-90, 90] degrees.")
    if not np.isfinite(args.wavelength) or args.wavelength <= 0 or args.radius_pix < 1:
        parser.error("Wavelength and field radius must be positive.")

    try:
        if args.tile is not None:
            preview = read_cube_preview(args.tile, args.ra, args.dec, args.wavelength, args.radius_pix)
        else:
            for url in find_cube_tiles(args.ra, args.dec):
                tile = cache_cube_tile(url, args.cache_dir)
                try:
                    preview = read_cube_preview(tile, args.ra, args.dec, args.wavelength, args.radius_pix)
                    break
                except PositionOutsideTile:
                    continue
            else:
                raise ValueError("None of the returned DR1 cube tiles contains this position.")
        write_preview(preview, args.output)
    except (ValueError, OSError, KeyError, requests.RequestException) as exc:
        parser.exit(1, f"Cube preview failed: {exc}\n")
    print(f"Saved {args.output.with_suffix('.ecsv')} and {args.output.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
