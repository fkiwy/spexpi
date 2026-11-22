"""Edit the settings below and run with an installed copy of spexpi."""
import sys

from spexpi.cube_preview import main

RA_DEG = 265.349
DEC_DEG = 25.882
WAVELENGTH_UM = 4.0
RADIUS_PIX = 20
OUTPUT = "../outputs/cube_preview/t9"
CACHE_DIR = "../outputs/cube_preview/cache"
TILE = "../outputs/cube_preview/cache/hips_20261026/Norder5/Dir0/Npix8160_cube.fits"  # Set to a local FITS filename to run offline.

if __name__ == "__main__":
    settings = [
        str(RA_DEG),
        str(DEC_DEG),
        "--wavelength", str(WAVELENGTH_UM),
        "--radius-pix", str(RADIUS_PIX),
        "--output", OUTPUT,
        "--cache-dir", CACHE_DIR,
    ]
    if TILE is not None:
        settings.extend(["--tile", str(TILE)])
    raise SystemExit(main(settings + sys.argv[1:]))
