"""Verify cube pixel sampling and coverage without a network connection."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

from spexpi.cube_preview import PositionOutsideTile, read_cube_preview


class CubePreviewTests(unittest.TestCase):
    def test_samples_target_and_keeps_missing_channel(self):
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crval = [265., 25.]
        wcs.wcs.crpix = [1., 2.]
        wcs.wcs.cdelt = [-0.004, 0.004]
        header = wcs.to_header()
        header["BUNIT"] = "MJy/sr"
        brightness = np.arange(36, dtype=np.float32).reshape(3, 3, 4)
        counts = np.ones_like(brightness, dtype=np.int32)
        flags = np.zeros_like(counts)
        brightness[1, 1, 0] = np.nan
        counts[1, 1, 0] = 0
        flags[2, 1, 0] = 8
        wavelengths = fits.Column(name="WAVELENGTHS", format="3E", unit="um", array=[[1., 2., 3.]])
        with TemporaryDirectory() as directory:
            tile = Path(directory) / "test_cube.fits"
            fits.HDUList([
                fits.PrimaryHDU(),
                fits.ImageHDU(brightness, header=header, name="IMAGE"),
                fits.ImageHDU(counts, name="NHITS"),
                fits.ImageHDU(flags, name="FLAGS"),
                fits.BinTableHDU.from_columns([wavelengths], name="WCS-WAVE"),
            ]).writeto(tile)
            result = read_cube_preview(tile, 265., 25., 2.9, radius_pix=1)
            with self.assertRaises(PositionOutsideTile):
                read_cube_preview(tile, 266., 25., 2.9)
        np.testing.assert_array_equal(result.spectrum["surface_brightness"], [4., np.nan, 28.])
        np.testing.assert_array_equal(result.spectrum["nhits"], [1, 0, 1])
        np.testing.assert_array_equal(result.spectrum["flags"], [0, 0, 8])
        np.testing.assert_array_equal(result.spectrum["valid"], [True, False, True])
        self.assertEqual(result.field_wavelength, 3.)
        np.testing.assert_array_equal(result.field, brightness[2, :, :2])
        np.testing.assert_allclose(result.marker, [0., 1.], atol=1e-10)
