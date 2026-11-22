# DR1 spectral-image validation

The [frozen sample](dr1-manifest.json) uses six IRSA Level 2 cutouts of
SIMP J013656.5+093347.3, one per detector. The exposures come from planning
periods 2025W28 and 2025W30. The online run uses the pipeline's TAP query to
check that all six exposures are discoverable.

| Check | Result on November 22, 2026 |
| --- | --- |
| TAP discovery | 214 DR1 spectral images matched the target, including the six frozen exposures |
| Aperture and ePSF photometry | Both methods returned finite fluxes and positive uncertainties on all six detectors |
| Wavelength calibration | Full-resolution wavelengths and bandwidths were finite and positive on all six detectors |
| Independent ePSF fit | All six agreed with the Photutils and SVD reference, with a maximum relative flux difference of about 3.1 × 10⁻⁸ |
| Spectrum FITS products | Both methods produced six points with channel binning enabled, and headers recorded DR1, calibration collections, and flag masks |
| Image products | Six finite detector images and a diagnostic plot |

The sample checks compatibility and numerical agreement with an independent
fit at a fixed centroid. Absolute flux accuracy and agreement with IRSA's
spectrophotometry service are outside this validation.

From an installed checkout, use a fresh cache directory outside the repository.

```bash
MPLBACKEND=Agg python scripts/validate_qr3.py --release dr1 --cache-dir /tmp/spexpi-dr1-validation --online
MPLBACKEND=Agg python scripts/validate_qr3.py --release dr1 --cache-dir /tmp/spexpi-dr1-validation
```

The second command reuses the cutouts and calibrations offline. The cache also
contains `results.json` with per-detector measurements, FITS hashes, and product
checks.
