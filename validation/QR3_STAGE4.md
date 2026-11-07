# QR3 compatibility validation — stage four

Validation date: 7 November 2026. Scientific pipeline revision tested:
`585f121e4568f13babb72b98b71a3c046f46ad73`.
This is a historical snapshot before the subsequent [flag-policy and background
uncertainty fixes](QR3_FLAGS.md). The current runner records intentional aperture
error corrections separately while enforcing legacy QR2 flux and PSF-error checks.

Original `master` baseline: `96b9ff95b4497f18b7cfc75dc6b38bcb3aff1b0f`.

The local compatibility checks pass. The separate comparison with IRSA's remote
spectrophotometry service remains unavailable because that service returned an
internal error. This is sufficient evidence for implementation review, not a
survey-wide photometric accuracy claim or authorization to merge.

## Results

| Check | Outcome |
| --- | --- |
| Offline regression suite | 34 tests passed |
| Live archive sample | 12/12 cutouts passed: QR2 and QR3, detectors 1–6 |
| QR2 extraction versus original `master` | Zero difference in aperture/PSF flux and formal error on all six cases |
| QR3 independent native-model/SVD fit | Six cases passed; maximum relative flux difference about 3.1e-8 |
| End-to-end spectrum products | Four FITS spectra passed: two releases × two methods, six detector points each, channel binning and provenance checked |
| Image processing | Both releases produced six finite detector FITS images and a diagnostic plot |
| Remote IRSA/Tractor comparison | Unavailable: remote job ended in `ERROR`, with no photometry returned |

The frozen [sample manifest](qr3-stage4-manifest.json) identifies the exact
exposures. The [numeric results](qr3-stage4-results.json) include input FITS hashes,
source-code hashes, tool versions, fitted fluxes/errors, wavelengths, quality
statistics, reference fits and product checks. FITS hashes describe the local
pipeline cache, including release metadata stamped during download.

All source FITS, calibration files and generated products were kept outside the
checkout. The bundled science products and production pipeline were unchanged
in this validation stage.

## Sample and methods

The sample is one exposure per detector in each release, covering
SIMP J013656.5+093347.3. It uses reference position
(24.2412498, 9.5630705) degrees, epoch 2016.0, and proper motion
(1238.244, -16.156) mas/year. Coordinates are propagated to each exposure.
QR2 frames come from 2025W28/2025W30; QR3 frames come from 2026W30.
The manifest was frozen before numerical comparisons; a transient QR2 D5 `503`
was resolved by retrieving the same exposure again, without changing the sample.

Each case measures a 2-pixel aperture and a fixed-position PSF fit with a
constant and linear background. Centroid search is disabled to isolate the
rendering/fit comparison; separate regression tests exercise centroid search,
subpixel phases, masks, partial fitting windows, detector edges, injected-noise
uncertainties, malformed models and release/calibration failure paths.

For QR2, the original `master` module receives the same prepared cutout and
pixel-area map as the new module. Extraction results agree exactly. This does
not assert that rebinned spectra are identical: parsing `SUBCHAN`, `WL_MIN` and
`WL_MAX` was deliberately corrected in stage two.

For QR3, the reference implementation independently:

1. Uses the linear alternate detector WCS to map source and pixel coordinates.
2. Selects the nearest ePSF zone center directly from the FITS table.
3. Renders native pixel values with Photutils `ImagePSF`, following
   [IRSA's tutorial](https://github.com/Caltech-IPAC/irsa-tutorials/blob/b4e0c3541fd926c8456cf12bd1f4ec7fab7bb84d/tutorials/spherex/spherex_psf.md).
4. Normalizes the full model before clipping it to the observed image.
5. Converts the raw image, zodiacal background and variance using the pixel-area
   map, then solves the whitened fit with `numpy.linalg.lstsq` and an SVD
   pseudoinverse covariance.

It does not call the production kernel builder or weighted fitter. It does share
Photutils' interpolator and the archive calibration inputs; it is a check of
coordinate bookkeeping and numerical implementation, not an independent
astrophysical calibration. Agreement tolerances are 1e-6 relative and 1e-6 uJy
absolute for flux/error. QR2's baseline tolerance is 1e-8 uJy absolute.

The reference checker initially applied celestial SIP distortion to the linear
alternate detector WCS. This was corrected to use the low-level linear transform,
and a regression test now covers SIP headers and clipped model stamps. No
corresponding production bug was found: the production code uses the detector
reference offsets directly, as IRSA documents.

## Scientific limits and fit quality

QR3 uses the expected 21 × 21 zone grid in detectors 1, 2, 4, 5 and 6, and the
11 × 41 grid in detector 3. Its fixed-position PSF fits have the following
reduced chi-square values:

| Detector | Wavelength (um) | Reduced chi-square |
| --- | ---: | ---: |
| 1 | 1.067658 | 7.687 |
| 2 | 1.307617 | 13.607 |
| 3 | 1.788901 | 5.183 |
| 4 | 3.647407 | 3.056 |
| 5 | 4.071297 | 1.340 |
| 6 | 4.547261 | 1.061 |

Numerical agreement does not make high-residual fits scientifically acceptable.
The field has neighbouring sources, the fitter models one source, and formal
errors do not include the archive's extra systematic floor or uncertainty in a
searched centroid. Use fit diagnostics and quality cuts. The image check has only
one exposure per detector, so it exercises reprojection, masking and file-format
compatibility rather than a multi-epoch stacking accuracy benchmark.

Do not compare these QR2 and QR3 fluxes as a direct calibration-ratio test: they
sample different wavelengths and epochs in a variable brown dwarf. This run does
not validate mixed-release combination, absolute flux accuracy, all-sky behavior,
or equivalence to the full archive Tractor processing.

## IRSA service comparison

The deployed Firefly interface accepted a single-exposure photometry request at
`https://irsa.ipac.caltech.edu/applications/spherex/CmdSrv/async`, with
`cmd=tableSearch` and a `SpectrophotometryProcessor` request. The backend confirmed:

```json
{
  "POINT": "24.244929417,9.563023138",
  "TIME": "61241.1108,61241.1123",
  "DETECTOR": "D3"
}
```

The job transitioned from `QUEUED` through `EXECUTING` to `ERROR`, reporting:

> Internal error; contact IRSA user support (Unexpected failure during VOTable file creation or modification: )

No remote flux or uncertainty was returned, so no service-comparison pass is
claimed. Revisit this comparison when the external service succeeds. The
independent reference tests above do not substitute for that missing result.

## Reproduce

Use an environment with the project installed. From the checkout:

```bash
python -m unittest discover -s tests -v

git show 96b9ff95b4497f18b7cfc75dc6b38bcb3aff1b0f:src/spexpi/spherex_pipeline.py > /tmp/spexpi-master-baseline.py

MPLBACKEND=Agg python scripts/validate_qr3.py \
  --cache-dir /tmp/spexpi-qr3-validation \
  --baseline-file /tmp/spexpi-master-baseline.py \
  --online
```

The runner uses the committed manifest by default, writes caches/products outside
the repository, and exits nonzero if a case or product check fails. Network access
to IRSA and the public SPHEREx S3 bucket is required for the first run. Individual
archive retrievals can fail transiently; cached successful inputs are retained.
It does not submit remote photometry jobs.

After the cache is populated, omit `--online` to repeat without archive queries
or calibration downloads:

```bash
MPLBACKEND=Agg python scripts/validate_qr3.py \
  --cache-dir /tmp/spexpi-qr3-validation \
  --baseline-file /tmp/spexpi-master-baseline.py
```

Omitting `--baseline-file` skips the comparison with original QR2 extraction;
that omission is recorded in the generated results. The committed report used
the baseline. The suite and final archive/product checks passed in the prepared
cloud environment; a fresh-environment installation was not part of this stage.
