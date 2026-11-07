# QR3 pixel-mask implementation and validation

Implemented on the migration feature branch after the original stage-four
snapshot. Policy source: [SPHEREx Explanatory Supplement v2](https://irsa.ipac.caltech.edu/data/SPHEREx/docs/SPHEREx_Expsupp_QR.pdf),
Table 16 and sections 3.3/3.4.3. Validation date: 7 November 2026.

## Policies

`spexpi.flag_policy` centralizes L2 bit positions. Source, annulus, diagnostic and
mosaic policies are independently resolved, rather than reusing one list.

- QR3 source: conservative exclusions, adding CROSSTALK (20), GHOST_FPA (23,
  currently unused) and STREAK (25), and omitting undefined bit 4.
- QR3 background: the conservative source default plus SOURCE (21).
- QR3 mosaics: the archive mosaic exclusions, retaining OVERFLOW (1), OUTLIER
  (19) and SOURCE (21). Input contamination is masked before interpolation.
- QR2: legacy source and mosaic bit lists remain unchanged. Background-count
  and reprojection corrections apply to both releases.

The QR3 source policy deliberately exceeds the shorter archive source-fit list.
It retains DICHROIC and PHANMISS for calibration quality and masks transient,
overflow, outlier and optical-artifact pixels conservatively. It is not an exact
reproduction of archive Tractor photometry.

`None` selects defaults by release; an explicit tuple overrides one policy and
`()` disables that bit mask. Direct helper callers should use
`resolved_flag_bits(cfg, purpose)` instead of assuming configuration fields
contain a materialized tuple. L3 measurement bit 32 is rejected as an L2 bit index.

## Aperture behavior and uncertainties

QR3 defaults to `aperture_masked_policy="reject"`: any missing source-aperture area
from masking, invalid variance or detector/image boundaries rejects that aperture.
The comparison uses exact fractional aperture weights, not only pixel centers.
Explicit `"allow"` reports the surviving flux without fabricating a source-flux
correction; `aperture_missing_flux_allowed` and `aperture_masked_fraction` identify
that condition. QR2 retains its legacy allow default.

The annulus sample count is taken from `ApertureStats.data_cutout` after actual
geometric masking, configured bit masking, nonfinite-data rejection and sigma
clipping. It therefore matches the samples used for background statistics. The
existing background variance approximation remains `std**2 / N`; this work fixes
N, not estimator-specific statistical assumptions or systematic uncertainty floors.

QR3 does not silently retry with SOURCE pixels admitted. An explicit
`allow_source_background_fallback=True` enables that behavior, and diagnostics
record it. QR2 retains the legacy fallback default but now reports its use.
The raw background mask is the actual mask used after any relaxation; the FITS
`BGMASK` describes the configured policy. `SRCMASK`, `APMASK`, `BGCOUNT` and
`BGSRCFB` provide additional product provenance. Original L2 flags near the source
are retained as `warning_flags`, independently of fit exclusions.

## Image stacking

Science pixels are masked in their input detector grid before reprojection.
Contributor FLAGS are propagated using bitwise OR of the nonzero bilinear input
contributors, so no interpolated flag values are cast into fabricated bit patterns.
Missing coverage carries MISSING_DATA (9); fractional input flag values are rejected.
Signed 32-bit input fields preserve their original bit patterns.

Each cutout retains the actual fatal mask used, including custom overrides.
Stacks reject inconsistent mask-policy labels within a detector. `FLAGPOL` and
`MASKBITS` are retained in stacked FITS headers. Reprojection and masking fixes
can change existing image products, even when the exclusion list is unchanged.

## Validation

- **43 automated tests passed**, including all earlier migration regressions.
- Added tests cover policy separation/overrides; real sigma-clipped annulus counts
  and their effect on uncertainty; explicit/reported SOURCE relaxation; rejected
  source area/invalid variance; crosstalk removal from a PSF fit; preservation of
  SOURCE pixels in the fit; original warning bits; contributor OR; missing
  coverage; signed fields; exclusion of a bright contaminated pixel before
  interpolation; release-specific mosaic policies; and mixed-policy rejection.
- The previous masked-aperture regression now verifies both the default QR3
  rejection and the explicit allow mode, which retains the old partial-flux result.
- The cached 12-cutout matrix passed: QR2/QR3 detectors 1–6. Four six-detector
  aperture/PSF spectra and both six-detector image workflows passed.
- All six QR2 aperture and PSF fluxes, and all PSF formal errors, still agree
  with original `master`. Aperture errors intentionally increase because the
  effective background sample counts are now correct. In these examples the
  increases range from about 1.38 to 173.77 uJy, depending on the annulus data.
- QR3 PSF fits still agree with the independent native-rendering/SVD reference.
  The cached small sample contains no crosstalk, streak or GHOST_FPA pixels;
  synthetic regressions exercise those policies rather than claiming empirical
  coverage of contaminated fields from this sample.

The [numeric results](qr3-flags-results.json) record hashes and numerical
comparisons. The [original stage-four report](QR3_STAGE4.md) remains historical
and refers to its earlier pipeline revision. Its exact aperture-error equality
claim is not a claim about this corrected version. The validation runner now
checks unchanged QR2 flux/PSF errors while separately recording the intentional
aperture-error change.

Reproduce with the stage-four commands and frozen manifest. No additional data
are needed once that cache is populated. The earlier remote IRSA photometry job
failed inside the service; this work does not claim that comparison succeeded.
