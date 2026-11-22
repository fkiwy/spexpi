"""Offline comparison of the bundled QR2 target and QR2 templates.

Run from src/. For QR3, change data_release AND both input locations to newly
extracted QR3 products; the bundled standards are QR2 and are not interchangeable.
"""
import os
import warnings

import astropy.units as u
from astropy.io import fits

from spexpi.spectrum_comparator import Spectrum, SpectrumTemplateComparator
from spexpi.spectrum_plotter import create_comparison_plot


data_release = "qr2"  # This demo intentionally uses the bundled QR2 library.


def extract_spectrum_from_fits(fits_file_path, expected_release):
    try:
        with fits.open(fits_file_path) as hdul:
            actual_release = str(hdul[0].header.get("DATAREL", "unknown")).lower()
            if actual_release != expected_release:
                raise ValueError(f"Expected {expected_release} spectrum, found {actual_release}; "
                                 "use target and templates from the same release")
            data = hdul[1].data
            wavelength = data["WAVELENGTH"]
            flux = data["FLUX"]
            error = data["ERR"]
            return wavelength, flux, error
    except Exception as e:
        raise RuntimeError(f"Cannot read spectrum {fits_file_path}: {e}") from e


warnings.simplefilter('ignore', category=Warning)

# Spectral units
wl_unit = u.um
flux_unit = u.Jy

# Load template spectra
data_dir = "../resources/standards/sequence/psf"
fits_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".fits"))

templates = []
for i, filename in enumerate(fits_files):
    full_path = os.path.join(data_dir, filename)
    wavelength, flux, flux_error = extract_spectrum_from_fits(full_path, data_release)

    wavelength *= wl_unit
    flux *= flux_unit
    flux_error *= flux_unit

    spt = filename.split()[0].replace("_", "")

    template = Spectrum(
        wavelength=wavelength,
        flux=flux,
        uncertainty=flux_error,
        label=spt,
        model_params={"spt": spt}
    )

    templates.append(template)

# Load observed spectrum
object_name = "T2 SIMP J013656.5+093347.3"
fits_path = "../resources/objects/spectra/fits/T2 SIMP J013656.5+093347.3_psf_spectrum.fits"

plot_observed_error = True
plot_template_error = False
plot_format = "pdf"
output_dir = f"../outputs/comparisons/{data_release}"

wavelength, flux, flux_error = extract_spectrum_from_fits(fits_path, data_release)

wavelength *= wl_unit
flux *= flux_unit
flux_error *= flux_unit

observed = Spectrum(
    label=object_name,
    wavelength=wavelength,
    flux=flux,
    uncertainty=flux_error
)

comparator = SpectrumTemplateComparator()
result = comparator.compare(
    observed,
    templates,
    trim_wave=True,
    number_of_matches=3,
    metric="reduced-chi2",
    normalize=True,
    scaling_mode="least-squares",
)

create_comparison_plot(
    result,
    plot_observed_error=plot_observed_error,
    plot_template_error=plot_template_error,
    object_name=object_name,
    output_dir=output_dir,
    plot_format=plot_format,
    open_plot=False,
)
