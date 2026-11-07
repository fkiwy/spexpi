#!/usr/bin/env python3
"""Opt-in archive validation; keep downloaded FITS and outputs outside the checkout.

First run: python scripts/validate_qr3.py --cache-dir /path/to/cache --online
Repeat:    python scripts/validate_qr3.py --cache-dir /path/to/cache
Use --baseline-file for an exported pre-migration spherex_pipeline.py.
This is a fixed small compatibility sample, not a survey accuracy benchmark.
"""
import argparse
import concurrent.futures
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from photutils.psf import ImagePSF
import requests
from spexpi import spherex_pipeline as p
from spexpi import flag_policy

TARGET = dict(name='SIMP_J013656_validation', ra=24.2412498, dec=9.5630705,
              reference_epoch_yr=2016., pmra_masyr=1238.244, pmdec_masyr=-16.156)


def discover(release, detector):
    query = f"""SELECT TOP 1 a.uri FROM spherex.artifact a
        JOIN spherex.plane p ON a.planeid=p.planeid
        WHERE 1=CONTAINS(POINT('ICRS',{TARGET['ra']},{TARGET['dec']}),p.poly)
        AND a.uri LIKE '%/spherex/{release}/%/{detector}/level2_%'"""
    response = requests.get('https://irsa.ipac.caltech.edu/TAP/sync', timeout=90,
        params=dict(REQUEST='doQuery', LANG='ADQL', FORMAT='csv', QUERY=query))
    response.raise_for_status()
    rows = list(csv.DictReader(io.StringIO(response.text)))
    if not rows or 'uri' not in rows[0]:
        raise RuntimeError(f'No archive coverage for {release} D{detector}: {response.text[:200]}')
    return dict(release=release, detector=detector, uri=rows[0]['uri'])


def reference_epsf_fit(path, row, cfg, sapm, sapm_header):
    """Independent native rendering + whitened SVD fit at a fixed source position.

    Uses FITS alternate WCS directly, IRSA's nearest-zone/ImagePSF recipe, raw
    calibrated arrays and numpy.linalg.lstsq (not pipeline kernel/fitter helpers).
    It tests bookkeeping/numerics, not equivalence to the full Tractor service.
    """
    with fits.open(path) as hdul:
        header = hdul['IMAGE'].header
        x, y = row['x_src'], row['y_src']
        # The alternate detector WCS is linear. Do not apply the celestial SIP
        # distortion also stored in this header (all_pix2world would do that).
        xp, yp = WCS(header, key='A').wcs_pix2world(x, y, 0)
        table, eh = hdul['EPSF'].data, hdul['EPSF'].header
        index = int(np.argmin((table['XCENTER']-xp)**2+(table['YCENTER']-yp)**2))
        model = ImagePSF(table['EPSF'][index], oversampling=(eh['OVSMPY'],eh['OVSMPX']))
        radius = cfg.psf_kernel_radius_native
        yy, xx = np.mgrid[-radius:radius+1,-radius:radius+1]
        xx = xx + int(np.round(x)); yy = yy + int(np.round(y))
        native = model.evaluate(xx, yy, flux=1., x_0=x, y_0=y)
        native = native/native.sum()
        # Normalize full support before restricting to pixels present in the cutout.
        ny,nx=hdul['IMAGE'].data.shape
        inside=(xx>=0)&(yy>=0)&(xx<nx)&(yy<ny)
        xx,yy,native=xx[inside],yy[inside],native[inside]
        parent_x, parent_y = WCS(header,key='A').wcs_pix2world(xx, yy, 0)
        area = sapm[np.rint(parent_y).astype(int),np.rint(parent_x).astype(int)]
        scale = (u.Unit(header['BUNIT'])*area*u.Unit(sapm_header['BUNIT'])).to(u.uJy).value
        data = (hdul['IMAGE'].data[yy,xx]-hdul['ZODI'].data[yy,xx])*scale
        variance = hdul['VARIANCE'].data[yy,xx]*scale**2
        flags = hdul['FLAGS'].data[yy,xx]
        mask = sum(1<<bit for bit in p.resolved_flag_bits(cfg))
        valid = np.isfinite(data)&np.isfinite(variance)&(variance>0)&((flags&mask)==0)
        design = np.column_stack([native[valid], np.ones(valid.sum()),
                                  (xx-x)[valid], (yy-y)[valid]])
        weighted = design/np.sqrt(variance[valid,None])
        coefficients, _, rank, _ = np.linalg.lstsq(weighted, data[valid]/np.sqrt(variance[valid]),rcond=None)
        if rank != design.shape[1]:
            raise ValueError('Reference fit is rank deficient')
        covariance = np.linalg.pinv(weighted) @ np.linalg.pinv(weighted).T
        return dict(flux_uJy=float(coefficients[0]), error_uJy=float(np.sqrt(covariance[0,0])),
                    binx=int(table['BINX'][index]), biny=int(table['BINY'][index]),
                    grid=[len(np.unique(table['BINX'])),len(np.unique(table['BINY']))])


def load_baseline(path):
    spec = importlib.util.spec_from_file_location('spexpi_validation_baseline',path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_case(case, root, online, baseline):
    release, detector = case['release'], case['detector']
    cfg = p.PipelineConfig(data_release=release,photometry_method='both',
        download_dir=str(root/'data'),sapm_cache_dir=str(root/'sapm'),
        spectral_wcs_cache_dir=str(root/'swcs'),spectral_channels_cache_dir=str(root/'channels'),
        enable_psf_local_grid_search=False,psf_fit_background_gradient=True,
        aperture_radii_pix=(2.,),selected_aperture_radius_pix=2.,
        show_download_progress=False,retries=1,reduce_oversampling=False,plot_spectrum=False,
        **{k:TARGET[k] for k in ('reference_epoch_yr','pmra_masyr','pmdec_masyr')})
    url = 'https://irsa.ipac.caltech.edu/'+case['uri']+f"?center={TARGET['ra']},{TARGET['dec']}d&size=0.016667"
    directory = p.get_object_download_dir(cfg,TARGET['name'])
    path = directory/p.filename_from_uri(url,detector)
    if online:
        download = p.prepare_cutout_row_file(dict(uri=url,cutout_index=detector),directory,cfg,TARGET['ra'],TARGET['dec'])
        if download['download_status'].startswith('download_failed'):
            raise RuntimeError(download['download_error'])
    if not path.exists():
        raise FileNotFoundError(f'{path}; populate this cache with --online')
    if not online:
        def no_download(*args,**kwargs):
            raise RuntimeError('Offline validation attempted a calibration download')
        original = p.download_calibration
        p.download_calibration = no_download
    try:
        row = p.process_cutout_row(dict(local_uri=str(path),uri=url,data_release=release,cutout_index=detector),
                                  TARGET['ra'],TARGET['dec'],cfg)
        if 'image_hdu' not in row:
            raise RuntimeError(row.get('open_error',str(row)))
        if row['detector'] != detector:
            raise AssertionError('Archive detector mismatch')
        maps = {}
        measurements = p.measure_cutout_photometry(row,maps,cfg)
        output = dict(case, status='passed',path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      wavelength_um=row['central_wavelength_um'],mjd=float(row['mjd_avg']))
        for measurement in measurements:
            method=measurement['photometry_method']
            if measurement['status']!='ok' or not np.isfinite(measurement['flux_uJy']) or measurement['err_uJy']<=0:
                raise AssertionError(f'{method} failed: {measurement.get("psf_status",measurement["status"])}')
            output[method]=dict(flux_uJy=float(measurement['flux_uJy']),error_uJy=float(measurement['err_uJy']))
            if method=='psf':
                output[method]['reduced_chi2']=float(measurement['psf_reduced_chi2'])
        if release=='qr3':
            reference=reference_epsf_fit(path,row,cfg,*maps[(release,detector)])
            output['reference']=reference
            for name,key in [('flux_uJy','flux_uJy'),('error_uJy','error_uJy')]:
                if not np.isclose(output['psf'][name],reference[key],rtol=1.e-6,atol=1.e-6):
                    raise AssertionError(f'Independent fit mismatch: {output["psf"]} vs {reference}')
        elif baseline is not None:
            old_cfg=baseline.PipelineConfig(photometry_method='both',enable_psf_local_grid_search=False,
                psf_fit_background_gradient=True,aperture_radii_pix=(2.,),selected_aperture_radius_pix=2.)
            # Identical FITS and calibration inputs isolate numerical extraction changes.
            old=baseline.measure_cutout_photometry(row,dict(maps),old_cfg)
            if len(old)!=len(measurements):
                raise AssertionError('QR2 baseline measurement count differs')
            flux_difference=max(abs(a['flux_uJy']-b['flux_uJy']) for a,b in zip(measurements,old))
            psf_error_difference=max(abs(a['err_uJy']-b['err_uJy']) for a,b in zip(measurements,old)
                                     if a['photometry_method']=='psf')
            aperture_error_change=[a['err_uJy']-b['err_uJy'] for a,b in zip(measurements,old)
                                   if a['photometry_method']=='aperture']
            output['baseline_max_abs_flux_difference_uJy']=flux_difference
            output['baseline_max_abs_psf_error_difference_uJy']=psf_error_difference
            output['aperture_background_error_change_uJy']=aperture_error_change
            # Corrected annulus counts intentionally change aperture errors.
            # Flux and PSF-covariance behavior must retain the legacy baseline.
            if flux_difference>1.e-8 or psf_error_difference>1.e-8:
                raise AssertionError('QR2 flux/PSF error differs from original master')
            if any(change < -1.e-8 for change in aperture_error_change):
                raise AssertionError('Aperture error unexpectedly decreased versus old background count')

        return output
    finally:
        if not online:
            p.download_calibration=original


def validate_products(root, online):
    """Exercise cache discovery, channel binning, both output paths and provenance."""
    results=[]
    original=p.download_calibration
    if not online:
        def no_download(*args,**kwargs):
            raise RuntimeError('Offline product validation attempted a download')
        p.download_calibration=no_download
    try:
        for release in ('qr2','qr3'):
            cfg=p.PipelineConfig(data_release=release,photometry_method='both',query_irsa_for_cutouts=False,
                download_dir=str(root/'data'),sapm_cache_dir=str(root/'sapm'),
                spectral_wcs_cache_dir=str(root/'swcs'),spectral_channels_cache_dir=str(root/'channels'),
                output_dir=str(root/'products'),enable_psf_local_grid_search=False,
                psf_fit_background_gradient=True,aperture_radii_pix=(2.,),selected_aperture_radius_pix=2.,
                show_download_progress=False,retries=1,reduce_oversampling=True,
                spectral_channels_bin_by='edges',plot_spectrum=False,save_raw_measurements=True,
                **{k:TARGET[k] for k in ('reference_epoch_yr','pmra_masyr','pmdec_masyr')})
            spectra,measurements=p.run_spherex_pipeline(TARGET['ra'],TARGET['dec'],TARGET['name'],cfg)
            for method,table in spectra.items():
                if len(table)!=6 or set(table['DETECTOR'])!=set(range(1,7)):
                    raise AssertionError(f'{release} {method} does not contain all six detectors')
                path=root/'products'/release/'fits'/f"{TARGET['name']}_{method}_spectrum.fits"
                with fits.open(path) as hdul:
                    if hdul[0].header['DATAREL']!=release or hdul[0].header['PHOTMET']!=method:
                        raise AssertionError('Incorrect output provenance')
                    if release=='qr3' and method=='psf' and hdul[0].header['PSFMODEL']!='EPSF':
                        raise AssertionError('Missing EPSF provenance')
                results.append(dict(release=release,method=method,points=len(table),status='passed'))
        return results
    finally:
        p.download_calibration=original


def validate_images(root):
    """Check QR2/QR3 image reprojection, masking, FITS writing and plotting."""
    from spexpi.image_stacker import stack_images
    from spexpi.image_plotter import plot_images
    results=[]
    for release in ('qr2','qr3'):
        output=root/'images'/release
        # This small sample has one input per detector; retain the reference frame.
        images=stack_images(24.244929417,9.563023138,
            str(root/'data'/release/TARGET['name']),str(output),object_name=TARGET['name'],
            image_size_arcsec=30,save_fits=True,include_reference_image=True,verbose=False)
        if len(images)!=6 or not all(np.isfinite(i['hdu'].data).any() for i in images):
            raise AssertionError(f'{release} did not produce six finite detector images')
        path=plot_images(24.244929417,9.563023138,images,30,object_name=TARGET['name'],
                         output_dir=str(output),open_plot=False,plot_format='png')
        if Path(path).stat().st_size<1000:
            raise AssertionError('Image plot is missing or empty')
        results.append(dict(release=release,status='passed',detectors=6,
                            finite_data=True,plot_written=True))
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir',type=Path,required=True)
    parser.add_argument('--online',action='store_true')
    parser.add_argument('--baseline-file',type=Path)
    parser.add_argument('--manifest',type=Path,
        default=Path(__file__).resolve().parents[1]/'validation/qr3-stage4-manifest.json',
        help='Frozen sample manifest; defaults to the checked-in stage-four sample')
    args=parser.parse_args()
    root=args.cache_dir.resolve()
    checkout=Path(__file__).resolve().parents[1]
    if root==checkout or checkout in root.parents:
        parser.error('Use a cache directory outside the repository checkout')
    root.mkdir(parents=True,exist_ok=True)
    manifest=root/'manifest.json'
    if not manifest.exists() and args.manifest.exists():
        manifest.write_text(args.manifest.read_text())
    if not manifest.exists():
        if not args.online:
            parser.error('No cached manifest; use --online to select and download the sample')
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures=[executor.submit(discover,r,d) for r in ('qr2','qr3') for d in range(1,7)]
            cases=[future.result() for future in futures]
        manifest.write_text(json.dumps(dict(target=TARGET,cases=cases),indent=2)+'\n')
    frozen=json.loads(manifest.read_text())
    if frozen['target'] != TARGET:
        parser.error('Cached target differs from the fixed validation target')
    cases=frozen['cases']
    expected={(release,detector) for release in ('qr2','qr3') for detector in range(1,7)}
    if len(cases)!=12 or {(case['release'],case['detector']) for case in cases} != expected:
        parser.error('Manifest must contain exactly one case per release and detector')
    baseline=load_baseline(args.baseline_file) if args.baseline_file else None
    provenance=dict(flag_policy_sha256=hashlib.sha256(Path(flag_policy.__file__).read_bytes()).hexdigest(),
                    validation_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    baseline_comparison_requested=bool(args.baseline_file),pipeline_sha256=hashlib.sha256(Path(p.__file__).read_bytes()).hexdigest(),
                    baseline_sha256=(hashlib.sha256(args.baseline_file.read_bytes()).hexdigest()
                                     if args.baseline_file else None))
    results=[]
    for case in cases:
        print(f"Validating {case['release']} D{case['detector']}",flush=True)
        try:
            results.append(validate_case(case,root,args.online,baseline))
        except Exception as error:
            results.append(dict(case,status='failed',error=f'{type(error).__name__}: {error}'))
        (root/'results.json').write_text(json.dumps(dict(target=TARGET,provenance=provenance,cases=results),indent=2,allow_nan=False)+'\n')
        print(results[-1]['status'],results[-1].get('error',''),flush=True)
    failed=sum(case['status']!='passed' for case in results)
    products=[]
    images=[]
    if not failed:
        try:
            products=validate_products(root,args.online)
            images=validate_images(root)
        except Exception as error:
            products=[dict(status='failed',error=f'{type(error).__name__}: {error}')]
            failed+=1
    (root/'results.json').write_text(json.dumps(dict(target=TARGET,provenance=provenance,
        cases=results,products=products,images=images),indent=2,allow_nan=False)+'\n')
    print(f'{sum(c["status"]=="passed" for c in results)}/{len(results)} cases passed; product checks: {products}',flush=True)
    return bool(failed)


if __name__=='__main__':
    raise SystemExit(main())
