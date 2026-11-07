"""Synthetic QR3 calibration and aperture extraction, without archive access."""
import gzip
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from spexpi import spherex_pipeline as p


class QR3ApertureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = p.PipelineConfig(data_release='qr3', photometry_method='aperture',
            download_dir=str(self.root/'data'), output_dir=str(self.root/'out'),
            sapm_cache_dir=str(self.root/'sapm'), spectral_wcs_cache_dir=str(self.root/'swcs'),
            spectral_channels_cache_dir=str(self.root/'channels'),query_irsa_for_cutouts=False,
            reduce_oversampling=False,plot_spectrum=False,save_raw_measurements=True,
            show_download_progress=False,retries=1,aperture_radii_pix=(3.,),selected_aperture_radius_pix=3.)
        y,x = np.mgrid[:80,:90]
        self.cwave=1.+x*.001+y*.002
        self.cband=np.full_like(self.cwave,.02)
        self.area=36.+x*.01+y*.02
        for detector in (1,4,5,6):
            coll=p.calibration_collection(self.cfg,'qr3','spectral_wcs')
            directory=Path(self.cfg.spectral_wcs_cache_dir)/'qr3'/coll
            directory.mkdir(parents=True,exist_ok=True)
            hdus=[fits.PrimaryHDU()]
            for name,data in [('CWAVE',self.cwave),('CBAND',self.cband)]:
                hdus.append(fits.ImageHDU(data,header=fits.Header({'BUNIT':'um','DETCOORD':'sky',
                            'DETECTOR':detector}),name=name))
            fits.HDUList(hdus).writeto(directory/f'spectral_wcs_D{detector}_spx_{coll}.fits')
            fits.HDUList([fits.PrimaryHDU(),fits.ImageHDU(self.area, name='IMAGE',
                header=fits.Header({'BUNIT':'arcsec2','DETCOORD':'sky','DETECTOR':detector}))]).writeto(
                    p.sapm_local_path(detector,self.cfg,'qr3'))

    def make_cutout(self,detector=1):
        wcs=WCS(naxis=2)
        wcs.wcs.ctype=['RA---TAN','DEC--TAN'];wcs.wcs.crpix=[16,16]
        wcs.wcs.crval=[24.24,9.56];wcs.wcs.cdelt=[-6.2/3600,6.2/3600]
        header=wcs.to_header()
        header.update({'DETCOORD':'sky','CRPIX1A':-19.,'CRPIX2A':-29.,
                       'CRVAL1A':0.,'CRVAL2A':0.,'CDELT1A':1.,'CDELT2A':1.,
                       'BUNIT':'MJy/sr','DETECTOR':detector,'DATAREL':'qr3','MJD-OBS':61250.})
        y,x=np.mgrid[:31,:31]
        source=np.exp(-((x-15.)**2+(y-15.)**2)/(2*.6**2))
        source*=1000./source.sum()
        scale=(u.MJy/u.sr*self.area[30:61,20:51]*u.arcsec**2).to_value(u.uJy)
        # Nonuniform pixel area and zodi require the real calibration path to recover flux.
        image=(source+100.)/scale+2.
        directory=Path(self.cfg.download_dir)/'qr3'/'synthetic';directory.mkdir(parents=True,exist_ok=True)
        path=directory/'cutout.fits'
        fits.HDUList([fits.PrimaryHDU(header=fits.Header({'VERSION':'7.0.5','DATAREL':'qr3'})),
            fits.ImageHDU(image,header=header,name='IMAGE'),
            fits.ImageHDU(np.ones_like(image)/scale**2,name='VARIANCE'),
            fits.ImageHDU(np.zeros_like(image,dtype=np.int32),name='FLAGS'),
            fits.ImageHDU(np.full_like(image,2.),name='ZODI')]).writeto(path,overwrite=True)
        return path,header

    def test_end_to_end_flux_wavelength_uncertainty_and_provenance(self):
        self.make_cutout()
        with patch.object(p,'update_psf_header',side_effect=AssertionError('old PSF path used')), \
             patch.object(p,'download_calibration',side_effect=AssertionError('network used')):
            spec,measurements=p.run_spherex_pipeline(24.24,9.56,'synthetic',self.cfg)
        self.assertEqual(len(spec),1)
        self.assertAlmostEqual(spec['FLUX'][0],.001,delta=1.e-7)
        self.assertGreater(spec['ERR'][0],0)
        self.assertAlmostEqual(spec['WAVELENGTH'][0],1.+35*.001+45*.002,places=8)
        self.assertEqual(measurements[0]['wavelength_source'],'CWAVE')
        self.assertEqual(measurements[0]['sapm_x0_parent'],20)
        self.assertEqual(measurements[0]['sapm_y0_parent'],30)
        with fits.open(self.root/'out/qr3/fits/synthetic_aperture_spectrum.fits') as h:
            self.assertEqual(h[0].header['DATAREL'],'qr3')
            self.assertEqual(h[0].header['WAVESRC'],'CWAVE')
            self.assertEqual(h[0].header['SWCSCOLL'],'cal-swcs-v5-2026-191')
        self.assertEqual(self.cfg.output_dir,str(self.root/'out'))

    def test_subpixel_mapping_mwir_orientation_and_memory_cache(self):
        for detector in (4,5,6):
            _,header=self.make_cutout(detector)
            wave,band=p.qr3_wavelength(header,(31,31),15.25,14.75,detector,self.cfg)
            self.assertAlmostEqual(wave,1.+35.25*.001+44.75*.002)
            self.assertAlmostEqual(band,.02)
        first=p.load_spectral_wcs(4,self.cfg,'qr3')
        with patch.object(p.fits,'open',side_effect=AssertionError('map loaded twice')):
            self.assertIs(p.load_spectral_wcs(4,self.cfg,'qr3'),first)

    def test_missing_offsets_bad_orientation_and_off_detector_fail(self):
        _,header=self.make_cutout()
        for change in ({'DETCOORD':'detector'},{'CRPIX1A':np.nan},{'CDELT1A':-1}):
            h=header.copy()
            # FITS headers cannot contain NaN; use a mapping for invalid-value checks.
            h=dict(h);h.update(change)
            with self.assertRaises(ValueError):
                p.qr3_wavelength(h,(31,31),15,15,1,self.cfg)
        h=header.copy();del h['CRPIX1A']
        with self.assertRaisesRegex(ValueError,'missing'):
            p.qr3_wavelength(h,(31,31),15,15,1,self.cfg)
        with self.assertRaisesRegex(ValueError,'outside'):
            p.qr3_wavelength(header,(31,31),-200,15,1,self.cfg)

    def test_pixel_area_units_and_variance(self):
        area=np.full((2,2),36.)
        sky=fits.Header({'BUNIT':'MJy/sr'})
        arc=fits.Header({'BUNIT':'arcsec2'});sr=fits.Header({'BUNIT':'sr'})
        area_sr=(area*u.arcsec**2).to_value(u.sr)
        converted=p.convert_to_ujy(np.ones((2,2)),sky,area,arc)
        np.testing.assert_allclose(converted,p.convert_to_ujy(np.ones((2,2)),sky,area_sr,sr))
        np.testing.assert_allclose(p.convert_variance_to_ujy2(np.full((2,2),4.),sky,area,arc),4*converted**2)

    def test_published_channel_centers_and_overlapping_envelopes(self):
        table=Table({'DETECTOR':[5,5,5],'SUBCHAN':[3,1,2],
                     'WAVELENGTH':[4.3,4.1,4.2],'WL_MIN':[4.24,4.04,4.14],
                     'WL_MAX':[4.36,4.16,4.26]})
        edges,centers=p.spectral_channel_edges_for_detector(table,5)
        np.testing.assert_allclose(centers,[4.1,4.2,4.3])
        np.testing.assert_allclose(edges,[4.04,4.15,4.25,4.36])
        self.cfg.spectral_channels_bin_by='edges'
        w,f,e,d=p.reduce_oversampling_with_spectral_channels(
            np.array([4.1,4.15,4.2,4.3]),np.array([1.,2.,2.,3.]),np.ones(4),np.full(4,5),table,self.cfg)
        np.testing.assert_allclose(w,centers)
        np.testing.assert_allclose(f,[1.,2.,3.])

    def test_masked_source_pixel_is_excluded(self):
        path,_=self.make_cutout()
        with fits.open(path,mode='update') as h:
            h['FLAGS'].data[15,15]=1
        spec,measurements=p.run_spherex_pipeline(24.24,9.56,'synthetic',self.cfg)
        y,x=np.mgrid[:31,:31]
        source=np.exp(-((x-15.)**2+(y-15.)**2)/(2*.6**2))
        missing=1000./source.sum()
        self.assertAlmostEqual(spec['FLUX'][0],(1000.-missing)*1.e-6,delta=1.e-7)
        self.assertEqual(measurements[0]['n_bad_aperture_pixels'],1)

    def test_detector_edge_clipping_keeps_area_map_aligned(self):
        header=fits.Header({'CRPIX1A':5.,'CRPIX2A':-9.,'DETCOORD':'sky'})
        area,slices,info=p.extract_sapm_for_cutout(self.area,header,(20,20))
        np.testing.assert_array_equal(area,self.area[10:30,:16])
        self.assertEqual(slices,(slice(0,20),slice(4,20)))
        self.assertEqual(info['sapm_x0_parent'],0)
        self.assertEqual(p.qr3_parent_coordinates(header,(20,20),self.area.shape),(-4.,10.))

    def test_no_cache_sapm_preserves_existing_file(self):
        path=p.sapm_local_path(1,self.cfg,'qr3')
        original=path.read_bytes()
        self.cfg.cache_sapm_files=False
        def download(key,temporary,timeout):
            self.assertNotEqual(temporary,path)
            temporary.write_bytes(original)
        with patch.object(p,'download_calibration',side_effect=download):
            area,header=p.load_sapm(1,self.cfg,'qr3')
        np.testing.assert_array_equal(area,self.area)
        self.assertEqual(path.read_bytes(),original)

    def test_download_preserves_compression_and_failed_refresh_preserves_cache(self):
        raw=io.BytesIO();fits.PrimaryHDU().writeto(raw)
        compressed=gzip.compress(raw.getvalue())
        path=self.root/'channel.fits.gz'
        with patch.object(p.requests,'get') as get:
            response=get.return_value.__enter__.return_value
            response.iter_content.return_value=[compressed]
            p.download_calibration('nasa-irsa-spherex/qr2/channel.fits.gz',path)
            self.assertEqual(path.read_bytes(),compressed)
            with fits.open(path) as h:
                self.assertEqual(len(h),1)
            response.iter_content.return_value=[b'invalid FITS']
            with self.assertRaises(OSError):
                p.download_calibration('nasa-irsa-spherex/qr2/channel.fits.gz',path)
        self.assertEqual(path.read_bytes(),compressed)
        self.assertFalse(path.with_name(path.name+'.part').exists())

    def test_qr3_channels_fail_closed_and_sparse_bins_keep_measured_wavelength(self):
        table=Table({'WAVELENGTH':[1.1,1.2],'DETECTOR':[1,1]})
        self.cfg.reduce_oversampling=True
        self.cfg.use_spectral_channels_for_binning=True
        spec=Table({'WAVELENGTH':[1.15],'FLUX':[.001],'ERR':[.0001],'DETECTOR':[1]})
        with patch.object(p,'load_spectral_channels_calibration',return_value=table), \
             patch.object(p,'reduce_oversampling_per_detector',side_effect=AssertionError('legacy bins used')):
            result=p.process_spectrum_table(spec,self.cfg)
        self.assertEqual(list(result['WAVELENGTH']),[1.15])
        with self.assertRaisesRegex(ValueError,'missing detector'):
            p.reduce_oversampling_with_spectral_channels(np.array([1.]),np.array([1.]),
                np.array([.1]),np.array([6]),table,self.cfg)
        with patch.object(p,'download_calibration',side_effect=OSError('offline')):
            with self.assertRaisesRegex(RuntimeError,'QR3 spectral-channel'):
                p.load_spectral_channels_calibration(self.cfg,'qr3')

    def test_qr3_download_without_channel_table_fails(self):
        def download(key,path,timeout):
            fits.PrimaryHDU().writeto(path,overwrite=True)
        with patch.object(p,'download_calibration',side_effect=download):
            with self.assertRaisesRegex(RuntimeError,'QR3 spectral-channel'):
                p.load_spectral_channels_calibration(self.cfg,'qr3')

    def test_qr3_psf_and_both_fail_before_processing(self):
        path,_=self.make_cutout()
        for method in ('psf','both'):
            self.cfg.photometry_method=method
            with self.assertRaisesRegex(NotImplementedError,'EPSF'):
                p.process_cutout_row({'local_uri':str(path),'data_release':'qr3'},24.24,9.56,self.cfg)


if __name__=='__main__':
    unittest.main()
