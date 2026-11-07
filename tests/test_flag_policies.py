"""Pixel policies, aperture quality, background counts and discrete reprojection."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.wcs import WCS
import astropy.units as u
from photutils.aperture import CircularAnnulus
from spexpi import spherex_pipeline as p
from spexpi import image_stacker as stack
from spexpi.flag_policy import *


class FlagPolicyTests(unittest.TestCase):
    def test_release_defaults_overrides_and_diagnostics_are_independent(self):
        self.assertEqual(p.resolved_flag_bits(p.PipelineConfig()),LEGACY_FATAL_BITS)
        cfg=p.PipelineConfig(data_release='qr3')
        self.assertEqual(p.resolved_flag_bits(cfg),QR3_SOURCE_BITS)
        for bit in (20,23,25):self.assertIn(bit,p.resolved_flag_bits(cfg))
        for bit in (4,12,21):self.assertNotIn(bit,p.resolved_flag_bits(cfg))
        self.assertIn(21,p.resolved_flag_bits(cfg,'background'))
        cfg.bad_flag_bits=()
        self.assertEqual(p.resolved_flag_bits(cfg),())
        self.assertIn(20,p.resolved_flag_bits(cfg,'warning'))
        self.assertIn(20,p.resolved_flag_bits(cfg,'background'))
        self.assertEqual(bitmask((20,20)),1<<20)
        self.assertEqual(bitmask(iter((20,))),1<<20)
        signed=np.array([[-2147483648]],dtype=np.int32)
        self.assertTrue(p.make_bad_pixel_mask(signed,np.ones((1,1)),(31,))[0,0])
        for bits in ((32,),(-1,),(1.5,),(True,)):
            with self.assertRaises(ValueError):p.validate_config(p.PipelineConfig(bad_flag_bits=bits))

    def test_actual_masked_and_clipped_background_count_controls_uncertainty(self):
        rng=np.random.default_rng(24)
        image=rng.normal(10,1,(41,41));image[20,20]+=100
        flags=np.zeros(image.shape,dtype=np.uint32)
        ann=CircularAnnulus((20,20),4,6).to_mask(method='center').to_image(image.shape)>0
        yy,xx=np.indices(image.shape)
        flags[ann&(xx<20)]=1<<21
        available=ann&(xx>=20)
        outliers=np.flatnonzero(available)[:2];image.flat[outliers]=1000
        cfg=p.PipelineConfig(data_release='qr3')
        flux,error,area,bad,info=p.aperture_flux_one_radius(image,np.ones_like(image),flags,20,20,2,cfg,return_info=True)
        clipped=SigmaClip(sigma=cfg.sigma_clip_sigma,maxiters=cfg.sigma_clip_maxiters)(image[available])
        self.assertEqual(info['background_npix'],clipped.count())
        self.assertLess(info['background_npix'],ann.sum())
        expected_variance=area+area**2*clipped.std()**2/clipped.count()
        self.assertAlmostEqual(error**2,expected_variance,places=8)
        self.assertTrue(np.isfinite(flux))

    def test_source_background_relaxation_is_explicit_and_reported(self):
        image=np.ones((31,31));image[15,15]=100
        flags=np.full(image.shape,1<<21,dtype=np.uint32)
        cfg=p.PipelineConfig(data_release='qr3')
        result=p.aperture_flux_one_radius(image,np.ones_like(image),flags,15,15,2,cfg,return_info=True)
        self.assertTrue(np.isnan(result[0]))
        self.assertEqual(result[-1]['background_status'],'unusable')
        cfg.allow_source_background_fallback=True
        result=p.aperture_flux_one_radius(image,np.ones_like(image),flags,15,15,2,cfg,return_info=True)
        self.assertTrue(np.isfinite(result[0]))
        self.assertTrue(result[-1]['background_source_mask_relaxed'])
        self.assertEqual(result[-1]['background_exclusion_mask'] & (1<<21),0)
        cfg.data_release='qr2';cfg.allow_source_background_fallback=None
        self.assertTrue(p.aperture_flux_one_radius(image,np.ones_like(image),flags,15,15,2,cfg,return_info=True)[-1]['background_source_mask_relaxed'])

    def test_missing_aperture_area_and_invalid_variance_rejected_by_default(self):
        image=np.ones((31,31));image[15,15]=100
        cfg=p.PipelineConfig(data_release='qr3')
        flags=np.zeros(image.shape,dtype=np.uint32);flags[15,15]=1<<20
        result=p.aperture_flux_one_radius(image,np.ones_like(image),flags,15,15,2,cfg,return_info=True)
        self.assertTrue(np.isnan(result[0]))
        self.assertGreater(result[-1]['aperture_masked_fraction'],0)
        result=p.aperture_flux_one_radius(image,None,None,.1,15,2,cfg,return_info=True)
        self.assertLess(result[-1]['aperture_coverage_fraction'],1)
        self.assertTrue(np.isnan(result[0]))
        var=np.ones_like(image);var[15,15]=0
        self.assertTrue(np.isnan(p.aperture_flux_one_radius(image,var,None,15,15,2,cfg)[0]))
        cfg.data_release='qr2'
        self.assertTrue(np.isfinite(p.aperture_flux_one_radius(image,np.ones_like(image),flags,15,15,2,cfg)[0]))

    def test_psf_rejects_crosstalk_but_keeps_source_and_reports_original_bits(self):
        y,x=np.mgrid[-5:6,-5:6];kernel=np.exp(-(x*x+y*y)/2);kernel/=kernel.sum()
        data=1000*kernel+10;data[5,5]+=10000
        flags=np.full(data.shape,1<<21,dtype=np.uint32);flags[5,5]|=1<<20
        cfg=p.PipelineConfig(data_release='qr3')
        flux,error,info=p.psf_flux_fit(data,np.ones_like(data),flags,5,5,kernel,cfg)
        self.assertAlmostEqual(flux,1000,places=7)
        self.assertEqual(info['psf_n_valid'],120)
        self.assertEqual(p.source_warning_flags(flags,5,5),(1<<20)|(1<<21))
        cfg.bad_flag_bits=()
        self.assertGreater(abs(p.psf_flux_fit(data,np.ones_like(data),flags,5,5,kernel,cfg)[0]-1000),100)


class FlagReprojectionTests(unittest.TestCase):
    def wcs(self):
        w=WCS(naxis=2);w.wcs.ctype=['RA---TAN','DEC--TAN']
        w.wcs.crval=[10,20];w.wcs.crpix=[5,5];w.wcs.cdelt=[-1/3600,1/3600]
        return w

    def test_bitwise_or_does_not_manufacture_bits_and_marks_missing_coverage(self):
        w=self.wcs();out=w.deepcopy();out.wcs.crpix[0]+=.5
        flags=np.zeros((9,9),dtype=np.uint32);flags[:,4:]=1<<20;flags[4,3]|=1<<21
        result=stack._reproject_flags(flags,w.to_header(),out,(9,9))
        self.assertEqual(result[4,4],(1<<20)|(1<<21))
        self.assertEqual(result[3,4],1<<20)
        self.assertTrue(np.any(result&(1<<9)))
        self.assertEqual(int(np.bitwise_or.reduce(result.ravel())),(1<<9)|(1<<20)|(1<<21))
        with self.assertRaises(ValueError):stack._as_uint_flags(np.array([.5]))
        self.assertEqual(stack._as_uint_flags(np.array([np.nan]))[0],1<<9)
        self.assertEqual(stack._as_uint_flags(np.array([-2147483648],dtype=np.int32))[0],1<<31)

    def write_scene(self,directory,release,bit=20):
        w=self.wcs();header=w.to_header();header['DETECTOR']=1
        data=np.ones((9,9));data[4,4]=1.e12
        flags=np.zeros(data.shape,dtype=np.uint32);flags[4,4]=1<<bit
        path=Path(directory)/f'{release}.fits'
        fits.HDUList([fits.PrimaryHDU(header=fits.Header({'DATAREL':release})),
            fits.ImageHDU(data,header=header,name='IMAGE'),
            fits.ImageHDU(flags,header=header,name='FLAGS'),
            fits.ImageHDU(np.zeros_like(data),header=header,name='ZODI')]).writeto(path)
        return path,w

    def test_flagged_flux_is_masked_before_interpolation(self):
        with tempfile.TemporaryDirectory() as directory:
            path,w=self.write_scene(directory,'qr3')
            out=w.deepcopy();out.wcs.crpix[0]+=.5
            with patch.object(stack,'find_optimal_celestial_wcs',return_value=(out,(9,9))):
                detector,cutout=stack._extract_detector_cutout(path,SkyCoord(10*u.deg,20*u.deg),9)
            valid=cutout.data[np.isfinite(cutout.data)]
            self.assertGreater(valid.size,0)
            np.testing.assert_allclose(valid,1.)
            self.assertEqual(cutout.header['FLAGPOL'],'qr3_mosaic')
            self.assertTrue(np.any(cutout.flags&(1<<20)))

    def test_qr3_mosaic_policy_differs_from_legacy_and_supports_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path,w=self.write_scene(directory,'qr3',bit=19)
            with fits.open(path) as h:
                bits,policy=stack._stack_mask_policy(h,path,None)
                self.assertNotIn(19,bits);self.assertNotIn(1,bits)
                self.assertIn(20,bits);self.assertIn(25,bits)
                self.assertEqual(stack._stack_mask_policy(h,path,())[0],())
            path,w=self.write_scene(directory,'qr2')
            with fits.open(path) as h:
                self.assertEqual(stack._stack_mask_policy(h,path,None)[0],LEGACY_FATAL_BITS)

    def test_mixed_release_stack_is_rejected_even_with_custom_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            for release in ('qr2','qr3'):
                self.write_scene(directory,release)
            with self.assertRaisesRegex(ValueError,'different release/mask'):
                stack.stack_images(10,20,directory,directory,image_size_arcsec=9,
                                   fatal_flag_bits=(),verbose=False)


if __name__=='__main__':unittest.main()
