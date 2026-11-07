"""ePSF regression tests with independent analytic source injections."""
import unittest
from unittest.mock import patch
import numpy as np
import astropy.units as u
from astropy.io import fits
from spexpi import spherex_pipeline as p
import test_qr3_aperture as fixtures


def model_table(detector=1):
    # Deliberately rectangular and reverse row order to expose square/ordering assumptions.
    by,bx=np.mgrid[:2,:3];bx=bx.ravel()[::-1];by=by.ravel()[::-1]
    yy,xx=np.mgrid[-16:17,-16:17]/5.
    arrays=[]
    for i,j in zip(bx,by):
        sigma=.65+.1*i+.05*j
        arr=np.exp(-((xx-.17)**2+(yy+.11)**2)/(2*sigma**2))
        arrays.append(arr/arr.sum())
    cols=[fits.Column(name=n,format='J',array=a) for n,a in [('BINX',bx),('BINY',by)]]
    cols += [fits.Column(name=n,format='E',array=a) for n,a in [
        ('XCENTER',bx*30+14.5),('YCENTER',by*40+19.5),('XWIDTH',np.full(6,30)),('YWIDTH',np.full(6,40))]]
    cols.append(fits.Column(name='EPSF',format='1089D',dim='(33,33)',array=arrays))
    return fits.BinTableHDU.from_columns(cols,name='EPSF',header=fits.Header({
        'DETECTOR':detector,'DETCOORD':'sky','OVSMPX':5,'OVSMPY':5}))


class EPSFTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.QR3ApertureTests()
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.cfg=self.fixture.cfg
        self.cfg.photometry_method='both'
        self.cfg.enable_psf_local_grid_search=False
        self.cfg.psf_kernel_radius_native=5
        self.cfg.psf_fit_background_gradient=True

    def scene(self,x=15.25,y=14.75,detector=1):
        path,header=self.fixture.make_cutout(detector)
        hdul=fits.open(path,memmap=False)
        image=hdul['IMAGE'];image.header.add_history('[CALIB] epsf test-calibration-id')
        wcs=p.WCS(image.header)
        ra,dec=wcs.pixel_to_world_values(x,y)
        xp,yp=x+20,y+30
        bx=int((xp+.5)//30);by=int((yp+.5)//40)
        sigma=.65+.1*bx+.05*by
        yy,xx=np.indices((31,31))
        source=np.exp(-((xx-x-.17)**2+(yy-y+.11)**2)/(2*sigma**2))
        yf,xf=np.mgrid[-20:51,-20:51]
        full=np.exp(-((xf-x-.17)**2+(yf-y+.11)**2)/(2*sigma**2))
        source*=1000/full.sum()
        scale=(u.MJy/u.sr*self.fixture.area[30:61,20:51]*u.arcsec**2).to_value(u.uJy)
        image.data=(source+100.+.03*xx-.05*yy)/scale+2.
        hdul.append(model_table(detector))
        hdul.writeto(path,overwrite=True);hdul.close()
        return path,float(ra),float(dec),source

    def test_analytic_flux_recovery_across_phases_and_mwir(self):
        for detector in (1,4,5,6):
            for phase in (.05,.49,.51,.95):
                with self.subTest(detector=detector,phase=phase):
                    path,ra,dec,_=self.scene(15+phase,15-phase,detector)
                    row=p.process_cutout_row({'local_uri':str(path),'data_release':'qr3','cutout_index':1},ra,dec,self.cfg)
                    measurements=p.measure_cutout_photometry(row,{},self.cfg)
                    result=next(r for r in measurements if r['photometry_method']=='psf')
                    self.assertEqual(result['psf_status'],'ok')
                    self.assertAlmostEqual(result['flux_uJy'],1000,delta=2.)
                    self.assertGreater(result['err_uJy'],0)
                    self.assertEqual((result['epsf_binx'],result['epsf_biny']),(1,1))

    def test_rectangular_zone_selection_uses_parent_coordinates(self):
        path,_,_,_=self.scene()
        with fits.open(path) as hdul:
            for x,y,expected in [(0,0,(0,0)),(15,15,(1,1)),(30,30,(1,1))]:
                index,_=p.select_epsf_zone(hdul,x,y)
                row=hdul['EPSF'].data[index]
                self.assertEqual((row['BINX'],row['BINY']),expected)

    def test_masking_and_small_fit_window_do_not_renormalize_flux(self):
        path,ra,dec,_=self.scene()
        self.cfg.psf_kernel_radius_native=2
        self.cfg.psf_min_valid_pixels=15
        with fits.open(path,mode='update') as h:
            h['FLAGS'].data[15,15]=1
        row=p.process_cutout_row({'local_uri':str(path),'data_release':'qr3','cutout_index':1},ra,dec,self.cfg)
        result=next(m for m in p.measure_cutout_photometry(row,{},self.cfg) if m['photometry_method']=='psf')
        self.assertAlmostEqual(result['flux_uJy'],1000,delta=3.)
        self.assertLess(result['psf_kernel_sum'],1.)
        self.assertEqual(result['psf_n_valid'],24)

    def test_image_edge_clipping_does_not_rescale_observed_flux(self):
        path,ra,dec,_=self.scene(.25,1.3)
        row=p.process_cutout_row({'local_uri':str(path),'data_release':'qr3','cutout_index':1},ra,dec,self.cfg)
        result=next(m for m in p.measure_cutout_photometry(row,{},self.cfg) if m['photometry_method']=='psf')
        self.assertEqual(result['psf_status'],'ok')
        self.assertAlmostEqual(result['flux_uJy'],1000,delta=3.)

    def test_parent_detector_clipping_keeps_epsf_and_source_aligned(self):
        path,ra,dec,source=self.scene(5.25,14.75)
        with fits.open(path,mode='update') as h:
            h['IMAGE'].header['CRPIX1A']=5.
            # Valid detector pixels start at cutout column 4 (parent column 0).
            scale=(u.MJy/u.sr*self.fixture.area[30:61,:27]*u.arcsec**2).to_value(u.uJy)
            yy,xx=np.indices((31,31))
            h['IMAGE'].data[:,4:]=(source[:,4:]+100.+.03*xx[:,4:]-.05*yy[:,4:])/scale+2.
            h['VARIANCE'].data[:,4:]=1./scale**2
        row=p.process_cutout_row({'local_uri':str(path),'data_release':'qr3','cutout_index':1},ra,dec,self.cfg)
        result=next(m for m in p.measure_cutout_photometry(row,{},self.cfg) if m['photometry_method']=='psf')
        self.assertAlmostEqual(result['psf_x_fit'],1.25,places=6)
        self.assertEqual((result['epsf_binx'],result['epsf_biny']),(0,1))
        self.assertAlmostEqual(result['flux_uJy'],1000,delta=3.)

    def test_grid_search_recovers_shifted_centroid(self):
        path,ra,dec,_=self.scene(15.25,14.75)
        self.cfg.enable_psf_local_grid_search=True
        self.cfg.psf_local_grid_half_range_pix=.5
        self.cfg.psf_local_grid_step_pix=.25
        self.cfg.psf_local_grid_metric='chi2'
        row=p.process_cutout_row({'local_uri':str(path),'data_release':'qr3','cutout_index':1},ra,dec,self.cfg)
        row['x_src']+=.25;row['y_src']-=.25
        result=next(m for m in p.measure_cutout_photometry(row,{},self.cfg) if m['photometry_method']=='psf')
        self.assertAlmostEqual(result['psf_x_fit'],15.25,places=6)
        self.assertAlmostEqual(result['psf_y_fit'],14.75,places=6)
        self.assertAlmostEqual(result['flux_uJy'],1000,delta=2.)

    def test_both_writes_independent_spectra_and_model_provenance(self):
        _,ra,dec,_=self.scene()
        spectra,measurements=p.run_spherex_pipeline(ra,dec,'synthetic',self.cfg)
        self.assertEqual(set(spectra),{'aperture','psf'})
        self.assertEqual(len(spectra['psf']),1)
        self.assertEqual({m['photometry_method'] for m in measurements},{'psf','aperture'})
        with fits.open(self.fixture.root/'out/qr3/fits/synthetic_psf_spectrum.fits') as h:
            self.assertEqual(h[0].header['PSFMODEL'],'EPSF')
            self.assertEqual(h[0].header['ERRTYPE'],'FORMAL')
            self.assertEqual(h[0].header['EPSFREF'],'test-calibration-id')
            self.assertEqual(h[0].header['DATAREL'],'qr3')

    def test_fit_quality_and_mask_rejections_are_reported(self):
        path,ra,dec,_=self.scene()
        row=p.process_cutout_row({'local_uri':str(path),'data_release':'qr3','cutout_index':1},ra,dec,self.cfg)
        row['flags_hdu'].data[:]=1
        result=next(m for m in p.measure_cutout_photometry(row,{},self.cfg) if m['photometry_method']=='psf')
        self.assertEqual(result['psf_status'],'too_few_valid_pixels')
        self.assertEqual(result['status'],'bad_photometry')
        row['flags_hdu'].data[:]=0
        row['image_hdu'].data[15,15]+=10.
        self.cfg.psf_max_reduced_chi2=1.
        result=next(m for m in p.measure_cutout_photometry(row,{},self.cfg) if m['photometry_method']=='psf')
        self.assertEqual(result['psf_status'],'rejected_high_reduced_chi2')
        self.assertTrue(np.isnan(result['flux']))

    def test_formal_errors_match_injected_noise_scatter(self):
        path,_,_,source=self.scene(15.25,14.75)
        with fits.open(path) as h:
            kernel=p.extract_spherex_epsf_kernel(h,15.25,14.75,self.cfg)
        rng=np.random.default_rng(8531)
        variance=np.full(source.shape,4.)
        self.cfg.psf_fit_background_gradient=False
        fluxes=[];errors=[]
        for i in range(128):
            flux,error,info=p.psf_flux_fit(source+100+rng.normal(0,2,source.shape),
                variance,None,15.25,14.75,kernel,self.cfg,normalize_psf=False)
            self.assertEqual(info['psf_status'],'ok')
            fluxes.append(flux);errors.append(error)
        self.assertAlmostEqual(np.mean(fluxes),1000,delta=2.)
        self.assertAlmostEqual(np.std(fluxes,ddof=1)/np.mean(errors),1.,delta=.25)

    def test_qr2_optical_psf_keeps_legacy_normalization(self):
        y,x=np.mgrid[-5:6,-5:6]
        kernel=np.exp(-(x*x+y*y)/2)
        image=50+1000*kernel/kernel.sum()
        self.cfg.psf_fit_background_gradient=False
        flux,error,info=p.psf_flux_fit(image,np.ones_like(image),None,5.,5.,kernel,self.cfg)
        self.assertAlmostEqual(flux,1000,places=7)
        self.assertGreater(error,0)
        self.assertEqual(info['psf_status'],'ok')

    def test_malformed_epsf_fails_without_optical_psf_fallback(self):
        path,_,_,_=self.scene()
        for key,value in [('OVSMPX',0),('DETCOORD','detector'),('DETECTOR',2)]:
            with self.subTest(key=key),fits.open(path) as h:
                h['EPSF'].header[key]=value
                with self.assertRaises(ValueError):
                    p.extract_spherex_psf_kernel(h,15,15,self.cfg)
        with fits.open(path) as h:
            h['EPSF'].data['EPSF'][0]=np.nan
            with self.assertRaises(ValueError):
                # Select row zero: high x and y zone.
                p.extract_spherex_psf_kernel(h,50,15,self.cfg)


if __name__=='__main__':
    unittest.main()
