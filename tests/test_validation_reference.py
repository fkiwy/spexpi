"""Regression tests for the independent archive-validation reference."""
import unittest
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scripts.validate_qr3 import reference_epsf_fit
from spexpi import spherex_pipeline as p
import test_epsf as scenes


class ValidationReferenceTests(unittest.TestCase):
    def setUp(self):
        self.scene=scenes.EPSFTests()
        self.scene.setUp()
        self.addCleanup(self.scene.doCleanups)

    def test_reference_handles_clipped_stamp_and_celestial_sip(self):
        path,_,_,_=self.scene.scene(2.25,3.25)
        with fits.open(path,mode='update') as h:
            header=h['IMAGE'].header
            header['CTYPE1']='RA---TAN-SIP';header['CTYPE2']='DEC--TAN-SIP'
            header['CTYPE1A']='LINEAR';header['CTYPE2A']='LINEAR'
            header['A_ORDER']=2;header['B_ORDER']=2
            header['A_2_0']=.02;header['B_0_2']=.02
            ra,dec=WCS(header).pixel_to_world_values(2.25,3.25)
        cfg=self.scene.cfg
        row=p.process_cutout_row(dict(local_uri=str(path),data_release='qr3',cutout_index=1),ra,dec,cfg)
        maps={}
        fitted=next(m for m in p.measure_cutout_photometry(row,maps,cfg) if m['photometry_method']=='psf')
        reference=reference_epsf_fit(path,row,cfg,*maps[('qr3',1)])
        self.assertEqual((reference['binx'],reference['biny']),(0,0))
        np.testing.assert_allclose(reference['flux_uJy'],fitted['flux_uJy'],rtol=1.e-8)
        np.testing.assert_allclose(reference['error_uJy'],fitted['err_uJy'],rtol=1.e-8)


if __name__=='__main__':
    unittest.main()
