"""Release/provenance regression tests; no archive access required."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from astropy.io import fits
from astropy.table import Table
from spexpi import spherex_pipeline as p


class ReleaseTests(unittest.TestCase):
    def test_supported_aliases_and_boundaries(self):
        for release in ('qr1', 'qr2', 'qr2_deep', 'qr3'):
            with self.subTest(release=release):
                self.assertEqual(p.normalize_data_release('SPHEREx_' + release.upper()), release)
                for uri in (f's3://nasa-irsa-spherex/{release}/level2/a.fits',
                            f'https://irsa.ipac.caltech.edu/ibe/data/spherex/{release}/a.fits?x=1',
                            f'{release}/a.fits'):
                    self.assertEqual(p.infer_data_release_from_uri(uri), release)
        for bad in ('qr30', 'my_qr2', 'dr1', None):
            self.assertEqual(p.normalize_data_release(bad), 'unknown')
        self.assertEqual(p.infer_data_release_from_uri('/data/qr30/a.fits'), 'unknown')
        self.assertEqual(p.s3_key_from_irsa_uri('qr3/level2/a.fits'), 'qr3/level2/a.fits')

    def test_fits_unknown_and_conflicting_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d)/'unknown.fits'
            fits.PrimaryHDU().writeto(f)
            self.assertEqual(p.infer_data_release_from_fits(f), 'unknown')
            with fits.open(f, mode='update') as h:
                h[0].header['DATAREL'] = 'qr3'
            self.assertEqual(p.infer_data_release_from_fits(f), 'qr3')
            with self.assertRaisesRegex(ValueError, 'Refusing'):
                p.stamp_data_release_in_fits(f, 'qr2')
            self.assertEqual(p.infer_data_release_from_fits(f), 'qr3')
            conflict = Path(d)/'qr2'/'conflict.fits'
            conflict.parent.mkdir()
            fits.PrimaryHDU(header=fits.Header({'DATAREL': 'qr3'})).writeto(conflict)
            with self.assertRaisesRegex(ValueError, 'Conflicting'):
                p.infer_data_release_from_fits(conflict)
        with self.assertRaises(FileNotFoundError):
            p.infer_data_release_from_fits('/nonexistent/file.fits')

    def test_single_release_required(self):
        rows = lambda *rs: [{'data_release': r} for r in rs]
        self.assertEqual(p.require_single_data_release(rows('qr3', 'qr3')), 'qr3')
        for inputs in ([], rows('unknown'), rows('qr2', 'unknown'), rows('qr2', 'qr3')):
            with self.assertRaises(ValueError):
                p.require_single_data_release(inputs)
        with self.assertRaises(ValueError):
            p.require_single_data_release(rows('qr3'), 'qr2')

    def test_calibration_defaults_and_overrides(self):
        cfg = p.PipelineConfig()
        self.assertIn('/qr2/solid_angle_pixel_map/cal-sapm-v2-2025-164/', p.make_sapm_s3_paths(cfg, 'qr2')[1])
        self.assertIn('/qr3/solid_angle_pixel_map/cal-sapm-v3-2026-191/', p.make_sapm_s3_paths(cfg, 'qr3')[1])
        self.assertTrue(p._default_spectral_channels_s3_candidates(cfg, 'qr2')[0].endswith('cal-sch-v1-2026-106.fits.gz'))
        self.assertTrue(p._default_spectral_channels_s3_candidates(cfg, 'qr3')[0].endswith('cal-sch-v2-2026-229.fits'))
        cfg.sapm_collection = 'custom'
        self.assertIn('/custom/', p.make_sapm_s3_paths(cfg, 'qr3')[1])
        with self.assertRaises(ValueError):
            p.make_sapm_s3_paths(cfg, 'unknown')
        with self.assertRaises(ValueError):
            p.validate_config(p.PipelineConfig(data_release='qr30'))

    def test_separate_calibration_and_cutout_caches_with_legacy_reads(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = p.PipelineConfig(download_dir=d, sapm_cache_dir=d, spectral_channels_cache_dir=d,
                                   show_download_progress=False)
            a = p.get_sapm_cache_dir(cfg, 'qr2')
            b = p.get_sapm_cache_dir(cfg, 'qr3')
            self.assertNotEqual(a, b)
            self.assertNotEqual(p.get_spectral_channels_cache_dir(cfg, 'qr2'),
                                p.get_spectral_channels_cache_dir(cfg, 'qr3'))
            legacy = Path(d)/'target';legacy.mkdir()
            fits.PrimaryHDU(header=fits.Header({'DATAREL': 'qr2'})).writeto(legacy/'old.fits')
            modern = Path(d)/'qr3'/'target';modern.mkdir(parents=True)
            fits.PrimaryHDU(header=fits.Header({'DATAREL': 'qr3'})).writeto(modern/'new.fits')
            rows = p.prepare_cached_cutout_rows('target', cfg)
            with self.assertRaisesRegex(ValueError, 'Mixed'):
                p.require_single_data_release(rows)
            cfg.data_release = 'qr2'
            self.assertEqual(len(p.prepare_cached_cutout_rows('target', cfg)), 1)
            cfg.data_release = 'qr3'
            self.assertEqual(p.prepare_cached_cutout_rows('target', cfg)[0]['data_release'], 'qr3')

    def test_downloads_do_not_collide_across_releases(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = p.PipelineConfig(download_dir=d, retries=1)
            paths = []
            def download(uri, path, cfg, **kwargs):
                fits.PrimaryHDU().writeto(path, overwrite=True)
                paths.append(path)
            with patch.object(p, 'download_uri_to_file', side_effect=download):
                for release in ('qr2', 'qr3'):
                    row = p.prepare_cutout_row_file(
                        {'uri': f'https://irsa.ipac.caltech.edu/ibe/data/spherex/{release}/same.fits',
                         'cutout_index': 1}, Path(d)/'target', cfg)
                    self.assertEqual(row['data_release'], release)
                    self.assertEqual(Path(row['local_uri']).parent, Path(d)/release/'target')
            self.assertNotEqual(paths[0], paths[1])

    def test_query_guard_rejects_mixed_results_before_download(self):
        table = Table({'uri': ['qr2/a.fits', 'qr3/b.fits']})
        with patch.object(p, 'query_spherex_cutouts', return_value=table), \
             patch.object(p, 'download_cutout_files_parallel') as downloaded:
            with self.assertRaisesRegex(ValueError, 'Mixed'):
                p.run_spherex_pipeline(1, 2)
            downloaded.assert_not_called()
        with patch.object(p, 'query_spherex_cutouts') as query:
            with self.assertRaises(NotImplementedError):
                p.run_spherex_pipeline(1, 2, cfg=p.PipelineConfig(data_release='qr3'))
            query.assert_not_called()

    def test_explicit_calibration_release_beats_previous_runtime_state(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = p.PipelineConfig(spectral_channels_cache_dir=d)
            p.set_runtime_data_release(cfg, 'qr2')
            cache = p.get_spectral_channels_cache_dir(cfg, 'qr3')
            fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(Table({'WAVELENGTH': [1.0]}))]).writeto(cache/'channels.fits')
            table = p.load_spectral_channels_calibration(cfg, 'qr3')
            self.assertEqual(list(table['WAVELENGTH']), [1.0])

    def test_query_filters_selected_release(self):
        with patch.object(p.pyvo.dal, 'TAPService') as service:
            service.return_value.search.return_value.to_table.return_value = Table({'uri': []})
            p.query_spherex_cutouts(1., 2., p.PipelineConfig(data_release='qr3'))
            query = service.return_value.search.call_args.args[0]
            self.assertIn("a.uri LIKE '%/spherex/qr3/%'", query)

    def test_pipeline_guards_before_photometry_and_isolates_outputs(self):
        for releases, error in ((['qr2', 'qr3'], ValueError), (['unknown'], ValueError),
                                (['qr3'], NotImplementedError)):
            with self.subTest(releases=releases), \
                 patch.object(p, 'prepare_cached_cutout_rows', return_value=[{'data_release': r} for r in releases]), \
                 patch.object(p, 'open_cutouts_parallel') as opened:
                with self.assertRaises(error):
                    p.run_spherex_pipeline(1, 2, cfg=p.PipelineConfig(query_irsa_for_cutouts=False))
                opened.assert_not_called()
        cfg = p.PipelineConfig(query_irsa_for_cutouts=False, output_dir='/unused', data_release='qr2')
        with patch.object(p, 'prepare_cached_cutout_rows', return_value=[{'data_release': 'qr2'}]), \
             patch.object(p, 'ensure_output_directories') as mkdir, \
             patch.object(p, 'open_cutouts_parallel', side_effect=RuntimeError('stop before science')):
            with self.assertRaisesRegex(RuntimeError, 'stop'):
                p.run_spherex_pipeline(1, 2, cfg=cfg)
            mkdir.assert_called_once_with('/unused/qr2')
        self.assertEqual(cfg.output_dir, '/unused')
        self.assertFalse(hasattr(cfg, '_active_data_release'))


if __name__ == '__main__':
    unittest.main()
