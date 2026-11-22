"""Release/provenance regression tests; no archive access required."""
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from astropy.io import fits
from astropy.table import Table
from spexpi import spherex_pipeline as p


class ReleaseTests(unittest.TestCase):
    def test_supported_aliases_and_boundaries(self):
        for release in ('qr1', 'qr2', 'qr2_deep', 'qr3', 'dr1'):
            with self.subTest(release=release):
                self.assertEqual(p.normalize_data_release('SPHEREx_' + release.upper()), release)
                for uri in (f's3://nasa-irsa-spherex/{release}/level2/a.fits',
                            f'https://irsa.ipac.caltech.edu/ibe/data/spherex/{release}/a.fits?x=1',
                            f'{release}/a.fits'):
                    self.assertEqual(p.infer_data_release_from_uri(uri), release)
        for bad in ('qr30', 'my_qr2', 'dr10', 'spherex_dr1_cube', None):
            self.assertEqual(p.normalize_data_release(bad), 'unknown')
        self.assertEqual(p.infer_data_release_from_uri('/data/qr30/a.fits'), 'unknown')
        self.assertEqual(p.normalize_data_release('SPHEREx_DR1_DEEP'), 'dr1')
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
        for inputs in ([], rows('unknown'), rows('qr2', 'unknown'), rows('qr2', 'qr3'), rows('dr1', 'qr3')):
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
        self.assertIn('/dr1/solid_angle_pixel_map/cal-sapm-v3-2026-191/', p.make_sapm_s3_paths(cfg, 'dr1')[1])
        self.assertIn('/dr1/spectral_channels/', p._default_spectral_channels_s3_candidates(cfg, 'dr1')[0])
        self.assertTrue(p._default_spectral_channels_s3_candidates(cfg, 'dr1')[0].endswith('cal-sch-v2-2026-229.fits'))
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

    def test_http_cutout_recenter_refreshes_old_cache_then_reuses_it(self):
        from urllib.parse import parse_qs, urlsplit

        with tempfile.TemporaryDirectory() as directory:
            cfg = p.PipelineConfig(
                data_release='dr1', download_dir=directory, retries=1,
                reference_epoch_yr=1998.87, pmra_masyr=2193, pmdec_masyr=527,
            )
            uri = 'https://irsa.ipac.caltech.edu/ibe/data/spherex/dr1/frame.fits?center=63.831432,-9.585187d&size=0.033333'
            row = dict(uri=uri, cutout_index=1, time_bounds_lower=60922., time_bounds_upper=60922.1)
            cache = p.get_object_download_dir(cfg, 'target')
            path = cache / p.filename_from_uri(uri, 1)
            fits.PrimaryHDU().writeto(path)
            content = io.BytesIO()
            fits.PrimaryHDU().writeto(content)

            with patch.object(p.requests, 'get') as get:
                get.return_value.__enter__.return_value.iter_content.return_value = [content.getvalue()]
                received = p.prepare_cutout_row_file(row, cache, cfg, 63.831432, -9.585187)
                self.assertEqual(received['download_status'], 'downloaded_http')
                sent = get.call_args.args[0]
                query = parse_qs(urlsplit(sent).query)
                self.assertEqual(query['size'], ['0.033333'])
                ra, dec, _ = p.project_to_epoch(63.831432, -9.585187, 1998.87, 2193, 527, 60922.05)
                self.assertEqual(query['center'], [f'{ra:.9f},{dec:.9f}d'])
                self.assertEqual(received['local_uri'], str(path))
                self.assertEqual(fits.getval(path, 'DATAREL'), 'dr1')
                self.assertEqual(p.prepare_cutout_row_file(row, cache, cfg, 63.831432, -9.585187)
                                 ['download_status'], 'cached')
                get.assert_called_once()
            self.assertFalse(cfg.redownload_existing_files)

    def test_combined_discovery_queries_each_release_with_its_own_cache_indices(self):
        cfg = p.PipelineConfig(combine_dr1_qr3=True)

        def query(ra, dec, release_cfg):
            return Table({'uri': [f'{release_cfg.data_release}/level2/a.fits'], 'cutout_index': [1]})

        def download(table, name, release_cfg, **kwargs):
            self.assertEqual(list(table['cutout_index']), [1])
            self.assertFalse(release_cfg.combine_dr1_qr3)
            return [{'data_release': release_cfg.data_release, 'cutout_index': 1}]

        with patch.object(p, 'query_spherex_cutouts', side_effect=query) as queries, \
             patch.object(p, 'download_cutout_files_parallel', side_effect=download):
            rows = p.prepare_combined_cutouts(1., 2., 'target', cfg)
        self.assertEqual([call.args[2].data_release for call in queries.call_args_list], ['dr1', 'qr3'])
        self.assertEqual([(row['data_release'], row['cutout_index']) for row in rows], [('dr1', 1), ('qr3', 2)])

    def test_combined_mode_requires_both_releases_and_rejects_conflicting_selector(self):
        with self.assertRaisesRegex(ValueError, 'requires data_release=None'):
            p.validate_config(p.PipelineConfig(combine_dr1_qr3=True, data_release='dr1'))
        with self.assertRaisesRegex(ValueError, 'both DR1 and QR3'):
            p.require_combined_releases([{'data_release': 'dr1'}])
        with self.assertRaisesRegex(ValueError, 'only cutouts'):
            p.require_combined_releases([{'data_release': r} for r in ('dr1', 'qr3', 'qr2')])

    def test_query_guard_rejects_mixed_results_before_download(self):
        table = Table({'uri': ['qr2/a.fits', 'qr3/b.fits']})
        with patch.object(p, 'query_spherex_cutouts', return_value=table), \
             patch.object(p, 'download_cutout_files_parallel') as downloaded:
            with self.assertRaisesRegex(ValueError, 'Mixed'):
                p.run_spherex_pipeline(1, 2)
            downloaded.assert_not_called()
        with patch.object(p, 'query_spherex_cutouts') as query:
            with self.assertRaises(ValueError):
                p.run_spherex_pipeline(1, 2, cfg=p.PipelineConfig(data_release='qr3', photometry_method='invalid'))
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
            p.query_spherex_cutouts(1., 2., p.PipelineConfig(data_release='dr1'))
            query = service.return_value.search.call_args.args[0]
            self.assertIn("a.uri LIKE '%/spherex/dr1/level2/%'", query)

    def test_pipeline_guards_before_photometry_and_isolates_outputs(self):
        for releases, error in ((['qr2', 'qr3'], ValueError), (['unknown'], ValueError)):
            with self.subTest(releases=releases), \
                 patch.object(p, 'prepare_cached_cutout_rows', return_value=[{'data_release': r} for r in releases]), \
                 patch.object(p, 'open_cutouts_parallel') as opened:
                with self.assertRaises(error):
                    p.run_spherex_pipeline(1, 2, cfg=p.PipelineConfig(query_irsa_for_cutouts=False, photometry_method="psf"))
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
