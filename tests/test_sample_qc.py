"""Sample QC checks A-D and the summary on synthetic results (small CSVs, fake TIFFs)."""
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.qc import distribution, review_images as rv, seam_consistency, snr_qc, summary, unaligned

BOX = ['x1', 'y1', 'x2', 'y2', 'z']


def _config(root, n_slices=6, blank=(1,)):
    """Two channels (RFP ref, Sox9), one tile 100_200 of 64 px; slices in `blank` are blank."""
    paths = {}
    for ch, key in (('RFP', 'rfp_dir'), ('Sox9', 'sox9_dir')):
        import tifffile
        folder = Path(root) / 'img' / ch / '100' / '100_200'
        folder.mkdir(parents=True)
        for z in range(1, n_slices + 1):
            if z in blank:
                (folder / f'100_200_{z:03d}.tif').write_bytes(b'\0' * 100)
            else:
                tifffile.imwrite(folder / f'100_200_{z:03d}.tif',
                                 np.random.default_rng(z).integers(0, 4000, (64, 64), dtype=np.uint16))
        paths[key] = str(Path(root) / 'img' / ch)
    return {'pipeline_mode': 'pre_align', 'paths': paths,
            'channels_routing': [{'id': 'RFP', 'type': 'soma', 'dir_key': 'rfp_dir'},
                                 {'id': 'Sox9', 'type': 'tf', 'dir_key': 'sox9_dir'}],
            'detection_params': {'tILESIZE': 64},
            'contrast_filter': {'Sox9': {'snr_min': 5, 'noise': 'slice', 'max_removed_frac': 0.4,
                                         'snr_step': 1, 'snr_floor': 3}},
            'qc': {'review': {'min_image_bytes': 1000, 'crop_px': 32, 'max_items_per_check': 1,
                              'layers_per_item': 2}}}


class UnalignedTests(unittest.TestCase):
    def test_classify_and_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            res = Path(tmp) / 'res'
            (res / '3_2d_aligned').mkdir(parents=True)
            (res / '2_2d_filtered').mkdir(parents=True)
            for tile, score, n in (('100_200', 0.0, 300), ('100_300', 0.0, 10), ('100_400', 0.3, 300)):
                (res / '3_2d_aligned' / f'{tile}_measured_offsets.json').write_text(json.dumps(
                    {'reference_channel': 'RFP', 'z_center': 3, 'scores': {'RFP': 1.0, 'Sox9': score}}))
                (res / '3_2d_aligned' / f'{tile}_offsets.json').write_text(json.dumps(
                    {'RFP': {'dx': 0, 'dy': 0, 'dz': 0}, 'Sox9': {'dx': 1, 'dy': 0, 'dz': 0}}))
                rows = pd.DataFrame({'x1': 10, 'y1': 10, 'x2': 20, 'y2': 20, 'z': np.arange(n) % 5 + 2})
                rows.to_csv(res / '2_2d_filtered' / f'{tile}_RFP_result.csv', index=False)
                rows.to_csv(res / '3_2d_aligned' / f'{tile}_RFP_result.csv', index=False)
            table = unaligned.classify(str(res), unaligned.DEFAULTS)
            self.assertEqual(dict(zip(table.tile, table.status)),
                             {'100_200': 'unaligned', '100_300': 'empty', '100_400': 'ok'})
            out = unaligned.run(config, str(res))
            self.assertEqual(out['status'], 'WARN')
            self.assertEqual(out['unaligned_tiles'], ['100_200'])
            self.assertEqual(len(out['images']), 2)         # max 1 item x 2 real layers
            for png in out['images']:
                self.assertNotIn('_z0001', png)             # slice 1 is blank
            config['qc']['alignment'] = {'fail_on_unaligned': True}
            with self.assertRaises(RuntimeError):
                unaligned.run(config, str(res))


class SeamTests(unittest.TestCase):
    def test_consistent_vs_single(self):
        seams = pd.DataFrame([
            dict(tile='a', nb_tile='b', direction='EAST', channel=c, status='ok', n_match=500,
                 meas_dx=m[0], meas_dy=m[1], meas_dz=0) for c, m in (('RFP', (60, 0)), ('Sox9', (65, 3)))] + [
            dict(tile='a', nb_tile='c', direction='SOUTH', channel=c, status='ok', n_match=n,
                 meas_dx=m, meas_dy=0, meas_dz=0) for c, m, n in (('RFP', 30, 50), ('Sox9', 0, 500))])
        pos = pd.DataFrame({'tile': ['a', 'b', 'c'],
                            **{f'P_{c}_{a}': [0.0, 100.0 if a == 'x' else 0.0, 0.0]
                               for c in ('RFP', 'Sox9') for a in 'xyz'}})
        res = seam_consistency.residuals(seams, pos)
        np.testing.assert_allclose(res.rx.to_numpy()[:2], [40, 35])
        table = seam_consistency.classify(res, seam_consistency.DEFAULTS)
        self.assertEqual(dict(zip(table.nb_tile, table.status)), {'b': 'consistent', 'c': 'single_outlier'})


class SnrTests(unittest.TestCase):
    def test_saturated_slices(self):
        with tempfile.TemporaryDirectory() as tmp:
            filt = Path(tmp)
            kept = pd.DataFrame({'x1': 1, 'y1': 1, 'x2': 5, 'y2': 5, 'z': [1] * 4 + [2] * 9,
                                 'source_image': ['s1'] * 4 + ['s2'] * 9, 'snr': 9.0,
                                 'snr_threshold': [3.0] * 4 + [5.0] * 9})
            rej = pd.DataFrame({'x1': 1, 'y1': 1, 'x2': 5, 'y2': 5, 'z': [1] * 6 + [2] * 1,
                                'source_image': ['s1'] * 6 + ['s2'], 'snr': 1.0, 'snr_threshold': np.nan,
                                'rejection_reason': 'snr_min'})
            rej = pd.concat([rej, rej.assign(rejection_reason='bbox')])   # bbox rejects do not count
            kept.to_csv(filt / 't_Sox9_result.csv', index=False)
            rej.to_csv(filt / 't_Sox9_rejected.csv', index=False)
            params = {'snr_min': 5, 'max_removed_frac': 0.4, 'snr_step': 1, 'snr_floor': 3}
            t = snr_qc.slice_table(str(filt), 'Sox9', params).set_index('source_image')
            self.assertEqual(t.loc['s1', 'n_in'], 10)
            self.assertTrue(t.loc['s1', 'saturated'])
            self.assertTrue(t.loc['s1', 'lowered'])
            self.assertFalse(t.loc['s2', 'saturated'])


class DistributionTests(unittest.TestCase):
    def test_bad_layer_and_outlier(self):
        layers = pd.DataFrame({'channel': 'RFP', 'tile': 't', 'z': range(1, 41),
                               'count': [100] * 20 + [10] + [100] * 19})
        bad, tissue = distribution.bad_layers(layers, distribution.DEFAULTS)
        self.assertEqual(bad.z.tolist(), [21])
        with tempfile.TemporaryDirectory() as tmp:
            rows = []
            for k in range(6):
                share = 0.9 if k == 5 else 0.4 + 0.01 * k
                n = 600
                rows.append(pd.DataFrame({'tile_name': f't{k}', 'cz': np.arange(n) % 50,
                                          'soma_positive_count': 1,
                                          'tf_positive_count': (np.arange(n) < share * n).astype(int)}))
            pd.concat(rows).to_csv(Path(tmp) / 'c.csv', index=False)
            tiles, by_z, tf = distribution.tf_share(str(Path(tmp) / 'c.csv'), distribution.DEFAULTS)
            self.assertEqual(tiles[tiles.outlier].tile_name.tolist(), ['t5'])


class SummaryTests(unittest.TestCase):
    def test_summary_statuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            qc = Path(tmp) / 'qc'
            (qc / 'snr').mkdir(parents=True)
            (qc / 'snr' / 'summary.json').write_text(json.dumps({'status': 'WARN', 'channels': []}))
            (qc / 'distribution_error.json').write_text(json.dumps({'error': 'boom'}))
            s = summary.run({}, tmp)
            self.assertEqual(s['status'], 'FAIL')
            self.assertEqual(s['checks']['snr']['status'], 'WARN')
            self.assertEqual(s['checks']['seam_consistency']['status'], 'NOT_RUN')
            self.assertTrue((qc / 'index.html').is_file())


class ReviewTests(unittest.TestCase):
    def test_pick_layers_and_blank(self):
        self.assertEqual(rv.pick_layers(range(1, 11), 3), [3, 5, 8])
        self.assertEqual(rv.pick_layers(range(1, 11), 3, ok=lambda z: z < 3), [1, 2])
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp, blank=(2,))
            images = rv.TileImages(config, str(Path(tmp) / 'res'), 1000)
            self.assertFalse(images.is_real('Sox9', '100_200', 2))
            self.assertTrue(images.all_real('100_200', 3))
            self.assertTrue(np.isnan(images.read('RFP', '100_200', 2, (0, 32, 0, 32))).all())
            crop = images.read('RFP', '100_200', 3, (0, 32, 0, 32), shift=(5, 0, 0))
            self.assertTrue(np.isnan(crop[:, :5]).all() and np.isfinite(crop[:, 5:]).all())


if __name__ == '__main__':
    unittest.main()
