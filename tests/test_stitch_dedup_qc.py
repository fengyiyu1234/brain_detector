"""Seam deduplication QC: residual counting, random review images, and skipping existing outputs."""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.qc.stitch_dedup import DEFAULTS, one_to_one_pairs, run, sample_seam_segments


class PairsTest(unittest.TestCase):
    def test_one_to_one_nearest_first_and_z_overlap(self):
        a = np.array([[0, 0, 1, 3], [10, 0, 1, 3]], float)
        b = np.array([[1, 0, 2, 4], [2, 0, 1, 2], [10, 0, 5, 6]], float)
        # a0 takes b0 (nearest); b1 is left over; a1-b2 do not overlap in z
        self.assertEqual(one_to_one_pairs(a, b, 3.0), 1)
        self.assertEqual(one_to_one_pairs(a, b, 3.0, check_z=False), 2)


class StitchDedupQcTest(unittest.TestCase):
    """Two 200 px tiles side by side, B at x=160 (40 px XML overlap)."""

    def _write_sample(self, root, n_dup):
        import tifffile
        d = {k: os.path.join(root, v) for k, v in (('pATH_GLOBAL_2D', '5_2d_global'),
                                                      ('pATH_CHANNEL_3D', '6_3d_global'),
                                                      ('pATH_ALIGN_OFFSETS', '3_2d_aligned'))}
        for path in d.values():
            os.makedirs(path, exist_ok=True)
        os.makedirs(os.path.join(d['pATH_GLOBAL_2D'], 'seam_offsets'), exist_ok=True)
        images = os.path.join(root, 'img')
        for tile in ('r/a', 'r/b'):
            os.makedirs(os.path.join(images, tile), exist_ok=True)
            for z in range(6):
                tifffile.imwrite(os.path.join(images, tile, f'{os.path.basename(tile)}_{z:03d}.tif'),
                                 np.full((200, 200), 100 + z, np.uint16))
        geom = {'dir_dict': {'r/a': (0, 0), 'r/b': (0, 1)}, 'tile_size': 200, 'z_start': 0, 'Z': 6,
                'disp_mat_fin': np.array([[[0, 0, 0], [160, 0, 0]]], dtype=object), 'num_tiles': 2,
                'seam_offsets_csv': os.path.join(d['pATH_GLOBAL_2D'], 'seam_offsets', 'seam_offsets.csv'),
                'seam_seg_layers': 50}
        # 300 cells in the overlap from A; n_dup of them also kept from B at the same place.
        rng = np.random.default_rng(0)
        cx, cy = rng.uniform(165, 195, 300), np.linspace(5, 195, 300)
        tracks, members, boxes = [], [], []
        for k in range(300):
            for tile in (['a', 'b'] if k < n_dup else ['a']):
                tid = f't{k}{tile}'
                tracks.append(dict(x1=cx[k] - 3, y1=cy[k] - 3, x2=cx[k] + 3, y2=cy[k] + 3, z_min=2, z_max=3,
                                   track_id=tid))
                for z in (2, 3):
                    members.append(dict(track_id=tid, tile_name=tile))
                    boxes.append(dict(x1=cx[k] - 3, y1=cy[k] - 3, x2=cx[k] + 3, y2=cy[k] + 3, score=1, mean=1,
                                      **{'class': 'nucleus'}, z=z + 1, detection_id=f'{tid}{z}',
                                      tile_name=tile, slice_name=''))
        pd.DataFrame(tracks).to_csv(os.path.join(d['pATH_CHANNEL_3D'], 'Olig2_3d_tracked.csv'), index=False)
        pd.DataFrame(members).to_csv(os.path.join(d['pATH_CHANNEL_3D'], 'Olig2_track_members.csv'), index=False)
        pd.DataFrame(boxes).to_csv(os.path.join(d['pATH_GLOBAL_2D'], 'Olig2_2d_global.csv'), index=False)
        pd.DataFrame(columns=['detection_id', 'tile_name', 'slice_name', 'z', 'decision', 'reason',
                              'other_detection_id', 'iou']).to_csv(
            os.path.join(d['pATH_GLOBAL_2D'], 'Olig2_stitch_decisions.csv'), index=False)
        pd.DataFrame([dict(channel='Olig2', tile='a', nb_tile='b', direction='EAST', seg=0, n_a=600, n_b=600,
                           n_dup=600, snr=30, iou=0.5, outlier=False, nom_dx=0, nom_dy=0, nom_dz=0,
                           meas_dx=0, meas_dy=0, meas_dz=0, dx=0.0, dy=0.0, dz=0, source='own')]).to_csv(
            geom['seam_offsets_csv'], index=False)
        config = {'pipeline_mode': 'pre_align', 'paths': {'olig2_dir': images},
                  'qc': {'stitch_dedup': {'workers': 1}}}
        routing = [{'id': 'Olig2', 'type': 'tf', 'dir_key': 'olig2_dir'}]
        return config, geom, routing, d

    def test_counts_residual_and_skips_existing_outputs(self):
        with tempfile.TemporaryDirectory() as root:
            config, geom, routing, derived = self._write_sample(root, n_dup=60)
            report = run(config, root, geom, routing, {'Olig2': root}, derived)
            self.assertAlmostEqual(report['summary'].iloc[0].residual_3d_pct, 20.0, delta=1.0)  # 60 of 300
            self.assertEqual(len(report['images_written']), 1)     # only one measured segment exists
            png = report['images_written'][0]
            mtime = os.stat(png).st_mtime_ns
            again = run(config, root, geom, routing, {'Olig2': root}, derived)
            self.assertEqual(again['images_written'], [])          # existing image kept
            self.assertEqual(os.stat(png).st_mtime_ns, mtime)

    def test_clean_seam_has_no_residual(self):
        with tempfile.TemporaryDirectory() as root:
            config, geom, routing, derived = self._write_sample(root, n_dup=0)
            report = run(config, root, geom, routing, {'Olig2': root}, derived)
            self.assertEqual(report['summary'].iloc[0].residual_3d, 0)


class SampleSeamSegmentsTest(unittest.TestCase):
    @staticmethod
    def _offsets(n_seams, n_segs):
        return pd.DataFrame([dict(channel='Sox9', tile=f'a{k}', nb_tile=f'b{k}', seg=s, source='own', n_dup=50)
                             for k in range(n_seams) for s in range(n_segs)] +
                            [dict(channel='Sox9', tile='a0', nb_tile='b0', seg=99, source='frame', n_dup=0)])

    def test_distinct_seams_first_and_deterministic(self):
        offsets = self._offsets(20, 5)
        picks = sample_seam_segments('Sox9', offsets, DEFAULTS)
        self.assertEqual(len(picks), 10)
        self.assertEqual(len({(a, b) for a, b, _ in picks}), 10)
        again = sample_seam_segments('Sox9', offsets, DEFAULTS)
        self.assertEqual([(a, b, int(r.seg)) for a, b, r in picks], [(a, b, int(r.seg)) for a, b, r in again])

    def test_fills_with_more_segments_when_few_seams(self):
        picks = sample_seam_segments('Sox9', self._offsets(3, 5), DEFAULTS)
        keys = [(a, b, int(r.seg)) for a, b, r in picks]
        self.assertEqual(len(keys), 10)
        self.assertEqual(len(set(keys)), 10)
        self.assertNotIn(99, [k[2] for k in keys])                 # fallback segments are not drawn
        self.assertEqual(len(sample_seam_segments('Sox9', self._offsets(1, 4), DEFAULTS)), 4)


if __name__ == '__main__':
    unittest.main()
