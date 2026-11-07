import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.core.patch_fragments import (
    candidate_paths, fragment_keep, patch_cut, resolve_patch_fragment, signature,
    suppress_patch_fragments)
from src.core.resume import raw_detection_settings

GEOMETRY = {'iomin': 0.5, 'edge_tol': 2.0, 'xsize': 512, 'ysize': 512, 'step': 384,
            'tile_size': 2048}


def _config(value=0.5, model='yolo'):
    return {'detection_params': {'xsize': 512, 'ysize': 512, 'step': 384, 'tILESIZE': 2048,
                                 model: {'patch_fragment_iomin': value}},
            'channels_routing': [{'id': 'RFP', 'model': model}]}


# ET10 316000_365900 RFP, raw z=423: two kept boxes of one cell, each clipped by a patch edge.
ET10_ROWS = [
    # detection_id, patch_x, patch_y, x1, y1, x2, y2
    ('raw2d:bec6', 1920, 1152, 1920.0, 1253.29, 1967.90, 1312.69),   # clipped at x=1920
    ('raw2d:e7d5', 1536, 768, 1909.86, 1254.10, 1963.91, 1280.0),    # clipped at y=1280
    ('raw2d:far', 0, 0, 100.0, 100.0, 140.0, 140.0),
]


def _frame(rows, z=423):
    return pd.DataFrame([{'detection_id': d, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                          'class': 'glia', 'score': 0.3, 'mean': 1.0, 'z': z}
                         for d, _, _, x1, y1, x2, y2 in rows])


def _origins(rows):
    return pd.DataFrame({'patch_x': [r[1] for r in rows], 'patch_y': [r[2] for r in rows]},
                        index=[r[0] for r in rows])


class PatchFragmentTests(unittest.TestCase):
    def test_et10_fragment_dropped_in_favour_of_larger_box(self):
        df = _frame(ET10_ROWS)
        cut = patch_cut(df, _origins(ET10_ROWS), GEOMETRY)
        self.assertEqual(cut.tolist(), [True, True, False])
        keep, winners, iomin = fragment_keep(df, cut, 0.5)
        self.assertEqual(keep.tolist(), [True, False, True])
        self.assertEqual(winners[1], 'raw2d:bec6')
        self.assertGreater(iomin[1], 0.8)

    def test_uncut_boxes_are_never_dropped(self):
        rows = [('a', 0, 0, 100.0, 100.0, 160.0, 160.0), ('b', 0, 0, 110.0, 110.0, 150.0, 150.0)]
        df = _frame(rows)
        cut = patch_cut(df, _origins(rows), GEOMETRY)
        self.assertFalse(cut.any())
        self.assertTrue(fragment_keep(df, cut, 0.5)[0].all())

    def test_tile_border_is_not_a_patch_cut(self):
        rows = [('edge', 1536, 0, 2000.0, 10.0, 2048.0, 50.0),
                ('left', 0, 0, 0.0, 10.0, 30.0, 50.0)]
        self.assertFalse(patch_cut(_frame(rows), _origins(rows), GEOMETRY).any())

    def test_other_slice_and_smaller_cover_do_not_count(self):
        rows = list(ET10_ROWS[:2])
        df = pd.concat([_frame(rows[:1], z=424), _frame(rows[1:], z=423)], ignore_index=True)
        cut = patch_cut(df, _origins(rows), GEOMETRY)
        self.assertTrue(fragment_keep(df, cut, 0.5)[0].all())
        # The larger box is never dropped in favour of the smaller fragment.
        df = _frame(rows)
        keep, _, _ = fragment_keep(df, patch_cut(df, _origins(rows), GEOMETRY), 0.5)
        self.assertTrue(keep[0])

    def test_fused_rows_use_winner_patch(self):
        df = _frame(ET10_ROWS)
        df['winner_detection_id'] = df['detection_id']
        df['detection_id'] = ['fused2d:1', 'fused2d:2', 'fused2d:3']
        self.assertEqual(patch_cut(df, _origins(ET10_ROWS), GEOMETRY).tolist(),
                         [True, True, False])

    def test_suppress_with_candidates_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame([{'detection_id': d, 'patch_x': px, 'patch_y': py, 'decision': 'kept'}
                          for d, px, py, *_ in ET10_ROWS]
                         + [{'detection_id': None, 'patch_x': 0, 'patch_y': 0,
                             'decision': 'suppressed_nms'}]
                         ).to_csv(Path(tmp) / 'T_RFP_candidates.csv', index=False)
            paths = candidate_paths(tmp, 'T', ['RFP'])
            filtered = _frame(ET10_ROWS)
            stats = {'before': 5, 'after': 3, 'removed': {'bbox': 2}, 'removed_total': 2}
            rejected = pd.DataFrame(columns=[*filtered.columns, 'rejection_reason'])
            out, rej, st = suppress_patch_fragments(filtered, rejected, stats, paths, GEOMETRY)
            self.assertEqual(out.detection_id.tolist(), ['raw2d:bec6', 'raw2d:far'])
            self.assertEqual(rej.rejection_reason.tolist(), ['patch_fragment'])
            self.assertEqual(rej.suppressed_by_detection_id.tolist(), ['raw2d:bec6'])
            self.assertEqual((st['after'], st['removed_total'], st['removed']['patch_fragment']),
                             (2, 3, 1))
            self.assertEqual(len(signature(GEOMETRY, paths)['candidates']), 1)
            with self.assertRaises(FileNotFoundError):
                candidate_paths(tmp, 'T', ['GFP'])

    def test_resolve(self):
        self.assertIsNone(resolve_patch_fragment(_config(None), {'id': 'RFP', 'model': 'yolo'}))
        self.assertEqual(resolve_patch_fragment(_config(), {'id': 'RFP', 'model': 'yolo'}),
                         GEOMETRY)
        with self.assertRaises(ValueError):
            resolve_patch_fragment(_config(1.5), {'id': 'RFP', 'model': 'yolo'})
        with self.assertRaises(ValueError):
            resolve_patch_fragment(_config(0.5, 'stardist'), {'id': 'RFP', 'model': 'stardist'})

    def test_key_is_not_part_of_raw_detection_identity(self):
        self.assertEqual(raw_detection_settings(_config(0.5)),
                         raw_detection_settings(_config(None)))


if __name__ == '__main__':
    unittest.main()
