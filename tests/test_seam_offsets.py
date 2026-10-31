"""Seam offset measurement and seam-aware overlap deduplication."""

import unittest

import numpy as np
import pandas as pd

from src.core.seam_offsets import DEFAULT_PARAMS, measure_seam, resolve_offsets
from src.core.stitcher import _seam_shift, combine_predictions


def _boxes(centres, layers, size=10.0):
    h = size / 2
    return np.column_stack([centres[:, 0] - h, centres[:, 1] - h,
                            centres[:, 0] + h, centres[:, 1] + h, layers])


class MeasureSeamTest(unittest.TestCase):
    def test_recovers_offset_per_segment_including_dz(self):
        rng = np.random.default_rng(0)
        n = 4000
        centres = np.column_stack([rng.uniform(0, 300, n), rng.uniform(0, 2000, n)])
        layers = rng.integers(0, 100, n).astype(float)
        a = _boxes(centres, layers)
        shift = np.where(layers[:, None] < 50, [6.0, -4.0], [-3.0, 2.0])
        dz = np.where(layers < 50, 0, 1)
        keep = rng.random(n) < 0.7          # tile b misses some cells
        b = _boxes(centres[keep] + shift[keep], layers[keep] + dz[keep])
        noise = _boxes(np.column_stack([rng.uniform(0, 300, 1500), rng.uniform(0, 2000, 1500)]),
                       rng.integers(0, 101, 1500).astype(float))
        b = np.vstack([b, noise])
        rows = {r['seg']: r for r in measure_seam(a, b, (0, 300, 0, 2000), 3.0, DEFAULT_PARAMS)}
        self.assertAlmostEqual(rows[0]['meas_dx'], 6.0, delta=0.5)
        self.assertAlmostEqual(rows[0]['meas_dy'], -4.0, delta=0.5)
        self.assertEqual(rows[0]['meas_dz'], 0)
        self.assertAlmostEqual(rows[1]['meas_dx'], -3.0, delta=0.5)
        self.assertAlmostEqual(rows[1]['meas_dy'], 2.0, delta=0.5)
        self.assertEqual(rows[1]['meas_dz'], 1)
        self.assertGreater(rows[0]['n_dup'], 1000)


class ResolveOffsetsTest(unittest.TestCase):
    def _row(self, ch, seg, dx, n_dup=100, contrast=50.0):
        return dict(channel=ch, tile='a', nb_tile='b', direction='EAST', seg=seg, n_a=500,
                    n_dup=n_dup, contrast=contrast, meas_dx=dx, meas_dy=0.0, meas_dz=0)

    def test_fallback_order(self):
        measured = pd.DataFrame([
            self._row('Olig2', 0, 10.0), self._row('Olig2', 1, 20.0), self._row('Olig2', 2, 30.0),
            # GFP: valid in 0 and 1 (constant +2 vs Olig2), too few pairs in 2
            self._row('GFP', 0, 12.0), self._row('GFP', 1, 22.0), self._row('GFP', 2, 99.0, n_dup=3),
            # RFP: a single valid segment, no constant
            self._row('RFP', 0, 5.0),
        ])
        out = resolve_offsets(measured, 'Olig2', 4, DEFAULT_PARAMS).set_index(['channel', 'seg'])
        self.assertEqual(out.loc[('GFP', 0), 'source'], 'own')
        self.assertEqual(out.loc[('GFP', 2), 'source'], 'frame_plus_const')
        self.assertAlmostEqual(out.loc[('GFP', 2), 'dx'], 32.0)
        self.assertEqual(out.loc[('RFP', 1), 'source'], 'own_nearest')
        self.assertAlmostEqual(out.loc[('RFP', 1), 'dx'], 5.0)
        self.assertEqual(out.loc[('Olig2', 3), 'source'], 'own_nearest')
        self.assertAlmostEqual(out.loc[('Olig2', 3), 'dx'], 30.0)

    def test_frame_used_when_channel_has_nothing(self):
        measured = pd.DataFrame([self._row('Olig2', 0, 10.0), self._row('GFP', 0, 1.0, n_dup=2)])
        out = resolve_offsets(measured, 'Olig2', 2, DEFAULT_PARAMS).set_index(['channel', 'seg'])
        self.assertEqual(out.loc[('GFP', 0), 'source'], 'frame')
        self.assertEqual(out.loc[('GFP', 1), 'source'], 'frame_nearest')


class SeamShiftTest(unittest.TestCase):
    def test_direct_reverse_and_diagonal(self):
        names = {(0, 0): 'nw', (0, 1): 'n', (1, 0): 'w', (1, 1): 't'}
        seams = {('nw', 'n'): {0: (10, 1, 0)}, ('n', 't'): {0: (2, 20, 1)},
                 ('nw', 'w'): {0: (1, 10, 0)}, ('w', 't'): {0: (11, 11, 1)}}
        np.testing.assert_allclose(_seam_shift(seams, names, (0, 1), (1, 1), 0), [2, 20, 1])
        np.testing.assert_allclose(_seam_shift(seams, names, (1, 1), (0, 1), 0), [-2, -20, -1])
        # nw -> t: (10+2, 1+20, 1) and (1+11, 10+11, 1) averaged
        np.testing.assert_allclose(_seam_shift(seams, names, (0, 0), (1, 1), 0), [12, 21, 1])
        self.assertIsNone(_seam_shift(seams, names, (0, 0), (1, 1), 5))


class SeamDedupTest(unittest.TestCase):
    """Two 100 px tiles, the right one placed at x=80 (20 px overlap)."""

    @staticmethod
    def _row(name, x1, y1, x2, y2, cls="glia", z=1):
        return [name, x1, y1, x2, y2, cls, 0.5, 100, z]

    def _stitch(self, left, right, seams):
        predictions = [[np.empty((0, 8)) for _ in range(2)] for _ in range(3)]
        disp = np.array([[[0, 0, 0], [80, 0, 0]]], dtype=object)
        metadata, trace, ids, decisions = [], {}, {}, []
        extra = {} if seams is None else dict(
            seam_offsets=seams, grid_names={(0, 0): 'tile_0', (0, 1): 'tile_1'},
            processed_tiles=set())
        for col, rows in enumerate((left, right)):
            combine_predictions(predictions, rows, None, 0, 3, (0, col), disp, (100, 180),
                                metadata, f"tile_{col}", tILESIZE=100, row_meta=trace,
                                provenance_meta=ids, decision_meta=decisions, **extra)
        return predictions, trace, decisions

    def test_offset_duplicate_removed_only_with_seam_offsets(self):
        left = [self._row("left_cell", 85, 40, 95, 50)]
        right = [self._row("right_copy", 13, 40, 23, 50)]   # global 93..103: 8 px off
        old, _, _ = self._stitch(left, right, None)
        self.assertEqual(len(old[0][0]), 2)
        seams = {('tile_0', 'tile_1'): {0: (8.0, 0.0, 0)}}
        new, trace, decisions = self._stitch(left, right, seams)
        self.assertEqual(trace[(0, 0)], [("tile_0", "left_cell")])
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0]['reason'], 'cross_tile_overlap')

    def test_dz_offset_matches_neighbouring_layer(self):
        left = [self._row("left_cell", 85, 40, 95, 50, z=2)]
        right = [self._row("right_copy", 5, 40, 15, 50, z=1)]   # same xy, one layer lower
        seams = {('tile_0', 'tile_1'): {0: (0.0, 0.0, -1)}}
        new, trace, _ = self._stitch(left, right, seams)
        self.assertEqual(sum(len(v) for v in trace.values()), 1)

    def test_unmatched_overlap_detection_moves_into_neighbour_frame(self):
        left = [self._row("left_other", 85, 70, 95, 80)]
        right = [self._row("right_unique", 5, 10, 15, 20)]
        seams = {('tile_0', 'tile_1'): {0: (8.0, -2.0, 0)}}
        new, trace, _ = self._stitch(left, right, seams)
        self.assertEqual(len(new[0][0]), 2)
        moved = new[0][0][1]
        self.assertEqual([float(v) for v in moved[:4]], [77.0, 12.0, 87.0, 22.0])

    def test_detection_outside_overlap_is_untouched(self):
        right = [self._row("right_inner", 50, 10, 60, 20)]
        seams = {('tile_0', 'tile_1'): {0: (8.0, -2.0, 0)}}
        new, _, _ = self._stitch([], right, seams)
        self.assertEqual([float(v) for v in new[0][0][0][:4]], [130.0, 10.0, 140.0, 20.0])


if __name__ == "__main__":
    unittest.main()
