"""Tile flagging from completed alignment QC records."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.qc.alignment_flags import flag_tiles, resolve_flag_options, write_flags

SETTINGS = {'xy_resolution_um': .65, 'z_resolution_um': 8.}
OPTIONS = {'sampling': {'min_pairs': 10, 'seed': 42}, 'reference_channel': 'RFP'}


def tf_record(tile, shift_px, n=40, seed=0):
    rng = np.random.default_rng(seed)
    soma = rng.uniform(0, 1000, (n, 3))
    noise = np.c_[rng.normal(0, 6, (n, 2)), np.zeros(n)]
    nuclear = soma + noise + np.asarray(shift_px)
    pairs = [dict(soma_points=s.tolist(), nuclear_points=u.tolist()) for s, u in zip(soma, nuclear)]
    return dict(tile=tile, pairs={'Sox9': pairs}, curves=[])


def iou_curves(tile, peak_px, at_zero=.2, peak=.3, unaligned=.1):
    rows = []
    for axis in 'xyz':
        for delta in range(-30, 31) if axis != 'z' else range(-5, 6):
            value = peak if (axis == 'x' and delta == peak_px) else (at_zero if delta == 0 else .05)
            scale = SETTINGS['z_resolution_um' if axis == 'z' else 'xy_resolution_um']
            rows.append(dict(tile=tile, channel='GFP', metric='voxel_iou', axis=axis, delta=delta,
                             delta_um=delta*scale, value=value, n_pairs=0, unaligned_iou=unaligned,
                             status='diagnostic'))
    return rows


class FlagTests(unittest.TestCase):
    def report(self, completed, failures=None):
        return dict(options=OPTIONS, settings=SETTINGS, completed=completed, failures=failures or {})

    def test_tf_offset_is_flagged_and_aligned_tile_passes(self):
        channels, tiles, _ = flag_tiles(self.report([tf_record('a', [0, 0, 0]), tf_record('b', [8, 0, 0])]))
        verdict = dict(zip(tiles.tile, tiles.verdict))
        self.assertEqual(verdict, {'a': 'ok', 'b': 'bad'})
        self.assertEqual(tiles.tile.iloc[0], 'b')
        self.assertAlmostEqual(channels.set_index('tile').loc['b', 'dx_um'], 8*.65, delta=1.5)

    def test_sparse_pairs_are_insufficient_not_bad(self):
        _, tiles, _ = flag_tiles(self.report([tf_record('a', [20, 0, 0], n=5)]))
        self.assertEqual(tiles.verdict.tolist(), ['insufficient'])

    def test_soma_iou_peak_away_from_zero_is_flagged(self):
        good = dict(tile='a', pairs={}, curves=iou_curves('a', 0, peak=.2))
        bad = dict(tile='b', pairs={}, curves=iou_curves('b', 20))
        unhelpful = dict(tile='c', pairs={}, curves=iou_curves('c', 0, at_zero=.1, peak=.1, unaligned=.1),
                         offsets={'RFP': {'dx': 0, 'dy': 0, 'dz': 0}, 'GFP': {'dx': 5, 'dy': 0, 'dz': 0}})
        # Same saved offset as the reference: the unaligned scan is identical, not a failure.
        same = dict(tile='d', pairs={}, curves=iou_curves('d', 0, at_zero=.1, peak=.1, unaligned=.1),
                    offsets={'RFP': {'dx': 0, 'dy': 0, 'dz': 0}, 'GFP': {'dx': 0, 'dy': 0, 'dz': 0}})
        _, tiles, _ = flag_tiles(self.report([good, bad, unhelpful, same]))
        self.assertEqual(dict(zip(tiles.tile, tiles.verdict)), {'a': 'ok', 'b': 'bad', 'c': 'bad', 'd': 'ok'})

    def test_failed_tiles_are_reported(self):
        _, tiles, _ = flag_tiles(self.report([tf_record('a', [0, 0, 0])], {'z': 'Traceback'}))
        self.assertEqual(tiles.iloc[0].to_dict()['verdict'], 'error')

    def test_write_flags_outputs_only_non_ok_tiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_flags(tmp, self.report([tf_record('a', [0, 0, 0]), tf_record('b', [8, 0, 0])]))
            text = (Path(tmp)/'flagged_tiles.csv').read_text()
            self.assertIn('b,bad', text)
            self.assertNotIn('a,ok', text)

    def test_invalid_options_rejected(self):
        with self.assertRaises(ValueError):
            resolve_flag_options({'max_offset_xy': 3})
        with self.assertRaises(ValueError):
            resolve_flag_options({'min_z_score': -1})


if __name__ == '__main__':
    unittest.main()
