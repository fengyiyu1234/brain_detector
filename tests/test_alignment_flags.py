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
        # Aligned no better than unaligned is inconclusive, not bad (small QC region).
        self.assertEqual(dict(zip(tiles.tile, tiles.verdict)),
                         {'a': 'ok', 'b': 'bad', 'c': 'insufficient', 'd': 'ok'})

    def test_soma_z_rules_need_interior_peak_two_slices_and_gain(self):
        def z_curve(tile, values):
            rows = []
            for axis in 'xyz':
                deltas = range(-5, 6) if axis == 'z' else range(-30, 31)
                for delta in deltas:
                    # zero shift is the same voxel IoU on every axis
                    value = values[delta + 5] if axis == 'z' or delta == 0 else .01
                    scale = SETTINGS['z_resolution_um' if axis == 'z' else 'xy_resolution_um']
                    rows.append(dict(tile=tile, channel='GFP', metric='voxel_iou', axis=axis,
                                     delta=delta, delta_um=delta*scale, value=value, n_pairs=0,
                                     unaligned_iou=.05, status='diagnostic'))
            return dict(tile=tile, pairs={}, curves=rows)
        # tsc18 325300_342500: clear interior peak at +2 slices, x1.95 -> bad
        real = z_curve('real', [.037, .039, .044, .052, .061, .088, .136, .172, .160, .126, .094])
        # Like TSC10 401700_394600 (x1.25 there, now below the gain limit anyway), but with
        # x1.67: still a monotonic rise to the +5 scan edge -> insufficient, not bad
        edge = z_curve('edge', [.001, .005, .010, .016, .026, .036, .040, .044, .050, .055, .060])
        # one slice off with a small gain -> ok
        small = z_curve('small', [.05, .05, .05, .05, .09, .10, .12, .05, .05, .05, .05])
        # two slices off but only x1.1 -> ok
        weak = z_curve('weak', [.05, .05, .05, .05, .05, .10, .05, .11, .05, .05, .05])
        channels, tiles, _ = flag_tiles(self.report([real, edge, small, weak]))
        verdict = dict(zip(tiles.tile, tiles.verdict))
        self.assertEqual(verdict, {'real': 'bad', 'edge': 'insufficient', 'small': 'ok', 'weak': 'ok'})
        self.assertIn('inconclusive', channels.set_index('tile').loc['edge', 'reasons'])

    def test_refresh_uses_current_defaults_not_stored_ones(self):
        import json
        from src.qc.alignment_flags import refresh_flags
        old = dict(OPTIONS, flagging={'bootstrap_samples': 1000, 'iou_min_gain': 1.05,
                                      'iou_min_vs_unaligned': 1.1, 'max_offset_xy_um': 3.0,
                                      'max_offset_z_um': 4.0, 'min_z_score': 3.0})
        # x peak at +6 px (3.9 um) with x1.1 gain: bad under the old rules only
        record = dict(tile='a', pairs={}, curves=iou_curves('a', 6, at_zero=.2, peak=.22))
        with tempfile.TemporaryDirectory() as tmp:
            report = dict(options=old, settings=SETTINGS, completed=[record], failures={})
            (Path(tmp)/'run_details.json').write_text(json.dumps(report), encoding='utf-8')
            (Path(tmp)/'qc_report.md').write_text('# Alignment\n\nkeep me\n\n## Flagged tiles\nold\n',
                                                  encoding='utf-8')
            tiles, _ = refresh_flags(tmp)
            self.assertEqual(tiles.verdict.tolist(), ['ok'])
            details = json.loads((Path(tmp)/'run_details.json').read_text())
            self.assertEqual(details['options']['flagging']['iou_min_gain'], 1.3)
            self.assertEqual(details['flagged'], [])
            md = (Path(tmp)/'qc_report.md').read_text()
            self.assertIn('keep me', md)
            self.assertNotIn('\nold\n', md)
            self.assertEqual(md.count('## Flagged tiles'), 1)

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
