"""Stage 2.5 nonblank_center sampling window and its compatibility with older settings."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.core.align_window import (nonblank_center_window, nonblank_indices, resolve_window_mode,
                                   soma_dense_window, tile_windows)
from src.core.point_cloud_aligner import align_tile, resolve_align_settings

ROUTING = [{'id': 'GFP', 'type': 'soma'}, {'id': 'RFP', 'type': 'soma'}]


def _tile(root, channel, tile, sizes):
    folder = Path(root) / channel / tile.split('_')[0] / tile
    folder.mkdir(parents=True, exist_ok=True)
    for i, size in enumerate(sizes, 1):
        (folder / f'{tile}_{i:04d}.tif').write_bytes(b'\0' * size)
    return folder


class WindowTests(unittest.TestCase):
    def test_centre_and_clamp(self):
        self.assertEqual(nonblank_center_window(list(range(101, 701)), 100), (350, 450, 100))
        self.assertEqual(nonblank_center_window(list(range(1, 41)), 100), (1, 41, 40))
        self.assertIsNone(nonblank_center_window([], 100))

    def test_fewer_than_count_uses_all_even_with_gaps(self):
        self.assertEqual(nonblank_center_window([5, 6, 200], 10), (5, 201, 3))

    def test_window_stays_inside_nonblank_range(self):
        start, stop, n = nonblank_center_window(list(range(1, 60)) + list(range(400, 420)), 50)
        self.assertGreaterEqual(start, 1)
        self.assertLessEqual(stop, 420)
        self.assertEqual(stop - start, 50)

    def test_blank_slices_at_both_ends_and_all_channels_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            sizes = [10] * 3 + [100] * 6 + [10] * 3
            gfp = _tile(tmp, 'g', '100_200', sizes)
            rfp_sizes = list(sizes)
            rfp_sizes[4] = 10                      # one channel blank -> slice blank
            _tile(tmp, 'r', '100_200', rfp_sizes)
            idx, n = nonblank_indices(str(gfp), [f'{tmp}/g', f'{tmp}/r'], f'{tmp}/g', 50)
            self.assertEqual(n, 12)
            self.assertEqual(idx, [4, 6, 7, 8, 9])
            w = tile_windows([str(gfp)], f'{tmp}/g', [f'{tmp}/g', f'{tmp}/r'], 5, 50)['100_200']
            self.assertEqual((w['start'], w['stop']), (4, 10))   # <= count non-blank: all used
            w = tile_windows([str(gfp)], f'{tmp}/g', [f'{tmp}/g', f'{tmp}/r'], 3, 50)['100_200']
            self.assertEqual((w['start'], w['stop'], w['n_nonblank_in_window']), (6, 9, 3))

    def test_all_blank_and_mismatched_depth(self):
        with tempfile.TemporaryDirectory() as tmp:
            gfp = _tile(tmp, 'g', '100_200', [10] * 5)
            _tile(tmp, 'r', '100_200', [10] * 5)
            w = tile_windows([str(gfp)], f'{tmp}/g', [f'{tmp}/g', f'{tmp}/r'], 4, 50)['100_200']
            self.assertIsNone(w['start'])
            _tile(tmp, 'o', '100_200', [10] * 4)
            with self.assertRaisesRegex(ValueError, 'slices'):
                nonblank_indices(str(gfp), [f'{tmp}/o'], f'{tmp}/g', 50)


class SomaDenseTests(unittest.TestCase):
    def test_densest_window_by_smaller_channel(self):
        # RFP alone is densest at 300-399; GFP+RFP together at 100-199 (tsc18-like).
        rfp = list(range(100, 200, 2)) + list(range(300, 400))
        gfp = list(range(100, 200, 2)) + [350]
        self.assertEqual(soma_dense_window([rfp, gfp], 100, 1, 700), (100, 200, 50))

    def test_bounds_ties_and_short_range(self):
        cells = [[50], [50]]
        start, stop, n = soma_dense_window(cells, 10, 1, 100)
        self.assertEqual((stop - start, n), (10, 1))
        self.assertTrue(41 <= start <= 50)
        self.assertEqual(start, 46)        # ties go towards the middle of the range (46-55)
        self.assertEqual(soma_dense_window([[5, 6], [7]], 100, 3, 40), (3, 41, 1))
        self.assertEqual(soma_dense_window([[500], [500]], 100, 450, 700)[:2], (500, 600))

    def test_none_without_cells_of_every_channel(self):
        self.assertIsNone(soma_dense_window([[10, 20], []], 100, 1, 700))
        self.assertIsNone(soma_dense_window([[10], [10]], 100, 200, 700))
        self.assertIsNone(soma_dense_window([[10], [10]], 100, None, None))

    def test_tile_windows_record_nonblank_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            gfp = _tile(tmp, 'g', '100_200', [10] * 3 + [100] * 6 + [10] * 3)
            w = tile_windows([str(gfp)], f'{tmp}/g', [f'{tmp}/g'], 3, 50)['100_200']
            self.assertEqual((w['first_nonblank'], w['last_nonblank']), (4, 9))


class SettingsTests(unittest.TestCase):
    def test_old_configs_unchanged(self):
        for pa in ({'sample_z_start_slice': 501, 'sample_z_count': 100}, {'sample_z_center_count': 100}, {}):
            config = {'pre_align_params': pa}
            base = resolve_align_settings(config, ROUTING)
            self.assertEqual(resolve_window_mode(config, dict(base)), base)

    def test_mode_adds_keys_and_excludes_fixed_start(self):
        config = {'pre_align_params': {'sample_z_mode': 'nonblank_center', 'sample_z_count': 100}}
        s = resolve_window_mode(config, resolve_align_settings(config, ROUTING))
        self.assertEqual((s['sample_z_mode'], s['blank_image_max_bytes'], s['sample_z_count']),
                         ('nonblank_center', 1_000_000, 100))
        bad = {'pre_align_params': {'sample_z_mode': 'nonblank_center', 'sample_z_start_slice': 5,
                                    'sample_z_count': 10}}
        with self.assertRaises(ValueError):
            resolve_window_mode(bad, resolve_align_settings(bad, ROUTING))
        dense = {'pre_align_params': {'sample_z_mode': 'soma_dense', 'sample_z_count': 100}}
        self.assertEqual(resolve_window_mode(dense, resolve_align_settings(dense, ROUTING))
                         ['sample_z_mode'], 'soma_dense')
        unknown = {'pre_align_params': {'sample_z_mode': 'densest', 'sample_z_count': 100}}
        with self.assertRaises(ValueError):
            resolve_window_mode(unknown, resolve_align_settings(unknown, ROUTING))

    def test_previous_aligner_hash_is_compatible(self):
        from src.core import point_cloud_aligner as pca
        source = Path(pca.__file__).read_text(encoding='utf-8')
        self.assertIn('95f6dba865c6552761c2f15a0698cc3e426442755432a2bed9bac0def55da44f', source)
        # last version before soma_dense: ERH/t70/DT47 alignments of 2026-11-17
        self.assertIn('df829b13c6dd2aa7b0aa1da3029ae33ce293e09163d1b94069e8092837ddd08c', source)


class AlignTileTests(unittest.TestCase):
    def _run(self, override, mode='nonblank_center'):
        config = {'pre_align_params': {'sample_z_mode': mode, 'sample_z_count': 100}}
        settings = resolve_window_mode(config, resolve_align_settings(config, ROUTING))
        cells = [{'cz': z, 'z_min': z, 'z_max': z} for z in (10, 300, 349, 360)]
        tmp = tempfile.mkdtemp()
        root = Path(tmp)
        tile, raw, out = root / 'tile', root / 'raw', root / 'aligned'
        tile.mkdir()
        raw.mkdir()
        for ch in ROUTING:
            pd.DataFrame([[f's{z:04d}', 1, 2, 10, 12, 'glia', .9, 20, z] for z in (1, 300, 700)],
                         columns=['slice_name', 'x1', 'y1', 'x2', 'y2', 'class', 'score', 'mean', 'z']
                         ).to_csv(raw / f"tile_{ch['id']}_result.csv", index=False)
        names = [f's{i:04d}.tif' for i in range(1, 801)]
        with patch('src.core.point_cloud_aligner.os.listdir', return_value=names), patch(
                'src.core.z_linker.run_z_linker', return_value=([], cells)), patch(
                'src.core.point_cloud_aligner.compute_tile_channel_shifts',
                return_value=({'GFP': (0, 0, 1), 'RFP': (0, 0, 0)}, {'GFP': .5})) as solve:
            align_tile(str(tile), str(raw), str(out), ROUTING, settings, override)
        return solve, json.loads((out / 'tile_measured_offsets.json').read_text()), out

    def test_window_override_used(self):
        solve, saved, _ = self._run({'start': 300, 'stop': 400, 'n_nonblank_in_window': 100})
        self.assertEqual(solve.call_args.kwargs['z_center'], 350)
        self.assertEqual([c['cz'] for c in solve.call_args.args[0]['GFP']], [300, 349, 360])
        self.assertEqual(saved['sample_window']['mode'], 'nonblank_center')
        self.assertFalse(saved['sample_window']['no_tissue'])

    def test_soma_dense_moves_window_to_cells(self):
        # cells at 300/349/360 for both soma channels; nonblank_center would use 500-600
        override = {'start': 500, 'stop': 600, 'first_nonblank': 1, 'last_nonblank': 800}
        solve, saved, _ = self._run(override, 'soma_dense')
        self.assertEqual(solve.call_args.kwargs['z_center'], 350)
        w = saved['sample_window']
        self.assertEqual((w['mode'], w['z_start'], w['z_stop_exclusive']), ('soma_dense', 300, 400))
        self.assertEqual((w['soma_cells_in_window'], w['fallback_nonblank_center']), (3, False))

    def test_soma_dense_falls_back_without_cells(self):
        override = {'start': 500, 'stop': 600, 'first_nonblank': 450, 'last_nonblank': 800}
        solve, saved, _ = self._run(override, 'soma_dense')
        self.assertEqual(solve.call_args.kwargs['z_center'], 550)
        self.assertTrue(saved['sample_window']['fallback_nonblank_center'])
        with self.assertRaisesRegex(ValueError, 'first/last_nonblank'):
            self._run({'start': 500, 'stop': 600}, 'soma_dense')

    def test_no_tissue_gives_zero_shifts(self):
        solve, saved, out = self._run({'start': None, 'stop': None})
        solve.assert_not_called()
        self.assertTrue(saved['sample_window']['no_tissue'])
        offsets = json.loads((out / 'tile_offsets.json').read_text())
        self.assertEqual({c: (v['dx'], v['dy'], v['dz']) for c, v in offsets.items()},
                         {'GFP': (0, 0, 0), 'RFP': (0, 0, 0)})
        self.assertEqual(pd.read_csv(out / 'tile_GFP_result.csv').z.tolist(), [1, 300, 700])


if __name__ == '__main__':
    unittest.main()
