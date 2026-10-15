"""Image-index sampling must use TIFF depth rather than detection density."""
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from src.core.point_cloud_aligner import alignment_sample_window, resolve_align_settings, align_tile


class AlignmentSampleWindowTests(unittest.TestCase):
    def test_800_images_use_exactly_501_through_600(self):
        self.assertEqual(alignment_sample_window(800, {
            'sample_z_start_fraction': .625, 'sample_z_center_count': 100}), (501, 601))

    def test_shorter_tiles_and_end_clamp(self):
        self.assertEqual(alignment_sample_window(400, {
            'sample_z_start_fraction': .625, 'sample_z_center_count': 100}), (251, 351))
        self.assertEqual(alignment_sample_window(120, {
            'sample_z_start_fraction': .625, 'sample_z_center_count': 100}), (21, 121))
        with self.assertRaisesRegex(ValueError, 'fewer'):
            alignment_sample_window(99, {'sample_z_start_fraction': .625,
                                        'sample_z_center_count': 100})

    def test_legacy_mode_unchanged(self):
        self.assertIsNone(alignment_sample_window(800, {'sample_z_center_count': 50}))

    def test_align_uses_image_window_and_records_exact_source_names(self):
        routing = [{'id': 'GFP', 'type': 'soma'}, {'id': 'RFP', 'type': 'soma'}]
        settings = resolve_align_settings({'pre_align_params': {
            'sample_z_start_slice': 501, 'sample_z_count': 100}}, routing)
        cells = [{'cz': z, 'z_min': z, 'z_max': z} for z in (500, 501, 600, 601)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tile, raw, out = root / 'tile', root / 'raw', root / 'aligned'
            tile.mkdir()
            raw.mkdir()
            for ch in routing:
                pd.DataFrame([[f's{z:04d}', 1, 2, 10, 12, 'glia', .9, 20, z]
                              for z in (1, 501, 800)],
                             columns=['slice_name', 'x1', 'y1', 'x2', 'y2',
                                      'class', 'score', 'mean', 'z']).to_csv(
                    raw / f"tile_{ch['id']}_result.csv", index=False)
            names = [f's{i:04d}.tif' for i in range(1, 801)]
            with patch('src.core.point_cloud_aligner.os.listdir', return_value=names), patch(
                    'src.core.z_linker.run_z_linker', return_value=([], cells)), patch(
                    'src.core.point_cloud_aligner.compute_tile_channel_shifts',
                    return_value=({'GFP': (0, 0, 0), 'RFP': (1, 2, 0)}, {})) as solve:
                align_tile(str(tile), str(raw), str(out), routing, settings)
            self.assertEqual(solve.call_args.kwargs['z_center'], 551)
            self.assertEqual(solve.call_args.kwargs['z_half_window'], 50)
            self.assertEqual([c['cz'] for c in solve.call_args.args[0]['GFP']], [501, 600])
            saved = json.loads((out / 'tile_measured_offsets.json').read_text())
            self.assertEqual(saved['sample_window']['first_slice_name'], 's0501')
            self.assertEqual(saved['sample_window']['last_slice_name'], 's0600')
            self.assertEqual(saved['sample_window']['count'], 100)
            # The calibration window must not crop the full-depth output.
            self.assertEqual(pd.read_csv(out / 'tile_GFP_result.csv').z.tolist(), [1, 501, 800])
            self.assertEqual(pd.read_csv(out / 'tile_RFP_result.csv').z.tolist(), [1, 501, 800])

    def test_explicit_window_is_independent_of_tile_depth(self):
        for depth in (751, 800, 900):
            self.assertEqual(alignment_sample_window(depth, {
                'sample_z_start_slice': 501, 'sample_z_count': 100}), (501, 601))
        self.assertEqual(alignment_sample_window(751, {
            'sample_z_start_slice': 501, 'sample_z_count': 101}), (501, 602))
        self.assertEqual(alignment_sample_window(751, {
            'sample_z_start_slice': 751, 'sample_z_count': 1}), (751, 752))
        with self.assertRaisesRegex(ValueError, 'only 751'):
            alignment_sample_window(751, {'sample_z_start_slice': 700, 'sample_z_count': 100})

    def test_odd_count_reaches_solver_exactly(self):
        routing = [{'id': 'GFP', 'type': 'soma'}]
        settings = resolve_align_settings({'pre_align_params': {
            'sample_z_start_slice': 11, 'sample_z_count': 3}}, routing)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tile, raw = root / 'tile', root / 'raw'
            tile.mkdir()
            raw.mkdir()
            pd.DataFrame([['s0011', 1, 2, 10, 12, 'glia', .9, 20, 11]],
                         columns=['slice_name', 'x1', 'y1', 'x2', 'y2',
                                  'class', 'score', 'mean', 'z']).to_csv(
                raw / 'tile_GFP_result.csv', index=False)
            with patch('src.core.point_cloud_aligner.os.listdir',
                       return_value=[f's{i:04d}.tif' for i in range(1, 21)]), patch(
                    'src.core.z_linker.run_z_linker', return_value=([], [])), patch(
                    'src.core.point_cloud_aligner.compute_tile_channel_shifts',
                    return_value=({'GFP': (0, 0, 0)}, {})) as solve:
                align_tile(str(tile), str(raw), str(root / 'aligned'), routing, settings)
            self.assertEqual(solve.call_args.kwargs['z_center'], 12.5)
            self.assertEqual(solve.call_args.kwargs['z_half_window'], 1.5)
            saved = json.loads((root / 'aligned/tile_measured_offsets.json').read_text())
            self.assertEqual(saved['sample_window']['last_slice_name'], 's0013')
            self.assertEqual(saved['sample_window']['count'], 3)

    def test_invalid_explicit_parameters(self):
        bad = [
            {'sample_z_start_slice': value, 'sample_z_count': 100}
            for value in (True, 0, -1, 1.5)
        ] + [
            {'sample_z_start_slice': 501, 'sample_z_count': value}
            for value in (True, 0, -1, 1.5)
        ] + [
            {'sample_z_start_slice': 501, 'sample_z_start_fraction': .625},
            {'sample_z_count': 100, 'sample_z_center_count': 50},
        ]
        for params in bad:
            with self.subTest(params=params), self.assertRaises(ValueError):
                resolve_align_settings({'pre_align_params': params},
                                       [{'id': 'GFP', 'type': 'soma'}])

    def test_invalid_fraction_and_count(self):
        for fraction, count in [(True, 100), (-.1, 100), (1, 100),
                                (float('nan'), 100), (.625, 0), (.625, True)]:
            with self.subTest(fraction=fraction, count=count), self.assertRaises(ValueError):
                resolve_align_settings({'pre_align_params': {
                    'sample_z_start_fraction': fraction, 'sample_z_center_count': count}},
                    [{'id': 'GFP', 'type': 'soma'}])


if __name__ == '__main__':
    unittest.main()
