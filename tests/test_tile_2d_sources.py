"""2D previews must overlay each saved stage in its own image frame."""

import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from qtpy.QtWidgets import QApplication

from src.utils.tile_2d import tile_2d_inputs
from src.utils.visualize import _run_2d
from src.utils.visualize_gui import VisualizeLauncher


class Tile2DTests(unittest.TestCase):
    def test_preview_uses_requested_csv_and_correct_image_shift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            offsets_dir = root / '3_2d_aligned'
            offsets_dir.mkdir()
            (offsets_dir / 'tile_offsets.json').write_text(
                json.dumps({'Olig2': dict(dx=7, dy=-3, dz=2)}), encoding='utf-8')
            # Each stage has a different box so stale/later results cannot pass.
            for index, (source, directory, shift) in enumerate([
                ('raw', '1_2d_raw', (0, 0, 0)),
                ('filtered', '2_2d_filtered', (0, 0, 0)),
                ('aligned', '3_2d_aligned', (7, -3, 2)),
                ('aligned_filtered', '4_2d_filtered', (7, -3, 2)),
            ]):
                with self.subTest(source=source):
                    stage_dir = root / directory
                    stage_dir.mkdir(exist_ok=True)
                    pd.DataFrame([dict(slice_name='tile_1', x1=10+index, y1=12,
                                       x2=20+index, y2=22, z=2, score=.8,
                                       mean=500, **{'class': 'nucleus_Olig2'})]).to_csv(
                        stage_dir / 'tile_Olig2_result.csv', index=False)
                    viewer = MagicMock()
                    with ExitStack() as stack:
                        for name in ('_add_contrast_panel', '_add_current_image_name',
                                     '_show_only_images_initially'):
                            stack.enter_context(patch('src.utils.visualize.' + name))
                        stack.enter_context(patch('src.utils.visualize.napari.Viewer', return_value=viewer))
                        stack.enter_context(patch('src.utils.visualize._resolve_z_range', return_value=(0, 4)))
                        stack.enter_context(patch('src.utils.visualize._canvas_shape_from_tile', return_value=(4, 32, 32)))
                        stack.enter_context(patch('src.utils.visualize._make_fn_recorder', return_value=None))
                        image = stack.enter_context(patch('src.utils.visualize.load_frame_volume',
                            return_value=(np.zeros((4, 32, 32)), (0, 1000))))
                        _run_2d({'2d_source': source},
                                {'pATHRESULT': tmp, 'olig2_dir': str(root / 'images')},
                                [{'id': 'Olig2', 'dir_key': 'olig2_dir'}],
                                str(root / 'images' / 'tile'), 'tile')
                    self.assertEqual(image.call_args.args[2], shift)
                    box = viewer.add_shapes.call_args.args[0][0]
                    self.assertEqual(box[0].tolist(), [1, 12, 10+index])

    def test_prefilter_needs_no_alignment_and_ignores_old_offsets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / '2_2d_filtered').mkdir()
            _, offsets = tile_2d_inputs(tmp, 'filtered', 'tile', ['Olig2'])
            self.assertEqual(offsets['Olig2'], dict(dx=0, dy=0, dz=0))
            (root / '4_2d_filtered').mkdir()
            with self.assertRaisesRegex(FileNotFoundError, 'offsets'):
                tile_2d_inputs(tmp, 'aligned_filtered', 'tile', ['Olig2'])

    def test_launcher_round_trips_source_keys_instead_of_display_labels(self):
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'vis.json'
            config.write_text(json.dumps({'mode': '2d', '2d_source': 'aligned_filtered'}), encoding='utf-8')
            window = VisualizeLauncher(str(config))
            try:
                self.assertEqual(window.source_2d.currentData(), 'aligned_filtered')
                window.source_2d.setCurrentIndex(window.source_2d.findData('filtered'))
                window._read_controls()
                self.assertEqual(window.config['2d_source'], 'filtered')
            finally:
                window.close()


if __name__ == '__main__':
    unittest.main()
