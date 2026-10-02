"""Tile reference conversion and integrated solver checkpoints."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from scripts.solve_tile_positions import (
    channel_xml_positions, write_channel_xml, publish_channel_xml,
)
from src.core.tile_position_pipeline import run_tile_position_stage, tile_position_frame


class TilePositionPipelineTests(unittest.TestCase):
    def test_one_setting_selects_the_final_frame(self):
        config = {'stitching_reference_channel': 'Olig2',
                  'pre_align_params': {'reference_channel': 'GFP'},
                  'tile_position_params': {'enabled': True}}
        self.assertEqual(tile_position_frame(config), 'Olig2')
        config['tile_position_params']['reference_channel'] = 'GFP'
        with self.assertRaisesRegex(ValueError, 'redundant'):
            tile_position_frame(config)

    def test_image_xml_uses_seam_geometry_and_channel_translation(self):
        rows = pd.DataFrame({
            'tile': ['t0', 't1'],
            'P_GFP_x': [10, 110], 'P_GFP_y': [20, 20], 'P_GFP_z': [4, 6],
            'P_Olig2_x': [5, 106], 'P_Olig2_y': [18, 18],
            'P_Olig2_z': [1, 4],
            # Cell alignment is measured with GFP as reference, then rebased
            # to the Olig2 tile frame. The residual is local to each tile.
            's_GFP_x': [8, 9], 's_GFP_y': [4, 5], 's_GFP_z': [6, 7],
        })
        positions = channel_xml_positions(rows, 'Olig2', 'GFP')
        np.testing.assert_allclose(positions, [[14, 22.5, 8],
                                               [114, 22.5, 10]])
        np.testing.assert_allclose(
            channel_xml_positions(rows, 'Olig2', 'Olig2'),
            [[5, 18, 1], [106, 18, 4]])
        with tempfile.TemporaryDirectory() as tmp:
            root = ET.Element('TeraStitcher')
            stacks = ET.SubElement(root, 'STACKS')
            for tile in ('t0', 't1'):
                ET.SubElement(stacks, 'Stack', DIR_NAME=tile)
            template = Path(tmp) / 'xml_import.xml'
            ET.ElementTree(root).write(template)
            output = Path(tmp) / 'xml_merging_GFP.xml'
            n, missing = write_channel_xml(
                template, output, ['t0', 't1'], positions,
                np.array([5, 18, 1]), str(Path(tmp)))
            self.assertEqual((n, missing), (2, []))
            stacks = list(ET.parse(output).getroot().find('STACKS'))
            self.assertEqual([(s.get('ABS_H'), s.get('ABS_D')) for s in stacks],
                             [('9', '7'), ('109', '9')])
            published = publish_channel_xml(output, tmp, str(template))
            self.assertEqual(Path(published).read_bytes(), output.read_bytes())
            self.assertTrue((Path(tmp) / 'xml_merging.original.xml').is_file())
            # Existing merging geometry is backed up before replacement.
            backup = Path(tmp) / 'xml_merging.original.xml'
            backup.unlink()
            target = Path(tmp) / 'xml_merging.xml'
            target.write_text('<old/>')
            publish_channel_xml(output, tmp, str(target))
            self.assertEqual(backup.read_text(), '<old/>')

    def test_solver_checkpoint_and_changed_input_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            det, align, report = (root / x for x in
                                  ('1_tile_2d_prefiltered', '0_channel_alignment',
                                   '5_analysis_report/tile_positions'))
            channels = [{'id': 'GFP', 'type': 'soma', 'dir_key': 'gfp'},
                        {'id': 'Olig2', 'type': 'tf', 'dir_key': 'olig2'}]
            paths = {}
            for ch in channels:
                directory = root / ch['id']
                directory.mkdir()
                (directory / 'xml_import.xml').write_text('<root/>')
                paths[ch['dir_key']] = str(directory)
            det.mkdir()
            align.mkdir()
            for ch in channels:
                (det / f"t0_{ch['id']}_result.csv").write_text('z,x\\n1,2\\n')
            (align / 't0_offsets.json').write_text('{}')
            script = root / 'solver.py'
            script.write_text('pass')
            cfg = root / 'config.json'
            cfg.write_text('{}')
            config = {'paths': paths, 'stitching_reference_channel': 'Olig2',
                      'pre_align_params': {'reference_channel': 'GFP'},
                      'tile_position_params': {'enabled': True}}
            calls = []
            def fake_run(command, check):
                calls.append(command)
                report.mkdir(parents=True, exist_ok=True)
                for ch in channels:
                    name = f"xml_merging_{ch['id']}.xml"
                    (report / name).write_text('<root/>')
                    (Path(paths[ch['dir_key']]) / 'xml_merging.xml').write_text('<root/>')
                    (Path(paths[ch['dir_key']]) / 'xml_merging.original.xml').write_text('<root/>')
            call = lambda: run_tile_position_stage(
                config, str(cfg), str(root), str(det), str(align), str(report),
                ['t0'], channels, str(script))
            with patch('src.core.tile_position_pipeline.subprocess.run', side_effect=fake_run):
                frame_xml, computed = call()
                self.assertTrue(computed)
                self.assertTrue(frame_xml.endswith('xml_merging_Olig2.xml'))
                self.assertIn('--alignment-from', calls[0])
                self.assertIn('--xml-into-channel-dirs', calls[0])
                self.assertFalse(call()[1])
                source = det / 't0_GFP_result.csv'
                source.write_text('z,x\\n1,2\\n2,3\\n')
                global_dir = root / '2_global_2d_raw'
                global_dir.mkdir()
                (global_dir / 'GFP_2d_global.csv').write_text('global')
                with self.assertRaisesRegex(RuntimeError, 'global checkpoints'):
                    call()
                self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
