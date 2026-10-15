"""Restart jobs must preserve raw input and reject damaged cached outputs."""
import tempfile
import json
import unittest
from pathlib import Path
import pandas as pd
from src.core.refilter_align import filter_one, validate_filtered_one


class RefilterAlignmentTests(unittest.TestCase):
    def test_raw_unchanged_resume_and_corrupt_output_rebuilt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'raw/tile_Sox9_result.csv'
            source.parent.mkdir()
            pd.DataFrame({
                'slice_name': ['s1', 's1', 's2'], 'z': [1, 1, 2],
                'x1': [0, 20, 40], 'y1': [0, 0, 0],
                'x2': [10, 30, 50], 'y2': [10, 10, 10],
                'class': ['nucleus'] * 3, 'score': [.9, .1, .8], 'mean': [30] * 3,
            }).to_csv(source, index=False)
            original = source.read_bytes()
            target = root / 'filtered/tile_Sox9_result.csv'
            job = (str(source), str(target), {'score_min': .35}, {'version': 1}, 'tile', 'Sox9')
            first = filter_one(job)
            self.assertEqual((first['before'], first['after']), (3, 2))
            self.assertFalse(first['cached'])
            expected = target.read_bytes()
            self.assertEqual(source.read_bytes(), original)
            self.assertTrue(filter_one(job)['cached'])
            # Keep the same length so checking only file size cannot detect damage.
            target.write_bytes(expected.replace(b'0.9', b'0.7', 1))
            self.assertEqual(target.stat().st_size, len(expected))
            self.assertFalse(filter_one(job)['cached'])
            self.assertEqual(target.read_bytes(), expected)
            self.assertEqual(source.read_bytes(), original)
            rejected = pd.read_csv(root / 'filtered/tile_Sox9_rejected.csv')
            self.assertEqual(rejected.rejection_reason.tolist(), ['score_min'])
            source.with_name('tile_Sox9_inputs.json').write_text(
                '{"output_sha256": "invalid-original-hash"}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'original inference fingerprint'):
                filter_one(job)


    def test_alignment_only_accepts_portable_paths_but_never_repairs_bad_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'raw/tile_Sox9_result.csv'
            source.parent.mkdir()
            pd.DataFrame({'slice_name': ['s1'], 'z': [1], 'x1': [0], 'y1': [0],
                          'x2': [10], 'y2': [10], 'class': ['nucleus'],
                          'score': [.9], 'mean': [30]}).to_csv(source, index=False)
            target = root / 'filtered/tile_Sox9_result.csv'
            job = (str(source), str(target), {'score_min': .35}, {'version': 1}, 'tile', 'Sox9')
            filter_one(job)
            original = target.read_bytes()
            manifest = target.with_name('tile_Sox9_filter_manifest.json')
            saved = json.loads(manifest.read_text())
            saved['input_signature']['source']['path'] = r'C:\old\raw\tile_Sox9_result.csv'
            saved['outputs'] = {
                r'C:\old\filtered' + '\\' + Path(name).name: stamp
                for name, stamp in saved['outputs'].items()
            }
            manifest.write_text(json.dumps(saved), encoding='utf-8')
            self.assertTrue(validate_filtered_one(job)['cached'])
            self.assertEqual(target.read_bytes(), original)
            damaged = original.replace(b'0.9', b'0.7', 1)
            target.write_bytes(damaged)
            with self.assertRaisesRegex(ValueError, 'modified'):
                validate_filtered_one(job)
            self.assertEqual(target.read_bytes(), damaged)
            changed_params = (*job[:2], {'score_min': .5}, *job[3:])
            with self.assertRaisesRegex(ValueError, 'different parameters'):
                validate_filtered_one(changed_params)


if __name__ == '__main__':
    unittest.main()
