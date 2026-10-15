"""Sparse-window fallbacks must remain explicit and preserve useful checkpoints."""
import tempfile
import json
import hashlib
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from src.utils.run_alignment_qc import quality_process_tile, cache_signature

class QualifiedWindowTests(unittest.TestCase):
    def test_qualified_window_records_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            params={'qc_checkpoint_dir':tmp,'eval':{'sampling':{'min_cells':15,'regions_per_tile':1}}}
            with patch('src.utils.run_alignment_qc.validation._process_tile',return_value=([{'a':1}],[])):
                rows,warnings=quality_process_tile('tile',params)
            self.assertTrue(rows[0]['signal_qualified'])
            self.assertFalse(warnings)

    def test_sparse_fallback_is_flagged_and_does_not_mutate_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            params={'qc_checkpoint_dir':tmp,'eval':{'sampling':{'min_cells':15,'regions_per_tile':1}}}
            with patch('src.utils.run_alignment_qc.validation._process_tile',side_effect=[ValueError('Cannot select signal-qualified windows'),([{'a':1}],[])]) as run:
                rows,warnings=quality_process_tile('tile',params)
                self.assertEqual(run.call_args.args[1]['eval']['sampling']['min_cells'],0)
            self.assertFalse(rows[0]['signal_qualified'])
            self.assertIn('accuracy unknown',warnings[0])
            self.assertEqual(params['eval']['sampling']['min_cells'],15)

    def test_other_errors_are_not_silently_relaxed(self):
        with tempfile.TemporaryDirectory() as tmp:
            params={'qc_checkpoint_dir':tmp,'eval':{'sampling':{'min_cells':15,'regions_per_tile':1}}}
            with patch('src.utils.run_alignment_qc.validation._process_tile',side_effect=ValueError('Invalid coordinates')) as run:
                with self.assertRaisesRegex(ValueError,'Invalid coordinates'):
                    quality_process_tile('tile',params)
                self.assertEqual(run.call_count,1)

    def test_signal_qualified_checkpoint_is_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame([dict(n_ref_cells=20,n_tgt_cells=30,region_id=0)]).to_csv(Path(tmp)/'tile_curves.csv',index=False)
            params={'qc_checkpoint_dir':tmp,'eval':{'sampling':{'min_cells':15,'regions_per_tile':1}}}
            checkpoint=Path(tmp)/'tile_curves.csv'
            checkpoint.with_suffix('.json').write_text(json.dumps(dict(input_signature=cache_signature('tile',params),
                csv_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())))
            with patch('src.utils.run_alignment_qc.validation._process_tile') as run:
                rows,_=quality_process_tile('tile',params)
                run.assert_not_called()
            self.assertTrue(rows[0]['signal_qualified'])


    def test_changed_scan_does_not_reuse_coarse_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            params={'qc_checkpoint_dir':tmp,'eval':{'sampling':{'min_cells':15,'regions_per_tile':1},'sweep':{'x':{'range_px':30}}}}
            rows=[dict(n_ref_cells=20,n_tgt_cells=30,region_id=0)]
            with patch('src.utils.run_alignment_qc.validation._process_tile',return_value=(rows,[])):
                quality_process_tile('tile',params)
            params['eval']['sweep']['x']['range_px']=8
            with patch('src.utils.run_alignment_qc.validation._process_tile',return_value=(rows,[])) as run:
                quality_process_tile('tile',params)
                self.assertEqual(run.call_count,1)

if __name__=='__main__':
    unittest.main()
