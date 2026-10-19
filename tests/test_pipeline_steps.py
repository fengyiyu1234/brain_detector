"""Stop boundaries reject typos and disabled stages before doing work."""
import unittest
from src.core.pipeline_steps import STEPS, resolve_stop_step, stop_after


class PipelineStepsTests(unittest.TestCase):
    def config(self):
        return {'pipeline_mode':'pre_align', 'channels_routing':[{'id':'GFP','double_exposure':True}],
                'tile_position_params':{'enabled':True},'direct_stitching':{'enabled':True},
                'qc':{'alignment':{'enabled':True}}}

    def test_all_named_and_numeric_boundaries(self):
        for name,number in STEPS.items():
            for value in (name,number,float(number)):
                config=self.config();config['stop_after_step']=value
                self.assertEqual(resolve_stop_step(config),name)
                with self.assertRaises(SystemExit) as exc:
                    stop_after(config,name)
                self.assertEqual(exc.exception.code,0)
                stop_after(config,'different_step')
        self.assertIsNone(resolve_stop_step({}))

    def test_invalid_disabled_and_removed_options(self):
        for value in (True,False,0,'typo','2.7'):
            with self.assertRaises(ValueError):resolve_stop_step({'stop_after_step':value})
        for step in ('alignment','alignment_qc','fusion','tile_positions','image_stitching'):
            with self.assertRaisesRegex(ValueError,'disabled'):
                resolve_stop_step({'stop_after_step':step})
        for key in ('stop_after_detection','stop_before_stitching'):
            with self.assertRaisesRegex(ValueError,'Removed'):
                resolve_stop_step({key:False})
