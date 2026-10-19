"""Regression tests for automatic resume and independent stage checkpoints."""
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from scripts.run_inference import prepare_alignment_inputs, validate_reused_provenance
from src.core.channel_stage3 import stitch_and_link_channel
from src.core.provenance import file_stamp, output_manifest_valid, write_output_manifest
from src.core.resume import channel_sources, raw_detection_settings, resume_config
from src.core.result_layout import result_paths


def row(z=1):
    return dict(slice_name=f'z{z}', x1=5, y1=5, x2=15, y2=15, score=.9,
                mean=100, z=z, detection_id=f'd{z}', **{'class':'neuron'})


class AutomaticResumeTests(unittest.TestCase):
    def test_mixed_channels_route_without_copies(self):
        derived = result_paths('sample')
        config = {'pipeline_mode':'pre_align', 'channels_routing':[
            {'id':'GFP'}, {'id':'RFP','double_exposure':True}]}
        sources = channel_sources(config,derived)
        self.assertEqual(sources['GFP'], derived['pATH_ALIGN_OFFSETS'])
        self.assertEqual(sources['RFP'], derived['pATH_DET_FILTERED'])
        config['pipeline_mode']='post_align'
        self.assertEqual(set(channel_sources(config,derived).values()), {derived['pATH_DET_FILTERED']})

    def test_missing_snapshot_bootstrap_retains_unknown_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'1_2d_raw').mkdir()
            (root/'1_2d_raw/tile_GFP_result.csv').write_text('z\n1\n')
            current={'paths':{'pATHRESULT':tmp}, 'pipeline_mode':'pre_align'}
            old=resume_config(tmp,current,'current.json')
            self.assertFalse(old['resume_bootstrap']['raw_origin_verified'])
            self.assertTrue((root/'resume_bootstrap_config.json').is_file())
            self.assertNotIn('resume_bootstrap',current)

    def test_filter_and_geometry_options_do_not_change_raw_identity(self):
        old={'detection_params':{'yolo':{'conf_thresh':.2,'score_min':.3},
                               'xy_resolution_um':.65,'sTARTID':1}, 'z_linker':{'soma':{}}}
        new=copy.deepcopy(old)
        new['detection_params']['yolo']['score_min']=.7
        new['detection_params']['xy_resolution_um']=1
        new['detection_params']['sTARTID']=2
        self.assertEqual(raw_detection_settings(old),raw_detection_settings(new))
        new['detection_params']['yolo']['conf_thresh']=.5
        self.assertNotEqual(raw_detection_settings(old),raw_detection_settings(new))

    def test_legacy_filter_other_channel_settings_do_not_rewrite_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); raw=root/'1_2d_raw'; raw.mkdir(); filtered=root/'2_2d_filtered'
            ch={'id':'GFP','type':'soma','model':'yolo'}
            cfg={'channels_routing':[ch], 'detection_params':{'yolo':{'score_min':.5}}}
            pd.DataFrame([row()]).to_csv(raw/'tile_GFP_result.csv',index=False)
            prepare_alignment_inputs(cfg,['tile'],[ch],str(raw),str(filtered))
            target=filtered/'tile_GFP_result.csv'; before=target.stat().st_mtime_ns
            manifest=filtered/'tile_GFP_filter_manifest.json'
            payload=json.loads(manifest.read_text())
            payload['input_signature']['settings']['filter_settings']['params']['Sox9']={'bbox_max':23}
            payload['input_signature']['source']['path']='Z:/relocated/tile_GFP_result.csv'
            payload['input_signature']['source']['mtime_ns']=0
            manifest.write_text(json.dumps(payload))
            prepare_alignment_inputs(cfg,['tile'],[ch],str(raw),str(filtered))
            self.assertEqual(before,target.stat().st_mtime_ns)

    def test_changed_filter_does_not_overwrite_existing_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); raw=root/'1_2d_raw'; raw.mkdir(); filtered=root/'2_2d_filtered'
            ch={'id':'GFP','type':'soma','model':'yolo'}
            cfg={'channels_routing':[ch], 'detection_params':{'yolo':{'score_min':.5}}}
            pd.DataFrame([row()]).to_csv(raw/'tile_GFP_result.csv',index=False)
            prepare_alignment_inputs(cfg,['tile'],[ch],str(raw),str(filtered))
            (root/'3_2d_aligned').mkdir(); (root/'3_2d_aligned/tile_GFP_result.csv').write_text('reviewed')
            cfg['detection_params']['yolo']['score_min']=.8
            with self.assertRaisesRegex(ValueError,'saved alignment'):
                prepare_alignment_inputs(cfg,['tile'],[ch],str(raw),str(filtered))

    def test_link_failure_and_parameter_change_preserve_global_2d(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'source'; glob=root/'global'; tracks=root/'tracks'
            for p in (source,glob,tracks):p.mkdir()
            pd.DataFrame([row(1),row(2)]).to_csv(source/'tile_GFP_result.csv',index=False)
            geom={'num_tiles':1,'dir_dict':{'tile':(0,0)},'disp_mat_fin':np.array([[[0,0,0]]]),
                  'z_start':0,'Z':2,'H':100,'W':100,'tile_size':100}
            params={'min_z_layers':1,'iou_thresh':.3,'max_cell_z_span':5,'max_z_gap':0}
            def run():
                return stitch_and_link_channel({'id':'GFP','type':'soma'},str(source),str(glob),str(tracks),geom,params)
            with patch('src.core.channel_stage3.run_z_linker',side_effect=RuntimeError('interrupted')):
                with self.assertRaisesRegex(RuntimeError,'interrupted'):run()
            self.assertTrue((glob/'GFP_global_manifest.json').is_file())
            with patch('src.core.channel_stage3._stitch_channel',side_effect=AssertionError('restitch')):
                run()
                params['min_z_layers']=2
                run()
                # A timestamp-only touch must not invalidate either stage.
                os.utime(source/'tile_GFP_result.csv',None)
                with patch('src.core.channel_stage3.run_z_linker',side_effect=AssertionError('relink')):
                    run()
            self.assertEqual(len(pd.read_csv(tracks/'GFP_3d_tracked.csv')),1)
            # Adopt a complete old combined checkpoint without recomputing either stage.
            (glob/'GFP_global_manifest.json').unlink()
            (tracks/'GFP_tracks_manifest.json').unlink()
            with patch('src.core.channel_stage3._stitch_channel',side_effect=AssertionError('restitch')), \
                 patch('src.core.channel_stage3.run_z_linker',side_effect=AssertionError('relink')):
                run()

    def test_relocated_manifest_validates_current_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); output=root/'out.csv'; output.write_text('id\n1\n')
            manifest=root/'manifest.json'; signature={'input':file_stamp(output,with_hash=True)}
            write_output_manifest(manifest,[output],signature)
            data=json.loads(manifest.read_text()); stamp=data['outputs'].pop(str(output))
            data['outputs']['/old/mount/'+root.name+'/out.csv']=stamp
            manifest.write_text(json.dumps(data))
            self.assertTrue(output_manifest_valid(manifest,signature))
            output.write_text('id\n2\n')
            self.assertFalse(output_manifest_valid(manifest,signature))

class PipelineEntryResumeTests(unittest.TestCase):
    def test_real_entry_reuses_reviewed_alignment_and_completes_downstream(self):
        import subprocess
        import sys
        from src.core.detection_filter import resolve_filter_params
        from src.core.point_cloud_aligner import resolve_align_settings, save_align_settings
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); results = root / 'results'; images = root / 'images'
            tile = '000000_000000'; tile_dir = images / '000000' / tile
            tile_dir.mkdir(parents=True)
            for z in (1, 2): (tile_dir / f'z{z}.tif').write_bytes(b'not read during resume')
            derived = result_paths(str(results))
            raw = Path(derived['pATH_DET_RES']); raw.mkdir(parents=True)
            ch = {'id': 'GFP', 'type': 'soma', 'model': 'yolo', 'dir_key': 'gfp_dir'}
            config = {'paths': {'pATHRESULT': str(results), 'gfp_dir': str(images), 'pATHXML': None},
                      'channels_routing': [ch], 'models': {}, 'pipeline_mode': 'pre_align',
                      'stitching_reference_channel': 'GFP', 'pre_align_params': {'reference_channel': 'GFP'},
                      'stage3_n_workers': 1, 'detection_params': {'yolo': {'score_min': .5},
                      'tILESIZE': 100, 'allow_grid_fallback': True, 'generate_histograms': False},
                      'z_linker': {'soma': {'min_z_layers': 1}}, 'generate_analysis_report': True}
            pd.DataFrame([row(1), row(2)]).to_csv(raw / f'{tile}_GFP_result.csv', index=False)
            prepare_alignment_inputs(config, [tile], [ch], str(raw), derived['pATH_DET_PREALIGN_FILTERED'])
            aligned = Path(derived['pATH_ALIGN_OFFSETS']); aligned.mkdir()
            filtered = Path(derived['pATH_DET_PREALIGN_FILTERED']) / f'{tile}_GFP_result.csv'
            accepted = aligned / filtered.name; accepted.write_bytes(filtered.read_bytes())
            (aligned / f'{tile}_offsets.json').write_text(json.dumps({'GFP': {'dx': 0, 'dy': 0, 'dz': 0}}))
            (aligned / '_align_done.flag').touch()
            settings = resolve_align_settings(config, [ch]); settings['input_stage'] = 'filtered_2d_v1'
            settings['filter_params'] = {'GFP': resolve_filter_params(config, ch)}
            save_align_settings(str(aligned), settings)
            config_path = root / 'config.json'; config_path.write_text(json.dumps(config))
            initial = accepted.stat().st_mtime_ns
            command = [sys.executable, 'scripts/run_inference.py', '--config', str(config_path)]
            env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
            # Every requested stop must return successfully before later products are made.
            for step in ('setup', 'detection', 'filtering', 'alignment', 'publish_filtered', 'z_linking', 'colocalization'):
                stopped = subprocess.run(command + ['--stop-after-step', step], capture_output=True,
                                         text=True, encoding='utf-8', env=env, timeout=120)
                self.assertEqual(stopped.returncode, 0, stopped.stdout + stopped.stderr)
                self.assertIn('stop_after_step=' + step, stopped.stdout + stopped.stderr)
                if step in ('setup', 'detection', 'filtering', 'alignment', 'publish_filtered'):
                    self.assertFalse((results / '5_2d_global/GFP_2d_global.csv').exists())
                if step != 'colocalization':
                    self.assertFalse((results / '7_colocalization/coloc_result.csv').exists())
                self.assertFalse(any((results / '7_colocalization/cell_centroids').glob('*.csv')))
            for iteration in range(2):
                result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', env=env, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(accepted.stat().st_mtime_ns, initial)
                self.assertFalse(Path(derived['pATH_DET_FILTERED']).exists())
                self.assertTrue((results / '7_colocalization/coloc_result.csv').is_file())
                if iteration == 0:
                    global_time = (results / '5_2d_global/GFP_2d_global.csv').stat().st_mtime_ns
                else:
                    self.assertEqual(global_time, (results / '5_2d_global/GFP_2d_global.csv').stat().st_mtime_ns)
                    self.assertIn('all pipeline checkpoints reused', result.stdout + result.stderr)
            self.assertTrue((results / 'runtime_config.json').is_file())
            self.assertEqual(len(list((results / 'run_configs').glob('*.json'))), 9)


class GeometryResumeTests(unittest.TestCase):
    def test_xml_failure_keeps_solver_checkpoint(self):
        from src.core.tile_position_pipeline import run_tile_position_stage
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); det=root/'det'; align=root/'align'; report=root/'positions'; images=root/'images'
            for directory in (det,align,report,images/'t0'/'t0'): directory.mkdir(parents=True)
            (det/'t0_GFP_result.csv').write_text('z\n1\n')
            (align/'t0_offsets.json').write_text('{}')
            solver=root/'solver.py'; solver.write_text('solver')
            xml=root/'xml.py'; xml.write_text('xml')
            cfg=root/'config.json'; cfg.write_text('{}')
            config={'paths':{'gfp_dir':str(images)},'stitching_reference_channel':'GFP',
                    'pre_align_params':{'reference_channel':'GFP'},'tile_position_params':{'enabled':True}}
            routing=[{'id':'GFP','type':'soma','dir_key':'gfp_dir'}]
            calls=[]
            def fake(command,check):
                calls.append(command[1])
                if command[1]==str(solver):
                    (report/'tile_positions.csv').write_text('tile\nt0\n')
                    (report/'solution.json').write_text('{}')
                elif len(calls)==2:
                    raise RuntimeError('XML interrupted')
                else:
                    (images/'xml_merging.xml').write_text('<root/>')
            def run():
                return run_tile_position_stage(config,str(cfg),str(root),str(det),str(align),
                                               str(report),['t0'],routing,str(solver),str(xml))
            with patch('src.core.tile_position_pipeline.subprocess.run',side_effect=fake):
                with self.assertRaisesRegex(RuntimeError,'XML interrupted'): run()
                self.assertTrue((report/'_solver_manifest.json').is_file())
                run()
            self.assertEqual(calls,[str(solver),str(xml),str(xml)])


if __name__=='__main__':unittest.main()
