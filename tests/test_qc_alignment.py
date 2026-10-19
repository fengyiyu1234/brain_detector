"""Scientific and end-to-end checks for fixed-landmark alignment QC."""
import copy
import json
import tempfile
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import tifffile
from src.core.point_cloud_aligner import resolve_align_settings
from src.qc.alignment import distance_curve, landmark_rows, run, scan_points, resolve_options
from src.qc.alignment_features import fixed_pairs, volume_features
from src.qc.run_qc import run_alignment_stage
from src.qc.common import save_json


def synthetic_sample(root):
    routing = [{'id':cid, 'type':kind, 'dir_key':cid} for cid,kind in
               [('GFP','soma'),('RFP','soma'),('Sox9','tf'),('Olig2','tf')]]
    config = dict(pipeline_mode='pre_align', channels_routing=routing,
        paths={cid:str(root/cid) for cid in ['GFP','RFP','Sox9','Olig2']},
        pre_align_params={'reference_channel':'GFP', 'sample_z_start_slice':20, 'sample_z_count':4},
        detection_params={'tILESIZE':64, 'xy_resolution_um':.65, 'z_resolution_um':8})
    settings = resolve_align_settings(config,routing)
    for folder in ['1_2d_raw','2_2d_filtered','3_2d_aligned']:
        (root/folder).mkdir()
    (root/'runtime_config.json').write_text(json.dumps(config))
    (root/'3_2d_aligned/_align_settings.json').write_text(json.dumps(settings))
    (root/'3_2d_aligned/tile_offsets.json').write_text(json.dumps({cid:{'dx':0,'dy':0,'dz':0} for cid in config['paths']}))
    (root/'3_2d_aligned/tile_measured_offsets.json').write_text(json.dumps({'z_center':22,'sample_window':{'z_start':20,'z_stop_exclusive':24}}))
    y,x = np.mgrid[:64,:64]
    volume = (100+6000*np.exp(-((x-20)**2+(y-20)**2)/12)).astype(np.uint16)
    for ch in routing:
        cid=ch['id']; directory=root/cid/'tile'; directory.mkdir(parents=True)
        for z in range(1,41):
            tifffile.imwrite(directory/f'{z:03d}.tif',volume)
        half=8 if ch['type']=='soma' else 4
        records=[dict(x1=20-half,y1=20-half,x2=20+half,y2=20+half,z=z,
                      score=.9,mean=100,**{'class':'neuron' if ch['type']=='soma' else 'nucleus'}) for z in (5,6)]
        pd.DataFrame(records).to_csv(root/f'2_2d_filtered/tile_{cid}_result.csv',index=False)
        (root/f'1_2d_raw/tile_{cid}_inputs.json').write_text(json.dumps({'images':[{'status':'processed'} for _ in range(40)]}))
    return config,settings


class AlignmentQCTests(unittest.TestCase):
    def test_signed_physical_distance_and_historical_third_center(self):
        self.assertEqual(np.argmin(distance_curve([[1.3,0,0]],0,[-2.6,-1.3,0,1.3])),1)
        self.assertEqual(distance_curve([[0,0,8]],2,[-8,0,8])[0],0)
        pairs={'Sox9':[{'soma_points':[[100,0,0],[100,0,0],[0,0,1]],
                         'nuclear_points':[[100,0,0],[100,0,0],[2,0,1]]}]}
        rows=landmark_rows('tile',pairs,[('x',-2),('x',0),('x',2)],
                           {'xy_resolution_um':.65,'z_resolution_um':8},10)
        self.assertEqual([r['value'] for r in rows],[0,1.3,2.6])
        self.assertTrue(all(r['status']=='low_pairs' for r in rows))

    def test_exact_scan_endpoints_and_empty_pairs(self):
        points=scan_points({'xy_range_um':20,'z_range_um':40},{'xy_resolution_um':.65,'z_resolution_um':8})
        self.assertEqual(len(points),137)
        self.assertAlmostEqual(min(d for a,d in points if a=='x')*.65,-20)
        rows=landmark_rows('tile',{'Sox9':[]},points,{'xy_resolution_um':.65,'z_resolution_um':8},10)
        self.assertTrue(all(np.isnan(r['value']) and r['status']=='no_pairs' for r in rows))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'empty.json'
            save_json(path,rows)
            def reject_constant(value):
                raise ValueError('nonstandard JSON: '+value)
            parsed=json.loads(path.read_text(),parse_constant=reject_constant)
            self.assertTrue(all(row['value'] is None for row in parsed))


    def test_ambiguity_and_bright_center_definition(self):
        soma={'per_z_boxes':{4:[0,0,10,10]}}
        nucleus={'per_z_boxes':{4:[3,3,7,7]}}
        self.assertEqual(fixed_pairs([soma],[nucleus],np.zeros(3),np.zeros(3))[0],[(0,0)])
        self.assertFalse(fixed_pairs([soma],[nucleus,nucleus],np.zeros(3),np.zeros(3))[0])
        y,x=np.mgrid[:31,:31]
        v=(100+2000*np.exp(-((x-15)**2+(y-15)**2)/18))[None].astype(np.uint16)
        f=volume_features(v,np.ones(v.shape,bool),np.array([.5,.5,7]),True)
        np.testing.assert_allclose(f[-1],[15.5,15.5,7],atol=.15)
        self.assertIsNone(volume_features(np.ones((1,10,10)),np.ones((1,10,10),bool),np.zeros(3),False))

    def test_disabled_hook_and_failure_policy(self):
        with patch('src.qc.alignment.run',side_effect=AssertionError('must not run')):
            self.assertEqual(run_alignment_stage({},'unused',[])['status'],'disabled')
        with tempfile.TemporaryDirectory() as tmp:
            cfg={'pipeline_mode':'post_align','qc':{'alignment':{'enabled':True}}}
            self.assertEqual(run_alignment_stage(cfg,tmp,[])['status'],'failed')
            cfg['qc']['alignment']['fail_on_error']=True
            with self.assertRaisesRegex(ValueError,'pre_align'):
                run_alignment_stage(cfg,tmp,[])

    def test_fresh_images_end_to_end_resume_and_image_invalidation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); config,settings=synthetic_sample(root)
            options={'enabled':True,'xy_range_um':1.3,'z_range_um':8,'workers':1,'formats':['png'],
                     'sampling':{'min_pairs':1,'max_attempts':40}}
            report=run_alignment_stage({**config,'qc':{'alignment':options}},str(root),['tile'])
            self.assertEqual(report['status'],'complete',report['failures'])
            record=report['completed'][0]
            self.assertEqual(record['counts']['Sox9']['common_pairs'],1)
            self.assertEqual(record['counts']['Olig2']['common_pairs'],1)
            self.assertTrue(record['region'][5]<20 or record['region'][4]>24)
            checkpoint=root/'qc/alignment/tiles/tile.json'
            before=checkpoint.stat().st_mtime_ns
            self.assertEqual(run(str(root),options=options)['status'],'complete')
            self.assertEqual(checkpoint.stat().st_mtime_ns,before)
            image=next(iter(record['images']))
            tifffile.imwrite(image,np.full((64,64),100,np.uint16))
            self.assertEqual(run(str(root),options=options)['status'],'complete')
            self.assertNotEqual(checkpoint.stat().st_mtime_ns,before)
            self.assertTrue((root/'qc/alignment/summary.csv').is_file())
            # Exercise spawn-based worker execution with two independent tile checkpoints.
            for folder in ('1_2d_raw', '2_2d_filtered', '3_2d_aligned'):
                for source in (root/folder).glob('tile_*'):
                    shutil.copyfile(source, source.with_name(source.name.replace('tile_', 'tile2_', 1)))
            for cid in config['paths']:
                shutil.copytree(root/cid/'tile', root/cid/'tile2')
            parallel=run(str(root),options={**options,'workers':2,'output_dir':'qc/parallel'})
            self.assertEqual(parallel['status'],'complete',parallel['failures'])
            self.assertEqual(len(parallel['completed']),2)

            with self.assertRaisesRegex(ValueError,'protected'):
                run(str(root),options={**options,'output_dir':'2_2d_filtered/bad'})
            with self.assertRaisesRegex(ValueError,'whole number'):
                resolve_options({'z_range_um':41},config['channels_routing'],settings)
