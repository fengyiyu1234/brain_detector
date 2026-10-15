"""Intensity QC geometry, fixed cohort, and baseline regression tests."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from src.utils.alignment_intensity_qc import rectangle_means, common_cohort, relative_shift, evaluate_tile, local_box_means, scan_specifications, summarize_curves

class IntensityQcTests(unittest.TestCase):
    def test_integral_matches_pixels_and_rejects_clipping(self):
        image=np.arange(48).reshape(6,8)
        integral=np.pad(image.astype(float).cumsum(0).cumsum(1),((1,0),(1,0)))
        got=rectangle_means(integral,np.array([[1,2,4,5],[0,0,8,6]],dtype=float))
        np.testing.assert_allclose(got,[image[2:5,1:4].mean(),image.mean()])
        with self.assertRaises(ValueError):
            rectangle_means(integral,np.array([[-1,0,3,4]],dtype=float))

    def test_common_cohort_covers_endpoints_baseline_and_fit_guard(self):
        boxes=np.array([[2,2,4,4,4],[0,0,2,2,4],[2,2,4,4,9],[2,2,4,4,1]],dtype=float)
        keep=common_cohort(boxes,[(0,0,0),(-1,0,0),(1,0,1)],8,8,10,(8,10),0)
        np.testing.assert_array_equal(keep,[True,False,False,True])
        keep=common_cohort(boxes,[(0,0,-2)],8,8,10)
        self.assertFalse(keep[-1])

    def test_relative_offset_sign_and_frame_rebase_invariance(self):
        nuclear=dict(dx=3,dy=-2,dz=4)
        soma=dict(dx=-1,dy=1,dz=2)
        self.assertEqual(relative_shift(nuclear,soma),(4,-3,2))
        self.assertEqual(relative_shift({k:v+7 for k,v in nuclear.items()},
                                        {k:v+7 for k,v in soma.items()}),(4,-3,2))

    def test_real_tiff_curve_fixed_cohort_and_old_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); result=root/'results'; qc=root/'qc'; tile='tile'
            (result/'3_2d_aligned').mkdir(parents=True)
            (result/'2_2d_filtered').mkdir()
            (result/'3_2d_aligned'/'_align_settings.json').write_text(json.dumps(dict(xy_resolution_um=.65,z_resolution_um=8)))
            olddir=result/'old_results'/'3_2d_aligned'; olddir.mkdir(parents=True)
            (qc/'tile_curves').mkdir(parents=True)
            offsets={cid:dict(dx=0,dy=0,dz=0) for cid in ('GFP','RFP','Sox9','Olig2')}
            offsets['Sox9']['dx']=1; offsets['Olig2']['dx']=1
            (result/'3_2d_aligned'/f'{tile}_offsets.json').write_text(json.dumps(offsets))
            (result/'3_2d_aligned'/f'{tile}_measured_offsets.json').write_text(json.dumps({'sample_window':{'z_start':8,'z_stop_exclusive':10}}))
            old={cid:dict(dx=0,dy=0,dz=0) for cid in offsets}
            (olddir/f'{tile}_offsets.json').write_text(json.dumps(old))
            rows=[]
            for nuc in ('Sox9','Olig2'):
                pd.DataFrame([[14,14,16,16,3]],columns=['x1','y1','x2','y2','z']).to_csv(result/'2_2d_filtered'/f'{tile}_{nuc}_result.csv',index=False)
                for axis in 'xyz':
                    for delta in (-1,0,1):
                        rows.append(dict(region_id=0,x0=1,x1=30,y0=1,y1=30,z0=3,z1=4,actual_z_guard_slices=0,
                                         channel=nuc,axis=axis,delta=delta,delta_um=delta))
            pd.DataFrame(rows).to_csv(qc/'tile_curves'/f'{tile}_curves.csv',index=False)
            paths={}; routing=[]
            for soma in ('GFP','RFP'):
                folder=root/soma/tile; folder.mkdir(parents=True)
                image=np.ones((32,32),dtype=np.uint16); image[14:18,15:19]=1000
                for z in range(1,11):
                    tifffile.imwrite(folder/f'{z:04d}.tif',image)
                paths[soma]=str(root/soma)
                routing.append(dict(id=soma,dir_key=soma))
            routing += [dict(id=nuc,dir_key=nuc) for nuc in ('Sox9','Olig2')]
            config=root/'config.json';config.write_text(json.dumps(dict(paths=paths,channels_routing=routing)))
            records,statuses=evaluate_tile((tile,str(result),str(qc),str(config),2000,42))
            df=pd.DataFrame(records)
            self.assertTrue(df.n_boxes.eq(1).all())
            self.assertEqual(len(statuses),4)
            zero=df.loc[df.axis.eq('x')&df.delta.eq(0)].iloc[0]
            self.assertAlmostEqual(zero.raw_mean_intensity,1000)
            self.assertAlmostEqual(zero.unaligned_raw_mean,500.5)
            self.assertAlmostEqual(zero.old_raw_mean,500.5)
            self.assertGreater(zero.intensity_enrichment,zero.unaligned_enrichment)
            self.assertAlmostEqual(zero.raw_ring_mean_intensity,1)
            self.assertGreater(zero.local_contrast,zero.unaligned_local_contrast)
            self.assertAlmostEqual(zero.old_local_contrast,zero.unaligned_local_contrast)
            self.assertEqual(df.loc[df.axis.eq('x'),'delta'].unique().tolist(),list(range(-8,9)))
            self.assertEqual(df.loc[df.axis.eq('z'),'delta'].unique().tolist(),list(range(-2,3)))


    def test_ring_excludes_core_and_gap(self):
        image=np.full((12,12),2.,dtype=float)
        image[4:8,4:8]=99
        image[5:7,5:7]=10
        integral=np.pad(image.cumsum(0).cumsum(1),((1,0),(1,0)))
        core,ring=local_box_means(integral,np.array([[5,5,7,7]],dtype=float),1,1)
        np.testing.assert_allclose(core,[10])
        np.testing.assert_allclose(ring,[2])

    def test_uniform_signal_has_zero_local_difference_and_dark_nucleus_is_negative(self):
        image=np.full((12,12),10.,dtype=float)
        boxes=np.array([[5,5,7,7]],dtype=float)
        integral=np.pad(image.cumsum(0).cumsum(1),((1,0),(1,0)))
        core,ring=local_box_means(integral,boxes,1,1)
        np.testing.assert_allclose(core-ring,[0])
        image[5:7,5:7]=2
        integral=np.pad(image.cumsum(0).cumsum(1),((1,0),(1,0)))
        core,ring=local_box_means(integral,boxes,1,1)
        np.testing.assert_allclose(core-ring,[-8])

    def test_ring_guard_preserves_a_fixed_cohort_without_clipping(self):
        boxes=np.array([[3,3,5,5,3],[6,6,8,8,3]],dtype=float)
        keep=common_cohort(boxes,[(-2,0,0),(2,0,0),(0,0,0)],16,16,10,xy_margin=2)
        np.testing.assert_array_equal(keep,[False,True])
        integral=np.zeros((17,17))
        with self.assertRaises(ValueError):
            local_box_means(integral,np.array([[0,0,2,2]],dtype=float),2,1)
        for width,gap in ((0,1),(1,-1),(True,1),(1,1.5)):
            with self.assertRaises(ValueError):
                local_box_means(integral,boxes[:,:4],width,gap)

    def test_fine_scan_is_symmetric_contains_zero_and_honors_custom_ranges(self):
        settings=dict(xy_resolution_um=.65,z_resolution_um=8)
        df=scan_specifications(settings)
        self.assertEqual(df.loc[df.axis.eq('x'),'delta'].tolist(),list(range(-8,9)))
        self.assertEqual(df.loc[df.axis.eq('z'),'delta'].tolist(),list(range(-2,3)))
        self.assertAlmostEqual(df.loc[df.axis.eq('x') & df.delta.eq(1),'delta_um'].iloc[0],.65)
        options={'sweep':{'x':{'range_px':5,'step_px':2},'y':{'range_px':0,'step_px':1},'z':{'range_slices':1,'step_slices':1}}}
        df=scan_specifications(settings,options)
        self.assertEqual(df.loc[df.axis.eq('x'),'delta'].tolist(),[-5,-3,-1,0,1,3,5])
        options['sweep']['x']['step_px']=0
        with self.assertRaises(ValueError):
            scan_specifications(settings,options)

    def test_contrast_summary_reports_signed_peak_and_trough(self):
        df=pd.DataFrame([dict(tile='tile',axis='x',delta=d,delta_um=d*.65,
                     intensity_enrichment=1,unaligned_enrichment=.8,old_enrichment=.9,n_boxes=10,
                     local_contrast=v,unaligned_local_contrast=.1,old_local_contrast=.2)
                     for d,v in [(-1,-.4),(0,-.2),(1,.3)]])
        row=summarize_curves(df,['tile','axis']).iloc[0]
        self.assertAlmostEqual(row.aligned_local_contrast,-.2)
        self.assertEqual(row.contrast_peak_delta,1)
        self.assertEqual(row.contrast_trough_delta,-1)
        self.assertAlmostEqual(row.contrast_trough,-.4)

if __name__=='__main__':
    unittest.main()
