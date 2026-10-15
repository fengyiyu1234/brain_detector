"""Regression tests for QC integrity and independent GT matching."""
from contextlib import closing
import csv
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from src.utils.pipeline_qc import PipelineQC, mapped_path, STAGES
from src.utils.compare_qc_gt import evaluate, maximum_matching


def table(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fixture(root):
    channel = "SomaX"
    config = {"channels_routing": [{"id": channel, "type": "soma", "model": "yolo", "dir_key": "image_dir"}],
              "paths": {"image_dir": str(root / "images")}, "pipeline_mode": "pre_align",
              "stitching_reference_channel": channel, "pre_align_params": {"reference_channel": channel},
              "detection_params": {"tILESIZE": 100, "xy_resolution_um": .5, "z_resolution_um": 8},
              "z_linker": {"soma": {"min_z_layers": 1, "max_cell_z_span": 5}},
              "tile_position_params": {"enabled": False}}
    (root / "runtime_config.json").write_text(json.dumps(config))
    row = dict(x1=10,y1=10,x2=20,y2=20,z=1,score=.9,mean=100,
               **{"class":"neuron_SomaX"},detection_id="d1",tile_name="tile",slice_name="slice1")
    row2 = {**row,"detection_id":"d2","z":2,"slice_name":"slice2"}
    for folder in ("1_2d_raw","2_2d_filtered","4_2d_filtered"):
        table(root / folder / "tile_SomaX_result.csv",[row,row2],list(row))
    raw=root / "1_2d_raw/tile_SomaX_result.csv"
    (raw.parent / "tile_SomaX_inputs.json").write_text(json.dumps({"output_size":raw.stat().st_size,
        "rows_written":2,"images":[{"status":"processed"}]}))
    table(root / "2_2d_filtered/tile_SomaX_rejected.csv",[],[*row,"rejection_reason"])
    aligned=root / "3_2d_aligned"
    aligned.mkdir()
    (aligned / "tile_offsets.json").write_text(json.dumps({channel:{"dx":0,"dy":0,"dz":0}}))
    xml=root / "frame.xml"
    xml.write_text('<TeraStitcher><dimensions stack_slices="10"/><STACKS><Stack DIR_NAME="tile" ABS_H="0" ABS_V="0" ABS_D="0"/></STACKS></TeraStitcher>')
    table(root / "5_2d_global/SomaX_2d_global.csv",[row,row2],list(row))
    table(root / "5_2d_global/SomaX_stitch_decisions.csv",[],["detection_id","other_detection_id","reason"])
    track={k:v for k,v in row.items() if k!="detection_id"}
    track.update(track_id="t1",member_count=2,z_min=1,z_max=2)
    table(root / "6_3d_global/SomaX_3d_tracked.csv",[track],list(track))
    members=[{**r,"track_id":"t1"} for r in (row,row2)]
    table(root / "6_3d_global/SomaX_track_members.csv",members,list(members[0]))
    table(root / "6_3d_global/SomaX_track_rejections.csv",[],["member_detection_ids"])
    coloc={k:v for k,v in row.items() if k!="detection_id"}
    coloc.update(coloc_id="c1",soma_positive_channels='["SomaX"]',tf_positive_channels="[]",
                 soma_positive_count=1,tf_positive_count=0)
    table(root / "7_colocalization/coloc_result.csv",[coloc],list(coloc))
    table(root / "7_colocalization/coloc_source_3d.csv",
          [dict(coloc_id="c1",channel="SomaX",track_id="t1",source_track=0,member_count=2)],
          ["coloc_id","channel","track_id","source_track","member_count"])
    table(root / "7_colocalization/coloc_source_boxes.csv",
          [dict(coloc_id="c1",source_track=0,channel="SomaX",track_id="t1",detection_id=r["detection_id"],z=r["z"]) for r in (row,row2)],
          ["coloc_id","source_track","channel","track_id","detection_id","z"])
    table(root / "7_colocalization/coloc_match_evidence.csv",[],["match_type","coloc_id","channel"])
    table(root / "7_colocalization/coloc_decisions.csv",[],["reason"])
    centroid=dict(coloc_id="c1",cx=15,cy=15,z=1,cx_um=7.5,cy_um=7.5,cz_um=0)
    table(root / "7_colocalization/cell_centroids/ob_neuron_SomaX.csv",[centroid],list(centroid))
    return {"id":"custom","result_dir":str(root),"xml":str(xml)}


class PipelineQCTest(unittest.TestCase):
    def run_qc(self, root, sample=None, options=None):
        with closing(sqlite3.connect(":memory:")) as db:
            return PipelineQC(sample or {"id":"test","result_dir":str(root),"xml":str(root/"frame.xml")},
                              options or {}, root/"qc", db).run()

    def test_complete_arbitrary_channel(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            report=self.run_qc(root,fixture(root))
            self.assertEqual(report["status"],"PASS", report["findings"])
            self.assertEqual(report["canvas"]["width"],100)
            self.assertEqual(report["statistics"]["stitch_counts:SomaX"]["eligible"],2)
            self.assertTrue((root/"qc/review_candidates.csv").is_file())

    def test_filter_count_equal_but_wrong_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);fixture(root)
            path=root/"2_2d_filtered/tile_SomaX_result.csv"
            path.write_text(path.read_text().replace("d2","invented"))
            report=self.run_qc(root)
            self.assertEqual(report["status"],"FAIL")
            self.assertIn("id_partition",{f["code"] for f in report["findings"]})

    def test_track_count_extent_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);fixture(root)
            path=root/"6_3d_global/SomaX_track_members.csv"
            lines=path.read_text().splitlines()
            path.write_text("\n".join(lines[:-1])+"\n")
            report=self.run_qc(root)
            self.assertIn("track_member_integrity",{f["code"] for f in report["findings"]})
            self.assertIn("detection_partition",{f["code"] for f in report["findings"]})

    def test_input_failure_and_frame_offset(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);fixture(root)
            path=root/"1_2d_raw/tile_SomaX_inputs.json"
            data=json.loads(path.read_text());data["images"].append({"status":"unreadable"})
            path.write_text(json.dumps(data))
            (root/"3_2d_aligned/tile_offsets.json").write_text('{"SomaX":{"dx":1,"dy":0,"dz":0}}')
            report=self.run_qc(root)
            codes={f["code"] for f in report["findings"]}
            self.assertTrue({"image_read_failures","invalid_offsets"}<=codes)

    def test_duplicate_global_and_wrong_export_units(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);fixture(root)
            path=root/"5_2d_global/SomaX_2d_global.csv"
            with path.open("a") as f:f.write(path.read_text().splitlines()[1]+"\n")
            path=root/"7_colocalization/cell_centroids/ob_neuron_SomaX.csv"
            path.write_text(path.read_text().replace("7.5","75"))
            report=self.run_qc(root)
            codes={f["code"] for f in report["findings"]}
            self.assertTrue({"duplicate","centroid_count_or_units"}<=codes)

    def test_missing_output_is_not_empty_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);fixture(root)
            (root/"6_3d_global/SomaX_3d_tracked.csv").unlink()
            report=self.run_qc(root)
            self.assertEqual(report["status"],"FAIL")

    def test_path_mapping_boundary(self):
        self.assertEqual(str(mapped_path("/cluster/data/a", {"/cluster/data":"C:/data"})).replace("\\","/"),"C:/data/a")
        self.assertNotIn("C:/data",str(mapped_path("/cluster/database/a",{"/cluster/data":"C:/data"})))


class GroundTruthTest(unittest.TestCase):
    def setup_case(self, root, pred, gt, **options):
        fields=["roi_id","id","channel","class","cx","cy","cz"]
        table(root/"pred.csv",pred,fields);table(root/"gt.csv",gt,fields)
        return dict(mode="points3d",predictions=str(root/"pred.csv"),ground_truth=str(root/"gt.csv"),
                    coordinate_frame="tile-local, z one-based",voxel_size_um=[1,1,8],
                    distance_threshold_um=2,rois=[{"id":"r","specimen_id":"brain1","exhaustive":True}],**options)

    @staticmethod
    def row(identity,x=0,z=1):
        return dict(roi_id="r",id=identity,channel="A",**{"class":"cell"},cx=x,cy=0,cz=z)

    def test_matching_reassigns_greedy_error(self):
        self.assertEqual(len(maximum_matching([[0,1],[0]],2)),2)

    def test_duplicate_predictions_one_tp_one_fp(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            config=self.setup_case(root,[self.row("p1"),self.row("p2")],[self.row("g")])
            report,_,_=evaluate(config)
            self.assertEqual(report["micro"]["tp"],1)
            self.assertEqual(report["micro"]["fp"],1)
            self.assertEqual(report["micro"]["recall"],1)

    def test_anisotropic_z_and_empty_roi(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            config=self.setup_case(root,[self.row("p",z=2)],[self.row("g")])
            config["rois"].append({"id":"empty","specimen_id":"brain1","exhaustive":True})
            report,rows,_=evaluate(config)
            self.assertEqual(report["micro"]["tp"],0)
            self.assertEqual(report["micro"]["fn"],1)
            self.assertIsNone(rows[1]["f1"])

    def test_incomplete_annotations_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);config=self.setup_case(root,[],[])
            config["rois"][0]["exhaustive"]=False
            with self.assertRaisesRegex(ValueError,"exhaustive"):
                evaluate(config)

    def test_unknown_roi_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            bad={**self.row("p"),"roi_id":"not_annotated"}
            config=self.setup_case(root,[bad],[])
            with self.assertRaisesRegex(ValueError,"unknown ROI"):evaluate(config)

    def test_landmarks_use_correspondence_not_nearest_neighbor(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            config=self.setup_case(root,[self.row("same",x=20)],[self.row("same")])
            config["mode"]="landmarks"
            report,_,_=evaluate(config)
            self.assertEqual(report["matched_distance_um"]["mean"],20)
            self.assertNotIn("micro",report)

    def test_class_error_on_matched_cells(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            p={**self.row("p"),"class":"wrong"}
            config=self.setup_case(root,[p],[self.row("g")],class_aware=False)
            report,_,_=evaluate(config)
            self.assertEqual(report["micro"]["tp"],1)
            self.assertEqual(report["class_confusion_matched_only"][0]["prediction"],"wrong")





class AdditionalQCTest(unittest.TestCase):
    def test_source_box_unknown_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);sample=fixture(root)
            path=root/"7_colocalization/coloc_source_boxes.csv"
            path.write_text(path.read_text().replace("d2","invented"))
            with closing(sqlite3.connect(":memory:")) as db:
                report=PipelineQC(sample,{},root/"qc",db).run()
            self.assertIn("source_box_integrity",{f["code"] for f in report["findings"]})

    def test_disabled_relations_are_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);sample=fixture(root)
            with closing(sqlite3.connect(":memory:")) as db:
                report=PipelineQC(sample,{"relational_checks":False},root/"qc",db).run()
            self.assertIn("relations_disabled",{f["code"] for f in report["findings"]})
            self.assertEqual(report["status"],"PASS")

    def test_post_align_uses_final_filtered(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);sample=fixture(root)
            config=json.loads((root/"runtime_config.json").read_text())
            config["pipeline_mode"]="post_align"
            (root/"runtime_config.json").write_text(json.dumps(config))
            raw=root/"1_2d_raw/tile_SomaX_result.csv"
            with raw.open(encoding="utf-8") as handle:
                fields = [*next(csv.DictReader(handle)), "rejection_reason"]
            table(root/"4_2d_filtered/tile_SomaX_rejected.csv",[],fields)
            with closing(sqlite3.connect(":memory:")) as db:
                report=PipelineQC(sample,{},root/"qc",db).run()
            self.assertEqual(report["status"],"PASS",report["findings"])

    def test_box_iou_and_class_threshold(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            fields=["roi_id","id","channel","class","x1","y1","x2","y2","z"]
            row=dict(roi_id="r",id="one",channel="GFP",**{"class":"neuron"},x1=0,y1=0,x2=10,y2=10,z=1)
            table(root/"gt.csv",[row],fields)
            table(root/"pred.csv",[{**row,"id":"p","x1":5,"x2":15}],fields)
            config=dict(mode="boxes2d",coordinate_frame="raw",predictions=str(root/"pred.csv"),
                        ground_truth=str(root/"gt.csv"),iou_threshold=.5,
                        rois=[dict(id="r",specimen_id="s",exhaustive=True)])
            report,_,_=evaluate(config)
            self.assertEqual(report["micro"]["tp"],0)
            config["iou_threshold"]=.3
            report,_,_=evaluate(config)
            self.assertEqual(report["micro"]["tp"],1)

    def test_two_samples_continue_after_missing_runtime(self):
        from src.utils.pipeline_qc import main
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);valid=root/"valid";valid.mkdir()
            sample=fixture(valid)
            cfg={"samples":[{"id":"missing","result_dir":str(root/"missing"),"output_dir":str(root/"errors")},sample]}
            config=root/"qc.json";config.write_text(json.dumps(cfg))
            self.assertEqual(main(["--config",str(config)]),1)
            self.assertEqual(json.loads((root/"errors/qc_report.json").read_text())["status"],"FAIL")
            self.assertTrue((valid/"pipeline_qc/qc_report.json").is_file())

    def test_maximum_matching_small_exhaustive(self):
        import itertools
        def brute(edges,index=0,used=frozenset()):
            if index==len(edges):return 0
            return max([brute(edges,index+1,used)]+[1+brute(edges,index+1,used|{g}) for g in edges[index] if g not in used])
        for bits in itertools.product((False,True),repeat=9):
            edges=[[g for g in range(3) if bits[p*3+g]] for p in range(3)]
            pairs=maximum_matching(edges,3)
            self.assertEqual(len(pairs),brute(edges))
            self.assertEqual(len({g for _,g in pairs}),len(pairs))


if __name__=="__main__":
    unittest.main()
