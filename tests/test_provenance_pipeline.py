"""Provenance survives filtering, fusion, stitching, and Z linking."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.detection_filter import filter_detection_df
from src.core.provenance import (output_manifest_valid, provenance_columns,
                                 write_output_manifest, validate_raw_input_manifest,
                                 file_sha256)
from src.core.stitcher import (combine_predictions, fuse_dual_intensity_2d,
                               stitchDetection)
from src.core.z_linker import run_z_linker
from src.core.channel_stage3 import stitch_and_link_channel
from scripts.run_inference import write_fusion_summary


def raw_row(det_id, x1, y1=0, x2=None, y2=20, z=1, score=0.8):
    return {
        "slice_name": "tile_000080", "x1": x1, "y1": y1,
        "x2": x2 if x2 is not None else x1 + 20, "y2": y2,
        "class": "neuron", "score": score, "mean": 100, "z": z,
        "detection_id": det_id, "source_image": f"{det_id}.tif",
        "raw_slice_name": "tile_000080", "raw_z": z,
    }


class ProvenancePipelineTest(unittest.TestCase):
    def test_filter_rejections_keep_parent_id(self):
        rows = pd.DataFrame([raw_row("keep", 0), raw_row("drop", 30, x2=35)])
        rows = provenance_columns(rows, "tile", "GFP")
        filtered, stats, rejected = filter_detection_df(
            rows, {"bbox_min": 10}, return_stats=True, return_rejected=True)
        self.assertEqual(filtered.detection_id.tolist(), ["keep"])
        self.assertEqual(rejected.detection_id.tolist(), ["drop"])
        self.assertEqual(rejected.rejection_reason.tolist(), ["bbox"])
        self.assertEqual(stats["removed_total"], 1)

    def test_containment_rejection_records_winner_and_metric(self):
        rows = pd.DataFrame([
            raw_row('winner', 0, x2=20, score=0.9),
            raw_row('loser', 5, y1=5, x2=10, y2=10, score=0.8),
        ])
        kept, _, rejected = filter_detection_df(
            rows, {'nms_containment_thresh': 0.8},
            return_stats=True, return_rejected=True)
        self.assertEqual(kept.detection_id.tolist(), ['winner'])
        self.assertEqual(rejected.suppressed_by_detection_id.tolist(), ['winner'])
        self.assertEqual(rejected.suppression_iomin.tolist(), [1.0])
        self.assertEqual(rejected.suppression_threshold.tolist(), [0.8])

    def test_fusion_records_both_parents_and_winner(self):
        low = pd.DataFrame([raw_row("low", 0, score=0.9)])
        high = pd.DataFrame([raw_row("high", 1, score=0.5)])
        fused, n_low, n_high, n_fused = fuse_dual_intensity_2d(low, high, 0.3)
        self.assertEqual((n_low, n_high, n_fused), (1, 1, 1))
        row = fused.iloc[0]
        self.assertEqual(json.loads(row.parent_detection_ids), ["high", "low"])
        self.assertEqual(row.winner_detection_id, "low")
        self.assertGreater(row.fusion_iou, 0.8)
        self.assertEqual(row.source_image, "low.tif")

    def test_fusion_summary_includes_cached_tiles(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for tile, counts in (
                ('first', {'n_low': 2, 'n_high': 1, 'n_fused': 2}),
                ('second', {'n_low': 3, 'n_high': 2, 'n_fused': 4}),
            ):
                output = root / f'{tile}_GFP_result.csv'
                output.write_text('id\n', encoding='utf-8')
                write_output_manifest(
                    root / f'{tile}_GFP_fusion_manifest.json',
                    [output], {'tile': tile}, metadata=counts)
            summary = write_fusion_summary(
                ['first', 'second'], [{'id': 'GFP'}], str(root))
            rows = pd.read_csv(summary)
            self.assertEqual(rows.tile.tolist(), ['first', 'second', 'TOTAL'])
            total = rows.iloc[-1]
            self.assertEqual((total.n_low, total.n_high, total.n_fused,
                              total.n_matched), (5, 3, 6, 2))

    def test_global_id_and_track_members_are_exact(self):
        rows = [raw_row("det-z1", 5), raw_row("det-z2", 6, z=2)]
        predictions = [[np.empty((0, 8), dtype=object)
                        for _ in range(2)] for _ in range(2)]
        disp = np.array([[[0, 0, 0]]], dtype=object)
        metadata, trace, provenance = [], {}, {}
        combine_predictions(
            predictions, iter(rows), None, 0, 2, (0, 0), disp, (100, 100),
            metadata, "tile", tILESIZE=100, row_meta=trace,
            provenance_meta=provenance)
        dets = []
        for z in range(2):
            row = predictions[z][1][0]
            dets.append([*row, provenance[(z, 1)][0], "tile",
                         trace[(z, 1)][0][1]])
        _, tracks = run_z_linker(np.asarray(dets, dtype=object),
                                 min_z_layers=2, channel_id="GFP")
        self.assertEqual(len(tracks), 1)
        track = tracks[0]
        self.assertEqual([m["detection_id"] for m in track["member_detections"]],
                         ["det-z1", "det-z2"])
        self.assertEqual([m["tile_name"] for m in track["member_detections"]],
                         ["tile", "tile"])
        self.assertEqual(track["observed_x1"], 5.0)
        self.assertEqual(track["observed_x2"], 26.0)
        rejected = []
        run_z_linker(np.asarray(dets, dtype=object), min_z_layers=3,
                     channel_id="GFP", rejected_tracks=rejected)
        self.assertEqual(rejected[0]["reason"], "min_z_layers")

    def test_stage3_rebuilds_when_source_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'filtered'
            global_dir = root / 'global'
            tracks_dir = root / 'tracks'
            for directory in (source, global_dir, tracks_dir):
                directory.mkdir()
            rows = [raw_row('one', 5), raw_row('two', 6, z=2)]
            source_csv = source / 'tile_GFP_result.csv'
            pd.DataFrame(rows).to_csv(source_csv, index=False)
            geom = {
                'num_tiles': 1, 'dir_dict': {'tile': (0, 0)},
                'disp_mat_fin': np.array([[[0, 0, 0]]]),
                'z_start': 0, 'Z': 2, 'H': 100, 'W': 100,
                'tile_size': 100,
            }
            params = dict(min_z_layers=2, iou_thresh=0.3,
                          max_cell_z_span=5, max_z_gap=0)
            run = lambda: stitch_and_link_channel(
                {'id': 'GFP', 'type': 'soma'}, str(source),
                str(global_dir), str(tracks_dir), geom, params)
            run()
            members = tracks_dir / 'GFP_track_members.csv'
            manifest = tracks_dir / 'GFP_stage3_manifest.json'
            self.assertTrue(manifest.is_file())
            self.assertEqual(pd.read_csv(members).detection_id.tolist(), ['one', 'two'])
            self.assertEqual(pd.read_csv(members)['class'].tolist(),
                             ['neuron_GFP', 'neuron_GFP'])
            summary = pd.read_csv(tracks_dir / 'GFP_3d_tracked.csv')
            self.assertEqual(summary.best_detection_id.tolist(), ['one'])
            self.assertEqual(summary.z_min.tolist(), [1])
            self.assertEqual(summary.z_max.tolist(), [2])
            rows[0]['score'] = 0.6
            pd.DataFrame(rows).to_csv(source_csv, index=False)
            run()
            global_csv = pd.read_csv(global_dir / 'GFP_2d_global.csv')
            self.assertEqual(global_csv.score.tolist(), [0.6, 0.8])
            self.assertEqual(len(pd.read_csv(members)), 2)

    def test_yolo_candidate_trace_identifies_suppressed_winner(self):
        boxes = np.asarray([[0, 0, 10, 10, 0.9],
                            [1, 1, 11, 11, 0.8],
                            [40, 40, 50, 50, 0.7]])
        kept, indices, suppressed = stitchDetection(boxes, return_trace=True)
        self.assertEqual(indices, [0, 2])
        self.assertEqual(len(kept), 2)
        self.assertEqual(suppressed[1][0], 0)
        self.assertGreater(suppressed[1][1], 0.6)

    def test_raw_manifest_detects_changed_tiff_and_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / 'image.tif'
            image.write_bytes(b'image one')
            output = root / 'tile_GFP_result.csv'
            output.write_bytes(b'id\n1\n')
            manifest = root / 'tile_GFP_inputs.json'
            from src.core.provenance import file_stamp
            manifest.write_text(json.dumps({
                'images': [{'status': 'processed', **file_stamp(image)}],
                'output_size': output.stat().st_size,
                'output_sha256': file_sha256(output),
            }), encoding='utf-8')
            validate_raw_input_manifest(manifest, output)
            image.write_bytes(b'image changed')
            with self.assertRaisesRegex(RuntimeError, 'input image changed'):
                validate_raw_input_manifest(manifest, output)

    def test_manifest_detects_modified_member(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "result.csv"
            output.write_text("id\n1\n", encoding="utf-8")
            manifest = root / "manifest.json"
            write_output_manifest(manifest, [output], {"config": "one"})
            self.assertTrue(output_manifest_valid(manifest, {"config": "one"}))
            output.write_text("id\n2\n", encoding="utf-8")
            self.assertFalse(output_manifest_valid(manifest, {"config": "one"}))


if __name__ == "__main__":
    unittest.main()
