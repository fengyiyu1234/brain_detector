"""Source boxes retain the exact channel tracks selected by Stage 4."""

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.core.coloc_sources import (cell_id, coloc_display_box, colocalization_status,
                                    primary_source_trace, source_3d_json,
                                    write_source_3d, write_source_boxes)
from src.core.stitcher import (annotate_soma_with_tf_containment,
                               match_soma_3d_iou, merge_soma_volumes_union)


def volume(cls, cx, cy, z, box, per_z):
    return {
        "class": cls, "cx": cx, "cy": cy, "cz": z,
        "x1_3d": box[0], "y1_3d": box[1],
        "x2_3d": box[2], "y2_3d": box[3],
        "z_min": min(per_z), "z_max": max(per_z),
        "per_z_boxes": per_z, "score": 0.8, "mean": 100.0,
    }


class ColocSourcesTest(unittest.TestCase):
    def test_matched_tf_source_and_per_z_boxes(self):
        soma = volume("neuron_GFP_RFP", 20, 20, 10, (0, 0, 40, 40),
                      {9: (1, 1, 39, 39), 10: (0, 0, 40, 40)})
        gfp = volume("neuron_GFP", 20, 20, 10, (1, 1, 39, 39),
                     {9: (1, 1, 39, 39), 10: (2, 2, 38, 38)})
        rfp = volume("neuron_RFP", 20, 20, 10, (0, 0, 40, 40),
                     {10: (0, 0, 40, 40)})
        sox9 = volume("nucleus_Sox9", 20, 20, 10, (15, 15, 25, 25),
                      {10: (15, 15, 25, 25)})
        distant = volume("nucleus_Sox9", 100, 100, 10, (95, 95, 105, 105),
                         {10: (95, 95, 105, 105)})
        soma["source_tracks"] = [("GFP", gfp), ("RFP", rfp)]

        annotate_soma_with_tf_containment(
            [soma], [sox9, distant], z_pad=1, source_channel="Sox9")
        self.assertEqual(soma["class"], "neuron_GFP_RFP_Sox9")
        self.assertEqual([ch for ch, _ in soma["source_tracks"]],
                         ["GFP", "RFP", "Sox9"])
        self.assertIs(soma["source_tracks"][-1][1], sox9)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sources.csv"
            write_source_boxes(path, [soma])
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 4)
        self.assertEqual({row["coloc_id"] for row in rows}, {cell_id(soma, 0)})
        self.assertEqual([row["channel"] for row in rows],
                         ["GFP", "GFP", "RFP", "Sox9"])
        self.assertEqual(
            (int(rows[-1]["z"]), float(rows[-1]["x1"]), float(rows[-1]["y1"])),
            (10, 15.0, 15.0))

        source_3d = json.loads(source_3d_json(soma, 0.65, 8.0))
        self.assertEqual(set(source_3d), {"GFP", "RFP", "Sox9"})
        self.assertEqual(source_3d["Sox9"][0]["z_min"], 10)
        self.assertEqual(source_3d["RFP"][0]["x2_3d"], 40.0)
        self.assertEqual(source_3d["Sox9"][0]["cx_um"], 13.0)
        self.assertEqual(source_3d["Sox9"][0]["cz_um"], 72.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sources_3d.csv"
            write_source_3d(path, [soma])
            with path.open(newline="", encoding="utf-8") as handle:
                flat = list(csv.DictReader(handle))
        self.assertEqual(len(flat), 3)
        self.assertEqual(flat[-1]["channel"], "Sox9")
        self.assertEqual(flat[-1]["x1_3d"], "15.0")

    def test_soma_union_is_used_for_both_independent_tf_matches(self):
        gfp = volume("neuron_GFP", 20, 20, 10, (10, 10, 30, 30),
                     {10: [10, 10, 30, 30], 11: [10, 10, 30, 30]})
        gfp["z_min"], gfp["z_max"] = 10, 11
        rfp = volume("neuron_RFP", 26, 22, 11, (16, 12, 36, 32),
                     {11: [16, 12, 36, 32], 12: [16, 12, 36, 32]})
        rfp["z_min"], rfp["z_max"] = 11, 12
        original_gfp = copy.deepcopy(gfp)
        gfp["source_tracks"] = [("GFP", original_gfp)]
        rfp["source_tracks"] = [("RFP", copy.deepcopy(rfp))]
        pairs, unmatched_gfp, unmatched_rfp = match_soma_3d_iou(
            [gfp], [rfp], iou_thresh=0.15, iomin_thresh=0.5)
        self.assertEqual(len(pairs), 1)
        self.assertEqual((unmatched_gfp, unmatched_rfp), ([], []))
        soma = merge_soma_volumes_union(gfp, rfp)
        soma["source_tracks"].extend(rfp["source_tracks"])
        soma["class"] = "neuron_GFP_RFP"

        self.assertEqual(tuple(soma[k] for k in
                         ("x1_3d", "y1_3d", "x2_3d", "y2_3d", "z_min", "z_max")),
                         (10, 10, 36, 32, 10, 12))
        self.assertEqual((soma["cx"], soma["cy"], soma["cz"]), (23, 21, 11))
        self.assertEqual(soma["per_z_boxes"][11], [10, 10, 36, 32])
        self.assertEqual(coloc_display_box(soma), ([10, 10, 36, 32], 11))
        self.assertEqual(original_gfp["x2_3d"], 30)
        self.assertEqual(original_gfp["per_z_boxes"][11], [10, 10, 30, 30])

        sox9 = volume("nucleus_Sox9", 31, 19, 11, (29, 17, 33, 21),
                      {11: [29, 17, 33, 21]})
        olig2 = volume("nucleus_Olig2", 22, 21, 11, (20, 19, 24, 23),
                       {11: [20, 19, 24, 23]})
        results = []
        for first, second in ((('Sox9', sox9), ('Olig2', olig2)),
                              (('Olig2', olig2), ('Sox9', sox9))):
            candidate = copy.deepcopy(soma)
            for channel, tf in (first, second):
                annotate_soma_with_tf_containment(
                    [candidate], [tf], z_pad=0, max_center_dist_ratio=0.6,
                    source_channel=channel)
            status = colocalization_status(
                candidate, ['GFP', 'RFP'], ['Sox9', 'Olig2'])
            results.append((candidate['class'], status))
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0][0], 'neuron_GFP_Olig2_RFP_Sox9')
        self.assertEqual(results[0][1]['soma_status'], 'double_positive')
        self.assertEqual(results[0][1]['tf_status'], 'double_positive')
        self.assertEqual(json.loads(results[0][1]['tf_positive_channels']),
                         ['Sox9', 'Olig2'])

    def test_all_rule_requires_tf_inside_every_soma_channel(self):
        # Real case (338600_388500 seam, z 419-423): the low-score RFP box is twice the
        # GFP box; a neighbouring Sox9 nucleus sits inside RFP but past GFP's lower edge.
        gfp = volume("neuron_GFP", 7010.119, 9390.447, 421,
                     (6991.020, 9371.415, 7029.218, 9409.480), {421: (0, 0, 1, 1)})
        rfp = volume("neuron_RFP", 7016.633, 9390.903, 421,
                     (6979.665, 9355.778, 7053.601, 9426.028), {421: (0, 0, 1, 1)})
        gfp["z_min"] = rfp["z_min"] = 419
        gfp["z_max"] = rfp["z_max"] = 423
        gfp["track_id"], rfp["track_id"] = "gfp-track", "rfp-track"
        sox9 = volume("nucleus_Sox9", 7021.5, 9407.5, 422,
                      (7017.0, 9401.5, 7026.0, 9413.5), {421: (0, 0, 1, 1)})
        sox9["z_min"], sox9["z_max"] = 421, 423
        # A nucleus at the centre of both boxes must still match under 'all'.
        olig2 = volume("nucleus_Olig2", 7012.0, 9391.0, 421,
                       (7007.0, 9386.0, 7017.0, 9396.0), {421: (0, 0, 1, 1)})

        def merged():
            soma = merge_soma_volumes_union(copy.deepcopy(gfp), copy.deepcopy(rfp))
            soma["class"] = "neuron_GFP_RFP"
            soma["source_tracks"] = [("GFP", gfp), ("RFP", rfp)]
            return soma

        union = merged()
        annotate_soma_with_tf_containment([union], [sox9], source_channel="Sox9")
        self.assertEqual(union["class"], "neuron_GFP_RFP_Sox9")

        strict = merged()
        for channel, tf in (("Sox9", sox9), ("Olig2", olig2)):
            annotate_soma_with_tf_containment(
                [strict], [tf], source_channel=channel, soma_channels={"GFP", "RFP"})
        self.assertEqual(strict["class"], "neuron_GFP_Olig2_RFP")
        self.assertEqual([ch for ch, _ in strict["source_tracks"]],
                         ["GFP", "RFP", "Olig2"])
        gates = json.loads(strict["match_evidence"][0]["tf_gates"])
        self.assertEqual([g["soma_channel"] for g in gates], ["GFP", "RFP"])
        self.assertEqual(gates[0]["soma_track_id"], "gfp-track")

    def test_all_rule_checks_z_on_merged_soma_range(self):
        # Real case (ET10 tile 327300_388500, coloc:3844a55f...): the GFP track was seen on
        # z427 only, RFP on z425-428; the Sox9 nucleus (z425-427) sits in the centre of
        # both boxes. Gating z on the one-layer GFP track rejected it; z must use the
        # merged soma range (425-428) while xy/center stay per channel.
        gfp = volume("glia_GFP", 6013.402, 7362.380, 427,
                     (5995.167, 7344.182, 6031.637, 7380.577), {427: (0, 0, 1, 1)})
        rfp = volume("glia_RFP", 6014.724, 7362.138, 426,
                     (5979.329, 7322.743, 6050.119, 7401.534), {426: (0, 0, 1, 1)})
        rfp["z_min"], rfp["z_max"] = 425, 428
        sox9 = volume("nucleus_Sox9", 6013.0, 7360.5, 426,
                      (6006.0, 7353.0, 6020.0, 7368.0), {426: (0, 0, 1, 1)})
        sox9["z_min"], sox9["z_max"] = 425, 427
        soma = merge_soma_volumes_union(copy.deepcopy(gfp), copy.deepcopy(rfp))
        soma["class"] = "glia_GFP_RFP"
        soma["source_tracks"] = [("GFP", gfp), ("RFP", rfp)]
        self.assertEqual((soma["z_min"], soma["z_max"]), (425, 428))
        annotate_soma_with_tf_containment(
            [soma], [sox9], z_pad=1, max_center_dist_ratio=0.35,
            source_channel="Sox9", soma_channels={"GFP", "RFP"})
        self.assertEqual(soma["class"], "glia_GFP_RFP_Sox9")

    def test_all_rule_without_soma_sources_uses_own_box(self):
        soma = volume("neuron_GFP", 20, 15, 10, (0, 0, 40, 30), {10: (0, 0, 40, 30)})
        soma["source_tracks"] = [("Olig2", {})]   # no soma-channel track saved
        sox9 = volume("nucleus_Sox9", 22, 15, 10, (20, 13, 24, 17), {10: (20, 13, 24, 17)})
        annotate_soma_with_tf_containment(
            [soma], [sox9], z_pad=0, source_channel="Sox9", soma_channels={"GFP"})
        self.assertEqual(soma["class"], "neuron_GFP_Sox9")

    def test_tf_on_center_distance_boundary_still_matches(self):
        # radius = hypot(40, 30) / 2 = 25, gate = 0.4 * 25 = 10; the TF sits at exactly 10px,
        # so the search radius (derived from the gate) must not drop it.
        soma = volume("neuron_GFP", 20, 15, 10, (0, 0, 40, 30), {10: (0, 0, 40, 30)})
        sox9 = volume("nucleus_Sox9", 30, 15, 10, (28, 13, 32, 17), {10: (28, 13, 32, 17)})
        annotate_soma_with_tf_containment(
            [soma], [sox9], z_pad=0, max_center_dist_ratio=0.4, source_channel="Sox9")
        self.assertEqual(soma["class"], "neuron_GFP_Sox9")
        self.assertEqual(soma["match_evidence"][0]["center_distance"], 10.0)

    def test_tf_positivity_counts_unique_channels(self):
        soma = {'source_tracks': [('GFP', {}), ('Sox9', {}), ('Sox9', {})]}
        status = colocalization_status(soma, ['GFP', 'RFP'], ['Sox9', 'Olig2'])
        self.assertEqual(status['soma_status'], 'single_positive')
        self.assertEqual(status['tf_status'], 'single_positive')
        self.assertEqual(status['tf_positive_count'], 1)
        self.assertEqual(colocalization_status(
            {'source_tracks': [('GFP', {})]}, ['GFP', 'RFP'],
            ['Sox9', 'Olig2'])['tf_status'], 'negative')

    def test_primary_trace_marks_synthetic_center_box(self):
        track = volume('neuron_GFP', 10, 10, 2, (0, 0, 20, 20),
                       {1: (0, 0, 20, 20), 3: (1, 1, 21, 21)})
        track['best_detection_id'] = 'best'
        track['member_detections'] = [
            {'z': 1, 'detection_id': 'best', 'tile_name': 'tile',
             'slice_name': 'slice_1'},
            {'z': 3, 'detection_id': 'other', 'tile_name': 'tile',
             'slice_name': 'slice_3'},
        ]
        soma = {'cz': 2, 'source_tracks': [('GFP', track)]}
        self.assertEqual(primary_source_trace(soma),
                         ('tile', 'slice_1',
                          'primary_best_member_synthetic_center_box'))

    def test_primary_trace_labels_merged_union_as_synthetic(self):
        source = volume('neuron_GFP', 10, 10, 2, (0, 0, 20, 20),
                        {2: (0, 0, 20, 20)})
        source['member_detections'] = [
            {'z': 2, 'detection_id': 'gfp-2',
             'tile_name': 'tile', 'slice_name': 'slice_2'}]
        soma = {'cz': 2, 'bounds_method': 'cross_channel_bbox_union',
                'source_tracks': [('GFP', source)]}
        self.assertEqual(primary_source_trace(soma),
                         ('tile', 'slice_2',
                          'merged_soma_union_primary_source_exact_primary_center_member'))

    def test_multiple_tracks_in_one_channel_are_preserved(self):
        a = volume("nucleus_Sox9", 20, 20, 10, (15, 15, 25, 25),
                   {10: (15, 15, 25, 25)})
        b = volume("nucleus_Sox9", 22, 22, 11, (16, 16, 26, 26),
                   {11: (16, 16, 26, 26)})
        soma = {"source_tracks": [("Sox9", a), ("Sox9", b), ("Olig2", a)]}
        sources = json.loads(source_3d_json(soma))
        self.assertEqual([track["source_track"] for track in sources["Sox9"]],
                         [0, 1])
        self.assertEqual(sources["Olig2"][0]["source_track"], 2)


if __name__ == "__main__":
    unittest.main()
