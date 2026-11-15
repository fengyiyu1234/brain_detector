"""Stage 3 soma linking options (z_linker.link_tracks) and their parameter defaults."""
import unittest

import numpy as np

from src.core import z_linker
from src.core.z_linker import (STAGE3_LINK_DEFAULTS, dedup_cross_tile, link_tracks,
                               run_z_linker, stage3_link_params)

BASE = dict(iou_thresh=0.35, min_z_layers=1, max_cell_z_span=7, max_z_gap=0)
OFF = dict(link_iomin_thresh=None, link_across_classes=False,
           cross_tile_dedup_iou=None, cross_tile_dedup_iomin=None)


def det(z, box, cls="glia_RFP", tile="t", score=0.5, i=0):
    return [*box, score, 100.0, cls, z, f"raw2d:{tile}_{z}_{i}", tile, f"{tile}_{z}"]


def layers(tracks):
    return sorted(sorted(t["per_z_boxes"]) for t in tracks)


class LinkTracksTest(unittest.TestCase):
    def test_all_options_off_is_run_z_linker(self):
        rng = np.random.default_rng(0)
        rows = []
        for z in range(1, 9):
            for i in range(30):
                x, y = rng.uniform(0, 400, 2)
                w = rng.uniform(15, 60)
                rows.append(det(z, (x, y, x + w, y + w), rng.choice(["neuron_RFP", "glia_RFP"]),
                                score=float(rng.uniform(0.2, 0.9)), i=i))
        mat = np.array(rows, dtype=object)
        summary_a, cells_a = run_z_linker(mat.copy(), **BASE)
        summary_b, cells_b = link_tracks(mat.copy(), **BASE, **OFF)
        self.assertEqual([c["track_id"] for c in cells_a], [c["track_id"] for c in cells_b])
        self.assertEqual(summary_a.tolist(), summary_b.tolist())

    def test_iomin_links_a_small_box_inside_a_large_one(self):
        small, big = (52, 5, 79, 42), (23, 0, 109, 84)        # IoU ~0.14, IoMin 1.0
        mat = np.array([det(1, small), det(2, small), det(3, big),
                        det(3, (300, 300, 330, 330), i=1), det(4, (302, 300, 332, 330))],
                       dtype=object)
        self.assertEqual(layers(link_tracks(mat, **BASE, **OFF)[1]), [[1, 2], [3], [3, 4]])
        on = dict(OFF, link_iomin_thresh=0.8)
        self.assertEqual(layers(link_tracks(mat, **BASE, **on)[1]), [[1, 2, 3], [3, 4]])
        # the swapped match functions are restored afterwards
        self.assertEqual(z_linker._match_sparse.__name__, "_match_sparse")
        self.assertEqual(z_linker._match_dense.__name__, "_match_dense")

    def test_neuron_and_glia_link_and_the_track_is_glia(self):
        box = (0, 0, 40, 40)
        mat = np.array([det(1, box, "neuron_RFP"), det(2, box, "glia_RFP"),
                        det(3, box, "neuron_RFP")], dtype=object)
        summary, cells = link_tracks(mat, **BASE, **OFF)
        self.assertEqual(layers(cells), [[1], [2], [3]])
        summary, cells = link_tracks(mat, **BASE, **dict(OFF, link_across_classes=True))
        self.assertEqual(layers(cells), [[1, 2, 3]])
        self.assertEqual(cells[0]["class"], "glia_RFP")
        self.assertEqual(summary[0][6], "glia_RFP")

    def test_gap_one_bridges_a_missing_layer(self):
        box = (0, 0, 40, 40)
        mat = np.array([det(1, box), det(3, box)], dtype=object)
        self.assertEqual(layers(link_tracks(mat, **BASE, **OFF)[1]), [[1], [3]])
        self.assertEqual(layers(link_tracks(mat, **dict(BASE, max_z_gap=1), **OFF)[1]),
                         [[1, 3]])

    def test_cross_tile_dedup_drops_the_lower_score_box_with_glia_priority(self):
        mat = np.array([
            det(5, (0, 0, 40, 40), "neuron_RFP", tile="a", score=0.8),
            det(5, (10, 10, 50, 50), "glia_RFP", tile="b", score=0.3),     # IoU 0.39
            det(5, (0, -20, 40, 20), "neuron_RFP", tile="a", score=0.9, i=1),  # same tile
            det(6, (200, 200, 240, 240), tile="a"),
        ], dtype=object)
        dropped = []
        kept = dedup_cross_tile(mat, 0.25, 0.45, dropped)
        self.assertEqual([r[8] for r in kept],
                         ["raw2d:a_5_0", "raw2d:a_5_1", "raw2d:a_6_0"])
        self.assertEqual(kept[0][6], "glia_RFP")               # took the dropped glia's class
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0]["detection_id"], "raw2d:b_5_0")
        self.assertEqual(dropped[0]["kept_detection_id"], "raw2d:a_5_0")
        self.assertTrue(dropped[0]["kept_relabelled_glia"])
        self.assertEqual(set(dropped[0]), set(z_linker.LINK_DEDUP_COLUMNS))
        self.assertEqual(mat[0][6], "neuron_RFP")              # input left unchanged

        rows = []
        link_tracks(mat, **BASE, **dict(OFF, cross_tile_dedup_iou=0.25,
                                       cross_tile_dedup_iomin=0.45), dedup_rows=rows)
        self.assertEqual([r["detection_id"] for r in rows], ["raw2d:b_5_0"])

    def test_stage3_params_defaults_override_and_null(self):
        params = stage3_link_params(None)
        self.assertEqual(params, STAGE3_LINK_DEFAULTS)
        self.assertEqual(params["soma"]["max_cell_z_span"], 7)
        self.assertEqual(params["soma"]["max_z_gap"], 1)
        self.assertEqual(set(params["tf"]),
                         {"iou_thresh", "min_z_layers", "max_cell_z_span", "max_z_gap"})
        params = stage3_link_params({"soma": {"max_cell_z_span": 5, "link_iomin_thresh": None,
                                              "iou_thresh_3d": 0.15}})
        self.assertEqual(params["soma"]["max_cell_z_span"], 5)
        self.assertIsNone(params["soma"]["link_iomin_thresh"])
        self.assertNotIn("iou_thresh_3d", params["soma"])      # Phase A keys stay out


if __name__ == "__main__":
    unittest.main()
