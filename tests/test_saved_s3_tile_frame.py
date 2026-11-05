"""Stage 3 summary rows drawn on one tile's own image in the saved local view."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.utils.saved_view import (
    neighbour_tiles, read_global_boxes, s3_rows_in_tile_frame,
)


class S3TileFrameTests(unittest.TestCase):
    # Tile B sits east of A. Stage 3: global = local + (abs_x, abs_y), global z = local z - z0.
    geometry = {
        "A": ((0, 0), 0, 0, 5),
        "B": ((0, 1), 100, 0, 2),
        "C": ((0, 3), 300, 0, 0),
    }
    # GUI tile position of B: XML origin and z0, as global_to_local_rows expects.
    position_b = SimpleNamespace(x=100, y=0, z=2)
    seam = {("A", "B"): {0: (3.0, -2.0, 1)}}  # B's copy minus A's copy

    def rows(self):
        return pd.DataFrame([
            dict(x1=111, y1=21, x2=131, y2=41, z=6, best_box_z=5, display_z=4,
                 best_detection_id="own"),
            dict(x1=150, y1=20, x2=170, y2=40, z=5, best_box_z=5, display_z=4,
                 best_detection_id="neighbour"),
            dict(x1=120, y1=10, x2=130, y2=20, z=5, best_box_z=5, display_z=4,
                 best_detection_id="unknown"),
        ])

    def sources(self):
        return {
            "own": ("B", 10.0, 20.0, 30.0, 40.0, 7),
            "neighbour": ("A", 150.0, 20.0, 170.0, 40.0, 10),
        }

    def test_own_neighbour_and_fallback_rows(self):
        local = s3_rows_in_tile_frame(
            self.rows(), "B", self.position_b, self.sources(),
            self.geometry, self.seam, seg_layers=50)
        own, neighbour, unknown = local.to_dict("records")
        # This tile's own detection: its tile-local box and layer, exactly.
        self.assertEqual((own["x1"], own["y1"], own["x2"], own["y2"]), (10, 20, 30, 40))
        self.assertEqual((own["display_z"], own["frame"]), (6, "own"))
        # Neighbour detection: A local + A origin - B origin + seam offset;
        # z: A global 10 - 5 = 5, +1 seam layer, +2 for B's z0, zero-based.
        self.assertEqual((neighbour["x1"], neighbour["y1"], neighbour["x2"], neighbour["y2"]),
                         (53, 18, 73, 38))
        self.assertEqual((neighbour["display_z"], neighbour["frame"]), (7, "seam"))
        # No source detection found: plain XML placement.
        self.assertEqual((unknown["x1"], unknown["y1"], unknown["display_z"], unknown["frame"]),
                         (20, 10, 6, "xml"))
        # Saved global box stays available for PKL track lookup.
        self.assertEqual((neighbour["stage3_x1"], neighbour["stage3_y2"]), (150, 40))
        self.assertEqual(list(local["source_tile"]), ["B", "A", ""])

    def test_neighbour_without_seam_entry_uses_xml_difference(self):
        local = s3_rows_in_tile_frame(
            self.rows(), "B", self.position_b, self.sources(),
            self.geometry, seam_offsets=None)
        neighbour = local.to_dict("records")[1]
        self.assertEqual((neighbour["x1"], neighbour["y1"], neighbour["display_z"]),
                         (50, 20, 6))

    def test_missing_geometry_keeps_neighbour_at_xml(self):
        local = s3_rows_in_tile_frame(
            self.rows(), "B", self.position_b, self.sources())
        self.assertEqual(list(local["frame"]), ["own", "xml", "xml"])
        self.assertEqual(local.loc[1, "x1"], 50)

    def test_neighbours_follow_stage3_grid(self):
        self.assertEqual(neighbour_tiles(self.geometry, "B"), ["A"])
        self.assertEqual(neighbour_tiles(None, "B"), [])

    def s4_rows(self):
        def track(source_track, box, z_range, best_id, cx, cy, best_z=5):
            return dict(source_track=source_track, x1_3d=box[0], y1_3d=box[1],
                        x2_3d=box[2], y2_3d=box[3], z_min=z_range[0], z_max=z_range[1],
                        cx=cx, cy=cy, best_box_z=best_z, best_detection_id=best_id)
        rep = "representative_centered_max_size"
        union = "cross_channel_bbox_union"
        gfp = track(0, (110, 20, 130, 40), (4, 6), "ownB2", 120, 30)
        rfp = track(1, (150, 20, 170, 40), (5, 7), "neighbour", 160, 30)
        sox9 = track(2, (0, 0, 500, 500), (1, 9), "tf", 250, 250)  # TF: not in the union
        rows = [
            # rep-z, exact centre member detected by neighbour A.
            dict(coloc_id="c1", bounds_method=rep, x1=150, y1=20, x2=170, y2=40, z=5,
                 x1_3d=150, y1_3d=20, x2_3d=170, y2_3d=40,
                 source_3d=json.dumps({"RFP": [rfp]})),
            # rep-z, synthetic box: moves with B's representative, saved 3 px east.
            dict(coloc_id="c2", bounds_method=rep, x1=105, y1=15, x2=135, y2=45, z=5,
                 x1_3d=105, y1_3d=15, x2_3d=135, y2_3d=45,
                 source_3d=json.dumps({"GFP": [dict(gfp, cx=123)]})),
            # Union of GFP (own) and RFP (neighbour A).
            dict(coloc_id="c3", bounds_method=union, x1=110, y1=20, x2=170, y2=40, z=6,
                 x1_3d=110, y1_3d=20, x2_3d=170, y2_3d=40,
                 source_3d=json.dumps({"GFP": [gfp], "RFP": [rfp], "Sox9": [sox9]})),
            # Union that is not the saved soma tracks' union: left at XML placement.
            dict(coloc_id="c4", bounds_method=union, x1=100, y1=20, x2=170, y2=40, z=6,
                 x1_3d=100, y1_3d=20, x2_3d=170, y2_3d=40,
                 source_3d=json.dumps({"GFP": [gfp], "RFP": [rfp]})),
        ]
        frame = pd.DataFrame(rows)
        frame["display_z"] = frame["z"] - 1
        return frame

    def test_stage4_rep_z_and_union_rows(self):
        from src.utils.saved_view import TileFrame, s4_rows_in_tile_frame
        frames = {c: TileFrame("B", self.geometry, self.seam) for c in ("GFP", "RFP")}
        sources = {
            "GFP": {"ownB2": ("B", 10.0, 20.0, 30.0, 40.0, 7)},
            "RFP": {"neighbour": ("A", 150.0, 20.0, 170.0, 40.0, 10)},
        }
        centers = {"c1": ("A", 150.0, 20.0, 170.0, 40.0, 10)}
        local = s4_rows_in_tile_frame(self.s4_rows(), self.position_b, frames, sources,
                                      ("GFP", "RFP"), centers)
        c1, c2, c3, c4 = local.to_dict("records")
        box = lambda r: (r["x1"], r["y1"], r["x2"], r["y2"])
        # Same placement as the Stage 3 neighbour row of the same detection.
        self.assertEqual((box(c1), c1["display_z"], c1["frame"]), ((53, 18, 73, 38), 7, "seam"))
        # Synthetic box follows its representative: 3 px west, same layer (6).
        self.assertEqual((box(c2), c2["display_z"], c2["frame"]), ((2, 15, 32, 45), 6, "own"))
        # GFP 3D box stays; RFP's moves (+3, -2, +1 layer): union x 10-73, y 18-40,
        # z 4-8 -> middle layer 6 (one-based global) -> display 6 - 1 + 2.
        self.assertEqual((box(c3), c3["display_z"], c3["frame"], c3["source_tile"]),
                         ((10, 18, 73, 40), 7, "seam", "A+B"))
        self.assertEqual((box(c4), c4["display_z"], c4["frame"]), ((0, 20, 70, 40), 7, "xml"))

    def test_summary_rows_select_and_draw_best_box_layer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s3.csv"
            pd.DataFrame([
                dict(x1=0, y1=0, x2=5, y2=5, z=5, best_box_z=3, **{"class": "glia"}),
                dict(x1=0, y1=0, x2=5, y2=5, z=3, best_box_z=5, **{"class": "glia"}),
            ]).to_csv(path, index=False)
            rows = read_global_boxes(str(path), (0, 0, 10, 10), (2, 3), z_column="best_box_z")
            self.assertEqual(list(rows["display_z"]), [2])
            self.assertEqual(list(rows["z"]), [5])
            legacy = read_global_boxes(str(path), (0, 0, 10, 10), (2, 3), z_column="missing")
            self.assertEqual(list(legacy["display_z"]), [2])
            self.assertEqual(list(legacy["best_box_z"]), [5])


if __name__ == "__main__":
    unittest.main()
