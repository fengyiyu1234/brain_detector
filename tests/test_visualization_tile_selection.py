"""Saved global cells must appear in every tile they spatially overlap."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.utils.visualize import (
    _load_coloc_s4_shapes,
    _load_global_csv_to_tile_shapes,
)


class GlobalTileSelectionTests(unittest.TestCase):
    def test_neighbor_sourced_cell_is_visible_in_current_tile(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "cells.csv"
            pd.DataFrame([
                # Local box [230, 345, 280, 395], local z=506. The cell
                # originated in a neighboring tile but falls in this canvas.
                dict(x1=5420, y1=7267, x2=5470, y2=7317, z=501,
                     score=.8, mean=10000, **{"class": "glia_GFP_RFP"},
                     tile_name="348800_349900"),
                # Matching provenance must not admit a box outside the canvas.
                dict(x1=9000, y1=9000, x2=9050, y2=9050, z=501,
                     score=.8, mean=10000, **{"class": "glia_GFP_RFP"},
                     tile_name="348800_361200"),
            ]).to_csv(p, index=False)
            groups = [dict(name="GFP+RFP", channels=["GFP", "RFP"])]
            displayed = _load_coloc_s4_shapes(
                str(p), "348800_361200", (500, 550), groups,
                tile_x0=5190, tile_y0=6922, tile_z0=6,
                canvas_w=2048, canvas_h=2048,
                left_margin=300, top_margin=300,
            )
            self.assertEqual(len(displayed[0][1]), 1)
            self.assertEqual(displayed[0][4][0]["z"], 6)
            self.assertAlmostEqual(displayed[0][4][0]["x1"], 230)

            (shapes, _, metadata), _ = _load_global_csv_to_tile_shapes(
                str(p), "348800_361200", 5190, 6922, 6,
                (500, 550), 2048, 2048,
                left_margin=300, top_margin=300,
            )
            self.assertEqual(len(shapes), 1)
            self.assertEqual(metadata[0]["z"], 6)
            self.assertAlmostEqual(metadata[0]["x1"], 230)

            kept, rejected = _load_global_csv_to_tile_shapes(
                str(p), "348800_361200", 5190, 6922, 6,
                (500, 550), 2048, 2048,
                filt={"bbox_min": 60},
            )
            self.assertEqual(len(kept[0]), 0)
            self.assertEqual(len(rejected[0]), 1)


if __name__ == "__main__":
    unittest.main()
