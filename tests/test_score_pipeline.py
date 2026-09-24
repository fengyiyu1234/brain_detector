"""Regression checks for StarDist probability and downstream score/coordinate flow."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.detection_filter import filter_detection_df
from src.core.point_cloud_aligner import apply_shift_to_csv
from src.core.stitcher import combine_predictions
from src.core.worker import stardist_regions_with_scores


class ScorePipelineTest(unittest.TestCase):
    def test_stardist_label_uses_its_original_probability(self):
        labels = np.array([[0, 2, 2], [0, 0, 0]], dtype=np.int32)
        image = np.ones_like(labels, dtype=np.uint16)
        regions = list(stardist_regions_with_scores(
            labels, {"prob": np.array([0.24, 0.83])}, image))
        self.assertEqual([(r.label, score) for r, score in regions],
                         [(2, 0.83)])
        with self.assertRaisesRegex(ValueError, "no instance probability"):
            list(stardist_regions_with_scores(labels, {"prob": [0.24]}, image))

    def test_score_and_offsets_survive_align_filter_stitch(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw.csv"
            aligned = Path(tmp) / "aligned.csv"
            df = pd.DataFrame([
                ["slice_3", 10, 20, 18, 28, "nucleus", 0.83, 100, 3],
                ["slice_3", 30, 20, 38, 28, "nucleus", 0.24, 100, 3],
            ], columns=["slice_name", "x1", "y1", "x2", "y2",
                        "class", "score", "mean", "z"])
            df.to_csv(raw, index=False)
            apply_shift_to_csv(str(raw), 7, -2, 1, str(aligned))
            filtered = filter_detection_df(
                pd.read_csv(aligned), {"score_min": 0.30})
            self.assertEqual(len(filtered), 1)
            self.assertAlmostEqual(filtered.iloc[0]["score"], 0.83)
            self.assertEqual(filtered.iloc[0]["x1"], 17)
            self.assertEqual(filtered.iloc[0]["y1"], 18)
            self.assertEqual(filtered.iloc[0]["z"], 4)

            disp = np.array([[[100, 200, 5]]])
            predictions = [[np.empty((0, 8)) for _ in range(2)]
                           for _ in range(20)]
            metadata, trace = [], {}
            rows = csv.reader(filtered.to_csv(index=False).splitlines()[1:])
            combine_predictions(predictions, rows, None, 0, 20, (0, 0),
                                disp, (2048, 2048), metadata, "tile_a",
                                row_meta=trace)
            result = predictions[8][1]
            self.assertEqual(result.shape, (1, 8))
            self.assertEqual(tuple(float(v) for v in result[0][0:4]),
                             (117, 218, 125, 226))
            self.assertAlmostEqual(float(result[0][4]), 0.83)
            self.assertEqual(int(result[0][7]), 9)
            self.assertEqual(trace[(8, 1)], [("tile_a", "slice_3")])


if __name__ == "__main__":
    unittest.main()