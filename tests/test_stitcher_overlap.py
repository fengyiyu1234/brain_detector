"""Regression checks for detections in overlapping tiles."""

import unittest

import numpy as np

from src.core.stitcher import combine_predictions


class StitcherOverlapTest(unittest.TestCase):
    def _stitch(self, left_rows, right_rows):
        predictions = [[np.empty((0, 8)) for _ in range(2)] for _ in range(3)]
        disp = np.array([[[0, 0, 0], [80, 0, 0]]])
        metadata, trace = [], {}
        for col, rows in enumerate((left_rows, right_rows)):
            combine_predictions(
                predictions, rows, None, 0, 3, (0, col), disp, (100, 180),
                metadata, f"tile_{col}", tILESIZE=100, row_meta=trace)
        return predictions, metadata, trace

    @staticmethod
    def _row(name, x1, y1, x2, y2, cls="glia", z=1):
        return [name, x1, y1, x2, y2, cls, 0.5, 100, z]

    def test_unique_cell_in_overlap_survives(self):
        left = [self._row("left_other", 2, 40, 12, 50)]
        right = [self._row("right_unique", 5, 40, 15, 50)]
        predictions, metadata, trace = self._stitch(left, right)
        self.assertEqual(len(predictions[0][0]), 2)
        self.assertEqual(trace[(0, 0)], [
            ("tile_0", "left_other"), ("tile_1", "right_unique")])
        self.assertEqual(len(metadata), 2)

    def test_same_cell_in_overlap_is_kept_once(self):
        left = [self._row("left_cell", 85, 40, 95, 50)]
        right = [self._row("right_duplicate", 5, 40, 15, 50)]
        predictions, metadata, trace = self._stitch(left, right)
        self.assertEqual(len(predictions[0][0]), 1)
        self.assertEqual(trace[(0, 0)], [("tile_0", "left_cell")])
        self.assertEqual(len(metadata), 1)

    def test_nearby_other_class_and_other_z_do_not_suppress(self):
        left = [self._row("left_neuron", 85, 40, 95, 50, "neuron"),
                self._row("left_other_z", 85, 40, 95, 50, "glia", z=2)]
        right = [self._row("right_glia", 5, 40, 15, 50)]
        predictions, _, trace = self._stitch(left, right)
        self.assertEqual(len(predictions[0][0]), 1)
        self.assertEqual(trace[(0, 0)], [("tile_1", "right_glia")])


if __name__ == "__main__":
    unittest.main()
