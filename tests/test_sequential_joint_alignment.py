"""Focused checks for the sequential joint scorer and frame conversion."""

import unittest

from src.core.point_cloud_aligner import (
    _choose_joint_nucleus_shift, compute_tile_channel_shifts,
    rebase_shifts, shifted_cell_boxes,
)


def cell(x, y, z=5, width=6):
    half = width / 2
    return {
        "cx": x, "cy": y, "cz": z,
        "x1_3d": x - half, "x2_3d": x + half,
        "y1_3d": y - half, "y2_3d": y + half,
        "z_min": z, "z_max": z,
        "per_z_boxes": {z: [x - half, y - half, x + half, y + half]},
    }


class SequentialJointTests(unittest.TestCase):
    def test_fixed_references_and_unique_soma_matches(self):
        gfp = [cell(20, 20, width=28)]
        rfp_raw = [cell(17, 20, width=28)]
        rfp = shifted_cell_boxes(rfp_raw, (3, 0, 0))
        self.assertEqual(rfp[0]["cx"], 20)
        self.assertEqual(rfp_raw[0]["cx"], 17)
        sox = [cell(20, 20)]
        olig_raw = [cell(16, 20)]
        shift, fraction, report = _choose_joint_nucleus_shift(
            gfp + rfp, olig_raw,
            {"GFP": (0, 0, 0), "GFP_RFP": (4, 0, 0), "Sox9": (4, 0, 0)},
            "GFP", 0.4, 0, 1, 0, 10, sox_cells=sox)
        self.assertEqual(shift, (4, 0, 0))
        self.assertEqual(fraction, 1.0)
        self.assertEqual(report["matched_nuclei"], 1)  # Both somata count once.

    def test_four_channel_alignment_from_detected_cells(self):
        sites = [(40, 40), (90, 45), (145, 70),
                 (55, 130), (115, 145), (170, 160)]
        cells = {
            "GFP": [cell(x, y, 10, 24) for x, y in sites],
            "RFP": [cell(x + 3, y - 2, 10, 24) for x, y in sites],
            "Sox9": [cell(x - 4, y + 3, 10, 6) for x, y in sites],
            "Olig2": [cell(x + 5, y - 4, 10, 6) for x, y in sites],
        }
        shifts, scores = compute_tile_channel_shifts(
            cells, ["GFP", "RFP"], ["Sox9", "Olig2"], 10, 5,
            bin_size=2, xy_range_px=10, z_range_slices=1,
            fine_xy_px=3, fine_z_slices=1,
            max_center_dist_ratio=.5, containment_z_pad=0,
            tf_align_mode="sequential_joint")
        expected = {"RFP": (-3, 2, 0), "Sox9": (4, -3, 0),
                    "Olig2": (-5, 4, 0)}
        for channel, target in expected.items():
            with self.subTest(channel=channel):
                self.assertTrue(all(
                    abs(shifts[channel][axis] - target[axis]) <= 1
                    for axis in range(3)))
                self.assertGreater(scores[channel], 0)

    def test_rebase_keeps_relative_positions_with_non_gfp_frame(self):
        measured = {"GFP": (0, 0, 0), "RFP": (3, 0, 0),
                    "Sox9": (5, -2, 1), "Olig2": (-4, 2, -1)}
        framed = rebase_shifts(measured, "Olig2")
        self.assertEqual(framed["Olig2"], (0, 0, 0))
        self.assertEqual(framed["GFP"], (4, -2, 1))
        self.assertEqual(
            tuple(framed["Sox9"][i] - framed["RFP"][i] for i in range(3)),
            (2, -2, 1))


if __name__ == "__main__":
    unittest.main()
