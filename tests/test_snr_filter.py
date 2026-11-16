"""Stage 2.25/2.75 SNR filter: slice noise, per-slice threshold lowering, plain form unchanged."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.core import local_contrast
from src.core.snr_filter import apply_snr_filter, slice_thresholds, snr_filter_signature

ADAPT = {"snr_min": 5.0, "noise": "ring", "max_removed_frac": 0.35, "snr_step": 1.0, "snr_floor": 2.0}


def stats_for(n):
    return {"before": n, "after": n, "removed": {}, "removed_total": 0}


class SnrFilterTests(unittest.TestCase):
    def write(self, root, slices):
        """slices: {image: [(ring_mad, mean), ...]} with ring_bg 0; snr column = mean / ring_mad."""
        rows = [(f"{image}_{i}", mad, mean, image)
                for image, boxes in slices.items() for i, (mad, mean) in enumerate(boxes)]
        frame = pd.DataFrame(rows, columns=["detection_id", "ring_mad", "mean", "source_image"])
        raw, contrast = root / "t_Sox9_result.csv", root / "t_Sox9_contrast.csv"
        frame[["detection_id", "mean", "source_image"]].to_csv(raw, index=False)
        frame.assign(ring_bg=0.0, ring_n=40, snr=frame["mean"] / frame.ring_mad)[
            ["detection_id", "ring_bg", "ring_mad", "ring_n", "snr"]].to_csv(contrast, index=False)
        return frame[["detection_id"]], raw, contrast

    def run_filter(self, slices, params):
        with tempfile.TemporaryDirectory() as tmp:
            filtered, raw, contrast = self.write(Path(tmp), slices)
            rejected = pd.DataFrame({"detection_id": ["x"], "rejection_reason": ["score_min"]})
            return apply_snr_filter(filtered, rejected, stats_for(len(filtered)), contrast, raw, params)

    def test_slice_noise_keeps_crowded_bright_box(self):
        slices = {"a": [(10, 80)] * 9 + [(100, 80)]}          # last ring is noisy: local SNR 0.8
        kept, rejected, _ = self.run_filter(slices, {"snr_min": 5.0})
        self.assertNotIn("a_9", kept.detection_id.tolist())
        params = dict(ADAPT, noise="slice", max_removed_frac=None)
        kept, rejected, stats = self.run_filter(slices, params)
        self.assertEqual(len(kept), 10)
        self.assertTrue(np.allclose(kept.snr, 8.0))
        self.assertTrue(np.allclose(kept.slice_noise, 10.0))
        self.assertEqual(stats["removed"]["snr_min"], 0)

    def test_threshold_drops_per_slice_until_share_ok_or_floor(self):
        slices = {"ok": [(1, 1)] * 2 + [(1, 6)] * 8,                       # 20% at 5: stays 5
                  "drop": [(1, 1)] * 3 + [(1, 4.5)] * 3 + [(1, 6)] * 4,    # 60% at 5, 30% at 4
                  "floor": [(1, 1)] * 5 + [(1, 6)] * 5}                    # 50% down to the floor
        kept, rejected, stats = self.run_filter(slices, ADAPT)
        by_slice = kept.assign(s=kept.detection_id.str.split("_").str[0]).groupby("s")
        self.assertEqual(by_slice.snr_threshold.first().to_dict(), {"drop": 4.0, "floor": 2.0, "ok": 5.0})
        self.assertEqual(by_slice.size().to_dict(), {"drop": 7, "floor": 5, "ok": 8})
        self.assertEqual(stats["removed"]["snr_min"], 10)
        removed = rejected[rejected.rejection_reason == "snr_min"]
        self.assertEqual(len(removed), 10)
        self.assertTrue((removed.snr < removed.snr_threshold).all())

    def test_nan_snr_counts_as_kept(self):
        thresholds = slice_thresholds(np.array([1, np.nan, np.nan, 6.0]), np.array(["s"] * 4), ADAPT)
        self.assertEqual(thresholds["s"], 5.0)                 # 1/4 removed <= 35%

    def test_plain_form_matches_previous_filter_and_signature(self):
        slices = {"a": [(10, 80)] * 9 + [(100, 80)], "b": [(1, 1)] * 6 + [(1, 6)] * 4}
        with tempfile.TemporaryDirectory() as tmp:
            filtered, raw, contrast = self.write(Path(tmp), slices)
            rejected = pd.DataFrame({"detection_id": ["x"], "rejection_reason": ["score_min"]})
            new = apply_snr_filter(filtered, rejected, stats_for(len(filtered)), contrast, raw,
                                   {"snr_min": 5.0})
            old = local_contrast.apply_snr_filter(filtered, rejected, stats_for(len(filtered)), contrast, 5.0)
            for a, b in zip(new[:2], old[:2]):
                pd.testing.assert_frame_equal(a, b)
            self.assertEqual(new[2], old[2])
            settings = local_contrast.contrast_settings()
            self.assertEqual(snr_filter_signature(contrast, {"snr_min": 5.0}, settings),
                             local_contrast.contrast_filter_signature(contrast, {"snr_min": 5.0}, settings))
            self.assertIn("filter_code_sha256", snr_filter_signature(contrast, ADAPT, settings))


if __name__ == "__main__":
    unittest.main()
