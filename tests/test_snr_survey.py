"""Threshold arithmetic of src/utils/snr_survey.py."""
import unittest

import numpy as np

from src.utils.snr_survey import matched_threshold, removed_share, sample_name


class SnrSurveyTest(unittest.TestCase):
    def test_removed_share_keeps_boxes_without_snr(self):
        snr = np.array([1.0, 3.5, 4.0, 9.0, np.nan])
        self.assertAlmostEqual(removed_share(snr, 4), 2 / 5)   # NaN is kept, as Stage 2.25

    def test_matched_threshold_removes_the_reference_share(self):
        snr = np.array([2.0, 3.0, 5.0, 6.0, 8.0, 9.0, 10.0, 12.0, np.nan, np.nan])
        t = matched_threshold(snr, 0.3)
        self.assertEqual(t, 6.0)
        self.assertAlmostEqual(removed_share(snr, t), 0.3)
        self.assertTrue(np.isnan(matched_threshold(np.array([]), 0.3)))

    def test_sample_name(self):
        self.assertEqual(sample_name("config/TSC2/config_TSC_FF11.json"), "TSC_FF11")


if __name__ == "__main__":
    unittest.main()
