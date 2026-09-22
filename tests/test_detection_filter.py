import unittest

import pandas as pd

from src.core.detection_filter import filter_detection_df, resolve_filter_params, validate_filter_params


def frame(rows):
    return pd.DataFrame(rows, columns=[
        "slice_name", "x1", "y1", "x2", "y2", "class", "score", "mean", "z",
    ])


class DetectionFilterTests(unittest.TestCase):
    def test_ordered_percentiles_and_input_stability(self):
        source = frame([
            ["a", 0, 0, 1, 1, "n", .9, 1, 1],
            ["a", 0, 0, 2, 2, "n", .8, 8, 1],
            ["a", 0, 0, 3, 3, "n", .7, 3, 2],
            ["a", 0, 0, 4, 4, "n", .6, 9, 2],
        ])
        original = source.copy(deep=True)
        params = {
            "bbox_min": None, "bbox_max": None, "bbox_max_aspect_ratio": None,
            "bbox_area_pct_min": 25, "bbox_area_pct_max": 50,
            "bbox_mean_pct_min": 50, "bbox_mean_min": None,
            "nms_containment_thresh": None,
        }
        result, stats = filter_detection_df(source, params, return_stats=True)
        pd.testing.assert_frame_equal(source, original)
        self.assertEqual(result.index.tolist(), [1])
        self.assertEqual(stats["thresholds"]["area_pct_min"], 3.25)
        self.assertEqual(stats["thresholds"]["area_pct_max"], 9.0)
        self.assertEqual(stats["thresholds"]["mean_pct_min"], 5.5)

    def test_containment_only_same_z_and_high_score_wins(self):
        source = frame([
            ["a", 0, 0, 10, 10, "n", .9, 3, 1],
            ["a", 1, 1, 2, 2, "n", .8, 3, 1],
            ["a", 1, 1, 2, 2, "n", .7, 3, 2],
        ])
        params = {key: None for key in (
            "bbox_min", "bbox_max", "bbox_max_aspect_ratio", "bbox_area_pct_min",
            "bbox_area_pct_max", "bbox_mean_pct_min", "bbox_mean_min",
            "nms_containment_thresh",
        )}
        params["nms_containment_thresh"] = .9
        result = filter_detection_df(source, params)
        self.assertEqual(result.index.tolist(), [0, 2])

    def test_channel_override_isolated_and_null_disables(self):
        config = {
            "detection_params": {"stardist": {"bbox_min": 8, "bbox_max": 22}},
            "channels_routing": [
                {"id": "Sox9", "model": "stardist"},
                {"id": "Olig2", "model": "stardist"},
            ],
            "channel_filter_overrides": {"Olig2": {"bbox_min": None, "bbox_max": 30}},
        }
        self.assertEqual(resolve_filter_params(config, "Sox9")["bbox_min"], 8)
        self.assertIsNone(resolve_filter_params(config, "Olig2")["bbox_min"])
        self.assertEqual(resolve_filter_params(config, "Olig2")["bbox_max"], 30)

    def test_invalid_params_and_missing_columns_fail(self):
        with self.assertRaisesRegex(ValueError, "bbox_min"):
            validate_filter_params({"bbox_min": 9, "bbox_max": 8})
        with self.assertRaisesRegex(ValueError, "missing required"):
            filter_detection_df(pd.DataFrame({"x1": [0]}), {})


if __name__ == "__main__":
    unittest.main()

