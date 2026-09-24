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


    def test_score_min_keeps_threshold_boundary_and_counts_removals(self):
        source = frame([
            ["a", 0, 0, 2, 2, "n", .19, 1, 1],
            ["a", 0, 0, 2, 2, "n", .20, 1, 1],
            ["a", 0, 0, 2, 2, "n", .30, 1, 1],
            ["a", 0, 0, 2, 2, "n", .80, 1, 1],
        ])
        original = source.copy(deep=True)
        result, stats = filter_detection_df(source, {"score_min": .30}, return_stats=True)
        self.assertEqual(result.score.tolist(), [.30, .80])
        self.assertEqual(result.index.tolist(), [2, 3])
        self.assertEqual(stats["removed"]["score_min"], 2)
        self.assertEqual(stats["removed_total"], 2)
        pd.testing.assert_frame_equal(source, original)

    def test_score_min_null_and_empty_frame_are_noops(self):
        source = frame([["a", 0, 0, 2, 2, "n", .2, 1, 1],
                        ["a", 0, 0, 2, 2, "n", .8, 1, 1]])
        result, stats = filter_detection_df(source, {"score_min": None}, return_stats=True)
        self.assertEqual(result.index.tolist(), source.index.tolist())
        self.assertEqual(stats["removed"]["score_min"], 0)
        empty, empty_stats = filter_detection_df(source.iloc[:0], {"score_min": .5}, return_stats=True)
        self.assertTrue(empty.empty)
        self.assertEqual(empty_stats["removed"]["score_min"], 0)

    def test_score_min_validation(self):
        for invalid in (-.01, 1.01, "0.3", True, float("nan"), float("inf")):
            with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, "score_min"):
                validate_filter_params({"score_min": invalid})
        validate_filter_params({"score_min": 0})
        validate_filter_params({"score_min": 1})
        validate_filter_params({"score_min": None})

    def test_score_min_channel_override_and_null_disable(self):
        config = {
            "detection_params": {"stardist": {"score_min": .25}},
            "channels_routing": [{"id": "Olig2", "model": "stardist"},
                                 {"id": "Sox9", "model": "stardist"}],
            "channel_filter_overrides": {"Olig2": {"score_min": .35}},
        }
        self.assertEqual(resolve_filter_params(config, "Olig2")["score_min"], .35)
        self.assertEqual(resolve_filter_params(config, "Sox9")["score_min"], .25)
        config["channel_filter_overrides"]["Olig2"]["score_min"] = None
        self.assertIsNone(resolve_filter_params(config, "Olig2")["score_min"])

    def test_score_filter_precedes_containment_nms(self):
        source = frame([
            ["a", 0, 0, 10, 10, "n", .9, 1, 1],
            ["a", 1, 1, 2, 2, "n", .2, 1, 1],
            ["a", 1, 1, 2, 2, "n", .8, 1, 1],
        ])
        result, stats = filter_detection_df(
            source, {"score_min": .5, "nms_containment_thresh": .9}, return_stats=True)
        self.assertEqual(result.index.tolist(), [0])
        self.assertEqual(stats["removed"]["score_min"], 1)
        self.assertEqual(stats["removed"]["containment_nms"], 1)
        self.assertEqual(stats["removed_total"], sum(stats["removed"].values()))


if __name__ == "__main__":
    unittest.main()

