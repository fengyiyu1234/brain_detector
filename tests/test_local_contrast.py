"""Stage 2.2 ring SNR and its use by the Stage 2.25 filter."""

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from scripts.run_inference import prepare_alignment_inputs, run_contrast_stage
from src.core.local_contrast import (
    CONTRAST_SUBDIR, resolve_contrast_filter, run_contrast_tile, contrast_settings, slice_ring_stats,
)

SOX9 = {"id": "Sox9", "type": "tf", "model": "stardist", "dir_key": "sox9_dir"}
GFP = {"id": "GFP", "type": "soma", "model": "yolo", "dir_key": "gfp_dir"}


def square(x, y, r):
    return np.array([[x - r, y - r], [x + r, y - r], [x + r, y + r], [x - r, y + r]], float)


def synthetic_slice(seed=0):
    """Background 1000 +- texture; nuclei at 30 / 70 / 110 px with means 1600 / 1030 / 1600."""
    rng = np.random.default_rng(seed)
    image = rng.normal(1000, 50, (64, 140)).astype(np.float32)
    centers, levels = (30, 70, 110), (1600, 1030, 1600)
    for x, level in zip(centers, levels):
        image[27:34, x - 3:x + 4] = level
    boxes = np.array([[x - 3, 27, x + 4, 34] for x in centers])
    return image, boxes, [square(x, 30, 3) for x in centers], np.array(levels, float)


class SliceRingStatsTests(unittest.TestCase):
    def test_bright_nucleus_has_high_snr_and_dim_one_low(self):
        image, boxes, polygons, means = synthetic_slice()
        stats = slice_ring_stats(image, boxes, polygons)
        snr = (means - stats[:, 0]) / stats[:, 1]
        self.assertTrue(np.all(np.abs(stats[:, 0] - 1000) < 15))
        self.assertGreater(snr[0], 8)
        self.assertLess(snr[1], 2)

    def test_ring_excludes_neighbouring_masks(self):
        image = np.full((40, 60), 100, np.float32)
        image[10:20, 10:20] = 900                     # own nucleus
        image[10:20, 23:33] = 5000                    # neighbour 3 px away, inside the ring
        boxes = np.array([[10, 10, 20, 20], [23, 10, 33, 20]])
        polygons = [np.array([[10, 10], [19.9, 10], [19.9, 19.9], [10, 19.9]]),
                    np.array([[23, 10], [32.9, 10], [32.9, 19.9], [23, 19.9]])]
        stats = slice_ring_stats(image, boxes, polygons)
        self.assertEqual(stats[0, 0], 100)
        self.assertGreater(stats[0, 2], 0)

    def test_missing_polygon_uses_box_and_tiny_ring_is_nan(self):
        image = np.full((12, 12), 100, np.float32)
        stats = slice_ring_stats(image, np.array([[1, 1, 11, 11]]), [None])
        self.assertTrue(np.isnan(stats[0, 0]))
        self.assertLess(stats[0, 2], 10)


class ResolveContrastFilterTests(unittest.TestCase):
    def config(self, spec):
        return {"channels_routing": [GFP, SOX9], "contrast_filter": spec}

    def test_valid_and_disabled(self):
        self.assertEqual(resolve_contrast_filter(self.config({"Sox9": {"snr_min": 5}})),
                         {"Sox9": {"snr_min": 5.0}})
        self.assertEqual(resolve_contrast_filter(self.config({"Sox9": {"snr_min": None}})), {})
        self.assertEqual(resolve_contrast_filter({"channels_routing": [SOX9]}), {})

    def test_rejects_bad_specs(self):
        for spec in ({"Olig2": {"snr_min": 5}}, {"GFP": {"snr_min": 5}},
                     {"Sox9": {"snr": 5}}, {"Sox9": {"snr_min": True}}, {"Sox9": {"snr_min": float("nan")}}):
            with self.assertRaises(ValueError):
                resolve_contrast_filter(self.config(spec))
        fused = dict(SOX9, double_exposure=True)
        with self.assertRaises(ValueError):
            resolve_contrast_filter({"channels_routing": [fused], "contrast_filter": {"Sox9": {"snr_min": 5}}})


class ContrastPipelineTests(unittest.TestCase):
    def write_tile(self, root):
        raw = root / "1_2d_raw"
        raw.mkdir()
        rows, instances = [], []
        for z in (1, 2):
            image, boxes, polygons, means = synthetic_slice(seed=z)
            path = root / f"img_z{z}.tif"
            cv2.imwrite(str(path), np.clip(image, 0, 65535).astype(np.uint16))
            for i, (box, polygon, mean) in enumerate(zip(boxes, polygons, means)):
                det = f"d{z}_{i}"
                rows.append([f"s{z}", *box, "nucleus", 0.9, mean, z, det, str(path)])
                instances.append([det, json.dumps(polygon.tolist())])
        pd.DataFrame(rows, columns=["slice_name", "x1", "y1", "x2", "y2", "class", "score", "mean", "z",
                                    "detection_id", "source_image"]).to_csv(raw / "t_Sox9_result.csv", index=False)
        pd.DataFrame(instances, columns=["detection_id", "polygon_xy_json"]).to_csv(
            raw / "t_Sox9_instances.csv", index=False)
        return raw

    def test_contrast_checkpoint_and_snr_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = self.write_tile(root)
            contrast_dir = raw / CONTRAST_SUBDIR
            config = {"channels_routing": [SOX9], "detection_params": {"stardist": {}},
                      "contrast_filter": {"Sox9": {"snr_min": 5}}}
            run_contrast_stage(["t"], resolve_contrast_filter(config), str(raw))
            table = pd.read_csv(contrast_dir / "t_Sox9_contrast.csv")
            self.assertEqual(table.detection_id.tolist(), [f"d{z}_{i}" for z in (1, 2) for i in range(3)])
            self.assertEqual((table.snr >= 5).tolist(), [True, False, True] * 2)
            self.assertEqual(run_contrast_tile("t", "Sox9", str(raw), str(contrast_dir),
                                               contrast_settings())[1], -1)   # reused

            filtered = root / "2_2d_filtered"
            prepare_alignment_inputs(config, ["t"], [SOX9], str(raw), str(filtered))
            kept = pd.read_csv(filtered / "t_Sox9_result.csv")
            rejected = pd.read_csv(filtered / "t_Sox9_rejected.csv")
            self.assertEqual(kept.detection_id.tolist(), ["d1_0", "d1_2", "d2_0", "d2_2"])
            self.assertTrue((kept.snr >= 5).all())
            self.assertEqual(rejected.rejection_reason.tolist(), ["snr_min", "snr_min"])
            manifest = filtered / "t_Sox9_filter_manifest.json"
            stamp = manifest.stat().st_mtime_ns
            prepare_alignment_inputs(config, ["t"], [SOX9], str(raw), str(filtered))
            self.assertEqual(manifest.stat().st_mtime_ns, stamp)              # checkpoint reused

            config["contrast_filter"]["Sox9"]["snr_min"] = 0.5
            prepare_alignment_inputs(config, ["t"], [SOX9], str(raw), str(filtered))
            self.assertEqual(len(pd.read_csv(filtered / "t_Sox9_result.csv")), 6)

            del config["contrast_filter"]
            prepare_alignment_inputs(config, ["t"], [SOX9], str(raw), str(filtered))
            plain = pd.read_csv(filtered / "t_Sox9_result.csv")
            self.assertEqual(len(plain), 6)
            self.assertNotIn("snr", plain.columns)


if __name__ == "__main__":
    unittest.main()
