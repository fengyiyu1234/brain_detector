"""snr_filter_preview keeps the same boxes as the pipeline's regular + SNR filter."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.detection_filter import filter_detection_df
from src.core.snr_filter import apply_snr_filter
from src.utils.snr_filter_preview import apply_filters

PARAMS = {"bbox_min": 8, "bbox_max": 28, "bbox_max_aspect_ratio": 3, "score_min": 0.35}
SNR = {"snr_min": 5.0, "noise": "slice", "max_removed_frac": 0.35, "snr_step": 1.0, "snr_floor": 4.0}


def features(seed=0, n=300):
    rng = np.random.default_rng(seed)
    x1, y1 = rng.integers(0, 2000, n), rng.integers(0, 2000, n)
    w, h = rng.integers(5, 32, n), rng.integers(5, 32, n)
    tile = np.where(np.arange(n) < n // 2, "t1", "t2")
    z = rng.integers(10, 13, n)
    ring_bg, ring_mad = rng.uniform(900, 1100, n), rng.uniform(20, 80, n)
    ring_mad[::17] = np.nan
    return pd.DataFrame({
        "x1": x1, "y1": y1, "x2": x1 + w, "y2": y1 + h, "score": rng.uniform(0.2, 1, n),
        "mean": ring_bg + rng.uniform(0, 12, n) * 50, "z": z, "tile": tile,
        "detection_id": [f"d{i}" for i in range(n)], "ring_bg": ring_bg, "ring_mad": ring_mad,
        "snr": np.nan, "kept": True})


class PreviewMatchesPipelineTests(unittest.TestCase):
    def test_same_kept_boxes(self):
        f = features()
        preview = apply_filters(f, PARAMS, SNR)
        kept = set(preview.loc[preview.kept, "detection_id"])
        pipeline = set()
        with tempfile.TemporaryDirectory() as tmp:
            for tile, part in f.groupby("tile"):
                raw = part.assign(source_image=[f"{tile}_z{z}.tif" for z in part.z])
                raw_csv, contrast_csv = Path(tmp) / f"{tile}_raw.csv", Path(tmp) / f"{tile}_c.csv"
                raw[["detection_id", "mean", "source_image"]].to_csv(raw_csv, index=False)
                raw.assign(ring_n=40)[["detection_id", "ring_bg", "ring_mad", "ring_n", "snr"]].to_csv(
                    contrast_csv, index=False)
                filtered, stats, rejected = filter_detection_df(raw, PARAMS, return_stats=True,
                                                                return_rejected=True)
                out, _, _ = apply_snr_filter(filtered, rejected, stats, contrast_csv, raw_csv, SNR)
                pipeline |= set(out.detection_id)
        self.assertEqual(kept, pipeline)
        self.assertTrue(0 < len(kept) < preview.regular_kept.sum())


if __name__ == "__main__":
    unittest.main()
