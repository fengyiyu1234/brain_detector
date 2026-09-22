import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class RefilterCliTests(unittest.TestCase):
    def test_dry_run_and_candidate_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "results"
            raw = results / "1_tile_2d_raw"
            raw.mkdir(parents=True)
            config = {
                "paths": {"pATHRESULT": str(results)},
                "pipeline_mode": "post_align",
                "channels_routing": [{"id": "Olig2", "model": "stardist", "active": True}],
                "detection_params": {"stardist": {
                    "bbox_min": 2, "bbox_max": None, "bbox_max_aspect_ratio": None,
                    "bbox_area_pct_min": None, "bbox_area_pct_max": None,
                    "bbox_mean_pct_min": None, "bbox_mean_min": None,
                    "nms_containment_thresh": None,
                }},
                "channel_filter_overrides": {"Olig2": {}},
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            pd.DataFrame([
                ["s", 0, 0, 1, 1, "nucleus", 1, 2, 1],
                ["s", 0, 0, 3, 3, "nucleus", 1, 2, 1],
            ], columns=["slice_name", "x1", "y1", "x2", "y2", "class", "score", "mean", "z"]
            ).to_csv(raw / "tile_Olig2_result.csv", index=False)
            script = Path(__file__).parents[1] / "scripts" / "refilter_detections.py"
            dry = subprocess.run(
                [sys.executable, str(script), "--config", str(config_path), "--dry-run"],
                text=True, capture_output=True, check=True,
            )
            self.assertIn("no files were written", dry.stdout)
            output = root / "candidate"
            subprocess.run(
                [sys.executable, str(script), "--config", str(config_path),
                 "--channels", "Olig2", "--output-dir", str(output)],
                text=True, capture_output=True, check=True,
            )
            self.assertTrue((output / "tile_Olig2_result.csv").is_file())
            self.assertTrue((output / "refilter_manifest.json").is_file())
            self.assertEqual(len(pd.read_csv(output / "tile_Olig2_result.csv")), 1)


if __name__ == "__main__":
    unittest.main()

