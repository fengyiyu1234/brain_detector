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
                    "score_min": .5, "bbox_min": None, "bbox_max": None, "bbox_max_aspect_ratio": None,
                    "bbox_area_pct_min": None, "bbox_area_pct_max": None,
                    "bbox_mean_pct_min": None, "bbox_mean_min": None,
                    "nms_containment_thresh": None,
                }},
                "channel_filter_overrides": {"Olig2": {}},
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            pd.DataFrame([
                ["s", 0, 0, 3, 3, "nucleus", .4, 2, 1],
                ["s", 0, 0, 3, 3, "nucleus", .5, 2, 1],
                ["s", 0, 0, 4, 4, "nucleus", .8, 2, 1],
            ], columns=["slice_name", "x1", "y1", "x2", "y2", "class", "score", "mean", "z"]
            ).to_csv(raw / "tile_Olig2_result.csv", index=False)
            script = Path(__file__).parents[1] / "scripts" / "refilter_detections.py"
            dry = subprocess.run(
                [sys.executable, str(script), "--config", str(config_path), "--dry-run"],
                text=True, capture_output=True, check=True,
            )
            self.assertIn("no files were written", dry.stdout)
            output = root / "candidate"
            self.assertFalse(output.exists())
            subprocess.run(
                [sys.executable, str(script), "--config", str(config_path),
                 "--channels", "Olig2", "--output-dir", str(output)],
                text=True, capture_output=True, check=True,
            )
            self.assertTrue((output / "tile_Olig2_result.csv").is_file())
            self.assertTrue((output / "refilter_manifest.json").is_file())
            result = pd.read_csv(output / "tile_Olig2_result.csv")
            self.assertEqual(result.score.tolist(), [.5, .8])
            manifest = json.loads((output / "refilter_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["filter_schema_version"], "2")
            record = manifest["records"][0]
            self.assertEqual(record["params"]["score_min"], .5)
            self.assertEqual(record["removed_by_step"]["score_min"], 1)
            self.assertEqual((record["before"], record["after"], record["removed_total"]), (3, 2, 1))
            subprocess.run(
                [sys.executable, str(script), "--config", str(config_path), "--channels", "Olig2",
                 "--output-dir", str(output), "--overwrite"],
                text=True, capture_output=True, check=True,
            )
            self.assertEqual(len(pd.read_csv(output / "tile_Olig2_result.csv")), 2)


if __name__ == "__main__":
    unittest.main()

