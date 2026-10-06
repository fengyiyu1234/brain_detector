"""Filtered tile detections are the input to pre-align checkpoints."""

import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.run_inference import prepare_alignment_inputs, validate_global_checkpoints
from src.core.point_cloud_aligner import tile_alignment_done


COLUMNS = ["slice_name", "x1", "y1", "x2", "y2", "class", "score", "mean", "z"]


class PipelineFilterInputTests(unittest.TestCase):
    def test_prefilter_removes_loose_detections_and_refreshes_changed_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "1_2d_raw"
            filtered = Path(tmp) / "2_2d_filtered"
            raw.mkdir()
            source = raw / "tile_a_GFP_result.csv"
            rows = [
                ["z1", 0, 0, 10, 10, "neuron", .2, 50, 1],
                ["z1", 20, 20, 30, 30, "neuron", .9, 50, 1],
            ]
            pd.DataFrame(rows, columns=COLUMNS).to_csv(source, index=False)
            channel = {"id": "GFP", "type": "soma", "model": "yolo"}
            config = {
                "channels_routing": [channel],
                "detection_params": {"yolo": {"score_min": .5}},
            }
            prepare_alignment_inputs(config, ["tile_a"], [channel], str(raw), str(filtered))
            target = filtered / source.name
            self.assertEqual(pd.read_csv(target).score.tolist(), [.9])
            self.assertEqual(len(pd.read_csv(source)), 2)

            rows[0][6] = .8
            pd.DataFrame(rows, columns=COLUMNS).to_csv(source, index=False)
            newer = os.path.getmtime(target) + 10
            os.utime(source, (newer, newer))
            prepare_alignment_inputs(config, ["tile_a"], [channel], str(raw), str(filtered))
            self.assertEqual(pd.read_csv(target).score.tolist(), [.8, .9])
            config["detection_params"]["yolo"]["score_min"] = .85
            prepare_alignment_inputs(config, ["tile_a"], [channel], str(raw), str(filtered))
            self.assertEqual(pd.read_csv(target).score.tolist(), [.9])

    def test_global_checkpoint_rejects_newer_filtered_tile_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            derived = {
                "pATH_DET_FILTERED": str(root / "filtered"),
                "pATH_GLOBAL_2D": str(root / "global"),
                "pATH_CHANNEL_3D": str(root / "tracked"),
                "pATH_COLOCALIZATION": str(root / "coloc"),
            }
            for folder in derived.values():
                Path(folder).mkdir()
            source = Path(derived["pATH_DET_FILTERED"]) / "tile_a_GFP_result.csv"
            global_csv = Path(derived["pATH_GLOBAL_2D"]) / "GFP_2d_global.csv"
            source.touch()
            global_csv.touch()
            older = os.path.getmtime(global_csv) + 10
            os.utime(source, (older, older))
            with self.assertRaisesRegex(RuntimeError, "Stale Stage 3 checkpoint"):
                validate_global_checkpoints(
                    derived, ["tile_a"], [{"id": "GFP"}])

    def test_alignment_checkpoint_is_stale_when_filtered_input_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "filtered"
            aligned = Path(tmp) / "aligned"
            source.mkdir()
            aligned.mkdir()
            name = "tile_a_GFP_result.csv"
            row = [["z1", 0, 0, 10, 10, "neuron", .9, 50, 1]]
            pd.DataFrame(row, columns=COLUMNS).to_csv(source / name, index=False)
            pd.DataFrame(row, columns=COLUMNS).to_csv(aligned / name, index=False)
            (aligned / "tile_a_offsets.json").write_text("{}", encoding="utf-8")
            old = os.path.getmtime(source / name) + 10
            os.utime(aligned / name, (old, old))
            self.assertTrue(tile_alignment_done(
                "tile_a", str(source), str(aligned), [{"id": "GFP"}]))
            newer = old + 10
            os.utime(source / name, (newer, newer))
            self.assertFalse(tile_alignment_done(
                "tile_a", str(source), str(aligned), [{"id": "GFP"}]))


if __name__ == "__main__":
    unittest.main()
