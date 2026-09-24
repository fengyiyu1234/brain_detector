"""Alignment reference and stitching reference may be different channels."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.core.point_cloud_aligner import (
    align_tile, rebase_shifts, resolve_align_settings,
    validate_alignment_frame, validate_stitching_xml_frame,
    validate_cached_geometry,
)


class AlignmentFrameTests(unittest.TestCase):
    def setUp(self):
        self.routing = [
            {"id": "GFP", "type": "soma"},
            {"id": "RFP", "type": "soma"},
            {"id": "Olig2", "type": "tf"},
            {"id": "Sox9", "type": "tf"},
        ]
        self.gfp_shifts = {
            "GFP": (0, 0, 0), "RFP": (4, 1, -2),
            "Olig2": (-7, 3, 5), "Sox9": (2, -3, 1),
        }

    def _settings(self, reference, frame):
        config = {
            "pre_align_params": {"reference_channel": reference,
                                 "tf_align_mode": "direct"},
            "stitching_reference_channel": frame,
            "detection_params": {},
        }
        return resolve_align_settings(config, self.routing)

    def test_rebased_offsets_are_reference_independent(self):
        gfp = self.gfp_shifts
        rfp = {ch: tuple(value[i] - gfp["RFP"][i] for i in range(3))
               for ch, value in gfp.items()}
        for frame in ("Olig2", "GFP", "RFP"):
            with self.subTest(frame=frame):
                self.assertEqual(rebase_shifts(gfp, frame),
                                 rebase_shifts(rfp, frame))
                self.assertEqual(rebase_shifts(gfp, frame)[frame], (0, 0, 0))

    def test_align_tile_writes_scores_and_frame_offsets_for_both_refs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tile = root / "tile_a"
            tile.mkdir()
            (tile / "slice_5.tif").touch()
            raw = root / "raw"
            raw.mkdir()
            row = [["slice_5", 10, 20, 18, 28, "nucleus", 0.83, 100, 5]]
            cols = ["slice_name", "x1", "y1", "x2", "y2",
                    "class", "score", "mean", "z"]
            for ch in self.routing:
                pd.DataFrame(row, columns=cols).to_csv(
                    raw / f"tile_a_{ch['id']}_result.csv", index=False)

            expected = rebase_shifts(self.gfp_shifts, "Olig2")
            for reference in ("GFP", "RFP"):
                with self.subTest(reference=reference):
                    out = root / reference
                    ref_offset = self.gfp_shifts[reference]
                    measured = {
                        ch: tuple(v[i] - ref_offset[i] for i in range(3))
                        for ch, v in self.gfp_shifts.items()
                    }
                    with patch("src.core.z_linker.run_z_linker",
                               return_value=([], [])), patch(
                        "src.core.point_cloud_aligner.compute_tile_channel_shifts",
                        return_value=(measured, {}),
                    ):
                        self.assertEqual(
                            align_tile(str(tile), str(raw), str(out),
                                       self.routing,
                                       self._settings(reference, "Olig2")),
                            [])
                    saved = json.loads(
                        (out / "tile_a_offsets.json").read_text(encoding="utf-8"))
                    for ch in measured:
                        self.assertEqual(
                            tuple(saved[ch][k] for k in ("dx", "dy", "dz")),
                            expected[ch])
                        aligned = pd.read_csv(out / f"tile_a_{ch}_result.csv")
                        self.assertAlmostEqual(aligned.iloc[0]["score"], 0.83)
                        self.assertEqual(aligned.iloc[0]["x1"], 10 + expected[ch][0])
                        self.assertEqual(aligned.iloc[0]["z"], 5 + expected[ch][2])
                    validate_alignment_frame(
                        str(out), ["tile_a"], list(measured), "Olig2")
                    with self.assertRaisesRegex(ValueError, "not in GFP frame"):
                        validate_alignment_frame(
                            str(out), ["tile_a"], list(measured), "GFP")

    def test_existing_cache_rejects_reference_or_xml_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            align = root / "0_channel_alignment"
            align.mkdir()
            (align / "tile_a_offsets.json").write_text("{}", encoding="utf-8")
            global_dir = root / "2_global_2d_raw"
            global_dir.mkdir()
            (global_dir / "GFP_2d_global.csv").touch()
            previous = {
                "channels_routing": self.routing,
                "pre_align_params": {"reference_channel": "GFP"},
                "paths": {"pATHXML": "/some/xml_merging_Olig2.xml"},
            }
            current = {
                **previous,
                "stitching_reference_channel": "Olig2",
            }
            validate_cached_geometry(
                previous, current, tmp, self._settings("GFP", "Olig2"))
            with self.assertRaisesRegex(ValueError, "Alignment reference changed"):
                validate_cached_geometry(
                    previous, current, tmp, self._settings("RFP", "Olig2"))
            changed_xml = {
                **current,
                "paths": {"pATHXML": "/other/xml_merging_Olig2.xml"},
            }
            with self.assertRaisesRegex(ValueError, "Stitching geometry"):
                validate_cached_geometry(
                    previous, changed_xml, tmp, self._settings("GFP", "Olig2"))
    def test_named_xml_must_match_stitching_frame(self):
        validate_stitching_xml_frame("xml_merging_Olig2.xml", "Olig2")
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_stitching_xml_frame("xml_merging_RFP.xml", "Olig2")


if __name__ == "__main__":
    unittest.main()