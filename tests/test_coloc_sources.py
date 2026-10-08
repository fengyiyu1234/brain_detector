"""Source boxes retain the exact channel tracks selected by Stage 4."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.core.coloc_sources import source_3d_json, write_source_3d, write_source_boxes
from src.core.stitcher import annotate_soma_with_tf_containment


def volume(cls, cx, cy, z, box, per_z):
    return {
        "class": cls, "cx": cx, "cy": cy, "cz": z,
        "x1_3d": box[0], "y1_3d": box[1],
        "x2_3d": box[2], "y2_3d": box[3],
        "z_min": min(per_z), "z_max": max(per_z),
        "per_z_boxes": per_z, "score": 0.8, "mean": 100.0,
    }


class ColocSourcesTest(unittest.TestCase):
    def test_matched_tf_source_and_per_z_boxes(self):
        soma = volume("neuron_GFP_RFP", 20, 20, 10, (0, 0, 40, 40),
                      {9: (1, 1, 39, 39), 10: (0, 0, 40, 40)})
        gfp = volume("neuron_GFP", 20, 20, 10, (1, 1, 39, 39),
                     {9: (1, 1, 39, 39), 10: (2, 2, 38, 38)})
        rfp = volume("neuron_RFP", 20, 20, 10, (0, 0, 40, 40),
                     {10: (0, 0, 40, 40)})
        sox9 = volume("nucleus_Sox9", 20, 20, 10, (15, 15, 25, 25),
                      {10: (15, 15, 25, 25)})
        distant = volume("nucleus_Sox9", 100, 100, 10, (95, 95, 105, 105),
                         {10: (95, 95, 105, 105)})
        soma["source_tracks"] = [("GFP", gfp), ("RFP", rfp)]

        annotate_soma_with_tf_containment(
            [soma], [sox9, distant], z_pad=1, source_channel="Sox9")
        self.assertEqual(soma["class"], "neuron_GFP_RFP_Sox9")
        self.assertEqual([ch for ch, _ in soma["source_tracks"]],
                         ["GFP", "RFP", "Sox9"])
        self.assertIs(soma["source_tracks"][-1][1], sox9)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sources.csv"
            write_source_boxes(path, [soma])
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 4)
        self.assertEqual({row["coloc_id"] for row in rows}, {"0"})
        self.assertEqual([row["channel"] for row in rows],
                         ["GFP", "GFP", "RFP", "Sox9"])
        self.assertEqual((rows[-1]["z"], rows[-1]["x1"], rows[-1]["y1"]),
                         ("10", "15.0", "15.0"))

        source_3d = json.loads(source_3d_json(soma))
        self.assertEqual(set(source_3d), {"GFP", "RFP", "Sox9"})
        self.assertEqual(source_3d["Sox9"][0]["z_min"], 10)
        self.assertEqual(source_3d["RFP"][0]["x2_3d"], 40.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sources_3d.csv"
            write_source_3d(path, [soma])
            with path.open(newline="", encoding="utf-8") as handle:
                flat = list(csv.DictReader(handle))
        self.assertEqual(len(flat), 3)
        self.assertEqual(flat[-1]["channel"], "Sox9")
        self.assertEqual(flat[-1]["x1_3d"], "15.0")

    def test_multiple_tracks_in_one_channel_are_preserved(self):
        a = volume("nucleus_Sox9", 20, 20, 10, (15, 15, 25, 25),
                   {10: (15, 15, 25, 25)})
        b = volume("nucleus_Sox9", 22, 22, 11, (16, 16, 26, 26),
                   {11: (16, 16, 26, 26)})
        soma = {"source_tracks": [("Sox9", a), ("Sox9", b), ("Olig2", a)]}
        sources = json.loads(source_3d_json(soma))
        self.assertEqual([track["source_track"] for track in sources["Sox9"]],
                         [0, 1])
        self.assertEqual(sources["Olig2"][0]["source_track"], 2)


if __name__ == "__main__":
    unittest.main()
