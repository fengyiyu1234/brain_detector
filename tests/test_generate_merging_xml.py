"""Checks for XML geometry, channel paths, and incomplete input rejection."""
import csv
from contextlib import redirect_stderr
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from src.core.generate_merging_xml import generate, grid_for_tiles, xml_image_directory


class GenerateMergingXmlTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.tiles = ("1000_2000", "1000_2100", "1100_2000", "1100_2100")
        self.channels = {}
        for channel in ("GFP", "Olig2"):
            directory = self.root / channel
            self.channels[channel] = directory
            for tile in self.tiles:
                row = tile.split("_")[0]
                image_dir = directory / row / tile
                image_dir.mkdir(parents=True)
                for z in (3000, 3010):
                    (image_dir / f"{tile}_{z}.tiff").touch()
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps({
            "channels_routing": [
                {"id": ch, "dir_key": ch, "active": True} for ch in self.channels
            ],
            "stitching_reference_channel": "Olig2",
            "paths": {ch: str(path) for ch, path in self.channels.items()},
            "detection_params": {"xy_resolution_um": 0.5, "z_resolution_um": 8},
        }), encoding="utf-8")
        self.positions = self.root / "tile_positions.csv"
        fields = ["tile"] + [
            f"P_{ch}_{axis}" for ch in self.channels for axis in ("x", "y", "z")
        ] + [f"s_GFP_{axis}" for axis in ("x", "y", "z")]
        with self.positions.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for tile in self.tiles:
                v, h = map(int, tile.split("_"))
                x, y = (h - 2000) / 5, (v - 1000) / 5
                if h == 2100:
                    x += 1
                z = int(h == 2100)
                writer.writerow({
                    "tile": tile,
                    "P_Olig2_x": x, "P_Olig2_y": y, "P_Olig2_z": z,
                    "P_GFP_x": x + 1, "P_GFP_y": y + 2, "P_GFP_z": z + 3,
                    "s_GFP_x": 5, "s_GFP_y": 6, "s_GFP_z": 7,
                })

    def test_generates_templateless_channel_xmls(self):
        stderr = StringIO()
        with redirect_stderr(stderr):
            outputs = generate(self.config, self.positions)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(set(outputs), {"GFP", "Olig2"})
        for ch, path in outputs.items():
            self.assertEqual(path, self.channels[ch] / "xml_merging.xml")
            data = path.read_bytes()
            self.assertIn(b'<!DOCTYPE TeraStitcher SYSTEM "TeraStitcher.DTD">', data)
            root = ET.parse(path).getroot()
            self.assertEqual(root.attrib, {
                "volume_format": "TiledXY|2Dseries", "input_plugin": "tiff2D"
            })
            self.assertEqual(root.find("stacks_dir").get("value"),
                             xml_image_directory(self.channels[ch]))
            self.assertEqual(root.find("mdata_bin").get("value"),
                             xml_image_directory(self.channels[ch]) + "/mdata.bin")
            self.assertEqual(root.find("dimensions").attrib, {
                "stack_rows": "2", "stack_columns": "2", "stack_slices": "2"
            })
            self.assertEqual(root.find("origin").attrib, {
                "V": "0.1", "H": "0.2", "D": "0.3"
            })
            stacks = list(root.find("STACKS"))
            self.assertEqual(len(stacks), 4)
            self.assertEqual([(s.get("ROW"), s.get("COL")) for s in stacks],
                             [("0", "0"), ("0", "1"), ("1", "0"), ("1", "1")])
            self.assertEqual(stacks[0].get("DIR_NAME"), "1000/1000_2000")
            self.assertEqual(stacks[0].get("Z_RANGES"), "[0,2)")
            east = stacks[0].find("EAST_displacements/Displacement")
            self.assertEqual(east.find("H").get("displ"), "21")
            self.assertEqual(east.find("H").get("default_displ"), "20")
            self.assertEqual(east.find("D").get("displ"), "1")
            self.assertEqual(stacks[1].find("WEST_displacements/Displacement/H")
                             .get("displ"), "-21")
            self.assertEqual(len(list(stacks[0])), 4)
        self.assertEqual(ET.parse(outputs["GFP"]).getroot().find("STACKS/Stack").get("ABS_H"), "5")
        self.assertEqual(ET.parse(outputs["Olig2"]).getroot().find("STACKS/Stack").get("ABS_H"), "0")
        with self.assertRaises(FileExistsError):
            generate(self.config, self.positions)

    def test_cluster_image_paths_use_y_drive(self):
        self.assertEqual(
            xml_image_directory(
                "/rsstu/users/a/agrinba/DeepDesign/Fengyi/EGFR_brain/T70/488nm"),
            "Y:/Fengyi/EGFR_brain/T70/488nm")
        self.assertEqual(
            xml_image_directory(r"Y:\Fengyi\EGFR_brain\T70\640nm"),
            "Y:/Fengyi/EGFR_brain/T70/640nm")

    def test_rejects_missing_tile_or_inconsistent_slices(self):
        with self.assertRaisesRegex(ValueError, "Incomplete tile grid"):
            grid_for_tiles(self.tiles[:-1])
        bad = self.channels["GFP"] / "1000" / "1000_2000" / "1000_2000_3010.tiff"
        bad.unlink()
        with self.assertRaisesRegex(ValueError, "TIFF slice coordinates differ"):
            generate(self.config, self.positions)


if __name__ == "__main__":
    unittest.main()
