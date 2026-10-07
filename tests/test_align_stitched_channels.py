"""End-to-end XY placement for independently merged TIFF channels."""
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from src.core.align_stitched_channels import run

try:
    from PIL import Image
except ImportError:
    Image = None


@unittest.skipIf(Image is None, "Pillow is not installed")
class AlignStitchedChannelsTests(unittest.TestCase):
    def write_xml(self, raw, marker, positions):
        tile_names = ("A", "B")
        root = ET.Element("TeraStitcher")
        ET.SubElement(root, "voxel_dims", H="1", V="1", D="1")
        stacks = ET.SubElement(root, "STACKS")
        for name, (x, y) in zip(tile_names, positions):
            tile_dir = raw / "row" / name
            tile_dir.mkdir(parents=True)
            Image.new("I;16", (8, 6)).save(tile_dir / "0000.tif")
            ET.SubElement(stacks, "Stack", DIR_NAME=f"row/{name}",
                          ABS_H=str(x), ABS_V=str(y), ABS_D="0")
        ET.ElementTree(root).write(raw / f"xml_merging_{marker}.xml")

    def test_two_channels_stack_and_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            raw = base / "raw"
            self.write_xml(raw / "640nm", "GFP", [(-2, -2), (6, -2)])
            self.write_xml(raw / "561nm", "RFP", [(0, 0), (6, 0)])
            stitched = base / "stitched"
            stitched.mkdir()
            gfp = Image.new("I;16", (8, 3))
            gfp.putpixel((1, 1), 1234)
            gfp2 = Image.new("I;16", (8, 3))
            gfp2.putpixel((2, 1), 2345)
            gfp.save(stitched / "GFP.tif", save_all=True, append_images=[gfp2])
            rfp_dir = stitched / "RFP"
            rfp_dir.mkdir()
            for index in range(2):
                rfp = Image.new("I;16", (7, 3))
                rfp.putpixel((0, 0), 5000 + index)
                rfp.save(rfp_dir / f"{index:04d}.tif")
            cfg = {
                "raw_sample_dir": str(raw),
                "stitched_images": {"GFP": str(stitched / "GFP.tif"),
                                    "RFP": str(rfp_dir)},
                "xy_downsample": 2,
                "output_dir": str(base / "aligned"),
            }
            config = base / "config.json"
            config.write_text(json.dumps(cfg), encoding="utf-8")
            run(config, dry_run=True)
            self.assertFalse((base / "aligned").exists())
            run(config)
            output = base / "aligned"
            manifest = json.loads((output / "alignment_manifest.json").read_text())
            self.assertEqual((manifest["canvas_width"], manifest["canvas_height"]), (8, 4))
            self.assertEqual((manifest["channels"]["RFP"]["left"],
                              manifest["channels"]["RFP"]["top"]), (1, 1))
            with Image.open(output / "GFP" / "GFP_z00000.tif") as image:
                self.assertEqual(image.size, (8, 4))
                self.assertEqual(image.getpixel((1, 1)), 1234)
            with Image.open(output / "RFP" / "0000.tif") as image:
                self.assertEqual(image.size, (8, 4))
                self.assertEqual(image.getpixel((1, 1)), 5000)
                self.assertEqual(image.getpixel((0, 0)), 0)


if __name__ == "__main__":
    unittest.main()


