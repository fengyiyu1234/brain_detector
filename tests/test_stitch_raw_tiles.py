"""Direct-stitch integration test with fractional tile offsets and Z shifts."""
import json
import os
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

try:
    from PIL import Image
    from scripts.stitch_raw_tiles import (output_path_value, platform_path,
                                          resolve_path, run)
except ImportError:
    Image = None


@unittest.skipIf(Image is None, "Pillow/numpy unavailable")
class DirectStitchTests(unittest.TestCase):
    def make_channel(self, root, wavelength, marker, placements, bases):
        channel = root / wavelength
        xml = ET.Element("TeraStitcher")
        ET.SubElement(xml, "voxel_dims", H="1", V="1", D="2")
        ET.SubElement(xml, "dimensions", stack_slices="6")
        stacks = ET.SubElement(xml, "STACKS")
        for name, (x, y, z) in placements.items():
            tile_dir = channel / "row" / name
            tile_dir.mkdir(parents=True)
            for index in range(6):
                Image.new("I;16", (4, 4), bases[name] + 10 * index).save(
                    tile_dir / f"{name}_{index:04d}.tif")
            ET.SubElement(stacks, "Stack", DIR_NAME=f"row/{name}",
                          ABS_H=str(x), ABS_V=str(y), ABS_D=str(z))
        ET.ElementTree(xml).write(channel / f"xml_merging_{marker}.xml")

    def test_shared_paths_and_output_selection_on_both_platforms(self):
        cluster = ("/rsstu/users/a/agrinba/DeepDesign/Fengyi/"
                   "EGFR_brain/T70/640nm")
        local = "Y:/Fengyi/EGFR_brain/T70/640nm"
        self.assertEqual(platform_path(cluster, "nt"), local)
        self.assertEqual(platform_path(local, "posix"), cluster)
        result = resolve_path(cluster, Path.cwd())
        self.assertEqual(result, Path(local if os.name == "nt" else cluster))
        params = {"output_dir": "J:/EGFR/T70/stitched2",
                  "output_dir_hpc": cluster.rsplit("/", 1)[0] + "/stitched2"}
        self.assertEqual(output_path_value(params, "nt"), params["output_dir"])
        self.assertEqual(output_path_value(params, "posix"),
                         params["output_dir_hpc"])
        with self.assertRaisesRegex(ValueError, "output_dir_hpc"):
            platform_path("J:/EGFR/T70/stitched2", "posix")

    def test_disabled_switch_skips_before_reading_raw_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            config = base / "config.json"
            config.write_text(json.dumps({
                "direct_stitching": {"enabled": False,
                                     "output_dir": str(base / "result")},
            }), encoding="utf-8")
            run(config)
            run(config, dry_run=True)
            self.assertFalse((base / "result").exists())

    def test_project_channel_alias_and_active_visualization_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            raw = base / "raw"
            self.make_channel(raw, "640nm_3", "GFP_3",
                              {"A": (0, 0, 0)}, {"A": 100})
            config = base / "config.json"
            config.write_text(json.dumps({
                "active_sample": "sample",
                "samples": {"sample": {
                    "channels_routing": [
                        {"id": "GFP_3", "dir_key": "gfp_dir", "active": True}],
                    "paths": {"gfp_dir": str(raw / "640nm_3")},
                }},
                "direct_stitching": {
                    "enabled": True,
                    "downsample_factor": 2,
                    "output_dir": str(base / "result"),
                },
            }), encoding="utf-8")
            run(config)
            self.assertTrue((base / "result" / "GFP_3" / "z00000.tif").is_file())

    def test_shared_canvas_blending_and_z_offsets(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            raw = base / "raw"
            self.make_channel(raw, "640nm", "GFP",
                              {"A": (0, 0, 0), "B": (2, 0, 1)},
                              {"A": 100, "B": 200})
            self.make_channel(raw, "561nm", "RFP",
                              {"A": (1, 0, 2), "B": (3, 0, 1)},
                              {"A": 300, "B": 400})
            config = base / "config.json"
            config.write_text(json.dumps({
                "raw_sample_dir": str(raw),
                "direct_stitching": {
                    "enabled": True,
                    "markers": ["GFP", "RFP"],
                    "downsample_factor": 2,
                    "blend_width_px": 2,
                    "output_dir": str(base / "result"),
                },
            }), encoding="utf-8")
            run(config, dry_run=True)
            self.assertFalse((base / "result").exists())
            run(config)
            result = base / "result"
            manifest = json.loads((result / "stitch_manifest.json").read_text())
            self.assertEqual((manifest["output_width"],
                              manifest["output_height"],
                              manifest["output_slices"]), (3, 2, 2))
            self.assertEqual(manifest["channels"]["GFP"]
                             ["z_offset_from_channel_frame_raw_slices"], 1)
            with Image.open(result / "GFP" / "z00000.tif") as image:
                self.assertEqual(image.size, (3, 2))
                self.assertEqual(image.getpixel((0, 0)), 125)
                self.assertTrue(125 < image.getpixel((1, 0)) < 215)
            with Image.open(result / "RFP" / "z00000.tif") as image:
                self.assertEqual(image.size, (3, 2))
                self.assertEqual(image.getpixel((0, 0)), 305)
            self.assertEqual(len(list((result / "GFP").glob("*.tif"))), 2)
            self.assertEqual(len(list((result / "RFP").glob("*.tif"))), 2)

            # The same script also accepts the project's comment-bearing config.
            project_config = base / "project_config.json"
            project_config.write_text("// project config\n" + json.dumps({
                "stitching_reference_channel": "GFP",
                "channels_routing": [
                    {"id": "GFP", "dir_key": "gfp_dir", "active": True},
                    {"id": "RFP", "dir_key": "rfp_dir", "active": True},
                ],
                "paths": {"gfp_dir": str(raw / "640nm"),
                          "rfp_dir": str(raw / "561nm")},
                "direct_stitching": {
                    "enabled": True,
                    "downsample_factor": 2,
                    "output_dir": str(base / "project_result"),
                },
            }), encoding="utf-8")
            run(project_config, dry_run=True)
            self.assertFalse((base / "project_result").exists())
            run(project_config)
            project_manifest = json.loads(
                (base / "project_result" / "stitch_manifest.json").read_text())
            self.assertEqual(project_manifest["reference_frame"]
                             ["leading_frame_z_slices_removed"], 1)
            run(project_config, skip_completed=True)
            self.assertEqual(len(list((base / "project_result" / "GFP")
                                      .glob("z*.tif"))), 2)
            xml = raw / "640nm" / "xml_merging_GFP.xml"
            changed = xml.stat().st_mtime_ns + 2_000_000_000
            os.utime(xml, ns=(changed, changed))
            with self.assertRaises(FileExistsError):
                run(project_config, skip_completed=True)


if __name__ == "__main__":
    unittest.main()
