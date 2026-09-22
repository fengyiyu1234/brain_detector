import json
import os
import tempfile
import unittest

from src.utils.coordinate_context import CoordinateContext, CoordinateContextError


def _xml(tile_abs):
    stacks = "".join(
        f'<Stack DIR_NAME="{name}/{name}" ABS_H="{x}" ABS_V="{y}" ABS_D="{z}" />'
        for name, (x, y, z) in tile_abs.items())
    return f"<TeraStitcher><STACKS>{stacks}</STACKS></TeraStitcher>"


class CoordinateContextTests(unittest.TestCase):
    tile = "360100_349900"

    def _make_context(self):
        tmp = tempfile.TemporaryDirectory()
        result = tmp.name
        xml_dir = os.path.join(result, "solver")
        os.makedirs(xml_dir)
        # The frame origin is the common solver ABS origin, not each XML's min.
        frame = {"origin": (0, 0, 11), self.tile: (3458, 8674, 6)}
        gfp = {"origin": (0, 0, 11), self.tile: (3466, 8668, 6)}
        for channel, positions in (("Olig2", frame), ("GFP", gfp)):
            with open(os.path.join(xml_dir, f"xml_merging_{channel}.xml"), "w", encoding="utf-8") as f:
                f.write(_xml(positions))
        runtime = {
            "pipeline_mode": "pre_align",
            "paths": {"pATHXML": os.path.join(xml_dir, "xml_merging_Olig2.xml")},
            "channels_routing": [{"id": "GFP", "active": True}, {"id": "Olig2", "active": True}],
            "detection_params": {"tILESIZE": 2048},
        }
        with open(os.path.join(result, "runtime_config.json"), "w", encoding="utf-8") as f:
            json.dump(runtime, f)
        align = os.path.join(result, "0_channel_alignment")
        os.makedirs(align)
        with open(os.path.join(align, f"{self.tile}_offsets.json"), "w", encoding="utf-8") as f:
            json.dump({"GFP": {"dx": 7, "dy": 28, "dz": -3},
                       "Olig2": {"dx": 0, "dy": 0, "dz": 0}}, f)
        return tmp, CoordinateContext.from_result_dir(result)

    def test_t4_coordinate_inverse_and_equivalent_expression(self):
        tmp, context = self._make_context()
        self.addCleanup(tmp.cleanup)
        offsets = context.offsets_for_tile(self.tile)
        self.assertEqual(tuple(context.position(self.tile)), (3458, 8674, 5))
        self.assertEqual(context.global_to_frame_local(self.tile, 3458, 8674, 2), (0, 0, 6))
        self.assertEqual(context.frame_local_to_global(self.tile, 0, 0, 6), (3458, 8674, 2))
        self.assertEqual(context.channel_residual(self.tile, "GFP", offsets), (-1, 34, -3))

        raw = (100, 200, 10)  # raw z is 1-based
        p_o = context.position(self.tile)
        p_g = context.position(self.tile, "GFP")
        q = context.channel_residual(self.tile, "GFP", offsets)
        pipeline = (raw[0] + offsets["GFP"]["dx"] + p_o.x,
                    raw[1] + offsets["GFP"]["dy"] + p_o.y,
                    raw[2] + offsets["GFP"]["dz"] - p_o.z)
        own_xml = (raw[0] + p_g.x + q[0], raw[1] + p_g.y + q[1], raw[2] - p_g.z + q[2])
        self.assertEqual(pipeline, own_xml)

    def test_prealign_requires_complete_offsets(self):
        tmp, context = self._make_context()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(context.result_dir, "0_channel_alignment", f"{self.tile}_offsets.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"Olig2": {"dx": 0, "dy": 0, "dz": 0}}, f)
        with self.assertRaisesRegex(CoordinateContextError, "GFP"):
            context.offsets_for_tile(self.tile)
