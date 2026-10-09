"""Saved pre-align GUI contracts using small synthetic pipeline outputs."""

import io
import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

from src.utils.saved_view import (
    global_to_local_rows, read_global_boxes, read_tile_boxes,
    saved_run_context, track_for_summary,
)
from src.utils.saved_gui import _add_tile_2d, run_saved_prealign
from src.utils.visualize import (
    _active_image_channel, _configure_text_output, _make_fn_recorder,
    _open_tiles_progressively,
    _show_only_images_initially, load_frame_volume, load_volume,
)


class FakeViewer:
    made = []

    def __init__(self, title):
        self.title = title
        self.layers = Layers()
        self.mouse_drag_callbacks = []
        self.cursor = SimpleNamespace(position=(0, 0, 0))
        self.dims = SimpleNamespace(ndisplay=2)
        self.status = ""
        FakeViewer.made.append(self)

    def add_shapes(self, data, **kwargs):
        layer = SimpleNamespace(data=data, **kwargs)
        self.layers.append(layer)
        return layer

    def add_points(self, data, **kwargs):
        layer = SimpleNamespace(data=data, **kwargs)
        self.layers.append(layer)
        return layer

    def add_image(self, data, **kwargs):
        layer = SimpleNamespace(data=data, **kwargs)
        self.layers.append(layer)
        return layer

    def reset_view(self):
        pass


class Layers(list):
    def __init__(self):
        super().__init__()
        self.selection = SimpleNamespace(active=None)


class SavedPrealignTests(unittest.TestCase):
    tile = "000100_000200"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.result = root / "results"
        self.result.mkdir()
        xml_dir = self.result / "5_2d_global" / "tile_positions"
        xml_dir.mkdir(parents=True)
        xml = xml_dir / "xml_merging_GFP.xml"
        xml.write_text(
            '<TeraStitcher><STACKS>'
            '<Stack DIR_NAME="origin/origin" ABS_H="0" ABS_V="0" ABS_D="10"/>'
            f'<Stack DIR_NAME="row/{self.tile}" ABS_H="20" ABS_V="30" ABS_D="8"/>'
            '</STACKS></TeraStitcher>', encoding="utf-8")
        (xml_dir / "xml_merging_RFP.xml").write_text(
            xml.read_text(encoding="utf-8"), encoding="utf-8")
        self.images = root / "images"
        for channel in ("GFP", "RFP"):
            folder = self.images / channel / "row" / self.tile
            folder.mkdir(parents=True)
            for z in range(5):
                cv2.imwrite(str(folder / f"{z:04d}.tif"),
                            np.zeros((16, 16), dtype=np.uint16))
        routing = [
            {"id": "GFP", "type": "soma", "dir_key": "gfp_dir", "active": True},
            {"id": "RFP", "type": "soma", "dir_key": "rfp_dir", "active": True},
        ]
        runtime = {
            "pipeline_mode": "pre_align",
            "stitching_reference_channel": "GFP",
            "channels_routing": routing,
            "paths": {
                "pATHRESULT": "/rsstu/example/results",
                "pATHXML": "/rsstu/example/results/5_2d_global/"
                            "tile_positions/xml_merging_GFP.xml",
            },
            "detection_params": {"tILESIZE": 16},
        }
        (self.result / "runtime_config.json").write_text(
            json.dumps(runtime), encoding="utf-8")
        align_dir = self.result / "3_2d_aligned"
        align_dir.mkdir(parents=True)
        (align_dir / f"{self.tile}_offsets.json").write_text(
            json.dumps({"GFP": {"dx": 0, "dy": 0, "dz": 0},
                        "RFP": {"dx": 2, "dy": 3, "dz": 1}}), encoding="utf-8")
        raw_dir = self.result / "1_2d_raw"
        raw_dir.mkdir()
        pd.DataFrame([dict(x1=1, y1=2, x2=5, y2=6, z=3,
                           score=.8, mean=100, **{"class": "glia"})]).to_csv(
            raw_dir / f"{self.tile}_RFP_result.csv", index=False)
        filtered = self.result / "4_2d_filtered"
        filtered.mkdir(exist_ok=True)
        pd.DataFrame([dict(x1=3, y1=5, x2=7, y2=9, z=4,
                           score=.8, mean=100, **{"class": "glia"})]).to_csv(
            filtered / f"{self.tile}_RFP_result.csv", index=False)
        global_row = dict(x1=23, y1=35, x2=27, y2=39, z=2,
                          score=.8, mean=100, **{"class": "glia_GFP_RFP"},
                          tile_name="neighbor")
        for directory, name in (
            ("5_2d_global", "RFP_2d_global.csv"),
            ("6_3d_global", "RFP_3d_tracked.csv"),
            ("7_colocalization", "coloc_result.csv"),
        ):
            folder = self.result / directory
            folder.mkdir(exist_ok=True)
            pd.DataFrame([global_row]).to_csv(folder / name, index=False)
        self.vis = {
            "paths": {
                "pATHRESULT": str(self.result),
                "gfp_dir": str(self.images / "GFP"),
                "rfp_dir": str(self.images / "RFP"),
            },
            "channels_routing": list(reversed(routing)),
            "z_start": 0, "z_count": 5, "no_images": True,
            "show_coloc": True, "show_zlinked": True,
            "fn_ann_path": None,
        }
        self.run, self.context = saved_run_context(self.vis)
        self.tile_path = str(self.images / "GFP" / "row" / self.tile)

    def test_images_only_initial_visibility_can_be_disabled(self):
        viewer = FakeViewer("visibility")
        image = viewer.add_image(None, name="[img] GFP", visible=False)
        boxes = viewer.add_shapes(None, name="[s3] GFP", visible=True)
        _show_only_images_initially(viewer, {})
        self.assertTrue(image.visible)
        self.assertFalse(boxes.visible)
        boxes.visible = True
        _show_only_images_initially(viewer, {"images_only_initially": False})
        self.assertTrue(boxes.visible)

    def test_windows_console_encoding_does_not_break_annotation_log(self):
        output = io.BytesIO()
        stdout = io.TextIOWrapper(output, encoding="cp1252", errors="strict")
        stderr = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
        with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
            _configure_text_output()
            _make_fn_recorder(
                {"fn_ann_path": "annotations.csv", "fn_crop_dir": "crops"},
                {"gfp_dir": str(self.images / "GFP")},
                [{"id": "GFP", "dir_key": "gfp_dir"}],
                str(self.images / "GFP"), self.tile_path,
                self.tile, (0, 5),
            )
            stdout.flush()
        self.assertIn(chr(0x2192).encode("utf-8"), output.getvalue())

    def test_runtime_frame_routing_and_cross_os_xml(self):
        self.assertEqual([r["id"] for r in self.context.routing], ["GFP", "RFP"])
        self.assertEqual(tuple(self.context.position(self.tile)), (20, 30, 2))
        self.assertTrue(os.path.isfile(self.context.frame_xml))

    def test_tile_and_global_z_and_xy(self):
        raw = read_tile_boxes(
            str(self.result / "1_2d_raw" / f"{self.tile}_RFP_result.csv"),
            (0, 5))
        self.assertEqual(int(raw.iloc[0]["display_z"]), 2)
        global_rows = read_global_boxes(
            str(self.result / "7_colocalization" / "coloc_result.csv"),
            (20, 30, 36, 46), (-2, 3))
        local = global_to_local_rows(global_rows, self.context.position(self.tile))
        self.assertEqual(int(local.iloc[0]["display_z"]), 3)
        self.assertEqual(float(local.iloc[0]["x1"]), 3)

    def test_track_lookup_refuses_nearest_guess(self):
        row = {"x1": 23, "x2": 27, "y1": 35, "y2": 39,
               "z": 2, "score": .8, "mean": 100, "class": "glia_RFP"}
        track = {"cx": 25, "cy": 37, "cz": 2, "score": .8, "mean": 100,
                 "class": "glia_RFP"}
        self.assertIs(track_for_summary(row, [track]), track)
        self.assertIsNone(track_for_summary(row, [track, track]))
        self.assertIsNone(track_for_summary(row, [{**track, "class": "glia_GFP"}]))

    def test_global_multiple_tiles_shares_saved_result_layer(self):
        second = "000100_000300"
        xml_dir = Path(self.context.frame_xml).parent
        for channel in ("GFP", "RFP"):
            xml = xml_dir / f"xml_merging_{channel}.xml"
            contents = xml.read_text(encoding="utf-8")
            contents = contents.replace(
                "</STACKS>",
                f'<Stack DIR_NAME="row/{second}" ABS_H="30" '
                'ABS_V="30" ABS_D="8"/></STACKS>')
            xml.write_text(contents, encoding="utf-8")
            folder = self.images / channel / "row" / second
            folder.mkdir(parents=True)
            for z in range(5):
                cv2.imwrite(str(folder / f"{z:04d}.tif"),
                            np.zeros((16, 16), dtype=np.uint16))
        align = self.result / "3_2d_aligned" / f"{second}_offsets.json"
        align.write_text(
            json.dumps({"GFP": {"dx": 0, "dy": 0, "dz": 0},
                        "RFP": {"dx": 2, "dy": 3, "dz": 1}}),
            encoding="utf-8")
        coloc = self.result / "7_colocalization" / "coloc_result.csv"
        data = pd.read_csv(coloc)
        data.loc[len(data)] = {
            **data.iloc[0].to_dict(),
            "x1": 32, "x2": 35, "tile_name": second}
        data.to_csv(coloc, index=False)
        run, context = saved_run_context(self.vis)
        class Viewer(FakeViewer):
            def __init__(self, title):
                super().__init__(title)
                self.layers = Layers()
        self.vis["view_space"] = "global"
        self.vis["no_images"] = False
        with patch("src.utils.saved_gui.napari.Viewer", Viewer):
            run_saved_prealign(
                self.vis, run["paths"], context.routing, context,
                [(self.tile_path, self.tile),
                 (str(self.images / "GFP" / "row" / second), second)])
        viewer = FakeViewer.made[-1]
        image_indices = [i for i, layer in enumerate(viewer.layers)
                         if layer.name.startswith("[img]")]
        self.assertTrue(image_indices)
        self.assertEqual(image_indices,
                         list(range(image_indices[0], image_indices[-1] + 1)))
        self.assertTrue(all(layer.visible == layer.name.startswith("[img]")
                            for layer in viewer.layers))
        coloc_layer = next(l for l in viewer.layers
                           if l.name == "[s4 saved rep-z exact] GFP+RFP")
        self.assertEqual(len(coloc_layer.data), 2)
        self.assertEqual({float(shape[0, 2]) for shape in coloc_layer.data},
                         {23.0, 32.0})

    def test_shift_click_reads_exact_saved_track_span(self):
        track = {"cx": 25.0, "cy": 37.0, "cz": 2.0,
                 "score": .8, "mean": 100.0, "class": "glia_GFP_RFP",
                 "z_min": 2, "z_max": 2,
                 "per_z_boxes": {2: [23, 35, 27, 39]}}
        path = self.result / "6_3d_global" / "RFP_3d_tracked.pkl"
        with path.open("wb") as handle:
            pickle.dump([track], handle)
        class Viewer(FakeViewer):
            def __init__(self, title):
                super().__init__(title)
                self.layers = Layers()
        with patch("src.utils.saved_gui.napari.Viewer", Viewer):
            run_saved_prealign(
                self.vis, self.run["paths"], self.context.routing, self.context,
                [(self.tile_path, self.tile)])
        viewer = FakeViewer.made[-1]
        viewer.layers.selection.active = next(
            layer for layer in viewer.layers if layer.name == "[s3 saved summary] RFP")
        viewer.layers.selection.active.visible = True
        viewer.cursor.position = (3, 7, 5)
        viewer.mouse_drag_callbacks[0](
            viewer, SimpleNamespace(type="mouse_press", modifiers=["Shift"]))
        span = next(layer for layer in viewer.layers
                    if layer.name == "[saved track span]")
        self.assertEqual(tuple(span.data[0][0]), (3, 5, 3))

    def test_regular_click_prints_complete_stage3_xyz(self):
        track = {"cx": 25.0, "cy": 37.0, "cz": 2.0,
                 "score": .8, "mean": 100.0, "class": "glia_GFP_RFP",
                 "z_min": 1, "z_max": 3,
                 "per_z_boxes": {1: [22, 34, 26, 38],
                                 3: [24, 36, 28, 40]}}
        path = self.result / "6_3d_global" / "RFP_3d_tracked.pkl"
        with path.open("wb") as handle:
            pickle.dump([track], handle)
        class Viewer(FakeViewer):
            def __init__(self, title):
                super().__init__(title)
                self.layers = Layers()
        with patch("src.utils.saved_gui.napari.Viewer", Viewer):
            run_saved_prealign(
                self.vis, self.run["paths"], self.context.routing, self.context,
                [(self.tile_path, self.tile)])
        viewer = FakeViewer.made[-1]
        viewer.layers.selection.active = next(
            layer for layer in viewer.layers if layer.name == "[s3 saved summary] RFP")
        viewer.layers.selection.active.visible = True
        viewer.cursor.position = (3, 7, 5)
        output = io.StringIO()
        with patch("sys.stdout", output):
            viewer.mouse_drag_callbacks[0](
                viewer, SimpleNamespace(type="mouse_press", modifiers=[]))
        printed = output.getvalue()
        self.assertIn("global XYZ x=[22, 28], y=[34, 40], z=[1, 3] (1-based), observed_layers=2", printed)
        self.assertIn("global z=1: x=[22, 26], y=[34, 38] | display z=2: x=[2, 6], y=[4, 8]", printed)
        self.assertIn("global z=3: x=[24, 28], y=[36, 40] | display z=4: x=[4, 8], y=[6, 10]", printed)
        self.assertFalse(any(layer.name == "[saved track span]" for layer in viewer.layers))

    def test_partial_run_keeps_local_2d_and_ignores_stale_vis_xml(self):
        frame_xml = Path(self.context.frame_xml)
        frame_xml.unlink()
        (frame_xml.parent / "xml_merging_RFP.xml").unlink()
        stale = self.result / "other_run" / "xml_merging_GFP.xml"
        stale.parent.mkdir()
        stale.write_text(
            '<TeraStitcher><STACKS>'
            f'<Stack DIR_NAME="row/{self.tile}" ABS_H="999" '
            'ABS_V="999" ABS_D="0"/>'
            '</STACKS></TeraStitcher>', encoding="utf-8")
        self.vis["paths"]["pATHXML"] = str(stale)
        run, context = saved_run_context(self.vis)
        self.assertIsNone(context)
        class Viewer(FakeViewer):
            def __init__(self, title):
                super().__init__(title)
                self.layers = Layers()
        with patch("src.utils.saved_gui.napari.Viewer", Viewer):
            run_saved_prealign(
                self.vis, run["paths"], run["channels_routing"], context,
                [(self.tile_path, self.tile)])
        self.assertTrue(any(l.name.startswith("[2d raw]")
                            for l in FakeViewer.made[-1].layers))
        self.assertFalse(any(l.name.startswith("[s4")
                             for l in FakeViewer.made[-1].layers))
        (self.result / "3_2d_aligned" /
         f"{self.tile}_offsets.json").unlink()
        with patch("src.utils.saved_gui.napari.Viewer", Viewer):
            run_saved_prealign(
                self.vis, run["paths"], run["channels_routing"], context,
                [(self.tile_path, self.tile)])
        names = [l.name for l in FakeViewer.made[-1].layers]
        self.assertTrue(any(name.startswith("[2d raw]") for name in names))
        self.assertFalse(any(name.startswith("[2d filtered]") for name in names))

    def test_local_image_preserves_false_negative_channel_selection(self):
        class Viewer(FakeViewer):
            def __init__(self, title):
                super().__init__(title)
                self.layers = Layers()
        self.vis["view_space"] = "local"
        self.vis["no_images"] = False
        with patch("src.utils.saved_gui.napari.Viewer", Viewer):
            run_saved_prealign(
                self.vis, self.run["paths"], self.context.routing, self.context,
                [(self.tile_path, self.tile)])
        viewer = FakeViewer.made[-1]
        viewer.layers.selection.active = next(
            l for l in viewer.layers if l.name == "[img] RFP")
        self.assertEqual(_active_image_channel(viewer), "RFP")

    def test_global_image_uses_saved_shift_and_xml_position(self):
        image_dir = self.images / "RFP" / "row" / self.tile
        cv2.imwrite(str(image_dir / "0002.tif"),
                    np.full((16, 16), 77, dtype=np.uint16))
        class Viewer(FakeViewer):
            def __init__(self, title):
                super().__init__(title)
                self.layers = Layers()
        self.vis["view_space"] = "global"
        self.vis["no_images"] = False
        with patch("src.utils.saved_gui.napari.Viewer", Viewer):
            run_saved_prealign(
                self.vis, self.run["paths"], self.context.routing, self.context,
                [(self.tile_path, self.tile)])
        viewer = FakeViewer.made[-1]
        rfp = next(l for l in viewer.layers
                   if l.name == f"[img] {self.tile} RFP")
        self.assertEqual(tuple(rfp.translate), (-2, 30, 20))
        self.assertEqual(rfp.data[3, 3, 2], 77)
        self.assertEqual(rfp.data[2, 3, 2], 0)

    def test_second_exposure_saved_2d_is_available(self):
        second = "RFP2"
        raw_path = self.result / "1_2d_raw" / f"{self.tile}_{second}_result.csv"
        pd.DataFrame([dict(x1=1, y1=2, x2=5, y2=6, z=3,
                           score=.8, mean=100, **{"class": "glia"})]).to_csv(
            raw_path, index=False)
        routing = [dict(self.context.routing[1],
                        double_exposure=True,
                        second_intensity_id=second)]
        viewer = FakeViewer("double")
        _add_tile_2d(
            viewer, str(self.result), self.tile, routing,
            {"RFP": {"dx": 2, "dy": 3, "dz": 1}}, (0, 5),
            self.context.position(self.tile), False, [], 2, self.vis)
        names = [layer.name for layer in viewer.layers]
        self.assertIn(f"[2d raw] {self.tile} {second}", names)
        self.assertNotIn(f"[2d filtered] {self.tile} {second}", names)

    def test_raw_boundary_uses_pre_shift_source_z(self):
        viewer = FakeViewer("boundary")
        registry = []
        _add_tile_2d(
            viewer, str(self.result), self.tile, self.context.routing,
            self.context.offsets_for_tile(self.tile), (3, 4),
            self.context.position(self.tile), False, registry, 2, self.vis)
        raw = next(l for l in viewer.layers if l.name.startswith("[2d raw]"))
        self.assertEqual(int(raw.data[0][0, 0]), 3)

    def test_image_shift_reads_source_slice_before_crop(self):
        image_dir = self.images / "RFP" / "row" / self.tile
        cv2.imwrite(str(image_dir / "0002.tif"),
                    np.full((16, 16), 77, dtype=np.uint16))
        volume, _ = load_frame_volume(
            str(image_dir), (3, 4), (0, 0, 1))
        self.assertEqual(volume.shape, (1, 16, 16))
        self.assertEqual(volume[0, 0, 0], 77)
        missing, _ = load_frame_volume(
            str(image_dir), (-10, -5), (0, 0, 0))
        self.assertIsNone(missing)

    def test_frame_volume_preserves_uint16_and_combines_xyz_shift(self):
        image_dir = self.images / "RFP" / "row" / self.tile
        source = np.zeros((16, 16), dtype=np.uint16)
        source[3, 4] = 40000
        cv2.imwrite(str(image_dir / "0002.tif"), source)
        volume, _ = load_frame_volume(
            str(image_dir), (3, 5), (2, -1, 1))
        self.assertEqual(volume.dtype, np.dtype("uint16"))
        self.assertEqual(volume.shape, (2, 16, 16))
        self.assertEqual(int(volume[0, 2, 6]), 40000)
        self.assertEqual(int(volume[0, 3, 4]), 0)
        direct, _ = load_frame_volume(str(image_dir), (2, 3))
        self.assertEqual(direct.dtype, np.dtype("uint16"))
        self.assertEqual(int(direct[0, 3, 4]), 40000)
        legacy_caller, _ = load_volume(str(image_dir), (2, 3))
        self.assertEqual(legacy_caller.dtype, np.dtype("float32"))

    def test_local_and_global_views_use_saved_rows(self):
        class Viewer(FakeViewer):
            def __init__(self, title):
                super().__init__(title)
                self.layers = Layers()
        with patch("src.utils.saved_gui.napari.Viewer", Viewer), \
             patch("src.utils.visualize.run_z_linker",
                   side_effect=AssertionError("GUI recalculated Z-link")):
            self.vis["view_space"] = "local"
            run_saved_prealign(
                self.vis, self.run["paths"], self.context.routing, self.context,
                [(self.tile_path, self.tile)])
            local = FakeViewer.made[-1]
            raw = next(l for l in local.layers if l.name.startswith("[2d raw]"))
            coloc = next(l for l in local.layers if l.name == "[s4 saved rep-z exact] GFP+RFP")
            self.assertEqual(tuple(raw.data[0][0]), (3, 5, 3))
            self.assertEqual(tuple(coloc.data[0][0]), (3, 5, 3))
            self.vis["view_space"] = "global"
            run_saved_prealign(
                self.vis, self.run["paths"], self.context.routing, self.context,
                [(self.tile_path, self.tile)])
            global_view = FakeViewer.made[-1]
            coloc = next(l for l in global_view.layers
                         if l.name == "[s4 saved rep-z exact] GFP+RFP")
            self.assertEqual(tuple(coloc.data[0][0]), (1, 35, 23))


    def test_local_tiles_open_after_first_viewer_starts(self):
        events, pending = [], []

        def open_tile(path, name):
            events.append(("open", name))

        def progress(current, total, name):
            events.append(("progress", name, current, total))

        def run_viewer():
            events.append(("viewer_running",))
            self.assertEqual(events[:3], [
                ("open", "first"),
                ("progress", "first", 1, 2),
                ("viewer_running",),
            ])
            while pending:
                pending.pop(0)()

        with patch("src.utils.visualize.QTimer.singleShot",
                   side_effect=lambda delay, callback: pending.append(callback)), \
             patch("src.utils.visualize.napari.run", side_effect=run_viewer):
            _open_tiles_progressively(
                [("a", "first"), ("b", "second")], open_tile, progress)

        self.assertEqual(events[-2:], [
            ("open", "second"),
            ("progress", "second", 2, 2),
        ])


if __name__ == "__main__":
    unittest.main()

