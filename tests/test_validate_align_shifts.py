"""Scientific regression checks for offset evaluation, fixed ROIs and pooling."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.core.point_cloud_aligner import resolve_align_settings
from src.utils import validate_align_shifts as validation


def cell(x, y, z, width=4):
    return {
        "cx": x, "cy": y, "cz": z,
        "x1_3d": x - width / 2, "x2_3d": x + width / 2,
        "y1_3d": y - width / 2, "y2_3d": y + width / 2,
        "z_min": z, "z_max": z,
        "per_z_boxes": {z: [x - width / 2, y - width / 2, x + width / 2, y + width / 2]},
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(root):
    routing = [
        {"id": "GFP", "type": "soma", "dir_key": "gfp_dir"},
        {"id": "RFP", "type": "soma", "dir_key": "rfp_dir"},
        {"id": "Sox9", "type": "tf", "dir_key": "sox9_dir"},
        {"id": "Olig2", "type": "tf", "dir_key": "olig2_dir"},
    ]
    config = {
        "channels_routing": routing,
        "pre_align_params": {"reference_channel": "GFP", "sample_z_center_count": 4},
        "stitching_reference_channel": "Olig2",
        "detection_params": {"tILESIZE": 64, "xy_resolution_um": .65, "z_resolution_um": 8},
        "z_linker": {"soma": {"max_cell_z_span": 5}, "tf": {"max_cell_z_span": 5}},
    }
    settings = resolve_align_settings(config, routing)
    offsets = {"GFP": {"dx": 3, "dy": 2, "dz": 1},
               "RFP": {"dx": -1, "dy": 2, "dz": 1},
               "Sox9": {"dx": 0, "dy": -1, "dz": 0},
               "Olig2": {"dx": 0, "dy": 0, "dz": 0}}
    align = root / "3_2d_aligned"
    filtered = root / "2_2d_filtered"
    filtered.mkdir(parents=True)
    write_json(root / "runtime_config.json", config)
    write_json(align / validation.ALIGN_SETTINGS_FILE, settings)
    write_json(align / "tile_offsets.json", offsets)
    write_json(align / "tile_measured_offsets.json", {"z_center": 16})
    per_ch = {}
    for ch in routing:
        cid = ch["id"]
        shift = tuple(offsets[cid][a] for a in ("dx", "dy", "dz"))
        desired = [cell(20, 20, 15), cell(40, 40, 17)]
        per_ch[cid] = validation._shift_cells(desired, *[-v for v in shift])
        records = []
        for c in per_ch[cid]:
            records.append({
                "x1": c["x1_3d"], "y1": c["y1_3d"], "x2": c["x2_3d"],
                "y2": c["y2_3d"], "z": c["cz"], "score": .9, "mean": 100,
                "class": "neuron" if ch["type"] == "soma" else "nucleus",
            })
        pd.DataFrame(records).to_csv(filtered / f"tile_{cid}_result.csv", index=False)
        write_json(root / "1_2d_raw" / f"tile_{cid}_inputs.json",
                   {"images": [{"status": "processed"} for _ in range(40)]})
    ev = copy.deepcopy(validation.DEFAULT_EVAL)
    ev["comparison"].update(fixed_channel="GFP", moving_channels=["RFP", "Sox9", "Olig2"])
    ev["sampling"].update(n_tiles=1, held_out=False, min_cells=1,
                          actual_z_size_slices=10, actual_z_depth_um=80, z_guard_slices=5)
    ev["metric"]["xy_bin_size_px"] = 1
    params = {
        "settings": settings, "routing": routing, "align_dir": str(align),
        "filtered_dir": str(filtered), "raw_dir": str(root / "1_2d_raw"),
        "eval": ev, "tile_size": 64, "sweep": ((2, 1), (2, 1), (1, 1)),
    }
    return params, per_ch


class AlignmentValidationTests(unittest.TestCase):
    def test_measured_checkpoint_is_not_an_extra_tile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a_offsets.json").touch()
            (root / "a_measured_offsets.json").touch()
            self.assertEqual(validation.list_tiles(root), ["a"])

    def test_legacy_support_is_lazy_and_uses_anchor_slice_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roots = {}
            for cid, names in (("GFP", ["s001.tif", "s002.tif", "s003.tiff"]),
                               ("RFP", ["s001.tif", "s003.tiff"])):
                directory = root / cid / "100" / "100_200"
                directory.mkdir(parents=True)
                for name in names + [".hidden.tif", "notes.csv"]:
                    (directory / name).touch()
                roots[cid] = str(root / cid)
            params = {"raw_dir": str(root / "1_2d_raw"), "channel_roots": roots,
                      "anchor_channel": "GFP", "candidate_tiles": ["100_200"]}
            support, metadata = validation.load_slice_support("100_200", "RFP", params)
            self.assertEqual(support, {1, 3})
            self.assertEqual(metadata["kind"], "legacy_tiff_index")
            self.assertEqual(params["_slice_cache"][("GFP", "100_200")],
                             ["s001", "s002", "s003"])

    def test_sweep_is_symmetric_even_when_step_does_not_divide_radius(self):
        values = [d for a, d in validation._sweep_deltas(((5, 2), (0, 1), (1, 1))) if a == "x"]
        self.assertEqual(values, [-5, -4, -2, 0, 2, 4, 5])

    def test_cells_outside_window_can_move_in(self):
        region = (0, 20, 0, 20, 10, 20)
        reference = validation._grid([cell(10, 10, 10)], region, 1, .65, 8)
        # This cell is entirely outside the ROI in Z until it moves by +3.
        moving = [cell(10, 10, 7)]
        self.assertEqual(validation.region_voxel_iou(reference, moving, region, 0, 0, 0, 1, .65, 8), 0)
        self.assertEqual(validation.region_voxel_iou(reference, moving, region, 0, 0, 3, 1, .65, 8), 1)

    def test_inverse_grid_with_z_halo_matches_complete_forward_shift(self):
        rng = np.random.default_rng(7)
        cells = [cell(float(rng.uniform(-10, 70)), float(rng.uniform(-10, 70)),
                      int(rng.integers(-20, 70)), width=float(rng.uniform(1, 40)))
                 for _ in range(300)]
        # Include a multi-slice cell and boxes with exact bin-boundary coordinates.
        multi = cell(20, 20, 12)
        multi.update(z_min=8, z_max=18)
        multi["per_z_boxes"][18] = [17, 17, 25, 25]
        cells.extend([multi, cell(4, 4, 11, width=8)])
        region = (0, 48, 0, 48, 10, 20)
        for bin_size in (1, 4):
            halo = validation.z_halo_cells(cells, region, 5, .65, 8)
            self.assertLess(len(halo), len(cells))
            for shift in ((0, 0, 0), (3, -5, 0), (-7, 11, 0), (0, 0, -5), (0, 0, 5)):
                forward = validation._grid(validation._shift_cells(cells, *shift),
                                           region, bin_size, .65, 8)
                inverse = validation.shifted_grid(halo, region, shift, bin_size, .65, 8)
                np.testing.assert_array_equal(inverse, forward)

    def test_center_start_is_deterministic_and_uses_image_support(self):
        starts = list(range(7, 28))
        self.assertEqual(validation.order_z_starts(starts, "center", np.random.default_rng(1)), [17])
        self.assertEqual(validation.order_z_starts(starts, "center", np.random.default_rng(999)), [17])
        self.assertEqual(validation.order_z_starts([1, 2, 10], "center", np.random.default_rng(0)), [2])

    def test_all_tiles_center_includes_empty_tile_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sample" / "detection_results"
            fixture(root)
            for folder in ("1_2d_raw", "2_2d_filtered", "3_2d_aligned"):
                for source in list((root / folder).glob("tile_*")):
                    target = source.with_name(source.name.replace("tile_", "empty_", 1))
                    if folder == "2_2d_filtered":
                        pd.read_csv(source).head(0).to_csv(target, index=False)
                    else:
                        target.write_bytes(source.read_bytes())
            config = Path(tmp) / "central.json"
            write_json(config, {
                "sampling": {"tile_selection": "all", "z_selection": "center",
                             "z_size_slices": 30, "z_depth_um": 240, "held_out": False, "min_cells": 0},
                "sweep": {"x": {"range_px": 2, "step_px": 1},
                          "y": {"range_px": 2, "step_px": 1},
                          "z": {"range_slices": 1, "step_slices": 1}},
                "metric": {"xy_bin_size_px": 1},
            })
            validation.main(["--sample", str(root), "--eval-config", str(config), "--no-plots"])
            output = root / "alignment_validation"
            manifest = json.loads((output / "sampling.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["selected_tiles"], ["empty", "tile"])
            df = pd.read_csv(output / "sample_curves.csv")
            self.assertEqual(df["tile"].nunique(), 2)
            self.assertTrue((df["z1"] - df["z0"]).eq(30).all())
            self.assertTrue(df["z_selection"].eq("center").all())
            self.assertFalse(df["held_out"].any())
            self.assertTrue(df.loc[df["tile"] == "empty", "voxel_iou"].isna().all())
            aggregate = pd.read_csv(output / "sample_pooled_curves.csv")
            self.assertTrue(aggregate["n_tiles"].eq(2).all())
            self.assertTrue(aggregate["n_defined_tiles"].eq(1).all())
            self.assertEqual(len(list((output / "tile_curves").glob("*.csv"))), 2)
            progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
            self.assertEqual(progress["processed_tiles"], 2)
            self.assertEqual(df[["z0", "z1"]].drop_duplicates().values.tolist(), [[6, 36]])

    def test_empty_union_is_undefined(self):
        grid = np.zeros((2, 2, 2), dtype=np.uint8)
        intersection, union, iou = validation.grid_counts(grid, grid)
        self.assertEqual((intersection, union), (0, 0))
        self.assertTrue(np.isnan(iou))

    def test_final_frame_nonzero_fixed_offset_and_unaligned_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            params, per_ch = fixture(Path(tmp))
            with patch.object(validation, "build_tile_vol_lists", return_value=(per_ch, 16)):
                rows, warnings = validation.process_tile("tile", params)
            self.assertTrue(rows, warnings)
            df = pd.DataFrame(rows)
            zero = df.loc[df["delta"] == 0]
            np.testing.assert_allclose(zero["voxel_iou"], 1)
            rfp = zero.loc[zero["channel"] == "RFP"]
            self.assertTrue((rfp["base_dx"] == -1).all())  # Saved offset, not RFP minus GFP.
            self.assertTrue((rfp["fixed_dx"] == 3).all())
            self.assertTrue((rfp["unaligned_iou"] < 1).all())
            self.assertEqual(df["coordinate_frame"].unique().tolist(), ["Olig2"])
            self.assertEqual(len(df[["x0", "x1", "y0", "y1", "z0", "z1"]].drop_duplicates()), 1)

    def test_held_out_checks_inverse_mapped_halo_and_missing_slices(self):
        supports = {"A": set(range(1, 61)), "B": set(range(1, 61)) - {8}}
        limits = {"A": ([0, 0, 0], [0, 0, 0]), "B": ([0, 0, -3], [0, 0, 3])}
        _, independent, valid = validation.valid_region_starts(
            supports, limits, 64, 10, (25, 35), 2, True)
        self.assertTrue(independent)
        self.assertNotIn(10, valid)  # Source halo contains missing slice 8.
        self.assertNotIn(14, independent)  # Source halo reaches the guarded solve window.
        self.assertIn(41, independent)

    def test_pooled_iou_is_ratio_of_counts_not_mean_of_ratios(self):
        df = pd.DataFrame([
            {"tile": "a", "channel": "RFP", "ref_channel": "GFP", "axis": "x", "delta": 0,
             "delta_um": 0, "intersection": 1, "union": 2, "voxel_iou": .5,
             "unaligned_intersection": 0, "unaligned_union": 2, "unaligned_iou": 0},
            {"tile": "b", "channel": "RFP", "ref_channel": "GFP", "axis": "x", "delta": 0,
             "delta_um": 0, "intersection": 9, "union": 10, "voxel_iou": .9,
             "unaligned_intersection": 1, "unaligned_union": 10, "unaligned_iou": .1},
        ])
        result = validation.aggregate_curves(df).iloc[0]
        self.assertAlmostEqual(result["pooled_iou"], 10 / 12)
        self.assertAlmostEqual(result["mean_iou"], .7)
        self.assertAlmostEqual(result["unaligned_pooled_iou"], 1 / 12)

    def test_tied_peak_selects_zero(self):
        group = pd.DataFrame({"delta": [-2, 0, 2], "delta_um": [-1.3, 0, 1.3],
                              "iou": [.5, .5, .5], "baseline": [.2, .2, .2]})
        summary = validation.curve_summary(group, "iou", "baseline")
        self.assertEqual(summary["peak_delta"], 0)
        self.assertTrue(summary["zero_at_peak"])
        self.assertAlmostEqual(summary["absolute_improvement"], .3)

    def test_one_config_controls_z_sample_quantity(self):
        settings = {"reference_channel": "GFP", "voxel_bin_size_px": 4,
                    "xy_resolution_um": .65, "z_resolution_um": 8,
                    "z_link": {"soma": {"max_cell_z_span": 5}, "tf": {"max_cell_z_span": 3}}}
        routing = [{"id": cid} for cid in ("GFP", "RFP", "Sox9", "Olig2")]
        template = Path(__file__).resolve().parents[1] / "config_example/align_validation.example.json"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.json"
            config = validation.load_config(template)
            for slices in (10, 30):
                with self.subTest(slices=slices):
                    config["sampling"]["z_size_slices"] = slices
                    write_json(path, config)
                    args = validation.parse_args(["--sample", "unused", "--eval-config", str(path)])
                    ev = validation.resolve_eval(args, routing, settings)
                    self.assertEqual(ev["sampling"]["actual_z_size_slices"], slices)
                    self.assertEqual(ev["sampling"]["actual_z_depth_um"], slices * 8)
            config["sampling"].update(tile_selection="all", z_selection="center",
                                      held_out=False, min_cells=0)
            write_json(path, config)
            ev = validation.resolve_eval(args, routing, settings)
            self.assertEqual(ev["sampling"]["actual_z_size_slices"], 30)
            self.assertEqual(ev["sampling"]["tile_selection"], "all")
            self.assertEqual(ev["sampling"]["z_selection"], "center")

    def test_eval_config_and_cli_overrides(self):
        settings = {"reference_channel": "GFP", "voxel_bin_size_px": 4,
                    "xy_resolution_um": .65, "z_resolution_um": 8,
                    "z_link": {"soma": {"max_cell_z_span": 5}, "tf": {"max_cell_z_span": 3}}}
        routing = [{"id": cid} for cid in ("GFP", "RFP", "Sox9", "Olig2")]
        path = Path(__file__).resolve().parents[1] / "config_example/align_validation.example.json"
        args = validation.parse_args(["--sample", "unused", "--eval-config", str(path),
                                      "--n-tiles", "2", "--z-depth-um", "96", "--xy-range", "12"])
        ev = validation.resolve_eval(args, routing, settings)
        self.assertEqual(ev["sampling"]["n_tiles"], 2)
        self.assertEqual(ev["sampling"]["actual_z_size_slices"], 12)
        self.assertEqual(ev["sweep"]["x"]["range_px"], 12)
        self.assertEqual(ev["sweep"]["y"]["range_px"], 12)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            validation._merge_config(copy.deepcopy(validation.DEFAULT_EVAL), {"samplng": {}})

    def test_main_saves_tables_manifest_and_reports_incomplete_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sample" / "detection_results"
            fixture(root)
            ev_path = Path(tmp) / "eval.json"
            write_json(ev_path, {
                "sampling": {"n_tiles": 1, "held_out": False, "min_cells": 1},
                "sweep": {"x": {"range_px": 2, "step_px": 1},
                          "y": {"range_px": 2, "step_px": 1},
                          "z": {"range_slices": 1, "step_slices": 1}},
                "metric": {"xy_bin_size_px": 1},
            })
            args = ["--sample", str(root), "--eval-config", str(ev_path), "--no-plots"]
            validation.main(args)
            out = root / "alignment_validation"
            manifest = json.loads((out / "sampling.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["selected_tiles"], ["tile"])
            self.assertEqual(len(list(out.glob("*.csv"))), 4)
            for region in manifest["regions"]:
                self.assertEqual(region["z1"] - region["z0"], 10)
            with self.assertRaisesRegex(ValueError, "Only 1/2"):
                validation.main(args + ["--n-tiles", "2"])
            manifest = json.loads((out / "sampling.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["complete"])


if __name__ == "__main__":
    unittest.main()
