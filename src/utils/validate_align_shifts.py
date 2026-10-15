# -*- coding: utf-8 -*-
"""Evaluate saved channel offsets by single-axis perturbation in their final tile frame.

Example (CPU only; existing detections and offsets are reused):
  python src/utils/validate_align_shifts.py --sample /path/to/detection_results3 \
      --eval-config config/EGFR/2/align_validation_t70.json

IoU is binary occupancy of z-linked detection boxes, using the alignment
rasterizer (including its inferred Z extent for single-slice cells), not raw
fluorescence or instance-mask IoU. Each channel receives its saved offset.
The fixed comparison channel does not define the coordinate frame.

Every curve uses a fixed ROI inside the common valid footprint of all shifts,
including the unaligned baseline. All tile cells are shifted BEFORE cropping,
so cells outside the ROI can move into it. The baseline applies the fixed
channel's offset to BOTH raw channels, preserving their unaligned relative
positions. Held-out sampling checks inverse-mapped source windows, including
the full sweep, against the raw alignment window with a cell-span guard.

Outputs are written before importing matplotlib: sampling.json, tile curves,
pooled curves, tile summaries, pooled summaries, and a channel-by-axis figure.
Missing tiles are replaced in deterministic random order; an incomplete sample
is saved and reported with a nonzero exit status. No inference is performed.
"""

import argparse
import copy
import json
import math
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import multiprocessing as mp

project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config.loader import load_config
from src.core.z_linker import run_z_linker
from src.core.point_cloud_aligner import (
    ALIGN_SETTINGS_FILE, resolve_align_settings, _voxelize_to_grid, _effective_z_extent,
    _cell_arrays, _prepare_soma_containment_index, _containment_score,
)

DEFAULT_EVAL = {
    "comparison": {
        "fixed_channel": None,
        "moving_channels": None,
        "coordinate_frame": "saved_offsets",
        "include_unaligned_baseline": True,
    },
    "sampling": {
        "n_tiles": 10, "seed": 42, "regions_per_tile": 1,
        "tile_selection": "random", "z_selection": "random",
        "xy_mode": "full_tile", "xy_size_px": 1024,
        "z_depth_um": 80, "z_size_slices": None,
        "held_out": True, "insufficient_depth_policy": "replace_tile",
        "z_guard_slices": None, "min_cells": 15, "max_attempts": 50,
    },
    "sweep": {
        "x": {"range_px": 8, "step_px": 1},
        "y": {"range_px": 8, "step_px": 1},
        "z": {"range_slices": 2, "step_slices": 1},
    },
    "metric": {"type": "voxel_iou", "xy_bin_size_px": 1, "aggregation": "pooled"},
    "output": {
        "dir": "alignment_validation", "plot_units": "um",
        "formats": ["png", "pdf"], "show_tile_curves": True, "per_tile_plots": False,
    },
    "workers": 1,
}


def resolve_results_dir(sample):
    for cand in (Path(sample), Path(sample) / "detection_results"):
        if (cand / "2_2d_filtered").is_dir() and (cand / "3_2d_aligned").is_dir():
            return str(cand.resolve())
    raise ValueError(f"{sample}: expected 2_2d_filtered/ and 3_2d_aligned/; "
                     "pass detection_results2/3 explicitly when applicable")


def load_settings(results_dir, config_path):
    align_dir = os.path.join(results_dir, "3_2d_aligned")
    runtime = os.path.join(results_dir, "runtime_config.json")
    cfg_path = runtime if os.path.isfile(runtime) else config_path
    if not cfg_path or not os.path.isfile(cfg_path):
        raise ValueError("Missing runtime_config.json; use --config with the original inference config")
    config = load_config(cfg_path)
    routing = [ch for ch in config.get("channels_routing", []) if ch.get("active", True)]
    if not routing:
        raise ValueError(f"No active channels in {cfg_path}")
    saved_path = os.path.join(align_dir, ALIGN_SETTINGS_FILE)
    if os.path.isfile(saved_path):
        settings = load_config(saved_path)
        source = saved_path
    else:
        settings = resolve_align_settings(config, routing)
        source = f"{cfg_path} (saved alignment settings unavailable)"
    return config, routing, settings, align_dir, source


def list_tiles(align_dir):
    # Measured offsets are a separate checkpoint, never an additional tile.
    return sorted(f[:-len("_offsets.json")] for f in os.listdir(align_dir)
                  if f.endswith("_offsets.json") and not f.endswith("_measured_offsets.json"))


def build_tile_vol_lists(filtered_dir, tile_name, routing, settings):
    per_ch, z_all = {}, []
    for ch in routing:
        cid, ctype = ch["id"], ch.get("type", "soma")
        path = os.path.join(filtered_dir, f"{tile_name}_{cid}_result.csv")
        if not os.path.isfile(path):
            raise ValueError(f"Missing filtered detections: {path}")
        df = pd.read_csv(path)
        cells = []
        if not df.empty:
            mat = df[["x1", "y1", "x2", "y2", "score", "mean", "class", "z"]].values.astype(object)
            mat[:, 6] = [f"{v}_{cid}" for v in mat[:, 6]]
            _, cells = run_z_linker(mat, **settings["z_link"]["soma" if ctype == "soma" else "tf"])
        per_ch[cid] = cells
        z_all.extend(c.get("cz", 0) for c in cells)
    return per_ch, float(np.median(z_all)) if z_all else None


def _merge_config(base, override, prefix=""):
    if not isinstance(override, dict):
        raise ValueError(f"{prefix or 'config'} must be an object")
    for key, value in override.items():
        name = f"{prefix}.{key}" if prefix else key
        if key not in base:
            raise ValueError(f"Unknown evaluation setting: {name}")
        if isinstance(base[key], dict):
            _merge_config(base[key], value, name)
        else:
            base[key] = value
    return base


def _positive(value, name, integer=False, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (not allow_zero and value == 0) or (integer and not isinstance(value, int)):
        raise ValueError(f"Invalid {name}: {value}")


def resolve_eval(args, routing, settings):
    ev = copy.deepcopy(DEFAULT_EVAL)
    if args.eval_config:
        _merge_config(ev, load_config(args.eval_config))
    comparison, sample, sweep = ev["comparison"], ev["sampling"], ev["sweep"]
    # Existing CLI options remain available and override the evaluation JSON.
    for attr, key in (("n_tiles", "n_tiles"), ("seed", "seed"),
                      ("regions_per_tile", "regions_per_tile"), ("z_guard", "z_guard_slices"),
                      ("min_cells", "min_cells"), ("max_attempts", "max_attempts")):
        if getattr(args, attr) is not None:
            sample[key] = getattr(args, attr)
    if args.n_tiles is not None:
        sample["tile_selection"] = "random"
    if args.all_tiles:
        sample["tile_selection"] = "all"
    if args.z_selection is not None:
        sample["z_selection"] = args.z_selection
    if args.xy_size is not None:
        sample.update(xy_mode="crop", xy_size_px=args.xy_size)
    if args.z_size is not None:
        sample["z_size_slices"] = args.z_size
    if args.z_depth_um is not None:
        sample.update(z_depth_um=args.z_depth_um, z_size_slices=None)
    if args.xy_range is not None:
        sweep["x"]["range_px"] = sweep["y"]["range_px"] = args.xy_range
    if args.xy_step is not None:
        sweep["x"]["step_px"] = sweep["y"]["step_px"] = args.xy_step
    for attr, key in (("z_range", "range_slices"), ("z_step", "step_slices")):
        if getattr(args, attr) is not None:
            sweep["z"][key] = getattr(args, attr)
    if args.fixed_channel is not None:
        comparison["fixed_channel"] = args.fixed_channel
    if args.workers is not None:
        ev["workers"] = args.workers
    if args.per_tile_plots:
        ev["output"]["per_tile_plots"] = True
    if args.out_dir:
        ev["output"]["dir"] = os.path.abspath(args.out_dir)
    if comparison["fixed_channel"] is None:
        comparison["fixed_channel"] = settings["reference_channel"]
    active = {ch["id"] for ch in routing}
    fixed = comparison["fixed_channel"]
    if comparison["moving_channels"] is None:
        comparison["moving_channels"] = [ch["id"] for ch in routing if ch["id"] != fixed]
    moving = comparison["moving_channels"]
    if (fixed not in active or not isinstance(moving, list) or not moving
            or any(cid not in active or cid == fixed for cid in moving)
            or len(set(moving)) != len(moving)):
        raise ValueError("comparison must select one active fixed channel and distinct active moving channels")
    if comparison["coordinate_frame"] != "saved_offsets":
        raise ValueError("Only coordinate_frame=saved_offsets is supported")
    if sample["tile_selection"] not in ("random", "all"):
        raise ValueError("sampling.tile_selection must be random or all")
    if sample["z_selection"] not in ("random", "center"):
        raise ValueError("sampling.z_selection must be random or center")
    if sample["z_selection"] == "center" and (sample["held_out"] or sample["regions_per_tile"] != 1):
        raise ValueError("Central sampling requires held_out=false and regions_per_tile=1")
    if sample["xy_mode"] not in ("full_tile", "crop"):
        raise ValueError("sampling.xy_mode must be full_tile or crop")
    if sample["insufficient_depth_policy"] not in ("replace_tile", "allow_in_window"):
        raise ValueError("insufficient_depth_policy must be replace_tile or allow_in_window")
    for key in ("held_out",):
        if not isinstance(sample[key], bool):
            raise ValueError(f"sampling.{key} must be boolean")
    if not isinstance(comparison["include_unaligned_baseline"], bool):
        raise ValueError("include_unaligned_baseline must be boolean")
    for key in ("n_tiles", "regions_per_tile", "max_attempts"):
        _positive(sample[key], f"sampling.{key}", integer=True)
    for key in ("seed", "min_cells"):
        _positive(sample[key], f"sampling.{key}", integer=True, allow_zero=True)
    _positive(sample["xy_size_px"], "sampling.xy_size_px")
    _positive(sample["z_depth_um"], "sampling.z_depth_um")
    if sample["z_size_slices"] is not None:
        _positive(sample["z_size_slices"], "sampling.z_size_slices", integer=True)
    if sample["z_guard_slices"] is None:
        sample["z_guard_slices"] = max(v["max_cell_z_span"] for v in settings["z_link"].values())
    _positive(sample["z_guard_slices"], "sampling.z_guard_slices", integer=True, allow_zero=True)
    for axis in ("x", "y", "z"):
        unit = "slices" if axis == "z" else "px"
        for kind in ("range", "step"):
            _positive(sweep[axis][f"{kind}_{unit}"], f"sweep.{axis}.{kind}",
                      integer=True, allow_zero=kind == "range")
    metric = ev["metric"]
    if metric["type"] != "voxel_iou" or metric["aggregation"] not in ("pooled", "mean"):
        raise ValueError("metric.type must be voxel_iou; aggregation must be pooled or mean")
    if metric["xy_bin_size_px"] is None:
        metric["xy_bin_size_px"] = settings["voxel_bin_size_px"]
    _positive(metric["xy_bin_size_px"], "metric.xy_bin_size_px", integer=True)
    _positive(ev["workers"], "workers", integer=True)
    for key in ("xy_resolution_um", "z_resolution_um"):
        _positive(settings[key], key)
    depth = sample["z_size_slices"]
    if depth is None:
        depth = max(1, int(math.floor(sample["z_depth_um"] / settings["z_resolution_um"] + 0.5)))
    sample["actual_z_size_slices"] = int(depth)
    sample["actual_z_depth_um"] = depth * settings["z_resolution_um"]
    out = ev["output"]
    if out["plot_units"] not in ("um", "native"):
        raise ValueError("output.plot_units must be um or native")
    if (not isinstance(out["formats"], list) or not out["formats"]
            or any(fmt not in ("png", "pdf", "svg") for fmt in out["formats"])):
        raise ValueError("output.formats must be a nonempty list of png/pdf/svg")
    for key in ("show_tile_curves", "per_tile_plots"):
        if not isinstance(out[key], bool):
            raise ValueError(f"output.{key} must be boolean")
    if not isinstance(out["dir"], str) or not out["dir"]:
        raise ValueError("output.dir must be a nonempty path")
    return ev


def _sweep_deltas(rng_ranges):
    out = []
    for axis, (radius, step) in zip(("x", "y", "z"), rng_ranges):
        # Always symmetric, including zero and both endpoints.
        positive = set(range(0, radius + 1, step)) | {radius}
        out.extend((axis, v) for v in sorted(positive | {-v for v in positive}))
    return out


def _shift_cells(cells, dx, dy, dz):
    out = []
    for cell in cells:
        c = dict(cell)
        for key in ("cx", "x1_3d", "x2_3d"):
            c[key] = cell[key] + dx
        for key in ("cy", "y1_3d", "y2_3d"):
            c[key] = cell[key] + dy
        for key in ("cz", "z_min", "z_max"):
            c[key] = cell[key] + dz
        c["per_z_boxes"] = {int(z) + dz: [b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy]
                            for z, b in cell.get("per_z_boxes", {}).items()}
        out.append(c)
    return out


def _grid(cells, region, bin_size, xy_res_um, z_res_um):
    x0, x1, y0, y1, z0, z1 = region
    return _voxelize_to_grid(cells, z0, z1, x0, x1, y0, y1,
                             bin_size, xy_res_um, z_res_um)



def shifted_grid(cells, region, shift, bin_size, xy_res_um, z_res_um):
    """Equivalent to shifting all boxes, without allocating new cell dictionaries."""
    dx, dy, dz = shift
    x0, x1, y0, y1, z0, z1 = region
    inverse_region = (x0 - dx, x1 - dx, y0 - dy, y1 - dy, z0 - dz, z1 - dz)
    return _grid(cells, inverse_region, bin_size, xy_res_um, z_res_um)


def z_halo_cells(cells, region, z_radius, xy_res_um, z_res_um):
    """Keep every cell whose rasterized Z extent can enter the ROI during scanning."""
    lower, upper = region[4] - z_radius, region[5] - 1 + z_radius
    selected = []
    for c in cells:
        effective_lo, effective_hi = _effective_z_extent(c, xy_res_um, z_res_um)
        if math.floor(effective_lo) <= upper and math.ceil(effective_hi) >= lower:
            selected.append(c)
    return selected


def grid_counts(grid_ref, grid_tgt):
    intersection = int(np.count_nonzero(np.logical_and(grid_ref, grid_tgt)))
    union = int(np.count_nonzero(grid_ref)) + int(np.count_nonzero(grid_tgt)) - intersection
    # Two empty grids have undefined IoU; do not report them as perfect alignment.
    return intersection, union, intersection / union if union else np.nan


def region_voxel_iou(grid_ref, tgt_cells, region, dx, dy, dz, bin_size, xy_res_um, z_res_um):
    return grid_counts(grid_ref, shifted_grid(tgt_cells, region, (dx, dy, dz),
                                             bin_size, xy_res_um, z_res_um))[2]


def _cells_overlapping(cells, region):
    x0, x1, y0, y1, z0, z1 = region
    return [c for c in cells if c["x2_3d"] >= x0 and c["x1_3d"] < x1
            and c["y2_3d"] >= y0 and c["y1_3d"] < y1
            and c["z_max"] >= z0 and c["z_min"] < z1]


def _cells_centroid_in(cells, region):
    x0, x1, y0, y1, z0, z1 = region
    return [c for c in cells if x0 <= c["cx"] < x1 and y0 <= c["cy"] < y1
            and z0 <= c["cz"] < z1]


def load_slice_support(tile, cid, params):
    """Actual processed one-based slice indices, never inferred from cell detections."""
    path = Path(params["raw_dir"]) / f"{tile}_{cid}_inputs.json"
    if path.is_file():
        manifest = load_config(path)
        if manifest.get("detection_params", {}).get("DOWNSAMPLE", False):
            raise ValueError(f"{cid}: downsampled detections cannot evaluate contiguous Z windows")
        images = manifest.get("images", [])
        if not images:
            raise ValueError(f"{cid}: empty input manifest")
        support = {i + 1 for i, row in enumerate(images) if row.get("status") == "processed"}
        if not support:
            raise ValueError(f"{cid}: no processed slices in manifest")
        return support, {"source": str(path), "count": len(images),
                         "processed": len(support), "kind": "processed_manifest"}

    # Legacy results: verify TIFF slice names against the anchor used by worker.py.
    files = tile_slice_names(cid, tile, params)
    anchor = tile_slice_names(params["anchor_channel"], tile, params)
    if not files or not anchor:
        raise ValueError(f"{cid}: no input manifest or readable raw TIFF index; "
                         "make original channel directories accessible")
    names = set(files)
    support = {i + 1 for i, name in enumerate(anchor) if name in names}
    if not support:
        raise ValueError(f"{cid}: no slices matching the acquisition anchor")
    return support, {"source": params["channel_roots"].get(cid), "count": len(anchor),
                     "processed": len(support), "kind": "legacy_tiff_index"}


def _shift_limits(offset, radii, moving, include_baseline, fixed_offset):
    lo = [offset[i] - (radii[i] if moving else 0) for i in range(3)]
    hi = [offset[i] + (radii[i] if moving else 0) for i in range(3)]
    if include_baseline:
        lo = [min(v, fixed_offset[i]) for i, v in enumerate(lo)]
        hi = [max(v, fixed_offset[i]) for i, v in enumerate(hi)]
    return lo, hi


def valid_region_starts(supports, limits, tile_size, depth, align_window, guard, held_out):
    """Common valid footprint; reject missing source slices and solving-window leakage."""
    x0 = math.ceil(max(hi[0] for lo, hi in limits.values()))
    x1 = math.floor(min(tile_size + lo[0] for lo, hi in limits.values()))
    y0 = math.ceil(max(hi[1] for lo, hi in limits.values()))
    y1 = math.floor(min(tile_size + lo[1] for lo, hi in limits.values()))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Offsets and scan ranges leave no common XY footprint")
    zlo = math.ceil(max(min(supports[cid]) + hi[2] for cid, (lo, hi) in limits.items()))
    zhi = math.floor(min(max(supports[cid]) + 1 + lo[2] for cid, (lo, hi) in limits.items()))
    valid, independent = [], []
    for start in range(zlo, zhi - depth + 1):
        is_valid, is_independent = True, True
        for cid, (lo, hi) in limits.items():
            source_lo, source_hi = start - hi[2], start + depth - 1 - lo[2]
            if not all(z in supports[cid] for z in range(source_lo, source_hi + 1)):
                is_valid = False
                break
            if not (source_hi < align_window[0] - guard or source_lo > align_window[1] + guard):
                is_independent = False
        if is_valid:
            valid.append(start)
            if is_independent:
                independent.append(start)
    return (x0, x1, y0, y1), (independent if held_out else valid), valid



def order_z_starts(starts, mode, rng):
    """Center the window in the common usable image support, independent of cell counts."""
    if mode == "random":
        return rng.permutation(starts).tolist()
    midpoint = (min(starts) + max(starts)) / 2
    return [min(starts, key=lambda start: (abs(start - midpoint), start))]


def process_tile(tile_name, params):
    """Return (curve rows, warnings); data eligibility failures permit replacement."""
    try:
        return _process_tile(tile_name, params)
    except (ValueError, FileNotFoundError, KeyError, pd.errors.EmptyDataError) as exc:
        return [], [f"{tile_name}: {exc}"]


def _process_tile(tile, params):
    settings, ev = params["settings"], params["eval"]
    sampling = ev["sampling"]
    fixed = ev["comparison"]["fixed_channel"]
    moving = ev["comparison"]["moving_channels"]
    channels = [fixed] + moving
    include_baseline = ev["comparison"]["include_unaligned_baseline"]
    offsets = load_config(os.path.join(params["align_dir"], f"{tile}_offsets.json"))
    shifts = {}
    for cid in channels:
        values = tuple(offsets[cid][key] for key in ("dx", "dy", "dz"))
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) or int(v) != v for v in values):
            raise ValueError(f"{cid}: offsets must be finite integer pixels/slices")
        shifts[cid] = tuple(int(v) for v in values)
    frame = settings.get("stitching_reference_channel", settings["reference_channel"])
    if frame not in offsets or any(offsets[frame][key] != 0 for key in ("dx", "dy", "dz")):
        raise ValueError(f"Offsets do not match the saved stitching frame {frame}")

    # Reconstruct every active channel to match the solver's z-center definition.
    per_ch, z_center = build_tile_vol_lists(params["filtered_dir"], tile, params["routing"], settings)
    measured_path = os.path.join(params["align_dir"], f"{tile}_measured_offsets.json")
    if os.path.isfile(measured_path):
        measured = load_config(measured_path)
        if measured.get("z_center") is not None:
            z_center = float(measured["z_center"])
    if z_center is None or not math.isfinite(z_center):
        raise ValueError("No finite alignment Z center")
    saved_window = measured.get("sample_window") if os.path.isfile(measured_path) else None
    if saved_window:
        align_window = (saved_window["z_start"], saved_window["z_stop_exclusive"])
    else:
        half = settings["sample_z_center_count"] // 2
        align_window = (z_center - half, z_center + half)
    supports, sources = {}, {}
    for cid in channels:
        supports[cid], sources[cid] = load_slice_support(tile, cid, params)
    radii = [r for r, step in params["sweep"]]
    limits = {cid: _shift_limits(shifts[cid], radii, cid != fixed, include_baseline, shifts[fixed])
              for cid in channels}
    # Inferred single-slice volumes can extend beyond the observed cell span.
    # Include that extent as well as the configured linking guard for independence.
    raster_guard = 0
    for cid in channels:
        for c in per_ch[cid]:
            if c["z_min"] <= align_window[1] and c["z_max"] >= align_window[0]:
                effective_lo, effective_hi = _effective_z_extent(
                    c, settings["xy_resolution_um"], settings["z_resolution_um"])
                raster_guard = max(raster_guard, math.ceil(max(
                    c["z_min"] - effective_lo, effective_hi - c["z_max"], 0)))
    actual_guard = sampling["z_guard_slices"] + raster_guard
    footprint, starts, valid = valid_region_starts(
        supports, limits, params["tile_size"], sampling["actual_z_size_slices"],
        align_window, actual_guard, sampling["held_out"])
    held_out = bool(sampling["held_out"])
    warnings = []
    if not starts:
        if (sampling["held_out"] and valid
                and sampling["insufficient_depth_policy"] == "allow_in_window"):
            starts, held_out = valid, False
            warnings.append(f"{tile}: no held-out window; explicitly allowed in-window fallback")
        else:
            raise ValueError("No valid Z window with requested depth/halo/held-out policy")
    # Shift the COMPLETE tile once, before any spatial selection.
    aligned = {cid: _shift_cells(per_ch[cid], *shifts[cid]) for cid in channels}
    baseline = {cid: _shift_cells(per_ch[cid], *shifts[fixed]) for cid in moving} if include_baseline else {}
    rng = np.random.default_rng(sampling["seed"] + zlib.crc32(tile.encode()))
    z_selection = sampling.get("z_selection", "random")
    starts = order_z_starts(starts, z_selection, rng)
    grid_args = (ev["metric"]["xy_bin_size_px"], settings["xy_resolution_um"], settings["z_resolution_um"])
    deltas = _sweep_deltas(params["sweep"])
    rows, accepted = [], set()
    for region_id in range(sampling["regions_per_tile"]):
        region = None
        for attempt in range(1 if z_selection == "center" else sampling["max_attempts"]):
            start = starts[attempt % len(starts)]
            x0, x1, y0, y1 = footprint
            if sampling["xy_mode"] == "crop":
                size = sampling["xy_size_px"]
                if size > min(x1 - x0, y1 - y0):
                    raise ValueError("Requested XY crop exceeds the common valid footprint")
                x0 = int(rng.integers(math.ceil(x0), math.floor(x1 - size) + 1))
                y0 = int(rng.integers(math.ceil(y0), math.floor(y1 - size) + 1))
                x1, y1 = x0 + size, y0 + size
            candidate = (x0, x1, y0, y1, int(start), int(start) + sampling["actual_z_size_slices"])
            if candidate in accepted:
                continue
            if all(len(_cells_overlapping(aligned[cid], candidate)) >= sampling["min_cells"]
                   for cid in channels):
                region = candidate
                accepted.add(region)
                break
        if region is None:
            raise ValueError(f"Cannot select {sampling['regions_per_tile']} distinct signal-qualified windows "
                             f"(min_cells={sampling['min_cells']}, attempts={sampling['max_attempts']})")
        ref_grid = _grid(aligned[fixed], region, *grid_args)
        n_ref = len(_cells_overlapping(aligned[fixed], region))
        # Auxiliary containment remains available for soma/TF comparisons.
        fixed_type = next(ch.get("type", "soma") for ch in params["routing"] if ch["id"] == fixed)
        soma_idx = _prepare_soma_containment_index(aligned[fixed]) if fixed_type == "soma" and aligned[fixed] else None
        for cid in moving:
            ctype = next(ch.get("type", "soma") for ch in params["routing"] if ch["id"] == cid)
            unaligned = grid_counts(ref_grid, _grid(baseline[cid], region, *grid_args)) if include_baseline else (np.nan,) * 3
            tf_cells = _cells_centroid_in(aligned[cid], region) if ctype == "tf" else []
            tf_arrays = _cell_arrays(tf_cells) if tf_cells else None
            target_halo = z_halo_cells(
                aligned[cid], region, radii[2], settings["xy_resolution_um"], settings["z_resolution_um"])
            for axis, delta in deltas:
                extra = tuple(delta if axis == a else 0 for a in ("x", "y", "z"))
                target = shifted_grid(target_halo, region, extra, *grid_args)
                intersection, union, iou = grid_counts(ref_grid, target)
                containment = np.nan
                if soma_idx is not None and tf_arrays is not None:
                    count, _ = _containment_score(
                        soma_idx, tf_arrays, *extra, settings["max_center_dist_ratio"],
                        0, settings["containment_z_pad"])
                    containment = count / len(tf_cells)
                total_shift = tuple(shifts[cid][i] + extra[i] for i in range(3))
                rows.append({
                    "tile": tile, "region_id": region_id, "held_out": held_out, "z_selection": z_selection,
                    **dict(zip(("x0", "x1", "y0", "y1", "z0", "z1"), region)),
                    "align_z_lo": align_window[0], "align_z_hi": align_window[1],
                    "actual_z_guard_slices": actual_guard,
                    "coordinate_frame": frame, "ref_channel": fixed, "channel": cid, "ch_type": ctype,
                    "axis": axis, "delta": delta,
                    "delta_um": delta * (settings["z_resolution_um"] if axis == "z" else settings["xy_resolution_um"]),
                    **dict(zip(("dx", "dy", "dz"), total_shift)),
                    **dict(zip(("base_dx", "base_dy", "base_dz"), shifts[cid])),
                    **dict(zip(("fixed_dx", "fixed_dy", "fixed_dz"), shifts[fixed])),
                    "intersection": intersection, "union": union, "voxel_iou": iou,
                    "unaligned_intersection": unaligned[0], "unaligned_union": unaligned[1],
                    "unaligned_iou": unaligned[2], "containment": containment,
                    "n_ref_cells": n_ref, "n_tgt_cells": len(_cells_overlapping(aligned[cid], region)),
                    "n_tf_in_region": len(tf_cells), "slice_support": json.dumps(sources, sort_keys=True),
                })
    return rows, warnings


def aggregate_curves(df):
    out = []
    for (channel, axis, delta), group in df.groupby(["channel", "axis", "delta"], sort=True):
        inter, union = int(group["intersection"].sum()), int(group["union"].sum())
        baseline = group.dropna(subset=["unaligned_union"])
        ui, uu = baseline["unaligned_intersection"].sum(), baseline["unaligned_union"].sum()
        # Equal tile weights even when more than one region is requested.
        tile_values = group.groupby("tile")["voxel_iou"].mean()
        out.append({
            "channel": channel, "ref_channel": group["ref_channel"].iloc[0],
            "axis": axis, "delta": delta, "delta_um": group["delta_um"].iloc[0],
            "intersection": inter, "union": union,
            "pooled_iou": inter / union if union else np.nan,
            "mean_iou": tile_values.mean(), "std_iou": tile_values.std(),
            "n_tiles": group["tile"].nunique(), "n_defined_tiles": int(tile_values.notna().sum()),
            "n_regions": len(group),
            "unaligned_intersection": ui if len(baseline) else np.nan,
            "unaligned_union": uu if len(baseline) else np.nan,
            "unaligned_pooled_iou": ui / uu if uu else np.nan,
            "unaligned_mean_iou": baseline.groupby("tile")["unaligned_iou"].mean().mean(),
        })
    return pd.DataFrame(out)


def curve_summary(group, column, baseline_column):
    g = group.sort_values("delta")
    valid = g.dropna(subset=[column])
    zero = g.loc[g["delta"] == 0].iloc[0]
    value = float(zero[column])
    baseline = float(zero[baseline_column])
    result = {
        "aligned_iou": value, "unaligned_iou": baseline,
        "absolute_improvement": value - baseline,
        "relative_improvement": (value - baseline) / baseline if baseline > 0 else np.nan,
        "peak_iou": np.nan, "peak_delta": np.nan, "peak_delta_um": np.nan,
        "zero_at_peak": False, "edge_mean": np.nan, "drop_ratio_at_edge": np.nan,
        "half_width_neg_um": np.nan, "half_width_pos_um": np.nan,
    }
    if valid.empty:
        return result
    # Flat/tied peaks choose the one nearest zero, rather than the negative endpoint.
    peak = valid[column].max()
    ties = valid.loc[np.isclose(valid[column], peak)].copy()
    peak_row = ties.assign(distance=ties["delta"].abs()).sort_values(["distance", "delta"]).iloc[0]
    edge = float(g.iloc[[0, -1]][column].mean())
    result.update(peak_iou=float(peak), peak_delta=int(peak_row["delta"]),
                  peak_delta_um=float(peak_row["delta_um"]),
                  zero_at_peak=bool(np.isfinite(value) and np.isclose(value, peak)),
                  edge_mean=edge, drop_ratio_at_edge=(value - edge) / value if value > 0 else np.nan)
    if value > 0:
        for side, mask in (("neg", valid["delta"] < 0), ("pos", valid["delta"] > 0)):
            distances = valid.loc[mask & (valid[column] < value / 2), "delta_um"].abs()
            if len(distances):
                result[f"half_width_{side}_um"] = float(distances.min())
    return result


def summarize(df, metric="voxel_iou"):
    records = []
    for (tile, region, channel, axis), group in df.groupby(["tile", "region_id", "channel", "axis"]):
        baseline = "unaligned_iou"
        if metric != "voxel_iou":
            group = group.assign(no_baseline=np.nan)
            baseline = "no_baseline"
        records.append({"tile": tile, "region_id": region, "channel": channel,
                        "axis": axis, "metric": metric, "held_out": bool(group["held_out"].iloc[0]),
                        **curve_summary(group, metric, baseline)})
    return pd.DataFrame(records)


def summarize_aggregate(aggregate):
    records = []
    for (channel, axis), group in aggregate.groupby(["channel", "axis"]):
        for name in ("pooled", "mean"):
            records.append({"channel": channel, "axis": axis, "aggregation": name,
                            "n_tiles": int(group["n_tiles"].iloc[0]),
                            **curve_summary(group, f"{name}_iou", f"unaligned_{name}_iou")})
    return pd.DataFrame(records)


def plot_overview(df, aggregate, ev, out_dir, sample_name, tile_name=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    channels = ev["comparison"]["moving_channels"]
    fig, axes = plt.subplots(len(channels), 3, figsize=(15, 3.7 * len(channels)), squeeze=False)
    xcol = "delta_um" if ev["output"]["plot_units"] == "um" else "delta"
    kind = ev["metric"]["aggregation"]
    curve_label = "tile IoU" if tile_name is not None else f"{kind} IoU"
    for row, channel in enumerate(channels):
        for col, axis in enumerate(("x", "y", "z")):
            ax = axes[row, col]
            sub = df.loc[(df["channel"] == channel) & (df["axis"] == axis)]
            curve = aggregate.loc[(aggregate["channel"] == channel) & (aggregate["axis"] == axis)].sort_values("delta")
            if ev["output"]["show_tile_curves"] and (tile_name is None or sub["region_id"].nunique() > 1):
                for _, tile_curve in sub.groupby(["tile", "region_id"]):
                    tile_curve = tile_curve.sort_values("delta")
                    ax.plot(tile_curve[xcol], tile_curve["voxel_iou"], color="0.65", alpha=.5, lw=.8)
            ax.plot(curve[xcol], curve[f"{kind}_iou"], color="crimson", lw=2,
                    marker="o", ms=3, label=curve_label)
            baseline = curve[f"unaligned_{kind}_iou"].iloc[0]
            if np.isfinite(baseline):
                ax.axhline(baseline, color="darkorange", ls="--", label="unaligned IoU")
            finite = curve.dropna(subset=[f"{kind}_iou"])
            if not finite.empty:
                peak = curve_summary(curve, f"{kind}_iou", f"unaligned_{kind}_iou")
                px = peak["peak_delta_um"] if xcol == "delta_um" else peak["peak_delta"]
                ax.scatter([px], [peak["peak_iou"]], color="black", s=20, zorder=5)
                ax.annotate(f"peak {px:g}", (px, peak["peak_iou"]),
                            xytext=(4, 7), textcoords="offset points", fontsize=8)
            elif tile_name is not None:
                message = "IoU undefined"
                if curve["union"].eq(0).all():
                    message += "\nBoth channels empty throughout scan"
                ax.text(.5, .5, message, transform=ax.transAxes,
                        ha="center", va="center", color="0.4", fontsize=10)
            ax.axvline(0, color="steelblue", ls=":", lw=1)
            unit = "µm" if xcol == "delta_um" else ("slices" if axis == "z" else "px")
            ax.set_xlabel(f"Additional Δ{axis} ({unit})")
            xmin, xmax = float(curve[xcol].min()), float(curve[xcol].max())
            if xmax > xmin:
                margin = .05 * (xmax - xmin)
                ax.set_xlim(xmin - margin, xmax + margin)
            ax.set_ylabel("3D box occupancy IoU")
            ax.set_title(f"{channel} vs {ev['comparison']['fixed_channel']} — {axis.upper()}")
            ax.set_ylim(bottom=0, top=max(.01, ax.get_ylim()[1] * 1.12))
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    central = ev["sampling"].get("z_selection") == "center"
    status = "center windows; may overlap solver data" if central else (
        "held-out" if df["held_out"].all() else "includes in-window evaluation")
    fig.suptitle(f"{sample_name}: saved-offset sensitivity; {df['tile'].nunique()} tiles; {status}")
    if tile_name is not None:
        windows = ", ".join(f"[{int(r.z0)}, {int(r.z1)})" for r in
                            df[["z0", "z1"]].drop_duplicates().itertuples())
        fig.suptitle(f"{sample_name}: tile {tile_name}; Z {windows} (slice indices); {status}")
    depth = ev["sampling"].get("actual_z_depth_um", (df["z1"].iloc[0] - df["z0"].iloc[0]))
    frame = ", ".join(sorted(df["coordinate_frame"].unique()))
    caption = (
        f"Each tile: {depth:g} um Z window; saved frame: {frame}. Gray: individual tiles; red points/line: {kind} IoU (no fit).\n"
        "Blue vertical: saved alignment (extra shift = 0); orange dashed: unaligned baseline; black point: scanned maximum.\n"
        "IoU uses 3D detection-box occupancy. Both-empty IoU is undefined. Each panel has its own Y-axis scale.")
    if tile_name is not None:
        caption = (
            f"This tile: {depth:g} um Z window; saved frame: {frame}. Red points/line: tile IoU (no fit).\n"
            "Blue vertical: saved alignment (extra shift = 0); orange dashed: this tile's unaligned baseline; black point: scanned maximum.\n"
            "IoU uses 3D detection-box occupancy. Both-empty IoU is undefined. Each panel has its own Y-axis scale.")
    fig.text(.5, .012, caption, ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, .075, 1, .97))
    try:
        for fmt in ev["output"]["formats"]:
            stem = f"{sample_name}_{tile_name}" if tile_name is not None else sample_name
            fig.savefig(os.path.join(out_dir, f"{stem}_alignment_sensitivity.{fmt}"), dpi=200)
    finally:
        plt.close(fig)



def plot_tiles(df, ev, out_dir, sample_name):
    """One channel-by-axis figure per tile, reusing the saved scan measurements."""
    target = Path(out_dir) / "tile_plots"
    target.mkdir(parents=True, exist_ok=True)
    index = []
    groups = df.groupby("tile", sort=True)
    for tile, tile_df in tqdm(groups, total=groups.ngroups, desc="Tile figures"):
        aggregate = aggregate_curves(tile_df)
        plot_overview(tile_df, aggregate, ev, str(target), sample_name, tile_name=tile)
        record = {"tile": tile, "regions": tile_df["region_id"].nunique(),
                  "z0": int(tile_df["z0"].min()), "z1": int(tile_df["z1"].max())}
        record.update({fmt: f"{sample_name}_{tile}_alignment_sensitivity.{fmt}"
                       for fmt in ev["output"]["formats"]})
        index.append(record)
    pd.DataFrame(index).to_csv(target / "index.csv", index=False)
    print(f"Saved {len(index)} tile figures in {target}", flush=True)
    return target


def replot_saved(args, results_dir):
    """Render existing tables without loading detections or recomputing any IoU."""
    if args.no_plots:
        raise ValueError("--plot-only cannot be combined with --no-plots")
    configured = load_config(args.eval_config).get("output", {}) if args.eval_config else {}
    out_dir = Path(args.out_dir).resolve() if args.out_dir else Path(results_dir) / configured.get("dir", "alignment_validation")
    manifest = load_config(out_dir / "sampling.json")
    ev = copy.deepcopy(manifest["evaluation"])
    ev["output"].update(configured)
    if args.per_tile_plots:
        ev["output"]["per_tile_plots"] = True
    sample_name = Path(results_dir).parent.name
    df = pd.read_csv(out_dir / f"{sample_name}_curves.csv")
    if args.tiles:
        wanted = [tile.strip() for tile in args.tiles.split(",") if tile.strip()]
        missing = set(wanted) - set(df["tile"])
        if not wanted or missing:
            raise ValueError(f"No saved curves for requested tiles: {sorted(missing)}")
        df = df.loc[df["tile"].isin(wanted)]
        plot_tiles(df, ev, out_dir, sample_name)
    else:
        plot_overview(df, aggregate_curves(df), ev, str(out_dir), sample_name)
        if ev["output"].get("per_tile_plots", False):
            plot_tiles(df, ev, out_dir, sample_name)


def tile_slice_names(cid, tile, params):
    """Read TIFF names only for the tile currently being evaluated."""
    cache = params.setdefault("_slice_cache", {})
    key = (cid, tile)
    if key in cache:
        return cache[key]
    root_value = params["channel_roots"].get(cid)
    if not root_value:
        return []
    root = Path(root_value)
    # The acquisition uses root/<row>/<row>_<column>; flat layouts also work.
    guesses = [root / tile, root / tile.split("_")[0] / tile]
    directory = next((p for p in guesses if p.is_dir()), None)
    if directory is None:
        # Locate alternative layouts without opening known tile directories.
        known_tiles = set(params.get("candidate_tiles", [tile]))
        matches = []
        for parent, dirs, _ in os.walk(root):
            for name in dirs:
                if name == tile:
                    matches.append(Path(parent) / name)
            dirs[:] = [name for name in dirs if name not in known_tiles]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous raw tile basename {tile} under {root}")
        directory = matches[0] if matches else None
    if directory is None:
        return []
    names = sorted(p.stem for p in directory.iterdir()
                   if p.suffix.lower() in (".tif", ".tiff") and not p.name.startswith("."))
    cache[key] = names
    return names


def build_legacy_indices(results_dir, config, routing, channels, candidates):
    """Resolve raw roots once; defer all TIFF listing until a tile is attempted."""
    roots = {}
    wanted = set(channels) | {routing[0]["id"]}
    needs_fallback = any(not (Path(results_dir) / "1_2d_raw" / f"{t}_{c}_inputs.json").is_file()
                         for t in candidates for c in channels)
    if not needs_fallback:
        return roots, {}
    for ch in routing:
        cid = ch["id"]
        if cid not in wanted:
            continue
        original = config.get("paths", {}).get(ch.get("dir_key"), "")
        options = [Path(original)] if original else []
        if original:
            options.append(Path(results_dir).parent / original.replace("\\", "/").rstrip("/").split("/")[-1])
        root = next((p for p in options if p.is_dir()), None)
        if root is not None:
            roots[cid] = str(root.resolve())
    return roots, {}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", required=True, help="Results directory or sample containing detection_results/")
    parser.add_argument("--config", help="Original inference config, used only if runtime_config.json is absent")
    parser.add_argument("--eval-config", help="Independent sampling/sweep/output JSON (supports // comments)")
    parser.add_argument("--out-dir", help="Output override, relative to current working directory")
    parser.add_argument("--tiles", help="Explicit comma-separated tile names; disables random replacement")
    parser.add_argument("--all-tiles", action="store_true", help="Evaluate every completed alignment tile")
    parser.add_argument("--z-selection", choices=("random", "center"), help="Choose random or image-center Z windows")
    parser.add_argument("--fixed-channel", help="Fixed comparison channel; coordinates still use saved offsets")
    for flag, help_text in (
        ("n-tiles", "Number of eligible random tiles (default 10)"),
        ("regions-per-tile", "Distinct windows per tile (default 1)"),
        ("z-size", "Z depth in slices; overrides physical depth"),
        ("z-guard", "Raw-source guard outside the alignment window in slices"),
        ("xy-range", "Symmetric X/Y sweep radius in pixels"),
        ("xy-step", "X/Y step in pixels"), ("z-range", "Z radius in slices"),
        ("z-step", "Z step in slices"), ("min-cells", "Minimum aligned boxes per channel/window"),
        ("max-attempts", "Window selection attempts per region"),
        ("seed", "Deterministic random seed"), ("workers", "CPU worker processes"),
    ):
        parser.add_argument(f"--{flag}", type=int, default=None, help=help_text)
    parser.add_argument("--xy-size", type=float, help="Use an XY crop of this size instead of full valid tile")
    parser.add_argument("--z-depth-um", type=float, help="Physical depth; overrides --z-size/config slices")
    parser.add_argument("--plot-only", action="store_true", help="Render saved curves without recomputing IoU")
    parser.add_argument("--per-tile-plots", action="store_true", help="Save a separate channel-by-axis figure for each tile")
    parser.add_argument("--no-plots", action="store_true", help="Save tables/manifest without importing matplotlib")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    results_dir = resolve_results_dir(args.sample)
    if args.plot_only:
        replot_saved(args, results_dir)
        return
    config, routing, settings, align_dir, source = load_settings(results_dir, args.config)
    if config.get("detection_params", {}).get("DOWNSAMPLE", False):
        raise ValueError("Contiguous Z evaluation requires detections without Z downsampling")
    ev = resolve_eval(args, routing, settings)
    all_tiles = list_tiles(align_dir)
    if not all_tiles:
        raise ValueError(f"No tile offsets in {align_dir}")
    explicit = args.tiles is not None
    all_mode = ev["sampling"]["tile_selection"] == "all"
    if explicit:
        candidates = [t.strip() for t in args.tiles.split(",") if t.strip()]
        if not candidates or len(candidates) != len(set(candidates)):
            raise ValueError("--tiles must contain distinct nonempty tile names")
        missing = set(candidates) - set(all_tiles)
        if missing:
            raise ValueError(f"Missing tile offsets: {sorted(missing)}")
    elif all_mode:
        candidates = all_tiles
    else:
        candidates = np.random.default_rng(ev["sampling"]["seed"]).permutation(all_tiles).tolist()
    requested = len(candidates) if explicit or all_mode else ev["sampling"]["n_tiles"]
    out_dir = os.path.join(results_dir, ev["output"]["dir"])
    os.makedirs(out_dir, exist_ok=True)
    channels = [ev["comparison"]["fixed_channel"]] + ev["comparison"]["moving_channels"]
    print("Resolving raw slice sources (TIFF lists are loaded per attempted tile)...", flush=True)
    roots, indices = build_legacy_indices(results_dir, config, routing, channels, candidates)
    sweep = tuple((ev["sweep"][a][f"range_{'slices' if a == 'z' else 'px'}"],
                   ev["sweep"][a][f"step_{'slices' if a == 'z' else 'px'}"]) for a in ("x", "y", "z"))
    tile_size = config.get("detection_params", {}).get("tILESIZE", 2048)
    _positive(tile_size, "detection_params.tILESIZE", integer=True)
    params = {
        "settings": settings, "eval": ev, "routing": routing, "align_dir": align_dir,
        "filtered_dir": os.path.join(results_dir, "2_2d_filtered"),
        "raw_dir": os.path.join(results_dir, "1_2d_raw"), "tile_size": tile_size,
        "sweep": sweep, "slice_files": indices, "channel_roots": roots,
        "anchor_channel": routing[0]["id"], "candidate_tiles": candidates,
    }
    sample_name = Path(results_dir).parent.name
    sampling = ev["sampling"]
    print(f"Results: {results_dir}\nSettings: {source}")
    print(f"Saved frame: {settings.get('stitching_reference_channel', settings['reference_channel'])}; "
          f"fixed comparison: {channels[0]}; moving: {channels[1:]}")
    print(f"Requested: {requested} tiles; depth {sampling['actual_z_size_slices']} slices "
          f"({sampling['actual_z_depth_um']:g} µm); XY {sampling['xy_mode']}")
    print("Full-tile mode uses the common valid interior for all offsets, sweep points and baseline.")
    print(f"Output: {out_dir}", flush=True)
    rows, warnings, selected, attempted = [], [], [], []
    worker_count = int(ev["workers"])
    pool = (ProcessPoolExecutor(max_workers=worker_count, mp_context=mp.get_context("spawn"))
            if worker_count > 1 else None)
    try:
        if all_mode:
            checkpoint_dir = Path(out_dir) / "tile_curves"
            checkpoint_dir.mkdir(exist_ok=True)
            if pool:
                pending = {pool.submit(process_tile, tile, params): tile for tile in candidates}
                completed = ((pending[future], future.result()) for future in as_completed(pending))
            else:
                completed = ((tile, process_tile(tile, params)) for tile in candidates)
            with tqdm(total=requested, desc="All tiles") as progress:
                for tile, (tile_rows, tile_warnings) in completed:
                    warnings.extend(tile_warnings)
                    attempted.append({"tile": tile, "accepted": bool(tile_rows), "warnings": tile_warnings})
                    if tile_rows:
                        rows.extend(tile_rows)
                        selected.append(tile)
                        pd.DataFrame(tile_rows).to_csv(checkpoint_dir / f"{tile}_curves.csv", index=False)
                    else:
                        for warning in tile_warnings:
                            print(f"Skipped: {warning}", flush=True)
                    progress.update(1)
                    with open(Path(out_dir) / "progress.json", "w", encoding="utf-8") as handle:
                        json.dump({"requested_tiles": requested, "processed_tiles": len(attempted),
                                   "accepted_tiles": len(selected), "attempted": attempted,
                                   "evaluation": ev}, handle, indent=2)
            # Output ordering remains reproducible across worker completion orders.
            rank = {tile: i for i, tile in enumerate(candidates)}
            selected.sort(key=rank.__getitem__)
            attempted.sort(key=lambda entry: rank[entry["tile"]])
        else:
            with tqdm(total=requested, desc="Eligible tiles") as progress:
                for start in range(0, len(candidates), worker_count):
                    batch = candidates[start:start + worker_count]
                    if pool:
                        futures = [pool.submit(process_tile, t, params) for t in batch]
                        results = [future.result() for future in futures]
                    else:
                        results = [process_tile(t, params) for t in batch]
                    # Consume in candidate order, independent of worker completion order.
                    for tile, (tile_rows, tile_warnings) in zip(batch, results):
                        warnings.extend(tile_warnings)
                        attempted.append({"tile": tile, "accepted": bool(tile_rows), "warnings": tile_warnings})
                        if tile_rows:
                            rows.extend(tile_rows)
                            selected.append(tile)
                            progress.update(1)
                        else:
                            for warning in tile_warnings:
                                print(f"Skipped: {warning}", flush=True)
                        if len(selected) == requested:
                            break
                    if len(selected) == requested:
                        break
    finally:
        if pool:
            pool.shutdown()
    complete = len(selected) == requested
    df = pd.DataFrame(rows)
    regions = []
    if rows:
        cols = ["tile", "region_id", "held_out", "z_selection", "x0", "x1", "y0", "y1", "z0", "z1",
                "align_z_lo", "align_z_hi", "actual_z_guard_slices", "coordinate_frame", "slice_support"]
        regions = df[cols].drop_duplicates(["tile", "region_id"]).to_dict("records")
        for region in regions:
            region["slice_support"] = json.loads(region["slice_support"])
    manifest = {
        "results_dir": results_dir, "settings_source": source,
        "pipeline_config_source": os.path.join(results_dir, "runtime_config.json")
        if os.path.isfile(os.path.join(results_dir, "runtime_config.json")) else args.config,
        "eval_config_source": os.path.abspath(args.eval_config) if args.eval_config else None,
        "evaluation": ev, "alignment_settings": settings,
        "requested_tiles": requested, "selected_tiles": selected, "complete": complete,
        "candidate_order": candidates, "attempted": attempted, "regions": regions, "warnings": warnings,
        "bounds": "Common valid footprint of all saved offsets, perturbations and baseline; Z is [z0,z1).",
        "baseline": "Both raw channels receive the fixed comparison channel's saved offset.",
        "metric": "Binary z-linked box occupancy; alignment rasterizer, including inferred single-slice Z extent.",
        "legacy_bounds": "TIFF-name fallback verifies presence, not image readability; manifests preferred.",
    }
    with open(os.path.join(out_dir, "sampling.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    if not rows:
        raise ValueError(f"No eligible tiles; see {out_dir}/sampling.json")
    df = df.sort_values(["tile", "region_id", "channel", "axis", "delta"])
    aggregate = aggregate_curves(df)
    summary = summarize(df)
    if df["containment"].notna().any():
        summary = pd.concat([summary, summarize(df, "containment")], ignore_index=True)
    pooled_summary = summarize_aggregate(aggregate)
    for name, table in (("curves", df), ("pooled_curves", aggregate),
                        ("summary", summary), ("pooled_summary", pooled_summary)):
        table.to_csv(os.path.join(out_dir, f"{sample_name}_{name}.csv"), index=False)
    print(pooled_summary.loc[pooled_summary["aggregation"] == ev["metric"]["aggregation"]].to_string(index=False))
    print(f"Saved {len(selected)}/{requested} tiles and tables in {out_dir}", flush=True)
    if not args.no_plots:
        plot_overview(df, aggregate, ev, out_dir, sample_name)
        if ev["output"]["per_tile_plots"]:
            plot_tiles(df, ev, out_dir, sample_name)
    if not complete:
        raise ValueError(f"Only {len(selected)}/{requested} eligible tiles; partial results saved. "
                         "Check sampling.json or adjust depth/held-out/min_cells settings.")


if __name__ == "__main__":
    mp.freeze_support()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        main()
    except ValueError as exc:
        raise SystemExit(f"Evaluation failed: {exc}") from exc
