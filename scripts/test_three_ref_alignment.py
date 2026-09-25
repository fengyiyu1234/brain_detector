"""Standalone T70 test: fixed GFP/RFP/Sox9, joint Olig2 alignment.

Reads existing raw detections and saved alignment, writes an independent result
root containing 0_channel_alignment/*.csv and *_offsets.json for all channels.
The main pipeline and its outputs are never changed.

Example:
  python scripts/test_three_ref_alignment.py --tiles 373300_353000
  python src/utils/visualize.py --config Y:/Fengyi/EGFR_brain/T70/detection_results_olig2_three_refs/gui_vis_config.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.config.loader import load_config
from src.core.point_cloud_aligner import (
    _cell_arrays, _containment_score, _prepare_soma_containment_index,
    apply_shift_to_csv, build_cell_boxes, find_shift, find_shift_containment,
    save_tile_offsets,
)
from src.core.z_linker import run_z_linker

CHANNELS = ("GFP", "RFP", "Sox9", "Olig2")
FIXED = CHANNELS[:3]
DEFAULT_RESULTS = Path("Y:/Fengyi/EGFR_brain/T70/detection_results")


def shifted_cells(cells, shift):
    """Translate z-link cells, including their per-slice boxes."""
    dx, dy, dz = (int(v) for v in shift)
    result = []
    for source in cells:
        cell = source.copy()
        for key, delta in (
            ("cx", dx), ("cy", dy), ("cz", dz),
            ("x1_3d", dx), ("x2_3d", dx),
            ("y1_3d", dy), ("y2_3d", dy),
            ("z_min", dz), ("z_max", dz),
        ):
            cell[key] = source[key] + delta
        cell["per_z_boxes"] = {
            int(z) + dz: [box[0] + dx, box[1] + dy,
                          box[2] + dx, box[3] + dy]
            for z, box in source.get("per_z_boxes", {}).items()
        }
        result.append(cell)
    return result


def load_volumes(raw_dir, tile, settings):
    """Use the same raw CSVs, z-link rules and central z window as Stage 2.5."""
    volumes, z_centers = {}, []
    for channel in CHANNELS:
        path = raw_dir / f"{tile}_{channel}_result.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if frame.empty:
            volumes[channel] = []
            continue
        mat = frame[["x1", "y1", "x2", "y2", "score", "mean", "class", "z"]].values
        mat[:, 6] = np.array([f"{value}_{channel}" for value in mat[:, 6]])
        kind = "soma" if channel in ("GFP", "RFP") else "tf"
        _, cells = run_z_linker(mat, **settings["z_link"][kind])
        volumes[channel] = cells
        z_centers.extend(cell.get("cz", 0) for cell in cells)
    center = float(np.median(z_centers)) if z_centers else 0.0
    half = settings["sample_z_center_count"] // 2
    return ({channel: build_cell_boxes(cells, center - half, center + half)
             for channel, cells in volumes.items()}, center)


def read_fixed_offsets(align_dir, tile):
    path = align_dir / f"{tile}_offsets.json"
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    for channel in CHANNELS:
        if channel not in payload or not all(
                axis in payload[channel] for axis in ("dx", "dy", "dz")):
            raise ValueError(f"{path} lacks {channel} dx/dy/dz")
    shifts = {
        channel: tuple(int(payload[channel][axis]) for axis in ("dx", "dy", "dz"))
        for channel in FIXED
    }
    if shifts["GFP"] != (0, 0, 0):
        raise ValueError(f"{path} is not in the T70 GFP frame")
    scores = {
        channel: float(payload[channel].get("iou_score", 0.0))
        for channel in CHANNELS
    }
    old_olig2 = tuple(int(payload["Olig2"][axis]) for axis in ("dx", "dy", "dz"))
    return shifts, scores, old_olig2


def sox_coincidence(tree, olig2_centers, shift, radius=10.0, z_scale=6.0):
    """Mean soft distance to the nearest aligned Sox9 nucleus."""
    if tree is None or not len(olig2_centers):
        return 0.0
    moved = olig2_centers + np.asarray(shift, dtype=float)
    moved[:, 2] *= z_scale
    distance, _ = tree.query(moved, k=1, workers=1)
    return float(np.maximum(0.0, 1.0 - distance / radius).mean())


def candidate_shifts(seeds, xy_radius, z_radius):
    return sorted({
        (sx + dx, sy + dy, sz + dz)
        for sx, sy, sz in seeds.values()
        for dx in range(-xy_radius, xy_radius + 1)
        for dy in range(-xy_radius, xy_radius + 1)
        for dz in range(-z_radius, z_radius + 1)
    })


def choose_joint_shift(cells, center, fixed_shifts, old_shift, settings,
                       local_xy=2, local_z=1, max_scored_cells=1500):
    """Score Olig2 against the union of two soma channels and Sox9 nuclei.

    The pipeline containment scorer counts a given Olig2 nucleus only once,
    even if it fits both GFP and RFP. The two signals are separately normalized
    across candidate shifts, so Sox9 density cannot dominate by itself.
    """
    gfp = cells["GFP"]
    rfp = shifted_cells(cells["RFP"], fixed_shifts["RFP"])
    sox = shifted_cells(cells["Sox9"], fixed_shifts["Sox9"])
    olig2 = cells["Olig2"]
    if not all((gfp, rfp, sox, olig2)):
        raise ValueError("All four channels need detections in the central z window")

    half = settings["sample_z_center_count"] // 2
    kwargs = dict(
        z_lo=center - half, z_hi=center + half,
        bin_size=settings["voxel_bin_size_px"],
        xy_res_um=settings["xy_resolution_um"],
        z_res_um=settings["z_resolution_um"],
        xy_range_px=settings["xy_search_range_px"],
        z_range_slices=settings["z_search_range_slices"],
        fine_xy_px=settings["xy_fine_search_px"],
        fine_z_slices=settings["z_fine_search_slices"],
    )
    soma = gfp + rfp
    soma_seed = find_shift_containment(
        soma, olig2,
        max_center_dist_ratio=settings["max_center_dist_ratio"],
        z_pad=settings["containment_z_pad"],
        coarse=settings.get("containment_coarse", "displacement_hist"),
        **kwargs,
    )[:3]
    sox_seed = find_shift(sox, olig2, **kwargs)[:3]
    seeds = {"original": old_shift, "GFP_RFP": soma_seed, "Sox9": sox_seed}

    # Even sampling caps the cost of scanning thousands of integer shifts.
    if len(olig2) > max_scored_cells:
        indices = np.linspace(0, len(olig2) - 1, max_scored_cells, dtype=int)
        scored_olig2 = [olig2[i] for i in indices]
    else:
        scored_olig2 = olig2
    soma_idx = _prepare_soma_containment_index(soma)
    olig2_arrays = _cell_arrays(scored_olig2)
    sox_centers = _cell_arrays(sox)[0].copy()
    sox_centers[:, 2] *= 6.0
    sox_tree = cKDTree(sox_centers)

    records = []
    for shift in candidate_shifts(seeds, local_xy, local_z):
        count, _ = _containment_score(
            soma_idx, olig2_arrays, *shift,
            settings["max_center_dist_ratio"], 0,
            settings["containment_z_pad"])
        soma_fraction = count / len(scored_olig2)
        sox_score = sox_coincidence(sox_tree, olig2_arrays[0], shift)
        records.append((shift, soma_fraction, sox_score, count))

    soma_values = np.array([row[1] for row in records])
    sox_values = np.array([row[2] for row in records])
    soma_span = float(np.ptp(soma_values))
    sox_span = float(np.ptp(sox_values))
    # A nearly flat signal does not cast a vote.
    soma_norm = ((soma_values - soma_values.min()) / soma_span
                 if soma_span >= 0.002 else np.zeros(len(records)))
    sox_norm = ((sox_values - sox_values.min()) / sox_span
                if sox_span >= 0.002 else np.zeros(len(records)))
    joint = soma_norm + sox_norm
    if not np.any(joint):
        chosen = old_shift
        status = "flat_evidence_kept_original"
    else:
        best = max(range(len(records)), key=lambda i: (
            joint[i], soma_norm[i], sox_norm[i],
            -sum((records[i][0][axis] - old_shift[axis]) ** 2
                 for axis in range(3))))
        chosen = records[best][0]
        status = "joint_peak"

    by_shift = {row[0]: row for row in records}
    selected = by_shift[chosen]
    # Report the pipeline-comparable containment fraction on every Olig2
    # nucleus; the capped sample above is used only to rank candidates.
    all_olig2_arrays = _cell_arrays(olig2)
    full_count, _ = _containment_score(
        soma_idx, all_olig2_arrays, *chosen,
        settings["max_center_dist_ratio"], 0,
        settings["containment_z_pad"])
    old_full_count, _ = _containment_score(
        soma_idx, all_olig2_arrays, *old_shift,
        settings["max_center_dist_ratio"], 0,
        settings["containment_z_pad"])
    gfp_count, _ = _containment_score(
        _prepare_soma_containment_index(gfp), all_olig2_arrays, *chosen,
        settings["max_center_dist_ratio"], 0,
        settings["containment_z_pad"])
    rfp_count, _ = _containment_score(
        _prepare_soma_containment_index(rfp), all_olig2_arrays, *chosen,
        settings["max_center_dist_ratio"], 0,
        settings["containment_z_pad"])
    full_fraction = full_count / len(olig2)
    full_sox_score = sox_coincidence(sox_tree, all_olig2_arrays[0], chosen)
    old_sox_score = sox_coincidence(sox_tree, all_olig2_arrays[0], old_shift)
    report = {
        "status": status,
        "old_shift": list(old_shift),
        "new_shift": list(chosen),
        "seeds": {
            name: {
                "shift": list(shift),
                "soma_fraction": by_shift[shift][1],
                "sox_score": by_shift[shift][2],
            }
            for name, shift in seeds.items()
        },
        "soma_fraction": full_fraction,
        "old_soma_fraction": old_full_count / len(olig2),
        "sampled_soma_fraction": selected[1],
        "sox_score": full_sox_score,
        "old_sox_score": old_sox_score,
        "sampled_sox_score": selected[2],
        "matched_olig2_soma_count": full_count,
        "matched_olig2_gfp_count": gfp_count,
        "matched_olig2_rfp_count": rfp_count,
        "matched_olig2_both_soma_count": gfp_count + rfp_count - full_count,
        "olig2_cells_scored": len(scored_olig2),
        "olig2_cells_in_window": len(olig2),
        "soma_score_span": soma_span,
        "sox_score_span": sox_span,
        "n_candidates": len(records),
        "reference_cells_in_window": {
            "GFP": len(gfp), "RFP": len(rfp), "Sox9": len(sox),
        },
    }
    return chosen, full_fraction, report


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    os.replace(part, path)


def write_tile(tile, raw_dir, original_align, output_align, tile_dir, settings,
               local_xy, local_z, max_scored_cells, overwrite):
    destinations = [
        output_align / f"{tile}_{channel}_result.csv" for channel in CHANNELS
    ] + [output_align / f"{tile}_offsets.json"]
    if any(path.exists() for path in destinations) and not overwrite:
        raise FileExistsError(f"Output for {tile} exists; use --overwrite")
    fixed_shifts, scores, old_shift = read_fixed_offsets(original_align, tile)
    cells, center = load_volumes(raw_dir, tile, settings)
    if all(cells[channel] for channel in CHANNELS):
        olig2_shift, olig2_score, report = choose_joint_shift(
            cells, center, fixed_shifts, old_shift, settings,
            local_xy, local_z, max_scored_cells)
    else:
        olig2_shift, olig2_score = old_shift, scores["Olig2"]
        report = {
            "status": "insufficient_cells_kept_original",
            "old_shift": list(old_shift),
            "new_shift": list(old_shift),
            "soma_fraction": olig2_score,
            "sox_score": 0.0,
            "reference_cells_in_window": {
                channel: len(cells[channel]) for channel in FIXED
            },
            "olig2_cells_in_window": len(cells["Olig2"]),
        }
    report["tile"] = tile
    report["z_center"] = center
    report["fixed_shifts"] = {k: list(v) for k, v in fixed_shifts.items()}
    shifts = {**fixed_shifts, "Olig2": olig2_shift}
    scores["Olig2"] = olig2_score

    output_align.mkdir(parents=True, exist_ok=True)
    # Keep all three visually verified channels byte-for-byte identical.
    for channel in FIXED:
        source = original_align / f"{tile}_{channel}_result.csv"
        if not source.is_file():
            raise FileNotFoundError(source)
        dest = output_align / source.name
        part = dest.with_name(dest.name + ".part")
        shutil.copyfile(source, part)
        os.replace(part, dest)

    slice_names = [
        path.stem for path in sorted(tile_dir.iterdir())
        if path.suffix.lower() in (".tif", ".tiff")
        and not path.name.startswith(".")
    ]
    if not slice_names:
        raise ValueError(f"No TIFF slices found in {tile_dir}")
    apply_shift_to_csv(
        str(raw_dir / f"{tile}_Olig2_result.csv"), *olig2_shift,
        str(output_align / f"{tile}_Olig2_result.csv"),
        slice_names=slice_names)

    write_json(output_align.parent / "diagnostics" / f"{tile}.json", report)
    # Offsets JSON is last, matching Stage 2.5's completion convention.
    save_tile_offsets(tile, shifts, scores, str(output_align))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-root", type=Path,
                        help="Default: sibling detection_results_olig2_three_refs")
    parser.add_argument("--vis-config", type=Path,
                        default=ROOT / "config" / "vis_config.json")
    parser.add_argument("--tiles", nargs="+",
                        help="Tile names; default: all tiles with original offsets")
    parser.add_argument("--local-xy", type=int, default=2)
    parser.add_argument("--local-z", type=int, default=1)
    parser.add_argument("--max-scored-cells", type=int, default=1500)
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace selected tiles in the separate output root")
    parser.add_argument("--resume", action="store_true",
                        help="Skip complete tiles and rebuild partial tiles")
    parser.add_argument("--workers", type=int, default=1,
                        help="Independent tile processes (default: 1)")
    args = parser.parse_args(argv)
    if args.local_xy < 0 or args.local_z < 0 or args.max_scored_cells < 1:
        parser.error("Search radii must be nonnegative and max-scored-cells >= 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.overwrite and args.resume:
        parser.error("Choose --overwrite or --resume, not both")

    results = args.results_dir.resolve()
    output = (args.output_root or
              results.with_name(results.name + "_olig2_three_refs")).resolve()
    original_align = results / "0_channel_alignment"
    raw_dir = results / "1_tile_2d_raw"
    output_align = output / "0_channel_alignment"
    if output in (results, original_align, raw_dir):
        parser.error("--output-root must be separate from original results")
    if not original_align.is_dir() or not raw_dir.is_dir():
        parser.error(f"Missing T70 input directories under {results}")

    settings_path = original_align / "_align_settings.json"
    if not settings_path.is_file():
        parser.error(f"Missing recorded alignment settings: {settings_path}")
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    if (settings.get("reference_channel") != "GFP"
            or settings.get("soma_ch_ids") != ["GFP", "RFP"]
            or settings.get("tf_ch_ids") != ["Sox9", "Olig2"]):
        parser.error("This experiment requires T70 channels in GFP frame")

    vis = load_config(args.vis_config)
    sample = vis.get("samples", {}).get("EGFR_t70")
    if sample is None:
        parser.error(f"{args.vis_config} lacks samples.EGFR_t70")
    gfp_dir = Path(sample["paths"]["gfp_dir"])
    if not gfp_dir.is_dir():
        parser.error(f"GFP image directory does not exist: {gfp_dir}")

    suffix = "_offsets.json"
    available = sorted(path.name[:-len(suffix)]
                       for path in original_align.glob(f"*{suffix}"))
    tiles = list(dict.fromkeys(args.tiles or available))
    if not tiles or any(tile not in available for tile in tiles):
        parser.error(f"Unknown tile(s): {sorted(set(tiles) - set(available))}")
    tile_set = set(tiles)
    tile_dirs = {}
    for root_dir, child_dirs, _ in os.walk(gfp_dir):
        if not child_dirs and Path(root_dir).name in tile_set:
            tile_dirs[Path(root_dir).name] = Path(root_dir)
    missing_images = sorted(tile_set - set(tile_dirs))
    if missing_images:
        parser.error(f"Missing GFP tile image directories: {missing_images[:3]}")
    if not (args.overwrite or args.resume):
        existing = [
            tile for tile in tiles
            if any((output_align / f"{tile}_{channel}_result.csv").exists()
                   for channel in CHANNELS)
            or (output_align / f"{tile}_offsets.json").exists()
        ]
        if existing:
            parser.error(f"Output already has {existing[:3]}; use --overwrite")

    output.mkdir(parents=True, exist_ok=True)
    vis["active_sample"] = "EGFR_t70"
    vis["samples"]["EGFR_t70"]["paths"]["pATHRESULT"] = str(output)
    vis["mode"] = "2d"
    vis["2d_source"] = "raw"
    vis["tile"] = tiles[0]
    write_json(output / "gui_vis_config.json", vis)
    for tile in tiles:
        vis["tile"] = tile
        write_json(output / "gui_tiles" / f"{tile}.json", vis)
    write_json(output / "experiment_settings.json", {
        "source_results": str(results),
        "base_settings": str(settings_path),
        "method": "fixed GFP/RFP/Sox9; Olig2 uses GFP+RFP soma and Sox9 nuclei",
        "local_xy": args.local_xy,
        "local_z": args.local_z,
        "max_scored_cells": args.max_scored_cells,
        "tiles": tiles,
        "workers": args.workers,
    })

    def complete(tile):
        return (output_align / f"{tile}_offsets.json").is_file() and all(
            (output_align / f"{tile}_{channel}_result.csv").is_file()
            for channel in CHANNELS) and (
            output / "diagnostics" / f"{tile}.json").is_file()
    todo = [tile for tile in tiles if not (args.resume and complete(tile))]
    print(f"Input: {results}\nOutput: {output}\nTiles: {len(tiles)}; "
          f"already complete: {len(tiles) - len(todo)}; to run: {len(todo)}",
          flush=True)
    def report_done(index, report):
        print(f"[{index}/{len(todo)}] {report['tile']}: Olig2 "
              f"{tuple(report['old_shift'])} -> {tuple(report['new_shift'])}; "
              f"soma={report['soma_fraction']:.4f}, "
              f"Sox9={report['sox_score']:.4f}; {report['status']}",
              flush=True)
    def job_args(tile):
        return (tile, raw_dir, original_align, output_align, tile_dirs[tile],
                settings, args.local_xy, args.local_z, args.max_scored_cells,
                args.overwrite or args.resume)
    if args.workers == 1:
        for index, tile in enumerate(todo, 1):
            report_done(index, write_tile(*job_args(tile)))
    elif todo:
        # Each worker writes a different tile. BLAS threads stay bounded;
        # individual cKDTree calls still use their own threads as needed.
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[variable] = "1"
        with ProcessPoolExecutor(max_workers=min(args.workers, len(todo))) as pool:
            futures = {pool.submit(write_tile, *job_args(tile)): tile
                       for tile in todo}
            for index, future in enumerate(as_completed(futures), 1):
                report_done(index, future.result())
    print("GUI: python src/utils/visualize.py --config "
          f"{output / 'gui_vis_config.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

