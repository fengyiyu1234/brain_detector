"""Read saved detections for GUI views without running inference."""

from __future__ import annotations

import json
import os
import posixpath

import numpy as np
import pandas as pd

from src.utils.coordinate_context import CoordinateContext, CoordinateContextError


BOX_COLUMNS = ("x1", "y1", "x2", "y2", "z", "class")


def saved_run_context(vis_config: dict) -> tuple[dict, CoordinateContext | None]:
    """Use runtime provenance and local visualization paths.

    Before tile-position solving completes, only tile-local 2D data can be shown.
    """
    vis_paths = vis_config.get("paths") or {}
    result_dir = vis_paths.get("pATHRESULT")
    if not result_dir:
        raise CoordinateContextError("Saved visualization needs paths.pATHRESULT")
    runtime_path = os.path.join(result_dir, "runtime_config.json")
    if not os.path.isfile(runtime_path):
        raise CoordinateContextError(f"Missing saved run metadata: {runtime_path}")
    with open(runtime_path, encoding="utf-8") as handle:
        runtime = json.load(handle)
    routing = [ch for ch in runtime.get("channels_routing", []) if ch.get("active", True)]
    if not routing:
        raise CoordinateContextError(f"No active channels in {runtime_path}")
    vis_routing = [ch["id"] for ch in vis_config.get("channels_routing", [])
                   if ch.get("active", True)]
    if vis_routing and set(vis_routing) != {ch["id"] for ch in routing}:
        raise CoordinateContextError(
            f"Visualization channels {vis_routing} differ from saved run channels "
            f"{[ch['id'] for ch in routing]}")
    paths = {**(runtime.get("paths") or {}), **vis_paths}
    paths["pATHRESULT"] = os.path.abspath(result_dir)
    frame = runtime.get("stitching_reference_channel") or (
        runtime.get("pre_align_params") or {}).get("reference_channel")
    runtime_xml = (runtime.get("paths") or {}).get("pATHXML")
    candidates = []
    runtime_xml_inside_result = False
    if runtime_xml:
        candidates.append(runtime_xml)
        old_root = (runtime.get("paths") or {}).get("pATHRESULT")
        if old_root:
            relative = posixpath.relpath(
                runtime_xml.replace("\\", "/"), old_root.replace("\\", "/"))
            if relative != ".." and not relative.startswith("../"):
                runtime_xml_inside_result = True
                candidates.append(os.path.join(result_dir, *relative.split("/")))
    if frame:
        candidates.append(os.path.join(
            result_dir, "5_2d_global", "tile_positions",
            f"xml_merging_{frame}.xml"))
    if vis_paths.get("pATHXML") and (
            not runtime_xml or
            (not runtime_xml_inside_result and
             os.path.basename(vis_paths["pATHXML"]) == os.path.basename(runtime_xml))):
        candidates.append(vis_paths["pATHXML"])
    frame_xml = next((p for p in candidates if p and os.path.isfile(p)), None)
    merged = {**runtime, "paths": paths, "channels_routing": routing}
    if frame_xml is None:
        return merged, None
    paths["pATHXML"] = frame_xml
    if frame:
        merged["frame_channel"] = frame
    return merged, CoordinateContext.from_vis_config(merged, runtime_path)


def _read_boxes(csv_path: str, select, chunksize: int = 100_000) -> pd.DataFrame:
    """Read intersecting rows only; global CSVs may have millions of detections."""
    if not os.path.isfile(csv_path):
        return pd.DataFrame(columns=BOX_COLUMNS)
    parts = []
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        missing = set(BOX_COLUMNS) - set(chunk.columns)
        if missing:
            raise ValueError(f"Saved result {csv_path} lacks columns {sorted(missing)}")
        subset = chunk.loc[select(chunk)]
        if not subset.empty:
            parts.append(subset.copy())
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=BOX_COLUMNS)


def read_tile_boxes(csv_path: str, z_range: tuple[int, int]) -> pd.DataFrame:
    """Tile CSV z is one-based; display_z is zero-based tile local."""
    z0, z1 = z_range
    rows = _read_boxes(csv_path, lambda c: (c["z"] >= z0 + 1) & (c["z"] < z1 + 1))
    if not rows.empty:
        rows["display_z"] = rows["z"].astype(int) - 1
    return rows


def read_global_boxes(csv_path: str, xy_bounds: tuple[float, float, float, float],
                      global_z_range: tuple[int, int]) -> pd.DataFrame:
    """Global CSV z is one-based; display_z is zero-based global."""
    x0, y0, x1, y1 = xy_bounds
    z0, z1 = global_z_range
    rows = _read_boxes(csv_path, lambda c: (
        (c["x2"] > x0) & (c["x1"] < x1) &
        (c["y2"] > y0) & (c["y1"] < y1) &
        (c["z"] >= z0 + 1) & (c["z"] < z1 + 1)))
    if not rows.empty:
        rows["display_z"] = rows["z"].astype(int) - 1
    return rows


def global_to_local_rows(rows: pd.DataFrame, tile_pos) -> pd.DataFrame:
    """Convert final-frame global rows to one tile's final-frame local space."""
    local = rows.copy()
    if local.empty:
        return local
    local[["x1", "x2"]] = local[["x1", "x2"]] - tile_pos.x
    local[["y1", "y2"]] = local[["y1", "y2"]] - tile_pos.y
    local["display_z"] = local["display_z"] + tile_pos.z
    return local


def records_and_shapes(rows: pd.DataFrame, shift=(0, 0, 0)):
    """Return napari rectangles and exact row metadata in the same order."""
    dx, dy, dz = shift
    shapes, records = [], []
    for row in rows.to_dict("records"):
        x1, y1, x2, y2 = (float(row[k]) for k in ("x1", "y1", "x2", "y2"))
        z = int(row["display_z"])
        shapes.append(np.array([
            [z, y1, x1], [z, y1, x2], [z, y2, x2], [z, y2, x1],
        ], dtype=float))
        records.append({**row, "world_z": z + dz, "world_x1": x1 + dx,
                        "world_x2": x2 + dx, "world_y1": y1 + dy,
                        "world_y2": y2 + dy})
    return shapes, records


def track_for_summary(summary_row: dict, tracks: list[dict]):
    """Find a unique saved PKL track for a Stage 3 summary row, or None.

    The pipeline does not save track IDs. Match fields written to both outputs;
    ambiguous records must never be assigned to a nearby cell.
    """
    cx = (float(summary_row["x1"]) + float(summary_row["x2"])) / 2
    cy = (float(summary_row["y1"]) + float(summary_row["y2"])) / 2
    matches = [c for c in tracks if
               c.get("class") == summary_row["class"] and
               int(round(c["cz"])) == int(summary_row["z"]) and
               np.isclose(c["cx"], cx, atol=1e-5) and
               np.isclose(c["cy"], cy, atol=1e-5) and
               np.isclose(c["score"], float(summary_row["score"]), atol=1e-5) and
               np.isclose(c["mean"], float(summary_row["mean"]), atol=1e-5)]
    return matches[0] if len(matches) == 1 else None

