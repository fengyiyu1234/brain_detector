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
        frame_route = next((ch for ch in routing if ch["id"] == frame), None)
        if frame_route and frame_route.get("dir_key") and paths.get(frame_route["dir_key"]):
            candidates.append(os.path.join(
                paths[frame_route["dir_key"]], "xml_merging.xml"))
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
                      global_z_range: tuple[int, int], z_column: str = "z") -> pd.DataFrame:
    """Global CSV z is one-based; display_z is zero-based global.

    z_column picks the layer used for selection and display. Stage 3 summaries pass
    best_box_z: their x/y are the representative box of that layer, while z is the
    track's median layer. Files without the column fall back to z.
    """
    x0, y0, x1, y1 = xy_bounds
    z0, z1 = global_z_range

    def select(c):
        column = z_column if z_column in c.columns else "z"
        return ((c["x2"] > x0) & (c["x1"] < x1) &
                (c["y2"] > y0) & (c["y1"] < y1) &
                (c[column] >= z0 + 1) & (c[column] < z1 + 1))

    rows = _read_boxes(csv_path, select)
    if not rows.empty:
        column = z_column if z_column in rows.columns else "z"
        rows["display_z"] = rows[column].astype(int) - 1
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


def stage3_geometry(xml_path: str, tile_size: int):
    """{tile: ((row, col), abs_x, abs_y, z0)} as combine_predictions places tiles, or None.

    Stage 3 global = local + (abs_x, abs_y) and global z = local z - z0.
    """
    from src.utils.io import loadTeraxml
    try:
        dir_dict, _, _, _, z_start, disp = loadTeraxml(xml_path, tile_size)
    except (AttributeError, KeyError, OSError, SyntaxError, TypeError, ValueError):
        return None
    geometry = {}
    for name, rc in dir_dict.items():
        abs_x, abs_y, abs_z = (int(v) for v in disp[rc])
        tile = os.path.basename(str(name).replace("\\", "/"))
        geometry[tile] = (tuple(rc), abs_x, abs_y, int(z_start) - abs_z)
    return geometry


def neighbour_tiles(geometry, tile_name: str) -> list[str]:
    """The up-to-8 grid neighbours Stage 3 compares a tile with."""
    if not geometry or tile_name not in geometry:
        return []
    row, col = geometry[tile_name][0]
    return [name for name, (rc, *_) in geometry.items()
            if name != tile_name and max(abs(rc[0] - row), abs(rc[1] - col)) == 1]


class TileFrame:
    """Maps Stage 3 input detections onto one tile's own image (tile-local display).

    In a tile overlap Stage 3 keeps a cell's box in the frame of whichever tile was
    stitched first, so a saved global box can sit a seam offset away from this tile's
    image. A detection is placed from its own tile-local box instead:

      own   detected by this tile: its tile-local box and layer, unchanged.
      seam  detected by a neighbour: the neighbour's tile-local box, moved through the
            XML placement plus the measured seam offset (neighbour -> this tile), the
            same offset Stage 3 removes when matching duplicates.

    A source is (tile, x1, y1, x2, y2, z) from a Stage 3 input CSV, z one-based
    tile-local. geometry is stage3_geometry(); seam_offsets is one channel's table.
    """

    def __init__(self, tile_name, geometry=None, seam_offsets=None, seg_layers=50):
        self.tile_name = tile_name
        self.geometry = geometry or {}
        self.seam_offsets = seam_offsets
        self.seg_layers = int(seg_layers)
        self.grid = {rc: name for name, (rc, *_) in self.geometry.items()}
        self.target = self.geometry.get(tile_name)
        self._shifts = {}

    def _seam_shift(self, src, seg):
        key = (src, seg)
        if key not in self._shifts:
            value = None
            if self.seam_offsets:
                from src.core.stitcher import _seam_shift
                value = _seam_shift(self.seam_offsets, self.grid,
                                    self.geometry[src][0], self.target[0], seg)
            self._shifts[key] = (None if value is None else
                                 (float(value[0]), float(value[1]),
                                  int(round(float(value[2])))))
        return self._shifts[key]

    def place(self, source):
        """((x1, y1, x2, y2), display_z, frame) in this tile, or None if unplaceable."""
        if source is None:
            return None
        src, x1, y1, x2, y2, z_src = source
        if src == self.tile_name:
            return (x1, y1, x2, y2), int(z_src) - 1, "own"
        if self.target is None or src not in self.geometry:
            return None
        _, sx, sy, sz0 = self.geometry[src]
        _, tx, ty, tz0 = self.target
        z_global = int(z_src) - sz0
        offset = self._seam_shift(src, (z_global - 1) // self.seg_layers)
        if offset is not None:
            # Stage 3 picks the segment from this tile's layer; refine once.
            refined = self._seam_shift(src, (z_global + offset[2] - 1) // self.seg_layers)
            offset = refined or offset
        ox, oy, oz = offset or (0.0, 0.0, 0)
        dx, dy = sx - tx + ox, sy - ty + oy
        return (x1 + dx, y1 + dy, x2 + dx, y2 + dy), z_global + oz + tz0 - 1, "seam"

    def delta(self, source, global_cx, global_cy, global_z, tile_pos):
        """(dx, dy, dz, frame) from XML placement of a saved global box to this tile.

        global_cx/cy: centre of the detection's saved global box; global_z: its
        one-based global layer. Adding the delta to any box placed with
        global_to_local_rows moves it by the same amount as that detection.
        """
        placed = self.place(source)
        if placed is None:
            return None
        (x1, y1, x2, y2), display_z, frame = placed
        return ((x1 + x2) / 2 - (global_cx - tile_pos.x),
                (y1 + y2) / 2 - (global_cy - tile_pos.y),
                display_z - (int(global_z) - 1 + tile_pos.z), frame)


def _local_with_saved_box(rows, tile_pos):
    local = global_to_local_rows(rows, tile_pos)
    if not local.empty:
        for key in ("x1", "y1", "x2", "y2"):
            local[f"stage3_{key}"] = rows[key].to_numpy()
        local["frame"] = "xml"
        local["source_tile"] = ""
    return local


def _apply_columns(local, columns, display_z, frame, source_tile):
    for name, values in columns.items():
        local[name] = values
    local["display_z"] = display_z
    local["frame"] = frame
    local["source_tile"] = source_tile
    return local


def _editable_columns(local):
    columns = {name: local[name].to_numpy(dtype=float).copy()
               for name in ("x1", "y1", "x2", "y2")}
    return (columns, local["display_z"].to_numpy(dtype=int).copy(),
            local["frame"].to_numpy(dtype=object).copy(),
            local["source_tile"].to_numpy(dtype=object).copy())


def s3_rows_in_tile_frame(rows: pd.DataFrame, tile_name: str, tile_pos, sources: dict,
                          geometry=None, seam_offsets=None, seg_layers: int = 50) -> pd.DataFrame:
    """Place Stage 3 summary rows on one tile's own image (see TileFrame).

    Each row is its representative detection (best_detection_id), so it is redrawn
    exactly as that detection; rows whose source is not found keep the XML placement
    (frame "xml"). sources maps detection_id to a TileFrame source. The saved global
    box stays in stage3_x1..stage3_y2 for track lookup; source_tile names the tile
    that detected the representative box.
    """
    local = _local_with_saved_box(rows, tile_pos)
    if local.empty or "best_detection_id" not in local.columns:
        return local
    frame_map = TileFrame(tile_name, geometry, seam_offsets, seg_layers)
    columns, display_z, frame, source_tile = _editable_columns(local)
    for i, det in enumerate(local["best_detection_id"].astype(str)):
        source = sources.get(det)
        placed = frame_map.place(source)
        if placed is None:
            continue
        box, display_z[i], frame[i] = placed
        for name, value in zip(("x1", "y1", "x2", "y2"), box):
            columns[name][i] = value
        source_tile[i] = source[0]
    return _apply_columns(local, columns, display_z, frame, source_tile)


def s4_source_tracks(row) -> list[tuple[str, dict]]:
    """[(channel, track record)] of one coloc_result row, ordered by source_track."""
    raw = row.get("source_3d")
    if not isinstance(raw, str) or not raw:
        return []
    tracks = [(channel, record) for channel, records in json.loads(raw).items()
              for record in records]
    return sorted(tracks, key=lambda item: int(item[1].get("source_track", 0)))


def s4_rows_in_tile_frame(rows: pd.DataFrame, tile_pos, frames: dict, sources: dict,
                          soma_channels, center_sources: dict,
                          tracks_by_row=None) -> pd.DataFrame:
    """Place Stage 4 rows on one tile's own image (see TileFrame).

    frames / sources: per channel, a TileFrame and {detection_id: source}.
    center_sources: {coloc_id: source} of the primary track's member on the row's
    layer, which is exactly the box a rep-z row shows. tracks_by_row: the rows'
    s4_source_tracks, if already parsed.

    rep-z rows (one soma track) move with that member, or with the primary track's
    representative detection when the row's box is synthetic. Union rows rebuild the
    cross-channel 3D union from each soma track's 3D box, each moved with its own
    representative detection, drawn at the union's middle layer as Stage 4 does.
    A row that cannot be rebuilt keeps the XML placement (frame "xml").
    """
    local = _local_with_saved_box(rows, tile_pos)
    if local.empty or "source_3d" not in local.columns:
        return local
    soma_channels = set(soma_channels)

    def track_delta(channel, record):
        frame_map = frames.get(channel)
        source = sources.get(channel, {}).get(str(record.get("best_detection_id")))
        if frame_map is None or source is None:
            return None
        moved = frame_map.delta(source, float(record["cx"]), float(record["cy"]),
                                int(record["best_box_z"]), tile_pos)
        return None if moved is None else (*moved, source[0])

    columns, display_z, frame, source_tile = _editable_columns(local)
    records = local.to_dict("records")
    if tracks_by_row is None:
        tracks_by_row = [s4_source_tracks(row) for row in records]
    for i, (row, tracks) in enumerate(zip(records, tracks_by_row)):
        if not tracks:
            continue
        if row.get("bounds_method") == "cross_channel_bbox_union":
            somas = [(c, r) for c, r in tracks if c in soma_channels]
            if not somas:
                continue
            saved = [min(float(r["x1_3d"]) for _, r in somas),
                     min(float(r["y1_3d"]) for _, r in somas),
                     max(float(r["x2_3d"]) for _, r in somas),
                     max(float(r["y2_3d"]) for _, r in somas)]
            if not np.allclose(saved, [float(row[k]) for k in
                                       ("x1_3d", "y1_3d", "x2_3d", "y2_3d")]):
                continue  # Not a plain union of the saved soma tracks.
            moves = [track_delta(c, r) for c, r in somas]
            if any(m is None for m in moves):
                continue
            boxes = [(float(r["x1_3d"]) - tile_pos.x + m[0],
                      float(r["y1_3d"]) - tile_pos.y + m[1],
                      float(r["x2_3d"]) - tile_pos.x + m[0],
                      float(r["y2_3d"]) - tile_pos.y + m[1],
                      int(r["z_min"]) + m[2], int(r["z_max"]) + m[2])
                     for (_, r), m in zip(somas, moves)]
            columns["x1"][i] = min(b[0] for b in boxes)
            columns["y1"][i] = min(b[1] for b in boxes)
            columns["x2"][i] = max(b[2] for b in boxes)
            columns["y2"][i] = max(b[3] for b in boxes)
            # Stage 4 draws a union at round((z_min + z_max) / 2); z here is
            # global one-based plus this tile's per-track layer shift.
            z_min, z_max = min(b[4] for b in boxes), max(b[5] for b in boxes)
            display_z[i] = int(round((z_min + z_max) / 2)) - 1 + tile_pos.z
            frame[i] = "seam" if any(m[3] == "seam" for m in moves) else "own"
            source_tile[i] = "+".join(sorted({m[4] for m in moves}))
            continue
        channel, primary = tracks[0]
        center = center_sources.get(row.get("coloc_id"))
        moved = None
        if center is not None and channel in frames:
            moved = frames[channel].delta(
                center, (row["stage3_x1"] + row["stage3_x2"]) / 2,
                (row["stage3_y1"] + row["stage3_y2"]) / 2, int(row["z"]), tile_pos)
            moved = None if moved is None else (*moved, center[0])
        if moved is None:
            moved = track_delta(channel, primary)
        if moved is None:
            continue
        dx, dy, dz, frame[i], source_tile[i] = moved
        columns["x1"][i] += dx
        columns["x2"][i] += dx
        columns["y1"][i] += dy
        columns["y2"][i] += dy
        display_z[i] += dz
    return _apply_columns(local, columns, display_z, frame, source_tile)


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

