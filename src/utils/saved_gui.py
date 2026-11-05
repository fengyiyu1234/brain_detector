"""Napari views of saved pre-align pipeline outputs.

All cell layers come from CSV/PKL checkpoints. Image shifts and coordinate
translations are the only operations performed by this module.
"""

from __future__ import annotations

import json
import os
import pickle

import numpy as np
import napari
import pandas as pd

from src.utils.video_export import add_video_export_panel
from src.utils.markers import class_markers
from src.utils.io import resolve_tile_dir
from src.utils.saved_view import (
    TileFrame, global_to_local_rows, neighbour_tiles, read_global_boxes,
    read_tile_boxes, records_and_shapes, s3_rows_in_tile_frame, s4_rows_in_tile_frame,
    s4_source_tracks, stage3_geometry, track_for_summary,
)
from src.utils.visualize import (
    _canvas_shape_from_tile, _ch_vis, _list_tiffs, _make_fn_recorder,
    _resolve_z_range, _show_only_images_initially, load_frame_volume,
)


def _image_tile_dir(paths, ch, anchor_dir, tile_path):
    return resolve_tile_dir(paths[ch["dir_key"]], os.path.relpath(tile_path, anchor_dir))


def _shift_rows(rows, dx=0, dy=0, dz=0):
    shifted = rows.copy()
    if not shifted.empty:
        shifted[["x1", "x2"]] = shifted[["x1", "x2"]] + dx
        shifted[["y1", "y2"]] = shifted[["y1", "y2"]] + dy
        shifted["display_z"] = shifted["display_z"] + dz
    return shifted


def _to_global_rows(rows, position):
    return _shift_rows(rows, position.x, position.y, -position.z)


def _tile_bounds(tile_path, z_range, position):
    _, h, w = _canvas_shape_from_tile(tile_path, z_range)
    return (position.x, position.y, position.x + w, position.y + h)


def _intersects_any(rows, bounds):
    if rows.empty:
        return rows
    mask = np.zeros(len(rows), dtype=bool)
    for x0, y0, x1, y1 in bounds:
        mask |= ((rows["x2"] > x0) & (rows["x1"] < x1) &
                 (rows["y2"] > y0) & (rows["y1"] < y1)).to_numpy()
    return rows.loc[mask].reset_index(drop=True)


def _add_box_layer(viewer, rows, name, *, visible, color=None,
                   width=3, registry=None, channel=None, source=None):
    if rows.empty:
        return
    shapes, records = records_and_shapes(rows)
    if color is None:
        colors = [([1.0, 0.55, 0.0, 1.0] if str(r["class"]).startswith("glia")
                   else [0.0, 0.55, 1.0, 1.0]) for r in records]
    elif callable(color):
        colors = [color(record) for record in records]
    else:
        colors = [color] * len(shapes)
    viewer.add_shapes(
        shapes, shape_type="rectangle", name=name, visible=visible,
        edge_color=colors, face_color=[0, 0, 0, 0], edge_width=width)
    if registry is not None:
        registry.extend({**r, "layer_name": name, "channel": channel,
                         "source": source} for r in records)
    print(f"[saved] {name}: {len(records)} records ({'visible' if visible else 'hidden'})")


def _add_s4_layers(viewer, rows, registry, width=4):
    if rows.empty:
        return
    groups = {}
    for cls in rows["class"].astype(str).unique():
        markers = tuple(sorted(class_markers(cls)))
        groups.setdefault(markers, []).append(cls)
    for markers, classes in sorted(groups.items(), key=lambda item: (len(item[0]), item[0])):
        selected = rows.loc[rows["class"].isin(classes)]
        label = "+".join(markers) if markers else "unmarked"
        base_colors = {
            "GFP": (0.2, 1.0, 0.2), "RFP": (1.0, 0.2, 0.2),
            "Sox9": (0.0, 0.8, 1.0), "Olig2": (1.0, 0.2, 1.0),
        }
        components = [base_colors.get(marker, (0.7, 0.7, 0.7))
                      for marker in markers]
        rgb = np.mean(components, axis=0) if components else np.array([.7, .7, .7])
        rgb = rgb / max(float(np.max(rgb)), 1e-6)
        def row_color(record):
            scale = .65 if str(record["class"]).startswith("glia") else 1.0
            return [*(rgb * scale), 1.0]
        box_label = ("[s4 saved 3d union]" if
                     "bounds_method" in selected and
                     selected["bounds_method"].eq("cross_channel_bbox_union").all()
                     else "[s4 saved rep-z exact]")
        _add_box_layer(
            viewer, selected, f"{box_label} {label}",
            visible=len(markers) >= 2, color=row_color,
            width=width, registry=registry, source="s4")


def _add_images(viewer, vis_cfg, paths, routing, anchor_dir, tile_path,
                tile_name, z_range, offsets, origin, global_view):
    if vis_cfg.get("no_images", False):
        return
    pct = tuple(vis_cfg.get("contrast_pct", [0.1, 99.9]))
    for ch in routing:
        cid = ch["id"]
        shift = offsets.get(cid, {"dx": 0, "dy": 0, "dz": 0})
        image_dir = _image_tile_dir(paths, ch, anchor_dir, tile_path)
        vol, limits = load_frame_volume(
            image_dir, z_range,
            (shift["dx"], shift["dy"], shift["dz"]), pct)
        if vol is None:
            print(f"[saved] image unavailable: {image_dir}")
            continue
        if limits is None or not np.isfinite(limits).all() or limits[1] <= limits[0]:
            limits = (0, 65535)
        label = (f"[img] {tile_name} {cid}" if global_view
                 else f"[img] {cid}")
        layer = viewer.add_image(
            vol, name=label, translate=origin,
            contrast_limits=limits, **_ch_vis(cid))
        layer.contrast_limits_range = (0, 65535)
        if ch.get("double_exposure"):
            second = ch["second_intensity_id"]
            second_dir = resolve_tile_dir(
                paths[ch["second_intensity_dir_key"]],
                os.path.relpath(tile_path, anchor_dir))
            second_shift = offsets.get(second, shift)
            second_vol, second_limits = load_frame_volume(
                second_dir, z_range,
                (second_shift["dx"], second_shift["dy"], second_shift["dz"]), pct)
            if second_vol is not None:
                if (second_limits is None or
                        not np.isfinite(second_limits).all() or
                        second_limits[1] <= second_limits[0]):
                    second_limits = (0, 65535)
                viewer.add_image(
                    second_vol,
                    name=(f"[img] {tile_name} {second}" if global_view
                          else f"[img] {second}"),
                    translate=origin, contrast_limits=second_limits,
                    visible=False, **_ch_vis(cid))


def _add_tile_2d(viewer, result_dir, tile_name, routing, offsets,
                 z_range, position, global_view, registry, width, vis_cfg):
    stages = (
        ("1_2d_raw", "raw", bool(vis_cfg.get("show_before", False)), True),
        ("2_2d_filtered", "prefiltered", False, True),
        ("4_2d_filtered", "filtered",
         bool(vis_cfg.get("show_filtered_2d", False)), False),
    )
    for ch in routing:
        primary = ch["id"]
        channel_ids = [(primary, True)]
        if ch.get("double_exposure"):
            channel_ids.append((ch["second_intensity_id"], False))
        for cid, is_primary in channel_ids:
            o = offsets.get(cid, offsets.get(
                primary, {"dx": 0, "dy": 0, "dz": 0}))
            for directory, label, visible, needs_shift in stages:
                if not offsets and directory != "1_2d_raw":
                    continue  # No saved transform for aligned stages yet.
                from src.core.resume import saved_channel_directory
                stage_dir = (saved_channel_directory(result_dir, ch) if directory == '4_2d_filtered'
                             else os.path.join(result_dir, directory))
                path = os.path.join(stage_dir, f"{tile_name}_{cid}_result.csv")
                if not os.path.isfile(path):
                    continue
                source_range = ((z_range[0] - o["dz"], z_range[1] - o["dz"])
                                if needs_shift else z_range)
                rows = read_tile_boxes(path, source_range)
                if needs_shift:
                    rows = _shift_rows(rows, o["dx"], o["dy"], o["dz"])
                if global_view:
                    rows = _to_global_rows(rows, position)
                _add_box_layer(
                    viewer, rows, f"[2d {label}] {tile_name} {cid}",
                    visible=visible and is_primary, width=width, registry=registry,
                    channel=cid, source=directory)


def _stage3_seam_inputs(result_dir, cid, context):
    """(geometry, seam_offsets, seg_layers) as Stage 3 used them for one channel.

    geometry is None when the frame XML no longer matches the Stage 3 manifest, or
    when Stage 3 used a seam table that is missing or changed; neighbour rows then
    keep their global (XML) placement.
    """
    from src.core.provenance import file_sha256
    from src.core.seam_offsets import load_offsets
    folder = os.path.join(result_dir, "5_2d_global")
    try:
        with open(os.path.join(folder, f"{cid}_global_manifest.json"),
                  encoding="utf-8") as handle:
            signature = json.load(handle)["input_signature"]
    except (OSError, ValueError, KeyError):
        signature = {}
    geometry = stage3_geometry(context.frame_xml, context.tile_size)
    saved = signature.get("geometry") or {}
    if geometry and saved.get("positions") is not None:
        z_start = int(saved["z_start"])
        expected = {key: [int(v) for v in value] for key, value in saved["positions"]}
        actual = {str(rc): [ax, ay, z_start - z0]
                  for rc, ax, ay, z0 in geometry.values()}
        if actual != expected:
            print(f"[saved s3] {cid}: frame XML differs from the Stage 3 manifest; "
                  "neighbour rows keep their XML placement")
            geometry = None
    seam = signature.get("seam_offsets")
    if not seam:
        return geometry, None, 50
    seg_layers = int(seam.get("seg_layers", 50))
    path = os.path.join(folder, "seam_offsets", "seam_offsets.csv")
    if (not os.path.isfile(path) or
            file_sha256(path) != (seam.get("table") or {}).get("sha256")):
        print(f"[saved s3] {cid}: seam table used by Stage 3 is missing or changed "
              f"({path}); neighbour rows keep their XML placement")
        return None, None, seg_layers
    return geometry, load_offsets(path, cid), seg_layers


def _tile_table(result_dir, ch, tile, cache):
    """One Stage 3 input CSV (detection_id, slice_name, box, z), read once per view."""
    key = (ch["id"], tile)
    if key not in cache:
        from src.core.resume import saved_channel_directory
        folder = cache.setdefault(("folder", ch["id"]),
                                  saved_channel_directory(result_dir, ch))
        path = os.path.join(folder, f"{tile}_{ch['id']}_result.csv")
        cache[key] = (pd.read_csv(path, usecols=lambda c: c in (
            "detection_id", "slice_name", "x1", "y1", "x2", "y2", "z"))
            if os.path.isfile(path) else None)
    return cache[key]


def _stage3_sources(result_dir, ch, tile_names, wanted, cache=None):
    """{detection_id: (tile, x1, y1, x2, y2, z)} from the CSVs Stage 3 stitched."""
    cache = {} if cache is None else cache
    wanted = set(wanted)
    sources = {}
    for name in tile_names:
        if not wanted:
            break
        table = _tile_table(result_dir, ch, name, cache)
        if table is None or "detection_id" not in table.columns:
            continue
        id_series = table["detection_id"].astype(str)
        ids = id_series.to_numpy()
        hit = np.flatnonzero(id_series.isin(wanted).to_numpy())  # hash lookup
        x1, y1, x2, y2, z = (table[c].to_numpy() for c in ("x1", "y1", "x2", "y2", "z"))
        for k in hit:
            sources[ids[k]] = (name, float(x1[k]), float(y1[k]), float(x2[k]),
                               float(y2[k]), int(z[k]))
        wanted -= set(ids[hit])
    return sources


def _s3_local_rows(rows, result_dir, ch, tile_name, position, context, cache=None):
    """Stage 3 summary rows drawn on this tile's own image (see s3_rows_in_tile_frame)."""
    if rows.empty or "best_detection_id" not in rows.columns or context is None:
        return s3_rows_in_tile_frame(rows, tile_name, position, {})
    cache = {} if cache is None else cache
    geometry, seam_offsets, seg_layers = _seam_inputs_cached(
        result_dir, ch["id"], context, cache)
    tiles = [tile_name] + neighbour_tiles(geometry, tile_name)
    sources = _stage3_sources(
        result_dir, ch, tiles, rows["best_detection_id"].astype(str), cache)
    local = s3_rows_in_tile_frame(rows, tile_name, position, sources,
                                  geometry, seam_offsets, seg_layers)
    counts = local["frame"].value_counts().to_dict()
    print(f"[saved s3] {ch['id']} tile frame: own={counts.get('own', 0)}, "
          f"neighbour+seam={counts.get('seam', 0)}, xml-only={counts.get('xml', 0)}")
    return local


def _seam_inputs_cached(result_dir, cid, context, cache):
    key = ("seam", cid)
    if key not in cache:
        cache[key] = _stage3_seam_inputs(result_dir, cid, context)
    return cache[key]


def _s4_center_sources(result_dir, routing, rows, cache=None, tracks_by_row=None):
    """{coloc_id: source} of the primary member each rep-z row shows exactly.

    coloc_result names that member's tile and slice; a pure translation keeps the box
    width and height, so the member is the unique same-size box on that slice.
    tracks_by_row: s4_source_tracks of each row, if already parsed.
    """
    cache = {} if cache is None else cache
    by_channel = {ch["id"]: ch for ch in routing}
    records = rows.to_dict("records")
    if tracks_by_row is None:
        tracks_by_row = [s4_source_tracks(row) for row in records]
    wanted = {}
    for row, tracks in zip(records, tracks_by_row):
        if (row.get("bounds_method") == "cross_channel_bbox_union" or
                not str(row.get("source_trace_status", "")).startswith(
                    "exact_primary_center_member")):
            continue
        if not tracks or tracks[0][0] not in by_channel:
            continue
        wanted.setdefault((tracks[0][0], str(row["tile_name"])), []).append(row)

    def size_key(slice_name, w, h):
        return (str(slice_name), round(float(w), 3), round(float(h), 3))

    found = {}
    for (channel, tile), members in wanted.items():
        table = _tile_table(result_dir, by_channel[channel], tile, cache)
        if table is None or "slice_name" not in table.columns:
            continue
        names = table["slice_name"].astype(str)
        table = table.loc[names.isin({str(row["slice_name"]) for row in members})]
        x1, y1, x2, y2, z = (table[c].to_numpy() for c in ("x1", "y1", "x2", "y2", "z"))
        index = {}
        for k, name in enumerate(table["slice_name"].astype(str).to_numpy()):
            index.setdefault(size_key(name, x2[k] - x1[k], y2[k] - y1[k]), []).append(k)
        for row in members:
            hits = index.get(size_key(row["slice_name"], row["x2"] - row["x1"],
                                      row["y2"] - row["y1"]), [])
            if len(hits) == 1:
                k = hits[0]
                found[row["coloc_id"]] = (tile, float(x1[k]), float(y1[k]),
                                          float(x2[k]), float(y2[k]), int(z[k]))
    return found


def _s4_local_rows(rows, result_dir, routing, tile_name, position, context, cache=None):
    """Stage 4 rows drawn on this tile's own image (see s4_rows_in_tile_frame)."""
    if rows.empty or "source_3d" not in rows.columns or context is None:
        return s4_rows_in_tile_frame(rows, position, {}, {}, (), {})
    cache = {} if cache is None else cache
    soma_channels = [ch["id"] for ch in routing if ch.get("type", "soma") == "soma"]
    records = rows.to_dict("records")
    tracks_by_row = [s4_source_tracks(row) for row in records]
    wanted = {}
    for row, tracks in zip(records, tracks_by_row):
        union = row.get("bounds_method") == "cross_channel_bbox_union"
        for index, (channel, record) in enumerate(tracks):
            if index == 0 or (union and channel in soma_channels):
                wanted.setdefault(channel, set()).add(str(record.get("best_detection_id")))
    frames, sources = {}, {}
    for ch in routing:
        if ch["id"] not in wanted:
            continue
        geometry, seam_offsets, seg_layers = _seam_inputs_cached(
            result_dir, ch["id"], context, cache)
        frames[ch["id"]] = TileFrame(tile_name, geometry, seam_offsets, seg_layers)
        tiles = [tile_name] + neighbour_tiles(geometry, tile_name)
        sources[ch["id"]] = _stage3_sources(
            result_dir, ch, tiles, wanted[ch["id"]], cache)
    local = s4_rows_in_tile_frame(
        rows, position, frames, sources, soma_channels,
        _s4_center_sources(result_dir, routing, rows, cache, tracks_by_row),
        tracks_by_row)
    counts = local["frame"].value_counts().to_dict()
    print(f"[saved s4] tile frame: own={counts.get('own', 0)}, "
          f"neighbour+seam={counts.get('seam', 0)}, xml-only={counts.get('xml', 0)}")
    return local


def _add_global_results(viewer, result_dir, routing, xy_bounds,
                        global_z_range, tile_bounds, position, global_view,
                        registry, width, show_coloc, show_zlinked, show_spheres,
                        tile_name=None, context=None):
    cache = {}  # Stage 3 input CSVs and seam tables shared by the s3 and s4 layers.
    for ch in routing:
        cid = ch["id"]
        for directory, filename, label in (
            ("5_2d_global", f"{cid}_2d_global.csv", "global"),
            ("6_3d_global", f"{cid}_3d_tracked.csv", "s3"),
        ):
            saved_path = os.path.join(result_dir, directory, filename)
            if not os.path.isfile(saved_path):
                print(f"[saved] Stage unavailable: {saved_path}")
            # A Stage 3 summary's x/y belong to its representative box's layer.
            rows = read_global_boxes(saved_path, xy_bounds, global_z_range,
                                     z_column="best_box_z" if label == "s3" else "z")
            rows = _intersects_any(rows, tile_bounds)
            if not global_view and label == "s3":
                rows = _s3_local_rows(rows, result_dir, ch, tile_name, position, context,
                                      cache)
            elif not global_view:
                rows = global_to_local_rows(rows, position)
            layer_label = ("[s3 saved summary]" if label == "s3"
                           else "[global saved]")
            _add_box_layer(
                viewer, rows, f"{layer_label} {cid}",
                visible=(label == "s3" and show_zlinked),
                width=width, registry=registry, channel=cid, source=directory)
            if label == "s3" and show_spheres and not rows.empty:
                centers = np.column_stack((
                    rows["display_z"].to_numpy(float),
                    ((rows["y1"] + rows["y2"]) / 2).to_numpy(float),
                    ((rows["x1"] + rows["x2"]) / 2).to_numpy(float)))
                sizes = np.maximum(
                    (rows["x2"] - rows["x1"]).to_numpy(float),
                    (rows["y2"] - rows["y1"]).to_numpy(float))
                viewer.add_points(
                    centers, size=sizes, name=f"[s3 saved spheres] {cid}",
                    face_color=_ch_vis(cid).get("colormap", "white"),
                    n_dimensional=True)
    if show_coloc:
        path = os.path.join(result_dir, "7_colocalization", "coloc_result.csv")
        rows = read_global_boxes(path, xy_bounds, global_z_range)
        rows = _intersects_any(rows, tile_bounds)
        if not global_view:
            rows = _s4_local_rows(rows, result_dir, routing, tile_name, position, context,
                                  cache)
        _add_s4_layers(viewer, rows, registry, width)
        if not os.path.isfile(path):
            print(f"[saved] Stage 4 unavailable: {path}")


def _attach_clicks(viewer, registry, result_dir, context, tile_contexts,
                   global_view, vis_cfg, paths, routing, anchor_dir):
    """Inspect saved rows and print complete Stage 3 coordinates on click."""
    track_cache = {}
    fn_recorders = {}
    if not global_view:
        for tile_name, (tile_path, z_range, offsets, _) in tile_contexts.items():
            fn_recorders[tile_name] = _make_fn_recorder(
                vis_cfg, paths, routing, anchor_dir, tile_path,
                tile_name, z_range, offsets=offsets)

    def on_click(v, event):
        if event.type != "mouse_press":
            return
        mods = [str(m).lower() for m in getattr(event, "modifiers", [])]
        is_shift = any("shift" in m for m in mods)
        is_ctrl = any("ctrl" in m or "control" in m for m in mods)
        z, y, x = v.cursor.position[:3]
        z = int(round(z))
        if is_ctrl:
            if global_view:
                v.status = "Use the local tile view to record false negatives"
            else:
                tile_name = next(iter(tile_contexts))
                recorder = fn_recorders[tile_name]
                if recorder is not None:
                    z0 = tile_contexts[tile_name][1][0]
                    recorder(v, z - z0, x, y, [])
                else:
                    v.status = "False-negative recording is disabled"
            return
        visible = {layer.name for layer in v.layers if layer.visible}
        hits = [r for r in registry if r["layer_name"] in visible
                and r["world_z"] == z
                and r["world_x1"] <= x <= r["world_x2"]
                and r["world_y1"] <= y <= r["world_y2"]]
        if not hits:
            v.status = f"No saved box at z={z}, x={x:.0f}, y={y:.0f}"
            return
        active = v.layers.selection.active
        if active is not None:
            preferred = [r for r in hits if r["layer_name"] == active.name]
            if preferred:
                hits = preferred
        row = min(hits, key=lambda r: (
            (r["world_x2"] - r["world_x1"]) *
            (r["world_y2"] - r["world_y1"])))
        label = row["layer_name"]
        v.status = (f"{label} {row['class']} z={row['z']} "
                    f"score={row.get('score', float('nan')):.3f} "
                    f"mean={row.get('mean', float('nan')):.1f}")
        print(f"[saved row] {label}: {row}")
        if row["source"] != "6_3d_global":
            if is_shift and row["source"] == "s4":
                v.status += " | Stage 4 stores representative z only"
            return
        cid = row["channel"]
        pkl_path = os.path.join(result_dir, "6_3d_global", f"{cid}_3d_tracked.pkl")
        if not os.path.isfile(pkl_path):
            v.status = f"Saved track unavailable: {pkl_path}"
            print(f"[saved 3d] {v.status}")
            return
        if cid not in track_cache:
            with open(pkl_path, "rb") as handle:
                track_cache[cid] = pickle.load(handle)
        summary = dict(row)
        if not global_view:
            tile_name = next(iter(tile_contexts))
            position = tile_contexts[tile_name][3]
            # Local s3 rows may be redrawn in the tile's own frame; match the saved row.
            for key, shift in (("x1", position.x), ("x2", position.x),
                               ("y1", position.y), ("y2", position.y)):
                summary[key] = (summary[f"stage3_{key}"] if f"stage3_{key}" in summary
                                else summary[key] + shift)
        track = track_for_summary(summary, track_cache[cid])
        if track is None:
            v.status = "No unique saved PKL track matches this Stage 3 row"
            print(f"[saved 3d] {v.status}")
            return
        per_z_boxes = track["per_z_boxes"]
        z_values = sorted(int(tz) for tz in per_z_boxes)
        x1 = min(float(box[0]) for box in per_z_boxes.values())
        y1 = min(float(box[1]) for box in per_z_boxes.values())
        x2 = max(float(box[2]) for box in per_z_boxes.values())
        y2 = max(float(box[3]) for box in per_z_boxes.values())
        print(f"[saved 3d] {cid} {track['class']}: global XYZ "
              f"x=[{x1:g}, {x2:g}], y=[{y1:g}, {y2:g}], "
              f"z=[{z_values[0]}, {z_values[-1]}] (1-based), "
              f"observed_layers={len(z_values)}", flush=True)
        for tz in z_values:
            gx1, gy1, gx2, gy2 = map(float, per_z_boxes[tz])
            if global_view:
                dz, dx, dy = tz - 1, gx1, gy1
                dx2, dy2 = gx2, gy2
            else:
                dz = tz - 1 + position.z
                dx, dy = gx1 - position.x, gy1 - position.y
                dx2, dy2 = gx2 - position.x, gy2 - position.y
            print(f"[saved 3d]   global z={tz}: "
                  f"x=[{gx1:g}, {gx2:g}], y=[{gy1:g}, {gy2:g}] | "
                  f"display z={dz}: x=[{dx:g}, {dx2:g}], "
                  f"y=[{dy:g}, {dy2:g}]", flush=True)
        v.status = (f"Saved {cid} track spans global z={z_values[0]}-{z_values[-1]} "
                    f"({len(z_values)} observed layers)")
        if not is_shift:
            return
        for layer in list(v.layers):
            if layer.name == "[saved track span]":
                v.layers.remove(layer)
        span = []
        for tz, box in track["per_z_boxes"].items():
            gx1, gy1, gx2, gy2 = map(float, box)
            gz = int(tz) - 1
            if global_view:
                tz_display = gz
                tx1, ty1, tx2, ty2 = gx1, gy1, gx2, gy2
            else:
                tile_name = next(iter(tile_contexts))
                position = tile_contexts[tile_name][3]
                tz_display = gz + position.z
                tx1, ty1 = gx1 - position.x, gy1 - position.y
                tx2, ty2 = gx2 - position.x, gy2 - position.y
            span.append(np.array([
                [tz_display, ty1, tx1], [tz_display, ty1, tx2],
                [tz_display, ty2, tx2], [tz_display, ty2, tx1],
            ], dtype=float))
        if span:
            v.add_shapes(
                span, shape_type="rectangle", name="[saved track span]",
                edge_color="yellow", face_color=[0, 0, 0, 0],
                edge_width=2)

    viewer.mouse_drag_callbacks.append(on_click)


def run_saved_prealign(vis_cfg, paths, routing, context, selected_tiles, progress=None):
    """Render one tile in local coordinates, or selected tiles in one global view."""
    view_space = vis_cfg.get("view_space", "local")
    if view_space not in ("local", "global"):
        raise ValueError("view_space must be 'local' or 'global'")
    if view_space == "global" and context is None:
        raise ValueError("Global view needs the saved final-frame XML")
    result_dir = paths["pATHRESULT"]
    anchor_dir = paths[routing[0]["dir_key"]]
    width = int(vis_cfg.get("outline_width", 3))
    show_coloc = bool(vis_cfg.get("show_coloc", True))
    if view_space == "local":
        for tile_index, (tile_path, tile_name) in enumerate(selected_tiles, 1):
            z_range = _resolve_z_range(vis_cfg, tile_path)
            offsets = (context.offsets_for_tile(tile_name) if context else
                       _load_offsets_if_present(result_dir, tile_name))
            pos = context.position(tile_name) if context else None
            viewer = napari.Viewer(title=f"Saved pre-align local — {tile_name}")
            registry = []
            _add_images(viewer, vis_cfg, paths, routing, anchor_dir, tile_path,
                        tile_name, z_range, offsets, (z_range[0], 0, 0), False)
            _add_tile_2d(viewer, result_dir, tile_name, routing, offsets,
                         z_range, pos, False, registry, width, vis_cfg)
            if pos is not None:
                bounds = _tile_bounds(tile_path, z_range, pos)
                global_range = (z_range[0] - pos.z, z_range[1] - pos.z)
                _add_global_results(
                    viewer, result_dir, routing, bounds, global_range,
                    [bounds], pos, False, registry, width, show_coloc,
                    bool(vis_cfg.get("show_zlinked", False)),
                    bool(vis_cfg.get("spheres", False)),
                    tile_name=tile_name, context=context)
            else:
                print("[saved] Final-frame XML not available; showing tile 2D only")
            _attach_clicks(
                viewer, registry, result_dir, context,
                {tile_name: (tile_path, z_range, offsets, pos)},
                False, vis_cfg, paths, routing, anchor_dir)
            if vis_cfg.get("spheres", False):
                viewer.dims.ndisplay = 3
            _show_only_images_initially(viewer, vis_cfg)
            viewer.reset_view()
            add_video_export_panel(viewer, vis_cfg)
            if progress:
                progress(tile_index, len(selected_tiles), tile_name)
        return

    first_path, first_name = selected_tiles[0]
    first_pos = context.position(first_name)
    first_range = _resolve_z_range(vis_cfg, first_path)
    global_z0 = vis_cfg.get("global_z_start")
    global_z0 = int(global_z0) if global_z0 is not None else first_range[0] - first_pos.z
    count = int(vis_cfg.get("z_count") or (first_range[1] - first_range[0]))
    global_range = (global_z0, global_z0 + count)
    viewer = napari.Viewer(title=f"Saved pre-align global — {context.frame_channel}")
    registry, tile_contexts, bounds_list = [], {}, []
    for tile_index, (tile_path, tile_name) in enumerate(selected_tiles, 1):
        pos = context.position(tile_name)
        offsets = context.offsets_for_tile(tile_name)
        local_range = (global_range[0] + pos.z, global_range[1] + pos.z)
        bounds = _tile_bounds(tile_path, local_range, pos)
        tile_contexts[tile_name] = (tile_path, local_range, offsets, pos)
        bounds_list.append(bounds)
        _add_images(viewer, vis_cfg, paths, routing, anchor_dir, tile_path,
                    tile_name, local_range, offsets,
                    (global_range[0], pos.y, pos.x), True)
        if progress:
            progress(tile_index, len(selected_tiles), tile_name)
    # Keep every tile's image layers together in Napari's layer list.
    for tile_name, (_, local_range, offsets, pos) in tile_contexts.items():
        _add_tile_2d(viewer, result_dir, tile_name, routing, offsets,
                     local_range, pos, True, registry, width, vis_cfg)
    union = (min(b[0] for b in bounds_list), min(b[1] for b in bounds_list),
             max(b[2] for b in bounds_list), max(b[3] for b in bounds_list))
    _add_global_results(
        viewer, result_dir, routing, union, global_range, bounds_list,
        None, True, registry, width, show_coloc,
        bool(vis_cfg.get("show_zlinked", False)),
        bool(vis_cfg.get("spheres", False)))
    _attach_clicks(
        viewer, registry, result_dir, context, tile_contexts, True,
        vis_cfg, paths, routing, anchor_dir)
    if vis_cfg.get("spheres", False):
        viewer.dims.ndisplay = 3
    _show_only_images_initially(viewer, vis_cfg)
    viewer.reset_view()
    add_video_export_panel(viewer, vis_cfg)


def _load_offsets_if_present(result_dir, tile_name):
    path = os.path.join(result_dir, "3_2d_aligned", f"{tile_name}_offsets.json")
    if not os.path.isfile(path):
        return {}
    import json
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)

