"""Napari views of saved pre-align pipeline outputs.

All cell layers come from CSV/PKL checkpoints. Image shifts and coordinate
translations are the only operations performed by this module.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import napari

from src.utils.markers import class_markers
from src.utils.saved_view import (
    global_to_local_rows, read_global_boxes, read_tile_boxes,
    records_and_shapes, track_for_summary,
)
from src.utils.visualize import (
    _canvas_shape_from_tile, _ch_vis, _list_tiffs, _make_fn_recorder,
    _resolve_z_range, _show_only_images_initially, load_frame_volume,
)


def _image_tile_dir(paths, ch, anchor_dir, tile_path):
    return os.path.join(paths[ch["dir_key"]], os.path.relpath(tile_path, anchor_dir))


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
            second_dir = os.path.join(
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
                path = os.path.join(
                    result_dir, directory, f"{tile_name}_{cid}_result.csv")
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


def _add_global_results(viewer, result_dir, routing, xy_bounds,
                        global_z_range, tile_bounds, position, global_view,
                        registry, width, show_coloc, show_zlinked, show_spheres):
    for ch in routing:
        cid = ch["id"]
        for directory, filename, label in (
            ("5_2d_global", f"{cid}_2d_global.csv", "global"),
            ("6_3d_global", f"{cid}_3d_tracked.csv", "s3"),
        ):
            saved_path = os.path.join(result_dir, directory, filename)
            if not os.path.isfile(saved_path):
                print(f"[saved] Stage unavailable: {saved_path}")
            rows = read_global_boxes(saved_path, xy_bounds, global_z_range)
            rows = _intersects_any(rows, tile_bounds)
            if not global_view:
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
            rows = global_to_local_rows(rows, position)
        _add_s4_layers(viewer, rows, registry, width)
        if not os.path.isfile(path):
            print(f"[saved] Stage 4 unavailable: {path}")


def _attach_clicks(viewer, registry, result_dir, context, tile_contexts,
                   global_view, vis_cfg, paths, routing, anchor_dir):
    """Inspect exact saved rows; load Stage 3 PKL only when a track is requested."""
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
        if not is_shift or row["source"] != "6_3d_global":
            if is_shift and row["source"] == "s4":
                v.status += " | Stage 4 stores representative z only"
            return
        cid = row["channel"]
        pkl_path = os.path.join(result_dir, "6_3d_global", f"{cid}_3d_tracked.pkl")
        if not os.path.isfile(pkl_path):
            v.status = f"Saved track unavailable: {pkl_path}"
            return
        if cid not in track_cache:
            with open(pkl_path, "rb") as handle:
                track_cache[cid] = pickle.load(handle)
        summary = dict(row)
        if not global_view:
            tile_name = next(iter(tile_contexts))
            position = tile_contexts[tile_name][3]
            summary["x1"] += position.x
            summary["x2"] += position.x
            summary["y1"] += position.y
            summary["y2"] += position.y
        track = track_for_summary(summary, track_cache[cid])
        if track is None:
            v.status = "No unique saved PKL track matches this Stage 3 row"
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
        v.status = f"Saved {cid} track spans z={track['z_min']}–{track['z_max']}"

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
                    bool(vis_cfg.get("spheres", False)))
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


def _load_offsets_if_present(result_dir, tile_name):
    path = os.path.join(result_dir, "3_2d_aligned", f"{tile_name}_offsets.json")
    if not os.path.isfile(path):
        return {}
    import json
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)

