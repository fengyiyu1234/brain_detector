#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Napari multi-tile visualization of brain_detector pipeline results.

Loads raw images for one or more tiles, stitches them into a single canvas
using TeraStitcher merge XML coordinates.  Overlap is resolved by cropping
each tile's trailing (right/bottom) edge at the start of its nearest selected
neighbour — no blending needed for quality inspection.

Layers added
------------
  [img] <channel>   stitched Z-volume per active channel
  [s1] <channel>    tile-level 2D raw detections  (hidden by default)
  [s3] <channel>    per-channel Z-linked 3D cells
  [s4] colocalized  final result from global_bboxes.csv

Usage
-----
  python src/utils/visualize.py                       # interactive tile selection
  python src/utils/visualize.py --tiles 0 1 4 5      # select by index
  python src/utils/visualize.py --list-tiles
  python src/utils/visualize.py --z-range 0 80
  python src/utils/visualize.py --no-images
  python src/utils/visualize.py --stage s4
"""

import argparse
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np
import pandas as pd
import napari

from src.config.loader import load_config
from src.utils.io import listTile, loadTeraxml

# ── Display config ──────────────────────────────────────────────────────────────
CHANNEL_VIS = {
    'RFP':  dict(colormap='red',   blending='additive', opacity=0.8),
    'GFP':  dict(colormap='green', blending='additive', opacity=0.8),
    'Sox9': dict(colormap='cyan',  blending='additive', opacity=0.8),
}
DEFAULT_VIS = dict(colormap='gray', blending='additive', opacity=0.7)

CLASS_COLOR = {
    'neuron':  [0.0, 0.55, 1.0, 1.0],
    'glia':    [1.0, 0.55, 0.0, 1.0],
    'nucleus': [0.2, 0.9,  0.2, 1.0],
}

def _color(class_str):
    base = str(class_str).split('_')[0]
    return CLASS_COLOR.get(base, [1.0, 1.0, 1.0, 1.0])


# ── Tile selection ──────────────────────────────────────────────────────────────

def select_tiles_interactive(tile_paths):
    """Print numbered tile list, prompt user to enter indices, return selected paths."""
    print(f"\nAvailable tiles ({len(tile_paths)}):")
    for i, p in enumerate(tile_paths):
        print(f"  [{i:3d}] {os.path.basename(p)}")
    print("\nEnter tile indices (space-separated, e.g. '0 1 4 5'): ", end='', flush=True)
    raw = input().strip()
    if not raw:
        print("No input — loading tile 0.")
        return [tile_paths[0]]
    indices = [int(x) for x in raw.split()]
    selected = [tile_paths[i] for i in indices]
    print(f"Selected: {[os.path.basename(p) for p in selected]}")
    return selected


# ── Tile position helpers ───────────────────────────────────────────────────────

def get_tile_pos(tile_name, dir_dict, disp_mat):
    """Return (abs_x, abs_y, abs_z) for tile_name from disp_mat, or None."""
    for dir_name, (row, col) in dir_dict.items():
        if os.path.basename(dir_name) == tile_name or dir_name == tile_name:
            ax, ay, az = disp_mat[row, col]
            return int(ax), int(ay), int(az)
    return None


def compute_valid_size(abs_x, abs_y, selected_positions_xy, tile_size):
    """
    For a tile at (abs_x, abs_y), return (valid_w, valid_h): the pixel extent
    that does not overlap with any adjacent selected tile to the right / below.
    """
    all_xs = sorted(set(p[0] for p in selected_positions_xy))
    all_ys = sorted(set(p[1] for p in selected_positions_xy))
    right  = [x for x in all_xs if x > abs_x]
    below  = [y for y in all_ys if y > abs_y]
    valid_w = (min(right) - abs_x) if right else tile_size
    valid_h = (min(below) - abs_y) if below else tile_size
    return min(valid_w, tile_size), min(valid_h, tile_size)


# ── Image loading ───────────────────────────────────────────────────────────────

def load_volume(tile_dir, z_range=None):
    """Stack TIFFs from tile_dir into (Z, H, W) float32 normalised to [0, 1]."""
    if not os.path.isdir(tile_dir):
        return None
    files = sorted(f for f in os.listdir(tile_dir)
                   if f.lower().endswith(('.tif', '.tiff')))
    if z_range:
        files = files[z_range[0]:z_range[1]]
    slices = []
    for f in files:
        img = cv2.imread(os.path.join(tile_dir, f), cv2.IMREAD_ANYDEPTH)
        if img is None:
            continue
        if img.ndim == 3:
            img = img[:, :, 0]
        slices.append(img.astype(np.float32))
    if not slices:
        return None
    vol = np.stack(slices)
    lo, hi = np.percentile(vol, 0.1), np.percentile(vol, 99.9)
    return np.clip((vol - lo) / (hi - lo + 1e-6), 0.0, 1.0).astype(np.float32)


def build_canvas(selected_tile_paths, ch_base_dir, anchor_root,
                 tile_size,
                 canvas_x0, canvas_y0, canvas_w, canvas_h,
                 tile_positions_xy, z_range=None):
    """
    Load selected tiles for one channel and stitch into a (Z, H, W) canvas.
    Overlap is resolved by cropping each tile's trailing edge at the boundary
    of its nearest right/bottom selected neighbour.
    Returns the canvas array, or None if no tiles could be loaded.
    """
    canvas = None
    for tile_path in selected_tile_paths:
        name = os.path.basename(tile_path)
        pos  = tile_positions_xy.get(name)
        if pos is None:
            continue
        abs_x, abs_y = pos

        rel      = os.path.relpath(tile_path, anchor_root)
        tile_dir = os.path.join(ch_base_dir, rel)
        vol      = load_volume(tile_dir, z_range)
        if vol is None:
            print(f"  Not found: {tile_dir}")
            continue

        if canvas is None:
            canvas = np.zeros((vol.shape[0], canvas_h, canvas_w), dtype=np.float32)

        valid_w, valid_h = compute_valid_size(
            abs_x, abs_y, list(tile_positions_xy.values()), tile_size
        )
        cx, cy = abs_x - canvas_x0, abs_y - canvas_y0
        canvas[:, cy:cy + valid_h, cx:cx + valid_w] = vol[:, :valid_h, :valid_w]
        print(f"  {name}: canvas[y={cy}:{cy+valid_h}, x={cx}:{cx+valid_w}]  "
              f"(valid {valid_w}x{valid_h} / {tile_size}x{tile_size})")

    return canvas


# ── Shape building ──────────────────────────────────────────────────────────────

def _rect(z, y1, x1, y2, x2):
    return np.array([[z, y1, x1], [z, y1, x2],
                     [z, y2, x2], [z, y2, x1]], dtype=float)


def tile_csv_to_shapes(csv_path, canvas_ox, canvas_oy, z_range_start=0):
    """
    Tile-level 2D raw CSV (local pixel coords).
    canvas_ox / canvas_oy are the tile's top-left offset within the canvas.
    """
    if not os.path.exists(csv_path):
        return [], []
    df = pd.read_csv(
        csv_path,
        names=['slice_name', 'x1', 'y1', 'x2', 'y2', 'class', 'score', 'mean', 'z'],
        skiprows=1,
    )
    df['z'] = df['z'].astype(float).astype(int) - 1 - z_range_start
    df = df[df['z'] >= 0]
    shapes, colors = [], []
    for _, r in df.iterrows():
        shapes.append(_rect(r['z'],
                            r['y1'] + canvas_oy, r['x1'] + canvas_ox,
                            r['y2'] + canvas_oy, r['x2'] + canvas_ox))
        colors.append(_color(r['class']))
    return shapes, colors


def global_csv_to_shapes(csv_path, canvas_x0, canvas_y0, z0,
                          canvas_w, canvas_h, z_range_start=0,
                          tile_name_filter=None):
    """
    Global detection CSV (absolute pixel coords) converted to canvas-local shapes.
    Boxes entirely outside the canvas are dropped.
    tile_name_filter: set of tile names; if provided and 'tile_name' column exists,
    only rows matching that set are kept.
    """
    if not os.path.exists(csv_path):
        return [], []
    df = pd.read_csv(csv_path)

    if tile_name_filter is not None and 'tile_name' in df.columns:
        df = df[df['tile_name'].isin(tile_name_filter)]

    df = df.copy()
    df['cx1'] = df['x1'].astype(float) - canvas_x0
    df['cx2'] = df['x2'].astype(float) - canvas_x0
    df['cy1'] = df['y1'].astype(float) - canvas_y0
    df['cy2'] = df['y2'].astype(float) - canvas_y0
    df['z_disp'] = df['z'].astype(float).astype(int) + z0 - 1 - z_range_start

    df = df[(df['z_disp'] >= 0) &
            (df['cx1'] < canvas_w) & (df['cx2'] > 0) &
            (df['cy1'] < canvas_h) & (df['cy2'] > 0)]

    shapes, colors = [], []
    for _, r in df.iterrows():
        shapes.append(_rect(r['z_disp'], r['cy1'], r['cx1'], r['cy2'], r['cx2']))
        colors.append(_color(r['class']))
    return shapes, colors


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Napari multi-tile visualization of brain_detector results.'
    )
    parser.add_argument('--config', default='D:/Fengyi/brain_detector/config/config.json')
    parser.add_argument('--tiles', nargs='+', type=int, default=None,
                        help='Tile indices to load (e.g. --tiles 0 1 4 5). '
                             'Omit for interactive prompt.')
    parser.add_argument('--z-range', nargs=2, type=int, default=None,
                        metavar=('Z_START', 'Z_END'))
    parser.add_argument('--list-tiles', action='store_true',
                        help='Print available tile names and exit.')
    parser.add_argument('--no-images', action='store_true',
                        help='Skip raw image loading; show bounding boxes only.')
    parser.add_argument('--stage', choices=['all', 'tile', 's3', 's4'], default='all')
    args = parser.parse_args()

    config = load_config(args.config)
    paths  = config['paths']
    dp     = config['detection_params']
    routing_config = [ch for ch in config['channels_routing'] if ch.get('active', True)]

    anchor_ch   = routing_config[0]
    anchor_dir  = paths[anchor_ch['dir_key']]
    anchor_root = os.path.abspath(anchor_dir)

    _, tile_paths = listTile(anchor_dir)

    if args.list_tiles:
        print(f"Available tiles ({len(tile_paths)}):")
        for i, p in enumerate(tile_paths):
            print(f"  [{i:3d}] {os.path.basename(p)}")
        return

    # ── Select tiles ──────────────────────────────────────────────────────────
    if args.tiles is not None:
        selected_paths = [tile_paths[i] for i in args.tiles]
        print(f"Selected: {[os.path.basename(p) for p in selected_paths]}")
    else:
        selected_paths = select_tiles_interactive(tile_paths)

    selected_names = {os.path.basename(p) for p in selected_paths}

    z_range       = args.z_range
    z_range_start = z_range[0] if z_range else 0

    # ── XML tile positions ─────────────────────────────────────────────────────
    xml_name = ('xml_merging.xml'
                if os.path.isfile(os.path.join(anchor_dir, 'xml_merging.xml'))
                else 'xml_import.xml')
    pATHxml   = os.path.join(anchor_dir, xml_name)
    tile_size = dp.get('tILESIZE', 2048)

    dir_dict, _, _, _, z_start, disp_mat = loadTeraxml(pATHxml, tile_size)

    tile_positions_xy = {}  # name → (abs_x, abs_y)
    tile_abs_z        = {}  # name → abs_z
    for p in selected_paths:
        name = os.path.basename(p)
        pos  = get_tile_pos(name, dir_dict, disp_mat)
        if pos is None:
            print(f"Warning: {name} not found in XML — skipping.")
            continue
        tile_positions_xy[name] = (pos[0], pos[1])
        tile_abs_z[name]        = pos[2]

    if not tile_positions_xy:
        sys.exit("No valid tiles found in XML.")

    xs = [v[0] for v in tile_positions_xy.values()]
    ys = [v[1] for v in tile_positions_xy.values()]
    canvas_x0 = min(xs)
    canvas_y0 = min(ys)
    canvas_w  = max(xs) + tile_size - canvas_x0
    canvas_h  = max(ys) + tile_size - canvas_y0

    abs_z_ref = next(iter(tile_abs_z.values()))
    z0 = z_start - abs_z_ref

    print(f"\nCanvas: {canvas_w}x{canvas_h} px  "
          f"(global offset x0={canvas_x0}, y0={canvas_y0})")
    print(f"z0={z0}, z_range={z_range}\n")

    # ── Napari viewer ─────────────────────────────────────────────────────────
    viewer = napari.Viewer(title=f"brain_detector — {len(selected_paths)} tile(s)")

    # ── Raw images ────────────────────────────────────────────────────────────
    if not args.no_images:
        for ch in routing_config:
            ch_id   = ch['id']
            ch_base = os.path.abspath(paths[ch['dir_key']])
            print(f"[img] Loading {ch_id}...")
            canvas_vol = build_canvas(
                selected_paths, ch_base, anchor_root,
                tile_size,
                canvas_x0, canvas_y0, canvas_w, canvas_h,
                tile_positions_xy, z_range,
            )
            if canvas_vol is None:
                print(f"[img] {ch_id}: no data loaded")
                continue
            vis = CHANNEL_VIS.get(ch_id, DEFAULT_VIS)
            viewer.add_image(canvas_vol, name=f"[img] {ch_id}", **vis)
            print(f"[img] {ch_id}: {canvas_vol.shape}\n")

    base_res    = paths['pATHRESULT']
    det_res_dir = os.path.join(base_res, '1_tile_2d_raw')
    s3_dir      = os.path.join(base_res, '3_global_3d_single')
    path_s4     = os.path.join(base_res, '4_global_3d_final', 'global_bboxes.csv')

    # ── Stage 1: tile-level 2D raw ────────────────────────────────────────────
    if args.stage in ('all', 'tile'):
        for ch in routing_config:
            ch_id = ch['id']
            all_shapes, all_colors = [], []
            for tile_path in selected_paths:
                name = os.path.basename(tile_path)
                pos  = tile_positions_xy.get(name)
                if pos is None:
                    continue
                canvas_ox = pos[0] - canvas_x0
                canvas_oy = pos[1] - canvas_y0
                csv_p = os.path.join(det_res_dir, f"{name}_{ch_id}_result.csv")
                sh, co = tile_csv_to_shapes(csv_p, canvas_ox, canvas_oy, z_range_start)
                all_shapes.extend(sh)
                all_colors.extend(co)
            if all_shapes:
                viewer.add_shapes(
                    all_shapes, shape_type='rectangle',
                    edge_color=all_colors, face_color='transparent',
                    edge_width=1.5, name=f"[s1] {ch_id}", visible=False,
                )
                print(f"[s1] {ch_id}: {len(all_shapes)} raw 2D boxes")

    # ── Stage 3: per-channel Z-linked ─────────────────────────────────────────
    if args.stage in ('all', 's3'):
        for ch in routing_config:
            ch_id  = ch['id']
            csv_s3 = os.path.join(s3_dir, f"{ch_id}_3d_tracked.csv")
            shapes, colors = global_csv_to_shapes(
                csv_s3, canvas_x0, canvas_y0, z0,
                canvas_w, canvas_h, z_range_start,
            )
            if shapes:
                viewer.add_shapes(
                    shapes, shape_type='rectangle',
                    edge_color=colors, face_color='transparent',
                    edge_width=2.0, name=f"[s3] {ch_id}", visible=True,
                )
                print(f"[s3] {ch_id}: {len(shapes)} 3D tracked cells")

    # ── Stage 4: final colocalized result ─────────────────────────────────────
    if args.stage in ('all', 's4'):
        shapes, colors = global_csv_to_shapes(
            path_s4, canvas_x0, canvas_y0, z0,
            canvas_w, canvas_h, z_range_start,
            tile_name_filter=selected_names,
        )
        if shapes:
            viewer.add_shapes(
                shapes, shape_type='rectangle',
                edge_color=colors, face_color='transparent',
                edge_width=3.0, name='[s4] colocalized', visible=True,
            )
            print(f"[s4] colocalized: {len(shapes)} cells")

        if os.path.exists(path_s4):
            df_s4 = pd.read_csv(path_s4)
            df_sel = (df_s4[df_s4['tile_name'].isin(selected_names)]
                      if 'tile_name' in df_s4.columns else df_s4)
            print("\nClass breakdown:")
            for cls, cnt in df_sel['class'].value_counts().items():
                print(f"  {cls:35s} {cnt}")

    perm_path = os.path.join(base_res, '5_analysis_report', 'colocalization_significance.csv')
    if os.path.exists(perm_path):
        print("\nColocalization significance:")
        print(pd.read_csv(perm_path).to_string(index=False))

    napari.run()


if __name__ == '__main__':
    main()
