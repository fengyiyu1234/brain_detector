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
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

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
    print("\nEnter tile indices (space-separated, e.g. 0 1 2): ", end='', flush=True)
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

def _read_tiff(fpath):
    img = cv2.imread(fpath, cv2.IMREAD_ANYDEPTH)
    if img is None:
        return None
    return img[:, :, 0].astype(np.float32) if img.ndim == 3 else img.astype(np.float32)


def load_volume(tile_dir, z_range=None, downsample_step=1):
    """Stack TIFFs from tile_dir into (Z, H, W) float32 normalised to [0, 1].
    TIFFs are read in parallel; percentile is estimated on a spatial subsample."""
    if not os.path.isdir(tile_dir):
        return None
    files = sorted(f for f in os.listdir(tile_dir)
                   if f.lower().endswith(('.tif', '.tiff')))
    if z_range:
        files = files[z_range[0]:z_range[1]:downsample_step]
    elif downsample_step > 1:
        files = files[::downsample_step]
    if not files:
        return None
    fpaths = [os.path.join(tile_dir, f) for f in files]
    with ThreadPoolExecutor() as ex:
        slices = list(ex.map(_read_tiff, fpaths))
    slices = [s for s in slices if s is not None]
    if not slices:
        return None
    vol = np.stack(slices)
    sample = vol[::2, ::4, ::4]          # subsample for fast percentile
    lo, hi = np.percentile(sample, 0.1), np.percentile(sample, 99.9)
    return np.clip((vol - lo) / (hi - lo + 1e-6), 0.0, 1.0).astype(np.float32)


def build_canvas(selected_tile_paths, ch_base_dir, anchor_root,
                 tile_size,
                 canvas_x0, canvas_y0, canvas_w, canvas_h,
                 tile_positions_xy, z_range=None, downsample_step=1):
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
        vol      = load_volume(tile_dir, z_range, downsample_step)
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

def _build_shapes(df, z, y1, x1, y2, x2):
    """Vectorised rectangle builder. Returns (shapes_list, colors_list)."""
    n = len(df)
    arr = np.empty((n, 4, 3), dtype=np.float64)
    arr[:, 0] = np.column_stack([z, y1, x1])
    arr[:, 1] = np.column_stack([z, y1, x2])
    arr[:, 2] = np.column_stack([z, y2, x2])
    arr[:, 3] = np.column_stack([z, y2, x1])
    colors = [_color(c) for c in df['class']]
    return list(arr), colors


def tile_csv_to_shapes(csv_path, canvas_ox, canvas_oy, z_range_start=0, downsample_step=1):
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
    df['z'] = (df['z'].astype(float).astype(int) - 1 - z_range_start) // downsample_step
    df = df[df['z'] >= 0]
    if df.empty:
        return [], []
    return _build_shapes(df,
                         z=df['z'].values,
                         y1=df['y1'].values + canvas_oy,
                         x1=df['x1'].values + canvas_ox,
                         y2=df['y2'].values + canvas_oy,
                         x2=df['x2'].values + canvas_ox)


def _prepare_global_df(csv_path, canvas_x0, canvas_y0, z0,
                        canvas_w, canvas_h, z_range_start=0,
                        tile_name_filter=None, downsample_step=1):
    """Load a global detection CSV and convert to canvas-local coordinates."""
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if tile_name_filter is not None and 'tile_name' in df.columns:
        df = df[df['tile_name'].isin(tile_name_filter)]
    df = df.copy()
    df['cx1'] = df['x1'].astype(float) - canvas_x0
    df['cx2'] = df['x2'].astype(float) - canvas_x0
    df['cy1'] = df['y1'].astype(float) - canvas_y0
    df['cy2'] = df['y2'].astype(float) - canvas_y0
    df['z_disp'] = (df['z'].astype(float).astype(int) + z0 - 1 - z_range_start) // downsample_step
    df = df[(df['z_disp'] >= 0) &
            (df['cx1'] < canvas_w) & (df['cx2'] > 0) &
            (df['cy1'] < canvas_h) & (df['cy2'] > 0)]
    return df if not df.empty else None


def global_csv_to_shapes(csv_path, canvas_x0, canvas_y0, z0,
                          canvas_w, canvas_h, z_range_start=0,
                          tile_name_filter=None, downsample_step=1):
    """
    Global detection CSV (absolute pixel coords) converted to canvas-local shapes.
    Boxes entirely outside the canvas are dropped.
    tile_name_filter: set of tile names; if provided and 'tile_name' column exists,
    only rows matching that set are kept.
    """
    df = _prepare_global_df(csv_path, canvas_x0, canvas_y0, z0,
                             canvas_w, canvas_h, z_range_start, tile_name_filter,
                             downsample_step)
    if df is None:
        return [], []
    return _build_shapes(df,
                         z=df['z_disp'].values,
                         y1=df['cy1'].values, x1=df['cx1'].values,
                         y2=df['cy2'].values, x2=df['cx2'].values)


# ── Dynamic shapes layer (per-z swap) ──────────────────────────────────────────

def _add_dynamic_shapes_layer_slow(viewer, shapes, colors, **kwargs):
    """
    ShapesLayer variant: swaps per-z rectangles on each z-change.
    NOTE: napari re-triangulates + re-uploads to GPU on every layer.data
    assignment, making this O(N_shapes) per z-tick.  Use _add_labels_layer
    instead unless you need per-box interactivity on a sparse layer (< ~200
    boxes total).
    """
    by_z_shapes = defaultdict(list)
    by_z_colors = defaultdict(list)
    for arr, col in zip(shapes, colors):
        z = int(round(arr[0, 0]))
        by_z_shapes[z].append(arr)
        by_z_colors[z].append(col)

    layer = viewer.add_shapes([], shape_type='rectangle', ndim=3, **kwargs)

    def _on_z_change(_=None):
        z = int(viewer.dims.current_step[0])
        s = by_z_shapes.get(z, [])
        c = by_z_colors.get(z, [])
        layer.data = s
        if s:
            layer.edge_color = c

    viewer.dims.events.current_step.connect(_on_z_change)
    _on_z_change()          # populate the initial z-slice on startup
    return layer


# ── Labels volume (fast z-scroll) ─────────────────────────────────────────────

def _rasterize_shapes_to_labels(shapes, colors, Z, H, W, outline_width=2):
    """Draw box outlines into a (Z, H, W) uint32 volume.

    Returns (vol, color_map) where color_map is {label_id: [r,g,b,a]} suitable
    for viewer.add_labels(color=...).  Label 0 is the transparent background.
    The rasterization runs once at startup; z-sliding then costs only a numpy
    view + one GPU texture upload — no per-frame triangulation.
    """
    vol = np.zeros((Z, H, W), dtype=np.uint32)
    color_to_id: dict = {}
    color_map: dict = {0: [0, 0, 0, 0]}
    next_id = 1

    hw = max(1, outline_width // 2)

    for arr, col in zip(shapes, colors):
        col_key = tuple(col)
        if col_key not in color_to_id:
            color_to_id[col_key] = next_id
            color_map[next_id] = list(col)
            next_id += 1
        label_id = color_to_id[col_key]

        z  = int(round(arr[0, 0]))
        y1 = int(round(arr[:, 1].min()))
        y2 = int(round(arr[:, 1].max()))
        x1 = int(round(arr[:, 2].min()))
        x2 = int(round(arr[:, 2].max()))

        if z < 0 or z >= Z:
            continue
        y1c = max(0, y1);  y2c = min(H - 1, y2)
        x1c = max(0, x1);  x2c = min(W - 1, x2)

        for d in range(hw):
            ty = np.clip(y1c + d, 0, H - 1);  by = np.clip(y2c - d, 0, H - 1)
            lx = np.clip(x1c + d, 0, W - 1);  rx = np.clip(x2c - d, 0, W - 1)
            vol[z, ty, x1c:x2c + 1] = label_id
            vol[z, by, x1c:x2c + 1] = label_id
            vol[z, y1c:y2c + 1, lx] = label_id
            vol[z, y1c:y2c + 1, rx] = label_id

    return vol, color_map


def _add_labels_layer(viewer, shapes, colors, canvas_shape, **kwargs):
    """Rasterize shapes into a labels volume and add to viewer (zero-lag z-scroll).

    Only 'name' and 'visible' are forwarded from **kwargs; ShapesLayer-specific
    kwargs like face_color / edge_width are silently dropped.
    """
    if not shapes:
        return None
    Z, H, W = canvas_shape
    vol, color_map = _rasterize_shapes_to_labels(shapes, colors, Z, H, W)
    labels_kwargs = {k: kwargs[k] for k in ('name', 'visible', 'opacity') if k in kwargs}
    layer = viewer.add_labels(vol, **labels_kwargs)
    layer.color = color_map
    return layer


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Napari multi-tile visualization of brain_detector results.'
    )
    parser.add_argument('--config', default=os.path.join(project_root, 'config', 'config.json'))
    parser.add_argument('--tiles', nargs='+', type=int, default=None,
                        help='Tile indices to load (e.g. --tiles 0 1 4 5). '
                             'Omit for interactive prompt.')
    parser.add_argument('--downsample-step', type=int, default=None,
                        help='只显示每隔 N 层的检测平面；省略时自动读取 config 的 DOWNSAMPLE_Z_STEP')
    parser.add_argument('--list-tiles', action='store_true',
                        help='Print available tile names and exit.')
    parser.add_argument('--no-images', action='store_true',
                        help='Skip raw image loading; show bounding boxes only.')
    parser.add_argument('--stage', choices=['all', 'tile', 's2', 's3', 's4'], default='all')
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

    # ── 直接在此处修改加载范围 ─────────────────────────────────────────────────
    z_range = 574, 861          # [start, end]，例如 [200, 500]；None = 加载全部
    # ──────────────────────────────────────────────────────────────────────────

    z_range_start = z_range[0] if z_range else 0

    if args.downsample_step is not None:
        downsample_step = args.downsample_step
    elif dp.get('DOWNSAMPLE', False):
        downsample_step = int(dp.get('DOWNSAMPLE_Z_STEP', 2))
    else:
        downsample_step = 1

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
    print(f"z0={z0}, z_range={z_range}")
    print("Tile positions in canvas:")
    for name, (ax, ay) in tile_positions_xy.items():
        vw, vh = compute_valid_size(ax, ay, list(tile_positions_xy.values()), tile_size)
        print(f"  {name}: canvas_x={ax-canvas_x0}  canvas_y={ay-canvas_y0}  "
              f"valid={vw}x{vh}")
    print()

    # ── Napari viewer ─────────────────────────────────────────────────────────
    viewer = napari.Viewer(title=f"brain_detector — {len(selected_paths)} tile(s)")

    # ── Raw images ────────────────────────────────────────────────────────────
    canvas_shape = None   # (Z, H, W) — set on first successful canvas load
    if not args.no_images:
        for ch in routing_config:
            ch_id   = ch['id']
            ch_base = os.path.abspath(paths[ch['dir_key']])
            print(f"[img] Loading {ch_id}...")
            canvas_vol = build_canvas(
                selected_paths, ch_base, anchor_root,
                tile_size,
                canvas_x0, canvas_y0, canvas_w, canvas_h,
                tile_positions_xy, z_range, downsample_step,
            )
            if canvas_vol is None:
                print(f"[img] {ch_id}: no data loaded")
                continue
            if canvas_shape is None:
                canvas_shape = canvas_vol.shape
            vis = CHANNEL_VIS.get(ch_id, DEFAULT_VIS)
            viewer.add_image(canvas_vol, name=f"[img] {ch_id}", **vis)
            print(f"[img] {ch_id}: {canvas_vol.shape}\n")

    if canvas_shape is None:
        # --no-images mode: derive Z from TIFF count in the anchor channel
        _ch_base = os.path.abspath(paths[routing_config[0]['dir_key']])
        _rel = os.path.relpath(selected_paths[0], anchor_root)
        _tile_dir = os.path.join(_ch_base, _rel)
        _files = sorted(
            f for f in os.listdir(_tile_dir) if f.lower().endswith(('.tif', '.tiff'))
        ) if os.path.isdir(_tile_dir) else []
        if z_range:
            _files = _files[z_range[0]:z_range[1]:downsample_step]
        elif downsample_step > 1:
            _files = _files[::downsample_step]
        canvas_shape = (len(_files), canvas_h, canvas_w)

    base_res    = paths['pATHRESULT']
    det_res_dir = os.path.join(base_res, '1_tile_2d_raw')
    s3_dir      = os.path.join(base_res, '3_channel_3d')
    path_s4     = os.path.join(base_res, '4_colocalization', 'coloc_result.csv')

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
                sh, co = tile_csv_to_shapes(csv_p, canvas_ox, canvas_oy, z_range_start, downsample_step)
                all_shapes.extend(sh)
                all_colors.extend(co)
            if all_shapes:
                _add_labels_layer(
                    viewer, all_shapes, all_colors, canvas_shape,
                    name=f"[s1] {ch_id}", visible=False,
                )
                print(f"[s1] {ch_id}: {len(all_shapes)} raw 2D boxes")

    # ── Stage 2: per-channel Z-link QC (center-Z slice of each 3D track) ────
    if args.stage in ('all', 's2'):
        for ch in routing_config:
            ch_csv = os.path.join(s3_dir, f"{ch['id']}_3d_tracked.csv")
            if not os.path.exists(ch_csv):
                continue
            shapes, colors = global_csv_to_shapes(
                ch_csv, canvas_x0, canvas_y0, z0,
                canvas_w, canvas_h, z_range_start,
                downsample_step=downsample_step,
            )
            if shapes:
                _add_labels_layer(
                    viewer, shapes, colors, canvas_shape,
                    name=f'[s2] {ch["id"]}', visible=False,
                )
                print(f"[s2] {ch['id']}: {len(shapes)} z-linked cells")

    # ── Stage 3: per-channel Z-link (3_channel_3d) ───────────────────────────
    if args.stage in ('all', 's3'):
        s3_files = []
        for ch in routing_config:
            if ch.get('type', 'soma') == 'soma':
                ch_csv = os.path.join(s3_dir, f"{ch['id']}_3d_tracked.csv")
                if os.path.exists(ch_csv):
                    s3_files.append((ch_csv, f"[s3] {ch['id']}"))
        for ch in routing_config:
            if ch.get('type', 'soma') == 'tf':
                tf_csv = os.path.join(s3_dir, f"{ch['id']}_3d_tracked.csv")
                if os.path.exists(tf_csv):
                    s3_files.append((tf_csv, f"[s3] {ch['id']}"))

        for csv_s3, layer_name in s3_files:
            shapes, colors = global_csv_to_shapes(
                csv_s3, canvas_x0, canvas_y0, z0,
                canvas_w, canvas_h, z_range_start,
                downsample_step=downsample_step,
            )
            if shapes:
                _add_labels_layer(
                    viewer, shapes, colors, canvas_shape,
                    name=layer_name, visible=True,
                )
                print(f"{layer_name}: {len(shapes)} 3D tracked cells")

    # ── Stage 4: 3D colocalization result (4_colocalization/coloc_result.csv) ──
    if args.stage in ('all', 's4'):
        df_s4 = _prepare_global_df(
            path_s4, canvas_x0, canvas_y0, z0,
            canvas_w, canvas_h, z_range_start,
            tile_name_filter=selected_names,
            downsample_step=downsample_step,
        )
        if df_s4 is not None:
            total = 0
            for cls_name, df_cls in df_s4.groupby('class'):
                shapes, colors = _build_shapes(
                    df_cls,
                    z=df_cls['z_disp'].values,
                    y1=df_cls['cy1'].values, x1=df_cls['cx1'].values,
                    y2=df_cls['cy2'].values, x2=df_cls['cx2'].values,
                )
                _add_labels_layer(
                    viewer, shapes, colors, canvas_shape,
                    name=f'[s4] {cls_name}', visible=True,
                )
                print(f"[s4] {cls_name}: {len(shapes)} cells")
                total += len(shapes)
            print(f"[s4] total: {total} colocalized cells")
            print("\nClass breakdown:")
            for cls, cnt in df_s4['class'].value_counts().items():
                print(f"  {cls:35s} {cnt}")

    perm_path = os.path.join(base_res, '5_analysis_report', 'colocalization_significance.csv')
    if os.path.exists(perm_path):
        print("\nColocalization significance:")
        print(pd.read_csv(perm_path).to_string(index=False))

    viewer.reset_view()
    napari.run()


if __name__ == '__main__':
    main()
