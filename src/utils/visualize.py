# -*- coding: utf-8 -*-
"""
Brain-detector unified single-tile visualizer.

Reads config/vis_config.json and launches napari in one of two modes:

  mode: "prealign"
    Saved-result QC in tile-local or global coordinates. Images use saved
    channel offsets; 2D, 3D, and colocalization layers read pipeline files.
    No z-linking or colocalization runs in the GUI.

  mode: "2d"
    Images and tile-local 2D boxes from the selected pipeline stage.
    Only aligned stages apply channel offsets.  Ctrl+click records false negatives.
    Does not load XML or run z-linking.

Usage
-----
  python src/utils/visualize.py
  python src/utils/visualize.py --config config/vis_config.json
"""

import argparse
import datetime
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
from src.utils.coordinate_context import CoordinateContextError
import numpy as np
import pandas as pd
import napari
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication, QMessageBox, QWidget, QGridLayout, QLabel, QSpinBox, QLineEdit

from src.utils.video_export import add_video_export_panel
from src.config.loader import load_config
from src.utils.tile_2d import SOURCES_2D, tile_2d_inputs
from src.utils.io import listTile, resolve_tile_dir
from src.utils.markers import channel_marker


# ── Display constants ─────────────────────────────────────────────────────────

CHANNEL_VIS = {
    'RFP':  dict(colormap='red',     blending='additive', opacity=0.8),
    'GFP':  dict(colormap='green',   blending='additive', opacity=0.8),
    'Sox9': dict(colormap='cyan',    blending='additive', opacity=0.8),
}
_EXTRA_VIS = {
    'Olig2': dict(colormap='magenta', blending='additive', opacity=0.8),
}
DEFAULT_VIS = dict(colormap='gray', blending='additive', opacity=0.7)

CLASS_COLOR = {
    'neuron':  [0.0, 0.55, 1.0, 1.0],
    'glia':    [1.0, 0.55, 0.0, 1.0],
    'nucleus': [0.2, 0.9,  0.2, 1.0],
}


# ── Shared helpers ────────────────────────────────────────────────────────────

def _color(class_str):
    base = str(class_str).split('_')[0]
    return CLASS_COLOR.get(base, [1.0, 1.0, 1.0, 1.0])


def _ch_vis(ch_id):
    for key in (ch_id, channel_marker(ch_id)):
        if key in CHANNEL_VIS:
            return CHANNEL_VIS[key]
        if key in _EXTRA_VIS:
            return _EXTRA_VIS[key]
    return DEFAULT_VIS


def _read_tiff(fpath, native_dtype=False):
    img = cv2.imread(fpath, cv2.IMREAD_ANYDEPTH)
    if img is None:
        return None
    if img.ndim == 3:
        img = img[:, :, 0]
    return img if native_dtype else img.astype(np.float32)


def load_volume(tile_dir, z_range=None, contrast_pct=(0.1, 99.9),
                native_dtype=False, files=None):
    """Load selected TIFF slices and estimate contrast from a spatial subsample.

    The default float32 output preserves existing callers. The frame viewer can
    request the original TIFF dtype to avoid doubling memory for 16-bit images.
    """
    if files is None:
        files = _list_tiffs(tile_dir)
    if z_range:
        files = files[z_range[0]:z_range[1]]
    if not files:
        return None, None
    paths = [os.path.join(tile_dir, name) for name in files]
    with ThreadPoolExecutor() as ex:
        slices = list(ex.map(partial(_read_tiff, native_dtype=native_dtype), paths))
    slices = [image for image in slices if image is not None]
    if not slices:
        return None, None
    vol = np.stack(slices)
    sample = vol[::2, ::4, ::4]
    # Contrast is display-only. Bound percentile work for large Z stacks.
    while sample.size > 1_000_000:
        sample = sample[::2, ::2, ::2]
    lo, hi = np.percentile(sample, contrast_pct)
    return vol, (float(lo), float(hi))


def load_frame_volume(tile_dir, frame_z_range, shift=(0, 0, 0), contrast_pct=(0.1, 99.9)):
    """Load TIFFs into final-frame coordinates without extra full-volume copies."""
    dx, dy, dz = (int(value) for value in shift)
    frame_z0, frame_z1 = frame_z_range
    files = _list_tiffs(tile_dir)
    raw_z0 = max(0, frame_z0 - dz)
    raw_z1 = max(0, min(len(files), frame_z1 - dz))
    if raw_z1 <= raw_z0:
        return None, None
    source, climits = load_volume(
        tile_dir, (raw_z0, raw_z1), contrast_pct,
        native_dtype=True, files=files)
    if source is None:
        return None, None
    depth, height, width = source.shape
    canvas_depth = frame_z1 - frame_z0
    dst_z0 = raw_z0 + dz - frame_z0
    if dx == 0 and dy == 0 and dst_z0 == 0 and depth == canvas_depth:
        return source, climits

    src_x0, src_x1 = max(0, -dx), min(width, width - dx)
    src_y0, src_y1 = max(0, -dy), min(height, height - dy)
    src_z0, src_z1 = max(0, -dst_z0), min(depth, canvas_depth - dst_z0)
    out = np.zeros((canvas_depth, height, width), dtype=source.dtype)
    if src_x1 > src_x0 and src_y1 > src_y0 and src_z1 > src_z0:
        out[dst_z0 + src_z0:dst_z0 + src_z1,
            src_y0 + dy:src_y1 + dy,
            src_x0 + dx:src_x1 + dx] = source[
                src_z0:src_z1, src_y0:src_y1, src_x0:src_x1]
    return out, climits


def _rasterize_shapes_to_labels(shapes, colors, Z, H, W,
                                 outline_width=2, dash_sizes=None):
    vol = np.zeros((Z, H, W), dtype=np.uint32)
    color_to_id, color_map = {}, {0: [0, 0, 0, 0]}
    next_id = 1
    hw = max(1, outline_width // 2)
    for i, (arr, col) in enumerate(zip(shapes, colors)):
        col_key = tuple(col)
        if col_key not in color_to_id:
            color_to_id[col_key] = next_id
            color_map[next_id] = list(col)
            next_id += 1
        label_id = color_to_id[col_key]
        z  = int(round(arr[0, 0]))
        y1 = int(round(arr[:, 1].min())); y2 = int(round(arr[:, 1].max()))
        x1 = int(round(arr[:, 2].min())); x2 = int(round(arr[:, 2].max()))
        if z < 0 or z >= Z:
            continue
        y1c = max(0, y1); y2c = min(H - 1, y2)
        x1c = max(0, x1); x2c = min(W - 1, x2)
        dash = dash_sizes[i] if dash_sizes else 0
        for d in range(hw):
            ty = np.clip(y1c + d, 0, H - 1); by = np.clip(y2c - d, 0, H - 1)
            lx = np.clip(x1c + d, 0, W - 1); rx = np.clip(x2c - d, 0, W - 1)
            if dash > 0:
                dh = max(1, dash // 2)
                xs = np.arange(x1c, x2c + 1)
                hm = (xs // dh) % 2 == 0
                vol[z, ty, xs[hm]] = label_id; vol[z, by, xs[hm]] = label_id
                ys = np.arange(y1c, y2c + 1)
                vm = (ys // dh) % 2 == 0
                vol[z, ys[vm], lx] = label_id; vol[z, ys[vm], rx] = label_id
            else:
                vol[z, ty, x1c:x2c + 1] = label_id; vol[z, by, x1c:x2c + 1] = label_id
                vol[z, y1c:y2c + 1, lx] = label_id; vol[z, y1c:y2c + 1, rx] = label_id
    return vol, color_map


def _add_labels_layer(viewer, shapes, colors, canvas_shape,
                       dash_sizes=None, **kwargs):
    if not shapes:
        return None
    Z, H, W = canvas_shape
    outline_width = kwargs.pop('outline_width', 2)
    vol, color_map = _rasterize_shapes_to_labels(
        shapes, colors, Z, H, W,
        outline_width=outline_width, dash_sizes=dash_sizes,
    )
    labels_kwargs = {k: kwargs[k] for k in ('name', 'visible', 'opacity') if k in kwargs}
    layer = viewer.add_labels(vol, **labels_kwargs)
    layer.color = color_map
    return layer


# ── Tile selection & Z-range ──────────────────────────────────────────────────

def _get_tile_grid_from_names(tile_paths):
    """Derive a (row, col) grid position for each tile from its directory name.

    Tile leaf directories are named "<V>_<H>[...]" where V/H are the microscope's
    absolute XY stage coordinates (io.listTile's two-level raw layout: an outer
    dir named V containing leaves V_H). Ranking the unique V/H values gives a
    compact 0-based grid position — no TeraStitcher XML required.
    Returns {tile_name: (row, col)}, or None if names don't follow this pattern.
    """
    raw = []
    for p in tile_paths:
        name = os.path.basename(p)
        parts = name.split('_')
        if len(parts) < 2:
            return None
        try:
            v, h = int(parts[0]), int(parts[1])
        except ValueError:
            return None
        raw.append((name, v, h))
    rows = sorted({v for _, v, _ in raw})
    cols = sorted({h for _, _, h in raw})
    row_idx = {v: i for i, v in enumerate(rows)}
    col_idx = {h: i for i, h in enumerate(cols)}
    return {name: (row_idx[v], col_idx[h]) for name, v, h in raw}


def _print_tile_grid(grid, name_to_idx):
    """Print tiles laid out by (row, col) so spatial neighbors are visually obvious."""
    pos_to_name = {pos: name for name, pos in grid.items()}
    rows = sorted({r for r, _ in grid.values()})
    cols = sorted({c for _, c in grid.values()})
    idx_w = max(3, len(str(len(name_to_idx) - 1)))
    row_w = max(3, len(str(max(rows))))
    print(f"\nTile grid layout ({len(rows)} rows x {len(cols)} cols, {len(name_to_idx)} tiles):")
    print(" " * (row_w + 1) + " ".join(f"{c:>{idx_w}d}" for c in cols) + "   <- col")
    for r in rows:
        cells = []
        for c in cols:
            name = pos_to_name.get((r, c))
            idx = name_to_idx.get(name) if name is not None else None
            cells.append(f"{idx:>{idx_w}d}" if idx is not None else "-" * idx_w)
        print(f"{r:>{row_w}d} " + " ".join(cells))
    print("^ row")


def _select_tiles(vis_cfg, tile_paths):
    """Resolve which tile(s) to visualize -> list of (tile_path, tile_name).

    vis_cfg['tile'] may be a single tile-dir name, a list of names, or absent
    (None) — in which case the tile list/grid is printed and the user types one
    or more comma-separated indices (e.g. "1,3,10,15") to open several at once.
    """
    tile_arg = vis_cfg.get('tile', None)
    if tile_arg is not None:
        names = tile_arg if isinstance(tile_arg, list) else [tile_arg]
        selected = []
        for name in names:
            tile_path = next((p for p in tile_paths if os.path.basename(p) == name), None)
            if tile_path is None:
                sys.exit(f"Tile '{name}' not found.")
            selected.append((tile_path, name))
        return selected

    name_to_idx = {os.path.basename(p): i for i, p in enumerate(tile_paths)}
    grid = _get_tile_grid_from_names(tile_paths)
    if grid:
        _print_tile_grid(grid, name_to_idx)

    print(f"\nAvailable tiles ({len(tile_paths)}):")
    for i, p in enumerate(tile_paths):
        print(f"  [{i:3d}] {os.path.basename(p)}")
    raw = input("\nEnter tile index(es), comma-separated (e.g. 1,3,10,15): ").strip()
    idxs = [int(s) for s in raw.split(',') if s.strip()] if raw else [0]
    return [(tile_paths[i], os.path.basename(tile_paths[i])) for i in idxs]


def _list_tiffs(path):
    return sorted(f for f in os.listdir(path)
                  if f.lower().endswith(('.tif', '.tiff'))
                  ) if os.path.isdir(path) else []


def _add_current_image_name(viewer, tile_path, z_range, dz=0, source_range=None):
    """Show the anchor channel's source TIFF for the current Z slice."""
    files = _list_tiffs(tile_path)
    if not files:
        return

    panel = QWidget()
    layout = QGridLayout(panel)
    layout.setContentsMargins(4, 2, 4, 2)
    layout.addWidget(QLabel("Current image:"), 0, 0)
    name = QLineEdit()
    name.setReadOnly(True)
    layout.addWidget(name, 0, 1)

    def update_name(event=None):
        z = z_range[0] + viewer.dims.current_step[0] - dz
        if 0 <= z < len(files) and (source_range is None or source_range[0] <= z < source_range[1]):
            name.setText(files[z])
            name.setToolTip(os.path.join(tile_path, files[z]))
        else:
            name.setText("(no source image at this Z)")
            name.setToolTip("")

    viewer.dims.events.current_step.connect(update_name)
    update_name()
    viewer.window.add_dock_widget(panel, name="Source image", area="bottom")


def _resolve_z_range(vis_cfg, tile_path):
    tiffs   = _list_tiffs(tile_path)
    total_z = len(tiffs)
    z_start_cfg = vis_cfg.get('z_start', None)
    z_count_cfg = vis_cfg.get('z_count', None)
    s = int(z_start_cfg) if z_start_cfg is not None else 0
    c = int(z_count_cfg) if z_count_cfg is not None else total_z - s
    return (max(0, s), min(total_z, s + c))


def _canvas_shape_from_tile(tile_path, z_range):
    tiffs = _list_tiffs(tile_path)
    canvas_z = z_range[1] - z_range[0]
    if tiffs:
        _img = cv2.imread(os.path.join(tile_path, tiffs[0]), cv2.IMREAD_ANYDEPTH)
        if _img is not None:
            return (canvas_z, _img.shape[0], _img.shape[1])
    return (canvas_z, 2048, 2048)


# ── Layer visibility ──────────────────────────────────────────────────────────

def _show_only_images_initially(viewer, vis_cfg):
    """Start with TIFF layers visible and all result/overlay layers hidden."""
    if not vis_cfg.get('images_only_initially', True):
        return
    for layer in viewer.layers:
        layer.visible = layer.name.startswith('[img]')


# ── Contrast control panel ─────────────────────────────────────────────────────

def _add_contrast_panel(viewer, image_layers):
    """Dock a small panel with min/max intensity spin boxes, one row per channel.

    image_layers: list of (cid, layer) tuples for the channel image layers
    (not overlays). Two-way synced with each layer's contrast_limits:
    editing a spin box updates the layer live; dragging napari's own
    slider updates the spin boxes back. A per-row guard flag prevents
    feedback loops in both directions.
    """
    if not image_layers:
        return

    panel  = QWidget()
    layout = QGridLayout()
    panel.setLayout(layout)
    layout.addWidget(QLabel("channel"), 0, 0)
    layout.addWidget(QLabel("min"),     0, 1)
    layout.addWidget(QLabel("max"),     0, 2)

    for row, (cid, layer) in enumerate(image_layers, start=1):
        layout.addWidget(QLabel(cid), row, 0)

        min_box, max_box = QSpinBox(), QSpinBox()
        min_box.setRange(0, 65535)
        max_box.setRange(0, 65535)
        lo, hi = layer.contrast_limits
        min_box.setValue(int(round(lo)))
        max_box.setValue(int(round(hi)))
        layout.addWidget(min_box, row, 1)
        layout.addWidget(max_box, row, 2)

        guard = [False]  # True while this row is applying a programmatic update

        def _commit(role, layer=layer, min_box=min_box, max_box=max_box, guard=guard):
            if guard[0]:
                return
            lo_v, hi_v = min_box.value(), max_box.value()
            if lo_v >= hi_v:
                # clamp instead of letting layer.contrast_limits raise ValueError
                if role == 'min':
                    hi_v = lo_v + 1
                    if hi_v > 65535:
                        hi_v, lo_v = 65535, 65534
                else:
                    lo_v = hi_v - 1
                    if lo_v < 0:
                        lo_v, hi_v = 0, 1
                guard[0] = True
                try:
                    min_box.setValue(lo_v)
                    max_box.setValue(hi_v)
                finally:
                    guard[0] = False
            guard[0] = True
            try:
                layer.contrast_limits = (float(lo_v), float(hi_v))
            finally:
                guard[0] = False

        def _sync_from_layer(event=None, layer=layer, min_box=min_box, max_box=max_box, guard=guard):
            if guard[0]:
                return
            lo_v, hi_v = layer.contrast_limits
            guard[0] = True
            try:
                min_box.setValue(int(round(lo_v)))
                max_box.setValue(int(round(hi_v)))
            finally:
                guard[0] = False

        min_box.valueChanged.connect(lambda _v, c=_commit: c('min'))
        max_box.valueChanged.connect(lambda _v, c=_commit: c('max'))
        layer.events.contrast_limits.connect(_sync_from_layer)

    dock = viewer.window.add_dock_widget(panel, area='right', name='Contrast (manual)', tabify=False)
    # tabify explicitly with napari's built-in "layer controls" dock rather than
    # whatever else napari.window.add_dock_widget(tabify=True) would grab last
    # (e.g. the layer list) — puts our panel one click away as a sibling tab.
    qt_window = viewer.window._qt_window
    qt_window.tabifyDockWidget(viewer.window._qt_viewer.dockLayerControls, dock)
    dock.show()
    dock.raise_()


# ── CSV → shapes ──────────────────────────────────────────────────────────────

def _load_tile_csv_shapes(csv_path, z_range=None):
    """Load a tile-local detection CSV (1-indexed z) → (shapes, colors, meta).

    With z_range, boxes outside it are dropped and z becomes local to the range.
    """
    if not os.path.isfile(csv_path):
        return [], [], []
    df = pd.read_csv(csv_path)
    if df.empty:
        return [], [], []
    z = df['z'].astype(float).astype(int).to_numpy() - 1
    if z_range is not None:
        z = z - z_range[0]
        inside = (z >= 0) & (z < z_range[1] - z_range[0])
        df, z = df[inside], z[inside]
        if df.empty:
            return [], [], []
    x1, y1 = df['x1'].to_numpy(float), df['y1'].to_numpy(float)
    x2, y2 = df['x2'].to_numpy(float), df['y2'].to_numpy(float)
    means = df['mean'].to_numpy(float) if 'mean' in df.columns else np.zeros(len(df))
    classes = df['class'].to_numpy()
    arr = np.empty((len(df), 4, 3), dtype=np.float64)
    arr[:, 0] = np.column_stack([z, y1, x1])
    arr[:, 1] = np.column_stack([z, y1, x2])
    arr[:, 2] = np.column_stack([z, y2, x2])
    arr[:, 3] = np.column_stack([z, y2, x1])
    box_meta = [
        {'z': int(z[i]), 'x1': float(x1[i]), 'y1': float(y1[i]),
         'x2': float(x2[i]), 'y2': float(y2[i]), 'cls': str(classes[i]),
         'mean': float(means[i])}
        for i in range(len(df))
    ]
    return list(arr), [_color(c) for c in classes], box_meta


# ── False-negative annotation ─────────────────────────────────────────────────

def _append_false_negative(ann_path, tile_name, tile_path, z_abs, x, y,
                           channel='', crop_file=''):
    """Append one false-negative annotation to ann_path (CSV), creating header if new."""
    exists = os.path.isfile(ann_path)
    os.makedirs(os.path.dirname(os.path.abspath(ann_path)), exist_ok=True)
    with open(ann_path, 'a', encoding='utf-8', newline='') as f:
        if not exists:
            f.write("tile_name,tile_path,z,x,y,channel,crop_file,timestamp\n")
        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"{tile_name},{tile_path},{z_abs},{x:.0f},{y:.0f},"
                f"{channel},{crop_file},{ts}\n")


def _save_fn_crop(out_dir, tile_dir, z_file, x, y, size, channel):
    """Cut a size×size window centred on (x, y) out of the original slice at z_file.

    Saved as "<original slice name>_<channel>.tif", with a numeric suffix appended
    when that name is taken, so marking several cells on one slice keeps every crop.
    Reads the source TIFF unchanged so the crop keeps its original bit depth, and
    clamps the window inward instead of padding, so a full-size crop is all real
    pixels.  Returns (out_path, note); out_path is None when the crop could not
    be made and note then says why.
    """
    if not os.path.isdir(tile_dir):
        return None, f"tile dir not found: {tile_dir}"
    files = sorted(f for f in os.listdir(tile_dir)
                   if f.lower().endswith(('.tif', '.tiff')))
    if not 0 <= z_file < len(files):
        return None, f"z={z_file} outside slice range 0–{len(files) - 1}"
    src = os.path.join(tile_dir, files[z_file])
    img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, f"unreadable: {src}"
    h, w = img.shape[:2]
    x0 = int(min(max(round(x) - size // 2, 0), max(w - size, 0)))
    y0 = int(min(max(round(y) - size // 2, 0), max(h - size, 0)))
    crop = img[y0:y0 + size, x0:x0 + size]
    os.makedirs(out_dir, exist_ok=True)
    stem = f"{os.path.splitext(files[z_file])[0]}_{channel}"
    out_path = os.path.join(out_dir, f"{stem}.tif")
    dup = 2
    while os.path.exists(out_path):
        out_path = os.path.join(out_dir, f"{stem}_{dup}.tif")
        dup += 1
    if not cv2.imwrite(out_path, crop):
        return None, f"write failed: {out_path}"
    return out_path, f"{crop.shape[1]}×{crop.shape[0]} at ({x0},{y0}) from {files[z_file]}"


def _active_image_channel(viewer):
    """Return the channel id of the active [img] layer, or '' if none is active."""
    active = viewer.layers.selection.active
    if active is not None and active.name.startswith('[img] '):
        return active.name[len('[img] '):]
    if active is not None:
        for prefix in ('[raw] ', '[filtered] '):
            if active.name.startswith(prefix):
                channel = active.name[len(prefix):]
                if any(layer.name == f'[img] {channel}' for layer in viewer.layers):
                    return channel
    img_visible = [l for l in viewer.layers
                   if l.name.startswith('[img] ') and l.visible]
    if len(img_visible) == 1:
        return img_visible[0].name[len('[img] '):]
    return ''


def _make_fn_recorder(vis_cfg, paths, routing_config, anchor_dir, tile_path,
                      tile_name, z_range, offsets=None):
    """Build the Ctrl+click false-negative handler, or None when fn_ann_path is unset.

    Crops are cut from the original per-slice TIFFs, so in prealign mode the clicked
    position is un-shifted by the channel's alignment offset first.
    """
    ann_path = vis_cfg.get('fn_ann_path') or None
    if not ann_path:
        print("[info] Ctrl+click  disabled  (set 'fn_ann_path' in vis_config.json to enable)")
        return None
    crop_dir  = vis_cfg.get('fn_crop_dir') or None
    crop_size = int(vis_cfg.get('fn_crop_size', 256))
    offsets   = offsets or {}
    rel     = os.path.relpath(tile_path, anchor_dir)
    ch_dirs = {}
    for ch in routing_config:
        ch_dirs[ch['id']] = resolve_tile_dir(os.path.abspath(paths[ch['dir_key']]), rel)
        if ch.get('double_exposure'):
            ch_dirs[ch['second_intensity_id']] = resolve_tile_dir(
                os.path.abspath(paths[ch['second_intensity_dir_key']]), rel)
    print(f"[info] Ctrl+click  anywhere  → mark false-negative position → {ann_path}")
    if crop_dir:
        print(f"[info]            + save {crop_size}×{crop_size} TIFF crop → {crop_dir}")
    else:
        print(f"[info]            (set 'fn_crop_dir' to also save image crops)")

    def record(viewer, z_cur, x, y, box_registry):
        z_abs  = z_cur + z_range[0]
        ch_tag = _active_image_channel(viewer)
        crop_name, crop_note, crop_log = '', '', ''
        if crop_dir and not ch_tag:
            crop_note = "  (no crop — select a single [img] layer first)"
        elif crop_dir:
            o = offsets.get(ch_tag, {})
            # Displayed images use aligned coordinates (aligned = raw + shift), so subtract the shift to crop the original TIFF.
            z_f = z_abs - o.get('dz', 0)
            x_f = x - o.get('dx', 0)
            y_f = y - o.get('dy', 0)
            out_path, note = _save_fn_crop(crop_dir, ch_dirs.get(ch_tag, ''),
                                           z_f, x_f, y_f, crop_size, ch_tag)
            if out_path:
                crop_name = os.path.basename(out_path)
                crop_note = f"  → {crop_name}"
                crop_log  = f"[fn]   crop {note}"
            else:
                crop_note = "  (crop failed — see terminal)"
                crop_log  = f"[fn]   crop FAILED — {note}"
        _append_false_negative(ann_path, tile_name, tile_path, z_abs, x, y,
                               channel=ch_tag, crop_file=crop_name)
        hits = [b for b in box_registry
                if b['z'] == z_cur and b['x1'] <= x <= b['x2'] and b['y1'] <= y <= b['y2']]
        box_note = ""
        if hits:
            best_h = min(hits, key=lambda b: (b['x2'] - b['x1']) * (b['y2'] - b['y1']))
            box_note = f"  (inside [{best_h['layer_name']}] {best_h['cls']})"
        ch_note = f"  ch={ch_tag}" if ch_tag else ""
        print(f"[fn] {tile_name}  z={z_abs}  x={x:.0f}  y={y:.0f}{ch_note}{box_note}")
        if crop_log:
            print(crop_log)
        viewer.status = (f"⌃ FN marked: z={z_abs} x={x:.0f} y={y:.0f}"
                         f"{ch_note}{box_note}{crop_note}")

    return record


# ── Prealign-specific helpers ─────────────────────────────────────────────────

def load_offsets(align_dir, tile_name):
    json_path = os.path.join(align_dir, f"{tile_name}_offsets.json")
    if not os.path.isfile(json_path):
        return {}
    with open(json_path, encoding='utf-8') as f:
        return json.load(f)


# ── Dual-intensity helpers ─────────────────────────────────────────────────────

class InvalidImageContrastError(ValueError):
    """A selected image range has no usable intensity span for napari."""


def _check_image_contrast(climits, tile_name, channel_id, z_range):
    lo, hi = climits
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise InvalidImageContrastError(
            f"Tile {tile_name}, channel {channel_id}, Z {z_range[0]}–{z_range[1]}: "
            f"invalid image contrast limits {climits}. "
            "The selected image slices may be blank or have constant intensity."
        )


def _load_second_intensity_image(ch, paths, anchor_dir, tile_path, z_range,
                                 contrast_pct, offsets=None):
    """For a double_exposure channel, load its second-intensity volume as its own
    image layer (same colormap as the primary channel — same physical channel,
    just a different laser power). No-op for regular channels.

    offsets: per-channel shift dict for prealign or 2d, keyed by channel id — the pipeline
    already copies the primary channel's shift onto second_intensity_id when writing
    3_2d_aligned/*_offsets.json, so this needs no special-case resolution.
    Pass None in post mode (raw, unshifted images).
    Returns (second_id, volume, contrast_limits, display_id), or None.
    """
    if not ch.get('double_exposure'):
        return None
    second_id = ch['second_intensity_id']
    second_base = os.path.abspath(paths[ch['second_intensity_dir_key']])
    tile_dir = resolve_tile_dir(second_base, os.path.relpath(tile_path, anchor_dir))
    print(f"[img] Loading {second_id}  ({tile_dir})  [dual-intensity] ...")
    shift_info = (offsets.get(second_id) or offsets.get(ch['id'])) if offsets else None
    if shift_info is None:
        vol, climits = load_volume(tile_dir, z_range=z_range, contrast_pct=contrast_pct)
    else:
        shift = tuple(int(shift_info[key]) for key in ('dx', 'dy', 'dz'))
        vol, climits = load_frame_volume(tile_dir, z_range, shift, contrast_pct)
        if any(shift):
            print(f"  shifted ({shift[0]:+d}, {shift[1]:+d}, {shift[2]:+d})")
    if vol is None:
        print("  not found, skipping")
        return None
    _check_image_contrast(climits, os.path.basename(tile_path), second_id, z_range)
    print(f"  shape={vol.shape}  contrast_limits={climits}")
    return second_id, vol, climits, ch['id']


# ── Tile-local 2D view ────────────────────────────────────────────────────────────

def _run_2d(vis_cfg, paths, routing_config, tile_path, tile_name):
    """Show a saved 2D pipeline stage with images in the matching frame."""
    source = vis_cfg.get('2d_source', 'filtered')
    csv_dir, offsets = tile_2d_inputs(
        paths['pATHRESULT'], source, tile_name, [ch['id'] for ch in routing_config])
    print(f"[2d {source}] Reading {csv_dir}")

    z_range = _resolve_z_range(vis_cfg, tile_path)
    if z_range[1] <= z_range[0]:
        raise ValueError(f"No TIFF slices selected for tile '{tile_name}': {z_range}")
    canvas_shape = _canvas_shape_from_tile(tile_path, z_range)
    anchor_dir = os.path.abspath(paths[routing_config[0]['dir_key']])
    contrast_pct = tuple(vis_cfg.get('contrast_pct', [0.1, 99.9]))
    prepared_images = []
    if not vis_cfg.get('no_images', False):
        for ch in routing_config:
            cid = ch['id']
            ch_dir = os.path.abspath(paths[ch['dir_key']])
            tile_dir = resolve_tile_dir(ch_dir, os.path.relpath(tile_path, anchor_dir))
            print(f"[img] Loading {cid} ({tile_dir}) ...")
            shift = tuple(int(offsets[cid][key]) for key in ('dx', 'dy', 'dz'))
            vol, climits = load_frame_volume(tile_dir, z_range, shift, contrast_pct)
            if vol is None:
                print(f"  [img] {cid}: no TIFF slices found")
                continue
            _check_image_contrast(climits, tile_name, cid, z_range)
            if vol.shape != canvas_shape:
                raise ValueError(f"Image shape differs for {cid}: {vol.shape} vs {canvas_shape}")
            prepared_images.append((cid, vol, climits, cid))
            second = _load_second_intensity_image(
                ch, paths, anchor_dir, tile_path, z_range, contrast_pct,
                offsets=offsets)
            if second:
                second_id, second_vol, _, _ = second
                if second_vol.shape != canvas_shape:
                    raise ValueError(f"Image shape differs for {second_id}: {second_vol.shape} vs {canvas_shape}")
                prepared_images.append(second)

    viewer = napari.Viewer(title=f"2D {source} - {tile_name}")
    image_layers = []
    for cid, vol, climits, display_id in prepared_images:
        layer = viewer.add_image(vol, name=f"[img] {cid}",
                                 contrast_limits=climits, **_ch_vis(display_id))
        layer.contrast_limits_range = (0, 65535)
        image_layers.append((cid, layer))
    if image_layers:
        _add_contrast_panel(viewer, image_layers)
        _add_current_image_name(viewer, tile_path, z_range,
                                offsets[routing_config[0]['id']]['dz'])

    total = 0
    box_registry = []
    for ch in routing_config:
        cid = ch['id']
        if source == 'aligned_filtered':
            from src.core.resume import saved_channel_directory
            source_dir = saved_channel_directory(paths['pATHRESULT'], ch)
        else:
            source_dir = csv_dir
        csv_path = os.path.join(source_dir, f"{tile_name}_{cid}_result.csv")
        shapes, colors, box_meta = _load_tile_csv_shapes(csv_path, z_range)
        if not os.path.isfile(csv_path):
            print(f"[2d {source}] Missing {cid}: {csv_path}")
        if not shapes:
            continue
        # A Shapes layer avoids a full-volume labels array per channel.
        viewer.add_shapes(shapes, shape_type='rectangle', edge_color=colors,
                          face_color='transparent',
                          edge_width=vis_cfg.get('outline_width', 2),
                          name=f"[{source}] {cid}")
        box_registry.extend({**item, 'layer_name': f"[{source}] {cid}"}
                            for item in box_meta)
        total += len(shapes)
        print(f"[2d {source}] {cid}: {len(shapes)} boxes")
    print(f"[2d {source}] {total} boxes; Z {z_range[0]}-{z_range[1]}")

    fn_record = _make_fn_recorder(vis_cfg, paths, routing_config, anchor_dir,
                                  tile_path, tile_name, z_range, offsets=offsets)
    if fn_record is not None:
        @viewer.bind_key('Control')
        def _ctrl_hint(v):
            v.status = "Ctrl held - click anywhere to mark a false negative"
            yield
            v.status = ""

        def _on_click(v, event):
            if event.type != 'mouse_press':
                return
            try:
                modifiers = [str(mod).lower() for mod in event.modifiers]
            except (AttributeError, TypeError):
                return
            if not any('ctrl' in mod or 'control' in mod for mod in modifiers):
                return
            pos = v.cursor.position
            if len(pos) < 3:
                return
            fn_record(v, int(round(pos[0])), float(pos[2]), float(pos[1]),
                      box_registry)

        viewer.mouse_drag_callbacks.append(_on_click)
    _show_only_images_initially(viewer, vis_cfg)
    viewer.reset_view()
    add_video_export_panel(viewer, vis_cfg)


# ── Main ──────────────────────────────────────────────────────────────────────

def _configure_text_output():
    """Keep Windows console and GUI child-process logs UTF-8 compatible."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, 'reconfigure', None)
        if reconfigure is not None:
            try:
                reconfigure(encoding='utf-8', errors='backslashreplace')
            except (OSError, ValueError):
                pass



def _open_tiles_progressively(selected_tiles, open_tile, progress):
    """Show the first available tile before loading the remaining windows."""
    tiles = iter(selected_tiles)
    total = len(selected_tiles)
    opened = 0

    def open_next():
        nonlocal opened
        for tile_path, tile_name in tiles:
            print(f"\n{'=' * 70}\nTile: {tile_name}\n{'=' * 70}")
            try:
                open_tile(tile_path, tile_name)
            except InvalidImageContrastError as exc:
                print(f"[error] {exc}\nSkipping tile {tile_name}.")
                QApplication.instance() or QApplication(sys.argv)
                QMessageBox.critical(
                    None, "Cannot open tile",
                    f"{exc}\n\nThis tile was skipped. Other selected tiles will continue."
                )
                continue
            opened += 1
            progress(opened, total, tile_name)
            # Let Qt paint the completed window before opening another tile.
            QTimer.singleShot(100, open_next)
            return True
        return False

    if open_next():
        print("VIS_READY", flush=True)
        napari.run()

def main():
    _configure_text_output()
    parser = argparse.ArgumentParser(description='brain_detector unified visualizer.')
    parser.add_argument(
        '--config',
        default=os.path.join(project_root, 'config', 'vis_config.json'),
        help='Path to vis_config.json',
    )
    parser.add_argument('--mode', choices=('2d', 'prealign'),
                        help='Override mode from vis_config.json')
    parser.add_argument('--2d-source', dest='source_2d', choices=tuple(SOURCES_2D),
                        help='Select raw, filtered, aligned, or aligned_filtered pipeline outputs')
    parser.add_argument('--direct', action='store_true',
                        help='Use the original command-line tile selection instead of the launcher')
    args = parser.parse_args()

    if not (args.direct or args.mode or args.source_2d):
        from src.utils.visualize_gui import launch_gui
        launch_gui(args.config)
        return

    vis_cfg_path = args.config
    vis_cfg = load_config(vis_cfg_path) if os.path.isfile(vis_cfg_path) else {}

    samples = vis_cfg.get('samples')
    if samples:
        active_sample = vis_cfg.get('active_sample')
        if not active_sample:
            sys.exit(f"'samples' is defined, but 'active_sample' is not set. Options: {', '.join(samples)}")
        if active_sample not in samples:
            sys.exit(f"Unknown sample '{active_sample}'. Options: {', '.join(samples)}")
        vis_cfg = {**vis_cfg, **samples[active_sample]}
        print(f"Sample: {active_sample}")

    mode = args.mode or vis_cfg.get('mode', 'prealign')
    if args.source_2d:
        vis_cfg['2d_source'] = args.source_2d
    print(f"Mode: {mode}\n")

    if mode not in ('2d', 'prealign'):
        sys.exit(f"Unknown mode '{mode}'. Use '2d' or 'prealign' "
                 "(post-alignment mode was removed).")
    if mode == '2d':
        paths = vis_cfg.get('paths') or {}
        routing_config = [ch for ch in vis_cfg.get('channels_routing', [])
                          if ch.get('active', True)]
        if not paths.get('pATHRESULT') or not routing_config:
            sys.exit("2d mode requires paths.pATHRESULT and active channels_routing.")
        if vis_cfg.get('2d_source', 'filtered') not in SOURCES_2D:
            sys.exit(f"2d_source must be one of {', '.join(SOURCES_2D)}")
        context = None
    else:
        from src.utils.saved_view import saved_run_context
        try:
            saved_run, context = saved_run_context(vis_cfg)
        except (CoordinateContextError, OSError, ValueError) as exc:
            sys.exit(f"Saved run error: {exc}")
        paths = saved_run['paths']
        routing_config = saved_run['channels_routing']

    anchor_ch  = routing_config[0]
    anchor_dir = os.path.abspath(paths[anchor_ch['dir_key']])
    _, tile_paths = listTile(anchor_dir)
    selected_tiles = _select_tiles(vis_cfg, tile_paths)
    print(f"\n{len(selected_tiles)} tile(s) selected: "
          f"{', '.join(name for _, name in selected_tiles)}")

    def report_progress(current, total, tile_name):
        print(f"VIS_PROGRESS {current} {total} {tile_name}", flush=True)

    if mode == 'prealign' and vis_cfg.get('view_space', 'local') == 'global':
        from src.utils.saved_gui import run_saved_prealign
        run_saved_prealign(
            vis_cfg, paths, routing_config, context, selected_tiles,
            progress=report_progress)
        print("VIS_READY", flush=True)
        napari.run()
        return

    if mode == 'prealign':
        from src.utils.saved_gui import run_saved_prealign

        def open_tile(tile_path, tile_name):
            run_saved_prealign(
                vis_cfg, paths, routing_config, context,
                [(tile_path, tile_name)])
    else:
        def open_tile(tile_path, tile_name):
            _run_2d(vis_cfg, paths, routing_config, tile_path, tile_name)

    _open_tiles_progressively(selected_tiles, open_tile, report_progress)


if __name__ == '__main__':
    main()
