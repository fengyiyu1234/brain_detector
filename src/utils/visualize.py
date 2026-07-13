# -*- coding: utf-8 -*-
"""
Brain-detector unified single-tile visualizer.

Reads config/vis_config.json and launches napari in one of two modes:

  mode: "prealign"
    Pre-alignment QC for one tile.  Images are shifted by channel-alignment
    offsets; detection boxes come from 0_channel_alignment/ (shifted coords).
    Optional [raw] boxes show pre-alignment positions for comparison.
    [zlinked] and [coloc] are computed in-memory.

  mode: "post"
    Post-pipeline results for one tile.  Raw (unshifted) images.
    [s1] raw 2D detections, [s3] saved z-linked tracks from 3_channel_3d/,
    [s4] colocalization result.

Usage
-----
  python src/utils/visualize.py
  python src/utils/visualize.py --config config/vis_config.json
"""

import argparse
import datetime
import json
import os
import pickle
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np
import pandas as pd
import napari
from qtpy.QtWidgets import QWidget, QGridLayout, QLabel, QSpinBox

from src.config.loader import load_config
from src.utils.io import listTile, compute_grid_fallback_offsets
from src.core.z_linker import run_z_linker
from src.core.stitcher import (
    match_soma_3d_iou, annotate_soma_with_tf_containment,
    suppress_cross_class_overlap, _merge_class,
)


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

SPHERE_COLORS = {
    'RFP':   [1.0,  0.25, 0.25, 0.85],
    'GFP':   [0.25, 1.0,  0.25, 0.85],
    'Sox9':  [0.0,  0.85, 1.0,  0.85],
    'Olig2': [1.0,  0.2,  1.0,  0.85],
}
_DEFAULT_SPHERE_COLOR = [0.9, 0.9, 0.9, 0.7]
_REJECTED_COLOR       = [0.5, 0.5, 0.5, 0.6]   # gray — boxes removed by vis filter

_MARKER_RGB = {
    'rfp':   [1.0,  0.20, 0.20],
    'gfp':   [0.20, 1.0,  0.20],
    'sox9':  [0.0,  0.80, 1.0],
    'olig2': [1.0,  0.20, 1.0],
}


# ── Shared helpers ────────────────────────────────────────────────────────────

def _color(class_str):
    base = str(class_str).split('_')[0]
    return CLASS_COLOR.get(base, [1.0, 1.0, 1.0, 1.0])


def _get_ch_filter(filter_cfg, ch):
    """Look up vis filter by channel type ('soma'/'tf'), fall back to channel id."""
    return filter_cfg.get(ch.get('type', 'soma')) or filter_cfg.get(ch.get('id', '')) or None


def _extract_markers(class_str):
    return set(str(class_str).split('_')[1:])


def _ch_vis(ch_id):
    return CHANNEL_VIS.get(ch_id, _EXTRA_VIS.get(ch_id, DEFAULT_VIS))


def _marker_combo_color(markers):
    rgbs = [_MARKER_RGB.get(m.lower(), [0.7, 0.7, 0.7]) for m in markers]
    if not rgbs:
        rgb = [0.9, 0.9, 0.9]
    else:
        rgb = [sum(c[i] for c in rgbs) / len(rgbs) for i in range(3)]
        mx = max(rgb) if max(rgb) > 0 else 1.0
        rgb = [min(c / mx, 1.0) for c in rgb]
    return rgb + [1.0], [c * 0.65 for c in rgb] + [1.0]


def _auto_groups_from_classes(classes, outline_width=4, dash_size=8):
    """Build one group per unique multi-marker combination from an iterable of class strings.

    Single-marker classes are skipped (not colocalization).
    Uses superset matching (exact=False): a cell with GFP+RFP+Sox9 also appears
    in the GFP+RFP layer, matching the expected Venn-diagram colocalization view.
    Colors are auto-assigned by blending per-marker base colors in _MARKER_RGB.
    """
    seen = {}
    for cls in classes:
        markers = tuple(sorted(str(cls).split('_')[1:]))
        if len(markers) >= 2:
            seen[markers] = None
    groups = []
    for markers in sorted(seen.keys(), key=lambda m: (len(m), m)):
        ncol, gcol = _marker_combo_color(markers)
        groups.append({
            'name':          '+'.join(markers),
            'channels':      list(markers),
            'neuron_color':  ncol,
            'glia_color':    gcol,
            'outline_width': outline_width,
            'dash_size':     dash_size,
            'exact':         False,
        })
    return groups


def _read_tiff(fpath):
    img = cv2.imread(fpath, cv2.IMREAD_ANYDEPTH)
    if img is None:
        return None
    return img[:, :, 0].astype(np.float32) if img.ndim == 3 else img.astype(np.float32)


def load_volume(tile_dir, z_range=None, contrast_pct=(0.1, 99.9)):
    """Load a tile volume preserving original 16-bit intensity values (no normalization).

    Returns (vol, contrast_limits): vol is float32 with raw 16-bit values; contrast_limits
    is an (lo, hi) tuple from contrast_pct percentiles on a subsample, meant to be passed
    straight to napari's contrast_limits so the slider reads real 16-bit intensities instead
    of a normalized 0-1 range.
    """
    if not os.path.isdir(tile_dir):
        return None, None
    files = sorted(f for f in os.listdir(tile_dir)
                   if f.lower().endswith(('.tif', '.tiff')))
    if z_range:
        files = files[z_range[0]:z_range[1]]
    if not files:
        return None, None
    with ThreadPoolExecutor() as ex:
        slices = list(ex.map(_read_tiff, [os.path.join(tile_dir, f) for f in files]))
    slices = [s for s in slices if s is not None]
    if not slices:
        return None, None
    vol = np.stack(slices)
    sample = vol[::2, ::4, ::4]
    lo, hi = np.percentile(sample, contrast_pct[0]), np.percentile(sample, contrast_pct[1])
    return vol, (float(lo), float(hi))


def load_raw_volume(tile_dir, z_range=None):
    """Load tile images as float32 preserving original 16-bit values (no normalization)."""
    if not os.path.isdir(tile_dir):
        return None
    files = sorted(f for f in os.listdir(tile_dir)
                   if f.lower().endswith(('.tif', '.tiff')))
    if z_range:
        files = files[z_range[0]:z_range[1]]
    if not files:
        return None
    with ThreadPoolExecutor() as ex:
        slices = list(ex.map(_read_tiff, [os.path.join(tile_dir, f) for f in files]))
    slices = [s for s in slices if s is not None]
    return np.stack(slices) if slices else None


def _compute_box_means_from_csv(csv_path, z_range, raw_vol):
    """Compute mean 16-bit intensity for every box in csv from raw_vol (no filtering)."""
    if not os.path.isfile(csv_path) or raw_vol is None:
        return np.array([], dtype=np.float32)
    df = pd.read_csv(csv_path,
                     names=['slice_name', 'x1', 'y1', 'x2', 'y2', 'class', 'score', 'mean', 'z'],
                     skiprows=1)
    if df.empty:
        return np.array([], dtype=np.float32)
    df['z'] = df['z'].astype(float).astype(int) - 1
    if z_range is not None:
        df['z_local'] = df['z'] - z_range[0]
        df = df[(df['z_local'] >= 0) & (df['z_local'] < z_range[1] - z_range[0])]
        z_col_arr = df['z_local'].values
    else:
        z_col_arr = df['z'].values
    if df.empty:
        return np.array([], dtype=np.float32)
    x1v = df['x1'].values.astype(float); x2v = df['x2'].values.astype(float)
    y1v = df['y1'].values.astype(float); y2v = df['y2'].values.astype(float)
    Z_v, H_v, W_v = raw_vol.shape
    n = len(df)
    means = np.zeros(n, dtype=np.float32)
    for i in range(n):
        zi = int(z_col_arr[i])
        if 0 <= zi < Z_v:
            x1c = max(0, int(round(x1v[i]))); x2c = min(W_v, int(round(x2v[i])))
            y1c = max(0, int(round(y1v[i]))); y2c = min(H_v, int(round(y2v[i])))
            if y2c > y1c and x2c > x1c:
                means[i] = raw_vol[zi, y1c:y2c, x1c:x2c].mean()
    return means


def _compute_box_areas_from_csv(csv_path, z_range):
    """Compute 2D box area (w*h px²) for every box in csv (no filtering)."""
    if not os.path.isfile(csv_path):
        return np.array([], dtype=np.float32)
    df = pd.read_csv(csv_path,
                     names=['slice_name', 'x1', 'y1', 'x2', 'y2', 'class', 'score', 'mean', 'z'],
                     skiprows=1)
    if df.empty:
        return np.array([], dtype=np.float32)
    df['z'] = df['z'].astype(float).astype(int) - 1
    if z_range is not None:
        df = df[(df['z'] >= z_range[0]) & (df['z'] < z_range[1])]
    if df.empty:
        return np.array([], dtype=np.float32)
    w = df['x2'].values.astype(float) - df['x1'].values.astype(float)
    h = df['y2'].values.astype(float) - df['y1'].values.astype(float)
    return (w * h).astype(np.float32)


def _plot_box_area_histograms(ch_areas_dict, tile_name='', out_dir=None):
    """Save per-channel box area histograms as PNG to out_dir."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    channels = [cid for cid, a in ch_areas_dict.items() if len(a) > 0]
    if not channels:
        print("[area_hist] No area data to plot.")
        return
    n_ch = len(channels)
    fig, axes = plt.subplots(1, n_ch, figsize=(6 * n_ch, 4), squeeze=False)
    fig.suptitle(f"Box area distribution  —  {tile_name}")
    pct_marks  = [10, 50, 90]
    pct_colors = ['orange', 'steelblue', 'red']
    for ax, cid in zip(axes[0], channels):
        areas = ch_areas_dict[cid]
        ax.hist(areas, bins=120, color='steelblue', alpha=0.75, log=True)
        for pct, col in zip(pct_marks, pct_colors):
            v = float(np.percentile(areas, pct))
            ax.axvline(v, color=col, linestyle='--', linewidth=1.2,
                       label=f'p{pct} = {v:.0f}')
        ax.set_title(f"{cid}  (n={len(areas):,})")
        ax.set_xlabel('Box area (px²)')
        ax.set_ylabel('Box count (log scale)')
        ax.legend(fontsize=8)
    plt.tight_layout()
    save_dir = out_dir or '.'
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"box_area_hist_{tile_name}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[area_hist] Saved → {out_path}")


def _plot_intensity_histograms(ch_means_dict, tile_name='', out_dir=None):
    """Save per-channel box intensity histograms as PNG to out_dir."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    channels = [cid for cid, m in ch_means_dict.items() if len(m) > 0]
    if not channels:
        print("[hist] No intensity data to plot.")
        return
    n_ch = len(channels)
    fig, axes = plt.subplots(1, n_ch, figsize=(6 * n_ch, 4), squeeze=False)
    fig.suptitle(f"Box intensity distribution  —  {tile_name}")
    pct_marks  = [10, 20, 50]
    pct_colors = ['orange', 'red', 'steelblue']
    for ax, cid in zip(axes[0], channels):
        means = ch_means_dict[cid]
        ax.hist(means, bins=120, color='steelblue', alpha=0.75, log=True)
        for pct, col in zip(pct_marks, pct_colors):
            v = float(np.percentile(means, pct))
            ax.axvline(v, color=col, linestyle='--', linewidth=1.2,
                       label=f'p{pct} = {v:.0f}')
        ax.set_title(f"{cid}  (n={len(means):,})")
        ax.set_xlabel('Mean intensity (16-bit raw)')
        ax.set_ylabel('Box count (log scale)')
        ax.legend(fontsize=8)
    plt.tight_layout()
    save_dir = out_dir or '.'
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"intensity_hist_{tile_name}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[hist] Saved → {out_path}")


def _filter_df_by_size_and_intensity(x1, y1, x2, y2, z_col, raw_vol, filt):
    """Return (keep_mask, comp_means_or_None).

    Filter order (each stage operates on survivors of the previous):
      1. bbox_min / bbox_max  — absolute size bounds (width AND height)
      2. bbox_area_pct_min    — area percentile on size-surviving rows
      3. bbox_mean_pct_min / bbox_mean_min — intensity on area-surviving rows
    raw_vol: float32 (Z, H, W) with original 16-bit values, tile-local coords.
    """
    n = len(x1)
    keep = np.ones(n, dtype=bool)

    bbox_min = filt.get('bbox_min')
    bbox_max = filt.get('bbox_max')
    if bbox_min is not None:
        keep &= (x2 - x1) >= bbox_min
        keep &= (y2 - y1) >= bbox_min
    if bbox_max is not None:
        keep &= (x2 - x1) <= bbox_max
        keep &= (y2 - y1) <= bbox_max

    aspect_max = filt.get('bbox_max_aspect_ratio')
    if aspect_max is not None:
        w = x2 - x1
        h = y2 - y1
        keep &= np.maximum(w, h) <= aspect_max * np.maximum(np.minimum(w, h), 1e-6)

    area_pct_min = filt.get('bbox_area_pct_min')
    if area_pct_min is not None:
        areas = (x2 - x1) * (y2 - y1)
        surviving_areas = areas[keep]
        if surviving_areas.size > 0:
            keep &= areas >= float(np.percentile(surviving_areas, area_pct_min))

    pct_min = filt.get('bbox_mean_pct_min')
    abs_min = filt.get('bbox_mean_min')
    comp_means = None

    if raw_vol is not None and (pct_min is not None or abs_min is not None):
        Z_v, H_v, W_v = raw_vol.shape
        comp_means = np.zeros(n, dtype=np.float32)
        for i in range(n):
            zi = int(z_col[i])
            if 0 <= zi < Z_v:
                x1c = max(0, int(round(x1[i]))); x2c = min(W_v, int(round(x2[i])))
                y1c = max(0, int(round(y1[i]))); y2c = min(H_v, int(round(y2[i])))
                if y2c > y1c and x2c > x1c:
                    comp_means[i] = raw_vol[zi, y1c:y2c, x1c:x2c].mean()
        surviving = comp_means[keep]
        if pct_min is not None and surviving.size > 0:
            keep &= comp_means >= float(np.percentile(surviving, pct_min))
        if abs_min is not None:
            keep &= comp_means >= abs_min

    return keep, comp_means


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


# ── XML helpers ───────────────────────────────────────────────────────────────

def _get_tile_offset(channel_dir, tile_name, base_res=None, tile_paths=None):
    """Return (tile_x0, tile_y0, tile_z0) for coordinate conversion.

    Resolution order:
      1. xml_merging.xml / xml_import.xml (real TeraStitcher inter-tile offsets)
      2. grid fallback, replicating run_inference.py's pre_align behavior when
         no TeraStitcher XML exists — reads tile_size/overlap_pct from the run's
         own runtime_config.json (under base_res) so this always matches whatever
         offset was actually baked into coloc_result.csv / 3d_tracked.csv.
      3. (0, 0, 0)
    """
    for xml_name in ('xml_merging.xml', 'xml_import.xml'):
        xml_path = os.path.join(channel_dir, xml_name)
        if not os.path.isfile(xml_path):
            continue
        try:
            root   = ET.parse(xml_path).getroot()
            stacks = list(root.find('STACKS'))
            z_start = max(int(s.get('ABS_D', 0)) for s in stacks)
            x_min   = min(int(s.get('ABS_H', 0)) for s in stacks)
            y_min   = min(int(s.get('ABS_V', 0)) for s in stacks)
            for stack in stacks:
                if os.path.basename(stack.get('DIR_NAME', '')) == tile_name:
                    return (int(stack.get('ABS_H', 0)) - x_min,
                            int(stack.get('ABS_V', 0)) - y_min,
                            z_start - int(stack.get('ABS_D', 0)))
        except Exception:
            pass

    if base_res and tile_paths:
        try:
            with open(os.path.join(base_res, 'runtime_config.json'), encoding='utf-8') as f:
                runtime_cfg = json.load(f)
            if runtime_cfg.get('pipeline_mode') == 'pre_align':
                tile_size   = runtime_cfg.get('detection_params', {}).get('tILESIZE', 2048)
                overlap_pct = runtime_cfg.get('pre_align_params', {}).get('tile_overlap_pct', 15)
                xy_res_um   = runtime_cfg.get('detection_params', {}).get('xy_resolution_um', 0.65)
                dir_dict, disp_mat_fin = compute_grid_fallback_offsets(tile_paths, tile_size, overlap_pct, xy_res_um)
                if tile_name in dir_dict:
                    gi, gj = dir_dict[tile_name]
                    ax, ay, az = disp_mat_fin[gi, gj]
                    return int(ax), int(ay), int(az)
        except Exception:
            pass

    return 0, 0, 0


def _get_tile_overlap_margins(channel_dir, tile_name, canvas_w, canvas_h):
    xml_path = os.path.join(channel_dir, 'xml_merging.xml')
    if not os.path.isfile(xml_path):
        return 0, 0
    try:
        root   = ET.parse(xml_path).getroot()
        stacks = list(root.find('STACKS'))
        info   = {}
        for s in stacks:
            name = os.path.basename(s.get('DIR_NAME', ''))
            info[name] = {'row': int(s.get('ROW', 0)), 'col': int(s.get('COL', 0)),
                          'h': int(s.get('ABS_H', 0)), 'v': int(s.get('ABS_V', 0))}
        if tile_name not in info:
            return 0, 0
        cur = info[tile_name]
        left_margin = top_margin = 0
        for t in info.values():
            if t['row'] == cur['row'] and t['col'] == cur['col'] - 1:
                left_margin = max(0, t['h'] + canvas_w - cur['h'])
            if t['row'] == cur['row'] - 1 and t['col'] == cur['col']:
                top_margin  = max(0, t['v'] + canvas_h - cur['v'])
        return left_margin, top_margin
    except Exception:
        return 0, 0


def _add_margin_overlay(viewer, canvas_shape, left_m, top_m):
    """Add a semi-transparent red overlay covering the excluded left/top margin region."""
    if left_m <= 0 and top_m <= 0:
        return
    Z, H, W = canvas_shape
    mask = np.zeros((H, W), dtype=np.uint8)
    if left_m > 0:
        mask[:, :int(left_m)] = 1
    if top_m > 0:
        mask[:int(top_m), :] = 1
    # broadcast_to avoids copying the full Z×H×W volume in memory
    vol = np.broadcast_to(mask[np.newaxis], (Z, H, W))
    viewer.add_image(
        vol,
        name='[overlap] excluded margin',
        colormap='red',
        blending='additive',
        opacity=0.35,
        visible=True,
    )


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

def _load_tile_csv_shapes(csv_path, z_range=None, raw_vol=None, filt=None,
                           left_margin=0, top_margin=0):
    """Load a tile-local detection CSV (1-indexed z) → napari shapes.

    raw_vol: optional float32 (Z,H,W) with 16-bit values for intensity filtering.
    filt: optional dict — bbox_min, bbox_max, bbox_mean_pct_min, bbox_mean_min.
    Returns ((shapes, colors, meta), (rej_shapes, rej_colors, rej_meta)).
    rej_* holds boxes removed by filt; both lists are empty when filt is None.
    """
    _empty = ([], [], [])
    if not os.path.isfile(csv_path):
        return _empty, _empty
    df = pd.read_csv(
        csv_path,
        names=['slice_name', 'x1', 'y1', 'x2', 'y2', 'class', 'score', 'mean', 'z'],
        skiprows=1,
    )
    if df.empty:
        return _empty, _empty
    df['z'] = df['z'].astype(float).astype(int) - 1
    if z_range is not None:
        df['z_local'] = df['z'] - z_range[0]
        df = df[(df['z_local'] >= 0) & (df['z_local'] < z_range[1] - z_range[0])]
        if df.empty:
            return _empty, _empty
        z_col = df['z_local'].values
    else:
        z_col = df['z'].values
    n_before = len(df)
    y1 = df['y1'].values.astype(float); y2 = df['y2'].values.astype(float)
    x1 = df['x1'].values.astype(float); x2 = df['x2'].values.astype(float)
    if filt:
        keep, comp_means = _filter_df_by_size_and_intensity(
            x1, y1, x2, y2, z_col, raw_vol, filt)
        rej_idx = np.where(~keep)[0]
        idx     = np.where(keep)[0]
        # Build rejected shapes
        if len(rej_idx) > 0:
            rx1, ry1 = x1[rej_idx], y1[rej_idx]
            rx2, ry2 = x2[rej_idx], y2[rej_idx]
            rz = z_col[rej_idx]; rn = len(rej_idx)
            rarr = np.empty((rn, 4, 3), dtype=np.float64)
            rarr[:, 0] = np.column_stack([rz, ry1, rx1])
            rarr[:, 1] = np.column_stack([rz, ry1, rx2])
            rarr[:, 2] = np.column_stack([rz, ry2, rx2])
            rarr[:, 3] = np.column_stack([rz, ry2, rx1])
            rdf = df.iloc[rej_idx]; rcls = rdf['class'].values
            rmeans = comp_means[rej_idx] if comp_means is not None else rdf['mean'].values.astype(float)
            rej_meta = [
                {'z': int(rz[i]), 'x1': float(rx1[i]), 'y1': float(ry1[i]),
                 'x2': float(rx2[i]), 'y2': float(ry2[i]), 'cls': str(rcls[i]),
                 'mean': float(rmeans[i])}
                for i in range(rn)
            ]
            rej_data = (list(rarr), [_REJECTED_COLOR] * rn, rej_meta)
        else:
            rej_data = _empty
        print(f"  [filter] {n_before} → {len(idx)} kept, {len(rej_idx)} rejected")
        if not keep.any():
            return _empty, rej_data
        x1, y1, x2, y2 = x1[idx], y1[idx], x2[idx], y2[idx]
        z_col = z_col[idx]
        df    = df.iloc[idx]
        means = comp_means[idx] if comp_means is not None else df['mean'].values.astype(float)
    else:
        means    = df['mean'].values.astype(float) if 'mean' in df.columns else np.zeros(n_before)
        rej_data = _empty
    # Margin filter: hide boxes whose center falls in the left/top tile overlap region
    if left_margin > 0 or top_margin > 0:
        cx = (x1 + x2) / 2; cy = (y1 + y2) / 2
        m_keep = (cx >= left_margin) & (cy >= top_margin)
        x1, y1, x2, y2 = x1[m_keep], y1[m_keep], x2[m_keep], y2[m_keep]
        z_col = z_col[m_keep]; means = means[m_keep]
        df = df.iloc[np.where(m_keep)[0].tolist()]
    if len(df) == 0:
        return _empty, rej_data
    n = len(df)
    arr = np.empty((n, 4, 3), dtype=np.float64)
    arr[:, 0] = np.column_stack([z_col, y1, x1])
    arr[:, 1] = np.column_stack([z_col, y1, x2])
    arr[:, 2] = np.column_stack([z_col, y2, x2])
    arr[:, 3] = np.column_stack([z_col, y2, x1])
    classes = df['class'].values
    box_meta = [
        {'z': int(z_col[i]), 'x1': float(x1[i]), 'y1': float(y1[i]),
         'x2': float(x2[i]), 'y2': float(y2[i]), 'cls': str(classes[i]),
         'mean': float(means[i])}
        for i in range(n)
    ]
    return (list(arr), [_color(c) for c in classes], box_meta), rej_data


def _load_global_csv_to_tile_shapes(csv_path, tile_name, tile_x0, tile_y0, tile_z0,
                                     z_range, canvas_w, canvas_h,
                                     raw_vol=None, filt=None,
                                     left_margin=0, top_margin=0):
    """Load a global 3D CSV (absolute x,y; tile-local 1-indexed z) → tile-local shapes.

    Filters by tile_name column if present, otherwise by spatial bounding box.
    z convention: z_local = z_csv + tile_z0 - z_range[0] - 1
    raw_vol: optional float32 (Z,H,W) with 16-bit values for intensity filtering.
    filt: optional dict — bbox_min, bbox_max, bbox_mean_pct_min, bbox_mean_min.
    Returns ((shapes, colors, meta), (rej_shapes, rej_colors, rej_meta)).
    """
    _empty = ([], [], [])
    if not os.path.isfile(csv_path):
        return _empty, _empty
    df = pd.read_csv(csv_path)
    if df.empty:
        return _empty, _empty
    if 'tile_name' in df.columns:
        df = df[df['tile_name'] == tile_name]
    else:
        df = df[
            (df['x1'].astype(float) >= tile_x0) &
            (df['x2'].astype(float) <  tile_x0 + canvas_w) &
            (df['y1'].astype(float) >= tile_y0) &
            (df['y2'].astype(float) <  tile_y0 + canvas_h)
        ]
    if df.empty:
        return _empty, _empty
    z0 = z_range[0] if z_range is not None else 0
    z1 = z_range[1] if z_range is not None else float('inf')
    df = df.copy()
    df['z_local'] = df['z'].astype(float).astype(int) + tile_z0 - z0 - 1
    df = df[(df['z_local'] >= 0) & (df['z_local'] < z1 - z0)]
    if df.empty:
        return _empty, _empty
    n_before = len(df)
    y1 = df['y1'].values.astype(float) - tile_y0
    y2 = df['y2'].values.astype(float) - tile_y0
    x1 = df['x1'].values.astype(float) - tile_x0
    x2 = df['x2'].values.astype(float) - tile_x0
    z  = df['z_local'].values
    if filt:
        keep, comp_means = _filter_df_by_size_and_intensity(
            x1, y1, x2, y2, z, raw_vol, filt)
        rej_idx = np.where(~keep)[0]
        idx     = np.where(keep)[0]
        # Build rejected shapes
        if len(rej_idx) > 0:
            rx1, ry1 = x1[rej_idx], y1[rej_idx]
            rx2, ry2 = x2[rej_idx], y2[rej_idx]
            rz = z[rej_idx]; rn = len(rej_idx)
            rarr = np.empty((rn, 4, 3), dtype=np.float64)
            rarr[:, 0] = np.column_stack([rz, ry1, rx1])
            rarr[:, 1] = np.column_stack([rz, ry1, rx2])
            rarr[:, 2] = np.column_stack([rz, ry2, rx2])
            rarr[:, 3] = np.column_stack([rz, ry2, rx1])
            rdf = df.iloc[rej_idx]
            rcls = rdf['class'].values if 'class' in rdf.columns else ['unknown'] * rn
            rmeans = comp_means[rej_idx] if comp_means is not None else rdf['mean'].values.astype(float)
            rej_meta = [
                {'z': int(rz[i]), 'x1': float(rx1[i]), 'y1': float(ry1[i]),
                 'x2': float(rx2[i]), 'y2': float(ry2[i]), 'cls': str(rcls[i]),
                 'mean': float(rmeans[i])}
                for i in range(rn)
            ]
            rej_data = (list(rarr), [_REJECTED_COLOR] * rn, rej_meta)
        else:
            rej_data = _empty
        print(f"  [filter] {n_before} → {len(idx)} kept, {len(rej_idx)} rejected")
        if not keep.any():
            return _empty, rej_data
        x1, y1, x2, y2, z = x1[idx], y1[idx], x2[idx], y2[idx], z[idx]
        df    = df.iloc[idx]
        means = comp_means[idx] if comp_means is not None else df['mean'].values.astype(float)
    else:
        means    = df['mean'].values.astype(float) if 'mean' in df.columns else np.zeros(n_before)
        rej_data = _empty
    # Margin filter: hide boxes whose center falls in the left/top tile overlap region
    # x1/y1/x2/y2 are already tile-local (tile_x0/y0 subtracted above)
    if left_margin > 0 or top_margin > 0:
        cx = (x1 + x2) / 2; cy = (y1 + y2) / 2
        m_keep = (cx >= left_margin) & (cy >= top_margin)
        x1, y1, x2, y2 = x1[m_keep], y1[m_keep], x2[m_keep], y2[m_keep]
        z = z[m_keep]; means = means[m_keep]
        df = df.iloc[np.where(m_keep)[0].tolist()]
    if len(df) == 0:
        return _empty, rej_data
    n = len(df)
    arr = np.empty((n, 4, 3), dtype=np.float64)
    arr[:, 0] = np.column_stack([z, y1, x1])
    arr[:, 1] = np.column_stack([z, y1, x2])
    arr[:, 2] = np.column_stack([z, y2, x2])
    arr[:, 3] = np.column_stack([z, y2, x1])
    classes = df['class'].values if 'class' in df.columns else ['unknown'] * n
    box_meta = [
        {'z': int(z[i]), 'x1': float(x1[i]), 'y1': float(y1[i]),
         'x2': float(x2[i]), 'y2': float(y2[i]), 'cls': str(classes[i]),
         'mean': float(means[i])}
        for i in range(n)
    ]
    return (list(arr), [_color(c) for c in classes], box_meta), rej_data


# ── Coloc helpers ─────────────────────────────────────────────────────────────

def _auto_coloc_groups(coloc_csv, outline_width=4, dash_size=8):
    """Read coloc_result.csv and return one group per unique marker combination."""
    if not os.path.isfile(coloc_csv):
        return None
    df = pd.read_csv(coloc_csv)
    if df.empty or 'class' not in df.columns:
        return []
    return _auto_groups_from_classes(df['class'].dropna().unique(), outline_width, dash_size)


def _load_coloc_s4_shapes(coloc_csv, tile_name, z_range, s4_groups,
                           tile_x0=0, tile_y0=0, tile_z0=0,
                           canvas_w=None, canvas_h=None,
                           left_margin=0, top_margin=0):
    if not os.path.isfile(coloc_csv):
        return None
    df = pd.read_csv(coloc_csv)
    needed = {'x1', 'y1', 'x2', 'y2', 'class', 'z'}
    if df.empty or not needed.issubset(df.columns):
        return None
    # Primary filter: tile_name column
    if 'tile_name' in df.columns:
        df_tile = df[df['tile_name'] == tile_name]
    else:
        df_tile = pd.DataFrame()
    # Fallback: spatial bounding box filter when tile_name column is missing or unhelpful
    if df_tile.empty:
        if canvas_w is not None and canvas_h is not None:
            df_tile = df[
                (df['x1'].astype(float) >= tile_x0) &
                (df['x2'].astype(float) <  tile_x0 + canvas_w) &
                (df['y1'].astype(float) >= tile_y0) &
                (df['y2'].astype(float) <  tile_y0 + canvas_h)
            ]
            if df_tile.empty:
                print(f"  [coloc] no rows in tile bbox "
                      f"x=[{tile_x0},{tile_x0+canvas_w}) y=[{tile_y0},{tile_y0+canvas_h})")
                return []
            print(f"  [coloc] tile_name filter skipped (CSV has no usable tile_name); "
                  f"spatial fallback: {len(df_tile)} rows")
        else:
            print(f"  [coloc] no rows for tile_name='{tile_name}' and no canvas size for spatial filter")
            return []
    df = df_tile.copy()
    z0 = z_range[0] if z_range is not None else 0
    z1 = z_range[1] if z_range is not None else float('inf')
    df['z_local'] = df['z'].astype(float).astype(int) + tile_z0 - z0 - 1
    z_csv_min, z_csv_max = df['z'].min(), df['z'].max()
    df = df[(df['z_local'] >= 0) & (df['z_local'] < z1 - z0)]
    if df.empty:
        print(f"  [coloc] z-range [{z0}, {z1}) filtered out all {len(df_tile)} rows "
              f"(z_csv: {z_csv_min:.0f}–{z_csv_max:.0f}, "
              f"tile_z0={tile_z0}, z_local = z_csv+{tile_z0}-{z0}-1)")
        return []
    base_col = df['class'].apply(lambda c: str(c).split('_')[0])
    results = []
    for grp in s4_groups:
        req = frozenset(c.lower() for c in grp['channels'])
        neuron_color = grp.get('neuron_color', [1.0, 1.0, 0.0, 1.0])
        glia_color   = grp.get('glia_color',   [1.0, 0.65, 0.0, 1.0])
        dash_size    = grp.get('dash_size', 8)
        if grp.get('exact', False):
            mask = df['class'].apply(
                lambda c: frozenset(p.lower() for p in str(c).split('_')[1:]) == req)
        else:
            mask = df['class'].apply(
                lambda c: req.issubset(frozenset(p.lower() for p in str(c).split('_')[1:])))
        df_grp = df[mask]
        if df_grp.empty:
            results.append((grp, [], [], [], []))
            continue
        bc = base_col[df_grp.index]
        n  = len(df_grp)
        arr = np.empty((n, 4, 3), dtype=np.float64)
        z_col = df_grp['z_local'].values
        y1v = df_grp['y1'].values.astype(float) - tile_y0
        y2v = df_grp['y2'].values.astype(float) - tile_y0
        x1v = df_grp['x1'].values.astype(float) - tile_x0
        x2v = df_grp['x2'].values.astype(float) - tile_x0
        # Margin filter on tile-local coordinates
        if left_margin > 0 or top_margin > 0:
            cx = (x1v + x2v) / 2; cy = (y1v + y2v) / 2
            m_keep = (cx >= left_margin) & (cy >= top_margin)
            z_col  = z_col[m_keep]
            y1v, y2v = y1v[m_keep], y2v[m_keep]
            x1v, x2v = x1v[m_keep], x2v[m_keep]
            bc       = bc.iloc[np.where(m_keep)[0].tolist()]
            df_grp   = df_grp.iloc[np.where(m_keep)[0].tolist()]
        if len(df_grp) == 0:
            results.append((grp, [], [], [], []))
            continue
        n = len(df_grp)
        arr = np.empty((n, 4, 3), dtype=np.float64)
        arr[:, 0] = np.column_stack([z_col, y1v, x1v])
        arr[:, 1] = np.column_stack([z_col, y1v, x2v])
        arr[:, 2] = np.column_stack([z_col, y2v, x2v])
        arr[:, 3] = np.column_stack([z_col, y2v, x1v])
        colors     = [glia_color if b == 'glia' else neuron_color for b in bc]
        dash_sizes = [dash_size  if b == 'glia' else 0             for b in bc]
        cls_vals   = df_grp['class'].values
        meta_g = [
            {'z': int(z_col[i]), 'x1': float(x1v[i]), 'y1': float(y1v[i]),
             'x2': float(x2v[i]), 'y2': float(y2v[i]), 'cls': str(cls_vals[i])}
            for i in range(n)
        ]
        results.append((grp, list(arr), colors, dash_sizes, meta_g))
    return results


def _add_coloc_layers(viewer, coloc_csv, tile_name, z_range,
                       tile_x0, tile_y0, tile_z0, canvas_shape,
                       coloc_opacity, outline_width, box_registry=None,
                       left_margin=0, top_margin=0):
    """Auto-detect all marker combinations from coloc_result.csv and add layers."""
    groups = _auto_coloc_groups(coloc_csv, outline_width=outline_width, dash_size=8)
    if groups is None:
        print("[coloc] coloc_result.csv not found — skipping coloc layer.")
        return
    if not groups:
        print("[coloc] no marker combinations found in coloc_result.csv")
        return
    print(f"\n[coloc] Loaded from {coloc_csv}")
    print(f"  Auto-detected {len(groups)} combination(s): "
          + ", ".join(g['name'] for g in groups))
    _, canvas_h, canvas_w = canvas_shape
    grp_results = _load_coloc_s4_shapes(
        coloc_csv, tile_name, z_range, groups,
        tile_x0=tile_x0, tile_y0=tile_y0, tile_z0=tile_z0,
        canvas_w=canvas_w, canvas_h=canvas_h,
        left_margin=left_margin, top_margin=top_margin,
    )
    if not grp_results:
        return
    any_shown = False
    for grp, shapes_c, colors_c, dash_c, meta_c in grp_results:
        if shapes_c:
            layer_name = f"[coloc] {grp['name']}"
            _add_labels_layer(
                viewer, shapes_c, colors_c, canvas_shape,
                dash_sizes=dash_c,
                name=layer_name,
                visible=False, opacity=coloc_opacity,
                outline_width=grp.get('outline_width', outline_width),
            )
            if box_registry is not None:
                for m in meta_c:
                    box_registry.append({**m, 'layer_name': layer_name})
            n_solid = sum(1 for d in dash_c if d == 0)
            print(f"[coloc] {grp['name']}: {len(shapes_c)} boxes "
                  f"({n_solid} neuron, {len(dash_c) - n_solid} glia)")
            any_shown = True
    if not any_shown:
        print("[coloc] no cells in z-range for any group")


# ── Colocalization debug trace helpers ────────────────────────────────────────

def _build_merged_soma_vols(per_ch_vol_lists, soma_ch_ids, tf_ch_ids, zl_soma_cfg):
    """Replicate Phase A of run_inference: merge soma channels by 3D IoU.
    Returns (merged_soma_list, tf_vols_by_ch_dict).
    """
    import copy
    iou_thresh_3d = zl_soma_cfg.get('iou_thresh_3d', 0.15)
    z_pad_3d      = zl_soma_cfg.get('z_pad_3d', 2)
    cross_iou     = zl_soma_cfg.get('cross_class_iou_thresh', 0.5)
    first_id      = soma_ch_ids[0] if soma_ch_ids else ''
    merged = [dict(copy.copy(c), _soma_cid=first_id, _soma_peers=[])
              for c in per_ch_vol_lists.get(first_id, [])]
    for cid_b in soma_ch_ids[1:]:
        cells_b = [copy.copy(c) for c in per_ch_vol_lists.get(cid_b, [])]
        matched_pairs, _, unmatched_b = match_soma_3d_iou(
            merged, cells_b, iou_thresh=iou_thresh_3d, z_pad=z_pad_3d)
        for a_cell, b_cell in matched_pairs:
            a_cell['class'] = _merge_class(a_cell['class'], b_cell['class'])
            a_cell['_soma_peers'].append((cid_b, b_cell))
        merged = merged + [dict(c, _soma_cid=cid_b, _soma_peers=[]) for c in unmatched_b]
    merged = suppress_cross_class_overlap(merged, iou_thresh=cross_iou, z_pad=z_pad_3d)
    tf_vols_by_ch = {cid: per_ch_vol_lists.get(cid, []) for cid in tf_ch_ids}
    return merged, tf_vols_by_ch


def _trace_coloc_for_soma(soma_idx, soma_vol_list, tf_vol_list,
                           z_pad=2, xy_margin=0, max_center_dist_ratio=0.5):
    """Mirror annotate_soma_with_tf_containment for one soma and return per-TF breakdown.

    Categories in returned dict:
      matched     — TF assigned to this soma
      stolen      — TF passed all checks here but a closer soma won
      failed_xy   — TF bbox not fully inside soma XY bbox (records per-side overhang)
      failed_z    — TF z-span outside soma z-span (records margin on each side)
      failed_dist — TF bbox contained but centroid > max_center_dist_ratio * radius
      out_of_range — count of TF cells outside the global search sphere (max_radius*2)
    """
    from scipy.spatial import cKDTree

    soma       = soma_vol_list[soma_idx]
    soma_arr   = np.array([[s['cx'], s['cy'], s['cz']] for s in soma_vol_list], dtype=float)
    soma_radii = np.maximum(1.0, np.array([
        np.sqrt((s['x2_3d'] - s['x1_3d'])**2 + (s['y2_3d'] - s['y1_3d'])**2) / 2
        for s in soma_vol_list], dtype=float))
    max_radius = float(soma_radii.max())
    soma_r     = float(soma_radii[soma_idx])
    soma_ctr   = soma_arr[soma_idx]
    tree       = cKDTree(soma_arr)

    res = dict(soma=soma, soma_radius=soma_r, max_radius=max_radius,
               matched=[], stolen=[], failed_xy=[], failed_z=[], failed_dist=[],
               out_of_range=0)

    def _z_contained(s_z_min, s_z_max, tf_z_min, tf_z_max):
        # z_pad slices of tolerance on exactly one side only, never both at once
        # (padding both sides simultaneously lets a truncated box balloon into
        # an oversized capture window) — applies the same way to single- and
        # multi-layer soma boxes.
        return ((tf_z_min >= s_z_min - z_pad and tf_z_max <= s_z_max) or
                (tf_z_min >= s_z_min          and tf_z_max <= s_z_max + z_pad))

    for tf in tf_vol_list:
        tf_pt  = np.array([tf['cx'], tf['cy'], tf['cz']], dtype=float)
        d_this = float(np.linalg.norm(soma_ctr - tf_pt))

        if d_this > max_radius * 2:
            res['out_of_range'] += 1
            continue

        xy_ok = (soma['x1_3d'] - xy_margin <= tf['x1_3d'] and
                 tf['x2_3d']  <= soma['x2_3d'] + xy_margin and
                 soma['y1_3d'] - xy_margin <= tf['y1_3d'] and
                 tf['y2_3d']  <= soma['y2_3d'] + xy_margin)
        z_ok  = _z_contained(soma['z_min'], soma['z_max'], tf['z_min'], tf['z_max'])

        if not xy_ok:
            res['failed_xy'].append({
                'tf': tf, 'dist': d_this, 'z_ok': z_ok,
                'dx_left':  (soma['x1_3d'] - xy_margin) - tf['x1_3d'],
                'dx_right': tf['x2_3d'] - (soma['x2_3d'] + xy_margin),
                'dy_top':   (soma['y1_3d'] - xy_margin) - tf['y1_3d'],
                'dy_bot':   tf['y2_3d'] - (soma['y2_3d'] + xy_margin),
            })
            continue

        if not z_ok:
            # margin for reporting only — actual rule above is one-sided (OR), so
            # these two values are each that option's shortfall, not simultaneous
            res['failed_z'].append({
                'tf': tf, 'dist': d_this,
                'dz_min': (soma['z_min'] - z_pad) - tf['z_min'],
                'dz_max': tf['z_max'] - (soma['z_max'] + z_pad),
            })
            continue

        gate = max_center_dist_ratio * soma_r
        if d_this > gate:
            res['failed_dist'].append({'tf': tf, 'dist': d_this, 'gate': gate})
            continue

        # Passed all checks — find the globally closest winning soma for this TF
        best_d, best_s = float('inf'), None
        for s_idx in tree.query_ball_point(tf_pt, r=max_radius * 2):
            s2 = soma_vol_list[s_idx]
            if not (s2['x1_3d'] - xy_margin <= tf['x1_3d'] and
                    tf['x2_3d'] <= s2['x2_3d'] + xy_margin and
                    s2['y1_3d'] - xy_margin <= tf['y1_3d'] and
                    tf['y2_3d'] <= s2['y2_3d'] + xy_margin and
                    _z_contained(s2['z_min'], s2['z_max'], tf['z_min'], tf['z_max'])):
                continue
            d = float(np.linalg.norm(soma_arr[s_idx] - tf_pt))
            if d > max_center_dist_ratio * soma_radii[s_idx]:
                continue
            if d < best_d:
                best_d, best_s = d, s_idx

        if best_s == soma_idx:
            res['matched'].append({'tf': tf, 'dist': d_this})
        else:
            res['stolen'].append({
                'tf': tf, 'dist': d_this,
                'winner': soma_vol_list[best_s] if best_s is not None else None,
                'winner_dist': best_d,
            })
    return res


def _print_coloc_trace(res, coloc_params):
    soma = res['soma']
    sep  = '═' * 72
    print(f'\n{sep}')
    print(f"COLOC TRACE  —  soma class={soma['class']}")
    print(f"  bbox    x=[{soma['x1_3d']:.0f},{soma['x2_3d']:.0f}]  "
          f"y=[{soma['y1_3d']:.0f},{soma['y2_3d']:.0f}]  "
          f"z=[{soma['z_min']},{soma['z_max']}]")
    print(f"  center  ({soma['cx']:.1f}, {soma['cy']:.1f}, {soma['cz']:.1f})  "
          f"radius≈{res['soma_radius']:.1f}px  "
          f"search_r={res['max_radius']*2:.1f}px")
    print(f"  params  z_pad={coloc_params['z_pad']}  "
          f"xy_margin={coloc_params['xy_margin']}  "
          f"max_center_dist_ratio={coloc_params['max_center_dist_ratio']}")

    n_cand = (len(res['matched']) + len(res['stolen']) +
              len(res['failed_xy']) + len(res['failed_z']) + len(res['failed_dist']))
    print(f"\n  TF pool: {n_cand + res['out_of_range']} total  "
          f"→  {n_cand} within search radius  "
          f"({res['out_of_range']} out_of_range)")

    if res['matched']:
        print(f"\n  ✓ MATCHED ({len(res['matched'])} TF):")
        for it in res['matched']:
            tf = it['tf']
            print(f"      {tf['class']}  dist={it['dist']:.1f}px  "
                  f"z=[{tf['z_min']},{tf['z_max']}]  "
                  f"x=[{tf['x1_3d']:.0f},{tf['x2_3d']:.0f}]  "
                  f"y=[{tf['y1_3d']:.0f},{tf['y2_3d']:.0f}]")
    else:
        print(f"\n  ✗ NO MATCH — soma carries no TF annotation from this channel")

    def _show(items, limit, header, detail_fn):
        if not items:
            return
        print(f"\n  {header} ({len(items)}):")
        for it in sorted(items, key=lambda x: x['dist'])[:limit]:
            print(f"      {detail_fn(it)}")
        if len(items) > limit:
            print(f"      … and {len(items)-limit} more")

    _show(res['stolen'], 8,
          '⚠  STOLEN — passed checks but closer soma won',
          lambda it: (
              f"{it['tf']['class']}  dist_here={it['dist']:.1f}px  "
              f"winner_dist={it['winner_dist']:.1f}px"
              + (f"  winner={it['winner']['class']}"
                 f" @({it['winner']['cx']:.0f},{it['winner']['cy']:.0f},{it['winner']['cz']:.0f})"
                 if it['winner'] else '')))

    _show(res['failed_dist'], 5,
          '✗  FAILED center-dist gate (bbox contained but centroid too far)',
          lambda it: (
              f"{it['tf']['class']}  dist={it['dist']:.1f}px  gate={it['gate']:.1f}px  "
              f"z=[{it['tf']['z_min']},{it['tf']['z_max']}]  "
              f"x=[{it['tf']['x1_3d']:.0f},{it['tf']['x2_3d']:.0f}]  "
              f"y=[{it['tf']['y1_3d']:.0f},{it['tf']['y2_3d']:.0f}]"))

    _show(res['failed_z'], 5,
          '✗  FAILED Z containment',
          lambda it: (
              f"{it['tf']['class']}  dist={it['dist']:.1f}px  "
              f"tf_z=[{it['tf']['z_min']},{it['tf']['z_max']}]  "
              f"soma_z=[{res['soma']['z_min']},{res['soma']['z_max']}]"
              + (f"  starts {it['dz_min']:.0f}sl early" if it['dz_min'] > 0 else '')
              + (f"  ends {it['dz_max']:.0f}sl late"   if it['dz_max'] > 0 else '')))

    _show(res['failed_xy'], 5,
          '✗  FAILED XY containment',
          lambda it: (
              f"{it['tf']['class']}  dist={it['dist']:.1f}px  "
              f"[Z{'✓' if it['z_ok'] else '✗'}]  overhang px: "
              + str({k: f"{v:.0f}" for k, v in [
                  ('L', it['dx_left']),  ('R', it['dx_right']),
                  ('T', it['dy_top']),   ('B', it['dy_bot'])] if v > 0})))

    print(f'{sep}\n')


def _print_coloc_all_channels(res, soma_ch_ids, per_ch_vols):
    """Print _print_s3_cell_info for the soma channel(s) and each matched TF."""
    soma = res['soma']
    # Primary soma channel (stored directly in the merged soma entry)
    _print_s3_cell_info(soma, soma.get('_soma_cid', soma_ch_ids[0] if soma_ch_ids else '?'))
    # Peer soma channels matched via IoU (e.g. GFP matched to RFP)
    for peer_cid, peer_cell in soma.get('_soma_peers', []):
        _print_s3_cell_info(peer_cell, peer_cid)
    # Matched TF channels (e.g. Sox9 contained within soma)
    for it in res.get('matched', []):
        tf = it['tf']
        _print_s3_cell_info(tf, tf.get('_cid', tf.get('class', '?')))


def _find_soma_idx_for_box(box, z_range, soma_vol_list):
    """Match a box_registry entry (tile-local coords) to the nearest soma in vol_list."""
    if not soma_vol_list:
        return None
    z0  = z_range[0] if z_range else 0
    cx  = (box['x1'] + box['x2']) / 2
    cy  = (box['y1'] + box['y2']) / 2
    z_a = box['z'] + z0
    best_idx, best_d = None, float('inf')
    for idx, soma in enumerate(soma_vol_list):
        if abs(soma['cz'] - z_a) > 3:
            continue
        d = np.sqrt((soma['cx'] - cx) ** 2 + (soma['cy'] - cy) ** 2)
        if d < best_d:
            best_d, best_idx = d, idx
    return best_idx if best_d < 60 else None


def _find_vollist_entry_for_box(box, z_range, per_ch_vol_lists, layer_name):
    """Match a box_registry entry from a [s3]/[zlinked] layer to its vol_list dict.
    Returns (cell_dict, cid) or (None, None).
    """
    cid = None
    for prefix in ('[s3]', '[zlinked]'):
        if layer_name.startswith(prefix):
            cid = layer_name[len(prefix):].strip()
            break
    if cid is None:
        return None, None
    vols = per_ch_vol_lists.get(cid, [])
    if not vols:
        return None, None
    z0  = z_range[0] if z_range else 0
    cx  = (box['x1'] + box['x2']) / 2
    cy  = (box['y1'] + box['y2']) / 2
    z_a = box['z'] + z0
    best_entry, best_d = None, float('inf')
    for entry in vols:
        if abs(entry['cz'] - z_a) > 3:
            continue
        d = np.sqrt((entry['cx'] - cx) ** 2 + (entry['cy'] - cy) ** 2)
        if d < best_d:
            best_d, best_entry = d, entry
    return (best_entry, cid) if best_d < 60 else (None, None)


def _print_s3_cell_info(cell, cid):
    """Print all vol_list fields for a z-linked cell to terminal."""
    pz  = cell.get('per_z_boxes', {})
    sep = '─' * 64
    # Tile offsets present when cell was loaded via _load_s3_pkl_as_tile_local
    tx0 = cell.get('_tile_x0')
    ty0 = cell.get('_tile_y0')
    tz0 = cell.get('_tile_z0')
    has_offsets = tx0 is not None

    print(f'\n{sep}')
    print(f"S3 CELL INFO  —  channel={cid}  class={cell['class']}")
    print(f"  center (local)   ({cell['cx']:.1f}, {cell['cy']:.1f}, {cell['cz']:.1f})")
    if has_offsets:
        gcx = cell['cx'] + tx0
        gcy = cell['cy'] + ty0
        gcz = cell['cz'] + 1 - tz0
        print(f"  center (global)  ({gcx:.1f}, {gcy:.1f}, {gcz:.1f})")
    print(f"  3D bbox (local)  x=[{cell['x1_3d']:.0f},{cell['x2_3d']:.0f}]  "
          f"y=[{cell['y1_3d']:.0f},{cell['y2_3d']:.0f}]  "
          f"z=[{cell['z_min']},{cell['z_max']}]")
    if has_offsets:
        gx1 = cell['x1_3d'] + tx0;  gx2 = cell['x2_3d'] + tx0
        gy1 = cell['y1_3d'] + ty0;  gy2 = cell['y2_3d'] + ty0
        gz1 = cell['z_min'] + 1 - tz0;  gz2 = cell['z_max'] + 1 - tz0
        print(f"  3D bbox (global) x=[{gx1:.0f},{gx2:.0f}]  "
              f"y=[{gy1:.0f},{gy2:.0f}]  z=[{gz1},{gz2}]")
    n_layers = cell['z_max'] - cell['z_min'] + 1
    print(f"  z-span    {n_layers} layers  ({len(pz)} detections)")
    score_str = f"{cell['score']:.3f}" if cell.get('score') is not None else '?'
    mean_str  = f"{cell['mean']:.1f}"  if cell.get('mean')  is not None else '?'
    print(f"  score={score_str}  mean={mean_str}")
    if pz:
        hdr = "local_z" if not has_offsets else "local_z (global_z)"
        print(f"  per-z boxes  ({hdr} : [x1, y1, x2, y2]  w  h):")
        for z_loc in sorted(pz.keys()):
            b = pz[z_loc]
            if has_offsets:
                g_z = z_loc + 1 - tz0
                z_label = f"{z_loc:5d} ({g_z:5d})"
            else:
                z_label = f"{z_loc:5d}"
            print(f"    z={z_label}  [{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}]"
                  f"  w={b[2]-b[0]:.0f}  h={b[3]-b[1]:.0f}")
    print(f'{sep}\n')


def _append_false_negative(ann_path, tile_name, tile_path, z_abs, x, y, channel=''):
    """Append one false-negative annotation to ann_path (CSV), creating header if new."""
    exists = os.path.isfile(ann_path)
    with open(ann_path, 'a', encoding='utf-8', newline='') as f:
        if not exists:
            f.write("tile_name,tile_path,z,x,y,channel,timestamp\n")
        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"{tile_name},{tile_path},{z_abs},{x:.0f},{y:.0f},{channel},{ts}\n")


def _active_image_channel(viewer):
    """Return the channel id of the active [img] layer, or '' if none is active."""
    active = viewer.layers.selection.active
    if active is not None and active.name.startswith('[img] '):
        return active.name[len('[img] '):]
    img_visible = [l for l in viewer.layers
                   if l.name.startswith('[img] ') and l.visible]
    if len(img_visible) == 1:
        return img_visible[0].name[len('[img] '):]
    return ''


def _build_s3_span_shapes(cell, z_range, canvas_shape):
    """Build one yellow rectangle per z from cell['per_z_boxes'] for _add_labels_layer."""
    pz  = cell.get('per_z_boxes', {})
    z0  = z_range[0] if z_range else 0
    Z   = canvas_shape[0]
    shapes, colors = [], []
    for z_abs in sorted(pz.keys()):
        z_l = z_abs - z0
        if z_l < 0 or z_l >= Z:
            continue
        b = pz[z_abs]
        x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        shapes.append(np.array(
            [[z_l, y1, x1], [z_l, y1, x2], [z_l, y2, x2], [z_l, y2, x1]],
            dtype=np.float64))
        colors.append([1.0, 1.0, 0.0, 1.0])
    return shapes, colors


def _load_s3_pkl_as_tile_local(pkl_path, tile_x0, tile_y0, tile_z0=0):
    """Load a _3d_tracked.pkl and convert global → tile-local coordinates.

    The pipeline stores global x/y (stitching adds ABS_X/ABS_Y) and global z
    (= local_z - tile_z0, 1-indexed for the reference tile).  Passing tile_z0
    (= z_start - ABS_Z for this tile, from _get_tile_offset) converts the z values
    back to 0-indexed tile-local, matching _run_zlink_for_csv output so that
    _find_vollist_entry_for_box / _build_s3_span_shapes work correctly.
    Returns None if the pkl doesn't exist.
    """
    if not os.path.isfile(pkl_path):
        return None
    with open(pkl_path, 'rb') as pf:
        raw_vols = pickle.load(pf)
    tile_vols = []
    for c in raw_vols:
        tc = dict(c)
        tc['cx']    = c['cx']    - tile_x0
        tc['cy']    = c['cy']    - tile_y0
        tc['x1_3d'] = c['x1_3d'] - tile_x0
        tc['x2_3d'] = c['x2_3d'] - tile_x0
        tc['y1_3d'] = c['y1_3d'] - tile_y0
        tc['y2_3d'] = c['y2_3d'] - tile_y0
        tc['cz']    = int(c['cz'])    - 1 + tile_z0   # global_z → 0-indexed local
        tc['z_min'] = int(c['z_min']) - 1 + tile_z0
        tc['z_max'] = int(c['z_max']) - 1 + tile_z0
        pzb = {}
        for z_k, bv in c.get('per_z_boxes', {}).items():
            pzb[int(z_k) - 1 + tile_z0] = [bv[0] - tile_x0, bv[1] - tile_y0,
                                             bv[2] - tile_x0, bv[3] - tile_y0]
        tc['per_z_boxes'] = pzb
        # Store tile offsets so _print_s3_cell_info can reconstruct global coords
        tc['_tile_x0'] = tile_x0
        tc['_tile_y0'] = tile_y0
        tc['_tile_z0'] = tile_z0
        tile_vols.append(tc)
    return tile_vols


# ── Prealign-specific helpers ─────────────────────────────────────────────────

def _shift_volume(vol, dx, dy, dz):
    if dx == 0 and dy == 0 and dz == 0:
        return vol
    Z, H, W = vol.shape
    out = np.zeros_like(vol)
    def _sd(shift, size):
        return ((slice(0, size - shift), slice(shift, size)) if shift >= 0
                else (slice(-shift, size), slice(0, size + shift)))
    sz_s, sz_d = _sd(dz, Z)
    sy_s, sy_d = _sd(dy, H)
    sx_s, sx_d = _sd(dx, W)
    out[sz_d, sy_d, sx_d] = vol[sz_s, sy_s, sx_s]
    return out


def load_offsets(align_dir, tile_name):
    json_path = os.path.join(align_dir, f"{tile_name}_offsets.json")
    if not os.path.isfile(json_path):
        return {}
    with open(json_path, encoding='utf-8') as f:
        return json.load(f)


def _run_zlink_for_csv(csv_path, z_range, iou_thresh, min_z_layers, max_cell_z_span,
                       max_z_gap=0, ch_id=None):
    """ch_id: if given, tagged onto each row's class ("neuron" -> "neuron_RFP"), mirroring
    run_inference.py's channel tagging so cross-channel class strings carry markers that
    _merge_class / annotate_soma_with_tf_containment / _auto_groups_from_classes expect."""
    if not os.path.isfile(csv_path):
        return []
    df = pd.read_csv(
        csv_path,
        names=['slice_name', 'x1', 'y1', 'x2', 'y2', 'class', 'score', 'mean', 'z'],
        skiprows=1,
    )
    if df.empty:
        return []
    df['z'] = df['z'].astype(float).astype(int) - 1
    if z_range is not None:
        df = df[(df['z'] >= z_range[0]) & (df['z'] < z_range[1])]
    if df.empty:
        return []
    if ch_id is not None:
        df['class'] = df['class'].astype(str) + '_' + ch_id
    matrix = df[['x1', 'y1', 'x2', 'y2', 'score', 'mean', 'class', 'z']].values
    _, vol_list = run_z_linker(
        matrix,
        iou_thresh=iou_thresh,
        min_z_layers=min_z_layers,
        max_cell_z_span=max_cell_z_span,
        max_z_gap=max_z_gap,
    )
    return vol_list


def _vol_list_to_center_z_shapes(vol_list, z_range):
    z0 = z_range[0] if z_range is not None else 0
    z1 = z_range[1] if z_range is not None else float('inf')
    shapes, colors, box_meta = [], [], []
    for cell in vol_list:
        cz = int(round(cell['cz']))
        if cz < z0 or cz >= z1:
            continue
        z_local = cz - z0
        per_z = cell.get('per_z_boxes', {})
        box = per_z.get(cz) or [
            cell.get('x1_3d', 0), cell.get('y1_3d', 0),
            cell.get('x2_3d', 0), cell.get('y2_3d', 0),
        ]
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        shapes.append(np.array([
            [z_local, y1, x1], [z_local, y1, x2],
            [z_local, y2, x2], [z_local, y2, x1],
        ], dtype=np.float64))
        cls_str = str(cell.get('class', 'neuron'))
        colors.append(_color(cls_str))
        box_meta.append({'z': z_local, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'cls': cls_str})
    return shapes, colors, box_meta


def _vol_list_to_s4_shapes(vol_list, z_range, s4_groups):
    z0 = z_range[0] if z_range is not None else 0
    z1 = z_range[1] if z_range is not None else float('inf')
    results = []
    for grp in s4_groups:
        req = frozenset(c.lower() for c in grp['channels'])
        neuron_color = grp.get('neuron_color', [1.0, 1.0, 0.0, 1.0])
        glia_color   = grp.get('glia_color',   [1.0, 0.65, 0.0, 1.0])
        dash_size    = grp.get('dash_size', 8)
        shapes, colors, dash_sizes, box_meta = [], [], [], []
        for cell in vol_list:
            parts        = str(cell.get('class', '')).split('_')
            base         = parts[0]
            cell_markers = frozenset(p.lower() for p in parts[1:])
            if not req.issubset(cell_markers):
                continue
            cz = int(round(cell['cz']))
            if cz < z0 or cz >= z1:
                continue
            z_local = cz - z0
            per_z = cell.get('per_z_boxes', {})
            box = per_z.get(cz) or [
                cell.get('x1_3d', 0), cell.get('y1_3d', 0),
                cell.get('x2_3d', 0), cell.get('y2_3d', 0),
            ]
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            shapes.append(np.array([
                [z_local, y1, x1], [z_local, y1, x2],
                [z_local, y2, x2], [z_local, y2, x1],
            ], dtype=np.float64))
            colors.append(glia_color if base == 'glia' else neuron_color)
            dash_sizes.append(dash_size if base == 'glia' else 0)
            box_meta.append({'z': z_local, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                             'cls': str(cell.get('class', ''))})
        results.append((grp, shapes, colors, dash_sizes, box_meta))
    return results


def _vol_list_to_points(vol_list, z_range):
    z0 = z_range[0] if z_range is not None else 0
    z1 = z_range[1] if z_range is not None else float('inf')
    centers, sizes = [], []
    for cell in vol_list:
        cz = cell['cz']
        if cz < z0 or cz >= z1:
            continue
        diam = max(cell['x2_3d'] - cell['x1_3d'], cell['y2_3d'] - cell['y1_3d'])
        centers.append([cz - z0, cell['cy'], cell['cx']])
        sizes.append(max(diam, 1.0))
    if not centers:
        return None, None
    return np.array(centers, dtype=float), np.array(sizes, dtype=float)


def _run_tile_colocalization(per_ch_vol_lists, soma_ch_ids, tf_ch_ids,
                              zl_soma_cfg, zl_tf_cfg):
    import copy
    iou_thresh_3d = zl_soma_cfg.get('iou_thresh_3d', 0.15)
    z_pad_3d      = zl_soma_cfg.get('z_pad_3d', 2)
    cross_iou     = zl_soma_cfg.get('cross_class_iou_thresh', 0.5)
    xy_margin       = zl_tf_cfg.get('containment_xy_margin', 0)
    z_pad_tf        = zl_tf_cfg.get('containment_z_pad', 2)
    max_center_dist = zl_tf_cfg.get('max_center_dist_ratio', 0.5)
    if not soma_ch_ids:
        return []
    merged = [copy.copy(c) for c in per_ch_vol_lists.get(soma_ch_ids[0], [])]
    for cid_b in soma_ch_ids[1:]:
        cells_b = [copy.copy(c) for c in per_ch_vol_lists.get(cid_b, [])]
        matched_pairs, _, unmatched_b = match_soma_3d_iou(
            merged, cells_b, iou_thresh=iou_thresh_3d, z_pad=z_pad_3d)
        for a_cell, b_cell in matched_pairs:
            a_cell['class'] = _merge_class(a_cell['class'], b_cell['class'])
        merged = merged + unmatched_b
    merged = suppress_cross_class_overlap(merged, iou_thresh=cross_iou, z_pad=z_pad_3d)
    for cid in tf_ch_ids:
        tf_vols = per_ch_vol_lists.get(cid, [])
        if tf_vols and merged:
            merged = annotate_soma_with_tf_containment(
                merged, tf_vols, z_pad=z_pad_tf, xy_margin=xy_margin,
                max_center_dist_ratio=max_center_dist)
    return merged


# ── Dual-intensity helpers ─────────────────────────────────────────────────────

def _add_second_intensity_layer(viewer, ch, paths, anchor_dir, tile_path, z_range,
                                 contrast_pct, offsets=None):
    """For a double_exposure channel, load its second-intensity volume as its own
    image layer (same colormap as the primary channel — same physical channel,
    just a different laser power). No-op for regular channels.

    offsets: prealign's per-channel shift dict, keyed by channel id — the pipeline
    already copies the primary channel's shift onto second_intensity_id when writing
    0_channel_alignment/*_offsets.json, so this needs no special-case resolution.
    Pass None in post mode (raw, unshifted images).
    Returns (second_id, layer), or None if nothing was loaded.
    """
    if not ch.get('double_exposure'):
        return None
    second_id = ch['second_intensity_id']
    second_base = os.path.abspath(paths[ch['second_intensity_dir_key']])
    tile_dir = os.path.join(second_base, os.path.relpath(tile_path, anchor_dir))
    print(f"[img] Loading {second_id}  ({tile_dir})  [dual-intensity] ...")
    vol, climits = load_volume(tile_dir, z_range=z_range, contrast_pct=contrast_pct)
    if vol is None:
        print("  → not found, skipping")
        return None
    if offsets and second_id in offsets:
        o = offsets[second_id]
        dx, dy, dz = o['dx'], o['dy'], o['dz']
        if dx != 0 or dy != 0 or dz != 0:
            vol = _shift_volume(vol, dx, dy, dz)
            print(f"  → shifted ({dx:+d}, {dy:+d}, {dz:+d})")
    layer = viewer.add_image(vol, name=f"[img] {second_id}", contrast_limits=climits,
                              **_ch_vis(ch['id']))
    layer.contrast_limits_range = (0, 65535)
    print(f"  → {vol.shape}  contrast_limits={climits}")
    return second_id, layer


def _channel_2d_csv(ch, tile_name, primary_dir, fallback_dir, fused_dir,
                     use_fallback=True):
    """Resolve the tile-local 2D-detection CSV path for a channel.

    double_exposure channels always read the fused result (1_tile_2d_fused) —
    the pre-fusion per-exposure CSV under primary_dir/fallback_dir would only
    hold that one exposure's detections, not the merged channel. Regular
    channels keep the existing primary_dir -> fallback_dir behavior.
    use_fallback: whether to fall back to fallback_dir when the file isn't at
    primary_dir (mirrors the callers' own has_aligned / os.path.exists checks).
    """
    cid = ch['id']
    if ch.get('double_exposure'):
        return os.path.join(fused_dir, f"{tile_name}_{cid}_result.csv")
    csv_path = os.path.join(primary_dir, f"{tile_name}_{cid}_result.csv")
    if not os.path.isfile(csv_path) and use_fallback:
        csv_path = os.path.join(fallback_dir, f"{tile_name}_{cid}_result.csv")
    return csv_path


# ── Mode: prealign ────────────────────────────────────────────────────────────

def _run_prealign(vis_cfg, paths, routing_config, tile_path, tile_name):
    zl_cfg  = vis_cfg.get('z_linker', {})
    zl_soma = zl_cfg.get('soma', {})
    zl_tf   = zl_cfg.get('tf',   {})
    soma_ch_ids = [ch['id'] for ch in routing_config if ch.get('type', 'soma') == 'soma']
    tf_ch_ids   = [ch['id'] for ch in routing_config if ch.get('type') == 'tf']

    contrast_pct = tuple(vis_cfg.get('contrast_pct', [0.1, 99.9]))
    no_images    = vis_cfg.get('no_images',     False)
    show_before  = vis_cfg.get('show_before',   False)
    show_spheres = vis_cfg.get('spheres',       False)
    show_zlinked = vis_cfg.get('show_zlinked',  False)
    show_coloc   = vis_cfg.get('show_coloc',    False)
    outline_width      = vis_cfg.get('outline_width',      2)
    aligned_opacity    = vis_cfg.get('aligned_opacity',    0.8)
    raw_opacity        = vis_cfg.get('raw_opacity',        0.6)
    zlinked_opacity    = vis_cfg.get('zlinked_opacity',    0.7)
    coloc_opacity      = vis_cfg.get('coloc_opacity',      0.9)
    sphere_opacity     = vis_cfg.get('sphere_opacity',     0.6)
    sphere_raw_opacity = vis_cfg.get('sphere_raw_opacity', 0.3)
    sphere_colors_cfg  = vis_cfg.get('sphere_colors',      {})

    base_res     = paths['pATHRESULT']
    align_dir    = os.path.join(base_res, '0_channel_alignment')
    raw_dir      = os.path.join(base_res, '1_tile_2d_raw')
    fused_dir    = os.path.join(base_res, '1_tile_2d_fused')
    filtered_dir = os.path.join(base_res, '1_tile_2d_filtered')

    anchor_ch  = routing_config[0]
    anchor_dir = os.path.abspath(paths[anchor_ch['dir_key']])
    _, tile_paths = listTile(anchor_dir)

    print(f"\nTile: {tile_name}")

    z_range      = _resolve_z_range(vis_cfg, tile_path)
    canvas_shape = _canvas_shape_from_tile(tile_path, z_range)
    canvas_z, canvas_h, canvas_w = canvas_shape
    print(f"Z-range: {z_range[0]} – {z_range[1]} ({z_range[1] - z_range[0]} slices)\n")

    offsets     = load_offsets(align_dir, tile_name)
    has_aligned = bool(offsets)

    print("=== Channel alignment summary ===")
    for ch in routing_config:
        cid = ch['id']
        if cid in offsets:
            o   = offsets[cid]
            tag = " [REFERENCE]" if (o['dx'] == 0 and o['dy'] == 0
                                      and o['dz'] == 0 and o['iou_score'] >= 1.0) else ""
            print(f"  [{cid:8s}]  shift=({o['dx']:+d}, {o['dy']:+d}, {o['dz']:+d})"
                  f"   iou={o['iou_score']:.4f}{tag}")
        else:
            print(f"  [{cid:8s}]  (no offset data)")
    print("=================================\n")

    viewer = napari.Viewer(title=f"Pre-Align QC — {tile_name}")

    # ── Z-linking ─────────────────────────────────────────────────────────────
    need_zlink = show_spheres or show_zlinked or show_coloc
    per_ch_vol_lists = {}
    if need_zlink:
        print("Running z-linking in memory...")
        for ch in routing_config:
            cid   = ch['id']
            ctype = ch.get('type', 'soma')
            filtered_csv = os.path.join(filtered_dir, f"{tile_name}_{cid}_result.csv")
            zl_p = zl_soma if ctype == 'soma' else zl_tf
            vol_list = _run_zlink_for_csv(
                filtered_csv, z_range,
                iou_thresh=zl_p.get('iou_thresh', 0.35),
                min_z_layers=zl_p.get('min_z_layers', 1),
                max_cell_z_span=zl_p.get('max_cell_z_span', 5),
                max_z_gap=zl_p.get('max_z_gap', 0),
                ch_id=cid,
            )
            per_ch_vol_lists[cid] = vol_list
            print(f"  [{cid}] z-linked: {len(vol_list)} cells")
        print()

    # ── Build merged soma/TF vol lists for Shift+click coloc trace ────────────
    _debug_soma_vols = []
    _debug_all_tf    = []
    _debug_cp        = {}
    if per_ch_vol_lists:
        _debug_soma_vols, _debug_tf_dict = _build_merged_soma_vols(
            per_ch_vol_lists, soma_ch_ids, tf_ch_ids, zl_soma)
        _debug_all_tf = [dict(c, _cid=_tf_cid)
                         for _tf_cid, _vols in _debug_tf_dict.items()
                         for c in _vols]
        _debug_cp = {
            'z_pad':                 zl_tf.get('containment_z_pad', 2),
            'xy_margin':             zl_tf.get('containment_xy_margin', 0),
            'max_center_dist_ratio': zl_tf.get('max_center_dist_ratio', 0.5),
        }
        print(f"[trace] Shift+click any [coloc] box to trace.  "
              f"({len(_debug_soma_vols)} soma, {len(_debug_all_tf)} TF)")

    # ── Image layers ──────────────────────────────────────────────────────────
    if not no_images:
        img_layers = []
        for ch in routing_config:
            cid      = ch['id']
            ch_base  = os.path.abspath(paths[ch['dir_key']])
            tile_dir = os.path.join(ch_base, os.path.relpath(tile_path, anchor_dir))
            print(f"[img] Loading {cid}  ({tile_dir}) ...")
            vol, climits = load_volume(tile_dir, z_range=z_range, contrast_pct=contrast_pct)
            if vol is None:
                print("  → not found, skipping")
                continue
            canvas_shape = vol.shape
            if cid in offsets:
                o = offsets[cid]
                dx, dy, dz = o['dx'], o['dy'], o['dz']
                if dx != 0 or dy != 0 or dz != 0:
                    vol = _shift_volume(vol, dx, dy, dz)
                    print(f"  → shifted ({dx:+d}, {dy:+d}, {dz:+d})")
            img_layer = viewer.add_image(vol, name=f"[img] {cid}", contrast_limits=climits,
                                          **_ch_vis(cid))
            img_layer.contrast_limits_range = (0, 65535)
            img_layers.append((cid, img_layer))
            print(f"  → {vol.shape}  contrast_limits={climits}")

            second = _add_second_intensity_layer(
                viewer, ch, paths, anchor_dir, tile_path, z_range, contrast_pct,
                offsets=offsets)
            if second:
                img_layers.append(second)
        _add_contrast_panel(viewer, img_layers)

    box_registry = []  # {z, x1, y1, x2, y2, cls, layer_name} — populated below

    # ── Load raw volumes for intensity filtering / histogram (prealign: apply shift) ─
    filter_cfg = vis_cfg.get('filter', {})
    show_hist  = vis_cfg.get('show_intensity_hist', False)
    raw_vols   = {}
    for ch in routing_config:
        cid = ch['id']
        need_vol = bool(_get_ch_filter(filter_cfg, ch)) or show_hist
        if not need_vol:
            continue
        ch_base  = os.path.abspath(paths[ch['dir_key']])
        tile_dir = os.path.join(ch_base, os.path.relpath(tile_path, anchor_dir))
        rv = load_raw_volume(tile_dir, z_range)
        if rv is not None and cid in offsets:
            o = offsets[cid]
            if o['dx'] or o['dy'] or o['dz']:
                rv = _shift_volume(rv, o['dx'], o['dy'], o['dz'])
        if rv is not None:
            raw_vols[cid] = rv
            active = [k for k, v in (_get_ch_filter(filter_cfg, ch) or {}).items() if v is not None]
            tag = f"— active filter: {active}" if active else "— hist only"
            print(f"[raw_vol] [{cid}] loaded {rv.shape} {tag}")

    # ── Intensity histogram (before any filtering) ────────────────────────────
    if show_hist and raw_vols:
        ch_means = {}
        ch_by_id = {ch['id']: ch for ch in routing_config}
        for cid, rv in raw_vols.items():
            aligned_csv = _channel_2d_csv(ch_by_id[cid], tile_name, align_dir, raw_dir, fused_dir)
            print(f"[hist] Computing box means for {cid} ...")
            ch_means[cid] = _compute_box_means_from_csv(aligned_csv, z_range, rv)
            print(f"[hist] {cid}: {len(ch_means[cid]):,} boxes")
        _plot_intensity_histograms(ch_means, tile_name=tile_name, out_dir=base_res)

    # ── Box area histogram ────────────────────────────────────────────────────
    show_box_area_hist = vis_cfg.get('show_box_area_hist', False)
    if show_box_area_hist:
        ch_areas = {}
        for ch in routing_config:
            cid = ch['id']
            aligned_csv = _channel_2d_csv(ch, tile_name, align_dir, raw_dir, fused_dir,
                                           use_fallback=not has_aligned)
            print(f"[area_hist] Computing box areas for {cid} ...")
            ch_areas[cid] = _compute_box_areas_from_csv(aligned_csv, z_range)
            print(f"[area_hist] {cid}: {len(ch_areas[cid]):,} boxes")
        _plot_box_area_histograms(ch_areas, tile_name=tile_name, out_dir=base_res)

    # ── [aligned] & [raw] boxes ───────────────────────────────────────────────
    left_m, top_m = _get_tile_overlap_margins(anchor_dir, tile_name, canvas_w, canvas_h)
    if left_m > 0 or top_m > 0:
        print(f"[overlap] hiding boxes in left={left_m}px / top={top_m}px margin")
    _add_margin_overlay(viewer, canvas_shape, left_m, top_m)
    for ch in routing_config:
        cid     = ch['id']
        iou_str = (f"  iou={offsets[cid]['iou_score']:.3f}" if cid in offsets else "")
        aligned_csv = os.path.join(filtered_dir, f"{tile_name}_{cid}_result.csv")
        ch_filt = _get_ch_filter(filter_cfg, ch)
        (shapes_a, colors_a, meta_a), (rej_sa, rej_ca, _) = _load_tile_csv_shapes(
            aligned_csv, z_range,
            raw_vol=raw_vols.get(cid), filt=ch_filt,
            left_margin=left_m, top_margin=top_m)
        if shapes_a:
            layer_name_a = f"[aligned] {cid}{iou_str}"
            _add_labels_layer(
                viewer, shapes_a, colors_a, canvas_shape,
                name=layer_name_a,
                visible=False, opacity=aligned_opacity, outline_width=outline_width,
            )
            for m in meta_a:
                box_registry.append({**m, 'layer_name': layer_name_a})
            print(f"[aligned] {cid}: {len(shapes_a)} boxes")
        if rej_sa:
            _add_labels_layer(
                viewer, rej_sa, rej_ca, canvas_shape,
                name=f"[aligned rejected] {cid}",
                visible=False, opacity=0.5, outline_width=outline_width,
            )
            print(f"[aligned rejected] {cid}: {len(rej_sa)} boxes (hidden)")
        if show_before:
            raw_csv = os.path.join(raw_dir, f"{tile_name}_{cid}_result.csv")
            (shapes_r, colors_r, meta_r), _ = _load_tile_csv_shapes(
                raw_csv, z_range,
                raw_vol=raw_vols.get(cid), filt=ch_filt,
                left_margin=left_m, top_margin=top_m)
            if shapes_r:
                layer_name_r = f"[raw] {cid}"
                _add_labels_layer(
                    viewer, shapes_r, colors_r, canvas_shape,
                    name=layer_name_r, visible=False,
                    opacity=raw_opacity, outline_width=outline_width,
                )
                for m in meta_r:
                    box_registry.append({**m, 'layer_name': layer_name_r})
                print(f"[raw]     {cid}: {len(shapes_r)} boxes (hidden)")

    # ── [zlinked] boxes ───────────────────────────────────────────────────────
    if show_zlinked:
        for ch in routing_config:
            cid = ch['id']
            shapes_z, colors_z, meta_z = _vol_list_to_center_z_shapes(
                per_ch_vol_lists.get(cid, []), z_range)
            if shapes_z:
                layer_name_z = f"[zlinked] {cid}"
                _add_labels_layer(
                    viewer, shapes_z, colors_z, canvas_shape,
                    name=layer_name_z, visible=False,
                    opacity=zlinked_opacity, outline_width=outline_width,
                )
                for m in meta_z:
                    box_registry.append({**m, 'layer_name': layer_name_z})
                print(f"[zlinked] {cid}: {len(shapes_z)} cells")

    # ── [spheres] ─────────────────────────────────────────────────────────────
    if show_spheres:
        for ch in routing_config:
            cid   = ch['id']
            color = sphere_colors_cfg.get(cid) or SPHERE_COLORS.get(cid, _DEFAULT_SPHERE_COLOR)
            centers, sizes = _vol_list_to_points(per_ch_vol_lists.get(cid, []), z_range)
            if centers is not None:
                viewer.add_points(
                    centers, size=sizes,
                    face_color=[color] * len(centers),
                    border_width=0, opacity=sphere_opacity,
                    n_dimensional=True, name=f"[spheres] {cid}",
                )
                print(f"[spheres] {cid}: {len(centers)} cells")
            if show_before:
                raw_csv = os.path.join(raw_dir, f"{tile_name}_{cid}_result.csv")
                ctype   = ch.get('type', 'soma')
                zl_p    = zl_soma if ctype == 'soma' else zl_tf
                vol_list_r = _run_zlink_for_csv(
                    raw_csv, z_range,
                    iou_thresh=zl_p.get('iou_thresh', 0.35),
                    min_z_layers=1,
                    max_cell_z_span=zl_p.get('max_cell_z_span', 5),
                    max_z_gap=zl_p.get('max_z_gap', 0),
                    ch_id=cid,
                )
                centers_r, sizes_r = _vol_list_to_points(vol_list_r, z_range)
                if centers_r is not None:
                    viewer.add_points(
                        centers_r, size=sizes_r,
                        face_color=[color] * len(centers_r),
                        border_width=0, opacity=sphere_raw_opacity,
                        n_dimensional=True, visible=False,
                        name=f"[spheres-raw] {cid}",
                    )
        viewer.dims.ndisplay = 3
        print("\nSphere view active — napari in 3-D mode.")

    # ── [coloc] ───────────────────────────────────────────────────────────────
    if show_coloc:
        coloc_csv = os.path.join(base_res, '4_colocalization', 'coloc_result.csv')
        tile_x0, tile_y0, tile_z0 = _get_tile_offset(anchor_dir, tile_name, base_res=base_res, tile_paths=tile_paths)
        print(f"  [coloc] tile offset: x0={tile_x0}, y0={tile_y0}, z0={tile_z0}")

        if os.path.isfile(coloc_csv):
            _add_coloc_layers(
                viewer, coloc_csv, tile_name, z_range,
                tile_x0, tile_y0, tile_z0, canvas_shape,
                coloc_opacity, outline_width, box_registry=box_registry,
                left_margin=left_m, top_margin=top_m,
            )
        else:
            print("\n[coloc] coloc_result.csv not found, running in-memory...")
            try:
                coloc_vols = _run_tile_colocalization(
                    per_ch_vol_lists,
                    soma_ch_ids=soma_ch_ids, tf_ch_ids=tf_ch_ids,
                    zl_soma_cfg=zl_soma, zl_tf_cfg=zl_tf,
                )
                print(f"  In-memory: {len(coloc_vols)} soma cells")
                if coloc_vols:
                    groups = _auto_groups_from_classes(
                        [c.get('class', '') for c in coloc_vols],
                        outline_width=outline_width,
                    )
                    if not groups:
                        print("[coloc] no marker combinations found (no soma cell "
                              "carries >=2 markers — check channel routing / z_linker.tf config)")
                    any_shown = False
                    for grp, shapes_c, colors_c, dash_c, meta_c in _vol_list_to_s4_shapes(
                            coloc_vols, z_range, groups):
                        if shapes_c:
                            layer_name_c = f"[coloc] {grp['name']}"
                            _add_labels_layer(
                                viewer, shapes_c, colors_c, canvas_shape,
                                dash_sizes=dash_c,
                                name=layer_name_c,
                                visible=False, opacity=coloc_opacity,
                                outline_width=grp.get('outline_width', outline_width),
                            )
                            for m in meta_c:
                                box_registry.append({**m, 'layer_name': layer_name_c})
                            n_solid = sum(1 for d in dash_c if d == 0)
                            print(f"[coloc] {grp['name']}: {len(shapes_c)} boxes "
                                  f"({n_solid} neuron, {len(dash_c) - n_solid} glia)")
                            any_shown = True
                    if groups and not any_shown:
                        print("[coloc] no cells in z-range for any group")
                else:
                    print("[coloc] no soma cells produced by in-memory merge — "
                          "check soma channel routing/config")
            except Exception as exc:
                print(f"[coloc] WARNING: in-memory coloc failed — {exc}")

        # ── Overlap suppression overlay ───────────────────────────────────────
        if left_m > 0 or top_m > 0:
            ovl = np.zeros((canvas_z, canvas_h, canvas_w), dtype=np.uint8)
            if left_m > 0: ovl[:, :, :left_m] = 255
            if top_m  > 0: ovl[:, :top_m, :]  = 255
            viewer.add_image(
                ovl, name=f"[overlap] left={left_m}px top={top_m}px",
                colormap='cyan', opacity=0.25, blending='additive', visible=True,
            )

    # ── Click-to-inspect  /  Shift+click traces  /  Ctrl+click FN annotation ────
    fn_ann_path = vis_cfg.get('fn_ann_path') or None
    print(f"\n[info] Click any detection box → dimensions in status bar.")
    print(f"[info] Shift+click [coloc]   box → TF colocalization trace in terminal.")
    print(f"[info] Shift+click [zlinked] box → per-z span highlighted in yellow + info.")
    if fn_ann_path:
        print(f"[info] Ctrl+click  anywhere  → mark false-negative position → {fn_ann_path}")
    else:
        print(f"[info] Ctrl+click  disabled  (set 'fn_ann_path' in vis_config.json to enable)")
    print(f"[info] {len(box_registry)} boxes indexed across all layers.")

    @viewer.bind_key('Shift')
    def _shift_hint(v):
        v.status = "⇧ Shift held — click a [zlinked] or [coloc] box to trace"
        yield
        v.status = ""

    @viewer.bind_key('Control')
    def _ctrl_hint(v):
        v.status = "⌃ Ctrl held — click anywhere to mark a false-negative position"
        yield
        v.status = ""

    def _on_click(viewer, event):
        if event.type != 'mouse_press':
            return
        try:
            mods = [str(m).lower() for m in event.modifiers]
            is_shift = any('shift' in m for m in mods)
            is_ctrl  = any('ctrl' in m or 'control' in m for m in mods)
        except Exception:
            is_shift = is_ctrl = False
        pos = viewer.cursor.position
        if len(pos) < 3:
            return
        z_cur = int(round(pos[0]))
        y_cur = float(pos[1])
        x_cur = float(pos[2])

        if is_ctrl:
            if not fn_ann_path:
                viewer.status = "⌃ Ctrl+click disabled — set 'fn_ann_path' in vis_config.json"
                return
            z_abs = z_cur + z_range[0]
            ch_tag = _active_image_channel(viewer)
            _append_false_negative(fn_ann_path, tile_name, tile_path, z_abs, x_cur, y_cur,
                                   channel=ch_tag)
            hits = [b for b in box_registry
                    if b['z'] == z_cur
                    and b['x1'] <= x_cur <= b['x2']
                    and b['y1'] <= y_cur <= b['y2']]
            box_note = ""
            if hits:
                best_h = min(hits, key=lambda b: (b['x2']-b['x1'])*(b['y2']-b['y1']))
                box_note = f"  (inside [{best_h['layer_name']}] {best_h['cls']})"
            ch_note = f"  ch={ch_tag}" if ch_tag else ""
            print(f"[fn] {tile_name}  z={z_abs}  x={x_cur:.0f}  y={y_cur:.0f}{ch_note}{box_note}")
            viewer.status = (f"⌃ FN marked: z={z_abs} x={x_cur:.0f} y={y_cur:.0f}"
                             f"{ch_note}{box_note}  → {os.path.basename(fn_ann_path)}")
            return

        hits = [
            b for b in box_registry
            if b['z'] == z_cur
            and b['x1'] <= x_cur <= b['x2']
            and b['y1'] <= y_cur <= b['y2']
        ]
        if hits:
            candidates = hits
            active_layer = viewer.layers.selection.active
            if active_layer is not None:
                active_hits = [b for b in candidates if b.get('layer_name') == active_layer.name]
                if active_hits:
                    candidates = active_hits
            best = min(candidates, key=lambda b: (b['x2'] - b['x1']) * (b['y2'] - b['y1']))
            w = best['x2'] - best['x1']
            h = best['y2'] - best['y1']
            lname = best.get('layer_name', '')
            mean_str = f"  mean={best['mean']:.0f}" if best.get('mean', 0) != 0 else ""
            if not is_shift:
                viewer.status = (f"[{lname}] {best['cls']}  "
                                 f"w={w:.0f}px  h={h:.0f}px  z={best['z']}{mean_str}")
            elif lname.startswith('[coloc]'):
                if _debug_soma_vols:
                    soma_idx = _find_soma_idx_for_box(best, z_range, _debug_soma_vols)
                    if soma_idx is not None:
                        res = _trace_coloc_for_soma(
                            soma_idx, _debug_soma_vols, _debug_all_tf, **_debug_cp)
                        _print_coloc_trace(res, _debug_cp)
                        _print_coloc_all_channels(res, soma_ch_ids, per_ch_vol_lists)
                        viewer.status = f"⇧ coloc trace for soma #{soma_idx} → see terminal"
                    else:
                        cx = (best['x1'] + best['x2']) / 2
                        cy = (best['y1'] + best['y2']) / 2
                        print(f"[trace] no soma matched — box cx={cx:.0f} cy={cy:.0f} z={best['z']}")
                        viewer.status = f"⇧ [coloc] no soma matched at cx={cx:.0f} cy={cy:.0f} z={best['z']}"
                else:
                    print("[trace] vol_lists not built — enable show_coloc/show_zlinked/spheres")
                    viewer.status = "⇧ [coloc] vol_lists not available — see terminal"
            elif lname.startswith(('[s3]', '[zlinked]')):
                for lyr in list(viewer.layers):
                    if lyr.name == '[debug] s3 span':
                        viewer.layers.remove(lyr)
                cell, cid = _find_vollist_entry_for_box(
                    best, z_range, per_ch_vol_lists, lname)
                if cell is not None:
                    _print_s3_cell_info(cell, cid)
                    sh, co = _build_s3_span_shapes(cell, z_range, canvas_shape)
                    if sh:
                        _add_labels_layer(viewer, sh, co, canvas_shape,
                                          name='[debug] s3 span', visible=True,
                                          opacity=0.9, outline_width=2)
                    viewer.status = (f"⇧ [{cid}] s3 info → terminal  "
                                     f"cz={cell['cz']}  z=[{cell['z_min']}-{cell['z_max']}]")
                else:
                    cx = (best['x1'] + best['x2']) / 2
                    cy = (best['y1'] + best['y2']) / 2
                    print(f"[trace] no vol_list match for s3 box "
                          f"cx={cx:.0f} cy={cy:.0f} z={best['z']}")
                    viewer.status = (f"⇧ no match in vol_list for {lname}  "
                                     f"cx={cx:.0f} cy={cy:.0f} z={best['z']} — see terminal")
            else:
                viewer.status = f"⇧ {lname} is not traceable — use [zlinked] or [coloc]"
        else:
            if is_shift:
                viewer.status = f"⇧ Shift+click: no box at z={z_cur} x={x_cur:.0f} y={y_cur:.0f}"
            else:
                viewer.status = f"(no box at z={z_cur} x={x_cur:.0f} y={y_cur:.0f})"

    viewer.mouse_drag_callbacks.append(_on_click)

    viewer.reset_view()


# ── Mode: post (single tile) ──────────────────────────────────────────────────

def _run_post(vis_cfg, paths, routing_config, tile_path, tile_name):
    zl_cfg  = vis_cfg.get('z_linker', {})
    zl_soma = zl_cfg.get('soma', {})
    zl_tf   = zl_cfg.get('tf',   {})

    no_images     = vis_cfg.get('no_images', False)
    contrast_pct  = tuple(vis_cfg.get('contrast_pct', [0.1, 99.9]))
    stage_cfg     = vis_cfg.get('stage', 'all')
    outline_width = vis_cfg.get('outline_width', 2)
    coloc_opacity = vis_cfg.get('coloc_opacity', 0.9)

    base_res     = paths['pATHRESULT']
    raw_dir      = os.path.join(base_res, '1_tile_2d_raw')
    filtered_dir = os.path.join(base_res, '1_tile_2d_filtered')
    fused_dir    = os.path.join(base_res, '1_tile_2d_fused')
    s3_dir    = os.path.join(base_res, '3_channel_3d')
    coloc_csv = os.path.join(base_res, '4_colocalization', 'coloc_result.csv')

    anchor_ch  = routing_config[0]
    anchor_dir = os.path.abspath(paths[anchor_ch['dir_key']])
    _, tile_paths = listTile(anchor_dir)

    print(f"\nTile: {tile_name}")

    z_range      = _resolve_z_range(vis_cfg, tile_path)
    canvas_shape = _canvas_shape_from_tile(tile_path, z_range)
    _, canvas_h, canvas_w = canvas_shape
    print(f"Z-range: {z_range[0]} – {z_range[1]} ({z_range[1] - z_range[0]} slices)\n")

    # Tile global offset (needed for s3 and coloc coordinate conversion)
    tile_x0, tile_y0, tile_z0 = _get_tile_offset(anchor_dir, tile_name, base_res=base_res, tile_paths=tile_paths)
    print(f"Tile global offset: x0={tile_x0}, y0={tile_y0}, z0={tile_z0}\n")

    viewer = napari.Viewer(title=f"Post-Pipeline — {tile_name}")

    box_registry = []  # {z, x1, y1, x2, y2, cls, layer_name} — populated below

    # ── Load raw volumes for intensity filtering / histogram ──────────────────
    filter_cfg    = vis_cfg.get('filter', {})
    show_hist     = vis_cfg.get('show_intensity_hist', False)
    raw_vols = {}
    for ch in routing_config:
        cid = ch['id']
        need_vol = bool(_get_ch_filter(filter_cfg, ch)) or show_hist
        if not need_vol:
            continue
        ch_base  = os.path.abspath(paths[ch['dir_key']])
        tile_dir = os.path.join(ch_base, os.path.relpath(tile_path, anchor_dir))
        rv = load_raw_volume(tile_dir, z_range)
        if rv is not None:
            raw_vols[cid] = rv
            active = [k for k, v in (_get_ch_filter(filter_cfg, ch) or {}).items() if v is not None]
            tag = f"— active filter: {active}" if active else "— hist only"
            print(f"[raw_vol] [{cid}] loaded {rv.shape} {tag}")

    # ── Intensity histogram (before any filtering) ────────────────────────────
    if show_hist and raw_vols:
        ch_means = {}
        ch_by_id = {ch['id']: ch for ch in routing_config}
        for cid, rv in raw_vols.items():
            csv_p = _channel_2d_csv(ch_by_id[cid], tile_name, raw_dir, raw_dir, fused_dir)
            print(f"[hist] Computing box means for {cid} ...")
            ch_means[cid] = _compute_box_means_from_csv(csv_p, z_range, rv)
            print(f"[hist] {cid}: {len(ch_means[cid]):,} boxes")
        _plot_intensity_histograms(ch_means, tile_name=tile_name, out_dir=base_res)

    # ── Box area histogram ────────────────────────────────────────────────────
    show_box_area_hist = vis_cfg.get('show_box_area_hist', False)
    if show_box_area_hist:
        ch_areas = {}
        for ch in routing_config:
            cid = ch['id']
            csv_p = _channel_2d_csv(ch, tile_name, raw_dir, raw_dir, fused_dir)
            print(f"[area_hist] Computing box areas for {cid} ...")
            ch_areas[cid] = _compute_box_areas_from_csv(csv_p, z_range)
            print(f"[area_hist] {cid}: {len(ch_areas[cid]):,} boxes")
        _plot_box_area_histograms(ch_areas, tile_name=tile_name, out_dir=base_res)

    # ── Image layers (raw, no shift) ──────────────────────────────────────────
    if not no_images:
        img_layers = []
        for ch in routing_config:
            cid      = ch['id']
            ch_base  = os.path.abspath(paths[ch['dir_key']])
            tile_dir = os.path.join(ch_base, os.path.relpath(tile_path, anchor_dir))
            print(f"[img] Loading {cid}  ({tile_dir}) ...")
            vol, climits = load_volume(tile_dir, z_range=z_range, contrast_pct=contrast_pct)
            if vol is None:
                print("  → not found, skipping")
                continue
            canvas_shape = vol.shape
            img_layer = viewer.add_image(vol, name=f"[img] {cid}", contrast_limits=climits,
                                          **_ch_vis(cid))
            img_layer.contrast_limits_range = (0, 65535)
            img_layers.append((cid, img_layer))
            print(f"  → {vol.shape}  contrast_limits={climits}")

            second = _add_second_intensity_layer(
                viewer, ch, paths, anchor_dir, tile_path, z_range, contrast_pct)
            if second:
                img_layers.append(second)
        _add_contrast_panel(viewer, img_layers)

    # ── Overlap margins (hide inaccurate boxes in left/top tile boundaries) ──────
    left_m, top_m = _get_tile_overlap_margins(anchor_dir, tile_name, canvas_w, canvas_h)
    if left_m > 0 or top_m > 0:
        print(f"[overlap] hiding boxes in left={left_m}px / top={top_m}px margin")
    _add_margin_overlay(viewer, canvas_shape, left_m, top_m)

    # ── [s1] 2D detections (from 1_tile_2d_filtered — regular channels are filtered
    # directly, double_exposure channels are fused then filtered; either way this is
    # the single source of truth Stage 3 itself reads from) ──────────────────────
    if stage_cfg in ('all', 's1'):
        for ch in routing_config:
            cid = ch['id']
            csv_p = os.path.join(filtered_dir, f"{tile_name}_{cid}_result.csv")
            (shapes, colors, meta), (rej_s, rej_c, _) = _load_tile_csv_shapes(
                csv_p, z_range,
                raw_vol=raw_vols.get(cid), filt=_get_ch_filter(filter_cfg, ch),
                left_margin=left_m, top_margin=top_m)
            if shapes:
                layer_name_s1 = f"[s1] {cid}"
                _add_labels_layer(
                    viewer, shapes, colors, canvas_shape,
                    name=layer_name_s1, visible=False,
                    opacity=0.7, outline_width=outline_width,
                )
                for m in meta:
                    box_registry.append({**m, 'layer_name': layer_name_s1})
                print(f"[s1] {cid}: {len(shapes)} 2D boxes (filtered) (hidden)")
            if rej_s:
                _add_labels_layer(
                    viewer, rej_s, rej_c, canvas_shape,
                    name=f"[s1 rejected] {cid}", visible=False,
                    opacity=0.5, outline_width=outline_width,
                )
                print(f"[s1 rejected] {cid}: {len(rej_s)} boxes (hidden)")

    # ── [s3] Saved z-linked tracks ────────────────────────────────────────────
    if stage_cfg in ('all', 's3'):
        for ch in routing_config:
            cid    = ch['id']
            ch_csv = os.path.join(s3_dir, f"{cid}_3d_tracked.csv")
            (shapes, colors, meta), (rej_s, rej_c, _) = _load_global_csv_to_tile_shapes(
                ch_csv, tile_name, tile_x0, tile_y0, tile_z0,
                z_range, canvas_w, canvas_h,
                raw_vol=raw_vols.get(cid), filt=_get_ch_filter(filter_cfg, ch),
                left_margin=left_m, top_margin=top_m)
            if shapes:
                layer_name_s3 = f"[s3] {cid}"
                _add_labels_layer(
                    viewer, shapes, colors, canvas_shape,
                    name=layer_name_s3, visible=False,
                    opacity=0.8, outline_width=outline_width,
                )
                for m in meta:
                    box_registry.append({**m, 'layer_name': layer_name_s3})
                print(f"[s3] {cid}: {len(shapes)} z-linked cells")
            else:
                print(f"[s3] {cid}: no cells in tile/z-range  ({ch_csv})")
            if rej_s:
                _add_labels_layer(
                    viewer, rej_s, rej_c, canvas_shape,
                    name=f"[s3 rejected] {cid}", visible=False,
                    opacity=0.5, outline_width=outline_width,
                )
                print(f"[s3 rejected] {cid}: {len(rej_s)} boxes (hidden)")

    # ── [s4] Colocalization result ────────────────────────────────────────────
    if stage_cfg in ('all', 's4'):
        _add_coloc_layers(
            viewer, coloc_csv, tile_name, z_range,
            tile_x0, tile_y0, tile_z0, canvas_shape,
            coloc_opacity, outline_width, box_registry=box_registry,
            left_margin=left_m, top_margin=top_m,
        )

    # ── Build merged soma/TF vol lists for Shift+click coloc trace ──────────────
    soma_ch_ids_p = [ch['id'] for ch in routing_config if ch.get('type', 'soma') == 'soma']
    tf_ch_ids_p   = [ch['id'] for ch in routing_config if ch.get('type') == 'tf']
    print("[trace] Building vol_lists for Shift+click coloc trace...")
    _per_ch_vols = {}
    for ch in routing_config:
        cid   = ch['id']
        ctype = ch.get('type', 'soma')
        pkl_p = os.path.join(s3_dir, f"{cid}_3d_tracked.pkl")
        vols  = _load_s3_pkl_as_tile_local(pkl_p, tile_x0, tile_y0, tile_z0)
        if vols is not None:
            print(f"[trace] {cid}: loaded {len(vols)} cells from pkl")
            _per_ch_vols[cid] = vols
        else:
            csv_p = os.path.join(filtered_dir, f"{tile_name}_{cid}_result.csv")
            zl_p  = zl_soma if ctype == 'soma' else zl_tf
            print(f"[trace] {cid}: pkl not found, falling back to CSV re-z-link")
            _per_ch_vols[cid] = _run_zlink_for_csv(
                csv_p, z_range,
                iou_thresh=zl_p.get('iou_thresh', 0.35),
                min_z_layers=zl_p.get('min_z_layers', 1),
                max_cell_z_span=zl_p.get('max_cell_z_span', 5),
                max_z_gap=zl_p.get('max_z_gap', 0),
                ch_id=cid,
            )
    _debug_soma_vols, _debug_tf_dict = _build_merged_soma_vols(
        _per_ch_vols, soma_ch_ids_p, tf_ch_ids_p, zl_soma)
    _debug_all_tf = [dict(c, _cid=_tf_cid)
                     for _tf_cid, _vols in _debug_tf_dict.items()
                     for c in _vols]
    _debug_cp = {
        'z_pad':                 zl_tf.get('containment_z_pad', 2),
        'xy_margin':             zl_tf.get('containment_xy_margin', 0),
        'max_center_dist_ratio': zl_tf.get('max_center_dist_ratio', 0.5),
    }
    print(f"[trace] Shift+click any [coloc] box to trace.  "
          f"({len(_debug_soma_vols)} soma, {len(_debug_all_tf)} TF)")

    # ── Click-to-inspect  /  Shift+click traces  /  Ctrl+click FN annotation ────
    fn_ann_path = vis_cfg.get('fn_ann_path') or None
    print(f"\n[info] Click any detection box → dimensions in status bar.")
    print(f"[info] Shift+click [coloc] box → TF colocalization trace in terminal.")
    print(f"[info] Shift+click [s3]    box → per-z span highlighted in yellow + info.")
    if fn_ann_path:
        print(f"[info] Ctrl+click  anywhere  → mark false-negative position → {fn_ann_path}")
    else:
        print(f"[info] Ctrl+click  disabled  (set 'fn_ann_path' in vis_config.json to enable)")
    print(f"[info] {len(box_registry)} boxes indexed across all layers.")

    @viewer.bind_key('Shift')
    def _shift_hint(v):
        v.status = "⇧ Shift held — click a [s3] or [coloc] box to trace"
        yield
        v.status = ""

    @viewer.bind_key('Control')
    def _ctrl_hint(v):
        v.status = "⌃ Ctrl held — click anywhere to mark a false-negative position"
        yield
        v.status = ""

    def _on_click(viewer, event):
        if event.type != 'mouse_press':
            return
        try:
            mods = [str(m).lower() for m in event.modifiers]
            is_shift = any('shift' in m for m in mods)
            is_ctrl  = any('ctrl' in m or 'control' in m for m in mods)
        except Exception:
            is_shift = is_ctrl = False
        pos = viewer.cursor.position
        if len(pos) < 3:
            return
        z_cur = int(round(pos[0]))
        y_cur = float(pos[1])
        x_cur = float(pos[2])

        if is_ctrl:
            if not fn_ann_path:
                viewer.status = "⌃ Ctrl+click disabled — set 'fn_ann_path' in vis_config.json"
                return
            z_abs = z_cur + z_range[0]
            ch_tag = _active_image_channel(viewer)
            _append_false_negative(fn_ann_path, tile_name, tile_path, z_abs, x_cur, y_cur,
                                   channel=ch_tag)
            hits = [b for b in box_registry
                    if b['z'] == z_cur
                    and b['x1'] <= x_cur <= b['x2']
                    and b['y1'] <= y_cur <= b['y2']]
            box_note = ""
            if hits:
                best_h = min(hits, key=lambda b: (b['x2']-b['x1'])*(b['y2']-b['y1']))
                box_note = f"  (inside [{best_h['layer_name']}] {best_h['cls']})"
            ch_note = f"  ch={ch_tag}" if ch_tag else ""
            print(f"[fn] {tile_name}  z={z_abs}  x={x_cur:.0f}  y={y_cur:.0f}{ch_note}{box_note}")
            viewer.status = (f"⌃ FN marked: z={z_abs} x={x_cur:.0f} y={y_cur:.0f}"
                             f"{ch_note}{box_note}  → {os.path.basename(fn_ann_path)}")
            return

        hits = [
            b for b in box_registry
            if b['z'] == z_cur
            and b['x1'] <= x_cur <= b['x2']
            and b['y1'] <= y_cur <= b['y2']
        ]
        if hits:
            if is_shift:
                trace_hits = [b for b in hits
                              if b.get('layer_name', '').startswith(('[s3]', '[zlinked]', '[coloc]'))]
                candidates = trace_hits if trace_hits else hits
            else:
                candidates = hits
            active_layer = viewer.layers.selection.active
            if active_layer is not None:
                active_hits = [b for b in candidates if b.get('layer_name') == active_layer.name]
                if active_hits:
                    candidates = active_hits
            best = min(candidates, key=lambda b: (b['x2'] - b['x1']) * (b['y2'] - b['y1']))
            w = best['x2'] - best['x1']
            h = best['y2'] - best['y1']
            lname = best.get('layer_name', '')
            mean_str = f"  mean={best['mean']:.0f}" if best.get('mean', 0) != 0 else ""
            if not is_shift:
                viewer.status = (f"[{lname}] {best['cls']}  "
                                 f"w={w:.0f}px  h={h:.0f}px  z={best['z']}{mean_str}")
            elif lname.startswith('[coloc]'):
                if _debug_soma_vols:
                    soma_idx = _find_soma_idx_for_box(best, z_range, _debug_soma_vols)
                    if soma_idx is not None:
                        res = _trace_coloc_for_soma(
                            soma_idx, _debug_soma_vols, _debug_all_tf, **_debug_cp)
                        _print_coloc_trace(res, _debug_cp)
                        _print_coloc_all_channels(res, soma_ch_ids_p, _per_ch_vols)
                        viewer.status = f"⇧ coloc trace for soma #{soma_idx} → see terminal"
                    else:
                        cx = (best['x1'] + best['x2']) / 2
                        cy = (best['y1'] + best['y2']) / 2
                        print(f"[trace] no soma matched — box cx={cx:.0f} cy={cy:.0f} z={best['z']}")
                        viewer.status = f"⇧ [coloc] no soma matched at cx={cx:.0f} cy={cy:.0f} z={best['z']}"
                else:
                    print("[trace] vol_lists not available")
                    viewer.status = "⇧ [coloc] vol_lists not available — see terminal"
            elif lname.startswith(('[s3]', '[zlinked]')):
                for lyr in list(viewer.layers):
                    if lyr.name == '[debug] s3 span':
                        viewer.layers.remove(lyr)
                cell, cid = _find_vollist_entry_for_box(
                    best, z_range, _per_ch_vols, lname)
                if cell is not None:
                    _print_s3_cell_info(cell, cid)
                    sh, co = _build_s3_span_shapes(cell, z_range, canvas_shape)
                    if sh:
                        _add_labels_layer(viewer, sh, co, canvas_shape,
                                          name='[debug] s3 span', visible=True,
                                          opacity=0.9, outline_width=2)
                    viewer.status = (f"⇧ [{cid}] s3 info → terminal  "
                                     f"cz={cell['cz']}  z=[{cell['z_min']}-{cell['z_max']}]")
                else:
                    cx = (best['x1'] + best['x2']) / 2
                    cy = (best['y1'] + best['y2']) / 2
                    print(f"[trace] no vol_list match for s3 box "
                          f"cx={cx:.0f} cy={cy:.0f} z={best['z']}")
                    viewer.status = (f"⇧ no match in vol_list for {lname}  "
                                     f"cx={cx:.0f} cy={cy:.0f} z={best['z']} — see terminal")
            else:
                viewer.status = f"⇧ {lname} is not traceable — only [s3], [zlinked], [coloc] are traceable"
        else:
            if is_shift:
                viewer.status = f"⇧ Shift+click: no box at z={z_cur} x={x_cur:.0f} y={y_cur:.0f}"
            else:
                viewer.status = f"(no box at z={z_cur} x={x_cur:.0f} y={y_cur:.0f})"

    viewer.mouse_drag_callbacks.append(_on_click)

    viewer.reset_view()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='brain_detector unified visualizer.')
    parser.add_argument(
        '--config',
        default=os.path.join(project_root, 'config', 'vis_config.json'),
        help='Path to vis_config.json',
    )
    args = parser.parse_args()

    vis_cfg_path = args.config
    vis_cfg = load_config(vis_cfg_path) if os.path.isfile(vis_cfg_path) else {}

    mode = vis_cfg.get('mode', 'prealign')
    print(f"Mode: {mode}\n")

    if mode not in ('prealign', 'post'):
        sys.exit(f"Unknown mode '{mode}'. Use 'prealign' or 'post'.")

    paths = vis_cfg.get('paths') or {}
    routing_config = [ch for ch in vis_cfg.get('channels_routing', []) if ch.get('active', True)]

    anchor_ch  = routing_config[0]
    anchor_dir = os.path.abspath(paths[anchor_ch['dir_key']])
    _, tile_paths = listTile(anchor_dir)
    selected_tiles = _select_tiles(vis_cfg, tile_paths)
    print(f"\n{len(selected_tiles)} tile(s) selected: "
          f"{', '.join(name for _, name in selected_tiles)}")

    # One napari.Viewer() (= one OS window) per tile, all built up-front; napari.run()
    # is called once at the end so every window stays open and you can Alt-Tab / click
    # between them, instead of blocking on a single tile at a time.
    for tile_path, tile_name in selected_tiles:
        print(f"\n{'=' * 70}\nTile: {tile_name}\n{'=' * 70}")
        if mode == 'prealign':
            _run_prealign(vis_cfg, paths, routing_config, tile_path, tile_name)
        else:
            _run_post(vis_cfg, paths, routing_config, tile_path, tile_name)

    napari.run()


if __name__ == '__main__':
    main()
