# -*- coding: utf-8 -*-
"""
Pre-align channel alignment visualizer.

Opens a napari viewer for a single tile showing:
  - Raw images for each active channel (tile-local coordinates)
  - [aligned]  Detection boxes AFTER alignment  (from 0_channel_alignment/)
  - [raw]      Detection boxes BEFORE alignment (from 1_tile_2d_raw/, hidden)
  - [zlinked]  Per-slice boxes for z-linked 3D cells per channel
  - [spheres]  3D sphere centroids from z-linking (napari 3D view)
  - [coloc]    In-memory colocalization result (soma multi-channel + TF annotation)

Usage
-----
  # 通常直接运行即可，主 config 自动从 vis_config_prealign.json 的 pATHRESULT/runtime_config.json 读取
  python src/utils/visualize_prealign.py

  # 手动指定主 config（覆盖自动推导）
  python src/utils/visualize_prealign.py --config config/config_IMARIS_preAlign.json --tile 000000_001000_000000
"""

import argparse
import json
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import napari

from src.config.loader import load_config
from src.utils.io import listTile
from src.core.z_linker import run_z_linker
from src.core.stitcher import (
    match_soma_3d_iou, annotate_soma_with_tf_gmm,
    suppress_cross_class_overlap, _merge_class,
)

from src.utils.visualize import (
    load_volume,
    _rasterize_shapes_to_labels,
    _add_labels_layer,
    CHANNEL_VIS,
    DEFAULT_VIS,
    _color,
)

# ── Color for Olig2 (not in the original CHANNEL_VIS) ───────────────────────
_EXTRA_VIS = {
    'Olig2': dict(colormap='magenta', blending='additive', opacity=0.8),
}

# ── Per-marker base RGB colors for auto-coloc group color blending ────────────
_MARKER_RGB = {
    'rfp':   [1.0, 0.20, 0.20],
    'gfp':   [0.20, 1.0, 0.20],
    'sox9':  [0.0,  0.80, 1.0],
    'olig2': [1.0,  0.20, 1.0],
}


def _marker_combo_color(markers):
    """Return (neuron_rgba, glia_rgba) for a tuple of marker names."""
    rgbs = [_MARKER_RGB.get(m.lower(), [0.7, 0.7, 0.7]) for m in markers]
    if not rgbs:
        rgb = [0.9, 0.9, 0.9]
    else:
        rgb = [sum(c[i] for c in rgbs) / len(rgbs) for i in range(3)]
        mx = max(rgb) if max(rgb) > 0 else 1.0
        rgb = [min(c / mx, 1.0) for c in rgb]
    neuron = rgb + [1.0]
    glia   = [c * 0.65 for c in rgb] + [1.0]
    return neuron, glia


# ── Sphere colors per channel (RGBA, 0-1) ────────────────────────────────────
SPHERE_COLORS = {
    'RFP':   [1.0,  0.25, 0.25, 0.85],
    'GFP':   [0.25, 1.0,  0.25, 0.85],
    'Sox9':  [0.0,  0.85, 1.0,  0.85],
    'Olig2': [1.0,  0.2,  1.0,  0.85],
}
_DEFAULT_SPHERE_COLOR = [0.9, 0.9, 0.9, 0.7]


def _ch_vis(ch_id):
    return CHANNEL_VIS.get(ch_id, _EXTRA_VIS.get(ch_id, DEFAULT_VIS))


# ── Image shift helper ────────────────────────────────────────────────────────

def _shift_volume(vol, dx, dy, dz):
    """Shift a (Z, H, W) volume by (dx, dy, dz) with zero-padding.

    Same sign convention as apply_shift_to_csv: aligned_coord = raw_coord + d.
    """
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


# ── Z-linking helper ──────────────────────────────────────────────────────────

def _run_zlink_for_csv(csv_path, z_range, iou_thresh, min_z_layers, max_cell_z_span):
    """Z-link a per-tile detection CSV and return the volumetric_list."""
    if not os.path.isfile(csv_path):
        return []
    df = pd.read_csv(
        csv_path,
        names=['slice_name', 'x1', 'y1', 'x2', 'y2', 'class', 'score', 'mean', 'z'],
        skiprows=1,
    )
    if df.empty:
        return []
    df['z'] = df['z'].astype(float).astype(int) - 1  # CSV z is 1-indexed; convert to 0-indexed
    if z_range is not None:
        df = df[(df['z'] >= z_range[0]) & (df['z'] < z_range[1])]
    if df.empty:
        return []
    matrix = df[['x1', 'y1', 'x2', 'y2', 'score', 'mean', 'class', 'z']].values
    _, vol_list = run_z_linker(
        matrix,
        iou_thresh=iou_thresh,
        min_z_layers=min_z_layers,
        max_cell_z_span=max_cell_z_span,
    )
    return vol_list


# ── vol_list → napari shapes at center-z only (one box per cell) ─────────────

def _vol_list_to_center_z_shapes(vol_list, z_range):
    """Convert a volumetric_list to napari shapes using only the center z layer.

    Returns (shapes_list, colors_list) with one rectangle per cell, placed at
    the cell's median z (cz).  Cells whose center falls outside z_range are skipped.
    """
    z0 = z_range[0] if z_range is not None else 0
    z1 = z_range[1] if z_range is not None else float('inf')

    shapes, colors = [], []
    for cell in vol_list:
        cz = int(round(cell['cz']))
        if cz < z0 or cz >= z1:
            continue
        z_local = cz - z0
        cls_color = _color(cell.get('class', 'neuron'))

        per_z = cell.get('per_z_boxes', {})
        box = per_z.get(cz) or [
            cell.get('x1_3d', 0), cell.get('y1_3d', 0),
            cell.get('x2_3d', 0), cell.get('y2_3d', 0),
        ]
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        rect = np.array([
            [z_local, y1, x1],
            [z_local, y1, x2],
            [z_local, y2, x2],
            [z_local, y2, x1],
        ], dtype=np.float64)
        shapes.append(rect)
        colors.append(cls_color)

    return shapes, colors


# ── vol_list → s4-group shapes for colocalization ────────────────────────────

def _vol_list_to_s4_shapes(vol_list, z_range, s4_groups):
    """Build per-group shapes/colors/dash_sizes for colocalized cells.

    Each group matches cells whose marker set is a superset of group['channels'].
    Neuron → solid (dash=0), glia → dashed (dash=group['dash_size']).
    Only the center z of each cell is drawn.

    Returns list of (grp_dict, shapes, colors, dash_sizes).
    """
    z0 = z_range[0] if z_range is not None else 0
    z1 = z_range[1] if z_range is not None else float('inf')

    results = []
    for grp in s4_groups:
        req = frozenset(c.lower() for c in grp['channels'])
        neuron_color = grp.get('neuron_color', [1.0, 1.0, 0.0, 1.0])
        glia_color   = grp.get('glia_color',   [1.0, 0.65, 0.0, 1.0])
        dash_size    = grp.get('dash_size', 8)

        shapes, colors, dash_sizes = [], [], []
        for cell in vol_list:
            parts = str(cell.get('class', '')).split('_')
            base = parts[0]
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
            rect = np.array([
                [z_local, y1, x1],
                [z_local, y1, x2],
                [z_local, y2, x2],
                [z_local, y2, x1],
            ], dtype=np.float64)
            shapes.append(rect)
            colors.append(glia_color if base == 'glia' else neuron_color)
            dash_sizes.append(dash_size if base == 'glia' else 0)

        results.append((grp, shapes, colors, dash_sizes))
    return results


# ── Load saved coloc_result.csv → s4-group shapes ────────────────────────────

def _load_coloc_s4_shapes(coloc_csv, tile_name, z_range, s4_groups,
                           tile_x0=0, tile_y0=0, tile_z0=0):
    """Read 4_colocalization/coloc_result.csv and build per-group shapes.

    Returns list of (grp_dict, shapes, colors, dash_sizes), same format as
    _vol_list_to_s4_shapes.  Each row in the CSV becomes one 2-D box at its
    own z slice (no center-z aggregation — the CSV is already 2D per-slice).

    tile_z0: z_start - ABS_D from TeraStitcher XML.  combine_predictions stores
    z_coloc = z_tile_local_1indexed - z0_combine in the CSV.  To recover the
    0-indexed napari layer: z_local = z_coloc + tile_z0 - z_range[0] - 1.
    """
    if not os.path.isfile(coloc_csv):
        return None  # signal: file not found, caller should fall back

    df = pd.read_csv(coloc_csv)
    needed = {'x1', 'y1', 'x2', 'y2', 'class', 'z', 'tile_name'}
    if df.empty or not needed.issubset(df.columns):
        return None

    df = df[df['tile_name'] == tile_name]
    if df.empty:
        return []

    z0 = z_range[0] if z_range is not None else 0
    z1 = z_range[1] if z_range is not None else float('inf')
    df = df.copy()
    df['z'] = df['z'].astype(float).astype(int)
    # coloc z = (tile-local 1-indexed z) - z0_combine, so tile-local 0-indexed = z_csv + tile_z0 - 1
    # then subtract z_range[0] to get napari layer index
    df['z_local'] = df['z'] + tile_z0 - z0 - 1
    df = df[(df['z_local'] >= 0) & (df['z_local'] < z1 - z0)]
    if df.empty:
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
                lambda c: frozenset(p.lower() for p in str(c).split('_')[1:]) == req
            )
        else:
            mask = df['class'].apply(
                lambda c: req.issubset(frozenset(p.lower() for p in str(c).split('_')[1:]))
            )
        df_grp = df[mask]
        if df_grp.empty:
            results.append((grp, [], [], []))
            continue

        bc = base_col[df_grp.index]
        n = len(df_grp)
        arr = np.empty((n, 4, 3), dtype=np.float64)
        z_col = df_grp['z_local'].values
        y1v = df_grp['y1'].values.astype(float) - tile_y0
        y2v = df_grp['y2'].values.astype(float) - tile_y0
        x1v = df_grp['x1'].values.astype(float) - tile_x0
        x2v = df_grp['x2'].values.astype(float) - tile_x0
        arr[:, 0] = np.column_stack([z_col, y1v, x1v])
        arr[:, 1] = np.column_stack([z_col, y1v, x2v])
        arr[:, 2] = np.column_stack([z_col, y2v, x2v])
        arr[:, 3] = np.column_stack([z_col, y2v, x1v])

        colors    = [glia_color if b == 'glia' else neuron_color for b in bc]
        dash_sizes = [dash_size if b == 'glia' else 0 for b in bc]
        results.append((grp, list(arr), colors, dash_sizes))

    return results


# ── Auto-detect coloc groups from CSV ────────────────────────────────────────

def _auto_coloc_groups(coloc_csv, outline_width=4, dash_size=8):
    """Scan coloc_result.csv and build one group per unique marker combination.

    Each group uses exact matching (only cells whose marker set equals the group),
    so every cell appears in exactly one layer.  Colors are auto-assigned by
    blending the per-marker colors from _MARKER_RGB.
    """
    if not os.path.isfile(coloc_csv):
        return None
    df = pd.read_csv(coloc_csv)
    if df.empty or 'class' not in df.columns:
        return []

    # Collect unique sorted marker tuples (case-preserved for display, lower for lookup)
    seen = {}
    for cls in df['class'].dropna().unique():
        parts = str(cls).split('_')
        markers = tuple(sorted(parts[1:]))  # sorted preserves original case
        if markers:
            seen[markers] = None

    groups = []
    for markers in sorted(seen.keys(), key=lambda m: (len(m), m)):
        name = '+'.join(markers)
        ncol, gcol = _marker_combo_color(markers)
        groups.append({
            'name':          name,
            'channels':      list(markers),
            'neuron_color':  ncol,
            'glia_color':    gcol,
            'outline_width': outline_width,
            'dash_size':     dash_size,
            'exact':         False,
        })
    return groups


# ── vol_list → napari sphere points ──────────────────────────────────────────

def _vol_list_to_points(vol_list, z_range):
    """Convert a volumetric_list to napari Points arrays.

    Returns (centers (N,3) [z_local, y, x], sizes (N,) [XY diameter px])
    or (None, None) if no cells fall within z_range.
    """
    z0 = z_range[0] if z_range is not None else 0
    z1 = z_range[1] if z_range is not None else float('inf')
    centers, sizes = [], []
    for cell in vol_list:
        cz = cell['cz']
        if cz < z0 or cz >= z1:
            continue
        diam = max(
            cell['x2_3d'] - cell['x1_3d'],
            cell['y2_3d'] - cell['y1_3d'],
        )
        centers.append([cz - z0, cell['cy'], cell['cx']])
        sizes.append(max(diam, 1.0))
    if not centers:
        return None, None
    return np.array(centers, dtype=float), np.array(sizes, dtype=float)


# ── In-memory colocalization (replicates run_inference.py Phase 3A+3B) ────────

def _run_tile_colocalization(per_ch_vol_lists, soma_ch_ids, tf_ch_ids,
                             zl_soma_cfg, zl_tf_cfg):
    """Run colocalization in-memory on per-tile z-linked vol_lists.

    Phase A: soma × soma 3D IoU matching + cross-class dedup
    Phase B: soma × TF GMM annotation

    Returns merged_soma_vols list (dicts with updated 'class' field).
    """
    import copy

    iou_thresh_3d = zl_soma_cfg.get('iou_thresh_3d', 0.15)
    z_pad_3d      = zl_soma_cfg.get('z_pad_3d', 2)
    cross_iou     = zl_soma_cfg.get('cross_class_iou_thresh', 0.5)
    p_thresh      = zl_tf_cfg.get('gmm_p_thresh', 0.5)

    if not soma_ch_ids:
        return []

    # Deep-copy so we don't mutate the shared vol_lists
    merged = [copy.copy(c) for c in per_ch_vol_lists.get(soma_ch_ids[0], [])]

    # Phase A: soma × soma
    for cid_b in soma_ch_ids[1:]:
        cells_b = [copy.copy(c) for c in per_ch_vol_lists.get(cid_b, [])]
        matched_pairs, unmatched_a, unmatched_b = match_soma_3d_iou(
            merged, cells_b, iou_thresh=iou_thresh_3d, z_pad=z_pad_3d
        )
        for a_cell, b_cell in matched_pairs:
            a_cell['class'] = _merge_class(a_cell['class'], b_cell['class'])
        merged = merged + unmatched_b

    merged = suppress_cross_class_overlap(merged, iou_thresh=cross_iou, z_pad=z_pad_3d)

    # Phase B: TF GMM annotation
    for cid in tf_ch_ids:
        tf_vols = per_ch_vol_lists.get(cid, [])
        if tf_vols and merged:
            merged = annotate_soma_with_tf_gmm(merged, tf_vols, p_thresh=p_thresh)

    return merged


# ── 从 TeraStitcher XML 获取 tile 的全局像素偏移 ─────────────────────────────

def _get_tile_offset(channel_dir, tile_name):
    """从 xml_merging.xml 读取 tile 的全局偏移量。

    XY：combine_predictions 将 tile 本地坐标加上 (ABS_H - x_min, ABS_V - y_min)。
    Z ：combine_predictions 将 tile 本地 z 减去 z0 = (z_start - ABS_D)。

    返回 (tile_x0, tile_y0, tile_z0)，其中
      tile_x0 = ABS_H（假设 x_min=0，即 xml 中最小 ABS_H 为 0）
      tile_y0 = ABS_V（同上）
      tile_z0 = z_start - ABS_D（需加回 coloc z 值才还原 tile 本地 z）
    找不到 XML 或 tile 时返回 (0, 0, 0)。
    """
    import xml.etree.ElementTree as ET
    xml_path = os.path.join(channel_dir, 'xml_merging.xml')
    if not os.path.isfile(xml_path):
        return 0, 0, 0
    try:
        root = ET.parse(xml_path).getroot()
        stacks = list(root.find('STACKS'))
        z_start = max(int(s.get('ABS_D', 0)) for s in stacks)
        x_min   = min(int(s.get('ABS_H', 0)) for s in stacks)
        y_min   = min(int(s.get('ABS_V', 0)) for s in stacks)
        for stack in stacks:
            if os.path.basename(stack.get('DIR_NAME', '')) == tile_name:
                abs_h = int(stack.get('ABS_H', 0))
                abs_v = int(stack.get('ABS_V', 0))
                abs_d = int(stack.get('ABS_D', 0))
                return abs_h - x_min, abs_v - y_min, z_start - abs_d
    except Exception:
        pass
    return 0, 0, 0


# ── Overlap margin from TeraStitcher XML ─────────────────────────────────────

def _get_tile_overlap_margins(channel_dir, tile_name, canvas_w, canvas_h):
    """Return (left_margin, top_margin) in tile-local pixels.

    combine_predictions suppresses boxes in the region that overlaps with the
    left neighbour (col-1) and the top neighbour (row-1).  The suppressed band
    widths are:
      left_margin = (left_ABS_H + canvas_w) - cur_ABS_H   [pixels into this tile]
      top_margin  = (top_ABS_V  + canvas_h) - cur_ABS_V

    Returns (0, 0) when there are no neighbours or the XML is missing.
    """
    import xml.etree.ElementTree as ET
    xml_path = os.path.join(channel_dir, 'xml_merging.xml')
    if not os.path.isfile(xml_path):
        return 0, 0
    try:
        root  = ET.parse(xml_path).getroot()
        stacks = list(root.find('STACKS'))
        info = {}
        for s in stacks:
            name = os.path.basename(s.get('DIR_NAME', ''))
            info[name] = {
                'row': int(s.get('ROW', 0)),
                'col': int(s.get('COL', 0)),
                'h':   int(s.get('ABS_H', 0)),
                'v':   int(s.get('ABS_V', 0)),
            }
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


# ── CSV → napari shapes (tile-local coordinates, no global offset) ──────────

def _load_tile_csv_shapes(csv_path, z_range=None):
    """
    Load a per-tile detection CSV and return (shapes_list, colors_list) in
    tile-local (z, y, x) napari coordinates.
    """
    if not os.path.isfile(csv_path):
        return [], []

    df = pd.read_csv(
        csv_path,
        names=['slice_name', 'x1', 'y1', 'x2', 'y2', 'class', 'score', 'mean', 'z'],
        skiprows=1,
    )
    if df.empty:
        return [], []

    df['z'] = df['z'].astype(float).astype(int) - 1  # CSV z is 1-indexed; convert to 0-indexed

    if z_range is not None:
        df['z_local'] = df['z'] - z_range[0]
        df = df[(df['z_local'] >= 0) & (df['z_local'] < z_range[1] - z_range[0])]
        if df.empty:
            return [], []
        z_col = df['z_local'].values
    else:
        z_col = df['z'].values

    n = len(df)
    arr = np.empty((n, 4, 3), dtype=np.float64)
    y1 = df['y1'].values.astype(float)
    y2 = df['y2'].values.astype(float)
    x1 = df['x1'].values.astype(float)
    x2 = df['x2'].values.astype(float)

    arr[:, 0] = np.column_stack([z_col, y1, x1])
    arr[:, 1] = np.column_stack([z_col, y1, x2])
    arr[:, 2] = np.column_stack([z_col, y2, x2])
    arr[:, 3] = np.column_stack([z_col, y2, x1])

    colors = [_color(c) for c in df['class']]
    return list(arr), colors


# ── Offset JSON loading ──────────────────────────────────────────────────────

def load_offsets(align_dir, tile_name):
    """Load per-tile alignment offsets from 0_channel_alignment/{tile_name}_offsets.json."""
    json_path = os.path.join(align_dir, f"{tile_name}_offsets.json")
    if not os.path.isfile(json_path):
        return {}
    with open(json_path, encoding='utf-8') as f:
        return json.load(f)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Pre-align channel alignment visualizer (napari).'
    )
    parser.add_argument(
        '--config', default=os.path.join(project_root, 'config', 'config.json'),
        help='Path to main config.json (default: config/config.json)',
    )
    parser.add_argument(
        '--tile', default=None,
        help='Tile directory name (e.g. 000000_001000_000000).',
    )
    parser.add_argument(
        '--z-range', nargs=2, type=int, default=None, metavar=('START', 'END'),
        help='Z-slice range to load, e.g. --z-range 400 500.',
    )
    parser.add_argument(
        '--no-images', action='store_true',
        help='Skip raw image loading (faster, shows boxes only).',
    )
    parser.add_argument(
        '--show-before', action='store_true',
        help='Also add [raw] layers with pre-alignment detections (hidden by default).',
    )
    parser.add_argument(
        '--show-zlinked', action='store_true',
        help='Add [zlinked] layers showing per-slice boxes of z-linked 3D cells.',
    )
    parser.add_argument(
        '--show-coloc', action='store_true',
        help='Run in-memory colocalization and add [coloc] layer.',
    )
    parser.add_argument(
        '--list-tiles', action='store_true',
        help='Print available tile names and exit.',
    )
    parser.add_argument(
        '--spheres', action='store_true',
        help='Add 3-D sphere Points layers per channel. napari switches to 3-D view.',
    )
    args = parser.parse_args()

    # ── Load vis_config_prealign.json first (needed to resolve runtime_config) ─
    vis_cfg_path = os.path.join(project_root, 'config','vis' ,'vis_config_prealign.json')
    vis_cfg = load_config(vis_cfg_path) if os.path.isfile(vis_cfg_path) else {}

    # ── Resolve main config: CLI --config > pATHRESULT/runtime_config.json > default
    config_path = args.config
    if not args.config or args.config == os.path.join(project_root, 'config', 'config.json'):
        pat = (vis_cfg.get('paths') or {}).get('pATHRESULT')
        if pat:
            runtime_cfg_path = os.path.join(pat, 'runtime_config.json')
            if os.path.isfile(runtime_cfg_path):
                config_path = runtime_cfg_path
                print(f"[config] Using runtime_config.json from result dir: {runtime_cfg_path}")

    config = load_config(config_path)
    paths  = config['paths']
    pa_cfg = config.get('pre_align_params', {})
    routing_config = [ch for ch in config['channels_routing'] if ch.get('active', True)]

    for key, val in vis_cfg.get('paths', {}).items():
        if val:
            paths[key] = val

    no_images    = args.no_images    or vis_cfg.get('no_images',     False)
    show_before  = args.show_before  or vis_cfg.get('show_before',   False)
    show_spheres = args.spheres      or vis_cfg.get('spheres',       False)
    show_zlinked = args.show_zlinked or vis_cfg.get('show_zlinked',  False)
    show_coloc   = args.show_coloc   or vis_cfg.get('show_coloc',    False)

    tile_arg    = args.tile     or vis_cfg.get('tile',    None)
    z_range_arg = args.z_range  or vis_cfg.get('z_range', None)
    z_start_cfg = vis_cfg.get('z_start', None)
    z_count_cfg = vis_cfg.get('z_count',  None)
    s4_groups   = vis_cfg.get('s4_groups', [])

    outline_width      = vis_cfg.get('outline_width',      2)
    aligned_opacity    = vis_cfg.get('aligned_opacity',    0.8)
    raw_opacity        = vis_cfg.get('raw_opacity',        0.6)
    zlinked_opacity    = vis_cfg.get('zlinked_opacity',    0.7)
    coloc_opacity      = vis_cfg.get('coloc_opacity',      0.9)
    sphere_opacity     = vis_cfg.get('sphere_opacity',     0.6)
    sphere_raw_opacity = vis_cfg.get('sphere_raw_opacity', 0.3)
    sphere_colors_cfg  = vis_cfg.get('sphere_colors',      {})

    base_res  = paths['pATHRESULT']
    align_dir = os.path.join(base_res, '0_channel_alignment')
    raw_dir   = os.path.join(base_res, '1_tile_2d_raw')

    anchor_ch  = routing_config[0]
    anchor_dir = os.path.abspath(paths[anchor_ch['dir_key']])

    _, tile_paths = listTile(anchor_dir)

    if args.list_tiles:
        print(f"Available tiles ({len(tile_paths)}):")
        for i, p in enumerate(tile_paths):
            print(f"  [{i:3d}] {os.path.basename(p)}")
        return

    # ── Select tile ───────────────────────────────────────────────────────────
    if tile_arg is not None:
        tile_name = tile_arg
        tile_path = next((p for p in tile_paths if os.path.basename(p) == tile_name), None)
        if tile_path is None:
            sys.exit(f"Tile '{tile_name}' not found in {anchor_dir}.")
    else:
        print(f"\nAvailable tiles ({len(tile_paths)}):")
        for i, p in enumerate(tile_paths):
            print(f"  [{i:3d}] {os.path.basename(p)}")
        raw = input("\nEnter tile index: ").strip()
        idx = int(raw) if raw else 0
        tile_path = tile_paths[idx]
        tile_name = os.path.basename(tile_path)

    print(f"\nTile: {tile_name}")

    # ── Determine z-range ─────────────────────────────────────────────────────
    def _list_tiffs(path):
        return sorted(
            f for f in os.listdir(path)
            if f.lower().endswith(('.tif', '.tiff'))
        ) if os.path.isdir(path) else []

    if z_range_arg is not None:
        z_range = tuple(z_range_arg)
    elif z_start_cfg is not None or z_count_cfg is not None:
        tiffs   = _list_tiffs(tile_path)
        total_z = len(tiffs)
        s = int(z_start_cfg) if z_start_cfg is not None else 0
        c = int(z_count_cfg)  if z_count_cfg  is not None else total_z - s
        z_range = (max(0, s), min(total_z, s + c))
    else:
        sample_count = pa_cfg.get('sample_z_center_count', 50)
        half = sample_count // 2
        tiffs = _list_tiffs(tile_path)
        total_z  = len(tiffs)
        z_center = total_z // 2
        z_range  = (max(0, z_center - half), min(total_z, z_center + half))

    print(f"Z-range: {z_range[0]} – {z_range[1]} ({z_range[1] - z_range[0]} slices)\n")

    # ── Load alignment offsets ────────────────────────────────────────────────
    offsets   = load_offsets(align_dir, tile_name)
    has_aligned = bool(offsets)

    print("=== Channel alignment summary ===")
    for ch in routing_config:
        cid = ch['id']
        if cid in offsets:
            o = offsets[cid]
            tag = " [REFERENCE]" if (o['dx'] == 0 and o['dy'] == 0 and o['dz'] == 0
                                      and o['iou_score'] >= 1.0) else ""
            print(f"  [{cid:8s}]  shift=({o['dx']:+d}, {o['dy']:+d}, {o['dz']:+d})"
                  f"   iou={o['iou_score']:.4f}{tag}")
        else:
            print(f"  [{cid:8s}]  (no offset data — alignment not run yet)")
    print("=================================\n")

    # ── Canvas shape ──────────────────────────────────────────────────────────
    canvas_z = z_range[1] - z_range[0]
    tiffs = _list_tiffs(tile_path)
    if tiffs:
        import cv2
        _img = cv2.imread(os.path.join(tile_path, tiffs[0]), cv2.IMREAD_ANYDEPTH)
        canvas_h = _img.shape[0] if _img is not None else 2048
        canvas_w = _img.shape[1] if _img is not None else 2048
    else:
        canvas_h, canvas_w = 2048, 2048
    canvas_shape = (canvas_z, canvas_h, canvas_w)

    # ── Build napari viewer ───────────────────────────────────────────────────
    viewer = napari.Viewer(title=f"Pre-Align QC — {tile_name}")

    # ── Z-linker params ───────────────────────────────────────────────────────
    zl_cfg   = config.get('z_linker', {})
    zl_soma  = zl_cfg.get('soma', {})
    zl_tf    = zl_cfg.get('tf',   {})

    soma_ch_ids = [ch['id'] for ch in routing_config if ch.get('type', 'soma') == 'soma']
    tf_ch_ids   = [ch['id'] for ch in routing_config if ch.get('type') == 'tf']

    # ── Run z-linking once if any downstream layer needs it ──────────────────
    need_zlink = show_spheres or show_zlinked or show_coloc
    per_ch_vol_lists = {}

    if need_zlink:
        print("Running z-linking in memory...")
        anchor_root = anchor_dir
        for ch in routing_config:
            cid     = ch['id']
            ctype   = ch.get('type', 'soma')
            ch_base = os.path.abspath(paths[ch['dir_key']])
            rel     = os.path.relpath(tile_path, anchor_root)
            # aligned CSV first, fallback to raw
            aligned_csv = os.path.join(align_dir, f"{tile_name}_{cid}_result.csv")
            if not os.path.isfile(aligned_csv) and not has_aligned:
                aligned_csv = os.path.join(raw_dir, f"{tile_name}_{cid}_result.csv")
            zl_params = zl_soma if ctype == 'soma' else zl_tf
            vol_list = _run_zlink_for_csv(
                aligned_csv, z_range,
                iou_thresh=zl_params.get('iou_thresh', 0.35),
                min_z_layers=zl_params.get('min_z_layers', 1),
                max_cell_z_span=zl_params.get('max_cell_z_span', 5),
            )
            per_ch_vol_lists[cid] = vol_list
            print(f"  [{cid}] z-linked: {len(vol_list)} cells")
        print()

    # ── Image layers ──────────────────────────────────────────────────────────
    if not no_images:
        anchor_root = anchor_dir
        for ch in routing_config:
            cid      = ch['id']
            ch_base  = os.path.abspath(paths[ch['dir_key']])
            rel      = os.path.relpath(tile_path, anchor_root)
            tile_dir = os.path.join(ch_base, rel)
            print(f"[img] Loading {cid}  ({tile_dir}) ...")
            vol = load_volume(tile_dir, z_range=z_range)
            if vol is None:
                print(f"  → not found, skipping")
                continue
            canvas_shape = (vol.shape[0], vol.shape[1], vol.shape[2])
            if cid in offsets:
                o = offsets[cid]
                dx, dy, dz = o['dx'], o['dy'], o['dz']
                if dx != 0 or dy != 0 or dz != 0:
                    vol = _shift_volume(vol, dx, dy, dz)
                    print(f"  → shifted ({dx:+d}, {dy:+d}, {dz:+d})")
            vis = _ch_vis(cid)
            viewer.add_image(vol, name=f"[img] {cid}", **vis)
            print(f"  → {vol.shape}")

    # ── [aligned] Detection box layers ────────────────────────────────────────
    for ch in routing_config:
        cid     = ch['id']
        iou_str = (f"  iou={offsets[cid]['iou_score']:.3f}" if cid in offsets else "")

        aligned_csv = os.path.join(align_dir, f"{tile_name}_{cid}_result.csv")
        if not os.path.isfile(aligned_csv) and not has_aligned:
            aligned_csv = os.path.join(raw_dir, f"{tile_name}_{cid}_result.csv")

        shapes_a, colors_a = _load_tile_csv_shapes(aligned_csv, z_range)
        if shapes_a:
            _add_labels_layer(
                viewer, shapes_a, colors_a, canvas_shape,
                name=f"[aligned] {cid}{iou_str}",
                visible=True,
                opacity=aligned_opacity,
                outline_width=outline_width,
            )
            print(f"[aligned] {cid}: {len(shapes_a)} boxes")

        if show_before:
            raw_csv = os.path.join(raw_dir, f"{tile_name}_{cid}_result.csv")
            shapes_r, colors_r = _load_tile_csv_shapes(raw_csv, z_range)
            if shapes_r:
                _add_labels_layer(
                    viewer, shapes_r, colors_r, canvas_shape,
                    name=f"[raw] {cid}",
                    visible=False,
                    opacity=raw_opacity,
                    outline_width=outline_width,
                )
                print(f"[raw]     {cid}: {len(shapes_r)} boxes (hidden)")

    # ── [zlinked] Center-z box layers from z-linked cells ────────────────────
    if show_zlinked:
        for ch in routing_config:
            cid      = ch['id']
            vol_list = per_ch_vol_lists.get(cid, [])
            shapes_z, colors_z = _vol_list_to_center_z_shapes(vol_list, z_range)
            if shapes_z:
                _add_labels_layer(
                    viewer, shapes_z, colors_z, canvas_shape,
                    name=f"[zlinked] {cid}",
                    visible=True,
                    opacity=zlinked_opacity,
                    outline_width=outline_width,
                )
                print(f"[zlinked] {cid}: {len(shapes_z)} cells at center-z")

    # ── [spheres] 3D sphere layers ────────────────────────────────────────────
    if show_spheres:
        for ch in routing_config:
            cid      = ch['id']
            vol_list = per_ch_vol_lists.get(cid, [])
            color    = (sphere_colors_cfg.get(cid)
                        or SPHERE_COLORS.get(cid, _DEFAULT_SPHERE_COLOR))
            centers, sizes = _vol_list_to_points(vol_list, z_range)
            if centers is not None:
                viewer.add_points(
                    centers, size=sizes,
                    face_color=[color] * len(centers),
                    border_width=0,
                    opacity=sphere_opacity,
                    n_dimensional=True,
                    name=f"[spheres] {cid}",
                )
                print(f"[spheres]     {cid}: {len(centers)} cells")

            if show_before:
                raw_csv = os.path.join(raw_dir, f"{tile_name}_{cid}_result.csv")
                vol_list_r = _run_zlink_for_csv(
                    raw_csv, z_range,
                    iou_thresh=zl_soma.get('iou_thresh', 0.35) if ch.get('type', 'soma') == 'soma' else zl_tf.get('iou_thresh', 0.25),
                    min_z_layers=1,
                    max_cell_z_span=zl_soma.get('max_cell_z_span', 5) if ch.get('type', 'soma') == 'soma' else zl_tf.get('max_cell_z_span', 3),
                )
                centers_r, sizes_r = _vol_list_to_points(vol_list_r, z_range)
                if centers_r is not None:
                    viewer.add_points(
                        centers_r, size=sizes_r,
                        face_color=[color] * len(centers_r),
                        border_width=0,
                        opacity=sphere_raw_opacity,
                        n_dimensional=True,
                        visible=False,
                        name=f"[spheres-raw] {cid}",
                    )
                    print(f"[spheres-raw] {cid}: {len(centers_r)} cells (hidden)")

        viewer.dims.ndisplay = 3
        print("\nSphere view active — napari is in 3-D mode.")

    # ── [coloc] Colocalization layers ─────────────────────────────────────────
    if show_coloc:
        coloc_csv = os.path.join(base_res, '4_colocalization', 'coloc_result.csv')
        tile_x0, tile_y0, tile_z0 = _get_tile_offset(anchor_dir, tile_name)
        print(f"  [coloc] tile 偏移: x0={tile_x0}, y0={tile_y0}, z0={tile_z0}")

        # Auto-detect all marker combos from CSV; fall back to config s4_groups
        auto_groups = _auto_coloc_groups(
            coloc_csv, outline_width=outline_width, dash_size=8
        )
        active_groups = auto_groups if auto_groups else s4_groups

        grp_results = _load_coloc_s4_shapes(
            coloc_csv, tile_name, z_range, active_groups,
            tile_x0=tile_x0, tile_y0=tile_y0, tile_z0=tile_z0,
        )

        if grp_results is None:
            # CSV not found — fall back to in-memory computation
            print("\n[coloc] coloc_result.csv not found, running in-memory colocalization...")
            try:
                coloc_vols = _run_tile_colocalization(
                    per_ch_vol_lists,
                    soma_ch_ids=soma_ch_ids,
                    tf_ch_ids=tf_ch_ids,
                    zl_soma_cfg=zl_soma,
                    zl_tf_cfg=zl_tf,
                )
                grp_results = _vol_list_to_s4_shapes(coloc_vols, z_range, s4_groups)
                print(f"  In-memory: {len(coloc_vols)} soma cells")
            except Exception as exc:
                print(f"[coloc] WARNING: in-memory colocalization failed — {exc}")
                grp_results = []
        else:
            print(f"\n[coloc] Loaded from {coloc_csv}")
            if auto_groups:
                print(f"  Auto-detected {len(auto_groups)} marker combination(s): "
                      + ", ".join(g['name'] for g in auto_groups))

        if grp_results:
            any_shown = False
            for grp, shapes_c, colors_c, dash_sizes_c in grp_results:
                if shapes_c:
                    n_solid = sum(1 for d in dash_sizes_c if d == 0)
                    n_dash  = len(dash_sizes_c) - n_solid
                    _add_labels_layer(
                        viewer, shapes_c, colors_c, canvas_shape,
                        dash_sizes=dash_sizes_c,
                        name=f"[coloc] {grp['name']}",
                        visible=True,
                        opacity=coloc_opacity,
                        outline_width=grp.get('outline_width', outline_width),
                    )
                    print(f"[coloc] {grp['name']}: {len(shapes_c)} boxes "
                          f"({n_solid} neuron solid, {n_dash} glia dashed)")
                    any_shown = True
            if not any_shown:
                print("[coloc]: no cells in z-range for any group")

        # ── Overlap suppression overlay ───────────────────────────────────────
        left_m, top_m = _get_tile_overlap_margins(
            anchor_dir, tile_name, canvas_w, canvas_h
        )
        if left_m > 0 or top_m > 0:
            ovl = np.zeros((canvas_z, canvas_h, canvas_w), dtype=np.uint8)
            if left_m > 0:
                ovl[:, :, :left_m] = 255
            if top_m > 0:
                ovl[:, :top_m, :] = 255
            viewer.add_image(
                ovl,
                name=f"[overlap] suppressed zone (left={left_m}px top={top_m}px)",
                colormap='cyan',
                opacity=0.25,
                blending='additive',
                visible=True,
            )
            print(f"[overlap] suppressed zone: left={left_m}px, top={top_m}px "
                  f"(boxes in this region belong to neighbour tiles)")

    print("\nnapari viewer ready.")
    print("Tips:")
    print("  • Compare [aligned] layers across channels to assess overlap quality.")
    if show_zlinked:
        print("  • [zlinked] shows 3D cells at each z-slice they were detected in.")
    if show_before:
        print("  • Enable [raw] layers to see pre-alignment positions.")
    if show_spheres:
        print("  • Rotate the 3-D view to inspect channel co-localisation as coloured spheres.")
    if show_coloc:
        print("  • [coloc] shows in-memory soma+TF colocalization result for this tile.")

    viewer.reset_view()
    napari.run()


if __name__ == '__main__':
    main()
