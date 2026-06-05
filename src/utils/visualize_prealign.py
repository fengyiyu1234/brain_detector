# -*- coding: utf-8 -*-
"""
Pre-align channel alignment visualizer.

Opens a napari viewer for a single tile showing:
  - Raw images for each active channel (tile-local coordinates)
  - Detection bounding boxes AFTER alignment (from 0_channel_alignment/)
  - Detection bounding boxes BEFORE alignment (from 1_tile_2d_raw/, hidden by default)

Allows the user to assess:
  1. How well the aligned channels overlap (toggle [aligned] layers per channel)
  2. Whether detections match the raw image signal (toggle [img] + [aligned] layers)

Usage
-----
  python src/utils/visualize_prealign.py --tile 000000_001000_000000
  python src/utils/visualize_prealign.py --tile 000000_001000_000000 --z-range 400 450
  python src/utils/visualize_prealign.py --tile 000000_001000_000000 --no-images
  python src/utils/visualize_prealign.py --tile 000000_001000_000000 --show-before
  python src/utils/visualize_prealign.py --list-tiles
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

# Re-use image loading and rasterization from the existing visualizer
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


def _ch_vis(ch_id):
    return CHANNEL_VIS.get(ch_id, _EXTRA_VIS.get(ch_id, DEFAULT_VIS))


# ── CSV → napari shapes (tile-local coordinates, no global offset) ──────────

def _load_tile_csv_shapes(csv_path, z_range=None):
    """
    Load a per-tile detection CSV and return (shapes_list, colors_list) in
    tile-local (z, y, x) napari coordinates.

    Applies optional z_range filter so only detections within the visible
    z-window are loaded.

    CSV columns: slice_name, x1, y1, x2, y2, class, score, mean, z
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

    df['z'] = df['z'].astype(float).astype(int)

    if z_range is not None:
        # Remap z to volume-local index: loaded_z = original_z - z_range[0]
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
    """
    Load per-tile alignment offsets from 0_channel_alignment/{tile_name}_offsets.json.
    Returns dict  ch_id → {"dx": int, "dy": int, "dz": int, "iou_score": float}
    or empty dict if not found.
    """
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
        '--config', default='D:/Fengyi/brain_detector/config/config.json',
        help='Path to config.json',
    )
    parser.add_argument(
        '--tile', default=None,
        help='Tile directory name (e.g. 000000_001000_000000). '
             'Omit for interactive selection.',
    )
    parser.add_argument(
        '--z-range', nargs=2, type=int, default=None, metavar=('START', 'END'),
        help='Z-slice range to load, e.g. --z-range 400 500. '
             'Defaults to the sample_z_center_count window from config.',
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
        '--list-tiles', action='store_true',
        help='Print available tile names and exit.',
    )
    args = parser.parse_args()

    config = load_config(args.config)
    paths  = config['paths']
    dp     = config['detection_params']
    pa_cfg = config.get('pre_align_params', {})
    routing_config = [ch for ch in config['channels_routing'] if ch.get('active', True)]

    base_res   = paths['pATHRESULT']
    align_dir  = os.path.join(base_res, '0_channel_alignment')
    raw_dir    = os.path.join(base_res, '1_tile_2d_raw')

    anchor_ch  = routing_config[0]
    anchor_dir = os.path.abspath(paths[anchor_ch['dir_key']])

    _, tile_paths = listTile(anchor_dir)

    if args.list_tiles:
        print(f"Available tiles ({len(tile_paths)}):")
        for i, p in enumerate(tile_paths):
            print(f"  [{i:3d}] {os.path.basename(p)}")
        return

    # ── Select tile ───────────────────────────────────────────────────────────
    if args.tile is not None:
        tile_name = args.tile
        tile_path = None
        for p in tile_paths:
            if os.path.basename(p) == tile_name:
                tile_path = p
                break
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
    if args.z_range is not None:
        z_range = tuple(args.z_range)
    else:
        # Use central z window from config (same window used for alignment)
        sample_count = pa_cfg.get('sample_z_center_count', 50)
        half = sample_count // 2

        # Count TIFF files in anchor tile directory to find total z depth
        anchor_tile_dir = tile_path
        tiffs = sorted(
            f for f in os.listdir(anchor_tile_dir)
            if f.lower().endswith(('.tif', '.tiff'))
        ) if os.path.isdir(anchor_tile_dir) else []
        total_z = len(tiffs)
        z_center = total_z // 2
        z_range = (max(0, z_center - half), min(total_z, z_center + half))

    print(f"Z-range: {z_range[0]} – {z_range[1]} "
          f"({z_range[1] - z_range[0]} slices)\n")

    # ── Load alignment offsets ────────────────────────────────────────────────
    offsets = load_offsets(align_dir, tile_name)
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

    # ── Determine canvas shape (Z, H, W) from one channel ────────────────────
    canvas_z = z_range[1] - z_range[0]
    _sample_dir = tile_path   # anchor channel tile dir
    _tiffs = sorted(
        f for f in os.listdir(_sample_dir)
        if f.lower().endswith(('.tif', '.tiff'))
    ) if os.path.isdir(_sample_dir) else []
    if _tiffs:
        import cv2
        _img = cv2.imread(os.path.join(_sample_dir, _tiffs[0]), cv2.IMREAD_ANYDEPTH)
        canvas_h = _img.shape[0] if _img is not None else 2048
        canvas_w = _img.shape[1] if _img is not None else 2048
    else:
        canvas_h, canvas_w = 2048, 2048
    canvas_shape = (canvas_z, canvas_h, canvas_w)

    # ── Build napari viewer ───────────────────────────────────────────────────
    viewer = napari.Viewer(title=f"Pre-Align QC — {tile_name}")

    # ── Image layers ──────────────────────────────────────────────────────────
    if not args.no_images:
        anchor_root = anchor_dir
        for ch in routing_config:
            cid     = ch['id']
            ch_base = os.path.abspath(paths[ch['dir_key']])
            rel     = os.path.relpath(tile_path, anchor_root)
            tile_dir = os.path.join(ch_base, rel)
            print(f"[img] Loading {cid}  ({tile_dir}) ...")
            vol = load_volume(tile_dir, z_range=z_range)
            if vol is None:
                print(f"  → not found, skipping")
                continue
            canvas_shape = (vol.shape[0], vol.shape[1], vol.shape[2])
            vis = _ch_vis(cid)
            viewer.add_image(vol, name=f"[img] {cid}", **vis)
            print(f"  → {vol.shape}")

    # ── Detection box layers ──────────────────────────────────────────────────
    for ch in routing_config:
        cid = ch['id']
        iou_str = (f"  iou={offsets[cid]['iou_score']:.3f}" if cid in offsets else "")

        # [aligned] layer: from 0_channel_alignment/
        aligned_csv = os.path.join(align_dir, f"{tile_name}_{cid}_result.csv")
        # Fallback to raw if aligned CSV missing (alignment not yet run)
        if not os.path.isfile(aligned_csv) and not has_aligned:
            aligned_csv = os.path.join(raw_dir, f"{tile_name}_{cid}_result.csv")

        shapes_a, colors_a = _load_tile_csv_shapes(aligned_csv, z_range)
        if shapes_a:
            _add_labels_layer(
                viewer, shapes_a, colors_a, canvas_shape,
                name=f"[aligned] {cid}{iou_str}",
                visible=True,
                opacity=0.8,
            )
            print(f"[aligned] {cid}: {len(shapes_a)} boxes")

        # [raw] layer: from 1_tile_2d_raw/ (hidden unless --show-before)
        if args.show_before:
            raw_csv = os.path.join(raw_dir, f"{tile_name}_{cid}_result.csv")
            shapes_r, colors_r = _load_tile_csv_shapes(raw_csv, z_range)
            if shapes_r:
                _add_labels_layer(
                    viewer, shapes_r, colors_r, canvas_shape,
                    name=f"[raw] {cid}",
                    visible=False,
                    opacity=0.6,
                )
                print(f"[raw]     {cid}: {len(shapes_r)} boxes (hidden)")

    print("\nnapari viewer ready. Use layer list on the left to toggle visibility.")
    print("Tips:")
    print("  • Compare [aligned] layers across channels to assess overlap quality.")
    print("  • Toggle [img] layers to check if detections match image signal.")
    if args.show_before:
        print("  • Enable [raw] layers to see pre-alignment positions for comparison.")

    viewer.reset_view()
    napari.run()


if __name__ == '__main__':
    main()
