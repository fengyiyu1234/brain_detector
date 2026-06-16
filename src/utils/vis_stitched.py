#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
vis_stitched.py — Draw detection boxes onto stitched TIFF slices, save as 16-bit PNG.

Edit CONFIG below. No CLI arguments needed.

Outputs:
  {output_dir}/1_raw2d/{CH}/z{z:04d}.png    — per-channel grayscale + 2D raw detections
  {output_dir}/2_zlinked/{CH}/z{z:04d}.png  — per-channel grayscale + z-linked detections
  {output_dir}/3_coloc/{combo}/z{z:04d}.png — RGB composite + colocalization boxes
"""

import os
import sys
import re

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np
import pandas as pd

from src.utils.io import loadTeraxml

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS — edit this block before running
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # Per-channel paths: name → Y_X block folder containing per-slice .tif files
    'channels': {
        'RFP':  r'y:\Fengyi\EGFR_brain\batch4\sample5\stitched\RFP\RES(4003x3150x321)\422850\422850_302400',
        'GFP':  r'y:\Fengyi\EGFR_brain\batch4\sample5\stitched\GFP_3\RES(4003x3150x321)\422850\422850_302400',
        'Sox9': r'y:\Fengyi\EGFR_brain\batch4\sample5\stitched\sox9\RES(4003x3150x321)\422850\422850_302400',
    },
    # Colors for RGB composite (BGR, 0–255 scale)
    'channel_colors': {
        'RFP':  (0,   0,   255),   # red
        'GFP':  (0,   255, 0  ),   # green
        'Sox9': (255, 100, 0  ),   # blue-ish
    },
    'z_start':    100,        # 0-based file index of first slice to load
    'n_layers':   10,          # number of consecutive slices to load
    'result_dir': r'y:\Fengyi\EGFR_brain\batch4\sample5\detection_results_stardist',
    'xml':        r'y:\Fengyi\EGFR_brain\batch4\sample5\numorph_align\aligned\RFP\xml_merging.xml',
    'xy_scale':   None,       # None = auto-detect from RES folder name + XML
    'z_scale':    None,       # None = auto-detect from file count + XML
    'output_dir': r'y:\Fengyi\EGFR_brain\batch4\sample5\vis_output2',
    'box_thickness':  4,
    'normalise_pct': (0.1, 99.9),
}
# ─────────────────────────────────────────────────────────────────────────────


# ── Slice discovery ───────────────────────────────────────────────────────────

def _z_from_fname(fname):
    stem = os.path.splitext(fname)[0]
    try:
        return int(stem.split('_')[-1])
    except ValueError:
        return -1


def list_block_slices(block_dir):
    """Return absolute .tif paths in block_dir sorted by their Z value."""
    if not os.path.isdir(block_dir):
        raise FileNotFoundError(f"Block dir not found: {block_dir}")
    files = sorted(
        [f for f in os.listdir(block_dir) if f.lower().endswith('.tif')],
        key=_z_from_fname,
    )
    if not files:
        raise RuntimeError(f"No .tif files found in: {block_dir}")
    return [os.path.join(block_dir, f) for f in files]


# ── Scale auto-detection ──────────────────────────────────────────────────────

def _find_res_dims(path):
    """Walk path components upward to find RES(HxWxZ). Returns (H, W, Z) or (None,None,None)."""
    for part in reversed(os.path.normpath(path).split(os.sep)):
        m = re.search(r'RES\((\d+)x(\d+)x(\d+)\)', part)
        if m:
            return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None, None, None


def resolve_scales(ch_paths, xml_path, xy_override, z_override):
    """Return (xy_scale, z_scale) with auto-detection from RES folder + XML."""
    if xy_override is not None and z_override is not None:
        return float(xy_override), float(z_override)

    first_dir = next(iter(ch_paths.values()))
    _, W_lv, _ = _find_res_dims(first_dir)

    if xml_path and os.path.isfile(xml_path):
        _, _, W_global, Z_global, _, _ = loadTeraxml(xml_path)
        Z_lv = len(list_block_slices(first_dir))
        xy_s = float(xy_override) if xy_override is not None else (
            W_lv / max(W_global, 1) if W_lv else 1.0
        )
        z_s = float(z_override) if z_override is not None else Z_lv / max(Z_global, 1)
        print(f"  Scales: xy={xy_s:.5f}  z={z_s:.5f}  "
              f"(W_lv={W_lv}, W_global={W_global}, Z_lv={Z_lv}, Z_global={Z_global})")
        return xy_s, z_s

    print("  Warning: no XML — xy_scale=1.0, z_scale=1.0 (detection overlay may be misaligned)")
    return float(xy_override or 1.0), float(z_override or 1.0)


# ── Image loading / normalization ─────────────────────────────────────────────

def read_slice_raw(fpath):
    """Read a single-channel TIFF → (H, W) native-dtype array. None on failure."""
    try:
        import tifffile
        arr = tifffile.imread(fpath)
        while arr.ndim > 2:
            arr = arr[0] if arr.shape[0] <= 4 else arr[:, :, 0]
        return arr
    except Exception:
        pass
    img = cv2.imread(fpath, cv2.IMREAD_ANYDEPTH)
    if img is None:
        return None
    return img[:, :, 0] if img.ndim == 3 else img


def stretch_to_uint16(arr, pct_lo=0.5, pct_hi=99.5):
    """Percentile stretch → uint16 [0, 65535]."""
    f = arr.astype(np.float32)
    lo, hi = np.percentile(f, [pct_lo, pct_hi])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((f - lo) / (hi - lo) * 65535.0, 0, 65535).astype(np.uint16)


# ── RGB composite ─────────────────────────────────────────────────────────────

def make_rgb_composite(ch_gray_u16, ch_colors_8bit):
    """Blend per-channel uint16 grays into a uint16 BGR composite.

    ch_gray_u16: dict {ch: (H,W) uint16}
    ch_colors_8bit: dict {ch: (b,g,r) in 0–255}
    Returns (H,W,3) uint16 BGR.
    """
    H, W = next(iter(ch_gray_u16.values())).shape
    acc = np.zeros((H, W, 3), dtype=np.float64)
    for ch, gray in ch_gray_u16.items():
        color = ch_colors_8bit.get(ch, (255, 255, 255))
        g_norm = gray.astype(np.float64) / 65535.0
        for i, c in enumerate(color):
            acc[:, :, i] += g_norm * (c / 255.0 * 65535.0)
    return np.clip(acc, 0, 65535).astype(np.uint16)


# ── Detection loading ─────────────────────────────────────────────────────────

def parse_class(class_str):
    """'neuron_GFP_RFP_Sox9' → ('neuron', frozenset({'GFP','RFP','Sox9'}))."""
    parts = str(class_str).strip().split('_')
    return parts[0], frozenset(parts[1:]) if len(parts) > 1 else frozenset()


def load_detections(csv_path, xy_scale, z_scale, z_start, n_layers):
    """Load detection CSV → dict {z_disp: [(x1,y1,x2,y2,base_type,markers)]}.

    z_disp = round((z_global-1)*z_scale) - z_start, filtered to [0, n_layers).
    """
    if not os.path.isfile(csv_path):
        return {}
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  Warning: cannot read {os.path.basename(csv_path)}: {e}")
        return {}

    needed = {'x1', 'y1', 'x2', 'y2', 'class', 'z'}
    if df.empty or not needed.issubset(df.columns):
        return {}

    result = {}
    for _, row in df.iterrows():
        z_g = int(float(row['z']))
        z_d = round((z_g - 1) * z_scale) - z_start
        if z_d < 0 or z_d >= n_layers:
            continue
        x1 = round(float(row['x1']) * xy_scale)
        y1 = round(float(row['y1']) * xy_scale)
        x2 = round(float(row['x2']) * xy_scale)
        y2 = round(float(row['y2']) * xy_scale)
        base, markers = parse_class(row['class'])
        result.setdefault(z_d, []).append((x1, y1, x2, y2, base, markers))
    return result


# ── Box drawing ───────────────────────────────────────────────────────────────

def _draw_dashed_line(img, pt1, pt2, color, thickness, dash=12, gap=6):
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    dx, dy = x2 - x1, y2 - y1
    total = int(np.hypot(dx, dy))
    if total == 0:
        return
    step = dash + gap
    for s in range(0, total, step):
        t0 = s / total
        t1 = min(s + dash, total) / total
        p0 = (int(x1 + t0 * dx), int(y1 + t0 * dy))
        p1 = (int(x1 + t1 * dx), int(y1 + t1 * dy))
        cv2.line(img, p0, p1, color, thickness)


def _draw_dashed_rect(img, pt1, pt2, color, thickness, dash=12, gap=6):
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    _draw_dashed_line(img, (x1, y1), (x2, y1), color, thickness, dash, gap)
    _draw_dashed_line(img, (x1, y2), (x2, y2), color, thickness, dash, gap)
    _draw_dashed_line(img, (x1, y1), (x1, y2), color, thickness, dash, gap)
    _draw_dashed_line(img, (x2, y1), (x2, y2), color, thickness, dash, gap)


def draw_detections_on_img(img, det_list, color, thickness):
    """Draw boxes: neuron=solid rect, glia=dashed rect.

    img: uint16 (H,W) grayscale or (H,W,3) BGR.
    color: scalar (grayscale) or 3-tuple (BGR), both in 16-bit range.
    """
    for x1, y1, x2, y2, base, _ in det_list:
        pt1, pt2 = (int(x1), int(y1)), (int(x2), int(y2))
        if base == 'glia':
            _draw_dashed_rect(img, pt1, pt2, color, thickness)
        else:
            cv2.rectangle(img, pt1, pt2, color, thickness)


# ── Colocalization helpers ────────────────────────────────────────────────────

def get_coloc_combos(det_by_z):
    """All multi-marker frozensets present in coloc detections."""
    combos = set()
    for det_list in det_by_z.values():
        for *_, markers in det_list:
            if len(markers) >= 2:
                combos.add(markers)
    return sorted(combos, key=lambda s: '+'.join(sorted(s)))


def filter_for_combo(det_list, target):
    """Keep entries where target ⊆ entry_markers (superset inclusion)."""
    return [d for d in det_list if target.issubset(d[5])]


def combo_name(markers):
    return '+'.join(sorted(markers))


# ── Save ──────────────────────────────────────────────────────────────────────

def save_image(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok = cv2.imwrite(path, img)
    if not ok:
        print(f"  Warning: cv2.imwrite failed for {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg          = CONFIG
    channels     = cfg['channels']
    ch_colors    = cfg['channel_colors']
    z_start      = cfg['z_start']
    n_layers     = cfg['n_layers']
    result_dir   = cfg['result_dir']
    xml_path     = cfg.get('xml')
    output_dir   = cfg['output_dir']
    thickness    = cfg.get('box_thickness', 2)
    pct          = cfg.get('normalise_pct', (0.5, 99.5))
    WHITE        = 65535         # for grayscale images
    WHITE_BGR    = (65535, 65535, 65535)  # for BGR images

    # 1. Resolve coordinate scales
    print("Resolving scales...")
    xy_scale, z_scale = resolve_scales(
        channels, xml_path, cfg.get('xy_scale'), cfg.get('z_scale')
    )

    # 2. Discover slice files
    print("Discovering slice files...")
    ch_files = {}
    for ch, block_dir in channels.items():
        ch_files[ch] = list_block_slices(block_dir)
        print(f"  {ch}: {len(ch_files[ch])} slices")

    n_slices = min(len(f) for f in ch_files.values())
    if z_start >= n_slices:
        raise ValueError(f"z_start={z_start} >= available slices ({n_slices})")
    z_end    = min(z_start + n_layers, n_slices)
    actual_n = z_end - z_start
    print(f"  Loading file indices {z_start}–{z_end-1} ({actual_n} slices)")

    # 3. Load detection CSVs
    print("Loading detections...")
    s2_dir = os.path.join(result_dir, '2_global_2d_raw')
    s3_dir = os.path.join(result_dir, '3_channel_3d')
    s4_dir = os.path.join(result_dir, '4_colocalization')

    det_2d, det_3d = {}, {}
    for ch in channels:
        p2 = os.path.join(s2_dir, f'{ch}_2d_global.csv')
        p3 = os.path.join(s3_dir, f'{ch}_3d_tracked.csv')
        det_2d[ch] = load_detections(p2, xy_scale, z_scale, z_start, actual_n)
        det_3d[ch] = load_detections(p3, xy_scale, z_scale, z_start, actual_n)
        n2 = sum(len(v) for v in det_2d[ch].values())
        n3 = sum(len(v) for v in det_3d[ch].values())
        print(f"  {ch}: {n2} 2D boxes, {n3} z-linked cells in loaded range")

    coloc_csv = os.path.join(s4_dir, 'coloc_result.csv')
    det_coloc = load_detections(coloc_csv, xy_scale, z_scale, z_start, actual_n)
    combos = get_coloc_combos(det_coloc)
    if combos:
        print(f"  Coloc combos: {[combo_name(c) for c in combos]}")
    else:
        print("  No coloc combos found (coloc CSV missing or no multi-marker entries in range)")

    # 4. Process each z-slice
    print(f"\nProcessing {actual_n} slices → {output_dir}")
    for i in range(actual_n):
        z_file = z_start + i   # index into sorted file list
        z_disp = i             # 0-based offset used for detection lookup

        print(f"  Slice file index {z_file} ...")

        # Load and stretch each channel
        ch_gray = {}
        for ch, files in ch_files.items():
            arr = read_slice_raw(files[z_file])
            if arr is None:
                raise RuntimeError(f"Failed to read: {files[z_file]}")
            ch_gray[ch] = stretch_to_uint16(arr, pct[0], pct[1])

        # Output 1: per-channel grayscale + 2D raw detections (white boxes)
        for ch in channels:
            img = ch_gray[ch].copy()
            draw_detections_on_img(img, det_2d[ch].get(z_disp, []), WHITE, thickness)
            path = os.path.join(output_dir, '1_raw2d', ch, f'z{z_file:04d}.png')
            save_image(img, path)

        # Output 2: per-channel grayscale + z-linked detections (white boxes)
        for ch in channels:
            img = ch_gray[ch].copy()
            draw_detections_on_img(img, det_3d[ch].get(z_disp, []), WHITE, thickness)
            path = os.path.join(output_dir, '2_zlinked', ch, f'z{z_file:04d}.png')
            save_image(img, path)

        # Output 3: RGB composite + coloc boxes per marker combo
        if combos:
            composite = make_rgb_composite(ch_gray, ch_colors)
            all_coloc = det_coloc.get(z_disp, [])
            for combo in combos:
                filtered = filter_for_combo(all_coloc, combo)
                img_c = composite.copy()
                draw_detections_on_img(img_c, filtered, WHITE_BGR, thickness)
                path = os.path.join(output_dir, '3_coloc', combo_name(combo),
                                    f'z{z_file:04d}.png')
                save_image(img_c, path)

    print(f"\nDone. Outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
