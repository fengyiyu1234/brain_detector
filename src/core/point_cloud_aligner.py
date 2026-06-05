# -*- coding: utf-8 -*-
"""
Point-cloud-based channel alignment (replaces numorph in pre_align mode).

Workflow per tile:
  1. build_point_cloud()  – extract (x, y, z) centroids from z-linked cells,
                            restricted to a central z window for robustness.
  2. find_shift()         – FFT cross-correlation in XY + brute-force in Z to
                            find the (dx, dy, dz) that maximises point-cloud IoU.
  3. apply_shift_to_csv() – add (dx, dy) to bbox columns, dz to z column of a
                            per-tile detection CSV.

Two-step alignment strategy for multi-channel data
(e.g., RFP/GFP soma  +  Sox9/Olig2 TF):

  Step 1a  intra-soma  : align each extra soma channel → reference soma channel
  Step 1b  intra-TF    : align each extra TF channel   → reference TF  channel
  Step 2   cross-group : align reference TF channel    → reference soma channel
  Final shift per channel:
    ref-soma : (0, 0, 0)
    soma-N   : intra-soma shift
    ref-TF   : cross-group shift
    TF-N     : intra-TF shift + cross-group shift
"""

import os
import json
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Build point cloud from z-linked volumetric cells
# ──────────────────────────────────────────────────────────────────────────────

def build_point_cloud(vol_list, z_center, z_half_window):
    """
    Extract (cx, cy, cz) centroids from vol_list, keeping only cells whose
    cz falls within [z_center - z_half_window, z_center + z_half_window].

    Returns ndarray of shape (N, 3) or empty (0, 3) if no cells qualify.
    """
    if not vol_list:
        return np.empty((0, 3), dtype=float)

    z_lo = z_center - z_half_window
    z_hi = z_center + z_half_window
    pts = []
    for cell in vol_list:
        cz = cell.get('cz', (cell.get('z_min', 0) + cell.get('z_max', 0)) / 2.0)
        if z_lo <= cz <= z_hi:
            pts.append([cell['cx'], cell['cy'], cz])
    if not pts:
        return np.empty((0, 3), dtype=float)
    return np.array(pts, dtype=float)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Find optimal (dx, dy, dz) shift between two point clouds
# ──────────────────────────────────────────────────────────────────────────────

def _point_cloud_iou(cloud_a, cloud_b_shifted, match_dist):
    """
    Fraction of paired points (nearest-neighbour within match_dist).
    Score = |matched| / (|A| + |B| - |matched|)  [point-cloud IoU].
    Uses a simple O(N·M) loop which is fast enough for small point clouds
    (< 2 000 cells per tile); falls back to cKDTree for larger sets.
    """
    if len(cloud_a) == 0 or len(cloud_b_shifted) == 0:
        return 0.0

    # Use cKDTree for efficiency
    from scipy.spatial import cKDTree
    tree_a = cKDTree(cloud_a[:, :2])          # XY only (z already filtered)
    matched = tree_a.query_ball_point(cloud_b_shifted[:, :2], r=match_dist)
    n_matched = sum(1 for m in matched if m)  # count B points that hit at least one A point
    union = len(cloud_a) + len(cloud_b_shifted) - n_matched
    return n_matched / union if union > 0 else 0.0


def _fft_xy_shifts(cloud_ref, cloud_tgt, xy_range_px, bin_size=2.0):
    """
    Estimate integer (dx, dy) via 2-D FFT cross-correlation on voxelised clouds.

    Returns a list of (score_proxy, dx, dy) sorted descending by score proxy.
    The score is the normalised cross-correlation peak height.
    """
    if len(cloud_ref) == 0 or len(cloud_tgt) == 0:
        return [(0.0, 0, 0)]

    pad = int(xy_range_px / bin_size) + 2

    # Determine common grid bounds
    all_x = np.concatenate([cloud_ref[:, 0], cloud_tgt[:, 0]])
    all_y = np.concatenate([cloud_ref[:, 1], cloud_tgt[:, 1]])
    x_min, x_max = all_x.min() - pad * bin_size, all_x.max() + pad * bin_size
    y_min, y_max = all_y.min() - pad * bin_size, all_y.max() + pad * bin_size

    nx = max(int((x_max - x_min) / bin_size) + 1, 1)
    ny = max(int((y_max - y_min) / bin_size) + 1, 1)

    def voxelise(cloud):
        grid = np.zeros((nx, ny), dtype=float)
        xi = np.clip(((cloud[:, 0] - x_min) / bin_size).astype(int), 0, nx - 1)
        yi = np.clip(((cloud[:, 1] - y_min) / bin_size).astype(int), 0, ny - 1)
        np.add.at(grid, (xi, yi), 1.0)
        return grid

    grid_ref = voxelise(cloud_ref)
    grid_tgt = voxelise(cloud_tgt)

    # FFT cross-correlation
    F_ref = np.fft.fft2(grid_ref)
    F_tgt = np.fft.fft2(grid_tgt)
    corr  = np.fft.ifft2(F_ref * np.conj(F_tgt)).real
    corr  = np.fft.fftshift(corr)

    # Restrict to xy_range_px search window
    cx, cy = corr.shape[0] // 2, corr.shape[1] // 2
    r = int(xy_range_px / bin_size)
    search = corr[max(cx - r, 0): cx + r + 1,
                  max(cy - r, 0): cy + r + 1]
    peak = np.unravel_index(np.argmax(search), search.shape)
    score = search[peak]

    # Convert peak index back to pixel shift
    dx_bins = peak[0] - min(r, cx)
    dy_bins = peak[1] - min(r, cy)
    dx = int(round(dx_bins * bin_size))
    dy = int(round(dy_bins * bin_size))
    return [(float(score), dx, dy)]


_Z_RANGE_SOFT_MAX = 5   # recommended maximum z search range (slices)
_Z_OFFSET_HARD_MAX = 10  # absolute hard cap on the returned dz


def find_shift(cloud_ref, cloud_target,
               xy_range_px=30, z_range_slices=3,
               match_dist_px=15):
    """
    Find (dx, dy, dz) that maximises point-cloud IoU between cloud_ref and
    cloud_target (shifted by the returned values).

    Strategy:
      - XY: FFT cross-correlation on voxelised clouds → best XY shift candidate
      - Z : brute-force ±z_range_slices around the FFT-suggested shift
      - Final score is the true point-cloud IoU at the winning (dx, dy, dz)

    Z-axis safety:
      - z_range_slices is soft-capped at _Z_RANGE_SOFT_MAX (5). Values above
        this print a warning and are clipped; light-sheet z-offsets are almost
        always < 5 slices.
      - The returned dz is hard-capped at ±_Z_OFFSET_HARD_MAX (10). If the
        optimiser picks a larger value it is almost certainly a false optimum;
        in that case dz is forced to 0 and a warning is emitted.

    Parameters
    ----------
    cloud_ref    : ndarray (N, 3)  reference channel centroids
    cloud_target : ndarray (M, 3)  target channel centroids (to be shifted)
    xy_range_px  : int   ±px search radius in XY
    z_range_slices : int ±slices to try in Z (soft-capped at 5)
    match_dist_px : float  nearest-neighbour threshold for IoU calculation

    Returns
    -------
    (dx, dy, dz, score) where dx/dy are integer pixels, dz is integer slices
    """
    import warnings

    if len(cloud_ref) == 0 or len(cloud_target) == 0:
        return 0, 0, 0, 0.0

    # Soft cap on z search range
    if z_range_slices > _Z_RANGE_SOFT_MAX:
        warnings.warn(
            f"z_range_slices={z_range_slices} exceeds recommended max "
            f"{_Z_RANGE_SOFT_MAX}; clipping to {_Z_RANGE_SOFT_MAX}.",
            UserWarning, stacklevel=2,
        )
        z_range_slices = _Z_RANGE_SOFT_MAX

    # Step 1: FFT candidate for XY
    fft_results = _fft_xy_shifts(cloud_ref, cloud_target, xy_range_px)
    _, dx_fft, dy_fft = fft_results[0]

    # Step 2: refine Z around zero (light-sheet z-offset usually minimal)
    best_score = -1.0
    best_dx, best_dy, best_dz = dx_fft, dy_fft, 0

    # Fine XY grid search ±5 px around FFT candidate
    xy_fine = 5
    for dx in range(dx_fft - xy_fine, dx_fft + xy_fine + 1):
        for dy in range(dy_fft - xy_fine, dy_fft + xy_fine + 1):
            for dz in range(-z_range_slices, z_range_slices + 1):
                shifted = cloud_target + np.array([dx, dy, dz], dtype=float)
                score = _point_cloud_iou(cloud_ref, shifted, match_dist_px)
                if score > best_score:
                    best_score = score
                    best_dx, best_dy, best_dz = dx, dy, dz

    # Hard cap on the returned z offset
    if abs(best_dz) > _Z_OFFSET_HARD_MAX:
        warnings.warn(
            f"Computed dz={best_dz} exceeds hard limit ±{_Z_OFFSET_HARD_MAX}; "
            "forcing dz=0. This usually indicates insufficient overlap between "
            "point clouds — check channel detections.",
            UserWarning, stacklevel=2,
        )
        best_dz = 0

    return int(best_dx), int(best_dy), int(best_dz), float(best_score)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Apply shift to a per-tile detection CSV
# ──────────────────────────────────────────────────────────────────────────────

def apply_shift_to_csv(in_csv_path, dx, dy, dz, out_csv_path):
    """
    Add (dx, dy) to bbox columns and dz to z column of a detection CSV.

    CSV columns expected: slice_name, x1, y1, x2, y2, class, score, mean, z
    The shift is applied to x1, x2 (+=dx), y1, y2 (+=dy), z (+=dz).
    """
    if not os.path.exists(in_csv_path):
        return

    df = pd.read_csv(in_csv_path)
    if df.empty:
        df.to_csv(out_csv_path, index=False)
        return

    for col in ['x1', 'x2']:
        if col in df.columns:
            df[col] = df[col].astype(float) + dx
    for col in ['y1', 'y2']:
        if col in df.columns:
            df[col] = df[col].astype(float) + dy
    if 'z' in df.columns:
        df['z'] = df['z'].astype(float) + dz

    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    df.to_csv(out_csv_path, index=False)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Two-step per-tile alignment orchestration
# ──────────────────────────────────────────────────────────────────────────────

def compute_tile_channel_shifts(per_ch_vol_lists, soma_ch_ids, tf_ch_ids,
                                z_center, z_half_window,
                                xy_range_px=30, z_range_slices=3,
                                match_dist_px=15):
    """
    Compute final (dx, dy, dz) per channel for one tile using two-step strategy.

    Parameters
    ----------
    per_ch_vol_lists : dict  ch_id → list of volumetric cell dicts
    soma_ch_ids      : list  ordered soma channel IDs; first is the reference
    tf_ch_ids        : list  ordered TF channel IDs; first is the reference TF
    z_center         : float z-center of the tile (slice index)
    z_half_window    : int   half-window in z for point cloud sampling
    xy_range_px      : int   FFT search radius in XY
    z_range_slices   : int   brute-force search range in Z
    match_dist_px    : float nearest-neighbour threshold for IoU

    Returns
    -------
    shifts : dict  ch_id → (dx, dy, dz)
    scores : dict  ch_id → IoU score at the optimal shift
    """
    shifts = {}
    scores = {}

    def _cloud(ch_id):
        vols = per_ch_vol_lists.get(ch_id, [])
        return build_point_cloud(vols, z_center, z_half_window)

    ref_soma = soma_ch_ids[0] if soma_ch_ids else None
    ref_tf   = tf_ch_ids[0]   if tf_ch_ids   else None

    # Reference soma: no shift
    if ref_soma:
        shifts[ref_soma] = (0, 0, 0)
        scores[ref_soma] = 1.0

    cloud_ref_soma = _cloud(ref_soma) if ref_soma else np.empty((0, 3))

    # Step 1a: intra-soma alignment (all extra soma channels → ref soma)
    for cid in soma_ch_ids[1:]:
        cloud_tgt = _cloud(cid)
        dx, dy, dz, sc = find_shift(
            cloud_ref_soma, cloud_tgt,
            xy_range_px=xy_range_px,
            z_range_slices=z_range_slices,
            match_dist_px=match_dist_px,
        )
        shifts[cid] = (dx, dy, dz)
        scores[cid] = sc

    # Step 1b: intra-TF alignment (all extra TF channels → ref TF)
    if ref_tf:
        cloud_ref_tf = _cloud(ref_tf)
        for cid in tf_ch_ids[1:]:
            cloud_tgt = _cloud(cid)
            dx, dy, dz, sc = find_shift(
                cloud_ref_tf, cloud_tgt,
                xy_range_px=xy_range_px,
                z_range_slices=z_range_slices,
                match_dist_px=match_dist_px,
            )
            shifts[cid] = (dx, dy, dz)   # intra-TF shift (to be updated in Step 2)
            scores[cid] = sc

        # Step 2: cross-group alignment (ref TF → ref soma)
        cloud_ref_tf = _cloud(ref_tf)
        cx, cy, cz, sc_cross = find_shift(
            cloud_ref_soma, cloud_ref_tf,
            xy_range_px=xy_range_px,
            z_range_slices=z_range_slices,
            match_dist_px=match_dist_px,
        )
        # ref TF final shift = cross-group shift
        shifts[ref_tf] = (cx, cy, cz)
        scores[ref_tf] = sc_cross

        # Chain: TF-N final shift = intra-TF shift + cross-group shift
        for cid in tf_ch_ids[1:]:
            ix, iy, iz = shifts[cid]
            shifts[cid] = (ix + cx, iy + cy, iz + cz)
            # score stays as intra-TF quality indicator

    return shifts, scores


def save_tile_offsets(tile_name, shifts, scores, out_dir):
    """Write per-tile offset summary to JSON for traceability."""
    payload = {
        ch_id: {
            "dx": int(v[0]), "dy": int(v[1]), "dz": int(v[2]),
            "iou_score": round(scores.get(ch_id, 0.0), 4),
        }
        for ch_id, v in shifts.items()
    }
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{tile_name}_offsets.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    return out_path
