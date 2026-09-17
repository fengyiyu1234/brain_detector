# -*- coding: utf-8 -*-
"""
3D volumetric channel alignment for pre_align mode.

Workflow per tile:
  1. build_cell_boxes()         – filter z-linked cells to the central z-window
  2. _effective_z_extent()      – expand single-z-slice cells to isotropic volume
  3. _voxelize_to_grid()        – rasterize 3D boxes into a binary occupancy grid
  4. _fft_3d_shifts()           – 3D FFT cross-correlation → coarse (dx, dy, dz)
  5. _voxel_iou()               – voxel-level IoU at a given bin shift
  6. find_shift()               – coarse FFT + fine voxel-IoU search (intra-soma / intra-TF)
  7. _containment_score()       – soma-TF containment count at a given pixel shift
  7a. _displacement_peaks()    – soma−TF centroid displacement histogram → coarse candidates
  8. find_shift_containment()   – coarse candidates + fine containment search (soma↔TF cross-group)
  9. compute_tile_channel_shifts() – two-step alignment strategy
  10. apply_shift_to_csv()       – applies (dx, dy, dz) to a detection CSV
  11. save_tile_offsets()        – writes per-tile offset JSON for traceability

Two-step alignment strategy (tf_align_mode='chain', default):
  Step 1a  intra-soma  : align each extra soma channel → reference soma channel
  Step 1b  intra-TF    : align each extra TF channel   → reference TF channel
  Step 2   cross-group : align reference TF channel    → reference soma channel
  Final shift per channel:
    ref-soma : (0, 0, 0)
    soma-N   : intra-soma shift
    ref-TF   : cross-group shift
    TF-N     : intra-TF shift + cross-group shift

tf_align_mode='direct': Step 1a as above, then every TF channel is aligned
independently to the reference soma channel with the cross-group containment
search (no intra-TF step). Use when TF markers label different populations.

Step 2 uses containment-based scoring (find_shift_containment) rather than
voxel-IoU: TF nucleus boxes (e.g. Sox9, ~8-17px) are much smaller than soma
boxes (whole-cell YOLO boxes), so shifting the TF box by a few pixels inside
a soma box barely changes bbox overlap area — the voxel-IoU landscape is
nearly flat in XY and its argmax is noisy. Containment (same strict 3D bbox
containment + centroid-distance gate used downstream for colocalization,
see stitcher.annotate_soma_with_tf_containment) is pixel-sensitive instead,
since a nucleus centroid crossing the gate radius flips the match on/off.
"""

import os
import json
import logging
import warnings
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Filter cells to the alignment z-window
# ──────────────────────────────────────────────────────────────────────────────

def build_cell_boxes(vol_list, z_lo, z_hi):
    """
    Return cells from vol_list whose z range overlaps [z_lo, z_hi].

    A cell is included if z_min <= z_hi AND z_max >= z_lo (partial overlap ok).
    Returns a subset list of original cell dicts (no copy).
    """
    if not vol_list:
        return []
    result = []
    for cell in vol_list:
        z_min = cell.get('z_min', cell.get('cz', 0))
        z_max = cell.get('z_max', cell.get('cz', 0))
        if z_min <= z_hi and z_max >= z_lo:
            result.append(cell)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Isotropic z-extent for single-z-slice cells
# ──────────────────────────────────────────────────────────────────────────────

def _effective_z_extent(cell, xy_res_um, z_res_um):
    """
    For cells spanning only one z-slice, infer z extent from XY diameter
    assuming an isotropic (spherical) shape.

    Returns (z_lo_f, z_hi_f) as float slice indices.
    Multi-slice cells are returned unchanged.
    """
    z_min = float(cell.get('z_min', 0))
    z_max = float(cell.get('z_max', 0))
    if z_max > z_min:
        return z_min, z_max

    xy_diam_px = max(
        cell.get('x2_3d', 0) - cell.get('x1_3d', 0),
        cell.get('y2_3d', 0) - cell.get('y1_3d', 0),
        1.0,
    )
    z_span = max(1.0, xy_diam_px * xy_res_um / z_res_um)
    z_ctr  = z_min
    return z_ctr - z_span / 2.0, z_ctr + z_span / 2.0


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Voxelize cells into a 3D binary occupancy grid
# ──────────────────────────────────────────────────────────────────────────────

def _voxelize_to_grid(cells, z_lo, z_hi, x_min, x_max, y_min, y_max,
                      bin_size, xy_res_um=0.65, z_res_um=8.0):
    """
    Rasterize cell 3D boxes into a binary occupancy grid of shape (Nx, Ny, Nz).

    Grid axes: x (column) → axis 0, y (row) → axis 1, z (slice) → axis 2.
    XY bin size = bin_size px.  Z bin size = 1 slice.

    Uses per_z_boxes for per-slice accuracy; falls back to the aggregate 3D box.
    Single-z-slice cells get their z extent expanded isotropically via
    _effective_z_extent() before rasterization.
    """
    nz = max(1, int(round(z_hi - z_lo)))
    nx = max(1, int(np.ceil((x_max - x_min) / bin_size)))
    ny = max(1, int(np.ceil((y_max - y_min) / bin_size)))
    grid = np.zeros((nx, ny, nz), dtype=np.uint8)

    for cell in cells:
        cell_z_lo, cell_z_hi = _effective_z_extent(cell, xy_res_um, z_res_um)
        z_start = max(int(z_lo), int(np.floor(cell_z_lo)))
        z_end   = min(int(z_hi) - 1, int(np.ceil(cell_z_hi)))

        per_z = cell.get('per_z_boxes', {})

        # Fall-back aggregate XY box
        agg_x1 = cell.get('x1_3d', cell.get('cx', 0))
        agg_y1 = cell.get('y1_3d', cell.get('cy', 0))
        agg_x2 = cell.get('x2_3d', cell.get('cx', 0))
        agg_y2 = cell.get('y2_3d', cell.get('cy', 0))

        for z_slice in range(z_start, z_end + 1):
            z_bin = z_slice - int(z_lo)
            if z_bin < 0 or z_bin >= nz:
                continue

            if z_slice in per_z:
                b = per_z[z_slice]
                bx1, by1, bx2, by2 = b[0], b[1], b[2], b[3]
            else:
                bx1, by1, bx2, by2 = agg_x1, agg_y1, agg_x2, agg_y2

            xi1 = max(0, int((bx1 - x_min) / bin_size))
            xi2 = min(nx - 1, int(np.ceil((bx2 - x_min) / bin_size)))
            yi1 = max(0, int((by1 - y_min) / bin_size))
            yi2 = min(ny - 1, int(np.ceil((by2 - y_min) / bin_size)))

            if xi2 >= xi1 and yi2 >= yi1:
                grid[xi1:xi2 + 1, yi1:yi2 + 1, z_bin] = 1

    return grid


# ──────────────────────────────────────────────────────────────────────────────
# 4.  3D FFT cross-correlation for coarse shift estimate
# ──────────────────────────────────────────────────────────────────────────────

def _fft_3d_shifts(grid_ref, grid_tgt, xy_range_px, z_range_slices, bin_size):
    """
    Estimate (dx, dy, dz) via 3D FFT cross-correlation on binary occupancy grids.

    corr(dx, dy, dz) = Σ grid_ref(x,y,z) * grid_tgt(x+dx, y+dy, z+dz)
    Computed efficiently as IFFT3(FFT3(ref) * conj(FFT3(tgt))).

    Returns (dx, dy, dz) in pixel / pixel / slice units.
    XY shifts are quantised to multiples of bin_size; dz is in whole slices.
    """
    if grid_ref.sum() == 0 or grid_tgt.sum() == 0:
        return 0, 0, 0

    F_ref = np.fft.fftn(grid_ref.astype(np.float32))
    F_tgt = np.fft.fftn(grid_tgt.astype(np.float32))
    corr  = np.fft.ifftn(F_ref * np.conj(F_tgt)).real
    corr  = np.fft.fftshift(corr)

    cx, cy, cz = corr.shape[0] // 2, corr.shape[1] // 2, corr.shape[2] // 2
    rx = min(int(xy_range_px / bin_size), cx)
    ry = min(int(xy_range_px / bin_size), cy)
    rz = min(z_range_slices, cz)

    search = corr[
        max(cx - rx, 0): cx + rx + 1,
        max(cy - ry, 0): cy + ry + 1,
        max(cz - rz, 0): cz + rz + 1,
    ]
    peak = np.unravel_index(np.argmax(search), search.shape)

    dx_bins = peak[0] - min(rx, cx)
    dy_bins = peak[1] - min(ry, cy)
    dz_bins = peak[2] - min(rz, cz)

    dx = int(round(dx_bins * bin_size))
    dy = int(round(dy_bins * bin_size))
    dz = int(dz_bins)
    return dx, dy, dz


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Voxel-level IoU at a given bin-space shift
# ──────────────────────────────────────────────────────────────────────────────

def _voxel_iou(grid_ref, grid_tgt, dx_bin, dy_bin, dz_bin):
    """
    Compute voxel-level IoU after shifting grid_tgt by (dx_bin, dy_bin, dz_bin) bins.

    Uses array slicing — no data copy.
    Union is computed over the total voxels in both grids (not just the overlap
    region), so large shifts that clip most cells are penalised.
    """
    nx, ny, nz = grid_ref.shape

    x0 = max(0, dx_bin);  x1 = min(nx, nx + dx_bin)
    y0 = max(0, dy_bin);  y1 = min(ny, ny + dy_bin)
    z0 = max(0, dz_bin);  z1 = min(nz, nz + dz_bin)

    if x0 >= x1 or y0 >= y1 or z0 >= z1:
        return 0.0

    tx0 = x0 - dx_bin;  tx1 = x1 - dx_bin
    ty0 = y0 - dy_bin;  ty1 = y1 - dy_bin
    tz0 = z0 - dz_bin;  tz1 = z1 - dz_bin

    ref_crop = grid_ref[x0:x1, y0:y1, z0:z1]
    tgt_crop = grid_tgt[tx0:tx1, ty0:ty1, tz0:tz1]

    intersection = int(np.logical_and(ref_crop, tgt_crop).sum())
    total = int(grid_ref.sum()) + int(grid_tgt.sum())
    union = total - intersection
    return intersection / union if union > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Full find_shift: voxelize → 3D FFT → fine voxel-IoU search
# ──────────────────────────────────────────────────────────────────────────────

def find_shift(cells_ref, cells_tgt, z_lo, z_hi,
               bin_size=4, xy_res_um=0.65, z_res_um=8.0,
               xy_range_px=30, z_range_slices=5,
               fine_xy_px=8, fine_z_slices=2):
    """
    Find (dx, dy, dz) that maximises 3D volumetric voxel IoU.

    Strategy:
      1. Determine spatial bounds from all cells in both channels.
      2. Voxelize each channel into a 3D binary occupancy grid (bin_size px/bin, 1 slice/bin).
      3. 3D FFT cross-correlation → coarse (dx_fft, dy_fft, dz_fft).
      4. Fine voxel-IoU search: ±fine_xy_px and ±fine_z_slices around FFT peak.

    Parameters
    ----------
    cells_ref, cells_tgt : list of vol_list cell dicts
    z_lo, z_hi           : float  alignment z-window (slice indices)
    bin_size             : int    XY voxel size in pixels (default 4)
    xy_res_um, z_res_um  : float  physical resolution for isotropic z expansion
    xy_range_px          : int    FFT search radius in XY (pixels)
    z_range_slices       : int    FFT search radius in Z (slices)
    fine_xy_px           : int    fine search ±radius in XY around FFT peak
    fine_z_slices        : int    fine search ±radius in Z around FFT peak

    Returns
    -------
    (dx, dy, dz, score)  in pixel / pixel / slice / [0, 1]
    """
    if not cells_ref or not cells_tgt:
        return 0, 0, 0, 0.0

    all_cells = cells_ref + cells_tgt
    x_min = min(c.get('x1_3d', c.get('cx', 0)) for c in all_cells)
    x_max = max(c.get('x2_3d', c.get('cx', 0)) for c in all_cells)
    y_min = min(c.get('y1_3d', c.get('cy', 0)) for c in all_cells)
    y_max = max(c.get('y2_3d', c.get('cy', 0)) for c in all_cells)

    # Pad so the FFT search window never touches the grid boundary
    pad = xy_range_px + fine_xy_px
    x_min -= pad;  x_max += pad
    y_min -= pad;  y_max += pad

    grid_ref = _voxelize_to_grid(cells_ref, z_lo, z_hi,
                                  x_min, x_max, y_min, y_max,
                                  bin_size, xy_res_um, z_res_um)
    grid_tgt = _voxelize_to_grid(cells_tgt, z_lo, z_hi,
                                  x_min, x_max, y_min, y_max,
                                  bin_size, xy_res_um, z_res_um)

    if grid_ref.sum() == 0 or grid_tgt.sum() == 0:
        return 0, 0, 0, 0.0

    # Coarse: 3D FFT
    dx_fft, dy_fft, dz_fft = _fft_3d_shifts(
        grid_ref, grid_tgt, xy_range_px, z_range_slices, bin_size
    )

    # Fine search in bin space around FFT peak
    fine_bins = max(1, int(np.ceil(fine_xy_px / bin_size)))
    dx_fft_bin = int(round(dx_fft / bin_size))
    dy_fft_bin = int(round(dy_fft / bin_size))

    best_score = -1.0
    best_dx_bin, best_dy_bin, best_dz = dx_fft_bin, dy_fft_bin, dz_fft

    for ddx in range(-fine_bins, fine_bins + 1):
        for ddy in range(-fine_bins, fine_bins + 1):
            for ddz in range(-fine_z_slices, fine_z_slices + 1):
                dx_b = dx_fft_bin + ddx
                dy_b = dy_fft_bin + ddy
                dz_b = dz_fft     + ddz
                score = _voxel_iou(grid_ref, grid_tgt, dx_b, dy_b, dz_b)
                if score > best_score:
                    best_score  = score
                    best_dx_bin = dx_b
                    best_dy_bin = dy_b
                    best_dz     = dz_b

    return (int(best_dx_bin * bin_size),
            int(best_dy_bin * bin_size),
            int(best_dz),
            float(best_score))


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Soma-TF containment score at a given pixel/slice shift
# ──────────────────────────────────────────────────────────────────────────────

def _cell_arrays(cells):
    """Vectorize a list of vol_list cell dicts into centroid + bbox numpy arrays."""
    centroids = np.array([[c.get('cx', 0), c.get('cy', 0), c.get('cz', 0)] for c in cells],
                         dtype=float)
    x1 = np.array([c.get('x1_3d', 0) for c in cells], dtype=float)
    y1 = np.array([c.get('y1_3d', 0) for c in cells], dtype=float)
    x2 = np.array([c.get('x2_3d', 0) for c in cells], dtype=float)
    y2 = np.array([c.get('y2_3d', 0) for c in cells], dtype=float)
    z1 = np.array([c.get('z_min', 0) for c in cells], dtype=float)
    z2 = np.array([c.get('z_max', 0) for c in cells], dtype=float)
    return centroids, x1, y1, x2, y2, z1, z2


def _prepare_soma_containment_index(soma_cells):
    """
    Precompute the soma centroid/bbox arrays, bounding-sphere radii and
    cKDTree used by _containment_score(). Call once per
    find_shift_containment() invocation and reuse across all candidate shifts.
    """
    centroids, x1, y1, x2, y2, z1, z2 = _cell_arrays(soma_cells)
    radii = np.maximum(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / 2, 1.0)
    max_radius = float(radii.max())
    tree = cKDTree(centroids)
    return dict(centroids=centroids, x1=x1, y1=y1, x2=x2, y2=y2, z1=z1, z2=z2,
                radii=radii, max_radius=max_radius, tree=tree)


def _containment_score(soma_idx, tf_arrays, dx, dy, dz,
                       max_center_dist_ratio, xy_margin, z_pad):
    """
    Score how well TF cells satisfy strict 3D bbox containment inside a soma
    cell after shifting TF coordinates by (dx, dy, dz). Vectorized batch
    query (cKDTree.query_ball_point with workers=-1) + NumPy masking,
    mirroring stitcher.match_soma_3d_iou's approach.

    Mirrors stitcher.annotate_soma_with_tf_containment(): XY/Z bbox
    containment (with xy_margin / z_pad tolerance) AND centroid distance
    <= max_center_dist_ratio * soma_radius.

    Returns (count, margin_sum):
      count      : number of TF cells matched to some soma (primary score).
      margin_sum : sum, over matched TF cells, of the *closest* match's
                   (gate_radius - dist) margin. Many nearby shifts can tie
                   on count when TF nuclei sit well inside the gate radius,
                   so margin_sum breaks ties toward the shift that centers
                   nuclei most tightly on their matched soma, rather than
                   an arbitrary edge of the tied plateau.

    soma_idx  : dict from _prepare_soma_containment_index(), precomputed once.
    tf_arrays : tuple from _cell_arrays(tf_cells), precomputed once.
    """
    tf_centroids, t_x1, t_y1, t_x2, t_y2, t_z1, t_z2 = tf_arrays
    n_tf = len(tf_centroids)
    if n_tf == 0:
        return 0, 0.0

    shift = np.array([dx, dy, dz], dtype=float)
    tf_pts = tf_centroids + shift

    candidate_lists = soma_idx['tree'].query_ball_point(
        tf_pts, r=soma_idx['max_radius'] * 2, workers=-1
    )
    counts = np.array([len(c) for c in candidate_lists], dtype=np.int64)
    if counts.sum() == 0:
        return 0, 0.0

    i_arr = np.repeat(np.arange(n_tf, dtype=np.int64), counts)
    j_arr = np.concatenate([np.asarray(c, dtype=np.int64) for c in candidate_lists if len(c) > 0])

    tf_x1c, tf_x2c = t_x1[i_arr] + dx, t_x2[i_arr] + dx
    tf_y1c, tf_y2c = t_y1[i_arr] + dy, t_y2[i_arr] + dy
    tf_z1c, tf_z2c = t_z1[i_arr] + dz, t_z2[i_arr] + dz

    s_x1, s_y1, s_x2, s_y2 = soma_idx['x1'][j_arr], soma_idx['y1'][j_arr], soma_idx['x2'][j_arr], soma_idx['y2'][j_arr]
    s_z1, s_z2 = soma_idx['z1'][j_arr], soma_idx['z2'][j_arr]

    contained_xy = ((s_x1 - xy_margin <= tf_x1c) & (tf_x2c <= s_x2 + xy_margin) &
                    (s_y1 - xy_margin <= tf_y1c) & (tf_y2c <= s_y2 + xy_margin))
    # z_pad slices of tolerance on exactly one side only, never both at once
    # (padding both sides simultaneously lets a truncated box balloon into an
    # oversized capture window) — applies the same way to single- and multi-layer
    # soma boxes.
    contained_z = ((tf_z1c >= s_z1 - z_pad) & (tf_z2c <= s_z2)) | ((tf_z1c >= s_z1) & (tf_z2c <= s_z2 + z_pad))
    ok = contained_xy & contained_z
    if not ok.any():
        return 0, 0.0

    dists = np.linalg.norm(soma_idx['centroids'][j_arr] - tf_pts[i_arr], axis=1)
    gate_radius = max_center_dist_ratio * soma_idx['radii'][j_arr]
    gate_ok = dists <= gate_radius
    matched = ok & gate_ok
    if not matched.any():
        return 0, 0.0

    m_i = i_arr[matched]
    m_margin = gate_radius[matched] - dists[matched]

    order = np.argsort(m_i)
    m_i_sorted, m_margin_sorted = m_i[order], m_margin[order]
    unique_i, start_idx = np.unique(m_i_sorted, return_index=True)
    best_margin_per_tf = np.maximum.reduceat(m_margin_sorted, start_idx)

    return int(unique_i.size), float(best_margin_per_tf.sum())


# ──────────────────────────────────────────────────────────────────────────────
# 7a.  Containment-aware coarse search: displacement histogram
# ──────────────────────────────────────────────────────────────────────────────

def _displacement_peaks(soma_idx, tf_arrays, xy_range_px, z_range_slices,
                        max_center_dist_ratio, n_peaks=3, bin_px=2):
    """
    Coarse soma<->TF shift candidates from the histogram of (soma centroid - TF centroid).

    A TF nucleus contributes to the containment count at shift s only when some
    soma centroid lies within the gate radius of (TF centroid + s), i.e. when
    s ~= soma - TF. Histogramming those displacements over the search window and
    smoothing with the gate radius therefore approximates the containment count
    for every candidate shift at once, from a single KD-tree query. The 3D FFT
    of occupancy grids used by find_shift() does not: large soma boxes and dense
    small nuclei give correlation peaks unrelated to containment, and the fine
    search that follows only looks +/-fine_xy_px around whatever FFT returned.

    Returns up to n_peaks (dx, dy, dz) candidates, strongest first, separated by
    at least the smoothing radius.
    """
    from scipy.ndimage import uniform_filter

    soma_c = soma_idx['centroids']
    tf_c = tf_arrays[0]
    if len(soma_c) == 0 or len(tf_c) == 0:
        return [(0, 0, 0)]
    r_xy, r_z = float(xy_range_px), max(int(z_range_slices), 0)
    scale = np.array([1.0, 1.0, r_xy / max(r_z, 0.5)])
    groups = cKDTree(soma_c * scale).query_ball_point(tf_c * scale, r=r_xy, p=np.inf, workers=-1)
    counts = np.fromiter((len(g) for g in groups), dtype=np.int64, count=len(groups))
    if counts.sum() == 0:
        return [(0, 0, 0)]
    ti = np.repeat(np.arange(len(tf_c)), counts)
    si = np.fromiter((j for g in groups for j in g), dtype=np.int64, count=int(counts.sum()))
    d = soma_c[si] - tf_c[ti]

    n_xy = int(np.ceil(r_xy / bin_px))
    edges_xy = (np.arange(-n_xy, n_xy + 2) - 0.5) * bin_px
    edges_z = np.arange(-r_z, r_z + 2) - 0.5
    hist, _ = np.histogramdd(d, bins=(edges_xy, edges_xy, edges_z))
    gate = max_center_dist_ratio * float(np.median(soma_idx['radii']))
    k_xy = 2 * int(np.ceil(gate / bin_px)) + 1
    hist = uniform_filter(hist, size=(k_xy, k_xy, 3), mode='constant')

    centers_xy = (edges_xy[:-1] + edges_xy[1:]) / 2
    centers_z = (edges_z[:-1] + edges_z[1:]) / 2
    sup_xy = max(k_xy, int(np.ceil(4 / bin_px)))
    peaks = []
    work = hist.copy()
    for _ in range(n_peaks):
        if work.max() <= 0:
            break
        i, j, k = np.unravel_index(np.argmax(work), work.shape)
        peaks.append((int(round(centers_xy[i])), int(round(centers_xy[j])), int(round(centers_z[k]))))
        work[max(i - sup_xy, 0):i + sup_xy + 1, max(j - sup_xy, 0):j + sup_xy + 1,
             max(k - 1, 0):k + 2] = 0
    return peaks or [(0, 0, 0)]


# ──────────────────────────────────────────────────────────────────────────────
# 7b.  Full find_shift_containment: voxelize → 3D FFT → fine containment search
# ──────────────────────────────────────────────────────────────────────────────

def find_shift_containment(cells_ref_soma, cells_tf, z_lo, z_hi,
                           bin_size=4, xy_res_um=0.65, z_res_um=8.0,
                           xy_range_px=30, z_range_slices=5,
                           fine_xy_px=8, fine_z_slices=2,
                           max_center_dist_ratio=0.3, xy_margin=0, z_pad=0,
                           coarse='displacement_hist'):
    """
    Find (dx, dy, dz) that maximises soma-TF containment count (see module
    docstring for why this replaces voxel-IoU for the soma<->TF cross-group
    alignment step).

    Strategy:
      1. Coarse, within +/-xy_range_px / +/-z_range_slices:
           'displacement_hist' (default) - histogram of soma-TF centroid
             displacements (_displacement_peaks); the top candidates are each
             checked with a small containment search and the best one is kept.
           'fft' - the 3D-FFT occupancy cross-correlation of find_shift(). Kept
             for reproducing old results only: on a dense TF channel (Olig2 vs a
             GFP soma reference) it locked onto spurious peaks in most tiles.
      2. Fine: instead of voxel-IoU, score each candidate pixel/slice shift
         by how many TF cells satisfy strict containment inside a soma cell
         (same test as stitcher.annotate_soma_with_tf_containment).

    Parameters
    ----------
    cells_ref_soma, cells_tf : list of vol_list cell dicts
    z_lo, z_hi                : float  alignment z-window (slice indices)
    bin_size                  : int    XY voxel size in pixels (FFT stage only)
    xy_res_um, z_res_um       : float  physical resolution for isotropic z expansion
    xy_range_px               : int    FFT search radius in XY (pixels)
    z_range_slices            : int    FFT search radius in Z (slices)
    fine_xy_px                : int    fine search +/-radius in XY around FFT peak (pixels)
    fine_z_slices              : int    fine search +/-radius in Z around FFT peak (slices)
    max_center_dist_ratio     : float  centroid gate, see stitcher.annotate_soma_with_tf_containment
    xy_margin, z_pad          : containment tolerance, see stitcher.annotate_soma_with_tf_containment

    Returns
    -------
    (dx, dy, dz, score)  in pixel / pixel / slice / [0, 1] (score = matched fraction of TF cells)
    """
    if not cells_ref_soma or not cells_tf:
        return 0, 0, 0, 0.0
    if coarse not in ('displacement_hist', 'fft'):
        raise ValueError(f"coarse 只能是 'displacement_hist' 或 'fft'，收到 {coarse!r}")

    soma_idx  = _prepare_soma_containment_index(cells_ref_soma)
    tf_arrays = _cell_arrays(cells_tf)

    def _score(dx, dy, dz):
        return _containment_score(soma_idx, tf_arrays, dx, dy, dz,
                                  max_center_dist_ratio, xy_margin, z_pad)

    if coarse == 'fft':
        all_cells = cells_ref_soma + cells_tf
        x_min = min(c.get('x1_3d', c.get('cx', 0)) for c in all_cells)
        x_max = max(c.get('x2_3d', c.get('cx', 0)) for c in all_cells)
        y_min = min(c.get('y1_3d', c.get('cy', 0)) for c in all_cells)
        y_max = max(c.get('y2_3d', c.get('cy', 0)) for c in all_cells)

        pad = xy_range_px + fine_xy_px
        x_min -= pad;  x_max += pad
        y_min -= pad;  y_max += pad

        grid_ref = _voxelize_to_grid(cells_ref_soma, z_lo, z_hi,
                                      x_min, x_max, y_min, y_max,
                                      bin_size, xy_res_um, z_res_um)
        grid_tgt = _voxelize_to_grid(cells_tf, z_lo, z_hi,
                                      x_min, x_max, y_min, y_max,
                                      bin_size, xy_res_um, z_res_um)

        if grid_ref.sum() == 0 or grid_tgt.sum() == 0:
            return 0, 0, 0, 0.0

        dx_fft, dy_fft, dz_fft = _fft_3d_shifts(
            grid_ref, grid_tgt, xy_range_px, z_range_slices, bin_size
        )
    else:
        # 每个候选峰先在 ±2 px / ±1 层内做小范围包含打分，取最好的作为精搜索中心
        best_c, best_c_score = (0, 0, 0), (-1, -1.0)
        for cx, cy, cz in _displacement_peaks(soma_idx, tf_arrays, xy_range_px, z_range_slices,
                                              max_center_dist_ratio):
            for ddx in range(-2, 3):
                for ddy in range(-2, 3):
                    for ddz in range(-1, 2):
                        sc = _score(cx + ddx, cy + ddy, cz + ddz)
                        if sc > best_c_score:
                            best_c_score, best_c = sc, (cx + ddx, cy + ddy, cz + ddz)
        dx_fft, dy_fft, dz_fft = best_c

    # Fine search in pixel/slice space around the coarse peak, scored by containment

    # (count, margin_sum) compared lexicographically: count is the primary
    # objective, margin_sum breaks ties toward the best-centered shift when
    # several nearby candidates all achieve the same containment count.
    best_score = (-1, -1.0)
    best_dx, best_dy, best_dz = dx_fft, dy_fft, dz_fft

    for ddx in range(-fine_xy_px, fine_xy_px + 1):
        for ddy in range(-fine_xy_px, fine_xy_px + 1):
            for ddz in range(-fine_z_slices, fine_z_slices + 1):
                dx_c = dx_fft + ddx
                dy_c = dy_fft + ddy
                dz_c = dz_fft + ddz
                score = _score(dx_c, dy_c, dz_c)
                if score > best_score:
                    best_score = score
                    best_dx, best_dy, best_dz = dx_c, dy_c, dz_c

    best_count = best_score[0]
    return (int(best_dx), int(best_dy), int(best_dz),
            float(best_count) / max(1, len(cells_tf)))


# ──────────────────────────────────────────────────────────────────────────────
# 8.  Two-step per-tile alignment orchestration
# ──────────────────────────────────────────────────────────────────────────────

def compute_tile_channel_shifts(per_ch_vol_lists, soma_ch_ids, tf_ch_ids,
                                z_center, z_half_window,
                                bin_size=4, xy_res_um=0.65, z_res_um=8.0,
                                xy_range_px=30, z_range_slices=5,
                                fine_xy_px=8, fine_z_slices=2,
                                max_center_dist_ratio=0.3, xy_margin=0,
                                containment_z_pad=0, tf_align_mode='chain',
                                containment_coarse='displacement_hist'):
    """
    Compute final (dx, dy, dz) per channel for one tile using two-step strategy.

    Parameters
    ----------
    per_ch_vol_lists : dict  ch_id → list of volumetric cell dicts
    soma_ch_ids      : list  ordered soma channel IDs; first is the reference
    tf_ch_ids        : list  ordered TF channel IDs; first is the reference TF
                             (only meaningful in 'chain' mode)
    tf_align_mode    : str   'chain'  — TF-N → ref TF (voxel IoU), ref TF → ref soma
                                        (containment); TF-N shift is the sum
                             'direct' — every TF channel → ref soma by containment,
                                        independently (use when TF markers label
                                        different cell populations)
    z_center         : float z-center of the alignment window (slice index)
    z_half_window    : int   half-width of the alignment z-window (slices)
    bin_size         : int   XY voxel bin size in pixels
    xy_res_um        : float XY pixel physical size (µm)
    z_res_um         : float Z slice spacing (µm)
    xy_range_px      : int   FFT search radius in XY (pixels)
    z_range_slices   : int   FFT search radius in Z (slices)
    fine_xy_px       : int   fine search ±radius in XY (pixels)
    fine_z_slices    : int   fine search ±radius in Z (slices)
    max_center_dist_ratio : float centroid gate for Step 2 containment search,
                             see stitcher.annotate_soma_with_tf_containment
    xy_margin, containment_z_pad : containment tolerance for Step 2, see
                             stitcher.annotate_soma_with_tf_containment
    containment_coarse : str coarse search of the soma<->TF step, see find_shift_containment

    Returns
    -------
    shifts : dict  ch_id → (dx, dy, dz)
    scores : dict  ch_id → alignment score at optimal shift (voxel IoU for
             intra-soma/intra-TF steps, matched-fraction for the soma↔TF
             cross-group step)
    """
    z_lo = z_center - z_half_window
    z_hi = z_center + z_half_window

    shifts = {}
    scores = {}

    def _boxes(ch_id):
        return build_cell_boxes(per_ch_vol_lists.get(ch_id, []), z_lo, z_hi)

    align_kwargs = dict(
        z_lo=z_lo, z_hi=z_hi,
        bin_size=bin_size, xy_res_um=xy_res_um, z_res_um=z_res_um,
        xy_range_px=xy_range_px, z_range_slices=z_range_slices,
        fine_xy_px=fine_xy_px, fine_z_slices=fine_z_slices,
    )

    ref_soma = soma_ch_ids[0] if soma_ch_ids else None
    ref_tf   = tf_ch_ids[0]   if tf_ch_ids   else None

    if ref_soma:
        shifts[ref_soma] = (0, 0, 0)
        scores[ref_soma] = 1.0

    boxes_ref_soma = _boxes(ref_soma) if ref_soma else []

    # Step 1a: intra-soma alignment (extra soma channels → ref soma)
    for cid in soma_ch_ids[1:]:
        dx, dy, dz, sc = find_shift(boxes_ref_soma, _boxes(cid), **align_kwargs)
        shifts[cid] = (dx, dy, dz)
        scores[cid] = sc

    # direct mode: every TF channel → ref soma by containment, no TF-to-TF step
    if tf_align_mode == 'direct':
        for cid in tf_ch_ids:
            dx, dy, dz, sc = find_shift_containment(
                boxes_ref_soma, _boxes(cid),
                max_center_dist_ratio=max_center_dist_ratio,
                xy_margin=xy_margin, z_pad=containment_z_pad, coarse=containment_coarse,
                **align_kwargs
            )
            shifts[cid] = (dx, dy, dz)
            scores[cid] = sc
        return shifts, scores

    # Step 1b: intra-TF alignment (extra TF channels → ref TF)
    if ref_tf:
        boxes_ref_tf = _boxes(ref_tf)
        for cid in tf_ch_ids[1:]:
            dx, dy, dz, sc = find_shift(boxes_ref_tf, _boxes(cid), **align_kwargs)
            shifts[cid] = (dx, dy, dz)
            scores[cid] = sc

        # Step 2: cross-group alignment (ref TF → ref soma), scored by
        # soma-TF containment instead of voxel-IoU (see module docstring)
        cx, cy, cz, sc_cross = find_shift_containment(
            boxes_ref_soma, boxes_ref_tf,
            max_center_dist_ratio=max_center_dist_ratio,
            xy_margin=xy_margin, z_pad=containment_z_pad, coarse=containment_coarse,
            **align_kwargs
        )
        shifts[ref_tf] = (cx, cy, cz)
        scores[ref_tf] = sc_cross

        # Chain: TF-N final shift = intra-TF shift + cross-group shift
        for cid in tf_ch_ids[1:]:
            ix, iy, iz = shifts[cid]
            shifts[cid] = (ix + cx, iy + cy, iz + cz)

    return shifts, scores


# ──────────────────────────────────────────────────────────────────────────────
# 8.  Apply shift to a per-tile detection CSV
# ──────────────────────────────────────────────────────────────────────────────

def apply_shift_to_csv(in_csv_path, dx, dy, dz, out_csv_path, slice_names=None):
    """
    Add (dx, dy) to bbox columns and dz to z column of a detection CSV.

    CSV columns expected: slice_name, x1, y1, x2, y2, class, score, mean, z
    Sign convention: aligned_coord = raw_coord + shift  (same as visualizer).

    slice_names : optional list of the tile's slice names (no extension) in z
                  order, so z (1-based) <-> slice_names[z - 1]. When given and
                  dz != 0, slice_name is rewritten to match the shifted z. Rows
                  shifted outside [1, len(slice_names)] get an empty slice_name
                  (Stage 3 drops them anyway: they fall outside the stitched z range).
    """
    if not os.path.exists(in_csv_path):
        return

    df = pd.read_csv(in_csv_path)
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    if df.empty:
        _atomic_to_csv(df, out_csv_path)
        return

    for col in ['x1', 'x2']:
        if col in df.columns:
            df[col] = df[col].astype(float) + dx
    for col in ['y1', 'y2']:
        if col in df.columns:
            df[col] = df[col].astype(float) + dy
    if 'z' in df.columns:
        df['z'] = df['z'].astype(int) + int(dz)
        if dz != 0 and slice_names is not None and 'slice_name' in df.columns:
            n = len(slice_names)
            df['slice_name'] = [slice_names[z - 1] if 1 <= z <= n else '' for z in df['z']]

    _atomic_to_csv(df, out_csv_path)


def _atomic_to_csv(df, path):
    """先写 .part 再改名：进程中途被杀不会留下半截 CSV 被当成已完成。"""
    tmp = path + '.part'
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


# ──────────────────────────────────────────────────────────────────────────────
# 9.  Save per-tile offset JSON for traceability
# ──────────────────────────────────────────────────────────────────────────────

def save_tile_offsets(tile_name, shifts, scores, out_dir):
    """Write per-tile offset summary to JSON."""
    payload = {
        ch_id: {
            "dx": int(v[0]), "dy": int(v[1]), "dz": int(v[2]),
            "iou_score": round(scores.get(ch_id, 0.0), 4),
        }
        for ch_id, v in shifts.items()
    }
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{tile_name}_offsets.json")
    with open(out_path + '.part', 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    os.replace(out_path + '.part', out_path)
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# 9b. One tile end to end (Stage 2.5 runs this in a CPU process pool)
# ──────────────────────────────────────────────────────────────────────────────

def _aligned_channel_ids(routing):
    """每个 tile 要写对齐 CSV 的通道 id（含 double_exposure 的第二曝光）。"""
    ids = []
    for ch in routing:
        ids.append(ch['id'])
        if ch.get('double_exposure'):
            ids.append(ch['second_intensity_id'])
    return ids


def _count_lines(path):
    with open(path, 'rb') as f:
        return sum(buf.count(b'\n') for buf in iter(lambda: f.read(1 << 20), b''))


def tile_alignment_done(tile_name, det_dir, align_dir, routing):
    """
    offsets JSON 最后写，是完成标记；另外核对每个通道的对齐 CSV 与原始 CSV 行数一致
    （apply_shift_to_csv 不增删行），防止旧版本流程先写 JSON、CSV 写到一半被杀的情况被当成已完成。
    """
    if not os.path.isfile(os.path.join(align_dir, f"{tile_name}_offsets.json")):
        return False
    primary = {ch['id'] for ch in routing}
    for cid in _aligned_channel_ids(routing):
        in_csv = os.path.join(det_dir, f"{tile_name}_{cid}_result.csv")
        out_csv = os.path.join(align_dir, f"{tile_name}_{cid}_result.csv")
        if not os.path.isfile(in_csv):
            if cid in primary:
                return False   # 让 align_tile 重跑并把缺失报出来，而不是悄悄当成已完成
            continue
        if not os.path.isfile(out_csv) or _count_lines(in_csv) != _count_lines(out_csv):
            return False
    return True


def align_tile(tile_path, det_dir, align_dir, routing, settings):
    """
    Stage 2.5 的单 tile 流程：各通道轻量 z-link → 两步对齐 → 写对齐 CSV → 最后写 offsets JSON。
    返回缺失的检测 CSV 路径列表。只做 CPU 计算，参数全可 pickle，供进程池直接调用。
    """
    from src.core.z_linker import run_z_linker   # 局部导入：z_linker 与本模块互不依赖，避免加载顺序问题

    tile_name = os.path.basename(tile_path)
    # 与 worker 相同的切片排序：CSV 里的 z（从 1 开始）对应 slice_names[z-1]，
    # apply_shift_to_csv 施加 dz 时据此同步更新 slice_name
    slice_names = [os.path.splitext(f)[0] for f in sorted(
        f for f in os.listdir(tile_path)
        if f.lower().endswith(('.tif', '.tiff')) and not f.startswith('.'))]

    # 1. 对每个通道轻量 z-link，得到 per-tile 3D vol_list
    per_ch_vol_lists, z_counts, missing = {}, [], []
    for ch in routing:
        cid, ctype = ch['id'], ch.get('type', 'soma')
        csv_path = os.path.join(det_dir, f"{tile_name}_{cid}_result.csv")
        per_ch_vol_lists[cid] = []
        if not os.path.isfile(csv_path):
            missing.append(csv_path)
            continue
        df_tile = pd.read_csv(csv_path)
        if df_tile.empty:
            continue
        mat = df_tile[["x1", "y1", "x2", "y2", "score", "mean", "class", "z"]].values
        mat[:, 6] = np.array([f"{v}_{cid}" for v in mat[:, 6]])
        _, vol_list = run_z_linker(mat, **settings['z_link']['soma' if ctype == 'soma' else 'tf'])
        per_ch_vol_lists[cid] = vol_list
        z_counts.extend(c.get('cz', 0) for c in vol_list)

    # 2. 估计 tile z 中心；3. 两步体素对齐
    z_center = float(np.median(z_counts)) if z_counts else 0.0
    shifts, scores = compute_tile_channel_shifts(
        per_ch_vol_lists,
        soma_ch_ids=settings['soma_ch_ids'],
        tf_ch_ids=settings['tf_ch_ids'],
        z_center=z_center,
        z_half_window=settings['sample_z_center_count'] // 2,
        bin_size=settings['voxel_bin_size_px'],
        xy_res_um=settings['xy_resolution_um'],
        z_res_um=settings['z_resolution_um'],
        xy_range_px=settings['xy_search_range_px'],
        z_range_slices=settings['z_search_range_slices'],
        fine_xy_px=settings['xy_fine_search_px'],
        fine_z_slices=settings['z_fine_search_slices'],
        max_center_dist_ratio=settings['max_center_dist_ratio'],
        containment_z_pad=settings['containment_z_pad'],
        tf_align_mode=settings['tf_align_mode'],
        containment_coarse=settings.get('containment_coarse', 'fft'),
    )

    # 3b. double_exposure 通道的第二曝光复用主曝光的偏移量
    # （同一物理通道/视野，只是曝光不同，无需独立点云配准）。
    for ch in routing:
        if ch.get('double_exposure'):
            second_id = ch['second_intensity_id']
            shifts[second_id] = shifts.get(ch['id'], (0, 0, 0))
            scores[second_id] = scores.get(ch['id'], 0.0)

    # 4. 对每个通道的原始 CSV 应用偏移（覆盖写：设置已由 check_align_settings 保证一致，
    #    偏移是确定性的，重算的结果与上次相同）
    for cid in _aligned_channel_ids(routing):
        dx, dy, dz = shifts.get(cid, (0, 0, 0))
        apply_shift_to_csv(os.path.join(det_dir, f"{tile_name}_{cid}_result.csv"), dx, dy, dz,
                           os.path.join(align_dir, f"{tile_name}_{cid}_result.csv"),
                           slice_names=slice_names)

    # 5. 最后写 offsets JSON，作为该 tile 的完成标记
    save_tile_offsets(tile_name, shifts, scores, align_dir)
    return missing


# ──────────────────────────────────────────────────────────────────────────────
# 10.  Alignment settings: resolve once, persist, guard against stale outputs
# ──────────────────────────────────────────────────────────────────────────────

ALIGN_SETTINGS_FILE = "_align_settings.json"


def resolve_align_settings(config, routing_config):
    """
    Resolve every setting that influences the Stage 2.5 shifts, defaults applied,
    so one dict both drives the alignment and is persisted next to its outputs.

    Raises ValueError for an invalid reference_channel / tf_align_mode.
    """
    pa = config.get('pre_align_params', {})
    dp = config.get('detection_params', {})
    zl = config.get('z_linker', {})
    soma_ids = [ch['id'] for ch in routing_config if ch.get('type', 'soma') == 'soma']
    tf_ids   = [ch['id'] for ch in routing_config if ch.get('type') == 'tf']

    ref = pa.get('reference_channel') or (soma_ids[0] if soma_ids else None)
    if soma_ids and ref not in soma_ids:
        raise ValueError(f"❌ pre_align_params.reference_channel='{ref}' 不是已激活的 soma 通道 {soma_ids}")
    mode = pa.get('tf_align_mode', 'chain')
    if mode not in ('chain', 'direct'):
        raise ValueError(f"❌ pre_align_params.tf_align_mode='{mode}'，只能是 'chain' 或 'direct'")

    def _z_link(p):
        return {'iou_thresh': p.get('iou_thresh', 0.35), 'min_z_layers': p.get('min_z_layers', 1),
                'max_cell_z_span': p.get('max_cell_z_span', 5), 'max_z_gap': p.get('max_z_gap', 0)}

    zl_tf = zl.get('tf', {})
    return {
        'reference_channel': ref,
        'tf_align_mode': mode,
        # compute_tile_channel_shifts treats soma_ch_ids[0] as the reference (stable sort)
        'soma_ch_ids': sorted(soma_ids, key=lambda c: c != ref),
        'tf_ch_ids': tf_ids,
        'z_link': {'soma': _z_link(zl.get('soma', {})), 'tf': _z_link(zl_tf)},
        'sample_z_center_count': pa.get('sample_z_center_count', 50),
        'voxel_bin_size_px': pa.get('voxel_bin_size_px', 4),
        'xy_search_range_px': pa.get('xy_search_range_px', 30),
        'z_search_range_slices': pa.get('z_search_range_slices', 5),
        'xy_fine_search_px': pa.get('xy_fine_search_px', 8),
        'z_fine_search_slices': pa.get('z_fine_search_slices', 2),
        'xy_resolution_um': dp.get('xy_resolution_um', 0.65),
        'z_resolution_um': dp.get('z_resolution_um', 8.0),
        'max_center_dist_ratio': zl_tf.get('max_center_dist_ratio', 0.3),
        'containment_z_pad': zl_tf.get('containment_z_pad', 0),
        'containment_coarse': pa.get('containment_coarse', 'displacement_hist'),
    }


def check_align_settings(align_dir, settings):
    """
    Refuse to reuse alignment outputs produced with different settings.

    apply_shift_to_csv outputs are never overwritten, so re-running Stage 2.5 with
    changed settings would silently mix old and new shifts.
    Returns True when saved settings match, False when there is nothing to compare
    (fresh directory, or legacy outputs from before settings were saved — logged).
    Raises ValueError on a mismatch.
    """
    path = os.path.join(align_dir, ALIGN_SETTINGS_FILE)
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as f:
            saved = json.load(f)
        # 这个键出现之前的对齐结果都是用 FFT 粗搜索算的
        saved.setdefault('containment_coarse', 'fft')
        current = json.loads(json.dumps(settings))
        if saved != current:
            changed = sorted(k for k in set(saved) | set(current) if saved.get(k) != current.get(k))
            details = "; ".join(f"{k}: {saved.get(k)!r} -> {current.get(k)!r}" for k in changed)
            raise ValueError(
                f"❌ 对齐设置与 {align_dir} 里已有的结果不一致（{details}）。"
                f"已有的对齐 CSV 不会被覆盖，请删除该目录后重跑 Stage 2.5。"
                + ('（只想沿用旧结果的话，在 pre_align_params 里设 "containment_coarse": "fft"。）'
                   if changed == ['containment_coarse'] else "")
            )
        return True
    if os.path.isdir(align_dir) and any(f.endswith(('_result.csv', '_offsets.json'))
                                        for f in os.listdir(align_dir)):
        logging.warning(f"⚠️ {align_dir} 已有对齐结果但没有 {ALIGN_SETTINGS_FILE}（旧版本生成），"
                        f"无法校验参数是否一致；如果改过对齐参数，请删除该目录后重跑。")
    return False


def save_align_settings(align_dir, settings):
    """Write the resolved settings next to the alignment outputs."""
    os.makedirs(align_dir, exist_ok=True)
    with open(os.path.join(align_dir, ALIGN_SETTINGS_FILE), 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)
