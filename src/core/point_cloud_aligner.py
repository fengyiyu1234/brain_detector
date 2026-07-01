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
  8. find_shift_containment()   – coarse FFT + fine containment search (soma↔TF cross-group)
  9. compute_tile_channel_shifts() – two-step alignment strategy
  10. apply_shift_to_csv()       – applies (dx, dy, dz) to a detection CSV
  11. save_tile_offsets()        – writes per-tile offset JSON for traceability

Two-step alignment strategy:
  Step 1a  intra-soma  : align each extra soma channel → reference soma channel
  Step 1b  intra-TF    : align each extra TF channel   → reference TF channel
  Step 2   cross-group : align reference TF channel    → reference soma channel
  Final shift per channel:
    ref-soma : (0, 0, 0)
    soma-N   : intra-soma shift
    ref-TF   : cross-group shift
    TF-N     : intra-TF shift + cross-group shift

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
    eff_z_pad = np.where(s_z2 > s_z1, z_pad, 0)
    contained_z = (s_z1 - eff_z_pad <= tf_z1c) & (tf_z2c <= s_z2 + eff_z_pad)
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
# 7b.  Full find_shift_containment: voxelize → 3D FFT → fine containment search
# ──────────────────────────────────────────────────────────────────────────────

def find_shift_containment(cells_ref_soma, cells_tf, z_lo, z_hi,
                           bin_size=4, xy_res_um=0.65, z_res_um=8.0,
                           xy_range_px=30, z_range_slices=5,
                           fine_xy_px=8, fine_z_slices=2,
                           max_center_dist_ratio=0.3, xy_margin=0, z_pad=0):
    """
    Find (dx, dy, dz) that maximises soma-TF containment count (see module
    docstring for why this replaces voxel-IoU for the soma<->TF cross-group
    alignment step).

    Strategy:
      1. Coarse: same 3D-FFT cross-correlation as find_shift() (unchanged),
         to get near the true peak even for large shifts.
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

    # Coarse: 3D FFT (identical to find_shift)
    dx_fft, dy_fft, dz_fft = _fft_3d_shifts(
        grid_ref, grid_tgt, xy_range_px, z_range_slices, bin_size
    )

    # Fine search in pixel/slice space around FFT peak, scored by containment
    soma_idx  = _prepare_soma_containment_index(cells_ref_soma)
    tf_arrays = _cell_arrays(cells_tf)

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
                score = _containment_score(
                    soma_idx, tf_arrays, dx_c, dy_c, dz_c,
                    max_center_dist_ratio, xy_margin, z_pad
                )
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
                                containment_z_pad=0):
    """
    Compute final (dx, dy, dz) per channel for one tile using two-step strategy.

    Parameters
    ----------
    per_ch_vol_lists : dict  ch_id → list of volumetric cell dicts
    soma_ch_ids      : list  ordered soma channel IDs; first is the reference
    tf_ch_ids        : list  ordered TF channel IDs; first is the reference TF
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
            xy_margin=xy_margin, z_pad=containment_z_pad,
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

def apply_shift_to_csv(in_csv_path, dx, dy, dz, out_csv_path):
    """
    Add (dx, dy) to bbox columns and dz to z column of a detection CSV.

    CSV columns expected: slice_name, x1, y1, x2, y2, class, score, mean, z
    Sign convention: aligned_coord = raw_coord + shift  (same as visualizer).
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
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    return out_path
