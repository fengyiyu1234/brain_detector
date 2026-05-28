import os
import numpy as np
import pandas as pd
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment


# ──────────────────────────────────────────────────────────────────────────────
# Module-level permutation worker (must be at top level for multiprocessing pickle)
# ──────────────────────────────────────────────────────────────────────────────

def _perm_batch_worker(args):
    """
    Runs a batch of permutations on one CPU core.
    Called in parallel by ProcessPoolExecutor inside permutation_test_colocalization.
    """
    (tf_cx, tf_cy, tf_z, tf_marker, soma_phys,
     unique_markers, distance_thresh_um, n_batch, seed_base) = args

    rng     = np.random.default_rng(seed_base)
    x_range = float(tf_cx.max() - tf_cx.min())
    y_range = float(tf_cy.max() - tf_cy.min())
    x_min   = float(tf_cx.min())
    y_min   = float(tf_cy.min())

    soma_tree = cKDTree(soma_phys)
    random_counts = {m: np.zeros(n_batch) for m in unique_markers}

    for i in range(n_batch):
        dx    = rng.uniform(0.1 * x_range, 0.9 * x_range)
        dy    = rng.uniform(0.1 * y_range, 0.9 * y_range)
        new_cx = (tf_cx - x_min + dx) % x_range + x_min
        new_cy = (tf_cy - y_min + dy) % y_range + y_min
        tf_tree = cKDTree(np.column_stack([new_cx, new_cy, tf_z]))
        pairs   = soma_tree.query_ball_tree(tf_tree, r=distance_thresh_um)

        counts_i = {m: 0 for m in unique_markers}
        for nbr_indices in pairs:
            found = {tf_marker[j] for j in nbr_indices if tf_marker[j]}
            for m in found:
                counts_i[m] += 1
        for m in unique_markers:
            random_counts[m][i] = counts_i[m]

    return random_counts


# ──────────────────────────────────────────────────────────────────────────────
# Internal GPU helpers (imported lazily so torch is optional)
# ──────────────────────────────────────────────────────────────────────────────

def _try_cuda():
    """Return (torch, device) if CUDA is available, else (None, None)."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch, torch.device('cuda')
    except ImportError:
        pass
    return None, None


def _gpu_colocalize_sphere(soma_centers, nuc_centers,
                            distance_thresh, torch, device, chunk=512):
    """
    GPU-accelerated sphere colocalization.
    Returns dict: soma_idx (int) → set of matched nuc class strings.
    """
    soma_t = torch.from_numpy(soma_centers.astype(np.float32)).to(device)
    nuc_t  = torch.from_numpy(nuc_centers.astype(np.float32)).to(device)
    r      = float(distance_thresh)

    soma_to_nucs = {}
    for start in range(0, soma_t.shape[0], chunk):
        d      = torch.cdist(soma_t[start:start + chunk], nuc_t)   # (chunk, N_nuc)
        rows_t, cols_t = torch.where(d < r)
        if rows_t.numel() == 0:
            continue
        for si_local, ni in zip(rows_t.cpu().numpy(), cols_t.cpu().numpy()):
            si = int(si_local) + start
            if si not in soma_to_nucs:
                soma_to_nucs[si] = []
            soma_to_nucs[si].append(int(ni))

    return soma_to_nucs


def _gpu_colocalize_box(soma_x1, soma_y1, soma_x2, soma_y2, soma_z,
                         nuc_cx, nuc_cy, nuc_cz, z_tolerance_slices,
                         torch, device, chunk=512):
    """
    GPU-accelerated centroid-in-box colocalization.
    Returns dict: soma_idx (int) → list of matched nuc indices.
    """
    sx1 = torch.from_numpy(soma_x1.astype(np.float32)).to(device)
    sy1 = torch.from_numpy(soma_y1.astype(np.float32)).to(device)
    sx2 = torch.from_numpy(soma_x2.astype(np.float32)).to(device)
    sy2 = torch.from_numpy(soma_y2.astype(np.float32)).to(device)
    sz  = torch.from_numpy(soma_z.astype(np.float32)).to(device)

    ncx = torch.from_numpy(nuc_cx.astype(np.float32)).to(device)
    ncy = torch.from_numpy(nuc_cy.astype(np.float32)).to(device)
    ncz = torch.from_numpy(nuc_cz.astype(np.float32)).to(device)
    ztol = float(z_tolerance_slices)

    soma_to_nucs = {}
    for start in range(0, sx1.shape[0], chunk):
        end   = start + chunk
        x1_c  = sx1[start:end].unsqueeze(1)   # (chunk, 1)
        y1_c  = sy1[start:end].unsqueeze(1)
        x2_c  = sx2[start:end].unsqueeze(1)
        y2_c  = sy2[start:end].unsqueeze(1)
        z_c   = sz[start:end].unsqueeze(1)

        in_x   = (ncx >= x1_c) & (ncx <= x2_c)    # (chunk, N_nuc)
        in_y   = (ncy >= y1_c) & (ncy <= y2_c)
        in_z   = torch.abs(ncz - z_c) <= ztol
        inside = in_x & in_y & in_z

        rows_t, cols_t = torch.where(inside)
        if rows_t.numel() == 0:
            continue
        for si_local, ni in zip(rows_t.cpu().numpy(), cols_t.cpu().numpy()):
            si = int(si_local) + start
            if si not in soma_to_nucs:
                soma_to_nucs[si] = []
            soma_to_nucs[si].append(int(ni))

    return soma_to_nucs


# ──────────────────────────────────────────────────────────────────────────────
# Colocalization functions
# ──────────────────────────────────────────────────────────────────────────────

def colocalize_3d(soma_3d_boxes, nuc_3d_boxes, xy_res=1.0, z_res=0.008, distance_thresh_um=15.0):
    """
    3D 空间点对体共定位：判断 TF/核 是否在 Soma 附近。
    GPU acceleration via torch.cdist when CUDA is available; CPU fallback uses
    cKDTree.query_ball_tree for a single batch query (replaces per-soma loop).
    """
    if len(soma_3d_boxes) == 0:
        return np.empty((0, 8), dtype=object)
    if len(nuc_3d_boxes) == 0:
        return soma_3d_boxes.copy()

    soma_centers = np.column_stack((
        (soma_3d_boxes[:, 0].astype(float) + soma_3d_boxes[:, 2].astype(float)) / 2 * xy_res,
        (soma_3d_boxes[:, 1].astype(float) + soma_3d_boxes[:, 3].astype(float)) / 2 * xy_res,
        soma_3d_boxes[:, 7].astype(float) * z_res
    ))
    nuc_centers = np.column_stack((
        (nuc_3d_boxes[:, 0].astype(float) + nuc_3d_boxes[:, 2].astype(float)) / 2 * xy_res,
        (nuc_3d_boxes[:, 1].astype(float) + nuc_3d_boxes[:, 3].astype(float)) / 2 * xy_res,
        nuc_3d_boxes[:, 7].astype(float) * z_res
    ))

    nuc_classes = [str(nuc_3d_boxes[i][6]) for i in range(len(nuc_3d_boxes))]

    torch, device = _try_cuda()
    if torch is not None:
        soma_to_nucs = _gpu_colocalize_sphere(
            soma_centers, nuc_centers,
            distance_thresh_um, torch, device
        )
    else:
        soma_tree = cKDTree(soma_centers)
        nuc_tree  = cKDTree(nuc_centers)
        pairs     = soma_tree.query_ball_tree(nuc_tree, r=distance_thresh_um)
        soma_to_nucs = {i: idxs for i, idxs in enumerate(pairs) if idxs}

    final_merged = soma_3d_boxes.copy()
    for i, nbr_indices in soma_to_nucs.items():
        matched_markers = []
        for idx in nbr_indices:
            parts = nuc_classes[idx].split('_')
            if len(parts) > 1:
                matched_markers.extend(parts[1:])
        if not matched_markers:
            continue
        current_class = str(final_merged[i][6])
        soma_parts    = current_class.split('_')
        base_type     = soma_parts[0]
        existing      = soma_parts[1:] if len(soma_parts) > 1 else []
        final_merged[i][6] = f"{base_type}_" + "_".join(sorted(set(existing + matched_markers)))

    return final_merged


def colocalize_3d_centroid_in_box(soma_3d_boxes, nuc_3d_boxes,
                                   xy_res=1.0, z_res=8.0,
                                   xy_tolerance_px=5, z_tolerance_slices=5):
    """
    Colocalize by checking whether a TF nucleus centroid falls inside the soma YOLO bbox,
    expanded by xy_tolerance_px (XY) and z_tolerance_slices (Z).
    GPU acceleration when CUDA available; CPU fallback uses cKDTree.query_ball_tree.
    """
    if len(soma_3d_boxes) == 0:
        return np.empty((0, 8), dtype=object)
    if len(nuc_3d_boxes) == 0:
        return soma_3d_boxes.copy()

    soma_x1 = soma_3d_boxes[:, 0].astype(float) - xy_tolerance_px
    soma_y1 = soma_3d_boxes[:, 1].astype(float) - xy_tolerance_px
    soma_x2 = soma_3d_boxes[:, 2].astype(float) + xy_tolerance_px
    soma_y2 = soma_3d_boxes[:, 3].astype(float) + xy_tolerance_px
    soma_z  = soma_3d_boxes[:, 7].astype(float)

    nuc_cx = (nuc_3d_boxes[:, 0].astype(float) + nuc_3d_boxes[:, 2].astype(float)) / 2
    nuc_cy = (nuc_3d_boxes[:, 1].astype(float) + nuc_3d_boxes[:, 3].astype(float)) / 2
    nuc_cz = nuc_3d_boxes[:, 7].astype(float)

    nuc_classes = [str(nuc_3d_boxes[i][6]) for i in range(len(nuc_3d_boxes))]

    torch, device = _try_cuda()
    if torch is not None:
        soma_to_nucs = _gpu_colocalize_box(
            soma_x1, soma_y1, soma_x2, soma_y2, soma_z,
            nuc_cx, nuc_cy, nuc_cz, z_tolerance_slices,
            torch, device
        )
    else:
        z_scale       = z_res / xy_res
        nuc_pts       = np.column_stack([nuc_cx, nuc_cy, nuc_cz * z_scale])
        nuc_tree      = cKDTree(nuc_pts)
        z_tol_scaled  = z_tolerance_slices * z_scale
        half_w        = (soma_x2 - soma_x1) / 2
        half_h        = (soma_y2 - soma_y1) / 2
        global_r      = float(np.sqrt(half_w.max()**2 + half_h.max()**2 + z_tol_scaled**2) + 1.0)
        soma_cx_c     = (soma_x1 + soma_x2) / 2
        soma_cy_c     = (soma_y1 + soma_y2) / 2
        soma_pts      = np.column_stack([soma_cx_c, soma_cy_c, soma_z * z_scale])
        soma_tree     = cKDTree(soma_pts)
        pairs         = soma_tree.query_ball_tree(nuc_tree, r=global_r)

        soma_to_nucs = {}
        for i, candidates in enumerate(pairs):
            x1, y1, x2, y2, z_s = soma_x1[i], soma_y1[i], soma_x2[i], soma_y2[i], soma_z[i]
            matched = [
                idx for idx in candidates
                if (x1 <= nuc_cx[idx] <= x2 and y1 <= nuc_cy[idx] <= y2
                    and abs(nuc_cz[idx] - z_s) <= z_tolerance_slices)
            ]
            if matched:
                soma_to_nucs[i] = matched

    final_merged = soma_3d_boxes.copy()
    for i, nbr_indices in soma_to_nucs.items():
        matched_markers = []
        for idx in nbr_indices:
            parts = nuc_classes[idx].split('_')
            if len(parts) > 1:
                matched_markers.extend(parts[1:])
        if not matched_markers:
            continue
        current_class = str(final_merged[i][6])
        soma_parts    = current_class.split('_')
        base_type     = soma_parts[0]
        existing      = soma_parts[1:] if len(soma_parts) > 1 else []
        final_merged[i][6] = f"{base_type}_" + "_".join(sorted(set(existing + matched_markers)))

    return final_merged


def colocalize_soma_channels(ch_a_boxes, ch_b_boxes,
                              xy_tolerance_px=10, z_tolerance_slices=5):
    """
    Merge overlapping detections from two soma channels (e.g. RFP and GFP)
    into single deduplicated entries. Matched pairs → one record with combined
    markers (higher-score bbox wins). Unmatched entries pass through unchanged.
    """
    if len(ch_a_boxes) == 0:
        return ch_b_boxes.copy()
    if len(ch_b_boxes) == 0:
        return ch_a_boxes.copy()

    def _centroids(boxes):
        cx = (boxes[:, 0].astype(float) + boxes[:, 2].astype(float)) / 2
        cy = (boxes[:, 1].astype(float) + boxes[:, 3].astype(float)) / 2
        cz = boxes[:, 7].astype(float)
        return cx, cy, cz

    ax, ay, az = _centroids(ch_a_boxes)
    bx, by, bz = _centroids(ch_b_boxes)

    dx = ax[:, None] - bx[None, :]
    dy = ay[:, None] - by[None, :]
    dz = np.abs(az[:, None] - bz[None, :])
    xy_dist = np.sqrt(dx**2 + dy**2)

    cost = xy_dist.copy()
    cost[(xy_dist > xy_tolerance_px) | (dz > z_tolerance_slices)] = 1e6

    row_ind, col_ind = linear_sum_assignment(cost)

    matched_a, matched_b = set(), set()
    merged = []
    for ri, ci in zip(row_ind, col_ind):
        if cost[ri, ci] >= 1e6:
            continue
        a, b = ch_a_boxes[ri], ch_b_boxes[ci]
        winner = a if float(a[4]) >= float(b[4]) else b
        a_parts = str(a[6]).split('_')
        b_parts = str(b[6]).split('_')
        base = a_parts[0]
        markers = sorted(set(a_parts[1:]) | set(b_parts[1:]))
        merged_row = winner.copy()
        merged_row[6] = f"{base}_" + "_".join(markers) if markers else base
        merged.append(merged_row)
        matched_a.add(ri)
        matched_b.add(ci)

    unmatched_a = [ch_a_boxes[i] for i in range(len(ch_a_boxes)) if i not in matched_a]
    unmatched_b = [ch_b_boxes[j] for j in range(len(ch_b_boxes)) if j not in matched_b]

    all_rows = merged + unmatched_a + unmatched_b
    return np.array(all_rows, dtype=object)


# ──────────────────────────────────────────────────────────────────────────────
# 2D pre-z_link colocalization helpers
# ──────────────────────────────────────────────────────────────────────────────

def _iou_matrix_2d(rows_a, rows_b):
    """Vectorised 2D IoU: list-of-rows × list-of-rows → (n_a, n_b) array."""
    a = np.array([[r[0], r[1], r[2], r[3]] for r in rows_a], dtype=float)
    b = np.array([[r[0], r[1], r[2], r[3]] for r in rows_b], dtype=float)
    ix1 = np.maximum(a[:, 0:1], b[:, 0])
    iy1 = np.maximum(a[:, 1:2], b[:, 1])
    ix2 = np.minimum(a[:, 2:3], b[:, 2])
    iy2 = np.minimum(a[:, 3:4], b[:, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-8)


def _merge_class(cls_a, cls_b):
    """Merge two class strings, combining markers: 'neuron_RFP' + 'neuron_GFP' → 'neuron_GFP_RFP'.
    If cls_b has no underscore (e.g. 'Sox9'), the whole string is treated as a bare marker."""
    parts_a = str(cls_a).split('_')
    parts_b = str(cls_b).split('_')
    base = parts_a[0]
    markers_a = set(parts_a[1:])
    markers_b = set(parts_b[1:]) if len(parts_b) > 1 else set(parts_b)
    markers = sorted(markers_a | markers_b)
    return f"{base}_" + "_".join(markers) if markers else base


def merge_soma_detections_2d(ch_a_matrix, ch_b_matrix, iou_thresh=0.1):
    """
    Per-Z-slice IoU matching of two soma channel detection matrices.
    Matched pairs → one merged row (higher-score bbox, combined class markers).
    Unmatched rows pass through unchanged.
    """
    if len(ch_a_matrix) == 0:
        return ch_b_matrix.copy() if len(ch_b_matrix) else np.empty((0, 8), dtype=object)
    if len(ch_b_matrix) == 0:
        return ch_a_matrix.copy()

    all_z = sorted(set(int(r[7]) for r in ch_a_matrix) |
                   set(int(r[7]) for r in ch_b_matrix))
    a_by_z = defaultdict(list)
    b_by_z = defaultdict(list)
    for r in ch_a_matrix:
        a_by_z[int(r[7])].append(r)
    for r in ch_b_matrix:
        b_by_z[int(r[7])].append(r)

    result = []
    for z in all_z:
        a_rows, b_rows = a_by_z[z], b_by_z[z]
        if not a_rows:
            result.extend(b_rows)
            continue
        if not b_rows:
            result.extend(a_rows)
            continue

        iou_mat = _iou_matrix_2d(a_rows, b_rows)
        cost = 1.0 - iou_mat
        cost[iou_mat < iou_thresh] = 1e6
        row_ind, col_ind = linear_sum_assignment(cost)

        matched_a, matched_b = set(), set()
        for ri, ci in zip(row_ind, col_ind):
            if cost[ri, ci] >= 1e6:
                continue
            a, b = a_rows[ri], b_rows[ci]
            winner = a if float(a[4]) >= float(b[4]) else b
            merged = list(winner)
            merged[6] = _merge_class(str(a[6]), str(b[6]))
            result.append(merged)
            matched_a.add(ri)
            matched_b.add(ci)

        result.extend(a_rows[i] for i in range(len(a_rows)) if i not in matched_a)
        result.extend(b_rows[j] for j in range(len(b_rows)) if j not in matched_b)

    return np.array(result, dtype=object) if result else np.empty((0, 8), dtype=object)


def annotate_soma_with_tf_2d(soma_matrix, tf_matrix, z_tolerance_slices=0):
    """
    For each soma 2D detection, find TF nucleus detections whose bbox is fully
    contained within the soma bbox and within a Z window (default 0 = same slice).
    Keeps only the highest-score TF match per TF channel.
    """
    if len(soma_matrix) == 0 or len(tf_matrix) == 0:
        return soma_matrix.copy() if len(soma_matrix) else np.empty((0, 8), dtype=object)

    # Build per-TF-channel lookup: ch_id → z → [(tx1, ty1, tx2, ty2, score, cls)]
    tf_by_channel = defaultdict(lambda: defaultdict(list))
    for row in tf_matrix:
        cls = str(row[6])
        parts = cls.split('_')
        ch_id = parts[1] if len(parts) > 1 else cls
        tf_by_channel[ch_id][int(row[7])].append(
            (float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), cls)
        )

    result = [list(r) for r in soma_matrix]
    for i, row in enumerate(result):
        z = int(row[7])
        s_x1, s_y1 = float(row[0]), float(row[1])
        s_x2, s_y2 = float(row[2]), float(row[3])

        new_markers = []
        for ch_id, z_dict in tf_by_channel.items():
            candidates = []
            for dz in range(-z_tolerance_slices, z_tolerance_slices + 1):
                for (tx1, ty1, tx2, ty2, score, cls) in z_dict.get(z + dz, []):
                    # TF bbox must be fully inside soma bbox (strict containment)
                    if s_x1 <= tx1 and s_y1 <= ty1 and tx2 <= s_x2 and ty2 <= s_y2:
                        candidates.append((score, cls))
            if candidates:
                best_cls = max(candidates, key=lambda x: x[0])[1]
                parts = best_cls.split('_')
                new_markers.extend(parts[1:] if len(parts) > 1 else [ch_id])

        if new_markers:
            row[6] = _merge_class(str(row[6]), '_'.join(sorted(set(new_markers))))

    return np.array(result, dtype=object)


# ──────────────────────────────────────────────────────────────────────────────
# 3-D volumetric colocalization (replaces 2-D channel merging)
# ──────────────────────────────────────────────────────────────────────────────

def _iou_3d(a, b):
    """3D IoU between two volumetric cell dicts."""
    ix = max(0.0, min(a['x2_3d'], b['x2_3d']) - max(a['x1_3d'], b['x1_3d']))
    iy = max(0.0, min(a['y2_3d'], b['y2_3d']) - max(a['y1_3d'], b['y1_3d']))
    iz = max(0.0, min(a['z_max'], b['z_max']) - max(a['z_min'], b['z_min']) + 1)
    inter = ix * iy * iz
    if inter == 0.0:
        return 0.0
    va = (a['x2_3d'] - a['x1_3d']) * (a['y2_3d'] - a['y1_3d']) * (a['z_max'] - a['z_min'] + 1)
    vb = (b['x2_3d'] - b['x1_3d']) * (b['y2_3d'] - b['y1_3d']) * (b['z_max'] - b['z_min'] + 1)
    return inter / (va + vb - inter + 1e-8)


def match_soma_3d_iou(cells_a, cells_b, iou_thresh=0.15):
    """
    Match two lists of volumetric soma cells using 3D IoU (Hungarian assignment).
    Returns (matched_pairs, unmatched_a, unmatched_b).
      matched_pairs: list of (cell_a, cell_b) dicts
      unmatched_a/b: lists of unmatched cell dicts
    """
    if not cells_a or not cells_b:
        return [], list(cells_a), list(cells_b)

    n, m = len(cells_a), len(cells_b)
    iou_mat = np.zeros((n, m), dtype=float)
    for i, a in enumerate(cells_a):
        for j, b in enumerate(cells_b):
            iou_mat[i, j] = _iou_3d(a, b)

    cost = 1.0 - iou_mat
    cost[iou_mat < iou_thresh] = 1e6
    row_ind, col_ind = linear_sum_assignment(cost)

    matched_pairs, matched_a, matched_b = [], set(), set()
    for ri, ci in zip(row_ind, col_ind):
        if cost[ri, ci] >= 1e6:
            continue
        matched_pairs.append((cells_a[ri], cells_b[ci]))
        matched_a.add(ri)
        matched_b.add(ci)

    unmatched_a = [cells_a[i] for i in range(n) if i not in matched_a]
    unmatched_b = [cells_b[j] for j in range(m) if j not in matched_b]
    return matched_pairs, unmatched_a, unmatched_b


def annotate_soma_with_tf_gmm(soma_vol_list, tf_vol_list, p_thresh=0.5):
    """
    Annotate soma cells with TF markers using per-sample GMM on normalized distances.

    For each TF cell, finds the nearest soma centroid and computes:
        d_norm = euclidean_3d(tf_centroid, soma_centroid) / soma_radius
    Fits a 2-component GMM on the sample's d_norm distribution.
    Annotates soma whose nearest TF has P(coloc component | d_norm) > p_thresh.

    Modifies soma dicts in-place (updates 'class' field). Returns soma_vol_list.
    """
    from sklearn.mixture import GaussianMixture

    if not soma_vol_list or not tf_vol_list:
        return soma_vol_list

    soma_arr = np.array([[s['cx'], s['cy'], s['cz']] for s in soma_vol_list], dtype=float)
    soma_radii = np.array([
        np.sqrt((s['x2_3d'] - s['x1_3d']) * (s['y2_3d'] - s['y1_3d'])) / 2
        for s in soma_vol_list
    ], dtype=float)
    soma_radii = np.maximum(soma_radii, 1.0)  # guard against zero-size

    # For each TF cell find the nearest soma and compute normalised distance
    tf_soma_pairs = []
    dists_norm = []
    for tf in tf_vol_list:
        tf_pt = np.array([tf['cx'], tf['cy'], tf['cz']], dtype=float)
        diffs = soma_arr - tf_pt
        raw_dists = np.sqrt((diffs ** 2).sum(axis=1))
        nearest_idx = int(np.argmin(raw_dists))
        d_norm = raw_dists[nearest_idx] / soma_radii[nearest_idx]
        dists_norm.append(d_norm)
        tf_soma_pairs.append((tf, nearest_idx, d_norm))

    dists_arr = np.array(dists_norm, dtype=float).reshape(-1, 1)
    dists_arr = np.clip(dists_arr, 0, np.percentile(dists_arr, 99))  # clip outliers for GMM

    gmm = GaussianMixture(n_components=2, random_state=0).fit(dists_arr)
    coloc_comp = int(np.argmin(gmm.means_))

    # Group best TF match (highest P_coloc) per soma
    soma_markers = defaultdict(set)   # soma_idx → set of marker strings
    for tf, soma_idx, d_norm in tf_soma_pairs:
        proba = gmm.predict_proba([[d_norm]])[0][coloc_comp]
        if proba > p_thresh:
            tf_marker = tf['class'].split('_')[-1]  # e.g. "nucleus_Sox9" → "Sox9"
            soma_markers[soma_idx].add(tf_marker)

    for idx, soma in enumerate(soma_vol_list):
        if idx in soma_markers:
            soma['class'] = _merge_class(
                soma['class'], '_'.join(sorted(soma_markers[idx]))
            )

    return soma_vol_list


# ──────────────────────────────────────────────────────────────────────────────
# Permutation test
# ──────────────────────────────────────────────────────────────────────────────

def permutation_test_colocalization(soma_3d_boxes, tf_3d_boxes,
                                    xy_res=1.0, z_res=8.0,
                                    distance_thresh_um=15.0,
                                    n_permutations=1000,
                                    max_soma_sample=50_000):
    """
    对 3D 共定位结果进行置换检验，评估每个 TF marker 的共定位是否显著高于随机水平。

    加速策略：
      - Soma 随机采样（当总数 > max_soma_sample 时）
      - TF 裁剪到采样 soma 的空间边界框 + 缓冲区
      - 置换循环并行化：使用 CPU 多进程（28 核），每个进程处理若干次置换
    """
    if len(soma_3d_boxes) == 0 or len(tf_3d_boxes) == 0:
        return pd.DataFrame()

    tf_marker_all = []
    for box in tf_3d_boxes:
        parts = str(box[6]).split('_')
        tf_marker_all.append(parts[1] if len(parts) > 1 else None)

    unique_markers = sorted({m for m in tf_marker_all if m})
    if not unique_markers:
        return pd.DataFrame()

    # --- Soma 随机采样 ---
    n_soma_total = len(soma_3d_boxes)
    rng = np.random.default_rng(seed=42)
    if n_soma_total > max_soma_sample:
        sample_idx  = rng.choice(n_soma_total, size=max_soma_sample, replace=False)
        soma_sample = soma_3d_boxes[sample_idx]
        sampled     = True
    else:
        soma_sample = soma_3d_boxes
        sampled     = False
    n_soma_used = len(soma_sample)

    # --- Soma 物理坐标（μm）---
    soma_phys = np.column_stack([
        (soma_sample[:, 0].astype(float) + soma_sample[:, 2].astype(float)) / 2 * xy_res,
        (soma_sample[:, 1].astype(float) + soma_sample[:, 3].astype(float)) / 2 * xy_res,
        soma_sample[:, 7].astype(float) * z_res,
    ])

    # --- 全量 TF 物理坐标 ---
    tf_cx_all = (tf_3d_boxes[:, 0].astype(float) + tf_3d_boxes[:, 2].astype(float)) / 2 * xy_res
    tf_cy_all = (tf_3d_boxes[:, 1].astype(float) + tf_3d_boxes[:, 3].astype(float)) / 2 * xy_res
    tf_z_all  =  tf_3d_boxes[:, 7].astype(float) * z_res

    # --- 将 TF 裁剪到采样 soma 的边界框 + 距离缓冲区 ---
    buf = distance_thresh_um
    tf_mask = (
        (tf_cx_all >= soma_phys[:, 0].min() - buf) & (tf_cx_all <= soma_phys[:, 0].max() + buf) &
        (tf_cy_all >= soma_phys[:, 1].min() - buf) & (tf_cy_all <= soma_phys[:, 1].max() + buf) &
        (tf_z_all  >= soma_phys[:, 2].min() - buf) & (tf_z_all  <= soma_phys[:, 2].max() + buf)
    )
    keep      = np.where(tf_mask)[0]
    tf_cx     = tf_cx_all[keep]
    tf_cy     = tf_cy_all[keep]
    tf_z      = tf_z_all[keep]
    tf_marker = [tf_marker_all[i] for i in keep]

    x_range = tf_cx.max() - tf_cx.min() if len(tf_cx) > 1 else 0.0
    y_range = tf_cy.max() - tf_cy.min() if len(tf_cy) > 1 else 0.0
    if x_range < 1e-6 or y_range < 1e-6:
        return pd.DataFrame()

    # --- 计算真实共定位数（CPU, 单次，快速）---
    soma_tree = cKDTree(soma_phys)

    def count_coloc(cx_arr, cy_arr):
        tf_tree = cKDTree(np.column_stack([cx_arr, cy_arr, tf_z]))
        pairs   = soma_tree.query_ball_tree(tf_tree, r=distance_thresh_um)
        counts  = {m: 0 for m in unique_markers}
        for nbr_indices in pairs:
            found = {tf_marker[ti] for ti in nbr_indices if tf_marker[ti]}
            for m in found:
                counts[m] += 1
        return counts

    actual_counts = count_coloc(tf_cx, tf_cy)

    # --- 置换循环：CPU 多进程并行 ---
    n_workers    = min(os.cpu_count() or 1, 28, n_permutations)
    batch_size   = max(1, (n_permutations + n_workers - 1) // n_workers)
    batches = []
    for i in range(0, n_permutations, batch_size):
        n_batch = min(batch_size, n_permutations - i)
        batches.append((
            tf_cx.copy(), tf_cy.copy(), tf_z.copy(), list(tf_marker),
            soma_phys.copy(), list(unique_markers),
            distance_thresh_um, n_batch,
            42 + i * 1337,   # unique seed per batch
        ))

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        batch_results = list(ex.map(_perm_batch_worker, batches))

    random_counts = {
        m: np.concatenate([r[m] for r in batch_results])
        for m in unique_markers
    }

    # --- 统计显著性 ---
    rows = []
    for m in unique_markers:
        n_act   = actual_counts[m]
        rand    = random_counts[m]
        z_score = (n_act - rand.mean()) / (rand.std() + 1e-8)
        p_value = float(np.mean(rand >= n_act))
        rows.append({
            'tf_marker':        m,
            'n_soma_total':     n_soma_total,
            'n_soma_used':      n_soma_used,
            'sampled':          sampled,
            'n_coloc_actual':   n_act,
            'pct_coloc_actual': round(n_act / n_soma_used * 100, 2),
            'n_random_mean':    round(rand.mean(), 1),
            'n_random_std':     round(rand.std(), 1),
            'pct_random_mean':  round(rand.mean() / n_soma_used * 100, 2),
            'z_score':          round(z_score, 3),
            'p_value':          round(p_value, 4),
            'significant_p05':  p_value < 0.05,
        })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Tile-level detection stitching (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def stitchDetection(detections, H=None, W=None, xsize=None, ysize=None, step=None):
    if len(detections) == 0:
        return detections
    boxes = np.array(list(detections))
    return non_max_suppression_iou(boxes, overlapThresh=0.4, sort_idx=4)


def non_max_suppression_iou(boxes, overlapThresh=0.45, sort_idx=4):
    if len(boxes) == 0:
        return []

    pick = []
    x1 = boxes[:, 0].astype(float)
    y1 = boxes[:, 1].astype(float)
    x2 = boxes[:, 2].astype(float)
    y2 = boxes[:, 3].astype(float)

    w_init = np.maximum(0, x2 - x1)
    h_init = np.maximum(0, y2 - y1)
    area   = w_init * h_init
    idxs   = np.argsort(boxes[:, sort_idx].astype(float))

    while len(idxs) > 0:
        last = len(idxs) - 1
        i    = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w   = np.maximum(0, xx2 - xx1)
        h   = np.maximum(0, yy2 - yy1)
        inter_area = w * h
        iou = inter_area / (area[i] + area[idxs[:last]] - inter_area + 1e-6)

        idxs = np.delete(idxs, np.concatenate(([last], np.where(iou > overlapThresh)[0])))

    return boxes[pick]


def combine_predictions(all_predictions, csv_reader, classes, z_start, Z, pos, disp_mat, size, metadata_registry, tile_name, tILESIZE=2048, file_z0=None):
    row, col = pos
    ABS_X, ABS_Y, ABS_Z = disp_mat[pos]
    H, W = size
    mask = np.zeros((H, W), dtype=float)
    if col > 0:
        x_pre_start = disp_mat[row, col - 1][0]; y_pre_start = disp_mat[row, col - 1][1]
        mask[max(ABS_Y, y_pre_start):min(ABS_Y + tILESIZE, y_pre_start + tILESIZE),
             max(ABS_X, x_pre_start):min(ABS_X + tILESIZE, x_pre_start + tILESIZE)] = 1
    if row > 0:
        x_pre_start = disp_mat[row - 1, col][0]; y_pre_start = disp_mat[row - 1, col][1]
        mask[max(ABS_Y, y_pre_start):min(ABS_Y + tILESIZE, y_pre_start + tILESIZE),
             max(ABS_X, x_pre_start):min(ABS_X + tILESIZE, x_pre_start + tILESIZE)] = 1
    z0 = z_start - ABS_Z
    z1 = z0 + Z

    for row_data in csv_reader:
        slice_name, x1, y1, x2, y2, class_name, score, mean, z = row_data[:9]
        z    = int(float(z))
        x1   = float(x1); x2 = float(x2); y1 = float(y1); y2 = float(y2)
        score = float(score); mean = float(mean)
        if z - 1 in range(z0, z1):
            x1 += ABS_X; x2 += ABS_X; y1 += ABS_Y; y2 += ABS_Y; z = z - z0
            if not mask[int((y1 + y2) // 2), int((x1 + x2) // 2)] > 0:
                cell_type_index = 0 if 'glia' in class_name.lower() else 1
                new_box = np.array([[x1, y1, x2, y2, score, mean, class_name, z]], dtype=object)
                all_predictions[z - 1][cell_type_index] = np.concatenate(
                    (all_predictions[z - 1][cell_type_index], new_box)
                )
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                metadata_registry.append([cx, cy, z, tile_name, slice_name])

    return all_predictions
