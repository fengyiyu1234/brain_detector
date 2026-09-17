import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

from src.utils.markers import class_markers, split_class


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
            matched_markers.extend(class_markers(nuc_classes[idx]))
        if not matched_markers:
            continue
        base_type, existing = split_class(final_merged[i][6])
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
            matched_markers.extend(class_markers(nuc_classes[idx]))
        if not matched_markers:
            continue
        base_type, existing = split_class(final_merged[i][6])
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
        base, a_markers = split_class(a[6])
        _,    b_markers = split_class(b[6])
        markers = sorted(set(a_markers) | set(b_markers))
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


_CELL_TYPE_PRIORITY = {'glia': 1, 'neuron': 0}

# Private provenance tags used only inside fuse_dual_intensity_2d to dodge _merge_class's
# bare-single-token branch (see that function's docstring). Never written to disk — stripped
# back to the bare base type before the fused rows are returned.
_DUAL_LO_TAG = "dilo"
_DUAL_HI_TAG = "dihi"


def _merge_class(cls_a, cls_b):
    """Merge two class strings, combining markers: 'neuron_RFP' + 'neuron_GFP' → 'neuron_GFP_RFP'.
    Glia takes priority over neuron when base types differ.
    If cls_b has no underscore (e.g. 'Sox9'), the whole string is treated as a bare marker."""
    base_a, mk_a = split_class(cls_a)
    base_b, mk_b = split_class(cls_b)
    pri_a = _CELL_TYPE_PRIORITY.get(base_a, -1)
    pri_b = _CELL_TYPE_PRIORITY.get(base_b, -1)
    base = base_a if pri_a >= pri_b else base_b
    markers_a = set(mk_a)
    markers_b = set(mk_b) if mk_b else {base_b}
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


_RAW_COLS = ["slice_name", "x1", "y1", "x2", "y2", "class", "score", "mean", "z"]
_BOX_COLS = ["x1", "y1", "x2", "y2", "score", "mean", "class", "z"]


def _raw_df_to_tagged_box_matrix(df, tag):
    """Raw 9-col tile dataframe -> (_BOX_COLS 8-col object matrix with tagged class, z->slice_name map).
    Tagging avoids _merge_class's bare-single-token branch (see fuse_dual_intensity_2d)."""
    if df.empty:
        return np.empty((0, 8), dtype=object), {}
    z_to_slice = dict(zip(df["z"].astype(int), df["slice_name"]))
    mat = df[_BOX_COLS].values.copy()
    mat[:, 6] = np.array([f"{v}_{tag}" for v in mat[:, 6]], dtype=object)
    return mat, z_to_slice


def fuse_dual_intensity_2d(low_df, high_df, iou_thresh=0.3):
    """
    Fuse two per-tile raw detection CSVs (same physical channel, two exposures/laser
    powers) at the 2D per-z-slice level via merge_soma_detections_2d, before any
    downstream filtering.

    low_df/high_df: raw 9-col dataframes (slice_name,x1,y1,x2,y2,class,score,mean,z),
    e.g. loaded from 1_tile_2d_raw or 0_channel_alignment.

    Classes are tagged internally before merging (so _merge_class's bare-token branch
    doesn't fire on plain untagged YOLO/StarDist classes like "neuron"/"glia" and
    fabricate a bogus "neuron_neuron" class), then stripped back to the bare base type
    on the fused output — no provenance tag ever reaches the caller.

    Returns (fused_df, n_low, n_high, n_fused) — fused_df in the same raw 9-col format.
    """
    n_low, n_high = len(low_df), len(high_df)

    low_mat, low_z2s = _raw_df_to_tagged_box_matrix(low_df, _DUAL_LO_TAG)
    high_mat, high_z2s = _raw_df_to_tagged_box_matrix(high_df, _DUAL_HI_TAG)

    fused_mat = merge_soma_detections_2d(low_mat, high_mat, iou_thresh=iou_thresh)
    n_fused = len(fused_mat)

    z_to_slice = dict(low_z2s)
    z_to_slice.update(high_z2s)

    rows = []
    for row in fused_mat:
        x1, y1, x2, y2, score, mean, cls, z = row
        base_cls = str(cls).split('_')[0]
        z_int = int(float(z))
        slice_name = z_to_slice.get(z_int, str(z_int))
        rows.append([slice_name, x1, y1, x2, y2, base_cls, score, mean, z_int])

    fused_df = pd.DataFrame(rows, columns=_RAW_COLS)
    return fused_df, n_low, n_high, n_fused


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
        _, cls_markers = split_class(cls)
        ch_id = cls_markers[0] if cls_markers else cls
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
                best_markers = class_markers(best_cls)
                new_markers.extend(best_markers if best_markers else [ch_id])

        if new_markers:
            row[6] = _merge_class(str(row[6]), '_'.join(sorted(set(new_markers))))

    return np.array(result, dtype=object)


# ──────────────────────────────────────────────────────────────────────────────
# 3-D volumetric colocalization (replaces 2-D channel merging)
# ──────────────────────────────────────────────────────────────────────────────

def _cells_to_arrays(cells, z_pad=0):
    """Extract bounding-box fields from a list of cell dicts into float32 NumPy arrays.
    Returns (x1,y1,x2,y2,z1,z2, coords[N,3], radii[N]) where coords/radii are for spatial queries.
    z_pad only widens the KDTree candidate-search radius here (conservative upper bound);
    the actual one-sided gap-bridging is applied later in _iou_3d_batch, not to these raw boxes."""
    x1 = np.array([c['x1_3d'] for c in cells], dtype=np.float32)
    y1 = np.array([c['y1_3d'] for c in cells], dtype=np.float32)
    x2 = np.array([c['x2_3d'] for c in cells], dtype=np.float32)
    y2 = np.array([c['y2_3d'] for c in cells], dtype=np.float32)
    z1 = np.array([c['z_min']  for c in cells], dtype=np.float32)
    z2 = np.array([c['z_max']  for c in cells], dtype=np.float32)
    cx = (x1 + x2) * 0.5;  cy = (y1 + y2) * 0.5;  cz = (z1 + z2) * 0.5
    rx = (x2 - x1) * 0.5;  ry = (y2 - y1) * 0.5;  rz = (z2 - z1 + 1 + z_pad) * 0.5
    radii = np.sqrt(rx**2 + ry**2 + rz**2)
    coords = np.stack([cx, cy, cz], axis=1).astype(np.float64)
    return x1, y1, x2, y2, z1, z2, coords, radii


def _iou_3d_batch(x1a, y1a, x2a, y2a, z1a, z2a,
                   x1b, y1b, x2b, y2b, z1b, z2b,
                   i_arr, j_arr, z_pad):
    """Vectorized 3D IoU and IoMin for candidate pairs given by index arrays i_arr, j_arr.
    Returns (iou_vals, iomin_vals) where iomin = intersection / min(vol_a, vol_b).

    z_pad is a one-sided gap-bridging tolerance, not a symmetric box inflation: it only
    fills in a real z-gap between two boxes that don't already overlap in z (up to z_pad
    slices), mirroring the single-side tolerance used for TF containment. Boxes that already
    overlap in z use their raw (unpadded) overlap/volume, so z_pad never dilutes the IoU of
    pairs that don't need bridging."""
    ix = np.maximum(0.0, np.minimum(x2a[i_arr], x2b[j_arr]) - np.maximum(x1a[i_arr], x1b[j_arr]))
    iy = np.maximum(0.0, np.minimum(y2a[i_arr], y2b[j_arr]) - np.maximum(y1a[i_arr], y1b[j_arr]))
    raw_iz  = (np.minimum(z2a[i_arr], z2b[j_arr]) - np.maximum(z1a[i_arr], z1b[j_arr]) + 1)
    depth_a = z2a[i_arr] - z1a[i_arr] + 1
    depth_b = z2b[j_arr] - z1b[j_arr] + 1
    # Bridged credit can never exceed either box's own depth, or inter would exceed va/vb.
    bridged_iz = np.minimum(np.maximum(0.0, raw_iz + z_pad), np.minimum(depth_a, depth_b))
    iz = np.where(raw_iz > 0, raw_iz, bridged_iz)
    inter = ix * iy * iz
    va = (x2a[i_arr] - x1a[i_arr]) * (y2a[i_arr] - y1a[i_arr]) * (z2a[i_arr] - z1a[i_arr] + 1)
    vb = (x2b[j_arr] - x1b[j_arr]) * (y2b[j_arr] - y1b[j_arr]) * (z2b[j_arr] - z1b[j_arr] + 1)
    iou   = inter / (va + vb - inter + 1e-8)
    iomin = inter / (np.minimum(va, vb) + 1e-8)
    return iou, iomin


def match_soma_3d_iou(cells_a, cells_b, iou_thresh=0.15, iomin_thresh=0.5, z_pad=0):
    """
    Match two lists of volumetric soma cells using 3D IoU (greedy matching).
    Vectorized: batch cKDTree query (workers=-1) + NumPy IoU, ~50-200x faster than
    the per-pair Python loop.
    Returns (matched_pairs, unmatched_a, unmatched_b).
    """
    if not cells_a or not cells_b:
        return [], list(cells_a), list(cells_b)

    n, m = len(cells_a), len(cells_b)

    x1a, y1a, x2a, y2a, z1a, z2a, coords_a, radii_a = _cells_to_arrays(cells_a, z_pad)
    x1b, y1b, x2b, y2b, z1b, z2b, coords_b, radii_b = _cells_to_arrays(cells_b, z_pad)

    tree_b = cKDTree(coords_b)

    # Batch query across all cores; use global max radius as conservative upper bound
    all_nearby = tree_b.query_ball_point(coords_a, r=radii_a + radii_b.max(), workers=-1)

    # Build flat (i, j) index arrays for all candidate pairs
    counts = np.array([len(nb) for nb in all_nearby], dtype=np.int64)
    if counts.sum() == 0:
        return [], list(cells_a), list(cells_b)

    i_arr = np.repeat(np.arange(n, dtype=np.int64), counts)
    j_arr = np.concatenate([np.asarray(nb, dtype=np.int64) for nb in all_nearby])

    # Refine with per-cell bounding-sphere check to discard false candidates
    # introduced by the global max-radius query
    dists_sq = np.sum((coords_a[i_arr] - coords_b[j_arr]) ** 2, axis=1)
    sphere_ok = dists_sq <= (radii_a[i_arr] + radii_b[j_arr]) ** 2
    i_arr, j_arr = i_arr[sphere_ok], j_arr[sphere_ok]

    if len(i_arr) == 0:
        return [], list(cells_a), list(cells_b)

    # Vectorized 3D IoU + IoMin for all remaining candidates
    iou_vals, iomin_vals = _iou_3d_batch(x1a, y1a, x2a, y2a, z1a, z2a,
                                          x1b, y1b, x2b, y2b, z1b, z2b,
                                          i_arr, j_arr, z_pad)

    # Match on standard IoU OR containment (IoMin): handles size-asymmetric pairs
    keep = (iou_vals >= iou_thresh) | (iomin_vals >= iomin_thresh)
    i_arr, j_arr, iou_vals = i_arr[keep], j_arr[keep], iou_vals[keep]

    # Greedy matching: highest IoU first
    order = np.argsort(-iou_vals)
    i_arr, j_arr = i_arr[order], j_arr[order]

    matched_a, matched_b = set(), set()
    matched_pairs = []
    for i, j in zip(i_arr.tolist(), j_arr.tolist()):
        if i not in matched_a and j not in matched_b:
            matched_pairs.append((cells_a[i], cells_b[j]))
            matched_a.add(i)
            matched_b.add(j)

    unmatched_a = [cells_a[i] for i in range(n) if i not in matched_a]
    unmatched_b = [cells_b[j] for j in range(m) if j not in matched_b]
    return matched_pairs, unmatched_a, unmatched_b


def suppress_cross_class_overlap(cells, iou_thresh=0.5, z_pad=2):
    """
    For any (neuron, glia) pair with 3D IoU > iou_thresh, drop the neuron.
    Glia takes priority.  Runs after Phase-A soma merging, before TF annotation.
    """
    neurons = [c for c in cells if str(c['class']).split('_')[0] == 'neuron']
    glias   = [c for c in cells if str(c['class']).split('_')[0] == 'glia']
    others  = [c for c in cells if str(c['class']).split('_')[0] not in ('neuron', 'glia')]

    if not neurons or not glias:
        return cells

    x1n, y1n, x2n, y2n, z1n, z2n, coords_n, radii_n = _cells_to_arrays(neurons, z_pad)
    x1g, y1g, x2g, y2g, z1g, z2g, coords_g, radii_g = _cells_to_arrays(glias,   z_pad)

    tree_g = cKDTree(coords_g)
    all_nearby = tree_g.query_ball_point(coords_n, r=radii_n + radii_g.max(), workers=-1)

    counts = np.array([len(nb) for nb in all_nearby], dtype=np.int64)
    suppressed = set()
    if counts.sum() > 0:
        i_arr = np.repeat(np.arange(len(neurons), dtype=np.int64), counts)
        j_arr = np.concatenate([np.asarray(nb, dtype=np.int64) for nb in all_nearby])

        dists_sq = np.sum((coords_n[i_arr] - coords_g[j_arr]) ** 2, axis=1)
        sphere_ok = dists_sq <= (radii_n[i_arr] + radii_g[j_arr]) ** 2
        i_arr, j_arr = i_arr[sphere_ok], j_arr[sphere_ok]

        if len(i_arr) > 0:
            iou_vals, _ = _iou_3d_batch(x1n, y1n, x2n, y2n, z1n, z2n,
                                         x1g, y1g, x2g, y2g, z1g, z2g,
                                         i_arr, j_arr, z_pad)
            suppressed = set(i_arr[iou_vals > iou_thresh].tolist())

    surviving = [nn for i, nn in enumerate(neurons) if i not in suppressed]
    logging.info(f"Cross-class dedup: suppressed {len(suppressed)} neuron(s) "
                 f"overlapping glia (iou_thresh={iou_thresh})")
    return surviving + glias + others


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
            tf_base, tf_mk = split_class(tf['class'])
            tf_marker = tf_mk[-1] if tf_mk else tf_base   # "nucleus_Sox9" → "Sox9"
            soma_markers[soma_idx].add(tf_marker)

    for idx, soma in enumerate(soma_vol_list):
        if idx in soma_markers:
            soma['class'] = _merge_class(
                soma['class'], '_'.join(sorted(soma_markers[idx]))
            )

    return soma_vol_list


_TF_CHUNK = 100_000   # 3B 每块处理的 TF 数；候选对数组的内存随块大小线性增长


def annotate_soma_with_tf_containment(soma_vol_list, tf_vol_list, z_pad=2, xy_margin=0,
                                       max_center_dist_ratio=0.5):
    """
    Annotate soma cells with TF markers using strict 3D bbox containment.
    A TF is colocalized only if its 3D bounding box is fully enclosed within
    a soma's 3D bounding box (with optional xy_margin and z_pad tolerance),
    AND its centroid is within max_center_dist_ratio * soma_radius of the soma center.
    When multiple somas contain the same TF, the closest one wins.
    """
    if not soma_vol_list or not tf_vol_list:
        return soma_vol_list

    soma_arr = np.array([[s['cx'], s['cy'], s['cz']] for s in soma_vol_list], dtype=float)
    soma_radii = np.array([
        np.sqrt((s['x2_3d'] - s['x1_3d']) ** 2 + (s['y2_3d'] - s['y1_3d']) ** 2) / 2
        for s in soma_vol_list
    ], dtype=float)
    soma_radii = np.maximum(soma_radii, 1.0)
    max_radius = float(soma_radii.max())

    tree = cKDTree(soma_arr)
    s_box = np.array([[s['x1_3d'], s['y1_3d'], s['x2_3d'], s['y2_3d']] for s in soma_vol_list],
                     dtype=float)
    s_zmin = np.array([s['z_min'] for s in soma_vol_list])
    s_zmax = np.array([s['z_max'] for s in soma_vol_list])

    # 分块批量处理：一次查询一整块 TF 的候选 soma，包含判断用数组运算（与逐个比较逐位相同）；
    # 只有通过包含判断的配对才进入原来的逐对距离判定，候选顺序与单点查询相同，
    # 所以「最近者胜、并列取先出现者」的结果与逐个处理完全一致。
    marker_of = {}
    soma_markers = defaultdict(set)
    for c0 in range(0, len(tf_vol_list), _TF_CHUNK):
        chunk = tf_vol_list[c0:c0 + _TF_CHUNK]
        tf_pts = np.array([[t['cx'], t['cy'], t['cz']] for t in chunk], dtype=float)
        t_box = np.array([[t['x1_3d'], t['y1_3d'], t['x2_3d'], t['y2_3d']] for t in chunk], dtype=float)
        t_zmin = np.array([t['z_min'] for t in chunk])
        t_zmax = np.array([t['z_max'] for t in chunk])

        cands = tree.query_ball_point(tf_pts, r=max_radius * 2, return_sorted=False, workers=-1)
        counts = np.fromiter((len(c) for c in cands), dtype=np.int64, count=len(cands))
        if counts.sum() == 0:
            continue
        ti = np.repeat(np.arange(len(chunk)), counts)
        si = np.fromiter((j for c in cands for j in c), dtype=np.int64, count=int(counts.sum()))
        sb, tb = s_box[si], t_box[ti]
        contained_xy = ((sb[:, 0] - xy_margin <= tb[:, 0]) & (tb[:, 2] <= sb[:, 2] + xy_margin) &
                        (sb[:, 1] - xy_margin <= tb[:, 1]) & (tb[:, 3] <= sb[:, 3] + xy_margin))
        # z_pad slices of tolerance on exactly one side only, never both at once
        # (padding both sides simultaneously lets a truncated box balloon into
        # an oversized capture window) — applies the same way to single- and
        # multi-layer soma boxes.
        zmin_t, zmax_t, zmin_s, zmax_s = t_zmin[ti], t_zmax[ti], s_zmin[si], s_zmax[si]
        contained_z = (((zmin_t >= zmin_s - z_pad) & (zmax_t <= zmax_s)) |
                       ((zmin_t >= zmin_s) & (zmax_t <= zmax_s + z_pad)))
        keep = np.flatnonzero(contained_xy & contained_z)

        best = {}   # tf 在块内的下标 → (best_dist, best_idx)
        for k in keep:
            t, idx = int(ti[k]), int(si[k])
            dist = float(np.linalg.norm(soma_arr[idx] - tf_pts[t]))
            # Hard gate: nucleus centroid must be near soma center, not just inside bbox
            if dist > max_center_dist_ratio * soma_radii[idx]:
                continue
            if dist < best.get(t, (float('inf'), None))[0]:
                best[t] = (dist, idx)

        for t, (_, best_idx) in best.items():
            cls = chunk[t]['class']
            if cls not in marker_of:
                tf_base, tf_mk = split_class(cls)
                marker_of[cls] = tf_mk[-1] if tf_mk else tf_base
            soma_markers[best_idx].add(marker_of[cls])

    for idx, soma in enumerate(soma_vol_list):
        if idx in soma_markers:
            soma['class'] = _merge_class(
                soma['class'], '_'.join(sorted(soma_markers[idx]))
            )

    return soma_vol_list


# ──────────────────────────────────────────────────────────────────────────────
# Tile-level detection stitching (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def stitchDetection(detections, H=None, W=None, xsize=None, ysize=None, step=None):
    if len(detections) == 0:
        return detections
    boxes = np.array(list(detections))
    return non_max_suppression_iou(boxes, overlapThresh=0.4, sort_idx=4)


def non_max_suppression_iou(boxes, overlapThresh=0.45, sort_idx=4, containment_thresh=None):
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
        iou        = inter_area / (area[i] + area[idxs[:last]] - inter_area + 1e-6)
        suppress   = iou > overlapThresh
        if containment_thresh is not None:
            # IoMin = inter / min(area_i, area_j): equals 1.0 when the smaller box is
            # fully inside the larger box, catching nested detections that IoU misses.
            iomin    = inter_area / (np.minimum(area[i], area[idxs[:last]]) + 1e-6)
            suppress = suppress | (iomin > containment_thresh)

        idxs = np.delete(idxs, np.concatenate(([last], np.where(suppress)[0])))

    return boxes[pick]


def combine_predictions(all_predictions, csv_reader, classes, z_start, Z, pos, disp_mat, size, metadata_registry, tile_name, tILESIZE=2048, file_z0=None, row_meta=None):
    """
    row_meta: 可选 dict，(层, 类别) → [(tile_name, slice_name), ...]，与写进
              all_predictions[层][类别] 的行一一对应、顺序相同（用于把溯源信息存进全局 2D CSV）。
    """
    row, col = pos
    ABS_X, ABS_Y, ABS_Z = disp_mat[pos]
    # 重叠区判定只涉及当前 tile 自身的局部范围，用 tile 大小的局部 mask
    # 即可，不要用全局拼接画布尺寸 (size)，否则超大画布会尝试分配 TB 级内存。
    mask = np.zeros((tILESIZE, tILESIZE), dtype=bool)
    if col > 0:
        x_pre_start = disp_mat[row, col - 1][0]; y_pre_start = disp_mat[row, col - 1][1]
        mask[max(ABS_Y, y_pre_start) - ABS_Y:min(ABS_Y + tILESIZE, y_pre_start + tILESIZE) - ABS_Y,
             max(ABS_X, x_pre_start) - ABS_X:min(ABS_X + tILESIZE, x_pre_start + tILESIZE) - ABS_X] = True
    if row > 0:
        x_pre_start = disp_mat[row - 1, col][0]; y_pre_start = disp_mat[row - 1, col][1]
        mask[max(ABS_Y, y_pre_start) - ABS_Y:min(ABS_Y + tILESIZE, y_pre_start + tILESIZE) - ABS_Y,
             max(ABS_X, x_pre_start) - ABS_X:min(ABS_X + tILESIZE, x_pre_start + tILESIZE) - ABS_X] = True
    z0 = z_start - ABS_Z
    z1 = z0 + Z

    # 先按 (层, 类别) 收集成列表，tile 读完后每组只 concatenate 一次。
    # 以前每个框都把整层数组复制一遍，稠密核通道（每层上万个框）耗时按平方增长。
    # 行的先后顺序与逐个追加时完全相同。
    new_rows = {}
    for row_data in csv_reader:
        slice_name, x1, y1, x2, y2, class_name, score, mean, z = row_data[:9]
        z    = int(float(z))
        x1   = float(x1); x2 = float(x2); y1 = float(y1); y2 = float(y2)
        score = float(score); mean = float(mean)
        if z - 1 in range(z0, z1):
            cy_local = min(max(int((y1 + y2) // 2), 0), tILESIZE - 1)
            cx_local = min(max(int((x1 + x2) // 2), 0), tILESIZE - 1)
            x1 += ABS_X; x2 += ABS_X; y1 += ABS_Y; y2 += ABS_Y; z = z - z0
            cell_type_index = 0 if 'glia' in class_name.lower() else 1

            if not mask[cy_local, cx_local]:
                # 非重叠区：写入
                new_rows.setdefault((z - 1, cell_type_index), []).append(
                    [x1, y1, x2, y2, score, mean, class_name, z])
                if row_meta is not None:
                    row_meta.setdefault((z - 1, cell_type_index), []).append((tile_name, slice_name))
                metadata_registry.append([(x1 + x2) / 2, (y1 + y2) / 2, z, tile_name, slice_name])
            # 重叠区：丢弃（保留左/上方 tile 的结果，右/下方 tile 的重叠区检测一律舍弃）

    for (zi, ti), rows in new_rows.items():
        all_predictions[zi][ti] = np.concatenate(
            (all_predictions[zi][ti], np.array(rows, dtype=object)))

    return all_predictions
