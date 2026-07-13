import numpy as np
from scipy.optimize import linear_sum_assignment


def _iou_matrix(det_boxes, track_boxes):
    """Vectorized IoU: (n_d, 4) × (n_t, 4) → (n_d, n_t)."""
    x1 = np.maximum(det_boxes[:, 0:1], track_boxes[:, 0])
    y1 = np.maximum(det_boxes[:, 1:2], track_boxes[:, 1])
    x2 = np.minimum(det_boxes[:, 2:3], track_boxes[:, 2])
    y2 = np.minimum(det_boxes[:, 3:4], track_boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_d = (det_boxes[:, 2] - det_boxes[:, 0]) * (det_boxes[:, 3] - det_boxes[:, 1])
    area_t = (track_boxes[:, 2] - track_boxes[:, 0]) * (track_boxes[:, 3] - track_boxes[:, 1])
    union = area_d[:, None] + area_t[None, :] - inter
    return inter / (union + 1e-8)


def parse_class_string(cls_str):
    """
    解析动态生成的类别字符串，例如 "neuron_RFP_Sox9"
    返回: base_type ("neuron"), markers_set ({"RFP", "Sox9"})
    """
    parts = str(cls_str).split('_')
    base_type = parts[0]
    markers = set(parts[1:]) if len(parts) > 1 else set()
    return base_type, markers


def run_z_linker(full_stack_matrix, iou_thresh=0.45, min_z_layers=2,
                 max_cell_z_span=5, max_z_gap=0):
    """
    Returns (summary_array, volumetric_list).
      summary_array:    np.ndarray (N, 8) — one row per cell, center_z, for visualization CSV
      volumetric_list:  list[dict]        — per-cell volumetric info for 3D colocalization
    """
    if isinstance(full_stack_matrix, list):
        full_stack_matrix = np.array(full_stack_matrix, dtype=object)
    if full_stack_matrix.size == 0:
        return np.empty((0, 8), dtype=object), []

    z_min, z_max = int(np.min(full_stack_matrix[:, 7])), int(np.max(full_stack_matrix[:, 7]))
    z_groups = {z: [] for z in range(z_min, z_max + 1)}
    for det in full_stack_matrix:
        z_groups[int(det[7])].append(det)

    active_tracks = []
    finished_tracks = []

    for z in range(z_min, z_max + 1):
        curr_detections = z_groups[z]

        for track in active_tracks:
            if (z - track['first_z'] >= max_cell_z_span or
                    z - track['last_z'] - 1 > max_z_gap):
                track['active'] = False

        matched_det_indices = set()
        active_t_indices = [ti for ti, t in enumerate(active_tracks) if t['active']]

        if curr_detections and active_t_indices:
            # Pre-parse all class strings for this z-slice once
            det_parsed   = [parse_class_string(d[6]) for d in curr_detections]
            det_bases    = np.array([p[0] for p in det_parsed])
            track_bases  = np.array([active_tracks[ti]['base_type'] for ti in active_t_indices])

            # Vectorized IoU matrix replaces the nested per-pair Python loop
            det_boxes    = np.array([d[:4] for d in curr_detections], dtype=float)
            track_boxes  = np.array([active_tracks[ti]['last_box'] for ti in active_t_indices], dtype=float)
            iou_mat      = _iou_matrix(det_boxes, track_boxes)     # (n_d, n_t)

            cost_mat = 1.0 - iou_mat
            cost_mat[det_bases[:, None] != track_bases[None, :]] = 1e6  # cross-type → infeasible

            row_ind, col_ind = linear_sum_assignment(cost_mat)
            for d_idx, t_pos in zip(row_ind, col_ind):
                if cost_mat[d_idx, t_pos] > 1.0 - iou_thresh:
                    continue
                ti    = active_t_indices[t_pos]
                track = active_tracks[ti]
                det   = curr_detections[d_idx]
                _, det_markers = det_parsed[d_idx]   # reuse pre-parsed result
                track['all_boxes'].append(det)
                track['last_box'] = det[:4]
                track['last_z']   = z
                track['all_markers'].update(det_markers)
                track['per_z_boxes'][z] = [float(det[0]), float(det[1]), float(det[2]), float(det[3])]
                matched_det_indices.add(d_idx)

        finished_tracks.extend([t for t in active_tracks if not t['active']])
        active_tracks = [t for t in active_tracks if t['active']]

        for idx, det in enumerate(curr_detections):
            if idx not in matched_det_indices:
                curr_base_type, curr_markers = parse_class_string(det[6])
                active_tracks.append({
                    'all_boxes':   [det],
                    'last_box':    det[:4],
                    'last_z':      z,
                    'first_z':     z,
                    'base_type':   curr_base_type,
                    'all_markers': curr_markers,
                    'active':      True,
                    'per_z_boxes': {z: [float(det[0]), float(det[1]), float(det[2]), float(det[3])]},
                })

    finished_tracks.extend(active_tracks)

    final_rows = []
    volumetric_list = []
    for track in finished_tracks:
        if len(track['all_boxes']) >= min_z_layers:
            boxes    = track['all_boxes']
            best_det = max(boxes, key=lambda x: (len(str(x[6]).split('_')) - 1, float(x[4])))
            best_x1, best_y1, best_x2, best_y2, best_score, best_mean = best_det[:6]
            z_list   = [b[7] for b in boxes]
            center_z = int(np.median(z_list))
            marker_str = "_".join(sorted(track['all_markers']))
            final_class_name = (f"{track['base_type']}_{marker_str}" if marker_str
                                else track['base_type'])
            final_rows.append([
                best_x1, best_y1, best_x2, best_y2,
                best_score, best_mean,
                final_class_name, center_z
            ])

            # Build cubic 3D bounding box (each axis independently)
            pzb = track['per_z_boxes']
            all_w = [v[2] - v[0] for v in pzb.values()]
            all_h = [v[3] - v[1] for v in pzb.values()]
            cx = (float(best_x1) + float(best_x2)) / 2
            cy = (float(best_y1) + float(best_y2)) / 2
            half_w = max(all_w) / 2
            half_h = max(all_h) / 2
            z_sorted = sorted(pzb.keys())
            volumetric_list.append({
                'cx': cx,
                'cy': cy,
                'cz': float(center_z),
                'x1_3d': cx - half_w,
                'y1_3d': cy - half_h,
                'x2_3d': cx + half_w,
                'y2_3d': cy + half_h,
                'z_min': z_sorted[0],
                'z_max': z_sorted[-1],
                'per_z_boxes': pzb,
                'class': final_class_name,
                'score': float(best_score),
                'mean':  float(best_mean),
            })

    summary = np.array(final_rows, dtype=object) if final_rows else np.empty((0, 8), dtype=object)
    return summary, volumetric_list
