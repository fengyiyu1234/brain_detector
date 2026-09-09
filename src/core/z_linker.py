import numpy as np
from scipy.optimize import linear_sum_assignment

from src.utils.markers import class_markers, split_class


# 每次只物化 _IOU_BLOCK 行的中间矩阵。TF 通道（如 Sox9）单层可能有
# 两万个框对两万条轨迹，一次性开 x1/y1/x2/y2/inter/union 等十几个
# (n_d, n_t) 中间数组会占几十 GB，在内存吃紧的机器上直接 MemoryError。
_IOU_BLOCK = 2048


def _iou_matrix(det_boxes, track_boxes):
    """Vectorized IoU: (n_d, 4) × (n_t, 4) → (n_d, n_t).

    按行分块计算，峰值内存 ≈ 结果矩阵 + _IOU_BLOCK×n_t 的中间量，
    而不是十几个完整 (n_d, n_t)。逐元素结果与不分块版本完全一致。
    """
    n_d = det_boxes.shape[0]
    n_t = track_boxes.shape[0]
    out = np.empty((n_d, n_t), dtype=np.float64)
    area_t = (track_boxes[:, 2] - track_boxes[:, 0]) * (track_boxes[:, 3] - track_boxes[:, 1])
    for s in range(0, n_d, _IOU_BLOCK):
        e = min(s + _IOU_BLOCK, n_d)
        blk = det_boxes[s:e]
        x1 = np.maximum(blk[:, 0:1], track_boxes[:, 0])
        y1 = np.maximum(blk[:, 1:2], track_boxes[:, 1])
        x2 = np.minimum(blk[:, 2:3], track_boxes[:, 2])
        y2 = np.minimum(blk[:, 3:4], track_boxes[:, 3])
        inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        area_d = (blk[:, 2] - blk[:, 0]) * (blk[:, 3] - blk[:, 1])
        union = area_d[:, None] + area_t[None, :] - inter
        np.divide(inter, union + 1e-8, out=out[s:e])
    return out


def parse_class_string(cls_str):
    """
    解析动态生成的类别字符串，例如 "neuron_RFP_Sox9"
    返回: base_type ("neuron"), markers_set ({"RFP", "Sox9"})
    伪 marker（旧结果里 "GFP_3" 通道名拆出的 "3"）由 split_class 过滤。
    """
    base_type, markers = split_class(cls_str)
    return base_type, set(markers)


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

            # 原地转成 cost，省掉一份和 iou_mat 同样大的副本
            cost_mat = iou_mat
            np.subtract(1.0, cost_mat, out=cost_mat)
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
            best_det = max(boxes, key=lambda x: (len(class_markers(x[6])), float(x[4])))
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
