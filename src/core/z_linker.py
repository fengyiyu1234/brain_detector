from functools import lru_cache

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from src.utils.markers import split_class


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


def _match_dense(det_boxes, det_bases, track_boxes, track_bases, iou_thresh):
    """原始做法：整层一个 (n_d, n_t) 代价矩阵做匈牙利匹配。返回被接受的 (d_idx, t_pos)。"""
    cost_mat = _iou_matrix(det_boxes, track_boxes)
    # 原地转成 cost，省掉一份和 iou_mat 同样大的副本
    np.subtract(1.0, cost_mat, out=cost_mat)
    cost_mat[det_bases[:, None] != track_bases[None, :]] = 1e6  # cross-type → infeasible
    row_ind, col_ind = linear_sum_assignment(cost_mat)
    keep = cost_mat[row_ind, col_ind] <= 1.0 - iou_thresh
    return list(zip(row_ind[keep], col_ind[keep]))


def _candidate_pairs(det_boxes, track_boxes):
    """
    所有可能 IoU > 0 的 (det, track) 对（可能多给，不会漏）。
    两框相交要求中心在每个轴上相距 < (w_a + w_b)/2 ≤ 2·max(half_a, half_b)，
    half = max(w, h)/2。所以分别以每个 det、每个 track 自己的 2·half 为半径查一次再取并集，
    个别超大框只放大它自己的查询半径，不会拖慢其它框。
    """
    def centers_half(b):
        c = np.column_stack(((b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2))
        return c, np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]) / 2
    dc, dh = centers_half(det_boxes)
    tc, th = centers_half(track_boxes)
    pairs = []
    for q_c, q_h, tree, flip in ((dc, dh, cKDTree(tc), False), (tc, th, cKDTree(dc), True)):
        hits = tree.query_ball_point(q_c, r=2 * q_h + 1e-6, p=np.inf)
        counts = np.fromiter((len(h) for h in hits), dtype=np.int64, count=len(hits))
        if counts.sum() == 0:
            continue
        q = np.repeat(np.arange(len(hits)), counts)
        o = np.fromiter((j for h in hits for j in h), dtype=np.int64, count=int(counts.sum()))
        pairs.append(np.column_stack((o, q)) if flip else np.column_stack((q, o)))
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    return np.unique(np.concatenate(pairs), axis=0)


def _match_sparse(det_boxes, det_bases, track_boxes, track_bases, iou_thresh):
    """
    与 _match_dense 等价（除了代价完全相同时可能选中另一组最优解），但只在 IoU>0 的连通块内求解。

    为什么等价：整层问题要求配满 min(n_d, n_t) 对，代价 = 1−IoU（IoU=0 的同类对代价恰为 1，
    跨类对 1e6）。任取一组正 IoU 的同类匹配 M，把它补满时所需的跨类对数只取决于各类别的数量，
    与 M 无关；其余位置用代价为 1 的 IoU=0 同类对补齐。所以整体最优 = 正 IoU 同类边上的
    最大权匹配，而它在连通块之间互相独立。块内仍按原来的代价矩阵求解，再用同样的阈值取舍。
    """
    pairs = _candidate_pairs(det_boxes, track_boxes)
    if len(pairs) == 0:
        return []
    di, ti = pairs[:, 0], pairs[:, 1]
    a, b = det_boxes[di], track_boxes[ti]
    # 与 _iou_matrix 完全相同的算式，保证阈值判定逐位一致
    inter = (np.maximum(0.0, np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0]))
             * np.maximum(0.0, np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1])))
    union = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
             + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter)
    iou = inter / (union + 1e-8)
    ok = (iou > 0) & (det_bases[di] == track_bases[ti])
    di, ti, cost = di[ok], ti[ok], 1.0 - iou[ok]
    if len(di) == 0:
        return []

    n_d = len(det_boxes)
    graph = coo_matrix((np.ones(len(di)), (di, n_d + ti)), shape=(n_d + len(track_boxes),) * 2)
    _, label = connected_components(graph, directed=False)
    comp = label[di]
    order = np.argsort(comp, kind='stable')
    di, ti, cost, comp = di[order], ti[order], cost[order], comp[order]
    starts = np.flatnonzero(np.r_[True, comp[1:] != comp[:-1]])
    ends = np.r_[starts[1:], len(comp)]

    accept_max = 1.0 - iou_thresh
    out = []
    for s, e in zip(starts, ends):
        if e - s == 1:   # 一条边的连通块（绝大多数）：直接判定
            if cost[s] <= accept_max:
                out.append((di[s], ti[s]))
            continue
        ud, rd = np.unique(di[s:e], return_inverse=True)
        ut, rt = np.unique(ti[s:e], return_inverse=True)
        sub = np.ones((len(ud), len(ut)))
        sub[det_bases[ud][:, None] != track_bases[ut][None, :]] = 1e6
        sub[rd, rt] = cost[s:e]
        r, c = linear_sum_assignment(sub)
        keep = sub[r, c] <= accept_max
        out.extend(zip(ud[r[keep]], ut[c[keep]]))
    return out


@lru_cache(maxsize=None)
def _parse_cached(cls_str):
    """class 字符串种类很少（每个通道几种），解析结果缓存起来。返回不可变的 tuple，调用方自行拷贝。"""
    base_type, markers = split_class(cls_str)
    return base_type, tuple(markers)


def _median(vals):
    """与 np.median 在整数 z 列表上的结果相同，但没有逐次调用 numpy 的开销。"""
    s = sorted(vals)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2


def parse_class_string(cls_str):
    """
    解析动态生成的类别字符串，例如 "neuron_RFP_Sox9"
    返回: base_type ("neuron"), markers_set ({"RFP", "Sox9"})
    伪 marker（旧结果里 "GFP_3" 通道名拆出的 "3"）由 split_class 过滤。
    """
    base_type, markers = _parse_cached(cls_str)
    return base_type, set(markers)


def run_z_linker(full_stack_matrix, iou_thresh=0.45, min_z_layers=2,
                 max_cell_z_span=5, max_z_gap=0, solver='sparse'):
    """
    Returns (summary_array, volumetric_list).
      summary_array:    np.ndarray (N, 8) — one row per cell, center_z, for visualization CSV
      volumetric_list:  list[dict]        — per-cell volumetric info for 3D colocalization

    solver: 'sparse'（默认）只在 IoU>0 的连通块内做匈牙利匹配，全脑稠密通道从小时级降到分钟级；
            'dense' 为原始的整层匹配，仅用于对照验证。两者只在代价完全相同的并列最优解上可能不同。
            iou_thresh ≤ 0 时 IoU=0 的对也会被接受，稀疏分解不再成立，自动退回 dense。
    """
    if solver not in ('sparse', 'dense'):
        raise ValueError(f"solver 只能是 'sparse' 或 'dense'，收到 {solver!r}")
    match = _match_sparse if (solver == 'sparse' and iou_thresh > 0) else _match_dense
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
            det_parsed   = [_parse_cached(d[6]) for d in curr_detections]
            det_bases    = np.array([p[0] for p in det_parsed])
            track_bases  = np.array([active_tracks[ti]['base_type'] for ti in active_t_indices])

            det_boxes    = np.array([d[:4] for d in curr_detections], dtype=float)
            track_boxes  = np.array([active_tracks[ti]['last_box'] for ti in active_t_indices], dtype=float)

            pairs = match(det_boxes, det_bases, track_boxes, track_bases, iou_thresh)
            for d_idx, t_pos in sorted((int(d), int(t)) for d, t in pairs):
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
                curr_base_type, curr_markers = parse_class_string(det[6])   # 新 set，之后会被 update
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
            best_det = max(boxes, key=lambda x: (len(_parse_cached(x[6])[1]), float(x[4])))
            best_x1, best_y1, best_x2, best_y2, best_score, best_mean = best_det[:6]
            z_list   = [b[7] for b in boxes]
            center_z = int(_median(z_list))
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
