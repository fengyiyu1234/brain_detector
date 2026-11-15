import json
from contextlib import contextmanager
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
                 max_cell_z_span=5, max_z_gap=0, solver='sparse',
                 channel_id=None, rejected_tracks=None):
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
        if len(track['all_boxes']) < min_z_layers:
            if rejected_tracks is not None:
                from src.core.provenance import stable_id
                ids = [
                    str(det[8]) if len(det) > 8 and det[8] else
                    stable_id('legacy_z_member', channel_id, i, *det[:8])
                    for i, det in enumerate(track['all_boxes'])
                ]
                rejected_tracks.append({
                    'track_id': stable_id('rejected_track3d', channel_id, sorted(ids)),
                    'channel': channel_id or '',
                    'decision': 'rejected',
                    'reason': 'min_z_layers',
                    'member_count': len(ids),
                    'threshold': min_z_layers,
                    'member_detection_ids': json.dumps(ids),
                })
            continue
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
            from src.core.provenance import stable_id
            members = []
            for member_index, det in enumerate(boxes):
                det_id = (str(det[8]) if len(det) > 8 and det[8] else
                          stable_id('legacy_z_member', channel_id, member_index,
                                    *det[:8]))
                members.append({
                    'detection_id': det_id,
                    'tile_name': str(det[9]) if len(det) > 9 else 'Unknown',
                    'slice_name': str(det[10]) if len(det) > 10 else 'Unknown',
                    'z': int(det[7]),
                    'x1': float(det[0]), 'y1': float(det[1]),
                    'x2': float(det[2]), 'y2': float(det[3]),
                    'score': float(det[4]), 'mean': float(det[5]),
                    'class': str(det[6]),
                })
            best_id = next(member['detection_id'] for member in members
                           if member['z'] == int(best_det[7]) and
                           member['x1'] == float(best_det[0]) and
                           member['y1'] == float(best_det[1]) and
                           member['score'] == float(best_det[4]))
            track_id = stable_id('track3d', channel_id,
                                 sorted(member['detection_id'] for member in members))
            volumetric_list.append({
                'track_id': track_id,
                'best_detection_id': best_id,
                'best_box_z': int(best_det[7]),
                'member_count': len(members),
                'member_detections': members,
                'bounds_method': 'representative_centered_max_size',
                'observed_x1': min(v[0] for v in pzb.values()),
                'observed_y1': min(v[1] for v in pzb.values()),
                'observed_x2': max(v[2] for v in pzb.values()),
                'observed_y2': max(v[3] for v in pzb.values()),
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


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3 linking options.
# run_z_linker and the helpers above are also hashed by Stage 2.9
# (tile_position_pipeline.SOLVE_CODE) and called by Stage 2.5 alignment, so options that
# only Stage 3 uses live below and leave them unchanged.
# ──────────────────────────────────────────────────────────────────────────────

# Stage 3 defaults per channel type; config z_linker.<type>.<key> overrides, null disables
# an option. Stage 2.5/2.9 read pre_align_params.z_linker (or these four base keys of the
# top-level z_linker) through point_cloud_aligner.resolve_align_settings instead.
STAGE3_LINK_DEFAULTS = {
    'soma': {
        'iou_thresh': 0.35, 'min_z_layers': 1,
        'max_cell_z_span': 7,           # 5 cut off cells; 8 joins the cell below
        'max_z_gap': 1,                 # bridge one layer without a box
        'link_iomin_thresh': 0.8,       # also link a small box inside a large one
        'link_across_classes': True,    # neuron and glia boxes of one cell link
        'cross_tile_dedup_iou': 0.25,   # same-layer duplicates from different tiles
        'cross_tile_dedup_iomin': 0.45,
    },
    'tf': {'iou_thresh': 0.25, 'min_z_layers': 1, 'max_cell_z_span': 3, 'max_z_gap': 0},
}

LINK_DEDUP_COLUMNS = [
    'detection_id', 'tile_name', 'z', 'class', 'score', 'x1', 'y1', 'x2', 'y2',
    'kept_detection_id', 'kept_tile', 'kept_score', 'iou', 'iomin', 'kept_relabelled_glia',
]


def stage3_link_params(z_linker_config):
    """Stage 3 link_tracks keyword arguments per channel type: config over defaults."""
    z_linker_config = z_linker_config or {}
    return {kind: {key: z_linker_config.get(kind, {}).get(key, value)
                   for key, value in defaults.items()}
            for kind, defaults in STAGE3_LINK_DEFAULTS.items()}


def _base_and_rest(cls):
    base, _, rest = str(cls).partition('_')
    return base, rest


def _as_glia(cls):
    _, rest = _base_and_rest(cls)
    return f'glia_{rest}' if rest else 'glia'


def dedup_cross_tile(mat, iou_thresh, iomin_thresh, dropped=None):
    """Drop same-layer duplicates from different tiles before linking.

    On one layer, two boxes from different tiles (column 9) with IoU > iou_thresh or
    IoMin > iomin_thresh are one cell; pairs go by IoU, highest first, and the lower
    score box is dropped (ties keep the earlier row). Glia priority: dropping a glia box
    relabels a kept non-glia box glia. Stitching (stitcher.combine_predictions) removed
    the clear duplicates after the measured seam offset; this catches the rest in the
    stitched global frame. dropped: list collecting one LINK_DEDUP_COLUMNS dict per
    dropped box. Returns the kept rows (class column updated).
    """
    if len(mat) < 2 or mat.shape[1] < 10:
        return mat
    mat = mat.copy()
    x1, y1, x2, y2, score = (mat[:, k].astype(float) for k in range(5))
    z = mat[:, 7].astype(int)
    tile = mat[:, 9]
    drop = np.zeros(len(mat), bool)
    order = np.argsort(z, kind='stable')
    starts = np.flatnonzero(np.r_[True, z[order][1:] != z[order][:-1]])
    for rows in np.split(order, starts[1:]):
        if len(rows) < 2:
            continue
        centers = np.c_[(x1[rows] + x2[rows]) / 2, (y1[rows] + y2[rows]) / 2]
        # overlapping boxes have centers closer than the largest box side on each axis
        side = np.maximum(x2[rows] - x1[rows], y2[rows] - y1[rows]).max()
        pairs = cKDTree(centers).query_pairs(r=side * 1.5, output_type='ndarray')
        if not len(pairs):
            continue
        a, b = rows[pairs[:, 0]], rows[pairs[:, 1]]
        other = tile[a] != tile[b]
        a, b = a[other], b[other]
        inter = (np.clip(np.minimum(x2[a], x2[b]) - np.maximum(x1[a], x1[b]), 0, None) *
                 np.clip(np.minimum(y2[a], y2[b]) - np.maximum(y1[a], y1[b]), 0, None))
        area_a = (x2[a] - x1[a]) * (y2[a] - y1[a])
        area_b = (x2[b] - x1[b]) * (y2[b] - y1[b])
        iou = inter / (area_a + area_b - inter + 1e-8)
        iomin = inter / (np.minimum(area_a, area_b) + 1e-8)
        hit = np.flatnonzero((iou > iou_thresh) | (iomin > iomin_thresh))
        for k in hit[np.argsort(-iou[hit], kind='stable')]:
            i, j = int(a[k]), int(b[k])
            if drop[i] or drop[j]:
                continue
            kept, gone = (i, j) if score[i] >= score[j] else (j, i)
            relabel = (_base_and_rest(mat[gone, 6])[0] == 'glia'
                       and _base_and_rest(mat[kept, 6])[0] != 'glia')
            if dropped is not None:
                dropped.append({
                    'detection_id': mat[gone, 8], 'tile_name': tile[gone], 'z': int(z[gone]),
                    'class': mat[gone, 6], 'score': float(score[gone]),
                    'x1': float(x1[gone]), 'y1': float(y1[gone]),
                    'x2': float(x2[gone]), 'y2': float(y2[gone]),
                    'kept_detection_id': mat[kept, 8], 'kept_tile': tile[kept],
                    'kept_score': float(score[kept]), 'iou': float(iou[k]),
                    'iomin': float(iomin[k]), 'kept_relabelled_glia': relabel})
            if relabel:
                mat[kept, 6] = _as_glia(mat[kept, 6])
            drop[gone] = True
    return mat[~drop]


def _match_iou_or_iomin(iomin_thresh):
    """run_z_linker match function that also links a box contained in the other.

    A (detection, track) pair is linkable when IoU >= iou_thresh (the base rule) or
    IoMin >= iomin_thresh, same class base. Per connected block of linkable pairs, a
    one-to-one assignment maximises the number of links, then the summed IoU.
    """
    def match(det_boxes, det_bases, track_boxes, track_bases, iou_thresh):
        pairs = _candidate_pairs(det_boxes, track_boxes)
        if len(pairs) == 0:
            return []
        di, ti = pairs[:, 0], pairs[:, 1]
        a, b = det_boxes[di], track_boxes[ti]
        inter = (np.maximum(0.0, np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0]))
                 * np.maximum(0.0, np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1])))
        area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
        area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        iou = inter / (area_a + area_b - inter + 1e-8)
        iomin = inter / (np.minimum(area_a, area_b) + 1e-8)
        ok = ((((1.0 - iou) <= 1.0 - iou_thresh) | (iomin >= iomin_thresh))
              & (det_bases[di] == track_bases[ti]))
        di, ti, weight = di[ok], ti[ok], 1.0 + iou[ok]
        if len(di) == 0:
            return []
        n_d = len(det_boxes)
        graph = coo_matrix((np.ones(len(di)), (di, n_d + ti)),
                           shape=(n_d + len(track_boxes),) * 2)
        block = connected_components(graph, directed=False)[1][di]
        out = []
        for c in np.unique(block):
            sel = block == c
            ud, rd = np.unique(di[sel], return_inverse=True)
            ut, rt = np.unique(ti[sel], return_inverse=True)
            sub = np.zeros((len(ud), len(ut)))
            sub[rd, rt] = weight[sel]
            r, k = linear_sum_assignment(sub, maximize=True)
            keep = sub[r, k] > 0
            out.extend(zip(ud[r[keep]], ut[k[keep]]))
        return out
    return match


def _class_blind(match):
    """Match function that ignores the neuron/glia base of detections and tracks."""
    def run(det_boxes, det_bases, track_boxes, track_bases, iou_thresh):
        return match(det_boxes, np.zeros(len(det_bases), int), track_boxes,
                     np.zeros(len(track_bases), int), iou_thresh)
    return run


@contextmanager
def _matchers(sparse, dense):
    """Run run_z_linker with other match functions (it looks them up at call time).
    Not thread-safe; Stage 3 links each channel in its own process or in sequence."""
    global _match_sparse, _match_dense
    saved = _match_sparse, _match_dense
    _match_sparse, _match_dense = sparse, dense
    try:
        yield
    finally:
        _match_sparse, _match_dense = saved


def link_tracks(full_stack_matrix, iou_thresh, min_z_layers, max_cell_z_span, max_z_gap,
                link_iomin_thresh=None, link_across_classes=False,
                cross_tile_dedup_iou=None, cross_tile_dedup_iomin=None,
                channel_id=None, rejected_tracks=None, dedup_rows=None):
    """Stage 3 z linking: run_z_linker plus the soma options (STAGE3_LINK_DEFAULTS).

    cross_tile_dedup_iou / _iomin: dedup_cross_tile before linking (both None = off);
        dropped boxes are appended to dedup_rows.
    link_iomin_thresh: a box also continues a track when one of the two boxes contains
        the other (IoMin >= this), not only when IoU >= iou_thresh.
    link_across_classes: neuron and glia boxes link into one track; a track with any
        glia member is glia (as the Phase A neuron/glia dedup keeps glia).
    With every option off this is run_z_linker.
    """
    if isinstance(full_stack_matrix, list):
        full_stack_matrix = np.array(full_stack_matrix, dtype=object)
    mat = full_stack_matrix
    if mat.size and (cross_tile_dedup_iou is not None or cross_tile_dedup_iomin is not None):
        mat = dedup_cross_tile(
            mat, np.inf if cross_tile_dedup_iou is None else cross_tile_dedup_iou,
            np.inf if cross_tile_dedup_iomin is None else cross_tile_dedup_iomin, dedup_rows)
    sparse, dense = _match_sparse, _match_dense
    if link_iomin_thresh is not None:
        sparse = dense = _match_iou_or_iomin(link_iomin_thresh)
    if link_across_classes:
        sparse, dense = _class_blind(sparse), _class_blind(dense)
    with _matchers(sparse, dense):
        summary, volumes = run_z_linker(
            mat, iou_thresh=iou_thresh, min_z_layers=min_z_layers,
            max_cell_z_span=max_cell_z_span, max_z_gap=max_z_gap,
            channel_id=channel_id, rejected_tracks=rejected_tracks)
    if link_across_classes:
        # summary rows and volumes are built in the same order, one per kept track
        for row, cell in zip(summary, volumes):
            if (_base_and_rest(cell['class'])[0] != 'glia' and any(
                    _base_and_rest(m['class'])[0] == 'glia' for m in cell['member_detections'])):
                cell['class'] = row[6] = _as_glia(cell['class'])
    return summary, volumes
