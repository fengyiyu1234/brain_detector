import numpy as np
from scipy.optimize import linear_sum_assignment

def calculate_iou(boxA, boxB):
    """计算两个 2D 框的交并比 (IoU)"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def parse_class_string(cls_str):
    """
    解析动态生成的类别字符串，例如 "neuron_RFP_Sox9"
    返回: base_type ("neuron"), markers_set ({"RFP", "Sox9"})
    """
    parts = str(cls_str).split('_')
    base_type = parts[0]
    markers = set(parts[1:]) if len(parts) > 1 else set()
    return base_type, markers

def run_z_linker(full_stack_matrix, iou_thresh=0.45, max_gap=1, min_z_layers=2, max_cell_z_span=5):
    """
    针对全量矩阵进行 Z 轴串联 (2.5D 细胞追踪算法) - 动态多通道版
    输入格式: [x1, y1, x2, y2, score, mean, class_str, z] (8列)
    """
    # 如果是空数组直接返回
    if isinstance(full_stack_matrix, list):
        full_stack_matrix = np.array(full_stack_matrix, dtype=object)
    if full_stack_matrix.size == 0:
        return full_stack_matrix

    # 1. 按 Z 轴排序并分组
    z_min, z_max = int(np.min(full_stack_matrix[:, 7])), int(np.max(full_stack_matrix[:, 7]))
    z_groups = {z: [] for z in range(z_min, z_max + 1)}
    for det in full_stack_matrix:
        z_groups[int(det[7])].append(det)

    active_tracks = []   # 正在追踪的细胞串
    finished_tracks = [] # 已结束的细胞串

    # 2. 逐层连接
    for z in range(z_min, z_max + 1):
        curr_detections = z_groups[z]
        
        # 判定 track 失活
        for track in active_tracks:
            if z - track['last_z'] > max_gap:
                track['active'] = False
            # 防止极高密度下的“糖葫芦”过度合并
            if z - track['first_z'] >= max_cell_z_span:
                track['active'] = False

        matched_det_indices = set()

        # 仅对仍活跃的 track 建立候选列表
        active_t_indices = [ti for ti, t in enumerate(active_tracks) if t['active']]

        if curr_detections and active_t_indices:
            n_d, n_t = len(curr_detections), len(active_t_indices)
            # 代价矩阵：代价 = 1 - IoU；跨类型或无重叠保持 1e6（不可行）
            cost_mat = np.full((n_d, n_t), 1e6)
            for d_idx, det in enumerate(curr_detections):
                det_base, _ = parse_class_string(det[6])
                for t_pos, ti in enumerate(active_t_indices):
                    track = active_tracks[ti]
                    if det_base != track['base_type']:
                        continue
                    cost_mat[d_idx, t_pos] = 1.0 - calculate_iou(det[:4], track['last_box'])

            row_ind, col_ind = linear_sum_assignment(cost_mat)
            for d_idx, t_pos in zip(row_ind, col_ind):
                if cost_mat[d_idx, t_pos] > 1.0 - iou_thresh:
                    continue  # 不可行或 IoU 不足阈值
                ti = active_t_indices[t_pos]
                track = active_tracks[ti]
                det = curr_detections[d_idx]
                _, det_markers = parse_class_string(det[6])
                track['all_boxes'].append(det)
                track['last_box'] = det[:4]
                track['last_z'] = z
                track['all_markers'].update(det_markers)
                matched_det_indices.add(d_idx)

        # ==========================================
        # 归档已失活的 track
        # ==========================================
        finished_tracks.extend([t for t in active_tracks if not t['active']])
        active_tracks = [t for t in active_tracks if t['active']]

        # ==========================================
        # 当前层未匹配的框，作为新的 track 起点
        # ==========================================
        for idx, det in enumerate(curr_detections):
            if idx not in matched_det_indices:
                curr_base_type, curr_markers = parse_class_string(det[6])
                active_tracks.append({
                    'all_boxes': [det],
                    'last_box': det[:4],
                    'last_z': z,
                    'first_z': z,
                    'base_type': curr_base_type,
                    'all_markers': curr_markers, # 初始 Markers
                    'active': True
                })

    # 循环结束，将剩下的 active_tracks 全部归档
    finished_tracks.extend(active_tracks)

    # 3. 结果聚合：从每个 3D Track 中计算代表属性
    final_rows = []
    for track in finished_tracks:
        # 利用 3D 信息强力降噪。至少出现在 min_z_layers 层中的才算真细胞
        if len(track['all_boxes']) >= min_z_layers:
            # 取出所有的检测框
            boxes = track['all_boxes']
            
            # 方法：选取置信度 (score) 最高的那一层作为中心坐标和代表分数
            # box[4] 是 score 所在列
            best_det = max(boxes, key=lambda x: x[4])
            best_x1, best_y1, best_x2, best_y2, best_score, best_mean = best_det[:6]
            
            # Z 坐标可以使用最佳层，或者用跨度的几何中心
            # 这里取所有层 Z 坐标的中位数
            z_list = [b[7] for b in boxes]
            center_z = int(np.median(z_list))
            
            # === 核心修改 3: 重建最终 3D 细胞的名字 ===
            # 将搜集到的所有特征排序后拼回去，例如 "neuron" + "_" + "GFP_RFP_Sox9"
            marker_str = "_".join(sorted(list(track['all_markers'])))
            if marker_str:
                final_class_name = f"{track['base_type']}_{marker_str}"
            else:
                final_class_name = track['base_type']
            
            # 保持原始输出格式 [x1, y1, x2, y2, score, mean, class_str, z]
            final_rows.append([
                best_x1, best_y1, best_x2, best_y2, 
                best_score, best_mean, 
                final_class_name, center_z
            ])

    return np.array(final_rows, dtype=object) if final_rows else np.empty((0, 8), dtype=object)