import numpy as np

def run_z_linker(full_stack_matrix, iou_thresh=0.25, max_gap=1, min_z_layers=2):
    """
    针对全量矩阵进行 Z 轴串联
    输入格式: [x1, y1, x2, y2, score, mean, class, z] (8列)
    新增 min_z_layers: 真实细胞至少需要连续跨越的 Z 层数 (用于强力过滤 2D 噪点)
    """
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
        matched_indices = set()

        for track in active_tracks:
            # 如果当前层距离该 track 上次出现的层数超过 max_gap，则停止该 track
            if z - track['last_z'] > max_gap:
                track['active'] = False
                continue
            
            best_overlap = iou_thresh
            best_match_idx = -1
            
            # 在当前层寻找同类别且重叠度最大的框
            for idx, det in enumerate(curr_detections):
                if idx in matched_indices: continue
                if det[6] != track['class_idx']: continue # 类别校验
                
                boxA = track['last_box']
                boxB = det[:4]
                
                xA = max(boxA[0], boxB[0])
                yA = max(boxA[1], boxB[1])
                xB = min(boxA[2], boxB[2])
                yB = min(boxA[3], boxB[3])
                
                interArea = max(0, xB - xA) * max(0, yB - yA)
                boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
                boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
                
                if interArea > 0:
                    # 核心优化：计算 IoU 以及双向 IoA
                    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
                    ioa_A = interArea / float(boxAArea + 1e-6) # 涵盖前一层小框的情况
                    ioa_B = interArea / float(boxBArea + 1e-6) # 涵盖当前层小框的情况
                    
                    # 只要交并比、或者任意一方被绝大部分包围(IoA)，就认为是高度重合
                    max_overlap_score = max(iou, ioa_A, ioa_B)
                else:
                    max_overlap_score = 0

                if max_overlap_score > best_overlap:
                    best_overlap = max_overlap_score
                    best_match_idx = idx
            
            if best_match_idx != -1:
                track['all_boxes'].append(curr_detections[best_match_idx])
                track['last_box'] = curr_detections[best_match_idx][:4]
                track['last_z'] = z
                matched_indices.add(best_match_idx)

        # 归档已失活的 track
        finished_tracks.extend([t for t in active_tracks if not t['active']])
        active_tracks = [t for t in active_tracks if t['active']]

        # 当前层未匹配的框，作为新的 track 起点
        for idx, det in enumerate(curr_detections):
            if idx not in matched_indices:
                active_tracks.append({
                    'all_boxes': [det],
                    'last_box': det[:4],
                    'last_z': z,
                    'class_idx': det[6],
                    'active': True
                })

    finished_tracks.extend(active_tracks)

    # 3. 结果聚合：从每个 Track 中选出 Score 最高的代表
    final_rows = []
    for track in finished_tracks:
        # 核心优化：利用 3D 信息强力降噪。至少出现在 min_z_layers 层中的才算真细胞
        if len(track['all_boxes']) >= min_z_layers: 
            best_det = max(track['all_boxes'], key=lambda x: x[4]) # 选出置信度最高的一层作为代表
            final_rows.append(best_det)
            
    return np.array(final_rows) if len(final_rows) > 0 else np.empty((0, 8))