import numpy as np

def run_z_linker(full_stack_matrix, iou_thresh=0.3, max_gap=1):
    """
    针对全量矩阵进行 Z 轴串联
    输入格式: [x1, y1, x2, y2, score, mean, class, z] (8列)
    """
    if full_stack_matrix.size == 0:
        return full_stack_matrix

    # 1. 按 Z 轴排序并分组
    # 注意：你的 Z 在最后一列 (index 7)
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
            
            best_iou = iou_thresh
            best_match_idx = -1
            
            # 在当前层寻找同类别且 IoU 最大的框
            for idx, det in enumerate(curr_detections):
                if idx in matched_indices: continue
                if det[6] != track['class_idx']: continue # 类别校验 (index 6)
                
                # 计算 2D IoU
                boxA = track['last_box']
                boxB = det[:4]
                
                xA = max(boxA[0], boxB[0])
                yA = max(boxA[1], boxB[1])
                xB = min(boxA[2], boxB[2])
                yB = min(boxA[3], boxB[3])
                interArea = max(0, xB - xA) * max(0, yB - yA)
                boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
                boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
                iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)

                if iou > best_iou:
                    best_iou = iou
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
    # 你也可以选择计算中值层，这里采用 Score 最高原则
    final_rows = []
    for track in finished_tracks:
        # 可选：过滤掉只出现过 1 层的噪声（如果是生物样本通常跨层）
        if len(track['all_boxes']) >= 1: 
            best_det = max(track['all_boxes'], key=lambda x: x[4]) # index 4 是 score
            final_rows.append(best_det)
            
    return np.array(final_rows) if len(final_rows) > 0 else np.empty((0, 8))
