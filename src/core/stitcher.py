import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

def colocalize_3d(soma_3d_boxes, nuc_3d_boxes, xy_res=1.0, z_res=0.008, distance_thresh_um=15.0):
    """
    3D 空间点对体共定位：判断 TF/核 是否在 Soma 附近
    xy_res: XY 像素物理尺寸 (um/pixel), 例如 0.5
    z_res: Z step 物理跨度 (um), 8nm 即 0.008
    """
    if len(soma_3d_boxes) == 0:
        return np.empty((0, 8), dtype=object)
    if len(nuc_3d_boxes) == 0:
        return soma_3d_boxes

    # 计算 Soma 的 3D 物理质心 (x, y, z)
    soma_centers = np.column_stack((
        (soma_3d_boxes[:, 0] + soma_3d_boxes[:, 2]) / 2.0 * xy_res,
        (soma_3d_boxes[:, 1] + soma_3d_boxes[:, 3]) / 2.0 * xy_res,
        soma_3d_boxes[:, 7].astype(float) * z_res
    ))
    
    # 计算 Nucleus 的 3D 物理质心 (x, y, z)
    nuc_centers = np.column_stack((
        (nuc_3d_boxes[:, 0] + nuc_3d_boxes[:, 2]) / 2.0 * xy_res,
        (nuc_3d_boxes[:, 1] + nuc_3d_boxes[:, 3]) / 2.0 * xy_res,
        nuc_3d_boxes[:, 7].astype(float) * z_res
    ))

    tree = cKDTree(nuc_centers)
    final_merged = []
    
    for i, soma in enumerate(soma_3d_boxes):
        soma_center = soma_centers[i]
        
        # 查找物理距离距离阈值内的所有细胞核
        indices = tree.query_ball_point(soma_center, r=distance_thresh_um)
        
        current_class = str(soma[6])
        matched_markers = []
        
        for idx in indices:
            nuc = nuc_3d_boxes[idx]
            nuc_class = str(nuc[6])
            
            parts = nuc_class.split('_')
            if len(parts) > 1:
                matched_markers.extend(parts[1:])
            # else: TF detection without channel suffix — skip, never use base type as marker
        
        # 组合最终标签
        if matched_markers:
            soma_parts = current_class.split('_')
            base_type = soma_parts[0]
            existing_markers = soma_parts[1:] if len(soma_parts) > 1 else []
            
            all_markers = sorted(list(set(existing_markers + matched_markers)))
            new_class = f"{base_type}_" + "_".join(all_markers)
        else:
            new_class = current_class
            
        merged_cell = soma.copy()
        merged_cell[6] = new_class
        final_merged.append(merged_cell)
        
    return np.array(final_merged, dtype=object)

def colocalize_3d_centroid_in_box(soma_3d_boxes, nuc_3d_boxes,
                                   xy_res=1.0, z_res=8.0,
                                   xy_tolerance_px=5, z_tolerance_slices=5):
    """
    Colocalize by checking whether a TF nucleus centroid falls inside the soma YOLO bbox,
    expanded by xy_tolerance_px (XY) and z_tolerance_slices (Z).

    Anatomically correct alternative to a distance sphere: the nucleus must be physically
    inside its cell body. Works for both Cellpose 2D+Z-linker and Cellpose 3D centroids.
    """
    if len(soma_3d_boxes) == 0:
        return np.empty((0, 8), dtype=object)
    if len(nuc_3d_boxes) == 0:
        return soma_3d_boxes

    nuc_cx = (nuc_3d_boxes[:, 0].astype(float) + nuc_3d_boxes[:, 2].astype(float)) / 2
    nuc_cy = (nuc_3d_boxes[:, 1].astype(float) + nuc_3d_boxes[:, 3].astype(float)) / 2
    nuc_cz = nuc_3d_boxes[:, 7].astype(float)

    # Scale Z to pixel-equivalent so KDTree distance is meaningful
    z_scale = z_res / xy_res  # e.g. 8/0.65 ≈ 12.3
    nuc_pts = np.column_stack([nuc_cx, nuc_cy, nuc_cz * z_scale])
    nuc_tree = cKDTree(nuc_pts)

    z_tol_scaled = z_tolerance_slices * z_scale
    final_merged = []

    for soma in soma_3d_boxes:
        x1 = float(soma[0]) - xy_tolerance_px
        y1 = float(soma[1]) - xy_tolerance_px
        x2 = float(soma[2]) + xy_tolerance_px
        y2 = float(soma[3]) + xy_tolerance_px
        z_s = float(soma[7])

        soma_cx_c = (x1 + x2) / 2
        soma_cy_c = (y1 + y2) / 2

        # Pre-filter: bounding sphere that contains the expanded box
        half_w = (x2 - x1) / 2
        half_h = (y2 - y1) / 2
        rough_r = np.sqrt(half_w**2 + half_h**2 + z_tol_scaled**2) + 1.0
        candidates = nuc_tree.query_ball_point([soma_cx_c, soma_cy_c, z_s * z_scale], r=rough_r)

        matched_markers = []
        for idx in candidates:
            cx, cy, cz_raw = nuc_cx[idx], nuc_cy[idx], nuc_cz[idx]
            if (x1 <= cx <= x2) and (y1 <= cy <= y2) and (abs(cz_raw - z_s) <= z_tolerance_slices):
                parts = str(nuc_3d_boxes[idx][6]).split('_')
                if len(parts) > 1:
                    matched_markers.extend(parts[1:])

        current_class = str(soma[6])
        if matched_markers:
            soma_parts = current_class.split('_')
            base_type = soma_parts[0]
            existing = soma_parts[1:] if len(soma_parts) > 1 else []
            new_class = f"{base_type}_" + "_".join(sorted(set(existing + matched_markers)))
        else:
            new_class = current_class

        merged = soma.copy()
        merged[6] = new_class
        final_merged.append(merged)

    return np.array(final_merged, dtype=object)


def permutation_test_colocalization(soma_3d_boxes, tf_3d_boxes,
                                    xy_res=1.0, z_res=8.0,
                                    distance_thresh_um=15.0,
                                    n_permutations=1000,
                                    max_soma_sample=50_000):
    """
    对 3D 共定位结果进行置换检验，评估每个 TF marker 的共定位是否显著高于随机水平。

    加速策略（适用于大脑全卷数据）：
      1. 当 soma 总数 > max_soma_sample 时，随机抽取子集，降低 query_ball_tree 的行数。
      2. 将 TF 核裁剪到抽样 soma 的空间边界框 + distance_thresh_um 缓冲区，
         进一步减少 TF KDTree 的节点数。
      两步叠加可将计算量降低 ~100 倍（500k soma → 50k；TF 随之缩减）。

    置换方法：对 TF 核的 XY 坐标做环形随机平移（toroidal shift），保持 Z 分布不变。
    返回：DataFrame，每行对应一个 TF marker，包含 z-score 和 p-value。
    """
    if len(soma_3d_boxes) == 0 or len(tf_3d_boxes) == 0:
        return pd.DataFrame()

    # --- 提取每个 TF 检测框的 marker 名 ---
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
        sample_idx = rng.choice(n_soma_total, size=max_soma_sample, replace=False)
        soma_sample = soma_3d_boxes[sample_idx]
        sampled = True
    else:
        soma_sample = soma_3d_boxes
        sampled = False
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
    keep = np.where(tf_mask)[0]
    tf_cx       = tf_cx_all[keep]
    tf_cy       = tf_cy_all[keep]
    tf_z        = tf_z_all[keep]
    tf_marker   = [tf_marker_all[i] for i in keep]

    x_range = tf_cx.max() - tf_cx.min() if len(tf_cx) > 1 else 0.0
    y_range = tf_cy.max() - tf_cy.min() if len(tf_cy) > 1 else 0.0
    if x_range < 1e-6 or y_range < 1e-6:
        return pd.DataFrame()

    # soma_tree 只需构建一次
    soma_tree = cKDTree(soma_phys)

    def count_coloc(cx_arr, cy_arr):
        """统计有 ≥1 个指定 marker TF 核的 soma 数量（每种 marker 每个 soma 最多计 1 次）。"""
        tf_tree = cKDTree(np.column_stack([cx_arr, cy_arr, tf_z]))
        pairs   = soma_tree.query_ball_tree(tf_tree, r=distance_thresh_um)
        counts  = {m: 0 for m in unique_markers}
        for nbr_indices in pairs:
            found = {tf_marker[ti] for ti in nbr_indices if tf_marker[ti]}
            for m in found:
                counts[m] += 1
        return counts

    actual_counts = count_coloc(tf_cx, tf_cy)

    # --- 置换循环（环形平移）---
    random_counts = {m: np.zeros(n_permutations) for m in unique_markers}
    for _ in range(n_permutations):
        dx = rng.uniform(0.1 * x_range, 0.9 * x_range)
        dy = rng.uniform(0.1 * y_range, 0.9 * y_range)
        new_cx = (tf_cx - tf_cx.min() + dx) % x_range + tf_cx.min()
        new_cy = (tf_cy - tf_cy.min() + dy) % y_range + tf_cy.min()
        perm = count_coloc(new_cx, new_cy)
        for m in unique_markers:
            random_counts[m][_] = perm[m]

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


def stitchDetection(detections, H=None, W=None, xsize=None, ysize=None, step=None):
    """
    单 Tile 内部的检测框去重。
    直接对全量检测结果使用矩阵化 NMS，放弃容易产生边界 Bug 的网格切分。
    """
    if len(detections) == 0:
        return detections
    
    # 将格式规范化，假设输入为 [x1, y1, x2, y2, score, class/other...]
    boxes = np.array(list(detections))
    
    return non_max_suppression_iou(boxes, overlapThresh=0.4, sort_idx=4)

def non_max_suppression_iou(boxes, overlapThresh=0.45, sort_idx=4):
    """
    标准交并比 (IoU) NMS，专为高密度细胞优化。
    """
    if len(boxes) == 0:
        return []
        
    pick = []
    x1 = boxes[:, 0].astype(float)
    y1 = boxes[:, 1].astype(float)
    x2 = boxes[:, 2].astype(float)
    y2 = boxes[:, 3].astype(float)
    
    w_init = np.maximum(0, x2 - x1)
    h_init = np.maximum(0, y2 - y1)
    area = w_init * h_init
    
    idxs = np.argsort(boxes[:, sort_idx].astype(float))
    
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        # 计算得分最高的框与其余所有框的交集坐标
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        # 交集宽和高
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        inter_area = w * h
        
        # 核心修正：使用标准 IoU = 交集 / (A + B - 交集)
        iou = inter_area / (area[i] + area[idxs[:last]] - inter_area + 1e-6)
        
        # 剔除 IoU 大于阈值的框
        idxs = np.delete(idxs, np.concatenate(([last], np.where(iou > overlapThresh)[0])))
        
    return boxes[pick]

def combine_predictions(all_predictions, csv_reader, classes, z_start, Z, pos, disp_mat, size, metadata_registry, tile_name, tILESIZE = 2048, file_z0 = None):
    row, col = pos
    ABS_X, ABS_Y, ABS_Z = disp_mat[pos]
    H, W = size
    mask = np.zeros((H,W), dtype=float)
    if col > 0: 
        x_pre_start = disp_mat[row,col-1][0]; y_pre_start = disp_mat[row,col-1][1]       
        mask[max(ABS_Y,y_pre_start):min(ABS_Y+tILESIZE,y_pre_start+tILESIZE),
             max(ABS_X,x_pre_start):min(ABS_X+tILESIZE,x_pre_start+tILESIZE)] = 1
    if row > 0:
        x_pre_start = disp_mat[row-1,col][0]; y_pre_start = disp_mat[row-1,col][1]       
        mask[max(ABS_Y,y_pre_start):min(ABS_Y+tILESIZE,y_pre_start+tILESIZE),
             max(ABS_X,x_pre_start):min(ABS_X+tILESIZE,x_pre_start+tILESIZE)] = 1       
    z0 = z_start - ABS_Z
    z1 = z0 + Z
    
    for row_data in csv_reader:
        slice_name, x1, y1, x2, y2, class_name, score, mean, z = row_data[:9]
        z = int(float(z))
        x1 = float(x1); x2 = float(x2); y1 = float(y1); y2 = float(y2)
        score = float(score); mean = float(mean)
        if z-1 in range(z0,z1):
            x1 += ABS_X; x2 += ABS_X; y1 += ABS_Y; y2 += ABS_Y; z = z - z0
            if not mask[int((y1+y2)//2),int((x1+x2)//2)] > 0:
                # [核心修改 1] 用字符串包含判定基础类型，取代 classes.index()
                cell_type_index = 0 if 'glia' in class_name.lower() else 1
                
                # [核心修改 2] 保持 class_name 为字符串塞进数组，必须指定 dtype=object
                new_box = np.array([[x1, y1, x2, y2, score, mean, class_name, z]], dtype=object)
                
                all_predictions[z-1][cell_type_index] = np.concatenate(
                    (all_predictions[z-1][cell_type_index], new_box)
                )
                
                # 将该细胞的全局质心与名字注册到内存中
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                metadata_registry.append([cx, cy, z, tile_name, slice_name])
                
    return all_predictions
