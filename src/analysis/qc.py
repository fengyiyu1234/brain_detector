
import cv2
import os
import numpy as np
from scipy.ndimage import maximum_filter
import glob
import pandas as pd

def calculate_comprehensive_qc(img, detections, z_index, prev_img_small=None):
    """
    综合 QC 计算：
    1. 图像质量 (模糊/过曝/条纹/漂移)
    2. 检测可靠性 - 假阴性 (漏检率)
    3. 检测可靠性 - 置信度分布 (新增)
    """
    if img is None: return None, None

    # --- 1. 图像预处理 ---
    h, w = img.shape[:2]
    target_size = 512
    scale = target_size / max(h, w)
    small_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    
    if len(small_img.shape) == 3:
        small_img_gray = small_img[:,:,2] # 假设 Red 为主通道
    else:
        small_img_gray = small_img

    # 计算平均亮度
    mean_intensity = float(np.mean(small_img_gray))

    metrics = {
        "z": z_index,
        "mean_intensity": mean_intensity,
        "detection_count": len(detections)
    }

    # --- A. 图像质量指标 ---
    metrics['sharpness'] = float(cv2.Laplacian(small_img_gray, cv2.CV_64F).var())
    
    thresh_sat = 60000 if small_img_gray.dtype == np.uint16 else 250
    metrics['saturation_ratio'] = float(np.count_nonzero(small_img_gray > thresh_sat) / small_img_gray.size)

    # -----------------------------------------------------
    # [修改点] 计算 Drift (过滤黑图)
    # -----------------------------------------------------
    drift = 0.0
    
    # 设定黑图阈值: 
    # 如果是 16-bit 图像，阈值设为 100 (通常背景底噪会有几十)
    # 如果是 8-bit 图像，阈值设为 5
    black_thresh = 100.0 if small_img_gray.dtype == np.uint16 else 5.0

    # 仅当：1.有上一帧数据 且 2.当前帧平均亮度大于黑图阈值 时，才计算漂移
    if prev_img_small is not None and metrics['mean_intensity'] > black_thresh:
        try:
            # 确保数据类型一致转换为 float32 进行 FFT 计算
            curr_32 = small_img_gray.astype(np.float32)
            prev_32 = prev_img_small.astype(np.float32)
            
            # 额外的安全性检查：如果上一张图也是黑图（方差极小），也跳过
            if np.var(prev_32) > 1e-5: 
                shift = cv2.phaseCorrelate(prev_32, curr_32)[0]
                drift = float(np.sqrt(shift[0]**2 + shift[1]**2))
        except Exception as e:
            # 可以在这里打印日志，或者静默失败
            pass
            
    metrics['drift'] = drift
    # -----------------------------------------------------

    # --- B. 检测可靠性：假阴性 (Potential False Negatives) ---
    thresh_bright = np.percentile(small_img_gray, 98)
    min_bright_limit = 100 if small_img_gray.dtype == np.uint16 else 10
    if thresh_bright < min_bright_limit: thresh_bright = min_bright_limit
    
    local_max = maximum_filter(small_img_gray, size=5)
    peaks = (small_img_gray == local_max) & (small_img_gray > thresh_bright)
    peak_y, peak_x = np.where(peaks)
    bright_spots = len(peak_x)
    metrics['cv_bright_spots'] = bright_spots
    
    hit_count = 0
    # 转换为 numpy 方便操作
    dets_arr = np.array(detections) if len(detections) > 0 else np.empty((0, 6))

    if bright_spots > 0 and len(dets_arr) > 0:
        dets_small = dets_arr[:, :4] * scale
        for py, px in zip(peak_y, peak_x):
            in_box = (px >= dets_small[:, 0]) & (px <= dets_small[:, 2]) & \
                     (py >= dets_small[:, 1]) & (py <= dets_small[:, 3])
            if np.any(in_box): hit_count += 1
            
    metrics['miss_ratio'] = round(1.0 - (hit_count / bright_spots), 3) if bright_spots > 0 else 0.0

    # --- C. [新增] 检测可靠性：置信度统计 (Confidence Stats) ---
    avg_conf = 0.0
    median_conf = 0.0
    low_conf_ratio = 0.0 
    
    if len(dets_arr) > 0:
        # 假设 detections 格式: [x1, y1, x2, y2, score, cls]
        scores = dets_arr[:, 4]
        
        avg_conf = float(np.mean(scores))
        median_conf = float(np.median(scores))
        
        weak_threshold = 0.6 
        weak_count = np.count_nonzero(scores < weak_threshold)
        low_conf_ratio = float(weak_count / len(scores))
    
    metrics['avg_conf'] = round(avg_conf, 3)
    metrics['median_conf'] = round(median_conf, 3)
    metrics['low_conf_ratio'] = round(low_conf_ratio, 3)

    return metrics, small_img_gray

def calculate_channel_logic_qc(img_c1, img_c2, detections, z_index):
    """
    检查检测结果与通道信号的生物学一致性 (Red/Green/Yellow 逻辑)
    """
    if img_c1 is None or img_c2 is None or len(detections) == 0:
        return {}

    h, w = img_c1.shape[:2]
    scale = 512.0 / max(h, w)
    small_c1 = cv2.resize(img_c1, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    small_c2 = cv2.resize(img_c2, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    
    bg_c1 = np.mean(small_c1) + 1e-6
    bg_c2 = np.mean(small_c2) + 1e-6

    logic_errors = 0
    total_checks = 0
    yellow_ratios = []

    # 抽样检查
    dets_arr = np.array(detections)
    if len(dets_arr) > 50:
        indices = np.random.choice(len(dets_arr), 50, replace=False)
        sample_dets = dets_arr[indices]
    else:
        sample_dets = dets_arr

    for det in sample_dets:
        x1, y1, x2, y2 = (det[:4].astype(float) * scale).astype(int)
        cls_id = int(det[5]) # 必须是 ID
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(small_c1.shape[1], x2), min(small_c1.shape[0], y2)
        if x2 <= x1 or y2 <= y1: continue

        roi_c1 = small_c1[y1:y2, x1:x2]
        roi_c2 = small_c2[y1:y2, x1:x2]
        
        snr_c1 = np.mean(roi_c1) / bg_c1
        snr_c2 = np.mean(roi_c2) / bg_c2
        
        has_c1 = snr_c1 > 1.2
        has_c2 = snr_c2 > 1.2
        is_error = False

        # Red Logic (0, 3)
        if cls_id in [0, 3]:
            if not has_c1: is_error = True
        # Green Logic (1, 4)
        elif cls_id in [1, 4]:
            if not has_c2: is_error = True
        # Yellow Logic (2, 5)
        elif cls_id in [2, 5]:
            if not (has_c1 and has_c2): is_error = True
            if has_c1 and has_c2:
                yellow_ratios.append(snr_c1 / (snr_c2 + 1e-6))

        if is_error: logic_errors += 1
        total_checks += 1

    metrics = {
        "logic_mismatch_rate": round(logic_errors / total_checks, 3) if total_checks > 0 else 0.0,
        "yellow_balance": round(np.mean(yellow_ratios), 2) if yellow_ratios else 0.0
    }
    return metrics


def generate_global_summary(result_root_path, output_name="Global_Project_Summary.csv"):
    print("正在生成全局 QC 汇总报告...")
    
    # 1. 找到所有的 QC CSV 文件
    # 假设文件名格式为 *_qc_metrics.csv
    qc_files = glob.glob(os.path.join(result_root_path, "*_qc_metrics.csv"))
    
    if not qc_files:
        print("未找到任何 QC 文件，跳过汇总。")
        return

    summary_list = []

    for fpath in qc_files:
        try:
            # 读取单个 Tile 的 CSV
            df = pd.read_csv(fpath)
            
            if df.empty:
                continue
                
            # 获取文件名作为 Tile ID
            tile_name = os.path.basename(fpath).replace('_qc_metrics.csv', '')
            
            # --- 计算该 Tile 的统计指标 ---
            total_cells = df['detection_count'].sum()
            avg_conf = df['avg_conf'].mean() if 'avg_conf' in df.columns else 0
            
            # 漂移统计 (如果有 drift 列)
            # 假设 drift 列叫 'c1_drift'
            max_drift = df['c1_drift'].max() if 'c1_drift' in df.columns else 0
            avg_drift = df['c1_drift'].mean() if 'c1_drift' in df.columns else 0
            
            # 逻辑错误统计 (如果有 logic_mismatch_rate)
            avg_logic_error = df['logic_mismatch_rate'].mean() if 'logic_mismatch_rate' in df.columns else 0
            
            # 清晰度统计
            avg_sharpness = df['c1_sharpness'].mean() if 'c1_sharpness' in df.columns else 0

            # 汇总成一行数据
            summary_row = {
                "Tile_Name": tile_name,
                "Total_Cells": total_cells,
                "Avg_Confidence": round(avg_conf, 3),
                "Avg_Sharpness": round(avg_sharpness, 2),
                "Max_Drift_px": round(max_drift, 2),
                "Avg_Logic_Error": round(avg_logic_error, 4),
                "Total_Layers": len(df),
                "QC_Status": "FAIL" if (max_drift > 10 or avg_logic_error > 0.15) else "PASS"
            }
            summary_list.append(summary_row)
            
        except Exception as e:
            print(f"读取文件 {fpath} 失败: {e}")

    # 2. 生成全局 DataFrame
    if summary_list:
        global_df = pd.DataFrame(summary_list)
        
        # 按照 Tile 名字排序
        global_df = global_df.sort_values("Tile_Name")
        
        # 保存
        save_path = os.path.join(result_root_path, output_name)
        global_df.to_csv(save_path, index=False)
        print(f"全局汇总报告已生成: {save_path}")
        print(global_df.to_string()) # 在控制台打印预览
    else:
        print("没有生成任何有效数据。")