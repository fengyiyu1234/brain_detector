import sys
sys.path.insert(0, '../')
import cv2
import csv
import os
import numpy as np
from ultralytics import YOLO
import logging
from tqdm import tqdm
from contextlib import nullcontext
from .stitcher import stitchDetection
from src.utils.io import load_cached_detections,listFile
from src.utils.image import normalize_for_detection
from src.analysis.qc import calculate_comprehensive_qc, calculate_channel_logic_qc

def draw_dashed_rectangle(img, pt1, pt2, color, thickness=2, dash_length=8):
    """
    在 OpenCV 图像上绘制虚线矩形框的辅助函数
    """
    x1, y1 = pt1
    x2, y2 = pt2
    
    # 定义四条边：(起点, 终点)
    lines = [
        ((x1, y1), (x2, y1)), # 顶边
        ((x2, y1), (x2, y2)), # 右边
        ((x2, y2), (x1, y2)), # 底边
        ((x1, y2), (x1, y1))  # 左边
    ]
    
    for (start_x, start_y), (end_x, end_y) in lines:
        length = np.hypot(end_x - start_x, end_y - start_y)
        dashes = max(1, int(length / dash_length))
        
        for i in range(dashes):
            if i % 2 == 0: # 只在偶数段画线，奇数段留空
                p1 = (int(start_x + (end_x - start_x) * i / dashes), 
                      int(start_y + (end_y - start_y) * i / dashes))
                p2 = (int(start_x + (end_x - start_x) * (i + 1) / dashes), 
                      int(start_y + (end_y - start_y) * (i + 1) / dashes))
                cv2.line(img, p1, p2, color, thickness)

def process_single_tile(i, pATHTEST, config):
    current_logger = logging.getLogger(__name__)
    dir_name = os.path.basename(pATHTEST)
    #current_logger.info(f"开始处理 Tile: {dir_name}")

    # --- 1. 初始化变量 ---
    dp = config['detection_params']
    run_qc_flag = dp.get('rUN_QC', True)
    derived = config['derived_paths']
    paths = config['paths']
    labels_to_names = config['labels_to_names']
    num_class = len(labels_to_names)
    device = config['device']
    names_to_labels = {v: int(k) for k, v in labels_to_names.items()}

    sample_step = config.get('vISUALIZATIONSAMPLESTEP', 100)
    sample_count = config.get('vISUALIZATIONSAMPLECOUNT', 5)
    visualize_tile = dp.get('vISUALIZE_TILE', False)

    CSV_PATH = os.path.join(derived['pATH_DET_RES'], dir_name + '_result.csv')
    pATH_VIS_TILE_CURRENT = os.path.join(derived['pATH_VIS_TILE'], dir_name)
    QC_CSV_PATH = os.path.join(derived['pATH_DET_RES'], dir_name + '_qc_metrics.csv')

    # === Checkpoint 1: 完美完成 ===
    # 如果 QC 文件存在，说明跑完了，直接跳过
    if os.path.exists(QC_CSV_PATH):
        print(f"[{dir_name}] QC Report exists. Skipping entire tile.")
        # 这里需要返回一些东西以保持主程序兼容，通常返回空列表或读取现有结果
        return [], dir_name

    # === Checkpoint 2: 检测已完成，补跑 QC (QC Mode) ===
    should_run_yolo = True
    cached_detections_map = {}
    
    if os.path.exists(CSV_PATH):
        # 尝试读取现有的 CSV 结果
        cached_detections_map = load_cached_detections(CSV_PATH)
        # 如果读取到了数据，说明检测已完成，进入 QC Only 模式
        if len(cached_detections_map) > 0:
            print(f"[{dir_name}] Found result.csv. Running QC ONLY mode...")
            should_run_yolo = False

    if os.path.isfile(CSV_PATH):
        return [], dir_name


    # 准备循环
    # --- 2. 环境准备 ---
    all_tile_detections = [] 

    if should_run_yolo:
        try:
            model = YOLO(config['model_path']) # 确保 config 里有这个 key
        except Exception as e:
            current_logger.error(f"模型加载失败: {e}")
            return [], dir_name
    testnames, testpaths = listFile(pATHTEST, '.tiff')

    if config['detection_params'].get('dOWNSAMPLE_Z_2X', False):
        current_logger.info(f"Downsampling enabled: reducing Z-stack by 2x for {dir_name}")
        testnames = testnames[::2]  # 从索引0开始，每隔2个取一个
        testpaths = testpaths[::2]

    if not testpaths: return [], dir_name

    # 建立 Channel 2 索引
    c1_root = os.path.abspath(paths['channel1_dir'])
    c2_root = os.path.abspath(paths['channel2_dir'])
    rel_tile_path = os.path.relpath(pATHTEST, c1_root)
    pATHTEST_C2 = os.path.join(c2_root, rel_tile_path)

    c2_files_index = {}
    if os.path.exists(pATHTEST_C2):
        c2_files_index = {os.path.splitext(f)[0]: os.path.join(pATHTEST_C2, f) 
                          for f in os.listdir(pATHTEST_C2) 
                          if f.lower().endswith(('.tif', '.tiff')) and not f.startswith('.')}

    mode_str = "Full" if should_run_yolo else "QC-Only"
    pbar = tqdm(total=len(testpaths), desc=f"[{mode_str}] {dir_name[:10]}", position=i+1, leave=False)

    #for qc
    qc_data_list = []
    last_small_img_c1 = None
    last_small_img_c2 = None    

    # --- 3. 核心循环 (Z-axis) ---
    ctx = open(CSV_PATH, 'w', newline='') if should_run_yolo else nullcontext()
    
    with ctx as csvfile:
        filewriter = csv.writer(csvfile) if should_run_yolo else None

        for z_idx, testpath in enumerate(testpaths):
            current_z_real = z_idx + 1 # 1-based index

            # 3.1 鲁棒性读取 Channel 1 (Red)
            # 无论什么模式，QC 必须读图
            img_c1 = cv2.imread(testpath, cv2.IMREAD_ANYDEPTH)
            if img_c1 is None:
                pbar.update(1)
                continue
            if len(img_c1.shape) == 3: img_c1 = img_c1[:, :, 0]
            H0, W0 = img_c1.shape[:2]

            # 3.2 鲁棒性读取 Channel 2 (Green)
            name_no_ext = os.path.splitext(testnames[z_idx])[0]
            c2_full_path = c2_files_index.get(name_no_ext)
            img_c2 = None
            if c2_full_path and os.path.exists(c2_full_path):
                img_c2 = cv2.imread(c2_full_path, cv2.IMREAD_ANYDEPTH)
                if img_c2 is not None and len(img_c2.shape) == 3:
                    img_c2 = img_c2[:, :, 0]

            # C2 缺失保护
            if img_c2 is None: 
                img_c2 = np.zeros((H0, W0), dtype=np.uint16)
            # 尺寸对齐保护
            elif img_c2.shape[:2] != (H0, W0):
                img_c2_resized = np.zeros((H0, W0), dtype=img_c2.dtype)
                h_l, w_l = min(H0, img_c2.shape[0]), min(W0, img_c2.shape[1])
                img_c2_resized[:h_l, :w_l] = img_c2[:h_l, :w_l]
                img_c2 = img_c2_resized

            # =========================================================
            # 步骤 2: 获取检测结果 (Run YOLO or Read Cache)
            # =========================================================
            final_layer_boxes = []
            if should_run_yolo:

                # 3.3 分通道归一化 (核心修改)
                p_low, p_high = config['dOWNSAMPLE_PERCENTILE_LOW'], config['dOWNSAMPLE_PERCENTILE_HIGH']
                red_8bit = normalize_for_detection(img_c1, p_low, p_high)
                
                # 对 C2 进行尺寸对齐（防止红绿通道图尺寸微差）
                if img_c2.shape[:2] != (H0, W0):
                    img_c2_resized = np.zeros((H0, W0), dtype=np.uint16)
                    h_l, w_l = min(H0, img_c2.shape[0]), min(W0, img_c2.shape[1])
                    img_c2_resized[:h_l, :w_l] = img_c2[:h_l, :w_l]
                    img_c2 = img_c2_resized
                    
                green_8bit = normalize_for_detection(img_c2, p_low, p_high)

                # 合成 8-bit BGR (B:0, G:green, R:red)
                fulldraw = np.zeros((H0, W0, 3), dtype=np.uint8)
                fulldraw[:, :, 1] = green_8bit
                fulldraw[:, :, 2] = red_8bit

                # 用于计算均值的原始 16-bit 堆叠图 (用于后续 mean_val 计算)
                fullimg_raw_16bit = np.zeros((H0, W0, 3), dtype=np.uint16)
                fullimg_raw_16bit[:, :, 1] = img_c2
                fullimg_raw_16bit[:, :, 2] = img_c1

                # 3.4 采样可视化判定
                is_sampled_frame = visualize_tile and (z_idx % sample_step < sample_count)
                fulldraw_vis = fulldraw.copy() if is_sampled_frame else None

                # 3.5 滑动窗口检测 (使用 8-bit RG 融合图)
                xsize, ysize, step_win = dp['xsize'], dp['ysize'], dp['step']
                H_pad = H0 if (H0-ysize)%step_win == 0 else H0-H0%step_win+ysize
                W_pad = W0 if (W0-xsize)%step_win == 0 else W0-W0%step_win+xsize
                conf_thresh = dp.get('conf_thresh', dp.get('tHRESHOLD', 0.25))
                nms_iou = dp.get('nms_iou', dp.get('mINIOU', 0.45))

                fullimg_pad = np.zeros((H_pad, W_pad, 3), dtype=np.uint8)
                fullimg_pad[0:H0, 0:W0] = fulldraw
                
                raw_detections = np.empty((0, 6))
                for x in range(0, W_pad, step_win):
                    for y in range(0, H_pad, step_win):
                        patch = fullimg_pad[y:y+ysize, x:x+xsize]
                        if patch.max() < 10: continue # 过滤全黑块
                        
                        results = model.predict(patch, device=device, verbose=False,conf=conf_thresh, iou=nms_iou )
                        res = results[0]
                        if len(res.boxes) > 0:
                            boxes = res.boxes.xyxy.cpu().numpy() + np.array([x, y, x, y])
                            scores = res.boxes.conf.cpu().numpy()
                            labels = res.boxes.cls.cpu().numpy()
                            patch_res = np.hstack((boxes, scores[:, np.newaxis], labels[:, np.newaxis]))
                            raw_detections = np.append(raw_detections, patch_res, axis=0)

                # 3.6 XY 平面去重与保存
                for label_idx in range(num_class):
                    layer_label_data = raw_detections[raw_detections[:, -1] == label_idx, :-1]
                    if layer_label_data.size > 0:
                        cleaned_layer_boxes = stitchDetection(layer_label_data, H0, W0, xsize, ysize, step_win)
                        class_name = labels_to_names.get(int(label_idx), f"C{label_idx}")
                        
                        for box in cleaned_layer_boxes:
                            x1, y1, x2, y2, score = box
                            final_layer_boxes.append([x1, y1, x2, y2, score, label_idx])
                            ix1, iy1, ix2, iy2 = map(int, [x1, y1, x2, y2])
                            
                            # 在原始 16-bit 融合数据上计算均值（包含红绿分量）
                            cell_crop = fullimg_raw_16bit[max(0,iy1):min(H0,iy2), max(0,ix1):min(W0,ix2)]
                            mean_val = cell_crop.mean() if cell_crop.size > 0 else 0
                            
                            filewriter.writerow([testnames[z_idx], x1, y1, x2, y2, class_name, score, mean_val, z_idx+1])
                            all_tile_detections.append([x1, y1, x2, y2, score, mean_val, int(label_idx), z_idx+1])

                            if is_sampled_frame and fulldraw_vis is not None:
                                # 1. 获取颜色映射
                                c_name = config['colors_map'].get(int(label_idx), "white")
                                bgr_c = config['bgr_colors'].get(c_name, (255, 255, 255))
                                
                                # 2. 获取类别名称 (例如 "red glia", "red neuron")
                                class_name_str = labels_to_names.get(int(label_idx), "").lower()
                                
                                # 3. 判断细胞类型：如果是 neuron 就画虚线，否则画实线
                                if "neuron" in class_name_str:
                                    draw_dashed_rectangle(fulldraw_vis, (ix1, iy1), (ix2, iy2), bgr_c, thickness=1, dash_length=3)
                                else:
                                    cv2.rectangle(fulldraw_vis, (ix1, iy1), (ix2, iy2), bgr_c, thickness=1)
                # 3.7 保存可视化图
                if is_sampled_frame and fulldraw_vis is not None:
                    os.makedirs(pATH_VIS_TILE_CURRENT, exist_ok=True)
                    cv2.imwrite(os.path.join(pATH_VIS_TILE_CURRENT, f"{name_no_ext}_Z{z_idx+1:03d}.jpg"), fulldraw_vis)                  
            else:
                # --- B. QC Only Mode (读取缓存) ---
                cached_list = cached_detections_map.get(current_z_real, [])
                # 缓存格式: [x1, y1, x2, y2, score, class_name]
                # 需要转换 class_name -> cls_id (为了 calculate_channel_logic_qc)
                for item in cached_list:
                    c_name = item[5]
                    c_id = names_to_labels.get(c_name, -1) # 如果找不到 ID，给 -1
                    if c_id != -1:
                        # 替换最后一项为 ID
                        new_item = item[:5] + [c_id]
                        final_layer_boxes.append(new_item)
            
            # =========================================================
            # 步骤 3: 双通道 QC 计算 (Dual Channel QC)
            # =========================================================
            if run_qc_flag:
            # 3.1 C1 通道质量 (Red)
                qc_c1, small_c1 = calculate_comprehensive_qc(
                    img_c1, final_layer_boxes, current_z_real, prev_img_small=last_small_img_c1
                )
                
                # 3.2 C2 通道质量 (Green)
                qc_c2, small_c2 = calculate_comprehensive_qc(
                    img_c2, final_layer_boxes, current_z_real, prev_img_small=last_small_img_c2
                )

                # 3.3 [新增] 生物学逻辑检查 (检测结果 vs 信号强度)
                qc_logic = calculate_channel_logic_qc(
                    img_c1, img_c2, final_layer_boxes, current_z_real
                )
                
                # --- 数据合并 ---
                combined_qc = {"z": current_z_real, "detection_count": len(final_layer_boxes)}
                
                # 合并 C1 结果 (加前缀区分)
                if qc_c1:
                    for k, v in qc_c1.items():
                        if k not in ["z", "detection_count"]:
                            combined_qc[f"c1_{k}"] = v

                # 合并 C2 结果 (加前缀区分)
                if qc_c2:
                    for k, v in qc_c2.items():
                        if k not in ["z", "detection_count"]:
                            combined_qc[f"c2_{k}"] = v
                
                # 合并 Logic 结果 (不需要前缀，因为字段名已经是独特的，如 logic_mismatch_rate)
                if qc_logic:
                    combined_qc.update(qc_logic)

                qc_data_list.append(combined_qc)
                
                last_small_img_c1 = small_c1
                last_small_img_c2 = small_c2

                pbar.update(1)
    
    # --- 4. 保存 QC ---
    if run_qc_flag and qc_data_list:
        try:
            # 收集所有可能的 key 以确保表头完整
            all_keys = set().union(*(d.keys() for d in qc_data_list))
            # 排序：z, detection_count 在前，其他字母序
            sorted_keys = ['z', 'detection_count'] + sorted([k for k in all_keys if k not in ['z', 'detection_count']])
            
            with open(QC_CSV_PATH, 'w', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, fieldnames=sorted_keys)
                dict_writer.writeheader()
                dict_writer.writerows(qc_data_list)
        except Exception as e:
            current_logger.error(f"QC Save Failed: {e}")

    pbar.close()
    if model: del model
    return all_tile_detections, dir_name

def process_single_tile_wrapper(args):
    # args 就是那个元组 (i, path, config, pos)
    # 使用 *args 自动解包传给原函数
    return process_single_tile(*args)