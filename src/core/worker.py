import sys
sys.path.insert(0, '../')
import cv2
import os
cv2.setNumThreads(0)
os.environ["OMP_NUM_THREADS"] = "1"
import csv
import numpy as np
from ultralytics import YOLO
import logging
from tqdm import tqdm
from .stitcher import stitchDetection
from src.utils.image import normalize_for_detection
import warnings
warnings.filterwarnings("ignore", message=".*channels deprecated.*")

try:
    from cellpose import models as cp_models
except ImportError:
    cp_models = None

import tempfile
import shutil

def fast_cloud_read(cloud_path):
    """
    极速云端图片读取器 (F 盘 SSD 增强版)：
    将缓存路径指向空间更大的 F 盘 SSD，确保 C 盘系统盘安全。
    """
    # 1. 自定义 F 盘的缓存目录
    temp_dir = r"F:\temp_cache" 
    
    # 确保文件夹存在，如果不存在则自动创建
    if not os.path.exists(temp_dir):
        try:
            os.makedirs(temp_dir, exist_ok=True)
        except Exception as e:
            # 如果 F 盘没有权限或不存在，退回到 C 盘系统临时目录
            temp_dir = tempfile.gettempdir()
    
    # 2. 构造防冲突的本地临时文件名 (加入当前进程的 PID)
    file_name = os.path.basename(cloud_path)
    pid = os.getpid()
    local_temp_path = os.path.join(temp_dir, f"gpu_worker_{pid}_{file_name}")
    
    try:
        # 3. 操作系统底层级大文件流式拉取
        shutil.copy2(cloud_path, local_temp_path)
        
        # 4. 从 F 盘 SSD 极速读取 (使用 IMREAD_ANYDEPTH 保证 0 损耗)
        img = cv2.imread(local_temp_path, cv2.IMREAD_ANYDEPTH) 
        
        return img
    
    except Exception as e:
        logging.getLogger(__name__).error(f"❌ F 盘读取失败 {cloud_path}: {e}")
        return None
        
    finally:
        # 5. 阅后即焚：读取完成后立即释放空间
        if os.path.exists(local_temp_path):
            try:
                os.remove(local_temp_path)
            except:
                pass

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def calculate_ioa(box_nuc, box_soma):
    """计算细胞核在胞体内的占比 (Intersection over Area of Nucleus)"""
    xA = max(box_nuc[0], box_soma[0])
    yA = max(box_nuc[1], box_soma[1])
    xB = min(box_nuc[2], box_soma[2])
    yB = min(box_nuc[3], box_soma[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0
    nucArea = (box_nuc[2] - box_nuc[0]) * (box_nuc[3] - box_nuc[1])
    return interArea / float(nucArea)

def extract_boxes_from_masks(masks):
    """将 Cellpose 得到的 2D 实例 mask 转换为 Bounding Box"""
    boxes = []
    # 0 是背景，细胞从 1 开始
    for val in np.unique(masks):
        if val == 0: continue
        y_idx, x_idx = np.where(masks == val)
        if len(y_idx) == 0: continue
        x1, x2 = np.min(x_idx), np.max(x_idx)
        y1, y2 = np.min(y_idx), np.max(y_idx)
        # 固定赋予 nucleus 类别和 1.0 的分数
        boxes.append([x1, y1, x2, y2, 1.0, "nucleus"])
    return boxes

def colocalization(layer_channel_boxes, iou_thresh=0.5, ioa_thresh=0.6):
    target_cells = []
    
    rfp_boxes = layer_channel_boxes.get("RFP", [])
    gfp_boxes = layer_channel_boxes.get("GFP", [])
    tf_boxes = layer_channel_boxes.get("Sox9", []) # 取决于 config 里的 id
    
    matched_gfp_indices = set()
    
    # 1. 胞体融合 (RFP 与 GFP)
    for r_box in rfp_boxes:
        best_iou = 0
        best_g_idx = -1
        
        for g_idx, g_box in enumerate(gfp_boxes):
            if g_idx in matched_gfp_indices: continue
            iou = calculate_iou(r_box[:4], g_box[:4])
            if iou > best_iou:
                best_iou = iou
                best_g_idx = g_idx
                
        r_type = r_box[5] # 假设 YOLO 传入的已经是字符串 'neuron' 或 'glia'
        
        cell_info = {
            "box": r_box[:4], 
            "cell_type": r_type,
            "markers": ["RFP"],
            "score": r_box[4]
        }
        
        if best_iou >= iou_thresh:
            cell_info["markers"].append("GFP")
            # 坐标取包络框 (包含两者的最大范围)
            g_box = gfp_boxes[best_g_idx]
            cell_info["box"] = [
                min(r_box[0], g_box[0]), min(r_box[1], g_box[1]),
                max(r_box[2], g_box[2]), max(r_box[3], g_box[3])
            ]
            matched_gfp_indices.add(best_g_idx)
            
            # --- 冲突解决：Glia 优先级最高 ---
            g_type = g_box[5]
            if r_type != g_type:
                if r_type == 'glia' or g_type == 'glia':
                    cell_info["cell_type"] = 'glia'
                else:
                    # 如果都不是 glia（理论上不可能，除非类别增多），听高置信度的
                    cell_info["cell_type"] = g_type if g_box[4] > r_box[4] else r_type

        target_cells.append(cell_info)
        
    # 将剩下的独立 GFP 放入池中
    for g_idx, g_box in enumerate(gfp_boxes):
        if g_idx not in matched_gfp_indices:
            target_cells.append({
                "box": g_box[:4],
                "cell_type": g_box[5],
                "markers": ["GFP"],
                "score": g_box[4]
            })

    # 2. 将 TF (Sox9) 核填入胞体池
    for tf_box in tf_boxes:
        for cell in target_cells:
            # 计算核在胞体内的面积占比
            ioa = calculate_ioa(tf_box[:4], cell["box"]) 
            if ioa >= ioa_thresh:
                cell["markers"].append("Sox9") # 具体标记名视需要可参数化
                break
                
    # 3. 输出格式化
    final_merged_boxes = []
    for cell in target_cells:
        marker_str = "_".join(sorted(cell["markers"]))
        final_class_name = f"{cell['cell_type']}_{marker_str}"
        x1, y1, x2, y2 = cell["box"]
        # 输出 [x1, y1, x2, y2, score, class_name] 以供 stitchDetection 使用
        final_merged_boxes.append([x1, y1, x2, y2, cell["score"], final_class_name])
        
    return final_merged_boxes

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

    dp = config['detection_params']
    derived = config['derived_paths']
    paths = config['paths']
    device = config['device']

    CSV_PATH = os.path.join(derived['pATH_DET_RES'], dir_name + '_result.csv')
    
    # --- 1.Checkpoint ---
    if os.path.exists(CSV_PATH):
        # 结果已存在，直接跳过整个 Tile
        print(f"[{dir_name}] Result CSV already exists. Skipping tile.")
        return [], dir_name
    
    # --- 2. 解析路由与打印确认 ---
    routing_config = [ch for ch in config.get('channels_routing', []) if ch.get('active', True)]

    print(f"\n[{dir_name}] === 确认通道推理路由 ===")
    if not routing_config:
        current_logger.error("配置文件中没有激活的 channels_routing！")
        return [], dir_name

    required_models = set()
    for ch in routing_config:
        ch_dir = paths.get(ch['dir_key'], 'Unknown')
        print(f" -> 通道: {ch['id']} ({ch.get('type', 'N/A')}) | 模型: {ch['model'].upper()} | 路径: {ch_dir}")
        required_models.add(ch['model'].lower())
    print("================================\n")

    # --- 3. 动态加载所需模型 ---
    models_dict = {}
    if 'yolo' in required_models:
        try:
            models_dict['yolo'] = YOLO(config['models']['yolo_path'])
            # 注意获取新的模型类别映射
            yolo_classes = config.get('model_classes', {}).get('yolo', {"0":"neuron", "1":"glia"})
            print(f"[{dir_name}] ✔️ YOLO 模型加载成功")
        except Exception as e:
            current_logger.error(f"YOLO 加载失败: {e}")
            return [], dir_name
    
    if 'cellpose' in required_models:
        try:
            from cellpose import models as cp_models
            models_dict['cellpose'] = cp_models.CellposeModel(
                pretrained_model=config['models']['cellpose_path'], 
                gpu=(device == 'cuda')
            )
            print(f"[{dir_name}] ✔️ Cellpose 模型加载成功")
        except Exception as e:
            current_logger.error(f"Cellpose 加载失败: {e}")
            return [], dir_name

    # --- 4. 建立多通道文件索引 (完美镜像结构版) ---
    anchor_ch = routing_config[0]
    anchor_root = os.path.abspath(paths[anchor_ch['dir_key']])
    
    print(f"\n[{dir_name}] ====== 文件检索 Checkpoint ======")
    
    # 1. 获取锚点通道的所有图片 (完美兼容 .tif 和 .tiff 混用)
    testnames = []
    if os.path.exists(pATHTEST):
        for f in os.listdir(pATHTEST):
            if f.lower().endswith(('.tif', '.tiff')) and not f.startswith('.'):
                testnames.append(f)
        testnames.sort()
        
    print(f" -> 锚点通道 [{anchor_ch['id']}] (路径: {pATHTEST})")
    print(f"    共找到切片: {len(testnames)} 张")

    if not testnames:
        current_logger.error(f"❌ 严重错误: 目录中没找到任何 tif/tiff，跳过此 Tile！")
        return [], dir_name

    # 提取无后缀名作为 Key (e.g., '452800_377900_001')
    testnames_no_ext = [os.path.splitext(f)[0] for f in testnames]

    if dp.get('dOWNSAMPLE_Z_2X', False):
        testnames = testnames[::2]
        testnames_no_ext = testnames_no_ext[::2]

    # 获取相对的二级结构路径 (例如: 452800\452800_377900)
    rel_tile_path = os.path.relpath(pATHTEST, anchor_root)
    channel_files_indices = {}

    # 2. 精准映射镜像通道
    for ch in routing_config:
        ch_id = ch['id']
        ch_root = os.path.abspath(paths[ch['dir_key']])
        
        # 因为结构一模一样，直接将根目录和二级相对路径拼起来
        target_dir = os.path.join(ch_root, rel_tile_path)
        
        if os.path.exists(target_dir):
            files_dict = {
                os.path.splitext(f)[0]: os.path.join(target_dir, f) 
                for f in os.listdir(target_dir) 
                if f.lower().endswith(('.tif', '.tiff')) and not f.startswith('.')
            }
            channel_files_indices[ch_id] = files_dict
            print(f" -> 关联通道 [{ch_id}] 找到切片: {len(files_dict)} 张")
        else:
            channel_files_indices[ch_id] = {}
            current_logger.warning(f"⚠️ 警告: 找不到对应的镜像文件夹: {target_dir}")
            
    print("======================================\n")

    # --- 5. 核心循环准备 ---
    all_tile_detections = []
    pbar = tqdm(total=len(testnames_no_ext), desc=f"[{dir_name[:10]}] 准备启动...", position=i+1, leave=False)

    file_handles = {}
    csv_writers = {}

    # 1. 打开综合结果的主 CSV
    f_combined = open(CSV_PATH, 'w', newline='', encoding='utf-8')
    writer_combined = csv.writer(f_combined)

    # 2. 为每个激活的通道打开专属的 CSV
    for ch in routing_config:
        ch_id = ch['id']
        ch_csv_path = os.path.join(derived['pATH_DET_RES'], f"{dir_name}_{ch_id}_result.csv")
        f_ch = open(ch_csv_path, 'w', newline='', encoding='utf-8')
        file_handles[ch_id] = f_ch
        csv_writers[ch_id] = csv.writer(f_ch)
        
    try:
        for z_idx, name_no_ext in enumerate(testnames_no_ext):
            current_z_real = z_idx + 1 # 1-based index
            # 格式: { "RFP": [[x1, y1, x2, y2, score, "neuron"], ...], "Sox9": [...] }
            layer_channel_boxes = {ch['id']: [] for ch in routing_config}
        
            H0, W0 = 0, 0 

            # =========================================================
            # 步骤 2.1: 遍历路由表，进行独立的单通道检测
            # =========================================================
            for ch in routing_config:
                ch_id = ch['id']
                pbar.set_description(f"Tile:[{dir_name[:10]}] | Z层:[{current_z_real}/{len(testnames_no_ext)}] | 检测通道:[{ch_id}]")
                ch_model = ch['model']
                img_path = channel_files_indices[ch_id].get(name_no_ext)
                
                # 如果当前 Z 层该通道缺图，直接跳过
                if not img_path or not os.path.exists(img_path):
                    continue
                    
                img_raw = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
                if img_raw is None: continue
                if len(img_raw.shape) == 3: img_raw = img_raw[:, :, 0]
                
                if H0 == 0: H0, W0 = img_raw.shape[:2] # 记录当前层的实际尺寸

                # 单通道图像归一化 (拉伸到 0-255)
                norm_img = normalize_for_detection(img_raw, dp['dOWNSAMPLE_PERCENTILE_LOW'], dp['dOWNSAMPLE_PERCENTILE_HIGH'])

                # --------- YOLO 单通道滑动窗口检测 ---------
                if ch_model == 'yolo':
                    img_infer = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR) # YOLO 仍需要 3 通道输入
                    
                    xsize, ysize, step_win = dp['xsize'], dp['ysize'], dp['step']
                    H_pad = H0 if (H0-ysize)%step_win == 0 else H0-H0%step_win+ysize
                    W_pad = W0 if (W0-xsize)%step_win == 0 else W0-W0%step_win+xsize
                    
                    fullimg_pad = np.zeros((H_pad, W_pad, 3), dtype=np.uint8)
                    fullimg_pad[0:H0, 0:W0] = img_infer
                    
                    conf_thresh = dp.get('conf_thresh', dp.get('tHRESHOLD', 0.25))
                    nms_iou = dp.get('nms_iou', dp.get('mINIOU', 0.45))
                    
                    raw_detections = np.empty((0, 6))
                    for x in range(0, W_pad, step_win):
                        for y in range(0, H_pad, step_win):
                            patch = fullimg_pad[y:y+ysize, x:x+xsize]
                            if patch.max() < 10: continue # 过滤纯黑无信号背景
                            
                            results = models_dict['yolo'].predict(patch, device=device, verbose=False, conf=conf_thresh, iou=nms_iou)
                            res = results[0]
                            if len(res.boxes) > 0:
                                boxes = res.boxes.xyxy.cpu().numpy() + np.array([x, y, x, y])
                                scores = res.boxes.conf.cpu().numpy()
                                labels = res.boxes.cls.cpu().numpy()
                                patch_res = np.hstack((boxes, scores[:, np.newaxis], labels[:, np.newaxis]))
                                raw_detections = np.append(raw_detections, patch_res, axis=0)

                    # 针对单通道内进行拼接和去重 (stitchDetection)
                    unique_labels = np.unique(raw_detections[:, 5]) if raw_detections.size > 0 else []
                    temp_yolo_boxes = []
                    
                    for lbl in unique_labels:
                        # 提取该类别坐标 [x1, y1, x2, y2, score] 进行局部拼接 NMS
                        layer_label_data = raw_detections[raw_detections[:, 5] == lbl, :-1]
                        if layer_label_data.size > 0:
                            cleaned_boxes = stitchDetection(layer_label_data, H0, W0, xsize, ysize, step_win)
                            class_str = yolo_classes.get(str(int(lbl)), "unknown")
                            for box in cleaned_boxes:
                                temp_yolo_boxes.append([box[0], box[1], box[2], box[3], box[4], class_str])
                                
                    # --- 单通道内部的冲突解决 (如模型在同个位置既框了 Neuron 又框了 Glia) ---
                    kept_yolo_boxes = []
                    temp_yolo_boxes.sort(key=lambda x: (1 if x[5]=='glia' else 0, x[4]), reverse=True) # Glia优先，得分高的优先
                    for box_curr in temp_yolo_boxes:
                        is_suppressed = False
                        cx1, cy1, cx2, cy2 = box_curr[:4]
                        area_c = (cx2 - cx1) * (cy2 - cy1)
                        for box_kept in kept_yolo_boxes:
                            kx1, ky1, kx2, ky2 = box_kept[:4]
                            area_k = (kx2 - kx1) * (ky2 - ky1)
                            iw = max(0, min(cx2, kx2) - max(cx1, kx1))
                            ih = max(0, min(cy2, ky2) - max(cy1, ky1))
                            if iw > 0 and ih > 0:
                                inter_area = iw * ih
                                ioa_curr = inter_area / area_c if area_c > 0 else 0
                                if ioa_curr > 0.70 or (inter_area / (area_c + area_k - inter_area) > 0.35):
                                    is_suppressed = True
                                    break
                        if not is_suppressed:
                            kept_yolo_boxes.append(box_curr)
                            
                            # ✨ 写入纯 YOLO 单通道的独立结果
                            # 格式: [slice_name, x1, y1, x2, y2, class, score, mean, z_val]
                            row = [name_no_ext, box_curr[0], box_curr[1], box_curr[2], box_curr[3], box_curr[5], box_curr[4], 0.0, current_z_real]
                            csv_writers[ch_id].writerow(row)
                            
                    layer_channel_boxes[ch_id].extend(kept_yolo_boxes)

                # --------- Cellpose 单通道密集细胞核检测 ---------
                elif ch_model == 'cellpose':
                    # eval 直接接受 2D 矩阵
                    masks, flows, styles = models_dict['cellpose'].eval(norm_img, diameter=None, channels=[0,0])
                    cp_boxes = extract_boxes_from_masks(masks) # 调用前面定义的辅助函数
                    
                    for box in cp_boxes:
                        # ✨ 写入纯 Cellpose 单通道的独立结果
                        row = [name_no_ext, box[0], box[1], box[2], box[3], box[5], box[4], 0.0, current_z_real]
                        csv_writers[ch_id].writerow(row)
                        
                    layer_channel_boxes[ch_id].extend(cp_boxes)

            # =========================================================
            # 步骤 2.2: 跨通道空间重叠合并 (Colocalization)
            # =========================================================
            # 将各个通道提取好的 Box 融合，填入 Sox9 核，并且合并 RFP/GFP
            final_merged_boxes = colocalization(layer_channel_boxes, iou_thresh=0.5, ioa_thresh=0.6)

            # =========================================================
            # 步骤 2.3: 写入综合结果并处理可视化
            # =========================================================
            for item in final_merged_boxes:
                # final_merged_boxes 的格式是 [x1, y1, x2, y2, score, final_class_name]
                x1, y1, x2, y2, score, class_name = item
                mean_val = 0.0 # 因为解耦了通道，伪彩平均强度失去意义，置为 0，不影响空间坐标拼接
                
                # ✨ 保存至融合主 CSV
                writer_combined.writerow([name_no_ext, x1, y1, x2, y2, class_name, score, mean_val, current_z_real])
                
                # 追加至全局列表供 Z-linker 3D 追踪使用
                all_tile_detections.append([x1, y1, x2, y2, score, mean_val, class_name, current_z_real])

            # 如果开启了可视化 (按需采样，将综合结果画在某一层上做记录)
            visualize_tile = dp.get('vISUALIZE_TILE', False)
            sample_step = config.get('vISUALIZATIONSAMPLESTEP', 100)
            sample_count = config.get('vISUALIZATIONSAMPLECOUNT', 5)
            
            if visualize_tile and (z_idx % sample_step < sample_count) and H0 > 0:
                pATH_VIS_TILE_CURRENT = os.path.join(derived['pATH_VIS_TILE'], dir_name)
                os.makedirs(pATH_VIS_TILE_CURRENT, exist_ok=True)
                
                # 创建一个全黑底板用于绘制综合结果
                vis_img = np.zeros((H0, W0, 3), dtype=np.uint8)
                for box in final_merged_boxes:
                    ix1, iy1, ix2, iy2 = map(int, box[:4])
                    c_name = box[5] # e.g. "glia_GFP_Sox9"
                    
                    # 简单按字符串包含上色
                    color = (255, 255, 255) # Default white
                    if "RFP" in c_name and "GFP" in c_name: color = (0, 255, 255) # Yellow for dual
                    elif "RFP" in c_name: color = (0, 0, 255)
                    elif "GFP" in c_name: color = (0, 255, 0)
                    
                    if "glia" in c_name.lower():
                        draw_dashed_rectangle(vis_img, (ix1, iy1), (ix2, iy2), color, thickness=2)
                    else:
                        cv2.rectangle(vis_img, (ix1, iy1), (ix2, iy2), color, 2, cv2.LINE_8)
                        
                cv2.imwrite(os.path.join(pATH_VIS_TILE_CURRENT, f"{name_no_ext}_Z{current_z_real:03d}.jpg"), vis_img)

            # ✅ 所有的通道、融合、可视化都搞定了，这一层结束，进度条前进一格
            pbar.update(1)

    finally:
        f_combined.close()
        for f in file_handles.values():
            f.close()
        pbar.close()
    
    # =========================================================
    # 步骤 6: 生成 Summary 统计文件
    # =========================================================
    from collections import Counter
    
    # 1. 提取所有检测到的类别标签
    # all_tile_detections 的每个 item 格式: [x1, y1, x2, y2, score, mean, class_name, z]
    all_labels = [item[6] for item in all_tile_detections]
    
    # 2. 使用 Counter 进行细分统计 (各种 marker 组合)
    detail_counts = Counter(all_labels)
    
    # 3. 计算大类总数 (Neuron / Glia)
    total_cells = len(all_labels)
    total_neurons = sum(count for label, count in detail_counts.items() if label.startswith('neuron'))
    total_glia = sum(count for label, count in detail_counts.items() if label.startswith('glia'))
    
    # 4. 写入 Summary CSV 文件
    SUMMARY_PATH = os.path.join(derived['pATH_DET_RES'], f"{dir_name}_summary.csv")
    with open(SUMMARY_PATH, 'w', newline='', encoding='utf-8') as f_sum:
        sum_writer = csv.writer(f_sum)
        
        # 写入标题行和核心指标
        sum_writer.writerow(["Metric", "Count"])
        sum_writer.writerow(["Total_Detected_Cells", total_cells])
        sum_writer.writerow(["Total_Neurons", total_neurons])
        sum_writer.writerow(["Total_Glia", total_glia])
        sum_writer.writerow([]) # 空行隔开
        
        # 写入每个细分类别的详细统计
        sum_writer.writerow(["Detailed_Class_Name", "Cell_Count"])
        # 按类别名称排序，让结果整齐一点
        for label in sorted(detail_counts.keys()):
            sum_writer.writerow([label, detail_counts[label]])

    print(f"✅ Tile {dir_name} 统计完成: 共发现 {total_cells} 个细胞 (Neuron: {total_neurons}, Glia: {total_glia})")

    # 释放显存
    for mod_name in list(models_dict.keys()):
        del models_dict[mod_name]
        
    return all_tile_detections, dir_name

def process_single_tile_wrapper(args):
    # args 就是那个元组 (i, path, config, pos)
    # 使用 *args 自动解包传给原函数
    return process_single_tile(*args)