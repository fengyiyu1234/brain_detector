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
from cellpose import models as cp_models
from concurrent.futures import ThreadPoolExecutor
import time
import logging
from src.utils.logger import setup_logging  

logger = logging.getLogger(__name__)

def save_raw_qc_visualization(norm_img, boxes, save_path, color=(0, 255, 0)):
    """
    保存单通道原始质控图：将原始预测框直接画在归一化后的 8-bit 底图上
    """
    # 确保图片是 3 通道 BGR，这样才能画出彩色的框
    if len(norm_img.shape) == 2:
        vis_img = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)
    else:
        vis_img = norm_img.copy()

    # 将模型输出的框原封不动地画上去
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, thickness=2)

    # 确保输出目录存在并保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, vis_img)

# 确保你外部定义的这个函数现在长这样就行了：
def fast_cloud_read(local_path):
    # 没有任何乱七八糟的拷贝，纯粹的极速直读
    return cv2.imread(local_path, cv2.IMREAD_ANYDEPTH)

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
        matched = False
        for cell in target_cells:
            ioa = calculate_ioa(tf_box[:4], cell["box"]) 
            if ioa >= ioa_thresh:
                cell["markers"].append("Sox9") 
                matched = True
                break

        if not matched:
            target_cells.append({
                "box": tf_box[:4],
                "cell_type": "nucleus", # 或者 "unknown_cell"
                "markers": ["Sox9"],
                "score": tf_box[4]
            })
                
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

    output_dir = derived.get('pATH_DET_RES')
    if output_dir:
        setup_logging(log_path=output_dir)
        
    current_logger = logging.getLogger(__name__)
    current_logger.info(f"🚀 Worker (PID:{os.getpid()}) 开始接管 Tile: {dir_name}，日志已绑定至输出目录！")

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

    is_downsample = dp.get('DOWNSAMPLE', False)
    
    if is_downsample:
        z_step = dp.get('DOWNSAMPLE_Z_STEP', 2) # 默认设个底线为2
        if z_step <= 1:
            current_logger.warning("⚠️ 开启了 DOWNSAMPLE 但 DOWNSAMPLE_Z_STEP <= 1，已自动修正为步长 2")
            z_step = 2
            
        current_logger.info(f"🚀 跳层检测开启：每隔 {z_step-1} 张处理一张 (抽样步长={z_step})")
        testnames = testnames[::z_step]
        testnames_no_ext = testnames_no_ext[::z_step]
    else:
        z_step = 1 # 未开启抽样，步长强制为 1

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
        PREFETCH_DEPTH = 10 # 预取深度：永远保持未来 10 层的数据在后台下载
        prefetch_futures = {} # 存放后台下载任务的字典: {(z_idx, ch_id): future}
       
        with ThreadPoolExecutor(max_workers=4) as downloader_pool:
            
            pbar.set_description(f"[{dir_name[:10]}] 正在填装初始流水线...")

            # --- 1. 填装初始弹药：提前把前 PREFETCH_DEPTH 层抛给后台 ---
            for pre_z in range(min(PREFETCH_DEPTH, len(testnames_no_ext))):
                pre_name = testnames_no_ext[pre_z]
                for ch in routing_config:
                    ch_id = ch['id']
                    img_path = channel_files_indices[ch_id].get(pre_name)
                    
                    if img_path and os.path.exists(img_path):
                        # submit 不会阻塞，它只是把任务扔给后台线程，然后立刻返回一个 Future 对象
                        prefetch_futures[(pre_z, ch_id)] = downloader_pool.submit(fast_cloud_read, img_path)
                    else:
                        prefetch_futures[(pre_z, ch_id)] = None

            # --- 2. 正式启动 GPU 核心消费者循环 ---
            for z_idx, name_no_ext in enumerate(testnames_no_ext):
                # 如果开启了 Z 轴降采样，那么真实的物理层数应该是 1, 3, 5, 7...
                current_z_real = z_idx * z_step + 1
                    
                layer_channel_boxes = {ch['id']: [] for ch in routing_config}
            
                H0, W0 = 0, 0 
                
                visualize_tile = dp.get('vISUALIZE_TILE', False)
                sample_step = config.get('vISUALIZATIONSAMPLESTEP', 100)
                sample_count = config.get('vISUALIZATIONSAMPLECOUNT', 5)
                need_vis = visualize_tile and (z_idx % sample_step < sample_count)

                for ch in routing_config:
                    ch_id = ch['id']
                    pbar.set_description(f"Tile:[{dir_name[:10]}] | Z层:[{current_z_real}/{len(testnames_no_ext)}] | 检测通道:[{ch_id}]")
                    ch_model = ch['model']
                    img_path = channel_files_indices[ch_id].get(name_no_ext)
                    
                    future = prefetch_futures.get((z_idx, ch_id))
                    
                    # 1. 检查物理文件本来就不存在的情况
                    if future is None:
                        error_msg = f"‼️ [数据缺失] Tile: {dir_name} | Z层: {current_z_real} | 通道: {ch_id} | 路径: {img_path}"
                        logger.error(error_msg) 
                        continue

                    # 2. .result() 会索要结果。
                    img_raw = future.result()
                    
                    # 3. 拿到货后，立刻释放字典中的任务，防止撑爆内存
                    del prefetch_futures[(z_idx, ch_id)]
                    
                    # 4. 检查读取内容是否合法
                    if img_raw is None:
                        error_msg = f"❌ [读取失败] Tile: {dir_name} | Z层: {current_z_real} | 通道: {ch_id} | 文件可能损坏: {img_path}"
                        logger.error(error_msg)
                        continue

                    # ---------------------------------------------------------
                    # 📤 维持流水线运转 (补充未来的预取任务)
                    # ---------------------------------------------------------
                    # 我消费了当前这层，立刻通知后台去下载未来 (z_idx + PREFETCH_DEPTH) 的那层
                    future_z_idx = z_idx + PREFETCH_DEPTH
                    if future_z_idx < len(testnames_no_ext):
                        future_name = testnames_no_ext[future_z_idx]
                        future_img_path = channel_files_indices[ch_id].get(future_name)
                        
                        if future_img_path and os.path.exists(future_img_path):
                            prefetch_futures[(future_z_idx, ch_id)] = downloader_pool.submit(fast_cloud_read, future_img_path)
                        else:
                            prefetch_futures[(future_z_idx, ch_id)] = None

                    # ---------------------------------------------------------
                    # 🧠 开始 GPU 推理逻辑 (以下代码和原来完全一样)
                    # ---------------------------------------------------------
                    if len(img_raw.shape) == 3: img_raw = img_raw[:, :, 0]
                    
                    if H0 == 0: H0, W0 = img_raw.shape[:2] # 记录当前层的实际尺寸

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
                        
                        # batch processing
                        batch_patches = []
                        batch_coords = []
                        BATCH_SIZE = 8  # 你可以根据 P5000 的显存调节，比如 16 或 32
                        
                        for x in range(0, W_pad, step_win):
                            for y in range(0, H_pad, step_win):
                                patch = fullimg_pad[y:y+ysize, x:x+xsize]
                                if patch.max() < 10: continue # 过滤纯黑背景
                                
                                batch_patches.append(patch)
                                batch_coords.append((x, y))
                                
                                if len(batch_patches) == BATCH_SIZE:
                                    results = models_dict['yolo'].predict(batch_patches, device=device, verbose=False, conf=conf_thresh, iou=nms_iou)
                                    
                                    # 批量解析结果
                                    for i, res in enumerate(results):
                                        if len(res.boxes) > 0:
                                            bx, by = batch_coords[i]
                                            boxes = res.boxes.xyxy.cpu().numpy() + np.array([bx, by, bx, by])
                                            scores = res.boxes.conf.cpu().numpy()
                                            labels = res.boxes.cls.cpu().numpy()
                                            patch_res = np.hstack((boxes, scores[:, np.newaxis], labels[:, np.newaxis]))
                                            raw_detections = np.append(raw_detections, patch_res, axis=0)
                                    
                                    # 清空队列，准备装载下一批
                                    batch_patches = []
                                    batch_coords = []

                        if len(batch_patches) > 0:
                            results = models_dict['yolo'].predict(batch_patches, device=device, verbose=False, conf=conf_thresh, iou=nms_iou)
                            for i, res in enumerate(results):
                                if len(res.boxes) > 0:
                                    bx, by = batch_coords[i]
                                    boxes = res.boxes.xyxy.cpu().numpy() + np.array([bx, by, bx, by])
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

                        if need_vis and H0 > 0:
                            # 动态颜色：根据通道名分配，GFP标绿，RFP标红，其他标黄
                            color = (0, 255, 0) if "gfp" in ch_id.lower() else (0, 0, 255) if "rfp" in ch_id.lower() else (0, 255, 255)
                            qc_path = os.path.join(derived['pATH_VIS_TILE'], dir_name, f"{name_no_ext}_Z{current_z_real:03d}_{ch_id}_RAW.jpg")
                            save_raw_qc_visualization(norm_img, kept_yolo_boxes, qc_path, color=color)

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

                        if need_vis and H0 > 0:
                            # 细胞核(Sox9)默认用白色框
                            qc_path = os.path.join(derived['pATH_VIS_TILE'], dir_name, f"{name_no_ext}_Z{current_z_real:03d}_{ch_id}_RAW.jpg")
                            save_raw_qc_visualization(norm_img, cp_boxes, qc_path, color=(255, 255, 255))
                    
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # 跨通道空间重叠合并 (Colocalization)
                # 将各个通道提取好的 Box 融合，填入 Sox9 核，并且合并 RFP/GFP
                final_merged_boxes = colocalization(layer_channel_boxes, iou_thresh=0.5, ioa_thresh=0.6)

                for item in final_merged_boxes:
                    # final_merged_boxes 的格式是 [x1, y1, x2, y2, score, final_class_name]
                    x1, y1, x2, y2, score, class_name = item
                    mean_val = 0.0
                    
                    writer_combined.writerow([name_no_ext, x1, y1, x2, y2, class_name, score, mean_val, current_z_real])
                    
                    # 追加至全局列表供 Z-linker 3D 追踪使用
                    all_tile_detections.append([x1, y1, x2, y2, score, mean_val, class_name, current_z_real])

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
    
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return all_tile_detections, dir_name

def process_single_tile_wrapper(args):
    # args 就是那个元组 (i, path, config, pos)
    # 使用 *args 自动解包传给原函数
    return process_single_tile(*args)