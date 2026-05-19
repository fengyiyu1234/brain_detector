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

_global_models = {}

def init_worker(config):
    """Pool initializer: load models once per worker process instead of once per tile."""
    global _global_models
    device = config['device']
    routing_config = [ch for ch in config.get('channels_routing', []) if ch.get('active', True)]
    required_models = {ch['model'].lower() for ch in routing_config}

    if 'yolo' in required_models:
        try:
            _global_models['yolo'] = YOLO(config['models']['yolo_path'])
            _global_models['yolo_classes'] = config.get('model_classes', {}).get('yolo', {"0": "neuron", "1": "glia"})
            print(f"[Worker PID:{os.getpid()}] ✔️ YOLO 模型加载成功")
        except Exception as e:
            print(f"[Worker PID:{os.getpid()}] ❌ YOLO 加载失败: {e}")

    if 'cellpose' in required_models:
        try:
            _global_models['cellpose'] = cp_models.CellposeModel(
                pretrained_model=config['models']['cellpose_path'],
                gpu=(device == 'cuda')
            )
            print(f"[Worker PID:{os.getpid()}] ✔️ Cellpose 模型加载成功")
        except Exception as e:
            print(f"[Worker PID:{os.getpid()}] ❌ Cellpose 加载失败: {e}")

def fast_cloud_read(local_path):
    # 没有任何乱七八糟的拷贝，纯粹的极速直读
    return cv2.imread(local_path, cv2.IMREAD_ANYDEPTH)

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

    # --- 1. 解析路由 ---
    routing_config = [ch for ch in config.get('channels_routing', []) if ch.get('active', True)]
    
    if not routing_config:
        current_logger.error("配置文件中没有激活的 channels_routing！")
        return [], dir_name

    # --- 2. Checkpoint (断点续传检查) ---
    # 检查是否当前 Tile 所有激活通道的专属 CSV 都已经存在
    all_channels_done = True
    for ch in routing_config:
        ch_csv_path = os.path.join(derived['pATH_DET_RES'], f"{dir_name}_{ch['id']}_result.csv")
        if not os.path.exists(ch_csv_path):
            all_channels_done = False
            break

    if all_channels_done:
        print(f"[{dir_name}] 所有通道的单据 (Result CSVs) 均已存在，跳过该 Tile。")
        return [], dir_name

    # --- 3. 打印确认通道路由 (模型已在 init_worker 中加载) ---
    print(f"\n[{dir_name}] === 确认通道推理路由 ===")
    for ch in routing_config:
        ch_dir = paths.get(ch['dir_key'], 'Unknown')
        print(f" -> 通道: {ch['id']} ({ch.get('type', 'N/A')}) | 模型: {ch['model'].upper()} | 路径: {ch_dir}")
    print("================================\n")

    yolo_classes = _global_models.get('yolo_classes', config.get('model_classes', {}).get('yolo', {"0": "neuron", "1": "glia"}))

    # --- 4. 建立多通道文件索引 (完美镜像结构版) ---
    anchor_ch = routing_config[0]
    anchor_root = os.path.abspath(paths[anchor_ch['dir_key']])
    
    print(f"\n[{dir_name}] ====== 文件检索 Checkpoint ======")
    
    # 获取锚点通道的所有图片
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

    # 提取无后缀名作为 Key
    testnames_no_ext = [os.path.splitext(f)[0] for f in testnames]

    is_downsample = dp.get('DOWNSAMPLE', False)
    
    if is_downsample:
        z_step = dp.get('DOWNSAMPLE_Z_STEP', 2)
        if z_step <= 1:
            current_logger.warning("⚠️ 开启了 DOWNSAMPLE 但 DOWNSAMPLE_Z_STEP <= 1，已自动修正为步长 2")
            z_step = 2
            
        current_logger.info(f"🚀 跳层检测开启：每隔 {z_step-1} 张处理一张 (抽样步长={z_step})")
        testnames = testnames[::z_step]
        testnames_no_ext = testnames_no_ext[::z_step]
    else:
        z_step = 1

    # 获取相对的二级结构路径
    rel_tile_path = os.path.relpath(pATHTEST, anchor_root)
    channel_files_indices = {}

    # 精准映射镜像通道
    for ch in routing_config:
        ch_id = ch['id']
        ch_root = os.path.abspath(paths[ch['dir_key']])
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
    pbar = tqdm(total=len(testnames_no_ext), desc=f"[{dir_name[:10]}] 准备启动...", position=i+1, leave=False)

    file_handles = {}
    csv_writers = {}

    for ch in routing_config:
        ch_id = ch['id']
        ch_csv_path = os.path.join(derived['pATH_DET_RES'], f"{dir_name}_{ch_id}_result.csv")
        f_ch = open(ch_csv_path, 'w', newline='', encoding='utf-8')
        file_handles[ch_id] = f_ch
        csv_writers[ch_id] = csv.writer(f_ch)
        
        # 写入规范的表头，方便后期跑全脑 3D Z-linker 时精准读取
        csv_writers[ch_id].writerow(['slice_name', 'x1', 'y1', 'x2', 'y2', 'class', 'score', 'mean', 'z'])
        
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
                current_z_real = z_idx * z_step + 1
                
                H0, W0 = 0, 0 

                for ch in routing_config:
                    ch_id = ch['id']
                    pbar.set_description(f"Tile:[{dir_name[:10]}] | Z层:[{current_z_real}/{len(testnames_no_ext)}] | 检测通道:[{ch_id}]")
                    ch_model = ch['model']
                    
                    # --- 1. 获取预取数据并维持流水线运转 ---
                    future = prefetch_futures.get((z_idx, ch_id))
                    if future is None: continue
                    img_raw = future.result()
                    del prefetch_futures[(z_idx, ch_id)]
                    if img_raw is None: continue

                    future_z_idx = z_idx + PREFETCH_DEPTH
                    if future_z_idx < len(testnames_no_ext):
                        future_name = testnames_no_ext[future_z_idx]
                        future_img_path = channel_files_indices[ch_id].get(future_name)
                        if future_img_path and os.path.exists(future_img_path):
                            prefetch_futures[(future_z_idx, ch_id)] = downloader_pool.submit(fast_cloud_read, future_img_path)
                        else:
                            prefetch_futures[(future_z_idx, ch_id)] = None

                    # --- 2. GPU 推理图像预处理 ---
                    if len(img_raw.shape) == 3: img_raw = img_raw[:, :, 0]
                    if H0 == 0: H0, W0 = img_raw.shape[:2]

                    norm_img = normalize_for_detection(img_raw, dp['normalize_PERCENTILE_LOW'], dp['normalize_PERCENTILE_HIGH'])

                    # --------- YOLO 单通道滑动窗口检测 ---------
                    if ch_model == 'yolo':
                        img_infer = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR) 
                        
                        xsize, ysize, step_win = dp['xsize'], dp['ysize'], dp['step']
                        H_pad = H0 if (H0-ysize)%step_win == 0 else H0-H0%step_win+ysize
                        W_pad = W0 if (W0-xsize)%step_win == 0 else W0-W0%step_win+xsize
                        
                        fullimg_pad = np.zeros((H_pad, W_pad, 3), dtype=np.uint8)
                        fullimg_pad[0:H0, 0:W0] = img_infer
                        
                        conf_thresh = dp.get('conf_thresh', dp.get('tHRESHOLD', 0.25))
                        nms_iou = dp.get('nms_iou', dp.get('mINIOU', 0.45))
                        
                        raw_detections = np.empty((0, 6))
                        
                        # 滑动窗口批处理
                        batch_patches = []
                        batch_coords = []
                        BATCH_SIZE = 8
                        
                        for x in range(0, W_pad, step_win):
                            for y in range(0, H_pad, step_win):
                                patch = fullimg_pad[y:y+ysize, x:x+xsize]
                                if patch.max() < 10: continue 
                                
                                batch_patches.append(patch)
                                batch_coords.append((x, y))
                                
                                if len(batch_patches) == BATCH_SIZE:
                                    results = _global_models['yolo'].predict(batch_patches, device=device, verbose=False, conf=conf_thresh, iou=nms_iou)
                                    for i, res in enumerate(results):
                                        if len(res.boxes) > 0:
                                            bx, by = batch_coords[i]
                                            boxes = res.boxes.xyxy.cpu().numpy() + np.array([bx, by, bx, by])
                                            scores = res.boxes.conf.cpu().numpy()
                                            labels = res.boxes.cls.cpu().numpy()
                                            patch_res = np.hstack((boxes, scores[:, np.newaxis], labels[:, np.newaxis]))
                                            raw_detections = np.append(raw_detections, patch_res, axis=0)
                                    batch_patches = []
                                    batch_coords = []

                        if len(batch_patches) > 0:
                            results = _global_models['yolo'].predict(batch_patches, device=device, verbose=False, conf=conf_thresh, iou=nms_iou)
                            for i, res in enumerate(results):
                                if len(res.boxes) > 0:
                                    bx, by = batch_coords[i]
                                    boxes = res.boxes.xyxy.cpu().numpy() + np.array([bx, by, bx, by])
                                    scores = res.boxes.conf.cpu().numpy()
                                    labels = res.boxes.cls.cpu().numpy()
                                    patch_res = np.hstack((boxes, scores[:, np.newaxis], labels[:, np.newaxis]))
                                    raw_detections = np.append(raw_detections, patch_res, axis=0)

                        unique_labels = np.unique(raw_detections[:, 5]) if raw_detections.size > 0 else []
                        for lbl in unique_labels:
                            layer_label_data = raw_detections[raw_detections[:, 5] == lbl, :-1]
                            if layer_label_data.size > 0:
                                # 调用基于 IoU 的新版 NMS
                                cleaned_boxes = stitchDetection(layer_label_data) 
                                class_str = yolo_classes.get(str(int(lbl)), "unknown")
                                
                                for box in cleaned_boxes:
                                    # 格式: [slice_name, x1, y1, x2, y2, class, score, mean, z_val]
                                    row = [name_no_ext, box[0], box[1], box[2], box[3], class_str, box[4], 0.0, current_z_real]
                                    csv_writers[ch_id].writerow(row)

                    # --------- Cellpose 单通道密集细胞核检测 ---------
                    elif ch_model == 'cellpose':
                        masks, flows, styles = _global_models['cellpose'].eval(norm_img, diameter=None, channels=[0,0])
                        cp_boxes = extract_boxes_from_masks(masks)
                        
                        for box in cp_boxes:
                            # 格式: [slice_name, x1, y1, x2, y2, class, score, mean, z_val]
                            row = [name_no_ext, box[0], box[1], box[2], box[3], box[5], box[4], 0.0, current_z_real]
                            csv_writers[ch_id].writerow(row)
                    
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                pbar.update(1)
    finally:
        for f in file_handles.values():
            f.close()
        pbar.close()
    
    # =========================================================
    # 步骤 6: 彻底移除局部统计，仅做内存清理与退出
    # =========================================================
    print(f"✅ Tile {dir_name} 单通道 2D 特征提取完成，已移交至全局 CSV。")

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # 返回空列表，因为我们不再向主进程回传冗余的内存数据
    return [], dir_name

def process_single_tile_wrapper(args):
    return process_single_tile(*args)