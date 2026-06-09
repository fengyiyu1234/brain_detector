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
import torch
from src.utils.logger import setup_logging  

logger = logging.getLogger(__name__)

_global_models = {}
_worker_device = 'cpu'

def init_worker(config, gpu_queue=None):
    """Pool initializer: load models once per worker process instead of once per tile."""
    global _global_models, _worker_device
    if gpu_queue is not None and torch.cuda.is_available():
        gpu_id = gpu_queue.get()
        torch.cuda.set_device(gpu_id)
        _worker_device = f'cuda:{gpu_id}'
    else:
        _worker_device = config['device']
    device = _worker_device
    routing_config = [ch for ch in config.get('channels_routing', []) if ch.get('active', True)]
    required_models = {ch['model'].lower() for ch in routing_config}

    if 'yolo' in required_models:
        try:
            _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            yolo_path = config['models']['yolo_path']
            if not os.path.isabs(yolo_path):
                yolo_path = os.path.join(_project_root, yolo_path)
            _global_models['yolo'] = YOLO(yolo_path, task='detect')
            _global_models['yolo_classes'] = config.get('model_classes', {}).get('yolo', {"0": "neuron", "1": "glia"})
            print(f"[Worker PID:{os.getpid()}] ✔️ YOLO 模型加载成功")
        except Exception as e:
            print(f"[Worker PID:{os.getpid()}] ❌ YOLO 加载失败: {e}")

    if 'cellpose' in required_models:
        try:
            _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            cellpose_path = config['models']['cellpose_path']
            if not os.path.isabs(cellpose_path):
                cellpose_path = os.path.join(_project_root, cellpose_path)
            _global_models['cellpose'] = cp_models.CellposeModel(
                pretrained_model=cellpose_path,
                gpu=device.startswith('cuda')
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
    for val in np.unique(masks):
        if val == 0: continue
        y_idx, x_idx = np.where(masks == val)
        if len(y_idx) == 0: continue
        x1, x2 = np.min(x_idx), np.max(x_idx)
        y1, y2 = np.min(y_idx), np.max(y_idx)
        boxes.append([x1, y1, x2, y2, 1.0, "nucleus"])
    return boxes


def boxes_from_prob_map(prob_map, threshold=0.5, min_area=10):
    """从 Cellpose 概率图直接提取 BBox，跳过 flow dynamics 后处理（更快）"""
    from skimage.measure import label as sk_label, regionprops
    binary = prob_map > threshold
    if not binary.any():
        return []
    labeled = sk_label(binary)
    boxes = []
    for region in regionprops(labeled):
        if region.area < min_area:
            continue
        min_r, min_c, max_r, max_c = region.bbox
        boxes.append([int(min_c), int(min_r), int(max_c), int(max_r), 1.0, "nucleus"])
    return boxes


def process_single_tile(i, pATHTEST, config):
    current_logger = logging.getLogger(__name__)
    dir_name = os.path.basename(pATHTEST)

    dp = config['detection_params']
    derived = config['derived_paths']
    paths = config['paths']
    device = _worker_device

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

    # --- 2. Checkpoint (按通道精细跳过) ---
    channels_to_run = [
        ch for ch in routing_config
        if not os.path.exists(
            os.path.join(derived['pATH_DET_RES'], f"{dir_name}_{ch['id']}_result.csv")
        )
    ]
    skipped = [ch['id'] for ch in routing_config if ch not in channels_to_run]
    if skipped:
        print(f"[{dir_name}] 已有结果，跳过通道: {skipped}")
    if not channels_to_run:
        print(f"[{dir_name}] 所有通道均已完成，跳过该 Tile。")
        return [], dir_name

    # --- 3. 打印确认通道路由 (模型已在 init_worker 中加载) ---
    print(f"\n[{dir_name}] === 确认通道推理路由 ===")
    for ch in channels_to_run:
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
    for ch in channels_to_run:
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

    for ch in channels_to_run:
        ch_id = ch['id']
        ch_csv_path = os.path.join(derived['pATH_DET_RES'], f"{dir_name}_{ch_id}_result.csv")
        f_ch = open(ch_csv_path, 'w', newline='', encoding='utf-8')
        file_handles[ch_id] = f_ch
        csv_writers[ch_id] = csv.writer(f_ch)
        
        # 写入规范的表头，方便后期跑全脑 3D Z-linker 时精准读取
        csv_writers[ch_id].writerow(['slice_name', 'x1', 'y1', 'x2', 'y2', 'class', 'score', 'mean', 'z'])
        
    try:
        PREFETCH_DEPTH = 24  # 预取深度：网络盘 latency 高，多预取更多层掩盖延迟
        prefetch_futures = {}

        cellpose_channels = [ch for ch in channels_to_run if ch['model'] == 'cellpose']
        cellpose_buffer = {ch['id']: [] for ch in cellpose_channels}

        with ThreadPoolExecutor(max_workers=16) as downloader_pool:

            pbar.set_description(f"[{dir_name[:10]}] 正在填装初始流水线...")

            # --- 1. 填装初始弹药：提前把前 PREFETCH_DEPTH 层抛给后台 ---
            for pre_z in range(min(PREFETCH_DEPTH, len(testnames_no_ext))):
                pre_name = testnames_no_ext[pre_z]
                for ch in channels_to_run:
                    ch_id = ch['id']
                    img_path = channel_files_indices[ch_id].get(pre_name)

                    if img_path and os.path.exists(img_path):
                        prefetch_futures[(pre_z, ch_id)] = downloader_pool.submit(fast_cloud_read, img_path)
                    else:
                        prefetch_futures[(pre_z, ch_id)] = None

            # --- 2. 正式启动 GPU 核心消费者循环 ---
            for z_idx, name_no_ext in enumerate(testnames_no_ext):
                current_z_real = z_idx * z_step + 1

                H0, W0 = 0, 0

                for ch in channels_to_run:
                    ch_id = ch['id']
                    pbar.set_description(f"Tile:[{dir_name[:10]}] | Z层:[{current_z_real}/{len(testnames_no_ext)}] | 检测通道:[{ch_id}]")
                    ch_model = ch['model']

                    # --- 获取预取数据并维持流水线运转 ---
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

                    # --- GPU 推理图像预处理 ---
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

                        raw_det_chunks = []
                        batch_patches = []
                        batch_coords = []
                        BATCH_SIZE = 8

                        def _flush_yolo_batch(patches, coords):
                            results = _global_models['yolo'].predict(patches, device=device, verbose=False, conf=conf_thresh, iou=nms_iou)
                            for k, res in enumerate(results):
                                if len(res.boxes) > 0:
                                    bx, by = coords[k]
                                    boxes = res.boxes.xyxy.cpu().numpy() + np.array([bx, by, bx, by])
                                    scores = res.boxes.conf.cpu().numpy()
                                    labels = res.boxes.cls.cpu().numpy()
                                    raw_det_chunks.append(np.hstack((boxes, scores[:, np.newaxis], labels[:, np.newaxis])))

                        for x in range(0, W_pad, step_win):
                            for y in range(0, H_pad, step_win):
                                patch = fullimg_pad[y:y+ysize, x:x+xsize]
                                if patch.max() < 10: continue

                                batch_patches.append(patch)
                                batch_coords.append((x, y))

                                if len(batch_patches) == BATCH_SIZE:
                                    _flush_yolo_batch(batch_patches, batch_coords)
                                    batch_patches = []
                                    batch_coords = []

                        if batch_patches:
                            _flush_yolo_batch(batch_patches, batch_coords)

                        raw_detections = np.concatenate(raw_det_chunks, axis=0) if raw_det_chunks else np.empty((0, 6))

                        unique_labels = np.unique(raw_detections[:, 5]) if raw_detections.size > 0 else []
                        for lbl in unique_labels:
                            layer_label_data = raw_detections[raw_detections[:, 5] == lbl, :-1]
                            if layer_label_data.size > 0:
                                cleaned_boxes = stitchDetection(layer_label_data)
                                class_str = yolo_classes.get(str(int(lbl)), "unknown")
                                for box in cleaned_boxes:
                                    row = [name_no_ext, box[0], box[1], box[2], box[3], class_str, box[4], 0.0, current_z_real]
                                    csv_writers[ch_id].writerow(row)

                    # --------- Cellpose: 累积切片，循环后统一批量推断 ---------
                    elif ch_model == 'cellpose':
                        cellpose_buffer[ch_id].append((name_no_ext, current_z_real, norm_img))

                pbar.update(1)

        # --- 3. Cellpose 批量推断（所有切片一次性送 GPU）---
        for ch in cellpose_channels:
            ch_id = ch['id']
            buffer = cellpose_buffer[ch_id]
            if not buffer:
                continue

            pbar.set_description(f"Tile:[{dir_name[:10]}] | Cellpose 批量推断 [{ch_id}] {len(buffer)} 层...")
            names_cp, z_reals_cp, imgs_cp = zip(*buffer)

            _, flows, _ = _global_models['cellpose'].eval(
                list(imgs_cp), diameter=None, channels=[0, 0],
                batch_size=dp.get('cellpose_batch_size', 32),
                compute_masks=False
            )
            cp_threshold = dp.get('cellpose_prob_threshold', 0.5)
            prob_maps = flows[2]  # list of [H, W] cell probability maps，由 GPU 前向传播生成

            for slice_name, z_real, prob_map in zip(names_cp, z_reals_cp, prob_maps):
                cp_boxes = boxes_from_prob_map(prob_map, threshold=cp_threshold)
                for box in cp_boxes:
                    row = [slice_name, box[0], box[1], box[2], box[3], box[5], box[4], 0.0, z_real]
                    csv_writers[ch_id].writerow(row)

            current_logger.info(f"[{dir_name}][{ch_id}] Cellpose 批量推断完成: {len(buffer)} 层")
    finally:
        for f in file_handles.values():
            f.close()
        pbar.close()
    
    # =========================================================
    # 步骤 6: 彻底移除局部统计，仅做内存清理与退出
    # =========================================================
    print(f"✅ Tile {dir_name} 单通道 2D 特征提取完成，已移交至全局 CSV。")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # 返回空列表，因为我们不再向主进程回传冗余的内存数据
    return [], dir_name

def process_single_tile_wrapper(args):
    return process_single_tile(*args)