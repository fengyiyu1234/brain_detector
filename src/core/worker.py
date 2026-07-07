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
from concurrent.futures import ThreadPoolExecutor
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
    routing_config = config.get('channels_routing_detect') or [
        ch for ch in config.get('channels_routing', []) if ch.get('active', True)
    ]
    required_models = {ch['model'].lower() for ch in routing_config}

    if 'yolo' in required_models:
        try:
            _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            yolo_path = config['models']['yolo_path']
            if not os.path.isabs(yolo_path):
                yolo_path = os.path.join(_project_root, yolo_path)
            _global_models['yolo'] = YOLO(yolo_path, task='detect')
            _global_models['yolo'].model.to(device)  # 提前占 GPU 显存，防止 TF 后来挤占
            _global_models['yolo_classes'] = config.get('model_classes', {}).get('yolo', {"0": "neuron", "1": "glia"})
            print(f"[Worker PID:{os.getpid()}] ✔️ YOLO 模型加载成功")
        except Exception as e:
            print(f"[Worker PID:{os.getpid()}] ❌ YOLO 加载失败: {e}")


    if 'stardist' in required_models:
        try:
            import tensorflow as tf
            tf_gpus = tf.config.list_physical_devices('GPU')
            if tf_gpus:
                gpu_idx = int(_worker_device.split(':')[-1]) if ':' in _worker_device else 0
                # 多卡时 TF 放另一块 GPU 避免与 PyTorch 争显存；单卡共用同一块
                if len(tf_gpus) > 1:
                    tf_gpu_idx = (gpu_idx + 1) % len(tf_gpus)
                else:
                    tf_gpu_idx = 0
                target_gpu = tf_gpus[tf_gpu_idx]
                tf.config.set_visible_devices(target_gpu, 'GPU')
                tf.config.experimental.set_memory_growth(target_gpu, True)
            from stardist.models import StarDist2D
            _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            sd_basedir = config.get('models', {}).get('stardist_basedir', None)
            sd_name    = config.get('models', {}).get('stardist_name', '2D_versatile_fluo')
            if sd_basedir:
                if not os.path.isabs(sd_basedir):
                    sd_basedir = os.path.join(_project_root, sd_basedir)
                _global_models['stardist'] = StarDist2D(None, name=sd_name, basedir=sd_basedir)
            else:
                _global_models['stardist'] = StarDist2D.from_pretrained('2D_versatile_fluo')
            print(f"[Worker PID:{os.getpid()}] ✔️ StarDist 模型加载成功")
        except Exception as e:
            print(f"[Worker PID:{os.getpid()}] ❌ StarDist 加载失败: {e}")
            raise RuntimeError(f"StarDist model failed to load: {e}") from e

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



def _iomin_nms_rows(rows, containment_thresh):
    """Suppress lower-score boxes whose center is almost fully inside a higher-score box.

    rows: list of [name, x1, y1, x2, y2, class_str, score, mean, z]
    Only boxes with IoMin > containment_thresh are suppressed (IoMin = inter / min_area).
    """
    if len(rows) < 2:
        return rows

    # group by z so we only compare detections on the same slice
    z_to_indices = {}
    for idx, row in enumerate(rows):
        z = row[8]
        z_to_indices.setdefault(z, []).append(idx)

    keep_mask = [True] * len(rows)

    for indices in z_to_indices.values():
        if len(indices) < 2:
            continue
        z_rows = [rows[i] for i in indices]
        n = len(z_rows)
        x1 = np.array([r[1] for r in z_rows], dtype=np.float32)
        y1 = np.array([r[2] for r in z_rows], dtype=np.float32)
        x2 = np.array([r[3] for r in z_rows], dtype=np.float32)
        y2 = np.array([r[4] for r in z_rows], dtype=np.float32)
        scores = np.array([r[6] for r in z_rows], dtype=np.float32)
        areas = (x2 - x1) * (y2 - y1)

        order = np.argsort(-scores)  # highest score first
        suppressed = np.zeros(n, dtype=bool)

        for i in range(n):
            ai = order[i]
            if suppressed[ai]:
                continue
            for j in range(i + 1, n):
                aj = order[j]
                if suppressed[aj]:
                    continue
                ix1 = max(x1[ai], x1[aj])
                iy1 = max(y1[ai], y1[aj])
                ix2 = min(x2[ai], x2[aj])
                iy2 = min(y2[ai], y2[aj])
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                if inter == 0.0:
                    continue
                iomin = inter / (min(areas[ai], areas[aj]) + 1e-6)
                if iomin > containment_thresh:
                    suppressed[aj] = True  # suppress lower-score (smaller) box

        for local_idx, global_idx in enumerate(indices):
            if suppressed[local_idx]:
                keep_mask[global_idx] = False

    return [row for i, row in enumerate(rows) if keep_mask[i]]


def _apply_bbox_filters(rows, model_dp):
    """Apply size/intensity filters to detection rows at write time.

    rows: list of [name, x1, y1, x2, y2, class, score, mean, z]
    Filters applied in order: bbox_min/max → aspect_ratio → area_pct → mean_pct → mean_min.
    Percentile filters are computed over the full tile (all z-slices) so thresholds match
    what Stage 2.75 would produce.
    """
    if not rows or not model_dp:
        return rows
    r = np.array(rows, dtype=object)
    x1 = r[:, 1].astype(float); y1 = r[:, 2].astype(float)
    x2 = r[:, 3].astype(float); y2 = r[:, 4].astype(float)
    mean_vals = r[:, 7].astype(float)
    w = x2 - x1; h = y2 - y1
    mask = np.ones(len(r), dtype=bool)

    bbox_min = model_dp.get('bbox_min')
    if bbox_min is not None:
        mask &= (w >= bbox_min) & (h >= bbox_min)
    bbox_max = model_dp.get('bbox_max')
    if bbox_max is not None:
        mask &= (w <= bbox_max) & (h <= bbox_max)
    aspect_max = model_dp.get('bbox_max_aspect_ratio')
    if aspect_max is not None:
        mask &= np.maximum(w, h) <= aspect_max * np.maximum(np.minimum(w, h), 1e-6)

    areas = w * h
    area_pct_min = model_dp.get('bbox_area_pct_min')
    if area_pct_min is not None and mask.any():
        thresh = float(np.percentile(areas[mask], area_pct_min))
        mask &= areas >= thresh
    area_pct_max = model_dp.get('bbox_area_pct_max')
    if area_pct_max is not None and mask.any():
        thresh = float(np.percentile(areas[mask], area_pct_max))
        mask &= areas <= thresh
    mean_pct_min = model_dp.get('bbox_mean_pct_min')
    if mean_pct_min is not None and mask.any():
        thresh = float(np.percentile(mean_vals[mask], mean_pct_min))
        mask &= mean_vals >= thresh
    mean_min = model_dp.get('bbox_mean_min') or 0
    if mean_min > 0:
        mask &= mean_vals >= mean_min

    return [rows[i] for i in range(len(rows)) if mask[i]]


def _write_filtered_detections(det_buf, csv_writers, dp, ch_routing_map):
    """Write detections to CSV, applying bbox/intensity filters and per-z IoMin NMS.

    Filters (bbox_min/max, aspect_ratio, area_pct, mean_pct, mean_min) are applied here
    so 1_tile_2d_raw already contains clean boxes.  Stage 2.75 can then be disabled
    (stage_2_75_enabled=false) to skip redundant re-filtering.
    """
    yolo_dp = dp.get('yolo', {})
    sd_dp   = dp.get('stardist', {})
    containment_thresh = yolo_dp.get('nms_containment_thresh', None)

    for ch_id, rows in det_buf.items():
        ch    = ch_routing_map.get(ch_id, {})
        model = ch.get('model', '').lower()
        model_dp = sd_dp if model == 'stardist' else yolo_dp
        rows = _apply_bbox_filters(rows, model_dp)
        if containment_thresh is not None and model == 'yolo':
            rows = _iomin_nms_rows(rows, containment_thresh)
        for row in rows:
            csv_writers[ch_id].writerow(row)


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

    # --- 1. 解析路由 (含 double_exposure 展开出的第二曝光合成通道，仅用于检测) ---
    routing_config = config.get('channels_routing_detect') or [
        ch for ch in config.get('channels_routing', []) if ch.get('active', True)
    ]

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
        PREFETCH_DEPTH = 8  # 预取深度：降低以减少内存峰值压力
        prefetch_futures = {}

        det_buf        = {ch['id']: [] for ch in channels_to_run}
        ch_routing_map = {ch['id']: ch for ch in channels_to_run}

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
                        BATCH_SIZE = 4

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
                                    x1r = max(0, int(round(box[0]))); x2r = min(W0, int(round(box[2])))
                                    y1r = max(0, int(round(box[1]))); y2r = min(H0, int(round(box[3])))
                                    mean_val = float(img_raw[y1r:y2r, x1r:x2r].mean()) if y2r > y1r and x2r > x1r else 0.0
                                    det_buf[ch_id].append([name_no_ext, box[0], box[1], box[2], box[3], class_str, box[4], mean_val, current_z_real])

                    # --------- StarDist: 逐切片推断，直接写出 BBox ---------
                    elif ch_model == 'stardist':
                        from csbdeep.utils import normalize as csbdeep_normalize
                        from skimage.measure import regionprops
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        sd_dp = dp.get('stardist', {})
                        sd_img = csbdeep_normalize(
                            img_raw.astype(np.float32),
                            sd_dp.get('norm_low', 1),
                            sd_dp.get('norm_high', 99.8),
                        )
                        labels, _ = _global_models['stardist'].predict_instances(
                            sd_img, axes='YX',
                            n_tiles=tuple(sd_dp.get('n_tiles', [2, 2])),
                            prob_thresh=sd_dp.get('prob_thresh', 0.5),
                            nms_thresh=sd_dp.get('nms_thresh', 0.4),
                        )
                        for prop in regionprops(labels, intensity_image=img_raw):
                            min_r, min_c, max_r, max_c = prop.bbox
                            det_buf[ch_id].append([
                                name_no_ext, int(min_c), int(min_r), int(max_c), int(max_r),
                                "nucleus", 1.0, float(prop.mean_intensity), current_z_real,
                            ])

                pbar.update(1)

        # --- 3. Per-tile filter and write ---
        _write_filtered_detections(det_buf, csv_writers, dp, ch_routing_map)

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