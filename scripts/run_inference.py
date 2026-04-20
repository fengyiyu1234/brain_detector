# -*- coding: utf-8 -*-
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import time
import csv
import logging
import multiprocessing as mp
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from scipy.spatial import cKDTree
from src.config.loader import load_config
from src.utils.logger import setup_logging
from src.utils.io import listTile, loadTeraxml, save_run_metadata
from src.core.worker import process_single_tile_wrapper
from src.core.stitcher import combine_predictions
from src.core.z_linker import run_z_linker


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    start_time = time.time()
    # ==========================================
    # 阶段 1: 准备工作 (配置读取、动态路径校验)
    # ==========================================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device.upper()}")

    # 1. 加载配置
    config = load_config('D:/Fengyi/brain_detector/config/config.json')
    config['device'] = device
    paths = config['paths']
    dp = config['detection_params']
    derived = config['derived_paths']

    # --- 核心修改：动态解析路由，确定基准(Anchor)通道 ---
    routing_config = [ch for ch in config.get('channels_routing', []) if ch.get('active', True)]
    if not routing_config:
        raise ValueError("❌ 配置文件中没有任何激活的通道路由 (channels_routing)！")
        
    anchor_ch = routing_config[0]
    anchor_dir = paths.get(anchor_ch['dir_key'])

    print("\n=== 多通道路径检查 ===")
    for ch in routing_config:
        d_path = paths.get(ch['dir_key'])
        if not d_path or not os.path.exists(d_path):
            raise FileNotFoundError(f"❌ 通道 {ch['id']} 路径不存在: {d_path}")
        print(f"✔️ {ch['id']}: {d_path}")
    print("======================\n")

    # 2. 获取 Tile 列表 (基于锚点通道)
    dirnames, pATHTILE_all = listTile(anchor_dir)
    if not pATHTILE_all:
        raise ValueError(f"❌ 在锚点目录 {anchor_dir} 中没有找到合法的 Tile！")

    setup_logging(paths.get('pATHRESULT', '')) 
    logging.info("starting Multi-Channel Inference...")
    logging.info(f"current device: {config['device']}")
    save_run_metadata(config, start_time)

    # 3. 范围筛选
    sTARTID = dp.get('sTARTID') or 1
    eNDID = dp.get('eNDID') or len(pATHTILE_all)
    target_indices = list(range(sTARTID - 1, eNDID))
    pATHTILE = [pATHTILE_all[i] for i in target_indices]
    
    os.makedirs(paths['pATHRESULT'], exist_ok=True)
    for p_name, p_path in derived.items():
        if not p_path.lower().endswith(('.txt', '.csv')):
            os.makedirs(p_path, exist_ok=True)
        else:
            os.makedirs(os.path.dirname(p_path), exist_ok=True)
    os.makedirs(os.path.join(paths['pATHRESULT'], 'cell_centroids'), exist_ok=True)

    # 4. 加载 XML
    xml_name = 'xml_merging.xml' if os.path.isfile(os.path.join(anchor_dir, 'xml_merging.xml')) else 'xml_import.xml'
    pATHxml = os.path.join(anchor_dir, xml_name)

    if not os.path.exists(pATHxml):
        raise FileNotFoundError(f"❌ 找不到拼接坐标文件！确保 {anchor_dir} 存在 {xml_name}")

    tile_size = dp.get('tILESIZE', 2048)
    dir_dict, H, W, Z, z_start, disp_mat_fin = loadTeraxml(pATHxml, tile_size)

    # ==========================================
    # 阶段 2: 线性 Checkpoint - Tile 级别检测 (无 QC)
    # ==========================================
    tasks_to_run = []
    for i, path in enumerate(pATHTILE):
        tile_name = os.path.split(path)[-1]
        csv_tile = os.path.join(derived['pATH_DET_RES'], f"{tile_name}_result.csv")
        
        if not os.path.exists(csv_tile):
            tasks_to_run.append((i, path, config))

    if tasks_to_run:
        logging.info(f"阶段 2: 发现 {len(tasks_to_run)} 个缺失结果，开始多进程独立推理...")
        num_processes = 4
        print("\n" * (num_processes + 2))
        with mp.Pool(processes=num_processes) as pool:
            with tqdm(total=len(tasks_to_run), desc="Tile Processing", position=0, leave=True) as pbar:
                for _ in pool.imap_unordered(process_single_tile_wrapper, tasks_to_run):
                    pbar.update(1)
    else:
        logging.info("✔️ Checkpoint 1 达成: 所有 Tile 检测完成。")

    # ==========================================
    # 阶段 3: 线性 Checkpoint - 全局拼接与 Z-Linker
    # ==========================================
    bbox_path = os.path.join(paths['pATHRESULT'], "global_bboxes.csv")
    final_results = None

    if os.path.exists(bbox_path):
        logging.info(f"✔️ Checkpoint 2 达成: 加载已有的全局检测结果 {bbox_path}")
        df_boxes = pd.read_csv(bbox_path)
        final_results = df_boxes[["x1", "y1", "x2", "y2", "score", "mean", "class", "z"]].values
        if final_results.ndim == 1:
            final_results = final_results.reshape(1, -1)
    else:
        logging.info("阶段 3: 开始合并 Tile 并运行 Z-Linker...")
        all_raw_detections = []
        metadata_registry = []
        num_tiles = len(pATHTILE_all)
        
        if num_tiles > 1:
            # 兼容旧 stitcher：初始化双类别矩阵 (Glia, Neuron)
            stitched_predictions = [[np.empty((0, 8)) for _ in range(2)] for _ in range(Z)]
            for dir_name in dir_dict:
                tile_name = os.path.split(dir_name)[-1]
                csv_tile = os.path.join(derived['pATH_DET_RES'], f"{tile_name}_result.csv")
                if os.path.isfile(csv_tile):
                    with open(csv_tile, newline='', encoding='utf-8') as tile_file:
                        csv_reader = csv.reader(tile_file)
                        # 注意：classes 参数传入 None 即可，stitcher 已经自己解析了
                        stitched_predictions = combine_predictions(
                            stitched_predictions, csv_reader, None, z_start, Z, 
                            dir_dict[dir_name], disp_mat_fin, (H, W), metadata_registry, tile_name, tILESIZE=2048
                        )
            for layer in stitched_predictions:
                for group in layer:
                    if group.size > 0: all_raw_detections.append(group)
        else:
            # 单 Tile 模式修正
            csv_list = [f for f in os.listdir(derived['pATH_DET_RES']) if f.endswith('_result.csv')]
            if csv_list:
                tile_name = csv_list[0].replace('_result.csv', '')
                with open(os.path.join(derived['pATH_DET_RES'], csv_list[0]), 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    temp_list = []
                    for r in reader:
                        if len(r) > 8:
                            slice_name = r[0]
                            # 核心修改：保持 c 为字符串，不再转 index
                            x1, y1, x2, y2, score, mean = map(float, r[1:5] + [r[6], r[7]])
                            c = str(r[5])
                            z = int(float(r[8]))
                            temp_list.append([x1, y1, x2, y2, score, mean, c, z])
                            
                            cx, cy = (x1+x2)/2, (y1+y2)/2
                            metadata_registry.append([cx, cy, z, tile_name, slice_name])
                    if temp_list: 
                        all_raw_detections.append(np.array(temp_list, dtype=object)) # 必须指定 object
        
        if len(all_raw_detections) > 0:
            full_stack_matrix = np.concatenate(all_raw_detections, axis=0)
            # 增加对 ENABLE_Z_LINKER 的判断
            if config.get('ENABLE_Z_LINKER', True) and dp.get('mERGEZ', True):
                logging.info(f"正在运行动态 Z-Linker (输入: {len(full_stack_matrix)} 个 2D 切片框)...")
                final_results = run_z_linker(full_stack_matrix, iou_thresh=0.35, max_gap=1, min_z_layers=2, max_cell_z_span=dp.get('z_distance_limit', 5))
            else:
                logging.info("⚠️ Z-Linker 已跳过（未开启 ENABLE_Z_LINKER 或采样步长过大）。")
                final_results = full_stack_matrix
            
            # Pandas 导出字符串表格
            df = pd.DataFrame(final_results, columns=["x1", "y1", "x2", "y2", "score", "mean", "class", "z"])
            
            if len(metadata_registry) > 0 and len(final_results) > 0:
                meta_np = np.array(metadata_registry, dtype=object)
                meta_coords = meta_np[:, :3].astype(float)
                meta_coords[:, 2] *= 10.0
                tree = cKDTree(meta_coords)
                
                final_coords = np.column_stack((
                    (final_results[:, 0] + final_results[:, 2]) / 2,
                    (final_results[:, 1] + final_results[:, 3]) / 2,
                    final_results[:, 7].astype(float) * 10.0
                ))
                
                _, indices = tree.query(final_coords)
                df['tile_name'] = meta_np[indices, 3]
                df['slice_name'] = meta_np[indices, 4]
            else:
                df['tile_name'] = 'Unknown'
                df['slice_name'] = 'Unknown'
            
            df.to_csv(bbox_path, index=False)
            logging.info(f"阶段 3 完成: 生成了 {len(final_results)} 个 3D 细胞记录。")
        else:
            logging.warning("⚠️ 警告：全局未检测到任何目标。")

    # ==========================================
    # 阶段 4: 生成分析级的统计报告与质心
    # ==========================================
    report_path = os.path.join(paths['pATHRESULT'], "global_summary_statistics.csv")

    if os.path.exists(report_path):
        logging.info("✔️ Checkpoint 3 达成: 全局统计报告已存在。")
    elif final_results is not None and len(final_results) > 0:
        logging.info("阶段 4: 开始生成动态标签质心文件与统计报告...")
        
        df_final = pd.read_csv(bbox_path)
        total_cells = len(df_final)

        # 1. 拆解分析动态标签 (例如把 "neuron_RFP_Sox9" 拆成类别和具体 Marker)
        df_final['base_type'] = df_final['class'].apply(lambda x: x.split('_')[0])
        
        # 统计组合情况 (e.g. neuron_RFP_Sox9: 150个)
        combo_counts = df_final['class'].value_counts()
        
        # 统计基类情况 (e.g. neuron: 800个, glia: 1200个)
        base_counts = df_final['base_type'].value_counts()
        
        # 提取所有的 Markers 并独立统计阳性率
        all_markers_found = set()
        for c in df_final['class'].unique():
            parts = str(c).split('_')
            if len(parts) > 1:
                all_markers_found.update(parts[1:])
                
        marker_counts = {}
        for m in all_markers_found:
            # 只要包含该 Marker 字符即算阳性
            marker_counts[m] = df_final[df_final['class'].str.contains(m)].shape[0]

        # 2. 写入极其详细的层级分析报告
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== Base Cell Type (基础细胞类型) ===\n")
            f.write("Type,Count,Percentage(%)\n")
            for t, count in base_counts.items():
                f.write(f"{t},{count},{count / total_cells * 100:.2f}%\n")
            
            f.write("\n=== Subtypes & Colocalization (具体组合) ===\n")
            f.write("Subtype,Count,Percentage(%)\n")
            for sub, count in combo_counts.items():
                f.write(f"{sub},{count},{count / total_cells * 100:.2f}%\n")
                
            f.write("\n=== Single Marker Positivity (各标记物全局阳性率) ===\n")
            f.write("Marker,Count,Percentage(%)\n")
            for m, count in marker_counts.items():
                f.write(f"{m},{count},{count / total_cells * 100:.2f}%\n")

        # 3. 按最终组合输出质心 (用下划线替代特殊字符保证文件名合法)
        df_final['cx'] = (df_final['x1'] + df_final['x2']) / 2
        df_final['cy'] = (df_final['y1'] + df_final['y2']) / 2
        
        for label, group in df_final.groupby('class'):
            group_sorted = group.sort_values('z')
            out_df = group_sorted[['cx', 'cy', 'z', 'score', 'slice_name', 'tile_name']]
            
            # 净化文件名 (如 neuron_RFP_Sox9 -> ob_neuron_RFP_Sox9.csv)
            safe_label = str(label).replace('/', '_').replace('\\', '_')
            save_path = os.path.join(derived['pATH_CENTROIDS'], f"ob_{safe_label}.csv")
            out_df.to_csv(save_path, index=False)
            
        logging.info(f"已生成所有 {len(combo_counts)} 种子类型的质心文件，保存在: {derived['pATH_CENTROIDS']}")
        logging.info("阶段 4 完成: 全局统计报告生成完毕。")

    # (删除了原先的 Stage 5: QC 报告环节)

    logging.info(f"🎉 动态多通道推断全部完成！总耗时: {(time.time() - start_time)/60:.2f} 分钟。")