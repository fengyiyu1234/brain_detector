# -*- coding: utf-8 -*-
"""
主程序入口：MADM 细胞检测与拼接
"""

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
from src.analysis.qc import generate_global_summary

if __name__ == '__main__':
    start_time = time.time()
    
    # ==========================================
    # 阶段 1: 准备工作 (配置读取、路径创建)
    # ==========================================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device.upper()}")

    # 1. 加载配置
    config = load_config('D:/Fengyi/brain_detector/config/config.json')
    config['device'] = device

    config['labels_to_names'] = {int(k): v for k, v in config['labels_to_names'].items()}
    config['colors_map'] = {int(k): v for k, v in config['colors_map'].items()}
    config['type_map'] = {int(k): v for k, v in config['type_map'].items()}

    paths = config['paths']
    dp = config['detection_params']
    derived = config['derived_paths']
    classes = list(config['labels_to_names'].values())
    num_class = len(classes)

    # ------------------------------------------
    # 🔴 新增：严密的输入路径校验 (Path Validation)
    # ------------------------------------------
    c1_dir = paths.get('channel1_dir')
    c2_dir = paths.get('channel2_dir')
    
    if not c1_dir or not os.path.exists(c1_dir):
        raise FileNotFoundError(f"❌ Channel 1 路径不存在或未配置！请检查: {c1_dir}")
        
    if not c2_dir or not os.path.exists(c2_dir):
        print(f"⚠️ 警告：Channel 2 路径不存在 ({c2_dir})。程序将退化为单通道检测模式。")
    # ------------------------------------------

    # 2. 获取 Tile 列表
    dirnames, pATHTILE_all = listTile(c1_dir)
    if not pATHTILE_all:
        raise ValueError(f"❌ 在 {c1_dir} 中没有找到任何合法的 Tile 文件夹！")

    setup_logging(paths.get('pATHRESULT', '')) 
    logging.info("starting...")
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

    # 4. 加载 XML (无论断点在哪，这部分轻量数据都必须加载供后续使用)
    xml_name = 'xml_merging.xml' if os.path.isfile(os.path.join(c1_dir, 'xml_merging.xml')) else 'xml_import.xml'
    pATHxml = os.path.join(c1_dir, xml_name)

    if not os.path.exists(pATHxml):
        raise FileNotFoundError(f"❌ 找不到 TeraStitcher 的拼接坐标文件！请确保 {c1_dir} 目录下存在 xml_merging.xml 或 xml_import.xml")

    tile_size = dp.get('tILESIZE', 2048)
    dir_dict, H, W, Z, z_start, disp_mat_fin = loadTeraxml(pATHxml, tile_size)
    
    run_qc_flag = dp.get('rUN_QC', True)

    if dp.get('dOWNSAMPLE_Z_2X', False):
        logging.info(f"Global Flag Detected: Downsampling Z-axis (Old Z={Z})")
        Z = (Z + 1) // 2 
        logging.info(f"New Z-axis depth: {Z}")


    # ==========================================
    # 🔴 阶段 2: 线性 Checkpoint - Tile 级别检测
    # ==========================================
    tasks_to_run = []
    for i, path in enumerate(pATHTILE):
        tile_name = os.path.split(path)[-1]
        csv_tile = os.path.join(derived['pATH_DET_RES'], f"{tile_name}_result.csv")
        qc_tile = os.path.join(derived['pATH_DET_RES'], f"{tile_name}_qc_metrics.csv")

        is_done = (os.path.exists(csv_tile) and os.path.exists(qc_tile)) if run_qc_flag else os.path.exists(csv_tile)
        
        if not is_done:
            tasks_to_run.append((i, path, config))

    if tasks_to_run:
        logging.info(f"阶段 2: 发现 {len(tasks_to_run)} 个缺失或未完成的 Tile，开始多进程推理...")
        num_processes = 9
        print("\n" * (num_processes + 2))
        with mp.Pool(processes=num_processes) as pool:
            with tqdm(total=len(tasks_to_run), desc="Tile Processing", position=0, leave=True) as pbar:
                for _ in pool.imap_unordered(process_single_tile_wrapper, tasks_to_run):
                    pbar.update(1)
    else:
        logging.info("✔️ Checkpoint 1 达成: 所有单个 Tile 的检测与 QC 均已完成，跳过 YOLO 推理。")

    # ==========================================
    # 🔴 阶段 3: 线性 Checkpoint - 全局拼接与 Z-Linker
    # ==========================================
    bbox_path = os.path.join(paths['pATHRESULT'], "global_bboxes.csv")
    final_results = None

    if os.path.exists(bbox_path):
        logging.info(f"✔️ Checkpoint 2 达成: 发现已有的 {bbox_path}，直接加载全局检测结果。")
        # 改用 Pandas 读取，因为现在 CSV 里最后两列是字符串，np.loadtxt 会报错
        df_boxes = pd.read_csv(bbox_path)
        final_results = df_boxes[["x1", "y1", "x2", "y2", "score", "mean", "class", "z"]].values
        if final_results.ndim == 1:
            final_results = final_results.reshape(1, -1)
    else:
        logging.info("阶段 3: 未找到全局结果，开始合并 Tile 并运行 Z-Linker...")
        all_raw_detections = []
        metadata_registry = [] # 用于追踪数据的来源
        num_tiles = len(pATHTILE_all)
        
        if num_tiles > 1:
            stitched_predictions = [[np.empty((0, 8)) for _ in range(2)] for _ in range(Z)]
            for dir_name in dir_dict:
                tile_name = os.path.split(dir_name)[-1]
                csv_tile = os.path.join(derived['pATH_DET_RES'], f"{tile_name}_result.csv")
                if os.path.isfile(csv_tile):
                    with open(csv_tile, newline='') as tile_file:
                        csv_reader = csv.reader(tile_file, delimiter=',', quotechar='|')
                        stitched_predictions = combine_predictions(
                            stitched_predictions, csv_reader, classes, z_start, Z, 
                            dir_dict[dir_name], disp_mat_fin, (H, W), metadata_registry, tile_name, tILESIZE=2048
                        )
            for layer in stitched_predictions:
                for group in layer:
                    if group.size > 0: all_raw_detections.append(group)
        else:
            # 单 Tile 模式
            csv_list = [f for f in os.listdir(derived['pATH_DET_RES']) if f.endswith('_result.csv')]
            if csv_list:
                tile_name = csv_list[0].replace('_result.csv', '')
                with open(os.path.join(derived['pATH_DET_RES'], csv_list[0]), 'r') as f:
                    reader = csv.reader(f, delimiter=',', quotechar='|')
                    temp_list = []
                    for r in reader:
                        if len(r) > 8:
                            slice_name = r[0]
                            x1, y1, x2, y2, score, mean, c, z = float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[6]), float(r[7]), classes.index(r[5]), int(float(r[8]))
                            temp_list.append([x1, y1, x2, y2, score, mean, c, z])
                            cx, cy = (x1+x2)/2, (y1+y2)/2
                            metadata_registry.append([cx, cy, z, tile_name, slice_name])
                    if temp_list: all_raw_detections.append(np.array(temp_list))
        
        if len(all_raw_detections) > 0:
            full_stack_matrix = np.concatenate(all_raw_detections, axis=0)
            if dp['mERGEZ']:
                logging.info(f"正在运行 3D Z-Linker (输入: {len(full_stack_matrix)} 个检测框)...")
                final_results = run_z_linker(full_stack_matrix, iou_thresh=dp.get('overlapThresh', 0.3), max_gap=dp.get('z_distance_limit', 1))
            else:
                final_results = full_stack_matrix
            
            # --- 逆向溯源匹配，并用 Pandas 保存包含文本的 CSV ---
            df = pd.DataFrame(final_results, columns=["x1", "y1", "x2", "y2", "score", "mean", "class", "z"])
            
            if len(metadata_registry) > 0 and len(final_results) > 0:
                meta_np = np.array(metadata_registry, dtype=object)
                meta_coords = meta_np[:, :3].astype(float)
                meta_coords[:, 2] *= 10.0 # 增强 Z 轴距离权重，防止不同层间错误匹配
                tree = cKDTree(meta_coords)
                
                final_coords = np.column_stack((
                    (final_results[:, 0] + final_results[:, 2]) / 2,
                    (final_results[:, 1] + final_results[:, 3]) / 2,
                    final_results[:, 7] * 10.0
                ))
                
                # 为最终保留下来的每一个框寻找它的来源标签
                _, indices = tree.query(final_coords)
                df['tile_name'] = meta_np[indices, 3]
                df['slice_name'] = meta_np[indices, 4]
            else:
                df['tile_name'] = 'Unknown'
                df['slice_name'] = 'Unknown'
            
            df.to_csv(bbox_path, index=False)
            logging.info(f"阶段 3 完成: 生成了 {len(final_results)} 个细胞坐标。")
        else:
            logging.warning("警告：在所有 Tile 中均未检测到任何目标。")

    # ==========================================
    # 🔴 阶段 4: 线性 Checkpoint - 生成统计和质心
    # ==========================================
    centroid_path = os.path.join(paths['pATHRESULT'], "global_centroids.csv")
    report_path = os.path.join(paths['pATHRESULT'], "global_summary_statistics.csv")

    if os.path.exists(centroid_path) and os.path.exists(report_path):
        logging.info("✔️ Checkpoint 3 达成: 全局统计和质心文件已存在，跳过重新计算。")
    elif final_results is not None and len(final_results) > 0:
        logging.info("阶段 4: 开始生成质心分类文件与统计报告...")
        
        x_cent = (final_results[:, 0] + final_results[:, 2]) / 2
        y_cent = (final_results[:, 1] + final_results[:, 3]) / 2
        z_vals, scores, clses, means = final_results[:, 7], final_results[:, 4], final_results[:, 6], final_results[:, 5]
        
        global_centroids = np.column_stack((x_cent, y_cent, z_vals, scores, clses, means))
        np.savetxt(centroid_path, global_centroids, delimiter=",", header="x,y,z,score,class,mean", comments='')

        l2n, c_map, t_map = config['labels_to_names'], config['colors_map'], config['type_map']
        total_cells = len(final_results)
        stats_class, stats_color, stats_type = {k: 0 for k in l2n.keys()}, {color: 0 for color in set(c_map.values())}, {t: 0 for t in set(t_map.values())}
        class_specific_data = {k: [] for k in l2n.keys()} 

        for i in range(total_cells):
            label = int(clses[i])
            if label in stats_class: stats_class[label] += 1
            if label in c_map: stats_color[c_map[label]] += 1
            if label in t_map: stats_type[t_map[label]] += 1
            class_specific_data[label].append([x_cent[i], y_cent[i], int(z_vals[i])])

        # 直接读取Stage 3 生成的包含详尽信息的全局坐标文件
        if os.path.exists(bbox_path):
            import pandas as pd
            df_final = pd.read_csv(bbox_path)

            df_final['cx'] = (df_final['x1'] + df_final['x2']) / 2
            df_final['cy'] = (df_final['y1'] + df_final['y2']) / 2
            
            df_final['z'] = df_final['z'].astype(int)
            
            for label, group in df_final.groupby('class'):
                group_sorted = group.sort_values('z')
                out_df = group_sorted[['cx', 'cy', 'z', 'score', 'slice_name', 'tile_name']]
                
                save_path = os.path.join(derived['pATH_CENTROIDS'], f"ob_{int(label)}.csv")
                out_df.to_csv(save_path, index=False)
                
                logging.info(f"已保存带有溯源信息的质心文件: {save_path}")
        else:
            logging.error(f"未找到 {bbox_path}，无法生成质心文件。")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Group,Category,Count,Percentage(%)\n")
            for cid, name in l2n.items():
                f.write(f"Class,{name},{stats_class[cid]},{stats_class[cid] / total_cells * 100:.2f}\n")
            for color, count in stats_color.items():
                f.write(f"Color,{color},{count},{count / total_cells * 100:.2f}\n")
            for t, count in stats_type.items():
                f.write(f"Type,{t},{count},{count / total_cells * 100:.2f}\n")

        logging.info("阶段 4 完成: 统计报告生成完毕。")
    
    # ==========================================
    # 阶段 5:  QC 报告
    # ===========================
    if run_qc_flag:
        try:
            generate_global_summary(derived['pATH_DEfT_RES'])
        except Exception as e:
            logging.error(f"生成全局 QC 报告失败: {e}")

    logging.info(f"🎉 所有任务已完成！总耗时: {(time.time() - start_time)/60:.2f} 分钟。")