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

from src.config.loader import load_config
from src.utils.logger import setup_logging
from src.utils.io import listTile, loadTeraxml, save_run_metadata
from src.core.worker import process_single_tile_wrapper
from src.core.stitcher import combine_predictions
from src.core.z_linker import run_z_linker
from src.visualization.plotter import save_visualization_samples
from src.analysis.qc import generate_global_summary

if __name__ == '__main__':
    start_time = time.time()
    
    # ==========================================
    # 阶段 1: 准备工作 (配置读取、路径创建、断点续传检查)
    # ==========================================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device.upper()}")

    # 1. 加载配置
    config = load_config('D:/Fengyi/brain_detector/config/config.json')
    config['device'] = device

    # 修复 JSON 中 key 变为 string 的问题
    config['labels_to_names'] = {int(k): v for k, v in config['labels_to_names'].items()}
    config['colors_map'] = {int(k): v for k, v in config['colors_map'].items()}
    config['type_map'] = {int(k): v for k, v in config['type_map'].items()}

    paths = config['paths']
    dp = config['detection_params']
    derived = config['derived_paths']
    classes = list(config['labels_to_names'].values())
    num_class = len(classes)
    
    # 2. 获取 Tile 列表
    dirnames, pATHTILE_all = listTile(paths['channel1_dir'])
    
    # 3. 设置项目日志
    # 💡 提示: 如果您希望日志保存到结果文件夹，请确保您的 setup_logging 支持传入路径
    setup_logging(paths.get('pATHRESULT', '')) 
    
    logging.info("starting...")
    logging.info(f"current device: {config['device']}")
    save_run_metadata(config, start_time)
    logging.info("parameters saved")

    # 4. 范围筛选 (sTARTID, eNDID)
    sTARTID = dp.get('sTARTID') or 1
    eNDID = dp.get('eNDID') or len(pATHTILE_all)
    target_indices = list(range(sTARTID - 1, eNDID))
    pATHTILE = [pATHTILE_all[i] for i in target_indices]
    
    os.makedirs(paths['pATHRESULT'], exist_ok=True)
    
    # 5. 统一处理子目录
    for p_name, p_path in derived.items():
        if not p_path.lower().endswith(('.txt', '.csv')):
            os.makedirs(p_path, exist_ok=True)
            print(f"Directory ready: {p_name} -> {p_path}")
        else:
            os.makedirs(os.path.dirname(p_path), exist_ok=True)

    os.makedirs(os.path.join(paths['pATHRESULT'], 'cell_centroids'), exist_ok=True)
    
    # 6. 断点续传检查
    bbox_path = os.path.join(paths['pATHRESULT'], "global_bboxes.csv")
    centroid_path = os.path.join(paths['pATHRESULT'], "global_centroids.csv")
    report_path = os.path.join(paths['pATHRESULT'], "global_summary_statistics.csv")

    global_detection_done = all(os.path.exists(p) for p in [bbox_path, centroid_path, report_path])

    # 无论是否分析完成，如果需要拼接，都要加载 XML
    xml_name = 'xml_merging.xml' if os.path.isfile(os.path.join(paths['channel1_dir'], 'xml_merging.xml')) else 'xml_import.xml'
    pATHxml = os.path.join(paths['channel1_dir'], xml_name)
    tile_size = dp.get('tILESIZE', 2048)
    dir_dict, H, W, Z, z_start, disp_mat_fin = loadTeraxml(pATHxml,tile_size)
    
    # 判断是否所有 Tile 都完成了 QC
    missing_qc_tiles = []
    for i, path in enumerate(pATHTILE):
        tile_name = os.path.split(path)[-1]
        qc_path = os.path.join(derived['pATH_DET_RES'], f"{tile_name}_qc_metrics.csv")
        if not os.path.exists(qc_path):
            missing_qc_tiles.append((i, path, config))

    qc_needs_run = len(missing_qc_tiles) > 0

    # 处理 Z 轴降采样标志
    if dp.get('dOWNSAMPLE_Z_2X', False):
        logging.info(f"Global Flag Detected: Downsampling Z-axis (Old Z={Z})")
        Z = (Z + 1) // 2 
        logging.info(f"New Z-axis depth: {Z}")

    # ==========================================
    # 阶段 2: 核心多进程并行检测与 QC
    # ==========================================
    
    # 任务分发逻辑
    if global_detection_done and not qc_needs_run:
        logging.info("Checkpoint: 全局检测与 QC 均已完成，直接加载数据。")
        final_results = np.loadtxt(bbox_path, delimiter=",", skiprows=1)

    else:
        if global_detection_done and qc_needs_run:
            logging.info(f"Checkpoint: 检测已完成，但缺少 {len(missing_qc_tiles)} 个 QC 报告。准备补跑 QC...")
            tasks_to_run = missing_qc_tiles
        else:
            all_tasks = [(i, path, config) for i, path in enumerate(pATHTILE)]
            tasks_to_run = []
            for i, path, cfg in all_tasks:
                tile_name = os.path.split(path)[-1]
                csv_tile = os.path.join(derived['pATH_DET_RES'], f"{tile_name}_result.csv")
                qc_tile = os.path.join(derived['pATH_DET_RES'], f"{tile_name}_qc_metrics.csv")
                
                if os.path.exists(csv_tile) and os.path.exists(qc_tile):
                    # 🔴 统一改用 logging.info
                    logging.info(f"Checkpoint: Tile {i} ({tile_name}) 检测+QC均完成，跳过。")
                else:
                    tasks_to_run.append((i, path, cfg))
    
        # 执行并行任务
        if len(tasks_to_run) > 0:
            num_processes = 20
            print("\n" * (num_processes + 2))

            with mp.Pool(processes=num_processes) as pool:
                results = []
                with tqdm(total=len(tasks_to_run), desc="Overall Progress", position=0, leave=True) as pbar:
                    for res in pool.imap_unordered(process_single_tile_wrapper, tasks_to_run):
                        results.append(res)
                        pbar.update(1)
        else:
            # 🔴 统一改用 logging.info
            logging.info("所有 Tile 均已存在检测结果，直接进入合并阶段。")

        # ==========================================
        # 阶段 3: 拼合阶段：转换坐标与 Z-Merge
        # ==========================================
        all_raw_detections = []
        
        if global_detection_done:
            logging.info("加载现有的全局检测结果...")
            final_results = np.loadtxt(bbox_path, delimiter=",", skiprows=1)
        else:
            num_tiles = len(pATHTILE_all)
            
            if num_tiles > 1:
                logging.info("Multi-tile mode detected. Loading XML and stitching...")
                stitched_predictions = [[np.empty((0, 8)) for _ in range(2)] for _ in range(Z)]
                
                for dir_name in dir_dict:
                    tile_name = os.path.split(dir_name)[-1]
                    csv_tile = os.path.join(derived['pATH_DET_RES'], f"{tile_name}_result.csv")
                    if os.path.isfile(csv_tile):
                        with open(csv_tile, newline='') as tile_file:
                            csv_reader = csv.reader(tile_file, delimiter=',', quotechar='|')
                            stitched_predictions = combine_predictions(
                                stitched_predictions, csv_reader, classes, z_start, Z, 
                                dir_dict[dir_name], disp_mat_fin, (H, W), tILESIZE=2048
                            )
                
                for layer in stitched_predictions:
                    for group in layer:
                        if group.size > 0: all_raw_detections.append(group)
            else:
                csv_list = [f for f in os.listdir(derived['pATH_DET_RES']) if f.endswith('_result.csv')]
                if csv_list:
                    csv_path = os.path.join(derived['pATH_DET_RES'], csv_list[0])
                    with open(csv_path, 'r') as f:
                        reader = csv.reader(f, delimiter=',', quotechar='|')
                        temp_list = [[float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[6]), float(r[7]), classes.index(r[5]), int(float(r[8]))] for r in reader if len(r)>5]
                        if temp_list: all_raw_detections.append(np.array(temp_list))
        
        # ==========================================
        # 阶段 4: 3D 关联与结果保存 (Z-Linker)
        # ==========================================
        if len(all_raw_detections) > 0:
            full_stack_matrix = np.concatenate(all_raw_detections, axis=0)
            
            if dp['mERGEZ']:
                logging.info(f"Global 3D Z-Linker Starting... (Input: {len(full_stack_matrix)} entries)")
                final_results = run_z_linker(
                    full_stack_matrix, 
                    iou_thresh=dp.get('overlapThresh', 0.3),
                    max_gap=dp.get('z_distance_limit', 1)
                )
                logging.info(f"Z-Linker Finished. Output: {len(final_results)} cells.")
            else:
                final_results = full_stack_matrix

            # 1. 输出全量 Bounding Box 文件
            bbox_path = os.path.join(paths['pATHRESULT'], "global_bboxes.csv")
            np.savetxt(bbox_path, final_results, delimiter=",", 
                    header="x1,y1,x2,y2,score,mean,class,z", comments='')

            # 2. 输出全量 Centroid (质心) 文件
            centroid_path = os.path.join(paths['pATHRESULT'], "global_centroids.csv")
            
            x_cent = (final_results[:, 0] + final_results[:, 2]) / 2
            y_cent = (final_results[:, 1] + final_results[:, 3]) / 2
            z_vals = final_results[:, 7]
            scores = final_results[:, 4]
            clses  = final_results[:, 6]
            means  = final_results[:, 5]
            
            global_centroids = np.column_stack((x_cent, y_cent, z_vals, scores, clses, means))
            np.savetxt(centroid_path, global_centroids, delimiter=",", 
                    header="x,y,z,score,class,mean", comments='')

            # 3. 全局统计
            l2n = config['labels_to_names']
            c_map = config['colors_map']
            t_map = config['type_map']
            total_cells = len(final_results)
            
            stats_class = {k: 0 for k in l2n.keys()}
            stats_color = {color: 0 for color in set(c_map.values())}
            stats_type = {t: 0 for t in set(t_map.values())}
            class_specific_data = {k: [] for k in l2n.keys()} 

            for i in range(total_cells):
                label = int(clses[i])
                if label in stats_class: stats_class[label] += 1
                if label in c_map: stats_color[c_map[label]] += 1
                if label in t_map: stats_type[t_map[label]] += 1
                class_specific_data[label].append([x_cent[i], y_cent[i], int(z_vals[i])])

            for label, data in class_specific_data.items():
                if data:
                    data_np = np.array(data)
                    data_np = data_np[data_np[:, 2].argsort()]
                    cat_csv_path = os.path.join(derived['pATH_CENTROIDS'], f"ob_{label}.csv")
                    np.savetxt(cat_csv_path, data_np, delimiter=",", comments='')

            report_path = os.path.join(paths['pATHRESULT'], "global_summary_statistics.csv")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("Group,Category,Count,Percentage(%)\n")
                print(f"\n{'GROUP':<10} | {'CATEGORY':<15} | {'COUNT':<8} | {'PERCENT'}")
                print("-" * 55)
                
                for cid, name in l2n.items():
                    count = stats_class[cid]
                    perc = (count / total_cells * 100) if total_cells > 0 else 0
                    f.write(f"Class,{name},{count},{perc:.2f}\n")
                    print(f"{'Class':<10} | {name:<15} | {count:<8} | {perc:.2f}%")
                
                for color, count in stats_color.items():
                    perc = (count / total_cells * 100) if total_cells > 0 else 0
                    f.write(f"Color,{color},{count},{perc:.2f}\n")
                
                for t, count in stats_type.items():
                    perc = (count / total_cells * 100) if total_cells > 0 else 0
                    f.write(f"Type,{t},{count},{perc:.2f}\n")

            logging.info(f"Final global results saved to: {paths['pATHRESULT']}")
    
    # ==========================================
    # 阶段 5: 可视化与全量 QC 报告
    # ==========================================
    if dp.get('vISUALIZE', True) and 'final_results' in locals():
        save_visualization_samples(
            final_results, dir_dict, disp_mat_fin, Z, H, W, config
        )
    
    try:
        generate_global_summary(derived['pATH_DET_RES'])
    except Exception as e:
        logging.error(f"生成全局 QC 报告失败: {e}")

    logging.info(f"所有任务已完成！总耗时: {(time.time() - start_time)/60:.2f} 分钟。")