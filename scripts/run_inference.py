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
from src.core.worker import process_single_tile_wrapper, init_worker
from src.core.stitcher import combine_predictions
from src.core.z_linker import run_z_linker
from src.core.stitcher import colocalize_3d, colocalize_3d_centroid_in_box, permutation_test_colocalization


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    start_time = time.time()
    # ==========================================
    # 阶段 1: 准备工作 (配置读取、动态路径构建与校验)
    # ==========================================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device.upper()}")

    # 1. 加载基础配置 (此时 load_config 仅作纯粹的 JSON 读取)
    config = load_config('D:/Fengyi/brain_detector/config/config.json')
    config['device'] = device
    paths = config['paths']
    dp = config['detection_params']

    # --- ✨ 核心重构：统一接管派生目录生成，建立 5 层数字化输出结构 ---
    base_res_path = paths.get('pATHRESULT')
    if not base_res_path:
        raise ValueError("❌ 配置文件中缺失 pATHRESULT 输出根目录！")
        
    derived = {}
    derived['pATH_DET_RES'] = os.path.join(base_res_path, "1_tile_2d_raw")
    derived['pATH_GLOBAL_2D'] = os.path.join(base_res_path, "2_global_2d_raw")
    derived['pATH_GLOBAL_3D_SINGLE'] = os.path.join(base_res_path, "3_global_3d_single")
    derived['pATH_GLOBAL_3D_FINAL'] = os.path.join(base_res_path, "4_global_3d_final")
    derived['pATH_REPORT'] = os.path.join(base_res_path, "5_analysis_report")
    derived['pATH_CENTROIDS'] = os.path.join(derived['pATH_REPORT'], "cell_centroids")

    # 将构建好的字典挂载回 config，供 worker.py 及后续流程使用
    config['derived_paths'] = derived

    # 自动生成所有物理文件夹
    os.makedirs(base_res_path, exist_ok=True)
    for p_key, p_path in derived.items():
        if not p_path.lower().endswith(('.txt', '.csv')):
            os.makedirs(p_path, exist_ok=True)

    # 2. 动态解析路由，确定基准(Anchor)通道
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
        print(f"✔️ [{ch['type'].upper()}] {ch['id']}: {d_path}")
    print("======================\n")

    # 3. 获取 Tile 列表 (基于锚点通道)
    dirnames, pATHTILE_all = listTile(anchor_dir)
    if not pATHTILE_all:
        raise ValueError(f"❌ 在锚点目录 {anchor_dir} 中没有找到合法的 Tile！")

    # 日志绑定到输出根目录
    setup_logging(base_res_path) 
    logging.info("Starting Multi-Channel Inference...")
    logging.info(f"Current device: {config['device']}")
    save_run_metadata(config, start_time)

    # 4. 范围筛选
    sTARTID = dp.get('sTARTID') or 1
    eNDID = dp.get('eNDID') or len(pATHTILE_all)
    target_indices = list(range(sTARTID - 1, eNDID))
    pATHTILE = [pATHTILE_all[i] for i in target_indices]

    # 5. 加载 TeraStitcher XML
    xml_name = 'xml_merging.xml' if os.path.isfile(os.path.join(anchor_dir, 'xml_merging.xml')) else 'xml_import.xml'
    pATHxml = os.path.join(anchor_dir, xml_name)

    if not os.path.exists(pATHxml):
        raise FileNotFoundError(f"❌ 找不到拼接坐标文件！确保 {anchor_dir} 存在 {xml_name}")

    tile_size = dp.get('tILESIZE', 2048)
    dir_dict, H, W, Z, z_start, disp_mat_fin = loadTeraxml(pATHxml, tile_size)

    # ==========================================
    # 阶段 2: 线性 Checkpoint - Tile 级别检测 
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
        with mp.Pool(processes=num_processes, initializer=init_worker, initargs=(config,)) as pool:
            with tqdm(total=len(tasks_to_run), desc="Tile Processing", position=0, leave=True) as pbar:
                for _ in pool.imap_unordered(process_single_tile_wrapper, tasks_to_run):
                    pbar.update(1)
    else:
        logging.info("✔️ Checkpoint 1 达成: 所有 Tile 检测完成。")

    # ==========================================
    # 阶段 3: 线性 Checkpoint - 全局拼接与 Z-Linker共定位
    # ==========================================
    bbox_path = os.path.join(paths['pATHRESULT'], "global_bboxes.csv") # 目标 4 的最终文件
    final_results = None
    soma_3d = None   # 供阶段 5 置换检验使用（仅本次运行计算时有值，从 checkpoint 加载时为 None）
    tf_3d   = None
    xy_res      = dp.get('xy_resolution_um', 0.65)
    z_res       = dp.get('z_resolution_um', 8.0)
    dist_thresh = dp.get('coloc_distance_um', 15.0)

    if os.path.exists(bbox_path):
        logging.info(f"✔️ Checkpoint 2 达成: 加载已有的全局检测结果 {bbox_path}")
        df_boxes = pd.read_csv(bbox_path)
        final_results = df_boxes[["x1", "y1", "x2", "y2", "score", "mean", "class", "z"]].values
        if final_results.ndim == 1:
            final_results = final_results.reshape(1, -1)
    else:
        logging.info("阶段 3: 开始合并 Tile 并运行 Z-Linker (先独立 3D 追踪，再 3D 共定位)...")
        
        routing_config = [ch for ch in config.get('channels_routing', []) if ch.get('active', True)]
        
        soma_raw_detections = []
        tf_raw_detections = []
        metadata_registry = []
        num_tiles = len(pATHTILE_all)
        
        # ====== 1. 遍历每个通道，独立进行 2D 全局边缘拼接 ======
        for ch in routing_config:
            ch_id = ch['id']
            ch_type = ch.get('type', 'soma')
            logging.info(f" -> 正在拼接通道 2D 框: [{ch_id}] (类型: {ch_type})")
            
            current_ch_global_2d = []
            
            if num_tiles > 1:
                # 每个通道重新初始化拼接矩阵，避免跨通道干扰
                stitched_predictions = [[np.empty((0, 8)) for _ in range(2)] for _ in range(Z)]
                for dir_name in dir_dict:
                    tile_name = os.path.split(dir_name)[-1]
                    # 读取目标 1 的结果
                    csv_tile = os.path.join(derived['pATH_DET_RES'], f"{tile_name}_{ch_id}_result.csv")
                    
                    if os.path.isfile(csv_tile):
                        with open(csv_tile, newline='', encoding='utf-8') as tile_file:
                            next(tile_file, None) # 跳过表头
                            csv_reader = csv.reader(tile_file)
                            # 自动映射偏移量去重
                            stitched_predictions = combine_predictions(
                                stitched_predictions, csv_reader, None, z_start, Z, 
                                dir_dict[dir_name], disp_mat_fin, (H, W), metadata_registry, tile_name, tILESIZE=tile_size
                            )
                
                # 展平该通道的全局结果
                for layer in stitched_predictions:
                    for group in layer:
                        if group.size > 0: current_ch_global_2d.append(group)
            else:
                # 单 Tile 处理
                csv_list = [f for f in os.listdir(derived['pATH_DET_RES']) if f.endswith(f'_{ch_id}_result.csv')]
                if csv_list:
                    tile_name = csv_list[0].split(f"_{ch_id}_")[0]
                    with open(os.path.join(derived['pATH_DET_RES'], csv_list[0]), 'r', encoding='utf-8') as f:
                        next(f, None) # 跳过表头
                        reader = csv.reader(f)
                        temp_list = []
                        for r in reader:
                            if len(r) > 8 and r[0] != 'tile_id':
                                slice_name = r[0]
                                x1, y1, x2, y2, score, mean = map(float, r[1:5] + [r[6], r[7]])
                                c = str(r[5])
                                z = int(float(r[8]))
                                temp_list.append([x1, y1, x2, y2, score, mean, c, z])
                                
                                cx, cy = (x1+x2)/2, (y1+y2)/2
                                metadata_registry.append([cx, cy, z, tile_name, slice_name])
                        if temp_list:
                            current_ch_global_2d.append(np.array(temp_list, dtype=object))
            
            # 给这批全局 2D 框打上 Marker 并按类型分流
            if len(current_ch_global_2d) > 0:
                ch_matrix = np.concatenate(current_ch_global_2d, axis=0)
                for row in ch_matrix:
                    original_class = str(row[6])
                    row[6] = f"{original_class}_{ch_id}" # e.g. "neuron" -> "neuron_RFP"
                    
                    if ch_type == 'tf':
                        tf_raw_detections.append(row)
                    else:
                        soma_raw_detections.append(row)

        # 转换为 Numpy 矩阵
        soma_matrix = np.array(soma_raw_detections, dtype=object) if soma_raw_detections else np.empty((0, 8), dtype=object)
        tf_matrix = np.array(tf_raw_detections, dtype=object) if tf_raw_detections else np.empty((0, 8), dtype=object)

        # 导出全局 2D 原始坐标 (Z-Linker 之前)
        global_2d_raw_path = os.path.join(paths['pATHRESULT'], "global_2D_raw.csv")
        global_2d_all = np.concatenate((soma_matrix, tf_matrix), axis=0) if len(soma_matrix) > 0 and len(tf_matrix) > 0 else (soma_matrix if len(soma_matrix) > 0 else tf_matrix)
        if len(global_2d_all) > 0:
            pd.DataFrame(global_2d_all, columns=["x1", "y1", "x2", "y2", "score", "mean", "class", "z"]).to_csv(global_2d_raw_path, index=False)
            logging.info(f"✔️ 已输出 目标2 (全局2D 原始): {global_2d_raw_path}")

        if config.get('ENABLE_Z_LINKER', True):
            logging.info(f"正在按 Config Type 分发 Z-Linker (先进行 3D 追踪)...")
            
            soma_3d = np.empty((0, 8), dtype=object)
            tf_3d = np.empty((0, 8), dtype=object)
            
            if len(soma_matrix) > 0:
                soma_3d = run_z_linker(
                    soma_matrix, 
                    iou_thresh=0.35, 
                    max_gap=dp.get('max_gap', 0), 
                    min_z_layers=2, # Soma 至少需出现 2 层
                    max_cell_z_span=dp.get('z_distance_limit', 5)
                )
            
            if len(tf_matrix) > 0:
                tf_3d = run_z_linker(
                    tf_matrix, 
                    iou_thresh=0.25, 
                    max_gap=dp.get('max_gap', 0), 
                    min_z_layers=1,
                    max_cell_z_span=3
                )
                
            logging.info(f"✔️ Z-Linker 追踪完毕: {len(soma_3d)} 个 3D 胞体，{len(tf_3d)} 个 3D 核/TF。")

            #导出全局 3D 单通道追踪坐标 (Colocalize 之前) 
            global_3d_single_path = os.path.join(paths['pATHRESULT'], "global_3D_single_channel.csv")
            global_3d_single_all = np.concatenate((soma_3d, tf_3d), axis=0) if len(soma_3d) > 0 and len(tf_3d) > 0 else (soma_3d if len(soma_3d) > 0 else tf_3d)
            if len(global_3d_single_all) > 0:
                pd.DataFrame(global_3d_single_all, columns=["x1", "y1", "x2", "y2", "score", "mean", "class", "z"]).to_csv(global_3d_single_path, index=False)
                logging.info(f"✔️ 已输出 目标3 (全局3D 单通道): {global_3d_single_path}")

            # ====== 3. 3D 物理空间共定位 ======
            logging.info("开始执行 3D 物理空间共定位...")
            use_centroid_box = dp.get('coloc_use_centroid_box', False)
            if use_centroid_box:
                xy_tol = dp.get('coloc_xy_tolerance_px', 5)
                z_tol  = dp.get('z_distance_limit', 5)
                final_results = colocalize_3d_centroid_in_box(
                    soma_3d, tf_3d,
                    xy_res=xy_res, z_res=z_res,
                    xy_tolerance_px=xy_tol,
                    z_tolerance_slices=z_tol,
                )
                logging.info(f"共定位方法: centroid-in-box (xy_tol=±{xy_tol}px, z_tol=±{z_tol} slices)")
            else:
                final_results = colocalize_3d(soma_3d, tf_3d, xy_res=xy_res, z_res=z_res, distance_thresh_um=dist_thresh)
                logging.info(f"共定位方法: 距离球 (radius={dist_thresh}μm)")
            
        else:
            logging.info("⚠️ Z-Linker 已跳过，合并裸数据。")
            final_results = np.concatenate((soma_matrix, tf_matrix), axis=0) if len(soma_matrix) > 0 else tf_matrix
        
        # ====== 4. 保存全局 3D 报告 (目标 4) ======
        if final_results is not None and len(final_results) > 0:
            df = pd.DataFrame(final_results, columns=["x1", "y1", "x2", "y2", "score", "mean", "class", "z"])
            
            if len(metadata_registry) > 0:
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
            
            df.to_csv(bbox_path, index=False) # 保存目标 4: global_bboxes.csv
            logging.info(f"✔️ 已输出 目标4 (全局3D 共定位): {bbox_path}")
        else:
            logging.warning("⚠️ 全局未检测到任何 3D 目标。")

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

    # ==========================================
    # 阶段 5: 共定位置换检验 (统计显著性)
    # ==========================================
    perm_path = os.path.join(derived['pATH_REPORT'], "colocalization_significance.csv")

    if os.path.exists(perm_path):
        logging.info("✔️ Checkpoint 4 达成: 置换检验结果已存在。")
    elif soma_3d is not None and tf_3d is not None and len(soma_3d) > 0 and len(tf_3d) > 0:
        n_perm = dp.get('n_permutations', 1000)
        logging.info(f"阶段 5: 开始共定位置换检验 (n={n_perm})，请稍候...")
        perm_df = permutation_test_colocalization(
            soma_3d, tf_3d,
            xy_res=xy_res, z_res=z_res,
            distance_thresh_um=dist_thresh,
            n_permutations=n_perm,
            max_soma_sample=dp.get('max_soma_sample', 50_000),
        )
        if len(perm_df) > 0:
            perm_df.to_csv(perm_path, index=False)
            logging.info(f"✔️ 置换检验完成，结果保存至: {perm_path}")
            for _, row in perm_df.iterrows():
                sig = "显著" if row['significant_p05'] else "不显著"
                logging.info(
                    f"  [{row['tf_marker']}] 实际={row['pct_coloc_actual']}%  随机均值={row['pct_random_mean']}%"
                    f"  z={row['z_score']}  p={row['p_value']}  {sig}"
                )
    else:
        logging.warning(
            "⚠️ 置换检验跳过：soma_3d/tf_3d 不在本次内存中。"
            "如需重新运行，请删除 global_bboxes.csv 以跳过 Checkpoint 2。"
        )

    logging.info(f"🎉 动态多通道推断全部完成！总耗时: {(time.time() - start_time)/60:.2f} 分钟。")