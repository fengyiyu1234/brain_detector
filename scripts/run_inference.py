# -*- coding: utf-8 -*-
import argparse
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
from src.core.stitcher import (match_soma_3d_iou, annotate_soma_with_tf_gmm,
                               permutation_test_colocalization, _merge_class)
from src.core.point_cloud_aligner import (
    build_point_cloud, compute_tile_channel_shifts,
    apply_shift_to_csv, save_tile_offsets,
)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    start_time = time.time()
    # ==========================================
    # 阶段 1: 准备工作 (配置读取、动态路径构建与校验)
    # ==========================================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. 加载基础配置 (此时 load_config 仅作纯粹的 JSON 读取)
    parser = argparse.ArgumentParser(description='Brain detector inference pipeline.')
    parser.add_argument(
        '--config',
        default=os.path.join(project_root, 'config', 'config.json'),
        help='Path to config.json (default: <project_root>/config/config.json)',
    )
    args = parser.parse_args()
    config = load_config(args.config)
    config['device'] = device
    paths = config['paths']
    dp = config['detection_params']

    # --- ✨ 核心重构：统一接管派生目录生成，建立 5 层数字化输出结构 ---
    base_res_path = paths.get('pATHRESULT')
    if not base_res_path:
        raise ValueError("❌ 配置文件中缺失 pATHRESULT 输出根目录！")
        
    pipeline_mode = config.get('pipeline_mode', 'post_align')  # "post_align" | "pre_align"
    pre_align_cfg = config.get('pre_align_params', {})

    derived = {}
    derived['pATH_ALIGN_OFFSETS'] = os.path.join(base_res_path, "0_channel_alignment")
    derived['pATH_DET_RES'] = os.path.join(base_res_path, "1_tile_2d_raw")
    derived['pATH_GLOBAL_2D'] = os.path.join(base_res_path, "2_global_2d_raw")
    derived['pATH_CHANNEL_3D']    = os.path.join(base_res_path, "3_channel_3d")
    derived['pATH_COLOCALIZATION'] = os.path.join(base_res_path, "4_colocalization")
    derived['pATH_REPORT'] = os.path.join(base_res_path, "5_analysis_report")
    derived['pATH_CENTROIDS'] = os.path.join(derived['pATH_REPORT'], "cell_centroids")

    # 将构建好的字典挂载回 config，供 worker.py 及后续流程使用
    config['derived_paths'] = derived

    # 自动生成所有物理文件夹
    os.makedirs(base_res_path, exist_ok=True)
    for p_key, p_path in derived.items():
        if not p_path.lower().endswith(('.txt', '.csv')):
            os.makedirs(p_path, exist_ok=True)

    setup_logging(base_res_path)
    logging.info("Starting Multi-Channel Inference...")
    logging.info(f"Device: {device.upper()}, Pipeline mode: {pipeline_mode.upper()}")

    # 2. 动态解析路由，确定基准(Anchor)通道
    routing_config = [ch for ch in config.get('channels_routing', []) if ch.get('active', True)]
    if not routing_config:
        raise ValueError("❌ 配置文件中没有任何激活的通道路由 (channels_routing)！")
        
    anchor_ch = routing_config[0]
    anchor_dir = paths.get(anchor_ch['dir_key'])

    logging.info("=== 多通道路径检查 ===")
    for ch in routing_config:
        d_path = paths.get(ch['dir_key'])
        if not d_path or not os.path.exists(d_path):
            raise FileNotFoundError(f"❌ 通道 {ch['id']} 路径不存在: {d_path}")
        logging.info(f"[{ch['type'].upper()}] {ch['id']}: {d_path}")
    logging.info("=====================")

    # 3. 获取 Tile 列表 (基于锚点通道)
    dirnames, pATHTILE_all = listTile(anchor_dir)
    if not pATHTILE_all:
        raise ValueError(f"❌ 在锚点目录 {anchor_dir} 中没有找到合法的 Tile！")

    save_run_metadata(config, start_time)

    # 4. 范围筛选
    sTARTID = dp.get('sTARTID') or 1
    eNDID = dp.get('eNDID') or len(pATHTILE_all)
    target_indices = list(range(sTARTID - 1, eNDID))
    pATHTILE = [pATHTILE_all[i] for i in target_indices]

    # 5. 加载 TeraStitcher XML（pre_align 模式下如果不存在则回退到文件名 Grid 推算）
    tile_size = dp.get('tILESIZE', 2048)
    xml_name = 'xml_merging.xml' if os.path.isfile(os.path.join(anchor_dir, 'xml_merging.xml')) else 'xml_import.xml'
    pATHxml = os.path.join(anchor_dir, xml_name)
    has_tera_xml = os.path.exists(pATHxml)

    if has_tera_xml:
        dir_dict, H, W, Z, z_start, disp_mat_fin = loadTeraxml(pATHxml, tile_size)
        logging.info(f"✔️ 已加载 TeraStitcher XML: {pATHxml}")
    elif pipeline_mode == 'pre_align':
        # 回退：从 tile 目录名解析行列号，用均匀 Grid 推算全局偏移
        overlap = pre_align_cfg.get('tile_overlap_pct', 15) / 100.0
        step_px = int(tile_size * (1.0 - overlap))
        logging.warning(f"⚠️ 未找到 TeraStitcher XML（路径: {pATHxml}），pre_align 模式回退到文件名解析全局偏移。")
        dir_dict = {}   # dir_name → (grid_row, grid_col)
        raw_entries = []  # (tile_name, raw_row, raw_col)
        for tile_path in sorted(pATHTILE_all):
            tile_name = os.path.split(tile_path)[-1]
            parts = tile_name.split('_')
            try:
                raw_row = int(parts[0]) if len(parts) >= 2 else 0
                raw_col = int(parts[1]) if len(parts) >= 2 else 0
            except ValueError:
                raw_row, raw_col = 0, 0
            raw_entries.append((tile_name, raw_row, raw_col))
        if raw_entries:
            # 目录名可能是像素偏移坐标而非网格索引，映射为紧凑 0..N-1 索引避免 OOM
            sorted_rows = sorted(set(r for _, r, _ in raw_entries))
            sorted_cols = sorted(set(c for _, _, c in raw_entries))
            row_idx = {v: i for i, v in enumerate(sorted_rows)}
            col_idx = {v: i for i, v in enumerate(sorted_cols)}
            n_row, n_col = len(sorted_rows), len(sorted_cols)
            use_raw_offset = (sorted_rows[-1] > step_px or sorted_cols[-1] > step_px)
            disp_mat_fin = np.zeros((n_row, n_col, 3), dtype=int)
            for tile_name, raw_row, raw_col in raw_entries:
                gi, gj = row_idx[raw_row], col_idx[raw_col]
                dir_dict[tile_name] = (gi, gj)
                ax = raw_col if use_raw_offset else gj * step_px
                ay = raw_row if use_raw_offset else gi * step_px
                disp_mat_fin[gi, gj] = [ax, ay, 0]
            logging.info(f"  解析到 {len(raw_entries)} 个 tile，网格 {n_row}×{n_col}，"
                         f"偏移模式: {'像素坐标' if use_raw_offset else '均匀Grid'}")
        else:
            disp_mat_fin = np.zeros((1, 1, 3), dtype=int)
        H = disp_mat_fin[:, :, 1].max() + tile_size
        W = disp_mat_fin[:, :, 0].max() + tile_size
        Z = len(os.listdir(pATHTILE_all[0])) if pATHTILE_all else 1
        z_start = 0
    else:
        raise FileNotFoundError(f"❌ 找不到拼接坐标文件！确保 {anchor_dir} 存在 {xml_name}")

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
        num_gpus = torch.cuda.device_count()
        num_processes = max(1, num_gpus)
        logging.info(f"阶段 2: 发现 {len(tasks_to_run)} 个缺失结果，启动 {num_processes} 个进程 ({num_gpus} GPU)...")
        gpu_queue = mp.Queue()
        for _i in range(num_gpus):
            gpu_queue.put(_i)
        with mp.Pool(processes=num_processes, initializer=init_worker, initargs=(config, gpu_queue)) as pool:
            with tqdm(total=len(tasks_to_run), desc="Tile Processing", position=0, leave=True) as pbar:
                for _ in pool.imap_unordered(process_single_tile_wrapper, tasks_to_run):
                    pbar.update(1)
    else:
        logging.info("✔️ Checkpoint 1 达成: 所有 Tile 检测完成。")

    # ==========================================
    # 阶段 2.5: 点云通道对齐 (仅 pre_align 模式)
    # ==========================================
    if pipeline_mode == 'pre_align':
        align_done_flag = os.path.join(derived['pATH_ALIGN_OFFSETS'], "_align_done.flag")
        if os.path.exists(align_done_flag):
            logging.info("✔️ Checkpoint 2.5 达成: 通道点云对齐已完成，直接读取对齐结果。")
        else:
            logging.info("阶段 2.5: 开始点云通道对齐 (pre_align 模式)...")
            os.makedirs(derived['pATH_ALIGN_OFFSETS'], exist_ok=True)

            routing_cfg_align = [ch for ch in config.get('channels_routing', []) if ch.get('active', True)]
            soma_ch_ids_align = [ch['id'] for ch in routing_cfg_align if ch.get('type', 'soma') == 'soma']
            tf_ch_ids_align   = [ch['id'] for ch in routing_cfg_align if ch.get('type') == 'tf']

            zl_pre   = config.get('z_linker', {})
            zl_soma  = zl_pre.get('soma', {})
            zl_tf    = zl_pre.get('tf', {})

            pa_sample_z   = pre_align_cfg.get('sample_z_center_count', 50)
            pa_xy_range   = pre_align_cfg.get('xy_search_range_px', 30)
            pa_z_range    = pre_align_cfg.get('z_search_range_slices', 3)
            pa_match_dist = pre_align_cfg.get('match_distance_px', 15)

            for tile_path in tqdm(pATHTILE, desc="Pre-Align Tiles"):
                tile_name = os.path.split(tile_path)[-1]

                # 1. 对每个通道轻量 z-link，得到 per-tile 3D vol_list
                per_ch_vol_lists = {}
                z_counts = []
                for ch in routing_cfg_align:
                    cid = ch['id']
                    ctype = ch.get('type', 'soma')
                    csv_path = os.path.join(derived['pATH_DET_RES'], f"{tile_name}_{cid}_result.csv")
                    if not os.path.isfile(csv_path):
                        per_ch_vol_lists[cid] = []
                        continue
                    df_tile = pd.read_csv(csv_path)
                    if df_tile.empty:
                        per_ch_vol_lists[cid] = []
                        continue

                    mat = df_tile[["x1", "y1", "x2", "y2", "score", "mean", "class", "z"]].values
                    mat[:, 6] = np.array([f"{v}_{cid}" for v in mat[:, 6]])

                    zl_params = zl_soma if ctype == 'soma' else zl_tf
                    _, vol_list = run_z_linker(
                        mat,
                        iou_thresh=zl_params.get('iou_thresh', 0.35),
                        min_z_layers=zl_params.get('min_z_layers', 1),
                        max_cell_z_span=zl_params.get('max_cell_z_span', 5),
                    )
                    per_ch_vol_lists[cid] = vol_list
                    if vol_list:
                        z_counts.extend([c.get('cz', 0) for c in vol_list])

                # 2. 估计 tile z 中心
                z_center = float(np.median(z_counts)) if z_counts else 0.0
                z_half   = pa_sample_z // 2

                # 3. 两步点云对齐
                shifts, scores = compute_tile_channel_shifts(
                    per_ch_vol_lists,
                    soma_ch_ids=soma_ch_ids_align,
                    tf_ch_ids=tf_ch_ids_align,
                    z_center=z_center,
                    z_half_window=z_half,
                    xy_range_px=pa_xy_range,
                    z_range_slices=pa_z_range,
                    match_dist_px=pa_match_dist,
                )

                # 4. 保存偏移 JSON
                save_tile_offsets(tile_name, shifts, scores, derived['pATH_ALIGN_OFFSETS'])

                # 5. 对每个通道的原始 CSV 应用偏移，写入 0_channel_alignment/
                for ch in routing_cfg_align:
                    cid = ch['id']
                    dx, dy, dz = shifts.get(cid, (0, 0, 0))
                    in_csv  = os.path.join(derived['pATH_DET_RES'], f"{tile_name}_{cid}_result.csv")
                    out_csv = os.path.join(derived['pATH_ALIGN_OFFSETS'], f"{tile_name}_{cid}_result.csv")
                    if os.path.isfile(in_csv) and not os.path.isfile(out_csv):
                        apply_shift_to_csv(in_csv, dx, dy, dz, out_csv)

            # 写完成标记
            open(align_done_flag, 'w').close()
            logging.info("✔️ [2.5] 所有 Tile 点云对齐完成。")

    # 决定 Phase 3 读取哪个 CSV 根目录
    pATH_SRC_CSV = (derived['pATH_ALIGN_OFFSETS']
                    if pipeline_mode == 'pre_align'
                    else derived['pATH_DET_RES'])

    # ==========================================
    # 阶段 3: 线性 Checkpoint - 全局拼接与 Z-Linker共定位
    # ==========================================
    bbox_path = os.path.join(derived['pATH_COLOCALIZATION'], "coloc_result.csv")
    final_results = None
    soma_3d = None
    tf_3d   = None
    xy_res  = dp.get('xy_resolution_um', 0.65)
    z_res   = dp.get('z_resolution_um', 8.0)

    if os.path.exists(bbox_path):
        logging.info(f"✔️ Checkpoint 2 达成: 加载已有的全局检测结果 {bbox_path}")
        df_boxes = pd.read_csv(bbox_path)
        final_results = df_boxes[["x1", "y1", "x2", "y2", "score", "mean", "class", "z"]].values
        if final_results.ndim == 1:
            final_results = final_results.reshape(1, -1)
    else:
        logging.info("阶段 3: 开始合并 Tile 并运行 Z-Linker (先独立 3D 追踪，再 3D 共定位)...")

        routing_config = [ch for ch in config.get('channels_routing', []) if ch.get('active', True)]

        per_ch_matrices = {}   # ch_id -> np.array, global 2D with marker applied
        metadata_registry = []
        num_tiles = len(pATHTILE_all)
        BOX_COLS  = ["x1", "y1", "x2", "y2", "score", "mean", "class", "z"]

        # ====== 1. 逐通道独立拼接全局 2D，每个通道单独保存 ======
        for ch in routing_config:
            ch_id   = ch['id']
            ch_type = ch.get('type', 'soma')
            logging.info(f" -> 正在拼接通道 2D 框: [{ch_id}] (类型: {ch_type})")

            current_ch_global_2d = []

            if num_tiles > 1:
                stitched_predictions = [[np.empty((0, 8)) for _ in range(2)] for _ in range(Z)]
                for dir_name in dir_dict:
                    tile_name = os.path.split(dir_name)[-1]
                    csv_tile  = os.path.join(pATH_SRC_CSV, f"{tile_name}_{ch_id}_result.csv")
                    if os.path.isfile(csv_tile):
                        with open(csv_tile, newline='', encoding='utf-8') as tile_file:
                            next(tile_file, None)
                            stitched_predictions = combine_predictions(
                                stitched_predictions, csv.reader(tile_file), None, z_start, Z,
                                dir_dict[dir_name], disp_mat_fin, (H, W),
                                metadata_registry, tile_name, tILESIZE=tile_size
                            )
                for layer in stitched_predictions:
                    for group in layer:
                        if group.size > 0: current_ch_global_2d.append(group)
            else:
                csv_list = [f for f in os.listdir(pATH_SRC_CSV) if f.endswith(f'_{ch_id}_result.csv')]
                if csv_list:
                    tile_name = csv_list[0].split(f"_{ch_id}_")[0]
                    with open(os.path.join(pATH_SRC_CSV, csv_list[0]), 'r', encoding='utf-8') as f:
                        next(f, None)
                        temp_list = []
                        for r in csv.reader(f):
                            if len(r) > 8 and r[0] != 'tile_id':
                                x1, y1, x2, y2 = float(r[1]), float(r[2]), float(r[3]), float(r[4])
                                score, mean, c, z = float(r[6]), float(r[7]), str(r[5]), int(float(r[8]))
                                temp_list.append([x1, y1, x2, y2, score, mean, c, z])
                                metadata_registry.append([(x1+x2)/2, (y1+y2)/2, z, tile_name, r[0]])
                        if temp_list:
                            current_ch_global_2d.append(np.array(temp_list, dtype=object))

            if current_ch_global_2d:
                ch_matrix = np.concatenate(current_ch_global_2d, axis=0)
                for row in ch_matrix:
                    row[6] = f"{row[6]}_{ch_id}"   # "neuron" → "neuron_RFP"
                per_ch_matrices[ch_id] = ch_matrix
                out_2d = os.path.join(derived['pATH_GLOBAL_2D'], f"{ch_id}_2d_global.csv")
                pd.DataFrame(ch_matrix, columns=BOX_COLS).to_csv(out_2d, index=False)
                logging.info(f"✔️ [{ch_id}] 全局2D: {len(ch_matrix)} 框 → {out_2d}")

        # ====== 2. 独立 Z-Link 每个 channel ======
        soma_ch_ids = [ch['id'] for ch in routing_config
                       if ch.get('type', 'soma') == 'soma' and ch.get('active', True)]
        tf_ch_ids   = [ch['id'] for ch in routing_config
                       if ch.get('type', 'tf')   == 'tf'   and ch.get('active', True)]

        zl      = config.get('z_linker', {})
        zl_soma = zl.get('soma', {})
        zl_tf   = zl.get('tf', {})

        soma_vol_by_ch = {}   # ch_id → volumetric_list (for 3D coloc)
        tf_vol_by_ch   = {}

        for cid in soma_ch_ids:
            ch_mat = per_ch_matrices.get(cid)
            if ch_mat is None or len(ch_mat) == 0:
                continue
            summary, vol_list = run_z_linker(
                ch_mat,
                iou_thresh=zl_soma.get('iou_thresh', 0.35),
                min_z_layers=zl_soma.get('min_z_layers', 1),
                max_cell_z_span=zl_soma.get('max_cell_z_span', 5),
            )
            soma_vol_by_ch[cid] = vol_list
            out_3d = os.path.join(derived['pATH_CHANNEL_3D'], f"{cid}_3d_tracked.csv")
            pd.DataFrame(summary, columns=BOX_COLS).to_csv(out_3d, index=False)
            logging.info(f"✔️ [{cid}] Z-Link soma: {len(vol_list)} 个细胞 → {out_3d}")

        for cid in tf_ch_ids:
            ch_mat = per_ch_matrices.get(cid)
            if ch_mat is None or len(ch_mat) == 0:
                continue
            summary, vol_list = run_z_linker(
                ch_mat,
                iou_thresh=zl_tf.get('iou_thresh', 0.25),
                min_z_layers=zl_tf.get('min_z_layers', 1),
                max_cell_z_span=zl_tf.get('max_cell_z_span', 3),
            )
            tf_vol_by_ch[cid] = vol_list
            out_3d = os.path.join(derived['pATH_CHANNEL_3D'], f"{cid}_3d_tracked.csv")
            pd.DataFrame(summary, columns=BOX_COLS).to_csv(out_3d, index=False)
            logging.info(f"✔️ [{cid}] Z-Link TF: {len(vol_list)} 个细胞 → {out_3d}")

        # ====== 3. 3D Colocalization ======
        # Phase A: soma × soma 3D IoU（逐对匹配，依次合并到主列表）
        iou_thresh_3d = zl_soma.get('iou_thresh_3d', 0.15)
        z_pad_3d      = zl_soma.get('z_pad_3d', 2)

        merged_soma_vols = list(soma_vol_by_ch.get(soma_ch_ids[0], [])) if soma_ch_ids else []
        for cid_b in soma_ch_ids[1:]:
            cells_b = soma_vol_by_ch.get(cid_b, [])
            matched_pairs, unmatched_a, unmatched_b = match_soma_3d_iou(
                merged_soma_vols, cells_b, iou_thresh=iou_thresh_3d, z_pad=z_pad_3d
            )
            for a_cell, b_cell in matched_pairs:
                a_cell['class'] = _merge_class(a_cell['class'], b_cell['class'])
            merged_soma_vols = merged_soma_vols + unmatched_b
        n_multi  = sum(1 for c in merged_soma_vols if len(c['class'].split('_')) > 2)
        n_single = len(merged_soma_vols) - n_multi
        logging.info(f"✔️ [3A] Soma 3D IoU 匹配: {len(merged_soma_vols)} 个 "
                     f"(多阳性 {n_multi}, 单阳性 {n_single})")

        # Phase B: soma × TF GMM 标注（逐通道独立GMM，结果累积到soma）
        p_thresh = zl_tf.get('gmm_p_thresh', 0.5)
        for cid in tf_ch_ids:
            tf_vols = tf_vol_by_ch.get(cid, [])
            if tf_vols and merged_soma_vols:
                merged_soma_vols = annotate_soma_with_tf_gmm(
                    merged_soma_vols, tf_vols, p_thresh=p_thresh
                )
                logging.info(f"✔️ [3B] [{cid}] TF GMM 标注完成")
        n_tf_annotated = sum(
            1 for c in merged_soma_vols
            if any(cid in c['class'] for cid in tf_ch_ids)
        )
        logging.info(f"✔️ [3B] 全部TF标注完成: {n_tf_annotated} 个 soma 有 TF marker")

        # Phase C: 输出为 2D 形式（center_z处的bbox），排除TF单阳性
        output_rows = []
        for soma in merged_soma_vols:
            center_z = int(round(soma['cz']))
            bbox = soma['per_z_boxes'].get(
                center_z,
                [soma['x1_3d'], soma['y1_3d'], soma['x2_3d'], soma['y2_3d']]
            )
            output_rows.append([
                bbox[0], bbox[1], bbox[2], bbox[3],
                soma['score'], soma['mean'], soma['class'], center_z
            ])

        soma_3d = (np.array(output_rows, dtype=object)
                   if output_rows else np.empty((0, 8), dtype=object))
        tf_3d   = np.empty((0, 8), dtype=object)   # TF单阳性不输出

        out_coloc = os.path.join(derived['pATH_COLOCALIZATION'], 'coloc_result.csv')
        pd.DataFrame(soma_3d, columns=BOX_COLS).to_csv(out_coloc, index=False)
        logging.info(f"✔️ [3C] 共定位结果: {len(soma_3d)} 个细胞 → {out_coloc}")

        final_results = soma_3d

        # ====== 4. 保存全局 3D 报告 (目标 4) ======
        if final_results is not None and len(final_results) > 0:
            df = pd.DataFrame(final_results, columns=BOX_COLS)

            if len(metadata_registry) > 0:
                meta_np     = np.array(metadata_registry, dtype=object)
                meta_coords = meta_np[:, :3].astype(float)
                meta_coords[:, 2] *= 10.0
                tree = cKDTree(meta_coords)
                final_coords = np.column_stack((
                    (final_results[:, 0] + final_results[:, 2]) / 2,
                    (final_results[:, 1] + final_results[:, 3]) / 2,
                    final_results[:, 7].astype(float) * 10.0
                ))
                _, indices = tree.query(final_coords)
                df['tile_name']  = meta_np[indices, 3]
                df['slice_name'] = meta_np[indices, 4]
            else:
                df['tile_name']  = 'Unknown'
                df['slice_name'] = 'Unknown'

            # Checkpoint CSV (全量，供 Stage 4/5 读取)
            df.to_csv(bbox_path, index=False)
            logging.info(f"✔️ 已输出 目标4 checkpoint: {bbox_path}")

            # 按细胞类型分别保存 CSV (neuron_GFP.csv, glia_RFP_Sox9.csv, ...)
            for cls, cls_df in df.groupby('class'):
                safe_cls = str(cls).replace('/', '_').replace('\\', '_')
                cls_df.to_csv(os.path.join(derived['pATH_COLOCALIZATION'], f"{safe_cls}.csv"), index=False)
            logging.info(f"✔️ 已输出 目标4 ({df['class'].nunique()} 种细胞类型) → {derived['pATH_COLOCALIZATION']}")
        else:
            logging.warning("⚠️ 全局未检测到任何 3D 目标。")

    # ==========================================
    # 阶段 4: 生成分析级的统计报告与质心
    # ==========================================
    report_path = os.path.join(derived['pATH_REPORT'], "global_summary_statistics.csv")

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
        logging.warning("⚠️ 置换检验跳过：soma_3d/tf_3d 不在本次内存中。")

    logging.info(f"🎉 动态多通道推断全部完成！总耗时: {(time.time() - start_time)/60:.2f} 分钟。")