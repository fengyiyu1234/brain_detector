# -*- coding: utf-8 -*-
#python scripts/run_inference.py --config config/config.json
import argparse
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import time
import csv
import logging
import pickle
import multiprocessing as mp
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from src.config.loader import load_config, expand_double_exposure_channels
from src.utils.logger import setup_logging
from src.utils.io import (listTile, listTile_from_local_csvs, loadTeraxml,
                          save_run_metadata, compute_grid_fallback_offsets)
from src.core.worker import process_single_tile_wrapper, init_worker
from src.core.stitcher import combine_predictions, fuse_dual_intensity_2d
from src.core.z_linker import run_z_linker
from src.core.stitcher import (match_soma_3d_iou, annotate_soma_with_tf_containment,
                               permutation_test_colocalization, _merge_class,
                               suppress_cross_class_overlap)
from src.core.point_cloud_aligner import (
    compute_tile_channel_shifts,
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

    base_res_path = paths.get('pATHRESULT')
    if not base_res_path:
        raise ValueError("❌ 配置文件中缺失 pATHRESULT 输出根目录！")
        
    pipeline_mode = config.get('pipeline_mode', 'post_align')  # "post_align" | "pre_align"
    start_from_stage = config.get('start_from_stage', 1)
    pre_align_cfg = config.get('pre_align_params', {})

    derived = {}
    derived['pATH_ALIGN_OFFSETS'] = os.path.join(base_res_path, "0_channel_alignment")
    derived['pATH_DET_RES']      = os.path.join(base_res_path, "1_tile_2d_raw")
    derived['pATH_DET_FILTERED'] = os.path.join(base_res_path, "1_tile_2d_filtered")
    derived['pATH_DET_FUSED']    = os.path.join(base_res_path, "1_tile_2d_fused")
    derived['pATH_GLOBAL_2D']    = os.path.join(base_res_path, "2_global_2d_raw")
    derived['pATH_CHANNEL_3D']    = os.path.join(base_res_path, "3_channel_3d")
    derived['pATH_COLOCALIZATION'] = os.path.join(base_res_path, "4_colocalization")
    derived['pATH_REPORT'] = os.path.join(base_res_path, "5_analysis_report")
    derived['pATH_CENTROIDS']   = os.path.join(derived['pATH_REPORT'], "cell_centroids")
    derived['pATH_HISTOGRAMS']  = os.path.join(base_res_path, "1_tile_2d_histograms")

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

    # 2b. 展开 double_exposure 通道，得到检测阶段(Stage 2)专用的路由列表（含合成的第二曝光通道）。
    # Stage 2 之后的所有阶段（2.5 部分、2.75、3、4、5）继续使用未展开的 routing_config——
    # 融合(Stage 2.6)之后两个曝光就是同一个逻辑通道了。
    detect_routing_config = expand_double_exposure_channels(routing_config)
    config['channels_routing_detect'] = detect_routing_config

    if start_from_stage < 2:
        logging.info("=== 多通道路径检查 ===")
        for ch in detect_routing_config:
            d_path = paths.get(ch['dir_key'])
            if not d_path or not os.path.exists(d_path):
                raise FileNotFoundError(f"❌ 通道 {ch['id']} 路径不存在: {d_path}")
            logging.info(f"[{ch['type'].upper()}] {ch['id']}: {d_path}")
        logging.info("=====================")
    else:
        logging.info(f"[start_from_stage={start_from_stage}] 跳过网络通道路径校验。")

    # 3. 获取 Tile 列表 (基于锚点通道)
    anchor_ch_id = anchor_ch['id']
    if start_from_stage >= 2:
        logging.info(f"[start_from_stage={start_from_stage}] 跳过网络扫描，从本地 CSV 推导 Tile 列表...")
        dirnames, pATHTILE_all = listTile_from_local_csvs(
            derived['pATH_DET_RES'], anchor_ch_id, anchor_dir
        )
        if not pATHTILE_all:
            raise ValueError(
                f"❌ start_from_stage={start_from_stage} 但在 {derived['pATH_DET_RES']} "
                f"中未找到 *_{anchor_ch_id}_result.csv 文件。请先完成 Stage 2 检测。"
            )
        logging.info(f"  从本地 CSV 推导出 {len(pATHTILE_all)} 个 Tile。")
    else:
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
        overlap_pct = pre_align_cfg.get('tile_overlap_pct', 15)
        logging.warning(f"⚠️ 未找到 TeraStitcher XML（路径: {pATHxml}），pre_align 模式回退到文件名解析全局偏移。")
        dir_dict, disp_mat_fin = compute_grid_fallback_offsets(pATHTILE_all, tile_size, overlap_pct)
        logging.info(f"  解析到 {len(dir_dict)} 个 tile，网格 {disp_mat_fin.shape[0]}×{disp_mat_fin.shape[1]}")
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
        all_done = all(
            os.path.exists(os.path.join(derived['pATH_DET_RES'], f"{tile_name}_{ch['id']}_result.csv"))
            for ch in detect_routing_config
        )
        if not all_done:
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

    if config.get('stop_after_detection', False):
        logging.info("🛑 stop_after_detection=true：Stage 1 完成，正常退出。Stage 2~5 可在 CPU 或单 GPU 上单独运行。")
        sys.exit(0)

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

            pa_sample_z      = pre_align_cfg.get('sample_z_center_count', 50)
            pa_bin_size      = pre_align_cfg.get('voxel_bin_size_px', 4)
            pa_xy_range      = pre_align_cfg.get('xy_search_range_px', 30)
            pa_z_range       = pre_align_cfg.get('z_search_range_slices', 5)
            pa_fine_xy       = pre_align_cfg.get('xy_fine_search_px', 8)
            pa_fine_z        = pre_align_cfg.get('z_fine_search_slices', 2)
            pa_xy_res        = dp.get('xy_resolution_um', 0.65)
            pa_z_res         = dp.get('z_resolution_um', 8.0)
            pa_max_center_dist_ratio = zl_tf.get('max_center_dist_ratio', 0.3)
            pa_containment_z_pad     = zl_tf.get('containment_z_pad', 0)

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
                        max_z_gap=zl_params.get('max_z_gap', 0),
                    )
                    per_ch_vol_lists[cid] = vol_list
                    if vol_list:
                        z_counts.extend([c.get('cz', 0) for c in vol_list])

                # 2. 估计 tile z 中心
                z_center = float(np.median(z_counts)) if z_counts else 0.0
                z_half   = pa_sample_z // 2

                # 3. 两步体素对齐
                shifts, scores = compute_tile_channel_shifts(
                    per_ch_vol_lists,
                    soma_ch_ids=soma_ch_ids_align,
                    tf_ch_ids=tf_ch_ids_align,
                    z_center=z_center,
                    z_half_window=z_half,
                    bin_size=pa_bin_size,
                    xy_res_um=pa_xy_res,
                    z_res_um=pa_z_res,
                    xy_range_px=pa_xy_range,
                    z_range_slices=pa_z_range,
                    fine_xy_px=pa_fine_xy,
                    fine_z_slices=pa_fine_z,
                    max_center_dist_ratio=pa_max_center_dist_ratio,
                    containment_z_pad=pa_containment_z_pad,
                )

                # 3b. double_exposure 通道的第二曝光复用主曝光的偏移量
                # （同一物理通道/视野，只是曝光不同，无需独立点云配准）。
                for ch in routing_cfg_align:
                    if ch.get('double_exposure'):
                        second_id = ch['second_intensity_id']
                        shifts[second_id] = shifts.get(ch['id'], (0, 0, 0))
                        scores[second_id] = scores.get(ch['id'], 0.0)

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

                    if ch.get('double_exposure'):
                        second_id = ch['second_intensity_id']
                        dx2, dy2, dz2 = shifts.get(second_id, (0, 0, 0))
                        in_csv2  = os.path.join(derived['pATH_DET_RES'], f"{tile_name}_{second_id}_result.csv")
                        out_csv2 = os.path.join(derived['pATH_ALIGN_OFFSETS'], f"{tile_name}_{second_id}_result.csv")
                        if os.path.isfile(in_csv2) and not os.path.isfile(out_csv2):
                            apply_shift_to_csv(in_csv2, dx2, dy2, dz2, out_csv2)

            # 写完成标记
            open(align_done_flag, 'w').close()
            logging.info("✔️ [2.5] 所有 Tile 点云对齐完成。")

    # ==========================================
    # 阶段 2.6: 双曝光强度融合 (Dual-Intensity Fusion)
    # 输入: 0_channel_alignment (pre_align) 或 1_tile_2d_raw (post_align) —— 与 Stage 2.75 相同的
    #       原始来源，在过滤之前完成两个曝光的融合。
    # 输出: 1_tile_2d_fused/{tile}_{primary_id}_result.csv （落在 Stage 2.75 期望主通道数据的位置）
    # ==========================================
    _de_channels = [ch for ch in routing_config if ch.get('double_exposure')]
    _tile_names_all = [os.path.split(p)[-1] for p in pATHTILE_all]

    if not _de_channels:
        logging.info("⏭️ [2.6] 未配置任何 double_exposure 通道，跳过融合阶段。")
    else:
        _fusion_src = (derived['pATH_ALIGN_OFFSETS']
                       if pipeline_mode == 'pre_align'
                       else derived['pATH_DET_RES'])
        _fusion_dst = derived['pATH_DET_FUSED']

        _fusion_done = all(
            os.path.exists(os.path.join(_fusion_dst, f"{tn}_{ch['id']}_result.csv"))
            for tn in _tile_names_all
            for ch in _de_channels
        )

        if _fusion_done:
            logging.info("✔️ Checkpoint 2.6 达成: 融合后 tile CSV 已全部存在。")
        else:
            logging.info(f"阶段 2.6: 融合 {len(_de_channels)} 个双曝光通道 ...")
            _fusion_rows = []
            _agg = {ch['id']: {'low': 0, 'high': 0, 'fused': 0} for ch in _de_channels}
            for _tn in tqdm(_tile_names_all, desc="Fuse dual-intensity tiles"):
                for ch in _de_channels:
                    ch_id = ch['id']
                    second_id = ch['second_intensity_id']
                    out_csv = os.path.join(_fusion_dst, f"{_tn}_{ch_id}_result.csv")
                    if os.path.exists(out_csv):
                        continue
                    low_csv  = os.path.join(_fusion_src, f"{_tn}_{ch_id}_result.csv")
                    high_csv = os.path.join(_fusion_src, f"{_tn}_{second_id}_result.csv")
                    low_df  = pd.read_csv(low_csv)  if os.path.isfile(low_csv)  else pd.DataFrame(columns=[
                        "slice_name", "x1", "y1", "x2", "y2", "class", "score", "mean", "z"])
                    high_df = pd.read_csv(high_csv) if os.path.isfile(high_csv) else pd.DataFrame(columns=[
                        "slice_name", "x1", "y1", "x2", "y2", "class", "score", "mean", "z"])
                    if low_df.empty and high_df.empty:
                        continue

                    fused_df, n_low, n_high, n_fused = fuse_dual_intensity_2d(
                        low_df, high_df, iou_thresh=ch.get('fusion_iou_thresh', 0.3)
                    )
                    fused_df.to_csv(out_csv, index=False)

                    n_matched = n_low + n_high - n_fused
                    _agg[ch_id]['low']   += n_low
                    _agg[ch_id]['high']  += n_high
                    _agg[ch_id]['fused'] += n_fused
                    _fusion_rows.append({
                        "channel": ch_id, "tile": _tn, "n_low": n_low, "n_high": n_high,
                        "n_fused": n_fused, "n_matched": n_matched,
                    })
                    logging.info(f"  [2.6][{ch_id}] tile={_tn}: low={n_low} high={n_high} "
                                 f"-> fused={n_fused} (matched={n_matched})")

            for ch_id, c in _agg.items():
                n_matched_total = c['low'] + c['high'] - c['fused']
                logging.info(f"✔️ [2.6] [{ch_id}] 汇总: low={c['low']:,} high={c['high']:,} "
                             f"fused={c['fused']:,} (matched={n_matched_total:,})")

            if _fusion_rows:
                _df_fusion_summary = pd.DataFrame(_fusion_rows)
                for ch_id, c in _agg.items():
                    _df_fusion_summary = pd.concat([_df_fusion_summary, pd.DataFrame([{
                        "channel": ch_id, "tile": "TOTAL", "n_low": c['low'], "n_high": c['high'],
                        "n_fused": c['fused'], "n_matched": c['low'] + c['high'] - c['fused'],
                    }])], ignore_index=True)
                _df_fusion_summary.to_csv(
                    os.path.join(_fusion_dst, "fusion_summary.csv"), index=False
                )

    # ==========================================
    # 阶段 2.75: Tile 级 CSV 强度/尺寸过滤
    # 输入: 1_tile_2d_raw (post_align) 或 0_channel_alignment (pre_align)
    # 输出: 1_tile_2d_filtered  (Stage 3 从此读取)
    # stage_2_75_enabled=true : 应用过滤（适合旧的未过滤 1_tile_2d_raw）
    # stage_2_75_enabled=false: 直接透传（检测阶段已过滤时跳过重复处理）
    # 修改过滤参数后删除 1_tile_2d_filtered 即可重跑，无需重新检测
    # ==========================================
    _filter_src = (derived['pATH_ALIGN_OFFSETS']
                   if pipeline_mode == 'pre_align'
                   else derived['pATH_DET_RES'])
    _filter_dst = derived['pATH_DET_FILTERED']
    _stage_2_75_enabled = config.get('stage_2_75_enabled', True)

    def _src_for(ch):
        """double_exposure 通道读取融合(Stage 2.6)后的结果，其余通道读取常规来源。"""
        return derived['pATH_DET_FUSED'] if ch.get('double_exposure') else _filter_src

    _tile_names_all = [os.path.split(p)[-1] for p in pATHTILE_all]
    _filter_done = all(
        os.path.exists(os.path.join(_filter_dst, f"{tn}_{ch['id']}_result.csv"))
        for tn in _tile_names_all
        for ch in routing_config
        if os.path.exists(os.path.join(_src_for(ch), f"{tn}_{ch['id']}_result.csv"))
    )

    if _filter_done:
        logging.info("✔️ Checkpoint 2.75 达成: 过滤后 tile CSV 已全部存在。")
    elif not _stage_2_75_enabled:
        import shutil
        logging.info("阶段 2.75: stage_2_75_enabled=false，直接透传 raw CSV → filtered（跳过过滤）...")
        for _tn in tqdm(_tile_names_all, desc="Passthrough tiles"):
            for _ch in routing_config:
                _ch_id  = _ch['id']
                _in_csv = os.path.join(_src_for(_ch), f"{_tn}_{_ch_id}_result.csv")
                _out_csv = os.path.join(_filter_dst, f"{_tn}_{_ch_id}_result.csv")
                if os.path.exists(_in_csv) and not os.path.exists(_out_csv):
                    shutil.copy2(_in_csv, _out_csv)
        logging.info("✔️ [2.75] 透传完成。")
    else:
        logging.info("阶段 2.75: 对 tile CSV 应用强度/尺寸过滤...")
        _sd_dp   = dp.get('stardist', {})
        _yolo_dp = dp.get('yolo', {})
        _n_filtered_total = 0
        for _tn in tqdm(_tile_names_all, desc="Filter tiles"):
            for _ch in routing_config:
                _ch_id   = _ch['id']
                _in_csv  = os.path.join(_src_for(_ch), f"{_tn}_{_ch_id}_result.csv")
                _out_csv = os.path.join(_filter_dst, f"{_tn}_{_ch_id}_result.csv")
                if not os.path.exists(_in_csv) or os.path.exists(_out_csv):
                    continue
                _model_dp = _sd_dp if _ch['model'] == 'stardist' else _yolo_dp
                _df = pd.read_csv(_in_csv)
                _n_before = len(_df)
                if not _df.empty:
                    _bbox_min      = _model_dp.get('bbox_min')
                    _bbox_max      = _model_dp.get('bbox_max')
                    _area_pct_min  = _model_dp.get('bbox_area_pct_min')
                    _area_pct_max  = _model_dp.get('bbox_area_pct_max')
                    _pct_min       = _model_dp.get('bbox_mean_pct_min')
                    _abs_min       = (_model_dp.get('bbox_mean_min') or
                                      _model_dp.get('nucleus_mean_min', 0)) or 0
                    # 1. 绝对尺寸
                    if _bbox_min is not None:
                        _df = _df[(_df['x2'] - _df['x1'] >= _bbox_min) &
                                  (_df['y2'] - _df['y1'] >= _bbox_min)]
                    if _bbox_max is not None:
                        _df = _df[(_df['x2'] - _df['x1'] <= _bbox_max) &
                                  (_df['y2'] - _df['y1'] <= _bbox_max)]
                    # 1b. 宽高比过滤
                    _aspect_max = _model_dp.get('bbox_max_aspect_ratio')
                    if _aspect_max is not None and not _df.empty:
                        _w = (_df['x2'] - _df['x1']).values
                        _h = (_df['y2'] - _df['y1']).values
                        _df = _df[np.maximum(_w, _h) <= _aspect_max * np.maximum(np.minimum(_w, _h), 1e-6)]
                    # 2. 面积百分位（在通过绝对尺寸过滤的子集上计算）
                    if _area_pct_min is not None and not _df.empty:
                        _areas  = (_df['x2'] - _df['x1']) * (_df['y2'] - _df['y1'])
                        _thresh = float(_areas.quantile(_area_pct_min / 100.0))
                        _df = _df[_areas >= _thresh]
                    # 2b. 面积百分位上限（过滤掉面积最大的框）
                    if _area_pct_max is not None and not _df.empty:
                        _areas  = (_df['x2'] - _df['x1']) * (_df['y2'] - _df['y1'])
                        _thresh = float(_areas.quantile(_area_pct_max / 100.0))
                        _df = _df[_areas <= _thresh]
                    # 3. 亮度百分位（在通过面积过滤的子集上计算）
                    if _pct_min is not None and not _df.empty:
                        _thresh = float(_df['mean'].quantile(_pct_min / 100.0))
                        _df = _df[_df['mean'] >= _thresh]
                    # 4. 亮度绝对下限
                    if _abs_min > 0:
                        _df = _df[_df['mean'] >= _abs_min]
                    # 5. Per-z-slice IoMin containment NMS（抑制大框套小框的重复检测）
                    _containment_thresh = _model_dp.get('nms_containment_thresh', None)
                    if _containment_thresh is not None and not _df.empty:
                        _keep_mask = np.ones(len(_df), dtype=bool)
                        _df_reset = _df.reset_index(drop=True)
                        for _z_val in _df_reset['z'].unique():
                            _z_mask = (_df_reset['z'] == _z_val).values
                            _idx    = np.where(_z_mask)[0]
                            _x1 = _df_reset['x1'].values[_idx].astype(float)
                            _y1 = _df_reset['y1'].values[_idx].astype(float)
                            _x2 = _df_reset['x2'].values[_idx].astype(float)
                            _y2 = _df_reset['y2'].values[_idx].astype(float)
                            _sc = _df_reset['score'].values[_idx].astype(float)
                            _ar = np.maximum(0, _x2 - _x1) * np.maximum(0, _y2 - _y1)
                            _order      = np.argsort(-_sc)
                            _local_keep = np.ones(len(_idx), dtype=bool)
                            for _ki in range(len(_order)):
                                _i = _order[_ki]
                                if not _local_keep[_i]:
                                    continue
                                _rest = _order[_ki + 1:]
                                _rest = _rest[_local_keep[_rest]]
                                if len(_rest) == 0:
                                    continue
                                _ix1   = np.maximum(_x1[_i], _x1[_rest])
                                _iy1   = np.maximum(_y1[_i], _y1[_rest])
                                _ix2   = np.minimum(_x2[_i], _x2[_rest])
                                _iy2   = np.minimum(_y2[_i], _y2[_rest])
                                _inter = np.maximum(0, _ix2 - _ix1) * np.maximum(0, _iy2 - _iy1)
                                _iomin = _inter / (np.minimum(_ar[_i], _ar[_rest]) + 1e-6)
                                _local_keep[_rest[_iomin > _containment_thresh]] = False
                            _keep_mask[_idx[~_local_keep]] = False
                        _df = _df[_keep_mask]
                _df.to_csv(_out_csv, index=False)
                _n_filtered_total += _n_before - len(_df)
        logging.info(f"✔️ [2.75] Tile 过滤完成，共移除 {_n_filtered_total:,} 个 box。")

    pATH_SRC_CSV = _filter_dst

    # ==========================================
    # 阶段 2.8: 生成过滤前 Raw 2D 直方图（强度 & 面积）
    # 输入: _filter_src (raw CSV)
    # 输出: 1_tile_2d_histograms/{tile}_{ch}_hist.png
    # 删除输出目录可强制重建；不影响过滤及后续流程
    # 开关: detection_params.generate_histograms (默认 true)
    # ==========================================
    _hist_dir = derived['pATH_HISTOGRAMS']
    if not dp.get('generate_histograms', True):
        logging.info("⏭️ [2.8] generate_histograms=false，跳过 raw 直方图生成。")
        _hist_todo = []
    else:
        _hist_todo = [
            (_tn, _ch)
            for _tn in _tile_names_all
            for _ch in routing_config
            if _ch.get('active', True)
            and os.path.exists(os.path.join(_src_for(_ch), f"{_tn}_{_ch['id']}_result.csv"))
            and not os.path.exists(os.path.join(_hist_dir, f"{_tn}_{_ch['id']}_hist.png"))
        ]
        if not _hist_todo:
            logging.info("✔️ Checkpoint 2.8 达成: raw 直方图已全部存在。")
    if _hist_todo:
        logging.info(f"阶段 2.8: 生成 {len(_hist_todo)} 个 raw 2D 直方图 ...")
        for _tn, _ch in tqdm(_hist_todo, desc="Raw histograms"):
            _ch_id    = _ch['id']
            _csv_path = os.path.join(_src_for(_ch), f"{_tn}_{_ch_id}_result.csv")
            _df_raw   = pd.read_csv(_csv_path)
            if _df_raw.empty:
                continue
            _areas = ((_df_raw['x2'] - _df_raw['x1']) * (_df_raw['y2'] - _df_raw['y1'])).values
            _means = _df_raw['mean'].values
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].hist(_means, bins=50, color='steelblue', edgecolor='none')
            axes[0].set_title(f'Intensity (mean)  |  {_tn} / {_ch_id}  |  n={len(_means):,}')
            axes[0].set_xlabel('mean pixel value')
            axes[0].set_ylabel('count')
            axes[1].hist(_areas, bins=50, color='salmon', edgecolor='none')
            axes[1].set_title(f'Area (px²)  |  {_tn} / {_ch_id}  |  n={len(_areas):,}')
            axes[1].set_xlabel('area (px²)')
            axes[1].set_ylabel('count')
            fig.tight_layout()
            fig.savefig(os.path.join(_hist_dir, f"{_tn}_{_ch_id}_hist.png"), dpi=100)
            plt.close(fig)
        logging.info(f"✔️ [2.8] 直方图输出至: {_hist_dir}")

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

        # ====== Checkpoint 2a: 若 2_global_2d_raw 已有内容，直接加载，跳过 Tile 拼接 ======
        global_2d_paths = {
            ch['id']: os.path.join(derived['pATH_GLOBAL_2D'], f"{ch['id']}_2d_global.csv")
            for ch in routing_config
        }
        if routing_config and all(os.path.exists(p) for p in global_2d_paths.values()):
            logging.info("✔️ Checkpoint 2a 达成: 直接加载已有的全局 2D 结果，跳过 Tile 拼接。")
            for ch_id, p in global_2d_paths.items():
                per_ch_matrices[ch_id] = pd.read_csv(p)[BOX_COLS].values
        else:
            # 按通道跳过：已存在 global_2d CSV 的通道直接加载，只重新拼接缺失的通道
            _ch_cached = [ch for ch in routing_config if os.path.exists(global_2d_paths[ch['id']])]
            _ch_todo   = [ch for ch in routing_config if not os.path.exists(global_2d_paths[ch['id']])]
            if _ch_cached:
                logging.info(f"✔️ Checkpoint 2a (按通道): {[c['id'] for c in _ch_cached]} 已存在，直接加载。")
                for ch in _ch_cached:
                    per_ch_matrices[ch['id']] = pd.read_csv(global_2d_paths[ch['id']])[BOX_COLS].values

            # ====== 1. 逐通道独立拼接全局 2D，每个通道单独保存 ======
            for ch in _ch_todo:
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
                                    metadata_registry, tile_name, tILESIZE=tile_size,
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

        # ====== Checkpoint 2b: 若 3_channel_3d pkl 已有内容，直接加载，跳过 Z-Linker ======
        all_ch_ids = soma_ch_ids + tf_ch_ids
        pkl_paths = {
            cid: os.path.join(derived['pATH_CHANNEL_3D'], f"{cid}_3d_tracked.pkl")
            for cid in all_ch_ids
        }
        if all_ch_ids and all(os.path.exists(p) for p in pkl_paths.values()):
            logging.info("✔️ Checkpoint 2b 达成: 直接加载已有的 3D 追踪结果，跳过 Z-Linker。")
            for cid in soma_ch_ids:
                with open(pkl_paths[cid], 'rb') as pf:
                    soma_vol_by_ch[cid] = pickle.load(pf)
            for cid in tf_ch_ids:
                with open(pkl_paths[cid], 'rb') as pf:
                    tf_vol_by_ch[cid] = pickle.load(pf)
        else:
            # 按通道跳过：已存在 pkl 的通道直接加载，只对缺失的通道重新跑 Z-Linker
            for cid in soma_ch_ids:
                if os.path.exists(pkl_paths[cid]):
                    with open(pkl_paths[cid], 'rb') as pf:
                        soma_vol_by_ch[cid] = pickle.load(pf)
                    logging.info(f"✔️ [{cid}] 已存在 3D 追踪结果，直接加载（跳过 Z-Linker）。")
                    continue
                ch_mat = per_ch_matrices.get(cid)
                if ch_mat is None or len(ch_mat) == 0:
                    continue
                summary, vol_list = run_z_linker(
                    ch_mat,
                    iou_thresh=zl_soma.get('iou_thresh', 0.35),
                    min_z_layers=zl_soma.get('min_z_layers', 1),
                    max_cell_z_span=zl_soma.get('max_cell_z_span', 5),
                    max_z_gap=zl_soma.get('max_z_gap', 0),
                )
                soma_vol_by_ch[cid] = vol_list
                out_3d = os.path.join(derived['pATH_CHANNEL_3D'], f"{cid}_3d_tracked.csv")
                pd.DataFrame(summary, columns=BOX_COLS).to_csv(out_3d, index=False)
                with open(os.path.join(derived['pATH_CHANNEL_3D'], f"{cid}_3d_tracked.pkl"), 'wb') as pf:
                    pickle.dump(vol_list, pf)
                logging.info(f"✔️ [{cid}] Z-Link soma: {len(vol_list)} 个细胞 → {out_3d}")

            for cid in tf_ch_ids:
                if os.path.exists(pkl_paths[cid]):
                    with open(pkl_paths[cid], 'rb') as pf:
                        tf_vol_by_ch[cid] = pickle.load(pf)
                    logging.info(f"✔️ [{cid}] 已存在 3D 追踪结果，直接加载（跳过 Z-Linker）。")
                    continue
                ch_mat = per_ch_matrices.get(cid)
                if ch_mat is None or len(ch_mat) == 0:
                    continue
                summary, vol_list = run_z_linker(
                    ch_mat,
                    iou_thresh=zl_tf.get('iou_thresh', 0.25),
                    min_z_layers=zl_tf.get('min_z_layers', 1),
                    max_cell_z_span=zl_tf.get('max_cell_z_span', 3),
                    max_z_gap=zl_tf.get('max_z_gap', 0),
                )
                tf_vol_by_ch[cid] = vol_list
                out_3d = os.path.join(derived['pATH_CHANNEL_3D'], f"{cid}_3d_tracked.csv")
                pd.DataFrame(summary, columns=BOX_COLS).to_csv(out_3d, index=False)
                with open(os.path.join(derived['pATH_CHANNEL_3D'], f"{cid}_3d_tracked.pkl"), 'wb') as pf:
                    pickle.dump(vol_list, pf)
                logging.info(f"✔️ [{cid}] Z-Link TF: {len(vol_list)} 个细胞 → {out_3d}")

        # ====== 3. 3D Colocalization ======
        # Phase A: soma × soma 3D IoU（逐对匹配，依次合并到主列表）
        iou_thresh_3d   = zl_soma.get('iou_thresh_3d', 0.15)
        iomin_thresh_3d = zl_soma.get('iomin_thresh_3d', 0.5)
        z_pad_3d        = zl_soma.get('z_pad_3d', 2)

        merged_soma_vols = list(soma_vol_by_ch.get(soma_ch_ids[0], [])) if soma_ch_ids else []
        for cid_b in soma_ch_ids[1:]:
            cells_b = soma_vol_by_ch.get(cid_b, [])
            matched_pairs, unmatched_a, unmatched_b = match_soma_3d_iou(
                merged_soma_vols, cells_b,
                iou_thresh=iou_thresh_3d, iomin_thresh=iomin_thresh_3d, z_pad=z_pad_3d
            )
            for a_cell, b_cell in matched_pairs:
                a_cell['class'] = _merge_class(a_cell['class'], b_cell['class'])
            merged_soma_vols = merged_soma_vols + unmatched_b
        cross_iou = zl_soma.get('cross_class_iou_thresh', 0.5)
        merged_soma_vols = suppress_cross_class_overlap(
            merged_soma_vols, iou_thresh=cross_iou, z_pad=z_pad_3d
        )
        n_multi  = sum(1 for c in merged_soma_vols if len(c['class'].split('_')) > 2)
        n_single = len(merged_soma_vols) - n_multi
        logging.info(f"✔️ [3A] Soma 3D IoU 匹配: {len(merged_soma_vols)} 个 "
                     f"(多阳性 {n_multi}, 单阳性 {n_single})")

        # Phase B: soma × TF 严格包含标注（TF框必须完全在soma框内）
        xy_margin          = zl_tf.get('containment_xy_margin', 0)
        z_pad_tf           = zl_tf.get('containment_z_pad', 2)
        max_center_dist    = zl_tf.get('max_center_dist_ratio', 0.5)
        tf_bbox_max_w = zl_tf.get('bbox_max_w', None)
        tf_bbox_max_h = zl_tf.get('bbox_max_h', None)
        for cid in tf_ch_ids:
            tf_vols = tf_vol_by_ch.get(cid, [])
            if (tf_bbox_max_w is not None or tf_bbox_max_h is not None) and tf_vols:
                n_before = len(tf_vols)
                tf_vols = [
                    v for v in tf_vols
                    if (tf_bbox_max_w is None or v['x2_3d'] - v['x1_3d'] <= tf_bbox_max_w)
                    and (tf_bbox_max_h is None or v['y2_3d'] - v['y1_3d'] <= tf_bbox_max_h)
                ]
                logging.info(f"  [{cid}] TF size filter: {n_before} → {len(tf_vols)} "
                             f"(bbox_max_w={tf_bbox_max_w}, bbox_max_h={tf_bbox_max_h})")
            if tf_vols and merged_soma_vols:
                merged_soma_vols = annotate_soma_with_tf_containment(
                    merged_soma_vols, tf_vols, z_pad=z_pad_tf, xy_margin=xy_margin,
                    max_center_dist_ratio=max_center_dist
                )
                logging.info(f"✔️ [3B] [{cid}] TF containment 标注完成")
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