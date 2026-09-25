# -*- coding: utf-8 -*-
#python scripts/run_inference.py --config config/config.json
import argparse
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import time
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
from src.utils.markers import channel_marker, class_markers, split_class
import multiprocessing.connection
from src.core.worker import run_tile_process
from src.core.detection_filter import (
    atomic_write_csv, filter_detection_df, resolve_filter_params, source_dir_for_channel,
)
from src.core.stitcher import fuse_dual_intensity_2d
from src.core.channel_stage3 import stitch_and_link_channel
from src.core.stitcher import (match_soma_3d_iou, annotate_soma_with_tf_containment,
                               _merge_class, suppress_cross_class_overlap)
from src.core.point_cloud_aligner import (
    align_tile, tile_alignment_done,
    resolve_align_settings, check_align_settings, save_align_settings,
    validate_alignment_frame, validate_stitching_xml_frame,
    validate_cached_geometry, validate_measurement_source,
)
from concurrent.futures import ProcessPoolExecutor, as_completed


def align_worker_count(pre_align_cfg, n_tasks):
    """Stage 2.5 并行进程数：pre_align_params.n_workers，缺省为 slurm 分到的 CPU 数（没有则本机核数）。"""
    allocated = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))
    n = pre_align_cfg.get('n_workers') or allocated
    return max(1, min(int(n), n_tasks, allocated))


_THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


class _single_threaded_children:
    """
    进程池期间让子进程只用单线程 BLAS/OpenMP，避免 进程数 × 核数 的线程超订。
    spawn 出来的子进程在引导阶段就会 import numpy，所以只能在父进程里设环境变量让它继承。
    """
    def __enter__(self):
        self.saved = {k: os.environ.get(k) for k in _THREAD_VARS}
        os.environ.update({k: '1' for k in _THREAD_VARS})

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def stage3_worker_count(config, n_channels):
    """Stage 3 同时处理的通道数：stage3_n_workers，缺省 = 通道数（受 slurm 分到的 CPU 数限制）。"""
    cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))
    n = config.get('stage3_n_workers') or n_channels
    return max(1, min(int(n), n_channels, cpus))


def run_stage3_channel_pool(routing, n_workers, src_dir, global_2d_dir, channel_3d_dir,
                            geom, zl_params, log_dir):
    """Stage 3 前半段：每个通道一个进程做全局 2D 拼接 + Z-Link。返回 {ch_id: 结果}。"""
    def _args(ch):
        return (ch, src_dir, global_2d_dir, channel_3d_dir, geom,
                zl_params['soma' if ch.get('type', 'soma') == 'soma' else 'tf'])

    logging.info(f"阶段 3: {len(routing)} 个通道的拼接 + Z-Link，{n_workers} 个进程并行")
    if n_workers <= 1:
        return {ch['id']: stitch_and_link_channel(*_args(ch)) for ch in routing}
    results = {}
    with _single_threaded_children(), ProcessPoolExecutor(
            max_workers=n_workers, mp_context=mp.get_context('spawn'),
            initializer=setup_logging, initargs=(log_dir,)) as pool:
        futs = {pool.submit(stitch_and_link_channel, *_args(ch)): ch['id'] for ch in routing}
        for fut in as_completed(futs):
            try:
                results[futs[fut]] = fut.result()
            except Exception:
                logging.error(f"❌ [3] 通道 {futs[fut]} 拼接/Z-Link 失败，终止作业。"
                              "已完成通道的 checkpoint 保留，修复后重新提交即可续跑。")
                for f in futs:
                    f.cancel()
                raise
    return results


def run_align_pool(tile_paths, n_workers, det_dir, align_dir, routing, settings):
    """
    Stage 2.5：每个 tile 一个任务，在 CPU 进程池里并行跑 align_tile。返回缺失的检测 CSV 列表。
    任一 tile 抛异常即终止作业；已写完 offsets JSON 的 tile 保留，重新提交会跳过它们。
    """
    if not tile_paths:
        return []
    missing = []
    with _single_threaded_children(), ProcessPoolExecutor(
            max_workers=n_workers, mp_context=mp.get_context('spawn')) as pool:
        futs = {pool.submit(align_tile, p, det_dir, align_dir, routing, settings): p for p in tile_paths}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Pre-Align Tiles"):
            try:
                missing.extend(fut.result())
            except Exception:
                logging.error(f"❌ [2.5] Tile {os.path.basename(futs[fut])} 对齐失败，终止作业。"
                              "已完成的 tile 保留，修复后重新提交即可续跑。")
                for f in futs:
                    f.cancel()
                raise
    return missing


def run_detection_pool(tasks, gpu_ids, config):
    """并行执行 tile 检测：每块 GPU 一个槽位，每个 tile 在全新的子进程里跑完即退出。

    不用常驻 worker 池：TF/StarDist 在长寿命进程里持续泄漏内存，跑几小时后作业被 OOM 杀掉
    （见 worker.run_tile_process）。Python 3.10 的 ProcessPoolExecutor 没有 max_tasks_per_child，
    mp.Pool 的 maxtasksperchild 又会在进程崩溃后不断补新 worker 导致卡死，所以这里自己调度。
    任何子进程非零退出（抛异常，或被杀如 OOM）时终止其余子进程并抛出异常，作业以非零状态退出。
    已写完的 tile CSV 会保留（worker 先写 .part 再改名），修复问题后重新提交即可续跑。
    """
    ctx = mp.get_context('spawn')
    pending = list(tasks)
    free_gpus = list(gpu_ids)
    running = {}  # sentinel -> (process, gpu_id, tile_name)
    try:
        with tqdm(total=len(pending), desc="Tile Processing", position=0, leave=True) as pbar:
            while pending or running:
                while pending and free_gpus:
                    task, gpu_id = pending.pop(0), free_gpus.pop(0)
                    p = ctx.Process(target=run_tile_process, args=(config, gpu_id, task))
                    p.start()
                    running[p.sentinel] = (p, gpu_id, os.path.basename(task[1]))

                for sentinel in mp.connection.wait(list(running)):
                    p, gpu_id, tile_name = running.pop(sentinel)
                    p.join()
                    if p.exitcode != 0:
                        if p.exitcode < 0:
                            logging.error(f"❌ Tile {tile_name} 的检测进程被信号 {-p.exitcode} 杀死"
                                          "（SIGKILL=9 通常是内存不足），终止作业。已完成的 tile 保留，修复后重新提交即可续跑。")
                        else:
                            logging.error(f"❌ Tile {tile_name} 检测进程异常退出（exit code {p.exitcode}，"
                                          "堆栈见 .err），终止作业。已完成的 tile 保留，修复后重新提交即可续跑。")
                        raise RuntimeError(f"Tile {tile_name} detection process failed (exitcode={p.exitcode})")
                    free_gpus.append(gpu_id)
                    pbar.update(1)
    finally:
        for p, _, _ in running.values():
            if p.is_alive():
                p.terminate()
        for p, _, _ in running.values():
            p.join()


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
    raw_source = paths.get('pATH_RAW_DETECTIONS')
    if raw_source:
        if start_from_stage < 2:
            raise ValueError(
                "paths.pATH_RAW_DETECTIONS is read-only; use start_from_stage >= 2")
        if not os.path.isdir(raw_source):
            raise FileNotFoundError(
                f"Raw detection source does not exist: {raw_source}")

    derived = {}
    derived['pATH_ALIGN_OFFSETS'] = os.path.join(base_res_path, "0_channel_alignment")
    derived['pATH_DET_RES']      = raw_source or os.path.join(base_res_path, "1_tile_2d_raw")
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

    # 2a. pre_align 对齐设置：在检测之前解析并校验，免得配置写错、或与已有对齐结果不一致，
    #     要等检测跑完才报错。
    align_settings = None
    if pipeline_mode == 'pre_align':
        align_settings = resolve_align_settings(config, routing_config)
        validate_measurement_source(derived['pATH_ALIGN_OFFSETS'], align_settings)
        check_align_settings(derived['pATH_ALIGN_OFFSETS'], align_settings)
        validate_stitching_xml_frame(paths.get('pATHXML'), align_settings['stitching_reference_channel'])
        logging.info(f"Stitching frame: {align_settings['stitching_reference_channel']}")
        logging.info(f"Pre-align 参考通道: {align_settings['reference_channel']}，"
                     f"TF 对齐方式: {align_settings['tf_align_mode']}")

    # 2b. 展开 double_exposure 通道，得到检测阶段(Stage 2)专用的路由列表（含合成的第二曝光通道）。
    # Stage 2 之后的所有阶段（2.5 部分、2.75、3、4）继续使用未展开的 routing_config——
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

    if align_settings is not None:
        previous_path = os.path.join(base_res_path, 'runtime_config.json')
        previous = load_config(previous_path) if os.path.isfile(previous_path) else None
        validate_cached_geometry(previous, config, base_res_path, align_settings)

    save_run_metadata(config, start_time)

    # 4. 范围筛选
    sTARTID = dp.get('sTARTID') or 1
    eNDID = dp.get('eNDID') or len(pATHTILE_all)
    target_indices = list(range(sTARTID - 1, eNDID))
    pATHTILE = [pATHTILE_all[i] for i in target_indices]

    # 5. TeraStitcher XML 只有 Stage 3 全局拼接才用到，加载放在 Stage 3 之前，
    #    这样拼接尚未完成的样本也能先跑检测 + tile 级对齐/过滤。

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
        if raw_source:
            raise FileNotFoundError(
                f"Read-only raw detection source {raw_source} is missing "
                f"{len(tasks_to_run)} tile(s); complete Stage 2 in its original "
                "result directory before reusing it")
        num_gpus = torch.cuda.device_count()
        num_processes = max(1, num_gpus)
        logging.info(f"阶段 2: 发现 {len(tasks_to_run)} 个缺失结果，启动 {num_processes} 个进程 ({num_gpus} GPU)...")
        # 无 GPU 时 gpu_id=None，init_worker 回退到 config['device']
        gpu_ids = list(range(num_gpus)) if num_gpus > 0 else [None]
        run_detection_pool(tasks_to_run, gpu_ids, config)
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

            save_align_settings(derived['pATH_ALIGN_OFFSETS'], align_settings)

            routing_cfg_align = [ch for ch in config.get('channels_routing', []) if ch.get('active', True)]

            # 已完成的 tile 直接跳过（offsets JSON 是最后写的完成标记，另核对 CSV 行数）
            todo = [p for p in pATHTILE
                    if not tile_alignment_done(os.path.basename(p), derived['pATH_DET_RES'],
                                               derived['pATH_ALIGN_OFFSETS'], routing_cfg_align)]
            n_workers = align_worker_count(pre_align_cfg, len(todo))
            logging.info(f"  [2.5] {len(pATHTILE) - len(todo)} 个 tile 已完成，剩余 {len(todo)} 个，"
                         f"{n_workers} 个 CPU 进程并行（纯 CPU 计算，不占 GPU）")
            missing_csvs = run_align_pool(todo, n_workers, derived['pATH_DET_RES'],
                                          derived['pATH_ALIGN_OFFSETS'], routing_cfg_align, align_settings)

            if missing_csvs:
                raise RuntimeError(
                    f"❌ [2.5] {len(missing_csvs)} 个检测 CSV 缺失，对齐结果不完整，未写完成标记。"
                    f"请先补齐 Stage 2 检测后重跑。示例: {missing_csvs[:3]}"
                )
            # 写完成标记
            open(align_done_flag, 'w').close()
            logging.info("✔️ [2.5] 所有 Tile 点云对齐完成。")

    # ==========================================
    # 阶段 2.6: 双曝光强度融合 (Dual-Intensity Fusion)
    # 输入: 0_channel_alignment (pre_align) 或 1_tile_2d_raw (post_align) —— 与 Stage 2.75 相同的
    #       原始来源，在过滤之前完成两个曝光的融合。
    # 输出: 1_tile_2d_fused/{tile}_{primary_id}_result.csv （落在 Stage 2.75 期望主通道数据的位置）
    # ==========================================
    if pipeline_mode == 'pre_align':
        validate_alignment_frame(
            derived['pATH_ALIGN_OFFSETS'],
            [os.path.basename(p) for p in pATHTILE_all],
            [ch['id'] for ch in routing_config],
            align_settings['stitching_reference_channel'],
        )

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
    # Stage 2.75: mandatory tile-level filtering.
    # Stage 3 consumes only this directory; raw output is never passed through.
    # ==========================================
    _filter_dst = derived['pATH_DET_FILTERED']
    os.makedirs(_filter_dst, exist_ok=True)
    _tile_names_all = [os.path.split(p)[-1] for p in pATHTILE_all]
    _expected_filter_inputs = [
        (_tn, _ch, source_dir_for_channel(derived, pipeline_mode, _ch))
        for _tn in _tile_names_all for _ch in routing_config
    ]
    _missing_sources = [
        os.path.join(_src, f"{_tn}_{_ch['id']}_result.csv")
        for _tn, _ch, _src in _expected_filter_inputs
        if not os.path.isfile(os.path.join(_src, f"{_tn}_{_ch['id']}_result.csv"))
    ]
    if _missing_sources:
        preview = "\n  ".join(_missing_sources[:10])
        raise FileNotFoundError(
            "Stage 2.75 requires every expected source CSV before checkpointing; "
            f"missing {len(_missing_sources)} file(s):\n  {preview}"
        )

    _filter_done = bool(_expected_filter_inputs) and all(
        os.path.isfile(os.path.join(_filter_dst, f"{_tn}_{_ch['id']}_result.csv"))
        for _tn, _ch, _src in _expected_filter_inputs
    )
    if _filter_done:
        logging.info("Stage 2.75 checkpoint reached: all filtered tile CSVs exist.")
    else:
        logging.info("Stage 2.75: applying shared filter to tile CSVs...")
        _n_filtered_total = 0
        for _tn, _ch, _src in tqdm(_expected_filter_inputs, desc="Filter tiles"):
            _ch_id = _ch['id']
            _in_csv = os.path.join(_src, f"{_tn}_{_ch_id}_result.csv")
            _out_csv = os.path.join(_filter_dst, f"{_tn}_{_ch_id}_result.csv")
            if os.path.isfile(_out_csv):
                continue
            _params = resolve_filter_params(config, _ch)
            _filtered_df, _stats = filter_detection_df(
                pd.read_csv(_in_csv), _params, return_stats=True,
                context=f"{_in_csv} ({_ch_id})",
            )
            atomic_write_csv(_filtered_df, _out_csv)
            _n_filtered_total += _stats["removed_total"]
            logging.info(
                "[2.75][%s][%s] %s -> %s (removed=%s; score_min_removed=%s; params=%s)",
                _tn, _ch_id, _stats["before"], _stats["after"],
                _stats["removed_total"], _stats["removed"]["score_min"], _params,
            )
        logging.info("[2.75] filtered %s box(es).", _n_filtered_total)

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
            and os.path.exists(os.path.join(source_dir_for_channel(derived, pipeline_mode, _ch), f"{_tn}_{_ch['id']}_result.csv"))
            and not os.path.exists(os.path.join(_hist_dir, f"{_tn}_{_ch['id']}_hist.png"))
        ]
        if not _hist_todo:
            logging.info("✔️ Checkpoint 2.8 达成: raw 直方图已全部存在。")
    if _hist_todo:
        logging.info(f"阶段 2.8: 生成 {len(_hist_todo)} 个 raw 2D 直方图 ...")
        for _tn, _ch in tqdm(_hist_todo, desc="Raw histograms"):
            _ch_id    = _ch['id']
            _csv_path = os.path.join(source_dir_for_channel(derived, pipeline_mode, _ch), f"{_tn}_{_ch_id}_result.csv")
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
    # 可选停止点：tile 级阶段（检测 / 2.5 对齐 / 2.6 融合 / 2.75 过滤 / 2.8 直方图）都不需要
    # TeraStitcher XML。拼接还没做完的样本设 stop_before_stitching=true 先跑到这里；
    # 拼接完成后改回 false 重跑，前面各阶段按 checkpoint 自动跳过，从 Stage 3 继续。
    # ==========================================
    if config.get('stop_before_stitching', False):
        logging.info("🛑 stop_before_stitching=true：tile 级阶段（检测/对齐/过滤）已完成，"
                     "在需要 XML 的全局拼接之前退出。拼接完成后改为 false 重跑即可。")
        sys.exit(0)

    # 加载 TeraStitcher XML
    #    查找优先级: paths.pATHXML(显式) > anchor_dir/xml_merging.xml > anchor_dir/xml_import.xml
    #
    #    为什么要能显式指定：拼接位移是在参考通道（如 730nm 自发荧光）上算出来的，检测通道目录里
    #    未必有这份 XML、或者放着一份 ABS_D 全为 0 的旧版本。细胞坐标必须和「真正 merge 出注册用
    #    全脑图像」的那份 XML 共用同一套 ABS_H/ABS_V/ABS_D，否则 z 会按 tile 错位。
    tile_size = dp.get('tILESIZE', 2048)
    xml_candidates = []
    if paths.get('pATHXML'):
        xml_candidates.append(paths['pATHXML'])
    xml_candidates += [os.path.join(anchor_dir, n) for n in ('xml_merging.xml', 'xml_import.xml')]
    if paths.get('pATHXML') and not os.path.isfile(paths['pATHXML']):
        raise FileNotFoundError(
            f"Explicit paths.pATHXML does not exist: {paths['pATHXML']}")
    pATHxml = next((p for p in xml_candidates if os.path.isfile(p)), None)

    if pATHxml:
        if align_settings is not None:
            validate_stitching_xml_frame(pATHxml, align_settings['stitching_reference_channel'])
        dir_dict, H, W, Z, z_start, disp_mat_fin = loadTeraxml(pATHxml, tile_size)
        _absd = disp_mat_fin[:, :, 2]
        _dmin, _dmax = _absd.min(), _absd.max()
        logging.info(f"✔️ 已加载 TeraStitcher XML: {pATHxml}")
        logging.info(f"  画布 W×H = {W}×{H}, Z = {Z}, z_start = {z_start}, "
                     f"ABS_D 范围 = [{_dmin}, {_dmax}]")
        if _dmin == _dmax:
            logging.warning(
                f"⚠️ 这份 XML 的 ABS_D 全部等于 {_dmin}，即没有任何 z 方向拼接位移，"
                f"于是 Z = stack_slices、每个 tile 的 z0 都是 0。若注册用的全脑图像是用带 z 位移的 "
                f"XML merge 出来的，细胞 z 会逐 tile 错位（错位量 = ABS_D − max(ABS_D)）。"
                f"请确认这份 XML 就是 merge 出那张图的同一份。"
            )
    elif dp.get('allow_grid_fallback', False):
        # 回退：从 tile 目录名解析行列号，用均匀 Grid 推算全局偏移。
        # 这是均匀网格，拿不到 TeraStitcher 逐 tile 的真实位移——行/列间距会有十几像素的系统
        # 偏差，跨整个网格累积可达上百像素。只适合没有拼接结果时的探索性跑批。
        overlap_pct = pre_align_cfg.get('tile_overlap_pct', 15)
        logging.warning("⚠️ 未找到任何 TeraStitcher XML，allow_grid_fallback=true，"
                        "回退到文件名解析的【均匀网格】全局偏移。")
        logging.warning("   这套坐标不等于真实拼接坐标，不要用它做配准/图谱定量。已尝试的路径：")
        for _p in xml_candidates:
            logging.warning(f"     - {_p}")
        dir_dict, disp_mat_fin = compute_grid_fallback_offsets(
            pATHTILE_all, tile_size, overlap_pct, xy_res_um=dp.get('xy_resolution_um', 0.65)
        )
        logging.info(f"  解析到 {len(dir_dict)} 个 tile，网格 {disp_mat_fin.shape[0]}×{disp_mat_fin.shape[1]}")
        H = disp_mat_fin[:, :, 1].max() + tile_size
        W = disp_mat_fin[:, :, 0].max() + tile_size
        Z = len(os.listdir(pATHTILE_all[0])) if pATHTILE_all else 1
        z_start = 0
    else:
        raise FileNotFoundError(
            "❌ 找不到 TeraStitcher 拼接坐标文件，已尝试：\n"
            + "\n".join(f"    - {p}" for p in xml_candidates)
            + "\n  请在 config 的 paths 里加 \"pATHXML\" 指向 merge 出注册用图像的那份 "
              "xml_merging.xml，或确认 anchor 通道目录下存在该文件。\n"
              "  （若确实要用文件名推算的均匀网格跑，设 detection_params.allow_grid_fallback=true，"
              "但那套坐标不能用于配准。）"
        )

    # ==========================================
    # 阶段 3: 线性 Checkpoint - 全局拼接与 Z-Linker共定位
    # ==========================================
    bbox_path = os.path.join(derived['pATH_COLOCALIZATION'], "coloc_result.csv")
    final_results = None

    if os.path.exists(bbox_path):
        logging.info(f"✔️ Checkpoint 2 达成: 加载已有的全局检测结果 {bbox_path}")
        df_boxes = pd.read_csv(bbox_path)
        final_results = df_boxes[["x1", "y1", "x2", "y2", "score", "mean", "class", "z"]].values
        if final_results.ndim == 1:
            final_results = final_results.reshape(1, -1)
    else:
        logging.info("阶段 3: 开始合并 Tile 并运行 Z-Linker (先独立 3D 追踪，再 3D 共定位)...")

        routing_config = [ch for ch in config.get('channels_routing', []) if ch.get('active', True)]
        num_tiles = len(pATHTILE_all)
        BOX_COLS  = ["x1", "y1", "x2", "y2", "score", "mean", "class", "z"]

        # ====== 1+2. 各通道独立：全局 2D 拼接 → Z-Link，每个通道一个进程 ======
        # checkpoint 与以前相同：2_global_2d_raw/<ch>_2d_global.csv、3_channel_3d/<ch>_3d_tracked.pkl
        zl      = config.get('z_linker', {})
        zl_soma = zl.get('soma', {})
        zl_tf   = zl.get('tf', {})
        zl_params = {
            'soma': dict(iou_thresh=zl_soma.get('iou_thresh', 0.35),
                         min_z_layers=zl_soma.get('min_z_layers', 1),
                         max_cell_z_span=zl_soma.get('max_cell_z_span', 5),
                         max_z_gap=zl_soma.get('max_z_gap', 0)),
            'tf':   dict(iou_thresh=zl_tf.get('iou_thresh', 0.25),
                         min_z_layers=zl_tf.get('min_z_layers', 1),
                         max_cell_z_span=zl_tf.get('max_cell_z_span', 3),
                         max_z_gap=zl_tf.get('max_z_gap', 0)),
        }
        geom = {'dir_dict': dir_dict, 'disp_mat_fin': disp_mat_fin, 'z_start': z_start, 'Z': Z,
                'H': H, 'W': W, 'tile_size': tile_size, 'num_tiles': num_tiles}
        ch_results = run_stage3_channel_pool(
            routing_config, stage3_worker_count(config, len(routing_config)),
            pATH_SRC_CSV, derived['pATH_GLOBAL_2D'], derived['pATH_CHANNEL_3D'], geom, zl_params,
            base_res_path)

        # tile 元数据（给最终细胞找回 tile/slice 名）按通道顺序拼起来，与以前逐通道追加的顺序一致
        meta_chunks = [ch_results[ch['id']]['metadata'] for ch in routing_config
                       if ch_results[ch['id']]['metadata'] is not None]

        soma_ch_ids = [ch['id'] for ch in routing_config
                       if ch.get('type', 'soma') == 'soma' and ch.get('active', True)]
        tf_ch_ids   = [ch['id'] for ch in routing_config
                       if ch.get('type', 'tf')   == 'tf'   and ch.get('active', True)]

        soma_vol_by_ch = {}   # ch_id → volumetric_list (for 3D coloc)
        tf_vol_by_ch   = {}
        for cid in soma_ch_ids + tf_ch_ids:
            pkl_path = os.path.join(derived['pATH_CHANNEL_3D'], f"{cid}_3d_tracked.pkl")
            if os.path.exists(pkl_path):   # 没有检测结果的通道不会生成 pkl，与以前一样跳过
                with open(pkl_path, 'rb') as pf:
                    (soma_vol_by_ch if cid in soma_ch_ids else tf_vol_by_ch)[cid] = pickle.load(pf)

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
        n_multi  = sum(1 for c in merged_soma_vols if len(class_markers(c['class'])) > 1)
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
        tf_markers_set = {channel_marker(cid) for cid in tf_ch_ids}
        n_tf_annotated = sum(
            1 for c in merged_soma_vols
            if tf_markers_set & set(class_markers(c['class']))
        )
        logging.info(f"✔️ [3B] 全部TF标注完成: {n_tf_annotated} 个 soma 有 TF marker")

        # 统一规范 class 字符串：丢掉旧结果（如 "GFP_3" 通道名）带进来的伪 marker。
        # 只参与合并的细胞会经过 _merge_class，单通道细胞不会，所以这里补一次。
        for soma in merged_soma_vols:
            _base, _mk = split_class(soma['class'])
            soma['class'] = f"{_base}_" + "_".join(sorted(_mk)) if _mk else _base

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

        out_coloc = os.path.join(derived['pATH_COLOCALIZATION'], 'coloc_result.csv')
        pd.DataFrame(soma_3d, columns=BOX_COLS).to_csv(out_coloc, index=False)
        logging.info(f"✔️ [3C] 共定位结果: {len(soma_3d)} 个细胞 → {out_coloc}")

        final_results = soma_3d

        # ====== 4. 保存全局 3D 报告 (目标 4) ======
        if final_results is not None and len(final_results) > 0:
            df = pd.DataFrame(final_results, columns=BOX_COLS)

            if meta_chunks:
                meta_coords = np.concatenate([m['coords'] for m in meta_chunks])
                meta_coords[:, 2] *= 10.0
                tree = cKDTree(meta_coords)
                final_coords = np.column_stack((
                    (final_results[:, 0] + final_results[:, 2]) / 2,
                    (final_results[:, 1] + final_results[:, 3]) / 2,
                    final_results[:, 7].astype(float) * 10.0
                ))
                _, indices = tree.query(final_coords)
                # 各通道的 tile/slice 名是编码后传回的，这里只解码被查到的那些
                bounds = np.cumsum([0] + [len(m['coords']) for m in meta_chunks])
                chunk_of = np.searchsorted(bounds, indices, side='right') - 1
                tiles, slices = np.empty(len(indices), dtype=object), np.empty(len(indices), dtype=object)
                for k, m in enumerate(meta_chunks):
                    sel = chunk_of == k
                    local = indices[sel] - bounds[k]
                    tiles[sel] = m['tile_u'][m['tile_c'][local]]
                    slices[sel] = m['slice_u'][m['slice_c'][local]]
                df['tile_name']  = tiles
                df['slice_name'] = slices
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
        #    split_class 同时丢掉旧结果里由 "GFP_3" 这类通道名产生的伪 marker "3"
        parsed_cls = df_final['class'].apply(split_class)
        df_final['base_type']   = parsed_cls.apply(lambda pc: pc[0])
        df_final['marker_set']  = parsed_cls.apply(lambda pc: frozenset(pc[1]))
        df_final['class_clean'] = parsed_cls.apply(
            lambda pc: f"{pc[0]}_" + "_".join(sorted(pc[1])) if pc[1] else pc[0]
        )

        # 统计组合情况 (e.g. neuron_RFP_Sox9: 150个)
        combo_counts = df_final['class_clean'].value_counts()

        # 统计基类情况 (e.g. neuron: 800个, glia: 1200个)
        base_counts = df_final['base_type'].value_counts()

        # 提取所有的 Markers 并独立统计阳性率
        all_markers_found = set()
        for ms in df_final['marker_set']:
            all_markers_found.update(ms)

        marker_counts = {}
        for m in sorted(all_markers_found):
            # 按 marker token 精确匹配，避免 str.contains 的子串误判
            marker_counts[m] = int(df_final['marker_set'].apply(lambda s: m in s).sum())

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
        
        for label, group in df_final.groupby('class_clean'):
            group_sorted = group.sort_values('z')
            out_df = group_sorted[['cx', 'cy', 'z', 'score', 'slice_name', 'tile_name']]
            
            # 净化文件名 (如 neuron_RFP_Sox9 -> ob_neuron_RFP_Sox9.csv)
            safe_label = str(label).replace('/', '_').replace('\\', '_')
            save_path = os.path.join(derived['pATH_CENTROIDS'], f"ob_{safe_label}.csv")
            out_df.to_csv(save_path, index=False)
            
        logging.info(f"已生成所有 {len(combo_counts)} 种子类型的质心文件，保存在: {derived['pATH_CENTROIDS']}")
        logging.info("阶段 4 完成: 全局统计报告生成完毕。")

    logging.info(f"🎉 动态多通道推断全部完成！总耗时: {(time.time() - start_time)/60:.2f} 分钟。")