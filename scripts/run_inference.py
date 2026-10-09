# -*- coding: utf-8 -*-
#python scripts/run_inference.py --config config/config.json
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
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
from src.core.result_layout import migrate_result_layout, result_paths
from src.utils.logger import setup_logging
from src.utils.io import (listTile, listTile_from_local_csvs, loadTeraxml,
                          save_run_metadata, compute_grid_fallback_offsets)
from src.utils.markers import channel_marker, class_markers, split_class
import multiprocessing.connection
from src.core.worker import run_tile_process
from src.core.detection_filter import (
    FILTER_SCHEMA_VERSION, atomic_write_csv, filter_detection_df, resolve_filter_params, source_dir_for_channel,
)
from src.core.stitcher import fuse_dual_intensity_2d
from src.core.channel_stage3 import stitch_and_link_channel
from src.core.coloc_sources import (cell_id, coloc_display_box, colocalization_status,
                                    primary_source_trace, source_3d_json,
                                    write_match_evidence, write_source_3d,
                                    write_source_boxes)
from src.core.provenance import (provenance_columns, file_sha256, file_stamp,
                                 model_fingerprints, atomic_json,
                                 output_manifest_valid, write_output_manifest,
                                 validate_raw_input_manifest)
from src.core.tile_position_pipeline import run_tile_position_stage, tile_position_frame
from src.core.stitcher import (match_soma_3d_iou, merge_soma_volumes_union, soma_match_metrics,
                               annotate_soma_with_tf_containment,
                               _merge_class, suppress_cross_class_overlap)
from src.core.point_cloud_aligner import (
    align_tile, tile_alignment_done,
    resolve_align_settings, check_align_settings, save_align_settings,
    validate_alignment_frame, validate_stitching_xml_frame,
    validate_cached_geometry, validate_measurement_source,
)
from concurrent.futures import ProcessPoolExecutor, as_completed


def align_worker_count(pre_align_cfg, n_tasks):
    """Stage 2.5 worker count; defaults to the allocated CPUs or local CPU count."""
    allocated = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))
    n = pre_align_cfg.get('n_workers') or allocated
    return max(1, min(int(n), n_tasks, allocated))


_THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


class _single_threaded_children:
    """
    Keep each child process on one BLAS/OpenMP thread to avoid oversubscription.
    Spawned children import numpy during startup, so set these variables in the parent.
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
    """Stage 3 channel workers, capped by the allocated CPU count."""
    cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))
    n = config.get('stage3_n_workers') or n_channels
    return max(1, min(int(n), n_channels, cpus))


def run_stage3_channel_pool(routing, n_workers, src_dir, global_2d_dir, channel_3d_dir,
                            geom, zl_params, log_dir):
    """Run global 2D stitching and Z-Link for each channel; return results by channel ID."""
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
    Stage 2.5 aligns one tile per CPU task and returns missing detection CSV paths.
    Abort on any tile error; completed offset JSON files allow a later run to resume.
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


def write_fusion_summary(tile_names, channels, fusion_dir):
    """Aggregate all tile manifests, including tiles skipped during resume."""
    rows = []
    totals = {ch['id']: {'n_low': 0, 'n_high': 0,
                         'n_fused': 0, 'n_matched': 0}
              for ch in channels}
    for tile in tile_names:
        for channel in channels:
            cid = channel['id']
            manifest = os.path.join(
                fusion_dir, f"{tile}_{cid}_fusion_manifest.json")
            with open(manifest, encoding='utf-8') as handle:
                counts = json.load(handle)['metadata']
            row = {'channel': cid, 'tile': tile, **counts}
            row['n_matched'] = counts['n_low'] + counts['n_high'] - counts['n_fused']
            rows.append(row)
            for key in totals[cid]:
                totals[cid][key] += row[key]
    rows.extend({'channel': cid, 'tile': 'TOTAL', **counts}
                for cid, counts in totals.items())
    summary_path = os.path.join(fusion_dir, 'fusion_summary.csv')
    atomic_write_csv(pd.DataFrame(rows), summary_path)
    return summary_path


def filter_input_signature(source, settings):
    return {'source': file_stamp(source, with_hash=True), 'settings': settings}


def prepare_alignment_inputs(config, tile_names, detect_routing, raw_dir, filtered_dir):
    """Filter raw tile detections before estimating pre-align channel shifts."""
    os.makedirs(filtered_dir, exist_ok=True)
    logical_channels = {ch['id']: ch for ch in config['channels_routing']}
    for ch in config['channels_routing']:
        if ch.get('double_exposure'):
            logical_channels[ch['second_intensity_id']] = ch
    jobs = [(tile, ch) for tile in tile_names for ch in detect_routing]
    params_by_channel = {
        ch['id']: resolve_filter_params(config, logical_channels[ch['id']])
        for ch in detect_routing
    }
    filter_signature = {
        "schema_version": FILTER_SCHEMA_VERSION, "params": params_by_channel,
        "filter_code_sha256": file_sha256(os.path.join(
            project_root, 'src', 'core', 'detection_filter.py')),
    }
    settings_file = os.path.join(filtered_dir, "_filter_settings.json")
    try:
        with open(settings_file, encoding="utf-8") as handle:
            same_settings = json.load(handle) == filter_signature
    except (OSError, ValueError):
        same_settings = False
    missing = [os.path.join(raw_dir, f"{tile}_{ch['id']}_result.csv")
               for tile, ch in jobs
               if not os.path.isfile(os.path.join(raw_dir, f"{tile}_{ch['id']}_result.csv"))]
    if missing:
        raise FileNotFoundError(
            f"Pre-align filtering needs {len(missing)} raw detection CSV(s); "
            f"first missing: {missing[0]}")
    for tile, ch in tqdm(jobs, desc="Filter raw tiles for alignment"):
        name = f"{tile}_{ch['id']}_result.csv"
        target = os.path.join(filtered_dir, name)
        rejected_path = os.path.join(
            filtered_dir, f"{tile}_{ch['id']}_rejected.csv")
        source = os.path.join(raw_dir, name)
        manifest_path = os.path.join(
            filtered_dir, f"{tile}_{ch['id']}_filter_manifest.json")
        signature = filter_input_signature(
            source, {'stage': 'pre_align_filter',
                     'params': params_by_channel[ch['id']],
                     'filter_settings': filter_signature})
        if (same_settings and output_manifest_valid(manifest_path, signature)):
            continue
        params = params_by_channel[ch['id']]
        raw = provenance_columns(pd.read_csv(source), tile, ch['id'])
        filtered, stats, rejected = filter_detection_df(
            raw, params, return_stats=True, return_rejected=True, context=source)
        atomic_write_csv(filtered, target)
        atomic_write_csv(rejected, rejected_path)
        write_output_manifest(manifest_path, [target, rejected_path], signature)
        logging.info("[2.25][%s][%s] %s -> %s", tile, ch['id'],
                     stats['before'], stats['after'])

    part = settings_file + ".part"
    with open(part, "w", encoding="utf-8") as handle:
        json.dump(filter_signature, handle, indent=2)
    os.replace(part, settings_file)



def validate_reused_provenance(previous, current, derived, model_hashes):
    """Prevent old checkpoints from being relabeled with changed input settings."""
    raw_dir = derived['pATH_DET_RES']
    tracked_dir = derived['pATH_CHANNEL_3D']
    has_raw = os.path.isdir(raw_dir) and any(
        name.endswith('_result.csv') for name in os.listdir(raw_dir))
    has_tracks = os.path.isdir(tracked_dir) and any(
        name.endswith('_3d_tracked.pkl') for name in os.listdir(tracked_dir))
    if not (has_raw or has_tracks):
        return
    if previous is None:
        raise ValueError("Existing checkpoints lack runtime_config.json provenance")
    if has_raw:
        keys = ('models', 'model_classes', 'channels_routing', 'pipeline_mode')
        changed = [key for key in keys if previous.get(key) != current.get(key)]
        old_detection = dict(previous.get('detection_params') or {})
        new_detection = dict(current.get('detection_params') or {})
        for transient in ('generate_histograms', 'sTARTID', 'eNDID'):
            old_detection.pop(transient, None)
            new_detection.pop(transient, None)
        if old_detection != new_detection:
            changed.append('detection_params')
        old_hashes = (previous.get('provenance') or {}).get('model_sha256')
        def content_only(fingerprints):
            return {name: {key: value for key, value in record.items()
                           if key in ('sha256', 'files', 'configured_value')}
                    for name, record in fingerprints.items()}
        if (old_hashes is not None and
                content_only(old_hashes) != content_only(model_hashes)):
            changed.append('model_content')
        if changed:
            raise ValueError(
                "Raw detections were produced with different settings/model "
                f"content ({', '.join(changed)}). Use a new result directory or "
                "regenerate the affected checkpoints.")
        if old_hashes is None:
            logging.warning("Legacy raw checkpoints have no model-content fingerprint.")
    if has_tracks and previous.get('z_linker') != current.get('z_linker'):
        logging.info("Z-link settings changed; Stage 3 manifest will rebuild tracks.")


def validate_global_checkpoints(derived, tile_names, routing):
    """Report stale channel outputs; Stage 3 will verify and rebuild them."""
    stale = set()
    for ch in routing:
        cid = ch['id']
        source_times = [
            os.path.getmtime(os.path.join(
                derived['pATH_DET_FILTERED'], f"{tile}_{cid}_result.csv"))
            for tile in tile_names
        ]
        newest = max(source_times, default=0)
        global_csv = os.path.join(derived['pATH_GLOBAL_2D'], f"{cid}_2d_global.csv")
        tracked = os.path.join(derived['pATH_CHANNEL_3D'], f"{cid}_3d_tracked.pkl")
        if os.path.isfile(global_csv) and os.path.getmtime(global_csv) < newest:
            stale.add(cid)
        if os.path.isfile(tracked) and (
                not os.path.isfile(global_csv)
                or os.path.getmtime(tracked) < os.path.getmtime(global_csv)):
            stale.add(cid)
    if stale:
        logging.warning("Stage 3 checkpoint(s) stale and scheduled for rebuild: %s",
                        ', '.join(sorted(stale)))
    return stale



def stage4_input_signature(derived, routing, config, geometry_source=None):
    return {
        "channels": [ch["id"] for ch in routing if ch.get("active", True)],
        "geometry_source": (file_stamp(geometry_source, with_hash=True)
                            if geometry_source else None),
        "z_linker": config.get("z_linker", {}),
        "calibration_um": {
            "xy": config.get("detection_params", {}).get("xy_resolution_um", 0.65),
            "z": config.get("detection_params", {}).get("z_resolution_um", 8.0),
        },
        "coloc_code_sha256": {
            name: file_sha256(os.path.join(project_root, name))
            for name in ('scripts/run_inference.py', 'src/core/stitcher.py',
                         'src/core/coloc_sources.py')
        },
        "filtered_inputs": {
            ch["id"]: {
                name: file_stamp(os.path.join(derived["pATH_DET_FILTERED"], name),
                                 with_hash=True)
                for name in sorted(os.listdir(derived["pATH_DET_FILTERED"]))
                if name.endswith(f"_{ch['id']}_result.csv")
            }
            for ch in routing if ch.get("active", True)
        },
        "stage3_manifests": {
            ch["id"]: file_stamp(path, with_hash=True)
            if os.path.isfile(path) else None
            for ch in routing if ch.get("active", True)
            for path in [os.path.join(
                derived["pATH_CHANNEL_3D"], f"{ch['id']}_stage3_manifest.json")]
        },
        "stage3_outputs": {
            ch["id"]: {
                name: file_stamp(path, with_hash=True)
                if os.path.isfile(path) else None
                for name, path in (
                    ('global_2d', os.path.join(derived['pATH_GLOBAL_2D'],
                                               f"{ch['id']}_2d_global.csv")),
                    ('stitch_decisions', os.path.join(derived['pATH_GLOBAL_2D'],
                                                       f"{ch['id']}_stitch_decisions.csv")),
                    ('track_summary', os.path.join(derived['pATH_CHANNEL_3D'],
                                                    f"{ch['id']}_3d_tracked.csv")),
                    ('track_members', os.path.join(derived['pATH_CHANNEL_3D'],
                                                    f"{ch['id']}_track_members.csv")),
                    ('track_rejections', os.path.join(derived['pATH_CHANNEL_3D'],
                                                       f"{ch['id']}_track_rejections.csv")),
                )
            }
            for ch in routing if ch.get("active", True)
        },
        "tracks": {
            ch["id"]: file_stamp(path) if os.path.isfile(path) else None
            for ch in routing if ch.get("active", True)
            for path in [os.path.join(
                derived["pATH_CHANNEL_3D"], f"{ch['id']}_3d_tracked.pkl")]
        },
    }


def run_detection_pool(tasks, gpu_ids, config):
    """Run tile detection with one slot per GPU and a fresh process for each tile.

    Persistent TF/StarDist workers can leak memory during long runs and be killed by OOM.
    Python 3.10 ProcessPoolExecutor lacks max_tasks_per_child; mp.Pool restarts failed
    workers indefinitely, so this stage schedules fresh processes directly.
    Stop remaining workers and fail the job if any child exits unsuccessfully.
    Completed tile CSVs remain available for a resumed run after the issue is fixed.
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
    # Stage 1: load configuration and build and validate output paths.
    # ==========================================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the base configuration as JSON.
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
    tile_position_cfg = config.get('tile_position_params', {})
    solve_tile_positions = pipeline_mode == 'pre_align' and tile_position_cfg.get('enabled', False)
    migrate_result_layout(base_res_path)
    if solve_tile_positions:
        frame = tile_position_frame(config)
        frame_dir_key = next(ch['dir_key'] for ch in config['channels_routing']
                             if ch.get('active', True) and ch['id'] == frame)
        paths['pATHXML'] = os.path.join(paths[frame_dir_key], 'xml_merging.xml')
    derived = result_paths(base_res_path)
    # Attach derived paths for worker.py and downstream stages.
    config['derived_paths'] = derived

    # Create all required directories.
    os.makedirs(base_res_path, exist_ok=True)
    for p_key, p_path in derived.items():
        if not p_path.lower().endswith(('.txt', '.csv')):
            os.makedirs(p_path, exist_ok=True)

    setup_logging(base_res_path)
    logging.info("Starting Multi-Channel Inference...")
    logging.info(f"Device: {device.upper()}, Pipeline mode: {pipeline_mode.upper()}")

    # Resolve channel routing and the anchor channel.
    routing_config = [ch for ch in config.get('channels_routing', []) if ch.get('active', True)]
    if not routing_config:
        raise ValueError("❌ 配置文件中没有任何激活的通道路由 (channels_routing)！")
        
    anchor_ch = routing_config[0]
    anchor_dir = paths.get(anchor_ch['dir_key'])

    # Resolve and validate pre-align settings before detection to catch configuration
    # errors or conflicts with existing alignment results early.
    align_settings = None
    if pipeline_mode == 'pre_align':
        align_settings = resolve_align_settings(config, routing_config)
        align_settings['input_stage'] = 'filtered_2d_v1'
        align_settings['filter_params'] = {
            ch['id']: resolve_filter_params(config, ch) for ch in routing_config
        }
        validate_measurement_source(derived['pATH_ALIGN_OFFSETS'], align_settings)
        check_align_settings(derived['pATH_ALIGN_OFFSETS'], align_settings)
        validate_stitching_xml_frame(paths.get('pATHXML'), align_settings['stitching_reference_channel'])
        logging.info(f"Stitching frame: {align_settings['stitching_reference_channel']}")
        logging.info(f"Pre-align 参考通道: {align_settings['reference_channel']}，"
                     f"TF 对齐方式: {align_settings['tf_align_mode']}")

    # Expand double-exposure channels for Stage 2 detection.
    # Later stages use the original routing configuration because fusion
    # represents both exposures as one logical channel.
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

    # List tiles from the anchor channel.
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

    previous_path = os.path.join(base_res_path, 'runtime_config.json')
    previous = load_config(previous_path) if os.path.isfile(previous_path) else None
    model_hashes = model_fingerprints(config.get('models'), project_root)
    validate_reused_provenance(previous, config, derived, model_hashes)
    if align_settings is not None:
        validate_cached_geometry(previous, config, base_res_path, align_settings)

    save_run_metadata(config, start_time, model_hashes=model_hashes)

    # Select the requested tile range.
    sTARTID = dp.get('sTARTID') or 1
    eNDID = dp.get('eNDID') or len(pATHTILE_all)
    target_indices = list(range(sTARTID - 1, eNDID))
    pATHTILE = [pATHTILE_all[i] for i in target_indices]

    # TeraStitcher XML is needed only for global stitching in Stage 3.
    # Detection and tile-level alignment/filtering can run before stitching finishes.

    # ==========================================
    # Stage 2: tile-level detection with checkpoints.
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
        if start_from_stage >= 2:
            raise FileNotFoundError(
                f"start_from_stage={start_from_stage} reuses raw detections in "
                f"{derived['pATH_DET_RES']}; {len(tasks_to_run)} tile(s) have missing "
                "CSV files. Complete Stage 2 before resuming.")
        num_gpus = torch.cuda.device_count()
        num_processes = max(1, num_gpus)
        logging.info(f"阶段 2: 发现 {len(tasks_to_run)} 个缺失结果，启动 {num_processes} 个进程 ({num_gpus} GPU)...")
        # Without a GPU, init_worker falls back to config['device'].
        gpu_ids = list(range(num_gpus)) if num_gpus > 0 else [None]
        run_detection_pool(tasks_to_run, gpu_ids, config)
    else:
        logging.info("✔️ Checkpoint 1 达成: 所有 Tile 检测完成。")

    # Verify newly generated raw checkpoints before downstream reuse. Legacy
    # CSVs without manifests remain usable, but their TIFF content is unknown.
    for _tile_path in pATHTILE:
        _tile = os.path.basename(_tile_path)
        for _ch in detect_routing_config:
            _prefix = f"{_tile}_{_ch['id']}"
            _raw_csv = os.path.join(derived['pATH_DET_RES'], _prefix + '_result.csv')
            _raw_manifest = os.path.join(derived['pATH_DET_RES'], _prefix + '_inputs.json')
            if os.path.isfile(_raw_manifest):
                validate_raw_input_manifest(_raw_manifest, _raw_csv)
            else:
                logging.warning("Legacy raw CSV has no TIFF/input manifest: %s", _raw_csv)

    if config.get('stop_after_detection', False):
        logging.info("🛑 stop_after_detection=true：Stage 1 完成，正常退出。Stage 2~5 可在 CPU 或单 GPU 上单独运行。")
        sys.exit(0)

    # ==========================================
    # Stage 2.5: align channel point clouds in pre-align mode.
    # ==========================================
    if pipeline_mode == 'pre_align':
        prepare_alignment_inputs(
            config, [os.path.basename(p) for p in pATHTILE_all],
            detect_routing_config, derived['pATH_DET_RES'],
            derived['pATH_DET_PREALIGN_FILTERED'])

    if pipeline_mode == 'pre_align':
        align_done_flag = os.path.join(derived['pATH_ALIGN_OFFSETS'], "_align_done.flag")
        if (os.path.exists(align_done_flag) and all(
                tile_alignment_done(os.path.basename(p),
                                    derived['pATH_DET_PREALIGN_FILTERED'],
                                    derived['pATH_ALIGN_OFFSETS'], routing_config)
                for p in pATHTILE_all)):
            logging.info("✔️ Checkpoint 2.5 达成: 通道点云对齐已完成，直接读取对齐结果。")
        else:
            logging.info("阶段 2.5: 开始点云通道对齐 (pre_align 模式)...")
            os.makedirs(derived['pATH_ALIGN_OFFSETS'], exist_ok=True)

            save_align_settings(derived['pATH_ALIGN_OFFSETS'], align_settings)

            routing_cfg_align = [ch for ch in config.get('channels_routing', []) if ch.get('active', True)]

            # Skip completed tiles; the offsets JSON is written last as a completion marker.
            todo = [p for p in pATHTILE
                    if not tile_alignment_done(os.path.basename(p), derived['pATH_DET_PREALIGN_FILTERED'],
                                               derived['pATH_ALIGN_OFFSETS'], routing_cfg_align)]
            n_workers = align_worker_count(pre_align_cfg, len(todo))
            logging.info(f"  [2.5] {len(pATHTILE) - len(todo)} 个 tile 已完成，剩余 {len(todo)} 个，"
                         f"{n_workers} 个 CPU 进程并行（纯 CPU 计算，不占 GPU）")
            missing_csvs = run_align_pool(todo, n_workers, derived['pATH_DET_PREALIGN_FILTERED'],
                                          derived['pATH_ALIGN_OFFSETS'], routing_cfg_align, align_settings)

            if missing_csvs:
                raise RuntimeError(
                    f"❌ [2.5] {len(missing_csvs)} 个检测 CSV 缺失，对齐结果不完整，未写完成标记。"
                    f"请先补齐 Stage 2 检测后重跑。示例: {missing_csvs[:3]}"
                )
            # Write the completion marker.
            open(align_done_flag, 'w').close()
            logging.info("✔️ [2.5] 所有 Tile 点云对齐完成。")

    # ==========================================
    # Stage 2.6: dual-exposure intensity fusion.
    # Input: aligned filtered CSVs (pre_align) or raw CSVs (post_align).
    # Fusion feeds the final filtered tile directory used by Stage 3.
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

        def _fusion_signature(tile, channel):
            ids = (channel['id'], channel['second_intensity_id'])
            paths = [os.path.join(_fusion_src, f"{tile}_{cid}_result.csv")
                     for cid in ids]
            missing = [path for path in paths if not os.path.isfile(path)]
            if missing:
                raise FileNotFoundError(
                    f"Dual-exposure fusion requires both inputs; missing {missing}")
            return {
                'channel': channel['id'],
                'iou_threshold': channel.get('fusion_iou_thresh', 0.3),
                'inputs': {path: file_stamp(path, with_hash=True)
                           for path in paths},
                'fusion_code_sha256': file_sha256(os.path.join(
                    project_root, 'src', 'core', 'stitcher.py')),
            }

        def _fusion_manifest(tile, channel):
            return os.path.join(
                _fusion_dst, f"{tile}_{channel['id']}_fusion_manifest.json")

        def _fusion_ready(tile, channel):
            output = os.path.join(
                _fusion_dst, f"{tile}_{channel['id']}_result.csv")
            return output_manifest_valid(
                _fusion_manifest(tile, channel),
                _fusion_signature(tile, channel))

        _fusion_done = all(
            _fusion_ready(tn, ch)
            for tn in _tile_names_all for ch in _de_channels
        )
        if _fusion_done:
            logging.info("✔️ Checkpoint 2.6 达成: 融合后 tile CSV 已全部存在。")
        else:
            logging.info(f"阶段 2.6: 融合 {len(_de_channels)} 个双曝光通道 ...")
            for _tn in tqdm(_tile_names_all, desc="Fuse dual-intensity tiles"):
                for ch in _de_channels:
                    ch_id = ch['id']
                    second_id = ch['second_intensity_id']
                    out_csv = os.path.join(_fusion_dst, f"{_tn}_{ch_id}_result.csv")
                    if _fusion_ready(_tn, ch):
                        continue
                    low_csv  = os.path.join(_fusion_src, f"{_tn}_{ch_id}_result.csv")
                    high_csv = os.path.join(_fusion_src, f"{_tn}_{second_id}_result.csv")
                    low_df  = pd.read_csv(low_csv)  if os.path.isfile(low_csv)  else pd.DataFrame(columns=[
                        "slice_name", "x1", "y1", "x2", "y2", "class", "score", "mean", "z"])
                    high_df = pd.read_csv(high_csv) if os.path.isfile(high_csv) else pd.DataFrame(columns=[
                        "slice_name", "x1", "y1", "x2", "y2", "class", "score", "mean", "z"])
                    low_df = provenance_columns(low_df, _tn, ch_id)
                    high_df = provenance_columns(high_df, _tn, second_id)

                    fused_df, n_low, n_high, n_fused = fuse_dual_intensity_2d(
                        low_df, high_df, iou_thresh=ch.get('fusion_iou_thresh', 0.3)
                    )
                    atomic_write_csv(fused_df, out_csv)
                    write_output_manifest(
                        _fusion_manifest(_tn, ch), [out_csv],
                        _fusion_signature(_tn, ch),
                        metadata={'n_low': n_low, 'n_high': n_high,
                                  'n_fused': n_fused})

                    n_matched = n_low + n_high - n_fused
                    logging.info(f"  [2.6][{ch_id}] tile={_tn}: low={n_low} high={n_high} "
                                 f"-> fused={n_fused} (matched={n_matched})")

        write_fusion_summary(_tile_names_all, _de_channels, _fusion_dst)


    # ==========================================
    # Stage 2.75: publish filtered, aligned tile CSVs for Stage 3.
    # Pre-align copies already-filtered single channels; fused channels are filtered here.
    # ==========================================
    _filter_params_by_channel = {
        ch['id']: resolve_filter_params(config, ch) for ch in routing_config
    }
    _filter_signature = {
        'schema_version': FILTER_SCHEMA_VERSION,
        'pipeline_mode': pipeline_mode,
        'input_stage': 'aligned_filtered_v1' if pipeline_mode == 'pre_align' else 'raw_v1',
        'params': _filter_params_by_channel,
        'filter_code_sha256': file_sha256(os.path.join(
            project_root, 'src', 'core', 'detection_filter.py')),
    }
    _filter_dst = derived['pATH_DET_FILTERED']
    os.makedirs(_filter_dst, exist_ok=True)
    _filter_settings_file = os.path.join(_filter_dst, "_filter_settings.json")
    try:
        with open(_filter_settings_file, encoding="utf-8") as handle:
            _same_filter_settings = json.load(handle) == _filter_signature
    except (OSError, ValueError):
        _same_filter_settings = False
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

    def _filtered_checkpoint_ready(tile, channel, source_dir):
        ch_id = channel['id']
        target = os.path.join(_filter_dst, f"{tile}_{ch_id}_result.csv")
        rejected = os.path.join(_filter_dst, f"{tile}_{ch_id}_rejected.csv")
        source = os.path.join(source_dir, f"{tile}_{ch_id}_result.csv")
        manifest = os.path.join(_filter_dst, f"{tile}_{ch_id}_filter_manifest.json")
        signature = filter_input_signature(
            source, {'stage': 'final_filter', 'channel': ch_id,
                     'params': _filter_params_by_channel[ch_id],
                     'filter_settings': _filter_signature})
        return (os.path.isfile(target) and os.path.isfile(rejected)
                and output_manifest_valid(manifest, signature))

    _filter_done = _same_filter_settings and bool(_expected_filter_inputs) and all(
        _filtered_checkpoint_ready(_tn, _ch, _src)
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
            _rejected_path = os.path.join(
                _filter_dst, f"{_tn}_{_ch_id}_rejected.csv")
            _manifest_path = os.path.join(
                _filter_dst, f"{_tn}_{_ch_id}_filter_manifest.json")
            _signature = filter_input_signature(
                _in_csv, {'stage': 'final_filter', 'channel': _ch_id,
                          'params': _filter_params_by_channel[_ch_id],
                          'filter_settings': _filter_signature})
            if (_same_filter_settings and
                    _filtered_checkpoint_ready(_tn, _ch, _src)):
                continue
            if pipeline_mode == 'pre_align' and not _ch.get('double_exposure'):
                if os.path.abspath(_in_csv) != os.path.abspath(_out_csv):
                    part = _out_csv + '.part'
                    shutil.copyfile(_in_csv, part)
                    os.replace(part, _out_csv)
                empty_rejected = pd.read_csv(_in_csv, nrows=0)
                empty_rejected['rejection_reason'] = pd.Series(dtype=str)
                atomic_write_csv(empty_rejected, _rejected_path)
                write_output_manifest(
                    _manifest_path, [_out_csv, _rejected_path], _signature)
                continue
            _params = _filter_params_by_channel[_ch_id]
            _input_df = provenance_columns(
                pd.read_csv(_in_csv), _tn, _ch_id)
            _filtered_df, _stats, _rejected_df = filter_detection_df(
                _input_df, _params, return_stats=True, return_rejected=True,
                context=f"{_in_csv} ({_ch_id})",
            )
            atomic_write_csv(_filtered_df, _out_csv)
            atomic_write_csv(_rejected_df, _rejected_path)
            write_output_manifest(
                _manifest_path, [_out_csv, _rejected_path], _signature)
            _n_filtered_total += _stats["removed_total"]
            logging.info(
                "[2.75][%s][%s] %s -> %s (removed=%s; score_min_removed=%s; params=%s)",
                _tn, _ch_id, _stats["before"], _stats["after"],
                _stats["removed_total"], _stats["removed"]["score_min"], _params,
            )
        logging.info("[2.75] filtered %s box(es).", _n_filtered_total)
        part = _filter_settings_file + ".part"
        with open(part, "w", encoding="utf-8") as handle:
            json.dump(_filter_signature, handle, indent=2)
        os.replace(part, _filter_settings_file)

    pATH_SRC_CSV = _filter_dst

    # ==========================================
    # Stage 2.8: generate raw 2D histograms of intensity and area.
    # Input: _filter_src (raw CSV).
    # Output: 1_2d_raw/histograms/{tile}_{ch}_hist.png.
    # Delete the output directory to rebuild histograms without affecting later stages.
    # Controlled by detection_params.generate_histograms (default: true).
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
    # Tile-level stages do not need TeraStitcher XML.
    # Set stop_before_stitching=true to stop here while stitching is unfinished.
    # After stitching, set it to false; checkpoints resume at Stage 3.
    # Stage 2.9: optionally solve tile geometry and publish XMLs before Stage 3.
    # The stop point below leaves the generated XMLs ready for image merging.
    if solve_tile_positions:
        report_dir = derived['pATH_TILE_POSITIONS']
        solver_script = os.path.join(project_root, 'src', 'core', 'solve_tile_positions.py')
        xml_script = os.path.join(project_root, 'src', 'core', 'generate_merging_xml.py')
        frame_xml, recomputed = run_tile_position_stage(
            config, os.path.abspath(args.config), base_res_path,
            derived['pATH_DET_PREALIGN_FILTERED'], derived['pATH_ALIGN_OFFSETS'],
            report_dir, [os.path.basename(p) for p in pATHTILE_all],
            routing_config, solver_script, xml_script)
        paths['pATHXML'] = frame_xml
        logging.info("Stage 2.9 tile positions %s: %s",
                     "computed" if recomputed else "reused", frame_xml)

    if config.get('stop_before_stitching', False):
        logging.info("stop_before_stitching=true: tile stages and enabled tile "
                     "position solving are complete; exiting before Stage 3.")
        sys.exit(0)

    # Load TeraStitcher XML.
    # Use generated frame XML for tile-position solving; otherwise use configured paths.
    #
    # Explicit paths matter because stitching offsets may come from a reference channel
    # whose XML is absent from detection folders or replaced by an old zero-offset XML.
    # Cell coordinates must use the XML that merged the registered whole-brain image.
    tile_size = dp.get('tILESIZE', 2048)
    xml_candidates = []
    if paths.get('pATHXML'):
        xml_candidates.append(paths['pATHXML'])
    if not solve_tile_positions:
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
        # Fall back to grid offsets inferred from tile directory names.
        # Uniform spacing lacks tile-specific TeraStitcher offsets and may introduce
        # errors of tens of pixels per step, accumulating to hundreds across the grid.
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
    # Stage 3: global stitching, Z-Link, and colocalization with checkpoints.
    # ==========================================
    # Image stitching uses the per-channel XMLs produced above. It is separate
    # from Stage 3, which stitches cell coordinates rather than raw pixels.
    if config.get('direct_stitching', {}).get('enabled', False):
        from src.core.stitch_raw_tiles import run as run_direct_stitching
        logging.info("Direct image stitching enabled; stitching raw channels before Stage 3.")
        run_direct_stitching(Path(args.config), skip_completed=True)

    stale_stage3_channels = validate_global_checkpoints(
        derived, [os.path.basename(p) for p in pATHTILE_all], routing_config)
    bbox_path = os.path.join(derived['pATH_COLOCALIZATION'], "coloc_result.csv")
    source_3d_path = os.path.join(
        derived['pATH_COLOCALIZATION'], 'coloc_source_3d.csv')
    source_boxes_path = os.path.join(
        derived['pATH_COLOCALIZATION'], 'coloc_source_boxes.csv')
    manifest_path = os.path.join(
        derived['pATH_COLOCALIZATION'], '_provenance_manifest.json')
    final_results = None
    source_checkpoint_ready = (not stale_stage3_channels and output_manifest_valid(
        manifest_path, stage4_input_signature(derived, routing_config, config, pATHxml)))
    if source_checkpoint_ready:
        columns = set(pd.read_csv(bbox_path, nrows=0).columns)
        source_checkpoint_ready = {
            'coloc_id', 'source_3d', 'source_trace_status',
            'cx', 'cy', 'cz', 'x1_3d', 'y1_3d', 'x2_3d', 'y2_3d',
            'z_min', 'z_max', 'bounds_method', 'soma_status', 'tf_status'
        } <= columns
    if os.path.isfile(bbox_path) and not source_checkpoint_ready:
        logging.info("Stage 4 provenance checkpoint incomplete or stale; rebuilding.")

    if source_checkpoint_ready:
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

        # Run global 2D stitching and Z-Link independently for each channel.
        # Existing checkpoints: 5_2d_global/<ch>_2d_global.csv and 6_3d_global/<ch>_3d_tracked.pkl.
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
                'H': H, 'W': W, 'tile_size': tile_size, 'num_tiles': num_tiles,
                'xml_source': pATHxml}
        ch_results = run_stage3_channel_pool(
            routing_config, stage3_worker_count(config, len(routing_config)),
            pATH_SRC_CSV, derived['pATH_GLOBAL_2D'], derived['pATH_CHANNEL_3D'], geom, zl_params,
            base_res_path)

        soma_ch_ids = [ch['id'] for ch in routing_config
                       if ch.get('type', 'soma') == 'soma' and ch.get('active', True)]
        tf_ch_ids   = [ch['id'] for ch in routing_config
                       if ch.get('type', 'tf')   == 'tf'   and ch.get('active', True)]

        soma_vol_by_ch = {}   # ch_id → volumetric_list (for 3D coloc)
        tf_vol_by_ch   = {}
        for cid in soma_ch_ids + tf_ch_ids:
            pkl_path = os.path.join(derived['pATH_CHANNEL_3D'], f"{cid}_3d_tracked.pkl")
            if os.path.exists(pkl_path):   # Skip channels without detections and a pickle.
                with open(pkl_path, 'rb') as pf:
                    (soma_vol_by_ch if cid in soma_ch_ids else tf_vol_by_ch)[cid] = pickle.load(pf)

        # Keep a snapshot of each channel track before class labels are merged.
        for cid, volumes in soma_vol_by_ch.items():
            for cell in volumes:
                cell['source_tracks'] = [(cid, cell.copy())]

        # ====== 3. 3D Colocalization ======
        # Phase A: match soma volumes by 3D IoU and merge them into the main list.
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
                iou, iomin = soma_match_metrics(a_cell, b_cell, z_pad_3d)
                a_cell.setdefault('match_evidence', []).append({
                    'match_type': 'soma_iou',
                    'channel': cid_b,
                    'track_id': b_cell.get('track_id', ''),
                    'iou': iou, 'iomin': iomin,
                    'iou_threshold': iou_thresh_3d,
                    'iomin_threshold': iomin_thresh_3d,
                    'z_pad': z_pad_3d,
                })
                merge_soma_volumes_union(a_cell, b_cell)
                a_cell['source_tracks'].extend(b_cell['source_tracks'])
                a_cell['class'] = _merge_class(a_cell['class'], b_cell['class'])
            merged_soma_vols = merged_soma_vols + unmatched_b
        cross_iou = zl_soma.get('cross_class_iou_thresh', 0.5)
        decision_rows = []
        filtered_tf_objects = set()
        merged_soma_vols = suppress_cross_class_overlap(
            merged_soma_vols, iou_thresh=cross_iou, z_pad=z_pad_3d,
            decision_rows=decision_rows
        )
        n_multi  = sum(1 for c in merged_soma_vols if len(class_markers(c['class'])) > 1)
        n_single = len(merged_soma_vols) - n_multi
        logging.info(f"✔️ [3A] Soma 3D IoU 匹配: {len(merged_soma_vols)} 个 "
                     f"(多阳性 {n_multi}, 单阳性 {n_single})")

        # Phase B: annotate soma with TF boxes fully contained by soma boxes.
        xy_margin          = zl_tf.get('containment_xy_margin', 0)
        z_pad_tf           = zl_tf.get('containment_z_pad', 2)
        max_center_dist    = zl_tf.get('max_center_dist_ratio', 0.5)
        tf_bbox_max_w = zl_tf.get('bbox_max_w', None)
        tf_bbox_max_h = zl_tf.get('bbox_max_h', None)
        for cid in tf_ch_ids:
            tf_vols = tf_vol_by_ch.get(cid, [])
            if (tf_bbox_max_w is not None or tf_bbox_max_h is not None) and tf_vols:
                n_before = len(tf_vols)
                old_tf_vols = tf_vols
                tf_vols = [
                    v for v in tf_vols
                    if (tf_bbox_max_w is None or v['x2_3d'] - v['x1_3d'] <= tf_bbox_max_w)
                    and (tf_bbox_max_h is None or v['y2_3d'] - v['y1_3d'] <= tf_bbox_max_h)
                ]
                accepted_objects = {id(v) for v in tf_vols}
                for rejected_tf in old_tf_vols:
                    if id(rejected_tf) not in accepted_objects:
                        filtered_tf_objects.add(id(rejected_tf))
                        decision_rows.append({
                            'stage': 'tf_size_filter', 'channel': cid,
                            'track_id': rejected_tf.get('track_id', ''),
                            'other_track_id': '', 'decision': 'rejected_tf',
                            'reason': 'bbox_max_size', 'metric': '',
                            'threshold': f"{tf_bbox_max_w},{tf_bbox_max_h}",
                        })
                logging.info(f"  [{cid}] TF size filter: {n_before} → {len(tf_vols)} "
                             f"(bbox_max_w={tf_bbox_max_w}, bbox_max_h={tf_bbox_max_h})")
            if tf_vols and merged_soma_vols:
                merged_soma_vols = annotate_soma_with_tf_containment(
                    merged_soma_vols, tf_vols, z_pad=z_pad_tf, xy_margin=xy_margin,
                    max_center_dist_ratio=max_center_dist, source_channel=cid
                )
                logging.info(f"✔️ [3B] [{cid}] TF containment 标注完成")
        matched_tf_objects = {
            id(source) for soma in merged_soma_vols
            for channel, source in soma.get('source_tracks', ())
            if channel in tf_ch_ids
        }
        tf_markers_set = {channel_marker(cid) for cid in tf_ch_ids}
        n_tf_annotated = sum(
            1 for c in merged_soma_vols
            if tf_markers_set & set(class_markers(c['class']))
        )
        logging.info(f"✔️ [3B] 全部TF标注完成: {n_tf_annotated} 个 soma 有 TF marker")

        # Normalize class labels to remove pseudo markers from old channel names.
        # Merged cells already pass through _merge_class; normalize single-channel cells here.
        for soma in merged_soma_vols:
            _base, _mk = split_class(soma['class'])
            soma['class'] = f"{_base}_" + "_".join(sorted(_mk)) if _mk else _base

        # Phase C: output center-z 2D boxes and exclude TF-only detections.
        output_rows = []
        for soma in merged_soma_vols:
            bbox, center_z = coloc_display_box(soma)
            output_rows.append([
                bbox[0], bbox[1], bbox[2], bbox[3],
                soma['score'], soma['mean'], soma['class'], center_z
            ])

        soma_3d = (np.array(output_rows, dtype=object)
                   if output_rows else np.empty((0, 8), dtype=object))

        xy_um = float(dp.get('xy_resolution_um', 0.65))
        z_um = float(dp.get('z_resolution_um', 8.0))
        out_coloc = os.path.join(derived['pATH_COLOCALIZATION'], 'coloc_result.csv')
        write_source_boxes(source_boxes_path, merged_soma_vols)
        write_source_3d(source_3d_path, merged_soma_vols, xy_um, z_um)
        write_match_evidence(
            os.path.join(derived['pATH_COLOCALIZATION'], 'coloc_match_evidence.csv'),
            merged_soma_vols)
        decisions_path = os.path.join(
            derived['pATH_COLOCALIZATION'], 'coloc_decisions.csv')
        with open(decisions_path + '.part', 'w', newline='', encoding='utf-8') as handle:
            columns = ['stage', 'channel', 'track_id', 'other_track_id',
                       'decision', 'reason', 'metric', 'threshold']
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(decision_rows)
            for cid in tf_ch_ids:
                for tf_track in tf_vol_by_ch.get(cid, []):
                    if (id(tf_track) not in matched_tf_objects
                            and id(tf_track) not in filtered_tf_objects):
                        writer.writerow({
                            'stage': 'tf_containment', 'channel': cid,
                            'track_id': tf_track.get('track_id', ''),
                            'other_track_id': '', 'decision': 'unmatched_tf',
                            'reason': 'no_accepted_soma', 'metric': '',
                            'threshold': max_center_dist,
                        })
        os.replace(decisions_path + '.part', decisions_path)
        logging.info(f"✔️ [3C] 共定位结果: {len(soma_3d)} 个细胞 → {out_coloc}")

        final_results = soma_3d

        # Save the global 3D report.
        if final_results is not None and len(final_results) > 0:
            df = pd.DataFrame(final_results, columns=BOX_COLS)
            df['coloc_id'] = [cell_id(soma, index)
                              for index, soma in enumerate(merged_soma_vols)]
            df['source_3d'] = [source_3d_json(soma, xy_um, z_um)
                               for soma in merged_soma_vols]
            for field in ('cx', 'cy', 'cz', 'x1_3d', 'y1_3d',
                          'x2_3d', 'y2_3d', 'z_min', 'z_max',
                          'bounds_method'):
                df[field] = [soma.get(field) for soma in merged_soma_vols]
            status_rows = [colocalization_status(soma, soma_ch_ids, tf_ch_ids)
                           for soma in merged_soma_vols]
            for field in ('soma_positive_channels', 'soma_positive_count',
                          'soma_status', 'tf_positive_channels',
                          'tf_positive_count', 'tf_status'):
                df[field] = [row[field] for row in status_rows]
            df['cx_um'] = [float(soma['cx']) * xy_um for soma in merged_soma_vols]
            df['cy_um'] = [float(soma['cy']) * xy_um for soma in merged_soma_vols]
            df['cz_um'] = [(float(soma['cz']) - 1) * z_um
                           for soma in merged_soma_vols]
            df['xy_um_per_px'] = xy_um
            df['z_um_per_slice'] = z_um

            traces = [primary_source_trace(soma) for soma in merged_soma_vols]
            df['tile_name'] = [trace[0] for trace in traces]
            df['slice_name'] = [trace[1] for trace in traces]
            df['source_trace_status'] = [trace[2] for trace in traces]

            # Save the full checkpoint CSV for later stages.
            df.to_csv(bbox_path + '.part', index=False)
            os.replace(bbox_path + '.part', bbox_path)
            logging.info(f"✔️ 已输出 目标4 checkpoint: {bbox_path}")

            # Save one CSV per cell type.
            class_paths = []
            for cls, cls_df in df.groupby('class'):
                safe_cls = str(cls).replace('/', '_').replace('\\', '_')
                class_path = os.path.join(
                    derived['pATH_COLOCALIZATION'], f"{safe_cls}.csv")
                cls_df.to_csv(class_path + '.part', index=False)
                os.replace(class_path + '.part', class_path)
                class_paths.append(class_path)
            logging.info(f"✔️ 已输出 目标4 ({df['class'].nunique()} 种细胞类型) → {derived['pATH_COLOCALIZATION']}")
        else:
            class_paths = []
            pd.DataFrame(columns=[*BOX_COLS, 'coloc_id', 'source_3d',
                                  'cx', 'cy', 'cz', 'x1_3d', 'y1_3d',
                                  'x2_3d', 'y2_3d', 'z_min', 'z_max',
                                  'bounds_method',
                                  'soma_positive_channels', 'soma_positive_count',
                                  'soma_status', 'tf_positive_channels',
                                  'tf_positive_count', 'tf_status',
                                  'tile_name', 'slice_name',
                                  'source_trace_status', 'cx_um', 'cy_um', 'cz_um',
                                  'xy_um_per_px', 'z_um_per_slice']).to_csv(
                out_coloc + '.part', index=False)
            os.replace(out_coloc + '.part', out_coloc)
            logging.warning("⚠️ 全局未检测到任何 3D 目标。")

        write_output_manifest(
            manifest_path,
            [bbox_path, source_3d_path, source_boxes_path,
             os.path.join(derived['pATH_COLOCALIZATION'],
                          'coloc_match_evidence.csv'),
             decisions_path, *class_paths],
            stage4_input_signature(derived, routing_config, config, pATHxml))

    # ==========================================
    # Stage 4: generate analysis statistics and centroids.
    # ==========================================
    report_path = os.path.join(derived['pATH_COLOCALIZATION'], "global_summary_statistics.csv")

    if final_results is not None and len(final_results) > 0:
        df_final = pd.read_csv(bbox_path)
        total_cells = len(df_final)

        # Split class labels into base types and marker names.
        # split_class also drops pseudo markers from old names such as GFP_3.
        parsed_cls = df_final['class'].apply(split_class)
        df_final['base_type']   = parsed_cls.apply(lambda pc: pc[0])
        df_final['marker_set']  = parsed_cls.apply(lambda pc: frozenset(pc[1]))
        df_final['class_clean'] = parsed_cls.apply(
            lambda pc: f"{pc[0]}_" + "_".join(sorted(pc[1])) if pc[1] else pc[0]
        )

        # Count marker combinations.
        combo_counts = df_final['class_clean'].value_counts()

        # Count base cell types.
        base_counts = df_final['base_type'].value_counts()

        # Collect markers and calculate positivity independently.
        all_markers_found = set()
        for ms in df_final['marker_set']:
            all_markers_found.update(ms)

        marker_counts = {}
        for m in sorted(all_markers_found):
            # Match exact marker tokens to avoid substring matches.
            marker_counts[m] = int(df_final['marker_set'].apply(lambda s: m in s).sum())

        # Write the detailed analysis report.
        if config.get('generate_analysis_report', False):
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

        # Calculate centroids for each final class.
        df_final['cx'] = (df_final['x1'] + df_final['x2']) / 2
        df_final['cy'] = (df_final['y1'] + df_final['y2']) / 2
        df_final['cx_um'] = df_final['cx'] * df_final['xy_um_per_px']
        df_final['cy_um'] = df_final['cy'] * df_final['xy_um_per_px']
        df_final['cz_um'] = (df_final['z'] - 1) * df_final['z_um_per_slice']
        
        for label, group in df_final.groupby('class_clean'):
            group_sorted = group.sort_values('z')
            out_df = group_sorted[['cx', 'cy', 'z', 'cx_um', 'cy_um', 'cz_um',
                                   'score', 'slice_name', 'tile_name',
                                   'coloc_id', 'source_trace_status']]
            
            # Sanitize labels for output filenames.
            safe_label = str(label).replace('/', '_').replace('\\', '_')
            save_path = os.path.join(derived['pATH_CENTROIDS'], f"ob_{safe_label}.csv")
            out_df.to_csv(save_path, index=False)
            
        logging.info(f"已生成所有 {len(combo_counts)} 种子类型的质心文件，保存在: {derived['pATH_CENTROIDS']}")
        logging.info("Stage 4 complete: cell centroids written.")

    logging.info(f"🎉 动态多通道推断全部完成！总耗时: {(time.time() - start_time)/60:.2f} 分钟。")