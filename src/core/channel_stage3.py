# -*- coding: utf-8 -*-
"""
Stage 3 的单通道部分：全局 2D 拼接 → Z-Link → 落盘。

各通道之间互不依赖，run_inference.py 把每个通道交给一个进程并行执行；跨通道的
3D 共定位（3A/3B/3C）仍在主进程里做。每一步都保留原来的 checkpoint 语义：
  5_2d_global/<ch>_2d_global.csv  存在 → 不再拼接；tile/slice 溯源信息从它的 tile_name/slice_name 列恢复
  6_3d_global/<ch>_3d_tracked.pkl     存在 → 不再 Z-Link
"""

import csv
import logging
import os
import pickle

import numpy as np
import pandas as pd

from src.core.stitcher import combine_predictions
from src.core.z_linker import run_z_linker
from src.utils.markers import channel_marker
from src.core.provenance import (stable_id, file_stamp, file_sha256,
                                 output_manifest_valid, write_output_manifest)

BOX_COLS = ["x1", "y1", "x2", "y2", "score", "mean", "class", "z"]
TRACE_COLS = ["tile_name", "slice_name"]   # 全局 2D CSV 里每个框的来源，续跑时据此恢复溯源信息


def _stitch_channel(ch_id, src_dir, geom, metadata):
    """
    把一个通道所有 tile 的 CSV 拼成全局 2D 矩阵（与原先 run_inference 里的逻辑一致）。
    返回 (mat, trace)：trace 是与 mat 逐行对应的 (tile_name, slice_name)。
    """
    parts, trace, source_ids, decisions = [], [], [], []
    if geom['num_tiles'] > 1:
        stitched = [[np.empty((0, 8)) for _ in range(2)] for _ in range(geom['Z'])]
        row_meta, provenance_meta = {}, {}
        for dir_name, pos in geom['dir_dict'].items():
            tile_name = os.path.split(dir_name)[-1]
            csv_tile = os.path.join(src_dir, f"{tile_name}_{ch_id}_result.csv")
            if os.path.isfile(csv_tile):
                with open(csv_tile, newline='', encoding='utf-8') as tile_file:
                    stitched = combine_predictions(
                        stitched, csv.DictReader(tile_file), None, geom['z_start'], geom['Z'],
                        pos, geom['disp_mat_fin'], (geom['H'], geom['W']),
                        metadata, tile_name, tILESIZE=geom['tile_size'], row_meta=row_meta,
                        provenance_meta=provenance_meta, decision_meta=decisions,
                    )
        for zi, layer in enumerate(stitched):
            for ti, group in enumerate(layer):
                if group.size > 0:
                    parts.append(group)
                    trace.extend(row_meta[(zi, ti)])
                    source_ids.extend(provenance_meta[(zi, ti)])
    else:
        csv_list = [f for f in os.listdir(src_dir) if f.endswith(f'_{ch_id}_result.csv')]
        if csv_list:
            tile_name = csv_list[0].split(f"_{ch_id}_")[0]
            with open(os.path.join(src_dir, csv_list[0]), 'r', encoding='utf-8') as f:
                rows = []
                for row_number, r in enumerate(csv.DictReader(f)):
                    x1, y1, x2, y2 = (float(r[k]) for k in ('x1', 'y1', 'x2', 'y2'))
                    score, mean, c, z = (float(r['score']), float(r['mean']),
                                         str(r['class']), int(float(r['z'])))
                    rows.append([x1, y1, x2, y2, score, mean, c, z])
                    metadata.append([(x1 + x2) / 2, (y1 + y2) / 2, z, tile_name, r['slice_name']])
                    trace.append((tile_name, r['slice_name']))
                    source_ids.append(r.get('detection_id') or
                                      stable_id('legacy2d', tile_name, ch_id, row_number))
                if rows:
                    parts.append(np.array(rows, dtype=object))
    if not parts:
        return None, None, decisions
    mat = np.concatenate(parts, axis=0)
    assert len(trace) == len(mat) == len(source_ids)
    mat = np.column_stack((mat, np.asarray(source_ids, dtype=object),
                           np.asarray([t for t, _ in trace], dtype=object),
                           np.asarray([s for _, s in trace], dtype=object)))
    marker = channel_marker(ch_id)   # class 标签用规范化 marker（"GFP_3" → "GFP"）
    for row in mat:
        row[6] = f"{row[6]}_{marker}"   # "neuron" → "neuron_RFP"
    return mat, trace, decisions


def _metadata_from_global_csv(df):
    """从已存在的全局 2D CSV 恢复查找 tile/slice 用的元数据（续跑时用）。"""
    x1, y1, x2, y2 = (df[c].to_numpy(dtype=float) for c in ('x1', 'y1', 'x2', 'y2'))
    return {'coords': np.column_stack(((x1 + x2) / 2, (y1 + y2) / 2, df['z'].to_numpy(dtype=float))),
            **_encode_names(df['tile_name'].to_numpy(), df['slice_name'].to_numpy())}


def _encode_names(tiles, slices):
    tile_u, tile_c = np.unique(np.asarray(tiles, dtype=object).astype(str), return_inverse=True)
    slice_u, slice_c = np.unique(np.asarray(slices, dtype=object).astype(str), return_inverse=True)
    return {'tile_u': tile_u, 'tile_c': tile_c, 'slice_u': slice_u, 'slice_c': slice_c}


def _pack_metadata(metadata):
    """tile 元数据 → 紧凑数组（tile/slice 名重复极多，编码后再跨进程传回主进程）。"""
    if not metadata:
        return None
    coords = np.array([m[:3] for m in metadata], dtype=float)
    return {'coords': coords, **_encode_names([m[3] for m in metadata], [m[4] for m in metadata])}


def _stage3_input_signature(ch_id, src_dir, geom, zl_params):
    suffix = f"_{ch_id}_result.csv"
    source_files = sorted(
        os.path.join(src_dir, name) for name in os.listdir(src_dir)
        if name.endswith(suffix))
    if geom['num_tiles'] > 1:
        tile_names = {os.path.basename(name) for name in geom['dir_dict']}
        source_files = [path for path in source_files
                        if os.path.basename(path)[:-len(suffix)] in tile_names]
    positions = [
        (str(key), list(np.asarray(geom['disp_mat_fin'][key]).tolist()))
        for key in sorted(geom['dir_dict'].values())
    ]
    return {
        'channel': ch_id,
        'source_files': {path: file_stamp(path, with_hash=True)
                         for path in source_files},
        'geometry': {
            'xml_source': (file_stamp(geom['xml_source'], with_hash=True)
                           if geom.get('xml_source') else None),
            'positions': positions, 'z_start': int(geom['z_start']),
            'Z': int(geom['Z']), 'tile_size': int(geom['tile_size']),
            'H': int(geom['H']), 'W': int(geom['W']),
        },
        'z_linker': zl_params,
        'code_sha256': {
            name: file_sha256(os.path.join(os.path.dirname(__file__), name))
            for name in ('channel_stage3.py', 'stitcher.py', 'z_linker.py',
                         'provenance.py')
        },
    }


def stitch_and_link_channel(ch, src_dir, global_2d_dir, channel_3d_dir, geom, zl_params):
    """
    一个通道的 Stage 3 前半段。返回 {'ch_id', 'metadata'}；3D 结果写到 pkl/csv，由主进程读回。
    参数全可 pickle，供进程池直接调用。
    """
    ch_id = ch['id']
    ch_type = ch.get('type', 'soma')
    global_2d = os.path.join(global_2d_dir, f"{ch_id}_2d_global.csv")
    pkl_path = os.path.join(channel_3d_dir, f"{ch_id}_3d_tracked.pkl")
    metadata = []
    packed = None
    manifest_path = os.path.join(channel_3d_dir, f"{ch_id}_stage3_manifest.json")
    input_signature = _stage3_input_signature(ch_id, src_dir, geom, zl_params)
    manifest_ready = output_manifest_valid(manifest_path, input_signature)
    sidecars = [
        os.path.join(global_2d_dir, f"{ch_id}_stitch_decisions.csv"),
        os.path.join(channel_3d_dir, f"{ch_id}_3d_tracked.csv"),
        os.path.join(channel_3d_dir, f"{ch_id}_track_members.csv"),
        os.path.join(channel_3d_dir, f"{ch_id}_track_rejections.csv"),
    ]
    use_cached_global = manifest_ready and os.path.isfile(global_2d)
    if use_cached_global:
        cached_cols = set(pd.read_csv(global_2d, nrows=0).columns)
        use_cached_global = {'detection_id', *TRACE_COLS} <= cached_cols
    need_relink = not (manifest_ready and os.path.isfile(pkl_path)
                       and all(os.path.isfile(path) for path in sidecars))

    mat = None
    if use_cached_global:
        need_mat = need_relink
        header = pd.read_csv(global_2d, nrows=0).columns
        has_trace = all(c in header for c in TRACE_COLS)
        has_id = "detection_id" in header
        cols = BOX_COLS if need_mat else ["x1", "y1", "x2", "y2", "z"]
        extras = (TRACE_COLS if has_trace else []) + (["detection_id"] if has_id else [])
        df = pd.read_csv(global_2d, usecols=cols + extras,
                         dtype={c: str for c in extras}, keep_default_na=False)
        if need_mat:
            ids = (df["detection_id"].to_numpy(dtype=object) if has_id else
                   np.asarray([stable_id("legacy_global2d", ch_id, i)
                               for i in range(len(df))], dtype=object))
            tiles = (df["tile_name"].to_numpy(dtype=object) if has_trace else
                     np.full(len(df), "Unknown", dtype=object))
            slices = (df["slice_name"].to_numpy(dtype=object) if has_trace else
                      np.full(len(df), "Unknown", dtype=object))
            mat = np.column_stack((df[BOX_COLS].values, ids, tiles, slices))
        if has_trace:
            packed = _metadata_from_global_csv(df)
        else:
            logging.warning(f"⚠️ [{ch_id}] {global_2d} 是旧版本生成的，没有 tile_name/slice_name 列，"
                            f"这个通道的细胞无法溯源到 tile/切片。删除该文件重跑即可补上。")
        logging.info(f"✔️ [{ch_id}] 全局 2D 已存在，直接加载。")
    else:
        logging.info(f" -> 正在拼接通道 2D 框: [{ch_id}] (类型: {ch_type})")
        mat, trace, decisions = _stitch_channel(ch_id, src_dir, geom, metadata)
        decision_path = os.path.join(global_2d_dir, f"{ch_id}_stitch_decisions.csv")
        pd.DataFrame(decisions, columns=[
            'detection_id', 'tile_name', 'slice_name', 'z', 'decision',
            'reason', 'other_detection_id', 'iou',
        ]).to_csv(decision_path + '.part', index=False)
        os.replace(decision_path + '.part', decision_path)
        if mat is None:
            mat = np.empty((0, 11), dtype=object)
        if mat is not None:
            out = pd.DataFrame(mat[:, :8], columns=BOX_COLS)
            out['detection_id'] = mat[:, 8]
            out['tile_name'] = mat[:, 9]
            out['slice_name'] = mat[:, 10]
            tmp = global_2d + '.part'
            out.to_csv(tmp, index=False)
            os.replace(tmp, global_2d)
            logging.info(f"✔️ [{ch_id}] 全局2D: {len(mat)} 框 → {global_2d}")
        packed = _pack_metadata(metadata)
        need_relink = True

    if not need_relink:
        logging.info(f"✔️ [{ch_id}] 已存在 3D 追踪结果，直接加载（跳过 Z-Linker）。")
    elif mat is not None:
        rejected_tracks = []
        summary, vol_list = run_z_linker(
            mat, channel_id=ch_id, rejected_tracks=rejected_tracks, **zl_params)
        rejected_path = os.path.join(channel_3d_dir, f"{ch_id}_track_rejections.csv")
        pd.DataFrame(rejected_tracks, columns=[
            'track_id', 'channel', 'decision', 'reason',
            'member_count', 'threshold', 'member_detection_ids',
        ]).to_csv(rejected_path + '.part', index=False)
        os.replace(rejected_path + '.part', rejected_path)
        out_3d = os.path.join(channel_3d_dir, f"{ch_id}_3d_tracked.csv")
        summary_df = pd.DataFrame(summary, columns=BOX_COLS)
        for field in ('track_id', 'best_detection_id', 'best_box_z',
                      'z_min', 'z_max', 'bounds_method', 'observed_x1',
                      'observed_y1', 'observed_x2', 'observed_y2',
                      'member_count'):
            summary_df[field] = [cell.get(field) for cell in vol_list]
        summary_df.to_csv(out_3d + '.part', index=False)
        os.replace(out_3d + '.part', out_3d)
        members_path = os.path.join(channel_3d_dir, f"{ch_id}_track_members.csv")
        members = [
            {'track_id': cell['track_id'], 'detection_id': det['detection_id'],
             'tile_name': det['tile_name'], 'slice_name': det['slice_name'],
             'z': det['z'], 'x1': det['x1'], 'y1': det['y1'],
             'x2': det['x2'], 'y2': det['y2'], 'score': det['score'],
             'mean': det['mean'], 'class': det['class']}
            for cell in vol_list for det in cell['member_detections']
        ]
        pd.DataFrame(members, columns=[
            'track_id', 'detection_id', 'tile_name', 'slice_name',
            'z', 'x1', 'y1', 'x2', 'y2', 'score', 'mean', 'class',
        ]).to_csv(members_path + '.part', index=False)
        os.replace(members_path + '.part', members_path)
        # pkl 是 checkpoint，最后写且原子替换：进程被杀不会留下半截 pkl
        with open(pkl_path + '.part', 'wb') as pf:
            pickle.dump(vol_list, pf)
        os.replace(pkl_path + '.part', pkl_path)
        kind = 'soma' if ch_type == 'soma' else 'TF'
        logging.info(f"✔️ [{ch_id}] Z-Link {kind}: {len(vol_list)} 个细胞 → {out_3d}")

    if need_relink:
        outputs = [global_2d, pkl_path, *sidecars]
        if all(os.path.isfile(path) for path in outputs):
            write_output_manifest(manifest_path, outputs, input_signature)

    return {'ch_id': ch_id, 'metadata': packed}
