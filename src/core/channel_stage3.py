# -*- coding: utf-8 -*-
"""
Stage 3 的单通道部分：全局 2D 拼接 → Z-Link → 落盘。

各通道之间互不依赖，run_inference.py 把每个通道交给一个进程并行执行；跨通道的
3D 共定位（3A/3B/3C）仍在主进程里做。每一步都保留原来的 checkpoint 语义：
  2_global_2d_raw/<ch>_2d_global.csv  存在 → 不再拼接；tile/slice 溯源信息从它的 tile_name/slice_name 列恢复
  3_channel_3d/<ch>_3d_tracked.pkl     存在 → 不再 Z-Link
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

BOX_COLS = ["x1", "y1", "x2", "y2", "score", "mean", "class", "z"]
TRACE_COLS = ["tile_name", "slice_name"]   # 全局 2D CSV 里每个框的来源，续跑时据此恢复溯源信息


def _stitch_channel(ch_id, src_dir, geom, metadata):
    """
    把一个通道所有 tile 的 CSV 拼成全局 2D 矩阵（与原先 run_inference 里的逻辑一致）。
    返回 (mat, trace)：trace 是与 mat 逐行对应的 (tile_name, slice_name)。
    """
    parts, trace = [], []
    if geom['num_tiles'] > 1:
        stitched = [[np.empty((0, 8)) for _ in range(2)] for _ in range(geom['Z'])]
        row_meta = {}
        for dir_name, pos in geom['dir_dict'].items():
            tile_name = os.path.split(dir_name)[-1]
            csv_tile = os.path.join(src_dir, f"{tile_name}_{ch_id}_result.csv")
            if os.path.isfile(csv_tile):
                with open(csv_tile, newline='', encoding='utf-8') as tile_file:
                    next(tile_file, None)
                    stitched = combine_predictions(
                        stitched, csv.reader(tile_file), None, geom['z_start'], geom['Z'],
                        pos, geom['disp_mat_fin'], (geom['H'], geom['W']),
                        metadata, tile_name, tILESIZE=geom['tile_size'], row_meta=row_meta,
                    )
        for zi, layer in enumerate(stitched):
            for ti, group in enumerate(layer):
                if group.size > 0:
                    parts.append(group)
                    trace.extend(row_meta[(zi, ti)])
    else:
        csv_list = [f for f in os.listdir(src_dir) if f.endswith(f'_{ch_id}_result.csv')]
        if csv_list:
            tile_name = csv_list[0].split(f"_{ch_id}_")[0]
            with open(os.path.join(src_dir, csv_list[0]), 'r', encoding='utf-8') as f:
                next(f, None)
                rows = []
                for r in csv.reader(f):
                    if len(r) > 8 and r[0] != 'tile_id':
                        x1, y1, x2, y2 = float(r[1]), float(r[2]), float(r[3]), float(r[4])
                        score, mean, c, z = float(r[6]), float(r[7]), str(r[5]), int(float(r[8]))
                        rows.append([x1, y1, x2, y2, score, mean, c, z])
                        metadata.append([(x1 + x2) / 2, (y1 + y2) / 2, z, tile_name, r[0]])
                        trace.append((tile_name, r[0]))
                if rows:
                    parts.append(np.array(rows, dtype=object))
    if not parts:
        return None, None
    mat = np.concatenate(parts, axis=0)
    assert len(trace) == len(mat)
    marker = channel_marker(ch_id)   # class 标签用规范化 marker（"GFP_3" → "GFP"）
    for row in mat:
        row[6] = f"{row[6]}_{marker}"   # "neuron" → "neuron_RFP"
    return mat, trace


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

    mat = None
    if os.path.exists(global_2d):
        need_mat = not os.path.exists(pkl_path)
        header = pd.read_csv(global_2d, nrows=0).columns
        has_trace = all(c in header for c in TRACE_COLS)
        cols = BOX_COLS if need_mat else ["x1", "y1", "x2", "y2", "z"]
        df = pd.read_csv(global_2d, usecols=cols + (TRACE_COLS if has_trace else []),
                         dtype={c: str for c in TRACE_COLS}, keep_default_na=False)
        if need_mat:
            mat = df[BOX_COLS].values
        if has_trace:
            packed = _metadata_from_global_csv(df)
        else:
            logging.warning(f"⚠️ [{ch_id}] {global_2d} 是旧版本生成的，没有 tile_name/slice_name 列，"
                            f"这个通道的细胞无法溯源到 tile/切片。删除该文件重跑即可补上。")
        logging.info(f"✔️ [{ch_id}] 全局 2D 已存在，直接加载。")
    else:
        logging.info(f" -> 正在拼接通道 2D 框: [{ch_id}] (类型: {ch_type})")
        mat, trace = _stitch_channel(ch_id, src_dir, geom, metadata)
        if mat is not None:
            out = pd.DataFrame(mat, columns=BOX_COLS)
            out['tile_name'] = [t for t, _ in trace]
            out['slice_name'] = [s for _, s in trace]
            tmp = global_2d + '.part'
            out.to_csv(tmp, index=False)
            os.replace(tmp, global_2d)
            logging.info(f"✔️ [{ch_id}] 全局2D: {len(mat)} 框 → {global_2d}")
        packed = _pack_metadata(metadata)

    if os.path.exists(pkl_path):
        logging.info(f"✔️ [{ch_id}] 已存在 3D 追踪结果，直接加载（跳过 Z-Linker）。")
    elif mat is not None and len(mat) > 0:
        summary, vol_list = run_z_linker(mat, **zl_params)
        out_3d = os.path.join(channel_3d_dir, f"{ch_id}_3d_tracked.csv")
        pd.DataFrame(summary, columns=BOX_COLS).to_csv(out_3d, index=False)
        # pkl 是 checkpoint，最后写且原子替换：进程被杀不会留下半截 pkl
        with open(pkl_path + '.part', 'wb') as pf:
            pickle.dump(vol_list, pf)
        os.replace(pkl_path + '.part', pkl_path)
        kind = 'soma' if ch_type == 'soma' else 'TF'
        logging.info(f"✔️ [{ch_id}] Z-Link {kind}: {len(vol_list)} 个细胞 → {out_3d}")

    return {'ch_id': ch_id, 'metadata': packed}
