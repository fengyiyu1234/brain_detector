# -*- coding: utf-8 -*-
"""
Stage 3 的单通道部分：全局 2D 拼接 → Z-Link → 落盘。

各通道之间互不依赖，run_inference.py 把每个通道交给一个进程并行执行；跨通道的
3D 共定位（3A/3B/3C）仍在主进程里做。每一步都保留原来的 checkpoint 语义：
  2_global_2d_raw/<ch>_2d_global.csv  存在 → 不再拼接（也就不再产生该通道的 tile 元数据）
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


def _stitch_channel(ch_id, src_dir, geom, metadata):
    """把一个通道所有 tile 的 CSV 拼成全局 2D 矩阵（与原先 run_inference 里的逻辑一致）。"""
    parts = []
    if geom['num_tiles'] > 1:
        stitched = [[np.empty((0, 8)) for _ in range(2)] for _ in range(geom['Z'])]
        for dir_name, pos in geom['dir_dict'].items():
            tile_name = os.path.split(dir_name)[-1]
            csv_tile = os.path.join(src_dir, f"{tile_name}_{ch_id}_result.csv")
            if os.path.isfile(csv_tile):
                with open(csv_tile, newline='', encoding='utf-8') as tile_file:
                    next(tile_file, None)
                    stitched = combine_predictions(
                        stitched, csv.reader(tile_file), None, geom['z_start'], geom['Z'],
                        pos, geom['disp_mat_fin'], (geom['H'], geom['W']),
                        metadata, tile_name, tILESIZE=geom['tile_size'],
                    )
        for layer in stitched:
            for group in layer:
                if group.size > 0:
                    parts.append(group)
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
                if rows:
                    parts.append(np.array(rows, dtype=object))
    if not parts:
        return None
    mat = np.concatenate(parts, axis=0)
    marker = channel_marker(ch_id)   # class 标签用规范化 marker（"GFP_3" → "GFP"）
    for row in mat:
        row[6] = f"{row[6]}_{marker}"   # "neuron" → "neuron_RFP"
    return mat


def _pack_metadata(metadata):
    """tile 元数据 → 紧凑数组（tile/slice 名重复极多，编码后再跨进程传回主进程）。"""
    if not metadata:
        return None
    coords = np.array([m[:3] for m in metadata], dtype=float)
    tile_u, tile_c = np.unique(np.array([m[3] for m in metadata], dtype=object).astype(str),
                               return_inverse=True)
    slice_u, slice_c = np.unique(np.array([m[4] for m in metadata], dtype=object).astype(str),
                                 return_inverse=True)
    return {'coords': coords, 'tile_u': tile_u, 'tile_c': tile_c,
            'slice_u': slice_u, 'slice_c': slice_c}


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

    mat = None
    if os.path.exists(global_2d):
        if not os.path.exists(pkl_path):
            mat = pd.read_csv(global_2d)[BOX_COLS].values
        logging.info(f"✔️ [{ch_id}] 全局 2D 已存在，直接加载。")
    else:
        logging.info(f" -> 正在拼接通道 2D 框: [{ch_id}] (类型: {ch_type})")
        mat = _stitch_channel(ch_id, src_dir, geom, metadata)
        if mat is not None:
            tmp = global_2d + '.part'
            pd.DataFrame(mat, columns=BOX_COLS).to_csv(tmp, index=False)
            os.replace(tmp, global_2d)
            logging.info(f"✔️ [{ch_id}] 全局2D: {len(mat)} 框 → {global_2d}")

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

    return {'ch_id': ch_id, 'metadata': _pack_metadata(metadata)}
