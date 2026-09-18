# -*- coding: utf-8 -*-
"""
不用 TeraStitcher 的位移计算，直接从各通道自己的检测结果求每个 tile 的位置，
顺带得到通道之间的逐 tile 位移（= channel alignment 里的几何部分）。

为什么要这么做
--------------
原来的 Stage 2.5 是在**同一个 tile 内**把 TF 通道的核对到参考通道的细胞体上。
T4 上 GFP 稀疏、Olig2/Sox9 稠密且标记的是不同细胞群，这个匹配本身信噪比极低
（得分中位数 0.006），逐 tile 结果不可信。

本脚本换一个方向：每个通道**只和自己比**——相邻 tile 的重叠区里，同一个细胞
被两个 tile 各检测一次，把它们配上就得到这条接缝的位移。Olig2/Sox9 是最稠密的
通道，反而每条接缝有几千到上万个配对，位移中位数的标准误 < 0.03 px。
各通道各自解出一套 tile 位置之后，通道之间的逐 tile 位移就是两套位置之差。

模型
----
共用一套 tile 位置 + 每通道一个平滑的通道位移场（默认 model='joint'）：

    P_c(t) = P(t) + delta_c(t)
    delta_c(t) = a_c + b_c * 行号(t) + c_c * 列号(t)        参考通道 delta ≡ 0

- P(t)：所有通道共用。拼接里真正的大头（台面误差）与通道无关，让所有通道的接缝
  一起约束它，信息量最大；而且纯平移模型解释不了的东西（T4 上第 3/4 行之间有约
  0.4° 的相对旋转，各通道的 x 接缝一致地随列号从 +24 px 走到 −21 px）会被 P(t)
  统一吸收，不会各通道分摊得不一样、再冒充成通道位移。
- delta_c(t)：通道之间真实存在的倍率差 / 光片倾斜差（色差）。T4 实测 Olig2 相对
  GFP 是 +5.25 px/行（y，约 0.30% 的放大率差）、−2.10 层/列（z）；RFP、Sox9 的
  斜率按波长单调排序，730 的 z 斜率变号——是光学量，不是噪声。
- 接缝只约束 delta 的**差**，常数项 a_c 解不出来（Olig2 那个 dz≈−7 层就在里面）：
  把某个通道所有 tile 的位置整体平移，每条接缝约束依然精确成立，数据看不出区别。
  a_c 只能靠跨通道匹配定，但每通道只有 3 个数，所以本脚本把**全脑所有 tile 的细胞按解出来
  的位置摆进同一套坐标后一次估出来**（--const-from pooled，默认），而不是像旧 Stage 2.5
  那样每个 tile 各估一次：数据量大几十倍，未知数少十几倍。
  soma↔TF 用包含度打分（和 Stage 3B 判共定位同一个判据），同类型通道用质心配对。
  也可以用 --const 直接给死，或 --const-from offsets 取旧 Stage 2.5 结果的中位数。

model='free' 则是每个通道各解各的（不加平滑约束），留作对照。

两个「参考」是分开的
------------------
--ref   通道对齐的参考通道：通道场 delta_c 和常数 a_c 都相对它，也就是真正**测量**出来的量。
        MADM 流程里它应当是 GFP——下游 Stage 3B 的共定位判的就是「GFP 细胞体里有没有 TF 核」，
        所以要让 GFP↔各核通道 成为直接测量的那一对，误差才落在最该准的地方。
--frame 全局坐标系用哪个通道的几何：细胞和配准用的全脑图都落在它上面（例如用 488 做配准就给
        Olig2）。它只是换个规范，不重测任何东西：
            s_c(t) = [delta_c(t) + a_c] − [delta_frame(t) + a_frame]
        缺省 --frame = --ref，就是原来的行为。

输入
----
`1_tile_2d_raw/` 里的逐 tile 检测 CSV，以及 tile 目录名（台面坐标，单位 0.1 µm，
见 src/utils/io.STAGE_UNIT_UM）当初值。**不需要任何 TeraStitcher XML。**

输出（默认 <results_dir>/5_analysis_report/tile_positions/）
  seams.csv            每条 (通道, 接缝) 的实测位移与配对质量；再次运行时直接复用
  tile_positions.csv   每个 tile 的共用位置、各通道位置、各通道相对参考通道的偏移
  solution.json        通道场系数、常数来源、残差统计（可追溯）
  report.txt           屏幕摘要的副本

可选输出
  --write-xml     每个通道写一份 xml_merging.xml（TeraStitcher 只跑 merge）
  --write-aligned 写成 0_channel_alignment 的格式（offsets JSON + 平移后的 CSV），
                  可以直接顶替 Stage 2.5 的结果；默认写到新目录，不覆盖原结果

用法
----
  # 只求解 + 看报告
  python scripts/solve_tile_positions.py --sample Y:/Fengyi/EGFR_brain/T4 --workers 16

  # 求解并写出可供 Stage 3 使用的偏移（常数由 --const 给出）
  python scripts/solve_tile_positions.py --sample Y:/Fengyi/EGFR_brain/T4 --workers 16 \
      --const Olig2=-3,0,-7 --const Sox9=1,6,-4 --const RFP=0,0,-4 \
      --write-aligned --write-xml
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    sys.stderr.reconfigure(encoding='utf-8')

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import multiprocessing as mp
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.sparse import coo_matrix, vstack
from scipy.sparse.linalg import lsqr
from scipy.spatial import cKDTree
from tqdm import tqdm

import scripts.compare_stitching as cs
from src.config.loader import load_config
from src.core.point_cloud_aligner import (apply_shift_to_csv, resolve_align_settings,
                                          containment_shift_from_arrays, _containment_score)
from src.utils.io import STAGE_UNIT_UM

AXES = ('x', 'y', 'z')


# ──────────────────────────────────────────────────────────────────────────────
# 1.  网格：tile 名字 → 行列号 + 名义位置
# ──────────────────────────────────────────────────────────────────────────────

def nominal_grid(tiles, xy_res_um):
    """
    tile 名字是 '<V>_<H>' 的台面坐标（单位 STAGE_UNIT_UM）。
    返回 {tile: {'rc': (行, 列), 'pos': np.array([x, y, z])}}，pos 已平移到从 0 开始。
    """
    rows = sorted({int(t.split('_')[0]) for t in tiles})
    cols = sorted({int(t.split('_')[1]) for t in tiles})
    out = {}
    for t in tiles:
        v, h = (int(p) for p in t.split('_')[:2])
        out[t] = {'rc': (rows.index(v), cols.index(h)),
                  'pos': np.array([h * STAGE_UNIT_UM / xy_res_um,
                                   v * STAGE_UNIT_UM / xy_res_um, 0.0])}
    origin = np.min([d['pos'] for d in out.values()], axis=0)
    for d in out.values():
        d['pos'] = d['pos'] - origin
    return out


def channel_image_dirs(config, routing, sample_dir):
    """
    {通道 id: 图像目录}。config 里的路径可能是 HPC 上的，本机跑时优先用
    <sample>/<同名目录>。
    """
    out = {}
    for ch in routing:
        p = config.get('paths', {}).get(ch.get('dir_key'), '') or ''
        local = os.path.join(sample_dir, os.path.basename(p.rstrip('/\\'))) if p else ''
        out[ch['id']] = local if local and os.path.isdir(local) else p
    return out


def discover_tiles(det_dir, channels):
    """检测 CSV 名为 '<tile>_<CH>_result.csv'；返回所有通道都有 CSV 的 tile。"""
    names = {}
    for f in os.listdir(det_dir):
        if not f.endswith('_result.csv'):
            continue
        base = f[:-len('_result.csv')]
        for ch in channels:
            if base.endswith('_' + ch):
                names.setdefault(base[:-(len(ch) + 1)], set()).add(ch)
    full = sorted(t for t, chs in names.items() if chs >= set(channels))
    missing = sorted(set(names) - set(full))
    return full, missing


def count_slices(det_dir, tiles, channels, tile_dirs=None):
    """
    每个 tile 的切片数，只用于剔除贴着 z 两端（可能被截断）的细胞。
    优先数 tile 目录里的 TIFF；数不到就取各通道检测 CSV 里最大的 z——必须把稠密通道
    也算进来，稀疏通道（如 GFP）的最大 z 往往远小于真实层数。
    """
    for t in tiles[:3]:
        d = (tile_dirs or {}).get(t)
        if d and os.path.isdir(d):
            n = sum(1 for f in os.listdir(d)
                    if f.lower().endswith(('.tif', '.tiff')) and not f.startswith('.'))
            if n:
                return n, f"{t} 目录里的 TIFF 数"
    zmax = 0
    for t in tiles[:3]:
        for ch in channels:
            p = os.path.join(det_dir, f"{t}_{ch}_result.csv")
            if os.path.isfile(p):
                z = pd.read_csv(p, usecols=['z'])['z']
                if not z.empty:
                    zmax = max(zmax, int(z.max()))
    return zmax, "检测 CSV 里的最大 z（可能略小于真实层数）"


# ──────────────────────────────────────────────────────────────────────────────
# 2.  接缝实测位移
# ──────────────────────────────────────────────────────────────────────────────

def build_jobs(tiles, grid, channels, ch_types, settings, params, n_slices, res):
    """每个 (通道, 相邻 tile 对) 一个任务，origin 用名义网格。"""
    rc2t = {d['rc']: t for t, d in grid.items()}
    jobs = []
    for t in tiles:
        r, c = grid[t]['rc']
        for direction, dr, dc in cs.DIRECTIONS:
            u = rc2t.get((r + dr, c + dc))
            if u is None or u not in grid:
                continue
            for ch in channels:
                ctype = ch_types.get(ch, 'soma')
                p = dict(params, r_xy=params['r_xy_soma'] if ctype == 'soma' else params['r_xy_tf'])
                p.pop('r_xy_soma'); p.pop('r_xy_tf')
                jobs.append({
                    'xml': 'grid', 'channel': ch, 'frame': ch, 'direction': direction,
                    'row': r, 'col': c, 'nb_row': r + dr, 'nb_col': c + dc,
                    'tile': t, 'nb_tile': u,
                    'origin_a': grid[t]['pos'], 'origin_b': grid[u]['pos'],
                    'n_slices': n_slices, 'res': res, 'params': p,
                    'z_link': settings['z_link']['soma' if ctype == 'soma' else 'tf'],
                })
    return jobs


def add_measured(df, grid):
    """
    补上实测位移列。

    run_seam_job 返回的 dx/dy/dz 是「按 origin 摆放后，同一个细胞在邻居 tile 比在
    本 tile 多出来的量」，即摆放误差，理想值 0。所以实测位移 = 名义位移 − 误差。
    复用缓存的 seams.csv 时也重算一遍，免得跟着旧文件里的列走。
    """
    nom = np.array([grid[r.nb_tile]['pos'] - grid[r.tile]['pos'] for r in df.itertuples()])
    for i, ax in enumerate(AXES):
        df[f'nom_d{ax}'] = nom[:, i]
        df[f'meas_d{ax}'] = nom[:, i] - df[f'd{ax}']
    return df


def measure_seams(jobs, grid, workers):
    """跑所有接缝任务，返回 DataFrame。"""
    rows = []
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
            futs = [pool.submit(cs.run_seam_job, j) for j in jobs]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Seams"):
                rows.append(fut.result())
    else:
        for j in tqdm(jobs, desc="Seams"):
            rows.append(cs.run_seam_job(j))
    df = add_measured(pd.DataFrame(rows), grid)
    return df.sort_values(['channel', 'row', 'col', 'direction']).reset_index(drop=True)


def seam_weights(df, axis):
    """位移中位数的标准误 ≈ 1.4826 * MAD / sqrt(n)，权重取其倒数（加下限防止过度自信）。"""
    mad = df[f'mad_{axis}'].to_numpy(dtype=float)
    n = np.maximum(df['n_match'].to_numpy(dtype=float), 1.0)
    sigma = np.maximum(np.maximum(mad, 0.3) * 1.4826 / np.sqrt(n), 0.02)
    return 1.0 / sigma


# ──────────────────────────────────────────────────────────────────────────────
# 3.  求解
# ──────────────────────────────────────────────────────────────────────────────

def _irls(build, b, w0, huber, iters):
    """加权最小二乘 + Huber 重加权。build(w) → 稀疏矩阵；返回 (解, 残差, 最终权重)。"""
    sol = None
    w = w0.copy()
    res = np.zeros(len(b))
    for _ in range(iters):
        A, rhs = build(w)
        sol = lsqr(A, rhs, atol=1e-12, btol=1e-12, iter_lim=20000)[0]
        res = A[:len(b)].dot(sol) / np.where(w == 0, 1, w) - b
        scale = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-6
        w = w0 * np.minimum(1.0, huber * scale / np.maximum(np.abs(res), 1e-9))
    return sol, res, w


def solve_axis(seams, tiles, grid, channels, ref, axis, model, prior_w, huber, iters):
    """
    解一根轴。返回 dict:
      P      : 共用 tile 位置 (model='joint') 或参考通道的位置 (model='free')
      P_ch   : {通道: 位置数组}
      slope  : {通道: (b 每行, c 每列)}，model='free' 时为 None
      resid  : 每条接缝的拟合残差（与 seams 同序）
    """
    idx = {t: i for i, t in enumerate(tiles)}
    nom = np.array([grid[t]['pos'][AXES.index(axis)] for t in tiles])
    NT = len(tiles)

    def prior_block(n_cols):
        return (coo_matrix((np.full(NT, prior_w), (np.arange(NT), np.arange(NT))),
                           shape=(NT, n_cols)), nom * prior_w)

    if model == 'free':
        P_ch, resid = {}, np.full(len(seams), np.nan)
        for ch in channels:
            sel = np.where(seams['channel'].to_numpy() == ch)[0]
            g = seams.iloc[sel]
            i = np.array([idx[t] for t in g['tile']])
            j = np.array([idx[t] for t in g['nb_tile']])
            b = g[f'meas_d{axis}'].to_numpy(dtype=float)
            w0 = seam_weights(g, axis)
            rr = np.repeat(np.arange(len(g)), 2)
            cc = np.column_stack([i, j]).ravel()
            vv = np.column_stack([-np.ones(len(g)), np.ones(len(g))]).ravel()

            def build(w, rr=rr, cc=cc, vv=vv, b=b):
                A = coo_matrix((vv * np.repeat(w, 2), (rr, cc)), shape=(len(b), NT))
                pri, prhs = prior_block(NT)
                return vstack([A, pri]).tocsr(), np.concatenate([b * w, prhs])

            sol, res, _ = _irls(build, b, w0, huber, iters)
            P_ch[ch] = sol
            resid[sel] = res
        return {'P': P_ch[ref], 'P_ch': P_ch, 'slope': None, 'resid': resid}

    # joint：未知量 = [P(t) × NT] + 每个非参考通道的 (b 每行, c 每列)
    extra = [c for c in channels if c != ref]
    eidx = {c: NT + 2 * k for k, c in enumerate(extra)}
    n = NT + 2 * len(extra)
    i = np.array([idx[t] for t in seams['tile']])
    j = np.array([idx[t] for t in seams['nb_tile']])
    drow = np.array([grid[u]['rc'][0] - grid[t]['rc'][0]
                     for t, u in zip(seams['tile'], seams['nb_tile'])], dtype=float)
    dcol = np.array([grid[u]['rc'][1] - grid[t]['rc'][1]
                     for t, u in zip(seams['tile'], seams['nb_tile'])], dtype=float)
    b = seams[f'meas_d{axis}'].to_numpy(dtype=float)
    w0 = seam_weights(seams, axis)
    m = len(seams)

    rr, cc, vv = [np.arange(m), np.arange(m)], [i, j], [-np.ones(m), np.ones(m)]
    for ch in extra:
        sel = (seams['channel'].to_numpy() == ch)
        k = np.where(sel)[0]
        rr += [k, k]
        cc += [np.full(len(k), eidx[ch]), np.full(len(k), eidx[ch] + 1)]
        vv += [drow[k], dcol[k]]
    rr, cc, vv = np.concatenate(rr), np.concatenate(cc), np.concatenate(vv)

    def build(w):
        A = coo_matrix((vv * w[rr], (rr, cc)), shape=(m, n))
        pri, prhs = prior_block(n)
        return vstack([A, pri]).tocsr(), np.concatenate([b * w, prhs])

    sol, res, _ = _irls(build, b, w0, huber, iters)
    P = sol[:NT]
    slope = {ch: (sol[eidx[ch]], sol[eidx[ch] + 1]) for ch in extra}
    slope[ref] = (0.0, 0.0)
    P_ch = {ch: P + np.array([slope[ch][0] * grid[t]['rc'][0] + slope[ch][1] * grid[t]['rc'][1]
                              for t in tiles]) for ch in channels}
    return {'P': P, 'P_ch': P_ch, 'slope': slope, 'resid': res}


def seam_support(seams, tiles, channels):
    """每个 tile 在每个通道上参与了几条成功的接缝。"""
    deg = {ch: {t: 0 for t in tiles} for ch in channels}
    for r in seams.itertuples():
        deg[r.channel][r.tile] += 1
        deg[r.channel][r.nb_tile] += 1
    return deg


# ──────────────────────────────────────────────────────────────────────────────
# 4.  通道常数 a_c
# ──────────────────────────────────────────────────────────────────────────────

CELL_COLS = ('cx', 'cy', 'cz', 'x1_3d', 'y1_3d', 'x2_3d', 'y2_3d', 'z_min', 'z_max')


def tile_cell_arrays(job):
    """
    一个 (tile, 通道)：读检测 CSV → 取指定 z 窗 → z-link → 返回 (N, 9) 数组，列见 CELL_COLS。
    坐标保持 tile 局部原始坐标，由调用方决定加哪一套 tile 位置（要比较不同的位置方案）。
    参数全可 pickle，供进程池调用。
    """
    from src.core.z_linker import run_z_linker
    empty = (job['channel'], job['tile'], np.empty((0, 9)))
    df = cs._read_dets(job['csv'])
    if df is None:
        return empty
    df = df[(df['z'] >= job['z_lo']) & (df['z'] <= job['z_hi'])]
    if df.empty:
        return empty
    mat = df[cs.BOX_COLS].values.copy()
    mat[:, 6] = np.array([f"{v}_{job['channel']}" for v in mat[:, 6]])
    _, vol = run_z_linker(mat, **job['z_link'])
    if not vol:
        return empty
    arr = np.array([[c.get(k, 0) for k in CELL_COLS] for c in vol], dtype=float)
    return job['channel'], job['tile'], arr


def shift_arrays(arr, offset):
    """给 (N, 9) 细胞数组整体加一个 (dx, dy, dz)。"""
    out = arr.copy()
    out[:, [0, 3, 5]] += offset[0]      # cx, x1, x2
    out[:, [1, 4, 6]] += offset[1]      # cy, y1, y2
    out[:, [2, 7, 8]] += offset[2]      # cz, z_min, z_max
    return out


def pool_cells(tiles, channels, det_dir, z_window, ch_types, settings, offsets, workers, desc):
    """
    把每个 tile、每个通道的一段 z 窗 z-link 成紧凑数组，返回 {通道: {tile: (N, 9) 局部坐标}}。

    z_window : {tile: (全局 z 下界, 全局 z 上界)}
    offsets  : {(通道, tile): (dx, dy, dz)}，只用来把全局 z 窗换算成该通道的局部 z 窗
    """
    jobs = []
    for ch in channels:
        zl = settings['z_link']['soma' if ch_types.get(ch, 'soma') == 'soma' else 'tf']
        for t in tiles:
            lo, hi = z_window[t]
            dz = offsets[(ch, t)][2]
            jobs.append({'csv': os.path.join(det_dir, f"{t}_{ch}_result.csv"),
                         'channel': ch, 'tile': t, 'z_link': zl,
                         'z_lo': lo - dz, 'z_hi': hi - dz})
    parts = {ch: {} for ch in channels}
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
            futs = [pool.submit(tile_cell_arrays, j) for j in jobs]
            for fut in tqdm(as_completed(futs), total=len(futs), desc=desc):
                ch, t, arr = fut.result()
                parts[ch][t] = arr
    else:
        for j in tqdm(jobs, desc=desc):
            ch, t, arr = tile_cell_arrays(j)
            parts[ch][t] = arr
    return parts


def stack_cloud(parts_ch, offsets_ch):
    """{tile: 局部数组} + {tile: 偏移} → 一整块全局坐标点云。"""
    chunks = [shift_arrays(arr, offsets_ch[t]) for t, arr in parts_ch.items() if len(arr)]
    return np.vstack(chunks) if chunks else np.empty((0, 9))


def _soma_index(arr):
    """由 (N, 9) 数组直接拼出 _prepare_soma_containment_index 的等价结构。"""
    centroids = arr[:, 0:3].copy()
    x1, y1, x2, y2 = arr[:, 3], arr[:, 4], arr[:, 5], arr[:, 6]
    radii = np.maximum(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / 2, 1.0)
    return dict(centroids=centroids, x1=x1, y1=y1, x2=x2, y2=y2,
                z1=arr[:, 7], z2=arr[:, 8], radii=radii,
                max_radius=float(radii.max()) if len(radii) else 1.0,
                tree=cKDTree(centroids))


def _tf_arrays(arr):
    """_cell_arrays 的等价结构。"""
    return (arr[:, 0:3].copy(), arr[:, 3], arr[:, 4], arr[:, 5], arr[:, 6], arr[:, 7], arr[:, 8])


def estimate_constants_pooled(clouds, channels, ref, ch_types, settings, p, rng):
    """
    全脑汇总估计每个通道的常数 a_c：把所有 tile 的细胞按解出来的位置摆进同一套全局坐标后，
    a_c = 让通道 c 的细胞与参考通道对上所需要的平移（同一个物理结构 G_ref − G_c）。

    soma↔TF 用包含度打分（和 Stage 3B 判共定位是同一个判据）；
    同类型通道（都是 soma / 都是 TF）用质心的位移直方图 + 互为最近邻，和接缝估计同一套。

    返回 {ch: (a, info)}，info 里有得分和参与的细胞数。
    """
    out = {}
    ref_arr = clouds.get(ref, np.empty((0, 9)))
    for ch in channels:
        if ch == ref or len(clouds.get(ch, [])) == 0 or len(ref_arr) == 0:
            continue
        arr = clouds[ch]
        same_type = ch_types.get(ch, 'soma') == ch_types.get(ref, 'soma')
        if same_type:
            A, B = arr[:, 0:3], ref_arr[:, 0:3]     # median(B − A) = a_c
            res = cs.estimate_seam_error(A, B, {
                'win_xy': p['win_xy'], 'win_z': p['win_z'], 'bin_xy': p['bin_xy'],
                'r_xy': p['r_xy'], 'r_z': p['r_z'], 'min_matches': p['min_matches']})
            out[ch] = ((res['dx'], res['dy'], res['dz']),
                       {'method': 'point_match', 'n_match': res['n_match'],
                        'n_ref': len(ref_arr), 'n_ch': len(arr),
                        'mad': (res['mad_x'], res['mad_y'], res['mad_z'])})
            continue
        # soma ↔ TF：谁是 soma 谁当索引；包含度打分求的是「加到 TF 上」的平移
        tf_is_ch = ch_types.get(ch, 'soma') == 'tf'
        soma_arr, tf_arr = (ref_arr, arr) if tf_is_ch else (arr, ref_arr)
        if len(tf_arr) > p['max_cells']:
            tf_arr = tf_arr[rng.choice(len(tf_arr), p['max_cells'], replace=False)]
        if len(soma_arr) > p['max_cells']:
            soma_arr = soma_arr[rng.choice(len(soma_arr), p['max_cells'], replace=False)]
        # 粗搜索的候选点比精搜索还多，用更小的抽样跑（只需把中心定到 ±2 px 内）
        n_coarse = min(p['coarse_cells'], len(tf_arr))
        coarse_arr = tf_arr[rng.choice(len(tf_arr), n_coarse, replace=False)] \
            if n_coarse < len(tf_arr) else tf_arr
        dx, dy, dz, score = containment_shift_from_arrays(
            _soma_index(soma_arr), _tf_arrays(tf_arr),
            xy_range_px=p['win_xy'], z_range_slices=p['win_z'],
            fine_xy_px=p['fine_xy'], fine_z_slices=p['fine_z'],
            max_center_dist_ratio=settings['max_center_dist_ratio'],
            xy_margin=0, z_pad=settings.get('containment_z_pad', 0),
            coarse_tf_arrays=_tf_arrays(coarse_arr))
        sign = 1.0 if tf_is_ch else -1.0
        out[ch] = ((sign * dx, sign * dy, sign * dz),
                   {'method': 'containment', 'score': score,
                    'n_soma': len(soma_arr), 'n_tf': len(tf_arr)})
    return out


def containment_count(soma_arr, tf_arr, settings):
    """在给定摆放下，有多少 TF 核落在某个 soma 内（判据与 Stage 3B 相同）。不搜索。"""
    if len(soma_arr) == 0 or len(tf_arr) == 0:
        return 0, 0.0
    count, _margin = _containment_score(
        _soma_index(soma_arr), _tf_arrays(tf_arr), 0, 0, 0,
        settings['max_center_dist_ratio'], 0, settings.get('containment_z_pad', 0))
    return int(count), float(count) / max(1, len(tf_arr))


def compare_offsets_holdout(parts, tiles, channels, ref, ch_types, settings,
                            off_new, off_old, max_cells=None, rng=None):
    """
    在**没参与估计**的 z 段上，用同一批细胞、同一判据给两套偏移各打一次分：
    新方法（通道场 + 全局常数）vs 旧 Stage 2.5（逐 tile 各自搜出来的偏移）。

    这是唯一公平的"哪个更好"：旧方法是逐 tile 直接最大化这个分数的（45×3 个自由参数），
    在它自己用过的 z 窗上必然占便宜；换到没用过的 z 段，过拟合就不再免费。

    两点要记住：
    - 新方法的 tile 位置 P(t) 来自接缝求解，而接缝用的是整个 z 范围，所以它间接"见过"
      held-out 那一段的 xy/z 摆放。被检验的是**通道偏移**能不能外推，这一点两边对等。
    - 结论对常数的估计质量非常敏感：抽样太小（比如几千个核）时常数本身就是噪声，
      这个对照只会反映那一点，不能用来判方法。看 report 里常数那一行的包含率/配对数。

    返回 {ch: {'new': (count, rate), 'old': (count, rate), 'n_tf': N}}
    """
    out = {}
    soma = stack_cloud(parts[ref], {t: off_new[(ref, t)] for t in tiles})
    for ch in channels:
        if ch == ref or ch_types.get(ch, 'soma') != 'tf':
            continue
        if any((ch, t) not in off_old for t in tiles):
            continue
        tf_new = stack_cloud(parts[ch], {t: off_new[(ch, t)] for t in tiles})
        tf_old = stack_cloud(parts[ch], {t: off_old[(ch, t)] for t in tiles})
        # 两套摆放的行序完全一致（都来自同一个 parts[ch]），所以抽样抽同一批行，
        # 两边比的是**同一批细胞**，对照才公平
        n_all = len(tf_new)
        if max_cells and n_all > max_cells:
            idx = (rng or np.random.default_rng(0)).choice(n_all, max_cells, replace=False)
            tf_new, tf_old = tf_new[idx], tf_old[idx]
        out[ch] = {'new': containment_count(soma, tf_new, settings),
                   'old': containment_count(soma, tf_old, settings),
                   'n_tf': len(tf_new), 'n_tf_all': n_all, 'n_soma': len(soma)}
    return out


def parse_const(specs):
    """--const CH=dx,dy,dz（可重复）→ {CH: (dx, dy, dz)}。"""
    out = {}
    for s in specs or []:
        ch, _, vals = s.partition('=')
        parts = [v for v in vals.replace(' ', '').split(',') if v]
        if not ch or len(parts) != 3:
            raise SystemExit(f"❌ --const '{s}' 格式应为 CH=dx,dy,dz")
        out[ch] = tuple(float(v) for v in parts)
    return out


def load_old_offsets(align_dir):
    """读已有的 0_channel_alignment 逐 tile 偏移 → {tile: {ch: {dx, dy, dz, ...}}}。"""
    old = {}
    if not align_dir or not os.path.isdir(align_dir):
        return old
    for f in os.listdir(align_dir):
        if f.endswith('_offsets.json'):
            with open(os.path.join(align_dir, f), encoding='utf-8') as fh:
                old[f[:-len('_offsets.json')]] = json.load(fh)
    return old


def const_from_offsets(align_dir, tiles, field, channels, ref):
    """
    用已有的 0_channel_alignment 逐 tile 偏移定常数：a_c = median(旧偏移 − 通道场)。
    返回 ({CH: (dx, dy, dz)}, {CH: 各轴的离散度})。旧偏移越可信，离散度越小。
    """
    old = load_old_offsets(align_dir)
    const, spread = {}, {}
    for ch in channels:
        if ch == ref:
            continue
        rows = []
        for k, t in enumerate(tiles):
            o = old.get(t)
            if not o or ch not in o or ref not in o:
                continue
            rows.append([o[ch][f'd{ax}'] - o[ref][f'd{ax}'] - field[ch][k][a]
                         for a, ax in enumerate(AXES)])
        if not rows:
            continue
        arr = np.array(rows, dtype=float)
        med = np.median(arr, axis=0)
        const[ch] = tuple(med)
        spread[ch] = tuple(np.median(np.abs(arr - med), axis=0) * 1.4826)
    return const, spread


# ──────────────────────────────────────────────────────────────────────────────
# 5.  输出
# ──────────────────────────────────────────────────────────────────────────────

def write_channel_xml(template_path, out_path, tiles, positions, common_origin, stacks_dir):
    """
    把模板 XML 里每个 Stack 的 ABS_H/V/D 换成解出来的位置。

    所有通道用同一个原点（参考通道位置的最小值），这样各通道 XML 里的 ABS 可以直接
    互相比较；TeraStitcher merge 时还会各自再归一化一次，所以**细胞坐标必须和图像
    用同一份 XML 推**（Stage 3 的 loadTeraxml 已经是这么做的）。
    """
    tree = ET.parse(template_path)
    root = tree.getroot()
    for node, attr in (('stacks_dir', 'value'), ('mdata_bin', 'value')):
        el = root.find(node)
        if el is not None and stacks_dir:
            el.set(attr, stacks_dir + ('/mdata.bin' if node == 'mdata_bin' else ''))
    pos = {t: positions[k] for k, t in enumerate(tiles)}
    n_set, missing = 0, []
    for s in root.find('STACKS'):
        name = os.path.basename(s.get('DIR_NAME').replace('\\', '/'))
        if name not in pos:
            missing.append(name)
            continue
        p = pos[name] - common_origin
        s.set('ABS_H', str(int(round(p[0]))))
        s.set('ABS_V', str(int(round(p[1]))))
        s.set('ABS_D', str(int(round(p[2]))))
        s.set('STITCHABLE', 'yes')
        n_set += 1
    tree.write(out_path, encoding='UTF-8', xml_declaration=True)
    return n_set, missing


def write_aligned(out_dir, tiles, shifts, det_dir, tile_dirs, routing, mark_done):
    """
    按 0_channel_alignment 的格式写偏移 JSON + 平移后的检测 CSV。
    shifts: {tile: {ch: (dx, dy, dz)}}，取整后写入（下游按整数像素/层处理）。
    """
    os.makedirs(out_dir, exist_ok=True)
    ch_ids = [ch['id'] for ch in routing]
    for t in tqdm(tiles, desc="Aligned CSV"):
        slice_names = None
        tdir = tile_dirs.get(t)
        if tdir and os.path.isdir(tdir):
            slice_names = [os.path.splitext(f)[0] for f in sorted(
                f for f in os.listdir(tdir)
                if f.lower().endswith(('.tif', '.tiff')) and not f.startswith('.'))]
        payload = {}
        for cid in ch_ids:
            dx, dy, dz = (int(round(v)) for v in shifts[t][cid])
            apply_shift_to_csv(os.path.join(det_dir, f"{t}_{cid}_result.csv"), dx, dy, dz,
                               os.path.join(out_dir, f"{t}_{cid}_result.csv"),
                               slice_names=slice_names)
            payload[cid] = {"dx": dx, "dy": dy, "dz": dz, "source": "solve_tile_positions"}
        tmp = os.path.join(out_dir, f"{t}_offsets.json")
        with open(tmp + '.part', 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp + '.part', tmp)
    if mark_done:
        open(os.path.join(out_dir, '_align_done.flag'), 'w').close()


# ──────────────────────────────────────────────────────────────────────────────
# 6.  main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sample', required=True, help='样本根目录（或直接给 detection_results/）')
    ap.add_argument('--config', default=None, help='默认 <results_dir>/runtime_config.json')
    ap.add_argument('--det-dir', default=None, help='原始检测 CSV 目录，默认 1_tile_2d_raw')
    ap.add_argument('--out-dir', default=None,
                    help='默认 <results_dir>/5_analysis_report/tile_positions')
    ap.add_argument('--channels', default=None, help='逗号分隔；默认 config 里所有 active 通道')
    ap.add_argument('--ref', default=None,
                    help='通道对齐的参考通道（通道场和常数都相对它，MADM 流程里就是 GFP）；'
                         '默认 pre_align_params.reference_channel')
    ap.add_argument('--frame', default=None,
                    help='全局坐标系用哪个通道的几何（细胞和配准用的全脑图都落在它上面）；'
                         '默认与 --ref 相同。换成别的通道只是换个规范，不影响测量')
    ap.add_argument('--model', choices=['joint', 'free'], default='joint',
                    help="joint=共用 tile 位置 + 每通道平滑通道场（默认）；free=每通道各解各的")
    ap.add_argument('--workers', type=int, default=1, help='接缝测量的并行进程数')
    ap.add_argument('--redo-seams', action='store_true', help='忽略已有的 seams.csv，重新测')
    # 接缝测量参数（与 compare_stitching.py 的 B 部分同义）
    ap.add_argument('--win-xy', type=float, default=80, help='粗搜索 XY 半径（px），要盖住台面误差')
    ap.add_argument('--win-z', type=float, default=10, help='粗搜索 Z 半径（切片）')
    ap.add_argument('--bin-xy', type=float, default=2, help='差值直方图 XY bin（px）')
    ap.add_argument('--match-xy-soma', type=float, default=10.0, help='soma 通道配对半径（px）')
    ap.add_argument('--match-xy-tf', type=float, default=6.0, help='TF 通道配对半径（px）')
    ap.add_argument('--match-z', type=float, default=1.5, help='配对半径 Z（切片）')
    ap.add_argument('--edge-px', type=float, default=24, help='贴着 tile 边缘的细胞不用（质心有偏）')
    ap.add_argument('--min-matches', type=int, default=10, help='一条接缝至少配上多少细胞才算数')
    ap.add_argument('--n-slices', type=int, default=None, help='每个 tile 的切片数；默认按 CSV 的最大 z')
    # 求解参数
    ap.add_argument('--prior-w', type=float, default=1e-3,
                    help='把位置拉回名义网格的先验权重；只用于定全局平移和撑住没有接缝的 tile')
    ap.add_argument('--huber', type=float, default=3.0, help='鲁棒重加权阈值（相对残差 MAD）')
    ap.add_argument('--iters', type=int, default=8, help='IRLS 迭代次数')
    # 通道常数
    ap.add_argument('--const', action='append', metavar='CH=dx,dy,dz',
                    help='通道相对参考通道的全局常数偏移，可重复；接缝解不出这一项')
    ap.add_argument('--const-from', choices=['pooled', 'offsets', 'none'], default='pooled',
                    help="没给 --const 的通道怎么定常数：pooled=把全脑所有 tile 的细胞汇总起来"
                         "一次估 3 个数（默认）；offsets=取已有 0_channel_alignment 的中位数；none=当作 0")
    ap.add_argument('--const-z-window', type=int, default=None,
                    help='汇总估计取每个 tile 中心多少层；默认 pre_align_params.sample_z_center_count')
    ap.add_argument('--const-win-xy', type=float, default=60, help='常数粗搜索 XY 半径（px）')
    ap.add_argument('--const-win-z', type=int, default=15, help='常数粗搜索 Z 半径（层）')
    ap.add_argument('--const-fine-xy', type=int, default=3, help='包含度精搜索 XY 半径（px）')
    ap.add_argument('--const-fine-z', type=int, default=2, help='包含度精搜索 Z 半径（层）')
    ap.add_argument('--const-max-cells', type=int, default=80000,
                    help='包含度精搜索时每个点云最多用多少细胞（随机抽样，只为控制耗时）')
    ap.add_argument('--const-coarse-cells', type=int, default=25000,
                    help='包含度粗搜索用多少细胞；粗搜索候选多、精度要求低，用更小的抽样')
    ap.add_argument('--const-seed', type=int, default=0, help='上面那个抽样的随机种子')
    ap.add_argument('--no-compare-old', action='store_true',
                    help='不做 held-out 对照（默认只要有 0_channel_alignment 就做）')
    ap.add_argument('--holdout-gap', type=float, default=1.5,
                    help='held-out z 段离中心窗多远，单位是窗厚；1.5 = 隔开半个窗（默认）')
    ap.add_argument('--holdout-max-cells', type=int, default=300000,
                    help='held-out 对照最多用多少个核（两套偏移抽同一批，只为控制耗时/内存）')
    # 可选输出
    ap.add_argument('--write-xml', action='store_true', help='每通道写一份 xml_merging.xml')
    ap.add_argument('--xml-template', default=None,
                    help='XML 模板；默认找各通道目录下的 xml_merging.xml / xml_import.xml，'
                         '找不到就用参考通道的模板改写路径')
    ap.add_argument('--write-aligned', action='store_true',
                    help='写 0_channel_alignment 格式的偏移 JSON + 平移后的 CSV')
    ap.add_argument('--aligned-dir', default=None,
                    help='默认 <results_dir>/0_channel_alignment_solved（不覆盖原结果）')
    ap.add_argument('--mark-done', action='store_true',
                    help='在 --write-aligned 的目录里写 _align_done.flag，让流程跳过 Stage 2.5')
    ap.add_argument('--force', action='store_true', help='允许写入已有内容的目录')
    return ap.parse_args()


def main():
    args = parse_args()
    sample_dir = os.path.abspath(args.sample)
    results_dir = cs.resolve_results_dir(sample_dir)
    if results_dir is None:
        raise SystemExit(f"❌ 在 {sample_dir} 下找不到 detection_results（要有 1_tile_2d_raw）")
    cfg_path = args.config or os.path.join(results_dir, 'runtime_config.json')
    if not os.path.isfile(cfg_path):
        raise SystemExit(f"❌ 找不到 config：{cfg_path}")
    config = load_config(cfg_path)
    routing = [ch for ch in config.get('channels_routing', []) if ch.get('active', True)]
    ch_types = {ch['id']: ch.get('type', 'soma') for ch in routing}
    settings = resolve_align_settings(config, routing)
    dp = config.get('detection_params', {})
    xy_um = dp.get('xy_resolution_um', 0.65)
    z_um = dp.get('z_resolution_um', 8.0)
    tile_size = dp.get('tILESIZE', 2048)

    channels = [c.strip() for c in args.channels.split(',')] if args.channels else [ch['id'] for ch in routing]
    unknown = [c for c in channels if c not in ch_types]
    if unknown:
        raise SystemExit(f"❌ 通道 {unknown} 不在 channels_routing {list(ch_types)} 里")
    ref = args.ref or settings['reference_channel']
    if ref not in channels:
        raise SystemExit(f"❌ 参考通道 {ref} 不在 --channels {channels} 里")
    frame = args.frame or ref
    if frame not in channels:
        raise SystemExit(f"❌ 坐标系通道 {frame} 不在 --channels {channels} 里")

    det_dir = args.det_dir or os.path.join(results_dir, '1_tile_2d_raw')
    out_dir = args.out_dir or os.path.join(results_dir, '5_analysis_report', 'tile_positions')
    os.makedirs(out_dir, exist_ok=True)

    tiles, partial = discover_tiles(det_dir, channels)
    if not tiles:
        raise SystemExit(f"❌ {det_dir} 里没有 {channels} 都齐全的 tile")
    grid = nominal_grid(tiles, xy_um)
    n_row = max(d['rc'][0] for d in grid.values()) + 1
    n_col = max(d['rc'][1] for d in grid.values()) + 1
    ch_dirs = channel_image_dirs(config, routing, sample_dir)
    # 对齐后的 z 用的是坐标系通道的层号，所以 slice_name 也从它的 tile 目录取
    # （T4 上四个通道的层名和层数完全一致，这里只是让语义对上）
    frame_img_dir = ch_dirs.get(frame, '')
    tile_dirs = {t: (os.path.join(frame_img_dir, t.split('_')[0], t) if frame_img_dir else None)
                 for t in tiles}
    if args.n_slices:
        n_slices, z_src = args.n_slices, '--n-slices'
    else:
        n_slices, z_src = count_slices(det_dir, tiles, channels, tile_dirs)

    log = []

    def say(msg=''):
        print(msg)
        log.append(msg)

    say(f"样本     : {os.path.basename(sample_dir.rstrip(os.sep))}  ({sample_dir})")
    say(f"结果目录 : {results_dir}")
    say(f"通道     : {', '.join(channels)}   对齐参考 = {ref}"
        + (f"   坐标系 = {frame}（细胞和配准图都落在它上面）" if frame != ref else ""))
    say(f"网格     : {n_row} 行 × {n_col} 列，{len(tiles)} 个 tile，每 tile {n_slices} 层（{z_src}）")
    if partial:
        say(f"⚠️  {len(partial)} 个 tile 缺部分通道的 CSV，已跳过：{partial[:5]}")
    say(f"输出     : {out_dir}")

    # ── 接缝测量 ──
    seam_csv = os.path.join(out_dir, 'seams.csv')
    if os.path.isfile(seam_csv) and not args.redo_seams:
        seams_all = add_measured(pd.read_csv(seam_csv), grid)
        say(f"\n接缝     : 复用 {seam_csv}（要重测加 --redo-seams）")
    else:
        params = {'det_dir': det_dir, 'tile_size': tile_size, 'edge_px': args.edge_px,
                  'win_xy': args.win_xy, 'win_z': args.win_z, 'bin_xy': args.bin_xy,
                  'r_z': args.match_z, 'min_matches': args.min_matches,
                  'r_xy_soma': args.match_xy_soma, 'r_xy_tf': args.match_xy_tf}
        jobs = build_jobs(tiles, grid, channels, ch_types, settings, params,
                          n_slices, {'x': xy_um, 'y': xy_um, 'z': z_um})
        say(f"\n接缝     : {len(jobs)} 个任务（{len(jobs) // max(len(channels), 1)} 条接缝 × "
            f"{len(channels)} 通道），{args.workers} 进程")
        seams_all = measure_seams(jobs, grid, args.workers)
        seams_all.to_csv(seam_csv, index=False)

    ok = seams_all[seams_all['status'] == 'ok'].reset_index(drop=True)
    say("\n接缝测量质量（成功的接缝）")
    say(f"  {'通道':<8}{'成功/总数':>12}{'配对数 p50':>12}{'配对率 p50':>12}"
        f"{'配对差 MAD (x,y,z)':>24}{'位移标准误 (x,y)':>20}")
    for ch in channels:
        g = ok[ok.channel == ch]
        a = seams_all[seams_all.channel == ch]
        if g.empty:
            say(f"  {ch:<8}{'0/' + str(len(a)):>12}   —")
            continue
        sem = [float(np.median(seam_weights(g, ax) ** -1)) for ax in ('x', 'y')]
        say(f"  {ch:<8}{f'{len(g)}/{len(a)}':>12}{g.n_match.median():>12.0f}"
            f"{g.match_frac.median():>12.2f}"
            f"{f'{g.mad_x.median():.2f}, {g.mad_y.median():.2f}, {g.mad_z.median():.2f}':>24}"
            f"{f'{sem[0]:.3f}, {sem[1]:.3f}':>20}")
    bad = seams_all[seams_all['status'] != 'ok']
    if not bad.empty:
        say(f"  没结果的接缝 {len(bad)} 条：{dict(bad.status.value_counts())}")
        empty = sorted(set(bad.tile) | set(bad.nb_tile))
        say(f"  涉及 tile：{empty}")

    # ── 求解 ──
    sol = {ax: solve_axis(ok, tiles, grid, channels, ref, ax, args.model,
                          args.prior_w, args.huber, args.iters) for ax in AXES}
    say(f"\n求解（model={args.model}）：接缝拟合残差 |resid| p50 / p90 / max，单位 px（z 为层）")
    for ax in AXES:
        r = np.abs(sol[ax]['resid'])
        r = r[np.isfinite(r)]
        q = np.percentile(r, [50, 90, 100]) if r.size else [np.nan] * 3
        say(f"  {ax}: {q[0]:6.2f} {q[1]:6.2f} {q[2]:6.2f}")
    if args.model == 'joint':
        say("\n通道位移场斜率（相对参考通道；px/行、px/列，z 为层）")
        for ch in channels:
            if ch == ref:
                continue
            s = [sol[ax]['slope'][ch] for ax in AXES]
            say(f"  {ch:<8}" + "  ".join(f"{ax}: ({s[k][0]:+.2f} 每行, {s[k][1]:+.2f} 每列)"
                                         for k, ax in enumerate(AXES)))

    # 每 tile 的通道场（不含常数）
    field = {ch: np.array([[sol[ax]['P_ch'][ch][k] - sol[ax]['P_ch'][ref][k] for ax in AXES]
                           for k in range(len(tiles))]) for ch in channels}

    # ── 通道常数 a_c ──
    # 接缝只约束通道场的差，常数解不出来，必须靠跨通道匹配。这里把全脑所有 tile 的细胞
    # 按解出来的位置摆进同一套全局坐标，一次只估 3 个数（旧 Stage 2.5 是每个 tile 各估一次）。
    const = parse_const(args.const)
    const_src = {ch: 'cli' for ch in const}
    const_info = {}
    solution_cmp = {}
    align_dir = os.path.join(results_dir, '0_channel_alignment')
    todo = [ch for ch in channels if ch != ref and ch not in const]

    if args.const_from == 'pooled' and todo:
        z_half = (args.const_z_window or settings['sample_z_center_count']) / 2.0
        say(f"\n全脑汇总估常数：每个 tile 取中心 ±{z_half:.0f} 层，"
            f"粗搜索 ±{args.const_win_xy:.0f} px / ±{args.const_win_z} 层")
        pooled_chs = [ref] + todo
        # 每个 tile 的中心 z（参考通道检测的中位数）→ 换成全局 z；各通道的局部窗由 pool_cells 换算
        z_center = {}
        for k, t in enumerate(tiles):
            z = pd.read_csv(os.path.join(det_dir, f"{t}_{ref}_result.csv"), usecols=['z'])['z']
            z_center[t] = (float(z.median()) if len(z) else 0.0) + sol['z']['P_ch'][ref][k]
        # 解出来的 tile 位置（各通道各一套）
        off_P = {(ch, t): np.array([sol[ax]['P_ch'][ch][k] for ax in AXES])
                 for ch in pooled_chs for k, t in enumerate(tiles)}
        win_fit = {t: (z_center[t] - z_half, z_center[t] + z_half) for t in tiles}
        parts = pool_cells(tiles, pooled_chs, det_dir, win_fit, ch_types, settings,
                           off_P, args.workers, "Pooling")
        clouds = {ch: stack_cloud(parts[ch], {t: off_P[(ch, t)] for t in tiles})
                  for ch in pooled_chs}
        say("  汇总到的细胞数：" + "，".join(f"{ch} {len(clouds[ch])}" for ch in pooled_chs))
        est = estimate_constants_pooled(
            clouds, pooled_chs, ref, ch_types, settings,
            {'win_xy': args.const_win_xy, 'win_z': args.const_win_z, 'bin_xy': args.bin_xy,
             'r_xy': args.match_xy_soma, 'r_z': args.match_z, 'min_matches': args.min_matches,
             'fine_xy': args.const_fine_xy, 'fine_z': args.const_fine_z,
             'max_cells': args.const_max_cells, 'coarse_cells': args.const_coarse_cells},
            np.random.default_rng(args.const_seed))
        for ch, (v, info) in est.items():
            const[ch], const_src[ch], const_info[ch] = v, 'pooled', info

        # ── held-out 对照：新方法 vs 旧 Stage 2.5，同一批细胞、同一判据 ──
        old_off = load_old_offsets(align_dir)
        if not args.no_compare_old and old_off:
            shift = z_half * 2 * args.holdout_gap
            win_hold = {t: (z_center[t] + shift - z_half, z_center[t] + shift + z_half)
                        for t in tiles}
            say(f"\nheld-out 对照：另取每个 tile 中心 +{shift:.0f} 层处的同样厚度 z 段"
                f"（没参与估计），给两套偏移各打一次包含度")
            parts_h = pool_cells(tiles, pooled_chs, det_dir, win_hold, ch_types, settings,
                                 off_P, args.workers, "Holdout")
            # 新方法：各通道用自己的 tile 位置；旧方法：参考通道的位置 + 旧的逐 tile 偏移
            off_new = {k: v for k, v in off_P.items()}
            off_old = {}
            for ch in pooled_chs:
                for k, t in enumerate(tiles):
                    o = old_off.get(t, {})
                    if ch == ref or (ch in o and ref in o):
                        d = np.zeros(3) if ch == ref else np.array(
                            [o[ch][f'd{ax}'] - o[ref][f'd{ax}'] for ax in AXES], dtype=float)
                        off_old[(ch, t)] = off_P[(ref, t)] + d
            cmp_res = compare_offsets_holdout(parts_h, tiles, pooled_chs, ref, ch_types,
                                              settings, off_new, off_old,
                                              max_cells=args.holdout_max_cells,
                                              rng=np.random.default_rng(args.const_seed))
            for ch, r in cmp_res.items():
                n, rate = r['new']
                o, orate = r['old']
                verdict = '新方法更好' if n > o else ('旧方法更好' if o > n else '打平')
                say(f"  {ch:<8}新方法 {n:6d} 个核落在 soma 内（{rate:.4f}）   "
                    f"旧 Stage2.5 {o:6d}（{orate:.4f}）   {verdict}"
                    f"   [用了 {r['n_tf']}/{r['n_tf_all']} 核 vs {r['n_soma']} soma]")
            say("  注：旧方法是逐 tile 直接最大化这个分数的（45×3 个自由参数），"
                "在它自己用过的 z 窗上必然占便宜；换到没用过的 z 段才公平。")
            solution_cmp = {ch: {k: (list(v) if isinstance(v, tuple) else v)
                                 for k, v in r.items()} for ch, r in cmp_res.items()}
        else:
            solution_cmp = {}
    elif args.const_from == 'offsets' and os.path.isdir(align_dir) and todo:
        auto, _ = const_from_offsets(align_dir, tiles, field, channels, ref)
        for ch, v in auto.items():
            if ch not in const:
                const[ch], const_src[ch] = v, 'old_offsets_median'

    say("\n通道常数 a_c（相对对齐参考 %s；接缝解不出这一项）" % ref)
    old_auto, old_spread = ({}, {})
    if os.path.isdir(align_dir):
        old_auto, old_spread = const_from_offsets(align_dir, tiles, field, channels, ref)
    for ch in channels:
        if ch == ref:
            continue
        v = const.get(ch, (0.0, 0.0, 0.0))
        info = const_info.get(ch, {})
        detail = ''
        if info.get('method') == 'containment':
            detail = f"  包含率 {info['score']:.3f}（{info['n_tf']} 核 vs {info['n_soma']} soma）"
        elif info.get('method') == 'point_match':
            m = info['mad']
            detail = (f"  配对 {info['n_match']}，配对差 MAD "
                      f"({m[0]:.1f}, {m[1]:.1f}, {m[2]:.1f})")
        say(f"  {ch:<8}({v[0]:+6.1f}, {v[1]:+6.1f}, {v[2]:+6.1f})  "
            f"来源={const_src.get(ch, 'zero')}{detail}")
        if ch in old_auto:
            o, sp = old_auto[ch], old_spread[ch]
            say(f"  {'':<8}对照旧 Stage 2.5：中位数 ({o[0]:+6.1f}, {o[1]:+6.1f}, {o[2]:+6.1f})"
                f"  逐 tile 离散度 ({sp[0]:.1f}, {sp[1]:.1f}, {sp[2]:.1f})"
                f"  与本次差 ({v[0] - o[0]:+.1f}, {v[1] - o[1]:+.1f}, {v[2] - o[2]:+.1f})")
    if not const:
        say("  ⚠️  没有任何常数（--const-from none），通道场只有相对变化，绝对位移按 0 处理。")

    # ── 逐 tile 结果表 ──
    # field_/const 都是相对对齐参考 ref（= 实际测量出来的量）；
    # s_ 是把该通道的原始 tile 坐标搬到**坐标系通道 frame** 上要加的量：
    #     s_c(t) = [delta_c(t) + a_c] − [delta_frame(t) + a_frame]
    # frame == ref 时后一项为 0，就是原来的定义。换 frame 只是换个规范，不重测任何东西。
    deg = seam_support(ok, tiles, channels)
    zero3 = (0.0, 0.0, 0.0)
    rows = []
    for k, t in enumerate(tiles):
        row = {'tile': t, 'grid_row': grid[t]['rc'][0], 'grid_col': grid[t]['rc'][1]}
        for a, ax in enumerate(AXES):
            row[f'nom_{ax}'] = grid[t]['pos'][a]
            row[f'P_{ax}'] = sol[ax]['P'][k]
        c_frame = const.get(frame, zero3)
        for ch in channels:
            row[f'n_seams_{ch}'] = deg[ch][t]
            c = const.get(ch, zero3)
            for a, ax in enumerate(AXES):
                row[f'P_{ch}_{ax}'] = sol[ax]['P_ch'][ch][k]
                row[f'field_{ch}_{ax}'] = field[ch][k][a]
                row[f's_{ch}_{ax}'] = ((field[ch][k][a] + c[a])
                                       - (field[frame][k][a] + c_frame[a]))
        rows.append(row)
    pos_df = pd.DataFrame(rows)
    pos_df.to_csv(os.path.join(out_dir, 'tile_positions.csv'), index=False)

    unsupported = [t for t in tiles if all(deg[ch][t] == 0 for ch in channels)]
    if unsupported:
        say(f"\n⚠️  {len(unsupported)} 个 tile 一条接缝都没有，位置只能用名义网格："
            f"{unsupported}（通常是没有细胞的边角 tile）")
    say("\n解出来的位置相对名义网格的偏离（px，z 为层）")
    for ax in AXES:
        d = pos_df[f'P_{ax}'] - pos_df[f'nom_{ax}']
        say(f"  {ax}: p50 {d.abs().median():6.1f}   max {d.abs().max():6.1f}")

    # ── solution.json ──
    solution = {
        'created': datetime.now().isoformat(timespec='seconds'),
        'sample': sample_dir, 'model': args.model, 'reference_channel': ref,
        'frame_channel': frame,
        'channels': channels, 'n_tiles': len(tiles), 'grid': [n_row, n_col],
        'xy_resolution_um': xy_um, 'z_resolution_um': z_um, 'tile_size': tile_size,
        'seam_params': {k: getattr(args, k) for k in
                        ('win_xy', 'win_z', 'bin_xy', 'match_xy_soma', 'match_xy_tf',
                         'match_z', 'edge_px', 'min_matches')},
        'solver': {'prior_w': args.prior_w, 'huber': args.huber, 'iters': args.iters},
        'slopes': ({ch: {ax: list(sol[ax]['slope'][ch]) for ax in AXES} for ch in channels}
                   if args.model == 'joint' else None),
        'const': {ch: list(v) for ch, v in const.items()},
        'const_source': const_src,
        'const_info': {ch: {k: (list(v) if isinstance(v, tuple) else v) for k, v in info.items()}
                       for ch, info in const_info.items()},
        'holdout_compare': solution_cmp,
        'const_params': {'z_window': args.const_z_window or settings['sample_z_center_count'],
                         'win_xy': args.const_win_xy, 'win_z': args.const_win_z,
                         'fine_xy': args.const_fine_xy, 'fine_z': args.const_fine_z,
                         'max_cells': args.const_max_cells,
                         'coarse_cells': args.const_coarse_cells, 'seed': args.const_seed},
        'seam_fit_resid_p50': {ax: float(np.nanpercentile(np.abs(sol[ax]['resid']), 50))
                               for ax in AXES},
        'tiles_without_seams': unsupported,
    }
    with open(os.path.join(out_dir, 'solution.json'), 'w', encoding='utf-8') as f:
        json.dump(solution, f, indent=2, ensure_ascii=False)

    # ── 可选：各通道 XML ──
    if args.write_xml:
        say("\n写各通道 xml_merging.xml")
        common_origin = np.array([pos_df[f'P_{frame}_{ax}'].min() for ax in AXES])
        ref_tmpl = args.xml_template
        if not ref_tmpl:
            for cand in ('xml_merging.xml', 'xml_import.xml'):
                p = os.path.join(ch_dirs.get(ref, ''), cand)
                if os.path.isfile(p):
                    ref_tmpl = p
                    break
        for ch in channels:
            tmpl = None
            for cand in ('xml_merging.xml', 'xml_import.xml'):
                p = os.path.join(ch_dirs.get(ch, ''), cand)
                if os.path.isfile(p):
                    tmpl = p
                    break
            tmpl = tmpl or ref_tmpl
            if not tmpl:
                say(f"  {ch:<8}跳过：找不到 XML 模板（用 --xml-template 指定）")
                continue
            out_xml = os.path.join(out_dir, f'xml_merging_{ch}.xml')
            P = np.column_stack([pos_df[f'P_{ch}_{ax}'].to_numpy() for ax in AXES])
            n_set, missing = write_channel_xml(tmpl, out_xml, list(pos_df.tile), P,
                                               common_origin, ch_dirs.get(ch, '').replace('/', '\\'))
            note = f"，模板里有 {len(missing)} 个 tile 不在结果里" if missing else ""
            say(f"  {ch:<8}{n_set} 个 tile ← 模板 {os.path.basename(tmpl)}"
                f"{'（借用 ' + ref + ' 的）' if tmpl == ref_tmpl and ch != ref else ''}{note}")
        say("  注意：TeraStitcher merge 会按各自 XML 的最小 ABS 再归一化一次，"
            "细胞坐标必须和图像用同一份 XML 推。")
        say(f"  细胞落在 {frame} 的坐标系里"
            + (f"（通道对齐仍以 {ref} 为参考，只是最后换算过去）" if frame != ref else "") + "，所以：")
        say(f"    1) 配准用的全脑图必须用 xml_merging_{frame}.xml merge（哪怕图像本身是别的通道）；")
        say(f"    2) config 的 paths.pATHXML 指到同一份，否则流程会退回去捡通道目录里的旧 XML：")
        say(f'       "pATHXML": "{os.path.join(out_dir, f"xml_merging_{frame}.xml")}"')
        if frame == ref:
            say(f"    想换个通道的几何当坐标系（比如拿 488 做配准），重跑时加 --frame <通道>。")

    # ── 可选：写 0_channel_alignment 格式 ──
    if args.write_aligned:
        aligned_dir = args.aligned_dir or os.path.join(results_dir, '0_channel_alignment_solved')
        if os.path.isdir(aligned_dir) and os.listdir(aligned_dir) and not args.force:
            raise SystemExit(f"❌ {aligned_dir} 非空；换个 --aligned-dir 或加 --force")
        missing_const = [ch for ch in channels if ch != ref and ch not in const]
        if missing_const:
            raise SystemExit(f"❌ 通道 {missing_const} 没有常数 a_c，写出来的偏移会差一个整体平移。"
                             f"用 --const 指定，或 --const-from offsets。")
        shifts = {t: {ch: tuple(pos_df.loc[k, f's_{ch}_{ax}'] for ax in AXES) for ch in channels}
                  for k, t in enumerate(tiles)}
        say(f"\n写对齐结果到 {aligned_dir}（偏移是搬到 {frame} 坐标系的量，{frame} 自身为 0）")
        write_aligned(aligned_dir, tiles, shifts, det_dir, tile_dirs, routing, args.mark_done)
        say(f"  {len(tiles)} 个 tile × {len(channels)} 通道"
            + ("，已写 _align_done.flag" if args.mark_done else
               "，没写 _align_done.flag（流程仍会跑 Stage 2.5，除非把这个目录顶替 0_channel_alignment）"))

    with open(os.path.join(out_dir, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(log) + '\n')
    print(f"\n结果写到 {out_dir}")


if __name__ == '__main__':
    mp.freeze_support()
    main()
