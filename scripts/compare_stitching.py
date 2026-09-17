# -*- coding: utf-8 -*-
"""
比较同一个样本用不同通道做出来的 TeraStitcher 拼接（例如 640nm vs 488nm）哪个更准，
以及「通道对齐参考 ≠ 拼接参考」时，细胞最终落到拼接坐标系里的误差有多大。

三部分检查，输入到了哪一步就跑到哪一步：

A. TeraStitcher 自评（只要 XML）
   每对相邻 tile 读 xml_displcomp / displproj / displthres / merging：
     comp_std     同一对 tile 在各 z 子块上算出的位移的离散度（只算 reliability ≥ 阈值的子块）
     replaced     该轴在 displthres 里 reliability=0，即被换成了机械默认位移（等于没拼上）
     place_resid  merging 里两 tile 的最终位置差 − 该对 tile 自己的位移。TeraStitcher 按
                  最小生成树摆放 tile，不在树上的边会出现残差，残差大 = 环路不自洽
   注意：nccPeak / reliability 的绝对值受图像内容影响（稠密核信号天然偏高），不宜跨通道直接比，
   replaced 比例和 comp_std 更可比。

B. 接缝残差（需要 XML + 1_tile_2d_raw 检测结果）——独立于 TeraStitcher 的检验
   相邻 tile 重叠区里，同一个细胞在两个 tile 各被检测一次。只取重叠区的检测做轻量 z-link，
   用 XML 把两边的 3D 细胞质心放到全局坐标，先用差值直方图找粗略平移，再做互为最近邻配对，
   配对差值的中位数 (B − A) 就是这条接缝的拼接误差，理想值为 0。
   每份 XML 都用它自己那个通道的原始检测（NAME:CHANNEL 里的 CHANNEL）测一遍；
   --also 里的通道则先用 0_channel_alignment 的逐 tile 偏移换到该通道坐标系再测：
       p_frame = p_raw + s_ch(tile) − s_frame(tile)
   例如 --xml 488nm:Olig2 --also GFP 测的就是「GFP 细胞经对齐后放进 488 拼接」的最终误差。

C. 两份 XML 交叉验证（需要两份带通道的 XML + 0_channel_alignment 偏移）
   同一 tile 在两份 XML 里的位置差应等于两个通道的逐 tile 偏移差（外加一个全局常数）：
       pos_A(t) − pos_B(t) = s_a(t) − s_b(t) + C
   残差 R(t) 去掉中位数后应接近 0；R 的离散度明显小于位置差本身的离散度，说明两份拼接和
   通道对齐互相印证。个别 tile 的 R 很大 = 该 tile 的拼接或对齐有一方出错。

偏移符号约定与 point_cloud_aligner.apply_shift_to_csv 一致：aligned = raw + s。
TeraStitcher 约定：邻居位置 − 本 tile 位置 = displ（EAST/SOUTH 分别对应右/下邻居）。

用法
----
  # 488 拼接还没出来时，先看 640 的自评和接缝残差
  python scripts/compare_stitching.py --sample Y:/Fengyi/EGFR_brain/T4 --xml 640nm:GFP

  # 两份都齐了
  python scripts/compare_stitching.py --sample Y:/Fengyi/EGFR_brain/T4 \
      --xml 640nm:GFP --xml 488nm:Olig2 --also GFP --workers 4

  --xml 的格式是 NAME[:CHANNEL][=PATH]：
    NAME     标签，同时也是默认目录名（<sample>/<NAME>/xml_*.xml）
    CHANNEL  该拼接所用图像对应的检测通道 id（channels_routing 里的 id）；不给就只做 A 部分
    PATH     XML 所在目录或 xml_merging.xml 文件本身，不在默认位置时用

输出（默认 <results_dir>/5_analysis_report/stitch_compare/）
  <NAME>_pairs.csv          A 部分，每对相邻 tile 一行
  seams.csv                 B 部分，每个 (XML, 通道, 接缝) 一行
  cross_<A>_vs_<B>.csv      C 部分，每个 tile 一行
  *.png                     对应的网格图 / 散点图（--no-plots 跳过）
"""

import argparse
import itertools
import json
import os
import sys
import xml.etree.ElementTree as ET
from functools import lru_cache

if hasattr(sys.stdout, 'reconfigure'):
    # 逐行刷新：画图环境崩溃时进程会直接退出，块缓冲里的输出会丢
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    sys.stderr.reconfigure(encoding='utf-8')

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree
from tqdm import tqdm

from src.config.loader import load_config
from src.core.z_linker import run_z_linker
from src.core.point_cloud_aligner import ALIGN_SETTINGS_FILE, resolve_align_settings

# XML 轴名 → 本项目坐标轴。H=列=x，V=行=y，D=切片=z
AXES = (('x', 'H'), ('y', 'V'), ('z', 'D'))
DIRECTIONS = (('EAST', 0, 1), ('SOUTH', 1, 0))
BOX_COLS = ["x1", "y1", "x2", "y2", "score", "mean", "class", "z"]


# ──────────────────────────────────────────────────────────────────────────────
# 输入解析
# ──────────────────────────────────────────────────────────────────────────────

def resolve_results_dir(sample):
    """--sample 可以是样本根目录，也可以直接是 detection_results；找不到返回 None。"""
    for cand in (os.path.join(sample, 'detection_results'), sample):
        if os.path.isfile(os.path.join(cand, 'runtime_config.json')) or \
                os.path.isdir(os.path.join(cand, '1_tile_2d_raw')):
            return os.path.abspath(cand)
    return None


def parse_xml_spec(spec, sample_dir):
    """NAME[:CHANNEL][=PATH] → dict。"""
    path = None
    if '=' in spec:
        spec, path = spec.split('=', 1)
    name, _, channel = spec.partition(':')
    if not name:
        raise SystemExit(f"❌ --xml '{spec}' 缺少 NAME")
    path = path or os.path.join(sample_dir, name)
    if os.path.isfile(path):
        xml_dir, merging = os.path.dirname(path), path
    else:
        xml_dir, merging = path, os.path.join(path, 'xml_merging.xml')
    return {'name': name, 'channel': channel or None, 'dir': xml_dir, 'merging': merging}


def load_channel_settings(results_dir, config_path):
    """
    返回 (routing, settings)。settings 只用到 z_link / 分辨率，与 Stage 2.5 同源：
    有 _align_settings.json 就用它，没有就按 config 解析。
    """
    cfg_path = config_path or (os.path.join(results_dir, 'runtime_config.json') if results_dir else None)
    if not cfg_path or not os.path.isfile(cfg_path):
        return None, None
    config = load_config(cfg_path)
    routing = [ch for ch in config.get('channels_routing', []) if ch.get('active', True)]
    saved = os.path.join(results_dir or '', '0_channel_alignment', ALIGN_SETTINGS_FILE)
    if results_dir and os.path.isfile(saved):
        with open(saved, encoding='utf-8') as f:
            settings = json.load(f)
    else:
        settings = resolve_align_settings(config, routing)
    return routing, settings


def load_offsets(align_dir):
    """tile → {ch: (dx, dy, dz)}；目录不存在返回 {}。"""
    out = {}
    if not align_dir or not os.path.isdir(align_dir):
        return out
    suffix = '_offsets.json'
    for fname in os.listdir(align_dir):
        if fname.endswith(suffix):
            with open(os.path.join(align_dir, fname), encoding='utf-8') as f:
                d = json.load(f)
            out[fname[:-len(suffix)]] = {ch: (v.get('dx', 0), v.get('dy', 0), v.get('dz', 0))
                                         for ch, v in d.items()}
    return out


def frame_shift(offsets, tile, ch, frame):
    """把 ch 通道原始坐标换到 frame 通道原始坐标系要加的量；缺数据返回 None。"""
    if ch == frame:
        return (0, 0, 0)
    off = offsets.get(tile)
    if not off or ch not in off or frame not in off:
        return None
    return tuple(a - b for a, b in zip(off[ch], off[frame]))


# ──────────────────────────────────────────────────────────────────────────────
# A. TeraStitcher 自评
# ──────────────────────────────────────────────────────────────────────────────

def read_terastitcher(path):
    """返回 {'meta': ..., 'stacks': {(r, c): Element}}；文件不存在返回 None。"""
    if not os.path.isfile(path):
        return None
    root = ET.parse(path).getroot()
    dims = root.find('dimensions')
    vox = root.find('voxel_dims')
    meta = {
        'n_row': int(dims.get('stack_rows')), 'n_col': int(dims.get('stack_columns')),
        'n_slices': int(dims.get('stack_slices')),
        'res': {'x': float(vox.get('H')), 'y': float(vox.get('V')), 'z': float(vox.get('D'))},
    }
    stacks = {(int(s.get('ROW')), int(s.get('COL'))): s for s in root.find('STACKS')}
    return {'meta': meta, 'stacks': stacks}


def stack_positions(ts):
    """(r, c) → (tile_name, np.array([ABS_H, ABS_V, ABS_D]), stitchable)。"""
    out = {}
    for rc, s in ts['stacks'].items():
        name = os.path.basename(s.get('DIR_NAME').replace('\\', '/'))
        pos = np.array([int(s.get('ABS_H')), int(s.get('ABS_V')), int(s.get('ABS_D'))])
        out[rc] = (name, pos, s.get('STITCHABLE', 'yes') == 'yes')
    return out


def _displacements(stack, direction):
    node = stack.find(f'{direction}_displacements')
    if node is None:
        return []
    out = []
    for d in node.findall('Displacement'):
        out.append({ax: {k: float(d.find(tag).get(k, 'nan'))
                         for k in ('displ', 'default_displ', 'reliability', 'nccPeak')}
                    for ax, tag in AXES})
    return out


def neighbor_pairs(positions):
    """[(direction, (r, c), (r2, c2))]，只含两边都在 XML 里的相邻对。"""
    out = []
    for (r, c) in sorted(positions):
        for direction, dr, dc in DIRECTIONS:
            if (r + dr, c + dc) in positions:
                out.append((direction, (r, c), (r + dr, c + dc)))
    return out


def self_report(spec, rel_thr):
    """A 部分：每对相邻 tile 一行。"""
    merging = read_terastitcher(spec['merging'])
    comp = read_terastitcher(os.path.join(spec['dir'], 'xml_displcomp.xml'))
    proj = read_terastitcher(os.path.join(spec['dir'], 'xml_displproj.xml'))
    thres = read_terastitcher(os.path.join(spec['dir'], 'xml_displthres.xml'))
    positions = stack_positions(merging)
    rows = []
    for direction, a, b in neighbor_pairs(positions):
        row = {'xml': spec['name'], 'direction': direction,
               'row': a[0], 'col': a[1], 'nb_row': b[0], 'nb_col': b[1],
               'tile': positions[a][0], 'nb_tile': positions[b][0]}
        placed = positions[b][1] - positions[a][1]
        c_list = _displacements(comp['stacks'][a], direction) if comp else []
        p_list = _displacements(proj['stacks'][a], direction) if proj else []
        t_list = _displacements(thres['stacks'][a], direction) if thres else []
        for i, (ax, _tag) in enumerate(AXES):
            rel_sub = [d[ax] for d in c_list if d[ax]['reliability'] >= rel_thr]
            vals = np.array([d['displ'] for d in rel_sub])
            row[f'{ax}_comp_n'] = len(c_list)
            row[f'{ax}_comp_n_rel'] = len(rel_sub)
            row[f'{ax}_comp_median'] = float(np.median(vals)) if vals.size else np.nan
            row[f'{ax}_comp_std'] = float(np.std(vals)) if vals.size > 1 else np.nan
            p = p_list[0][ax] if p_list else None
            t = t_list[0][ax] if t_list else None
            row[f'{ax}_default'] = (t or p or {}).get('default_displ', np.nan)
            row[f'{ax}_proj_displ'] = p['displ'] if p else np.nan
            row[f'{ax}_proj_rel'] = p['reliability'] if p else np.nan
            row[f'{ax}_ncc_peak'] = p['nccPeak'] if p else np.nan
            row[f'{ax}_thres_displ'] = t['displ'] if t else np.nan
            row[f'{ax}_replaced'] = bool(t['reliability'] == 0) if t else np.nan
            row[f'{ax}_placed'] = int(placed[i])
            ref = t['displ'] if t else (p['displ'] if p else np.nan)
            row[f'{ax}_place_resid'] = placed[i] - ref
        rows.append(row)
    n_unstitchable = sum(1 for v in positions.values() if not v[2])
    missing = [n for n, f in (('displcomp', comp), ('displproj', proj), ('displthres', thres)) if f is None]
    return pd.DataFrame(rows), merging['meta'], n_unstitchable, missing


def summarize_self_report(df):
    out = []
    for ax, _ in AXES:
        rep = df[f'{ax}_replaced'].astype(float)
        kept = df[df[f'{ax}_replaced'] != True]  # noqa: E712 — 列里可能是 NaN
        resid = kept[f'{ax}_place_resid'].abs()
        out.append({
            'axis': ax, 'n_pairs': len(df),
            'replaced_frac': rep.mean() if rep.notna().any() else np.nan,
            'rel_median': df[f'{ax}_proj_rel'].median(),
            'comp_std_median': df[f'{ax}_comp_std'].median(),
            'place_resid_p50': resid.median(), 'place_resid_max': resid.max(),
        })
    return pd.DataFrame(out)


# ──────────────────────────────────────────────────────────────────────────────
# B. 接缝残差
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=8)
def _read_dets(csv_path):
    if not os.path.isfile(csv_path):
        return None
    df = pd.read_csv(csv_path, usecols=BOX_COLS)
    if df.empty:
        return None
    df['gcx'] = (df['x1'] + df['x2']) / 2
    df['gcy'] = (df['y1'] + df['y2']) / 2
    return df


def _link_cells(df, ch_id, z_link):
    """z-link 一批 2D 检测 → (N, 5) [cx, cy, cz, z_min, z_max]（均为该 tile 局部原始坐标）。"""
    if df is None or df.empty:
        return np.empty((0, 5))
    mat = df[BOX_COLS].values.copy()
    mat[:, 6] = np.array([f"{v}_{ch_id}" for v in mat[:, 6]])
    _, vol_list = run_z_linker(mat, **z_link)
    out = []
    for c in vol_list:
        pzb = c['per_z_boxes']
        zs = np.array(sorted(pzb))
        boxes = np.array([pzb[z] for z in zs], dtype=float)
        # 质心取各层框中心的均值：比 z-linker 自带的「最高分那一层的框」稳，
        # 两个 tile 选中的最高分层不同也不会引入额外噪声
        out.append([(boxes[:, 0] + boxes[:, 2]).mean() / 2, (boxes[:, 1] + boxes[:, 3]).mean() / 2,
                    zs.mean(), zs[0], zs[-1]])
    return np.array(out, dtype=float).reshape(-1, 5)


def _tile_cells_in_overlap(det_dir, tile, ch_id, origin, overlap, pad, z_link):
    """
    读 tile 的检测，只保留（换到全局后）落在重叠区 ± pad 内的，z-link 后返回全局坐标的
    (N, 5) [gx, gy, gz, gz_min, gz_max]。
    """
    df = _read_dets(os.path.join(det_dir, f"{tile}_{ch_id}_result.csv"))
    if df is None:
        return None
    x0, x1, y0, y1 = overlap
    gx = df['gcx'].to_numpy() + origin[0]
    gy = df['gcy'].to_numpy() + origin[1]
    sel = (gx >= x0 - pad) & (gx <= x1 + pad) & (gy >= y0 - pad) & (gy <= y1 + pad)
    cells = _link_cells(df[sel], ch_id, z_link)
    cells[:, 0] += origin[0]
    cells[:, 1] += origin[1]
    cells[:, 2:] += origin[2]
    return cells


def _hist_peak(diffs, win_xy, win_z, bin_xy):
    """差值直方图（3D）平滑后的峰值位置与峰值对比度（峰值 / 均值）。"""
    ex = np.arange(-win_xy - bin_xy / 2, win_xy + bin_xy, bin_xy)
    ez = np.arange(-win_z - 0.5, win_z + 1.0, 1.0)
    h, _ = np.histogramdd(diffs, bins=(ex, ex, ez))
    h = uniform_filter(h, size=(3, 3, 3), mode='constant')
    i, j, k = np.unravel_index(np.argmax(h), h.shape)
    centers = lambda e: (e[:-1] + e[1:]) / 2  # noqa: E731
    contrast = float(h.max() / h.mean()) if h.mean() > 0 else np.nan
    return np.array([centers(ex)[i], centers(ex)[j], centers(ez)[k]]), contrast


def _mutual_nn(A, B, shift, r_xy, r_z):
    """A 平移 shift 后与 B 做互为最近邻配对（z 按 r_xy/r_z 缩放成各向同性）。"""
    scale = np.array([1.0, 1.0, r_xy / r_z])
    As, Bs = (A + shift) * scale, B * scale
    d_ab, j_ab = cKDTree(Bs).query(As, distance_upper_bound=r_xy)
    d_ba, j_ba = cKDTree(As).query(Bs, distance_upper_bound=r_xy)
    ia = np.where(np.isfinite(d_ab))[0]
    ia = ia[j_ba[j_ab[ia]] == ia]
    return ia, j_ab[ia]


def estimate_seam_error(A, B, p):
    """A、B：(N, 3) 全局质心。返回残差 dict（B − A 的中位数，理想值 0）。"""
    res = {'n_a': len(A), 'n_b': len(B), 'n_match': 0, 'peak_contrast': np.nan,
           'coarse_dx': np.nan, 'coarse_dy': np.nan, 'coarse_dz': np.nan,
           'dx': np.nan, 'dy': np.nan, 'dz': np.nan,
           'mad_x': np.nan, 'mad_y': np.nan, 'mad_z': np.nan}
    if len(A) < p['min_matches'] or len(B) < p['min_matches']:
        return res
    scale = np.array([1.0, 1.0, p['win_xy'] / p['win_z']])
    groups = cKDTree(B * scale).query_ball_point(A * scale, r=p['win_xy'], p=np.inf)
    ia = np.repeat(np.arange(len(A)), [len(g) for g in groups])
    if ia.size == 0:
        return res
    jb = np.concatenate([np.asarray(g, dtype=int) for g in groups])
    coarse, contrast = _hist_peak(B[jb] - A[ia], p['win_xy'], p['win_z'], p['bin_xy'])
    res.update(peak_contrast=contrast, coarse_dx=coarse[0], coarse_dy=coarse[1], coarse_dz=coarse[2])

    shift = coarse
    ia = jb = np.empty(0, dtype=int)
    for _ in range(3):
        ia, jb = _mutual_nn(A, B, shift, p['r_xy'], p['r_z'])
        if len(ia) < p['min_matches']:
            break
        new = np.median(B[jb] - A[ia], axis=0)
        if np.allclose(new, shift):
            break
        shift = new
    if len(ia) < p['min_matches']:
        res['n_match'] = int(len(ia))
        return res
    d = B[jb] - A[ia]
    med = np.median(d, axis=0)
    mad = np.median(np.abs(d - med), axis=0)
    res.update(n_match=int(len(ia)), dx=med[0], dy=med[1], dz=med[2],
               mad_x=mad[0], mad_y=mad[1], mad_z=mad[2])
    return res


def run_seam_job(job):
    """一个 (XML, 通道, 接缝) 任务。参数全可 pickle，供进程池直接调用。"""
    p = job['params']
    T, n_slices = p['tile_size'], job['n_slices']
    oa, ob = job['origin_a'], job['origin_b']
    ov = (max(oa[0], ob[0]) + p['edge_px'], min(oa[0], ob[0]) + T - p['edge_px'],
          max(oa[1], ob[1]) + p['edge_px'], min(oa[1], ob[1]) + T - p['edge_px'])
    # z：两边的原始 z 为 1..n_slices；碰到任一 tile 两端的细胞可能被截断，不用
    oz0 = max(oa[2], ob[2]) + 1
    oz1 = min(oa[2], ob[2]) + n_slices
    row = {k: job[k] for k in ('xml', 'channel', 'frame', 'direction', 'row', 'col',
                               'nb_row', 'nb_col', 'tile', 'nb_tile')}
    row.update(overlap_w=ov[1] - ov[0], overlap_h=ov[3] - ov[2])
    if ov[1] <= ov[0] or ov[3] <= ov[2]:
        row['status'] = 'no_overlap'
        return row

    pad = p['win_xy'] + 64
    cells = []
    for tile, origin in ((job['tile'], oa), (job['nb_tile'], ob)):
        c = _tile_cells_in_overlap(p['det_dir'], tile, job['channel'], origin, ov, pad, job['z_link'])
        if c is None:
            row['status'] = f'missing_csv:{tile}'
            return row
        keep = ((c[:, 0] >= ov[0]) & (c[:, 0] <= ov[1]) & (c[:, 1] >= ov[2]) & (c[:, 1] <= ov[3])
                & (c[:, 3] > oz0) & (c[:, 4] < oz1))
        cells.append(c[keep, :3])
    res = estimate_seam_error(cells[0], cells[1], p)
    row.update(res)
    row['match_frac'] = res['n_match'] / max(1, min(res['n_a'], res['n_b']))
    row['status'] = 'ok' if res['n_match'] >= p['min_matches'] else 'too_few_matches'
    xy_um, z_um = job['res']['x'], job['res']['z']
    row['err_xy_um'] = float(np.hypot(res['dx'], res['dy']) * xy_um)
    row['err_z_um'] = float(abs(res['dz']) * z_um)
    return row


def build_seam_jobs(spec, channel, frame, offsets, ts, z_link, params, warns):
    positions = stack_positions(ts)
    jobs, n_skip = [], 0
    for direction, a, b in neighbor_pairs(positions):
        (ta, pa, _), (tb, pb, _) = positions[a], positions[b]
        sa, sb = frame_shift(offsets, ta, channel, frame), frame_shift(offsets, tb, channel, frame)
        if sa is None or sb is None:
            n_skip += 1
            continue
        jobs.append({
            'xml': spec['name'], 'channel': channel, 'frame': frame, 'direction': direction,
            'row': a[0], 'col': a[1], 'nb_row': b[0], 'nb_col': b[1], 'tile': ta, 'nb_tile': tb,
            'origin_a': pa + np.array(sa), 'origin_b': pb + np.array(sb),
            'n_slices': ts['meta']['n_slices'], 'res': ts['meta']['res'],
            'z_link': z_link, 'params': params,
        })
    if n_skip:
        warns.append(f"[{spec['name']}] {channel}→{frame}: {n_skip} 条接缝缺 0_channel_alignment 偏移，跳过")
    return jobs


def summarize_seams(df):
    ok = df[df['status'] == 'ok']
    if ok.empty:
        return pd.DataFrame()
    g = ok.groupby(['xml', 'channel', 'frame'])
    return g.agg(
        n_ok=('status', 'size'),
        n_match_med=('n_match', 'median'),
        dx_med=('dx', 'median'), dy_med=('dy', 'median'), dz_med=('dz', 'median'),
        abs_dx_p50=('dx', lambda v: v.abs().median()), abs_dy_p50=('dy', lambda v: v.abs().median()),
        abs_dz_p50=('dz', lambda v: v.abs().median()),
        err_xy_um_p50=('err_xy_um', 'median'), err_xy_um_p90=('err_xy_um', lambda v: v.quantile(0.9)),
        err_z_um_p50=('err_z_um', 'median'), err_z_um_p90=('err_z_um', lambda v: v.quantile(0.9)),
    ).reset_index().merge(
        df.groupby(['xml', 'channel', 'frame']).size().rename('n_seams').reset_index(),
        on=['xml', 'channel', 'frame'])


# ──────────────────────────────────────────────────────────────────────────────
# C. 两份 XML 交叉验证
# ──────────────────────────────────────────────────────────────────────────────

def cross_check(spec_a, ts_a, spec_b, ts_b, offsets, flag_xy, flag_z):
    pos_a = {v[0]: v[1] for v in stack_positions(ts_a).values()}
    rc_a = {v[0]: rc for rc, v in stack_positions(ts_a).items()}
    pos_b = {v[0]: v[1] for v in stack_positions(ts_b).values()}
    ch_a, ch_b = spec_a['channel'], spec_b['channel']
    rows = []
    for tile in sorted(set(pos_a) & set(pos_b)):
        s = frame_shift(offsets, tile, ch_a, ch_b)   # s_a − s_b
        if s is None:
            continue
        d = pos_a[tile] - pos_b[tile]
        rows.append({'tile': tile, 'row': rc_a[tile][0], 'col': rc_a[tile][1],
                     **{f'D_{ax}': d[i] for i, (ax, _) in enumerate(AXES)},
                     **{f'E_{ax}': s[i] for i, (ax, _) in enumerate(AXES)}})
    df = pd.DataFrame(rows)
    if df.empty:
        return df, pd.DataFrame()
    summary = []
    for ax, _ in AXES:
        df[f'Dc_{ax}'] = df[f'D_{ax}'] - df[f'D_{ax}'].median()
        r = df[f'D_{ax}'] - df[f'E_{ax}']
        df[f'R_{ax}'] = r - r.median()
        summary.append({'axis': ax, 'n_tiles': len(df),
                        'const_C': float(r.median()),
                        'std_pos_diff': float(df[f'Dc_{ax}'].std()),
                        'std_residual': float(df[f'R_{ax}'].std()),
                        'max_abs_residual': float(df[f'R_{ax}'].abs().max())})
    df['flagged'] = ((np.hypot(df['R_x'], df['R_y']) > flag_xy) | (df['R_z'].abs() > flag_z))
    return df, pd.DataFrame(summary)


# ──────────────────────────────────────────────────────────────────────────────
# 作图
# ──────────────────────────────────────────────────────────────────────────────

def _plt():
    # matplotlib 只在真要画图时才导入：CSV 已经落盘了，画图环境坏掉不该连累数据
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _draw_seams(ax, df, value_col, n_row, n_col, vmax, title, fmt='{:.1f}', bad_mask=None, cmap='viridis'):
    from matplotlib.collections import LineCollection
    segs = [[(r.col, r.row), (r.nb_col, r.nb_row)] for r in df.itertuples()]
    vals = df[value_col].to_numpy(dtype=float)
    ax.scatter(*np.meshgrid(np.arange(n_col), np.arange(n_row)), s=30, c='0.8', zorder=1)
    lc = LineCollection(segs, cmap=cmap, linewidths=5, zorder=2)
    lc.set_array(np.ma.masked_invalid(vals))
    lc.set_clim(0, vmax if vmax and np.isfinite(vmax) and vmax > 0 else 1)
    ax.add_collection(lc)
    for (seg, v, bad) in zip(segs, vals, bad_mask if bad_mask is not None else [False] * len(vals)):
        (x0, y0), (x1, y1) = seg
        label = 'x' if bad else ('' if not np.isfinite(v) else fmt.format(v))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, label, fontsize=6, ha='center', va='center',
                color='red' if bad else 'black', zorder=3)
    ax.set_xlim(-0.5, n_col - 0.5)
    ax.set_ylim(n_row - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_xticks(range(n_col))
    ax.set_yticks(range(n_row))
    ax.set_xlabel('col')
    ax.set_ylabel('row')
    ax.set_title(title, fontsize=9)
    return lc


def plot_self_report(df, meta, name, out_path):
    plt = _plt()
    fig, axs = plt.subplots(2, 3, figsize=(3.2 * 3 + 1, 2 * 0.55 * meta['n_row'] + 2.5), squeeze=False)
    for i, (ax_name, _) in enumerate(AXES):
        bad = df[f'{ax_name}_replaced'].fillna(False).to_numpy(dtype=bool)
        lc = _draw_seams(axs[0, i], df, f'{ax_name}_proj_rel', meta['n_row'], meta['n_col'], 1.0,
                         f'{ax_name}: reliability (x = replaced)', fmt='{:.2f}', bad_mask=bad)
        fig.colorbar(lc, ax=axs[0, i], shrink=0.7)
        resid = df[f'{ax_name}_place_resid'].abs()
        lc = _draw_seams(axs[1, i], df.assign(_r=resid), '_r', meta['n_row'], meta['n_col'],
                         max(3.0, resid.max()) if resid.notna().any() else 3.0,
                         f'{ax_name}: |placed − pair displ| ({"slices" if ax_name == "z" else "px"})',
                         fmt='{:.0f}', bad_mask=bad, cmap='magma_r')
        fig.colorbar(lc, ax=axs[1, i], shrink=0.7)
    fig.suptitle(f'{name} — TeraStitcher self-report')
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_seams(df, meta, title, out_path):
    plt = _plt()
    fig, axs = plt.subplots(1, 3, figsize=(3.2 * 3 + 1, 0.55 * meta['n_row'] + 2.2), squeeze=False)
    bad = (df['status'] != 'ok').to_numpy()
    for i, (col, label, unit) in enumerate((('err_xy_um', '|xy error|', 'µm'),
                                            ('err_z_um', '|z error|', 'µm'),
                                            ('n_match', 'matched cells', ''))):
        v = df[col].to_numpy(dtype=float)
        vmax = np.nanpercentile(v, 95) if np.isfinite(v).any() else 1.0
        lc = _draw_seams(axs[0, i], df, col, meta['n_row'], meta['n_col'], vmax,
                         f'{label} {unit} (x = not ok)', fmt='{:.0f}', bad_mask=bad,
                         cmap='viridis' if col == 'n_match' else 'magma_r')
        fig.colorbar(lc, ax=axs[0, i], shrink=0.7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_cross(df, title, out_path):
    plt = _plt()
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (a, _) in zip(axs, AXES):
        e = df[f'E_{a}'] - df[f'E_{a}'].median()
        ax.scatter(e, df[f'Dc_{a}'], s=18, c=np.where(df['flagged'], 'crimson', 'steelblue'))
        lim = max(1.0, np.nanmax(np.abs(np.r_[e, df[f'Dc_{a}']])) * 1.1)
        ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.8)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_xlabel(f'channel shift difference s_a − s_b ({"slices" if a == "z" else "px"}, centered)')
        ax.set_ylabel('XML position difference (centered)')
        ax.set_title(f'{a}: std(resid)={df[f"R_{a}"].std():.2f}  std(pos diff)={df[f"Dc_{a}"].std():.2f}',
                     fontsize=9)
        ax.grid(alpha=0.25)
    fig.suptitle(title + '  (red = flagged tile)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sample', required=True, help='样本根目录（含各通道目录和 detection_results/）')
    ap.add_argument('--xml', action='append', required=True, metavar='NAME[:CHANNEL][=PATH]',
                    help='要比较的拼接，可重复给多次')
    ap.add_argument('--also', default='',
                    help='逗号分隔的通道：先用对齐偏移换到每份 XML 的通道坐标系再测接缝（B 部分）')
    ap.add_argument('--config', default=None, help='默认 <results_dir>/runtime_config.json')
    ap.add_argument('--det-dir', default=None,
                    help='原始（未对齐）检测 CSV 目录，默认 <results_dir>/1_tile_2d_raw')
    ap.add_argument('--out-dir', default=None,
                    help='默认 <results_dir>/5_analysis_report/stitch_compare')
    ap.add_argument('--tile-size', type=int, default=None, help='tile 边长 px，默认读 config 的 tILESIZE')
    ap.add_argument('--rel-thr', type=float, default=0.7, help='A 部分统计 comp_std 用的 reliability 阈值')
    ap.add_argument('--win-xy', type=float, default=60, help='B 部分粗搜索 XY 半径（px）')
    ap.add_argument('--win-z', type=float, default=8, help='B 部分粗搜索 Z 半径（切片）')
    ap.add_argument('--bin-xy', type=float, default=2, help='差值直方图 XY bin（px）')
    ap.add_argument('--match-xy', type=float, default=None,
                    help='配对半径 XY（px）；默认 soma 10、tf 6')
    ap.add_argument('--match-z', type=float, default=1.5, help='配对半径 Z（切片）')
    ap.add_argument('--edge-px', type=float, default=24,
                    help='离 tile 边缘这么近的细胞不用（被截断，质心有偏）')
    ap.add_argument('--min-matches', type=int, default=10, help='一条接缝至少配上多少个细胞才算数')
    ap.add_argument('--flag-xy', type=float, default=8, help='C 部分 XY 残差报警阈值（px）')
    ap.add_argument('--flag-z', type=float, default=2, help='C 部分 Z 残差报警阈值（切片）')
    ap.add_argument('--workers', type=int, default=1, help='B 部分并行进程数')
    ap.add_argument('--no-plots', action='store_true', help='只出 CSV 不画图')
    return ap.parse_args()


def _print_df(df, title):
    print(f"\n===== {title} =====")
    with pd.option_context('display.width', 200, 'display.max_rows', 200, 'display.max_columns', 50):
        print(df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


def main():
    args = parse_args()
    sample_dir = os.path.abspath(args.sample)
    sample_name = os.path.basename(sample_dir.rstrip('/\\'))
    results_dir = resolve_results_dir(sample_dir)
    routing, settings = load_channel_settings(results_dir, args.config)
    ch_types = {ch['id']: ch.get('type', 'soma') for ch in (routing or [])}
    config = load_config(args.config or os.path.join(results_dir, 'runtime_config.json')) \
        if (args.config or results_dir) else {}
    tile_size = args.tile_size or config.get('detection_params', {}).get('tILESIZE', 2048)

    out_dir = args.out_dir or (os.path.join(results_dir, '5_analysis_report', 'stitch_compare')
                               if results_dir else os.path.join(sample_dir, 'stitch_compare'))
    os.makedirs(out_dir, exist_ok=True)
    det_dir = args.det_dir or (os.path.join(results_dir, '1_tile_2d_raw') if results_dir else None)
    offsets = load_offsets(os.path.join(results_dir, '0_channel_alignment') if results_dir else None)

    warns = []
    specs, ts_by_name = [], {}
    for s in args.xml:
        spec = parse_xml_spec(s, sample_dir)
        ts = read_terastitcher(spec['merging'])
        if ts is None:
            warns.append(f"[{spec['name']}] 找不到 {spec['merging']}，这份拼接跳过（还没做完？）")
            continue
        if spec['channel'] and routing is not None and spec['channel'] not in ch_types:
            raise SystemExit(f"❌ --xml {s}: 通道 {spec['channel']} 不在 channels_routing {list(ch_types)} 里")
        specs.append(spec)
        ts_by_name[spec['name']] = ts
    for w in warns:
        print(f"⚠️  {w}")
    warns = []
    if not specs:
        raise SystemExit("❌ 没有可用的 xml_merging.xml。")

    also = [c.strip() for c in args.also.split(',') if c.strip()]
    for c in also:
        if routing is not None and c not in ch_types:
            raise SystemExit(f"❌ --also {c} 不在 channels_routing {list(ch_types)} 里")

    print(f"样本     : {sample_name}  ({sample_dir})")
    print(f"结果目录 : {results_dir or '（没有 detection_results，只做 A 部分）'}")
    for sp in specs:
        print(f"拼接     : {sp['name']}  通道={sp['channel'] or '-'}  {sp['merging']}")
    print(f"对齐偏移 : {len(offsets)} 个 tile")
    print(f"输出     : {out_dir}")

    # 所有 CSV 写完之后再统一画图：matplotlib 坏掉会直接崩进程，不能连累后面的数据
    plots = []

    # ── A ──
    for sp in specs:
        df, meta, n_unst, missing = self_report(sp, args.rel_thr)
        df.to_csv(os.path.join(out_dir, f"{sp['name']}_pairs.csv"), index=False)
        _print_df(summarize_self_report(df),
                  f"A. {sp['name']} TeraStitcher 自评（{meta['n_row']}×{meta['n_col']}，"
                  f"STITCHABLE=no 的 tile {n_unst} 个）")
        if missing:
            print(f"  （缺少 {', '.join(missing)}，对应列为空）")
        plots.append((plot_self_report, df, meta, sp['name'],
                      os.path.join(out_dir, f"{sp['name']}_self_report.png")))
    print("\n  replaced_frac   = 该轴被换成机械默认位移的相邻对比例（越低越好）")
    print("  rel_median      = displproj 里的 reliability 中位数（跨通道不宜直接比绝对值）")
    print("  comp_std_median = 同一对 tile 各子块位移的标准差中位数（越小越稳）")
    print("  place_resid_*   = 最终摆放与该对自身位移之差（未被替换的对；大 = 环路不自洽）")

    # ── B ──
    has_dets = det_dir and os.path.isdir(det_dir) and any(f.endswith('_result.csv') for f in os.listdir(det_dir))
    if not has_dets:
        print(f"\nB. 跳过接缝残差：{det_dir or '（无 results_dir）'} 里还没有检测 CSV。")
    elif settings is None:
        print("\nB. 跳过接缝残差：找不到 config（用 --config 指定）。")
    else:
        params = {'det_dir': det_dir, 'tile_size': tile_size, 'edge_px': args.edge_px,
                  'win_xy': args.win_xy, 'win_z': args.win_z, 'bin_xy': args.bin_xy,
                  'r_z': args.match_z, 'min_matches': args.min_matches}
        jobs = []
        for sp in specs:
            if not sp['channel']:
                continue
            for ch in dict.fromkeys([sp['channel']] + also):
                ctype = ch_types.get(ch, 'soma')
                p = dict(params, r_xy=args.match_xy or (10.0 if ctype == 'soma' else 6.0))
                jobs += build_seam_jobs(sp, ch, sp['channel'], offsets, ts_by_name[sp['name']],
                                        settings['z_link']['soma' if ctype == 'soma' else 'tf'], p, warns)
        rows = []
        if args.workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
                futs = [pool.submit(run_seam_job, j) for j in jobs]
                for fut in tqdm(as_completed(futs), total=len(futs), desc="Seams"):
                    rows.append(fut.result())
        else:
            for j in tqdm(jobs, desc="Seams"):
                rows.append(run_seam_job(j))
        for w in warns:
            print(f"⚠️  {w}")
        if rows:
            seams = pd.DataFrame(rows).sort_values(['xml', 'channel', 'row', 'col', 'direction'])
            seams.to_csv(os.path.join(out_dir, 'seams.csv'), index=False)
            bad = seams[seams['status'] != 'ok']
            if not bad.empty:
                print(f"⚠️  {len(bad)} 条接缝没有结果：{bad['status'].value_counts().to_dict()}")
            summ = summarize_seams(seams)
            if not summ.empty:
                _print_df(summ, "B. 接缝残差（B − A 的中位数；理想值 0）")
                print("\n  channel→frame 不同的行 = 该通道经对齐偏移换到拼接通道坐标系后的误差，"
                      "即细胞最终落在这份拼接图上的误差")
            for (xml, ch, frame), g in seams.groupby(['xml', 'channel', 'frame']):
                tag = ch if ch == frame else f"{ch}_in_{frame}"
                plots.append((plot_seams, g, ts_by_name[xml]['meta'],
                              f"{sample_name} — {xml} — {tag} seam residual",
                              os.path.join(out_dir, f"{xml}_{tag}_seams.png")))
        else:
            print("\nB. 没有可跑的接缝。")

    # ── C ──
    with_ch = [sp for sp in specs if sp['channel']]
    if len(with_ch) < 2:
        print("\nC. 跳过交叉验证：需要至少两份带通道的 XML。")
    elif not offsets:
        print("\nC. 跳过交叉验证：还没有 0_channel_alignment 偏移。")
    else:
        for sa, sb in itertools.combinations(with_ch, 2):
            df, summ = cross_check(sa, ts_by_name[sa['name']], sb, ts_by_name[sb['name']],
                                   offsets, args.flag_xy, args.flag_z)
            if len(df) < 3:
                print(f"\nC. {sa['name']} vs {sb['name']}: 同时具备两份位置和偏移的 tile 只有 {len(df)} 个"
                      f"（Stage 2.5 还没跑完？），至少要 3 个才有意义，跳过。")
                continue
            tag = f"{sa['name']}_vs_{sb['name']}"
            df.to_csv(os.path.join(out_dir, f"cross_{tag}.csv"), index=False)
            _print_df(summ, f"C. {sa['name']}({sa['channel']}) vs {sb['name']}({sb['channel']})")
            print("  std_residual 明显小于 std_pos_diff → 两份拼接的差异能被逐 tile 通道偏移解释，互相印证")
            flagged = df[df['flagged']]
            if not flagged.empty:
                print(f"⚠️  {len(flagged)} 个 tile 残差超阈值：{flagged['tile'].tolist()}")
            plots.append((plot_cross, df, f"{sample_name} — {sa['name']} vs {sb['name']}",
                          os.path.join(out_dir, f"cross_{tag}.png")))

    print(f"\nCSV 已写到 {out_dir}")
    if args.no_plots:
        print("图：跳过（--no-plots）")
    else:
        for fn, *fargs in plots:
            fn(*fargs)
        print(f"图：{len(plots)} 张，在 {out_dir}")


if __name__ == '__main__':
    mp.freeze_support()
    main()
