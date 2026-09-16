# -*- coding: utf-8 -*-
"""
验证 pre_align 通道偏移的可信度：在**没有参与求解**的 subvolume 里，把偏移从
最优解沿 x / y / z 各方向挪开，看通道间的细胞重合度是否显著下降。

如果最优解是真的，曲线应该在 delta=0 处有一个尖峰；如果曲线是平的、或者峰值
不在 0，说明该 tile 的偏移只是噪声里的一个随机点，不可信。

留出区域（held-out）怎么定义
---------------------------
Stage 2.5 求偏移时用的是「z 中心窗口内的整幅 XY」（见 run_inference.py 阶段 2.5
和 point_cloud_aligner.build_cell_boxes），所以 XY 上没有任何未被使用的区域，
严格未被使用的只有 z 窗口之外的切片。本脚本因此：
  1. 用与 Stage 2.5 完全相同的方式重算该 tile 的 z_center 和对齐窗口
     [z_center ± sample_z_center_count/2]；
  2. 在该窗口**之外**（再留 --z-guard 层安全边距，因为 build_cell_boxes 按
     「z 范围有重叠」收细胞，窗口边缘的细胞体本身会向外延伸几层）随机取一段
     长 --z-size 的 z；
  3. 在 XY 上再随机截一块 --xy-size 见方的子区域，使评估具有局部性。
z 深度不足以留出窗口外区域时会退化成「窗口内取样」并在输出里标 held_out=False。

指标
----
  voxel_iou   : 与 Stage 2.5 的 intra-soma / intra-TF 目标函数同源
                （point_cloud_aligner._voxel_iou），所有通道对都记。
  containment : 与 Stage 2.5 求 TF 偏移用的目标函数同源
                （point_cloud_aligner._containment_score），只对 TF 通道记，
                = 区域内落进某个参考 soma 里的 TF 核比例。
与求解时不同的是：候选偏移直接作用在细胞坐标上再体素化（像素级精确），而不是
在 bin 空间里平移网格，所以 --xy-step 不必是 voxel_bin_size_px 的整数倍。

扫描方式：单轴扫描。扫 dx 时 dy/dz 固定在最优解，依此类推。

用法
----
  python scripts/validate_align_shifts.py --sample Y:/Fengyi/TSC_brain/sample18
  python scripts/validate_align_shifts.py --sample Y:/Fengyi/TSC_brain/sample18 \
      --n-tiles 8 --regions-per-tile 2 --xy-size 1024 --z-size 30 --workers 4
  python scripts/validate_align_shifts.py --sample ... --tiles 291400_342500,302700_342500

注意：conda 环境 brain_detector / gt_sam 里的 matplotlib 画图时会直接崩掉进程
（Agg 后端加载 DLL 失败，Windows 异常 0xc06d007f），antsreg 环境正常。CSV 在画图
之前就已经写盘，崩了也不丢数据；这类环境加 --no-plots 跑，或换环境画图。

输出（默认写到 <results_dir>/5_analysis_report/align_validation/）
  <sample>_curves.csv    每个 (tile, 区域, 通道, 轴, delta) 一行的原始曲线数据
  <sample>_summary.csv   每个 (tile, 通道, 指标, 轴) 一行的峰值/半高宽/跌幅汇总
  <sample>_<channel>_<metric>.png  曲线图：细线=单个区域，粗线=所有区域均值
"""

import argparse
import json
import os
import sys
import zlib

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm

from src.config.loader import load_config
from src.core.z_linker import run_z_linker
from src.core.point_cloud_aligner import (
    ALIGN_SETTINGS_FILE, resolve_align_settings,
    _voxelize_to_grid, _voxel_iou,
    _cell_arrays, _prepare_soma_containment_index, _containment_score,
)

# soma 细胞体半径量级（px）：给 containment 的候选 soma 选择留的 XY 余量，
# 保证区域边缘外、但 bbox 仍能罩住区域内 TF 核的 soma 也被纳入。
_SOMA_PAD_PX = 64


# ──────────────────────────────────────────────────────────────────────────────
# 数据准备
# ──────────────────────────────────────────────────────────────────────────────

def resolve_results_dir(sample):
    """--sample 既可以是样本根目录，也可以直接是 detection_results 目录。"""
    for cand in (sample, os.path.join(sample, 'detection_results')):
        if os.path.isdir(os.path.join(cand, '1_tile_2d_raw')):
            return os.path.abspath(cand)
    raise SystemExit(f"❌ 在 {sample} 下找不到 1_tile_2d_raw/，请确认 --sample 指向样本目录或 detection_results 目录。")


def load_settings(results_dir, config_path):
    """
    取回 Stage 2.5 实际用过的对齐设置。

    优先读 0_channel_alignment/_align_settings.json（新版本流程会落盘，最可靠）；
    旧结果没有这个文件，就用 config 重新解析一份（要求 config 里的对齐参数没改过）。
    """
    align_dir = os.path.join(results_dir, '0_channel_alignment')
    cfg_path = config_path or os.path.join(results_dir, 'runtime_config.json')
    if not os.path.isfile(cfg_path):
        raise SystemExit(f"❌ 找不到配置文件 {cfg_path}，请用 --config 指定当初跑这个样本用的 config。")
    config = load_config(cfg_path)
    routing = [ch for ch in config.get('channels_routing', []) if ch.get('active', True)]
    if not routing:
        raise SystemExit(f"❌ {cfg_path} 里没有激活的 channels_routing。")

    saved_path = os.path.join(align_dir, ALIGN_SETTINGS_FILE)
    if os.path.isfile(saved_path):
        with open(saved_path, encoding='utf-8') as f:
            settings = json.load(f)
        src = saved_path
    else:
        settings = resolve_align_settings(config, routing)
        src = f"{cfg_path}（{ALIGN_SETTINGS_FILE} 不存在，按 config 重新解析；若对齐后改过参数，结果不可比）"
    return config, routing, settings, align_dir, src


def list_tiles(align_dir):
    """有 offsets JSON 的 tile 才算对齐过。"""
    suffix = '_offsets.json'
    return sorted(f[:-len(suffix)] for f in os.listdir(align_dir) if f.endswith(suffix))


def build_tile_vol_lists(raw_dir, tile_name, routing, settings):
    """
    与 Stage 2.5 同样的方式：每个通道读原始检测 CSV → 轻量 z-link → vol_list，
    并按同样的口径估 z_center（所有通道全部 3D 细胞 cz 的中位数）。
    """
    per_ch, z_all = {}, []
    for ch in routing:
        cid, ctype = ch['id'], ch.get('type', 'soma')
        csv_path = os.path.join(raw_dir, f"{tile_name}_{cid}_result.csv")
        if not os.path.isfile(csv_path):
            per_ch[cid] = []
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            per_ch[cid] = []
            continue
        mat = df[["x1", "y1", "x2", "y2", "score", "mean", "class", "z"]].values
        mat[:, 6] = np.array([f"{v}_{cid}" for v in mat[:, 6]])
        _, vol_list = run_z_linker(mat, **settings['z_link']['soma' if ctype == 'soma' else 'tf'])
        per_ch[cid] = vol_list
        z_all.extend([c.get('cz', 0) for c in vol_list])
    z_center = float(np.median(z_all)) if z_all else None
    return per_ch, z_center


# ──────────────────────────────────────────────────────────────────────────────
# 区域取样
# ──────────────────────────────────────────────────────────────────────────────

def _cells_bounds(per_ch):
    """所有通道细胞的 XY / Z 包围盒。"""
    xs1, ys1, xs2, ys2, zs1, zs2 = [], [], [], [], [], []
    for cells in per_ch.values():
        for c in cells:
            xs1.append(c['x1_3d']); xs2.append(c['x2_3d'])
            ys1.append(c['y1_3d']); ys2.append(c['y2_3d'])
            zs1.append(c['z_min']); zs2.append(c['z_max'])
    if not xs1:
        return None
    return (min(xs1), max(xs2), min(ys1), max(ys2), int(min(zs1)), int(max(zs2)))


def sample_region(rng, bounds, align_z_lo, align_z_hi, z_guard, xy_size, z_size):
    """
    随机取一个 (x0, x1, y0, y1, z0, z1) 子体积。

    z 优先取在对齐窗口 [align_z_lo, align_z_hi] ± z_guard 之外（真正的 held-out）；
    取不到就退回全 z 范围，并用返回的 held_out=False 标记这次取样不算独立验证。
    """
    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = bounds

    z_starts = [s for s in range(z_lo, z_hi - z_size + 2)
                if (s + z_size - 1) < (align_z_lo - z_guard) or s > (align_z_hi + z_guard)]
    held_out = bool(z_starts)
    if not held_out:
        z_starts = list(range(z_lo, max(z_lo + 1, z_hi - z_size + 2)))
    z0 = int(rng.choice(z_starts))
    z1 = min(z0 + z_size, z_hi + 1)

    def _axis(lo, hi, size):
        if hi - lo <= size:
            return lo, hi
        start = float(rng.uniform(lo, hi - size))
        return start, start + size

    x0, x1 = _axis(x_lo, x_hi, xy_size)
    y0, y1 = _axis(y_lo, y_hi, xy_size)
    return (x0, x1, y0, y1, z0, z1), held_out


def _cells_overlapping(cells, region, pad_xy=0.0, pad_z=0):
    """bbox 与区域（外扩 pad）有交集的细胞。"""
    x0, x1, y0, y1, z0, z1 = region
    x0 -= pad_xy; x1 += pad_xy; y0 -= pad_xy; y1 += pad_xy
    z0 -= pad_z;  z1 += pad_z
    return [c for c in cells
            if c['x2_3d'] >= x0 and c['x1_3d'] <= x1
            and c['y2_3d'] >= y0 and c['y1_3d'] <= y1
            and c['z_max'] >= z0 and c['z_min'] <= z1]


def _cells_centroid_in(cells, region):
    """质心落在区域内的细胞（用作 containment 的分母，使其不随偏移变化）。"""
    x0, x1, y0, y1, z0, z1 = region
    return [c for c in cells
            if x0 <= c['cx'] < x1 and y0 <= c['cy'] < y1 and z0 <= c['cz'] < z1]


# ──────────────────────────────────────────────────────────────────────────────
# 指标
# ──────────────────────────────────────────────────────────────────────────────

def _shift_cells(cells, dx, dy, dz):
    """把一批细胞整体平移 (dx, dy, dz)，返回新 dict（不改原对象）。"""
    out = []
    for c in cells:
        d = dict(c)
        d['cx'] = c['cx'] + dx; d['cy'] = c['cy'] + dy; d['cz'] = c['cz'] + dz
        d['x1_3d'] = c['x1_3d'] + dx; d['x2_3d'] = c['x2_3d'] + dx
        d['y1_3d'] = c['y1_3d'] + dy; d['y2_3d'] = c['y2_3d'] + dy
        d['z_min'] = c['z_min'] + dz; d['z_max'] = c['z_max'] + dz
        d['per_z_boxes'] = {int(z) + dz: [b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy]
                            for z, b in c.get('per_z_boxes', {}).items()}
        out.append(d)
    return out


def _grid(cells, region, bin_size, xy_res_um, z_res_um):
    """把细胞栅格化到区域体素网格；区域外的部分由 _voxelize_to_grid 自动裁掉。"""
    x0, x1, y0, y1, z0, z1 = region
    return _voxelize_to_grid(cells, z0, z1, x0, x1, y0, y1, bin_size, xy_res_um, z_res_um)


def region_voxel_iou(grid_ref, tgt_cells, region, dx, dy, dz, bin_size, xy_res_um, z_res_um):
    grid_tgt = _grid(_shift_cells(tgt_cells, dx, dy, dz), region, bin_size, xy_res_um, z_res_um)
    if grid_ref.sum() == 0 or grid_tgt.sum() == 0:
        return 0.0
    return _voxel_iou(grid_ref, grid_tgt, 0, 0, 0)


# ──────────────────────────────────────────────────────────────────────────────
# 单 tile 处理
# ──────────────────────────────────────────────────────────────────────────────

def _sweep_deltas(rng_ranges):
    """[(axis, delta), ...]；每个轴都含 delta=0，扫某轴时另两轴固定在最优解。"""
    (rx, sx), (ry, sy), (rz, sz) = rng_ranges
    out = []
    for axis, r, s in (('x', rx, sx), ('y', ry, sy), ('z', rz, sz)):
        vals = sorted(set(list(range(-r, r + 1, s)) + [0]))
        out.extend((axis, v) for v in vals)
    return out


def process_tile(tile_name, params):
    """返回 (rows, warnings)。设计成可被 ProcessPoolExecutor 直接调用（参数全可 pickle）。"""
    rows, warns = [], []
    settings = params['settings']
    routing = params['routing']
    ref_id = settings['reference_channel']

    offsets_path = os.path.join(params['align_dir'], f"{tile_name}_offsets.json")
    if not os.path.isfile(offsets_path):
        return rows, [f"{tile_name}: 缺少 {os.path.basename(offsets_path)}，跳过"]
    with open(offsets_path, encoding='utf-8') as f:
        offsets = json.load(f)

    per_ch, z_center = build_tile_vol_lists(params['raw_dir'], tile_name, routing, settings)
    if z_center is None:
        return rows, [f"{tile_name}: 所有通道都没有 3D 细胞，跳过"]
    bounds = _cells_bounds(per_ch)
    if bounds is None:
        return rows, [f"{tile_name}: 没有可用细胞，跳过"]

    z_half = settings['sample_z_center_count'] // 2
    align_z_lo, align_z_hi = z_center - z_half, z_center + z_half

    ref_cells_all = per_ch.get(ref_id, [])
    if not ref_cells_all:
        return rows, [f"{tile_name}: 参考通道 {ref_id} 没有细胞，跳过"]
    ref_shift = offsets.get(ref_id, {})
    ref_dxyz = (ref_shift.get('dx', 0), ref_shift.get('dy', 0), ref_shift.get('dz', 0))

    targets = [ch for ch in routing if ch['id'] != ref_id]
    deltas = _sweep_deltas(params['sweep'])
    pad_xy = float(max(params['sweep'][0][0], params['sweep'][1][0]))
    pad_z = int(params['sweep'][2][0])

    rng = np.random.default_rng(params['seed'] + zlib.crc32(tile_name.encode()))

    for region_id in range(params['regions_per_tile']):
        region, held_out = None, False
        for _ in range(params['max_attempts']):
            cand, cand_held = sample_region(rng, bounds, align_z_lo, align_z_hi,
                                            params['z_guard'], params['xy_size'], params['z_size'])
            n_ref = len(_cells_overlapping(ref_cells_all, cand))
            if n_ref < params['min_cells']:
                continue
            if all(len(_cells_overlapping(per_ch.get(ch['id'], []), cand)) >= params['min_cells']
                   for ch in targets):
                region, held_out = cand, cand_held
                break
        if region is None:
            warns.append(f"{tile_name}: 区域 #{region_id} 试了 {params['max_attempts']} 次都凑不够 "
                         f"{params['min_cells']} 个细胞，跳过（可以调大 --xy-size / --z-size 或调小 --min-cells）")
            continue
        if not held_out:
            warns.append(f"{tile_name}: 区域 #{region_id} 的 z 深度不足以避开对齐窗口 "
                         f"[{align_z_lo:.0f}, {align_z_hi:.0f}]±{params['z_guard']}，"
                         f"该区域参与过求解，held_out=False")

        ref_region = _cells_overlapping(ref_cells_all, region)
        grid_ref = _grid(ref_region, region, settings['voxel_bin_size_px'],
                         settings['xy_resolution_um'], settings['z_resolution_um'])
        # containment 的参考 soma 要外扩，否则区域边缘外的 soma 罩住区域内的核会被漏掉
        soma_idx = None
        ref_soma_pad = _cells_overlapping(ref_cells_all, region, pad_xy + _SOMA_PAD_PX, pad_z + 5)
        if ref_soma_pad:
            soma_idx = _prepare_soma_containment_index(ref_soma_pad)

        for ch in targets:
            cid, ctype = ch['id'], ch.get('type', 'soma')
            cells_all = per_ch.get(cid, [])
            if not cells_all:
                warns.append(f"{tile_name}: 通道 {cid} 没有细胞，跳过")
                continue
            off = offsets.get(cid)
            if off is None:
                warns.append(f"{tile_name}: offsets JSON 里没有通道 {cid}，跳过")
                continue
            # 通道相对参考通道的最优解（参考通道本身通常就是 (0,0,0)）
            base = (off.get('dx', 0) - ref_dxyz[0],
                    off.get('dy', 0) - ref_dxyz[1],
                    off.get('dz', 0) - ref_dxyz[2])

            tgt_region = _cells_overlapping(cells_all, region, pad_xy, pad_z)
            tf_region = _cells_centroid_in(cells_all, region) if ctype == 'tf' else []
            tf_arrays = _cell_arrays(tf_region) if tf_region else None

            for axis, delta in deltas:
                dx = base[0] + (delta if axis == 'x' else 0)
                dy = base[1] + (delta if axis == 'y' else 0)
                dz = base[2] + (delta if axis == 'z' else 0)

                iou = region_voxel_iou(grid_ref, tgt_region, region, dx, dy, dz,
                                       settings['voxel_bin_size_px'],
                                       settings['xy_resolution_um'], settings['z_resolution_um'])

                contain = np.nan
                if ctype == 'tf' and tf_arrays is not None and soma_idx is not None:
                    count, _margin = _containment_score(
                        soma_idx, tf_arrays, dx, dy, dz,
                        settings['max_center_dist_ratio'], 0, settings['containment_z_pad'])
                    contain = count / len(tf_region)

                rows.append({
                    'tile': tile_name, 'region_id': region_id, 'held_out': held_out,
                    'x0': round(region[0], 1), 'x1': round(region[1], 1),
                    'y0': round(region[2], 1), 'y1': round(region[3], 1),
                    'z0': region[4], 'z1': region[5],
                    'align_z_lo': round(align_z_lo, 1), 'align_z_hi': round(align_z_hi, 1),
                    'ref_channel': ref_id, 'channel': cid, 'ch_type': ctype,
                    'axis': axis, 'delta': delta,
                    'dx': dx, 'dy': dy, 'dz': dz,
                    'base_dx': base[0], 'base_dy': base[1], 'base_dz': base[2],
                    'voxel_iou': iou, 'containment': contain,
                    'n_ref_cells': len(ref_region), 'n_tgt_cells': len(tgt_region),
                    'n_tf_in_region': len(tf_region),
                })
    return rows, warns


# ──────────────────────────────────────────────────────────────────────────────
# 汇总与作图
# ──────────────────────────────────────────────────────────────────────────────

def summarize(df, metric):
    """
    每个 (tile, region, channel, axis) 一行：峰值是否在 delta=0、偏移到两端时跌多少、
    以及跌到峰值一半所需的偏移量（half_width，越小说明曲线越尖、解越可信）。
    """
    out = []
    sub = df.dropna(subset=[metric])
    for (tile, region_id, ch, axis), g in sub.groupby(['tile', 'region_id', 'channel', 'axis']):
        g = g.sort_values('delta')
        d = g['delta'].to_numpy()
        v = g[metric].to_numpy(dtype=float)
        i0 = int(np.argmin(np.abs(d)))
        v0, vmax = v[i0], v.max()
        edge = np.mean([v[0], v[-1]])

        half = 0.5 * v0
        left = d[:i0 + 1][v[:i0 + 1] < half]
        right = d[i0:][v[i0:] < half]
        hw_neg = abs(left.max()) if left.size else np.nan   # 向负方向跌破半高的最近 delta
        hw_pos = right.min() if right.size else np.nan

        out.append({
            'tile': tile, 'region_id': region_id, 'channel': ch, 'axis': axis,
            'metric': metric,
            'value_at_optimum': v0, 'value_max': vmax,
            'delta_at_max': int(d[int(np.argmax(v))]),
            'peak_at_optimum': bool(np.isclose(v0, vmax)),
            'edge_mean': edge,
            'drop_ratio_at_edge': (v0 - edge) / v0 if v0 > 0 else np.nan,
            'half_width_neg': hw_neg, 'half_width_pos': hw_pos,
            'n_ref_cells': int(g['n_ref_cells'].iloc[0]),
            'n_tgt_cells': int(g['n_tgt_cells'].iloc[0]),
            'held_out': bool(g['held_out'].iloc[0]),
        })
    return pd.DataFrame(out)


def plot_channel(df, channel, metric, out_path, sample_name):
    # matplotlib 只在真要画图时才导入：CSV 已经落盘了，画图环境坏掉不该连累数据。
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sub = df[(df['channel'] == channel)].dropna(subset=[metric])
    if sub.empty:
        return False
    axes_order = [a for a in ('x', 'y', 'z') if a in set(sub['axis'])]
    fig, axs = plt.subplots(1, len(axes_order), figsize=(4.6 * len(axes_order), 4.0), squeeze=False)
    for ax, axis in zip(axs[0], axes_order):
        g_axis = sub[sub['axis'] == axis]
        for (tile, region_id), g in g_axis.groupby(['tile', 'region_id']):
            g = g.sort_values('delta')
            ax.plot(g['delta'], g[metric], color='0.7', lw=0.8, alpha=0.8, zorder=1)
        mean = g_axis.groupby('delta')[metric].mean().sort_index()
        ax.plot(mean.index, mean.values, color='crimson', lw=2.2, marker='o', ms=3, zorder=3,
                label=f"mean (n={g_axis[['tile', 'region_id']].drop_duplicates().shape[0]})")
        ax.axvline(0, color='steelblue', ls='--', lw=1.2, zorder=2)
        ax.set_xlabel(f"Δ{axis} from optimal shift ({'slices' if axis == 'z' else 'px'})")
        ax.set_ylabel(metric)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"{sample_name} — {channel} vs {df['ref_channel'].iloc[0]} — {metric} "
                 f"(held-out subvolumes)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sample', required=True,
                    help='样本目录（含 detection_results/）或 detection_results 目录本身')
    ap.add_argument('--config', default=None,
                    help='对齐时用的 config；默认读 <results_dir>/runtime_config.json')
    ap.add_argument('--out-dir', default=None,
                    help='输出目录；默认 <results_dir>/5_analysis_report/align_validation')
    ap.add_argument('--tiles', default=None, help='逗号分隔的 tile 名；不给则从已对齐 tile 里随机抽')
    ap.add_argument('--n-tiles', type=int, default=5, help='随机抽多少个 tile（--tiles 未给时生效）')
    ap.add_argument('--regions-per-tile', type=int, default=2, help='每个 tile 取几个随机子区域')
    ap.add_argument('--xy-size', type=float, default=1024, help='子区域 XY 边长（px）')
    ap.add_argument('--z-size', type=int, default=30, help='子区域 z 厚度（切片数）')
    ap.add_argument('--z-guard', type=int, default=None,
                    help='子区域与对齐窗口之间的安全边距（切片）；默认 '
                         'max_cell_z_span + z_search_range + z_fine_search')
    ap.add_argument('--xy-range', type=int, default=20, help='XY 扫描半径（px）')
    ap.add_argument('--xy-step', type=int, default=2, help='XY 扫描步长（px）')
    ap.add_argument('--z-range', type=int, default=6, help='Z 扫描半径（切片）')
    ap.add_argument('--z-step', type=int, default=1, help='Z 扫描步长（切片）')
    ap.add_argument('--min-cells', type=int, default=15, help='子区域内每个通道最少细胞数')
    ap.add_argument('--max-attempts', type=int, default=50, help='区域取样最多重试次数')
    ap.add_argument('--seed', type=int, default=0, help='随机种子（同种子结果可复现）')
    ap.add_argument('--workers', type=int, default=1, help='并行处理 tile 的进程数')
    ap.add_argument('--no-plots', action='store_true',
                    help='只出 CSV 不画图（matplotlib 装坏的环境用）')
    return ap.parse_args()


def main():
    args = parse_args()
    results_dir = resolve_results_dir(args.sample)
    sample_name = os.path.basename(os.path.dirname(results_dir)) or os.path.basename(results_dir)
    config, routing, settings, align_dir, settings_src = load_settings(results_dir, args.config)

    all_tiles = list_tiles(align_dir)
    if not all_tiles:
        raise SystemExit(f"❌ {align_dir} 里没有 *_offsets.json，该样本还没跑过 Stage 2.5。")
    if args.tiles:
        tiles = [t.strip() for t in args.tiles.split(',') if t.strip()]
        missing = [t for t in tiles if t not in all_tiles]
        if missing:
            raise SystemExit(f"❌ 这些 tile 没有对齐结果：{missing}")
    else:
        rng = np.random.default_rng(args.seed)
        n = min(args.n_tiles, len(all_tiles))
        tiles = sorted(np.array(all_tiles)[rng.choice(len(all_tiles), n, replace=False)].tolist())

    z_guard = args.z_guard
    if z_guard is None:
        z_guard = (max(settings['z_link']['soma']['max_cell_z_span'],
                       settings['z_link']['tf']['max_cell_z_span'])
                   + settings['z_search_range_slices'] + settings['z_fine_search_slices'])

    out_dir = args.out_dir or os.path.join(results_dir, '5_analysis_report', 'align_validation')
    os.makedirs(out_dir, exist_ok=True)

    print(f"样本      : {sample_name}  ({results_dir})")
    print(f"对齐设置  : {settings_src}")
    print(f"参考通道  : {settings['reference_channel']}，TF 模式 {settings['tf_align_mode']}")
    print(f"待验证 tile: {len(tiles)} / {len(all_tiles)}  {tiles if len(tiles) <= 8 else tiles[:8] + ['...']}")
    print(f"子区域    : {args.xy_size:.0f}×{args.xy_size:.0f} px × {args.z_size} slices，"
          f"每 tile {args.regions_per_tile} 个，避开对齐窗口 ±{z_guard} 层")
    print(f"扫描      : Δxy ±{args.xy_range} step {args.xy_step}，Δz ±{args.z_range} step {args.z_step}（单轴）")
    print(f"输出      : {out_dir}\n")

    params = {
        'settings': settings, 'routing': routing, 'align_dir': align_dir,
        'raw_dir': os.path.join(results_dir, '1_tile_2d_raw'),
        'sweep': ((args.xy_range, args.xy_step), (args.xy_range, args.xy_step),
                  (args.z_range, args.z_step)),
        'regions_per_tile': args.regions_per_tile, 'xy_size': args.xy_size, 'z_size': args.z_size,
        'z_guard': z_guard, 'min_cells': args.min_cells, 'max_attempts': args.max_attempts,
        'seed': args.seed,
    }

    rows, warns = [], []
    if args.workers > 1 and len(tiles) > 1:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=mp.get_context('spawn')) as pool:
            futs = {pool.submit(process_tile, t, params): t for t in tiles}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Tiles"):
                r, w = fut.result()
                rows.extend(r); warns.extend(w)
    else:
        for t in tqdm(tiles, desc="Tiles"):
            r, w = process_tile(t, params)
            rows.extend(r); warns.extend(w)

    for w in warns:
        print(f"⚠️  {w}")
    if not rows:
        raise SystemExit("❌ 没有产出任何曲线，检查上面的警告。")

    df = pd.DataFrame(rows)
    curves_path = os.path.join(out_dir, f"{sample_name}_curves.csv")
    df.to_csv(curves_path, index=False)

    summaries = [summarize(df, 'voxel_iou')]
    if df['containment'].notna().any():
        summaries.append(summarize(df, 'containment'))
    summary = pd.concat(summaries, ignore_index=True)
    summary_path = os.path.join(out_dir, f"{sample_name}_summary.csv")
    summary.to_csv(summary_path, index=False)

    # 终端汇总：每个通道 × 指标 × 轴，聚合所有 tile/区域
    print("\n===== 汇总（所有 tile / 区域平均）=====")
    agg = (summary.groupby(['channel', 'metric', 'axis'])
           .agg(n=('value_at_optimum', 'size'),
                at_opt=('value_at_optimum', 'mean'),
                at_edge=('edge_mean', 'mean'),
                drop=('drop_ratio_at_edge', 'mean'),
                peak_hit=('peak_at_optimum', 'mean'),
                d_at_max=('delta_at_max', 'mean'))
           .reset_index())
    with pd.option_context('display.width', 160, 'display.max_rows', 200):
        print(agg.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n  at_opt   = 最优解处的指标值        at_edge = 扫描两端的指标均值")
    print("  drop     = (at_opt - at_edge)/at_opt，越接近 1 说明偏离后跌得越狠")
    print("  peak_hit = 曲线峰值正好落在 delta=0 的区域比例（1.0 = 每个区域都是）")
    print("  d_at_max = 峰值所在 delta 的均值，离 0 越远说明该通道的最优解越可疑")

    n_plots = 0
    if not args.no_plots:
        for ch in sorted(df['channel'].unique()):
            for metric in ('voxel_iou', 'containment'):
                p = os.path.join(out_dir, f"{sample_name}_{ch}_{metric}.png")
                if plot_channel(df, ch, metric, p, sample_name):
                    n_plots += 1

    held = df['held_out'].all()
    print(f"\n曲线数据 : {curves_path}")
    print(f"汇总     : {summary_path}")
    print(f"图       : {'跳过（--no-plots）' if args.no_plots else f'{n_plots} 张，在 {out_dir}'}")
    if not held:
        print("⚠️  有区域没能避开对齐窗口（held_out=False），这部分不算独立验证。")


if __name__ == '__main__':
    mp.freeze_support()
    main()
