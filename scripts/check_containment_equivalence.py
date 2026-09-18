# -*- coding: utf-8 -*-
"""
检验 point_cloud_aligner 的包含度搜索重构没有改变结果。

背景：为了让 solve_tile_positions.py 能用「全脑汇总点云」一次估通道常数，
`find_shift_containment` 被拆成了三块（`_coarse_from_peaks` / `_fine_containment` /
`containment_shift_from_arrays`），并且 `_fine_containment` 改成**候选 (soma, TF) 对只从
KD 树取一次**（半径放大到覆盖整个精搜索窗）。后者理论上是精确等价的：bbox 包含本身就要求
两个质心的距离不超过 soma 的半对角线（≤ max_radius），所以放大半径只会多出必然判不过的对。
本脚本在真实 tile 上把这件事测出来，而不是只靠推理。

两项检查：
  A. 逐候选调用 `_containment_score` 的**穷举**精搜索 vs 新的 `_fine_containment`
     —— 直接测那个优化本身（argmax 与匹配数都要一致）。
  B. 重构前的 `find_shift_containment` 函数体（原样抄在本文件里）vs 现在的实现，
     'displacement_hist' 和 'fft' 两种粗搜索都测。

用法
----
  python scripts/check_containment_equivalence.py --sample /path/to/T4
  python scripts/check_containment_equivalence.py --sample ... --tiles 326200_338600,337500_349900
  python scripts/check_containment_equivalence.py --sample ... --n-tiles 4 --fine-xy 8

退出码 0 = 全部一致；1 = 有不一致（会逐条打印）。
"""

import argparse
import os
import sys
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    sys.stderr.reconfigure(encoding='utf-8')

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd

import src.core.point_cloud_aligner as pca
from src.config.loader import load_config
from src.core.z_linker import run_z_linker

BOX_COLS = ["x1", "y1", "x2", "y2", "score", "mean", "class", "z"]


# ──────────────────────────────────────────────────────────────────────────────
# 重构前的实现，原样保留作为参照（只把内部 helper 换成从模块 import）
# ──────────────────────────────────────────────────────────────────────────────

def find_shift_containment_before_refactor(
        cells_ref_soma, cells_tf, z_lo, z_hi,
        bin_size=4, xy_res_um=0.65, z_res_um=8.0,
        xy_range_px=30, z_range_slices=5, fine_xy_px=8, fine_z_slices=2,
        max_center_dist_ratio=0.3, xy_margin=0, z_pad=0,
        coarse='displacement_hist'):
    if not cells_ref_soma or not cells_tf:
        return 0, 0, 0, 0.0
    soma_idx = pca._prepare_soma_containment_index(cells_ref_soma)
    tf_arrays = pca._cell_arrays(cells_tf)

    def _score(dx, dy, dz):
        return pca._containment_score(soma_idx, tf_arrays, dx, dy, dz,
                                      max_center_dist_ratio, xy_margin, z_pad)

    if coarse == 'fft':
        all_cells = cells_ref_soma + cells_tf
        x_min = min(c.get('x1_3d', c.get('cx', 0)) for c in all_cells)
        x_max = max(c.get('x2_3d', c.get('cx', 0)) for c in all_cells)
        y_min = min(c.get('y1_3d', c.get('cy', 0)) for c in all_cells)
        y_max = max(c.get('y2_3d', c.get('cy', 0)) for c in all_cells)
        pad = xy_range_px + fine_xy_px
        x_min -= pad; x_max += pad
        y_min -= pad; y_max += pad
        grid_ref = pca._voxelize_to_grid(cells_ref_soma, z_lo, z_hi, x_min, x_max, y_min, y_max,
                                         bin_size, xy_res_um, z_res_um)
        grid_tgt = pca._voxelize_to_grid(cells_tf, z_lo, z_hi, x_min, x_max, y_min, y_max,
                                         bin_size, xy_res_um, z_res_um)
        if grid_ref.sum() == 0 or grid_tgt.sum() == 0:
            return 0, 0, 0, 0.0
        dx_c, dy_c, dz_c = pca._fft_3d_shifts(grid_ref, grid_tgt, xy_range_px,
                                              z_range_slices, bin_size)
    else:
        best_c, best_c_score = (0, 0, 0), (-1, -1.0)
        for cx, cy, cz in pca._displacement_peaks(soma_idx, tf_arrays, xy_range_px,
                                                  z_range_slices, max_center_dist_ratio):
            for ddx in range(-2, 3):
                for ddy in range(-2, 3):
                    for ddz in range(-1, 2):
                        sc = _score(cx + ddx, cy + ddy, cz + ddz)
                        if sc > best_c_score:
                            best_c_score, best_c = sc, (cx + ddx, cy + ddy, cz + ddz)
        dx_c, dy_c, dz_c = best_c

    best_score = (-1, -1.0)
    best = (dx_c, dy_c, dz_c)
    for ddx in range(-fine_xy_px, fine_xy_px + 1):
        for ddy in range(-fine_xy_px, fine_xy_px + 1):
            for ddz in range(-fine_z_slices, fine_z_slices + 1):
                cand = (dx_c + ddx, dy_c + ddy, dz_c + ddz)
                score = _score(*cand)
                if score > best_score:
                    best_score, best = score, cand
    return (int(best[0]), int(best[1]), int(best[2]),
            float(best_score[0]) / max(1, len(cells_tf)))


def exhaustive_fine(soma_idx, tf_arrays, center, fine_xy_px, fine_z_slices,
                    max_center_dist_ratio, xy_margin, z_pad):
    """穷举版精搜索：每个候选位移都重新调用 _containment_score。"""
    dx0, dy0, dz0 = center
    best_score, best = (-1, -1.0), (dx0, dy0, dz0)
    for ddx in range(-fine_xy_px, fine_xy_px + 1):
        for ddy in range(-fine_xy_px, fine_xy_px + 1):
            for ddz in range(-fine_z_slices, fine_z_slices + 1):
                cand = (dx0 + ddx, dy0 + ddy, dz0 + ddz)
                sc = pca._containment_score(soma_idx, tf_arrays, *cand,
                                            max_center_dist_ratio, xy_margin, z_pad)
                if sc > best_score:
                    best_score, best = sc, cand
    return int(best[0]), int(best[1]), int(best[2]), max(best_score[0], 0)


# ──────────────────────────────────────────────────────────────────────────────

def load_cells(det_dir, tile, ch, z_link):
    path = os.path.join(det_dir, f"{tile}_{ch}_result.csv")
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path, usecols=BOX_COLS)
    if df.empty:
        return []
    mat = df[BOX_COLS].values.copy()
    mat[:, 6] = np.array([f"{v}_{ch}" for v in mat[:, 6]])
    _, vol = run_z_linker(mat, **z_link)
    return vol


def pick_tiles(det_dir, ref, n):
    """按参考通道 CSV 的大小挑 n 个细胞多的 tile（细胞太少的 tile 测不出什么）。"""
    sizes = []
    suffix = f"_{ref}_result.csv"
    for f in os.listdir(det_dir):
        if f.endswith(suffix):
            sizes.append((os.path.getsize(os.path.join(det_dir, f)), f[:-len(suffix)]))
    sizes.sort(reverse=True)
    return [t for _, t in sizes[:n]]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sample', required=True, help='样本根目录（或 detection_results/）')
    ap.add_argument('--config', default=None, help='默认 <results_dir>/runtime_config.json')
    ap.add_argument('--det-dir', default=None, help='默认 <results_dir>/1_tile_2d_raw')
    ap.add_argument('--tiles', default=None, help='逗号分隔的 tile 名；默认按细胞数自动挑')
    ap.add_argument('--n-tiles', type=int, default=2, help='自动挑几个 tile（默认 2）')
    ap.add_argument('--ref', default=None, help='soma 参考通道；默认 config 的 reference_channel')
    ap.add_argument('--tf-channels', default=None, help='要测的 TF 通道；默认 config 里所有 TF 通道')
    ap.add_argument('--fine-xy', type=int, default=4,
                    help='精搜索 XY 半径；穷举版是 (2n+1)^2*(2m+1) 次打分，调大很慢（默认 4）')
    ap.add_argument('--fine-z', type=int, default=2, help='精搜索 Z 半径（默认 2）')
    ap.add_argument('--skip-fft', action='store_true', help='跳过 fft 粗搜索那一组对照')
    return ap.parse_args()


def main():
    args = parse_args()
    sample = os.path.abspath(args.sample)
    results_dir = next((c for c in (os.path.join(sample, 'detection_results'), sample)
                        if os.path.isdir(os.path.join(c, '1_tile_2d_raw'))), None)
    if results_dir is None:
        raise SystemExit(f"❌ 在 {sample} 下找不到 1_tile_2d_raw")
    config = load_config(args.config or os.path.join(results_dir, 'runtime_config.json'))
    routing = [ch for ch in config.get('channels_routing', []) if ch.get('active', True)]
    settings = pca.resolve_align_settings(config, routing)
    det_dir = args.det_dir or os.path.join(results_dir, '1_tile_2d_raw')
    ref = args.ref or settings['reference_channel']
    tf_channels = ([c.strip() for c in args.tf_channels.split(',')] if args.tf_channels
                   else settings['tf_ch_ids'])
    tiles = ([t.strip() for t in args.tiles.split(',')] if args.tiles
             else pick_tiles(det_dir, ref, args.n_tiles))
    z_half = settings['sample_z_center_count'] // 2
    coarse_modes = ['displacement_hist'] if args.skip_fft else ['displacement_hist', 'fft']

    print(f"样本 {results_dir}")
    print(f"参考 {ref}   TF 通道 {tf_channels}   tile {tiles}")
    print(f"精搜索 ±{args.fine_xy} px / ±{args.fine_z} 层   "
          f"z 窗 ±{z_half} 层   粗搜索对照 {coarse_modes}\n")

    n_ok = n_bad = 0
    for tile in tiles:
        ref_all = load_cells(det_dir, tile, ref, settings['z_link']['soma'])
        if not ref_all:
            print(f"{tile}: 参考通道没有细胞，跳过")
            continue
        z_center = float(np.median([c.get('cz', 0) for c in ref_all]))
        z_lo, z_hi = z_center - z_half, z_center + z_half
        soma = pca.build_cell_boxes(ref_all, z_lo, z_hi)
        for ch in tf_channels:
            tf_all = load_cells(det_dir, tile, ch, settings['z_link']['tf'])
            if not tf_all:
                print(f"{tile} {ch}: 没有细胞，跳过")
                continue
            tf = pca.build_cell_boxes(tf_all, z_lo, z_hi)
            print(f"{tile} {ch}: soma {len(soma)} / TF {len(tf)}（z 窗内）")

            # ── A. 穷举精搜索 vs 新的一次取候选对 ──
            soma_idx = pca._prepare_soma_containment_index(soma)
            tf_arrays = pca._cell_arrays(tf)
            center = pca._coarse_from_peaks(soma_idx, tf_arrays,
                                            settings['xy_search_range_px'],
                                            settings['z_search_range_slices'],
                                            settings['max_center_dist_ratio'],
                                            0, settings.get('containment_z_pad', 0))
            t0 = time.time()
            a = exhaustive_fine(soma_idx, tf_arrays, center, args.fine_xy, args.fine_z,
                                settings['max_center_dist_ratio'], 0,
                                settings.get('containment_z_pad', 0))
            t_a = time.time() - t0
            t0 = time.time()
            b = pca._fine_containment(soma_idx, tf_arrays, center, args.fine_xy, args.fine_z,
                                      settings['max_center_dist_ratio'], 0,
                                      settings.get('containment_z_pad', 0))
            t_b = time.time() - t0
            same = a == b
            n_ok += same; n_bad += (not same)
            print(f"  A 精搜索  穷举 {a} ({t_a:.0f}s)   新 {b} ({t_b:.0f}s)   "
                  f"{'一致' if same else '❌ 不一致'}   加速 {t_a / max(t_b, 1e-6):.1f}x")

            # ── B. 重构前的整个函数 vs 现在的 ──
            for coarse in coarse_modes:
                kw = dict(bin_size=settings['voxel_bin_size_px'],
                          xy_res_um=settings['xy_resolution_um'],
                          z_res_um=settings['z_resolution_um'],
                          xy_range_px=settings['xy_search_range_px'],
                          z_range_slices=settings['z_search_range_slices'],
                          fine_xy_px=args.fine_xy, fine_z_slices=args.fine_z,
                          max_center_dist_ratio=settings['max_center_dist_ratio'],
                          xy_margin=0, z_pad=settings.get('containment_z_pad', 0),
                          coarse=coarse)
                old = find_shift_containment_before_refactor(soma, tf, z_lo, z_hi, **kw)
                new = pca.find_shift_containment(soma, tf, z_lo, z_hi, **kw)
                same = old == new
                n_ok += same; n_bad += (not same)
                print(f"  B {coarse:18s} 旧 {old}   新 {new}   "
                      f"{'一致' if same else '❌ 不一致'}")

    print(f"\n{n_ok} 项一致，{n_bad} 项不一致")
    return 1 if n_bad else 0


if __name__ == '__main__':
    sys.exit(main())
