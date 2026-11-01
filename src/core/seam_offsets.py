# -*- coding: utf-8 -*-
"""Measured offsets between adjacent tiles, per channel and z segment.

Stage 3 places every tile at its XML position. A cell imaged by two adjacent tiles
does not land on the same global box when that position is off; on ET10 the error
drifts with z by ~10-40 px, differs between channels by ~5 px (per-tile channel
alignment error), and cannot be removed by any per-tile translation (2x2 loop
closure ~5 px p50, ~14 px p90). Stage 3 therefore compares overlap detections only
after removing the offset measured on that seam.

The measurement mirrors channel alignment. For each (channel, seam, z segment) the
volume is the nominal overlap of the two tiles (tile_overlap_pct, z shift 0) over
seg_layers layers. Both tiles' boxes are rasterised into occupancy grids; a 3D FFT
cross-correlation searches +-search_xy_px / +-search_z around the nominal guess, the
seam's segments vote a common centre, and each segment then takes its peak within
+-drift_xy_px / +-drift_z of that centre. A fine search (1 px, 1 layer) around the
peak maximises the box IoU of the two volumes, computed from the box coordinates.

Offsets are stored relative to the XML placement Stage 3 uses (b centre minus a
centre, global px; layer = z - z0 - 1), plus the shift from the nominal guess.

Segments with too few matched pairs, a weak peak, or a jump from their neighbours
fall back, in order, to: frame channel's segment + this channel's constant difference
on the seam, nearest valid segment of this channel, frame channel's segment, nearest
valid frame segment. A seam with nothing usable gets no entry (offset treated as 0).

Output: <5_2d_global>/seam_offsets/seam_offsets.csv (+ manifest).
"""

import logging
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy import fft as sfft
from scipy.spatial import cKDTree

from src.core.provenance import (file_sha256, output_manifest_valid,
                                 write_output_manifest)

DEFAULT_PARAMS = {
    'seg_layers': 50,        # z segment height (global layers): one volume per segment
    'overlap_pct': None,     # nominal overlap of the initial guess; None -> pre_align_params.tile_overlap_pct (15)
    'search_xy_px': 160,     # coarse search radius around the nominal guess (px)
    'search_z': 10,          # coarse search radius around the nominal guess (layers)
    'drift_xy_px': 50,       # per-segment search radius around the seam's common centre (px)
    'drift_z': 3,            # ... (layers)
    'bin_px': 4,             # coarse occupancy grid bin (px)
    'fine_xy_px': 5,         # fine IoU search radius around the coarse peak (px)
    'fine_z': 1,             # ... (layers)
    'edge_px': 8,            # ignore this margin at the overlap border
    'min_match': 20,         # matched box pairs a segment needs to count as measured
    'min_snr': 6.0,          # coarse correlation peak above background, in robust SDs
    'max_jump_px': 15,       # max xy distance from the neighbouring segments' median
    'match_iou': 0.4,        # a box pair counts as matched above this IoU ...
    'match_iomin': 0.6,      # ... or this IoMin (same rule as Stage 3)
    'fft_threads': 4,        # scipy.fft workers per channel process
}
MEASURE_COLUMNS = ['channel', 'tile', 'nb_tile', 'direction', 'seg', 'n_a', 'n_b', 'n_dup',
                   'snr', 'iou', 'outlier', 'nom_dx', 'nom_dy', 'nom_dz',
                   'meas_dx', 'meas_dy', 'meas_dz']
OUTPUT_COLUMNS = MEASURE_COLUMNS + ['dx', 'dy', 'dz', 'source']


def resolve_params(config):
    cfg = config.get('seam_dedup', {}) or {}
    unknown = set(cfg) - set(DEFAULT_PARAMS) - {'enabled'}
    if unknown:
        raise ValueError(f"Unknown seam_dedup options: {sorted(unknown)}")
    params = dict(DEFAULT_PARAMS)
    params.update({k: v for k, v in cfg.items() if k != 'enabled'})
    if params['overlap_pct'] is None:
        params['overlap_pct'] = float(
            (config.get('pre_align_params') or {}).get('tile_overlap_pct', 15))
    return bool(cfg.get('enabled', True)), params


def grid_names(geom):
    """{(row, col): tile_name} from the Stage 3 geometry."""
    return {tuple(rc): os.path.basename(name) for name, rc in geom['dir_dict'].items()}


def tile_origin(geom, rc):
    """Global (x, y) origin and z0 exactly as combine_predictions computes them."""
    abs_x, abs_y, abs_z = geom['disp_mat_fin'][rc]
    return float(abs_x), float(abs_y), int(geom['z_start']) - int(abs_z)


def adjacent_seams(geom):
    """[(tile_a, tile_b, direction)] for each EAST/SOUTH neighbour pair."""
    names = grid_names(geom)
    seams = []
    for (r, c), name in sorted(names.items()):
        for direction, dr, dc in (('EAST', 0, 1), ('SOUTH', 1, 0)):
            nb = names.get((r + dr, c + dc))
            if nb is not None:
                seams.append((name, nb, direction))
    return seams


def nominal_step(geom, params):
    return geom['tile_size'] * (1.0 - float(params['overlap_pct']) / 100.0)


def nominal_offset(geom, rc_a, rc_b, params):
    """Offset (b copy minus a copy, XML placement) if the tiles sat on the nominal grid."""
    ax, ay, z0a = tile_origin(geom, rc_a)
    bx, by, z0b = tile_origin(geom, rc_b)
    step = nominal_step(geom, params)
    return np.array([bx - ax - (rc_b[1] - rc_a[1]) * step,
                     by - ay - (rc_b[0] - rc_a[0]) * step,
                     z0a - z0b], float)


def nominal_band(geom, rc_a, rc_b, params):
    """Global xy rect of tile a covered by tile b on the nominal grid, shrunk by edge_px."""
    ax, ay, _ = tile_origin(geom, rc_a)
    t, step, m = geom['tile_size'], nominal_step(geom, params), params['edge_px']
    x0, x1, y0, y1 = 0.0, float(t), 0.0, float(t)
    if rc_b[1] > rc_a[1]:
        x0 = step
    if rc_b[0] > rc_a[0]:
        y0 = step
    return (ax + x0 + m, ax + x1 - m, ay + y0 + m, ay + y1 - m)


def load_tile_band(csv_path, geom, rc, rects):
    """Rows of one tile CSV whose global centre lies in any rect.

    Returns float array [x1, y1, x2, y2, layer] in Stage 3 global coordinates.
    """
    if not os.path.isfile(csv_path):
        return None
    ox, oy, z0 = tile_origin(geom, rc)
    parts = []
    for df in pd.read_csv(csv_path, usecols=['x1', 'y1', 'x2', 'y2', 'z'], chunksize=2_000_000):
        arr = df[['x1', 'y1', 'x2', 'y2', 'z']].to_numpy(np.float64)
        layer = np.floor(arr[:, 4]) - z0 - 1
        keep = (layer >= 0) & (layer < geom['Z'])
        cx = (arr[:, 0] + arr[:, 2]) / 2 + ox
        cy = (arr[:, 1] + arr[:, 3]) / 2 + oy
        inside = np.zeros(len(arr), bool)
        for x0, x1, y0, y1 in rects:
            inside |= (cx >= x0) & (cx <= x1) & (cy >= y0) & (cy <= y1)
        keep &= inside
        arr = arr[keep]
        arr[:, [0, 2]] += ox
        arr[:, [1, 3]] += oy
        arr[:, 4] = layer[keep]
        parts.append(arr.astype(np.float32))   # whole-sample bands stay in memory
    return np.concatenate(parts) if parts else np.empty((0, 5), np.float32)


def _centres(boxes):
    return np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2])


def _in_rect(c, rect):
    return (c[:, 0] >= rect[0]) & (c[:, 0] <= rect[1]) & (c[:, 1] >= rect[2]) & (c[:, 1] <= rect[3])


def _raster(boxes, origin, shape, bin_px):
    """Binary occupancy grid (z, y, x) of 2D boxes [x1, y1, x2, y2, layer]."""
    nz, ny, nx = shape
    x0, y0, z0 = origin
    zi = np.floor(boxes[:, 4] - z0 + 0.5).astype(int)
    ix1 = np.clip(np.floor((boxes[:, 0] - x0) / bin_px).astype(int), 0, nx)
    ix2 = np.clip(np.ceil((boxes[:, 2] - x0) / bin_px).astype(int), 0, nx)
    iy1 = np.clip(np.floor((boxes[:, 1] - y0) / bin_px).astype(int), 0, ny)
    iy2 = np.clip(np.ceil((boxes[:, 3] - y0) / bin_px).astype(int), 0, ny)
    keep = (zi >= 0) & (zi < nz) & (ix2 > ix1) & (iy2 > iy1)
    zi, ix1, ix2, iy1, iy2 = zi[keep], ix1[keep], ix2[keep], iy1[keep], iy2[keep]
    diff = np.zeros((nz, ny + 1, nx + 1), np.float32)
    np.add.at(diff, (zi, iy1, ix1), 1)
    np.add.at(diff, (zi, iy1, ix2), -1)
    np.add.at(diff, (zi, iy2, ix1), -1)
    np.add.at(diff, (zi, iy2, ix2), 1)
    grid = diff.cumsum(1).cumsum(2)[:, :ny, :nx]
    return (grid > 0.5).astype(np.float32)


def _coarse_window(a, b, band, s0, s1, params):
    """Occupancy cross-correlation for shifts u (b minus a) within the search window.

    a: tile a boxes of the segment inside band; b: tile b boxes already moved by the
    nominal offset. Returns (window[dz, dy, dx], rb) for dz in [-search_z, search_z]
    and dx, dy in bins of bin_px within +-rb bins.
    """
    S, Sz, bp = params['search_xy_px'], params['search_z'], params['bin_px']
    rb = int(np.ceil(S / bp))
    origin = (band[0] - rb * bp, band[2] - rb * bp, s0 - Sz)
    shape = (int(s1 - s0 + 2 * Sz),
             int(np.ceil((band[3] - band[2]) / bp)) + 2 * rb + 1,
             int(np.ceil((band[1] - band[0]) / bp)) + 2 * rb + 1)
    grid_a = _raster(a, origin, shape, bp)
    grid_b = _raster(b, origin, shape, bp)
    threads = int(params.get('fft_threads', 1))
    corr = sfft.irfftn(np.conj(sfft.rfftn(grid_a, workers=threads)) *
                       sfft.rfftn(grid_b, workers=threads), s=shape, workers=threads)
    iz = np.arange(-Sz, Sz + 1) % shape[0]
    iy = np.arange(-rb, rb + 1) % shape[1]
    ix = np.arange(-rb, rb + 1) % shape[2]
    return corr[np.ix_(iz, iy, ix)].astype(np.float32), rb


def _snr(window, peak):
    background = float(np.median(window))
    mad = float(np.median(np.abs(window - background))) * 1.4826
    return (peak - background) / (mad + 1e-3)


def _restricted_peak(window, rb, centre, params):
    """Argmax of the window, within the drift range of centre if given (u in px / layers)."""
    Sz, bp = params['search_z'], params['bin_px']
    if centre is None:
        k = np.unravel_index(int(np.argmax(window)), window.shape)
    else:
        dz = np.arange(-Sz, Sz + 1)[:, None, None]
        dy = (np.arange(-rb, rb + 1) * bp)[None, :, None]
        dx = (np.arange(-rb, rb + 1) * bp)[None, None, :]
        ok = ((np.abs(dz - centre[2]) <= params['drift_z']) &
              (np.hypot(dx - centre[0], dy - centre[1]) <= params['drift_xy_px']))
        k = np.unravel_index(int(np.argmax(np.where(ok, window, -np.inf))), window.shape)
    u = np.array([(k[2] - rb) * bp, (k[1] - rb) * bp, k[0] - Sz], float)
    return u, float(window[k])


def _pair_overlap(a, b):
    """IoU and IoMin of paired boxes (n, 4) and (n, 4)."""
    ix = np.maximum(0, np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0]))
    iy = np.maximum(0, np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1]))
    inter = ix * iy
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a + area_b - inter + 1e-6), inter / (np.minimum(area_a, area_b) + 1e-6)


def _parabola_vertex(fm, f0, fp):
    den = fm - 2 * f0 + fp
    return float(np.clip(0.5 * (fm - fp) / den, -0.5, 0.5)) if den < 0 else 0.0


def _fine_search(a, b, r_coarse, params):
    """Maximise the box IoU of the two volumes around r_coarse (b minus a, px / layers).

    The IoU numerator is the summed intersection of all box pairs, the denominator the
    summed box areas of both volumes minus it. Returns (r_best, iou, n_match) or None.
    """
    F, Fz = int(params['fine_xy_px']), int(params['fine_z'])
    ca, cb = _centres(a), _centres(b)
    width = float(max(np.max(a[:, 2] - a[:, 0]), np.max(a[:, 3] - a[:, 1]),
                      np.max(b[:, 2] - b[:, 0]), np.max(b[:, 3] - b[:, 1])))
    radius = F + min(width, 120.0)
    tree_b = cKDTree(np.column_stack([cb, b[:, 4] * 1e4]))
    steps = np.arange(-F, F + 1, dtype=float)
    best = None
    for dz in range(int(r_coarse[2]) - Fz, int(r_coarse[2]) + Fz + 1):
        tree_a = cKDTree(np.column_stack([ca + r_coarse[:2], (a[:, 4] + dz) * 1e4]))
        sm = tree_a.sparse_distance_matrix(tree_b, radius, output_type='ndarray')
        i, j = sm['i'], sm['j']
        if not len(i):
            continue
        ai, bj = a[i], b[j]
        # Summed intersection for every (fy, fx): overlap_y(fy) . overlap_x(fx) over pairs.
        sx = r_coarse[0] + steps[:, None]
        sy = r_coarse[1] + steps[:, None]
        ox = np.maximum(0, np.minimum(ai[None, :, 2] + sx, bj[None, :, 2]) -
                        np.maximum(ai[None, :, 0] + sx, bj[None, :, 0]))
        oy = np.maximum(0, np.minimum(ai[None, :, 3] + sy, bj[None, :, 3]) -
                        np.maximum(ai[None, :, 1] + sy, bj[None, :, 1]))
        # Row by row instead of oy @ ox.T: the local numpy's BLAS crashes on matmul.
        inter = np.stack([(row[None, :] * ox).sum(axis=1) for row in oy])
        k = np.unravel_index(int(np.argmax(inter)), inter.shape)
        if best is None or inter[k] > best[0]:
            best = (float(inter[k]), dz, k, inter)
    if best is None:
        return None
    total, dz, (ky, kx), inter = best
    sub_x = _parabola_vertex(*inter[ky, kx - 1:kx + 2]) if 0 < kx < 2 * F else 0.0
    sub_y = _parabola_vertex(*inter[ky - 1:ky + 2, kx]) if 0 < ky < 2 * F else 0.0
    r = np.array([r_coarse[0] + steps[kx] + sub_x, r_coarse[1] + steps[ky] + sub_y, dz], float)
    area_a = float(np.sum((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])))
    area_b = float(np.sum((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])))
    union = area_a + area_b - total
    # One-to-one matched pairs at the best shift (the duplicates Stage 3 should remove).
    tree_a = cKDTree(np.column_stack([ca + r[:2], (a[:, 4] + dz) * 1e4]))
    sm = tree_a.sparse_distance_matrix(tree_b, radius, output_type='ndarray')
    i, j = sm['i'], sm['j']
    iou, iomin = _pair_overlap(a[i, :4] + [r[0], r[1], r[0], r[1]], b[j, :4])
    match = (iou > params['match_iou']) | (iomin > params['match_iomin'])
    order = np.argsort(-iou[match])
    used_a, used_b = set(), set()
    for p, q in zip(i[match][order].tolist(), j[match][order].tolist()):
        if p not in used_a and q not in used_b:
            used_a.add(p)
            used_b.add(q)
    return r, (total / union if union > 0 else 0.0), len(used_a)


def measure_seam(a, b, band, b_fov, r0, n_segments, params):
    """Per-segment offset of tile b relative to tile a (b centre minus a centre).

    a, b: [x1, y1, x2, y2, layer] in Stage 3 global coordinates. band: the nominal
    overlap rect in tile a. b_fov: tile b's global xy rect. r0: the nominal offset.
    """
    a, b = a.astype(np.float64), b.astype(np.float64)
    seg_n, m = params['seg_layers'], params['edge_px']
    a = a[_in_rect(_centres(a), band)]
    b_nom = b.copy()
    b_nom[:, [0, 2]] -= r0[0]
    b_nom[:, [1, 3]] -= r0[1]
    b_nom[:, 4] -= r0[2]
    seg_a = (a[:, 4] // seg_n).astype(int)
    windows = {}
    for seg in range(n_segments):
        s0, s1 = seg * seg_n, (seg + 1) * seg_n
        sel = seg_a == seg
        if sel.sum() < params['min_match']:
            continue
        bsel = (b_nom[:, 4] >= s0 - params['search_z']) & (b_nom[:, 4] < s1 + params['search_z'])
        if bsel.sum() < params['min_match']:
            continue
        windows[seg] = (_coarse_window(a[sel], b_nom[bsel], band, s0, s1, params), sel)
    # The seam's common centre: median of the strong segments' unrestricted peaks.
    strong = []
    for (window, rb), _ in windows.values():
        u, value = _restricted_peak(window, rb, None, params)
        if _snr(window, value) >= params['min_snr']:
            strong.append(u)
    centre = np.median(strong, axis=0) if strong else None
    rows = []
    for seg in range(n_segments):
        row = {'seg': seg, 'n_a': int((seg_a == seg).sum()), 'n_b': 0, 'n_dup': 0,
               'snr': 0.0, 'iou': 0.0, 'outlier': False,
               'nom_dx': np.nan, 'nom_dy': np.nan, 'nom_dz': np.nan,
               'meas_dx': np.nan, 'meas_dy': np.nan, 'meas_dz': np.nan}
        rows.append(row)
        if seg not in windows:
            continue
        (window, rb), sel = windows[seg]
        u, value = _restricted_peak(window, rb, centre, params)
        row['snr'] = float(_snr(window, value))
        r_coarse = r0 + u
        # Only the parts both tiles see at this shift take part in the fine search.
        fov = (b_fov[0] - r_coarse[0] + m, b_fov[1] - r_coarse[0] - m,
               b_fov[2] - r_coarse[1] + m, b_fov[3] - r_coarse[1] - m)
        a_seg = a[sel]
        a_seg = a_seg[_in_rect(_centres(a_seg), fov)]
        s0, s1 = seg * seg_n, (seg + 1) * seg_n
        lo = s0 + r_coarse[2] - params['fine_z']
        hi = s1 + r_coarse[2] + params['fine_z']
        cb = _centres(b) - r_coarse[:2]
        b_seg = b[(b[:, 4] >= lo) & (b[:, 4] < hi) & _in_rect(cb, band) & _in_rect(cb, fov)]
        row['n_b'] = int(len(b_seg))
        if len(a_seg) < params['min_match'] or len(b_seg) < params['min_match']:
            continue
        fine = _fine_search(a_seg, b_seg, r_coarse, params)
        if fine is None:
            continue
        r, iou, n_match = fine
        row.update(n_dup=int(n_match), iou=float(iou),
                   nom_dx=float(r[0] - r0[0]), nom_dy=float(r[1] - r0[1]),
                   nom_dz=int(round(r[2] - r0[2])),
                   meas_dx=float(r[0]), meas_dy=float(r[1]), meas_dz=int(round(r[2])))
    _flag_jumps(rows, params)
    return rows


def _flag_jumps(rows, params):
    """Mark measured segments that jump away from their neighbours' median."""
    good = {r['seg']: np.array([r['meas_dx'], r['meas_dy'], r['meas_dz']], float)
            for r in rows if r['n_dup'] >= params['min_match'] and r['snr'] >= params['min_snr']}
    for r in rows:
        if r['seg'] not in good:
            continue
        near = [v for s, v in good.items() if s != r['seg'] and abs(s - r['seg']) <= 2]
        if not near:
            continue
        ref = np.median(near, axis=0)
        here = good[r['seg']]
        if (np.hypot(*(here[:2] - ref[:2])) > params['max_jump_px'] or
                abs(here[2] - ref[2]) > 2):
            r['outlier'] = True


def measure_channel(ch_id, ch_type, src_dir, geom, params):
    """Measure every EAST/SOUTH seam of one channel; returns a DataFrame."""
    names = grid_names(geom)
    rc_of = {name: rc for rc, name in names.items()}
    t = geom['tile_size']
    pad = params['search_xy_px'] + params['fine_xy_px'] + 2
    rects, plan = {}, []
    for a, b, direction in adjacent_seams(geom):
        band = nominal_band(geom, rc_of[a], rc_of[b], params)
        r0 = nominal_offset(geom, rc_of[a], rc_of[b], params)
        bx, by, _ = tile_origin(geom, rc_of[b])
        plan.append((a, b, direction, band, (bx, bx + t, by, by + t), r0))
        rects.setdefault(a, []).append(band)
        rects.setdefault(b, []).append((band[0] + r0[0] - pad, band[1] + r0[0] + pad,
                                        band[2] + r0[1] - pad, band[3] + r0[1] + pad))
    bands = {name: load_tile_band(os.path.join(src_dir, f"{name}_{ch_id}_result.csv"),
                                  geom, rc_of[name], tile_rects)
             for name, tile_rects in rects.items()}
    n_segments = int(np.ceil(int(geom['Z']) / params['seg_layers']))
    rows = []
    for a, b, direction, band, b_fov, r0 in plan:
        if bands.get(a) is None or bands.get(b) is None or not len(bands[a]) or not len(bands[b]):
            continue
        for row in measure_seam(bands[a], bands[b], band, b_fov, r0, n_segments, params):
            rows.append({'channel': ch_id, 'tile': a, 'nb_tile': b, 'direction': direction, **row})
    return pd.DataFrame(rows, columns=MEASURE_COLUMNS)


def _valid(df, params):
    return ((df['n_dup'] >= params['min_match']) & (df['snr'] >= params['min_snr']) &
            ~df['outlier'].astype(bool))


def resolve_offsets(measured, frame, n_segments, params):
    """Fill every (channel, seam, segment) with an offset and record where it came from."""
    measured = measured.copy()
    measured['valid'] = _valid(measured, params)
    frame_valid = {}
    for row in measured[(measured.channel == frame) & measured.valid].itertuples():
        frame_valid.setdefault((row.tile, row.nb_tile), {})[row.seg] = np.array(
            [row.meas_dx, row.meas_dy, row.meas_dz], float)
    by_seam = {key: g.set_index('seg') for key, g in measured.groupby(['channel', 'tile', 'nb_tile'])}
    seams = sorted({(r.tile, r.nb_tile, r.direction) for r in measured.itertuples()})
    defaults = {col: (False if col == 'outlier' else
                      np.nan if col.startswith(('meas_', 'nom_')) else 0)
                for col in MEASURE_COLUMNS[5:]}
    out = []
    for ch in sorted(measured.channel.unique()):
        for tile, nb, direction in seams:
            g = by_seam.get((ch, tile, nb))
            own = {} if g is None else {
                seg: np.array([r.meas_dx, r.meas_dy, r.meas_dz], float)
                for seg, r in g[g.valid].iterrows()}
            fr = frame_valid.get((tile, nb), {})
            common = sorted(set(own) & set(fr))
            const = (np.median([own[s] - fr[s] for s in common], axis=0)
                     if ch != frame and len(common) >= 2 else None)
            for seg in range(n_segments):
                value, source = None, None
                if seg in own:
                    value, source = own[seg], 'own'
                elif const is not None and seg in fr:
                    value, source = fr[seg] + const, 'frame_plus_const'
                elif own:
                    near = min(own, key=lambda s: (abs(s - seg), s))
                    value, source = own[near], 'own_nearest'
                elif seg in fr:
                    value, source = fr[seg], 'frame'
                elif fr:
                    near = min(fr, key=lambda s: (abs(s - seg), s))
                    value, source = fr[near], 'frame_nearest'
                if value is None:
                    continue
                meas = g.loc[seg] if g is not None and seg in g.index else None
                record = {'channel': ch, 'tile': tile, 'nb_tile': nb, 'direction': direction,
                          'seg': seg}
                for col, default in defaults.items():
                    record[col] = meas[col] if meas is not None else default
                record.update(dx=float(value[0]), dy=float(value[1]),
                              dz=int(round(float(value[2]))), source=source)
                out.append(record)
    return pd.DataFrame(out, columns=OUTPUT_COLUMNS)


def load_offsets(path, channel):
    """{(tile_a, tile_b): {seg: (dx, dy, dz)}} for one channel (b minus a, global px/layers)."""
    df = pd.read_csv(path, usecols=['channel', 'tile', 'nb_tile', 'seg', 'dx', 'dy', 'dz'])
    df = df[df.channel.astype(str) == str(channel)]
    lookup = {}
    for row in df.itertuples():
        lookup.setdefault((row.tile, row.nb_tile), {})[int(row.seg)] = (
            float(row.dx), float(row.dy), int(row.dz))
    return lookup


def _source_stamps(src_dir, ch_id, tiles):
    stamps = {}
    for tile in tiles:
        path = os.path.join(src_dir, f"{tile}_{ch_id}_result.csv")
        if os.path.isfile(path):
            st = os.stat(path)
            stamps[f"{tile}_{ch_id}_result.csv"] = [st.st_size, st.st_mtime_ns]
    return stamps


def stage_signature(routing, src_dirs, geom, frame, params):
    tiles = sorted(grid_names(geom).values())
    return {
        'version': 2,
        'frame': frame,
        # fft_threads only changes speed, not the result.
        'params': {k: v for k, v in params.items() if k != 'fft_threads'},
        'channels': {ch['id']: ch.get('type', 'soma') for ch in routing},
        # size + mtime only: Stage 3 hashes these files itself, and it hashes this
        # stage's output, so a content change still reaches the global 2D checkpoint.
        'sources': {ch['id']: _source_stamps(src_dirs[ch['id']], ch['id'], tiles)
                    for ch in routing},
        'geometry': {
            'positions': [(name, [int(v) for v in geom['disp_mat_fin'][rc]])
                          for rc, name in sorted(grid_names(geom).items())],
            'z_start': int(geom['z_start']), 'Z': int(geom['Z']),
            'tile_size': int(geom['tile_size']),
        },
        'code_sha256': file_sha256(os.path.abspath(__file__)),
    }


def run_seam_offset_stage(routing, src_dirs, geom, out_dir, frame, params, n_workers=1):
    """Measure (or reuse) seam offsets for all channels; returns the CSV path."""
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, 'seam_offsets.csv')
    manifest = os.path.join(out_dir, '_seam_offsets_manifest.json')
    signature = stage_signature(routing, src_dirs, geom, frame, params)
    if output_manifest_valid(manifest, signature):
        logging.info("✔️ Seam offsets already measured: %s", out_csv)
        return out_csv
    logging.info("Measuring seam offsets per channel and z segment (%d channels)...", len(routing))
    parts = []
    n_workers = max(1, min(int(n_workers), len(routing)))
    if n_workers == 1:
        for ch in routing:
            parts.append(measure_channel(ch['id'], ch.get('type', 'soma'),
                                         src_dirs[ch['id']], geom, params))
    else:
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context('spawn')) as pool:
            futs = {pool.submit(measure_channel, ch['id'], ch.get('type', 'soma'),
                                src_dirs[ch['id']], geom, params): ch['id'] for ch in routing}
            for fut in as_completed(futs):
                parts.append(fut.result())
    measured = (pd.concat(parts, ignore_index=True) if parts
                else pd.DataFrame(columns=MEASURE_COLUMNS))
    if frame is None and len(measured):
        # Without a configured frame, sparse channels fall back to the densest one.
        dense = measured[_valid(measured, params)].groupby('channel')['n_dup'].sum()
        frame = dense.idxmax() if len(dense) else None
        logging.info("  Seam fallback channel: %s (most measured duplicates)", frame)
    n_segments = int(np.ceil(int(geom['Z']) / params['seg_layers']))
    resolved = resolve_offsets(measured, frame, n_segments, params)
    resolved.to_csv(out_csv + '.part', index=False)
    os.replace(out_csv + '.part', out_csv)
    write_output_manifest(manifest, [out_csv], signature)
    for ch, g in resolved.groupby('channel'):
        counts = g.source.value_counts().to_dict()
        own = g[g.source == 'own']
        logging.info("  [%s] seam segments: %s; measured |dxy| p50 %.1f px, p90 %.1f px",
                     ch, counts, float(np.median(np.hypot(own.dx, own.dy))) if len(own) else float('nan'),
                     float(np.percentile(np.hypot(own.dx, own.dy), 90)) if len(own) else float('nan'))
    return out_csv
