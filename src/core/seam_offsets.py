# -*- coding: utf-8 -*-
"""Measured residual offsets between adjacent tiles, per channel and z segment.

Stage 3 places every tile at its XML position. A cell imaged by two adjacent tiles
does not land on the same global box when that position is off; on ET10 the error
drifts with z by ~10-40 px, differs between channels by ~5 px (per-tile channel
alignment error), and cannot be removed by any per-tile translation (2x2 loop
closure ~5 px p50, ~14 px p90). Stage 3 therefore compares overlap detections only
after removing the offset measured on that seam.

For each (channel, seam, z segment) the offset is the peak of the 2D histogram of
center differences (B minus A, global placement) between the two tiles' detections,
searched over dz in [-max_dz, max_dz]. Offsets are measured in the same global frame
that `stitcher.combine_predictions` uses: x/y + ABS_X/ABS_Y, layer = z - z0 - 1.

Segments with too few duplicates fall back, in order, to:
  frame channel's segment + this channel's constant difference on the seam,
  nearest valid segment of this channel, frame channel's segment, nearest valid
  frame segment.  A seam with nothing usable gets no entry (offset treated as 0).

Output: <5_2d_global>/seam_offsets/seam_offsets.csv (+ manifest).
"""

import logging
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree

from src.core.provenance import (file_sha256, output_manifest_valid,
                                 write_output_manifest)

DEFAULT_PARAMS = {
    'seg_layers': 50,        # z segment height (global layers)
    'win_px': 40,            # xy search radius for the offset (px)
    'max_dz': 2,             # dz search range (layers)
    'edge_px': 8,            # ignore this margin at the overlap border
    'min_dup': 20,           # duplicate pairs a segment needs to count as measured
    'min_contrast': 5.0,     # histogram peak / median background
    'r_dup_tf': 3.0,         # pair-to-peak distance counted as duplicate (px)
    'r_dup_soma': 5.0,
}
OUTPUT_COLUMNS = ['channel', 'tile', 'nb_tile', 'direction', 'seg', 'n_a', 'n_dup',
                  'contrast', 'meas_dx', 'meas_dy', 'meas_dz', 'dx', 'dy', 'dz', 'source']


def resolve_params(config):
    cfg = config.get('seam_dedup', {}) or {}
    unknown = set(cfg) - set(DEFAULT_PARAMS) - {'enabled'}
    if unknown:
        raise ValueError(f"Unknown seam_dedup options: {sorted(unknown)}")
    params = dict(DEFAULT_PARAMS)
    params.update({k: v for k, v in cfg.items() if k != 'enabled'})
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


def _overlap(geom, rc_a, rc_b, margin):
    """Global xy rectangle shared by two tiles, shrunk by margin."""
    ax, ay, _ = tile_origin(geom, rc_a)
    bx, by, _ = tile_origin(geom, rc_b)
    t = geom['tile_size']
    return (max(ax, bx) + margin, min(ax, bx) + t - margin,
            max(ay, by) + margin, min(ay, by) + t - margin)


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


def measure_seam(a, b, rect, r_dup, params):
    """Per-segment offset of tile b relative to tile a (b centre minus a centre)."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    x0, x1, y0, y1 = rect
    win, seg_n = params['win_px'], params['seg_layers']
    ca, cb = _centres(a), _centres(b)
    in_a = (ca[:, 0] >= x0) & (ca[:, 0] <= x1) & (ca[:, 1] >= y0) & (ca[:, 1] <= y1)
    a, ca = a[in_a], ca[in_a]
    if len(a) < params['min_dup'] or len(b) < params['min_dup']:
        return []
    # Layers are scaled far apart so a 2D radius query never crosses layers.
    tree_b = cKDTree(np.column_stack([cb, b[:, 4] * 1e4]))
    seg_a = (a[:, 4] // seg_n).astype(int)
    pairs = {}
    for dz in range(-params['max_dz'], params['max_dz'] + 1):
        tree_a = cKDTree(np.column_stack([ca, (a[:, 4] + dz) * 1e4]))
        sm = tree_a.sparse_distance_matrix(tree_b, win, output_type='ndarray')
        pairs[dz] = (sm['i'], cb[sm['j']] - ca[sm['i']], sm['j'])
    rows = []
    for seg in np.unique(seg_a):
        sel = seg_a == seg
        best = None
        for dz, (i, d, _) in pairs.items():
            m = sel[i]
            if m.sum() < params['min_dup']:
                continue
            hist, _, _ = np.histogram2d(d[m, 0], d[m, 1], bins=2 * int(win),
                                        range=[[-win, win], [-win, win]])
            smooth = uniform_filter(hist, 3, mode='constant') * 9
            k = np.unravel_index(np.argmax(smooth), smooth.shape)
            nonzero = smooth[smooth > 0]
            background = float(np.median(nonzero)) if nonzero.size else 0.0
            if best is None or smooth[k] > best[1]:
                best = (dz, float(smooth[k]), background,
                        np.array([k[0] - win + 0.5, k[1] - win + 0.5]))
        row = {'seg': int(seg), 'n_a': int(sel.sum()), 'n_dup': 0, 'contrast': 0.0,
               'meas_dx': np.nan, 'meas_dy': np.nan, 'meas_dz': np.nan}
        if best is not None:
            dz, peak_h, background, peak = best
            i, d, j = pairs[dz]
            m = sel[i]
            i, d, j = i[m], d[m], j[m]
            for _ in range(3):
                near = np.hypot(*(d - peak).T) < r_dup
                if not near.any():
                    break
                peak = d[near].mean(axis=0)
            near = np.hypot(*(d - peak).T) < r_dup
            order = np.argsort(np.hypot(*(d[near] - peak).T))
            ii, jj = i[near][order], j[near][order]
            _, first = np.unique(ii, return_index=True)
            _, first_j = np.unique(jj[first], return_index=True)
            row.update(n_dup=int(len(first_j)), contrast=peak_h / (background + 1.0),
                       meas_dx=float(peak[0]), meas_dy=float(peak[1]), meas_dz=int(dz))
        rows.append(row)
    return rows


def measure_channel(ch_id, ch_type, src_dir, geom, params):
    """Measure every EAST/SOUTH seam of one channel; returns a DataFrame."""
    names = grid_names(geom)
    rc_of = {name: rc for rc, name in names.items()}
    seams = adjacent_seams(geom)
    margin = params['edge_px']
    rects = {}
    for a, b, _ in seams:
        rect = _overlap(geom, rc_of[a], rc_of[b], margin)
        pad = params['win_px'] + 2
        wide = (rect[0] - pad, rect[1] + pad, rect[2] - pad, rect[3] + pad)
        rects.setdefault(a, []).append(wide)
        rects.setdefault(b, []).append(wide)
    bands = {name: load_tile_band(os.path.join(src_dir, f"{name}_{ch_id}_result.csv"),
                                  geom, rc_of[name], tile_rects)
             for name, tile_rects in rects.items()}
    r_dup = params['r_dup_tf'] if ch_type == 'tf' else params['r_dup_soma']
    rows = []
    for a, b, direction in seams:
        if bands.get(a) is None or bands.get(b) is None:
            continue
        rect = _overlap(geom, rc_of[a], rc_of[b], margin)
        for row in measure_seam(bands[a], bands[b], rect, r_dup, params):
            rows.append({'channel': ch_id, 'tile': a, 'nb_tile': b, 'direction': direction, **row})
    return pd.DataFrame(rows, columns=['channel', 'tile', 'nb_tile', 'direction', 'seg', 'n_a',
                                       'n_dup', 'contrast', 'meas_dx', 'meas_dy', 'meas_dz'])


def _valid(df, params):
    return (df['n_dup'] >= params['min_dup']) & (df['contrast'] >= params['min_contrast'])


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
                out.append({
                    'channel': ch, 'tile': tile, 'nb_tile': nb, 'direction': direction,
                    'seg': seg,
                    'n_a': int(meas['n_a']) if meas is not None else 0,
                    'n_dup': int(meas['n_dup']) if meas is not None else 0,
                    'contrast': float(meas['contrast']) if meas is not None else 0.0,
                    'meas_dx': float(meas['meas_dx']) if meas is not None else np.nan,
                    'meas_dy': float(meas['meas_dy']) if meas is not None else np.nan,
                    'meas_dz': float(meas['meas_dz']) if meas is not None else np.nan,
                    'dx': float(value[0]), 'dy': float(value[1]),
                    'dz': int(round(float(value[2]))), 'source': source})
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
        'version': 1,
        'frame': frame,
        'params': params,
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
    measured = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=['channel', 'tile', 'nb_tile', 'direction', 'seg', 'n_a', 'n_dup',
                 'contrast', 'meas_dx', 'meas_dy', 'meas_dz'])
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
