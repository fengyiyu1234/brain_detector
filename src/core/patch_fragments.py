"""Drop YOLO boxes cut off by an inner sliding-window patch edge when another box covers them.

YOLO runs on xsize x ysize patches with stride `step`, and the per-tile NMS keeps
only boxes with IoU <= 0.4 to each other. A cell lying where patches overlap can
then keep two boxes on one slice: one patch sees the whole cell, while a neighbouring
patch sees only the part on its side of the patch edge, and the box is clipped there.
A full box and a fragment often have IoU < 0.4, so both survive. The same slice then
has two boxes for one cell, and the z-linker splits the cell into two 3D tracks.

Rule (per slice, all classes together, like the containment NMS): a box whose edge
lies on an inner edge of its own patch (within EDGE_TOL px; tile borders do not count)
is dropped when another kept box at least as large covers more than `iomin` of its
area. Boxes are checked smallest first. The patch of each box comes from
{tile}_{channel}_candidates.csv in 1_2d_raw (for fused rows, from winner_detection_id).

Enabled per model with detection_params.<model>.patch_fragment_iomin (null or absent:
off, so existing checkpoints are unchanged). The settings, this file's hash and the
candidate CSVs enter the filter checkpoint signature only when the rule is enabled.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.provenance import file_sha256, file_stamp

PATCH_FRAGMENT_KEY = 'patch_fragment_iomin'
EDGE_TOL = 2.0
REASON = 'patch_fragment'


def resolve_patch_fragment(config, channel):
    """Settings dict for a channel spec, or None when the rule is off."""
    model = str(channel.get('model', '')).lower()
    dp = config.get('detection_params', {})
    threshold = (dp.get(model) or {}).get(PATCH_FRAGMENT_KEY)
    if threshold is None:
        return None
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not 0 < threshold <= 1):
        raise ValueError(f'detection_params.{model}.{PATCH_FRAGMENT_KEY} must be in (0, 1] or null')
    if model != 'yolo':
        raise ValueError(f'{PATCH_FRAGMENT_KEY} only applies to YOLO channels (got {model!r})')
    missing = [key for key in ('xsize', 'ysize', 'step', 'tILESIZE') if dp.get(key) is None]
    if missing:
        raise ValueError(f'{PATCH_FRAGMENT_KEY} needs detection_params {missing}')
    return {'iomin': float(threshold), 'edge_tol': EDGE_TOL,
            'xsize': int(dp['xsize']), 'ysize': int(dp['ysize']),
            'step': int(dp['step']), 'tile_size': int(dp['tILESIZE'])}


def candidate_paths(raw_dir, tile, channel_ids):
    paths = [Path(raw_dir) / f'{tile}_{cid}_candidates.csv' for cid in channel_ids]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f'{PATCH_FRAGMENT_KEY} needs YOLO candidate CSVs; missing {missing}')
    return paths


def signature(settings, paths):
    return {'settings': settings,
            'code_sha256': file_sha256(os.path.abspath(__file__)),
            'candidates': [file_stamp(p, with_hash=True) for p in paths]}


def _padded(size, patch, step):
    # Same padding as the worker's sliding window.
    return size if (size - patch) % step == 0 else size - size % step + patch


def patch_cut(df, origins, settings):
    """Boolean array: the box touches an inner edge of the patch it was detected in."""
    ids = df['detection_id'].astype(str)
    px = ids.map(origins['patch_x'])
    py = ids.map(origins['patch_y'])
    if 'winner_detection_id' in df:
        winner = df['winner_detection_id'].astype(str)
        px = px.fillna(winner.map(origins['patch_x']))
        py = py.fillna(winner.map(origins['patch_y']))
    known = px.notna().to_numpy() & py.notna().to_numpy()
    px = px.to_numpy(float); py = py.to_numpy(float)
    size, tol = settings['tile_size'], settings['edge_tol']
    right = np.minimum(px + settings['xsize'], _padded(size, settings['xsize'], settings['step']))
    bottom = np.minimum(py + settings['ysize'], _padded(size, settings['ysize'], settings['step']))
    x1, y1 = df.x1.to_numpy(float), df.y1.to_numpy(float)
    x2, y2 = df.x2.to_numpy(float), df.y2.to_numpy(float)
    with np.errstate(invalid='ignore'):
        cut = (((px > 0) & (np.abs(x1 - px) <= tol))
               | ((py > 0) & (np.abs(y1 - py) <= tol))
               | ((right < size) & (np.abs(x2 - right) <= tol))
               | ((bottom < size) & (np.abs(y2 - bottom) <= tol)))
    return cut & known


def load_origins(paths):
    parts = [pd.read_csv(p, usecols=['detection_id', 'patch_x', 'patch_y', 'decision'])
             for p in paths]
    origins = pd.concat(parts, ignore_index=True)
    origins = origins[(origins.decision == 'kept') & origins.detection_id.notna()]
    return origins.drop_duplicates('detection_id').set_index(
        origins.detection_id.astype(str))[['patch_x', 'patch_y']]


def fragment_keep(df, cut, threshold):
    """(keep, winner_ids, iomin) with the rule from the module docstring."""
    n = len(df)
    keep = np.ones(n, dtype=bool)
    winners = np.full(n, '', dtype=object)
    iomin = np.full(n, np.nan)
    if n == 0 or not cut.any():
        return keep, winners, iomin
    x1, y1 = df.x1.to_numpy(float), df.y1.to_numpy(float)
    x2, y2 = df.x2.to_numpy(float), df.y2.to_numpy(float)
    area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    ids = df['detection_id'].astype(str).to_numpy() if 'detection_id' in df else np.full(n, '')
    for positions in df.groupby('z', sort=False).indices.values():
        positions = np.asarray(positions)
        if not cut[positions].any() or len(positions) < 2:
            continue
        for b in positions[np.argsort(area[positions], kind='stable')]:
            if not cut[b] or area[b] <= 0:
                continue
            others = positions[keep[positions] & (positions != b) & (area[positions] >= area[b])]
            if not len(others):
                continue
            inter = (np.maximum(0, np.minimum(x2[b], x2[others]) - np.maximum(x1[b], x1[others]))
                     * np.maximum(0, np.minimum(y2[b], y2[others]) - np.maximum(y1[b], y1[others])))
            cover = inter / area[b]
            best = int(np.argmax(cover))
            if cover[best] > threshold:
                keep[b] = False
                winners[b] = ids[others[best]]
                iomin[b] = float(cover[best])
    return keep, winners, iomin


def suppress_patch_fragments(filtered, rejected, stats, paths, settings):
    """Apply the rule to a filter result; returns (filtered, rejected, stats) like the SNR filter."""
    cut = patch_cut(filtered, load_origins(paths), settings)
    keep, winners, iomin = fragment_keep(filtered, cut, settings['iomin'])
    dropped = filtered.loc[~keep].copy()
    if len(dropped):
        dropped['rejection_reason'] = REASON
        dropped['suppressed_by_detection_id'] = winners[~keep]
        dropped['suppression_iomin'] = iomin[~keep]
        dropped['suppression_threshold'] = settings['iomin']
        rejected = (dropped.reset_index(drop=True) if rejected.empty
                    else pd.concat([rejected, dropped], ignore_index=True))
    filtered = filtered.loc[keep].copy()
    stats = {**stats, 'removed': {**stats.get('removed', {}), REASON: int((~keep).sum())},
             'patch_cut_kept_before': int(cut.sum())}
    stats['after'] = len(filtered)
    stats['removed_total'] = stats['before'] - stats['after']
    return filtered, rejected, stats
