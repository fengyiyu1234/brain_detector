"""Stage 2.5 sampling windows.

"nonblank_center": the middle of the images that are not blank.
"soma_dense": inside the non-blank range, the `count` slices holding the most soma cells
(max over windows of the smallest per-soma-channel cell count). TF channels such as Sox9
fill the whole depth and pull an all-channel median away from where GFP/RFP cells are;
tsc18 325300_342500 had 85 soma cells in its old window and 2978 in the soma-dense one.

A slice counts as blank when its TIFF in any channel is smaller than ``max_bytes``
(acquisition fills empty planes with small, highly compressible images; ERH's are
131,782 bytes, real images about 8 MB). Slice numbers are 1-based positions in the
anchor tile's sorted TIFF list, the same numbering as the detection CSV ``z``.
"""
import os

import numpy as np

from src.utils.io import anchor_slice_key, resolve_tile_dir

WINDOW_FILE = '_sample_windows.json'
WINDOW_MODES = ('nonblank_center', 'soma_dense')


def _tiffs(folder):
    return sorted(f for f in os.listdir(folder)
                  if f.lower().endswith(('.tif', '.tiff')) and not f.startswith('.'))


def nonblank_indices(anchor_tile_dir, channel_roots, anchor_root, max_bytes):
    """1-based slice numbers whose TIFF is >= max_bytes in every channel.

    anchor_tile_dir: the anchor channel's tile folder (defines slice order).
    channel_roots: image roots of all channels (incl. second exposures).
    Returns (indices, n_slices). Raises when a channel's slice count differs.
    """
    names = _tiffs(anchor_tile_dir)
    keys = [os.path.splitext(f)[0] for f in names]
    tile = os.path.basename(anchor_tile_dir)
    rel = os.path.relpath(anchor_tile_dir, anchor_root)
    ok = [True] * len(keys)
    for root in channel_roots:
        folder = resolve_tile_dir(os.path.abspath(root), rel)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f'Missing channel tile folder: {folder}')
        files = _tiffs(folder)
        if len(files) != len(keys):
            raise ValueError(f'{folder} has {len(files)} slices, anchor {anchor_tile_dir} has {len(keys)}')
        ch_tile = os.path.basename(folder)
        sizes = {anchor_slice_key(os.path.splitext(f)[0], ch_tile, tile): os.stat(os.path.join(folder, f)).st_size
                 for f in files}
        for i, key in enumerate(keys):
            if sizes.get(key, 0) < max_bytes:
                ok[i] = False
    return [i + 1 for i, good in enumerate(ok) if good], len(keys)


def nonblank_center_window(indices, count):
    """(start, stop_exclusive, n_nonblank_in_window) or None without non-blank slices.

    Centred on the median non-blank slice, `count` consecutive slices, kept between
    the first and the last non-blank slice. With at most `count` non-blank slices the
    window runs from the first to the last of them, so all are used.
    """
    if not indices:
        return None
    first, last = indices[0], indices[-1]
    span = last - first + 1
    if span <= count or len(indices) <= count:
        start, stop = first, last + 1
    else:
        centre = indices[(len(indices) - 1) // 2]
        start = centre - count // 2
        start = max(first, min(start, last + 1 - count))
        stop = start + count
    n_in = sum(1 for i in indices if start <= i < stop)
    return start, stop, n_in


def tile_windows(tile_paths, anchor_root, channel_roots, count, max_bytes):
    """{tile: {'start', 'stop', 'n_nonblank', 'n_slices'} or {'start': None, ...}}."""
    out = {}
    for path in tile_paths:
        indices, n_slices = nonblank_indices(path, channel_roots, anchor_root, max_bytes)
        window = nonblank_center_window(indices, count)
        out[os.path.basename(path)] = {
            'start': window[0] if window else None,
            'stop': window[1] if window else None,
            'n_nonblank_in_window': window[2] if window else 0,
            'n_nonblank': len(indices), 'n_slices': n_slices,
            'first_nonblank': indices[0] if indices else None,
            'last_nonblank': indices[-1] if indices else None,
        }
    return out


def soma_dense_window(soma_centres, count, first, last):
    """(start, stop_exclusive, min_cells) of the densest soma window, or None.

    soma_centres: one sequence of cell z centres (1-based slices) per soma channel.
    The window has `count` slices inside [first, last]; with a shorter range it is the
    whole range. Score = smallest per-channel count of cells centred in the window; ties
    go to the window nearest the middle of the range. None when no window holds a cell
    of every soma channel (callers keep the nonblank_center window).
    """
    if first is None or last is None or not soma_centres:
        return None
    span = last - first + 1
    width = min(count, span)
    starts = np.arange(first, last - width + 2)
    counts = []
    for centres in soma_centres:
        z = np.sort(np.asarray(centres, dtype=float))
        counts.append(np.searchsorted(z, starts + width, side='left')
                      - np.searchsorted(z, starts, side='left'))
    score = np.min(counts, axis=0)
    if score.max() <= 0:
        return None
    middle = (first + last + 1 - width) / 2
    best = max(range(len(starts)), key=lambda i: (score[i], -abs(starts[i] - middle)))
    start = int(starts[best])
    return start, start + width, int(score[best])


def resolve_window_mode(config, settings):
    """Add sample_z_mode / blank_image_max_bytes to the resolved Stage 2.5 settings.

    Kept out of point_cloud_aligner.resolve_align_settings, which Stage 2.9 hashes.
    Only configs that set sample_z_mode get the keys, so older _align_settings.json
    files still match.
    """
    pa = config.get('pre_align_params', {})
    mode = pa.get('sample_z_mode')
    if mode is None:
        return settings
    if mode not in WINDOW_MODES:
        raise ValueError(f"pre_align_params.sample_z_mode={mode!r} must be one of {WINDOW_MODES}")
    if pa.get('sample_z_start_slice') is not None or pa.get('sample_z_start_fraction') is not None:
        raise ValueError(f'sample_z_mode {mode} excludes sample_z_start_slice/fraction')
    count = settings['sample_z_center_count']
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError('sample_z_count must be a positive integer')
    blank = pa.get('blank_image_max_bytes', 1_000_000)
    if isinstance(blank, bool) or not isinstance(blank, int) or blank < 0:
        raise ValueError('blank_image_max_bytes must be a non-negative integer')
    settings['sample_z_mode'] = mode
    settings['blank_image_max_bytes'] = blank
    settings.setdefault('sample_z_count', count)
    return settings
