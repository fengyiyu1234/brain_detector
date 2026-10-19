"""Deterministic held-out sampling with fixed baseline containment pairs."""
import math
import zlib
from pathlib import Path
import numpy as np
from src.config.loader import load_config
from src.utils import validate_align_shifts as validation
from src.core.point_cloud_aligner import _effective_z_extent
from .alignment_features import eligible, fixed_pairs


def prepare_tile(tile, params):
    settings, options = params['settings'], params['options']
    offsets = load_config(Path(params['align_dir']) / f'{tile}_offsets.json')
    channels = [options['reference_channel'], *options['soma_channels'], *options['tf_channels']]
    for cid in channels:
        if cid not in offsets or any(isinstance(offsets[cid][a], bool) or
                not isinstance(offsets[cid][a], (float, int)) or
                not math.isfinite(offsets[cid][a]) or int(offsets[cid][a]) != offsets[cid][a]
                for a in ('dx', 'dy', 'dz')):
            raise ValueError(f'{tile}/{cid}: missing or invalid saved offsets')
    frame = settings.get('stitching_reference_channel', settings['reference_channel'])
    if frame not in offsets or any(offsets[frame][a] != 0 for a in ('dx', 'dy', 'dz')):
        raise ValueError(f'{tile}: saved offsets do not match frame {frame}')
    cells, center = validation.build_tile_vol_lists(params['filtered_dir'], tile, params['routing'], settings)
    measured_path = Path(params['align_dir']) / f'{tile}_measured_offsets.json'
    measured = load_config(measured_path) if measured_path.is_file() else {}
    if measured.get('sample_window'):
        window = measured['sample_window']
        fit = (window['z_start'], window['z_stop_exclusive'])
    elif settings.get('sample_z_start_slice') is not None:
        start = settings['sample_z_start_slice']
        fit = (start, start + settings['sample_z_count'])
    else:
        center = measured.get('z_center', center)
        if center is None or not math.isfinite(center):
            raise ValueError(f'{tile}: no saved or reconstructible alignment window')
        half = settings['sample_z_center_count'] // 2
        fit = (center - half, center + half)
    guard = options['sampling']['z_guard_slices']
    if guard is None:
        guard = max(v['max_cell_z_span'] for v in settings['z_link'].values())
    raster_guard = 0
    for cid in channels:
        for cell in cells[cid]:
            if cell['z_min'] <= fit[1] and cell['z_max'] >= fit[0]:
                lo, hi = _effective_z_extent(cell, settings['xy_resolution_um'], settings['z_resolution_um'])
                raster_guard = max(raster_guard, math.ceil(max(cell['z_min']-lo, hi-cell['z_max'], 0)))
    guard += raster_guard
    shifts = {cid: tuple(offsets[cid][a] for a in ('dx', 'dy', 'dz')) for cid in channels}
    supports = {cid: validation.load_slice_support(tile, cid, params)[0] for cid in channels}
    return cells, offsets, shifts, supports, fit, guard


def scan_footprint(params, shifts, supports, fit, guard, channels=None):
    options, settings = params['options'], params['settings']
    fixed = options['reference_channel']
    radii = [math.ceil(options['xy_range_um']/settings['xy_resolution_um'])]*2
    radii += [round(options['z_range_um']/settings['z_resolution_um'])]
    wanted = channels or list(shifts)
    limits = {cid: validation._shift_limits(shifts[cid], radii,
              cid in options['soma_channels'], True, shifts[fixed]) for cid in wanted}
    return validation.valid_region_starts(
        {cid: supports[cid] for cid in wanted}, limits, params['tile_size'],
        options['sampling']['z_size_slices'], fit, guard, True)


def cohort(cells, offsets, region, fit, guard, tile_size, fixed, targets):
    selected = {cid: [cell for cell in cells[cid]
        if eligible(cell, offsets[cid], region, fit, guard, [0, 0, 0], tile_size)]
        for cid in [fixed, *targets]}
    shift = {cid: np.array([offsets[cid][a] for a in ('dx', 'dy', 'dz')]) for cid in selected}
    pairs, counts = {}, {}
    for cid in targets:
        pairs[cid], edges = fixed_pairs(selected[fixed], selected[cid], shift[fixed], shift[cid])
        counts[cid] = dict(eligible_nuclei=len(selected[cid]), candidate_edges=edges,
                           unique_pairs=len(pairs[cid]))
    return selected, pairs, counts


def choose_region(tile, params, cells, offsets, shifts, supports, fit, guard, saved=None):
    footprint, starts, _ = scan_footprint(params, shifts, supports, fit, guard)
    sample = params['options']['sampling']
    if not starts:
        raise ValueError(f'{tile}: no valid held-out Z window')
    if saved is not None:
        region = saved
        if int(region[4]) not in starts or region[5]-region[4] != sample['z_size_slices']:
            raise ValueError(f'{tile}: saved ROI no longer supports the scan/held-out settings')
        if not (footprint[0] <= region[0] < region[1] <= footprint[1] and
                footprint[2] <= region[2] < region[3] <= footprint[3]):
            raise ValueError(f'{tile}: saved ROI is outside the valid footprint')
        return list(region)
    rng = np.random.default_rng(sample['seed'] + zlib.crc32(tile.encode()))
    best, best_count = None, -1
    fixed, targets = params['options']['reference_channel'], params['options']['tf_channels']
    for start in rng.permutation(starts)[:sample['max_attempts']]:
        region = [*footprint, int(start), int(start)+sample['z_size_slices']]
        _, pairs, _ = cohort(cells, offsets, region, fit, guard, params['tile_size'], fixed, targets)
        count = min((len(pairs[cid]) for cid in targets), default=sample['min_pairs'])
        if count > best_count:
            best, best_count = region, count
        if count >= sample['min_pairs']:
            break
    return best
