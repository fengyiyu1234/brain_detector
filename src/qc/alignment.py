"""Channel alignment sensitivity QC, runnable with python -m src.qc.alignment."""
from __future__ import annotations
import argparse
import copy
import json
import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
import traceback
import numpy as np
import pandas as pd
import tifffile
from src.config.loader import load_config
from src.core.provenance import file_sha256, file_stamp, output_manifest_valid, write_output_manifest
from src.utils import validate_align_shifts as validation
from src.utils.alignment_intensity_qc import image_files
from .alignment_features import extract_features
from .alignment_flags import resolve_flag_options, write_flags
from .alignment_plots import curve_summary, overview, plot_curves
from .common import mapped_path, save_json, stamps, unchanged, worker_count
from .sampling import choose_region, cohort, prepare_tile, scan_footprint

DEFAULTS = {
    'enabled': False, 'reference_channel': None, 'soma_channels': None, 'tf_channels': None,
    'xy_range_um': 20, 'z_range_um': 40, 'workers': 4, 'fail_on_error': False,
    'output_dir': 'qc/alignment', 'formats': ['png', 'pdf'], 'path_mappings': {},
    'previous_landmarks': None, 'roi_manifest': None, 'flagging': {},
    'sampling': {'seed': 42, 'z_size_slices': 10, 'z_guard_slices': None,
                 'max_attempts': 50, 'min_pairs': 10},
}


def resolve_options(options, routing, settings):
    options = copy.deepcopy(options or {})
    unknown = set(options) - DEFAULTS.keys()
    if unknown:
        raise ValueError(f'Unknown alignment QC options: {sorted(unknown)}')
    out = {**copy.deepcopy(DEFAULTS), **options}
    if not isinstance(out['sampling'], dict) or set(out['sampling']) - DEFAULTS['sampling'].keys():
        raise ValueError('Unknown/invalid QC sampling options')
    out['sampling'] = {**DEFAULTS['sampling'], **out['sampling']}
    out['flagging'] = resolve_flag_options(out['flagging'])
    types = {ch['id']: ch.get('type', 'soma') for ch in routing}
    fixed = out['reference_channel'] or settings['reference_channel']
    if types.get(fixed) != 'soma':
        raise ValueError('QC reference_channel must be an active soma channel')
    out['reference_channel'] = fixed
    for key, kind in [('soma_channels', 'soma'), ('tf_channels', 'tf')]:
        values = out[key] if out[key] is not None else [cid for cid in types if types[cid] == kind and cid != fixed]
        if not isinstance(values, list) or any(not isinstance(cid, str) for cid in values):
            raise ValueError(f'{key} must be a list of channel IDs')
        if len(values) != len(set(values)) or fixed in values or any(types.get(cid) != kind for cid in values):
            raise ValueError(f'{key}: duplicate, inactive or wrong-type channel')
        out[key] = values
    if not out['soma_channels'] and not out['tf_channels']:
        raise ValueError('Alignment QC requires at least one moving soma or TF channel')
    for key in ('xy_range_um', 'z_range_um'):
        value = out[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be finite and positive')
    for key in ('xy_resolution_um', 'z_resolution_um'):
        if not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f'Invalid saved {key}')
    z_radius = out['z_range_um']/settings['z_resolution_um']
    if not np.isclose(z_radius, round(z_radius)):
        raise ValueError('z_range_um must be a whole number of source slices')
    for key, value in [('workers', out['workers']), *out['sampling'].items()]:
        if value is None and key == 'z_guard_slices':
            continue
        minimum = 0 if key in ('seed', 'z_guard_slices') else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'{key} must be an integer >= {minimum}')
    if not isinstance(out['formats'], list) or not out['formats'] or set(out['formats']) - {'png', 'pdf'}:
        raise ValueError('formats must contain png and/or pdf')
    if not isinstance(out['path_mappings'], dict):
        raise ValueError('path_mappings must be a dictionary')
    for key in ('enabled', 'fail_on_error'):
        if not isinstance(out[key], bool):
            raise ValueError(f'{key} must be boolean')
    return out


def scan_points(options, settings):
    radius = options['xy_range_um']/settings['xy_resolution_um']
    xy = sorted(set([-radius, radius, 0.] + list(range(-int(np.floor(radius)), int(np.floor(radius))+1))))
    z_radius = round(options['z_range_um']/settings['z_resolution_um'])
    return [(axis, delta) for axis in 'xy' for delta in xy] + [('z', delta) for delta in range(-z_radius, z_radius+1)]


def distance_curve(residual_um, axis, deltas_um):
    residual_um = np.asarray(residual_um, dtype=float).reshape(-1, 3)
    if not len(residual_um):
        return np.full(len(deltas_um), np.nan)
    direction = np.eye(3)[axis]
    return np.array([np.median(np.linalg.norm(residual_um + direction*delta, axis=-1)) for delta in deltas_um])


def landmark_rows(tile, pairs, points, settings, minimum):
    scale = np.array([settings['xy_resolution_um']]*2 + [settings['z_resolution_um']])
    rows = []
    for cid, records in pairs.items():
        residual = []
        for pair in records:
            soma, nucleus = np.asarray(pair['soma_points']), np.asarray(pair['nuclear_points'])
            # Historical experiments stored three centers; new runs store only bright centers.
            if soma.ndim == 2:
                soma, nucleus = soma[-1], nucleus[-1]
            if soma.shape != (3,) or nucleus.shape != (3,) or not np.isfinite([soma, nucleus]).all():
                raise ValueError('Invalid saved landmark coordinates')
            residual.append((nucleus-soma)*scale)
        for ai, axis in enumerate('xyz'):
            deltas = [d for a, d in points if a == axis]
            values = distance_curve(residual, ai, np.asarray(deltas)*scale[ai])
            status = 'no_pairs' if not residual else ('low_pairs' if len(residual) < minimum else 'diagnostic')
            rows.extend(dict(tile=tile, channel=cid, metric='bright_region_distance', axis=axis,
                             delta=d, delta_um=d*scale[ai], value=float(v), n_pairs=len(residual),
                             status=status, unaligned_iou=np.nan) for d, v in zip(deltas, values))
    return rows


def validate_previous(record, mappings):
    for name, expected in {**record.get('inputs', {}), **record.get('images', {})}.items():
        path = mapped_path(name, mappings)
        stat = path.stat()
        size, mtime = ((expected['size'], expected['mtime_ns']) if isinstance(expected, dict) else expected)
        if (stat.st_size, stat.st_mtime_ns) != (size, mtime):
            raise ValueError(f'Previous landmark source changed: {path}')
        if isinstance(expected, dict) and expected.get('sha256') and file_sha256(path) != expected['sha256']:
            raise ValueError(f'Previous landmark source content changed: {path}')


def source_slice_names(tile, params):
    path = Path(params['raw_dir'])/f"{tile}_{params['anchor_channel']}_inputs.json"
    if path.is_file():
        images = load_config(path).get('images', [])
        if images and all(row.get('slice_name') for row in images):
            return [row['slice_name'] for row in images]
    return validation.tile_slice_names(params['anchor_channel'], tile, params)


def extract_pairs(tile, params, region, cells, offsets, fit, guard):
    options = params['options']
    fixed, targets = options['reference_channel'], options['tf_channels']
    selected, candidates, counts = cohort(cells, offsets, region, fit, guard, params['tile_size'], fixed, targets)
    readers, opened = {}, {}
    anchor_names = source_slice_names(tile, params)
    for cid in [fixed, *targets]:
        if not targets:
            break
        files = image_files(params['channel_roots'][cid], tile)
        by_name = {p.stem: p for p in files}
        def make_reader(paths):
            @lru_cache(maxsize=24)
            def read(z):
                if not 1 <= z <= len(anchor_names) or anchor_names[z-1] not in paths:
                    raise ValueError(f'Missing source image for one-based slice {z}')
                path = paths[anchor_names[z-1]]
                before = file_stamp(path)
                opened.setdefault(str(path), before)
                array = tifffile.imread(path)
                if file_stamp(path) != before:
                    raise ValueError(f'Source image changed while reading: {path}')
                return array
            return read
        readers[cid] = make_reader(by_name)
    cache, results = {}, {}
    def feature(cid, index):
        key = (cid, index)
        if key not in cache:
            cache[key] = extract_features(selected[cid][index], readers[cid], cid != fixed)
        return cache[key]
    shifts = {cid: np.array([offsets[cid][a] for a in ('dx', 'dy', 'dz')]) for cid in selected}
    for cid in targets:
        records = []
        for si, ni in candidates[cid]:
            soma, nucleus = feature(fixed, si), feature(cid, ni)
            if soma is None or nucleus is None or not np.isfinite([soma, nucleus]).all():
                continue
            records.append(dict(soma_track=selected[fixed][si]['track_id'], nuclear_track=selected[cid][ni]['track_id'],
                                soma_points=(soma[-1]+shifts[fixed]).tolist(),
                                nuclear_points=(nucleus[-1]+shifts[cid]).tolist()))
        results[cid] = records
        counts[cid]['common_pairs'] = len(records)
    unchanged(opened)
    return results, counts, opened


def iou_rows(tile, params, cells, shifts, supports, fit, guard, region, points):
    options, settings = params['options'], params['settings']
    fixed, moving = options['reference_channel'], options['soma_channels']
    if not moving:
        return [], region
    footprint, starts, _ = scan_footprint(params, shifts, supports, fit, guard, [fixed, *moving])
    roi = [max(region[0], footprint[0]), min(region[1], footprint[1]),
           max(region[2], footprint[2]), min(region[3], footprint[3]), *region[4:]]
    if int(roi[4]) not in starts or roi[0] >= roi[1] or roi[2] >= roi[3]:
        raise ValueError(f'{tile}: landmark ROI cannot support held-out IoU scan')
    args = (1, settings['xy_resolution_um'], settings['z_resolution_um'])
    aligned = {cid: validation._shift_cells(cells[cid], *shifts[cid]) for cid in [fixed, *moving]}
    ref = validation._grid(aligned[fixed], roi, *args)
    rows = []
    for cid in moving:
        baseline = validation._grid(validation._shift_cells(cells[cid], *shifts[fixed]), roi, *args)
        unaligned = validation.grid_counts(ref, baseline)[2]
        halo = validation.z_halo_cells(aligned[cid], roi,
            round(options['z_range_um']/settings['z_resolution_um']), *args[1:])
        for axis, delta in points:
            extra = tuple(delta if axis == a else 0 for a in 'xyz')
            # Fixed raster dimensions also at fractional-pixel scan endpoints.
            grid = validation._grid(validation._shift_cells(halo, *extra), roi, *args)
            value = validation.grid_counts(ref, grid)[2]
            scale = settings['z_resolution_um' if axis == 'z' else 'xy_resolution_um']
            rows.append(dict(tile=tile, channel=cid, metric='voxel_iou', axis=axis,
                             delta=delta, delta_um=delta*scale, value=value, n_pairs=0,
                             unaligned_iou=unaligned, status='no_signal' if np.isnan(value) else 'diagnostic'))
    return rows, roi


def input_signature(tile, params, previous=None, saved_region=None):
    paths = [Path(params['filtered_dir'])/f"{tile}_{ch['id']}_result.csv" for ch in params['routing']]
    paths += [Path(params['raw_dir'])/f"{tile}_{ch['id']}_inputs.json" for ch in params['routing']]
    paths += [Path(params['align_dir'])/f'{tile}_{suffix}.json' for suffix in ('offsets', 'measured_offsets')]
    paths += [Path(params['align_dir'])/'_align_settings.json']
    # Flag thresholds and post-processing modules do not change per-tile records.
    report_only = {'alignment_flags.py', 'run_qc.py'}
    code = {str(p): file_sha256(p) for p in [*(q for q in Path(__file__).parent.glob('*.py') if q.name not in report_only),
        Path(validation.__file__), Path(__file__).parents[1]/'core/point_cloud_aligner.py',
        Path(__file__).parents[1]/'core/z_linker.py', Path(__file__).parents[1]/'utils/alignment_intensity_qc.py']}
    options = {k: v for k, v in params['options'].items() if k != 'flagging'}
    return dict(options=options, settings=params['settings'], inputs=stamps(paths), code=code,
                roots=params['channel_roots'], previous=previous, saved_region=saved_region,
                source_index={cid: validation.tile_slice_names(cid, tile, params)
                              for cid in {params['anchor_channel'], *params['options']['tf_channels'], params['options']['reference_channel']}})


def evaluate_tile(tile, params, output, previous=None, saved_region=None, force=False):
    output = Path(output)
    checkpoint = output/'tiles'/f'{tile}.json'
    manifest = output/'tiles'/f'{tile}_manifest.json'
    signature = input_signature(tile, params, previous, saved_region)
    if not force and output_manifest_valid(manifest, signature):
        record = load_config(checkpoint)
        try:
            unchanged(record['images'])
        except (ValueError, OSError):
            pass
        else:
            return record
    cells, offsets, shifts, supports, fit, guard = prepare_tile(tile, params)
    options = params['options']
    if previous is not None:
        validate_previous(previous, options['path_mappings'])
        if previous['offsets'] != offsets:
            raise ValueError(f'{tile}: previous landmarks use different offsets')
        region = previous.get('landmark_region', previous['region'])
        pairs = {cid: previous['pairs'][cid] for cid in options['tf_channels']}
        counts = {cid: previous['counts'][cid] for cid in options['tf_channels']}
        images = {str(mapped_path(p, options['path_mappings'])): file_stamp(mapped_path(p, options['path_mappings']))
                  for p in previous.get('images', {})}
    else:
        region = choose_region(tile, params, cells, offsets, shifts, supports, fit, guard, saved_region)
        pairs, counts, images = extract_pairs(tile, params, region, cells, offsets, fit, guard)
    points = scan_points(options, params['settings'])
    rows, iou_region = iou_rows(tile, params, cells, shifts, supports, fit, guard, region, points)
    rows += landmark_rows(tile, pairs, points, params['settings'], options['sampling']['min_pairs'])
    unchanged(signature['inputs'])
    unchanged(images)
    if previous is not None:
        validate_previous(previous, options['path_mappings'])
    record = dict(tile=tile, region=iou_region, landmark_region=region, offsets=offsets, fit_window=fit,
                  actual_z_guard_slices=guard, pairs=pairs, counts=counts, images=images,
                  inputs=signature['inputs'], curves=rows)
    save_json(checkpoint, record)
    frame = pd.DataFrame(rows)
    plot_base = output/'tile_plots'/f'{tile}_alignment_sensitivity'
    plot_base.parent.mkdir(parents=True, exist_ok=True)
    plot_curves(frame, plot_base, f'{params["sample_name"]} | {tile} | saved offsets | held-out Z {region[4:]}',
                params['settings'], options['formats'])
    write_output_manifest(manifest, [checkpoint, *[Path(f'{plot_base}.{fmt}') for fmt in options['formats']]], signature)
    return record


def run(sample, config_path=None, options=None, tiles=None, force=False, config=None):
    results = Path(validation.resolve_results_dir(sample))
    saved_config, routing, settings, align_dir, source = validation.load_settings(str(results), config_path)
    # Pipeline supplies the current run's configuration. Standalone always prefers saved settings.
    saved_config = config if config is not None else saved_config
    options = resolve_options(options, routing, settings)
    if saved_config.get('detection_params', {}).get('DOWNSAMPLE', False):
        raise ValueError('Alignment QC requires contiguous, non-downsampled detections')
    all_tiles = validation.list_tiles(align_dir)
    tiles = all_tiles if tiles is None else tiles
    if not tiles or len(tiles) != len(set(tiles)) or set(tiles)-set(all_tiles):
        raise ValueError('Requested tiles must be distinct, nonempty and have saved offsets')
    output = mapped_path(options['output_dir'], options['path_mappings'])
    if not output.is_absolute():
        output = results/output
    # Never permit QC to overwrite pipeline products or historical T70 results.
    output = output.resolve()
    protected = [results.resolve(), *[p.resolve() for p in results.iterdir()
                 if p.is_dir() and (p.name[:1].isdigit() or p.name.startswith('alignment_qc'))]]
    if any(output == p or p in output.parents for p in protected[1:]) or output == protected[0] or output in results.resolve().parents:
        raise ValueError('QC output_dir overlaps protected pipeline/historical products')
    output.mkdir(parents=True, exist_ok=True)
    roots = {}
    for ch in routing:
        value = saved_config.get('paths', {}).get(ch.get('dir_key'))
        if value:
            path = mapped_path(value, options['path_mappings'])
            if not path.is_dir():
                fallback = results.parent/str(value).replace('\\', '/').rstrip('/').split('/')[-1]
                path = fallback if fallback.is_dir() else path
            roots[ch['id']] = str(path)
    params = dict(settings=settings, options=options, routing=routing, align_dir=align_dir,
        filtered_dir=str(results/'2_2d_filtered'), raw_dir=str(results/'1_2d_raw'),
        tile_size=saved_config['detection_params'].get('tILESIZE', 2048), channel_roots=roots,
        slice_files={}, anchor_channel=routing[0]['id'], candidate_tiles=tiles, sample_name=results.parent.name)
    previous, regions = {}, {}
    if options['previous_landmarks']:
        history = load_config(mapped_path(options['previous_landmarks'], options['path_mappings']))
        if history['status'] != 'complete' or history['settings'] != settings:
            raise ValueError('Previous landmark run is incomplete or uses different alignment settings')
        previous = {r['tile']: r for r in history['completed']}
        if set(tiles)-previous.keys():
            raise ValueError('Previous landmarks do not cover all requested tiles')
    if options['roi_manifest']:
        manifest = load_config(mapped_path(options['roi_manifest'], options['path_mappings']))
        regions = {r['tile']: [r[k] for k in ('x0','x1','y0','y1','z0','z1')] for r in manifest['regions']}
        if set(tiles)-regions.keys():
            raise ValueError('Saved ROI manifest does not cover requested tiles')
    report = dict(status='running', requested=tiles, completed=[], failures={}, settings=settings,
                  options=options, settings_source=source,
                  method=dict(pairing='unique baseline containment, frozen across shifts',
                              bright_center='P10-background-subtracted intensity centroid of upper-decile component containing peak',
                              aggregation='per-tile median 3D physical distance; overview uses equal tile weight',
                              interpretation='sensitivity diagnostic; alignment-based pair selection can favor zero'))
    save_json(output/'run_details.json', report)
    count = worker_count(options['workers'], len(tiles))
    thread_names = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
    old_env = {k: os.environ.get(k) for k in thread_names}
    try:
        os.environ.update({k:'1' for k in thread_names})
        def collect(tile, future=None):
            try:
                record = future.result() if future is not None else evaluate_tile(
                    tile, params, str(output), previous.get(tile), regions.get(tile), force)
                report['completed'].append(record)
                print(f'Alignment QC [{len(report["completed"])}/{len(tiles)}]: {tile}', flush=True)
            except Exception:
                report['failures'][tile] = traceback.format_exc()
                print(f'Alignment QC FAILED: {tile}: {report["failures"][tile]}', flush=True)
            save_json(output/'run_details.json', report)
        if count == 1:
            for tile in tiles:
                collect(tile)
        else:
            with ProcessPoolExecutor(max_workers=count, mp_context=mp.get_context('spawn')) as pool:
                futures = {pool.submit(evaluate_tile, tile, params, str(output), previous.get(tile), regions.get(tile), force): tile for tile in tiles}
                for future in as_completed(futures):
                    collect(futures[future], future)
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    report['completed'].sort(key=lambda r: r['tile'])
    report['status'] = 'failed' if report['failures'] else 'complete'
    rows = [row for record in report['completed'] for row in record['curves']]
    if rows:
        frame = pd.DataFrame(rows)
        frame.to_csv(output/'curves.csv', index=False)
        pd.DataFrame(curve_summary(frame)).to_csv(output/'summary.csv', index=False)
        overview(frame, output/'alignment_overview', params['sample_name'], settings, options['formats'])
    flagged, flag_lines = write_flags(output, report)
    report['flagged'] = flagged.loc[flagged.verdict.isin(['bad', 'error'])].to_dict('records')
    report['insufficient'] = flagged.loc[flagged.verdict.eq('insufficient'), 'tile'].tolist()
    save_json(output/'sampling.json', dict(regions=[dict(tile=r['tile'], **dict(zip(('x0','x1','y0','y1','z0','z1'), r['landmark_region']))) for r in report['completed']]))
    save_json(output/'run_details.json', report)
    lines = ['# Alignment sensitivity QC', '', f'Status: {report["status"]}; {len(report["completed"])}/{len(tiles)} tiles.', '',
             'RFP/other soma: voxel IoU. TF: bright-region median 3D distance in um.',
             'Pairs remain fixed across shifts. Pair selection uses saved alignment and can favor zero.',
             'Sparse/empty pairs are flagged in summary.csv; a completed computation is not an accuracy PASS.',
             '', '[Summary](summary.csv) | [Curves](curves.csv) | [Run details](run_details.json)', '', *flag_lines]
    (output/'qc_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', required=True, help='Result directory containing 2_2d_filtered and 3_2d_aligned')
    parser.add_argument('--config', help='Original inference config, fallback when runtime_config.json is absent')
    parser.add_argument('--qc-config', help='QC options JSON or inference config containing qc.alignment')
    parser.add_argument('--tiles', help='Comma-separated tile IDs')
    parser.add_argument('--workers', type=int)
    parser.add_argument('--out-dir')
    parser.add_argument('--previous-landmarks', help='Import frozen pairs from a prior run_details.json')
    parser.add_argument('--roi-manifest')
    parser.add_argument('--force', action='store_true', help='Recompute QC checkpoints')
    args = parser.parse_args(argv)
    if args.qc_config:
        raw = load_config(args.qc_config)
        options = raw.get('qc', {}).get('alignment', raw)
    else:
        snapshot = Path(args.sample)/'runtime_config.json'
        raw = load_config(snapshot if snapshot.is_file() else args.config) if snapshot.is_file() or args.config else {}
        options = raw.get('qc', {}).get('alignment', {})
    options = copy.deepcopy(options)
    for key in ('workers', 'previous_landmarks', 'roi_manifest'):
        value = getattr(args, key)
        if value is not None:
            options[key] = value
    if args.out_dir:
        options['output_dir'] = args.out_dir
    tiles = [t.strip() for t in args.tiles.split(',')] if args.tiles else None
    report = run(args.sample, args.config, options, tiles, args.force)
    return 1 if report['status'] == 'failed' else 0


if __name__ == '__main__':
    mp.freeze_support()
    raise SystemExit(main())
