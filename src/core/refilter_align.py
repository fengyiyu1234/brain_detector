"""CPU-only raw refiltering and channel alignment; stops before global stitching.

Example:
  python -m src.core.refilter_align --config config/EGFR/2/config_EGFR_t70.json \
      --local-root Y:/Fengyi/EGFR_brain/T70 --workers 4

Raw CSVs are read in place. Unselected channels require compatible prefiltered
inputs. No models, instance polygons, fluorescence intensities or colocalization
are used. Completed jobs can be resumed with the same arguments.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import time

from src.config.loader import load_config
from src.core.detection_filter import (
    FILTER_SCHEMA_VERSION, filter_detection_df, resolve_filter_params,
)
from src.core.point_cloud_aligner import (
    align_tile, alignment_sample_window, check_align_settings,
    resolve_align_settings, save_align_settings, tile_alignment_done,
    validate_alignment_frame,
)
from src.core.provenance import (
    atomic_json, file_sha256, file_stamp, output_manifest_valid,
    provenance_columns, write_output_manifest,
)


def local_config(config, local_root=None, results_dir=None):
    config = copy.deepcopy(config)
    if local_root:
        root = Path(local_root).resolve()
        for key, value in config['paths'].items():
            if key.endswith('_dir') and value:
                config['paths'][key] = str(root / Path(value).name)
        config['paths']['pATHRESULT'] = str(root / Path(config['paths']['pATHRESULT']).name)
    if results_dir:
        config['paths']['pATHRESULT'] = str(Path(results_dir).resolve())
    return config


def atomic_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + '.part')
    # Large buffers avoid thousands of small writes to network shares.
    with part.open('w', encoding='utf-8', newline='', buffering=8 * 1024 * 1024) as handle:
        df.to_csv(handle, index=False)
    os.replace(part, path)


def filter_one(job):
    import pandas as pd
    source, target, params, signature_settings, tile, cid = job
    source, target = Path(source), Path(target)
    rejected = target.with_name(target.name.replace('_result.csv', '_rejected.csv'))
    manifest = target.with_name(target.name.replace('_result.csv', '_filter_manifest.json'))
    signature = {
        'source': file_stamp(source, with_hash=True),
        'settings': {'stage': 'pre_align_filter', 'params': params,
                     'filter_settings': signature_settings},
    }
    raw_manifest = source.with_name(source.name.replace('_result.csv', '_inputs.json'))
    if raw_manifest.is_file():
        raw_provenance = load_config(raw_manifest)
        expected = raw_provenance.get('output_sha256')
        if expected and expected != signature['source']['sha256']:
            raise ValueError(f'Raw CSV differs from original inference fingerprint: {source}')
    if output_manifest_valid(manifest, signature):
        saved = load_config(manifest)
        return {'tile': tile, 'channel': cid, 'cached': True, **saved.get('metadata', {})}
    raw = provenance_columns(pd.read_csv(source), tile, cid)
    filtered, stats, discarded = filter_detection_df(
        raw, params, return_stats=True, return_rejected=True, context=str(source))
    atomic_csv(filtered, target)
    atomic_csv(discarded, rejected)
    write_output_manifest(manifest, [target, rejected], signature, metadata=stats)
    return {'tile': tile, 'channel': cid, 'cached': False, **stats}



def validate_filtered_one(job):
    """Reuse completed filters across Windows/HPC paths without rewriting CSVs."""
    source, target, params, signature_settings, tile, cid = job
    source, target = Path(source), Path(target)
    manifest = target.with_name(target.name.replace('_result.csv', '_filter_manifest.json'))
    saved = load_config(manifest)
    signature = saved['input_signature']
    if (signature['settings'].get('stage') != 'pre_align_filter'
            or signature['settings'].get('params') != params
            or signature['settings'].get('filter_settings') != signature_settings):
        raise ValueError(f'Filtered input has different parameters/code: {target}')
    current = file_stamp(source, with_hash=True)
    if (current['size'] != signature['source']['size']
            or current['sha256'] != signature['source']['sha256']):
        raise ValueError(f'Raw source changed since filtering: {source}')
    raw_manifest = source.with_name(source.name.replace('_result.csv', '_inputs.json'))
    if raw_manifest.is_file():
        original = load_config(raw_manifest).get('output_sha256')
        if original and original != current['sha256']:
            raise ValueError(f'Raw CSV differs from original inference fingerprint: {source}')
    outputs = {name.replace('\\', '/').rsplit('/', 1)[-1]: stamp
               for name, stamp in saved['outputs'].items()}
    expected = {target.name, target.name.replace('_result.csv', '_rejected.csv')}
    if set(outputs) != expected:
        raise ValueError(f'Incomplete filtered output manifest: {manifest}')
    for name, stamp in outputs.items():
        actual = file_stamp(target.parent / name, with_hash=True)
        if actual['size'] != stamp['size'] or actual['sha256'] != stamp['sha256']:
            raise ValueError(f'Filtered output was modified: {target.parent / name}')
    return {'tile': tile, 'channel': cid, 'cached': True, **saved.get('metadata', {})}


def run_jobs(function, jobs, workers, label):
    records = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
        futures = {pool.submit(function, job): job for job in jobs}
        try:
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                if label in ('filter', 'validate'):
                    logging.info('[%s %s/%s] %s %s: %s -> %s%s',
                                 label, len(records), len(jobs), record['tile'], record['channel'],
                                 record.get('before'), record.get('after'),
                                 ' (cached)' if record['cached'] else '')
                else:
                    logging.info('[align %s/%s] %s completed', len(records), len(jobs), record['tile'])
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return records


def align_one(job):
    tile_path, filtered_dir, align_dir, routing, settings = job
    missing = align_tile(tile_path, filtered_dir, align_dir, routing, settings)
    if missing:
        raise FileNotFoundError(str(missing))
    tile = Path(tile_path).name
    measured = load_config(Path(align_dir) / f'{tile}_measured_offsets.json')
    return {'tile': tile, **measured}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--local-root', help='Map channel/result paths to a local sample directory')
    parser.add_argument('--results-dir', help='Output root override; raw can remain elsewhere')
    parser.add_argument('--raw-dir', help='Existing raw CSV directory; read only')
    parser.add_argument('--filter-channels', nargs='+', default=['Sox9', 'Olig2'])
    parser.add_argument('--tiles', nargs='+', help='Optional subset for a pilot run')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--dry-run', action='store_true')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--filter-only', action='store_true')
    modes.add_argument('--align-only', action='store_true',
                       help='Verify and reuse completed filters; never refilter or write detection CSVs')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    started = time.time()
    if args.workers < 1:
        raise ValueError('--workers must be positive')
    config = local_config(load_config(args.config), args.local_root, args.results_dir)
    if config.get('pipeline_mode') != 'pre_align':
        raise ValueError('This entry point requires pipeline_mode=pre_align')
    routing = [ch for ch in config['channels_routing'] if ch.get('active', True)]
    if any(ch.get('double_exposure') for ch in routing):
        raise ValueError('Dual-exposure fusion requires the full pipeline')
    channels = {ch['id']: ch for ch in routing}
    if set(args.filter_channels) - channels.keys():
        raise ValueError('Unknown --filter-channels')
    base = Path(config['paths']['pATHRESULT'])
    raw = Path(args.raw_dir).resolve() if args.raw_dir else base / '1_2d_raw'
    filtered_dir, align_dir = base / '2_2d_filtered', base / '3_2d_aligned'
    if raw.resolve() in (filtered_dir.resolve(), align_dir.resolve()):
        raise ValueError('Raw CSVs must be separate from output directories')
    settings = resolve_align_settings(config, routing)
    params = {cid: resolve_filter_params(config, ch) for cid, ch in channels.items()}
    settings.update(input_stage='filtered_2d_v1', filter_params=params)
    check_align_settings(str(align_dir), settings)
    first = routing[0]['id']
    suffix = f'_{first}_result.csv'
    available = sorted(p.name[:-len(suffix)] for p in raw.glob(f'*{suffix}'))
    tiles = args.tiles or available
    if not tiles or len(tiles) != len(set(tiles)) or set(tiles) - set(available):
        raise ValueError('No tiles, duplicate tiles, or missing raw tiles')
    ref = channels[settings['reference_channel']]
    ref_root = Path(config['paths'][ref['dir_key']])
    paths = {p.name: p for row in ref_root.iterdir() if row.is_dir()
             for p in row.iterdir() if p.is_dir()}
    old_filter_settings = load_config(filtered_dir / '_filter_settings.json') if (
        filtered_dir / '_filter_settings.json').is_file() else {}
    reused = set(channels) - set(args.filter_channels)
    if any(old_filter_settings.get('params', {}).get(cid) != params[cid] for cid in reused):
        raise ValueError('Unselected channel filter settings differ; include those channels explicitly')
    windows = []
    for tile in tiles:
        for cid in channels:
            if not (raw / f'{tile}_{cid}_result.csv').is_file():
                raise FileNotFoundError(f'Missing raw CSV: {tile} {cid}')
            if cid in reused and not (filtered_dir / f'{tile}_{cid}_result.csv').is_file():
                raise FileNotFoundError(f'Missing reusable filtered CSV: {tile} {cid}')
        if tile not in paths:
            raise FileNotFoundError(f'Missing reference image tile: {tile}')
        names = sorted(p.stem for p in paths[tile].iterdir()
                       if p.suffix.lower() in ('.tif', '.tiff') and not p.name.startswith('.'))
        window = alignment_sample_window(len(names), settings)
        windows.append({'tile': tile, 'total_slices': len(names),
                        'z_start': window[0] if window else None,
                        'z_stop_exclusive': window[1] if window else None})
    print(json.dumps({'results': str(base), 'raw': str(raw), 'tiles': len(tiles),
                      'filter_channels': args.filter_channels, 'reused_channels': sorted(reused),
                      'workers': args.workers, 'filter_params': params,
                      'sampling': windows}, indent=2), flush=True)
    if args.dry_run:
        return 0
    base.mkdir(parents=True, exist_ok=True)
    filtered_dir.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(
                            base / 'refilter_align.log', encoding='utf-8')], force=True)
    signature_settings = {
        'schema_version': FILTER_SCHEMA_VERSION, 'params': params,
        'filter_code_sha256': file_sha256(Path(__file__).with_name('detection_filter.py')),
    }
    # Limit numerical libraries before spawned workers import NumPy/SciPy.
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    jobs = [(str(raw / f'{tile}_{cid}_result.csv'),
             str(filtered_dir / f'{tile}_{cid}_result.csv'), params[cid],
             signature_settings, tile, cid) for tile in tiles for cid in args.filter_channels]
    function = validate_filtered_one if args.align_only else filter_one
    records = run_jobs(function, jobs, min(args.workers, len(jobs)),
                       'validate' if args.align_only else 'filter')
    atomic_json(filtered_dir / '_refilter_alignment_manifest.json', {
        'config': str(Path(args.config).resolve()), 'raw_dir': str(raw),
        'reused_channels': sorted(reused), 'records': records,
    })
    if set(tiles) == set(available):
        atomic_json(filtered_dir / '_filter_settings.json', signature_settings)
    if args.filter_only:
        return 0
    align_dir.mkdir(exist_ok=True)
    save_align_settings(str(align_dir), settings)
    config['restart_provenance'] = {
        'mode': 'alignment_only' if args.align_only else 'raw_refilter_and_alignment_only',
        'filter_executed': not args.align_only, 'source_config': str(Path(args.config).resolve()),
        'raw_dir': str(raw), 'filter_channels': args.filter_channels,
        'reused_channels': sorted(reused), 'intensity_used_for_alignment': False,
    }
    atomic_json(base / 'alignment_runtime_config.json', config)
    # Leave the inference runtime_config.json intact, including model provenance.
    todo = [tile for tile in tiles if not tile_alignment_done(
        tile, str(filtered_dir), str(align_dir), routing)]
    logging.info('Alignment: %s pending, %s cached; %s CPU workers',
                 len(todo), len(tiles) - len(todo), args.workers)
    jobs = [(str(paths[tile]), str(filtered_dir), str(align_dir), routing, settings) for tile in todo]
    aligned = run_jobs(align_one, jobs, min(args.workers, len(jobs)), 'align') if jobs else []
    validate_alignment_frame(str(align_dir), tiles, list(channels), settings['stitching_reference_channel'])
    # Validate completed artifacts, including CSV hashes and input/output row counts.
    incomplete = [tile for tile in tiles if not tile_alignment_done(
        tile, str(filtered_dir), str(align_dir), routing)]
    if incomplete:
        raise RuntimeError(f'Incomplete alignment: {incomplete}')
    import pandas as pd
    summary = []
    comparison = []
    previous_dir = base / 'old_results' / '3_2d_aligned'
    uninformative = set()
    for tile in tiles:
        measured = load_config(align_dir / f'{tile}_measured_offsets.json')
        offsets = load_config(align_dir / f'{tile}_offsets.json')
        diagnostics_path = align_dir / 'diagnostics' / f'{tile}.json'
        diagnostics = load_config(diagnostics_path) if diagnostics_path.is_file() else {}
        previous_path = previous_dir / f'{tile}_offsets.json'
        previous = load_config(previous_path) if previous_path.is_file() else {}
        counts = diagnostics.get('cells_in_window', {})
        for cid in channels:
            count = counts.get(cid)
            score = measured['scores'].get(cid)
            reason = []
            if counts.get(settings['reference_channel']) == 0:
                reason.append('no_reference_cells')
            if count == 0:
                reason.append('no_channel_cells')
            if cid != settings['reference_channel'] and (score is None or score <= 0):
                reason.append('no_positive_fit_score')
            if reason:
                uninformative.add(tile)
            summary.append({
                'tile': tile, 'channel': cid, 'cells_in_sample_window': count,
                'fit_score': score, 'fit_warning': ';'.join(reason),
                **{key: offsets[cid][key] for key in ('dx', 'dy', 'dz')},
                'z_start': measured.get('sample_window', {}).get('z_start'),
                'z_stop_exclusive': measured.get('sample_window', {}).get('z_stop_exclusive'),
            })
            if cid in previous:
                row = {'tile': tile, 'channel': cid}
                ref = settings['reference_channel']
                for key in ('dx', 'dy', 'dz'):
                    row[f'old_{key}'] = previous[cid][key]
                    row[f'new_{key}'] = offsets[cid][key]
                    if ref in previous:
                        old_relative = previous[cid][key] - previous[ref][key]
                        new_relative = offsets[cid][key] - offsets[ref][key]
                        row[f'old_relative_{key}'] = old_relative
                        row[f'new_relative_{key}'] = new_relative
                        row[f'delta_relative_{key}'] = new_relative - old_relative
                comparison.append(row)
    atomic_csv(pd.DataFrame(summary), align_dir / 'alignment_fit_summary.csv')
    if comparison:
        atomic_csv(pd.DataFrame(comparison), align_dir / 'old_new_offsets_comparison.csv')
    if uninformative:
        logging.warning('%s tiles have channels without enough fit support: %s. '
                        'See alignment_fit_summary.csv; no automatic old-offset fallback.',
                        len(uninformative), ', '.join(sorted(uninformative)))
    atomic_json(align_dir / '_refilter_align_run.json', {
        'status': 'complete', 'tiles': tiles, 'sampling': windows,
        'elapsed_seconds': time.time() - started, 'new_measurements': aligned,
        'tiles_with_fit_warnings': sorted(uninformative),
        'config': config, 'stopped_before': 'global_stitching_and_colocalization',
    })
    if set(tiles) == set(available):
        (align_dir / '_align_done.flag').touch()
    logging.info('Complete: %s tiles. Stopped before global stitching and colocalization. %.1f seconds',
                 len(tiles), time.time() - started)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
