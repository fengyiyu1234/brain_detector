"""Run the sample QC checks on an existing result directory, without the pipeline.

    python -m src.qc.sample_qc --config config/EGFR/2/config_EGFR_t10.json [--only A,B,C,D,E,T,F] [--force]

Checks: A unaligned tiles, B seam consistency, C SNR filter, D distribution, E integrity
(src/utils/pipeline_qc.py; slow on large samples), T images of the worst dedup seams,
F summary (qc/qc_summary.json + qc/index.html). Default: A,B,C,D,T,F (E only when
qc.integrity.enabled or listed in --only). The existing alignment QC and seam dedup QC are
not rerun; their outputs are read. Nothing outside <results>/qc is written.
"""
import argparse
import json
import logging
import os
import time

from src.config.loader import load_config


def _geom(config, results_dir, derived):
    """Stage 3 geometry from the stitching XML recorded in runtime_config.json."""
    from src.core.seam_offsets import resolve_params
    from src.utils.io import loadTeraxml
    runtime = os.path.join(results_dir, 'runtime_config.json')
    xml = None
    if os.path.isfile(runtime):
        with open(runtime, encoding='utf-8') as handle:
            xml = (json.load(handle).get('paths') or {}).get('pATHXML')
    xml = xml or (config.get('paths') or {}).get('pATHXML')
    if not xml or not os.path.isfile(xml):
        return None
    tile_size = int((config.get('detection_params') or {}).get('tILESIZE', 2048))
    dir_dict, H, W, Z, z_start, disp = loadTeraxml(xml, tile_size)
    _, seam_params = resolve_params(config)
    return {'dir_dict': dir_dict, 'disp_mat_fin': disp, 'z_start': z_start, 'Z': Z, 'H': H, 'W': W,
            'tile_size': tile_size, 'num_tiles': len(dir_dict),
            'seam_offsets_csv': os.path.join(derived['pATH_GLOBAL_2D'], 'seam_offsets', 'seam_offsets.csv'),
            'seam_seg_layers': seam_params['seg_layers']}


def main(argv=None):
    from src.core.result_layout import result_paths
    from src.core.resume import channel_sources
    from .run_qc import run_check
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', required=True, help='pipeline config (JSONC)')
    parser.add_argument('--only', default=None, help='comma list of A,B,C,D,E,T,F')
    parser.add_argument('--force', action='store_true', help='redraw review images')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    config = load_config(args.config)
    results_dir = config['paths']['pATHRESULT']
    if '{date}' in results_dir or not os.path.isdir(results_dir):
        raise SystemExit(f'Result directory not found: {results_dir}')
    qc = config.get('qc') or {}
    wanted = set((args.only or 'A,B,C,D,T,F' + (',E' if (qc.get('integrity') or {}).get('enabled') else ''))
                 .upper().split(','))
    derived = result_paths(results_dir)
    routing = [ch for ch in config['channels_routing'] if ch.get('active', True)]
    src_dirs = channel_sources(config, derived)
    logging.info('[QC] %s: %s', results_dir, ','.join(sorted(wanted)))
    start = time.time()
    if 'A' in wanted and config.get('pipeline_mode') == 'pre_align':
        from .unaligned import run as run_a
        run_check('unaligned', run_a, results_dir, config, results_dir, args.force)
    if 'C' in wanted:
        from .snr_qc import run as run_c
        run_check('snr', run_c, results_dir, config, results_dir, args.force)
    if 'B' in wanted:
        from .seam_consistency import run as run_b
        run_check('seam_consistency', run_b, results_dir, config, results_dir, args.force)
    if 'T' in wanted:
        geom = _geom(config, results_dir, derived)
        if geom and os.path.isfile(geom['seam_offsets_csv']):
            from .stitch_dedup_top import run as run_t
            run_check('stitch_dedup_top', run_t, results_dir, config, results_dir, geom, routing, src_dirs,
                      derived, int((qc.get('review') or {}).get('top_dedup_seams', 3)), args.force)
        else:
            logging.info('[QC] top dedup seam images skipped: no stitching XML or seam offsets')
    if 'D' in wanted:
        from .distribution import run as run_d
        run_check('distribution', run_d, results_dir, config, results_dir, src_dirs, args.force)
    if 'E' in wanted:
        from .integrity import run as run_e
        run_check('integrity', run_e, results_dir, config, results_dir)
    if 'F' in wanted:
        from .summary import run as run_f
        run_check('summary', run_f, results_dir, config, results_dir)
    logging.info('[QC] done in %.1f min', (time.time() - start) / 60)


if __name__ == '__main__':
    main()
