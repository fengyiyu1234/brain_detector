"""Pipeline hooks; disabled QC does not import image-processing dependencies."""
import logging
from pathlib import Path
from .common import save_json


def run_alignment_stage(config, results_dir, tiles):
    options = config.get('qc', {}).get('alignment', {})
    if not options.get('enabled', False):
        return {'status': 'disabled'}
    try:
        if config.get('pipeline_mode') != 'pre_align':
            raise ValueError('Alignment QC requires pre_align saved offsets and filtered detections')
        from .alignment import run
        report = run(results_dir, options=options, tiles=tiles, config=config)
        if report['status'] == 'failed':
            raise RuntimeError(f"Alignment QC failed for {len(report['failures'])} tile(s)")
        return report
    except Exception as exc:
        logging.exception('Alignment QC failed')
        # Preserve detailed tile failures when they are already available.
        save_json(Path(results_dir)/'qc'/'alignment_error.json', {'status':'failed', 'error':str(exc)})
        if options.get('fail_on_error', False):
            raise
        return {'status': 'failed', 'error': str(exc)}
