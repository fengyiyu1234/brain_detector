"""QC E: saved-output integrity checks (src/utils/pipeline_qc.py) for one result directory.

Thresholds and options come from config/qc_config.json (its sample list is ignored);
qc.integrity in the pipeline config overrides them. Off by default in the pipeline
(qc.integrity.enabled) until its run time on large samples is known.
Output: qc/integrity/{qc_report.json,qc_report.md,findings.csv,...}.
"""
import os
import sqlite3
import tempfile
import time
from contextlib import closing
from pathlib import Path

from .common import save_json

ROOT = Path(__file__).resolve().parents[2]


def options_for(config):
    from src.config.loader import load_config
    base = ROOT / 'config' / 'qc_config.json'
    options = {k: v for k, v in load_config(str(base)).items() if k != 'samples'} if base.is_file() else {}
    for key, value in (((config.get('qc') or {}).get('integrity')) or {}).items():
        if key == 'enabled':
            continue
        options[key] = {**options.get(key, {}), **value} if isinstance(value, dict) else value
    return options


def run(config, results_dir, force=False):
    from src.utils.pipeline_qc import PipelineQC
    out = Path(results_dir) / 'qc' / 'integrity'
    out.mkdir(parents=True, exist_ok=True)
    options = options_for(config)
    sample = {'id': os.path.basename(os.path.normpath(results_dir)), 'result_dir': str(results_dir)}
    start = time.time()
    with tempfile.TemporaryDirectory(prefix='brain_qc_', dir=options.get('scratch_dir')) as scratch:
        with closing(sqlite3.connect(str(Path(scratch) / 'relations.sqlite'))) as db:
            db.execute('PRAGMA journal_mode=OFF')
            db.execute('PRAGMA synchronous=OFF')
            db.execute('PRAGMA cache_size=-32768')
            report = PipelineQC(sample, options, out, db).run()
    minutes = (time.time() - start) / 60
    counts = {}
    for f in report['findings']:
        counts[f['status']] = counts.get(f['status'], 0) + 1
    summary = {'status': report['status'], 'minutes': round(minutes, 1), 'finding_counts': counts,
               'fail_codes': sorted({f['code'] for f in report['findings'] if f['status'] == 'FAIL'}),
               'warn_codes': sorted({f['code'] for f in report['findings'] if f['status'] == 'WARN'}),
               'report': str(out / 'qc_report.md')}
    save_json(out / 'summary.json', summary)
    return summary
