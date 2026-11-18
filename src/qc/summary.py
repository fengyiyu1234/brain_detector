"""QC F: one summary of all sample QC checks: qc/qc_summary.json and qc/index.html.

Reads the outputs of the existing alignment QC and seam dedup QC and of the checks
A-E (unaligned, seam_consistency, snr, distribution, integrity). Status per check:
PASS / WARN / FAIL / NOT_RUN. Nothing is recomputed here.
"""
import glob
import html
import json
import logging
import os

import pandas as pd

from .common import save_json

DEFAULTS = {'stitch_dedup_warn_seam_pct': 10.0}


def _json(path):
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as handle:
            return json.load(handle)
    return None


def _alignment(qc):
    path = os.path.join(qc, 'alignment', 'tile_verdicts.csv')
    if not os.path.isfile(path):
        return {'status': 'NOT_RUN'}
    v = pd.read_csv(path, dtype={'tile': str})
    worst = v.groupby('tile').verdict.agg(lambda s: 'bad' if (s == 'bad').any() or (s == 'error').any()
                                          else 'insufficient' if (s == 'insufficient').any() else 'ok')
    bad = worst[worst == 'bad'].index.tolist()
    return {'status': 'WARN' if bad else 'PASS', 'tiles': int(len(worst)), 'bad_tiles': len(bad),
            'insufficient_tiles': int((worst == 'insufficient').sum()), 'bad': bad,
            'report': os.path.join(qc, 'alignment', 'qc_report.md'),
            'images': sorted(glob.glob(os.path.join(qc, 'alignment', 'tile_plots', '*.png')))[:20]}


def _stitch_dedup(qc, options):
    path = os.path.join(qc, 'stitch_dedup', 'summary.csv')
    if not os.path.isfile(path):
        return {'status': 'NOT_RUN'}
    s = pd.read_csv(path)
    high = s[s.seam_residual_3d_pct_max > options['stitch_dedup_warn_seam_pct']]
    images = sorted(glob.glob(os.path.join(qc, 'stitch_dedup', 'images_top', '*.png')))
    images += sorted(glob.glob(os.path.join(qc, 'stitch_dedup', 'images', '*.png')))
    return {'status': 'WARN' if len(high) else 'PASS',
            'channels': {r.channel: {'residual_3d_pct': round(float(r.residual_3d_pct), 3),
                                     'seam_max_pct': round(float(r.seam_residual_3d_pct_max), 2)}
                         for r in s.itertuples()},
            'note': f"WARN: a seam keeps > {options['stitch_dedup_warn_seam_pct']}% 3D duplicates",
            'report': os.path.join(qc, 'stitch_dedup', 'qc_report.md'), 'images': images}


CHECKS = [
    ('alignment', 'Alignment (existing QC)', None),
    ('unaligned', 'A unaligned tiles', 'alignment/unaligned_summary.json'),
    ('seam_consistency', 'B seam consistency (2.9)', 'seam_consistency/summary.json'),
    ('stitch_dedup', 'Seam duplicates (existing QC)', None),
    ('snr', 'C SNR filter', 'snr/summary.json'),
    ('distribution', 'D distribution', 'distribution/summary.json'),
    ('integrity', 'E integrity', 'integrity/summary.json'),
]
ERRORS = {'alignment': 'alignment_error.json', 'stitch_dedup': 'stitch_dedup_error.json'}


def _key_numbers(name, s):
    if s.get('status') == 'NOT_RUN':
        return s.get('reason', '')
    if name == 'alignment':
        return f"bad {s['bad_tiles']}/{s['tiles']} tiles, insufficient {s['insufficient_tiles']}"
    if name == 'unaligned':
        return f"counts {s.get('counts')}; unaligned {s.get('unaligned_tiles')}"
    if name == 'seam_consistency':
        return (f"{s.get('counts')}; max |r| x {s.get('max_abs_rx', 0):.1f} y {s.get('max_abs_ry', 0):.1f} px "
                f"z {s.get('max_abs_rz', 0):.1f}")
    if name == 'stitch_dedup':
        return '; '.join(f"{c}: {v['residual_3d_pct']:.2f}% (seam max {v['seam_max_pct']:.1f}%)"
                         for c, v in s['channels'].items())
    if name == 'snr':
        return '; '.join(f"{c['channel']}: lowered {100 * c['lowered_frac']:.0f}%, saturated "
                         f"{100 * c['saturated_frac']:.1f}% of slices" for c in s.get('channels', []))
    if name == 'distribution':
        drop = ', '.join(f"{c} {100 * v['dropout_frac']:.2f}%" for c, v in s.get('channels', {}).items())
        return (f"TF+ {s.get('tf_share', float('nan')):.3f}; outlier tiles {len(s.get('outlier_tiles', []))}; "
                f"dropout layers {drop}")
    if name == 'integrity':
        return f"{s.get('finding_counts')}; {s.get('minutes')} min; FAIL {s.get('fail_codes')}"
    return ''


def run(config, results_dir):
    options = dict(DEFAULTS)
    options.update({k: v for k, v in (((config.get('qc') or {}).get('summary')) or {}).items() if k in DEFAULTS})
    qc = os.path.join(results_dir, 'qc')
    checks = {}
    for name, label, rel in CHECKS:
        if name == 'alignment':
            s = _alignment(qc)
        elif name == 'stitch_dedup':
            s = _stitch_dedup(qc, options)
        else:
            s = _json(os.path.join(qc, rel)) or {'status': 'NOT_RUN'}
        err = _json(os.path.join(qc, ERRORS.get(name, f'{name}_error.json')))
        if err:
            s = {**s, 'status': 'FAIL', 'error': err.get('error')}
        s['label'] = label
        s['key_numbers'] = _key_numbers(name, s) if s['status'] != 'FAIL' else s.get('error', '')
        checks[name] = s
    order = {'FAIL': 3, 'WARN': 2, 'PASS': 1, 'NOT_RUN': 0}
    overall = max((c['status'] for c in checks.values()), key=lambda x: order.get(x, 0))
    summary = {'sample': os.path.basename(os.path.normpath(results_dir)), 'results_dir': results_dir,
               'status': overall, 'checks': checks}
    save_json(os.path.join(qc, 'qc_summary.json'), summary)
    _html(qc, summary)
    logging.info('[QC] 汇总（%s）: %s', os.path.join(qc, 'index.html'), overall)
    for name, c in checks.items():
        logging.info('    %-8s %-32s %s', c['status'], c['label'], c['key_numbers'][:160])
    return summary


def _html(qc, summary):
    def rel(p):
        return html.escape(os.path.relpath(p, qc).replace(os.sep, '/'))
    colour = {'PASS': '#2e7d32', 'WARN': '#ef6c00', 'FAIL': '#c62828', 'NOT_RUN': '#757575'}
    parts = ['<!DOCTYPE html><html><head><meta charset="utf-8"><title>QC ' + html.escape(summary['sample']) +
             '</title><style>body{font-family:sans-serif;margin:20px}td,th{border:1px solid #ccc;padding:4px 8px;'
             'vertical-align:top}table{border-collapse:collapse}img{width:220px;margin:2px}</style></head><body>',
             f"<h1>QC {html.escape(summary['sample'])}: <span style='color:{colour.get(summary['status'])}'>"
             f"{summary['status']}</span></h1><p>{html.escape(summary['results_dir'])}</p>",
             '<table><tr><th>check</th><th>status</th><th>key numbers</th><th>report</th></tr>']
    for c in summary['checks'].values():
        report = c.get('report') or c.get('table')
        link = f"<a href='{rel(report)}'>{rel(report)}</a>" if report and os.path.exists(report) else ''
        parts.append(f"<tr><td>{html.escape(c['label'])}</td><td style='color:{colour.get(c['status'])}'>"
                     f"<b>{c['status']}</b></td><td>{html.escape(str(c['key_numbers']))}</td><td>{link}</td></tr>")
    parts.append('</table>')
    for c in summary['checks'].values():
        imgs = [p for p in (c.get('plots') or []) + (c.get('images') or []) if os.path.isfile(p)]
        if imgs:
            parts.append(f"<h2>{html.escape(c['label'])}</h2>")
            parts += [f"<a href='{rel(p)}'><img src='{rel(p)}' title='{rel(p)}'></a>" for p in imgs[:40]]
    parts.append('</body></html>')
    with open(os.path.join(qc, 'index.html'), 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(parts))
