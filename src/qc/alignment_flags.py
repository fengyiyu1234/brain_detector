"""Flag tiles whose saved channel offsets look wrong, from completed alignment QC records.

TF channels: per-axis median of (nuclear - soma) intensity-center residuals over the fixed
containment pairs, with a bootstrap SE. Soma channels: voxel-IoU scan optimum vs. zero shift.
Runnable on an existing QC directory: python -m src.qc.alignment_flags --qc-dir <qc/alignment>.
"""
import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd
from src.config.loader import load_config
from .alignment_plots import curve_summary

FLAG_DEFAULTS = {
    'max_offset_xy_um': 3.0,      # TF median residual / IoU optimum beyond this in X or Y
    'max_offset_z_um': 4.0,       # same for Z (half of an 8 um slice)
    'min_z_score': 3.0,           # TF residual must also exceed this many bootstrap SEs
    'bootstrap_samples': 1000,
    'iou_min_gain': 1.05,         # IoU optimum must beat zero shift by this ratio
    'iou_min_vs_unaligned': 1.1,  # aligned IoU must beat the unaligned IoU by this ratio
}
SEVERITY = {'ok': 0, 'insufficient': 1, 'bad': 2, 'error': 3}


def resolve_flag_options(options):
    options = options or {}
    if not isinstance(options, dict) or set(options) - FLAG_DEFAULTS.keys():
        raise ValueError('Unknown/invalid QC flagging options')
    out = {**FLAG_DEFAULTS, **options}
    for key, value in out.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'flagging.{key} must be finite and positive')
    if not isinstance(out['bootstrap_samples'], int):
        raise ValueError('flagging.bootstrap_samples must be an integer')
    return out


def tf_rows(record, settings, flags, min_pairs, seed):
    scale = np.array([settings['xy_resolution_um']]*2 + [settings['z_resolution_um']])
    limits = np.array([flags['max_offset_xy_um']]*2 + [flags['max_offset_z_um']])
    rows = []
    for cid, pairs in sorted(record['pairs'].items()):
        residual = []
        for pair in pairs:
            soma, nucleus = np.asarray(pair['soma_points']), np.asarray(pair['nuclear_points'])
            if soma.ndim == 2:
                soma, nucleus = soma[-1], nucleus[-1]
            residual.append((nucleus-soma)*scale)
        residual = np.asarray(residual, dtype=float).reshape(-1, 3)
        row = dict(tile=record['tile'], channel=cid, method='intensity_center', n_pairs=len(residual))
        if len(residual) < min_pairs:
            rows.append(dict(row, verdict='insufficient', reasons=f'{len(residual)} pairs < {min_pairs}'))
            continue
        median = np.median(residual, axis=0)
        draws = np.random.default_rng(seed).integers(0, len(residual), (flags['bootstrap_samples'], len(residual)))
        se = np.median(residual[draws], axis=1).std(axis=0)
        reasons = [f'{axis} {m:+.1f} um ({m/s:+.1f} px, {abs(m)/e:.1f} SE)' if e > 0 else
                   f'{axis} {m:+.1f} um ({m/s:+.1f} px)'
                   for axis, m, e, s, limit in zip('xyz', median, se, scale, limits)
                   if abs(m) > limit and abs(m) >= flags['min_z_score']*e]
        row.update({f'd{a}_um': float(m) for a, m in zip('xyz', median)})
        row.update({f'se_{a}_um': float(e) for a, e in zip('xyz', se)})
        rows.append(dict(row, verdict='bad' if reasons else 'ok', reasons='; '.join(reasons)))
    return rows


def iou_rows(record, settings, flags, fixed):
    frame = pd.DataFrame([r for r in record['curves'] if r['metric'] == 'voxel_iou'])
    if frame.empty:
        return []
    limits = {'x': flags['max_offset_xy_um'], 'y': flags['max_offset_xy_um'], 'z': flags['max_offset_z_um']}
    rows = []
    for cid, group in frame.groupby('channel', sort=True):
        summary = {r['axis']: r for r in curve_summary(group)}
        unaligned = group.unaligned_iou.dropna()
        at_zero = next((r.get('value_at_zero') for r in summary.values() if r.get('value_at_zero') is not None), None)
        row = dict(tile=record['tile'], channel=cid, method='voxel_iou', n_pairs=0,
                   iou_at_zero=at_zero, unaligned_iou=float(unaligned.iloc[0]) if len(unaligned) else None)
        if at_zero is None or not at_zero > 0:
            rows.append(dict(row, verdict='insufficient', reasons='no soma overlap signal'))
            continue
        reasons = []
        for axis, item in summary.items():
            row[f'd{axis}_um'] = item.get('best_delta_um')
            gain = item.get('best_value', at_zero)/at_zero
            if abs(item.get('best_delta_um') or 0) > limits[axis] and gain >= flags['iou_min_gain']:
                unit = 'slice' if axis == 'z' else 'px'
                reasons.append(f"{axis} IoU peak {item['best_delta_um']:+.1f} um "
                               f"({item['best_delta']:+g} {unit}, x{gain:.2f})")
        # "Unaligned" applies the reference offset; identical saved offsets make it the same scan.
        moved = record.get('offsets', {}).get(cid) != record.get('offsets', {}).get(fixed)
        if moved and row['unaligned_iou'] and at_zero < flags['iou_min_vs_unaligned']*row['unaligned_iou']:
            reasons.append(f"aligned IoU {at_zero:.3f} vs unaligned {row['unaligned_iou']:.3f}")
        rows.append(dict(row, verdict='bad' if reasons else 'ok', reasons='; '.join(reasons)))
    return rows


def flag_tiles(report):
    """Per tile x channel verdicts plus a tile table sorted worst-first."""
    options = report['options']
    flags = resolve_flag_options(options.get('flagging'))
    min_pairs, seed = options['sampling']['min_pairs'], options['sampling']['seed']
    rows = []
    for record in report['completed']:
        rows += iou_rows(record, report['settings'], flags, options['reference_channel'])
        rows += tf_rows(record, report['settings'], flags, min_pairs, seed)
    rows += [dict(tile=tile, channel='all', method='', n_pairs=0, verdict='error',
                  reasons='QC failed; see run_details.json')
             for tile in sorted(report.get('failures', {}))]
    channels = pd.DataFrame(rows, columns=['tile', 'channel', 'method', 'n_pairs', 'verdict', 'reasons',
                                           'dx_um', 'dy_um', 'dz_um', 'se_x_um', 'se_y_um', 'se_z_um',
                                           'iou_at_zero', 'unaligned_iou'])
    tiles = []
    for tile, group in channels.groupby('tile', sort=True):
        worst = max(group.verdict, key=SEVERITY.get)
        tiles.append(dict(tile=tile, verdict=worst,
                          bad_channels=','.join(group.loc[group.verdict.isin(['bad', 'error']), 'channel']),
                          insufficient_channels=','.join(group.loc[group.verdict.eq('insufficient'), 'channel']),
                          reasons=' | '.join(f'{c}: {r}' for c, r, v in zip(group.channel, group.reasons, group.verdict)
                                             if v in ('bad', 'error'))))
    tiles = pd.DataFrame(tiles, columns=['tile', 'verdict', 'bad_channels', 'insufficient_channels', 'reasons'])
    tiles = tiles.sort_values(['verdict', 'tile'], key=lambda s: s.map(SEVERITY) if s.name == 'verdict' else s,
                              ascending=[False, True]).reset_index(drop=True)
    return channels, tiles, flags


def report_lines(tiles, flags):
    bad = tiles.loc[tiles.verdict.isin(['bad', 'error'])]
    sparse = tiles.loc[tiles.verdict.eq('insufficient')]
    lines = ['## Flagged tiles', '',
             f'{len(bad)} of {len(tiles)} tiles need review; {len(sparse)} have too few pairs/signal to judge.',
             f"TF: median intensity-center residual > {flags['max_offset_xy_um']:g} um (XY) / "
             f"{flags['max_offset_z_um']:g} um (Z) and >= {flags['min_z_score']:g} bootstrap SE. "
             f"Soma: IoU optimum beyond the same limits with >= x{flags['iou_min_gain']:g} gain, or aligned IoU "
             f"< x{flags['iou_min_vs_unaligned']:g} unaligned.", '']
    if len(bad):
        lines += ['| Tile | Channels | Reasons |', '|---|---|---|']
        lines += [f'| {r.tile} | {r.bad_channels} | {r.reasons} |' for r in bad.itertuples()]
        lines.append('')
    if len(sparse):
        lines += ['Insufficient data: ' + ', '.join(f'{r.tile} ({r.insufficient_channels})' for r in sparse.itertuples()), '']
    lines += ['[Flagged tiles](flagged_tiles.csv) | [Per-channel verdicts](tile_verdicts.csv)']
    return lines


def write_flags(output, report):
    """Write flag tables; returns the tile table."""
    output = Path(output)
    channels, tiles, flags = flag_tiles(report)
    channels.to_csv(output/'tile_verdicts.csv', index=False)
    tiles.loc[tiles.verdict.ne('ok')].to_csv(output/'flagged_tiles.csv', index=False)
    return tiles, report_lines(tiles, flags)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qc-dir', required=True, help='Alignment QC directory containing run_details.json')
    args = parser.parse_args(argv)
    tiles, lines = write_flags(args.qc_dir, load_config(Path(args.qc_dir)/'run_details.json'))
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
