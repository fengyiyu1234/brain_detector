"""QC C: Stage 2.25 SNR filter per tile and slice.

Boxes reaching the SNR step = rows of 2_2d_filtered/{tile}_{ch}_result.csv plus the rows of
_rejected.csv with rejection_reason snr_min; one slice = one source_image. For each slice:
the threshold used, the share removed, whether the threshold was lowered, and whether it
is saturated: the share removed still exceeds max_removed_frac (the threshold could not go
lower without passing snr_floor); only slices with >= min_boxes boxes count. Parameters come from the config's contrast_filter.

WARN when a channel has saturated slices on more than saturated_warn_frac of its slices.
Images: threshold and removed share along z (one line per tile) and, for the most removed
saturated slices, the raw crop with kept boxes green and SNR-removed boxes magenta.

Outputs: qc/snr/{slices.csv,summary.csv,summary.json,report.md,<ch>_by_z.png}, qc/review/snr/*.png.
"""
import json
import os

import numpy as np
import pandas as pd

from .common import save_json
from . import review_images as rv

DEFAULTS = {'saturated_warn_frac': 0.05, 'min_boxes': 20}
COLS = ['x1', 'y1', 'x2', 'y2', 'z', 'source_image', 'snr', 'snr_threshold']


def _tiles(filt_dir, ch):
    suffix = f'_{ch}_result.csv'
    return sorted(f[:-len(suffix)] for f in os.listdir(filt_dir) if f.endswith(suffix))


def _read(path, cols, reason=None):
    if not os.path.isfile(path):
        return pd.DataFrame(columns=cols)
    head = pd.read_csv(path, nrows=0).columns
    use = [c for c in cols + (['rejection_reason'] if reason else []) if c in head]
    df = pd.read_csv(path, usecols=use, dtype={'source_image': str})
    if reason:
        df = df[df.rejection_reason == reason] if 'rejection_reason' in df else df.iloc[:0]
    return df


def slice_table(filt_dir, ch, params, min_boxes=1):
    rows = []
    for tile in _tiles(filt_dir, ch):
        kept = _read(os.path.join(filt_dir, f'{tile}_{ch}_result.csv'), COLS)
        rem = _read(os.path.join(filt_dir, f'{tile}_{ch}_rejected.csv'), COLS, 'snr_min')
        if 'snr_threshold' not in kept:
            continue
        kept = kept.assign(removed=False)
        rem = rem.assign(removed=True)
        both = pd.concat([d for d in (kept, rem) if len(d)], ignore_index=True) if len(kept) + len(rem) else kept
        if both.empty:
            continue
        g = both.groupby('source_image', sort=False)
        t = pd.DataFrame({'z': g.z.first(), 'n_in': g.size(), 'n_removed': g.removed.sum(),
                          'threshold': g.snr_threshold.max()})
        t['tile'] = tile
        rows.append(t.reset_index())
    if not rows:
        return pd.DataFrame(columns=['tile', 'source_image', 'z', 'n_in', 'n_removed', 'threshold'])
    out = pd.concat(rows, ignore_index=True)
    out['removed_frac'] = out.n_removed / out.n_in
    out['threshold'] = out.threshold.fillna(params['snr_min'])
    out['lowered'] = out.threshold < params['snr_min'] - 1e-9
    max_frac = params.get('max_removed_frac')
    out['saturated'] = ((out.removed_frac > max_frac + 1e-12) & (out.n_in >= min_boxes)
                        if max_frac is not None else False)
    out['channel'] = ch
    return out


def plot_by_z(table, ch, params, png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for tile, g in table.groupby('tile'):
        g = g.sort_values('z')
        axes[0].plot(g.z, g.threshold, lw=0.6, alpha=0.5)
        axes[1].plot(g.z, g.removed_frac, lw=0.6, alpha=0.5)
    axes[0].axhline(params['snr_min'], color='k', ls='--', lw=0.8)
    axes[0].axhline(params['snr_floor'], color='r', ls='--', lw=0.8)
    axes[0].set_ylabel('SNR threshold')
    if params.get('max_removed_frac') is not None:
        axes[1].axhline(params['max_removed_frac'], color='r', ls='--', lw=0.8)
    axes[1].set_ylabel('share removed by SNR')
    axes[1].set_xlabel('z (slice)')
    axes[0].set_title(f'{ch}: one line per tile (dashed: snr_min, red: floor / max_removed_frac)')
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)


def slice_review(images, filt_dir, ch, row, out_dir, review):
    kept = _read(os.path.join(filt_dir, f'{row.tile}_{ch}_result.csv'), COLS)
    rem = _read(os.path.join(filt_dir, f'{row.tile}_{ch}_rejected.csv'), COLS, 'snr_min')
    kept, rem = kept[kept.source_image == row.source_image], rem[rem.source_image == row.source_image]
    pts = pd.concat([kept, rem])
    window = rv.densest_window((pts.x1 + pts.x2) / 2, (pts.y1 + pts.y2) / 2, int(review['crop_px']))
    # 2_2d_filtered is in raw tile coordinates: read without the Stage 2.5 shift
    img = images.read(ch, row.tile, int(row.z), window, shift=(0, 0, 0))
    pil = rv.to_pil(img)
    plain = pil.copy()
    rv.draw_boxes(pil, kept[['x1', 'y1', 'x2', 'y2']].to_numpy(), window, (0, 255, 0), 2)
    rv.draw_boxes(pil, rem[['x1', 'y1', 'x2', 'y2']].to_numpy(), window, (255, 0, 255), 2)
    png = os.path.join(out_dir, f'{ch}_{row.tile}_z{int(row.z):04d}.png')
    return rv.grid_png([[(plain, 'raw'), (pil, 'green kept, magenta removed by SNR')]], png,
                       f'{ch} {row.tile} slice {int(row.z)}: threshold {row.threshold:g}, '
                       f'removed {row.n_removed}/{row.n_in} ({100 * row.removed_frac:.0f}%)',
                       'saturated: the share removed is still above max_removed_frac at the floor')


def run(config, results_dir, force=False):
    from src.core.snr_filter import resolve_snr_filter
    filters = resolve_snr_filter(config)
    options = dict(DEFAULTS)
    options.update({k: v for k, v in (((config.get('qc') or {}).get('snr')) or {}).items() if k in DEFAULTS})
    review = rv.review_options(config)
    out_dir = os.path.join(results_dir, 'qc', 'snr')
    filt_dir = os.path.join(results_dir, '2_2d_filtered')
    if not filters or not os.path.isdir(filt_dir):
        summary = {'status': 'NOT_RUN', 'reason': 'no contrast_filter or 2_2d_filtered'}
        save_json(os.path.join(out_dir, 'summary.json'), summary)
        return summary
    os.makedirs(out_dir, exist_ok=True)
    tables, sums = [], []
    for ch, params in filters.items():
        t = slice_table(filt_dir, ch, {'snr_floor': 0, **params}, int(options['min_boxes']))
        if t.empty:
            continue
        tables.append(t)
        plot_by_z(t, ch, {'snr_floor': params.get('snr_floor', 0), **params},
                  os.path.join(out_dir, f'{ch}_by_z.png'))
        sat_tiles = t[t.saturated].groupby('tile').size()
        sums.append({'channel': ch, 'slices': len(t), 'lowered_frac': float(t.lowered.mean()),
                     'saturated_slices': int(t.saturated.sum()), 'saturated_frac': float(t.saturated.mean()),
                     'removed_frac_overall': float(t.n_removed.sum() / max(1, t.n_in.sum())),
                     'removed_frac_p95': float(t.removed_frac.quantile(0.95)),
                     'tiles_with_saturated': int(len(sat_tiles)),
                     'worst_tile': str(sat_tiles.idxmax()) if len(sat_tiles) else ''})
    slices = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    summ = pd.DataFrame(sums)
    slices.to_csv(os.path.join(out_dir, 'slices.csv'), index=False)
    summ.to_csv(os.path.join(out_dir, 'summary.csv'), index=False)
    warn = summ[summ.saturated_frac > options['saturated_warn_frac']] if len(summ) else summ
    summary = {'status': 'WARN' if len(warn) else 'PASS',
               'channels': summ.to_dict('records'), 'warn_channels': warn.channel.tolist() if len(warn) else [],
               'plots': [os.path.join(out_dir, f'{c}_by_z.png') for c in summ.channel] if len(summ) else [],
               'images': []}
    image_dir = os.path.join(results_dir, 'qc', 'review', 'snr')
    if review['enabled'] and len(slices):
        sat = slices[slices.saturated].sort_values(['n_removed', 'removed_frac'], ascending=False)
        picks = sat.groupby('channel', sort=False).head(int(review['max_items_per_check']))
        manifest = os.path.join(image_dir, '_manifest.json')
        files = sorted({os.path.join(filt_dir, f'{r.tile}_{r.channel}_result.csv') for r in picks.itertuples()})
        sig = rv.review_signature(files, {'picks': [f'{r.channel}:{r.tile}:{r.source_image}' for r in picks.itertuples()],
                                          'review': review, 'module': _code()})
        if not force and rv.images_current(manifest, sig):
            with open(manifest, encoding='utf-8') as handle:
                summary['images'] = json.load(handle)['images']
        else:
            images = rv.TileImages(config, results_dir, review['min_image_bytes'])
            if images.available():
                rv.clear_pngs(image_dir)
                written = [slice_review(images, filt_dir, r.channel, r, image_dir, review)
                           for r in picks.itertuples()]
                rv.save_images_manifest(manifest, sig, written)
                summary['images'] = written
            else:
                summary['images_note'] = 'image folders not reachable'
    lines = ['# SNR filter QC (Stage 2.25)', '', f"Status: **{summary['status']}** "
             f"(WARN when saturated slices > {100 * options['saturated_warn_frac']:.0f}% of a channel's slices)", '',
             '| channel | slices | lowered % | saturated % | removed % | removed p95 % | tiles w/ saturated |',
             '|---|---|---|---|---|---|---|']
    for r in summ.itertuples():
        lines.append(f'| {r.channel} | {r.slices} | {100 * r.lowered_frac:.1f} | {100 * r.saturated_frac:.1f} | '
                     f'{100 * r.removed_frac_overall:.1f} | {100 * r.removed_frac_p95:.1f} | {r.tiles_with_saturated} |')
    lines += ['', 'saturated = the threshold reached snr_floor and still removed more than max_removed_frac '
              f"(slices with >= {options['min_boxes']} boxes only).",
              'Plots: <channel>_by_z.png. Review images: ../review/snr/.']
    with open(os.path.join(out_dir, 'report.md'), 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
    save_json(os.path.join(out_dir, 'summary.json'), summary)
    return summary


def _code():
    from src.core.provenance import file_sha256
    return file_sha256(os.path.abspath(__file__))
