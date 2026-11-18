"""QC D: cell counts along z and across tiles, single-layer tracks, TF+ share.

* Bad layer (per tile and channel, tile-local aligned z from the Stage 3 input CSVs):
  inside the tile's tissue range, count < bad_layer_ratio x the median of the
  bad_layer_window layers on each side (that median >= bad_layer_min_neighbour).
  Layers whose image is blank (< min_image_bytes) are labelled blank_image, others dropout.
* Single-layer tracks: share of 6_3d_global/{ch}_3d_tracked.csv rows with z_min == z_max.
* TF+ share: soma cells (soma_positive_count > 0) of 7_colocalization/coloc_result.csv with
  tf_positive_count > 0, per tile and per 25-layer z bin. Outlier tile: |share - median| >
  outlier_mad x MAD over tiles with >= outlier_min_soma soma.

WARN when a channel's dropout layers exceed bad_layer_warn_frac of its tissue tile-layers
or any tile is an outlier. Images: worst dropout layers (layer and the one above/below,
4-channel overlay) and outlier tiles (overlay with soma boxes: green has TF, red none).

Outputs: qc/distribution/{layers.csv,bad_layers.csv,tiles.csv,tf_by_z.csv,summary.json,
report.md,counts_by_z.png,tf_by_z.png}, qc/review/distribution/*.png.
"""
import json
import os

import numpy as np
import pandas as pd

from .common import save_json
from . import review_images as rv

DEFAULTS = {'bad_layer_ratio': 0.3, 'bad_layer_window': 5, 'bad_layer_min_neighbour': 20,
            'bad_layer_warn_frac': 0.01, 'outlier_mad': 4.0, 'outlier_min_soma': 500, 'z_bin': 25}


def options_for(config):
    out = dict(DEFAULTS)
    out.update({k: v for k, v in (((config.get('qc') or {}).get('distribution')) or {}).items() if k in DEFAULTS})
    return out


def layer_counts(src_dirs, channels):
    rows = []
    for ch in channels:
        folder = src_dirs[ch]
        suffix = f'_{ch}_result.csv'
        for name in sorted(os.listdir(folder)):
            if not name.endswith(suffix):
                continue
            tile = name[:-len(suffix)]
            z = pd.read_csv(os.path.join(folder, name), usecols=['z']).z
            c = z.value_counts().sort_index()
            rows.append(pd.DataFrame({'channel': ch, 'tile': tile, 'z': c.index.astype(int), 'count': c.values}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=['channel', 'tile', 'z', 'count'])


def bad_layers(layers, options):
    w, out, tissue = int(options['bad_layer_window']), [], {}
    for (ch, tile), g in layers.groupby(['channel', 'tile'], sort=False):
        z0, z1 = int(g.z.min()), int(g.z.max())
        s = g.set_index('z')['count'].reindex(range(z0, z1 + 1), fill_value=0).astype(float)
        roll = s.rolling(2 * w + 1, center=True, min_periods=1).median()
        top = roll.max()
        inside = roll[roll >= 0.1 * top] if top > 0 else roll.iloc[:0]
        if inside.empty:
            continue
        lo, hi = int(inside.index.min()), int(inside.index.max())
        tissue[(ch, tile)] = hi - lo + 1
        vals = s.to_numpy()
        idx = s.index.to_numpy()
        for k in range(len(vals)):
            z = idx[k]
            if z < lo or z > hi:
                continue
            nb = np.r_[vals[max(0, k - w):k], vals[k + 1:k + 1 + w]]
            if not nb.size:
                continue
            med = float(np.median(nb))
            if med >= options['bad_layer_min_neighbour'] and vals[k] < options['bad_layer_ratio'] * med:
                out.append({'channel': ch, 'tile': tile, 'z': int(z), 'count': int(vals[k]),
                            'neighbour_median': med, 'ratio': vals[k] / med})
    return pd.DataFrame(out, columns=['channel', 'tile', 'z', 'count', 'neighbour_median', 'ratio']), tissue


def single_layer(channel_3d_dir, channels):
    rows = []
    for ch in channels:
        p = os.path.join(channel_3d_dir, f'{ch}_3d_tracked.csv')
        if not os.path.isfile(p):
            continue
        t = pd.read_csv(p, usecols=['z_min', 'z_max'])
        rows.append({'channel': ch, 'tracks': len(t), 'single_layer_frac': float((t.z_min == t.z_max).mean())
                     if len(t) else np.nan})
    return pd.DataFrame(rows)


def tf_share(coloc_csv, options):
    cols = ['tile_name', 'cz', 'soma_positive_count', 'tf_positive_count']
    parts = []
    for df in pd.read_csv(coloc_csv, usecols=cols, chunksize=2_000_000, dtype={'tile_name': str}):
        parts.append(df[df.soma_positive_count > 0])
    d = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
    d['tf'] = d.tf_positive_count > 0
    tiles = d.groupby('tile_name').agg(soma=('tf', 'size'), tf_pos=('tf', 'sum')).reset_index()
    tiles['tf_share'] = tiles.tf_pos / tiles.soma
    big = tiles[tiles.soma >= options['outlier_min_soma']]
    med = float(big.tf_share.median()) if len(big) else np.nan
    mad = float((big.tf_share - med).abs().median()) * 1.4826 if len(big) else np.nan
    tiles['outlier'] = ((tiles.soma >= options['outlier_min_soma'])
                        & ((tiles.tf_share - med).abs() > options['outlier_mad'] * max(mad, 1e-3)))
    zb = int(options['z_bin'])
    d['z_bin'] = (d.cz // zb * zb).astype(int)
    by_z = d.groupby('z_bin').agg(soma=('tf', 'size'), tf_pos=('tf', 'sum')).reset_index()
    by_z['tf_share'] = by_z.tf_pos / by_z.soma
    overall = float(d.tf.mean()) if len(d) else np.nan
    return tiles, by_z, {'tf_share': overall, 'tile_median': med, 'tile_mad': mad, 'soma': int(len(d))}


def _plots(layers, by_z, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    for ch, g in layers.groupby('channel'):
        s = g.groupby('z')['count'].sum()
        ax.plot(s.index, s.values, lw=0.8, label=ch)
    ax.set_yscale('log')
    ax.set_xlabel('tile-local aligned z')
    ax.set_ylabel('boxes (all tiles)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'counts_by_z.png'), dpi=110)
    plt.close(fig)
    if len(by_z):
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(by_z.z_bin, by_z.tf_share, marker='o', ms=3)
        ax.set_xlabel('global z (bin start)')
        ax.set_ylabel('TF+ share of soma')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'tf_by_z.png'), dpi=110)
        plt.close(fig)


def _local_offsets(global_csv, src_csv, tile):
    """Global minus tile-local (aligned) coordinates of a tile, from shared detection_ids."""
    local = pd.read_csv(src_csv, usecols=['detection_id', 'x1', 'y1', 'z'], dtype={'detection_id': str})
    parts = []
    for df in pd.read_csv(global_csv, usecols=['detection_id', 'x1', 'y1', 'z', 'tile_name'],
                          chunksize=3_000_000, dtype={'detection_id': str, 'tile_name': str}):
        parts.append(df[df.tile_name == tile])
        if sum(len(p) for p in parts) > 5000:
            break
    glob_ = pd.concat(parts)
    m = glob_.merge(local, on='detection_id', suffixes=('_g', '_l'))
    if m.empty:
        return None
    return tuple(float(np.median(m[f'{a}_g'] - m[f'{a}_l'])) for a in ('x1', 'y1', 'z'))


def run(config, results_dir, src_dirs, force=False):
    options = options_for(config)
    review = rv.review_options(config)
    out_dir = os.path.join(results_dir, 'qc', 'distribution')
    coloc = os.path.join(results_dir, '7_colocalization', 'coloc_result.csv')
    channels = [ch['id'] for ch in config['channels_routing'] if ch.get('active', True)]
    if not os.path.isfile(coloc) or not all(os.path.isdir(src_dirs.get(c, '')) for c in channels):
        summary = {'status': 'NOT_RUN', 'reason': 'needs Stage 4 coloc_result.csv and Stage 3 input CSVs'}
        save_json(os.path.join(out_dir, 'summary.json'), summary)
        return summary
    os.makedirs(out_dir, exist_ok=True)
    layers = layer_counts(src_dirs, channels)
    bad, tissue = bad_layers(layers, options)
    images = rv.TileImages(config, results_dir, review['min_image_bytes'])
    reach = images.available()
    if len(bad):
        bad['blank_image'] = [(not images.is_real(r.channel, r.tile, int(r.z) - images.shift(r.tile, r.channel)[2]))
                              if reach else None for r in bad.itertuples()]
    else:
        bad['blank_image'] = []
    sl = single_layer(os.path.join(results_dir, '6_3d_global'), channels)
    tiles, by_z, tf = tf_share(coloc, options)
    layers.to_csv(os.path.join(out_dir, 'layers.csv'), index=False)
    bad.to_csv(os.path.join(out_dir, 'bad_layers.csv'), index=False)
    tiles.to_csv(os.path.join(out_dir, 'tiles.csv'), index=False)
    by_z.to_csv(os.path.join(out_dir, 'tf_by_z.csv'), index=False)
    _plots(layers, by_z, out_dir)
    dropout = bad[bad.blank_image != True] if len(bad) else bad  # noqa: E712 (None = unknown)
    per_ch = {}
    for ch in channels:
        n_t = sum(v for (c, _), v in tissue.items() if c == ch)
        n_b = int((dropout.channel == ch).sum()) if len(dropout) else 0
        per_ch[ch] = {'tissue_tile_layers': n_t, 'dropout_layers': n_b,
                      'dropout_frac': n_b / n_t if n_t else 0.0,
                      'blank_bad_layers': int(((bad.channel == ch) & (bad.blank_image == True)).sum()) if len(bad) else 0}  # noqa: E712
    for r in sl.itertuples():
        per_ch[r.channel]['single_layer_frac'] = r.single_layer_frac
    outliers = tiles[tiles.outlier]
    warn = [c for c, v in per_ch.items() if v['dropout_frac'] > options['bad_layer_warn_frac']]
    summary = {'status': 'WARN' if warn or len(outliers) else 'PASS', 'channels': per_ch,
               'dropout_warn_channels': warn, **tf,
               'outlier_tiles': outliers[['tile_name', 'soma', 'tf_share']].to_dict('records'),
               'plots': [os.path.join(out_dir, 'counts_by_z.png'), os.path.join(out_dir, 'tf_by_z.png')],
               'images': []}
    image_dir = os.path.join(results_dir, 'qc', 'review', 'distribution')
    if review['enabled'] and reach:
        n = int(review['max_items_per_check'])
        worst = dropout.sort_values('ratio').drop_duplicates(['tile', 'z']).head(n) if len(dropout) else dropout
        outl = outliers.reindex((outliers.tf_share - tf['tile_median']).abs().sort_values(ascending=False).index).head(n)
        manifest = os.path.join(image_dir, '_manifest.json')
        sig = rv.review_signature([coloc], {'bad': [f'{r.channel}:{r.tile}:{r.z}' for r in worst.itertuples()],
                                            'outliers': outl.tile_name.tolist(), 'review': review,
                                            'module': _code()})
        if not force and rv.images_current(manifest, sig):
            with open(manifest, encoding='utf-8') as handle:
                summary['images'] = json.load(handle)['images']
        else:
            rv.clear_pngs(image_dir)
            written = []
            for r in worst.itertuples():
                written.append(_bad_layer_png(images, src_dirs, r, review, image_dir))
            for r in outl.itertuples():
                written += _outlier_png(images, config, results_dir, src_dirs, r, review, image_dir)
            written = [w for w in written if w]
            rv.save_images_manifest(manifest, sig, written)
            summary['images'] = written
    _report(out_dir, summary, options)
    save_json(os.path.join(out_dir, 'summary.json'), summary)
    return summary


def _bad_layer_png(images, src_dirs, r, review, out_dir):
    src = os.path.join(src_dirs[r.channel], f'{r.tile}_{r.channel}_result.csv')
    b = pd.read_csv(src, usecols=['x1', 'y1', 'x2', 'y2', 'z'])
    near = b[(b.z - r.z).abs() <= 1]
    window = rv.densest_window((near.x1 + near.x2) / 2, (near.y1 + near.y2) / 2, int(review['crop_px']))
    row = []
    for z in (r.z - 1, r.z, r.z + 1):
        imgs = {ch: images.read(ch, r.tile, int(z), window) for ch in images.channels}
        pil = rv.to_pil(rv.composite(imgs))
        on = b[b.z == z]
        rv.draw_boxes(pil, on[['x1', 'y1', 'x2', 'y2']].to_numpy(), window, (255, 255, 0), 1)
        row.append((pil, f'z {z}: {len(on)} {r.channel} boxes in tile'))
    png = os.path.join(out_dir, f'layer_{r.channel}_{r.tile}_z{int(r.z):04d}.png')
    return rv.grid_png([row], png, f'{r.channel} {r.tile}: layer {r.z} has {r.count} boxes '
                       f'vs neighbour median {r.neighbour_median:.0f}',
                       'RFP red, GFP green, Sox9 cyan, Olig2 magenta; yellow = boxes of the channel')


def _outlier_png(images, config, results_dir, src_dirs, r, review, out_dir):
    soma_ch = next((ch['id'] for ch in config['channels_routing']
                    if ch.get('active', True) and ch.get('type', 'soma') == 'soma'), images.anchor)
    src = os.path.join(src_dirs[soma_ch], f'{r.tile_name}_{soma_ch}_result.csv')
    glob_csv = os.path.join(results_dir, '5_2d_global', f'{soma_ch}_2d_global.csv')
    off = _local_offsets(glob_csv, src, r.tile_name) if os.path.isfile(glob_csv) else None
    if off is None:
        return []
    cols = ['tile_name', 'x1_3d', 'y1_3d', 'x2_3d', 'y2_3d', 'z_min', 'z_max', 'tf_positive_count',
            'soma_positive_count']
    parts = [df[(df.tile_name == r.tile_name) & (df.soma_positive_count > 0)]
             for df in pd.read_csv(os.path.join(results_dir, '7_colocalization', 'coloc_result.csv'),
                                   usecols=cols, chunksize=2_000_000, dtype={'tile_name': str})]
    cells = pd.concat(parts)
    for a, k in (('x1_3d', 0), ('x2_3d', 0), ('y1_3d', 1), ('y2_3d', 1)):
        cells[a] = cells[a] - off[k]
    cells['zl0'], cells['zl1'] = cells.z_min - off[2], cells.z_max - off[2]
    zc = ((cells.zl0 + cells.zl1) / 2).round()
    zs = rv.pick_layers(zc.value_counts().index[:50].tolist(), review['layers_per_item'],
                        ok=lambda z: images.all_real(r.tile_name, z))
    written = []
    for z in zs:
        on = cells[(cells.zl0 <= z) & (cells.zl1 >= z)]
        window = rv.densest_window((on.x1_3d + on.x2_3d) / 2, (on.y1_3d + on.y2_3d) / 2, int(review['crop_px']))
        imgs = {ch: images.read(ch, r.tile_name, int(z), window) for ch in images.channels}
        pil = rv.to_pil(rv.composite(imgs))
        plain = pil.copy()
        box = ['x1_3d', 'y1_3d', 'x2_3d', 'y2_3d']
        rv.draw_boxes(pil, on[on.tf_positive_count > 0][box].to_numpy(), window, (0, 255, 0), 1)
        rv.draw_boxes(pil, on[on.tf_positive_count == 0][box].to_numpy(), window, (255, 0, 0), 1)
        png = os.path.join(out_dir, f'tile_{r.tile_name}_z{int(z):04d}.png')
        written.append(rv.grid_png([[(plain, 'overlay'), (pil, 'soma: green has TF, red none')]], png,
                                   f'{r.tile_name}: TF+ share {r.tf_share:.2f} of {r.soma} soma (outlier), slice {int(z)}',
                                   'RFP red, GFP green, Sox9 cyan, Olig2 magenta'))
    return written


def _report(out_dir, summary, options):
    lines = ['# Distribution QC (counts along z, tiles, TF+ share)', '', f"Status: **{summary['status']}**", '',
             f"TF+ share of soma: {summary['tf_share']:.3f} (tile median {summary['tile_median']:.3f}, "
             f"MAD {summary['tile_mad']:.3f}, {summary['soma']} soma)", '',
             '| channel | tissue tile-layers | dropout layers | dropout % | blank-image bad layers | single-layer tracks % |',
             '|---|---|---|---|---|---|']
    for ch, v in summary['channels'].items():
        slf = v.get('single_layer_frac')
        lines.append(f"| {ch} | {v['tissue_tile_layers']} | {v['dropout_layers']} | {100 * v['dropout_frac']:.2f} | "
                     f"{v['blank_bad_layers']} | {'' if slf is None else f'{100 * slf:.1f}'} |")
    lines += ['', f"Outlier tiles (> {options['outlier_mad']} MAD, >= {options['outlier_min_soma']} soma): "
              + (', '.join(f"{o['tile_name']} ({o['tf_share']:.2f})" for o in summary['outlier_tiles']) or 'none'),
              '', 'Tables: layers.csv, bad_layers.csv, tiles.csv, tf_by_z.csv. Review images: ../review/distribution/.']
    with open(os.path.join(out_dir, 'report.md'), 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def _code():
    from src.core.provenance import file_sha256
    return file_sha256(os.path.abspath(__file__))
