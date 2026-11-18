"""QC A: tiles whose channels were never aligned, and review images of badly aligned tiles.

Stage 2.5 returns score 0 and zero shifts when the sampling window has no reference
cells (``missing_reference``). That is harmless on empty edge tiles but wrong when the
tile has cells elsewhere in z (ERH/t70 with a fixed 501-600 window, 11-17).

* ``empty``: reference-channel boxes in the tile < unaligned_min_ref_boxes; recorded only.
* ``unaligned``: at least that many reference boxes and every non-reference score is 0 -> WARN.
* ``no_tissue``: nonblank_center found no non-blank slice (offsets 0 on purpose).

Review images: the unaligned tiles plus the worst tiles of the existing alignment QC
(tile_verdicts.csv: soma |dz| >= 2 slices, |dx|,|dy| >= 6 um; TF residual > 6 um), up to
max_items_per_check, each as 4-channel overlays (left: Stage 2.5 shifts, right: QC's best
shift) and single-channel grey crops.

Outputs: qc/alignment/unaligned_tiles.csv, qc/alignment/unaligned_summary.json,
qc/review/alignment/*.png.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from .common import save_json
from . import review_images as rv

DEFAULTS = {'unaligned_min_ref_boxes': 200, 'fail_on_unaligned': False,
            'bad_dz_slices': 2, 'bad_dxy_um': 6.0, 'bad_tf_um': 6.0}


def options_for(config):
    out = dict(DEFAULTS)
    out.update({k: v for k, v in (((config.get('qc') or {}).get('alignment')) or {}).items() if k in DEFAULTS})
    return out


def classify(results_dir, options):
    align_dir = os.path.join(results_dir, '3_2d_aligned')
    filt_dir = os.path.join(results_dir, '2_2d_filtered')
    windows = {}
    wpath = os.path.join(align_dir, '_sample_windows.json')
    if os.path.isfile(wpath):
        with open(wpath, encoding='utf-8') as handle:
            windows = json.load(handle)
    rows = []
    for path in sorted(glob.glob(os.path.join(align_dir, '*_measured_offsets.json'))):
        tile = os.path.basename(path)[:-len('_measured_offsets.json')]
        with open(path, encoding='utf-8') as handle:
            m = json.load(handle)
        ref = m.get('reference_channel')
        scores = {k: float(v) for k, v in (m.get('scores') or {}).items()}
        others = {k: v for k, v in scores.items() if k != ref}
        win = m.get('sample_window') or {}
        lo = win.get('z_start') or (windows.get(tile) or {}).get('start')
        hi = win.get('z_stop_exclusive') or (windows.get(tile) or {}).get('stop')
        csv = os.path.join(filt_dir, f'{tile}_{ref}_result.csv')
        n_ref, n_win = 0, None
        if os.path.isfile(csv):
            z = pd.read_csv(csv, usecols=['z']).z.to_numpy()
            n_ref = int(len(z))
            if lo is not None and hi is not None:
                n_win = int(((z >= lo) & (z < hi)).sum())
        zero = bool(others) and all(v == 0 for v in others.values())
        if win.get('no_tissue'):
            status = 'no_tissue'
        elif not zero:
            status = 'ok'
        elif n_ref < options['unaligned_min_ref_boxes']:
            status = 'empty'
        else:
            status = 'unaligned'
        rows.append({'tile': tile, 'status': status, 'reference_channel': ref,
                     'ref_boxes_tile': n_ref, 'ref_boxes_window': n_win,
                     'window_start': lo, 'window_stop_exclusive': hi,
                     'z_center': m.get('z_center'),
                     **{f'score_{k}': v for k, v in scores.items()}})
    return pd.DataFrame(rows)


def worst_tiles(results_dir, options, settings):
    """Severe rows of the existing alignment QC, worst first: [(tile, channel, extra_shift, reason)]."""
    path = os.path.join(results_dir, 'qc', 'alignment', 'tile_verdicts.csv')
    if not os.path.isfile(path):
        return []
    v = pd.read_csv(path, dtype={'tile': str})
    v = v[v.verdict == 'bad']
    xy_um, z_um = settings['xy'], settings['z']
    out = []
    for r in v.itertuples():
        d = np.array([r.dx_um, r.dy_um, r.dz_um], float)
        d = np.nan_to_num(d)
        if r.method == 'voxel_iou':
            severity = max(abs(d[2]) / z_um / options['bad_dz_slices'],
                           max(abs(d[0]), abs(d[1])) / options['bad_dxy_um'])
            extra = (d[0] / xy_um, d[1] / xy_um, d[2] / z_um)
        else:
            severity = float(np.hypot(d[0], d[1])) / options['bad_tf_um']
            extra = (-d[0] / xy_um, -d[1] / xy_um, -d[2] / z_um)
        if severity >= 1:
            out.append((severity, r.tile, r.channel, extra, str(r.reasons)))
    out.sort(key=lambda t: -t[0])
    seen, picked = set(), []
    for severity, tile, ch, extra, reason in out:
        if tile not in seen:
            seen.add(tile)
            picked.append((tile, ch, extra, reason))
    return picked


def _boxes(results_dir, tile, ch, z_lo=None, z_hi=None):
    path = os.path.join(results_dir, '3_2d_aligned', f'{tile}_{ch}_result.csv')
    if not os.path.isfile(path):
        return pd.DataFrame(columns=['x1', 'y1', 'x2', 'y2', 'z'])
    df = pd.read_csv(path, usecols=['x1', 'y1', 'x2', 'y2', 'z'])
    if z_lo is not None:
        df = df[(df.z >= z_lo) & (df.z <= z_hi)]
    return df


def tile_review(images, results_dir, tile, extra_ch, extra, reason, out_dir, review, ref):
    """Overlay rows (pipeline vs QC best shift) + grey single channels at up to N real layers."""
    chans = images.channels
    ref_boxes = _boxes(results_dir, tile, ref)
    if ref_boxes.empty:
        ref_boxes = _boxes(results_dir, tile, chans[0])
    zs = pick_from(ref_boxes.z, images, tile, review['layers_per_item'])
    written = []
    size = int(review['crop_px'])
    for z in zs:
        on = ref_boxes[ref_boxes.z == z]
        window = rv.densest_window((on.x1 + on.x2) / 2, (on.y1 + on.y2) / 2, size)
        pipe = {ch: images.read(ch, tile, z, window) for ch in chans}
        best = dict(pipe)
        if extra_ch and extra is not None:
            s = images.shift(tile, extra_ch)
            best[extra_ch] = images.read(extra_ch, tile, z, window,
                                         (s[0] + extra[0], s[1] + extra[1], s[2] + extra[2]))
        rows = [[(rv.to_pil(rv.composite(pipe)), 'Stage 2.5 shifts (pipeline)'),
                 (rv.to_pil(rv.composite(best)),
                  f'QC best shift for {extra_ch}' if extra_ch else 'same (no QC estimate)')],
                [(rv.to_pil(pipe[ch]).resize((size // 2, size // 2)), f'{ch} (grey)') for ch in chans]]
        png = os.path.join(out_dir, f'{tile}_z{z:04d}.png')
        sub = (f'RFP red, GFP green, Sox9 cyan, Olig2 magenta; crop x {window[0]}-{window[1]}, '
               f'y {window[2]}-{window[3]}; {reason[:150]}')
        written.append(rv.grid_png(rows, png, f'{tile}  aligned slice {z}', sub))
    return written


def pick_from(z_values, images, tile, n):
    zs = pd.Series(z_values).value_counts()
    zs = zs[zs >= max(1, zs.max() * 0.2)] if len(zs) else zs
    return rv.pick_layers(zs.index.tolist(), n, ok=lambda z: images.all_real(tile, z))


def run(config, results_dir, force=False):
    options = options_for(config)
    review = rv.review_options(config)
    out_dir = os.path.join(results_dir, 'qc', 'alignment')
    table = classify(results_dir, options)
    if table.empty:
        summary = {'status': 'NOT_RUN', 'reason': 'no 3_2d_aligned/*_measured_offsets.json'}
        save_json(os.path.join(out_dir, 'unaligned_summary.json'), summary)
        return summary
    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(os.path.join(out_dir, 'unaligned_tiles.csv'), index=False)
    counts = table.status.value_counts().to_dict()
    unaligned = table[table.status == 'unaligned'].sort_values('ref_boxes_tile', ascending=False)
    dp = config.get('detection_params') or {}
    settings = {'xy': float(dp.get('xy_resolution_um', 0.65)), 'z': float(dp.get('z_resolution_um', 8.0))}
    worst = worst_tiles(results_dir, options, settings)
    summary = {'status': 'WARN' if len(unaligned) else 'PASS', 'counts': counts,
               'unaligned_tiles': unaligned.tile.tolist(),
               'severe_alignment_qc_tiles': [w[0] for w in worst],
               'table': os.path.join(out_dir, 'unaligned_tiles.csv'), 'images': []}

    image_dir = os.path.join(results_dir, 'qc', 'review', 'alignment')
    if review['enabled']:
        images = rv.TileImages(config, results_dir, review['min_image_bytes'])
        items = [(t, None, None, f'unaligned: {n} reference boxes, all scores 0')
                 for t, n in zip(unaligned.tile, unaligned.ref_boxes_tile)]
        items += [w for w in worst if w[0] not in set(unaligned.tile)]
        items = items[:int(review['max_items_per_check'])]
        manifest = os.path.join(image_dir, '_manifest.json')
        sources = []
        for t, *_ in items:
            sources += [os.path.join(results_dir, '3_2d_aligned', f'{t}_offsets.json')]
        sources.append(os.path.join(results_dir, 'qc', 'alignment', 'tile_verdicts.csv'))
        sig = rv.review_signature(sources, {'items': [[i[0], i[1], list(i[2]) if i[2] else None] for i in items],
                                            'review': review, 'module': _code()})
        if not force and rv.images_current(manifest, sig):
            with open(manifest, encoding='utf-8') as handle:
                summary['images'] = json.load(handle)['images']
        elif images.available():
            rv.clear_pngs(image_dir)
            written = []
            ref = table.reference_channel.dropna().iloc[0] if table.reference_channel.notna().any() else images.anchor
            for tile, ch, extra, reason in items:
                written += tile_review(images, results_dir, tile, ch, extra, reason, image_dir, review, ref)
            rv.save_images_manifest(manifest, sig, written)
            summary['images'] = written
        else:
            summary['images_note'] = 'image folders not reachable'
    save_json(os.path.join(out_dir, 'unaligned_summary.json'), summary)
    if len(unaligned) and options['fail_on_unaligned']:
        raise RuntimeError(f'{len(unaligned)} tile(s) with cells were not aligned: {unaligned.tile.tolist()}')
    return summary


def _code():
    from src.core.provenance import file_sha256
    return file_sha256(os.path.abspath(__file__))
