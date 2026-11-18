"""Seam dedup QC add-on: review images of the seams with the most residual duplicates.

The seam dedup QC (stitch_dedup.py) draws randomly sampled seams. This draws, per
channel, the top_seams seams with the highest residual_3d_pct (residual_3d > 0), at the
50-layer segment with the most duplicates Stage 2.97 measured, with the same three-panel
layout (stitch_dedup.review_image; imported, not changed).
Output: qc/stitch_dedup/images_top/*.png (+ _manifest.json).
"""
import json
import os

import pandas as pd

from . import review_images as rv
from . import stitch_dedup as sd


def run(config, results_dir, geom, routing, src_dirs, derived, top_seams=3, force=False):
    out_dir = os.path.join(results_dir, 'qc', 'stitch_dedup')
    seams_csv = os.path.join(out_dir, 'seams.csv')
    offsets_csv = geom.get('seam_offsets_csv')
    if not os.path.isfile(seams_csv) or not offsets_csv or not os.path.isfile(offsets_csv):
        return {'status': 'NOT_RUN', 'images': []}
    seams = pd.read_csv(seams_csv, dtype={'tile': str, 'nb_tile': str, 'channel': str})
    offsets = pd.read_csv(offsets_csv)
    image_dir = os.path.join(out_dir, 'images_top')
    picks = (seams[seams.residual_3d > 0].sort_values('residual_3d_pct', ascending=False)
             .groupby('channel', sort=False).head(int(top_seams)))
    manifest = os.path.join(image_dir, '_manifest.json')
    sig = rv.review_signature([seams_csv, offsets_csv],
                              {'picks': [f'{r.channel}:{r.tile}:{r.nb_tile}' for r in picks.itertuples()],
                               'code': _code()})
    if not force and rv.images_current(manifest, sig):
        with open(manifest, encoding='utf-8') as handle:
            return {'status': 'ok', 'images': json.load(handle)['images']}
    rv.clear_pngs(image_dir)
    os.makedirs(image_dir, exist_ok=True)
    seg_layers = int(geom.get('seam_seg_layers', 50))
    written = []
    by_ch = {ch['id']: ch for ch in routing}
    for ch_id, group in picks.groupby('channel', sort=False):
        ch = by_ch.get(ch_id)
        paths = sd._channel_paths(ch, src_dirs, derived, config)
        if not ch or not paths.get('image_root') or not os.path.isdir(paths['image_root']):
            continue
        own = offsets[(offsets.channel.astype(str) == ch_id) & (offsets.source == 'own')]
        rects = [r for r in sd.seam_rects(geom) if (r[0], r[1]) in set(zip(group.tile, group.nb_tile))]
        bands = sd._global_bands(os.path.join(paths['global_2d'], f'{ch_id}_2d_global.csv'), rects)
        decisions = pd.read_csv(os.path.join(paths['global_2d'], f'{ch_id}_stitch_decisions.csv'),
                                dtype={'detection_id': str, 'tile_name': str})
        renderer = sd._Renderer(geom, paths['image_root'], paths.get('align_dir'))
        for r in group.itertuples():
            rows = own[(own.tile.astype(str) == r.tile) & (own.nb_tile.astype(str) == r.nb_tile)]
            if rows.empty:
                continue
            row = rows.sort_values('n_dup', ascending=False).iloc[0]
            layer = min(int(row.seg) * seg_layers + seg_layers // 2, int(geom['Z']) - 1)
            png = os.path.join(image_dir, f'{ch_id}_{r.tile}__{r.nb_tile}_layer{layer:03d}.png')
            sd.review_image(ch_id, r.tile, r.nb_tile, layer, geom, renderer, row, bands, decisions,
                            paths['src_dir'], png)
            written.append(png)
    rv.save_images_manifest(manifest, sig, written)
    return {'status': 'ok', 'images': written}


def _code():
    from src.core.provenance import file_sha256
    return file_sha256(os.path.abspath(__file__))
