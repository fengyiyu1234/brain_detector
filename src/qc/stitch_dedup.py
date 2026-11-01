"""Seam deduplication QC after Stage 3 (read-only).

For every channel and EAST/SOUTH seam it measures what Stage 3 left behind in the
XML overlap of the two tiles:

* 3D: Z-linked cells whose members come mostly from tile A and mostly from tile B,
  centres within pair_radius px and overlapping z ranges (one-to-one, nearest
  first). Unmatched overlap boxes from B were moved into A's frame by Stage 3, so
  remaining duplicates sit at zero offset. A null count with B moved by
  null_shift_px estimates chance pairs; residual = pairs - null.
* 2D: the same count per 50-layer segment for boxes on the same layer, compared
  with the duplicates Stage 2.97 measured on the Stage 3 input.
* Decisions: counts of removed / relabelled boxes from the stitch decision CSVs.

Review PNGs: images_per_channel randomly sampled (seam, z segment) pairs per
channel (fixed seed, so the same seams are drawn while the results do not change)
show tile A, tile B moved by the measured offset, the final boxes, and B's removed
boxes. Existing PNGs are kept; they are rebuilt only when the Stage 3 results change.

Outputs (results/qc/stitch_dedup): summary.csv, seams.csv, segments_2d.csv,
qc_report.md, images/*.png.
"""
import argparse
import glob
import json
import logging
import os
import zlib
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.core.provenance import (file_sha256, file_stamp, output_manifest_valid,
                                 write_output_manifest)
from src.core.seam_offsets import adjacent_seams, grid_names, tile_origin

DEFAULTS = {
    'enabled': True,
    'output_dir': 'qc/stitch_dedup',
    'images_per_channel': 10,       # randomly sampled seam review PNGs per channel
    'image_seed': 0,
    'workers': 4,
    'edge_px': 8,
    'pair_radius_tf_px': 5.0,       # 3D centre distance counted as the same nucleus
    'pair_radius_soma_px': 10.0,    # ... soma
    'pair_radius_2d_tf_px': 3.0,    # same-layer 2D centre distance
    'pair_radius_2d_soma_px': 5.0,
    'null_shift_px': 30.0,
}
# Only these change the tables; the image options only affect the PNGs.
_METRIC_KEYS = ('edge_px', 'pair_radius_tf_px', 'pair_radius_soma_px',
                'pair_radius_2d_tf_px', 'pair_radius_2d_soma_px', 'null_shift_px')
_PAD = 60
_SCALE, _HALF_LEN, _MARGIN = 2, 256, 40


def resolve_options(config):
    options = dict(DEFAULTS)
    options.update((config.get('qc') or {}).get('stitch_dedup') or {})
    unknown = set(options) - set(DEFAULTS) - {'fail_on_error'}
    if unknown:
        raise ValueError(f"Unknown qc.stitch_dedup options: {sorted(unknown)}")
    return options


# ── geometry ──────────────────────────────────────────────────────────────────

def seam_rects(geom):
    """[(tile_a, tile_b, direction, (x0, x1, y0, y1))]: the XML overlap Stage 3 uses."""
    names = grid_names(geom)
    rc_of = {n: rc for rc, n in names.items()}
    t = geom['tile_size']
    out = []
    for a, b, direction in adjacent_seams(geom):
        ax, ay, _ = tile_origin(geom, rc_of[a])
        bx, by, _ = tile_origin(geom, rc_of[b])
        out.append((a, b, direction, (max(ax, bx), min(ax, bx) + t, max(ay, by), min(ay, by) + t)))
    return out


def _inside(x, y, rect, pad=0.0):
    return (x >= rect[0] - pad) & (x <= rect[1] + pad) & (y >= rect[2] - pad) & (y <= rect[3] + pad)


def one_to_one_pairs(a, b, radius, check_z=True):
    """Pairs within radius (nearest first, each cell used once).

    a, b: (n, 4) [cx, cy, z_lo, z_hi]; with check_z the z ranges must overlap.
    """
    if not len(a) or not len(b):
        return 0
    sm = cKDTree(a[:, :2]).sparse_distance_matrix(cKDTree(b[:, :2]), radius, output_type='ndarray')
    i, j, d = sm['i'], sm['j'], sm['v']
    if check_z:
        keep = (a[i, 2] <= b[j, 3]) & (b[j, 2] <= a[i, 3])
        i, j, d = i[keep], j[keep], d[keep]
    order = np.argsort(d, kind='stable')
    used_a, used_b, n = set(), set(), 0
    for p, q in zip(i[order].tolist(), j[order].tolist()):
        if p not in used_a and q not in used_b:
            used_a.add(p)
            used_b.add(q)
            n += 1
    return n


# ── per-channel metrics ───────────────────────────────────────────────────────

def _cells_3d(channel_3d_dir, ch_id):
    """[cx, cy, z_min, z_max] and majority source tile of every Z-linked cell."""
    cells = pd.read_csv(os.path.join(channel_3d_dir, f'{ch_id}_3d_tracked.csv'),
                        usecols=['x1', 'y1', 'x2', 'y2', 'z_min', 'z_max', 'track_id'],
                        dtype={'track_id': str})
    members = pd.read_csv(os.path.join(channel_3d_dir, f'{ch_id}_track_members.csv'),
                          usecols=['track_id', 'tile_name'], dtype=str)
    counts = members.groupby(['track_id', 'tile_name']).size().reset_index(name='n')
    major = counts.sort_values(['track_id', 'n', 'tile_name'], ascending=[True, False, True]) \
                  .drop_duplicates('track_id').set_index('track_id').tile_name
    info = np.column_stack([(cells.x1 + cells.x2) / 2, (cells.y1 + cells.y2) / 2,
                            cells.z_min, cells.z_max]).astype(float)
    return info, cells.track_id.map(major).fillna('').to_numpy(dtype=object)


def _global_bands(global_csv, rects):
    """Rows of the global 2D CSV whose centre lies near any seam overlap."""
    cols = ['x1', 'y1', 'x2', 'y2', 'class', 'z', 'detection_id', 'tile_name']
    parts = []
    for df in pd.read_csv(global_csv, usecols=cols, chunksize=3_000_000,
                          dtype={'detection_id': str, 'tile_name': str, 'class': str}):
        cx, cy = ((df.x1 + df.x2) / 2).to_numpy(), ((df.y1 + df.y2) / 2).to_numpy()
        keep = np.zeros(len(df), bool)
        for *_, rect in rects:
            keep |= _inside(cx, cy, rect, _PAD)
        parts.append(df[keep])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def channel_metrics(ch, paths, geom, offsets, options):
    """3D and 2D residual duplicates of one channel; returns (seams, segments, decisions)."""
    ch_id, soma = ch['id'], ch.get('type', 'soma') == 'soma'
    rects = seam_rects(geom)
    edge, shift = options['edge_px'], options['null_shift_px']
    r3 = options['pair_radius_soma_px'] if soma else options['pair_radius_tf_px']
    r2 = options['pair_radius_2d_soma_px'] if soma else options['pair_radius_2d_tf_px']
    seg_layers = int(geom.get('seam_seg_layers', 50))

    info, major = _cells_3d(paths['channel_3d'], ch_id)
    bands = _global_bands(os.path.join(paths['global_2d'], f'{ch_id}_2d_global.csv'), rects)
    bx = ((bands.x1 + bands.x2) / 2).to_numpy()
    by = ((bands.y1 + bands.y2) / 2).to_numpy()
    blayer = bands.z.to_numpy(float) - 1
    btile = bands.tile_name.to_numpy(dtype=object)
    own = offsets[(offsets.channel.astype(str) == ch_id) & (offsets.source == 'own')]
    n_dup = {(r.tile, r.nb_tile, int(r.seg)): int(r.n_dup) for r in own.itertuples()}

    seams, segments = [], []
    for a, b, direction, rect in rects:
        core = (rect[0] + edge, rect[1] - edge, rect[2] + edge, rect[3] - edge)
        inside = _inside(info[:, 0], info[:, 1], core)
        A, B = info[inside & (major == a)], info[inside & (major == b)]
        pairs = one_to_one_pairs(A, B, r3)
        moved = B.copy()
        moved[:, :2] += shift
        null = one_to_one_pairs(A, moved, r3)
        row = {'channel': ch_id, 'tile': a, 'nb_tile': b, 'direction': direction,
               'cells_a': len(A), 'cells_b': len(B), 'pairs_3d': pairs, 'null_3d': null,
               'residual_3d': max(0, pairs - null)}
        row['residual_3d_pct'] = 100.0 * row['residual_3d'] / len(A) if len(A) else 0.0
        # 2D: same-layer pairs per segment
        inside2 = _inside(bx, by, core)
        sel_a, sel_b = inside2 & (btile == a), inside2 & (btile == b)
        pts_a = np.column_stack([bx[sel_a], by[sel_a], blayer[sel_a], blayer[sel_a]])
        pts_b = np.column_stack([bx[sel_b], by[sel_b], blayer[sel_b], blayer[sel_b]])
        seg_a, seg_b = (pts_a[:, 2] // seg_layers).astype(int), (pts_b[:, 2] // seg_layers).astype(int)
        total2, measured2 = 0, 0
        for seg in sorted(set(seg_a.tolist()) | set(n_dup_seg for (ta, tb, n_dup_seg) in n_dup
                                                    if (ta, tb) == (a, b))):
            pa, pb = pts_a[seg_a == seg], pts_b[seg_b == seg]
            p2 = one_to_one_pairs(pa, pb, r2)
            nb_ = pb.copy()
            nb_[:, :2] += shift
            n2 = one_to_one_pairs(pa, nb_, r2)
            dup = n_dup.get((a, b, seg), 0)
            segments.append({'channel': ch_id, 'tile': a, 'nb_tile': b, 'seg': seg,
                             'boxes_a': len(pa), 'boxes_b': len(pb), 'pairs_2d': p2,
                             'null_2d': n2, 'residual_2d': max(0, p2 - n2),
                             'duplicates_before_dedup': dup})
            if dup:
                total2 += max(0, p2 - n2)
                measured2 += dup
        row['residual_2d'] = total2
        row['duplicates_before_dedup'] = measured2
        row['residual_2d_pct'] = 100.0 * total2 / measured2 if measured2 else np.nan
        seams.append(row)

    decisions_path = os.path.join(paths['global_2d'], f'{ch_id}_stitch_decisions.csv')
    decisions = {}
    if os.path.isfile(decisions_path):
        dec = pd.read_csv(decisions_path, usecols=['decision', 'reason'], dtype=str)
        decisions = {f'{d}:{r}': int(n) for (d, r), n in dec.groupby(['decision', 'reason']).size().items()}
    return pd.DataFrame(seams), pd.DataFrame(segments), {'channel': ch_id, **decisions}


# ── review images ─────────────────────────────────────────────────────────────

def _font(size):
    from PIL import ImageFont
    try:
        import matplotlib
        return ImageFont.truetype(os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf', 'DejaVuSans.ttf'), size)
    except Exception:
        return ImageFont.load_default()


class _Renderer:
    """Tile images placed in Stage 3 global coordinates (XML + Stage 2.5 offset)."""

    def __init__(self, geom, image_root, align_dir):
        from src.utils.io import resolve_tile_dir
        self.geom, self.root, self.align_dir = geom, image_root, align_dir
        self.rc_of = {n: rc for rc, n in grid_names(geom).items()}
        self.dir_of = {os.path.basename(d): d for d in geom['dir_dict']}
        self._resolve = resolve_tile_dir

    @lru_cache(maxsize=None)
    def offset(self, tile, ch_id):
        path = os.path.join(self.align_dir or '', f'{tile}_offsets.json')
        if self.align_dir and os.path.isfile(path):
            with open(path, encoding='utf-8') as handle:
                value = json.load(handle).get(ch_id)
            if value:
                return int(value['dx']), int(value['dy']), int(value['dz'])
        return 0, 0, 0

    @lru_cache(maxsize=None)
    def slices(self, tile):
        folder = str(self._resolve(self.root, self.dir_of[tile]))
        if not os.path.isdir(folder):
            return []
        return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                      if f.lower().endswith(('.tif', '.tiff')) and not f.startswith('.'))

    def render(self, ch_id, tile, layer, window, shift=(0, 0, 0)):
        import tifffile
        x0, x1, y0, y1 = window
        ox, oy, z0 = tile_origin(self.geom, self.rc_of[tile])
        dx, dy, dz = self.offset(tile, ch_id)
        raw_z = int(layer + round(shift[2])) + z0 + 1 - dz
        out = np.full((y1 - y0, x1 - x0), np.nan, np.float32)
        files = self.slices(tile)
        if not 1 <= raw_z <= len(files):
            return out
        img = tifffile.imread(files[raw_z - 1])
        sx = x0 + int(round(shift[0])) - int(ox) - dx
        sy = y0 + int(round(shift[1])) - int(oy) - dy
        h, w = img.shape[:2]
        ys, xs, ye, xe = max(0, sy), max(0, sx), min(h, sy + y1 - y0), min(w, sx + x1 - x0)
        if ye > ys and xe > xs:
            out[ys - sy:ye - sy, xs - sx:xe - sx] = img[ys:ye, xs:xe]
        return out


def _norm(img):
    v = img[np.isfinite(img)]
    if not v.size:
        return np.zeros_like(img)
    lo, hi = np.percentile(v, [1, 99.7])
    return np.nan_to_num(np.clip((img - lo) / max(hi - lo, 1), 0, 1))


def _window(geom, rc_of, a, b, bands=None, layer=None):
    """Crop around the XML overlap: across the seam the whole overlap plus a margin,
    along it 2 * _HALF_LEN px centred on the densest stretch of boxes on the layer."""
    t = geom['tile_size']
    ax, ay, _ = tile_origin(geom, rc_of[a])
    bx, by, _ = tile_origin(geom, rc_of[b])
    east = rc_of[b][1] > rc_of[a][1]
    lo, hi = (max(ay, by), min(ay, by) + t) if east else (max(ax, bx), min(ax, bx) + t)
    centre = (lo + hi) / 2
    if bands is not None and len(bands):
        rect = (max(ax, bx), min(ax, bx) + t, max(ay, by), min(ay, by) + t)
        cx, cy = (bands.x1 + bands.x2) / 2, (bands.y1 + bands.y2) / 2
        on = (bands.z == layer + 1) & _inside(cx, cy, rect)
        along = (cy if east else cx)[on].to_numpy(float)
        if along.size:
            edges = np.arange(lo, hi + 32, 32)
            counts, _ = np.histogram(along, bins=edges)
            k = max(1, int(2 * _HALF_LEN // 32))
            sums = np.convolve(counts, np.ones(k), mode='valid')
            start = int(np.argmax(sums))
            centre = float(edges[start] + k * 16)
    if east:
        return (int(bx) - _MARGIN, int(ax + t) + _MARGIN, int(centre - _HALF_LEN), int(centre + _HALF_LEN))
    return (int(centre - _HALF_LEN), int(centre + _HALF_LEN), int(by) - _MARGIN, int(ay + t) + _MARGIN)


def review_image(ch, a, b, layer, geom, renderer, offset_row, bands, decisions, src_dir, out_png):
    """Three panels in tile A's frame: A + moved B, final boxes, B's removed/kept boxes."""
    from PIL import Image, ImageDraw
    rc_of = renderer.rc_of
    window = _window(geom, rc_of, a, b, bands, layer)
    x0, x1, y0, y1 = window
    r = (float(offset_row.dx), float(offset_row.dy), int(offset_row.dz))
    na = _norm(renderer.render(ch, a, layer, window))
    nb = _norm(renderer.render(ch, b, layer, window, r))

    def in_win(df):
        cx, cy = (df.x1 + df.x2) / 2, (df.y1 + df.y2) / 2
        return df[(cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)]

    final = in_win(bands[bands.z == layer + 1])
    rej_ids = set(decisions[(decisions.tile_name == b) & (decisions.decision == 'rejected') &
                            (decisions.z == layer + r[2] + 1)].detection_id)
    removed = pd.DataFrame(columns=['x1', 'y1', 'x2', 'y2'])
    if rej_ids:
        ox, oy, _ = tile_origin(geom, rc_of[b])
        src = pd.read_csv(os.path.join(src_dir, f'{b}_{ch}_result.csv'),
                          usecols=['x1', 'y1', 'x2', 'y2', 'detection_id'], dtype={'detection_id': str})
        removed = src[src.detection_id.isin(rej_ids)].copy()
        removed[['x1', 'x2']] += ox - r[0]
        removed[['y1', 'y2']] += oy - r[1]
        removed = in_win(removed)
    relabelled = set(decisions[decisions.decision == 'relabelled'].detection_id)

    def rgb(img):
        img = np.dstack([img] * 3) if img.ndim == 2 else img
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(img).resize((img.shape[1] * _SCALE, img.shape[0] * _SCALE), Image.NEAREST)

    def boxes(canvas, df, color, width=2):
        d = ImageDraw.Draw(canvas)
        for q in df.itertuples():
            d.rectangle([(q.x1 - x0) * _SCALE, (q.y1 - y0) * _SCALE,
                         (q.x2 - x0) * _SCALE, (q.y2 - y0) * _SCALE], outline=color, width=width)

    def edges(canvas):
        t = geom['tile_size']
        ax, ay, _ = tile_origin(geom, rc_of[a])
        bx, by, _ = tile_origin(geom, rc_of[b])
        d = ImageDraw.Draw(canvas)
        W, H = canvas.size
        if rc_of[b][1] > rc_of[a][1]:
            for x in (bx - r[0], ax + t):
                for y in range(0, H, 16):
                    d.line([((x - x0) * _SCALE, y), ((x - x0) * _SCALE, y + 8)], fill=(255, 255, 0))
        else:
            for y in (by - r[1], ay + t):
                for x in range(0, W, 16):
                    d.line([(x, (y - y0) * _SCALE), (x + 8, (y - y0) * _SCALE)], fill=(255, 255, 0))

    panels = [rgb(np.dstack([nb, na, nb])), rgb(na), rgb(nb)]
    for canvas in panels:
        edges(canvas)
    from_a, from_b = final[final.tile_name == a], final[final.tile_name == b]
    boxes(panels[1], from_a, (0, 255, 0))
    boxes(panels[1], from_b, (255, 0, 255))
    boxes(panels[1], final[~final.tile_name.isin([a, b])], (170, 170, 170))
    boxes(panels[1], final[final.detection_id.isin(relabelled)], (0, 255, 255), 3)
    boxes(panels[2], removed, (255, 40, 40))
    boxes(panels[2], from_b, (255, 0, 255))
    titles = ['1 tile A (green) + tile B moved by the measured offset (magenta)',
              '2 final boxes: green = from A, magenta = from B (moved), cyan = relabelled glia',
              "3 tile B (moved), B's boxes: red = removed duplicate, magenta = kept"]
    pw, ph = panels[0].size
    gap, top, head = 12, 58, 30
    out = Image.new('RGB', (3 * pw + 2 * gap, top + head + ph))
    d = ImageDraw.Draw(out)
    big, small = _font(18), _font(15)
    d.text((8, 4), f'{ch}  seam {a} -> {b}  layer {layer}  offset B-A dx={r[0]:.1f} dy={r[1]:.1f} '
                   f'dz={r[2]} [{offset_row.source}]  pairs={int(offset_row.n_dup)}',
           fill=(255, 255, 255), font=big)
    d.text((8, 32), f'in crop: kept A={len(from_a)} B={len(from_b)}, removed B={len(removed)}; '
                    f'yellow dashes = edges of the two tile views', fill=(220, 220, 220), font=small)
    for k, (canvas, title) in enumerate(zip(panels, titles)):
        d.text((k * (pw + gap) + 4, top + 6), title, fill=(255, 255, 255), font=small)
        out.paste(canvas, (k * (pw + gap), top + head))
    out.save(out_png + '.part.png')
    os.replace(out_png + '.part.png', out_png)


def sample_seam_segments(ch_id, offsets, options):
    """Random (tile, nb_tile, segment row) picks among this channel's measured segments.

    Distinct seams first; if there are fewer seams than images, further segments of
    already drawn seams. The seed depends on image_seed and the channel only.
    """
    n = int(options['images_per_channel'])
    own = offsets[(offsets.channel.astype(str) == ch_id) & (offsets.source == 'own')]
    if n <= 0 or not len(own):
        return []
    rng = np.random.default_rng([int(options['image_seed']), zlib.crc32(ch_id.encode())])
    by_seam = {key: g.sort_values('seg') for key, g in own.groupby(['tile', 'nb_tile'])}
    order = [list(by_seam)[k] for k in rng.permutation(len(by_seam))]
    used, picks = set(), []
    while len(picks) < n and len(used) < len(own):
        progress = False
        for key in order:
            free = by_seam[key][~by_seam[key].seg.isin([s for (k, s) in used if k == key])]
            if not len(free):
                continue
            row = free.iloc[int(rng.integers(len(free)))]
            used.add((key, int(row.seg)))
            picks.append((key[0], key[1], row))
            progress = True
            if len(picks) == n:
                break
        if not progress:
            break
    return picks


def channel_images(ch, paths, geom, offsets, options, image_dir):
    """Randomly sampled review PNGs for one channel; existing files are skipped."""
    ch_id = ch['id']
    if not paths.get('image_root'):
        return []
    seg_layers = int(geom.get('seam_seg_layers', 50))
    jobs = []
    for a, b, row in sample_seam_segments(ch_id, offsets, options):
        layer = min(int(row.seg) * seg_layers + seg_layers // 2, int(geom['Z']) - 1)
        png = os.path.join(image_dir, f'{ch_id}_{a}__{b}_layer{layer:03d}.png')
        if not os.path.isfile(png):
            jobs.append((a, b, row, layer, png))
    if not jobs:
        return []
    wanted = {(a, b) for a, b, *_ in jobs}
    rects = [r for r in seam_rects(geom) if (r[0], r[1]) in wanted]
    bands = _global_bands(os.path.join(paths['global_2d'], f'{ch_id}_2d_global.csv'), rects)
    decisions = pd.read_csv(os.path.join(paths['global_2d'], f'{ch_id}_stitch_decisions.csv'),
                            dtype={'detection_id': str, 'tile_name': str})
    renderer = _Renderer(geom, paths['image_root'], paths.get('align_dir'))
    written = []
    for a, b, row, layer, png in jobs:
        review_image(ch_id, a, b, layer, geom, renderer, row, bands, decisions, paths['src_dir'], png)
        written.append(png)
    return written


# ── stage ─────────────────────────────────────────────────────────────────────

def _channel_paths(ch, src_dirs, derived, config):
    image_root = (config.get('paths') or {}).get(ch.get('dir_key', ''))
    return {'src_dir': src_dirs[ch['id']] if isinstance(src_dirs, dict) else src_dirs,
            'global_2d': derived['pATH_GLOBAL_2D'], 'channel_3d': derived['pATH_CHANNEL_3D'],
            'image_root': image_root,
            'align_dir': derived.get('pATH_ALIGN_OFFSETS') if config.get('pipeline_mode') == 'pre_align' else None}


def _signature(routing, derived, geom, options):
    channels = {}
    for ch in routing:
        files = [os.path.join(derived['pATH_GLOBAL_2D'], f"{ch['id']}_2d_global.csv"),
                 os.path.join(derived['pATH_GLOBAL_2D'], f"{ch['id']}_stitch_decisions.csv"),
                 os.path.join(derived['pATH_CHANNEL_3D'], f"{ch['id']}_3d_tracked.csv"),
                 os.path.join(derived['pATH_CHANNEL_3D'], f"{ch['id']}_track_members.csv")]
        channels[ch['id']] = [file_stamp(p) if os.path.isfile(p) else None for p in files]
    return {
        'version': 1,
        'metrics': {k: options[k] for k in _METRIC_KEYS},
        'channels': channels,
        'seam_offsets': file_stamp(geom['seam_offsets_csv'], with_hash=True),
        'geometry': [(name, [int(v) for v in geom['disp_mat_fin'][rc]])
                     for rc, name in sorted(grid_names(geom).items())],
        'code_sha256': file_sha256(os.path.abspath(__file__)),
    }


def _channel_job(args):
    ch, paths, geom, offsets_path, options = args
    offsets = pd.read_csv(offsets_path)
    return channel_metrics(ch, paths, geom, offsets, options)


def _image_job(args):
    ch, paths, geom, offsets_path, options, image_dir = args
    return channel_images(ch, paths, geom, pd.read_csv(offsets_path), options, image_dir)


def _pool_map(function, jobs, workers):
    if workers <= 1 or len(jobs) <= 1:
        return [function(job) for job in jobs]
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=mp.get_context('spawn')) as pool:
        return list(pool.map(function, jobs))


def _summary(seams, decisions):
    rows = []
    for ch, g in seams.groupby('channel'):
        dup = g.duplicates_before_dedup.sum()
        rows.append({'channel': ch, 'seams': len(g), 'cells_a': int(g.cells_a.sum()),
                     'residual_3d': int(g.residual_3d.sum()),
                     'residual_3d_pct': 100.0 * g.residual_3d.sum() / max(1, g.cells_a.sum()),
                     'seam_residual_3d_pct_p50': float(g.residual_3d_pct.median()),
                     'seam_residual_3d_pct_max': float(g.residual_3d_pct.max()),
                     'residual_2d_same_layer_pct': 100.0 * g.residual_2d.sum() / dup if dup else np.nan,
                     **{k: v for k, v in decisions.get(ch, {}).items() if k != 'channel'}})
    return pd.DataFrame(rows)


def run(config, results_dir, geom, routing, src_dirs, derived, n_workers=None):
    """Compute (or reuse) the QC tables and write missing review PNGs."""
    options = resolve_options(config)
    out_dir = os.path.join(results_dir, options['output_dir'])
    image_dir = os.path.join(out_dir, 'images')
    os.makedirs(image_dir, exist_ok=True)
    workers = int(n_workers or options['workers'])
    tables = {name: os.path.join(out_dir, f'{name}.csv') for name in ('seams', 'segments_2d', 'decisions')}
    manifest = os.path.join(out_dir, '_stitch_dedup_qc_manifest.json')
    signature = _signature(routing, derived, geom, options)
    paths = {ch['id']: _channel_paths(ch, src_dirs, derived, config) for ch in routing}
    offsets_path = geom['seam_offsets_csv']
    if output_manifest_valid(manifest, signature):
        logging.info("✔️ [QC] Seam dedup QC tables up to date: %s", out_dir)
        seams = pd.read_csv(tables['seams'], dtype={'tile': str, 'nb_tile': str, 'channel': str})
        dec_df = pd.read_csv(tables['decisions'], dtype={'channel': str})
    else:
        logging.info("[QC] Measuring residual seam duplicates (%d channels)...", len(routing))
        for stale in glob.glob(os.path.join(image_dir, '*.png')):
            os.remove(stale)          # images of older Stage 3 results
        results = _pool_map(_channel_job, [(ch, paths[ch['id']], geom, offsets_path, options)
                                           for ch in routing], workers)
        seams = pd.concat([r[0] for r in results], ignore_index=True)
        segments = pd.concat([r[1] for r in results], ignore_index=True)
        dec_df = pd.DataFrame([r[2] for r in results])
        for name, df in (('seams', seams), ('segments_2d', segments), ('decisions', dec_df)):
            df.to_csv(tables[name] + '.part', index=False)
            os.replace(tables[name] + '.part', tables[name])
        write_output_manifest(manifest, list(tables.values()), signature)
    decisions = {str(r['channel']): r for r in dec_df.to_dict('records')}
    summary = _summary(seams, decisions)
    summary.to_csv(os.path.join(out_dir, 'summary.csv'), index=False)
    written = _pool_map(_image_job, [(ch, paths[ch['id']], geom, offsets_path, options, image_dir)
                                     for ch in routing], workers)
    written = [p for part in written for p in part]
    _report(out_dir, summary)
    return {'status': 'ok', 'summary': summary, 'images_written': written,
            'report': os.path.join(out_dir, 'qc_report.md')}


def _report(out_dir, summary):
    lines = ['# Seam deduplication QC', '',
             'Residual duplicates in the XML overlap of each seam after Stage 3.',
             '3D: Z-linked cells mostly from tile A vs mostly from tile B, minus a null count;',
             '% of tile A cells in the overlap. 2D: same-layer box pairs as % of the duplicates',
             'Stage 2.97 measured before deduplication.', '',
             '| channel | seams | 3D residual % | seam p50 % | seam max % | 2D same-layer % |',
             '|---|---|---|---|---|---|']
    for r in summary.itertuples():
        lines.append(f'| {r.channel} | {r.seams} | {r.residual_3d_pct:.2f} | {r.seam_residual_3d_pct_p50:.2f} | '
                     f'{r.seam_residual_3d_pct_max:.2f} | {r.residual_2d_same_layer_pct:.2f} |')
    lines += ['', 'Review images: images/<channel>_<tileA>__<tileB>_layer<L>.png (randomly sampled seams).',
              'Panel 1: tile A green, tile B moved by the measured offset magenta; the two copies of',
              'a cell should merge into white. Panel 2: final boxes, green from A, magenta from B',
              '(moved into A\'s frame), cyan relabelled glia; magenta should only mark cells A has no',
              'box for. Panel 3: tile B, red = removed as duplicate (each should have a green box at',
              'the same place in panel 2), magenta = kept.']
    with open(os.path.join(out_dir, 'qc_report.md'), 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def main(argv=None):
    """Run the QC on an existing result directory (uses its runtime_config.json)."""
    from src.core.result_layout import result_paths
    from src.core.resume import channel_sources
    from src.core.seam_offsets import resolve_params
    from src.utils.io import loadTeraxml
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('results_dir')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--path-map', nargs=2, metavar=('FROM', 'TO'), default=None,
                        help='rewrite path prefixes in the saved config, e.g. /rsstu/users/a/agrinba/DeepDesign Y:')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    with open(os.path.join(args.results_dir, 'runtime_config.json'), encoding='utf-8') as handle:
        text = handle.read()
    if args.path_map:
        text = text.replace(args.path_map[0], args.path_map[1])
    config = json.loads(text)
    derived = result_paths(args.results_dir)
    routing = [ch for ch in config['channels_routing'] if ch.get('active', True)]
    xml = (config.get('paths') or {}).get('pATHXML')
    tile_size = int((config.get('detection_params') or {}).get('tILESIZE', 2048))
    dir_dict, H, W, Z, z_start, disp = loadTeraxml(xml, tile_size)
    _, seam_params = resolve_params(config)
    geom = {'dir_dict': dir_dict, 'disp_mat_fin': disp, 'z_start': z_start, 'Z': Z, 'H': H, 'W': W,
            'tile_size': tile_size, 'num_tiles': len(dir_dict),
            'seam_offsets_csv': os.path.join(derived['pATH_GLOBAL_2D'], 'seam_offsets', 'seam_offsets.csv'),
            'seam_seg_layers': seam_params['seg_layers']}
    report = run(config, args.results_dir, geom, routing, channel_sources(config, derived), derived,
                 args.workers)
    print(report['summary'].to_string(index=False))
    print(f"images written: {len(report['images_written'])}")


if __name__ == '__main__':
    main()
