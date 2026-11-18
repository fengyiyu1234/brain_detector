"""QC B: do the Stage 2.9 tile positions agree with the seam measurements?

Residual per seam, channel and axis: (P_ch[nb] - P_ch[tile]) - meas_d, from
5_2d_global/tile_positions/{seams,tile_positions}.csv (ok seams only); the same numbers as
report.txt. A channel exceeds when |rx| or |ry| > max_xy_px or |rz| > max_z_slices.

* ``consistent``: >= 2 channels exceed on the seam and their residual vectors point the same
  way (pairwise cosine >= agree_cos, or difference < agree_px; z converted to px by
  z_um / xy_um) -> the tile position itself is suspect -> WARN.
* ``single_outlier``: one channel exceeds and it has < min_match matches -> one bad measurement.
* ``single``/``mixed``: recorded only.

Review images (worst consistent seams): the overlap of the two raw tiles (no Stage 2.5 shift,
the frame Stage 2.9 measures and solves in), one row for the channel with the largest residual
and one for the stitching frame channel; tile A green, tile B magenta, left placed by the solved
positions, right by the measured displacement, at up to layers_per_item real slices.

Outputs: qc/seam_consistency/{seams.csv,summary.json,report.md}, qc/review/seams/*.png.
"""
import json
import os

import numpy as np
import pandas as pd

from .common import save_json
from . import review_images as rv

DEFAULTS = {'max_xy_px': 20.0, 'max_z_slices': 3.0, 'agree_px': 10.0, 'agree_cos': 0.5, 'min_match': 100}


def options_for(config):
    out = dict(DEFAULTS)
    out.update({k: v for k, v in (((config.get('qc') or {}).get('seam_consistency')) or {}).items()
                if k in DEFAULTS})
    return out


def residuals(seams, positions):
    """seams.csv rows with status ok plus rx, ry, rz (px, px, slices)."""
    ok = seams[seams.status == 'ok'].copy()
    pos = positions.set_index('tile')
    for axis in 'xyz':
        def p(tiles, ch):
            col = f'P_{ch}_{axis}'
            return pos[col].reindex(tiles).to_numpy(float) if col in pos else np.full(len(tiles), np.nan)
        solved = np.empty(len(ok))
        for ch, idx in ok.groupby('channel').groups.items():
            rows = ok.loc[idx]
            solved[ok.index.get_indexer(idx)] = p(rows.nb_tile, ch) - p(rows.tile, ch)
        ok[f'solved_d{axis}'] = solved
        ok[f'r{axis}'] = solved - ok[f'meas_d{axis}'].to_numpy(float)
    return ok


def _agree(v, options):
    if len(v) < 2:
        return False
    norm = np.linalg.norm(v, axis=1)
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            if np.linalg.norm(v[i] - v[j]) < options['agree_px']:
                continue
            if norm[i] == 0 or norm[j] == 0 or v[i] @ v[j] / (norm[i] * norm[j]) < options['agree_cos']:
                return False
    return True


def classify(res, options, z_px=8.0 / 0.65):
    rows = []
    for (a, b, direction), g in res.groupby(['tile', 'nb_tile', 'direction'], sort=False):
        exceed = g[(g.rx.abs() > options['max_xy_px']) | (g.ry.abs() > options['max_xy_px'])
                   | (g.rz.abs() > options['max_z_slices'])]
        status = 'ok'
        if len(exceed) >= 2:
            v = exceed[['rx', 'ry', 'rz']].to_numpy(float) * [1, 1, z_px]
            status = 'consistent' if _agree(v, options) else 'mixed'
        elif len(exceed) == 1:
            status = 'single_outlier' if exceed.n_match.iloc[0] < options['min_match'] else 'single'
        worst = g.loc[np.hypot(g.rx, g.ry).idxmax()]
        rows.append({'tile': a, 'nb_tile': b, 'direction': direction, 'status': status,
                     'n_channels': len(g), 'n_exceed': len(exceed),
                     'exceed_channels': ','.join(exceed.channel.astype(str)),
                     'max_abs_rx': float(g.rx.abs().max()), 'max_abs_ry': float(g.ry.abs().max()),
                     'max_abs_rz': float(g.rz.abs().max()),
                     'mean_rx': float(exceed.rx.mean()) if len(exceed) else 0.0,
                     'mean_ry': float(exceed.ry.mean()) if len(exceed) else 0.0,
                     'worst_channel': str(worst.channel)})
    out = pd.DataFrame(rows)
    if len(out):
        out['severity'] = np.hypot(out.max_abs_rx, out.max_abs_ry)
        out = out.sort_values('severity', ascending=False)
    return out


def seam_review(images, results_dir, seam, res_rows, out_dir, review, tile_size, frame=None):
    """Overlap images of one seam: one row per channel (worst channel, then the frame channel).

    Stage 2.9 measures seams and solves positions on the raw tiles (2_2d_filtered, no
    Stage 2.5 channel shift), so both tiles are read unshifted and B is placed at A + d.
    Adding the Stage 2.5 shifts here would move B by shift_B - shift_A on top of d.
    """
    ch = seam.worst_channel
    channels = [ch] + ([frame] if frame and frame != ch and (res_rows.channel == frame).any() else [])
    rows = {c: res_rows[res_rows.channel == c].iloc[0] for c in channels}
    r = rows[ch]
    a, b = seam.tile, seam.nb_tile
    meas = (r.meas_dx, r.meas_dy, r.meas_dz)
    east = seam.direction == 'EAST'
    size = int(review['crop_px'])
    margin = 40
    raw = (0, 0, 0)
    # overlap in tile A's raw frame, using the measured displacement
    if east:
        x0, x1 = int(max(0, meas[0] - margin)), int(min(tile_size, tile_size + margin))
    else:
        y0, y1 = int(max(0, meas[1] - margin)), int(min(tile_size, tile_size + margin))
    csv = os.path.join(results_dir, '2_2d_filtered', f'{a}_{ch}_result.csv')
    boxes = pd.read_csv(csv, usecols=['x1', 'y1', 'x2', 'y2', 'z']) if os.path.isfile(csv) else \
        pd.DataFrame(columns=['x1', 'y1', 'x2', 'y2', 'z'])
    cx, cy = (boxes.x1 + boxes.x2) / 2, (boxes.y1 + boxes.y2) / 2
    in_ov = boxes[(cx >= meas[0]) if east else (cy >= meas[1])]
    zs = in_ov.z.value_counts()
    zs = zs[zs >= max(1, zs.max() * 0.2)].index.tolist() if len(zs) else []

    def real(z):
        return all(images.is_real(c, a, z) and images.is_real(c, b, int(round(z - rows[c].meas_dz)))
                   for c in channels)

    zs = rv.pick_layers(zs, review['layers_per_item'], ok=real)
    written = []
    for z in zs:
        on = in_ov[in_ov.z == z]
        along = ((on.y1 + on.y2) / 2 if east else (on.x1 + on.x2) / 2).to_numpy(float)
        centre = float(np.median(along)) if along.size else tile_size / 2
        lo = int(np.clip(centre - size // 2, 0, max(0, tile_size - size)))
        window = (x0, x1, lo, lo + size) if east else (lo, lo + size, y0, y1)
        grid = []
        for c in channels:
            rc = rows[c]
            img_a = images.read(c, a, z, window, raw)
            na = rv.stretch(img_a)
            tag = f'{c}{" (frame)" if c == frame else ""}'
            panels = []
            for label, d in (('solved (Stage 2.9)', (rc.solved_dx, rc.solved_dy, rc.solved_dz)),
                             ('measured', (rc.meas_dx, rc.meas_dy, rc.meas_dz))):
                img_b = images.read(c, b, z, window, d)
                nb = rv.stretch(img_b)
                m = np.isfinite(img_a) & np.isfinite(img_b)
                corr = float(np.corrcoef(img_a[m], img_b[m])[0, 1]) if m.sum() > 100 else float('nan')
                panels.append((rv.to_pil(np.dstack([nb, na, nb])),
                               f'{tag} {label}: B-A = ({d[0]:.1f}, {d[1]:.1f}, {d[2]:.1f}), r={corr:.2f}'))
            grid.append(panels)
        png = os.path.join(out_dir, f'{ch}_{a}__{b}_z{z:04d}.png')
        res_text = '; '.join(f'{c} x {rows[c].rx:+.1f} y {rows[c].ry:+.1f} px z {rows[c].rz:+.1f}'
                             for c in channels)
        written.append(rv.grid_png(
            grid, png, f'seam {a} -> {b} ({seam.direction}), raw tiles, tile A slice {z}',
            f'A green, B magenta; matching cells turn white; r = overlap correlation. '
            f'residual {res_text}; exceeding: {seam.exceed_channels}'))
    return written


def run(config, results_dir, force=False):
    options = options_for(config)
    review = rv.review_options(config)
    tp = os.path.join(results_dir, '5_2d_global', 'tile_positions')
    out_dir = os.path.join(results_dir, 'qc', 'seam_consistency')
    seams_csv, pos_csv = os.path.join(tp, 'seams.csv'), os.path.join(tp, 'tile_positions.csv')
    if not (os.path.isfile(seams_csv) and os.path.isfile(pos_csv)):
        summary = {'status': 'NOT_RUN', 'reason': 'no Stage 2.9 seams.csv/tile_positions.csv'}
        save_json(os.path.join(out_dir, 'summary.json'), summary)
        return summary
    seams = pd.read_csv(seams_csv, dtype={'tile': str, 'nb_tile': str, 'channel': str})
    positions = pd.read_csv(pos_csv, dtype={'tile': str})
    res = residuals(seams, positions)
    dp = config.get('detection_params') or {}
    table = classify(res, options, float(dp.get('z_resolution_um', 8.0)) / float(dp.get('xy_resolution_um', 0.65)))
    os.makedirs(out_dir, exist_ok=True)
    res.to_csv(os.path.join(out_dir, 'seam_channel_residuals.csv'), index=False)
    table.to_csv(os.path.join(out_dir, 'seams.csv'), index=False)
    consistent = table[table.status == 'consistent'] if len(table) else table
    per_axis = {f'max_abs_r{a}': float(res[f'r{a}'].abs().max()) if len(res) else 0.0 for a in 'xyz'}
    summary = {'status': 'WARN' if len(consistent) else 'PASS',
               'counts': table.status.value_counts().to_dict() if len(table) else {},
               'consistent_seams': [f'{r.tile}->{r.nb_tile}' for r in consistent.itertuples()],
               **per_axis, 'table': os.path.join(out_dir, 'seams.csv'), 'images': []}
    image_dir = os.path.join(results_dir, 'qc', 'review', 'seams')
    picks = consistent.head(int(review['max_items_per_check'])) if len(consistent) else consistent
    if review['enabled'] and len(picks):
        manifest = os.path.join(image_dir, '_manifest.json')
        sig = rv.review_signature([seams_csv, pos_csv], {'picks': [f'{r.tile}->{r.nb_tile}' for r in picks.itertuples()],
                                                         'review': review, 'module': _code()})
        if not force and rv.images_current(manifest, sig):
            with open(manifest, encoding='utf-8') as handle:
                summary['images'] = json.load(handle)['images']
        else:
            images = rv.TileImages(config, results_dir, review['min_image_bytes'])
            if images.available():
                rv.clear_pngs(image_dir)
                tile_size = int((config.get('detection_params') or {}).get('tILESIZE', 2048))
                frame = config.get('stitching_reference_channel')
                written = []
                for seam in picks.itertuples():
                    rows = res[(res.tile == seam.tile) & (res.nb_tile == seam.nb_tile)]
                    written += seam_review(images, results_dir, seam, rows, image_dir, review, tile_size,
                                           frame)
                rv.save_images_manifest(manifest, sig, written)
                summary['images'] = written
            else:
                summary['images_note'] = 'image folders not reachable'
    _report(out_dir, table, summary, options)
    save_json(os.path.join(out_dir, 'summary.json'), summary)
    return summary


def _report(out_dir, table, summary, options):
    lines = ['# Seam consistency QC (Stage 2.9 tile positions)', '',
             f"Residual = solved (P_ch[nb] - P_ch[tile]) - measured. Exceeds: |x|,|y| > {options['max_xy_px']} px "
             f"or |z| > {options['max_z_slices']} slices. consistent = >= 2 channels exceed in the same direction "
             f"(pairwise cosine >= {options['agree_cos']} or within {options['agree_px']} px) -> tile position "
             "suspect.", '',
             f"Status: **{summary['status']}**; counts {summary['counts']}; max |r| x/y/z "
             f"{summary['max_abs_rx']:.1f} / {summary['max_abs_ry']:.1f} / {summary['max_abs_rz']:.1f}", '',
             '| seam | status | exceeding | max |rx| | max |ry| | max |rz| |', '|---|---|---|---|---|---|']
    for r in table[table.status != 'ok'].itertuples() if len(table) else []:
        lines.append(f'| {r.tile} -> {r.nb_tile} | {r.status} | {r.exceed_channels} | '
                     f'{r.max_abs_rx:.1f} | {r.max_abs_ry:.1f} | {r.max_abs_rz:.1f} |')
    with open(os.path.join(out_dir, 'report.md'), 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def _code():
    from src.core.provenance import file_sha256
    return file_sha256(os.path.abspath(__file__))
