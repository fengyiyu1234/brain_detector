"""Sparse local-contrast (SNR) survey of TF channels, one report per sample (read-only).

Helps choose contrast_filter.<TF>.snr_min per sample. For each config and each active
single-exposure StarDist TF channel: --tiles tissue-rich tiles (raw CSV size >= median,
seeded) and --slices slices per tile (depth terciles of the slices with >= --min-boxes
boxes kept by the regular Stage 2.25 filter). The Stage 2.2 SNR of every box on those
slices comes from 1_2d_raw/contrast/ when Stage 2.2 wrote it, else from
src.core.local_contrast.slice_ring_stats on the raw image (same formula). Among the boxes
the regular filter keeps (2_2d_filtered result rows plus rejections with reason snr_min),
the report gives the share an SNR threshold would remove.

The reference sample (--reference, default ET10) sets the target: --ref-threshold per
channel (Olig2 4: hand-picked on 15 ET10 slices on 2026-11-13; Sox9 5: 44 slices,
2026-11-03) and the share it removes there. Each sample also gets the matched threshold,
the SNR that removes the same share in that sample.

Writes <out>/<sample>/report.html with per-channel tables, the SNR distribution against
the reference, and one image per sampled slice (boxes coloured by SNR band, grey = removed
by the regular filter); <out>/summary.csv and <out>/index.html across samples. Each
finished sample/channel is cached in <out>/<sample>/<channel>/features.csv and reused.

Usage (project root on PYTHONPATH; Windows drive letters come from `net use`):
  python src/utils/snr_survey.py --out X:/Fengyi/snr_survey [config ...]
With no configs, every config/**/config_*.json is surveyed. On a Windows PC numpy needs
MKL_THREADING_LAYER=SEQUENTIAL (see compare notes in PROGRESS_LOG).
"""

import argparse
import glob
import html
import json
import os
import re
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.loader import expand_result_date, load_config
from src.core.local_contrast import CONTRAST_SUBDIR, _read_image, slice_ring_stats
from src.qc.common import mapped_path

THRESHOLDS = [3, 4, 5, 6, 7, 8]
BANDS = [  # (upper SNR bound, RGB, width, label)
    (3, (255, 60, 60), 1, '<3'), (4, (255, 150, 0), 1, '3-4'), (5, (255, 230, 0), 2, '4-5'),
    (6, (150, 230, 80), 2, '5-6'), (np.inf, (0, 200, 0), 2, '>=6')]
HALF, SCALE = 260, 2
RAW_COLS = ['slice_name', 'x1', 'y1', 'x2', 'y2', 'score', 'mean', 'z', 'detection_id',
            'source_image']


def drive_mappings():
    """{'/rsstu/users/...': 'X:/'} from `net use` on Windows; {} elsewhere."""
    if os.name != 'nt':
        return {}
    text = subprocess.run(['net', 'use'], capture_output=True, text=True).stdout
    return {'/' + share.replace('\\', '/'): f'{drive}/'
            for drive, share in re.findall(r'([A-Za-z]:)\s+\\\\[^\\\s]+\\(rsstu(?:\\[^\s\\]+)+)',
                                           text)}


def sample_name(config_path):
    return Path(config_path).stem.replace('config_', '', 1)


def tf_channels(config):
    return [ch['id'] for ch in config.get('channels_routing', [])
            if ch.get('active', True) and ch.get('type') == 'tf'
            and str(ch.get('model', '')).lower() == 'stardist' and not ch.get('double_exposure')]


def choose_slices(result_dir, channel, rng, n_tiles, n_slices, min_boxes):
    """[(tile, [raw z, ...])] and {tile: kept detection ids on those slices}."""
    raw_dir, filt_dir = result_dir / '1_2d_raw', result_dir / '2_2d_filtered'
    files = sorted(raw_dir.glob(f'*_{channel}_result.csv'))
    files = [f for f in files
             if (filt_dir / f.name).is_file()]
    sizes = np.array([f.stat().st_size for f in files])
    rich = [f.name[:-len(f'_{channel}_result.csv')] for f, s in zip(files, sizes)
            if s >= np.median(sizes)]
    tiles = sorted(rng.choice(rich, min(n_tiles, len(rich)), replace=False)) if rich else []
    plan, kept = [], {}
    for tile in tiles:
        filtered = pd.read_csv(filt_dir / f'{tile}_{channel}_result.csv',
                               usecols=['detection_id', 'z'], dtype={'detection_id': str})
        rejected_path = filt_dir / f'{tile}_{channel}_rejected.csv'
        if rejected_path.is_file():
            rejected = pd.read_csv(rejected_path, usecols=['detection_id', 'z', 'rejection_reason'],
                                   dtype={'detection_id': str, 'rejection_reason': str})
            # boxes the regular filter kept but Stage 2.25's SNR step removed
            filtered = pd.concat([filtered, rejected.loc[rejected.rejection_reason == 'snr_min',
                                                         ['detection_id', 'z']]])
        counts = filtered.groupby('z').size()
        good = np.sort(counts[counts >= min_boxes].index.to_numpy())
        if len(good) < n_slices:
            good = np.sort(counts.sort_values(ascending=False).index[:n_slices].to_numpy())
        zs = sorted(int(rng.choice(part)) for part in np.array_split(good, n_slices) if len(part))
        plan.append((tile, zs))
        kept[tile] = set(filtered.loc[filtered.z.isin(zs), 'detection_id'])
    return plan, kept


def slice_features(result_dir, channel, tile, zs, kept_ids, mappings):
    """{z: (features DataFrame, image)} for one tile's chosen slices."""
    raw_dir = result_dir / '1_2d_raw'
    raw = pd.concat(c[c.z.isin(zs)] for c in pd.read_csv(
        raw_dir / f'{tile}_{channel}_result.csv', usecols=RAW_COLS, chunksize=500_000,
        dtype={'detection_id': str, 'source_image': str}))
    contrast = raw_dir / CONTRAST_SUBDIR / f'{tile}_{channel}_contrast.csv'
    snr_table = (pd.read_csv(contrast, dtype={'detection_id': str}).set_index('detection_id')
                 if contrast.is_file() else None)
    polygons = {}
    if snr_table is None:
        ids = set(raw.detection_id)
        for chunk in pd.read_csv(raw_dir / f'{tile}_{channel}_instances.csv',
                                 usecols=['detection_id', 'polygon_xy_json'], dtype=str,
                                 chunksize=500_000):
            chunk = chunk[chunk.detection_id.isin(ids)]
            polygons.update(zip(chunk.detection_id, chunk.polygon_xy_json))
    out = {}
    for z, part in raw.groupby('z'):
        part = part.reset_index(drop=True)
        image = _read_image(str(mapped_path(part.source_image.iloc[0], mappings)))
        if snr_table is not None:
            stats = snr_table.reindex(part.detection_id)[['ring_bg', 'ring_mad', 'snr']].to_numpy()
            source = 'stage2.2 table'
        else:
            shapes = [np.asarray(json.loads(polygons[d]), float)
                      if isinstance(polygons.get(d), str) else None for d in part.detection_id]
            ring = slice_ring_stats(image, part[['x1', 'y1', 'x2', 'y2']].to_numpy(int), shapes)
            with np.errstate(divide='ignore', invalid='ignore'):
                snr = np.where(ring[:, 1] > 0, (part['mean'].to_numpy(float) - ring[:, 0]) / ring[:, 1],
                               np.nan)
            stats = np.c_[ring[:, :2], snr]
            source = 'computed'
        part = part.drop(columns='source_image').assign(
            ring_bg=stats[:, 0], ring_mad=stats[:, 1], snr=stats[:, 2],
            kept=part.detection_id.isin(kept_ids), tile=tile, snr_source=source)
        out[int(z)] = (part, image)
    return out


def band(snr):
    for upper, rgb, width, _ in BANDS:
        if snr < upper:
            return rgb, width
    return (255, 255, 255), 1   # NaN: the pipeline keeps boxes without an SNR


def render(path, title, d, image):
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 18)
    except OSError:
        font = ImageFont.load_default()
    kept = d[d.kept]
    c = np.c_[(kept.x1 + kept.x2) / 2, (kept.y1 + kept.y2) / 2]
    h, w = image.shape
    grid_x = range(HALF, max(HALF, w - HALF) + 1, 128)
    grid_y = range(HALF, max(HALF, h - HALF) + 1, 128)
    bright = (kept.snr >= 6).to_numpy()

    def score(p):
        inside = (np.abs(c[:, 0] - p[0]) < HALF) & (np.abs(c[:, 1] - p[1]) < HALF)
        return 3 * (inside & bright).sum() + inside.sum()
    cx, cy = max(((x, y) for x in grid_x for y in grid_y), key=score) if len(c) else (w // 2, h // 2)
    x0, y0 = int(min(max(0, cx - HALF), max(0, w - 2 * HALF))), int(min(max(0, cy - HALF), max(0, h - 2 * HALF)))
    lo, hi = np.percentile(image, 1), np.percentile(image, 99.8)
    g = np.clip((image[y0:y0 + 2 * HALF, x0:x0 + 2 * HALF] - lo) / max(hi - lo, 1e-6), 0, 1)
    im = Image.fromarray((np.dstack([g, g, g]) * 255).astype(np.uint8)).resize(
        (g.shape[1] * SCALE, g.shape[0] * SCALE), Image.NEAREST)
    draw = ImageDraw.Draw(im)
    inside = d[(d.x2 > x0) & (d.x1 < x0 + 2 * HALF) & (d.y2 > y0) & (d.y1 < y0 + 2 * HALF)]
    for r in inside.itertuples():
        box = [(r.x1 - x0) * SCALE, (r.y1 - y0) * SCALE, (r.x2 - x0) * SCALE, (r.y2 - y0) * SCALE]
        if not r.kept:
            draw.rectangle(box, outline=(110, 110, 110), width=1)
        else:
            rgb, width = band(r.snr)
            draw.rectangle(box, outline=rgb, width=width)
    shown = inside[inside.kept]
    lows = [-np.inf] + [b[0] for b in BANDS[:-1]]
    counts = '  '.join(f'{label} {int(((shown.snr >= low) & (shown.snr < up)).sum())}'
                       for low, (up, _, _, label) in zip(lows, BANDS))
    lines = [title, f'crop origin x={x0} y={y0}; kept boxes in crop by SNR: {counts}',
             'red <3  orange 3-4  yellow 4-5  light green 5-6  green >=6  grey = removed by regular filter']
    draw.rectangle([0, 0, im.width, 26 * len(lines) + 6], fill=(0, 0, 0))
    for j, text in enumerate(lines):
        draw.text((6, 4 + 26 * j), text, fill='white', font=font)
    im.save(path)


def survey_channel(config_path, channel, out, n_tiles, n_slices, min_boxes, seed, mappings):
    """Features of one sample/channel (cached in <out>/<sample>/<channel>/features.csv)."""
    name = sample_name(config_path)
    folder = Path(out) / name / channel
    cached = folder / 'features.csv'
    if cached.is_file():
        return name, channel, 'cached'
    t0 = time.time()
    config = expand_result_date(load_config(config_path))
    result_dir = mapped_path(config['paths']['pATHRESULT'], mappings)
    rng = np.random.default_rng([seed, sum(map(ord, name + channel))])
    plan, kept = choose_slices(result_dir, channel, rng, n_tiles, n_slices, min_boxes)
    (folder / 'images').mkdir(parents=True, exist_ok=True)
    parts = []
    for tile, zs in plan:
        for z, (d, image) in slice_features(result_dir, channel, tile, zs, kept[tile],
                                            mappings).items():
            image_name = f'{tile}_z{z}.png'
            bg = float(np.nanmedian(d.loc[d.kept, 'ring_bg'])) if d.kept.any() else np.nan
            render(folder / 'images' / image_name,
                   f'{name} {channel} {tile} raw z{z}  ring bg {bg:.0f}  kept {int(d.kept.sum())}',
                   d, image)
            parts.append(d.assign(image=image_name))
    features = pd.concat(parts) if parts else pd.DataFrame()
    features.to_csv(str(cached) + '.part', index=False)
    os.replace(str(cached) + '.part', cached)
    return name, channel, f'{len(plan)} tiles, {len(parts)} slices, {time.time() - t0:.0f}s'


def removed_share(snr, threshold):
    """Share of kept boxes with SNR < threshold (no SNR = kept, as Stage 2.25)."""
    return float(np.mean(snr < threshold)) if len(snr) else np.nan


def matched_threshold(snr, share):
    """SNR threshold removing `share` of these boxes (NaN SNR counts as kept)."""
    values = np.sort(np.where(np.isnan(snr), np.inf, snr))
    if not len(values) or not np.isfinite(share):
        return np.nan
    k = int(round(share * len(values)))
    return float(values[min(k, len(values) - 1)]) if k < len(values) else float(values[-1])


def summarize(features, channel, ref_snr, ref_threshold):
    kept = features[features.kept]
    snr = kept.snr.to_numpy(float)
    ref_share = removed_share(ref_snr, ref_threshold)
    per_slice = kept.groupby(['tile', 'z']).snr.apply(lambda s: removed_share(s.to_numpy(float), ref_threshold))
    row = {'channel': channel, 'slices': int(features.groupby(['tile', 'z']).ngroups),
           'kept_boxes': len(kept), 'ring_bg_median': float(np.nanmedian(kept.ring_bg)),
           'snr_median': float(np.nanmedian(snr)), 'no_snr': int(np.isnan(snr).sum()),
           'ref_threshold': ref_threshold, 'ref_removed': ref_share,
           'removed_at_ref': removed_share(snr, ref_threshold),
           'slice_removed_min': float(per_slice.min()), 'slice_removed_median': float(per_slice.median()),
           'slice_removed_max': float(per_slice.max()),
           'matched_threshold': matched_threshold(snr, ref_share)}
    row.update({f'removed_snr{t}': removed_share(snr, t) for t in THRESHOLDS})
    return row


def plot_distribution(path, snr, ref_snr, channel, ref_name, ref_threshold):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=120)
    for values, label, color in ((ref_snr, f'{ref_name} (reference)', '#a3a29c'),
                                 (snr, 'this sample', '#2a78d6')):
        v = np.sort(values[np.isfinite(values)])
        if len(v):
            ax.plot(v, np.arange(1, len(v) + 1) / len(values), color=color, lw=2, label=label)
    ax.axvline(ref_threshold, color='#eb6834', lw=1, ls='--', label=f'SNR {ref_threshold:g}')
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 1)
    ax.set_xlabel('SNR of boxes kept by the regular filter')
    ax.set_ylabel('cumulative share')
    ax.set_title(f'{channel}: share removed at each SNR threshold', fontsize=10)
    ax.grid(color='#e4e3df', lw=0.6)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def pct(v):
    return '' if v is None or (isinstance(v, float) and not np.isfinite(v)) else f'{100 * v:.1f}%'


def write_reports(out, samples, reference, ref_thresholds):
    out = Path(out)
    ref_snr = {}
    for channel in ref_thresholds:
        path = out / reference / channel / 'features.csv'
        if path.is_file():
            f = pd.read_csv(path)
            ref_snr[channel] = f.loc[f.kept, 'snr'].to_numpy(float)
    summary = []
    for name, channels in samples.items():
        sections = []
        for channel in channels:
            path = out / name / channel / 'features.csv'
            if not path.is_file() or channel not in ref_snr:
                continue
            features = pd.read_csv(path, dtype={'tile': str})
            if features.empty:
                continue
            row = summarize(features, channel, ref_snr[channel], ref_thresholds[channel])
            summary.append({'sample': name, **row})
            plot_distribution(out / name / f'{channel}_snr.png',
                              features.loc[features.kept, 'snr'].to_numpy(float),
                              ref_snr[channel], channel, reference, ref_thresholds[channel])
            per_slice = []
            for (tile, z), g in features.groupby(['tile', 'z']):
                k = g[g.kept]
                s = k.snr.to_numpy(float)
                per_slice.append(f"<tr><td>{tile}</td><td>{z}</td><td>{len(k)}</td>"
                                 f"<td>{np.nanmedian(k.ring_bg):.0f}</td><td>{np.nanmedian(s):.1f}</td>"
                                 + ''.join(f'<td>{pct(removed_share(s, t))}</td>' for t in THRESHOLDS)
                                 + f"<td><a href='{channel}/images/{g.image.iat[0]}'>image</a></td></tr>")
            images = ''.join(
                f"<a href='{channel}/images/{img}'><img src='{channel}/images/{img}' width='360'></a>"
                for img in features.image.drop_duplicates())
            head = ''.join(f'<th>SNR&lt;{t}</th>' for t in THRESHOLDS)
            sections.append(f"""
<h2>{channel}</h2>
<p>Reference {reference} at SNR {row['ref_threshold']:g} removes <b>{pct(row['ref_removed'])}</b> of the
boxes kept by the regular filter. Here SNR {row['ref_threshold']:g} removes <b>{pct(row['removed_at_ref'])}</b>
(per slice {pct(row['slice_removed_min'])} – {pct(row['slice_removed_max'])}); the same share as the reference
is removed at <b>SNR {row['matched_threshold']:.1f}</b>.</p>
<table><tr><th>slices</th><th>kept boxes</th><th>ring bg median</th><th>SNR median</th>{head}</tr>
<tr><td>{row['slices']}</td><td>{row['kept_boxes']}</td><td>{row['ring_bg_median']:.0f}</td>
<td>{row['snr_median']:.1f}</td>{''.join(f"<td>{pct(row[f'removed_snr{t}'])}</td>" for t in THRESHOLDS)}</tr></table>
<img src='{channel}_snr.png'>
<h3>Per slice</h3>
<table><tr><th>tile</th><th>raw z</th><th>kept</th><th>ring bg</th><th>SNR median</th>{head}<th></th></tr>
{''.join(per_slice)}</table>
<h3>Images</h3><p>Boxes kept by the regular filter, coloured by SNR: red &lt;3, orange 3–4, yellow 4–5,
light green 5–6, green ≥6; white = no SNR (kept by the pipeline); grey = removed by the regular filter.</p>
<div>{images}</div>""")
        if sections:
            (out / name / 'report.html').write_text(PAGE.format(
                title=f'SNR survey — {html.escape(name)}', body=''.join(sections)), encoding='utf-8')
    table = pd.DataFrame(summary)
    table.to_csv(out / 'summary.csv', index=False)
    rows = ''.join(
        f"<tr><td><a href='{r['sample']}/report.html'>{r['sample']}</a></td><td>{r['channel']}</td>"
        f"<td>{r['kept_boxes']}</td><td>{r['ring_bg_median']:.0f}</td><td>{r['snr_median']:.1f}</td>"
        f"<td>{pct(r['removed_at_ref'])}</td><td>{pct(r['ref_removed'])}</td>"
        f"<td>{r['matched_threshold']:.1f}</td></tr>"
        for r in table.sort_values(['channel', 'sample']).to_dict('records'))
    (out / 'index.html').write_text(PAGE.format(title='SNR survey', body=f"""
<h1>SNR survey</h1><p>Reference: {reference}; thresholds {', '.join(f'{c} {t:g}' for c, t in ref_thresholds.items())}.</p>
<table><tr><th>sample</th><th>channel</th><th>kept boxes</th><th>ring bg median</th><th>SNR median</th>
<th>removed at reference threshold</th><th>reference removes</th><th>matched threshold</th></tr>{rows}</table>"""),
        encoding='utf-8')
    return table


PAGE = """<!DOCTYPE html><html><head><meta charset='utf-8'><title>{title}</title><style>
body{{font-family:system-ui,sans-serif;margin:24px;color:#0b0b0b}} table{{border-collapse:collapse;margin:8px 0}}
td,th{{border:1px solid #e4e3df;padding:3px 8px;text-align:right}} th{{background:#f4f3ef}}
img{{margin:3px}}</style></head><body>{body}</body></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('configs', nargs='*', help='pipeline configs (default: config/**/config_*.json)')
    parser.add_argument('--out', required=True)
    parser.add_argument('--reference', default='config/EGFR/3/config_EGFR_ET10.json')
    parser.add_argument('--ref-threshold', action='append', default=[], metavar='CH=V',
                        help='reference SNR threshold per channel (default Olig2=4 Sox9=5)')
    parser.add_argument('--tiles', type=int, default=4)
    parser.add_argument('--slices', type=int, default=3)
    parser.add_argument('--min-boxes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20261113)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    ref_thresholds = {'Olig2': 4.0, 'Sox9': 5.0}
    for spec in args.ref_threshold:
        channel, _, value = spec.partition('=')
        ref_thresholds[channel] = float(value)
    configs = args.configs or sorted(glob.glob('config/**/config_*.json', recursive=True))
    configs = [os.path.normpath(p) for p in configs]
    reference = os.path.normpath(args.reference)
    configs = [reference] + [p for p in configs if p != reference]
    mappings = drive_mappings()
    print('path mappings:', mappings, flush=True)
    samples = {sample_name(p): tf_channels(load_config(p)) for p in configs}
    jobs = [(p, ch) for p in configs for ch in samples[sample_name(p)]]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(survey_channel, p, ch, args.out, args.tiles, args.slices,
                               args.min_boxes, args.seed, mappings): (p, ch) for p, ch in jobs}
        for future in as_completed(futures):
            try:
                print(*future.result(), flush=True)
            except Exception as error:   # report and continue with the other samples
                print('FAILED', *futures[future], repr(error), flush=True)
    table = write_reports(args.out, samples, sample_name(reference), ref_thresholds)
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
