"""Preview the configured Stage 2.25 filter of TF channels on the snr_survey slices (read-only).

For each config and each active single-exposure StarDist TF channel, reads the cached boxes
of src/utils/snr_survey.py (<survey>/<sample>/<channel>/features.csv: every raw box on 12
sampled slices, with its Stage 2.2 ring statistics) and applies what Stage 2.25 now does:
the regular filter from the config (per box only: the tool refuses percentile filters and
the patch-fragment rule, which need the whole tile), then contrast_filter with the same code
as the pipeline (src.core.snr_filter: slice noise, per-slice threshold lowering).

Writes <out>/<sample>/report.html (per channel: totals, per-slice table with the threshold
used and the share each SNR threshold would remove, one image per slice), <out>/summary.csv
and <out>/index.html. Images: boxes kept by the regular filter coloured by the SNR the
filter compares (same bands as snr_survey), a magenta dot on boxes the SNR step removes,
grey = removed by the regular filter.

Usage (project root on PYTHONPATH; on a Windows PC set MKL_THREADING_LAYER=SEQUENTIAL):
  python src/utils/snr_filter_preview.py --survey X:/Fengyi/snr_survey/01 \
      --out X:/Fengyi/snr_survey/03 [config ...]
"""

import argparse
import glob
import html
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.loader import expand_result_date, load_config
from src.core.detection_filter import filter_detection_df, resolve_filter_params
from src.core.local_contrast import _read_image
from src.core.patch_fragments import resolve_patch_fragment
from src.core.snr_filter import resolve_snr_filter, slice_snr, slice_thresholds
from src.qc.common import mapped_path
from src.utils.snr_slice_compare import source_images
from src.utils.snr_survey import BANDS, HALF, PAGE, SCALE, band, drive_mappings, pct, sample_name, tf_channels

THRESHOLDS = [2, 3, 4, 5, 6, 7, 8]
SLICE_WIDE = ("bbox_area_pct_min", "bbox_area_pct_max", "bbox_mean_pct_min", "nms_containment_thresh")
SNR_REMOVED = (255, 0, 255)


def channel_filter(config, channel):
    """(regular filter params, SNR params or None); refuses filters that need the whole tile."""
    spec = next(ch for ch in config['channels_routing'] if ch['id'] == channel)
    params = resolve_filter_params(config, spec)
    wide = [k for k in SLICE_WIDE if params.get(k) is not None]
    if wide or resolve_patch_fragment(config, spec) is not None:
        raise ValueError(f'{channel}: {wide or "patch_fragment"} needs whole tiles; '
                         'this preview only handles per-box filters')
    return params, resolve_snr_filter(config).get(channel)


def apply_filters(features, params, snr_params):
    """features plus regular_kept, snr (the value compared), slice_noise, threshold, kept.

    The survey's own `kept` (regular filter at survey time) becomes kept_old."""
    f = features.reset_index(drop=True).rename(columns={'kept': 'kept_old'})
    f['slice_key'] = f.tile.astype(str) + '/' + f.z.astype(str)
    regular = filter_detection_df(f.assign(row=np.arange(len(f))), params)
    f['regular_kept'] = np.isin(np.arange(len(f)), regular.row.to_numpy())
    f['slice_noise'] = np.nan
    f['threshold'] = np.nan
    if snr_params is None:
        f['kept'] = f.regular_kept
        return f
    if snr_params.get('noise', 'ring') == 'slice':
        snr, noise = slice_snr(f['mean'].to_numpy(float), f.ring_bg.to_numpy(float),
                               f.ring_mad.to_numpy(float), f.slice_key.to_numpy())
        f['snr'], f['slice_noise'] = snr, noise
    keep = f.regular_kept.to_numpy()
    if 'max_removed_frac' in snr_params:
        per_slice = slice_thresholds(f.snr.to_numpy(float)[keep], f.slice_key.to_numpy()[keep], snr_params)
        f['threshold'] = per_slice.reindex(f.slice_key).to_numpy(float)
    else:
        f['threshold'] = snr_params['snr_min']
    f.loc[~keep, 'threshold'] = np.nan
    with np.errstate(invalid='ignore'):
        f['kept'] = keep & ~(f.snr.to_numpy(float) < f.threshold.to_numpy(float))
    return f


def render(path, lines, d, image):
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 18)
    except OSError:
        font = ImageFont.load_default()
    reg = d[d.regular_kept]
    removed = (~reg.kept).to_numpy()
    c = np.c_[(reg.x1 + reg.x2) / 2, (reg.y1 + reg.y2) / 2]
    h, w = image.shape

    def score(p):   # crop with the most SNR-removed boxes, then the most boxes
        inside = (np.abs(c[:, 0] - p[0]) < HALF) & (np.abs(c[:, 1] - p[1]) < HALF)
        return 4 * (inside & removed).sum() + inside.sum()
    grid = [(x, y) for x in range(HALF, max(HALF, w - HALF) + 1, 128)
            for y in range(HALF, max(HALF, h - HALF) + 1, 128)]
    cx, cy = max(grid, key=score) if len(c) else (w // 2, h // 2)
    x0 = int(min(max(0, cx - HALF), max(0, w - 2 * HALF)))
    y0 = int(min(max(0, cy - HALF), max(0, h - 2 * HALF)))
    lo, hi = np.percentile(image, 1), np.percentile(image, 99.8)
    g = np.clip((image[y0:y0 + 2 * HALF, x0:x0 + 2 * HALF] - lo) / max(hi - lo, 1e-6), 0, 1)
    crop = Image.fromarray((np.dstack([g, g, g]) * 255).astype(np.uint8)).resize(
        (g.shape[1] * SCALE, g.shape[0] * SCALE), Image.NEAREST)
    draw = ImageDraw.Draw(crop)
    inside = d[(d.x2 > x0) & (d.x1 < x0 + 2 * HALF) & (d.y2 > y0) & (d.y1 < y0 + 2 * HALF)]
    for r in inside.itertuples():
        box = [(r.x1 - x0) * SCALE, (r.y1 - y0) * SCALE, (r.x2 - x0) * SCALE, (r.y2 - y0) * SCALE]
        if not r.regular_kept:
            draw.rectangle(box, outline=(110, 110, 110), width=1)
            continue
        rgb, width = band(r.snr)
        draw.rectangle(box, outline=rgb, width=width)
        if not r.kept:
            draw.ellipse([box[2] - 4, box[1] - 10, box[2] + 6, box[1]], fill=SNR_REMOVED)
    lines = lines + [f'crop origin x={x0} y={y0}',
                     'red <3  orange 3-4  yellow 4-5  light green 5-6  green >=6',
                     'magenta dot = removed by SNR   grey = removed by regular filter']
    head = 26 * len(lines) + 6
    im = Image.new('RGB', (crop.width, crop.height + head), (0, 0, 0))
    im.paste(crop, (0, head))
    text = ImageDraw.Draw(im)
    for j, line in enumerate(lines):
        text.text((6, 4 + 26 * j), line, fill='white', font=font)
    im.save(path)


def share_below(snr, t):
    return float(np.mean(snr < t)) if len(snr) else np.nan


def preview_channel(config_path, channel, survey, out, mappings):
    """Filter one sample/channel's survey slices, render images; returns (summary row, slice rows)."""
    t0 = time.time()
    name = sample_name(config_path)
    config = expand_result_date(load_config(config_path))
    params, snr_params = channel_filter(config, channel)
    features = pd.read_csv(Path(survey) / name / channel / 'features.csv', dtype={'tile': str})
    f = apply_filters(features, params, snr_params)
    result_dir = mapped_path(config['paths']['pATHRESULT'], mappings)
    folder = Path(out) / name / channel / 'images'
    folder.mkdir(parents=True, exist_ok=True)
    slices = []
    for tile, part in f.groupby('tile', sort=True):
        sources = source_images(result_dir, channel, tile, sorted(part.z.unique()))
        for z, d in part.groupby('z'):
            reg = d[d.regular_kept]
            snr = reg.snr.to_numpy(float)
            t = float(reg.threshold.iloc[0]) if len(reg) else np.nan
            row = {'tile': tile, 'z': int(z), 'raw': len(d), 'regular_kept': len(reg),
                   'ring_bg': float(np.nanmedian(reg.ring_bg)) if len(reg) else np.nan,
                   'slice_noise': float(d.slice_noise.iloc[0]), 'threshold': t,
                   'snr_removed': int((~reg.kept).sum()), 'kept': int(reg.kept.sum()),
                   **{f'below{v}': share_below(snr, v) for v in THRESHOLDS},
                   'image': f'{tile}_z{int(z)}.png'}
            slices.append(row)
            lines = [f'{name} {channel} {tile} raw z{int(z)}  slice noise {row["slice_noise"]:.0f}  '
                     f'ring bg {row["ring_bg"]:.0f}',
                     f'regular filter keeps {len(reg)} of {len(d)}; SNR threshold {t:g} removes '
                     f'{row["snr_removed"]} ({pct(row["snr_removed"] / max(len(reg), 1))})',
                     'share below: ' + '  '.join(f'<{v} {pct(row[f"below{v}"])}' for v in THRESHOLDS)]
            render(folder / row['image'], lines, d,
                   _read_image(str(mapped_path(sources[int(z)], mappings))))
    reg = f[f.regular_kept]
    table = pd.DataFrame(slices)
    snr_min = snr_params['snr_min'] if snr_params else np.nan
    summary = {'sample': name, 'channel': channel, 'snr_params': snr_params, 'slices': len(table),
               'raw': len(f), 'old_regular_kept': int(f.kept_old.sum()),
               'regular_kept': len(reg), 'snr_removed': int((~reg.kept).sum()),
               'snr_removed_share': float((~reg.kept).mean()) if len(reg) else np.nan,
               'removed_at_snr_min': share_below(reg.snr.to_numpy(float), snr_min),
               'slices_lowered': int((table.threshold < snr_min).sum()),
               'lowest_threshold': float(table.threshold.min()),
               'slice_noise_median': float(np.nanmedian(table.slice_noise)),
               'seconds': round(time.time() - t0)}
    table.to_csv(Path(out) / name / channel / 'slices.csv', index=False)
    return summary, table


def section(s, table):
    head = ''.join(f'<th>&lt;{v}</th>' for v in THRESHOLDS)
    rows = ''.join(
        f"<tr><td>{r.tile}</td><td>{r.z}</td><td>{r.raw}</td><td>{r.regular_kept}</td>"
        f"<td>{r.ring_bg:.0f}</td><td>{r.slice_noise:.0f}</td>"
        f"<td{' style=background:#fde2f4' if r.threshold < s['snr_params']['snr_min'] else ''}>{r.threshold:g}</td>"
        f"<td>{r.snr_removed} ({pct(r.snr_removed / max(r.regular_kept, 1))})</td>"
        + ''.join(f'<td>{pct(getattr(r, f"below{v}"))}</td>' for v in THRESHOLDS)
        + f"<td><a href='{s['channel']}/images/{r.image}'>image</a></td></tr>"
        for r in table.itertuples())
    images = ''.join(f"<a href='{s['channel']}/images/{img}'><img src='{s['channel']}/images/{img}' width='360'></a>"
                     for img in table.image)
    return f"""
<h2>{s['channel']}</h2>
<p>contrast_filter: <code>{html.escape(str(s['snr_params']))}</code></p>
<p>{s['slices']} slices, {s['raw']} raw boxes. Regular filter keeps <b>{s['regular_kept']}</b>
(before removing the percentile filters: {s['old_regular_kept']}). The SNR step removes <b>{s['snr_removed']}
({pct(s['snr_removed_share'])})</b>; a fixed SNR {s['snr_params']['snr_min']:g} would remove
{pct(s['removed_at_snr_min'])}. Threshold lowered on <b>{s['slices_lowered']} / {s['slices']}</b> slices
(lowest {s['lowest_threshold']:g}).</p>
<table><tr><th>tile</th><th>raw z</th><th>raw</th><th>regular kept</th><th>ring bg</th><th>slice noise</th>
<th>threshold</th><th>SNR removed</th>{head}<th></th></tr>{rows}</table>
<p>Share of the regular-filter boxes below each SNR (the value the filter compares). Pink threshold = lowered.</p>
<div>{images}</div>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('configs', nargs='*', help='pipeline configs (default: config/**/config_*.json)')
    parser.add_argument('--survey', required=True, help='snr_survey output with features.csv per sample/channel')
    parser.add_argument('--out', required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    configs = [os.path.normpath(p) for p in
               (args.configs or sorted(glob.glob('config/**/config_*.json', recursive=True)))]
    mappings = drive_mappings()
    print('path mappings:', mappings, flush=True)
    jobs = [(p, ch) for p in configs for ch in tf_channels(load_config(p))
            if (Path(args.survey) / sample_name(p) / ch / 'features.csv').is_file()]
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(preview_channel, p, ch, args.survey, args.out, mappings): (p, ch)
                   for p, ch in jobs}
        for future in as_completed(futures):
            p, ch = futures[future]
            try:
                summary, table = future.result()
            except Exception as error:   # report and continue with the other samples
                print('FAILED', p, ch, repr(error), flush=True)
                continue
            results[(sample_name(p), ch)] = (summary, table)
            print(sample_name(p), ch, f"{summary['seconds']}s", flush=True)
    out = Path(args.out)
    for name in sorted({n for n, _ in results}):
        body = ''.join(section(*results[k]) for k in sorted(results) if k[0] == name)
        (out / name / 'report.html').write_text(
            PAGE.format(title=f'SNR filter preview — {html.escape(name)}', body=f'<h1>{html.escape(name)}</h1>' + body),
            encoding='utf-8')
    summary = pd.DataFrame([s for s, _ in results.values()]).sort_values(['channel', 'sample'])
    summary.to_csv(out / 'summary.csv', index=False)
    rows = ''.join(
        f"<tr><td><a href='{r['sample']}/report.html'>{r['sample']}</a></td><td>{r['channel']}</td>"
        f"<td>{r['regular_kept']}</td><td>{r['old_regular_kept']}</td><td>{r['slice_noise_median']:.0f}</td>"
        f"<td>{pct(r['removed_at_snr_min'])}</td><td>{pct(r['snr_removed_share'])}</td>"
        f"<td>{r['slices_lowered']} / {r['slices']}</td><td>{r['lowest_threshold']:g}</td></tr>"
        for r in summary.to_dict('records'))
    (out / 'index.html').write_text(PAGE.format(title='SNR filter preview', body=f"""
<h1>SNR filter preview</h1><p>Survey slices: {html.escape(str(args.survey))}. Each sample uses its own config.</p>
<table><tr><th>sample</th><th>channel</th><th>regular kept</th><th>regular kept (old filters)</th>
<th>slice noise median</th><th>removed at fixed snr_min</th><th>removed (per-slice threshold)</th>
<th>slices lowered</th><th>lowest threshold</th></tr>{rows}</table>"""), encoding='utf-8')
    print(summary.drop(columns=['snr_params']).to_string(index=False))


if __name__ == '__main__':
    main()
