"""Compare the Stage 2.2 local SNR with a slice-noise SNR on the snr_survey slices (read-only).

Local SNR (current pipeline) = (mean - ring median) / ring noise, ring noise = 1.4826 * MAD
of the 2-8 px ring. A crowded nucleus gets halo light in its ring (high ring noise, low
SNR); a dim nucleus on flat black gets a very quiet ring (high SNR). Slice-noise SNR keeps
the numerator and divides by the slice's noise level instead: the median ring noise of all
boxes on that tile/slice.

Reads the cached <survey>/<sample>/<channel>/features.csv of src/utils/snr_survey.py (same
tiles and slices) and the raw images. For each sample/channel the slice-noise threshold is
set to remove as many boxes as the local SNR at the reference threshold (Olig2 4, Sox9 5),
so the images show which boxes change, not how many. Each image shows one crop twice: left
coloured by local SNR, right by slice-noise SNR (same colour bands); a cyan dot marks a box
only the local SNR removes, a magenta dot a box only the slice-noise SNR removes.

Writes <out>/<sample>/report.html, <out>/summary.csv and <out>/index.html.

Usage (project root on PYTHONPATH; on a Windows PC set MKL_THREADING_LAYER=SEQUENTIAL):
  python src/utils/snr_slice_compare.py --survey X:/Fengyi/snr_survey/01 \
      --out X:/Fengyi/snr_survey/02 [config ...]
"""

import argparse
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.loader import expand_result_date, load_config
from src.core.local_contrast import _read_image
from src.qc.common import mapped_path
from src.utils.snr_survey import (BANDS, HALF, PAGE, SCALE, THRESHOLDS, band, drive_mappings,
                                  matched_threshold, pct, removed_share, sample_name, tf_channels)

REF_THRESHOLDS = {'Olig2': 4.0, 'Sox9': 5.0}
ONLY_LOCAL, ONLY_SLICE = (0, 230, 255), (255, 0, 255)


def add_slice_snr(features):
    """Add slice_noise (median ring noise of the tile/slice) and slice_snr columns."""
    f = features.copy()
    noise = f.ring_mad.where(f.ring_mad > 0)
    f['slice_noise'] = noise.groupby([f.tile, f.z]).transform('median')
    f['slice_snr'] = (f['mean'] - f.ring_bg) / f.slice_noise
    return f


def flags(kept, local_t, slice_t):
    """(removed by local, removed by slice-noise) for kept boxes; NaN SNR is kept."""
    return (kept.snr < local_t).to_numpy(), (kept.slice_snr < slice_t).to_numpy()


def threshold_lines(kept):
    """Whole-slice box counts removed at each threshold, one line per SNR."""
    n = max(len(kept), 1)
    return [f'{label:<12}removes ' + '  '.join(
        f'<{t} {int((kept[col] < t).sum())} ({100 * (kept[col] < t).sum() / n:.0f}%)'
        for t in THRESHOLDS) for label, col in (('local', 'snr'), ('slice-noise', 'slice_snr'))]


def source_images(result_dir, channel, tile, zs):
    """{z: source image path} from the raw CSV."""
    found = {}
    for chunk in pd.read_csv(result_dir / '1_2d_raw' / f'{tile}_{channel}_result.csv',
                             usecols=['z', 'source_image'], dtype={'source_image': str},
                             chunksize=500_000):
        chunk = chunk[chunk.z.isin(set(zs) - set(found))].drop_duplicates('z')
        found.update(zip(chunk.z.astype(int), chunk.source_image))
        if len(found) == len(zs):
            break
    return found


def render(path, lines, d, image, local_t, slice_t):
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 18)
    except OSError:
        font = ImageFont.load_default()
    kept = d[d.kept]
    old, new = flags(kept, local_t, slice_t)
    changed = old != new
    c = np.c_[(kept.x1 + kept.x2) / 2, (kept.y1 + kept.y2) / 2]
    h, w = image.shape

    def score(p):   # crop with the most boxes whose fate changes, then the most boxes
        inside = (np.abs(c[:, 0] - p[0]) < HALF) & (np.abs(c[:, 1] - p[1]) < HALF)
        return 4 * (inside & changed).sum() + inside.sum()
    grid = [(x, y) for x in range(HALF, max(HALF, w - HALF) + 1, 128)
            for y in range(HALF, max(HALF, h - HALF) + 1, 128)]
    cx, cy = max(grid, key=score) if len(c) else (w // 2, h // 2)
    x0 = int(min(max(0, cx - HALF), max(0, w - 2 * HALF)))
    y0 = int(min(max(0, cy - HALF), max(0, h - 2 * HALF)))
    lo, hi = np.percentile(image, 1), np.percentile(image, 99.8)
    g = np.clip((image[y0:y0 + 2 * HALF, x0:x0 + 2 * HALF] - lo) / max(hi - lo, 1e-6), 0, 1)
    base = Image.fromarray((np.dstack([g, g, g]) * 255).astype(np.uint8)).resize(
        (g.shape[1] * SCALE, g.shape[0] * SCALE), Image.NEAREST)
    inside = d[(d.x2 > x0) & (d.x1 < x0 + 2 * HALF) & (d.y2 > y0) & (d.y1 < y0 + 2 * HALF)]
    panels = []
    for column in ('snr', 'slice_snr'):   # each panel drawn on its own, so boxes are clipped
        panel = base.copy()
        pd_ = ImageDraw.Draw(panel)
        for r in inside.itertuples():
            box = [(r.x1 - x0) * SCALE, (r.y1 - y0) * SCALE, (r.x2 - x0) * SCALE, (r.y2 - y0) * SCALE]
            if not r.kept:
                pd_.rectangle(box, outline=(110, 110, 110), width=1)
                continue
            rgb, width = band(getattr(r, column))
            pd_.rectangle(box, outline=rgb, width=width)
            gone_local, gone_slice = r.snr < local_t, r.slice_snr < slice_t
            if gone_local != gone_slice:
                dot = ONLY_LOCAL if gone_local else ONLY_SLICE
                pd_.ellipse([box[2] - 4, box[1] - 10, box[2] + 6, box[1]], fill=dot)
        panels.append(panel)
    panel = panels[0]
    gap, head = 12, 26 * len(lines) + 34
    im = Image.new('RGB', (2 * panel.width + gap, panel.height + head), (0, 0, 0))
    im.paste(panels[0], (0, head))
    im.paste(panels[1], (panel.width + gap, head))
    draw = ImageDraw.Draw(im)
    for j, text in enumerate(lines):
        draw.text((6, 4 + 26 * j), text, fill='white', font=font)
    y = head - 28
    draw.text((6, y), f'LEFT: local SNR (current), removed < {local_t:g}', fill='white', font=font)
    draw.text((panel.width + gap + 6, y), f'RIGHT: slice-noise SNR, removed < {slice_t:.1f}',
              fill='white', font=font)
    im.save(path)


def compare_channel(config_path, channel, survey, out, local_t, mappings):
    name = sample_name(config_path)
    features = pd.read_csv(Path(survey) / name / channel / 'features.csv',
                           dtype={'tile': str, 'detection_id': str})
    if features.empty:
        return name, channel, None
    t0 = time.time()
    f = add_slice_snr(features)
    kept = f[f.kept]
    slice_t = matched_threshold(kept.slice_snr.to_numpy(float),
                                removed_share(kept.snr.to_numpy(float), local_t))
    config = expand_result_date(load_config(config_path))
    result_dir = mapped_path(config['paths']['pATHRESULT'], mappings)
    folder = Path(out) / name / channel / 'images'
    folder.mkdir(parents=True, exist_ok=True)
    for tile, part in f.groupby('tile'):
        sources = source_images(result_dir, channel, tile, sorted(part.z.unique()))
        for z, d in part.groupby('z'):
            image = _read_image(str(mapped_path(sources[int(z)], mappings)))
            k = d[d.kept]
            old, new = flags(k, local_t, slice_t)
            lines = [f'{name} {channel} {tile} raw z{z}  kept {len(k)}  ring bg median '
                     f'{np.nanmedian(k.ring_bg):.0f}  slice noise {d.slice_noise.iloc[0]:.0f}',
                     *threshold_lines(k),
                     f'at the compared thresholds: local removes {old.sum()}, slice-noise removes '
                     f'{new.sum()}; only local {(old & ~new).sum()} (cyan dot), only slice-noise '
                     f'{(new & ~old).sum()} (magenta dot)',
                     'red <3  orange 3-4  yellow 4-5  light green 5-6  green >=6  '
                     'grey = removed by regular filter']
            render(folder / d.image.iat[0], lines, d, image, local_t, slice_t)
    f.to_csv(Path(out) / name / channel / 'features.csv', index=False)
    return name, channel, f'slice threshold {slice_t:.2f}, {time.time() - t0:.0f}s'


def channel_row(f, local_t, ref_slice_t):
    kept = f[f.kept]
    snr, slice_snr = kept.snr.to_numpy(float), kept.slice_snr.to_numpy(float)
    share = removed_share(snr, local_t)
    slice_t = matched_threshold(slice_snr, share)
    old, new = snr < local_t, slice_snr < slice_t
    row = {'kept_boxes': len(kept), 'local_threshold': local_t, 'local_removed': share,
           'slice_threshold_same_share': slice_t, 'removed_both': int((old & new).sum()),
           'only_local': int((old & ~new).sum()), 'only_slice': int((new & ~old).sum()),
           'slice_noise_median': float(np.nanmedian(kept.slice_noise)),
           'ref_slice_threshold': ref_slice_t,
           'slice_removed_at_ref': removed_share(slice_snr, ref_slice_t)}
    row.update({f'slice_removed_{t}': removed_share(slice_snr, t) for t in THRESHOLDS})
    return row


def write_reports(out, samples, reference):
    out = Path(out)
    ref = {}
    for channel, local_t in REF_THRESHOLDS.items():
        path = out / reference / channel / 'features.csv'
        if path.is_file():
            k = pd.read_csv(path).query('kept')
            ref[channel] = (removed_share(k.snr.to_numpy(float), local_t),
                            matched_threshold(k.slice_snr.to_numpy(float),
                                              removed_share(k.snr.to_numpy(float), local_t)))
    summary = []
    for name, channels in samples.items():
        sections = []
        for channel in channels:
            path = out / name / channel / 'features.csv'
            if not path.is_file() or channel not in ref:
                continue
            f = pd.read_csv(path, dtype={'tile': str})
            local_t = REF_THRESHOLDS[channel]
            ref_share, ref_slice_t = ref[channel]
            row = channel_row(f, local_t, ref_slice_t)
            row['slice_matched_to_ref'] = matched_threshold(
                f.loc[f.kept, 'slice_snr'].to_numpy(float), ref_share)
            summary.append({'sample': name, 'channel': channel, **row})
            per_slice = []
            for (tile, z), g in f.groupby(['tile', 'z']):
                k = g[g.kept]
                old = k.snr.to_numpy(float) < local_t
                new = k.slice_snr.to_numpy(float) < row['slice_threshold_same_share']
                per_slice.append(
                    f"<tr><td>{tile}</td><td>{z}</td><td>{len(k)}</td>"
                    f"<td>{g.slice_noise.iat[0]:.0f}</td><td>{pct(old.mean())}</td>"
                    f"<td>{pct(new.mean())}</td><td>{(old & ~new).sum()}</td><td>{(new & ~old).sum()}</td>"
                    f"<td><a href='{channel}/images/{g.image.iat[0]}'>image</a></td></tr>")
            images = ''.join(f"<a href='{channel}/images/{img}'><img src='{channel}/images/{img}' "
                             f"width='720'></a>" for img in f.image.drop_duplicates())
            head = ''.join(f'<th>&lt;{t}</th>' for t in THRESHOLDS)
            sections.append(f"""
<h2>{channel}</h2>
<p>Local SNR &lt; {local_t:g} removes <b>{pct(row['local_removed'])}</b> of the boxes kept by the regular
filter. Slice-noise SNR removes the same number at <b>{row['slice_threshold_same_share']:.1f}</b>:
{row['removed_both']} boxes removed by both, <b>{row['only_local']}</b> only by the local SNR (cyan dot,
would come back), <b>{row['only_slice']}</b> only by the slice-noise SNR (magenta dot, newly removed).</p>
<p>Against {reference}: the slice-noise threshold equivalent to local {local_t:g} there is
{ref_slice_t:.1f} (removes {pct(ref_share)} there); here it removes <b>{pct(row['slice_removed_at_ref'])}</b>;
the same share as {reference} is removed here at slice-noise SNR <b>{row['slice_matched_to_ref']:.1f}</b>.</p>
<table><tr><th colspan='{len(THRESHOLDS)}'>slice-noise SNR: share removed below</th></tr><tr>{head}</tr>
<tr>{''.join(f"<td>{pct(row[f'slice_removed_{t}'])}</td>" for t in THRESHOLDS)}</tr></table>
<h3>Per slice</h3>
<table><tr><th>tile</th><th>raw z</th><th>kept</th><th>slice noise</th><th>local removes</th>
<th>slice-noise removes</th><th>only local</th><th>only slice-noise</th><th></th></tr>{''.join(per_slice)}</table>
<h3>Images</h3><p>Same crop twice: left coloured by local SNR, right by slice-noise SNR. Red &lt;3,
orange 3–4, yellow 4–5, light green 5–6, green ≥6, white = no SNR, grey = removed by the regular
filter. Cyan dot: removed only by the local SNR. Magenta dot: removed only by the slice-noise SNR.
The crop is the part of the slice with the most such boxes.</p><div>{images}</div>""")
        if sections:
            (out / name / 'report.html').write_text(PAGE.format(
                title=f'Local vs slice-noise SNR — {name}', body=''.join(sections)), encoding='utf-8')
    table = pd.DataFrame(summary)
    table.to_csv(out / 'summary.csv', index=False)
    rows = ''.join(
        f"<tr><td><a href='{r['sample']}/report.html'>{r['sample']}</a></td><td>{r['channel']}</td>"
        f"<td>{r['kept_boxes']}</td><td>{pct(r['local_removed'])}</td>"
        f"<td>{r['slice_threshold_same_share']:.1f}</td><td>{r['only_local']}</td><td>{r['only_slice']}</td>"
        f"<td>{pct(r['slice_removed_at_ref'])}</td><td>{r['slice_matched_to_ref']:.1f}</td></tr>"
        for r in table.sort_values(['channel', 'sample']).to_dict('records'))
    (out / 'index.html').write_text(PAGE.format(title='Local vs slice-noise SNR', body=f"""
<h1>Local vs slice-noise SNR</h1><p>Local thresholds {', '.join(f'{c} {t:g}' for c, t in REF_THRESHOLDS.items())};
reference {reference}. "Same count" = slice-noise threshold removing as many boxes as the local threshold
in that sample.</p>
<table><tr><th>sample</th><th>channel</th><th>kept boxes</th><th>local removes</th><th>slice threshold, same count</th>
<th>only local removes</th><th>only slice removes</th><th>slice removes at {reference}-equivalent threshold</th>
<th>slice threshold matched to {reference}</th></tr>{rows}</table>"""), encoding='utf-8')
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('configs', nargs='*', help='pipeline configs (default: config/**/config_*.json)')
    parser.add_argument('--survey', required=True, help='snr_survey output with features.csv caches')
    parser.add_argument('--out', required=True)
    parser.add_argument('--reference', default='config/EGFR/3/config_EGFR_ET10.json')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    configs = args.configs or sorted(glob.glob('config/**/config_*.json', recursive=True))
    configs = [os.path.normpath(p) for p in configs]
    reference = os.path.normpath(args.reference)
    configs = [reference] + [p for p in configs if p != reference]
    mappings = drive_mappings()
    samples = {sample_name(p): [c for c in tf_channels(load_config(p)) if c in REF_THRESHOLDS]
               for p in configs}
    jobs = [(p, ch) for p in configs for ch in samples[sample_name(p)]
            if (Path(args.survey) / sample_name(p) / ch / 'features.csv').is_file()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(compare_channel, p, ch, args.survey, args.out, REF_THRESHOLDS[ch],
                               mappings): (p, ch) for p, ch in jobs}
        for future in as_completed(futures):
            try:
                print(*future.result(), flush=True)
            except Exception as error:   # report and continue with the other samples
                print('FAILED', *futures[future], repr(error), flush=True)
    table = write_reports(args.out, samples, sample_name(reference))
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
