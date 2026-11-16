"""Sensitivity of TF box counts to the contrast_filter parameters (read-only).

For each config and each active single-exposure StarDist TF channel, takes the snr_survey
slices (<survey>/<sample>/<channel>/features.csv: every raw box on 12 sampled slices) and
applies the configured Stage 2.25 filter (src.utils.snr_filter_preview.apply_filters, the
pipeline's own functions) with the config's contrast_filter as baseline and with one
parameter changed at a time: snr_min +- 0.5, max_removed_frac +- 0.05 and off, snr_floor
+- snr_step, the other step (0.5 <-> 1), and no SNR filter.

Between-sample comparisons are robust to a parameter when every sample's kept count moves
by about the same factor. Per channel and variant the report gives the kept/baseline ratio
across samples (min, median, max, max/min) and the Spearman correlation of the samples'
kept boxes per slice with the baseline.

Writes <out>/results.csv (sample x channel x variant), <out>/stability.csv and
<out>/index.html.

Usage (project root on PYTHONPATH; on a Windows PC set MKL_THREADING_LAYER=SEQUENTIAL):
  python src/utils/snr_sensitivity.py --survey X:/Fengyi/snr_survey/01 \
      --out X:/Fengyi/snr_survey/04 [config ...]
"""

import argparse
import glob
import html
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.config.loader import load_config
from src.utils.snr_filter_preview import apply_filters, channel_filter
from src.utils.snr_survey import PAGE, pct, sample_name, tf_channels


def variants(base):
    """[(name, contrast_filter params or None)] around the baseline params (full form)."""
    step, floor, snr_min = base['snr_step'], base['snr_floor'], base['snr_min']
    frac = base['max_removed_frac']
    out = [('baseline', base)]
    out += [(f'snr_min {v:g}', {**base, 'snr_min': v, 'snr_floor': min(floor, v)})
            for v in (snr_min - 0.5, snr_min + 0.5)]
    if frac is not None:
        out += [(f'max_removed_frac {v:.2f}', {**base, 'max_removed_frac': round(v, 4)})
                for v in (frac - 0.05, frac + 0.05)]
        out += [('no lowering', {**base, 'max_removed_frac': None})]
        out += [(f'snr_floor {v:g}', {**base, 'snr_floor': v})
                for v in (floor - step, floor + step) if v <= snr_min]
        other = 1.0 if step != 1.0 else 0.5
        out += [(f'snr_step {other:g}', {**base, 'snr_step': other})]
    out += [('no SNR filter', None)]
    return out


def run_channel(config_path, channel, survey):
    name = sample_name(config_path)
    params, base = channel_filter(load_config(config_path), channel)
    if base is None or 'noise' not in base:
        raise ValueError(f'{name} {channel}: needs the full contrast_filter form')
    features = pd.read_csv(Path(survey) / name / channel / 'features.csv', dtype={'tile': str})
    rows = []
    for variant, snr_params in variants(base):
        f = apply_filters(features, params, snr_params)
        reg = f[f.regular_kept]
        per_slice = reg.groupby('slice_key').threshold.first()
        rows.append({'sample': name, 'channel': channel, 'variant': variant,
                     'params': '' if snr_params is None else str(snr_params),
                     'slices': f.slice_key.nunique(), 'regular_kept': len(reg),
                     'kept': int(reg.kept.sum()), 'removed_share': float((~reg.kept).mean()),
                     'slices_lowered': int((per_slice < base['snr_min']).sum()) if snr_params else 0,
                     'mean_threshold': float(per_slice.mean()) if snr_params else np.nan})
    return rows


def stability(results):
    """Per channel x variant: spread of kept/baseline across samples and rank agreement."""
    base = results[results.variant == 'baseline'].set_index(['channel', 'sample'])
    out = []
    for (channel, variant), g in results.groupby(['channel', 'variant'], sort=False):
        g = g.set_index('sample')
        b = base.loc[channel].reindex(g.index)
        ratio = g.kept / b.kept
        density, base_density = g.kept / g.slices, b.kept / b.slices
        out.append({'channel': channel, 'variant': variant, 'samples': len(g),
                    'removed_median': float(g.removed_share.median()),
                    'ratio_min': float(ratio.min()), 'ratio_median': float(ratio.median()),
                    'ratio_max': float(ratio.max()), 'ratio_spread': float(ratio.max() / ratio.min()),
                    'spearman': float(spearmanr(density, base_density).statistic) if len(g) > 2 else np.nan,
                    'most_up': ratio.idxmax(), 'most_down': ratio.idxmin()})
    return pd.DataFrame(out)


def write_html(out, results, stab, survey):
    sections = []
    for channel, s in stab.groupby('channel', sort=False):
        rows = ''.join(
            f"<tr><td style='text-align:left'>{html.escape(r.variant)}</td><td>{pct(r.removed_median)}</td>"
            f"<td>{r.ratio_min:.3f}</td><td>{r.ratio_median:.3f}</td><td>{r.ratio_max:.3f}</td>"
            f"<td><b>{r.ratio_spread:.3f}</b></td><td>{r.spearman:.3f}</td>"
            f"<td>{html.escape(r.most_up)}</td><td>{html.escape(r.most_down)}</td></tr>"
            for r in s.itertuples())
        r = results[results.channel == channel]
        wide = r.pivot(index='sample', columns='variant', values='kept')
        order = list(s.variant)
        ratio = wide[order].div(wide['baseline'], axis=0)
        base_params = r.loc[r.variant == 'baseline', 'params'].unique()
        sample_rows = ''.join(
            f"<tr><td style='text-align:left'>{html.escape(sample)}</td><td>{int(wide.loc[sample, 'baseline'])}</td>"
            + ''.join(f'<td>{ratio.loc[sample, v]:.3f}</td>' for v in order[1:]) + '</tr>'
            for sample in ratio.index)
        sections.append(f"""
<h2>{channel}</h2>
<p>Baseline (each sample's config): {'; '.join(html.escape(p) for p in base_params)}</p>
<table><tr><th>variant</th><th>removed (median sample)</th><th>kept / baseline: min</th><th>median</th>
<th>max</th><th>max/min</th><th>Spearman vs baseline</th><th>most up</th><th>most down</th></tr>{rows}</table>
<h3>Kept boxes per sample relative to baseline</h3>
<table><tr><th>sample</th><th>baseline kept</th>{''.join(f'<th>{html.escape(v)}</th>' for v in order[1:])}</tr>
{sample_rows}</table>""")
    (Path(out) / 'index.html').write_text(PAGE.format(title='SNR filter sensitivity', body=f"""
<h1>SNR filter sensitivity</h1>
<p>Survey slices: {html.escape(str(survey))} (12 per sample). Each variant changes one parameter of the
sample's own contrast_filter; the regular filter is the config's. <b>max/min</b> near 1 = every sample's
count moves by the same factor, so comparisons between samples do not depend on that parameter.
Spearman = rank agreement of kept boxes per slice across samples.</p>{''.join(sections)}"""),
        encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('configs', nargs='*', help='pipeline configs (default: config/**/config_*.json)')
    parser.add_argument('--survey', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    configs = [os.path.normpath(p) for p in
               (args.configs or sorted(glob.glob('config/**/config_*.json', recursive=True)))]
    rows = []
    for path in configs:
        for channel in tf_channels(load_config(path)):
            if (Path(args.survey) / sample_name(path) / channel / 'features.csv').is_file():
                rows += run_channel(path, channel, args.survey)
    results = pd.DataFrame(rows)
    stab = stability(results)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    results.to_csv(Path(args.out) / 'results.csv', index=False)
    stab.to_csv(Path(args.out) / 'stability.csv', index=False)
    write_html(args.out, results, stab, args.survey)
    pd.set_option('display.width', 250)
    print(stab.to_string(index=False, float_format=lambda v: f'{v:.3f}'))


if __name__ == '__main__':
    main()
