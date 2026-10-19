"""Headless per-tile and equally weighted overview sensitivity figures."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def curve_summary(frame):
    records = []
    for (tile, channel, metric, axis), group in frame.groupby(['tile', 'channel', 'metric', 'axis'], sort=True):
        valid = group.loc[group.value.notna()]
        row = dict(tile=tile, channel=channel, metric=metric, axis=axis,
                   n_pairs=int(group.n_pairs.iloc[0]), status=group.status.iloc[0])
        if not valid.empty:
            extreme = valid.value.max() if metric == 'voxel_iou' else valid.value.min()
            candidates = valid.loc[np.isclose(valid.value, extreme)].copy()
            best = candidates.loc[candidates.delta_um.abs().idxmin()]
            zero = valid.loc[valid.delta.eq(0)]
            row.update(value_at_zero=float(zero.value.iloc[0]) if len(zero) else None,
                       best_value=float(best.value), best_delta=float(best.delta),
                       best_delta_um=float(best.delta_um),
                       extremum_at_boundary=bool(np.isclose(abs(best.delta_um), group.delta_um.abs().max())))
        records.append(row)
    return records


def plot_curves(frame, output, title, settings, formats):
    metrics = [m for m in ('voxel_iou', 'bright_region_distance') if m in set(frame.metric)]
    fig, axes = plt.subplots(len(metrics), 3, figsize=(16, 4.2*len(metrics)), squeeze=False)
    for mi, metric in enumerate(metrics):
        for ai, axis in enumerate('xyz'):
            ax = axes[mi, ai]
            rows = frame.loc[frame.metric.eq(metric) & frame.axis.eq(axis)]
            for cid, group in rows.groupby('channel', sort=True):
                group = group.sort_values('delta_um')
                valid = group.value.notna()
                n = int(group.n_pairs.iloc[0])
                label = cid if metric == 'voxel_iou' else f'{cid}: n={n}'
                if not valid.any():
                    ax.plot([], [], label=f'{label}; no valid data')
                    continue
                line, = ax.plot(group.delta_um, group.value, '.-', label=label)
                value = group.value.max() if metric == 'voxel_iou' else group.value.min()
                best = group.loc[np.isclose(group.value, value)]
                best = best.loc[best.delta_um.abs().idxmin()]
                ax.plot(best.delta_um, best.value, 'v', color=line.get_color())
                extreme_name = 'max' if metric == 'voxel_iou' else 'min'
                unit = 'sl' if axis == 'z' else 'px'
                line.set_label(f'{label}; {extreme_name} {best.delta:+g} {unit}')
                if metric == 'voxel_iou' and group.unaligned_iou.notna().any():
                    ax.axhline(group.unaligned_iou.dropna().iloc[0], color=line.get_color(), linestyle=':', alpha=.65)
            ax.axvline(0, color='black', linestyle='--', alpha=.5)
            ax.set_title(f'{axis.upper()} | ' + ('Soma voxel IoU' if metric == 'voxel_iou' else 'Bright-region center'))
            ax.set_xlabel(f'Extra {axis.upper()} shift (um; + = increasing coordinate)')
            ax.set_ylabel('Voxel IoU (higher is better)' if metric == 'voxel_iou' else 'Median 3D distance (um; lower is better)')
            ax.grid(alpha=.2)
            if not rows.empty:
                ax.set_xlim(rows.delta_um.min(), rows.delta_um.max())
            ax.legend(fontsize=8)
            scale = settings['z_resolution_um' if axis == 'z' else 'xy_resolution_um']
            top = ax.secondary_xaxis('top', functions=(lambda u, s=scale: u/s, lambda p, s=scale: p*s))
            top.set_xlabel('slices' if axis == 'z' else 'pixels', fontsize=8)
    fig.suptitle(title)
    fig.text(.5, .015, 'Fixed baseline containment pairs; no rematching. Pair selection can favor zero. '
             'Sensitivity is diagnostic, not independent registration accuracy.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, .94))
    for fmt in formats:
        fig.savefig(f'{output}.{fmt}', dpi=150)
    plt.close(fig)


def overview(frame, output, title, settings, formats):
    # Every tile contributes one value at each displacement, independent of pair count.
    usable = frame.loc[frame.value.notna()]
    if usable.empty:
        return
    keys = ['channel', 'metric', 'axis', 'delta', 'delta_um']
    aggregate = usable.groupby(keys, as_index=False).agg(
        value=('value', 'mean'), unaligned_iou=('unaligned_iou', 'mean'), n_pairs=('n_pairs', 'sum'))
    aggregate['tile'] = 'overview'
    aggregate['status'] = 'diagnostic'
    plot_curves(aggregate, output, title + ' | equal tile weight', settings, formats)
