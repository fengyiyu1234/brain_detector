"""QC-only nuclear-box fluorescence sensitivity; never fits or writes offsets.

The fixed cohort is selected from filtered nuclear detections in each saved QC
ROI, independently of soma fluorescence. All shifts, raw baseline, and old
relative offsets use the same boxes. Images remain unresampled: boxes are mapped
back into each soma channel's original TIFF coordinates. Scores are diagnostic
fluorescence enrichment, not a probability or proof of nuclear colocalization.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from functools import lru_cache
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
import zlib
import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.config.loader import load_config


def rectangle_means(integral, boxes):
    """Means over [floor(x1),ceil(x2)) x [floor(y1),ceil(y2)); no clipping."""
    x1, y1 = np.floor(boxes[:, :2]).astype(int).T
    x2, y2 = np.ceil(boxes[:, 2:4]).astype(int).T
    height, width = np.array(integral.shape) - 1
    if np.any((x1 < 0) | (y1 < 0) | (x2 > width) | (y2 > height) | (x2 <= x1) | (y2 <= y1)):
        raise ValueError('Box outside image or empty; cohort must be checked before scanning')
    return (integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]) / ((x2-x1)*(y2-y1))



def local_box_means(integral, boxes, ring_width_px=4, ring_gap_px=2):
    """Core and annulus means, with an excluded gap and no edge clipping."""
    if (isinstance(ring_width_px, bool) or not isinstance(ring_width_px, int)
            or ring_width_px <= 0 or isinstance(ring_gap_px, bool)
            or not isinstance(ring_gap_px, int) or ring_gap_px < 0):
        raise ValueError('Ring width must be a positive integer; gap a nonnegative integer')
    core = np.column_stack((np.floor(boxes[:, :2]), np.ceil(boxes[:, 2:4])))
    expansion = np.array([-1, -1, 1, 1])
    inner = core + ring_gap_px * expansion
    outer = core + (ring_gap_px + ring_width_px) * expansion
    inner_area = (inner[:, 2]-inner[:, 0]) * (inner[:, 3]-inner[:, 1])
    outer_area = (outer[:, 2]-outer[:, 0]) * (outer[:, 3]-outer[:, 1])
    core_mean = rectangle_means(integral, core)
    ring_mean = (rectangle_means(integral, outer)*outer_area
                 - rectangle_means(integral, inner)*inner_area) / (outer_area-inner_area)
    return core_mean, ring_mean


def scan_specifications(settings, options=None):
    """Fine QC scan, independent of the saved coarse IoU curve grid."""
    options = options or {}
    sweep = options.get('sweep', {
        'x': {'range_px': 8, 'step_px': 1},
        'y': {'range_px': 8, 'step_px': 1},
        'z': {'range_slices': 2, 'step_slices': 1},
    })
    rows = []
    for axis in 'xyz':
        unit = 'slices' if axis == 'z' else 'px'
        radius, step = sweep[axis][f'range_{unit}'], sweep[axis][f'step_{unit}']
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (radius, step)) or radius < 0 or step <= 0:
            raise ValueError('Scan radius must be a nonnegative integer and step a positive integer')
        deltas = sorted({0, -radius, radius} | set(range(-radius, radius+1, step))
                        | {-v for v in range(-radius, radius+1, step)})
        resolution = settings['z_resolution_um'] if axis == 'z' else settings['xy_resolution_um']
        rows.extend({'axis': axis, 'delta': d, 'delta_um': d*resolution} for d in deltas)
    return pd.DataFrame(rows)

def common_cohort(boxes, shifts, width, height, depth, fit_window=None, guard=0, xy_margin=0):
    """Fixed boxes valid for every evaluated relative shift, including baselines."""
    keep = np.isfinite(boxes).all(axis=1)
    for dx, dy, dz in shifts:
        keep &= ((np.floor(boxes[:, 0] + dx) >= xy_margin) & (np.floor(boxes[:, 1] + dy) >= xy_margin)
                 & (np.ceil(boxes[:, 2] + dx) <= width-xy_margin) & (np.ceil(boxes[:, 3] + dy) <= height-xy_margin)
                 & (boxes[:, 4] + dz >= 1) & (boxes[:, 4] + dz <= depth))
        if fit_window is not None:
            z = boxes[:, 4] + dz
            keep &= (z < fit_window[0] - guard) | (z >= fit_window[1] + guard)
    keep &= (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return keep


def relative_shift(nuclear, soma):
    return tuple(nuclear[k] - soma[k] for k in ('dx', 'dy', 'dz'))


def image_files(root, tile):
    root = Path(root)
    directory = next((p for p in (root / tile, root / tile.split('_')[0] / tile) if p.is_dir()), None)
    if directory is None:
        raise FileNotFoundError(f'{root}: {tile}')
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in ('.tif', '.tiff') and not p.name.startswith('.'))


def read_roi_boxes(path, shift, region):
    x0, x1, y0, y1, z0, z1 = region
    found = []
    for frame in pd.read_csv(path, usecols=['x1','y1','x2','y2','z'], chunksize=200000):
        b = frame[['x1','y1','x2','y2','z']].to_numpy(dtype=float)
        cx = (b[:,0]+b[:,2])/2 + shift['dx']
        cy = (b[:,1]+b[:,3])/2 + shift['dy']
        z = b[:,4] + shift['dz']
        keep = (cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1) & (z >= z0) & (z < z1)
        found.append(b[keep])
    return np.concatenate(found) if found else np.empty((0,5))


def evaluate_tile(job):
    tile, result, qc, config_path, max_boxes, seed = job[:6]
    options = job[6] if len(job) > 6 else {}
    ring_width = options.get('ring_width_px', 4)
    ring_gap = options.get('ring_gap_px', 2)
    result, qc = Path(result), Path(qc)
    config = load_config(config_path)
    routing = {ch['id']: ch for ch in config['channels_routing'] if ch.get('active', True)}
    offsets = load_config(result/'3_2d_aligned'/f'{tile}_offsets.json')
    measured = load_config(result/'3_2d_aligned'/f'{tile}_measured_offsets.json')
    fit = measured.get('sample_window')
    fit_window = (fit['z_start'], fit['z_stop_exclusive']) if fit else None
    prior_file = result/'old_results'/'3_2d_aligned'/f'{tile}_offsets.json'
    prior = load_config(prior_file) if prior_file.is_file() else {}
    curves = pd.read_csv(qc/'tile_curves'/f'{tile}_curves.csv')
    settings = load_config(result/'3_2d_aligned'/'_align_settings.json')
    specifications = scan_specifications(settings, options)
    regions = curves[['region_id','x0','x1','y0','y1','z0','z1','actual_z_guard_slices']].drop_duplicates()
    rows, statuses = [], []
    nuclear_rois = {}
    for soma in ('GFP', 'RFP'):
        files = image_files(config['paths'][routing[soma]['dir_key']], tile)
        @lru_cache(maxsize=64)
        def image_data(z):
            image = tifffile.imread(files[int(z)-1])
            if image.ndim != 2 or not np.isfinite(image).all():
                raise ValueError(f'Invalid 2D TIFF: {files[int(z)-1]}')
            # Fixed image-level reference statistics; never recomputed inside shifted boxes.
            bg, bright = np.percentile(image[::4,::4], [50,99.5])
            scale = float(bright-bg)
            integral = np.pad(image.astype(np.float64).cumsum(0).cumsum(1), ((1,0),(1,0)))
            return integral, float(bg), scale
        first = tifffile.imread(files[0])
        height, width = first.shape
        del first
        for nuc in ('Sox9','Olig2'):
            relative = relative_shift(offsets[nuc], offsets[soma])
            old = relative_shift(prior[nuc], prior[soma]) if nuc in prior and soma in prior else None
            for region in regions.itertuples(index=False):
                region_values = (region.x0,region.x1,region.y0,region.y1,region.z0,region.z1)
                roi_key = (nuc, region.region_id)
                if roi_key not in nuclear_rois:
                    nuclear_rois[roi_key] = read_roi_boxes(result/'2_2d_filtered'/f'{tile}_{nuc}_result.csv', offsets[nuc], region_values)
                boxes = nuclear_rois[roi_key]
                shifts = [tuple(relative[i]+(float(row.delta) if i == 'xyz'.index(row.axis) else 0)
                                for i in range(3)) for row in specifications.itertuples(index=False)]
                check = shifts + [(0,0,0)] + ([old] if old is not None else [])
                # Source nuclear detections and soma pixels must both stay outside fitting data.
                keep = common_cohort(boxes, check, width, height, len(files), fit_window,
                                     region.actual_z_guard_slices, xy_margin=ring_gap+ring_width)
                boxes = boxes[keep]
                available = len(boxes)
                if available > max_boxes:
                    rng = np.random.default_rng(seed + zlib.crc32(f'{tile}/{soma}/{nuc}/{region.region_id}'.encode()))
                    boxes = boxes[np.sort(rng.choice(available, max_boxes, replace=False))]
                # Remove boxes on constant/invalid normalization slices for ALL scan points.
                valid_z = np.ones(len(boxes), dtype=bool)
                for shift in check:
                    for z in np.unique(boxes[:,4]+shift[2]):
                        if image_data(int(z))[2] <= 0:
                            valid_z &= boxes[:,4]+shift[2] != z
                boxes = boxes[valid_z]
                def score(shift):
                    if not len(boxes):
                        return (np.nan,)*4
                    values, raw_values, ring_values, contrasts = [], [], [], []
                    for z in np.unique(boxes[:,4]+shift[2]):
                        b = boxes[boxes[:,4]+shift[2] == z, :4].copy()
                        b += np.array([shift[0],shift[1],shift[0],shift[1]])
                        integral, bg, scale = image_data(int(z))
                        means, rings = local_box_means(integral, b, ring_width, ring_gap)
                        raw_values.extend(means)
                        ring_values.extend(rings)
                        values.extend((means-bg)/scale)
                        contrasts.extend((means-rings)/scale)
                    return tuple(float(np.mean(v)) for v in (values, raw_values, ring_values, contrasts))
                baseline, baseline_raw, baseline_ring, baseline_contrast = score((0,0,0))
                old_value, old_raw, old_ring, old_contrast = score(old) if old is not None else (np.nan,)*4
                for point, shift in zip(specifications.itertuples(index=False), shifts):
                    value, raw_value, ring_value, contrast = score(shift)
                    rows.append(dict(tile=tile, region_id=region.region_id, soma=soma, nucleus=nuc,
                                     axis=point.axis, delta=point.delta, delta_um=point.delta_um,
                                     intensity_enrichment=value, raw_mean_intensity=raw_value,
                                     raw_ring_mean_intensity=ring_value, local_contrast=contrast,
                                     unaligned_ring_mean=baseline_ring, unaligned_local_contrast=baseline_contrast,
                                     old_ring_mean=old_ring, old_local_contrast=old_contrast,
                                     ring_width_px=ring_width, ring_gap_px=ring_gap,
                                     unaligned_enrichment=baseline, unaligned_raw_mean=baseline_raw,
                                     old_enrichment=old_value, old_raw_mean=old_raw, n_boxes=len(boxes),
                                     old_relative_dx=old[0] if old else np.nan,
                                     old_relative_dy=old[1] if old else np.nan,
                                     old_relative_dz=old[2] if old else np.nan,
                                     source_boxes_available=available, z0=region.z0,z1=region.z1,
                                     relative_dx=relative[0],relative_dy=relative[1],relative_dz=relative[2]))
                statuses.append(dict(tile=tile,soma=soma,nucleus=nuc,region_id=region.region_id,
                                     n_boxes=len(boxes),eligible_boxes=available,
                                     status='evaluated' if len(boxes) else 'no_valid_nuclear_boxes'))
        image_data.cache_clear()
    return rows, statuses


def summarize_curves(frame, groups):
    rows = []
    for key, g in frame.groupby(groups):
        row = dict(zip(groups, key if isinstance(key,tuple) else (key,)))
        zero = g.loc[g.delta.eq(0)].iloc[0]
        finite = g.dropna(subset=['intensity_enrichment'])
        row.update(aligned_enrichment=zero.intensity_enrichment, unaligned_enrichment=zero.unaligned_enrichment,
                   old_enrichment=zero.old_enrichment, n_boxes=zero.n_boxes, peak_delta=np.nan,
                   peak_delta_um=np.nan,peak_enrichment=np.nan,curve_range=np.nan)
        if len(finite):
            maximum = finite.intensity_enrichment.max()
            peak = finite.loc[np.isclose(finite.intensity_enrichment,maximum)].assign(distance=lambda x: x.delta.abs()).sort_values('distance').iloc[0]
            row.update(peak_delta=peak.delta,peak_delta_um=peak.delta_um,peak_enrichment=maximum,
                       curve_range=maximum-finite.intensity_enrichment.min())
        if 'local_contrast' in g:
            row.update(aligned_local_contrast=zero.local_contrast,
                       unaligned_local_contrast=zero.unaligned_local_contrast,
                       old_local_contrast=zero.old_local_contrast,
                       contrast_peak=np.nan, contrast_peak_delta=np.nan, contrast_peak_delta_um=np.nan,
                       contrast_trough=np.nan, contrast_trough_delta=np.nan, contrast_trough_delta_um=np.nan)
            contrast = g.dropna(subset=['local_contrast'])
            for kind, value in (('peak', contrast.local_contrast.max()), ('trough', contrast.local_contrast.min())):
                if len(contrast):
                    extremum = contrast.loc[np.isclose(contrast.local_contrast,value)].assign(
                        distance=lambda x: x.delta.abs()).sort_values(['distance','delta']).iloc[0]
                    row.update({f'contrast_{kind}': value, f'contrast_{kind}_delta': extremum.delta,
                                f'contrast_{kind}_delta_um': extremum.delta_um})
        rows.append(row)
    return pd.DataFrame(rows)


def plot_curves(frame, target, title, individual=False, metric="intensity_enrichment"):
    local = metric == "local_contrast"
    baseline_col = "unaligned_local_contrast" if local else "unaligned_enrichment"
    old_col = "old_local_contrast" if local else "old_enrichment"
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4,3,figsize=(15,14), squeeze=False)
    for i,(soma,nuc) in enumerate((('GFP','Sox9'),('GFP','Olig2'),('RFP','Sox9'),('RFP','Olig2'))):
        for j,axis in enumerate('xyz'):
            ax=axes[i,j]
            sub=frame.loc[frame.soma.eq(soma)&frame.nucleus.eq(nuc)&frame.axis.eq(axis)]
            if not individual:
                for _,g in sub.groupby(['tile','region_id']):
                    g=g.sort_values('delta')
                    ax.plot(g.delta_um,g[metric],color='.7',alpha=.4,lw=.6)
            curve=sub.groupby(['delta','delta_um'],as_index=False)[[metric,baseline_col,old_col]].mean().sort_values('delta')
            ax.plot(curve.delta_um,curve[metric],'o-',color='crimson',ms=3,
                    label='inside minus surrounding ring' if local else 'nuclear-box enrichment')
            if len(curve):
                for col,color,label in [(baseline_col,'darkorange','unaligned'),(old_col,'purple','old offsets')]:
                    value=curve[col].iloc[0]
                    if np.isfinite(value):
                        ax.axhline(value,color=color,ls='--',label=label)
                if curve[metric].notna().any():
                    peak=curve.loc[curve[metric].idxmax()]
                    ax.scatter([peak.delta_um],[peak[metric]],color='black',s=20)
                    if local:
                        trough=curve.loc[curve[metric].idxmin()]
                        ax.scatter([trough.delta_um],[trough[metric]],color='black',marker='v',s=20)
                else:
                    ax.text(.5,.5,'No valid signal / boxes',transform=ax.transAxes,ha='center')
            if len(curve):
                xmin,xmax=float(curve.delta_um.min()),float(curve.delta_um.max())
                margin=.05*(xmax-xmin) if xmax>xmin else .5
                ax.set_xlim(xmin-margin,xmax+margin)
            ax.axvline(0,color='steelblue',ls=':')
            ax.set_title(f'{nuc} boxes in {soma}: {axis.upper()}')
            ax.set_xlabel('Additional nuclear shift (um)')
            ax.set_ylabel('Signed local contrast' if local else 'Normalized nuclear-box intensity')
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
    fig.suptitle(title)
    definition = ('Contrast = (box mean - surrounding ring mean) / (image P99.5 - P50). Circle: maximum; triangle: minimum.'
                  if local else 'Intensity = (box mean - image P50) / (image P99.5 - P50).')
    fig.text(.02,.01,'Fixed cohort for all shifts and baselines; overview weights tiles equally. Peaks or troughs are diagnostic.\n'
             + definition + '\nNo offsets were fitted or changed.',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.97))
    for suffix in ('png','pdf'):
        fig.savefig(str(target)+'.'+suffix,dpi=150)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample',required=True)
    parser.add_argument('--config',required=True)
    parser.add_argument('--qc-dir',required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--max-boxes',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--eval-config',help='Read the fine sweep from this QC evaluation JSON')
    parser.add_argument('--ring-width-px',type=int,default=4)
    parser.add_argument('--ring-gap-px',type=int,default=2)
    parser.add_argument('--output-subdir',default='intensity_local_contrast',help='New output folder within qc-dir')
    parser.add_argument('--follow-progress',action='store_true',help='Process each IoU tile as its checkpoint becomes available')
    args=parser.parse_args()
    if args.workers < 1 or args.max_boxes < 1:
        parser.error('workers and max-boxes must be positive')
    if args.ring_width_px <= 0 or args.ring_gap_px < 0:
        parser.error('ring-width-px must be positive and ring-gap-px nonnegative')
    if Path(args.output_subdir).name != args.output_subdir or args.output_subdir in ('.','..',''):
        parser.error('output-subdir must be a single folder name within qc-dir')
    options = dict(ring_width_px=args.ring_width_px, ring_gap_px=args.ring_gap_px)
    if args.eval_config:
        options['sweep'] = load_config(args.eval_config)['sweep']
    result,qc=Path(args.sample),Path(args.qc_dir)
    settings=load_config(result/'3_2d_aligned'/'_align_settings.json')
    specifications=scan_specifications(settings, options)
    sampling_path = qc/'sampling.json'
    expected = sorted(p.name.removesuffix('_offsets.json') for p in (result/'3_2d_aligned').glob('*_offsets.json')
                      if not p.name.endswith('_measured_offsets.json'))
    tiles = expected if args.follow_progress else load_config(sampling_path)['selected_tiles']
    target=qc/args.output_subdir
    target.mkdir(exist_ok=True)
    all_rows,all_status=[],[]
    completed=set()
    submitted=set()
    last_activity=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
        pending={}
        while len(completed)<len(tiles):
            ready = set(tiles)
            if args.follow_progress and not sampling_path.is_file():
                try:
                    progress = load_config(qc/'progress.json')
                    ready = {r['tile'] for r in progress['attempted'] if r['accepted']}
                except (OSError, ValueError):
                    ready = set()
            for tile in tiles:
                checkpoint=qc/'tile_curves'/f'{tile}_curves.csv'
                if tile in ready and tile not in submitted and checkpoint.is_file():
                    job=(tile,str(result),str(qc),args.config,args.max_boxes,args.seed,options)
                    pending[pool.submit(evaluate_tile,job)]=tile
                    submitted.add(tile)
            if not pending:
                if time.monotonic()-last_activity>3600:
                    raise TimeoutError('No new completed IoU tile for one hour')
                if sampling_path.is_file():
                    available=set(load_config(sampling_path)['selected_tiles'])
                    missing=set(tiles)-available
                    if missing:
                        raise ValueError(f'IoU QC omitted tiles: {sorted(missing)}; completed intensity checkpoints retained')
                time.sleep(5)
                continue
            done,_=wait(pending,timeout=5,return_when=FIRST_COMPLETED)
            for future in done:
                tile=pending.pop(future)
                rows,statuses=future.result()
                frame=pd.DataFrame(rows)
                frame.to_csv(target/f'{tile}_intensity_curves.csv',index=False)
                plot_curves(frame,target/f'{tile}_intensity_sensitivity',f'T70: {tile}; fixed-cohort intensity QC',True)
                plot_curves(frame,target/f'{tile}_local_contrast_sensitivity',f'T70: {tile}; local contrast QC',True,metric='local_contrast')
                all_rows.extend(rows)
                all_status.extend(statuses)
                completed.add(tile)
                last_activity=time.monotonic()
                (target/'progress.json').write_text(json.dumps(dict(requested=len(tiles),completed=len(completed),tiles=sorted(completed)),indent=2),encoding='utf-8')
                print(f'Intensity [{len(completed)}/{len(tiles)}] {tile}',flush=True)
    # The IoU parent writes its full sampling manifest after its tile checkpoints.
    for _ in range(60):
        if sampling_path.is_file():
            break
        time.sleep(1)
    manifest=load_config(sampling_path)
    df=pd.DataFrame(all_rows)
    df.to_csv(target/'T70_intensity_curves.csv',index=False)
    pd.DataFrame(all_status).to_csv(target/'T70_intensity_status.csv',index=False)
    summarize_curves(df,['tile','region_id','soma','nucleus','axis']).to_csv(target/'T70_intensity_summary.csv',index=False)
    plot_curves(df,target/'T70_intensity_overview','T70: held-out intensity sensitivity; equal tile weight')
    plot_curves(df,target/'T70_local_contrast_overview','T70: held-out local contrast; equal tile weight',metric='local_contrast')
    (target/'method.json').write_text(json.dumps(dict(
        metric='Mean per-box (raw soma-channel mean - image P50) / (image P99.5 - P50)',
        local_contrast='Mean per-box (inside mean - surrounding ring mean) / (image P99.5 - P50); signed',
        ring_width_px=args.ring_width_px, ring_gap_px=args.ring_gap_px,
        scan_points=specifications.to_dict('records'),
        normalization_percentiles_spatial_stride=4, max_boxes_per_pair_region=args.max_boxes,
        cohort='Fixed filtered nuclear boxes selected independently of soma intensity; complete ring valid for every shift and baseline',
        image_transform='None: inverse mapping of boxes into raw soma TIFFs; one-based source slice indices',
        fitting_data_excluded=True,guard='Same raw alignment window plus recorded IoU QC guard, including old offsets',
        old_baseline='Old within-tile relative offsets on the SAME currently filtered nuclear boxes; not old filtered data',
        limitations=['Soma/cytoplasmic expression does not imply a nuclear brightness maximum.',
                     'All eligible nuclei are included; non-MADM nuclei dilute sensitivity.',
                     'One held-out 10-slice ROI per tile does not establish depth-invariant registration.'],
        source_sampling_sha256=hashlib.sha256((qc/'sampling.json').read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_config=args.config,selected_tiles=tiles,intensity_used_for_alignment=False),indent=2),encoding='utf-8')
    print(f'Intensity QC complete: {len(tiles)} tiles; {target}',flush=True)


if __name__=='__main__':
    mp.freeze_support()
    main()
