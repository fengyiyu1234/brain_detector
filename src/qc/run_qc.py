"""Pipeline hooks; disabled QC does not import image-processing dependencies."""
import logging
from pathlib import Path
from .common import save_json


def run_check(name, function, results_dir, *args, fail_on_error=False, **kwargs):
    """Run one sample QC check; failures are logged and saved to qc/<name>_error.json only."""
    error_path = Path(results_dir)/'qc'/f'{name}_error.json'
    try:
        summary = function(*args, **kwargs)
        if error_path.is_file():
            error_path.unlink()
        status = summary.get('status', '?') if isinstance(summary, dict) else '?'
        log = logging.warning if status in ('WARN', 'FAIL') else logging.info
        log(f"{'⚠️' if status == 'WARN' else '✔️'} [QC] {name}: {status}")
        return summary
    except Exception as exc:
        logging.exception(f'QC check {name} failed')
        save_json(error_path, {'status': 'failed', 'error': str(exc)})
        if fail_on_error:
            raise
        return {'status': 'FAIL', 'error': str(exc)}


def review_enabled(config):
    return (config.get('qc') or {}).get('review', {}).get('enabled', True)


def run_post_alignment_checks(config, results_dir):
    """A (unaligned tiles) and C (SNR filter) after Stage 2.5; never stop the pipeline,
    except A with qc.alignment.fail_on_unaligned."""
    qc = config.get('qc') or {}
    if config.get('pipeline_mode') == 'pre_align' and (qc.get('unaligned') or {}).get('enabled', True):
        from .unaligned import run as run_unaligned
        fail = bool((qc.get('alignment') or {}).get('fail_on_unaligned', False))
        try:
            summary = run_unaligned(config, results_dir)
        except RuntimeError:
            if fail:
                raise
            summary = None
        except Exception as exc:
            logging.exception('QC check unaligned failed')
            save_json(Path(results_dir)/'qc'/'unaligned_error.json', {'status': 'failed', 'error': str(exc)})
            summary = None
        if summary:
            (Path(results_dir)/'qc'/'unaligned_error.json').unlink(missing_ok=True)
            if summary.get('unaligned_tiles'):
                logging.warning(f"⚠️ [QC] {len(summary['unaligned_tiles'])} 个有细胞的 tile 对齐得分为 0（没对齐）: "
                                f"{', '.join(summary['unaligned_tiles'])}")
            else:
                logging.info(f"✔️ [QC] unaligned: {summary.get('status')} {summary.get('counts', '')}")
    if (qc.get('snr') or {}).get('enabled', True):
        from .snr_qc import run as run_snr
        run_check('snr', run_snr, results_dir, config, results_dir)


def run_seam_checks(config, results_dir, geom, routing, src_dirs, derived):
    """B (seam consistency of the Stage 2.9 positions) and top-seam dedup images after Stage 3."""
    qc = config.get('qc') or {}
    if (qc.get('seam_consistency') or {}).get('enabled', True):
        from .seam_consistency import run as run_seams
        run_check('seam_consistency', run_seams, results_dir, config, results_dir)
    dedup = qc.get('stitch_dedup') or {}
    if dedup.get('enabled', True) and review_enabled(config) and geom.get('seam_offsets_csv'):
        from .stitch_dedup_top import run as run_top
        run_check('stitch_dedup_top', run_top, results_dir, config, results_dir, geom, routing, src_dirs,
                  derived, int((qc.get('review') or {}).get('top_dedup_seams', 3)))


def run_final_checks(config, results_dir, src_dirs):
    """D (distribution), E (integrity, off by default) and the F summary after Stage 4."""
    qc = config.get('qc') or {}
    if (qc.get('distribution') or {}).get('enabled', True):
        from .distribution import run as run_distribution
        run_check('distribution', run_distribution, results_dir, config, results_dir, src_dirs)
    if (qc.get('integrity') or {}).get('enabled', False):
        from .integrity import run as run_integrity
        run_check('integrity', run_integrity, results_dir, config, results_dir)
    from .summary import run as run_summary
    return run_check('summary', run_summary, results_dir, config, results_dir)


def run_alignment_stage(config, results_dir, tiles):
    report = _run_alignment_qc(config, results_dir, tiles)
    run_post_alignment_checks(config, results_dir)
    return report


def _run_alignment_qc(config, results_dir, tiles):
    options = config.get('qc', {}).get('alignment', {})
    if not options.get('enabled', False):
        return {'status': 'disabled'}
    try:
        if config.get('pipeline_mode') != 'pre_align':
            raise ValueError('Alignment QC requires pre_align saved offsets and filtered detections')
        from .alignment import run
        report = run(results_dir, options=options, tiles=tiles, config=config)
        if report['status'] == 'failed':
            raise RuntimeError(f"Alignment QC failed for {len(report['failures'])} tile(s)")
        report_path = Path(results_dir)/options.get('output_dir', 'qc/alignment')/'qc_report.md'
        if report.get('flagged'):
            logging.warning(f"⚠️ [QC] {len(report['flagged'])}/{len(tiles)} 个 tile 对齐可疑，详见 {report_path}")
            for row in report['flagged']:
                logging.warning(f"    {row['tile']} [{row['bad_channels']}]: {row['reasons']}")
        else:
            logging.info(f"✔️ [QC] 未发现对齐可疑的 tile（{report_path}）")
        if report.get('insufficient'):
            logging.info(f"  [QC] 配对/信号不足、无法判断: {', '.join(report['insufficient'])}")
        return report
    except Exception as exc:
        logging.exception('Alignment QC failed')
        # Preserve detailed tile failures when they are already available.
        save_json(Path(results_dir)/'qc'/'alignment_error.json', {'status':'failed', 'error':str(exc)})
        if options.get('fail_on_error', False):
            raise
        return {'status': 'failed', 'error': str(exc)}


def run_stitch_dedup_stage(config, results_dir, geom, routing, src_dirs, derived, n_workers=None):
    """Seam deduplication QC after Stage 3; on by default, never stops the pipeline unless asked."""
    options = (config.get('qc') or {}).get('stitch_dedup') or {}
    if not options.get('enabled', True):
        return {'status': 'disabled'}
    if not geom.get('seam_offsets_csv') or geom.get('num_tiles', 1) <= 1:
        logging.info('[QC] Seam dedup QC skipped: needs Stage 2.97 seam offsets and more than one tile')
        return {'status': 'skipped'}
    try:
        from .stitch_dedup import run
        report = run(config, results_dir, geom, routing, src_dirs, derived, n_workers)
        for row in report['summary'].itertuples():
            logging.info(f"  [QC] {row.channel}: 接缝重叠区残留 3D 重复 {row.residual_3d_pct:.2f}% "
                         f"(单条接缝中位数 {row.seam_residual_3d_pct_p50:.2f}%, 最大 {row.seam_residual_3d_pct_max:.2f}%)")
        logging.info(f"✔️ [QC] 接缝去重 QC 完成（{report['report']}）")
        if report['images_written']:
            logging.info(f"  [QC] 新写出 {len(report['images_written'])} 张接缝检查图")
        return report
    except Exception as exc:
        logging.exception('Seam dedup QC failed')
        save_json(Path(results_dir)/'qc'/'stitch_dedup_error.json', {'status': 'failed', 'error': str(exc)})
        if options.get('fail_on_error', False):
            raise
        return {'status': 'failed', 'error': str(exc)}
