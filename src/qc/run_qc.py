"""Pipeline hooks; disabled QC does not import image-processing dependencies."""
import logging
from pathlib import Path
from .common import save_json


def run_alignment_stage(config, results_dir, tiles):
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
