"""Validated named boundaries for stopping and resuming the pipeline."""
import logging

STEPS = {
    'setup': '1', 'detection': '2', 'filtering': '2.25',
    'alignment': '2.5', 'alignment_qc': '2.55', 'fusion': '2.6',
    'publish_filtered': '2.75', 'histograms': '2.8',
    'tile_positions': '2.9', 'image_stitching': '2.95',
    'z_linking': '3', 'colocalization': '3.5', 'exports': '4',
}


def resolve_stop_step(config):
    removed = {'stop_after_detection', 'stop_before_stitching'} & config.keys()
    if removed:
        raise ValueError(f"Removed options {sorted(removed)}; use stop_after_step "
                         "(detection, tile_positions, or null for the complete pipeline).")
    value = config.get('stop_after_step')
    if value is None:
        return None
    aliases = {number: name for name, number in STEPS.items()}
    step = aliases.get(format(value, 'g') if isinstance(value, (int, float)) else str(value), value) if not isinstance(value, bool) else None
    if not isinstance(step, str) or step not in STEPS:
        raise ValueError(f"Invalid stop_after_step={value!r}; choose null or {', '.join(STEPS)}")
    pre = config.get('pipeline_mode', 'post_align') == 'pre_align'
    enabled = {
        'filtering': pre, 'alignment': pre,
        'alignment_qc': pre and config.get('qc', {}).get('alignment', {}).get('enabled', False),
        'fusion': any(ch.get('double_exposure') for ch in config.get('channels_routing', [])
                      if ch.get('active', True)),
        'histograms': config.get('detection_params', {}).get('generate_histograms', True),
        'tile_positions': pre and config.get('tile_position_params', {}).get('enabled', False),
        'image_stitching': config.get('direct_stitching', {}).get('enabled', False),
    }
    if step in enabled and not enabled[step]:
        raise ValueError(f"stop_after_step={step!r} refers to a disabled/inapplicable step; "
                         "enable that step or choose another boundary.")
    config['stop_after_step'] = step
    return step


def stop_after(config, step):
    if config.get('stop_after_step') == step:
        logging.info("stop_after_step=%s: step %s complete; checkpoints saved.", step, STEPS[step])
        raise SystemExit(0)
