"""Stage-local resume policy and input routing (no inference dependencies)."""
import copy
import logging
from pathlib import Path
from src.config.loader import load_config
from src.core.provenance import atomic_json
from src.core.result_layout import result_paths


def downstream_directory(derived, channel):
    return derived.get('channel_sources', {}).get(channel['id'], derived['pATH_DET_FILTERED'])


def channel_sources(config, derived):
    return {ch['id']: (derived['pATH_ALIGN_OFFSETS']
                      if config.get('pipeline_mode') == 'pre_align' and not ch.get('double_exposure')
                      else derived['pATH_DET_FILTERED'])
            for ch in config['channels_routing'] if ch.get('active', True)}


def saved_channel_directory(base, channel, config=None):
    """Resolve viewers with the same routing as inference, including old runs."""
    if config is None:
        for name in ('runtime_config.json', 'alignment_runtime_config.json'):
            path = Path(base) / name
            if path.is_file():
                config = load_config(path)
                break
    if config:
        selected = channel_sources(config, result_paths(str(base)))[channel['id']]
        if config.get('downstream_layout_version') == 2 or any(Path(selected).glob(f"*_{channel['id']}_result.csv")):
            return selected
        return str(Path(base) / '4_2d_filtered')
    final = Path(base) / '4_2d_filtered'
    if not any(final.glob(f"*_{channel['id']}_result.csv")) and not channel.get('double_exposure'):
        aligned = Path(base) / '3_2d_aligned'
        if any(aligned.glob(f"*_{channel['id']}_result.csv")):
            return str(aligned)
    return str(final)


def resume_config(base, current, source_config):
    """Bootstrap missing history without claiming current models produced old raw data."""
    path = Path(base) / 'runtime_config.json'
    if path.is_file():
        return load_config(path)
    if not any((Path(base) / '1_2d_raw').glob('*_result.csv')):
        return copy.deepcopy(current)
    historical = Path(base) / 'alignment_runtime_config.json'
    previous = load_config(historical) if historical.is_file() else copy.deepcopy(current)
    snapshot = copy.deepcopy(current)
    snapshot['resume_bootstrap'] = {
        'source_config': str(source_config),
        'historical_alignment_config': str(historical) if historical.is_file() else None,
        'raw_origin_verified': False,
    }
    atomic_json(Path(base) / 'resume_bootstrap_config.json', snapshot)
    previous['resume_bootstrap'] = snapshot['resume_bootstrap']
    logging.warning('No runtime_config.json: adopting existing artifacts using current config; '
                    'historical raw model/settings remain unverified.')
    return previous


def raw_detection_settings(config):
    """Exclude filtering, geometry and report settings from raw detection identity."""
    from src.core.detection_filter import FILTER_KEYS
    dp = copy.deepcopy(config.get('detection_params', {}))
    for key in ('generate_histograms', 'sTARTID', 'eNDID', 'xy_resolution_um',
                'z_resolution_um', 'tILESIZE', 'cross_tile_iomin_thresh', 'allow_grid_fallback'):
        dp.pop(key, None)
    for value in dp.values():
        if isinstance(value, dict):
            for key in (*FILTER_KEYS, 'nucleus_mean_min'):
                value.pop(key, None)
    return dp
