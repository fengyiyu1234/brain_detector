"""Checkpointed, template-free tile geometry and merging XML generation."""
import json
import os
import subprocess
import sys


def _stamp(path):
    st = os.stat(path)
    return [st.st_size, st.st_mtime_ns]




_LAYOUT_NAMES = (
    ('1_tile_2d_prefiltered', '2_2d_filtered'),
    ('0_channel_alignment', '3_2d_aligned'),
    ('5_analysis_report', '5_2d_global'),
)


def _canonical_manifest(value, script, xml_script):
    """Compare migrated checkpoints by content while ignoring script edit times."""
    if isinstance(value, str):
        for old, new in _LAYOUT_NAMES:
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [_canonical_manifest(item, script, xml_script) for item in value]
    if isinstance(value, dict):
        return {
            _canonical_manifest(key, script, xml_script):
                _canonical_manifest(item, script, xml_script)
            for key, item in value.items() if key not in (script, xml_script)
        }
    return value
def solver_command(script, config_path, results_dir, det_dir, ref, frame, workers, options):
    command = [sys.executable, script, '--sample', results_dir, '--config', config_path,
               '--det-dir', det_dir, '--alignment-from', 'old-offsets',
               '--ref', ref, '--frame', frame, '--workers', str(workers)]
    allowed = {'model', 'win_xy', 'win_z', 'bin_xy', 'match_xy_soma',
               'match_xy_tf', 'match_z', 'edge_px', 'min_matches',
               'prior_w', 'huber', 'iters', 'n_slices'}
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"Unknown tile_position_params.options: {sorted(unknown)}")
    for key in sorted(options):
        command.extend(['--' + key.replace('_', '-'), str(options[key])])
    return command


def tile_position_frame(config):
    """Use one setting for alignment output, solved tile geometry, and Stage 3."""
    params = config.get('tile_position_params', {})
    if 'reference_channel' in params:
        raise ValueError(
            "tile_position_params.reference_channel is redundant; "
            "set stitching_reference_channel instead")
    frame = config.get('stitching_reference_channel')
    if not frame:
        raise ValueError(
            "stitching_reference_channel is required when tile position solving is enabled")
    return frame


def run_tile_position_stage(config, config_path, results_dir, det_dir, align_dir,
                            report_dir, tile_names, routing, script, xml_script):
    """Solve positions, generate every channel XML, then reuse identical outputs."""
    params = config.get('tile_position_params', {})
    frame = tile_position_frame(config)
    ref = config.get('pre_align_params', {}).get('reference_channel') or next(
        ch['id'] for ch in routing if ch.get('type') == 'soma')
    channels = [ch['id'] for ch in routing]
    if frame not in channels or ref not in channels:
        raise ValueError(f"Tile frame {frame} and alignment reference {ref} must be active channels")
    workers = int(params.get('workers') or config.get('pre_align_params', {}).get('n_workers') or 1)
    workers = max(1, min(workers, int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))))
    command = solver_command(script, config_path, results_dir, det_dir, ref, frame,
                             workers, params.get('options') or {})
    os.makedirs(report_dir, exist_ok=True)
    position_csv = os.path.join(report_dir, 'tile_positions.csv')
    xml_command = [sys.executable, xml_script, '--config', config_path,
                   '--positions', position_csv, '--out-dir', report_dir,
                   '--bytes-per-channel', str(params.get('xml_bytes_per_channel', 2)),
                   '--force']
    files = {}
    for tile in tile_names:
        for ch in channels:
            csv = os.path.join(det_dir, f'{tile}_{ch}_result.csv')
            if not os.path.isfile(csv):
                raise FileNotFoundError(f"Tile solver needs filtered detections: {csv}")
            files[csv] = _stamp(csv)
        offset = os.path.join(align_dir, f'{tile}_offsets.json')
        if not os.path.isfile(offset):
            raise FileNotFoundError(f"Tile solver needs channel alignment: {offset}")
        files[offset] = _stamp(offset)
    for ch in routing:
        channel_dir = config['paths'].get(ch['dir_key'])
        if not channel_dir or not os.path.isdir(channel_dir):
            raise FileNotFoundError(f"Original image directory missing for {ch['id']}: {channel_dir}")
        for tile in tile_names:
            tile_dir = os.path.join(channel_dir, tile.split('_')[0], tile)
            if not os.path.isdir(tile_dir):
                raise FileNotFoundError(f"Original image tile directory missing: {tile_dir}")
            files[tile_dir] = _stamp(tile_dir)
    files[script] = _stamp(script)
    files[xml_script] = _stamp(xml_script)
    outputs = [position_csv, os.path.join(report_dir, 'solution.json')]
    outputs.extend(os.path.join(report_dir, f"xml_merging_{ch}.xml") for ch in channels)
    signature = {'version': 3, 'reference_channel': ref, 'frame_channel': frame,
                 'solver_command': command, 'xml_command': xml_command, 'inputs': files,
                 'geometry_config': {
                     'detection_params': {key: config.get('detection_params', {}).get(key)
                                          for key in ('xy_resolution_um', 'z_resolution_um', 'tILESIZE')},
                     'channels_routing': routing,
                     'paths': {ch['dir_key']: config['paths'][ch['dir_key']] for ch in routing},
                 }}
    manifest_path = os.path.join(report_dir, '_pipeline_manifest.json')
    try:
        with open(manifest_path, encoding='utf-8') as handle:
            previous = json.load(handle)
    except (OSError, ValueError):
        previous = None
    frame_xml = os.path.join(report_dir, f'xml_merging_{frame}.xml')
    output_stamps = {path: _stamp(path) for path in outputs} if all(
        os.path.isfile(path) for path in outputs) else None
    if previous and output_stamps:
        if previous.get('signature') == signature and previous.get('outputs') == output_stamps:
            return frame_xml, False
        legacy = any(old in str(previous.get('signature')) for old, _ in _LAYOUT_NAMES)
        if (legacy and
                _canonical_manifest(previous.get('signature'), script, xml_script) ==
                _canonical_manifest(signature, script, xml_script) and
                _canonical_manifest(previous.get('outputs'), script, xml_script) ==
                _canonical_manifest(output_stamps, script, xml_script)):
            return frame_xml, False
    global_dirs = ('5_2d_global', '6_3d_global', '7_colocalization')
    if any(os.path.isdir(os.path.join(results_dir, d)) and
           any(name.endswith(('.csv', '.pkl')) for name in os.listdir(os.path.join(results_dir, d)))
           for d in global_dirs):
        raise RuntimeError("Tile positions or their inputs changed while global checkpoints exist. "
                           "Archive/regenerate global 2D, 3D, and colocalization outputs first.")
    subprocess.run(command, check=True)
    if not os.path.isfile(position_csv):
        raise RuntimeError(f"Tile solver finished without positions CSV: {position_csv}")
    subprocess.run(xml_command, check=True)
    if not all(os.path.isfile(path) for path in outputs):
        raise RuntimeError("Tile solver/XML generator finished without all expected outputs")
    part = manifest_path + '.part'
    with open(part, 'w', encoding='utf-8') as handle:
        json.dump({'signature': signature,
                   'outputs': {path: _stamp(path) for path in outputs}},
                  handle, indent=2)
    os.replace(part, manifest_path)
    return frame_xml, True
