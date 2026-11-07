"""Checkpointed, template-free tile geometry and merging XML generation."""
import hashlib
import json
import os
import subprocess
import sys

from src.utils.io import resolve_tile_dir
from src.core.provenance import (file_sha256, file_stamp, output_manifest_valid,
                                  write_output_manifest, content_signature,
                                  manifest_outputs_valid, PROVENANCE_SCHEMA_VERSION,
                                  code_sha256)


# The Stage 2.9 checkpoint hashes only this code of src/core/solve_tile_positions.py
# (short names) and its helpers ('package.module:name'), via provenance.code_sha256:
# comments, formatting, line endings and statements calling SOLVE_LOG_CALLS are
# ignored. Add any new helper that main() reaches. SOLVE_CODE_SKIP holds solver code
# the pipeline never runs (it does not pass --write-xml, --write-aligned or
# --xml-into-channel-dirs); move a name into SOLVE_CODE if that changes.
# Kept here, not in the solver, so adding it left solve_tile_positions.py byte-identical
# and older whole-file manifests still migrate. Guarded by tests/test_checkpoint_code_lists.py.
SOLVE_LOG_CALLS = ('say', 'print', 'say_scores')
SOLVE_CODE_SKIP = ('write_channel_xml', 'channel_xml_positions', 'source_xml_template',
                   'publish_channel_xml', 'write_aligned')
SOLVE_CODE = (
    'main', 'parse_args', 'AXES', 'CELL_COLS', '_irls', '_refine_job', '_soma_index',
    '_tf_arrays', 'add_measured', 'build_jobs', 'channel_image_dirs', 'const_from_offsets',
    'containment_count', 'count_slices', 'discover_tiles', 'estimate_constants_pooled',
    'json_scores', 'load_old_offsets', 'measure_seams', 'nominal_grid', 'parse_const',
    'pool_cells', 'refine_per_tile', 'score_placements', 'seam_support', 'seam_weights',
    'shift_arrays', 'solve_axis', 'stack_cloud', 'tile_cell_arrays',
    'src.config.loader:_strip_comments', 'src.config.loader:load_config',
    'src.core.point_cloud_aligner:_coarse_from_peaks',
    'src.core.point_cloud_aligner:_containment_score',
    'src.core.point_cloud_aligner:_displacement_peaks',
    'src.core.point_cloud_aligner:_fine_containment',
    'src.core.point_cloud_aligner:containment_shift_from_arrays',
    'src.core.point_cloud_aligner:resolve_align_settings',
    'src.core.z_linker:_IOU_BLOCK', 'src.core.z_linker:_candidate_pairs',
    'src.core.z_linker:_iou_matrix', 'src.core.z_linker:_match_dense',
    'src.core.z_linker:_match_sparse', 'src.core.z_linker:_median',
    'src.core.z_linker:_parse_cached', 'src.core.z_linker:parse_class_string',
    'src.core.z_linker:run_z_linker',
    'src.utils.compare_stitching:BOX_COLS', 'src.utils.compare_stitching:DIRECTIONS',
    'src.utils.compare_stitching:_hist_peak', 'src.utils.compare_stitching:_link_cells',
    'src.utils.compare_stitching:_mutual_nn', 'src.utils.compare_stitching:_read_dets',
    'src.utils.compare_stitching:_tile_cells_in_overlap',
    'src.utils.compare_stitching:estimate_seam_error',
    'src.utils.compare_stitching:resolve_results_dir',
    'src.utils.compare_stitching:run_seam_job',
    'src.utils.io:TILE_NAME_TOLERANCE', 'src.utils.io:_name_coords',
    'src.utils.io:resolve_tile_dir',
    'src.utils.markers:clean_markers', 'src.utils.markers:split_class',
)


def solver_code_sha256():
    import src.core.solve_tile_positions as solver
    return code_sha256(solver, SOLVE_CODE, SOLVE_LOG_CALLS)


def _file_hashes(paths):
    return {path: file_sha256(path) for path in paths if os.path.isfile(path)}


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


def _solver_signature(signature, xml_script):
    """Compare geometry inputs while allowing XML format/location changes."""
    value = {key: item for key, item in signature.items()
             if key not in ('version', 'xml_command')}
    value['inputs'] = {path: stamp for path, stamp in value['inputs'].items()
                       if path != xml_script}
    return value


def _same_file_bytes(path, digest):
    """Whole-file hash recorded by older manifests, with either line ending."""
    if not os.path.isfile(path):
        return False
    with open(path, 'rb') as handle:
        data = handle.read().replace(b'\r\n', b'\n')
    return digest in (hashlib.sha256(data).hexdigest(),
                      hashlib.sha256(data.replace(b'\n', b'\r\n')).hexdigest())


def _legacy_solver_manifest_valid(path, solver_signature, script):
    """Accept a manifest that hashed the whole solver file, if that file is unchanged.

    Earlier versions hashed solve_tile_positions.py byte for byte. When the file
    still has those bytes (up to line endings), the code is identical, so the
    existing positions are adopted instead of being treated as stale.
    """
    try:
        with open(path, encoding='utf-8') as handle:
            payload = json.load(handle)
        old = dict(payload['input_signature'])
        inputs = dict(old['inputs'])
        entry = inputs.pop(os.path.basename(script))
        old['inputs'] = inputs
        old['solver_code'] = solver_signature['solver_code']
        return (payload.get('schema_version') == PROVENANCE_SCHEMA_VERSION
                and _same_file_bytes(script, entry['sha256'])
                and content_signature(old) == content_signature(solver_signature)
                and manifest_outputs_valid(path, payload))
    except (OSError, ValueError, KeyError, TypeError):
        return False


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
                            report_dir, tile_names, routing, script, xml_script,
                            solver_code=None):
    """Solve positions, generate every channel XML, then reuse identical outputs.

    solver_code: hash of the code that determines geometry (see
    solve_tile_positions.SOLVE_CODE). When given, it replaces the whole-file hash
    of `script`, so comment, formatting or logging edits do not invalidate positions.
    """
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
                   '--positions', position_csv,
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
            tile_dir = resolve_tile_dir(channel_dir, os.path.join(tile.split('_')[0], tile))
            if not os.path.isdir(tile_dir):
                raise FileNotFoundError(f"Original image tile directory missing: {tile_dir}")
            files[tile_dir] = _stamp(tile_dir)
    if solver_code is None:
        files[script] = _stamp(script)
    files[xml_script] = _stamp(xml_script)
    outputs = [position_csv, os.path.join(report_dir, 'solution.json')]
    outputs.extend(os.path.join(config['paths'][ch['dir_key']], 'xml_merging.xml')
                   for ch in routing)
    signature = {'version': 4, 'reference_channel': ref, 'frame_channel': frame,
                 'solver_command': command, 'xml_command': xml_command, 'inputs': files,
                 'solver_code': solver_code,
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
    frame_dir = config['paths'][next(ch['dir_key'] for ch in routing if ch['id'] == frame)]
    frame_xml = os.path.join(frame_dir, 'xml_merging.xml')
    output_stamps = {path: _stamp(path) for path in outputs} if all(
        os.path.isfile(path) for path in outputs) else None
    input_hashes = _file_hashes(files)
    output_hashes = _file_hashes(outputs)
    hashes_match = (not previous or not previous.get('hashes') or
                    previous['hashes'].get('inputs') == input_hashes and
                    previous['hashes'].get('outputs') == output_hashes)
    solver_hashes_match = (not previous or not previous.get('hashes') or
        {path: digest for path, digest in previous['hashes'].get('inputs', {}).items()
         if path != xml_script} ==
        {path: digest for path, digest in input_hashes.items()
         if path != xml_script} and
        {path: previous['hashes'].get('outputs', {}).get(path)
         for path in outputs[:2]} ==
        {path: output_hashes.get(path) for path in outputs[:2]})
    if previous and output_stamps and not previous.get('hashes'):
        previous['hashes'] = {'inputs': input_hashes, 'outputs': output_hashes}
        part = manifest_path + '.part'
        with open(part, 'w', encoding='utf-8') as handle:
            json.dump(previous, handle, indent=2)
        os.replace(part, manifest_path)
    if previous and output_stamps:
        if (hashes_match and previous.get('signature') == signature
                and previous.get('outputs') == output_stamps):
            return frame_xml, False
        legacy = any(old in str(previous.get('signature')) for old, _ in _LAYOUT_NAMES)
        if (legacy and hashes_match and
                _canonical_manifest(previous.get('signature'), script, xml_script) ==
                _canonical_manifest(signature, script, xml_script) and
                _canonical_manifest(previous.get('outputs'), script, xml_script) ==
                _canonical_manifest(output_stamps, script, xml_script)):
            return frame_xml, False
    solver_manifest = os.path.join(report_dir, '_solver_manifest.json')
    solver_signature = {
        'reference_channel': ref, 'frame_channel': frame,
        'options': params.get('options') or {},
        'geometry_config': {key: value for key, value in signature['geometry_config'].items() if key != 'paths'},
        'inputs': {os.path.basename(path): file_stamp(path, with_hash=True)
                   for path in files if os.path.isfile(path) and path != xml_script},
        'image_tiles': {ch['id']: list(tile_names) for ch in routing},
    }
    if solver_code is not None:
        solver_signature['solver_code'] = solver_code
    positions_reusable = bool(previous) and solver_hashes_match and all(
        os.path.isfile(path) and previous.get('hashes', {}).get('outputs', {}).get(path) == file_sha256(path)
        for path in outputs[:2])
    solver_unchanged = output_manifest_valid(solver_manifest, solver_signature) or (
        solver_code is not None and
        _legacy_solver_manifest_valid(solver_manifest, solver_signature, script)) or (
        positions_reusable and isinstance(previous.get('signature'), dict) and
        _solver_signature(previous['signature'], xml_script) == _solver_signature(signature, xml_script))
    if solver_unchanged:
        # A changed XML destination or serialization does not change tile geometry.
        write_output_manifest(solver_manifest, outputs[:2], solver_signature)
        subprocess.run(xml_command, check=True)
    else:
        global_dirs = ('5_2d_global', '6_3d_global', '7_colocalization')
        if any(os.path.isdir(os.path.join(results_dir, d)) and
               any(name.endswith(('.csv', '.pkl')) for name in os.listdir(os.path.join(results_dir, d)))
               for d in global_dirs):
            raise RuntimeError("Tile positions or their inputs changed while global checkpoints exist. "
                               "Archive/regenerate global 2D, 3D, and colocalization outputs first.")
        subprocess.run(command, check=True)
        if not os.path.isfile(position_csv):
            raise RuntimeError(f"Tile solver finished without positions CSV: {position_csv}")
        write_output_manifest(solver_manifest, outputs[:2], solver_signature)
        subprocess.run(xml_command, check=True)
    if not all(os.path.isfile(path) for path in outputs):
        raise RuntimeError("Tile solver/XML generator finished without all expected outputs")
    part = manifest_path + '.part'
    with open(part, 'w', encoding='utf-8') as handle:
        json.dump({'signature': signature,
                   'outputs': {path: _stamp(path) for path in outputs},
                   'hashes': {'inputs': _file_hashes(files),
                              'outputs': _file_hashes(outputs)}},
                  handle, indent=2)
    os.replace(part, manifest_path)
    return frame_xml, True
