"""Pipeline stages and image frames for tile-local 2D previews."""

import json
import os


SOURCES_2D = {
    "raw": ("1_2d_raw", "Raw detections", False),
    "filtered": ("2_2d_filtered", "Filtered detections (before alignment)", False),
    "aligned": ("3_2d_aligned", "Aligned detections", True),
    "aligned_filtered": ("4_2d_filtered", "Filtered detections (after alignment)", True),
}


def tile_2d_inputs(result_dir, source, tile_name, channel_ids):
    """Use zero image shifts for raw-frame CSVs, saved shifts for aligned CSVs."""
    if source not in SOURCES_2D:
        raise ValueError(f"Unknown 2d_source {source!r}; choose {', '.join(SOURCES_2D)}")
    directory, _, aligned = SOURCES_2D[source]
    csv_dir = os.path.join(result_dir, directory)
    if not os.path.isdir(csv_dir):
        raise FileNotFoundError(f"2D {source} result directory does not exist: '{csv_dir}'")
    if not aligned:
        return csv_dir, {cid: dict(dx=0, dy=0, dz=0) for cid in channel_ids}
    offset_path = os.path.join(result_dir, '3_2d_aligned', f'{tile_name}_offsets.json')
    if not os.path.isfile(offset_path):
        raise FileNotFoundError(f"Channel alignment offsets do not exist: '{offset_path}'")
    with open(offset_path, encoding='utf-8') as handle:
        offsets = json.load(handle)
    for cid in channel_ids:
        shift = offsets.get(cid)
        if not isinstance(shift, dict) or not all(k in shift for k in ('dx', 'dy', 'dz')):
            raise ValueError(f"Offsets '{offset_path}' lack dx/dy/dz for '{cid}'")
    return csv_dir, offsets
