"""Names and migration for pipeline result directories."""

from __future__ import annotations

import hashlib
import os
import shutil

DIRECTORIES = {
    "pATH_DET_RES": "1_2d_raw",
    "pATH_DET_PREALIGN_FILTERED": "2_2d_filtered",
    "pATH_ALIGN_OFFSETS": "3_2d_aligned",
    "pATH_DET_FUSED": "3_2d_aligned_fusion",
    "pATH_DET_FILTERED": "4_2d_filtered",
    "pATH_GLOBAL_2D": "5_2d_global",
    "pATH_CHANNEL_3D": "6_3d_global",
    "pATH_COLOCALIZATION": "7_colocalization",
}


def result_paths(base: str) -> dict[str, str]:
    paths = {key: os.path.join(base, name) for key, name in DIRECTORIES.items()}
    paths["pATH_CENTROIDS"] = os.path.join(paths["pATH_COLOCALIZATION"], "cell_centroids")
    paths["pATH_HISTOGRAMS"] = os.path.join(paths["pATH_DET_RES"], "histograms")
    paths["pATH_TILE_POSITIONS"] = os.path.join(paths["pATH_GLOBAL_2D"], "tile_positions")
    return paths


def _same_file(first: str, second: str) -> bool:
    if os.path.getsize(first) != os.path.getsize(second):
        return False
    def digest(path: str) -> bytes:
        h = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        return h.digest()
    return digest(first) == digest(second)


def _merge_without_overwrite(source: str, destination: str) -> None:
    if not os.path.isdir(source):
        return
    if not os.path.exists(destination):
        shutil.move(source, destination)
        return
    for name in os.listdir(source):
        old = os.path.join(source, name)
        new = os.path.join(destination, name)
        if os.path.isdir(old):
            _merge_without_overwrite(old, new)
        elif not os.path.exists(new):
            shutil.move(old, new)
        elif os.path.isfile(old) and os.path.isfile(new):
            if _same_file(old, new):
                os.remove(old)
            else:
                raise FileExistsError(f"Conflicting legacy and new checkpoints: {old} and {new}")
    if not os.listdir(source):
        os.rmdir(source)


def migrate_result_layout(base: str) -> None:
    """Reuse the original result layout without replacing different checkpoints."""
    moves = (
        ("1_tile_2d_raw", "1_2d_raw"),
        ("1_tile_2d_prefiltered", "2_2d_filtered"),
        ("1_tile_2d_fused", "3_2d_aligned_fusion"),
        ("0_channel_alignment", "3_2d_aligned"),
        ("1_tile_2d_filtered", "4_2d_filtered"),
        ("2_global_2d_raw", "5_2d_global"),
        ("3_channel_3d", "6_3d_global"),
        ("4_colocalization", "7_colocalization"),
        ("1_tile_2d_histograms", os.path.join("1_2d_raw", "histograms")),
        (os.path.join("5_analysis_report", "tile_positions"),
         os.path.join("5_2d_global", "tile_positions")),
        (os.path.join("5_analysis_report", "cell_centroids"),
         os.path.join("7_colocalization", "cell_centroids")),
    )
    for old, new in moves:
        _merge_without_overwrite(os.path.join(base, old), os.path.join(base, new))
    old_report = os.path.join(base, "5_analysis_report", "global_summary_statistics.csv")
    new_report = os.path.join(base, "7_colocalization", "global_summary_statistics.csv")
    if os.path.isfile(old_report) and not os.path.exists(new_report):
        shutil.move(old_report, new_report)
