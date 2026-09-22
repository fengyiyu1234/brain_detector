"""Read-only Napari review for raw TIFs with saved filtered detection boxes.

Tile selection is printed as a spatial grid in the terminal (the same interface as
src.utils.visualize); comma-separated tile indices open multiple review windows.
No CSV, image, or pipeline result is modified.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import napari
from qtpy.QtWidgets import (
    QApplication, QCheckBox, QDialog, QDialogButtonBox, QFormLayout, QLabel,
    QSpinBox, QVBoxLayout,
)

from src.config.loader import load_config
from src.utils.io import listTile
from src.utils.visualize import (
    _add_labels_layer, _ch_vis, _load_tile_csv_shapes, _select_tiles,
    load_frame_volume, load_offsets,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only QC: raw images plus saved filtered CSV boxes."
    )
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "config" / "config_EGFR_t4_local_gpu.json"),
        help="Pipeline JSON/JSONC config.",
    )
    parser.add_argument(
        "--filtered-dir",
        help="Directory containing <tile>_<channel>_result.csv; defaults to 1_tile_2d_filtered.",
    )
    return parser.parse_args()


def _tiff_count(tile_dir: str) -> int:
    return len([
        name for name in os.listdir(tile_dir)
        if name.lower().endswith((".tif", ".tiff"))
    ]) if os.path.isdir(tile_dir) else 0


class ReviewSettingsDialog(QDialog):
    """Choose the common image range and channel set after terminal tile selection."""

    def __init__(self, selected_tiles: list[tuple[str, str]], channels: list[dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Filtered-result QC settings")
        min_z = min(max(_tiff_count(path), 1) for path, _ in selected_tiles)
        default_count = min(50, min_z)

        layout = QVBoxLayout(self)
        names = ", ".join(name for _, name in selected_tiles)
        layout.addWidget(QLabel(f"Selected {len(selected_tiles)} tile(s): {names}"))
        layout.addWidget(QLabel("The same Z range is loaded for each selected tile."))
        form = QFormLayout()
        self.z_start = QSpinBox()
        self.z_start.setRange(0, min_z - 1)
        self.z_start.setValue(max(0, (min_z - default_count) // 2))
        self.z_count = QSpinBox()
        self.z_count.setRange(1, min_z)
        self.z_count.setValue(default_count)
        form.addRow("Start Z (0-based)", self.z_start)
        form.addRow("Number of images", self.z_count)
        layout.addLayout(form)

        layout.addWidget(QLabel("Channels"))
        self.channel_checks: dict[str, QCheckBox] = {}
        for channel in channels:
            check = QCheckBox(channel["id"])
            check.setChecked(channel["id"] == "Olig2")
            self.channel_checks[channel["id"]] = check
            layout.addWidget(check)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selection(self) -> tuple[int, int, list[str]]:
        return (
            self.z_start.value(),
            self.z_count.value(),
            [channel_id for channel_id, check in self.channel_checks.items() if check.isChecked()],
        )


def _open_tile_viewer(
    tile_path: str, z_start: int, z_count: int, channel_ids: list[str],
    channels: list[dict], paths: dict, anchor_dir: str, align_dir: str,
    pipeline_mode: str, filtered_dir: str,
) -> None:
    tile_name = os.path.basename(tile_path)
    total_z = _tiff_count(tile_path)
    z_range = (z_start, min(total_z, z_start + z_count))
    if z_range[0] >= z_range[1]:
        print(f"[skip] {tile_name}: requested Z range is outside its image stack.")
        return
    offsets = load_offsets(align_dir, tile_name) if pipeline_mode == "pre_align" else {}
    viewer = napari.Viewer(title=f"Filtered-result QC — {tile_name} — z {z_range[0]}:{z_range[1]}")
    canvas_shape = None

    for channel in channels:
        channel_id = channel["id"]
        if channel_id not in channel_ids:
            continue
        channel_root = os.path.abspath(paths[channel["dir_key"]])
        image_dir = os.path.join(channel_root, os.path.relpath(tile_path, anchor_dir))
        offset = offsets.get(channel_id, {}) if pipeline_mode == "pre_align" else {}
        shift = (offset.get("dx", 0), offset.get("dy", 0), offset.get("dz", 0))
        volume, contrast = load_frame_volume(image_dir, z_range, shift=shift)
        if volume is None:
            print(f"[skip] image stack unavailable: {channel_id} ({image_dir})")
            continue
        canvas_shape = volume.shape
        image_layer = viewer.add_image(
            volume, name=f"[raw image] {channel_id}", contrast_limits=contrast, **_ch_vis(channel_id)
        )
        image_layer.contrast_limits_range = (0, 65535)

        csv_path = os.path.join(filtered_dir, f"{tile_name}_{channel_id}_result.csv")
        (shapes, colors, _), _ = _load_tile_csv_shapes(csv_path, z_range, filt=None)
        if shapes:
            _add_labels_layer(
                viewer, shapes, colors, canvas_shape,
                name=f"[saved filtered] {channel_id}  n={len(shapes)}",
                visible=True, opacity=0.9, outline_width=3,
            )
        else:
            print(f"[info] no filtered boxes loaded: {csv_path}")
        if pipeline_mode == "pre_align" and any(shift):
            print(f"[{tile_name}][{channel_id}] image shifted dx={shift[0]}, dy={shift[1]}, dz={shift[2]}.")

    if canvas_shape is None:
        viewer.close()
        print(f"[skip] no selected image stacks could be loaded for {tile_name}.")


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)
    paths = config.get("paths")
    channels = [channel for channel in config.get("channels_routing", []) if channel.get("active", True)]
    if not paths or not channels:
        raise ValueError("Config requires paths and at least one active channels_routing entry.")

    base_result = paths["pATHRESULT"]
    filtered_dir = os.path.abspath(args.filtered_dir or os.path.join(base_result, "1_tile_2d_filtered"))
    align_dir = os.path.join(base_result, "0_channel_alignment")
    pipeline_mode = config.get("pipeline_mode", "post_align")
    anchor_dir = os.path.abspath(paths[channels[0]["dir_key"]])
    _, tile_paths = listTile(anchor_dir)
    if not tile_paths:
        raise FileNotFoundError(f"No tiles found under anchor channel: {anchor_dir}")

    # This prints the grid + indexed tile list and accepts e.g. "1,3,10".
    selected_tiles = _select_tiles({}, tile_paths)
    app = QApplication.instance() or QApplication([])
    dialog = ReviewSettingsDialog(selected_tiles, channels)
    if dialog.exec_() != QDialog.Accepted:
        return 0
    z_start, z_count, channel_ids = dialog.selection()
    if not channel_ids:
        raise ValueError("Select at least one channel.")

    for tile_path, _ in selected_tiles:
        _open_tile_viewer(
            tile_path, z_start, z_count, channel_ids, channels, paths, anchor_dir,
            align_dir, pipeline_mode, filtered_dir,
        )
    print(f"Read-only review loaded from filtered directory: {filtered_dir}")
    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

