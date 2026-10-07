#!/usr/bin/env python3
"""Stitch raw TIFF tiles directly from the per-channel merging XML geometry.

Config: config_example/direct_stitching.example.json, or a detection config
with a direct_stitching block. Set direct_stitching.enabled to false to skip;
set it to true to stitch, including when run_inference.py reaches Stage 3.
The existing XML generation step is unchanged.
Each channel's XML supplies
ABS_H/ABS_V/ABS_D; this script reads raw tile TIFFs, blends overlaps, averages
blocks in XYZ, and writes aligned 2D TIFF sequences without TeraStitcher.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts.align_stitched_channels import MARKER_TO_WAVELENGTH
from src.config.loader import load_config


@dataclass(frozen=True)
class Tile:
    name: str
    x: int
    y: int
    z: int
    files: tuple[Path, ...]


@dataclass(frozen=True)
class Channel:
    marker: str
    xml: Path
    tiles: tuple[Tile, ...]
    slice_keys: tuple[int, ...]
    tile_size: tuple[int, int]  # width, height
    dtype: np.dtype
    voxel_size: tuple[float, float, float]  # H, V, D


@dataclass(frozen=True)
class Patch:
    tile: Tile
    x0: int
    y0: int
    width: int
    height: int
    phase_x: int
    phase_y: int


CLUSTER_ROOT = "/rsstu/users/a/agrinba/DeepDesign"


def platform_path(value: str, platform: str | None = None) -> str:
    """Translate the shared Y: mount and the HPC /rsstu mount both ways."""
    platform = platform or os.name
    normalized = value.replace("\\", "/")
    if platform == "nt":
        if normalized == CLUSTER_ROOT or normalized.startswith(CLUSTER_ROOT + "/"):
            return "Y:" + normalized[len(CLUSTER_ROOT):]
        if normalized.startswith("/") and not normalized.startswith("//"):
            raise ValueError(f"Cannot map Linux path to Windows: {value}")
    else:
        if normalized[:3].lower() == "y:/":
            return CLUSTER_ROOT + normalized[2:]
        if len(normalized) >= 2 and normalized[1] == ":":
            raise ValueError(
                f"Windows drive path cannot be used on HPC: {value}; "
                "set direct_stitching.output_dir_hpc"
            )
    return value


def resolve_path(value: str, base: Path) -> Path:
    path = Path(platform_path(value)).expanduser()
    return path.absolute() if path.is_absolute() else (base / path).resolve()


def output_path_value(params: dict, platform: str | None = None) -> str:
    platform = platform or os.name
    if platform != "nt" and params.get("output_dir_hpc"):
        return params["output_dir_hpc"]
    return params["output_dir"]


def list_raw_slices(tile_dir: Path, tile_name: str):
    if not tile_dir.is_dir():
        raise FileNotFoundError(f"Missing raw tile directory: {tile_dir}")
    prefix = tile_name + "_"
    found = {}
    with os.scandir(tile_dir) as entries:
        for entry in entries:
            if not entry.is_file() or not entry.name.lower().endswith((".tif", ".tiff")):
                continue
            stem = Path(entry.name).stem
            if not stem.startswith(prefix) or not stem[len(prefix):].isdigit():
                raise ValueError(f"Unexpected raw TIFF name: {entry.path}")
            key = int(stem[len(prefix):])
            if key in found:
                raise ValueError(f"Duplicate raw Z coordinate {key}: {tile_dir}")
            found[key] = Path(entry.path)
    if not found:
        raise ValueError(f"No raw TIFF slices: {tile_dir}")
    keys = tuple(sorted(found))
    return keys, tuple(found[key] for key in keys)


def read_channel(raw_dir: Path, marker: str) -> Channel:
    named = raw_dir / f"xml_merging_{marker}.xml"
    generic = raw_dir / "xml_merging.xml"
    xml = named if named.is_file() else generic
    if not xml.is_file():
        raise FileNotFoundError(f"Missing merging XML for {marker}: {named} or {generic}")
    root = ET.parse(xml).getroot()
    stacks = root.find("STACKS")
    voxel = root.find("voxel_dims")
    if stacks is None or not len(stacks) or voxel is None:
        raise ValueError(f"Missing STACKS or voxel_dims in {xml}")
    voxel_size = tuple(float(voxel.attrib[axis]) for axis in ("H", "V", "D"))
    if min(voxel_size) <= 0:
        raise ValueError(f"Invalid voxel dimensions in {xml}")

    tiles = []
    names = set()
    reference_keys = None
    reference_size = None
    reference_dtype = None
    print(f"Scanning {marker}: {len(stacks)} raw tile directories", flush=True)
    for tile_number, stack in enumerate(stacks, 1):
        dir_name = stack.attrib["DIR_NAME"].replace("\\", "/")
        if dir_name.startswith("/") or ":" in dir_name:
            raise ValueError(f"DIR_NAME must be relative to {raw_dir}: {dir_name}")
        tile_name = Path(dir_name).name
        if tile_name in names:
            raise ValueError(f"Duplicate tile {tile_name} in {xml}")
        names.add(tile_name)
        keys, files = list_raw_slices(raw_dir / Path(dir_name), tile_name)
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            raise ValueError(f"Raw Z coordinates differ for {tile_name} in {raw_dir}")
        with Image.open(files[0]) as image:
            size = image.size
            mode = image.mode
        if mode == "L":
            dtype = np.dtype("uint8")
        elif mode in ("I;16", "I;16L", "I;16B"):
            dtype = np.dtype("uint16")
        else:
            raise ValueError(f"Expected grayscale 8/16-bit TIFF: {files[0]} (mode={mode})")
        if reference_size is None:
            reference_size, reference_dtype = size, dtype
        elif size != reference_size or dtype != reference_dtype:
            raise ValueError(f"Tile size or dtype differs for {tile_name} in {raw_dir}")
        tiles.append(Tile(tile_name, int(stack.attrib["ABS_H"]),
                          int(stack.attrib["ABS_V"]), int(stack.attrib["ABS_D"]), files))
        if tile_number % 10 == 0 or tile_number == len(stacks):
            print(f"  {marker}: {tile_number}/{len(stacks)} tile directories", flush=True)
    dimensions = root.find("dimensions")
    if dimensions is not None and "stack_slices" in dimensions.attrib:
        if int(dimensions.attrib["stack_slices"]) != len(reference_keys):
            raise ValueError(f"XML stack_slices differs from raw TIFF count: {xml}")
    return Channel(marker, xml, tuple(tiles), reference_keys,
                   reference_size, reference_dtype, voxel_size)


def geometry(channels: dict[str, Channel], factor: int):
    """Compute one XY canvas and the complete common Z interval."""
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
        raise ValueError("downsample_factor must be a positive integer")
    first = next(iter(channels.values()))
    first_names = {tile.name for tile in first.tiles}
    for channel in channels.values():
        if {tile.name for tile in channel.tiles} != first_names:
            raise ValueError(f"Tile set differs for {channel.marker}")
        if (channel.tile_size != first.tile_size or
                channel.slice_keys != first.slice_keys or
                channel.dtype != first.dtype or
                channel.voxel_size != first.voxel_size):
            raise ValueError(f"Tile size, Z coordinates, dtype, or voxel size differs for {channel.marker}")
    all_tiles = [tile for channel in channels.values() for tile in channel.tiles]
    tile_width, tile_height = first.tile_size
    min_x = min(tile.x for tile in all_tiles)
    min_y = min(tile.y for tile in all_tiles)
    max_x = max(tile.x + tile_width for tile in all_tiles)
    max_y = max(tile.y + tile_height for tile in all_tiles)
    min_z = min(tile.z for tile in all_tiles)
    max_z = max(tile.z for tile in all_tiles)
    usable_z = len(first.slice_keys) - (max_z - min_z)
    if usable_z < factor:
        raise ValueError("No complete downsampled Z slice is shared by all tiles")
    canvas = ((max_x - min_x) // factor, (max_y - min_y) // factor)
    depth = usable_z // factor
    if min(canvas) < 1:
        raise ValueError("Downsample factor exceeds mosaic size")
    return (min_x, min_y, min_z, max_z), canvas, depth, usable_z % factor


def feather_weights(tile_size: tuple[int, int], blend_width: int):
    width, height = tile_size
    if blend_width < 1:
        raise ValueError("blend_width_px must be a positive integer")
    x = np.arange(width)
    y = np.arange(height)
    wx = np.minimum(1.0, np.minimum(x + 1, width - x) / blend_width).astype(np.float32)
    wy = np.minimum(1.0, np.minimum(y + 1, height - y) / blend_width).astype(np.float32)
    return wy[:, None] * wx[None, :]


def block_sum(array: np.ndarray, phase_x: int, phase_y: int, factor: int):
    """Sum full-resolution pixels into the shared, globally anchored XY grid."""
    height, width = array.shape
    right = (-phase_x - width) % factor
    bottom = (-phase_y - height) % factor
    padded = np.pad(array, ((phase_y, bottom), (phase_x, right)),
                    mode="constant")
    return padded.reshape(padded.shape[0] // factor, factor,
                          padded.shape[1] // factor, factor).sum(axis=(1, 3),
                                                                 dtype=np.float64)


def make_patches(channel: Channel, origin, canvas, factor: int):
    min_x, min_y, _, _ = origin
    patches = []
    for tile in channel.tiles:
        dx, dy = tile.x - min_x, tile.y - min_y
        x0, y0 = dx // factor, dy // factor
        px, py = dx % factor, dy % factor
        width = min(canvas[0] - x0, math.ceil((px + channel.tile_size[0]) / factor))
        height = min(canvas[1] - y0, math.ceil((py + channel.tile_size[1]) / factor))
        if width > 0 and height > 0:
            patches.append(Patch(tile, x0, y0, width, height, px, py))
    return patches


def read_image(path: Path, expected_size, expected_dtype):
    with Image.open(path) as image:
        array = np.asarray(image)
    if (array.ndim != 2 or array.shape != (expected_size[1], expected_size[0])
            or array.dtype.kind != "u" or
            array.dtype.itemsize != expected_dtype.itemsize):
        raise ValueError(f"Raw TIFF shape or dtype changed: {path}")
    return array.astype(expected_dtype, copy=False)


def stitch_channel(channel: Channel, output: Path, origin, canvas, depth,
                   factor: int, blend_width: int):
    output.mkdir()
    patches = make_patches(channel, origin, canvas, factor)
    weight = feather_weights(channel.tile_size, blend_width)
    weight_maps = {}
    denominator = np.zeros((canvas[1], canvas[0]), dtype=np.float64)
    for patch in patches:
        phase = (patch.phase_x, patch.phase_y)
        if phase not in weight_maps:
            weight_maps[phase] = block_sum(weight, *phase, factor)
        small = weight_maps[phase][:patch.height, :patch.width]
        denominator[patch.y0:patch.y0 + patch.height,
                    patch.x0:patch.x0 + patch.width] += small
    if not np.any(denominator):
        raise ValueError(f"No source pixels fall inside output canvas for {channel.marker}")
    z_start = origin[3]  # max ABS_D across all active channels
    upper = np.iinfo(channel.dtype).max
    for output_z in range(depth):
        numerator = np.zeros_like(denominator)
        for patch in patches:
            summed = np.zeros((patch.height, patch.width), dtype=np.float64)
            first_raw = z_start - patch.tile.z + output_z * factor
            for dz in range(factor):
                source = read_image(patch.tile.files[first_raw + dz],
                                    channel.tile_size, channel.dtype)
                weighted = source.astype(np.float32) * weight
                small = block_sum(weighted, patch.phase_x, patch.phase_y, factor)
                summed += small[:patch.height, :patch.width]
            numerator[patch.y0:patch.y0 + patch.height,
                      patch.x0:patch.x0 + patch.width] += summed
        np.divide(numerator, denominator * factor, out=numerator,
                  where=denominator > 0)
        image = np.clip(np.rint(numerator), 0, upper).astype(channel.dtype)
        destination = output / f"z{output_z:05d}.tif"
        temporary = output / (destination.name + ".part")
        Image.fromarray(image).save(temporary, format="TIFF", compression="tiff_lzw")
        os.replace(temporary, destination)
        if (output_z + 1) % 10 == 0 or output_z + 1 == depth:
            print(f"  {channel.marker}: {output_z + 1}/{depth} slices", flush=True)


def completed_output_matches(output: Path, config_path: Path, channel_dirs: dict,
                             markers: list[str], factor: int, blend_width: int) -> bool:
    """A final manifest marks a finished image stitch for pipeline resumes."""
    manifest_path = output / "stitch_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        depth = manifest["output_slices"]
        if (manifest["source_config"] != str(config_path)
                or manifest["downsample_factor_xyz"] != factor
                or manifest["blend_width_px"] != blend_width
                or set(manifest["channels"]) != set(markers)
                or not isinstance(depth, int) or depth < 1):
            return False
        for marker in markers:
            raw_dir = channel_dirs[marker]
            named = raw_dir / f"xml_merging_{marker}.xml"
            xml = named if named.is_file() else raw_dir / "xml_merging.xml"
            details = manifest["channels"][marker]
            if (details["xml"] != str(xml)
                    or details["xml_mtime_ns"] != xml.stat().st_mtime_ns
                    or len(list((output / marker).glob("z*.tif"))) != depth):
                return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return True


def run(config_path: Path, dry_run: bool = False, skip_completed: bool = False):
    config_path = config_path.resolve()
    config = load_config(config_path)
    params = config["direct_stitching"]
    enabled = params.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("direct_stitching.enabled must be true or false")
    if not enabled:
        print(f"Direct stitching disabled in {config_path}; skipping.")
        return
    # Pipeline config paths follow the project working directory convention;
    # standalone stitching configs keep paths relative to their own file.
    path_base = PROJECT_ROOT if "paths" in config or "samples" in config else config_path.parent
    output = resolve_path(output_path_value(params), path_base)
    factor = params.get("downsample_factor", 4)
    blend_width = params.get("blend_width_px", 256)
    if isinstance(blend_width, bool) or not isinstance(blend_width, int) or blend_width < 1:
        raise ValueError("blend_width_px must be a positive integer")
    source = config
    if "samples" in config and "active_sample" in config:
        source = config["samples"][config["active_sample"]]
    routes = {route["id"]: route for route in source.get("channels_routing", [])
              if route.get("active", True)}
    route_names = {marker.lower(): marker for marker in routes}
    markers = params.get("markers") or list(routes)
    if not isinstance(markers, list) or not markers:
        raise ValueError("direct_stitching.markers must list the markers to stitch")
    canonical = []
    for value in markers:
        marker = None
        if isinstance(value, str):
            if routes:
                marker = route_names.get(value.lower())
            else:
                marker = next((m for m in MARKER_TO_WAVELENGTH
                               if m.lower() == value.lower()), None)
        if marker is None or marker in canonical:
            raise ValueError(f"Unknown, inactive, or repeated marker: {value}")
        canonical.append(marker)
    if "raw_sample_dir" in config:
        raw_root = resolve_path(config["raw_sample_dir"], config_path.parent)
        channel_dirs = {m: raw_root / MARKER_TO_WAVELENGTH[m] for m in canonical}
    else:
        paths = source.get("paths", {})
        channel_dirs = {
            m: resolve_path(paths[routes[m]["dir_key"]], path_base)
            for m in canonical
        }
        raw_root = None
    if output.exists():
        if skip_completed and completed_output_matches(
                output, config_path, channel_dirs, canonical, factor, blend_width):
            print(f"Direct stitching already complete at {output}; skipping.")
            return
        raise FileExistsError(f"Output already exists or is incomplete; "
                              f"choose a new output_dir: {output}")
    channels = {m: read_channel(channel_dirs[m], m) for m in canonical}
    origin, canvas, depth, discarded_z = geometry(channels, factor)
    print(f"Canvas={canvas[0]}x{canvas[1]}, Z={depth}, downsample={factor}x XYZ, "
          f"global ABS_D range={origin[2]}..{origin[3]}, "
          f"discarded tail slices={discarded_z}")
    for marker, channel in channels.items():
        print(f"  {marker}: {len(channel.tiles)} tiles, "
              f"{len(channel.slice_keys)} raw Z slices, {channel.xml}")
    if dry_run:
        return

    output.mkdir(parents=True)
    for marker, channel in channels.items():
        stitch_channel(channel, output / marker, origin, canvas, depth, factor, blend_width)
    voxel = next(iter(channels.values())).voxel_size
    manifest = {
        "source_config": str(config_path),
        "raw_sample_dir": str(raw_root) if raw_root else None,
        "raw_channel_dirs": {m: str(channel_dirs[m]) for m in canonical},
        "output_width": canvas[0], "output_height": canvas[1],
        "output_slices": depth, "downsample_factor_xyz": factor,
        "output_voxel_size_um": [value * factor for value in voxel],
        "global_min_abs_h": origin[0], "global_min_abs_v": origin[1],
        "global_min_abs_d": origin[2], "global_max_abs_d": origin[3],
        "raw_z_start_for_tile": "global_max_abs_d - tile_abs_d",
        "blend_width_px": blend_width,
        "channels": {m: {"xml": str(ch.xml),
                         "xml_mtime_ns": ch.xml.stat().st_mtime_ns,
                         "channel_max_abs_d": max(t.z for t in ch.tiles),
                         "z_offset_from_channel_frame_raw_slices":
                             origin[3] - max(t.z for t in ch.tiles)}
                     for m, ch in channels.items()},
    }
    frame_marker = config.get("stitching_reference_channel")
    if frame_marker in channels:
        frame_tiles = channels[frame_marker].tiles
        manifest["reference_frame"] = {
            "marker": frame_marker,
            "output_origin_shift_from_frame_raw_xy": [
                min(t.x for t in frame_tiles) - origin[0],
                min(t.y for t in frame_tiles) - origin[1],
            ],
            "leading_frame_z_slices_removed": (
                origin[3] - max(t.z for t in frame_tiles)
            ),
        }
    (output / "stitch_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Directly stitched TIFF sequences written to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate raw tiles and print output geometry")
    args = parser.parse_args()
    try:
        run(args.config, args.dry_run)
    except (OSError, ValueError, KeyError, ET.ParseError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
