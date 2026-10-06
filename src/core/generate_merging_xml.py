#!/usr/bin/env python3
r"""Generate TeraStitcher merging XMLs without reading an XML template.

Input positions come from solve_tile_positions.py's tile_positions.csv.
The generator writes XMLs into the chosen output directory and never modifies source images.

Example:
  python src/core/generate_merging_xml.py \
    --config config/EGFR/2/config_EGFR_t70.json \
    --positions /path/to/tile_positions.csv

"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.config.loader import load_config

TILE_RE = re.compile(r"^(\d+)_(\d+)$")
AXES = ("x", "y", "z")
DIRECTIONS = (("NORTH", -1, 0), ("EAST", 0, 1),
              ("SOUTH", 1, 0), ("WEST", 0, -1))


def tile_coordinates(name):
    match = TILE_RE.fullmatch(name)
    if not match:
        raise ValueError(f"Invalid tile name {name!r}; expected <V>_<H>")
    return int(match.group(1)), int(match.group(2))


def read_positions(path, channels, frame):
    required = {"tile"} | {f"P_{ch}_{axis}" for ch in channels for axis in AXES}
    required |= {f"s_{ch}_{axis}" for ch in channels if ch != frame for axis in AXES}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no tile positions")
    by_tile = {}
    for row in rows:
        tile = row["tile"]
        tile_coordinates(tile)
        if tile in by_tile:
            raise ValueError(f"Duplicate tile: {tile}")
        for key in required - {"tile"}:
            value = float(row[key])
            if not math.isfinite(value):
                raise ValueError(f"Non-finite {key} for {tile}")
            row[key] = value
        by_tile[tile] = row
    return by_tile


def grid_for_tiles(tiles):
    coords = {tile: tile_coordinates(tile) for tile in tiles}
    vs = sorted({v for v, _ in coords.values()})
    hs = sorted({h for _, h in coords.values()})
    coordinate_set = set(coords.values())
    missing = [(v, h) for v in vs for h in hs if (v, h) not in coordinate_set]
    if missing:
        raise ValueError(f"Incomplete tile grid; missing {missing[:8]}")
    v_index = {v: i for i, v in enumerate(vs)}
    h_index = {h: i for i, h in enumerate(hs)}
    grid = {(v_index[v], h_index[h]): tile for tile, (v, h) in coords.items()}
    return grid, vs, hs


def slice_coordinates(tile_dir, tile):
    if not tile_dir.is_dir():
        raise FileNotFoundError(f"Missing image tile directory: {tile_dir}")
    prefix = tile + "_"
    z_values = []
    with os.scandir(tile_dir) as entries:
        for entry in entries:
            if not entry.is_file() or not entry.name.lower().endswith((".tif", ".tiff")):
                continue
            stem = Path(entry.name).stem
            if not stem.startswith(prefix) or not stem[len(prefix):].isdigit():
                raise ValueError(f"Unexpected TIFF name: {entry.path}")
            z_values.append(int(stem[len(prefix):]))
    if not z_values or len(z_values) != len(set(z_values)):
        raise ValueError(f"Missing or duplicate TIFF slice coordinates: {tile_dir}")
    return tuple(sorted(z_values))


def validate_images(channel_dir, tiles):
    expected = None
    for tile in tiles:
        v, _ = tile_coordinates(tile)
        z_values = slice_coordinates(channel_dir / str(v) / tile, tile)
        if expected is None:
            expected = z_values
        elif z_values != expected:
            raise ValueError(f"TIFF slice coordinates differ: {channel_dir / str(v) / tile}")
    return expected


def image_positions(rows, channels, frame):
    """Use solved seam geometry plus global channel translation.

    Per-tile s_* refinements align detected cells; they are intentionally
    excluded from tile image geometry, as in solve_tile_positions.py.
    """
    tiles = list(rows)
    origin = {axis: min(rows[t][f"P_{frame}_{axis}"] for t in tiles) for axis in AXES}
    result = {}
    for ch in channels:
        translation = {}
        for axis in AXES:
            translation[axis] = (0.0 if ch == frame else statistics.median(
                rows[t][f"s_{ch}_{axis}"]
                - (rows[t][f"P_{ch}_{axis}"] - rows[t][f"P_{frame}_{axis}"])
                for t in tiles))
        result[ch] = {
            t: tuple(round(rows[t][f"P_{ch}_{axis}"] + translation[axis]
                           - origin[axis]) for axis in AXES)
            for t in tiles
        }
    return result


def displacement(parent, direction, source, target, nominal):
    node = ET.SubElement(parent, f"{direction}_displacements")
    if target is None:
        return
    element = ET.SubElement(node, "Displacement", TYPE="MIP_NCC")
    for tag, index in (("V", 1), ("H", 0), ("D", 2)):
        ET.SubElement(element, tag, {
            "displ": str(target[index] - source[index]),
            "default_displ": str(nominal[index]),
            "reliability": "1",
            "nccPeak": "0", "nccWidth": "26", "nccWRangeThr": "25",
            "nccInvWidth": "26", "delay": "25",
        })


def formatted(value):
    return f"{value:g}"


def mechanical_step(coords):
    return statistics.median(b - a for a, b in zip(coords, coords[1:])) / 10 if len(coords) > 1 else 0


def build_xml(channel_dir, grid, vs, hs, z_values, positions, xy_um, z_um, bytes_per_channel):
    root = ET.Element("TeraStitcher", volume_format="TiledXY|2Dseries", input_plugin="tiff2D")
    directory = str(channel_dir).rstrip("/\\")
    ET.SubElement(root, "stacks_dir", value=directory)
    ET.SubElement(root, "mdata_bin", value=directory + "/mdata.bin")
    ET.SubElement(root, "ref_sys", ref1="1", ref2="2", ref3="3")
    ET.SubElement(root, "voxel_dims", V=formatted(xy_um), H=formatted(xy_um), D=formatted(z_um))
    ET.SubElement(root, "origin", V=formatted(vs[0] / 10000),
                  H=formatted(hs[0] / 10000), D=formatted(z_values[0] / 10000))
    ET.SubElement(root, "mechanical_displacements",
                  V=formatted(mechanical_step(vs)), H=formatted(mechanical_step(hs)))
    ET.SubElement(root, "dimensions", stack_rows=str(len(vs)),
                  stack_columns=str(len(hs)), stack_slices=str(len(z_values)))
    stacks = ET.SubElement(root, "STACKS")
    for (row, col), tile in sorted(grid.items()):
        x, y, z = positions[tile]
        stack = ET.SubElement(stacks, "Stack", {
            "N_CHANS": "1", "N_BYTESxCHAN": str(bytes_per_channel),
            "ROW": str(row), "COL": str(col),
            "ABS_V": str(y), "ABS_H": str(x), "ABS_D": str(z),
            "STITCHABLE": "yes", "DIR_NAME": f"{vs[row]}/{tile}",
            "Z_RANGES": f"[0,{len(z_values)})", "IMG_REGEX": "",
        })
        for direction, dr, dc in DIRECTIONS:
            neighbor = grid.get((row + dr, col + dc))
            if neighbor is None:
                target, nominal = None, (0, 0, 0)
            else:
                target = positions[neighbor]
                nv, nh = tile_coordinates(neighbor)
                nominal = (round((nh - hs[col]) / 10 / xy_um),
                           round((nv - vs[row]) / 10 / xy_um), 0)
            displacement(stack, direction, (x, y, z), target, nominal)
    ET.indent(root, space="    ")
    return root


def write_xml(root, path):
    body = ET.tostring(root, encoding="utf-8")
    data = (b'<?xml version="1.0" encoding="UTF-8" ?>\n'
            b'<!DOCTYPE TeraStitcher SYSTEM "TeraStitcher.DTD">\n' + body + b"\n")
    temporary = path.with_name(path.name + ".part")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def generate(config_path, positions_path, out_dir, bytes_per_channel=2, force=False):
    config = load_config(config_path)
    routing = [ch for ch in config["channels_routing"] if ch.get("active", True)]
    channels = [ch["id"] for ch in routing]
    frame = config["stitching_reference_channel"]
    if frame not in channels:
        raise ValueError(f"Frame channel {frame!r} is not active")
    rows = read_positions(positions_path, channels, frame)
    grid, vs, hs = grid_for_tiles(rows)
    solution_path = positions_path.with_name("solution.json")
    if solution_path.is_file():
        solution = json.loads(solution_path.read_text(encoding="utf-8"))
        if solution.get("frame_channel") != frame or set(solution.get("channels", ())) != set(channels):
            raise ValueError(f"{solution_path}: frame or channels differ from config")
    dp = config["detection_params"]
    xy_um, z_um = float(dp["xy_resolution_um"]), float(dp["z_resolution_um"])
    if xy_um <= 0 or z_um <= 0 or bytes_per_channel <= 0:
        raise ValueError("Voxel sizes and bytes per channel must be positive")
    images, outputs = {}, {}
    for ch in routing:
        channel = ch["id"]
        channel_dir = Path(config["paths"][ch["dir_key"]])
        images[channel] = (channel_dir, validate_images(channel_dir, rows))
        outputs[channel] = out_dir / f"xml_merging_{channel}.xml"
        if outputs[channel].exists() and not force:
            raise FileExistsError(f"Output exists; pass --force to replace: {outputs[channel]}")
    if len({z for _, z in images.values()}) != 1:
        raise ValueError("Channels have different TIFF slice coordinates")
    positions = image_positions(rows, channels, frame)
    trees = {ch: build_xml(directory, grid, vs, hs, z_values, positions[ch],
                           xy_um, z_um, bytes_per_channel)
             for ch, (directory, z_values) in images.items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    for ch, tree in trees.items():
        write_xml(tree, outputs[ch])
        print(f"{ch}: {len(rows)} tiles -> {outputs[ch]}")

    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--positions", type=Path,
                        help="tile_positions.csv; default: config result directory")
    parser.add_argument("--out-dir", type=Path, help="Default: directory containing positions CSV")
    parser.add_argument("--bytes-per-channel", type=int, default=2,
                        help="TIFF bytes per channel, default 2 for 16-bit images")
    parser.add_argument("--force", action="store_true", help="Replace existing generated XMLs")
    args = parser.parse_args()
    config = load_config(args.config)
    positions = args.positions or (Path(config["paths"]["pATHRESULT"]) /
                                   "5_2d_global/tile_positions/tile_positions.csv")
    generate(args.config, positions, args.out_dir or positions.parent,
             args.bytes_per_channel, args.force)


if __name__ == "__main__":
    main()




