#!/usr/bin/env python3
"""Align independently stitched TIFF channels using their merging XML coordinates.

Set raw_sample_dir to a sample containing wavelength folders (640nm, 561nm,
730nm, 488nm). In stitched_images, list only the markers present in this
sample. Each value can be a TIFF stack, a flat TIFF sequence, or a
single-block TeraStitcher RES directory. Set xy_downsample=1 for native
resolution, 4 for a 1/4-resolution mosaic, etc. Use --dry-run to check the
geometry before writing.

Output is a TIFF sequence per marker, written one slice at a time. Only XY
canvas placement changes; Z order and image content are unchanged. This
script aligns mosaics after stitching; it does not merge the original tiles.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

MARKER_TO_WAVELENGTH = {
    "GFP": "640nm", "RFP": "561nm", "Sox9": "730nm", "Olig2": "488nm",
}


@dataclass(frozen=True)
class Geometry:
    marker: str
    xml: Path
    tiles: frozenset[str]
    voxel_xy: tuple[float, float]
    tile_size: tuple[int, int]
    min_x: int
    min_y: int
    max_x: int  # exclusive
    max_y: int  # exclusive


def resolve_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def read_geometry(raw_root: Path, marker: str, image_module) -> Geometry:
    raw_dir = raw_root / MARKER_TO_WAVELENGTH[marker]
    named = raw_dir / f"xml_merging_{marker}.xml"
    generic = raw_dir / "xml_merging.xml"
    xml = named if named.is_file() else generic
    if not xml.is_file():
        raise FileNotFoundError(f"No merging XML for {marker}: {named} or {generic}")
    root = ET.parse(xml).getroot()
    stacks = root.find("STACKS")
    voxel = root.find("voxel_dims")
    if stacks is None or not len(stacks) or voxel is None:
        raise ValueError(f"Missing STACKS or voxel_dims in {xml}")
    voxel_xy = (float(voxel.attrib["H"]), float(voxel.attrib["V"]))
    if min(voxel_xy) <= 0:
        raise ValueError(f"Invalid XY voxel size in {xml}")

    names, positions = set(), []
    for stack in stacks:
        name = Path(stack.attrib["DIR_NAME"].replace("\\", "/")).name
        if name in names:
            raise ValueError(f"Duplicate tile {name} in {xml}")
        names.add(name)
        positions.append((int(stack.attrib["ABS_H"]), int(stack.attrib["ABS_V"])))
    tile_dir = raw_dir / Path(stacks[0].attrib["DIR_NAME"].replace("\\", "/"))
    if not tile_dir.is_dir():
        raise FileNotFoundError(f"Raw tile directory missing: {tile_dir}")
    raw_tiffs = sorted(p for p in tile_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in (".tif", ".tiff"))
    if not raw_tiffs:
        raise FileNotFoundError(f"No raw TIFF in {tile_dir}")
    with image_module.open(raw_tiffs[0]) as image:
        tile_size = image.size
    width, height = tile_size
    return Geometry(marker, xml, frozenset(names), voxel_xy, tile_size,
                    min(x for x, _ in positions), min(y for _, y in positions),
                    max(x for x, _ in positions) + width,
                    max(y for _, y in positions) + height)


def calculate_canvas(geometries: dict[str, Geometry], factor: int,
                     image_sizes: dict[str, tuple[int, int]]):
    """Return common (width,height) and each marker's (left,top) in pixels."""
    if not isinstance(factor, int) or isinstance(factor, bool) or factor < 1:
        raise ValueError("xy_downsample must be a positive integer")
    first = next(iter(geometries.values()))
    for geo in geometries.values():
        if geo.tiles != first.tiles:
            raise ValueError(f"Tile set differs between {first.marker} and {geo.marker}")
        if geo.tile_size != first.tile_size or geo.voxel_xy != first.voxel_xy:
            raise ValueError(f"Tile size or XY voxel size differs for {geo.marker}")
    min_x = min(g.min_x for g in geometries.values())
    min_y = min(g.min_y for g in geometries.values())
    max_x = max(g.max_x for g in geometries.values())
    max_y = max(g.max_y for g in geometries.values())
    width = (max_x - min_x) // factor
    height = (max_y - min_y) // factor
    offsets = {}
    for marker, geo in geometries.items():
        expected = ((geo.max_x - geo.min_x) // factor,
                    (geo.max_y - geo.min_y) // factor)
        if image_sizes[marker] != expected:
            raise ValueError(
                f"{marker}: stitched size {image_sizes[marker]} differs from XML "
                f"prediction {expected} at xy_downsample={factor}; check RES level, "
                "XML version, and whether the image was cropped")
        # TeraStitcher normalizes each mosaic to its own minimum ABS_H/V.
        left = math.floor((geo.min_x - min_x) / factor + 0.5)
        top = math.floor((geo.min_y - min_y) / factor + 0.5)
        offsets[marker] = (left, top)
        width = max(width, left + expected[0])
        height = max(height, top + expected[1])
    return (width, height), offsets


def natural_key(path: Path):
    return [int(p) if p.isdigit() else p.lower()
            for p in re.split(r"(\d+)", path.name)]


def source_pages(path: Path, image_module):
    """List (TIFF path, page index, output name), rejecting tiled RES volumes."""
    if path.is_file():
        if path.suffix.lower() not in (".tif", ".tiff"):
            raise ValueError(f"Only TIFF is supported: {path}")
        with image_module.open(path) as image:
            count = image.n_frames
        return [(path, i, f"{path.stem}_z{i:05d}.tif") for i in range(count)]
    if not path.is_dir():
        raise FileNotFoundError(f"Stitched input missing: {path}")
    files = [p for p in path.rglob("*") if p.is_file()
             and p.suffix.lower() in (".tif", ".tiff")]
    if not files:
        raise ValueError(f"No TIFF files in {path}")
    if len({p.parent for p in files}) != 1:
        raise ValueError(
            f"{path} has multiple TIFF block directories. Export one complete "
            "2D TIFF per Z slice, or select a single-block RES directory.")
    files.sort(key=natural_key)
    for file in files:
        with image_module.open(file) as image:
            if image.n_frames != 1:
                raise ValueError(f"Sequence TIFF has multiple pages: {file}")
    return [(p, 0, p.stem + ".tif") for p in files]


def inspect_pages(pages, image_module):
    reference = None
    for file, page, _ in pages:
        with image_module.open(file) as image:
            image.seek(page)
            if len(image.getbands()) != 1:
                raise ValueError(f"Input is not grayscale: {file}")
            current = (image.size, image.mode)
        if reference is None:
            reference = current
        elif current != reference:
            raise ValueError(f"TIFF page size or pixel mode differs: {file}")
    return reference


def write_channel(pages, destination: Path, offset, canvas, image_module):
    destination.mkdir()
    for n, (file, page, name) in enumerate(pages, 1):
        with image_module.open(file) as source:
            source.seek(page)
            source.load()
            image = image_module.new(source.mode, canvas, color=0)
            image.paste(source, offset)
            temporary = destination / (name + ".part")
            image.save(temporary, format="TIFF", compression="tiff_lzw")
            os.replace(temporary, destination / name)
        if n % 25 == 0 or n == len(pages):
            print(f"  {destination.name}: {n}/{len(pages)}", flush=True)


def run(config_path: Path, dry_run: bool = False):
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Pillow is required: pip install Pillow") from exc
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    raw_root = resolve_path(config["raw_sample_dir"], base)
    output = resolve_path(config["output_dir"], base)
    factor = config["xy_downsample"]
    inputs = config["stitched_images"]
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("stitched_images must map at least one marker to a TIFF path")
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new output_dir: {output}")

    marker_paths = {}
    for key, value in inputs.items():
        marker = next((m for m in MARKER_TO_WAVELENGTH
                       if m.lower() == key.lower()), None)
        if marker is None or marker in marker_paths:
            raise ValueError(f"Unknown or repeated marker: {key}")
        marker_paths[marker] = resolve_path(value, base)
    geometries = {m: read_geometry(raw_root, m, Image) for m in marker_paths}
    pages = {m: source_pages(p, Image) for m, p in marker_paths.items()}
    descriptions = {m: inspect_pages(pages[m], Image) for m in marker_paths}
    counts = {m: len(pages[m]) for m in marker_paths}
    if len(set(counts.values())) != 1:
        raise ValueError(f"Different Z slice counts: {counts}; this script aligns XY only")
    modes = {m: descriptions[m][1] for m in marker_paths}
    if len(set(modes.values())) != 1:
        raise ValueError(f"Different TIFF pixel modes: {modes}")
    image_sizes = {m: descriptions[m][0] for m in marker_paths}
    canvas, offsets = calculate_canvas(geometries, factor, image_sizes)
    print(f"Canvas: width={canvas[0]}, height={canvas[1]}, "
          f"Z={next(iter(counts.values()))}, mode={next(iter(modes.values()))}")
    for marker in marker_paths:
        left, top = offsets[marker]
        w, h = image_sizes[marker]
        print(f"  {marker} ({MARKER_TO_WAVELENGTH[marker]}): "
              f"left={left}, top={top}, right={canvas[0]-left-w}, "
              f"bottom={canvas[1]-top-h}")
    if dry_run:
        return

    output.mkdir(parents=True)
    for marker in marker_paths:
        write_channel(pages[marker], output / marker, offsets[marker], canvas, Image)
    manifest = {
        "canvas_width": canvas[0], "canvas_height": canvas[1],
        "xy_downsample": factor, "z_slices": next(iter(counts.values())),
        "channels": {
            m: {"wavelength": MARKER_TO_WAVELENGTH[m],
                "stitched_input": str(marker_paths[m]),
                "merging_xml": str(geometries[m].xml),
                "left": offsets[m][0], "top": offsets[m][1]}
            for m in marker_paths
        },
    }
    (output / "alignment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Aligned TIFF sequences: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate inputs and print offsets without writing images")
    args = parser.parse_args()
    try:
        run(args.config, args.dry_run)
    except (OSError, ValueError, KeyError, ET.ParseError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

