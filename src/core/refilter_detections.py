"""Regenerate filtered tile CSVs from existing raw, aligned, or fused CSVs."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config.loader import load_config
from src.core.provenance import file_stamp, provenance_columns
from src.core.provenance import file_stamp, provenance_columns
from src.core.detection_filter import (
    FILTER_SCHEMA_VERSION, filter_detection_df, resolve_filter_params,
    source_dir_for_channel,
)


def _derived_paths(config: dict) -> dict[str, str]:
    base = config["paths"].get("pATHRESULT")
    if not base:
        raise ValueError("Config is missing paths.pATHRESULT")
    return {
        "pATH_DET_RES": os.path.join(base, "1_2d_raw"),
        "pATH_ALIGN_OFFSETS": os.path.join(base, "3_2d_aligned"),
        "pATH_DET_FUSED": os.path.join(base, "3_2d_aligned_fusion"),
        "pATH_DET_FILTERED": os.path.join(base, "4_2d_filtered"),
    }


def _tile_names(source_dir: str, channel_id: str) -> set[str]:
    path = Path(source_dir)
    suffix = f"_{channel_id}_result.csv"
    if not path.is_dir():
        return set()
    return {entry.name[:-len(suffix)] for entry in path.iterdir()
            if entry.is_file() and entry.name.endswith(suffix)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU-only regeneration of Stage 2.75 filtered detection CSVs."
    )
    parser.add_argument("--config", required=True, help="Pipeline JSON/JSONC config.")
    parser.add_argument("--channels", nargs="+", help="Logical channel IDs; default: all active.")
    parser.add_argument("--tiles", nargs="+", help="Tile names; default: discover from source CSVs.")
    parser.add_argument("--output-dir", help="Destination; default: pATHRESULT/4_2d_filtered.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and report without writing.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing target CSVs.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    started = time.time()
    config = load_config(args.config)
    channels = [ch for ch in config.get("channels_routing", []) if ch.get("active", True)]
    selected_ids = args.channels or [ch["id"] for ch in channels]
    by_id = {ch["id"]: ch for ch in channels}
    unknown = sorted(set(selected_ids) - set(by_id))
    if unknown:
        raise ValueError(f"Unknown or inactive --channels value(s): {unknown}")
    selected = [by_id[channel_id] for channel_id in selected_ids]
    derived = _derived_paths(config)
    pipeline_mode = config.get("pipeline_mode", "post_align")
    output_dir = os.path.abspath(args.output_dir or derived["pATH_DET_FILTERED"])

    if args.tiles:
        tiles = list(dict.fromkeys(args.tiles))
    else:
        tiles = sorted(set().union(*(
            _tile_names(source_dir_for_channel(derived, pipeline_mode, ch), ch["id"])
            for ch in selected
        )))
    if not tiles:
        raise FileNotFoundError("No source CSV tiles matched the selected channel(s).")

    jobs = []
    for tile in tiles:
        for ch in selected:
            source_dir = source_dir_for_channel(derived, pipeline_mode, ch)
            source = os.path.join(source_dir, f"{tile}_{ch['id']}_result.csv")
            target = os.path.join(output_dir, f"{tile}_{ch['id']}_result.csv")
            if not os.path.isfile(source):
                raise FileNotFoundError(
                    f"Required source for tile={tile}, channel={ch['id']} is missing: {source}"
                )
            if os.path.abspath(source) == os.path.abspath(target):
                raise ValueError("Source and destination are the same aligned CSV. "
                                 "Use a separate --output-dir or rerun the pipeline from Stage 2.25.")
            if os.path.exists(target) and not args.overwrite:
                raise FileExistsError(f"Target exists (pass --overwrite): {target}")
            jobs.append((tile, ch, source, target))

    print(f"Matched {len(tiles)} tile(s), {len(selected)} channel(s), {len(jobs)} CSV(s).")
    prepared = []
    for tile, ch, source, target in jobs:
        params = resolve_filter_params(config, ch)
        source_df = provenance_columns(pd.read_csv(source), tile, ch['id'])
        filtered, stats, rejected = filter_detection_df(
            source_df, params, return_stats=True, return_rejected=True,
            context=f"{source} ({ch['id']})",
        )
        prepared.append((tile, ch, source, target, params, filtered, stats, rejected))
        print(f"[{tile}][{ch['id']}] {stats['before']} -> {stats['after']} "
              f"(removed={stats['removed_total']}; score_min={stats['removed']['score_min']})")

    if args.dry_run:
        print("Dry run complete: no files were written.")
        return 0

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    parts: list[tuple[Path, Path]] = []
    try:
        for _, _, _, target, _, filtered, _, rejected in prepared:
            for destination, data in (
                (Path(target), filtered),
                (Path(target.replace('_result.csv', '_rejected.csv')), rejected),
            ):
                part = destination.with_name(destination.name + ".part")
                data.to_csv(part, index=False)
                parts.append((part, destination))
        for part, destination in parts:
            os.replace(part, destination)
    except Exception:
        for part, _ in parts:
            if part.exists():
                part.unlink()
        raise

    stale = [name for name in (
        "5_2d_global", "6_3d_global", "7_colocalization"
    ) if os.path.exists(os.path.join(derived["pATH_DET_RES"], "..", name))]
    if stale:
        print("WARNING: downstream outputs may now be stale (not deleted): " + ", ".join(stale))

    records = [{
        "tile": tile, "channel": ch["id"], "source": source, "target": target,
        "rejected": target.replace('_result.csv', '_rejected.csv'),
        "source_stamp": file_stamp(source, with_hash=True),
        "target_stamp": file_stamp(target, with_hash=True),
        "rejected_stamp": file_stamp(target.replace('_result.csv', '_rejected.csv'),
                                      with_hash=True),
        "params": params, "before": stats["before"], "after": stats["after"],
        "removed_total": stats["removed_total"], "removed_by_step": stats["removed"],
        "thresholds": stats["thresholds"],
    } for tile, ch, source, target, params, _, stats, _ in prepared]
    manifest = {
        "filter_schema_version": FILTER_SCHEMA_VERSION,
        "config": os.path.abspath(args.config),
        "pipeline_mode": pipeline_mode,
        "output_dir": output_dir,
        "elapsed_seconds": round(time.time() - started, 3),
        "records": records,
    }
    manifest_path = Path(output_dir) / "refilter_manifest.json"
    part = manifest_path.with_name(manifest_path.name + ".part")
    try:
        part.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(part, manifest_path)
    except Exception:
        if part.exists():
            part.unlink()
        raise
    print(f"Wrote {len(prepared)} filtered CSV(s) and {manifest_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

