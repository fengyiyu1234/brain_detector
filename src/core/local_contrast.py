"""Stage 2.2: local-contrast SNR of StarDist detections, used by the Stage 2.25 filter.

Each detection's background ring is the pixels 2-8 px outside its instance polygon,
excluding every instance mask of the slice. SNR = (mean - ring median) / (1.4826 * ring MAD),
with `mean` the raw mask mean saved by Stage 2. Tissue texture sets the ring noise
(MAD/background ~0.15), so one SNR threshold works across slices of different brightness.
Calibration (44 hand-thresholded slices, 9 EGFR samples): snr_min 5,
Y:/Fengyi/EGFR_brain/test/sox9_classifier_20261103.

Config (only listed channels are computed and filtered):
    "contrast_filter": {"Sox9": {"snr_min": 5}}
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.draw import polygon as draw_polygon

CONTRAST_SCHEMA_VERSION = "1"
CONTRAST_SUBDIR = "contrast"  # under 1_2d_raw
RING_INNER_PX, RING_OUTER_PX, MIN_RING_PIXELS = 2, 8, 10
CONTRAST_KEYS = frozenset({"snr_min"})
CONTRAST_COLUMNS = ("detection_id", "ring_bg", "ring_mad", "ring_n", "snr")
PREFETCH_IMAGES = 4


def resolve_contrast_filter(config: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """Return {channel_id: params} for active channels with an enabled snr_min."""
    spec = config.get("contrast_filter") or {}
    if not isinstance(spec, Mapping):
        raise ValueError("contrast_filter must be an object")
    channels = {ch["id"]: ch for ch in config.get("channels_routing", []) if ch.get("active", True)}
    resolved = {}
    for channel_id, params in spec.items():
        if channel_id not in channels:
            raise ValueError(f"contrast_filter: {channel_id!r} is not an active channel")
        if not isinstance(params, Mapping) or set(params) - CONTRAST_KEYS:
            raise ValueError(f"contrast_filter.{channel_id} accepts only {sorted(CONTRAST_KEYS)}")
        channel = channels[channel_id]
        if str(channel.get("model", "")).lower() != "stardist" or channel.get("double_exposure"):
            raise ValueError(f"contrast_filter.{channel_id}: needs a single-exposure StarDist channel "
                             "(SNR uses its instance polygons)")
        value = params.get("snr_min")
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"contrast_filter.{channel_id}.snr_min must be a finite number or null")
        resolved[channel_id] = {"snr_min": float(value)}
    return resolved


def contrast_settings() -> dict[str, Any]:
    """Everything that determines the SNR table, for checkpoint signatures."""
    from src.core.provenance import file_sha256
    return {"schema_version": CONTRAST_SCHEMA_VERSION, "ring_inner_px": RING_INNER_PX,
            "ring_outer_px": RING_OUTER_PX, "min_ring_pixels": MIN_RING_PIXELS,
            "code_sha256": file_sha256(__file__)}


def contrast_paths(contrast_dir: str | os.PathLike[str], tile: str, channel_id: str) -> tuple[Path, Path]:
    base = Path(contrast_dir)
    return base / f"{tile}_{channel_id}_contrast.csv", base / f"{tile}_{channel_id}_contrast_manifest.json"


def slice_ring_stats(image: np.ndarray, boxes: np.ndarray, polygons: list) -> np.ndarray:
    """Return (n, 3) ring median, ring MAD*1.4826 and ring pixel count for one slice.

    boxes: (n, 4) int x1, y1, x2, y2. polygons: per box an (m, 2) x/y array or None.
    Later instances overwrite earlier ones where polygons overlap, as in StarDist labels.
    A box whose polygon is missing or fully overwritten uses its rectangle as the mask.
    """
    height, width = image.shape
    labels = np.zeros(image.shape, np.int32)
    for index, xy in enumerate(polygons, 1):
        if xy is not None and len(xy):
            rows, cols = draw_polygon(xy[:, 1], xy[:, 0], image.shape)
            labels[rows, cols] = index
    any_mask = labels > 0
    out = np.full((len(boxes), 3), np.nan)
    for index, (x1, y1, x2, y2) in enumerate(boxes, 1):
        y0, y3 = max(0, y1 - RING_OUTER_PX - 1), min(height, y2 + RING_OUTER_PX + 1)
        x0, x3 = max(0, x1 - RING_OUTER_PX - 1), min(width, x2 + RING_OUTER_PX + 1)
        own = labels[y0:y3, x0:x3] == index
        if not own.any():
            own[y1 - y0:y2 - y0, x1 - x0:x2 - x0] = True
        dist = ndimage.distance_transform_edt(~own)
        ring = image[y0:y3, x0:x3][(dist > RING_INNER_PX) & (dist <= RING_OUTER_PX)
                                   & ~any_mask[y0:y3, x0:x3]]
        out[index - 1, 2] = ring.size
        if ring.size >= MIN_RING_PIXELS:
            median = np.median(ring)
            out[index - 1, :2] = median, 1.4826 * np.median(np.abs(ring - median))
    return out


def _read_image(path: str) -> np.ndarray:
    import cv2
    image = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    if image is None:
        raise FileNotFoundError(f"Cannot read source image {path}")
    return image.astype(np.float32)


def tile_contrast(raw_csv: str | os.PathLike[str], instances_csv: str | os.PathLike[str],
                  read_image=_read_image) -> pd.DataFrame:
    """SNR table for one tile x channel, in raw CSV row order."""
    raw = pd.read_csv(raw_csv, usecols=["detection_id", "x1", "y1", "x2", "y2", "mean", "source_image"],
                      dtype={"detection_id": str, "source_image": str})
    if raw.source_image.isna().any() or raw.source_image.eq("").any():
        raise ValueError(f"{raw_csv}: rows without source_image; cannot read the slice images")
    polygons = {}
    for chunk in pd.read_csv(instances_csv, usecols=["detection_id", "polygon_xy_json"],
                             dtype=str, chunksize=500_000):
        polygons.update(zip(chunk.detection_id, chunk.polygon_xy_json))
    stats = np.full((len(raw), 3), np.nan)
    groups = list(raw.groupby("source_image", sort=False).indices.items())

    def load(position):
        return read_image(groups[position][0])

    with ThreadPoolExecutor(max_workers=PREFETCH_IMAGES) as pool:
        pending = {i: pool.submit(load, i) for i in range(min(PREFETCH_IMAGES, len(groups)))}
        for position, (_path, rows) in enumerate(groups):
            image = pending.pop(position).result()
            ahead = position + PREFETCH_IMAGES
            if ahead < len(groups):
                pending[ahead] = pool.submit(load, ahead)
            part = raw.iloc[rows]
            shapes = []
            for det in part.detection_id:
                text = polygons.get(det)
                shapes.append(np.asarray(json.loads(text), float)
                              if isinstance(text, str) and text else None)
            stats[rows] = slice_ring_stats(
                image, part[["x1", "y1", "x2", "y2"]].to_numpy(int), shapes)
    out = pd.DataFrame(stats, columns=["ring_bg", "ring_mad", "ring_n"])
    out.insert(0, "detection_id", raw.detection_id.to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = (raw["mean"].to_numpy(float) - out.ring_bg.to_numpy()) / out.ring_mad.to_numpy()
    out["snr"] = np.where(out.ring_mad.to_numpy() > 0, snr, np.nan)
    out["ring_n"] = out.ring_n.astype(int)
    return out


def contrast_signature(raw_csv: str | os.PathLike[str], instances_csv: str | os.PathLike[str],
                       settings: Mapping[str, Any]) -> dict[str, Any]:
    from src.core.provenance import file_stamp
    stat = os.stat(instances_csv)
    # Instance CSVs are written together with the hashed raw CSV; hashing ~1 GB each is too slow.
    return {"stage": "contrast", "raw": file_stamp(raw_csv, with_hash=True),
            "instances": {"name": Path(instances_csv).name, "size": stat.st_size},
            "settings": dict(settings)}


def run_contrast_tile(tile: str, channel_id: str, raw_dir: str, contrast_dir: str,
                      settings: Mapping[str, Any]) -> tuple[str, int, int]:
    """Compute and checkpoint one tile; returns (tile, n boxes, n with SNR) or n=-1 when reused."""
    from src.core.detection_filter import atomic_write_csv
    from src.core.provenance import output_manifest_valid, write_output_manifest
    raw_csv = Path(raw_dir) / f"{tile}_{channel_id}_result.csv"
    instances_csv = Path(raw_dir) / f"{tile}_{channel_id}_instances.csv"
    target, manifest = contrast_paths(contrast_dir, tile, channel_id)
    signature = contrast_signature(raw_csv, instances_csv, settings)
    if output_manifest_valid(manifest, signature):
        return tile, -1, -1
    table = tile_contrast(raw_csv, instances_csv)
    atomic_write_csv(table, target)
    n_valid = int(table.snr.notna().sum())
    write_output_manifest(manifest, [target], signature,
                          metadata={"n_boxes": len(table), "n_snr": n_valid})
    return tile, len(table), n_valid


def contrast_ready(tile: str, channel_id: str, raw_dir: str, contrast_dir: str,
                   settings: Mapping[str, Any]) -> bool:
    from src.core.provenance import output_manifest_valid
    raw_csv = Path(raw_dir) / f"{tile}_{channel_id}_result.csv"
    instances_csv = Path(raw_dir) / f"{tile}_{channel_id}_instances.csv"
    _, manifest = contrast_paths(contrast_dir, tile, channel_id)
    return output_manifest_valid(manifest, contrast_signature(raw_csv, instances_csv, settings))


def contrast_filter_signature(contrast_csv: str | os.PathLike[str], params: Mapping[str, Any],
                              settings: Mapping[str, Any]) -> dict[str, Any]:
    """Extra Stage 2.25 signature for a contrast-filtered channel."""
    from src.core.provenance import file_stamp
    return {"params": dict(params), "settings": dict(settings),
            "source": file_stamp(contrast_csv, with_hash=True)}


def apply_snr_filter(filtered: pd.DataFrame, rejected: pd.DataFrame, stats: dict,
                     contrast_csv: str | os.PathLike[str], snr_min: float):
    """Drop filtered rows with snr < snr_min after the regular filter.

    Adds an `snr` column to both outputs. Rows whose ring is too small to measure
    (snr NaN) are kept. Returns (filtered, rejected, stats).
    """
    table = pd.read_csv(contrast_csv, usecols=["detection_id", "snr"], dtype={"detection_id": str})
    snr = table.set_index("detection_id").snr
    ids = filtered.detection_id.astype(str)
    missing = ~ids.isin(snr.index)
    if missing.any():
        raise ValueError(f"{contrast_csv} lacks {int(missing.sum())} filtered detection(s), "
                         f"e.g. {ids[missing].iloc[0]}; recompute Stage 2.2")
    filtered = filtered.assign(snr=snr.reindex(ids).to_numpy())
    low = filtered.snr.to_numpy() < snr_min
    removed = filtered.loc[low].assign(rejection_reason="snr_min")
    rejected = rejected.assign(snr=snr.reindex(rejected.detection_id.astype(str)).to_numpy()) \
        if "detection_id" in rejected else rejected
    if len(removed):
        rejected = (removed.reindex(columns=rejected.columns.union(removed.columns, sort=False))
                    .reset_index(drop=True) if rejected.empty
                    else pd.concat([rejected, removed], ignore_index=True))
    stats = {**stats, "after": int((~low).sum()),
             "removed": {**stats["removed"], "snr_min": int(low.sum())},
             "removed_total": stats["removed_total"] + int(low.sum()),
             "thresholds": {**stats.get("thresholds", {}), "snr_min": float(snr_min)}}
    return filtered.loc[~low].copy(), rejected, stats
