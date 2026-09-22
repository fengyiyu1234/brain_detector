"""Shared, CPU-only filtering for per-tile 2-D detection CSVs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

FILTER_SCHEMA_VERSION = "1"
FILTER_KEYS = frozenset({
    "bbox_min", "bbox_max", "bbox_max_aspect_ratio", "bbox_area_pct_min",
    "bbox_area_pct_max", "bbox_mean_pct_min", "bbox_mean_min",
    "nms_containment_thresh",
})
REQUIRED_COLUMNS = ("x1", "y1", "x2", "y2", "score", "mean", "z")


def _channel_spec(config: Mapping[str, Any], channel: str | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(channel, Mapping):
        return channel
    found = next((ch for ch in config.get("channels_routing", []) if ch.get("id") == channel), None)
    if found is None:
        raise ValueError(f"Unknown channel id {channel!r} in channel_filter_overrides")
    return found


def resolve_filter_params(config: Mapping[str, Any], channel: str | Mapping[str, Any]) -> dict[str, Any]:
    """Merge model defaults with a channel-id override (where null disables)."""
    ch = _channel_spec(config, channel)
    channel_id, model = ch.get("id"), str(ch.get("model", "")).lower()
    defaults = config.get("detection_params", {}).get(model)
    if not isinstance(defaults, Mapping):
        raise ValueError(f"Channel {channel_id!r} uses unknown model {model!r}")
    overrides = config.get("channel_filter_overrides", {})
    if not isinstance(overrides, Mapping):
        raise ValueError("channel_filter_overrides must be an object")
    known_channels = {c.get("id") for c in config.get("channels_routing", [])}
    unknown_channels = set(overrides) - known_channels
    if unknown_channels:
        raise ValueError(f"Unknown channel_filter_overrides channel(s): {sorted(unknown_channels)}")
    override = overrides.get(channel_id, {})
    if not isinstance(override, Mapping):
        raise ValueError(f"channel_filter_overrides.{channel_id} must be an object")
    unknown_keys = set(override) - FILTER_KEYS
    if unknown_keys:
        raise ValueError(f"Unknown filter key(s) for {channel_id}: {sorted(unknown_keys)}")
    params = {key: defaults.get(key) for key in FILTER_KEYS}
    if params["bbox_mean_min"] is None and "nucleus_mean_min" in defaults:
        params["bbox_mean_min"] = defaults["nucleus_mean_min"]
    params.update(override)
    validate_filter_params(params)
    return params


def validate_filter_params(params: Mapping[str, Any]) -> None:
    unknown = set(params) - FILTER_KEYS
    if unknown:
        raise ValueError(f"Unknown filter key(s): {sorted(unknown)}")

    def number(key: str, low: float | None = None, high: float | None = None) -> None:
        value = params.get(key)
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"{key} must be a finite number or null")
        if low is not None and value < low:
            raise ValueError(f"{key} must be >= {low}")
        if high is not None and value > high:
            raise ValueError(f"{key} must be <= {high}")

    number("bbox_min", 0); number("bbox_max", 0); number("bbox_max_aspect_ratio", 1)
    number("bbox_area_pct_min", 0, 100); number("bbox_area_pct_max", 0, 100)
    number("bbox_mean_pct_min", 0, 100); number("bbox_mean_min")
    number("nms_containment_thresh", 0, 1)
    if params.get("bbox_min") is not None and params.get("bbox_max") is not None and params["bbox_min"] > params["bbox_max"]:
        raise ValueError("bbox_min must be <= bbox_max")
    if params.get("bbox_area_pct_min") is not None and params.get("bbox_area_pct_max") is not None and params["bbox_area_pct_min"] > params["bbox_area_pct_max"]:
        raise ValueError("bbox_area_pct_min must be <= bbox_area_pct_max")


def _check_columns(df: pd.DataFrame, context: str | None) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        where = f" in {context}" if context else ""
        raise ValueError(f"Detection CSV is missing required column(s) {missing}{where}")


def _iomin_keep(df: pd.DataFrame, threshold: float) -> np.ndarray:
    keep = np.ones(len(df), dtype=bool)
    for _, group in df.groupby("z", sort=False):
        indices = group.index.to_numpy()
        x1, y1 = group.x1.to_numpy(float), group.y1.to_numpy(float)
        x2, y2 = group.x2.to_numpy(float), group.y2.to_numpy(float)
        areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        order = np.argsort(-group.score.to_numpy(float), kind="mergesort")
        local_keep = np.ones(len(group), dtype=bool)
        for pos, winner in enumerate(order):
            if not local_keep[winner]:
                continue
            for loser in order[pos + 1:]:
                if not local_keep[loser]:
                    continue
                inter = max(0., min(x2[winner], x2[loser]) - max(x1[winner], x1[loser]))
                inter *= max(0., min(y2[winner], y2[loser]) - max(y1[winner], y1[loser]))
                denom = min(areas[winner], areas[loser])
                if denom > 0 and inter / denom > threshold:
                    local_keep[loser] = False
        keep[indices[~local_keep]] = False
    return keep


def filter_detection_df(df: pd.DataFrame, params: Mapping[str, Any], return_stats: bool = False,
                        *, context: str | None = None) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """Filter a complete tile×channel dataframe without changing its input."""
    validate_filter_params(params); _check_columns(df, context)
    out = df.copy()
    stats: dict[str, Any] = {"before": len(out), "removed": {}, "thresholds": {}}

    def apply(name: str, mask: np.ndarray) -> None:
        nonlocal out
        before = len(out); out = out.loc[np.asarray(mask, dtype=bool)].copy()
        stats["removed"][name] = before - len(out)

    if not out.empty:
        w = out.x2.to_numpy(float) - out.x1.to_numpy(float); h = out.y2.to_numpy(float) - out.y1.to_numpy(float)
        mask = np.ones(len(out), dtype=bool)
        if params.get("bbox_min") is not None: mask &= (w >= params["bbox_min"]) & (h >= params["bbox_min"])
        if params.get("bbox_max") is not None: mask &= (w <= params["bbox_max"]) & (h <= params["bbox_max"])
        apply("bbox", mask)
    else: stats["removed"]["bbox"] = 0
    if not out.empty and params.get("bbox_max_aspect_ratio") is not None:
        w = out.x2.to_numpy(float) - out.x1.to_numpy(float); h = out.y2.to_numpy(float) - out.y1.to_numpy(float)
        apply("aspect_ratio", np.maximum(w, h) <= params["bbox_max_aspect_ratio"] * np.maximum(np.minimum(w, h), 1e-6))
    else: stats["removed"]["aspect_ratio"] = 0

    def percentile_step(name: str, values: np.ndarray, percentile: Any, keep_min: bool) -> None:
        if percentile is None or out.empty:
            stats["removed"][name] = 0; return
        threshold = float(np.percentile(values, percentile)); stats["thresholds"][name] = threshold
        apply(name, values >= threshold if keep_min else values <= threshold)

    area = (out.x2.to_numpy(float) - out.x1.to_numpy(float)) * (out.y2.to_numpy(float) - out.y1.to_numpy(float)) if not out.empty else np.array([])
    percentile_step("area_pct_min", area, params.get("bbox_area_pct_min"), True)
    area = (out.x2.to_numpy(float) - out.x1.to_numpy(float)) * (out.y2.to_numpy(float) - out.y1.to_numpy(float)) if not out.empty else np.array([])
    percentile_step("area_pct_max", area, params.get("bbox_area_pct_max"), False)
    percentile_step("mean_pct_min", out["mean"].to_numpy(float) if not out.empty else np.array([]), params.get("bbox_mean_pct_min"), True)
    if not out.empty and params.get("bbox_mean_min") is not None: apply("mean_min", out["mean"].to_numpy(float) >= params["bbox_mean_min"])
    else: stats["removed"]["mean_min"] = 0
    if not out.empty and params.get("nms_containment_thresh") is not None: apply("containment_nms", _iomin_keep(out, float(params["nms_containment_thresh"])))
    else: stats["removed"]["containment_nms"] = 0
    stats["after"] = len(out); stats["removed_total"] = stats["before"] - stats["after"]
    return (out, stats) if return_stats else out


def source_dir_for_channel(derived: Mapping[str, str], pipeline_mode: str, channel: Mapping[str, Any]) -> str:
    if channel.get("double_exposure"):
        return derived["pATH_DET_FUSED"]
    return derived["pATH_ALIGN_OFFSETS"] if pipeline_mode == "pre_align" else derived["pATH_DET_RES"]


def atomic_write_csv(df: pd.DataFrame, destination: str | os.PathLike[str]) -> None:
    target = Path(destination); target.parent.mkdir(parents=True, exist_ok=True); part = target.with_name(target.name + ".part")
    try:
        df.to_csv(part, index=False); os.replace(part, target)
    except Exception:
        if part.exists(): part.unlink()
        raise

