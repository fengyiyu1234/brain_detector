"""Stage 2.25 / 2.75 SNR filter on the Stage 2.2 contrast table (src/core/local_contrast.py).

Config, per single-exposure StarDist channel:
    "contrast_filter": {
        "Sox9": {"snr_min": 5, "noise": "slice", "max_removed_frac": 0.35,
                 "snr_step": 1, "snr_floor": 2}
    }

- noise "ring" (default): SNR = (mean - ring median) / ring noise of the box itself
  (the Stage 2.2 `snr` column).
- noise "slice": same numerator, divided by the slice noise = median ring noise
  (1.4826 * ring MAD, zeros ignored) of all raw boxes on the same tile and slice image.
- max_removed_frac (default null = off): per tile and slice image, if snr_min would remove more
  than this share of the boxes reaching the SNR step, the threshold drops by snr_step until it
  removes no more than that share or the next step would go below snr_floor.

Boxes without an SNR (ring too small, or a slice without any ring noise) are kept and count
in the share's denominator. Output rows get `snr` (the value compared) and, outside the plain
form, `snr_threshold` (the threshold of the box's slice) and, for slice noise, `slice_noise`.

A config with only snr_min (ring noise, no adaptation) is the plain form: it filters and signs
its checkpoints exactly as before these options existed, so such samples are not refiltered.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SNR_FILTER_KEYS = frozenset({"snr_min", "noise", "max_removed_frac", "snr_step", "snr_floor"})
NOISE_MODES = ("ring", "slice")
DEFAULT_SNR_STEP = 1.0
DEFAULT_SNR_FLOOR = 2.0


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def resolve_snr_filter(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return {channel_id: params} for active channels with an enabled snr_min.

    Plain form {"snr_min": x} when noise is ring and max_removed_frac is off, else all keys.
    """
    spec = config.get("contrast_filter") or {}
    if not isinstance(spec, Mapping):
        raise ValueError("contrast_filter must be an object")
    channels = {ch["id"]: ch for ch in config.get("channels_routing", []) if ch.get("active", True)}
    resolved = {}
    for channel_id, params in spec.items():
        name = f"contrast_filter.{channel_id}"
        if channel_id not in channels:
            raise ValueError(f"contrast_filter: {channel_id!r} is not an active channel")
        if not isinstance(params, Mapping) or set(params) - SNR_FILTER_KEYS:
            raise ValueError(f"{name} accepts only {sorted(SNR_FILTER_KEYS)}")
        channel = channels[channel_id]
        if str(channel.get("model", "")).lower() != "stardist" or channel.get("double_exposure"):
            raise ValueError(f"{name}: needs a single-exposure StarDist channel "
                             "(SNR uses its instance polygons)")
        if params.get("snr_min") is None:
            continue
        snr_min = _number(params["snr_min"], f"{name}.snr_min")
        noise = params.get("noise", "ring")
        if noise not in NOISE_MODES:
            raise ValueError(f"{name}.noise must be one of {NOISE_MODES}")
        max_frac = params.get("max_removed_frac")
        if max_frac is not None:
            max_frac = _number(max_frac, f"{name}.max_removed_frac")
            if not 0 < max_frac < 1:
                raise ValueError(f"{name}.max_removed_frac must be between 0 and 1 (or null)")
        step = _number(params.get("snr_step", DEFAULT_SNR_STEP), f"{name}.snr_step")
        if step <= 0:
            raise ValueError(f"{name}.snr_step must be > 0")
        floor = _number(params.get("snr_floor", DEFAULT_SNR_FLOOR), f"{name}.snr_floor")
        if noise == "ring" and max_frac is None:
            resolved[channel_id] = {"snr_min": snr_min}
            continue
        if max_frac is not None and floor > snr_min:
            raise ValueError(f"{name}.snr_floor must not exceed snr_min")
        resolved[channel_id] = {"snr_min": snr_min, "noise": noise, "max_removed_frac": max_frac,
                                "snr_step": step, "snr_floor": floor}
    return resolved


def is_plain(params: Mapping[str, Any]) -> bool:
    return set(params) == {"snr_min"}


def snr_filter_signature(contrast_csv: str | os.PathLike[str], params: Mapping[str, Any],
                         settings: Mapping[str, Any]) -> dict[str, Any]:
    """Extra Stage 2.25/2.75 signature for an SNR-filtered channel.

    settings: local_contrast.contrast_settings(). The plain form keeps the signature it had
    before this module existed; the raw CSV read for slice keys is the filter's own source.
    """
    from src.core.provenance import file_sha256, file_stamp
    signature = {"params": dict(params), "settings": dict(settings),
                 "source": file_stamp(contrast_csv, with_hash=True)}
    if not is_plain(params):
        signature["filter_code_sha256"] = file_sha256(__file__)
    return signature


def slice_thresholds(snr: np.ndarray, keys: np.ndarray, params: Mapping[str, Any]) -> pd.Series:
    """{slice key: threshold} for the boxes reaching the SNR step (NaN SNR counts as kept)."""
    snr_min, max_frac = params["snr_min"], params.get("max_removed_frac")
    out = {}
    for key, values in pd.Series(snr).groupby(keys, sort=False):
        values = values.to_numpy(float)
        steps = 0
        if max_frac is not None:
            with np.errstate(invalid="ignore"):
                while ((values < snr_min - steps * params["snr_step"]).sum() > max_frac * len(values)
                       and snr_min - (steps + 1) * params["snr_step"] >= params["snr_floor"] - 1e-9):
                    steps += 1
        out[key] = snr_min - steps * params["snr_step"] if steps else snr_min
    return pd.Series(out, dtype=float)


def _snr_table(contrast_csv, raw_csv, params) -> pd.DataFrame:
    """Per raw detection: snr (as configured) and, outside the plain form, slice and slice_noise."""
    if is_plain(params):
        return pd.read_csv(contrast_csv, usecols=["detection_id", "snr"],
                           dtype={"detection_id": str}).set_index("detection_id")
    table = pd.read_csv(contrast_csv, usecols=["detection_id", "ring_bg", "ring_mad", "snr"],
                        dtype={"detection_id": str})
    raw = pd.read_csv(raw_csv, usecols=["detection_id", "mean", "source_image"],
                      dtype={"detection_id": str, "source_image": str})
    if len(raw) != len(table) or not np.array_equal(raw.detection_id.to_numpy(),
                                                    table.detection_id.to_numpy()):
        raise ValueError(f"{contrast_csv} does not match {raw_csv} row by row; recompute Stage 2.2")
    table["slice"] = raw.source_image.to_numpy()
    if params["noise"] == "slice":
        table["snr"], table["slice_noise"] = slice_snr(
            raw["mean"].to_numpy(float), table.ring_bg.to_numpy(float),
            table.ring_mad.to_numpy(float), table["slice"].to_numpy())
    return table.drop(columns=["ring_bg", "ring_mad"]).set_index("detection_id")


def slice_snr(mean: np.ndarray, ring_bg: np.ndarray, ring_mad: np.ndarray,
              keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(snr, slice_noise) for all raw boxes; keys = slice of each box (tile/slice image).

    slice_noise = median of the slice's positive ring noise; snr NaN where it is missing.
    """
    noise = pd.Series(ring_mad).where(pd.Series(ring_mad) > 0)
    slice_noise = noise.groupby(keys, sort=False).transform("median").to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = (mean - ring_bg) / slice_noise
    return np.where(slice_noise > 0, snr, np.nan), slice_noise


def apply_snr_filter(filtered: pd.DataFrame, rejected: pd.DataFrame, stats: dict,
                     contrast_csv: str | os.PathLike[str], raw_csv: str | os.PathLike[str],
                     params: Mapping[str, Any]):
    """Drop filtered rows whose SNR is below their slice's threshold, after the regular filter.

    Adds the SNR column(s) to both outputs. Returns (filtered, rejected, stats).
    """
    table = _snr_table(contrast_csv, raw_csv, params)
    ids = filtered.detection_id.astype(str)
    missing = ~ids.isin(table.index)
    if missing.any():
        raise ValueError(f"{contrast_csv} lacks {int(missing.sum())} filtered detection(s), "
                         f"e.g. {ids[missing].iloc[0]}; recompute Stage 2.2")
    plain = is_plain(params)
    columns = ["snr"] + (["slice_noise"] if not plain and params["noise"] == "slice" else [])
    filtered = filtered.assign(**{c: table[c].reindex(ids).to_numpy() for c in columns})
    snr = filtered.snr.to_numpy(float)
    if plain:
        threshold = np.full(len(filtered), params["snr_min"])
    else:
        keys = table["slice"].reindex(ids).to_numpy()
        per_slice = slice_thresholds(snr, keys, params)
        threshold = per_slice.reindex(keys).to_numpy(float)
        filtered["snr_threshold"] = threshold
        lowered = per_slice[per_slice < params["snr_min"]]
        logging.info("[SNR] %s: threshold lowered on %s / %s slice(s)%s", Path(raw_csv).name,
                     len(lowered), len(per_slice),
                     f" (lowest {lowered.min():g})" if len(lowered) else "")
    with np.errstate(invalid="ignore"):
        low = snr < threshold
    removed = filtered.loc[low].assign(rejection_reason="snr_min")
    if "detection_id" in rejected:
        rejected_ids = rejected.detection_id.astype(str)
        rejected = rejected.assign(**{c: table[c].reindex(rejected_ids).to_numpy() for c in columns})
    if len(removed):
        rejected = (removed.reindex(columns=rejected.columns.union(removed.columns, sort=False))
                    .reset_index(drop=True) if rejected.empty
                    else pd.concat([rejected, removed], ignore_index=True))
    stats = {**stats, "after": int((~low).sum()),
             "removed": {**stats["removed"], "snr_min": int(low.sum())},
             "removed_total": stats["removed_total"] + int(low.sum()),
             "thresholds": {**stats.get("thresholds", {}), "snr_min": float(params["snr_min"])}}
    return filtered.loc[~low].copy(), rejected, stats
