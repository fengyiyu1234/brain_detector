"""Export the exact Stage 3 channel tracks assigned to Stage 4 cells."""

import csv
import json
import os


SOURCE_3D_COLUMNS = (
    "coloc_id", "source_track", "channel", "source_class",
    "source_score", "source_mean", "cx", "cy", "cz",
    "cx_um", "cy_um", "cz_um", "xy_um_per_px", "z_um_per_slice",
    "track_id", "best_detection_id", "best_box_z", "member_count",
    "bounds_method", "observed_x1", "observed_y1", "observed_x2", "observed_y2",
    "x1_3d", "y1_3d", "x2_3d", "y2_3d", "z_min", "z_max",
)
SOURCE_BOX_COLUMNS = (
    "coloc_id", "source_track", "channel", "track_id", "detection_id",
    "tile_name", "slice_name", "z", "x1", "y1", "x2", "y2", "score", "mean",
)



def cell_id(soma, row_index):
    """Stable across reruns when Stage 3 member IDs are available."""
    from src.core.provenance import stable_id
    track_ids = sorted(str(track.get("track_id", ""))
                       for _, track in soma.get("source_tracks", ()))
    if track_ids and all(track_ids):
        return stable_id("coloc", track_ids)
    return stable_id("legacy_coloc", row_index, soma.get("class"),
                     soma.get("cx"), soma.get("cy"), soma.get("cz"))


def coloc_display_box(soma):
    """Return the 2D CSV projection and representative Z of a soma volume."""
    center_z = int(round(soma["cz"]))
    if soma.get("bounds_method") == "cross_channel_bbox_union":
        box = [soma["x1_3d"], soma["y1_3d"],
               soma["x2_3d"], soma["y2_3d"]]
    else:
        box = soma["per_z_boxes"].get(
            center_z,
            [soma["x1_3d"], soma["y1_3d"],
             soma["x2_3d"], soma["y2_3d"]])
    return box, center_z


def colocalization_status(soma, soma_channels, tf_channels):
    """Report unique matched channel positivity, independent of TF track count."""
    sources = {channel for channel, _ in soma.get("source_tracks", ())}
    soma_positive = [channel for channel in soma_channels if channel in sources]
    tf_positive = [channel for channel in tf_channels if channel in sources]

    def status(count):
        return ("negative", "single_positive", "double_positive")[count] if count < 3 else "multi_positive"

    return {
        "soma_positive_channels": json.dumps(soma_positive, separators=(",", ":")),
        "soma_positive_count": len(soma_positive),
        "soma_status": status(len(soma_positive)),
        "tf_positive_channels": json.dumps(tf_positive, separators=(",", ":")),
        "tf_positive_count": len(tf_positive),
        "tf_status": status(len(tf_positive)),
    }


def primary_source_trace(soma):
    """Exact source of the primary soma's representative Z box, if retained."""
    sources = soma.get("source_tracks", ())
    if not sources:
        return ("Unknown", "Unknown", "source_track_missing")
    track = sources[0][1]
    members = track.get("member_detections")
    if not members:
        return ("Unknown", "Unknown", "legacy_member_trace_missing")
    center_z = int(round(soma["cz"]))
    member = next((m for m in members if int(m["z"]) == center_z), None)
    status = "exact_primary_center_member"
    if member is None:
        member = next((m for m in members
                       if m["detection_id"] == track.get("best_detection_id")), None)
        status = "primary_best_member_synthetic_center_box"
    if member is None:
        return ("Unknown", "Unknown", "member_trace_missing")
    if soma.get("bounds_method") == "cross_channel_bbox_union":
        status = "merged_soma_union_primary_source_" + status
    return (member["tile_name"], member["slice_name"], status)


def source_3d_records(soma, xy_um=None, z_um=None):
    """Return every matched source track in the global aligned coordinate frame."""
    records = []
    for source_track, (channel, track) in enumerate(soma.get("source_tracks", ())):
        records.append({
            "source_track": source_track,
            "channel": channel,
            "source_class": track["class"],
            "source_score": float(track["score"]),
            "source_mean": float(track["mean"]),
            "track_id": track.get("track_id", ""),
            "best_detection_id": track.get("best_detection_id", ""),
            "best_box_z": track.get("best_box_z", ""),
            "member_count": track.get("member_count", len(track["per_z_boxes"])),
            "bounds_method": track.get("bounds_method", "legacy_unknown"),
            "observed_x1": track.get("observed_x1", min(b[0] for b in track["per_z_boxes"].values())),
            "observed_y1": track.get("observed_y1", min(b[1] for b in track["per_z_boxes"].values())),
            "observed_x2": track.get("observed_x2", max(b[2] for b in track["per_z_boxes"].values())),
            "observed_y2": track.get("observed_y2", max(b[3] for b in track["per_z_boxes"].values())),
            "cx": float(track["cx"]),
            "cy": float(track["cy"]),
            "cz": float(track["cz"]),
            "cx_um": float(track["cx"]) * xy_um if xy_um is not None else "",
            "cy_um": float(track["cy"]) * xy_um if xy_um is not None else "",
            "cz_um": (float(track["cz"]) - 1) * z_um if z_um is not None else "",
            "xy_um_per_px": xy_um if xy_um is not None else "",
            "z_um_per_slice": z_um if z_um is not None else "",
            "x1_3d": float(track["x1_3d"]),
            "y1_3d": float(track["y1_3d"]),
            "x2_3d": float(track["x2_3d"]),
            "y2_3d": float(track["y2_3d"]),
            "z_min": int(track["z_min"]),
            "z_max": int(track["z_max"]),
        })
    return records


def source_3d_json(soma, xy_um=None, z_um=None):
    """Keep all channels, including multiple TF tracks in one channel, in one cell row."""
    by_channel = {}
    for record in source_3d_records(soma, xy_um, z_um):
        channel = record["channel"]
        by_channel.setdefault(channel, []).append({
            key: value for key, value in record.items() if key != "channel"
        })
    return json.dumps(by_channel, separators=(",", ":"), ensure_ascii=False)


def write_source_boxes(path, soma_volumes):
    """Write each original Stage 3 per-z box, linked by coloc_id/source_track."""
    path = os.fspath(path)
    with open(path + ".part", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SOURCE_BOX_COLUMNS)
        for row_index, soma in enumerate(soma_volumes):
            coloc_id = cell_id(soma, row_index)
            for source_track, (channel, track) in enumerate(
                soma.get("source_tracks", ())
            ):
                members = track.get("member_detections")
                if members is None:
                    members = [
                        {"z": z, "x1": box[0], "y1": box[1],
                         "x2": box[2], "y2": box[3],
                         "detection_id": "", "tile_name": "", "slice_name": "",
                         "score": "", "mean": ""}
                        for z, box in sorted(track["per_z_boxes"].items())
                    ]
                for member in members:
                    writer.writerow((
                        coloc_id, source_track, channel, track.get("track_id", ""),
                        member["detection_id"], member["tile_name"], member["slice_name"],
                        member["z"], member["x1"], member["y1"], member["x2"],
                        member["y2"], member["score"], member["mean"],
                    ))

    os.replace(path + ".part", path)


def write_source_3d(path, soma_volumes, xy_um=None, z_um=None):
    """Write one 3D coordinate row for each matched source track."""
    path = os.fspath(path)
    with open(path + ".part", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_3D_COLUMNS)
        writer.writeheader()
        for row_index, soma in enumerate(soma_volumes):
            coloc_id = cell_id(soma, row_index)
            for record in source_3d_records(soma, xy_um, z_um):
                writer.writerow({"coloc_id": coloc_id, **record})

    os.replace(path + ".part", path)


MATCH_COLUMNS = (
    "coloc_id", "match_type", "channel", "track_id",
    "iou", "iomin", "iou_threshold", "iomin_threshold",
    "center_distance", "center_distance_ratio",
    "center_distance_ratio_max", "xy_margin", "z_pad",
)


def write_match_evidence(path, soma_volumes):
    """Write the metrics and configured gates for each accepted channel match."""
    path = os.fspath(path)
    with open(path + ".part", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATCH_COLUMNS)
        writer.writeheader()
        for row_index, soma in enumerate(soma_volumes):
            for evidence in soma.get("match_evidence", ()):
                writer.writerow({"coloc_id": cell_id(soma, row_index), **evidence})
    os.replace(path + ".part", path)
