"""Export the exact Stage 3 channel tracks assigned to Stage 4 cells."""

import csv
import json


SOURCE_3D_COLUMNS = (
    "coloc_id", "source_track", "channel", "source_class",
    "source_score", "source_mean", "cx", "cy", "cz",
    "x1_3d", "y1_3d", "x2_3d", "y2_3d", "z_min", "z_max",
)
SOURCE_BOX_COLUMNS = (
    "coloc_id", "source_track", "channel", "z", "x1", "y1", "x2", "y2",
)


def source_3d_records(soma):
    """Return every matched source track in the global aligned coordinate frame."""
    records = []
    for source_track, (channel, track) in enumerate(soma.get("source_tracks", ())):
        records.append({
            "source_track": source_track,
            "channel": channel,
            "source_class": track["class"],
            "source_score": float(track["score"]),
            "source_mean": float(track["mean"]),
            "cx": float(track["cx"]),
            "cy": float(track["cy"]),
            "cz": float(track["cz"]),
            "x1_3d": float(track["x1_3d"]),
            "y1_3d": float(track["y1_3d"]),
            "x2_3d": float(track["x2_3d"]),
            "y2_3d": float(track["y2_3d"]),
            "z_min": int(track["z_min"]),
            "z_max": int(track["z_max"]),
        })
    return records


def source_3d_json(soma):
    """Keep all channels, including multiple TF tracks in one channel, in one cell row."""
    by_channel = {}
    for record in source_3d_records(soma):
        channel = record["channel"]
        by_channel.setdefault(channel, []).append({
            key: value for key, value in record.items() if key != "channel"
        })
    return json.dumps(by_channel, separators=(",", ":"), ensure_ascii=False)


def write_source_boxes(path, soma_volumes):
    """Write each original Stage 3 per-z box, linked by coloc_id/source_track."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SOURCE_BOX_COLUMNS)
        for coloc_id, soma in enumerate(soma_volumes):
            for source_track, (channel, track) in enumerate(
                soma.get("source_tracks", ())
            ):
                for z, box in sorted(track["per_z_boxes"].items()):
                    writer.writerow((
                        coloc_id, source_track, channel, int(z), *map(float, box),
                    ))


def write_source_3d(path, soma_volumes):
    """Write one 3D coordinate row for each matched source track."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_3D_COLUMNS)
        writer.writeheader()
        for coloc_id, soma in enumerate(soma_volumes):
            for record in source_3d_records(soma):
                writer.writerow({"coloc_id": coloc_id, **record})
