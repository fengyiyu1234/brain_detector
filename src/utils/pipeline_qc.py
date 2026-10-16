"""Streaming QC for saved brain_detector outputs; never runs inference."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
import csv
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.config.loader import load_config, expand_double_exposure_channels

BOX = {"x1", "y1", "x2", "y2", "z", "score", "mean", "class"}
STAGES = ("detection", "filtering", "alignment", "geometry", "global", "tracks", "coloc", "exports")
LIMITS = {"filter_retention_min": .25, "single_layer_fraction_max": .8,
          "span_cap_fraction_max": .5, "seam_success_fraction_min": .5,
          "seam_residual_xy_px_max": 5, "seam_residual_z_max": 1,
          "tf_center_margin": .05, "soma_match_margin": .03}


def finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite number")
    return value


def integer(value):
    number = finite(value)
    if number != int(number):
        raise ValueError("non-integer value")
    return int(number)


def dump_json(path, value):
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(part, path)


def write_csv(path, rows, fields):
    part = path.with_name(path.name + ".part")
    with part.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(part, path)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mapped_path(value, mappings, base=ROOT):
    value = str(value).replace("\\", "/")
    for old in sorted(mappings, key=len, reverse=True):
        prefix = old.replace("\\", "/").rstrip("/")
        if value == prefix or value.startswith(prefix + "/"):
            value = str(mappings[old]).rstrip("/\\") + value[len(prefix):]
            break
    path = Path(value)
    return path if path.is_absolute() else base / path


def effective_mappings(options):
    return {**options.get("path_mappings", {}),
            **(options.get("windows_path_mappings", {}) if os.name == "nt" else {})}


class Reservoir:
    def __init__(self, size, seed):
        self.size, self.n, self.values = size, 0, []
        self.rng = random.Random(seed)

    def add(self, value):
        self.n += 1
        if len(self.values) < self.size:
            self.values.append(value)
        elif self.size:
            index = self.rng.randrange(self.n)
            if index < self.size:
                self.values[index] = value

    def summary(self):
        values = sorted(self.values)
        if not values:
            return {"n": self.n, "sample_n": 0}
        return {"n": self.n, "sample_n": len(values), "approximate": self.n > len(values),
                **{name: values[round((len(values) - 1) * q)]
                   for name, q in (("p05", .05), ("p50", .5), ("p95", .95))}}


class PipelineQC:
    def __init__(self, sample, options, out, db):
        self.sample, self.options, self.out, self.db = sample, options, Path(out), db
        self.maps = effective_mappings(options)
        self.result = mapped_path(sample["result_dir"], self.maps)
        runtime = mapped_path(sample.get("runtime_config", self.result / "runtime_config.json"), self.maps)
        self.config = load_config(runtime)
        self.inputs, self.findings, self.files, self.tile_rows, self.z_rows, self.review = [], [], [], [], [], []
        self.routing = [c for c in self.config["channels_routing"] if c.get("active", True)]
        self.channels = {c["id"]: c for c in self.routing}
        self.limits = {**LIMITS, **options.get("thresholds", {})}
        self.enabled = {s: options.get("checks", {}).get(s, True) for s in STAGES}
        self.deep = options.get("relational_checks", True)
        self.seed = int(options.get("seed", 2026))
        self.canvas, self.positions, self.stats = None, {}, {}
        self.note_input(runtime)
        self.db.executescript("""
            CREATE TABLE ids(scope TEXT,id TEXT,parent TEXT,tile TEXT,z REAL,n INTEGER,
                             lo REAL,hi REAL,cls TEXT,PRIMARY KEY(scope,id));
            CREATE INDEX parents ON ids(scope,parent);
            CREATE TABLE assignments(channel TEXT,track TEXT,coloc TEXT,
                                     PRIMARY KEY(channel,track,coloc));
            CREATE TABLE positives(coloc TEXT,channel TEXT,PRIMARY KEY(coloc,channel));
            CREATE TABLE source_boxes(coloc TEXT,source TEXT,channel TEXT,track TEXT,det TEXT,z REAL,
                                      PRIMARY KEY(coloc,source,det));
        """)

    def finding(self, stage, status, code, detail, count=1, path=""):
        self.findings.append(dict(stage=stage, status=status, code=code,
                                  detail=str(detail), count=count, path=str(path)))

    def note_input(self, path):
        stat = path.stat()
        self.inputs.append({"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})

    def rows(self, path, required, stage):
        path = Path(path)
        if not path.is_file():
            self.finding(stage, "FAIL", "missing_file", "Expected output is missing", path=path)
            return
        self.note_input(path)
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            missing = required - set(reader.fieldnames or [])
            if missing:
                self.finding(stage, "FAIL", "missing_columns", sorted(missing), path=path)
                return
            yield from reader

    def insert(self, scope, identity, stage, path, **v):
        if not self.deep:
            return
        if not identity:
            self.stats["empty_id:" + scope] = self.stats.get("empty_id:" + scope, 0) + 1
            return
        try:
            self.db.execute("INSERT INTO ids VALUES(?,?,?,?,?,?,?,?,?)",
                            (scope, identity, v.get("parent"), v.get("tile"), v.get("z"),
                             v.get("n"), v.get("lo"), v.get("hi"), v.get("cls")))
        except sqlite3.IntegrityError:
            self.stats["duplicate:" + scope] = self.stats.get("duplicate:" + scope, 0) + 1

    def count(self, sql, args=()):
        return self.db.execute(sql, args).fetchone()[0]

    def scan_boxes(self, path, stage, channel, tile="", global_space=False, scope=None, track=False):
        required = BOX | ({"track_id", "member_count", "z_min", "z_max"} if track else {"detection_id"})
        counts, z_counts, tile_counts, histogram = Counter(), Counter(), Counter(), Counter()
        distributions = {k: Reservoir(2048, self.seed) for k in ("score", "area", "mean")}
        candidates = Reservoir(int(self.options.get("review_per_file", 3)), self.seed)
        for r in self.rows(path, required, stage):
            counts["rows"] += 1
            try:
                x1, y1, x2, y2, score, mean = (finite(r[k]) for k in ("x1", "y1", "x2", "y2", "score", "mean"))
                z = integer(r["z"])
                if x1 >= x2 or y1 >= y2:
                    counts["invalid_box"] += 1
                if not 0 <= score <= 1:
                    counts["invalid_score"] += 1
                for key, value in (("score", score), ("mean", mean), ("area", (x2-x1)*(y2-y1))):
                    distributions[key].add(value)
                z_counts[z] += 1
                tile_counts[r.get("tile_name", tile)] += 1
                if global_space and self.canvas:
                    if x1 < 0 or y1 < 0 or x2 > self.canvas["width"] or y2 > self.canvas["height"]:
                        counts["xy_out_of_bounds"] += 1
                    if z < 1 or z > self.canvas["depth"]:
                        counts["z_out_of_bounds"] += 1
                elif stage == "detection":
                    size = self.config.get("detection_params", {}).get("tILESIZE", 2048)
                    if x1 < 0 or y1 < 0 or x2 > size or y2 > size or z < 1:
                        counts["raw_out_of_bounds"] += 1
                identity = r["track_id" if track else "detection_id"]
                values = dict(tile=r.get("tile_name", tile), z=z, cls=r["class"])
                if track:
                    n, lo, hi = (integer(r[k]) for k in ("member_count", "z_min", "z_max"))
                    values.update(n=n, lo=lo, hi=hi)
                    kind = self.channels[channel].get("type", "soma")
                    params = self.config.get("z_linker", {}).get(kind, {})
                    cap = params.get("max_cell_z_span", 5 if kind == "soma" else 3)
                    counts["single_layer"] += n == 1
                    counts["at_span_cap"] += hi-lo+1 == cap
                    histogram[n] += 1
                    if n < params.get("min_z_layers", 1) or n > hi-lo+1 or hi-lo+1 > cap or not lo <= z <= hi:
                        counts["invalid_track_extent"] += 1
                if scope:
                    self.insert(scope, identity, stage, path, **values)
                candidates.add(dict(stage=stage, channel=channel, tile=values["tile"], id=identity,
                                    z=z, x1=x1, y1=y1, x2=x2, y2=y2, path=str(path)))
            except (ValueError, TypeError, KeyError, OverflowError):
                counts["invalid_numeric"] += 1
        for key in ("invalid_box", "invalid_score", "invalid_numeric", "invalid_track_extent", "raw_out_of_bounds"):
            if counts[key]:
                self.finding(stage, "FAIL", key, channel, counts[key], path)
        for key in ("xy_out_of_bounds", "z_out_of_bounds"):
            if counts[key]:
                self.finding(stage, "WARN", key, "Review shifted edge boxes and common XML Z range", counts[key], path)
        if path.is_file() and not counts["rows"] and scope != "rejected":
            self.finding(stage, "WARN", "empty_output", "Header-only output can be valid in empty tissue", path=path)
        if track and counts["rows"]:
            for key, limit in (("single_layer", "single_layer_fraction_max"), ("at_span_cap", "span_cap_fraction_max")):
                fraction = counts[key] / counts["rows"]
                if fraction > self.limits[limit]:
                    self.finding(stage, "WARN", key, f"{channel}: {fraction:.3f}; heuristic, requires image review", counts[key], path)
        self.files.append({**dict(counts), "stage": stage, "channel": channel, "tile": tile, "path": str(path),
                           "distributions": {k: v.summary() for k, v in distributions.items()},
                           "member_count_histogram": dict(histogram)})
        self.review.extend(candidates.values)
        self.z_rows.extend(dict(stage=stage, channel=channel, tile=tile, z=z, count=n) for z, n in sorted(z_counts.items()))
        self.tile_rows.extend(dict(stage=stage, channel=channel, tile=t, rows=n) for t, n in sorted(tile_counts.items()))
        return counts

    def geometry(self):
        configured = self.sample.get("xml") or self.config.get("paths", {}).get("pATHXML")
        frame = self.config.get("stitching_reference_channel") or self.config.get("pre_align_params", {}).get("reference_channel")
        if not configured and frame in self.channels:
            directory = self.config["paths"][self.channels[frame]["dir_key"]]
            configured = str(directory).rstrip("/\\") + "/xml_merging.xml"
        if not configured or not mapped_path(configured, self.maps).is_file():
            self.finding("geometry", "NOT_CHECKED", "xml_unavailable", "Set samples[].xml; canvas checks unavailable")
            return
        path = mapped_path(configured, self.maps)
        self.note_input(path)
        root = ET.parse(path).getroot()
        stacks = list(root.find("STACKS"))
        size = self.config.get("detection_params", {}).get("tILESIZE", 2048)
        positions = [(finite(s.get("ABS_H")), finite(s.get("ABS_V")), finite(s.get("ABS_D"))) for s in stacks]
        if not positions:
            raise ValueError("XML contains no tiles")
        xmin, ymin, zmin = map(min, zip(*positions))
        xmax, ymax, zmax = map(max, zip(*positions))
        self.canvas = dict(width=xmax-xmin+size, height=ymax-ymin+size,
                           depth=integer(root.find("dimensions").get("stack_slices"))-zmax+zmin)
        if self.canvas["depth"] <= 0:
            raise ValueError("XML common Z range is empty")
        for stack, (x, y, z) in zip(stacks, positions):
            tile = stack.get("DIR_NAME").replace("\\", "/").rstrip("/").split("/")[-1]
            if tile in self.positions:
                raise ValueError("Duplicate tile basename in XML: " + tile)
            self.positions[tile] = (x-xmin, y-ymin, zmax-z)

    def tile_stages(self, tiles):
        pre = self.config.get("pipeline_mode", "post_align") == "pre_align"
        for ch in expand_double_exposure_channels(self.routing):
            cid = ch["id"]
            for tile in tiles:
                self.db.execute("DELETE FROM ids WHERE scope IN ('raw','filtered','rejected')")
                raw = self.result / "1_2d_raw" / f"{tile}_{cid}_result.csv"
                raw_n = self.scan_boxes(raw, "detection", cid, tile, scope="raw") if self.enabled["detection"] else None
                if raw_n is not None:
                    self.raw_manifest(raw, raw_n)
                if self.enabled["filtering"]:
                    if not pre and (cid not in self.channels or ch.get("double_exposure")):
                        self.finding("filtering", "NOT_CHECKED", "post_fusion_filter", cid)
                        continue
                    folder = "2_2d_filtered" if pre else "4_2d_filtered"
                    target = self.result / folder / raw.name
                    kept = self.scan_boxes(target, "filtering", cid, tile, scope="filtered")
                    reject = target.with_name(f"{tile}_{cid}_rejected.csv")
                    rejected = self.scan_boxes(reject, "filter_rejected", cid, tile, scope="rejected")
                    if raw_n is not None and raw.is_file() and target.is_file() and reject.is_file():
                        if raw_n["rows"] != kept["rows"] + rejected["rows"]:
                            self.finding("filtering", "FAIL", "count_conservation", tile + ": " + cid)
                        if self.deep:
                            unknown = self.count("""SELECT COUNT(*) FROM ids a WHERE scope IN ('filtered','rejected')
                                AND NOT EXISTS(SELECT 1 FROM ids b WHERE b.scope='raw' AND b.id=a.id)""")
                            missing = self.count("""SELECT COUNT(*) FROM ids a WHERE scope='raw' AND NOT EXISTS
                                (SELECT 1 FROM ids b WHERE b.scope IN ('filtered','rejected') AND b.id=a.id)""")
                            overlap = self.count("SELECT COUNT(*) FROM ids a JOIN ids b ON a.id=b.id WHERE a.scope='filtered' AND b.scope='rejected'")
                            if unknown + missing + overlap:
                                self.finding("filtering", "FAIL", "id_partition", f"{tile}/{cid}: unknown={unknown}, missing={missing}, overlap={overlap}")
                        fraction = kept["rows"] / raw_n["rows"] if raw_n["rows"] else None
                        if fraction is not None and fraction < self.limits["filter_retention_min"]:
                            self.finding("filtering", "WARN", "low_retention", f"{tile}/{cid}: {fraction:.3f}")
                    self.stats[f"filter_reasons:{tile}:{cid}"] = dict(Counter(r["rejection_reason"] for r in self.rows(reject, {"rejection_reason"}, "filtering")))
                self.db.commit()
                print(f"  tile QC {cid}/{tile}", flush=True)

    def raw_manifest(self, raw, counts):
        path = raw.with_name(raw.name.replace("_result.csv", "_inputs.json"))
        if not path.is_file():
            self.finding("detection", "WARN", "raw_manifest_missing", "Legacy output: input completeness is unverified", path=path)
            return
        self.note_input(path)
        manifest = load_config(path)
        if raw.is_file():
            if manifest.get("output_size") != raw.stat().st_size or manifest.get("rows_written") != counts["rows"]:
                self.finding("detection", "FAIL", "raw_manifest_mismatch", raw)
            if self.options.get("verify_manifest_hashes", False) and sha(raw) != manifest.get("output_sha256"):
                self.finding("detection", "FAIL", "raw_hash_mismatch", raw)
        statuses = Counter(image.get("status", "unknown") for image in manifest.get("images", []))
        self.stats["image_status:" + raw.stem] = dict(statuses)
        failed = statuses["missing"] + statuses["unreadable"]
        if failed:
            self.finding("detection", "FAIL", "image_read_failures", raw.stem, failed)
        if self.options.get("verify_input_files", False):
            changed = 0
            for image in manifest.get("images", []):
                if image.get("status") != "processed":
                    continue
                source = mapped_path(image["path"], self.maps)
                if not source.is_file():
                    changed += 1
                    continue
                stat = source.stat()
                changed += stat.st_size != image.get("size") or stat.st_mtime_ns != image.get("mtime_ns")
            if changed:
                self.finding("detection", "FAIL", "raw_inputs_changed", raw.stem, changed)

    def output_manifests(self):
        patterns = []
        if self.enabled["filtering"]:
            patterns.extend(("2_2d_filtered/*_filter_manifest.json", "4_2d_filtered/*_filter_manifest.json"))
        if self.enabled["tracks"]:
            patterns.append("6_3d_global/*_stage3_manifest.json")
        if self.enabled["coloc"]:
            patterns.append("7_colocalization/*manifest.json")
        checked, invalid = 0, 0
        for pattern in patterns:
            for manifest_path in sorted(self.result.glob(pattern)):
                self.note_input(manifest_path)
                manifest = load_config(manifest_path)
                for configured, stamp in manifest.get("outputs", {}).items():
                    checked += 1
                    path = mapped_path(configured, self.maps)
                    if not path.is_file() or path.stat().st_size != stamp.get("size"):
                        invalid += 1
                    elif self.options.get("verify_manifest_hashes", False) and sha(path) != stamp.get("sha256"):
                        invalid += 1
        self.stats["manifest_outputs_checked"] = checked
        if invalid:
            self.finding("provenance", "FAIL", "modified_manifest_output", "Missing, resized, or hash-mismatched output", invalid)
        if not self.options.get("verify_manifest_hashes", False):
            self.finding("provenance", "NOT_CHECKED", "hashes_disabled", "Manifest output sizes checked; content hashes require verify_manifest_hashes=true")
        if self.enabled["detection"] and not self.options.get("verify_input_files", False):
            self.finding("detection", "NOT_CHECKED", "input_stat_disabled", "Recorded input statuses checked; TIFF existence/size/mtime require verify_input_files=true")

    def stitch_conservation(self, cid, tiles):
        if not self.canvas or not self.positions:
            self.finding("global", "NOT_CHECKED", "stitch_partition_no_xml", cid)
            return
        eligible, clipped = 0, 0
        for tile in tiles:
            if tile not in self.positions:
                self.finding("global", "FAIL", "tile_absent_in_xml", tile)
                continue
            from src.core.resume import saved_channel_directory
            path = Path(saved_channel_directory(self.result, self.channels[cid], self.config)) / f"{tile}_{cid}_result.csv"
            for row in self.rows(path, BOX | {"detection_id"}, "global"):
                z = integer(row["z"]) - self.positions[tile][2]
                if len(self.positions) == 1 or 1 <= z <= self.canvas["depth"]:
                    eligible += 1
                    self.insert("eligible:" + cid, row["detection_id"], "global", path)
                else:
                    clipped += 1
        decisions = self.result / "5_2d_global" / f"{cid}_stitch_decisions.csv"
        rejected = 0
        for row in self.rows(decisions, {"detection_id", "other_detection_id", "reason"}, "global"):
            rejected += 1
            self.insert("stitch_rejected:" + cid, row["detection_id"], "global", decisions, parent=row["other_detection_id"])
        self.stats["stitch_counts:" + cid] = dict(eligible=eligible, clipped_common_z=clipped, rejected=rejected)
        if self.deep:
            scope, kept, removed = "eligible:" + cid, "global:" + cid, "stitch_rejected:" + cid
            missing = self.count("""SELECT COUNT(*) FROM ids a WHERE scope=? AND NOT EXISTS
                (SELECT 1 FROM ids b WHERE b.scope IN (?,?) AND b.id=a.id)""", (scope, kept, removed))
            unknown = self.count("""SELECT COUNT(*) FROM ids a WHERE scope IN (?,?) AND NOT EXISTS
                (SELECT 1 FROM ids b WHERE b.scope=? AND b.id=a.id)""", (kept, removed, scope))
            overlap = self.count("SELECT COUNT(*) FROM ids a JOIN ids b ON a.id=b.id WHERE a.scope=? AND b.scope=?", (kept, removed))
            winner_missing = self.count("""SELECT COUNT(*) FROM ids a WHERE scope=? AND NOT EXISTS
                (SELECT 1 FROM ids b WHERE b.scope=? AND b.id=a.parent)""", (removed, kept))
            if missing + unknown + overlap + winner_missing:
                self.finding("global", "FAIL", "stitch_id_partition", f"{cid}: missing={missing}, unknown={unknown}, overlap={overlap}, unknown_winner={winner_missing}")

    def alignment(self, tiles):
        frame = self.config.get("stitching_reference_channel") or self.config.get("pre_align_params", {}).get("reference_channel")
        for tile in tiles:
            path = self.result / "3_2d_aligned" / f"{tile}_offsets.json"
            if not path.is_file():
                self.finding("alignment", "FAIL", "missing_offsets", tile, path=path)
                continue
            self.note_input(path)
            offsets = load_config(path)
            offsets = offsets.get("offsets", offsets)
            for cid in self.channels:
                try:
                    value = offsets[cid]
                    shift = [finite(value[k]) for k in ("dx", "dy", "dz")] if isinstance(value, dict) else [finite(v) for v in value]
                    if len(shift) != 3 or (cid == frame and any(abs(v) > 1e-9 for v in shift)):
                        raise ValueError("invalid final-frame shift")
                except (ValueError, KeyError, TypeError):
                    self.finding("alignment", "FAIL", "invalid_offsets", f"{tile}/{cid}", path=path)
        self.finding("alignment", "NOT_CHECKED", "accuracy_external", "Offset integrity only; run validate_align_shifts.py or evaluate independent landmarks.")

    def seams(self, tiles):
        directory = self.result / "5_2d_global" / "tile_positions"
        if not self.config.get("tile_position_params", {}).get("enabled", False) and not directory.is_dir():
            self.finding("geometry", "NOT_CHECKED", "solver_disabled", "No detection-based geometry diagnostics")
            return
        graph = {t: set() for t in tiles}
        channels = defaultdict(Counter)
        for r in self.rows(directory / "seams.csv", {"tile", "nb_tile", "channel", "status"}, "geometry"):
            channels[r["channel"]]["total"] += 1
            if r["status"] == "ok":
                channels[r["channel"]]["ok"] += 1
                if r["tile"] in graph and r["nb_tile"] in graph:
                    graph[r["tile"]].add(r["nb_tile"])
                    graph[r["nb_tile"]].add(r["tile"])
        seen, components = set(), []
        for tile in tiles:
            if tile in seen:
                continue
            todo, component = [tile], []
            seen.add(tile)
            while todo:
                t = todo.pop()
                component.append(t)
                for neighbor in graph[t] - seen:
                    seen.add(neighbor)
                    todo.append(neighbor)
            components.append(sorted(component))
        self.stats["seam_components"], self.stats["seam_channels"] = components, dict(channels)
        if len(components) > 1:
            self.finding("geometry", "WARN", "disconnected_seams", f"{len(components)} components; nominal priors constrain disconnected tiles")
        for cid, count in channels.items():
            if count["ok"] / count["total"] < self.limits["seam_success_fraction_min"]:
                self.finding("geometry", "WARN", "low_seam_success", f"{cid}: {count['ok']}/{count['total']}")
        path = directory / "solution.json"
        if path.is_file():
            self.note_input(path)
            solution = load_config(path)
            # Legacy solver JSON can contain NaN. Preserve it as null and flag it below.
            diagnostic = {k: solution.get(k) for k in ("tiles_without_seams", "seam_fit_resid_p50")}
            diagnostic["holdout_compare_available"] = bool(solution.get("holdout_compare"))
            for axis, value in diagnostic.get("seam_fit_resid_p50", {}).items():
                limit = self.limits["seam_residual_z_max" if axis == "z" else "seam_residual_xy_px_max"]
                if not math.isfinite(float(value)) or value > limit:
                    self.finding("geometry", "WARN", "large_fit_residual", f"{axis}: {value}; training-data fit only")
                if not math.isfinite(float(value)):
                    diagnostic["seam_fit_resid_p50"][axis] = None
            self.stats["solver_diagnostics"] = diagnostic
        else:
            self.finding("geometry", "FAIL", "missing_solution", path)
        self.finding("geometry", "NOT_CHECKED", "held_out_seams", "No new held-out seam measurement; use independent overlap landmarks/ROIs.")

    def global_and_tracks(self, tiles):
        for cid in self.channels:
            if self.enabled["global"]:
                path = self.result / "5_2d_global" / f"{cid}_2d_global.csv"
                self.scan_boxes(path, "global", cid, global_space=True, scope="global:" + cid)
                self.stitch_conservation(cid, tiles)
                self.stats["stitch_rejections:" + cid] = dict(Counter(r["reason"] for r in self.rows(path.with_name(f"{cid}_stitch_decisions.csv"), {"reason"}, "global")))
            if not self.enabled["tracks"]:
                continue
            directory = self.result / "6_3d_global"
            self.scan_boxes(directory / f"{cid}_3d_tracked.csv", "tracks", cid, global_space=True, scope="tracks:" + cid, track=True)
            for r in self.rows(directory / f"{cid}_track_members.csv", {"track_id", "detection_id", "z"}, "tracks"):
                try:
                    self.insert("members:" + cid, r["detection_id"], "tracks", directory, parent=r["track_id"], z=integer(r["z"]))
                except ValueError:
                    self.finding("tracks", "FAIL", "invalid_member_z", cid)
            for r in self.rows(directory / f"{cid}_track_rejections.csv", {"member_detection_ids"}, "tracks"):
                ids = json.loads(r["member_detection_ids"])
                if not isinstance(ids, list):
                    raise ValueError("member_detection_ids must be a JSON array")
                for identity in ids:
                    self.insert("rejected_members:" + cid, identity, "tracks", directory)
            if self.deep:
                tracks, members, global_scope = "tracks:" + cid, "members:" + cid, "global:" + cid
                bad = self.count("""SELECT COUNT(*) FROM ids m WHERE scope=? AND NOT EXISTS
                    (SELECT 1 FROM ids t WHERE t.scope=? AND t.id=m.parent)""", (members, tracks))
                mismatch = self.count("""SELECT COUNT(*) FROM ids t LEFT JOIN
                    (SELECT parent,COUNT(*) n,COUNT(DISTINCT z) nz,MIN(z) lo,MAX(z) hi FROM ids WHERE scope=? GROUP BY parent) m
                    ON t.id=m.parent WHERE t.scope=? AND (m.n IS NULL OR t.n!=m.n OR m.n!=m.nz OR t.lo!=m.lo OR t.hi!=m.hi)""", (members, tracks))
                kind = self.channels[cid].get("type", "soma")
                gap = self.config.get("z_linker", {}).get(kind, {}).get("max_z_gap", 0)
                bad_gaps = self.count("""SELECT COUNT(*) FROM
                    (SELECT z-LAG(z) OVER(PARTITION BY parent ORDER BY z)-1 gap FROM ids WHERE scope=?) WHERE gap>?""", (members,gap))
                if bad_gaps:
                    self.finding("tracks", "FAIL", "member_z_gap", cid, bad_gaps)
                if bad + mismatch:
                    self.finding("tracks", "FAIL", "track_member_integrity", f"{cid}: unknown_tracks={bad}, extent/count_mismatch={mismatch}")
                if self.enabled["global"]:
                    rejects = "rejected_members:" + cid
                    unknown = self.count("""SELECT COUNT(*) FROM ids m WHERE scope IN (?,?) AND NOT EXISTS
                        (SELECT 1 FROM ids g WHERE g.scope=? AND g.id=m.id)""", (members, rejects, global_scope))
                    missing = self.count("""SELECT COUNT(*) FROM ids g WHERE scope=? AND NOT EXISTS
                        (SELECT 1 FROM ids m WHERE m.scope IN (?,?) AND m.id=g.id)""", (global_scope, members, rejects))
                    overlap = self.count("SELECT COUNT(*) FROM ids a JOIN ids b ON a.id=b.id WHERE a.scope=? AND b.scope=?", (members, rejects))
                    if unknown + missing + overlap:
                        self.finding("tracks", "FAIL", "detection_partition", f"{cid}: unknown={unknown}, unaccounted={missing}, accepted_and_rejected={overlap}")
            self.db.commit()
            print(f"  global/track QC {cid}", flush=True)

    def colocalization(self):
        directory = self.result / "7_colocalization"
        counts, classes = Counter(), Counter()
        required = BOX | {"coloc_id", "soma_positive_channels", "tf_positive_channels", "soma_positive_count", "tf_positive_count"}
        candidates = Reservoir(20, self.seed)
        for r in self.rows(directory / "coloc_result.csv", required, "coloc"):
            counts["rows"] += 1
            classes[r["class"]] += 1
            try:
                values = {k: finite(r[k]) for k in ("x1", "y1", "x2", "y2", "z", "score", "mean")}
                if values["x1"] >= values["x2"] or values["y1"] >= values["y2"]:
                    raise ValueError("invalid box")
                integer(r["z"])
                if not 0 <= values["score"] <= 1:
                    raise ValueError("invalid score")
                if self.canvas and (values["z"] < 1 or values["z"] > self.canvas["depth"]):
                    counts["z_out_of_bounds"] += 1
                self.insert("coloc", r["coloc_id"], "coloc", directory, cls=r["class"])
                for kind in ("soma", "tf"):
                    positive = json.loads(r[kind + "_positive_channels"])
                    if not isinstance(positive, list) or len(set(positive)) != len(positive) or len(positive) != integer(r[kind + "_positive_count"]):
                        raise ValueError("invalid channel positivity")
                    for cid in positive:
                        if cid not in self.channels or self.channels[cid].get("type", "soma") != kind:
                            raise ValueError("unexpected positive channel")
                        if self.deep:
                            self.db.execute("INSERT OR IGNORE INTO positives VALUES(?,?)", (r["coloc_id"], cid))
                    counts[kind + "_positive_cells"] += bool(positive)
                    counts[kind + "_multi_positive"] += len(positive) > 1
                candidates.add(dict(stage="coloc", id=r["coloc_id"], channel="", tile=r.get("tile_name", ""),
                                    **{k: r[k] for k in ("z", "x1", "y1", "x2", "y2")}, path=str(directory / "coloc_result.csv")))
            except (ValueError, TypeError, KeyError):
                counts["invalid_rows"] += 1
        self.review.extend(candidates.values)
        if counts["invalid_rows"]:
            self.finding("coloc", "FAIL", "invalid_coloc_rows", "Numeric or positivity errors", counts["invalid_rows"])
        if counts["z_out_of_bounds"]:
            self.finding("coloc", "WARN", "z_out_of_bounds", "Review display-box centers", counts["z_out_of_bounds"])
        self.stats["coloc"], self.stats["class_counts"] = dict(counts), dict(classes)
        for r in self.rows(directory / "coloc_source_3d.csv", {"coloc_id", "channel", "track_id", "source_track", "member_count"}, "coloc"):
            if self.deep:
                try:
                    self.db.execute("INSERT INTO assignments VALUES(?,?,?)", (r["channel"], r["track_id"], r["coloc_id"]))
                except sqlite3.IntegrityError:
                    counts["duplicate_sources"] += 1
                self.insert("source:" + r["coloc_id"], r["source_track"], "coloc", directory,
                            parent=r["track_id"], n=integer(r["member_count"]), cls=r["channel"])
        if self.deep:
            orphan = self.count("SELECT COUNT(*) FROM assignments a WHERE NOT EXISTS(SELECT 1 FROM ids c WHERE c.scope='coloc' AND c.id=a.coloc)")
            unknown = self.count("SELECT COUNT(*) FROM assignments a WHERE NOT EXISTS(SELECT 1 FROM ids t WHERE t.scope='tracks:'||a.channel AND t.id=a.track)") if self.enabled["tracks"] else 0
            missing = self.count("SELECT COUNT(*) FROM positives p WHERE NOT EXISTS(SELECT 1 FROM assignments a WHERE a.coloc=p.coloc AND a.channel=p.channel)")
            extra = self.count("SELECT COUNT(*) FROM assignments a WHERE NOT EXISTS(SELECT 1 FROM positives p WHERE p.coloc=a.coloc AND p.channel=a.channel)")
            if orphan + unknown + missing + extra + counts["duplicate_sources"]:
                self.finding("coloc", "FAIL", "source_integrity", f"orphan={orphan}, unknown_tracks={unknown}, positivity_mismatch={missing+extra}, duplicate={counts['duplicate_sources']}")
            reused = self.count("SELECT COUNT(*) FROM (SELECT channel,track FROM assignments GROUP BY channel,track HAVING COUNT(*)>1)")
            if reused:
                self.finding("coloc", "WARN", "reused_source_track", "Review tracks assigned to multiple soma; TF assignment may permit this", reused)
        borderline, invalid = Counter(), 0
        for r in self.rows(directory / "coloc_match_evidence.csv", {"match_type", "coloc_id", "channel"}, "coloc"):
            try:
                if r["match_type"] == "soma_iou":
                    marginal = max(finite(r["iou"])-finite(r["iou_threshold"]), finite(r["iomin"])-finite(r["iomin_threshold"])) < self.limits["soma_match_margin"]
                else:
                    marginal = finite(r["center_distance_ratio_max"])-finite(r["center_distance_ratio"]) < self.limits["tf_center_margin"]
                borderline[r["channel"]] += marginal
                if marginal and sum(borderline.values()) <= 100:
                    self.review.append(dict(stage="coloc_borderline", channel=r["channel"], id=r["coloc_id"], path=str(directory / "coloc_match_evidence.csv")))
            except (ValueError, KeyError, TypeError):
                invalid += 1
        if invalid:
            self.finding("coloc", "FAIL", "invalid_match_evidence", "Missing/nonfinite evidence", invalid)
        if self.deep:
            box_errors = 0
            for r in self.rows(directory / "coloc_source_boxes.csv",
                               {"coloc_id", "source_track", "channel", "track_id", "detection_id", "z"}, "coloc"):
                try:
                    self.db.execute("INSERT INTO source_boxes VALUES(?,?,?,?,?,?)",
                                    (r["coloc_id"],r["source_track"],r["channel"],r["track_id"],r["detection_id"],integer(r["z"])))
                except (sqlite3.IntegrityError, ValueError):
                    box_errors += 1
            orphan_boxes = self.count("""SELECT COUNT(*) FROM source_boxes b WHERE NOT EXISTS
                (SELECT 1 FROM ids s WHERE s.scope='source:'||b.coloc AND s.id=b.source
                 AND s.parent=b.track AND s.cls=b.channel)""")
            mismatched = self.count("""SELECT COUNT(*) FROM ids s LEFT JOIN
                (SELECT 'source:'||coloc scope,source,COUNT(*) n FROM source_boxes GROUP BY coloc,source) b
                ON s.scope=b.scope AND s.id=b.source
                WHERE s.scope LIKE 'source:%' AND (b.n IS NULL OR s.n!=b.n)""")
            unknown_members = self.count("""SELECT COUNT(*) FROM source_boxes b WHERE NOT EXISTS
                (SELECT 1 FROM ids m WHERE m.scope='members:'||b.channel AND m.id=b.det AND m.parent=b.track AND m.z=b.z)""") if self.enabled["tracks"] else 0
            if box_errors + orphan_boxes + mismatched + unknown_members:
                self.finding("coloc", "FAIL", "source_box_integrity",
                             f"duplicate/invalid={box_errors}, orphan={orphan_boxes}, count_mismatch={mismatched}, unknown_member={unknown_members}")
        self.stats["borderline_matches"] = dict(borderline)
        self.stats["coloc_decisions"] = dict(Counter(r["reason"] for r in self.rows(directory / "coloc_decisions.csv", {"reason"}, "coloc")))
        self.db.commit()

    def exports(self):
        if not self.enabled["coloc"]:
            self.finding("exports", "NOT_CHECKED", "coloc_disabled", "Enable coloc to compare centroid exports")
            return
        directory = self.result / "7_colocalization" / "cell_centroids"
        for cls, expected in self.stats.get("class_counts", {}).items():
            safe = cls.replace("/", "_").replace("\\", "_")
            path = directory / f"ob_{safe}.csv"
            n, invalid = 0, 0
            for r in self.rows(path, {"coloc_id", "cx", "cy", "z", "cx_um", "cy_um", "cz_um"}, "exports"):
                n += 1
                try:
                    x, y, z, xu, yu, zu = (finite(r[k]) for k in ("cx", "cy", "z", "cx_um", "cy_um", "cz_um"))
                    dp = self.config.get("detection_params", {})
                    xy, dz = dp.get("xy_resolution_um", .65), dp.get("z_resolution_um", 8)
                    if not all(math.isclose(a, b, abs_tol=1e-6) for a, b in ((xu,x*xy), (yu,y*xy), (zu,(z-1)*dz))):
                        invalid += 1
                    self.insert("exports", r["coloc_id"], "exports", path, cls=cls)
                except (ValueError, TypeError):
                    invalid += 1
            if n != expected or invalid:
                self.finding("exports", "FAIL", "centroid_count_or_units", f"{cls}: rows={n}, expected={expected}, invalid={invalid}", path=path)
        if self.deep:
            mismatch = self.count("""SELECT COUNT(*) FROM ids c LEFT JOIN ids e ON e.scope='exports' AND e.id=c.id
                WHERE c.scope='coloc' AND (e.id IS NULL OR e.cls!=c.cls)""")
            unknown = self.count("SELECT COUNT(*) FROM ids e WHERE scope='exports' AND NOT EXISTS(SELECT 1 FROM ids c WHERE c.scope='coloc' AND c.id=e.id)")
            if mismatch + unknown:
                self.finding("exports", "FAIL", "centroid_id_partition", f"missing/wrong_class={mismatch}, unknown={unknown}")

    def plots(self):
        # Code-native SVG artifacts require no plotting dependencies.
        columns = sorted({(r["stage"],r["channel"]) for r in self.tile_rows})
        tiles = sorted({r["tile"] for r in self.tile_rows if r["tile"]})
        if not columns or not tiles:
            return
        values = Counter()
        for row in self.tile_rows:
            values[(row["tile"],row["stage"],row["channel"])] += row["rows"]
        maxima = {key:max((values[(t,*key)] for t in tiles),default=0) for key in columns}
        width,height = 200+150*len(columns),160+24*len(tiles)
        svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
               '<rect width="100%" height="100%" fill="white"/>',
               '<g font-family="sans-serif" font-size="11">',
               '<text x="10" y="20">Tile counts: color scales independently per column (log count).</text>']
        for col,(stage,channel) in enumerate(columns):
            x=200+150*col
            svg.append(f'<text x="{x}" y="70">{escape(stage)}</text>')
            svg.append(f'<text x="{x}" y="90">{escape(channel)}</text>')
        for index,tile in enumerate(tiles):
            y=120+24*index
            svg.append(f'<text x="10" y="{y+15}">{escape(tile)}</text>')
            for col,key in enumerate(columns):
                count=values[(tile,*key)]
                level=math.log1p(count)/math.log1p(maxima[key]) if maxima[key] else 0
                color=f"rgb({round(245-170*level)},{round(250-110*level)},255)"
                x=200+150*col
                svg.append(f'<rect x="{x}" y="{y}" width="145" height="22" fill="{color}"/>')
                svg.append(f'<text x="{x+5}" y="{y+15}">{count:,}</text>')
        svg.extend(["</g>","</svg>"])
        (self.out/"tile_counts.svg").write_text("\n".join(svg),encoding="utf-8")

    def run(self):
        self.geometry()
        tiles = self.sample.get("tiles") or sorted(self.positions)
        if not tiles:
            names = set()
            for ch in expand_double_exposure_channels(self.routing):
                suffix = f"_{ch['id']}_result.csv"
                names.update(p.name[:-len(suffix)] for p in (self.result / "1_2d_raw").glob("*" + suffix))
            tiles = sorted(names)
            self.finding("detection", "NOT_CHECKED", "tile_inventory_inferred", "Set tiles or XML to detect tiles absent from every raw channel")
        if not tiles:
            self.finding("detection", "FAIL", "no_tiles", self.result)
        if self.enabled["detection"] or self.enabled["filtering"]:
            self.tile_stages(tiles)
        if self.enabled["alignment"] and self.config.get("pipeline_mode") == "pre_align":
            self.alignment(tiles)
        if self.enabled["geometry"]:
            self.seams(tiles)
        self.global_and_tracks(tiles)
        if self.enabled["coloc"]:
            self.colocalization()
        if self.enabled["exports"]:
            self.exports()
        self.output_manifests()
        for key, n in self.stats.items():
            if (key.startswith("duplicate:") or key.startswith("empty_id:")) and n:
                self.finding("integrity", "FAIL", key.split(":")[0], key.split(":", 1)[1], n)
        for stage, enabled in self.enabled.items():
            if not enabled:
                self.finding(stage, "NOT_CHECKED", "disabled", "Disabled by QC configuration")
        if not self.enabled["alignment"] or self.config.get("pipeline_mode") != "pre_align":
            self.finding("alignment", "NOT_CHECKED", "offsets_not_applicable", "No pre_align offset check was run")
        if not self.deep:
            self.finding("integrity", "NOT_CHECKED", "relations_disabled", "ID uniqueness and cross-file relations were not checked")
        self.finding("accuracy", "NOT_CHECKED", "no_ground_truth", "Integrity checks do not establish detection/colocalization accuracy")
        status = "FAIL" if any(f["status"]=="FAIL" for f in self.findings) else "WARN" if any(f["status"]=="WARN" for f in self.findings) else "PASS"
        report = dict(schema_version=1, sample_id=self.sample["id"], status=status,
                      status_scope="Enabled saved-output integrity checks only; see NOT_CHECKED",
                      created_utc=datetime.now(timezone.utc).isoformat(), tiles=tiles, canvas=self.canvas,
                      frame=self.config.get("stitching_reference_channel"), thresholds=self.limits,
                      options=self.options, sample=self.sample, channels=self.routing,
                      findings=self.findings, files=self.files, statistics=self.stats,
                      inputs=self.inputs, qc_code_sha256=sha(Path(__file__)))
        self.out.mkdir(parents=True, exist_ok=True)
        dump_json(self.out / "qc_report.json", report)
        write_csv(self.out / "findings.csv", self.findings, ["stage","status","code","count","detail","path"])
        write_csv(self.out / "tile_metrics.csv", self.tile_rows, ["stage","channel","tile","rows"])
        write_csv(self.out / "z_counts.csv", self.z_rows, ["stage","channel","tile","z","count"])
        write_csv(self.out / "review_candidates.csv", self.review, ["stage","channel","tile","id","z","x1","y1","x2","y2","path"])
        if self.options.get("plots", True):
            self.plots()
        lines = [f"# QC: {self.sample['id']}", "", f"Status: **{status}** (enabled integrity checks only)", "",
                 "Quantiles use deterministic reservoirs (up to 2048 values/file); counts are exact.",
                 "Review candidates are diagnostic samples, not a representative GT test set.", "",
                 "| Stage | Status | Check | Count | Detail |", "|---|---|---|---:|---|"]
        lines.extend(f"| {f['stage']} | {f['status']} | {f['code']} | {f['count']} | {f['detail'].replace('|','/')} |" for f in self.findings)
        (self.out / "qc_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="QC JSON/JSONC; paths relative to project root")
    parser.add_argument("--samples", nargs="+", help="Override enabled sample IDs")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    samples = config["samples"]
    ids = [s["id"] for s in samples]
    if len(ids) != len(set(ids)):
        parser.error("Duplicate sample IDs")
    if args.samples and set(args.samples)-set(ids):
        parser.error("Unknown sample IDs: " + str(sorted(set(args.samples)-set(ids))))
    selected = [s for s in samples if s["id"] in args.samples] if args.samples else [s for s in samples if s.get("enabled", True)]
    if not selected:
        parser.error("No samples selected")
    failed = False
    for sample in selected:
        options = {k:v for k,v in config.items() if k != "samples"}
        for key, value in sample.get("options", {}).items():
            options[key] = {**options.get(key, {}), **value} if isinstance(value, dict) else value
        result = mapped_path(sample["result_dir"], effective_mappings(options))
        out = mapped_path(sample["output_dir"], effective_mappings(options)) if sample.get("output_dir") else result / "pipeline_qc"
        print(f"[{sample['id']}] QC starting: {result}", flush=True)
        out.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(prefix="brain_qc_", dir=options.get("scratch_dir")) as scratch:
                with closing(sqlite3.connect(str(Path(scratch) / "relations.sqlite"))) as db:
                    db.execute("PRAGMA journal_mode=OFF")
                    db.execute("PRAGMA synchronous=OFF")
                    db.execute("PRAGMA cache_size=-32768")
                    report = PipelineQC(sample, options, out, db).run()
            error_path = out / "qc_error.json"
            if error_path.is_file():
                error_path.unlink()
            failed |= report["status"] == "FAIL"
            print(f"[{sample['id']}] {report['status']}: {out / 'qc_report.md'}", flush=True)
        except (OSError, ValueError, KeyError, TypeError, csv.Error, ET.ParseError, sqlite3.Error) as exc:
            failed = True
            error = {"sample_id":sample["id"], "status":"FAIL", "error":str(exc),
                     "created_utc":datetime.now(timezone.utc).isoformat()}
            dump_json(out / "qc_error.json", error)
            dump_json(out / "qc_report.json", error)
            (out / "qc_report.md").write_text("QC failed: " + str(exc) + "\n", encoding="utf-8")
            print(f"[{sample['id']}] FAIL: {exc}", file=sys.stderr, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

