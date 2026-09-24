"""Authoritative coordinate metadata for pipeline-result visualizations.

Stage 3/4 results are always expressed in the final frame XML named by the
run's ``paths.pATHXML``. This module deliberately does not guess geometry
from channel order, image directories, or a regular tile grid.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import xml.etree.ElementTree as ET


class CoordinateContextError(RuntimeError):
    """Saved global results cannot be mapped unambiguously."""


@dataclass(frozen=True)
class TilePosition:
    x: int
    y: int
    z: int

    def __iter__(self):
        return iter((self.x, self.y, self.z))


def _tile_name(stack: ET.Element) -> str:
    name = os.path.basename((stack.get("DIR_NAME") or "").replace("\\", "/"))
    if not name:
        raise CoordinateContextError("XML Stack has an empty DIR_NAME")
    return name


def _read_xml_positions(xml_path: str) -> dict[str, tuple[int, int, int]]:
    try:
        stacks = list(ET.parse(xml_path).getroot().find("STACKS") or [])
    except (ET.ParseError, OSError, AttributeError) as exc:
        raise CoordinateContextError(f"Cannot parse merging XML '{xml_path}': {exc}") from exc
    if not stacks:
        raise CoordinateContextError(f"Merging XML has no STACKS: '{xml_path}'")
    positions = {}
    for stack in stacks:
        name = _tile_name(stack)
        if name in positions:
            raise CoordinateContextError(f"Duplicate DIR_NAME basename '{name}' in XML '{xml_path}'")
        try:
            positions[name] = tuple(int(stack.get(k)) for k in ("ABS_H", "ABS_V", "ABS_D"))
        except (TypeError, ValueError) as exc:
            raise CoordinateContextError(
                f"Invalid ABS_H/ABS_V/ABS_D for tile '{name}' in '{xml_path}'") from exc
    return positions


@dataclass
class CoordinateContext:
    """Runtime-derived mapping between global and final-frame tile coordinates."""

    result_dir: str
    runtime_path: str
    runtime: dict
    paths: dict
    routing: list[dict]
    pipeline_mode: str
    frame_channel: str
    frame_xml: str
    xml_dir: str
    tile_size: int
    frame_positions: dict[str, TilePosition]
    channel_positions: dict[str, dict[str, TilePosition]]


    @classmethod
    def from_vis_config(cls, config: dict, config_path: str = "vis_config.json") -> "CoordinateContext":
        """Build visualization coordinates solely from the selected vis sample."""
        paths = config.get("paths") or {}
        result_dir = paths.get("pATHRESULT")
        if not result_dir:
            raise CoordinateContextError("vis_config must specify paths.pATHRESULT")
        result_dir = os.path.abspath(result_dir)
        routing = [dict(ch) for ch in config.get("channels_routing", []) if ch.get("active", True)]
        if not routing:
            raise CoordinateContextError("vis_config has no active channels_routing")

        frame_channel = config.get("frame_channel") or routing[0]["id"]
        if frame_channel not in {ch["id"] for ch in routing}:
            raise CoordinateContextError(f"vis_config frame_channel '{frame_channel}' is not active")
        frame_xml = paths.get("pATHXML")
        if not frame_xml:
            channel_dir = paths.get(next(ch["dir_key"] for ch in routing if ch["id"] == frame_channel))
            if not channel_dir:
                raise CoordinateContextError(f"vis_config has no image directory for '{frame_channel}'")
            frame_xml = next((candidate for candidate in (
                os.path.join(channel_dir, "xml_merging.xml"),
                os.path.join(channel_dir, "xml_import.xml"),
            ) if os.path.isfile(candidate)), None)
        if not frame_xml:
            raise CoordinateContextError("vis_config needs paths.pATHXML or a frame-channel XML")
        frame_xml = os.path.abspath(frame_xml)
        if not os.path.isfile(frame_xml):
            raise CoordinateContextError(f"Visualization frame XML does not exist: '{frame_xml}'")
        named_channel = re.fullmatch(r"xml_merging_(.+)\.xml", os.path.basename(frame_xml), re.IGNORECASE)
        if named_channel and named_channel.group(1) != frame_channel:
            raise CoordinateContextError(
                f"vis_config frame_channel '{frame_channel}' disagrees with '{frame_xml}'")

        raw_frame = _read_xml_positions(frame_xml)
        x_min = min(p[0] for p in raw_frame.values())
        y_min = min(p[1] for p in raw_frame.values())
        z_start = max(p[2] for p in raw_frame.values())

        def normalize(raw: dict[str, tuple[int, int, int]], label: str) -> dict[str, TilePosition]:
            missing = set(raw_frame) - set(raw)
            extra = set(raw) - set(raw_frame)
            if missing or extra:
                raise CoordinateContextError(
                    f"Tile mapping differs between frame XML and {label}: "
                    f"missing={sorted(missing)}, extra={sorted(extra)}")
            return {name: TilePosition(p[0] - x_min, p[1] - y_min, z_start - p[2])
                    for name, p in raw.items()}

        xml_dir = os.path.dirname(frame_xml)
        per_channel = {ch["id"]: os.path.join(xml_dir, f"xml_merging_{ch['id']}.xml")
                       for ch in routing}
        available = {ch: path for ch, path in per_channel.items() if os.path.isfile(path)}
        if available and len(available) != len(per_channel):
            missing = sorted(set(per_channel) - set(available))
            raise CoordinateContextError(f"Missing per-channel XML beside '{frame_xml}': {missing}")
        if available:
            channel_positions = {ch: normalize(_read_xml_positions(path), path)
                                 for ch, path in per_channel.items()}
        else:
            # A shared TeraStitcher XML plus recorded alignment offsets can
            # describe prealignment views before per-channel XMLs are solved.
            channel_positions = {ch["id"]: normalize(raw_frame, frame_xml) for ch in routing}

        return cls(
            result_dir=result_dir, runtime_path=os.path.abspath(config_path), runtime=config,
            paths=dict(paths), routing=routing,
            pipeline_mode=str(config.get("pipeline_mode", "pre_align")),
            frame_channel=frame_channel, frame_xml=frame_xml, xml_dir=xml_dir,
            tile_size=int((config.get("detection_params") or {}).get("tILESIZE", 2048)),
            frame_positions=normalize(raw_frame, frame_xml), channel_positions=channel_positions,
        )

    @classmethod
    def from_result_dir(cls, result_dir: str) -> "CoordinateContext":
        result_dir = os.path.abspath(result_dir)
        runtime_path = os.path.join(result_dir, "runtime_config.json")
        if not os.path.isfile(runtime_path):
            raise CoordinateContextError(f"Missing runtime_config.json: '{runtime_path}'")
        try:
            with open(runtime_path, encoding="utf-8") as handle:
                runtime = json.load(handle)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise CoordinateContextError(f"Cannot read runtime config '{runtime_path}': {exc}") from exc

        paths = runtime.get("paths") or {}
        frame_xml = paths.get("pATHXML")
        if not frame_xml:
            raise CoordinateContextError(
                f"runtime config '{runtime_path}' has no paths.pATHXML; global Stage 3/4 results cannot be displayed")
        frame_xml = os.path.abspath(frame_xml)
        if not os.path.isfile(frame_xml):
            raise CoordinateContextError(
                f"Runtime frame XML does not exist: '{frame_xml}' (from '{runtime_path}')")

        match = re.fullmatch(r"xml_merging_(.+)\.xml", os.path.basename(frame_xml), re.IGNORECASE)
        if not match:
            raise CoordinateContextError(
                f"Runtime frame XML must be named xml_merging_<channel>.xml, got '{frame_xml}'")
        frame_channel = match.group(1)
        routing = [dict(ch) for ch in runtime.get("channels_routing", []) if ch.get("active", True)]
        if not routing:
            raise CoordinateContextError(f"No active channels in runtime config '{runtime_path}'")
        routed_ids = {ch.get("id") for ch in routing}
        if frame_channel not in routed_ids:
            raise CoordinateContextError(
                f"Frame channel '{frame_channel}' from '{frame_xml}' is not an active runtime channel")

        raw_frame = _read_xml_positions(frame_xml)
        x_min = min(p[0] for p in raw_frame.values())
        y_min = min(p[1] for p in raw_frame.values())
        z_start = max(p[2] for p in raw_frame.values())

        def normalize(raw: dict[str, tuple[int, int, int]], label: str) -> dict[str, TilePosition]:
            missing = set(raw_frame) - set(raw)
            extra = set(raw) - set(raw_frame)
            if missing or extra:
                raise CoordinateContextError(
                    f"Tile mapping differs between frame XML and {label}: "
                    f"missing={sorted(missing)}, extra={sorted(extra)}")
            # Solver XML files share their ABS origin. Never independently normalize
            # channels before computing P_c - P_frame.
            return {name: TilePosition(p[0] - x_min, p[1] - y_min, z_start - p[2])
                    for name, p in raw.items()}

        xml_dir = os.path.dirname(frame_xml)
        channel_positions = {frame_channel: normalize(raw_frame, "frame XML")}
        for ch in routing:
            channel = ch["id"]
            xml_path = os.path.join(xml_dir, f"xml_merging_{channel}.xml")
            if not os.path.isfile(xml_path):
                raise CoordinateContextError(
                    f"Missing per-channel XML for '{channel}': '{xml_path}'. "
                    f"It must be beside runtime frame XML '{frame_xml}'.")
            channel_positions[channel] = normalize(_read_xml_positions(xml_path), xml_path)

        return cls(
            result_dir=result_dir, runtime_path=runtime_path, runtime=runtime,
            paths=dict(paths), routing=routing,
            pipeline_mode=str(runtime.get("pipeline_mode", "")),
            frame_channel=frame_channel, frame_xml=frame_xml, xml_dir=xml_dir,
            tile_size=int((runtime.get("detection_params") or {}).get("tILESIZE", 2048)),
            frame_positions=channel_positions[frame_channel], channel_positions=channel_positions,
        )

    def offsets_for_tile(self, tile_name: str) -> dict:
        path = os.path.join(self.result_dir, "0_channel_alignment", f"{tile_name}_offsets.json")
        if not os.path.isfile(path):
            if self.pipeline_mode == "pre_align":
                raise CoordinateContextError(f"Missing pre_align offsets for tile '{tile_name}': '{path}'")
            return {}
        try:
            with open(path, encoding="utf-8") as handle:
                offsets = json.load(handle)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise CoordinateContextError(f"Cannot read offsets '{path}': {exc}") from exc
        for ch in self.routing:
            channel = ch["id"]
            value = offsets.get(channel)
            if not isinstance(value, dict) or not all(k in value for k in ("dx", "dy", "dz")):
                raise CoordinateContextError(f"Offsets '{path}' lack dx/dy/dz for active channel '{channel}'")
        frame_shift = offsets[self.frame_channel]
        if tuple(int(frame_shift[k]) for k in ("dx", "dy", "dz")) != (0, 0, 0):
            raise CoordinateContextError(
                f"Frame channel '{self.frame_channel}' must have shift (0,0,0) in '{path}', got {frame_shift}")
        return offsets

    def position(self, tile_name: str, channel: str | None = None) -> TilePosition:
        positions = self.frame_positions if channel is None else self.channel_positions.get(channel)
        if positions is None or tile_name not in positions:
            origin = self.frame_xml if channel is None else os.path.join(self.xml_dir, f"xml_merging_{channel}.xml")
            raise CoordinateContextError(f"Tile '{tile_name}' not found in '{origin}'")
        return positions[tile_name]

    def global_to_frame_local(self, tile_name: str, x: float, y: float, z_global: int) -> tuple[float, float, int]:
        pos = self.position(tile_name)
        return x - pos.x, y - pos.y, int(z_global) + pos.z - 1

    def frame_local_to_global(self, tile_name: str, x: float, y: float, z_local_0based: int) -> tuple[float, float, int]:
        pos = self.position(tile_name)
        return x + pos.x, y + pos.y, int(z_local_0based) - pos.z + 1

    def channel_residual(self, tile_name: str, channel: str, offsets: dict) -> tuple[int, int, int]:
        frame = self.position(tile_name)
        own = self.position(tile_name, channel)
        shift = offsets[channel]
        return (frame.x + int(shift["dx"]) - own.x,
                frame.y + int(shift["dy"]) - own.y,
                frame.z + int(shift["dz"]) - own.z)

    def print_provenance(self, tile_name: str, offsets: dict) -> None:
        frame = self.position(tile_name)
        print("=== Coordinate context ===")
        print(f"  config: {self.runtime_path}")
        print(f"  pipeline_mode={self.pipeline_mode}, final_frame={self.frame_channel}")
        print(f"  frame XML: {self.frame_xml}")
        print(f"  per-channel XML directory: {self.xml_dir}")
        print(f"  {self.frame_channel} P_O({tile_name})=({frame.x}, {frame.y}, {frame.z})")
        for ch in self.routing:
            channel = ch["id"]
            own = self.position(tile_name, channel)
            shift = offsets.get(channel, {"dx": 0, "dy": 0, "dz": 0})
            residual = self.channel_residual(tile_name, channel, offsets) if channel in offsets else None
            delta = (own.x - frame.x, own.y - frame.y, own.z - frame.z)
            print(
                f"  {channel}: P_c-P_O={delta}, "
                f"s=({shift['dx']}, {shift['dy']}, {shift['dz']}), q={residual}")
        print("==========================")

