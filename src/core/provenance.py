"""Stable record identifiers and lightweight provenance helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


PROVENANCE_SCHEMA_VERSION = 1


def stable_id(kind, *parts):
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"{kind}:{digest}"


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256(module, names):
    """Hash only the named functions/constants, so unrelated edits elsewhere in
    the module do not invalidate checkpoints that never call them.
    'package.module:name' refers to a helper defined in another module."""
    import importlib
    import inspect
    digest = hashlib.sha256()
    for name in names:
        owner, _, attr = name.rpartition(':')
        obj = getattr(importlib.import_module(owner) if owner else module, attr)
        text = inspect.getsource(obj) if callable(obj) else repr(obj)
        digest.update(f"{name}\0{text}\0".encode("utf-8"))
    return digest.hexdigest()


def file_stamp(path, with_hash=False):
    stat = os.stat(path)
    result = {"path": str(Path(path)), "size": stat.st_size,
              "mtime_ns": stat.st_mtime_ns}
    if with_hash:
        result["sha256"] = file_sha256(path)
    return result


def atomic_json(path, value):
    path = Path(path)
    part = path.with_name(path.name + ".part")
    with part.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
    os.replace(part, path)


def provenance_columns(df, tile_name, channel_id):
    """Supply explicit legacy IDs when older 2D checkpoints lack new columns."""
    out = df.copy()
    if "detection_id" not in out:
        out["detection_id"] = [
            stable_id("legacy2d", tile_name, channel_id, i)
            for i in range(len(out))
        ]
        out["provenance_status"] = "legacy_id_inferred_from_row_order"
    if "source_image" not in out:
        out["source_image"] = ""
    if "raw_slice_name" not in out:
        out["raw_slice_name"] = out["slice_name"]
    if "raw_z" not in out:
        out["raw_z"] = out["z"]
    if "score_type" not in out:
        out["score_type"] = ""
    if "intensity_region" not in out:
        out["intensity_region"] = ""
    if "parent_detection_ids" not in out:
        out["parent_detection_ids"] = [
            json.dumps([value], separators=(",", ":"))
            for value in out["detection_id"]
        ]
    return out


def write_output_manifest(path, outputs, input_signature, metadata=None):
    """Publish a completion marker only after every output has been closed."""
    payload = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "input_signature": input_signature,
        "metadata": metadata or {},
        "outputs": {str(name): file_stamp(name, with_hash=True)
                    for name in outputs},
    }
    atomic_json(path, payload)


def content_signature(value):
    """Compare content, not mount points or timestamps, when a hash is available."""
    if isinstance(value, (list, tuple)):
        return [content_signature(item) for item in value]
    if not isinstance(value, dict):
        return value
    if 'sha256' in value and 'size' in value:
        return {'sha256': value['sha256'], 'size': value['size']}
    result = {key: content_signature(item) for key, item in value.items()}
    # A filter checkpoint belongs to one channel. Older manifests embedded all
    # channels in filter_settings; those unrelated parameters are not dependencies.
    if result.get('stage') in ('pre_align_filter', 'final_filter'):
        settings = dict(result.get('filter_settings', {}))
        settings.pop('params', None)
        result['filter_settings'] = settings
    # File collections use stable result-relative names across Windows and HPC.
    return {(key.replace('\\', '/').rsplit('/', 1)[-1]
             if isinstance(item, dict) and 'sha256' in item and 'size' in item
             else key): item for key, item in result.items()}


def manifest_output_path(manifest, name):
    """Map an output to this run, preserving sibling stages and subdirectories."""
    parent = Path(manifest).parent
    parts = name.replace('\\', '/').split('/')
    if parent.name in parts:
        # Use the last occurrence; a mount ancestor may share the stage's name.
        index = len(parts) - 1 - parts[::-1].index(parent.name)
        return parent.joinpath(*parts[index + 1:])
    if len(parts) >= 2:
        return parent.parent / parts[-2] / parts[-1]
    return parent / parts[-1]


def manifest_outputs_valid(path, payload):
    """Resolve relocated outputs relative to their manifest, never to another run."""
    if not payload.get('outputs'):
        return False
    for name, stamp in payload['outputs'].items():
        actual = manifest_output_path(path, name)
        if (not actual.is_file() or actual.stat().st_size != stamp['size']
                or file_sha256(actual) != stamp['sha256']):
            return False
    return True


def output_manifest_valid(path, input_signature):
    """Reject incomplete groups; allow identical content on a different mount."""
    try:
        with open(path, encoding='utf-8') as handle:
            payload = json.load(handle)
        return (payload.get('schema_version') == PROVENANCE_SCHEMA_VERSION
                and content_signature(payload.get('input_signature')) == content_signature(input_signature)
                and manifest_outputs_valid(path, payload))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def model_fingerprints(models, project_root):
    """Hash local weight files; retain names for models loaded by external packages."""
    result = {}
    for name, configured in (models or {}).items():
        path = Path(configured)
        if not path.is_absolute():
            path = Path(project_root) / path
        if path.is_file():
            result[name] = {"path": str(path), "sha256": file_sha256(path)}
        elif path.is_dir():
            files = sorted(p for p in path.rglob("*") if p.is_file())
            result[name] = {
                "path": str(path),
                "files": {str(p.relative_to(path)): file_sha256(p) for p in files},
            }
        else:
            result[name] = {"configured_value": configured, "hash_available": False}
    return result


def validate_raw_input_manifest(manifest_path, output_path, image_dir=None):
    """Check raw outputs and TIFF stamps, resolving relocated inputs in image_dir."""
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if (os.path.getsize(output_path) != manifest.get("output_size") or
            file_sha256(output_path) != manifest.get("output_sha256")):
        raise RuntimeError(f"Raw detection CSV differs from its manifest: {output_path}")
    for suffix, prefix in (("_candidates.csv", "candidate"),
                           ("_instances.csv", "instance")):
        expected_hash = manifest.get(prefix + "_output_sha256")
        if expected_hash is not None:
            related_path = output_path.removesuffix("_result.csv") + suffix
            if (not os.path.isfile(related_path) or
                    os.path.getsize(related_path) != manifest[prefix + "_output_size"] or
                    file_sha256(related_path) != expected_hash):
                raise RuntimeError(f"Raw detection sidecar differs from manifest: {related_path}")
    for image in manifest.get("images", []):
        path = image.get("path")
        if not path:
            continue
        if image_dir is not None:
            # The caller supplies this channel's current tile directory. Never
            # fall back to another sample/channel at a recorded absolute path.
            parts = path.replace("\\", "/").split("/")
            if (len(parts) < 2 or parts[-2] != Path(image_dir).name
                    or parts[-1] in ("", ".", "..")):
                raise RuntimeError(f"Raw input image tile differs from current tile: {path}")
            path = os.path.join(image_dir, parts[-1])
        exists = os.path.isfile(path)
        if image.get("status") == "processed" and not exists:
            raise RuntimeError(f"Raw input image disappeared: {path}")
        if exists:
            now = file_stamp(path)
            recorded_mtime = image.get("mtime_ns")
            current_mtime = now["mtime_ns"]
            # SMB exposes Linux timestamps at Windows FILETIME's 100 ns
            # precision. Discard only the unrepresentable fractional ticks.
            if os.name == "nt" and isinstance(recorded_mtime, int):
                recorded_mtime //= 100
                current_mtime //= 100
            if (now["size"] != image.get("size") or
                    current_mtime != recorded_mtime):
                raise RuntimeError(f"Raw input image changed: {path}")
    return manifest
