"""Shared QC path mapping, provenance and CPU limits."""
import os
import math
from pathlib import Path
from src.core.provenance import atomic_json, file_stamp


def mapped_path(value, mappings):
    value = str(value).replace('\\', '/')
    for old, new in sorted(mappings.items(), key=lambda item: -len(item[0])):
        old = old.replace('\\', '/').rstrip('/')
        if value == old or value.startswith(old + '/'):
            value = str(new).replace('\\', '/').rstrip('/') + value[len(old):]
            break
    return Path(value)


def worker_count(requested, tasks):
    allocated = int(os.environ.get('SLURM_CPUS_PER_TASK') or
                    os.environ.get('LSB_DJOB_NUMPROC') or os.cpu_count() or 1)
    return max(1, min(requested, allocated, max(tasks, 1)))


def stamps(paths):
    return {str(p): file_stamp(p, with_hash=True) for p in paths if p.is_file()}


def unchanged(records):
    for name, record in records.items():
        if file_stamp(name, with_hash='sha256' in record) != record:
            raise ValueError(f'QC input changed during evaluation: {name}')


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    def finite_json(item):
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, dict):
            return {key: finite_json(val) for key, val in item.items()}
        if isinstance(item, (tuple, list)):
            return [finite_json(val) for val in item]
        return item
    atomic_json(path, finite_json(value))
