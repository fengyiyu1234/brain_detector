"""FUSION_CODE / STITCH_CODE must list every project helper the hashed code uses.

Stage 2.6 and Stage 3 checkpoints hash only these names (provenance.source_sha256).
A helper missing from the list can change without invalidating old results.
"""
import ast
import importlib
import inspect
import textwrap
import types
import unittest

from src.core import stitcher


def _resolve(name):
    """List entry -> (defining module, attribute); mirrors source_sha256."""
    owner, _, attr = name.rpartition(':')
    module = importlib.import_module(owner) if owner else stitcher
    obj = getattr(module, attr)
    home = getattr(obj, '__module__', None) if callable(obj) else None
    if home and home != module.__name__:
        module = importlib.import_module(home)
    return module, attr


def _entry(module, attr):
    return attr if module is stitcher else f'{module.__name__}:{attr}'


def _module_constants(module):
    tree = ast.parse(inspect.getsource(module))
    return {target.id for node in tree.body if isinstance(node, ast.Assign)
            for target in node.targets if isinstance(target, ast.Name)}


def _dependencies(module, attr):
    """Project functions/constants that module.attr references, as list entries."""
    obj = getattr(module, attr)
    if not callable(obj):
        return set()
    tree = ast.parse(textwrap.dedent(inspect.getsource(obj)))
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Name) or node.id == attr or node.id not in vars(module):
            continue
        value = vars(module)[node.id]
        if isinstance(value, types.ModuleType):
            continue
        if callable(value):
            home = getattr(value, '__module__', '') or ''
            if home.startswith('src.'):
                dep_module = importlib.import_module(home)
                # Names imported into stitcher keep their short stitcher spelling.
                found.add(node.id if module is stitcher else _entry(dep_module, node.id))
        elif node.id in _module_constants(module):
            found.add(_entry(module, node.id))
    return found


def _closure(names):
    pending, seen = list(names), set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        pending.extend(_dependencies(*_resolve(name)))
    return seen


class CheckpointCodeListTest(unittest.TestCase):
    def test_lists_are_closed_over_project_helpers(self):
        for list_name in ('FUSION_CODE', 'STITCH_CODE'):
            listed = set(getattr(stitcher, list_name))
            missing = _closure(listed) - listed
            self.assertFalse(
                missing, f"stitcher.{list_name} is missing {sorted(missing)}; "
                         "add them so checkpoint hashes cover the code actually run")

    def test_detects_an_omitted_helper(self):
        self.assertIn('src.utils.markers:clean_markers', _closure({'split_class'}))
        self.assertIn('_merge_class', _closure({'merge_soma_detections_2d'}))


if __name__ == '__main__':
    unittest.main()
