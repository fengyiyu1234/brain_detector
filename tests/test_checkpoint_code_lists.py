"""FUSION_CODE / STITCH_CODE / SOLVE_CODE must list every project helper the hashed code uses.

Stage 2.6 and Stage 3 checkpoints hash only the stitcher lists (provenance.source_sha256);
Stage 2.9 tile positions hash solve_tile_positions.SOLVE_CODE (provenance.code_sha256).
A helper missing from a list can change without invalidating old results.
"""
import ast
import importlib
import inspect
import textwrap
import types
import unittest
from unittest.mock import patch

from src.core import stitcher
from src.core import solve_tile_positions as solver
from src.core import tile_position_pipeline as tpp
from src.core.provenance import code_sha256, code_text, code_tree


def _resolve(name, base=stitcher):
    """List entry -> (defining module, attribute); mirrors source_sha256."""
    owner, _, attr = name.rpartition(':')
    module = importlib.import_module(owner) if owner else base
    obj = getattr(module, attr)
    home = getattr(obj, '__module__', None) if callable(obj) else None
    if home and home != module.__name__:
        module = importlib.import_module(home)
    return module, attr


def _entry(module, attr, base=stitcher):
    return attr if module is base else f'{module.__name__}:{attr}'


def _module_constants(module):
    tree = ast.parse(inspect.getsource(module))
    return {target.id for node in tree.body if isinstance(node, ast.Assign)
            for target in node.targets if isinstance(target, ast.Name)}


def _dependencies(module, attr, base=stitcher, drop_calls=None):
    """Project functions/constants that module.attr references, as list entries.

    drop_calls (code_sha256 lists only): logging calls left out of the hash, which also
    covers `module.helper` references such as `cs.run_seam_job`.
    """
    obj = getattr(module, attr)
    if not callable(obj):
        return set()
    if drop_calls is None:
        tree = ast.parse(textwrap.dedent(inspect.getsource(obj)))
    else:
        tree = code_tree(obj, drop_calls)
    found = set()
    for node in ast.walk(tree):
        if (drop_calls is not None and isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)):
            value = vars(module).get(node.value.id)
            if isinstance(value, types.ModuleType) and value.__name__.startswith('src.'):
                found.add(_entry(value, node.attr, base))
            continue
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
                found.add(node.id if module is stitcher and base is stitcher
                          else _entry(dep_module, node.id, base))
        elif node.id in _module_constants(module):
            found.add(_entry(module, node.id, base))
    return found


def _closure(names, base=stitcher, drop_calls=None, skip=()):
    pending, seen = list(names), set()
    while pending:
        name = pending.pop()
        if name in seen or name in skip:
            continue
        seen.add(name)
        pending.extend(_dependencies(*_resolve(name, base), base, drop_calls))
    return seen


class CheckpointCodeListTest(unittest.TestCase):
    def test_lists_are_closed_over_project_helpers(self):
        for list_name in ('FUSION_CODE', 'STITCH_CODE'):
            listed = set(getattr(stitcher, list_name))
            missing = _closure(listed) - listed
            self.assertFalse(
                missing, f"stitcher.{list_name} is missing {sorted(missing)}; "
                         "add them so checkpoint hashes cover the code actually run")

    def test_solver_list_is_closed_over_project_helpers(self):
        listed = set(tpp.SOLVE_CODE)
        self.assertIn('main', listed)
        self.assertFalse(listed & set(tpp.SOLVE_CODE_SKIP))
        missing = _closure(listed, solver, tpp.SOLVE_LOG_CALLS,
                           tpp.SOLVE_CODE_SKIP) - listed
        self.assertFalse(
            missing, f"tile_position_pipeline.SOLVE_CODE is missing {sorted(missing)}; "
                     "add them (or to SOLVE_CODE_SKIP if the pipeline never runs them)")
        self.assertIn('src.utils.compare_stitching:run_seam_job',
                      _closure({'measure_seams'}, solver, tpp.SOLVE_LOG_CALLS))
        tpp.solver_code_sha256()

    def test_code_hash_ignores_comments_line_endings_and_logging(self):
        def text(*source_lines, newline='\n'):
            source = newline.join(source_lines) + newline
            namespace = {}
            exec(compile(source.replace('\r', ''), '<fake>', 'exec'), namespace)
            with patch('inspect.getsource', return_value=source):
                return code_text(namespace['f'], ('say',))

        base = text('def f(x):', '    y = x + 1', '    say(f"y={y}")', '    return y')
        same = text('def f(x):', '    """Doc."""', '    # comment', '',
                    '    y = x + 1  # trailing', '    say("changed",', '        y)',
                    '    return y', newline='\r\n')
        changed = text('def f(x):', '    y = x + 2', '    say(f"y={y}")', '    return y')
        self.assertEqual(base, same)
        self.assertNotEqual(base, changed)
        # A log call sharing a line with real code keeps that line.
        self.assertIn('if x: say(x)', text('def f(x):', '    if x: say(x)', '    return x'))
        # '#' inside a string is not a comment.
        self.assertIn("'#'", text('def f():', "    return '#'  # c"))

    def test_detects_an_omitted_helper(self):
        self.assertIn('src.utils.markers:clean_markers', _closure({'split_class'}))
        self.assertIn('_merge_class', _closure({'merge_soma_detections_2d'}))


if __name__ == '__main__':
    unittest.main()
