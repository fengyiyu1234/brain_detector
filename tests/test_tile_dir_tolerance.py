"""Channel tile directories whose names carry slightly shifted stage coordinates."""
import os
import tempfile
import unittest

from src.utils.io import anchor_slice_key, resolve_tile_dir


class ResolveTileDirTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        for rel in ("496700/496700_329300", "496700/496700_340600", "508000/508000_340700"):
            os.makedirs(os.path.join(self.root, rel))

    def test_exact_directory_wins(self):
        rel = os.path.join("508000", "508000_340700")
        self.assertEqual(resolve_tile_dir(self.root, rel), os.path.join(self.root, rel))

    def test_shifted_name_within_tolerance(self):
        got = resolve_tile_dir(self.root, os.path.join("496700", "496700_340700"))
        self.assertEqual(got, os.path.join(self.root, "496700", "496700_340600"))

    def test_shifted_row_directory(self):
        got = resolve_tile_dir(self.root, os.path.join("496800", "496800_340700"))
        self.assertEqual(got, os.path.join(self.root, "496700", "496700_340600"))

    def test_out_of_tolerance_returns_missing_exact_path(self):
        rel = os.path.join("496700", "496700_352000")
        got = resolve_tile_dir(self.root, rel)
        self.assertEqual(got, os.path.join(self.root, rel))
        self.assertFalse(os.path.isdir(got))
        self.assertFalse(os.path.isdir(resolve_tile_dir(self.root, rel, tolerance=0)))

    def test_flat_layout(self):
        os.makedirs(os.path.join(self.root, "flat", "100_250"))
        got = resolve_tile_dir(os.path.join(self.root, "flat"), "100_200")
        self.assertEqual(got, os.path.join(self.root, "flat", "100_250"))

    def test_equidistant_candidates_raise(self):
        os.makedirs(os.path.join(self.root, "496700", "496700_340800"))
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            resolve_tile_dir(self.root, os.path.join("496700", "496700_340700"))

    def test_anchor_slice_key(self):
        self.assertEqual(anchor_slice_key("496700_340600_250700", "496700_340600", "496700_340700"),
                         "496700_340700_250700")
        self.assertEqual(anchor_slice_key("496700_340700_250700", "496700_340700", "496700_340700"),
                         "496700_340700_250700")
        self.assertEqual(anchor_slice_key("other_250700", "496700_340600", "496700_340700"),
                         "other_250700")


if __name__ == "__main__":
    unittest.main()
