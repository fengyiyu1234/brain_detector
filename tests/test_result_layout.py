import tempfile
import unittest
from pathlib import Path

from src.core.result_layout import migrate_result_layout, result_paths


class ResultLayoutTests(unittest.TestCase):
    def test_legacy_layout_migrates_without_losing_exposures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = {
                "1_tile_2d_raw": "tile_GFP_result.csv",
                "1_tile_2d_prefiltered": "tile_GFP_result.csv",
                "0_channel_alignment": "tile_offsets.json",
                "1_tile_2d_fused": "tile_GFP_result.csv",
                "1_tile_2d_filtered": "tile_GFP_result.csv",
                "2_global_2d_raw": "GFP_2d_global.csv",
                "3_channel_3d": "GFP_3d_tracked.pkl",
                "4_colocalization": "coloc_result.csv",
            }
            for directory, filename in legacy.items():
                folder = root / directory
                folder.mkdir()
                (folder / filename).write_text(directory)
            (root / "0_channel_alignment" / "tile_GFP_result.csv").write_text("aligned")
            report = root / "5_analysis_report"
            (report / "tile_positions").mkdir(parents=True)
            (report / "tile_positions" / "xml_merging_GFP.xml").write_text("<xml/>")
            (report / "cell_centroids").mkdir()
            (report / "cell_centroids" / "ob_neuron.csv").write_text("x,y,z")
            paths = result_paths(str(root))
            self.assertEqual(Path(paths["pATH_ALIGN_OFFSETS"]), root / "3_2d_aligned")
            self.assertEqual(Path(paths["pATH_DET_FUSED"]), root / "3_2d_aligned_fusion")
            self.assertEqual(Path(paths["pATH_DET_FILTERED"]), root / "4_2d_filtered")
            self.assertEqual(Path(paths["pATH_CENTROIDS"]), root / "7_colocalization" / "cell_centroids")
            migrate_result_layout(str(root))
            migrate_result_layout(str(root))
            self.assertEqual((root / "3_2d_aligned" / "tile_GFP_result.csv").read_text(), "aligned")
            self.assertEqual((root / "3_2d_aligned_fusion" / "tile_GFP_result.csv").read_text(), "1_tile_2d_fused")
            self.assertEqual((root / "4_2d_filtered" / "tile_GFP_result.csv").read_text(), "1_tile_2d_filtered")
            self.assertTrue((root / "5_2d_global" / "tile_positions" / "xml_merging_GFP.xml").is_file())
            self.assertTrue((root / "7_colocalization" / "cell_centroids" / "ob_neuron.csv").is_file())
            self.assertFalse((root / "1_tile_2d_raw").exists())

    def test_different_checkpoint_files_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "0_channel_alignment"
            new = root / "3_2d_aligned"
            old.mkdir()
            new.mkdir()
            (old / "tile_offsets.json").write_text("old")
            (new / "tile_offsets.json").write_text("new")
            with self.assertRaisesRegex(FileExistsError, "Conflicting legacy"):
                migrate_result_layout(str(root))
            self.assertEqual((new / "tile_offsets.json").read_text(), "new")
            self.assertEqual((old / "tile_offsets.json").read_text(), "old")


if __name__ == "__main__":
    unittest.main()
