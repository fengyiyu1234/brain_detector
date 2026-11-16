"""Slice-noise SNR arithmetic of src/utils/snr_slice_compare.py."""
import unittest

import numpy as np
import pandas as pd

from src.utils.snr_slice_compare import add_slice_snr, threshold_lines


class SnrSliceCompareTest(unittest.TestCase):
    def test_slice_noise_is_median_ring_noise_per_tile_slice(self):
        f = pd.DataFrame({'tile': ['a', 'a', 'a', 'a', 'b'], 'z': [1, 1, 1, 2, 1],
                          'mean': [150.0, 300.0, 120.0, 200.0, 200.0],
                          'ring_bg': [100.0, 100.0, 100.0, 100.0, 100.0],
                          'ring_mad': [10.0, 80.0, 0.0, 40.0, 20.0]})
        out = add_slice_snr(f)
        # tile a z1: zero ring noise is ignored, median of 10 and 80 is 45
        np.testing.assert_allclose(out.slice_noise, [45, 45, 45, 40, 20])
        np.testing.assert_allclose(out.slice_snr, [50 / 45, 200 / 45, 20 / 45, 2.5, 5])

    def test_threshold_lines_count_removed_boxes(self):
        kept = pd.DataFrame({'snr': [2.5, 3.5, 9.0, np.nan], 'slice_snr': [4.5, 7.5, 9.0, 1.0]})
        local, slice_line = threshold_lines(kept)
        self.assertIn('<3 1 (25%)', local)
        self.assertIn('<4 2 (50%)', local)   # NaN SNR is kept, as Stage 2.25
        self.assertIn('<5 2 (50%)', slice_line)
        self.assertIn('<8 3 (75%)', slice_line)


if __name__ == '__main__':
    unittest.main()
