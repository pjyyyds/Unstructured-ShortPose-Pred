import unittest

import numpy as np

from scripts.build_pinns_causal_maps import masked_motion


class MaskedMotionTest(unittest.TestCase):
    def test_anchor_translation_uses_only_valid_last_five_observations(self):
        past = np.full((20, 2), 91, dtype=np.float32)
        past[-5:] = [[10, 20], [1000, -1000], [12, 18], [0, 0], [14, 16]]
        hmask = np.ones(20, dtype=np.float32)
        hmask[[-4, -2]] = 0
        future = np.full((30, 2), 99, dtype=np.float32)
        future[[0, 4]] = [[0, 0], [3, 4]]
        fmask = np.zeros(30, dtype=np.float32)
        fmask[[0, 4]] = 1
        hist, hm, fut, fm = masked_motion({
            "hist_pos_20": past, "hist_mask_20": hmask, "p_t0": past[-1],
            "fut_xy": future, "fut_mask": fmask,
        })
        np.testing.assert_array_equal(hist, [[-4, 4], [0, 0], [-2, 2], [0, 0], [0, 0]])
        np.testing.assert_array_equal(hm, [True, False, True, False, True])
        np.testing.assert_array_equal(fut[fm], [[0, 0], [3, 4]])
        np.testing.assert_array_equal(fut[~fm], np.zeros((28, 2)))
        self.assertEqual(hist.dtype, np.float32)
        self.assertEqual(fut.dtype, np.float32)
        self.assertEqual(hm.dtype, bool)
        self.assertEqual(fm.dtype, bool)
