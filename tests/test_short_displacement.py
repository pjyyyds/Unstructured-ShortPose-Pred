import unittest

import numpy as np

from scripts.evaluate_pinns_short_displacement import describe, distances, group_scores
from scripts.score_pinns_annotation_history import FIRST, PRIOR


class TestShortDisplacement(unittest.TestCase):
    def test_cumulative_distance_at_threshold_with_return_to_origin(self):
        paths = np.zeros((2, 5, 2))
        paths[0, :, 0] = [0, 0.125, 0, 0.125, 0]
        paths[1, :, 1] = [0, 0.125, 0.25, 0.25, 0.25]
        route, net = distances(paths)
        np.testing.assert_array_equal(route, [0.5, 0.25])
        np.testing.assert_array_equal(net, [0, 0.25])
        _, mask = describe([("A", 1, 4), ("B", 3, 4)], paths, np.array(["A", "B"]), np.array([FIRST, PRIOR]))
        np.testing.assert_array_equal(mask["first_long"], [True, False])
        np.testing.assert_array_equal(mask["prior_short"], [False, True])

    def test_absent_site_does_not_become_a_zero(self):
        # Deliberately imbalanced. Equal-site difference is (1+3)/2, not the pooled 7/3.
        sites = np.array(["a", "b", "b", "c", "d"])
        clips = sites.copy()
        values = np.zeros((2, 3, 5, 2))
        values[:, 1, :, :] = np.array([1., 3., 3., 6., 6.])[None, :, None]
        out = group_scores(np.array([True, True, True, False, False]), clips, sites, np.zeros((5,5)), values, [42, 1024])
        self.assertIsNone(out["four_site_ade_difference_m"])
        self.assertEqual(out["contributing_sites"], 2)
        self.assertEqual(out["M1_minus_M0"]["ade_m"], 2.)
        self.assertEqual(out["conditional_paired_clip_95"]["ade_m"], [2., 2.])
