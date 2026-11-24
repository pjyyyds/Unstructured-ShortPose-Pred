"""Synthetic tests for the observed-history-only candidate selector. No real folds are scored."""
import inspect
import unittest

import numpy as np

from scripts.rerank_pinns_velocity_modes import (
    FIXED_SEEDS, MODELS, aggregate, last_two_future, score_forecasts, select_velocity_mode,
)


class TestVelocityModeSelector(unittest.TestCase):
    def test_exact_ties_use_probability_then_index_and_use_last_two_points(self):
        history = np.array([[[155., 99.], [13., 18.], [-8., 14.], [2., 3.], [3., 3.]]])
        reference = last_two_future(history)
        np.testing.assert_allclose(reference[0, [0, -1]], [[4., 3.], [33., 3.]])
        paths = np.repeat(reference[:, None], 6, axis=1)
        paths[:, 0, :, 1] += 4
        probs = np.array([[.6, .01, .13, .13, .12, .01]])
        self.assertEqual(select_velocity_mode(history, paths, probs).item(), 2)
        # Candidate 2 ends closest but its preceding 29 steps are farther than candidate 3's.
        paths[0, 2, :-1, 1] += 3
        paths[0, 3, -1, 1] += 1
        paths[0, [1, 4, 5], :, 1] += 2
        self.assertEqual(select_velocity_mode(history, paths, probs).item(), 3)

    def test_targets_change_the_oracle_but_cannot_change_selection(self):
        self.assertEqual(list(inspect.signature(select_velocity_mode).parameters),
                         ["history", "candidates", "probabilities"])
        history = np.zeros((1, 5, 2))
        probs = np.array([[.45, .38, .06, .05, .04, .02]])
        paths = np.zeros((1, 6, 30, 2))
        paths[0, 0, :, 0] = 1
        paths[0, 1, -1, 0] = 4
        paths[0, 2:, :, 0] = 9
        a, modes_a = score_forecasts(history, paths, probs, np.zeros((1, 30, 2)))
        b, modes_b = score_forecasts(history, paths, probs, paths[:, 0])
        np.testing.assert_array_equal(modes_a, [[0, 1, 1]])
        np.testing.assert_array_equal(modes_b, [[0, 1, 0]])
        # Minimum-ADE oracle's FDE is 4 here, though a different candidate ends at distance 1.
        np.testing.assert_allclose(a[0, 3], [4 / 30, 4])
        np.testing.assert_allclose(b[0, 3], [0, 0])
        np.testing.assert_allclose(a[0, 2], [0, 0])

    def test_equal_site_means_and_clip_intervals(self):
        sites = {"a": np.array([0]), "b": np.array([1, 2]), "c": np.array([3]), "d": np.array([4])}
        clips = np.array(["a0", "b0", "b0", "c0", "d0"])
        scores = np.zeros((len(FIXED_SEEDS), len(MODELS), len(clips), 4, 2))
        scores[:, :, :, 0, :] = 10
        modes = np.zeros(scores.shape[:3] + (3,), dtype=int)
        modes[:, :, 0, 1] = 2
        for s in range(len(FIXED_SEEDS)):
            scores[s, :, :, 1, :] = np.array([1, 5, 5, 1, 1])[:, None] + s
        report = aggregate(scores, modes, sites, clips)
        mean = report["four_site_five_seed_means"]["M0"]
        self.assertAlmostEqual(mean["changed_mode_rate"], .25)
        self.assertAlmostEqual(mean["methods"]["velocity_selected"]["ade_m"], 4)
        self.assertAlmostEqual(mean["reranked_minus_network_m"]["ade_m"], -6)
        ci = report["paired_clip_percentile95_m"]["M0"]
        np.testing.assert_allclose(ci["four_site_equal"]["ade_m"], [-6, -6])
        np.testing.assert_allclose(ci["by_site"]["b"]["fde_m"], [-3, -3])


if __name__ == "__main__":
    unittest.main()
