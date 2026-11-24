import unittest
from unittest.mock import patch

import numpy as np

from scripts import evaluate_pinns_neighbor_residuals as experiment


class NeighborResidualsTest(unittest.TestCase):
    def test_typed_ego_exclusion_ties_and_causal_velocity(self):
        # Target moves +1 m/s. A same-numbered car is stationary at x=1.4.
        # Two other people are equidistant at t0; lower ID is selected despite insertion order.
        history = []
        for i in range(5):
            snapshot = {("Person", 7): np.array([i / 10, 0.]),
                        ("Person", 20): np.array([i / 10, 2.]),
                        ("Car", 7): np.array([1.4, 0.]),
                        ("Car", 1): np.array([10.401, 0.])}
            if i == 4:
                snapshot[("Person", 3)] = np.array([0.4, -2.])
            history.append(snapshot)
        own = np.array([[i / 10 - .4, 0.] for i in range(5)])
        masks, geometry, counts = experiment.neighbor_features(history, 7, own)
        self.assertEqual(counts, [2, 1])
        np.testing.assert_allclose(masks, [np.log(3), 1, 0, np.log(2), 1, 1])
        np.testing.assert_allclose(geometry, [0., -2., 2., 0., 0., 0., 1., 0., 1., -1., 0., 1.])
        with self.assertRaises(ValueError):
            experiment.neighbor_features([*history, history[-1]], 7, own)

    def test_masked_absence_and_irregular_neighbor_times(self):
        hist = [{("Person", 4): np.zeros(2)} for _ in range(5)]
        hist[1][("Car", 2)] = np.array([3., 0.])
        hist[4][("Car", 2)] = np.array([3.6, 0.])
        masks, geometry, counts = experiment.neighbor_features(hist, 4, np.zeros((5, 2)))
        self.assertEqual(counts, [0, 1])
        np.testing.assert_equal(masks[:3], np.zeros(3))
        np.testing.assert_equal(geometry[:6], np.zeros(6))
        np.testing.assert_allclose(geometry[-3:], [2., 0., -2.])

    def test_fixed_ridge_closed_form_no_evaluation_scaling(self):
        x = np.array([[-1., 3.], [1., 3.]])
        target = np.broadcast_to(np.array([1., 5.])[:, None, None], (2, 30, 2))
        a = experiment.ridge_residual(x, target, np.array([[2., 1e5]]))
        b = experiment.ridge_residual(x, target, np.array([[2., 1e5], [1e9, -4.]]))
        np.testing.assert_allclose(a, 3 + 4 / (1 + experiment.ALPHA))
        np.testing.assert_array_equal(a, b[:1])

    def test_only_disjoint_training_rows_pass_to_solver(self):
        n = 20
        rng = np.random.default_rng(9)
        sites = np.repeat(["a", "b", "c", "d"], 5)
        folds = np.tile(np.arange(5), 4)
        clips = np.array([f"clip{i}" for i in range(n)])
        own = rng.normal(size=(n, 5, 2))
        own -= own[:, -1:, :].copy()
        velocities = experiment.constant_velocity(own)
        residuals = np.broadcast_to(np.arange(n)[:, None, None], (n, 30, 2))
        future = velocities + residuals
        features = {name: np.column_stack([np.arange(n), np.ones(n)]) for name in experiment.METHODS}
        cohort = experiment.Cohort([(c, 1, 4) for c in clips], sites, clips, folds, features,
                                   np.zeros((n, 2)), np.zeros((n, 2), dtype=bool), own, future, {})
        calls = []
        def synthetic_solver(x, y, held):
            train_ids, held_ids = x[:, 0].astype(int), held[:, 0].astype(int)
            self.assertFalse(set(train_ids).intersection(held_ids))
            self.assertEqual(len(set(folds[held_ids])), 1)
            np.testing.assert_allclose(y, residuals[train_ids], atol=1e-12)
            calls.append(held_ids)
            return np.zeros((len(held), 30, 2))

        with patch.object(experiment, "ridge_residual", side_effect=AssertionError("design fitted")):
            report = experiment.evaluate(cohort, True)
        self.assertNotIn("out_of_fold", report)
        with patch.object(experiment, "ridge_residual", side_effect=synthetic_solver):
            report = experiment.evaluate(cohort)
        self.assertEqual(len(calls), 5 * 3)
        self.assertEqual(report["strata"]["none"]["windows"], n)
        self.assertEqual(report["primary_percentile_95_m"], {"ade_m": [0., 0.], "fde_m": [0., 0.]})
        self.assertEqual(report["out_of_fold"]["errors"]["motion_neighbors"], report["out_of_fold"]["errors"]["ols_five"])
        np.testing.assert_allclose(np.asarray(report["out_of_fold"]["errors"]["motion"])[:, 0], np.arange(n) * np.sqrt(2))
