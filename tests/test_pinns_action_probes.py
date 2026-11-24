"""Check grouped splits, nested time-ordered features, train-only fitting, and OOF reconstruction."""
import unittest
from unittest.mock import patch

import numpy as np

from scripts import evaluate_pinns_action_probes as probe


class TestActionProbes(unittest.TestCase):
    def test_fold_assignment_is_closed_and_row_order_invariant(self):
        clips = np.repeat([f"{s}-{i}" for s in "ab" for i in range(7)], np.tile([5, 2, 1, 3, 9, 1, 1], 2))
        sites = np.array([c[0] for c in clips])
        folds = probe.assign_folds(sites, clips)
        perm = np.random.default_rng(4).permutation(len(clips))
        np.testing.assert_array_equal(folds[perm], probe.assign_folds(sites[perm], clips[perm]))
        for clip in set(clips):
            self.assertEqual(len(set(folds[clips == clip])), 1)
        for site in set(sites):
            self.assertEqual(set(folds[sites == site]), set(range(5)))
        sites[0] = "b"
        with self.assertRaisesRegex(ValueError, "unique"):
            probe.assign_folds(sites, clips)

    def test_features_exclude_invalid_and_changed_origins_and_keep_order(self):
        hist = np.zeros((1, 5, 2))
        xm = np.zeros((1, 5, 9))
        xp = np.zeros((1, 5, 17, 6))
        xp[0, :, 0, 0] = [0, 1, 5, 0, 12]
        xp[0, :, 0, 3] = [1, 1, 1, 0, 1]
        xp[0, :, 0, 4] = [0, 0, 1, 1, 1]
        frozen = xp.copy()
        features, eligible = probe.history_features(xm, xp, hist)
        motion, last, ordered = (features[k][0] for k in probe.NAMES)
        self.assertEqual([len(motion), len(last), len(ordered)], [28, 79, 283])
        np.testing.assert_array_equal(eligible, [[True, False]])
        np.testing.assert_array_equal(motion, last[:len(motion)])
        np.testing.assert_array_equal(last, ordered[:len(last)])
        changes = ordered[len(last):].reshape(4, 17, 3)
        np.testing.assert_array_equal(changes[:, 0], [[1, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        np.testing.assert_array_equal(xp, frozen)

    def test_fitting_uses_training_only_and_handles_single_classes(self):
        train = np.arange(40.).reshape(-1, 1)
        labels = np.array([[i >= 20, i < 20, False, True, i % 3 == 0] for i in range(40)])
        test = np.array([[4.], [35.]])
        a, fit = probe.fit_scores(train, labels, test)
        b, _ = probe.fit_scores(train, labels, np.concatenate([test, [[1.e9]]]))
        np.testing.assert_array_equal(a, b[:2])
        np.testing.assert_array_equal(a[:, 2:4], [[0., 1.], [0., 1.]])
        self.assertLess(a[0, 0], a[1, 0])
        self.assertEqual(fit["left"]["fit"], "single-class training frequency")
        train_norm, test_norm = probe.standardize(np.array([[1.,2.], [3.,2.]]), np.array([[9.,4.]]))
        np.testing.assert_array_equal(train_norm, [[-1.,0.], [1.,0.]])
        np.testing.assert_array_equal(test_norm, [[7.,2.]])

    def test_oof_scores_and_eligible_baseline_reconstruct_with_synthetic_data(self):
        rng = np.random.default_rng(12)
        clips = np.repeat([f"clip{i}" for i in range(10)], 3)
        sites = np.repeat(["a", "b"], 15)
        n = len(clips)
        x = rng.normal(size=(n, 5))
        eligible = np.column_stack([x[:, 0] <= 0, x[:, 0] > 0])
        truth = np.column_stack([eligible[:,0] & (x[:,1] < 0), eligible[:,1] & (x[:,1] < 0),
                                 np.zeros(n,dtype=bool), x[:,1] > 1, np.ones(n,dtype=bool)])
        folds = probe.assign_folds(sites, clips)
        features = {k:x[:, :i+1] for i,k in enumerate(probe.NAMES)}
        args = ([(c, i, 5) for i,c in enumerate(clips)], sites, clips, folds, features, eligible, truth, {})
        with patch.object(probe, "load_inputs", return_value=args):
            result = probe.evaluate(None, None)
            with patch.object(probe, "fit_scores", side_effect=AssertionError("design-only tried to fit")):
                self.assertTrue(probe.evaluate(None, None, True)["design_only"])
        oof = result["out_of_fold"]
        np.testing.assert_array_equal(oof["truth"], truth)
        for fold in range(5):
            test = folds == fold
            for j in range(2):
                rate = truth[(~test) & eligible[:, j], j].mean()
                np.testing.assert_array_equal(np.asarray(oof["training_eligible_prevalences"])[test,j],
                                              np.full(test.sum(), rate))
        for name in probe.NAMES:
            predicted = np.asarray(oof["probabilities"][name])
            self.assertIsNone(result["pooled"]["methods"][name]["per_class"]["left"]["ap"])
            for j,action in enumerate(probe.ACTIONS):
                self.assertEqual(probe.class_scores(truth[:,j], predicted[:,j]),
                                 result["pooled"]["methods"][name]["per_class"][action])
        tied = probe.class_scores(np.array([True,False,True]), np.array([.7,.7,.7]))
        self.assertAlmostEqual(tied["ap"], 2/3)


if __name__ == "__main__":
    unittest.main()
