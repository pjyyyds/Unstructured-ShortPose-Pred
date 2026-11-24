import unittest
from unittest.mock import patch

import numpy as np

from scripts import evaluate_pinns_nested_start_stop as run


def sample():
    sites = np.repeat(["a", "b", "c", "d"], 12)
    ids = np.arange(len(sites))
    within = ids % 12
    eligible = np.column_stack([within < 6, within >= 6])
    truth = np.column_stack([eligible[:, 0] & (within % 3 == 0), eligible[:, 1] & (within % 3 == 1)])
    clips = np.array([f"{s}_{i//6}" for s, i in zip(sites, within)])
    rng = np.random.default_rng(29)
    x = rng.normal(size=(len(ids), 5))
    features = {name: x[:, :i+2].copy() for i, name in enumerate(run.NAMES)}
    keys = [(clip, int(i), 10) for i, clip in enumerate(clips)]
    return keys, sites, clips, None, features, eligible, truth, {}


class NestedStartStopTests(unittest.TestCase):
    def test_threshold_exact_tie_gating_and_degenerate_rules(self):
        s, y = np.array([.9,.8,.7,.6]), np.array([1,0,0,1], dtype=bool)
        # At .9 and .6 F1 is exactly 2/3. The higher threshold wins.
        self.assertEqual(run.select_threshold(s, y, np.ones(4, bool))[0], .9)
        self.assertEqual(run.select_threshold(s, y, np.zeros(4, bool))[0], .5)
        self.assertEqual(run.select_threshold(s, ~np.ones(4, bool), np.ones(4, bool))[0], .5)
        eligibility = np.array([False, True])
        t = run.select_threshold(np.zeros(2), np.array([False,True]), eligibility)[0]
        self.assertEqual(t, 0.)
        np.testing.assert_array_equal(eligibility & (np.zeros(2) >= t), [False,True])

    def test_empty_and_single_class_fits_never_use_eval_prevalence(self):
        for y, expected in (([], 0), ([False, False], 0), ([True], 1)):
            fitted = run.fit_binary(np.zeros((len(y), 2)), np.asarray(y,dtype=bool))
            np.testing.assert_array_equal(run.predict(fitted, np.zeros((2,2)), np.array([True,False])),[expected,0])

    def test_outer_labels_do_not_change_any_inner_cutoff_or_held_predictions(self):
        inputs = sample()
        public, private = run.evaluate(inputs)
        modified = list(inputs)
        y = inputs[6].copy()
        held = inputs[1] == "d"
        y[held] = inputs[5][held] & ~y[held]
        modified[6] = y
        other_public, other_private = run.evaluate(modified)
        self.assertEqual(private["outer"]["d"], other_private["outer"]["d"])
        for tag, fit in private["fits"].items():
            if "d" not in fit["training_sites"]:
                self.assertEqual(fit, other_private["fits"][tag])
        self.assertNotEqual(public["sites"]["d"]["event_support"], other_public["sites"]["d"]["event_support"])
        self.assertNotIn("fits", public)
        self.assertNotIn("window_keys", public)

    def test_replay_coefficients_and_three_inner_validation_sites(self):
        inputs = sample()
        _, sites, _, _, xs, eligible, truth, _ = inputs
        public, private = run.evaluate(inputs)
        for held, record in private["outer"].items():
            outer, train = map(np.array, [record["outer_test_rows"],record["outer_train_rows"]])
            for name in run.NAMES:
                for j,event in enumerate(run.EVENTS):
                    item = record["methods"][name][event]
                    actual_inner = np.empty(len(train))
                    for inner, tag in item["inner_fit_by_validation_site"].items():
                        fit = private["fits"][tag]
                        self.assertEqual(set(fit["training_sites"]), set(sites) - {held, inner})
                        train_ids = np.flatnonzero(np.isin(sites,fit["training_sites"]) & eligible[:,j])
                        self.assertEqual(fit["training_rows"], train_ids.tolist())
                        local = sites[train] == inner
                        actual_inner[local] = run.predict(fit, xs[name][train[local]], eligible[train[local],j])
                        if fit["kind"] == "logistic":
                            np.testing.assert_allclose(fit["mean"],xs[name][train_ids].mean(axis=0))
                    np.testing.assert_array_equal(actual_inner,item["inner_oof_scores_outer_train_order"])
                    threshold = run.select_threshold(actual_inner, truth[train,j], eligible[train,j])[0]
                    self.assertEqual(threshold, item["inner_f1_threshold"])
                    final = private["fits"][item["final_fit"]]
                    self.assertEqual(set(final["training_sites"]),set(sites) - {held})
                    outer_score = run.predict(final,xs[name][outer],eligible[outer,j])
                    np.testing.assert_array_equal(outer_score,item["outer_scores"])
                    selected = eligible[outer,j] & (outer_score >= threshold)
                    np.testing.assert_array_equal(selected,item["inner_threshold_decisions"])
                    metrics = public["sites"][held]["methods"][name][event]["inner_f1"]
                    self.assertEqual(metrics["tp"], int((selected & truth[outer,j]).sum()))
        global_tp = public["pooled"][run.DUMMY]["start"]["inner_f1"]["tp"]
        self.assertEqual(global_tp, int(truth[:,0].sum()))
        self.assertEqual(public["equal_site_macro"][run.DUMMY]["start"]["fixed_0_5"]["tp_mean_per_site"], global_tp / 4)

    def test_design_only_never_fits_and_absent_positive_metrics_are_explicit(self):
        with patch.object(run, "fit_binary", side_effect=RuntimeError("Unexpected fit")):
            public, private = run.evaluate(sample(), True)
        self.assertTrue(public["design_only"])
        self.assertIsNone(private)
        zero=np.zeros(2, bool)
        measures=run.event_metrics(zero,~zero,np.ones(2),~zero,~zero)
        self.assertIsNone(measures["ap"])
        self.assertEqual(measures["inner_f1"], {"tp":0,"fp":2,"fn":0,"f1":None})
