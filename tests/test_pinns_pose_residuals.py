import unittest

import numpy as np

from scripts.evaluate_pinns_pose_residuals import (
    ANCHOR,
    GRID,
    NAMES,
    T_FUT,
    evaluate,
    frames_from_history,
    make_features,
    outer_fold,
    predict_world_residual,
    selected_index,
)


def cohort():
    rng = np.random.default_rng(81)
    n = 24
    sites = np.repeat(list("ABCD"), 6)
    clips = np.array([s + str(i % 2) for i, s in enumerate(sites)])
    history = rng.normal(size=(n, 5, 2))
    history -= history[:, -1:].copy()
    frames, moving = frames_from_history(history)
    anchor = (history[:, -1:] - history[:, -2:-1]) * np.arange(1, T_FUT + 1)[None, :, None]
    x = rng.normal(size=(n, 4))
    features = {name: x[:, : i + 2] for i, name in enumerate(NAMES)}
    truth = anchor + rng.normal(size=anchor.shape)
    keys = [(str(c), i, 4) for i, c in enumerate(clips)]
    return dict(
        keys=keys,
        clips=clips,
        sites=sites,
        frames=frames,
        moving=moving,
        anchor=anchor,
        truth=truth,
        features=features,
    )


class ResidualTests(unittest.TestCase):
    def test_history_only_heading_and_pose_representations(self):
        h = np.zeros((2, 5, 2))
        h[0, :, 1] = np.arange(-4, 1) * 0.1
        h[1, -2, 0] = 0.01
        pose = np.zeros((2, 5, 17, 6), dtype=np.float32)
        pose[:, :, 0, 3] = 1
        pose[:, 3, 0, 0] = 0.4
        feat, frame, moving = make_features(h, pose)
        np.testing.assert_array_equal(moving, [True, False])
        np.testing.assert_array_equal(frame[0], [[0, -1], [1, 0]])
        np.testing.assert_array_equal(frame[1], np.eye(2))
        np.testing.assert_allclose(h @ frame @ frame.transpose(0, 2, 1), h)
        pose[:, 3, 0, 0] = 0.8
        revised, _, _ = make_features(h, pose)
        for name in NAMES[:2]:
            np.testing.assert_array_equal(feat[name], revised[name])
        self.assertFalse(np.array_equal(feat[NAMES[2]], revised[NAMES[2]]))
        self.assertEqual([a.shape[1] for a in feat.values()], [28, 79, 283])

    def test_shared_grid_equal_arm_site_weights_and_conservative_ties(self):
        cost = np.full((3, 3, len(GRID)), 9.0)
        cost[:, :, 2:4] = 2.0
        cost[0, 0, 2] = 1.0
        self.assertEqual(selected_index(cost), 2)
        cost[0, 0, 2] = 2.0
        self.assertEqual(selected_index(cost), 2)
        self.assertEqual(selected_index(np.zeros_like(cost)), 0)

    def test_outer_truth_and_features_cannot_leak_into_fit_or_selection(self):
        data = cohort()
        rows, predictions, state = outer_fold(data, "A")
        test_rows = data["sites"] == "A"
        for j, site in enumerate(state["inner_validation_sites"]):
            train_rows, val_rows = (state[f"inner_{j}_{label}_rows"] for label in ("training", "validation"))
            self.assertEqual(set(data["sites"][val_rows]), {site})
            self.assertEqual(set(data["sites"][train_rows]), set("BCD") - {site})
        for name in NAMES:
            fit = {f: state[f"final_{name}_{f}"] for f in ("mean", "scale", "intercept", "weights")}
            delta = predict_world_residual(fit, data["features"][name][rows], data["frames"][rows])
            np.testing.assert_allclose(
                predictions[name], data["anchor"][rows] + state["selected_multiplier"] * delta
            )
        data["truth"][test_rows] += 2e5
        _, new_predictions, _ = outer_fold(data, "A")
        for name in NAMES:
            np.testing.assert_array_equal(predictions[name], new_predictions[name])
            data["features"][name] = data["features"][name].copy()
            data["features"][name][test_rows] += 100
        _, _, new_state = outer_fold(data, "A")
        for k in state:
            np.testing.assert_array_equal(state[k], new_state[k])

    def test_design_and_zero_fallback(self):
        data = cohort()
        design, empty = evaluate(data, {}, design_only=True)
        self.assertIsNone(empty)
        self.assertNotIn("equal_site_macro_m", design)
        data["truth"] = data["anchor"].copy()
        report, private = evaluate(data, {})
        self.assertTrue(all(x["selected_multiplier"] == 0 for x in report["sites"].values()))
        for name in (ANCHOR, *NAMES):
            np.testing.assert_array_equal(private["prediction_" + name], data["anchor"])
            np.testing.assert_array_equal(private["errors_" + name], 0)
        self.assertEqual(report["primary_ordered_minus_anchor_percentile_95_m"]["ade_m"], [0.0, 0.0])
