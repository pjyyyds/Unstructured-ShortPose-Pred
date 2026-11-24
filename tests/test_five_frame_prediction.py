import copy
import unittest

import numpy as np
import torch

from scripts.predict_five_frame_pedestrian import action_coordinate_convention, forecast, observation_inputs
from src.data.actions import compute_action_labels
from src.data.constants import ACTION_NAMES
from reviews.pinns_individual_actions.run_individual_joint_actions import model_for


def observation():
    return {"timestamps_s": [2.0, 2.1, 2.2, 2.3, 2.4],
            "ground_xy_m": [[3.0 + i * 0.1, 4.0] for i in range(5)],
            "image_height_px": 720, "pose_frames": [None] * 5}


class FiveFramePredictionTests(unittest.TestCase):
    def test_complete_motion_and_missing_joints(self):
        batch, world, last = observation_inputs(observation(), True)
        self.assertEqual(last, 2.4)
        np.testing.assert_allclose(world[-1], [3.4, 4.0])
        np.testing.assert_allclose(batch["x_m"][0, -1, :5], [0, 0, 1, 0, 1])
        self.assertEqual(float(batch["q_vec"][0, 0]), 0.)
        self.assertEqual(int(batch["ego_type_oh"].argmax()), 0)

    def test_archived_turn_names_follow_signed_heading_and_reflect_with_y(self):
        hist = np.column_stack([np.arange(5) * 0.1, np.zeros(5)])
        future = hist[-1] + np.column_stack([np.zeros(30), np.arange(1, 31) * 0.1])
        convention = action_coordinate_convention()
        self.assertEqual(convention["pinns_calibration"], {
            "u_axis": "x", "v_axis": "-y", "archived_displacement_in_declared_bev_axes": "[dx, -dy]"})
        for y_sign, expected in [(1, "positive"), (-1, "negative")]:
            bits, valid = compute_action_labels(hist, np.ones(5), future * [1, y_sign], np.ones(30))
            self.assertEqual(valid, 1)
            self.assertEqual(bits.sum(), 1)
            self.assertEqual(convention["archived_turn_sign"][ACTION_NAMES[bits.argmax()]], expected)

    def test_reject_irregular_observations_and_invalid_boxes(self):
        sample = observation()
        sample["timestamps_s"][2] = 2.25
        with self.assertRaisesRegex(ValueError, "spaced"):
            observation_inputs(sample, True)
        sample = observation()
        sample["pose_frames"][0] = {"bbox_center_xywh_px": [50, 100, -5, 20]}
        with self.assertRaisesRegex(ValueError, "positive"):
            observation_inputs(sample, True)

    def test_inference_matches_joint_training_graph_and_translates_with_origin(self):
        torch.set_num_threads(2)
        train_model = model_for("detached").eval()
        state = {"identity": {
            "arm": "detached", "mode": "stop_grad", "heldout": "fixture",
            "registration_sha256": "0eb60b827089822cffa096596848f6e0d0e8b47436c715afda9345a2f84ac1cc"},
            "model_state": train_model.state_dict()}
        sample = observation()
        batch, world, _ = observation_inputs(sample, True)
        with torch.inference_mode():
            raw = train_model(batch)
        decoded = forecast(state, sample)
        self.assertEqual(decoded["action_coordinate_convention"], action_coordinate_convention())
        np.testing.assert_allclose(decoded["trajectories_ground_xy_m"], raw["trajs"][0].numpy() + world[-1])
        np.testing.assert_array_equal(decoded["mode_probabilities"], raw["mode_probs"][0].numpy())
        np.testing.assert_array_equal(list(decoded["observable_action_probabilities"].values()),
                                      raw["act_probs"][0].numpy())
        self.assertEqual(decoded["observable_action_eligibility"], {"Start": False, "Stop": True})
        self.assertAlmostEqual(decoded["pre_event_mean_speed_m_s"], 1.0)
        original = decoded["observable_action_probabilities"]
        filtered = decoded["eligible_action_probabilities"]
        self.assertEqual(filtered, {**original, "Start": 0.0})
        for velocity, flags in [(0.0, (True, False)), (0.4, (False, False))]:
            changed = copy.deepcopy(sample)
            changed["ground_xy_m"] = [[i * 0.1 * velocity, 0.0] for i in range(5)]
            reported = forecast(state, changed)
            self.assertEqual(tuple(reported["observable_action_eligibility"].values()), flags)
            for name in ("Turn-Left", "Turn-Right", "Maintain"):
                self.assertEqual(reported["eligible_action_probabilities"][name],
                                 reported["observable_action_probabilities"][name])
        shifted = copy.deepcopy(sample)
        shift = np.array([100., -200.])
        shifted["ground_xy_m"] = (world + shift).tolist()
        result = forecast(state, shifted)
        np.testing.assert_allclose(result["trajectories_ground_xy_m"],
                                   np.asarray(decoded["trajectories_ground_xy_m"]) + shift)
        np.testing.assert_allclose(result["last_pair_constant_velocity_ground_xy_m"][-1], world[-1] + shift + [3., 0.])


if __name__ == "__main__":
    unittest.main()
