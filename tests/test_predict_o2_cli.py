"""Unit tests for variable-clock O(2) CLI inference (`scripts/predict_o2_pedestrian.py`)."""
from __future__ import annotations

import math
import unittest

import numpy as np

from scripts.predict_o2_pedestrian import forecast_observation


class TestPredictO2CLI(unittest.TestCase):
    def test_cli_rotation_and_reflection_equivariance(self) -> None:
        obs = {
            "timestamps_s": [0.0, 0.12, 0.25, 0.37, 0.50],
            "ground_xy_m": [[1.0, 2.0], [1.08, 2.05], [1.19, 2.11], [1.31, 2.18], [1.44, 2.27]],
            "neighbors": [
                {"kind": "Person", "ground_xy_m": [2.2, 2.8], "velocity_xy_mps": [-0.5, 0.2]},
                {"kind": "Car", "ground_xy_m": [5.0, 1.5], "velocity_xy_mps": [-3.0, 0.0]},
            ],
        }
        res0 = forecast_observation(obs)
        p0 = np.asarray(res0["bayesian_trajectory_xy_m"])
        self.assertEqual(p0.shape, (30, 2))

        c, s = math.cos(1.2), math.sin(1.2)
        for mat in (np.array([[c, -s], [s, c]]), np.array([[1.0, 0.0], [0.0, -1.0]])):
            obs_r = {
                "timestamps_s": obs["timestamps_s"],
                "ground_xy_m": (np.asarray(obs["ground_xy_m"]) @ mat.T).tolist(),
                "neighbors": [
                    {
                        "kind": nb["kind"],
                        "ground_xy_m": (np.asarray(nb["ground_xy_m"]) @ mat.T).tolist(),
                        "velocity_xy_mps": (np.asarray(nb["velocity_xy_mps"]) @ mat.T).tolist(),
                    }
                    for nb in obs["neighbors"]
                ],
            }
            pr = np.asarray(forecast_observation(obs_r)["bayesian_trajectory_xy_m"]) @ mat
            self.assertLess(float(np.max(np.abs(p0 - pr))), 5e-6)


if __name__ == "__main__":
    unittest.main()
