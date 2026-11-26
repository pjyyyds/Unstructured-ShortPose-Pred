"""Unit tests for Wave 19b: usable pose box/q1 boundary audit & locally-weighted zero-retrain split-conformal."""
from __future__ import annotations

import json
import unittest

import numpy as np
import torch

from scripts.run_wave19_adaptive_conformal_12_and_72pair_5seed import run_adaptive_conformal_benchmark
from scripts.run_wave19_usable_pose_boundary_audit_5seed import run_boundary_audit_on_cohort


class TestWave19bScripts(unittest.TestCase):
    def test_boundary_and_adaptive_conformal_end_to_end(self) -> None:
        dev = torch.device("cpu")
        rng = np.random.default_rng(19)
        n = 60
        grp = np.array(["g0"] * 20 + ["g1"] * 20 + ["g2"] * 20, dtype=str)
        clips = np.array([f"c{i // 10}" for i in range(n)], dtype=str)
        hist = rng.normal(0.0, 0.35, size=(n, 5, 2)).astype(np.float64)
        hist -= hist[:, -1:]
        v0 = np.linalg.norm(hist[:, -1] - hist[:, -2], axis=-1) / 0.1
        truth = rng.normal(0.0, 0.5, size=(n, 30, 2)).astype(np.float64) * (0.4 + v0[:, None, None])
        q_np = rng.uniform(0.05, 0.95, size=(n, 4)).astype(np.float32)
        mp_np = np.ones(n, dtype=np.float32)
        mp_np[:6] = 0.0
        bbox_h = rng.uniform(35.0, 160.0, size=n).astype(np.float64)
        ya_np = rng.uniform(size=(n, 5)) > 0.65
        va_np = np.ones((n, 5), dtype=bool)
        gd = {
            "history": torch.from_numpy(hist), "truth": torch.from_numpy(truth),
            "bio_even": torch.randn(n, 5, 11), "pose_vec_world": torch.randn(n, 5, 2, 2, dtype=torch.float64) * 0.2,
            "signed_knee": torch.randn(n, 5, 1), "quality": torch.from_numpy(q_np), "m_pose": torch.from_numpy(mp_np),
            "edges": torch.randn(n, 12, 9).abs(), "is_car": torch.zeros(n, 12), "nbr_mask": torch.ones(n, 12),
            "y_act": torch.from_numpy(ya_np), "valid_act": torch.from_numpy(va_np),
            "obs_dt": torch.full((n, 4), 0.1, dtype=torch.float64),
            "fut_s": torch.arange(1, 31, dtype=torch.float64)[None].expand(n, -1) * 0.1,
        }
        d_cv = np.linalg.norm(((hist[:, -1:] - hist[:, -2:-1]) / 0.1) * (np.arange(1, 31)[None, :, None] * 0.1) - truth, axis=-1)

        b_res = run_boundary_audit_on_cohort(
            "pinns_syn", grp, clips, True, gd, truth, q_np, mp_np,
            np.column_stack((d_cv.mean(1), d_cv[:, -1])), bbox_h, 20, (1, 1), (2e-3, 1e-3), [42], dev,
        )
        self.assertIn("B0_tiny_lt50px", b_res["strata"])
        self.assertIn("tau_h_65px", b_res["bbox_height_px_cutoff_sweep_vs_M0"])
        self.assertIsNotNone(b_res["strata"]["all"]["paired_top1_ci"]["M1_mono_q_minus_M0"]["ci95"])

        c_res = run_adaptive_conformal_benchmark(
            "pinns_syn", grp, gd, truth, v0, d_cv.max(1), 20, (1, 1, 1, 1), (2e-3, 1e-3, 1e-3, 1e-3), [42], [42], dev,
        )
        self.assertEqual(c_res["n_ordered_cross_pairs"], 6)
        # Strict JSON serialization check (allow_nan=False)
        encoded = json.dumps({"b": b_res, "c": c_res}, allow_nan=False)
        self.assertGreater(len(encoded), 200)


if __name__ == "__main__":
    unittest.main()
