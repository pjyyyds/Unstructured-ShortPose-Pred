"""Unit tests for Wave 20: representation-decoupled M3 synergy & true-20f monotonic 3x3 matrix."""
from __future__ import annotations

import json
import unittest

import numpy as np
import torch

from scripts.run_wave20_decoupled_m2b_m3_synergy_5seed import run_decoupled_synergy_on_cohort
from scripts.run_wave20_true20f_mono_3x3_matrix_5seed import run_true20f_mono_matrix


class TestWave20Scripts(unittest.TestCase):
    def test_decoupled_synergy_zero_drift_and_true20f_mono(self) -> None:
        dev = torch.device("cpu")
        rng = np.random.default_rng(29)
        n = 36
        grp = np.array(["s0"] * 12 + ["s1"] * 12 + ["s2"] * 12, dtype=str)
        clips = np.array([f"c{i // 6}" for i in range(n)], dtype=str)
        hist = rng.normal(0.0, 0.35, size=(n, 5, 2)).astype(np.float64)
        hist -= hist[:, -1:]
        truth = rng.normal(0.0, 0.6, size=(n, 30, 2)).astype(np.float64)
        ya_np = rng.uniform(size=(n, 5)) > 0.6
        va_np = np.ones((n, 5), dtype=bool)
        gd = {
            "history": torch.from_numpy(hist), "truth": torch.from_numpy(truth),
            "bio_even": torch.randn(n, 5, 11), "pose_vec_world": torch.randn(n, 5, 2, 2, dtype=torch.float64) * 0.2,
            "signed_knee": torch.randn(n, 5, 1), "quality": torch.from_numpy(rng.uniform(0.1, 0.9, size=(n, 4)).astype(np.float32)),
            "m_pose": torch.ones(n, dtype=torch.float32), "edges": torch.randn(n, 12, 9).abs(),
            "is_car": torch.zeros(n, 12), "nbr_mask": torch.ones(n, 12),
            "y_act": torch.from_numpy(ya_np), "valid_act": torch.from_numpy(va_np),
            "obs_dt": torch.full((n, 4), 0.1, dtype=torch.float64),
            "fut_s": torch.arange(1, 31, dtype=torch.float64)[None].expand(n, -1) * 0.1,
        }
        d_cv = np.linalg.norm(((hist[:, -1:] - hist[:, -2:-1]) / 0.1) * (np.arange(1, 31)[None, :, None] * 0.1) - truth, axis=-1)
        res_d = run_decoupled_synergy_on_cohort(
            "pinns_syn", grp, clips, True, gd, hist, truth, ya_np, va_np,
            np.column_stack((d_cv.mean(1), d_cv[:, -1])), {"all": np.ones(n, dtype=bool)},
            0.1, 18, (1, 1, 1, 1), (2e-3, 1e-3, 1e-3, 1e-3), [42], dev,
        )
        self.assertEqual(res_d["max_abs_traj_drift_vs_m2b_m"], 0.0)
        self.assertIsNotNone(res_d["strata"]["all"]["paired_top1_ci"]["decoupled_rerank_minus_coadapt"]["ci95"])

        hist20 = torch.from_numpy(rng.normal(0.0, 0.3, size=(n, 20, 2)).astype(np.float64))
        hist20 = hist20 - hist20[:, -1:]
        per_dur = {}
        for dur in (5, 10, 20):
            qv = rng.uniform(0.1, 0.95, size=(n, 4)).astype(np.float32)
            qb = np.array([0] * 12 + [1] * 12 + [2] * 12, dtype=int)
            per_dur[dur] = {
                "bio_even": torch.randn(n, dur, 11), "pose_vec_world": torch.randn(n, dur, 2, 2, dtype=torch.float64) * 0.2,
                "signed_knee": torch.randn(n, dur, 1), "quality": torch.from_numpy(qv),
                "m_pose": torch.ones(n, dtype=torch.float32), "q_buckets": qb,
            }
        res_20 = run_true20f_mono_matrix({"site": grp, "clip": clips, "hist20": hist20, "truth": torch.from_numpy(truth), "per_dur": per_dur}, [42], dev)
        self.assertGreater(res_20["T20"]["gate_audit"]["M1_mono_q"]["spearman_gate_vs_q1"], 0.5)
        self.assertGreater(len(json.dumps({"d": res_d, "m": res_20}, allow_nan=False)), 200)


if __name__ == "__main__":
    unittest.main()
