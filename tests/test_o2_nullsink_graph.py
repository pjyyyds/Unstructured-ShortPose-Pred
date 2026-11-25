"""Unit tests for O2NullSinkGraphForecaster, pose corruption regimes, and conformal quantiles."""
from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from scripts.run_o2_hetero_graph_pinns_and_dut import O2NullSinkGraphForecaster
from scripts.run_pinns_conformal_and_events import conformal_quantile
from scripts.run_pinns_o2_occlusion_stress import REGIMES, corrupt_pose_bank
from src.models.bioeq_o2_joint import O2BioEqJointForecaster


class TestWave6O2GraphAndRobustness(unittest.TestCase):
    def test_o2_nullsink_graph_rotation_reflection_and_permutation_equivariance(self) -> None:
        torch.manual_seed(17)
        base = O2BioEqJointForecaster(pose_mode="bio_pose", action_mode="none").eval()
        net = O2NullSinkGraphForecaster(base).eval()
        with torch.no_grad():
            net.out_proj.weight.normal_(0.0, 0.08)

        hist = torch.randn(4, 5, 2, dtype=torch.float64) * 0.4
        hist[:, -1] = 0.0
        edges = torch.randn(4, 12, 9, dtype=torch.float32)
        edges[:, :, 4] = torch.linalg.vector_norm(edges[:, :, 0:2], dim=-1)
        edges[:, :, 6] = (torch.rand(4, 12) > 0.2).float()
        is_car = (torch.rand(4, 12) > 0.6).float()
        mask = (torch.rand(4, 12) > 0.3).float()
        bio = torch.randn(4, 5, 11)
        qbio = torch.rand(4)

        out0 = net(hist, edges, is_car, mask, bio, qbio, graph_scale=0.8)
        perm = torch.randperm(12)
        out_perm = net(hist, edges[:, perm], is_car[:, perm], mask[:, perm], bio, qbio, graph_scale=0.8)
        self.assertLess(float((out0["paths"] - out_perm["paths"]).abs().max()), 1e-6)

        theta = 1.13
        c, s = math.cos(theta), math.sin(theta)
        rot = torch.tensor([[c, -s], [s, c]], dtype=torch.float64)
        refl = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.float64)
        for mat, swap_modes in ((rot, False), (refl, True)):
            er = edges.clone()
            er[:, :, 0:2] = (er[:, :, 0:2].double() @ mat.T).float()
            er[:, :, 2:4] = (er[:, :, 2:4].double() @ mat.T).float()
            outr = net(hist @ mat.T, er, is_car, mask, bio, qbio, graph_scale=0.8)
            p_back = outr["paths"] @ mat
            q_r = outr["logits"]
            if swap_modes:
                order = [0, 1, 2, 4, 3, 5]
                p_back = p_back[:, order]
                q_r = q_r[:, order]
            self.assertLess(float((out0["paths"] - p_back).abs().max()), 5e-6)
            self.assertLess(float((out0["logits"] - q_r).abs().max()), 5e-6)

    def test_corrupt_pose_bank_and_conformal_quantile(self) -> None:
        rng = np.random.default_rng(7)
        kp = rng.uniform(10.0, 120.0, size=(3, 5, 17, 3)).astype(np.float32)
        kp[..., 2] = 0.9
        bx = np.broadcast_to(np.array([20.0, 15.0, 40.0, 110.0], dtype=np.float32), (3, 5, 4)).copy()
        ih = np.array([1080.0, 1080.0, 720.0], dtype=np.float32)
        for reg in REGIMES:
            xp, qv, bs, qb = corrupt_pose_bank(kp, bx, ih, reg)
            self.assertEqual(xp.shape, (3, 5, 17, 6))
            self.assertEqual(bs.shape, (3, 5, 11))
            self.assertTrue(torch.isfinite(xp).all() and torch.isfinite(bs).all())
        self.assertTrue(torch.all(corrupt_pose_bank(kp, bx, ih, "complete_blackout")[3] == 0.0))
        q90 = conformal_quantile(np.arange(1, 100, dtype=np.float64), 0.10)
        self.assertEqual(q90, 90.0)

    def test_fit_and_eval_graph_pipeline_smoke(self) -> None:
        from scripts.run_o2_hetero_graph_pinns_and_dut import _fit_base_and_graph, eval_pinns_rows, select_graph_scale
        torch.manual_seed(3)
        data = {
            "site": np.array(["s1"] * 4 + ["s2"] * 4 + ["s3"] * 4 + ["s4"] * 4, dtype="<U24"),
            "clip": np.array([f"c{i // 2}" for i in range(16)], dtype=str),
            "history": torch.randn(16, 5, 2, dtype=torch.float64) * 0.2,
            "truth": torch.randn(16, 30, 2, dtype=torch.float64) * 0.5,
            "bio_seq": torch.randn(16, 5, 11),
            "q_bio": torch.rand(16),
            "edges": torch.randn(16, 12, 9),
            "is_car": (torch.rand(16, 12) > 0.5).float(),
            "nbr_mask": (torch.rand(16, 12) > 0.3).float(),
        }
        data["history"][:, -1] = 0.0
        tr_ids = np.arange(12)
        te_ids = np.arange(12, 16)
        gn, bsc = _fit_base_and_graph(42, data, tr_ids)
        gsc = select_graph_scale(42, data, tr_ids)
        res = eval_pinns_rows(gn, data, te_ids, bsc, gsc, "real")
        self.assertEqual(res.shape, (4, 4))
        self.assertTrue(np.isfinite(res).all())


if __name__ == "__main__":
    unittest.main()
