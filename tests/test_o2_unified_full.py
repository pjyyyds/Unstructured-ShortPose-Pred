"""Unit tests for unified Ours_O2_Full (O2UnifiedFullForecaster)."""
from __future__ import annotations

import math
import unittest

import torch

from src.models.o2_unified_full import O2UnifiedFullForecaster


class TestO2UnifiedFull(unittest.TestCase):
    def test_param_budget_equivariance_and_safety_fallbacks(self) -> None:
        torch.manual_seed(29)
        net = O2UnifiedFullForecaster().eval()
        n_params = sum(p.numel() for p in net.parameters())
        self.assertLess(n_params, 25000)

        with torch.no_grad():
            net.bio_par_central.weight.normal_(0.0, 0.05)
            net.bio_par_pair.weight.normal_(0.0, 0.05)
            net.graph_proj.weight.normal_(0.0, 0.05)
            net.bridge_even.weight.normal_(0.0, 0.05)
            net.bridge_odd.weight.normal_(0.0, 0.05)

        hist = torch.randn(4, 5, 2, dtype=torch.float64) * 0.35
        hist[:, -1] = 0.0
        bio = torch.randn(4, 5, 11)
        qbio = torch.rand(4) * 0.8 + 0.1
        edg = torch.randn(4, 12, 9)
        edg[:, :, 4] = torch.linalg.vector_norm(edg[:, :, 0:2], dim=-1)
        car = (torch.rand(4, 12) > 0.6).float()
        msk = (torch.rand(4, 12) > 0.3).float()

        # Blackout & empty-graph exact fallback identities
        o_b0 = net(hist, bio, torch.zeros_like(qbio), edg, car, msk, bio_scale=1.0, graph_scale=1.0)
        o_bs = net(hist, bio, qbio, edg, car, msk, bio_scale=0.0, graph_scale=1.0)
        self.assertEqual(float((o_b0["paths"] - o_bs["paths"]).abs().max()), 0.0)
        o_g0 = net(hist, bio, qbio, edg, car, torch.zeros_like(msk), bio_scale=1.0, graph_scale=1.0)
        o_gs = net(hist, bio, qbio, edg, car, msk, bio_scale=1.0, graph_scale=0.0)
        self.assertEqual(float((o_g0["paths"] - o_gs["paths"]).abs().max()), 0.0)

        # Full active O(2) rotation + reflection equivariance across paths, modes, and 5 actions
        out0 = net(hist, bio, qbio, edg, car, msk, bio_scale=0.8, graph_scale=0.8, use_action_bridge=True)
        c, s = math.cos(0.79), math.sin(0.79)
        rot = torch.tensor([[c, -s], [s, c]], dtype=torch.float64)
        refl = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.float64)
        for mat, is_refl in ((rot, False), (refl, True)):
            er = edg.clone()
            er[:, :, 0:2] = (er[:, :, 0:2].double() @ mat.T).float()
            er[:, :, 2:4] = (er[:, :, 2:4].double() @ mat.T).float()
            outr = net(hist @ mat.T, bio, qbio, er, car, msk, bio_scale=0.8, graph_scale=0.8, use_action_bridge=True)
            pb = outr["paths"] @ mat
            qr = outr["logits"]
            ar = outr["act_logits"]
            if is_refl:
                pb = pb[:, [0, 1, 2, 4, 3, 5]]
                qr = qr[:, [0, 1, 2, 4, 3, 5]]
                ar = ar[:, [0, 1, 3, 2, 4]]
            self.assertLess(float((out0["paths"] - pb).abs().max()), 5e-6)
            self.assertLess(float((out0["logits"] - qr).abs().max()), 5e-6)
            self.assertLess(float((out0["act_logits"] - ar).abs().max()), 5e-6)


if __name__ == "__main__":
    unittest.main()
