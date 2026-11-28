"""Unit tests for Wave 29 Passivity-Constrained Speed-Proportional Viewing-Ray O(2) Model (PCR6)."""
from __future__ import annotations

import math
import unittest
import torch

from src.models.emergence_kalman_residual_6slot import compute_split_anatomical_covisibility
from src.models.passivity_ray_pcr6 import O2PassivityRay6SlotNet


class TestWave29PCR6(unittest.TestCase):
    def test_longitudinal_passivity_prevents_backward_reversal_under_extreme_braking(self) -> None:
        torch.manual_seed(31)
        net = O2PassivityRay6SlotNet(use_pose=True, use_interaction=True, alpha_prop=1.5).eval()
        for p in net.parameters():
            if p.ndim >= 1:
                torch.nn.init.normal_(p, std=0.5)
        with torch.no_grad():
            net.slot0_dx.bias.fill_(-10.0)  # extreme negative braking output

        # Slow pedestrian moving along +X at 0.25 m/s (0.025m per 0.1s step)
        h = torch.tensor([[[0.0, 0.0], [0.025, 0.0], [0.050, 0.0], [0.075, 0.0], [0.100, 0.0]]], dtype=torch.float64)
        be = torch.randn(1, 5, 11)
        pvw = torch.zeros(1, 5, 2, 2, dtype=torch.float64)
        sk = torch.zeros(1, 5, 1)
        qv, mp = torch.ones(1, 4), torch.ones(1)
        out = net(h, be, pvw, sk, qv, mp)
        dx_world = out["paths"][0, 0, :, 0].detach() - 0.100
        # Must NEVER reverse backward past x_t0 = 0.100 (dx >= -1e-12)!
        self.assertGreaterEqual(float(dx_world.min()), -1e-12)

    def test_pcr6_full_o2_reflection_rotation_equivariance(self) -> None:
        torch.manual_seed(32)
        net = O2PassivityRay6SlotNet(use_pose=True, use_interaction=True).eval()
        for p in net.parameters():
            if p.ndim >= 1:
                torch.nn.init.normal_(p, std=0.08)

        h = torch.randn(4, 5, 2, dtype=torch.float64) * 0.35
        be = torch.randn(1, 5, 11).expand(4, -1, -1)
        pvw = (torch.randn(1, 5, 2, 2, dtype=torch.float64) * 0.15).expand(4, -1, -1, -1)
        sk = torch.randn(1, 5, 1).expand(4, -1, -1)
        qv, mp = torch.ones(4, 4), torch.ones(4)
        jac = torch.tensor([[[[0.015, -0.005], [0.024, 0.078]]]], dtype=torch.float64).expand(4, 5, 2, 2).clone()
        ed = torch.zeros(4, 12, 8); ed[:, :2, :4] = torch.randn(4, 2, 4); ed[:, :2, 4] = 2.0; ed[:, :2, 5] = 0.9
        ic, nm = torch.ones(4, 12), torch.zeros(4, 12); nm[:, :2] = 1.0
        sc0 = compute_split_anatomical_covisibility(None, None, be, pvw, sk, qv)
        o0 = net(h, be, pvw, sk, qv, mp, sc0, jac, None, None, ed, ic, nm)

        ang = -0.94
        q_m = torch.tensor([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]], dtype=torch.float64) @ torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.float64)
        pvw_r = torch.einsum("btvk,jk->btvj", pvw, q_m)
        jac_r = torch.einsum("btvk,jk->btvj", jac, q_m)
        ed_r = ed.clone()
        ed_r[:, :, 0:2] = (ed[:, :, 0:2].double() @ q_m.T).float()
        ed_r[:, :, 2:4] = (ed[:, :, 2:4].double() @ q_m.T).float()
        sc_r = compute_split_anatomical_covisibility(None, None, be, pvw_r, -sk, qv)
        o_r = net(h @ q_m.T, be, pvw_r, -sk, qv, mp, sc_r, jac_r, None, None, ed_r, ic, nm)
        self.assertLess(float(((o0["paths"] @ q_m.T)[:, [0, 1, 2, 4, 3, 5]] - o_r["paths"]).detach().abs().max()), 1e-6)


if __name__ == "__main__":
    unittest.main()
