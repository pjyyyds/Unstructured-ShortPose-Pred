"""Unit tests for Wave 28 Terminal-Mobility & Superlinear Ramp Viewing-Ray O(2) Model (TM-RKR6)."""
from __future__ import annotations

import math
import unittest
import torch

from src.models.emergence_kalman_residual_6slot import compute_split_anatomical_covisibility
from src.models.tm_ray_rkr6 import O2TerminalMobilityRay6SlotNet


class TestWave28TMRKR6(unittest.TestCase):
    def test_stopped_beside_passing_car_has_exact_zero_drift_even_with_prior_chord(self) -> None:
        torch.manual_seed(28)
        net = O2TerminalMobilityRay6SlotNet(use_pose=True, use_interaction=True).eval()
        for p in net.parameters():
            if p.ndim >= 1:
                torch.nn.init.normal_(p, std=0.06)

        # Pedestrian decelerated from x=-0.30 to x=0.0 at frame 3 and is STOPPED at frame 3..4 (p_3 == p_4 == 0),
        # while a fast car passes 1.8m away! Previous models pushed this pedestrian; TM-RKR6 must have 0.0m drift!
        h = torch.tensor(
            [[[-0.30, 0.0], [-0.15, 0.0], [-0.04, 0.0], [0.0, 0.0], [0.0, 0.0]]],
            dtype=torch.float64,
        ).expand(2, 5, 2)
        be = torch.randn(2, 5, 11)
        pvw = torch.randn(2, 5, 2, 2, dtype=torch.float64) * 0.2
        sk = torch.randn(2, 5, 1)
        qv, mp = torch.ones(2, 4), torch.ones(2)
        ed = torch.zeros(2, 12, 8); ed[:, 0, :2] = torch.tensor([1.5, 0.4]); ed[:, 0, 4] = 1.6; ed[:, 0, 5] = 2.2
        ic, nm = torch.ones(2, 12), torch.zeros(2, 12); nm[:, 0] = 1.0
        jac = torch.tensor([[[[0.015, 0.005], [-0.020, 0.080]]]], dtype=torch.float64).expand(2, 5, 2, 2)
        out = net(h, be, pvw, sk, qv, mp, jac_uv_world=jac, edges=ed, is_car=ic, nbr_mask=nm)
        self.assertLess(float(out["paths"][:, 0].detach().abs().max()), 1e-12)

    def test_tm_rkr6_full_o2_reflection_rotation_equivariance(self) -> None:
        torch.manual_seed(29)
        net = O2TerminalMobilityRay6SlotNet(use_pose=True, use_interaction=True).eval()
        for p in net.parameters():
            if p.ndim >= 1:
                torch.nn.init.normal_(p, std=0.05)

        h = torch.randn(4, 5, 2, dtype=torch.float64) * 0.35
        be = torch.randn(1, 5, 11).expand(4, -1, -1)
        pvw = (torch.randn(1, 5, 2, 2, dtype=torch.float64) * 0.15).expand(4, -1, -1, -1)
        sk = torch.randn(1, 5, 1).expand(4, -1, -1)
        qv, mp = torch.ones(4, 4), torch.ones(4)
        jac = torch.tensor([[[[0.016, -0.004], [0.022, 0.075]]]], dtype=torch.float64).expand(4, 5, 2, 2).clone()
        ed = torch.zeros(4, 12, 8); ed[:, :2, :4] = torch.randn(4, 2, 4); ed[:, :2, 4] = 2.0; ed[:, :2, 5] = 0.8
        ic, nm = torch.ones(4, 12), torch.zeros(4, 12); nm[:, :2] = 1.0
        sc0 = compute_split_anatomical_covisibility(None, None, be, pvw, sk, qv)
        o0 = net(h, be, pvw, sk, qv, mp, sc0, jac, None, None, ed, ic, nm)

        ang = 1.27
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
