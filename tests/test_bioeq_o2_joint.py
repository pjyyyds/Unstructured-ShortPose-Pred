"""Unit tests for O(2) parity-equivariant multi-duration & joint action-trajectory forecaster."""
from __future__ import annotations

import math
import unittest

import torch

from src.models.bioeq_neural import biomech_sequence_torch
from src.models.bioeq_o2_joint import O2BioEqJointForecaster, o2_joint_loss

COCO_SWAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


class TestO2BioEqJointForecaster(unittest.TestCase):
    def test_o2_rotation_and_reflection_equivariance_across_durations(self) -> None:
        torch.manual_seed(2026)
        net = O2BioEqJointForecaster(pose_mode="bio_pose", action_mode="stop_grad").eval()
        for m in (net.bio_par_central, net.bio_par_pair, net.bio_score_central, net.bio_score_pair):
            torch.nn.init.normal_(m.weight, std=0.05)

        for t_obs in (5, 10, 20):
            hist = torch.cumsum(torch.randn(4, t_obs, 2, dtype=torch.float64) * 0.15 + 0.12, dim=1)
            kpts = torch.rand(4, t_obs, 17, 3, dtype=torch.float32)
            kpts[..., :2] = kpts[..., :2] * 80.0 + 200.0
            kpts[..., 2] = 0.35 + 0.6 * kpts[..., 2]
            bbox = torch.full((4, t_obs, 4), 120.0, dtype=torch.float32)
            bio_seq, q_bio = biomech_sequence_torch(kpts, bbox)
            out0 = net(hist, bio_seq, q_bio)
            p0 = out0["logits"].softmax(dim=1).double()
            bayes0 = torch.einsum("bk,bktw->btw", p0, out0["paths"])
            act0 = torch.sigmoid(out0["act_logits"])

            for deg, reflect in ((37.0, False), (90.0, False), (180.0, False), (53.0, True), (0.0, True)):
                rad = math.radians(deg)
                rot = torch.tensor([[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]], dtype=torch.float64)
                op = rot @ (torch.diag(torch.tensor([1.0, -1.0], dtype=torch.float64)) if reflect else torch.eye(2, dtype=torch.float64))
                h_g = hist @ op.T
                kp_g = kpts.clone()
                if reflect:
                    kp_g[..., 0] = 640.0 - kp_g[..., 0]
                    kp_g = kp_g[:, :, COCO_SWAP, :]
                bs_g, qb_g = biomech_sequence_torch(kp_g, bbox)
                out_g = net(h_g, bs_g, qb_g)
                p_g = out_g["logits"].softmax(dim=1).double()
                bayes_g = torch.einsum("bk,bktw->btw", p_g, out_g["paths"]) @ op
                self.assertLess((bayes_g - bayes0).abs().max().item(), 5e-6)
                top0 = out0["paths"][torch.arange(4), p0.argmax(1)]
                top_g = out_g["paths"][torch.arange(4), p_g.argmax(1)] @ op
                self.assertLess((top_g - top0).abs().max().item(), 5e-6)
                act_g = torch.sigmoid(out_g["act_logits"])
                for idx in (0, 1, 4):
                    self.assertLess((act_g[:, idx] - act0[:, idx]).abs().max().item(), 2e-6)
                if reflect:
                    self.assertLess((act_g[:, 2] - act0[:, 3]).abs().max().item(), 2e-6)
                    self.assertLess((act_g[:, 3] - act0[:, 2]).abs().max().item(), 2e-6)
                else:
                    self.assertLess((act_g[:, 2:4] - act0[:, 2:4]).abs().max().item(), 2e-6)

    def test_stop_grad_isolation_and_variable_clock(self) -> None:
        torch.manual_seed(7)
        hist = torch.cumsum(torch.randn(3, 5, 2, dtype=torch.float64) * 0.2 + 0.15, dim=1)
        tgt = hist[:, -1:] + torch.randn(3, 30, 2, dtype=torch.float64) * 0.2
        w = torch.ones(3)
        for mode, expect_grad in (("stop_grad", False), ("end_to_end", True)):
            net = O2BioEqJointForecaster(pose_mode="motion", action_mode=mode)
            out = net(hist)
            loss = o2_joint_loss(out, tgt, w, act_weight=0.0)
            loss.backward()
            has_grad = net.act_turn.weight.grad is not None and net.act_turn.weight.grad.abs().sum().item() > 0.0
            self.assertEqual(has_grad, expect_grad)
        net_clk = O2BioEqJointForecaster(pose_mode="motion", action_mode="none").eval()
        dt = torch.full((3, 4), 0.125, dtype=torch.float64)
        ft = torch.arange(1, 31, dtype=torch.float64)[None, :].repeat(3, 1) * 0.1001
        out_clk = net_clk(hist, obs_dt=dt, fut_times_s=ft)
        expected_cv = hist[:, -1:] + ((hist[:, -1:] - hist[:, -2:-1]) / 0.125) * ft[:, :, None]
        self.assertLess((out_clk["paths"][:, 0] - expected_cv).abs().max().item(), 1e-12)


if __name__ == "__main__":
    unittest.main()
