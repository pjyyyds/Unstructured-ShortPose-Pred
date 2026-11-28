"""Unit tests for Wave 26 Coherence-Kalman Slot-0 (EKR6) & JAAD Full-17 Split AutoPose + NP Conformal."""
from __future__ import annotations

import unittest
import numpy as np
import torch

from src.models.emergence_kalman_residual_6slot import (
    O2EmergenceKalmanRes6SlotNet,
    compute_coherence_kalman_velocity,
    compute_split_anatomical_covisibility,
)
from src.models.jaad_full17_reappear_models import (
    calibrate_conformal_np_thresholds,
    kpts5_to_full17_split_autopose,
)


class TestWave26EKR6AndJAADFull17(unittest.TestCase):
    def test_coherence_kalman_separates_jitter_from_accelerating_emergence(self) -> None:
        # Row 0: pure oscillatory footpoint jitter (+3cm, -3cm, +3cm, -3cm -> chord=0, v_last=-0.30 m/s)
        # Row 1: smooth co-directional emergence (+5cm, +8cm, +11cm, +14cm -> chord=0.38m, kappa=1, cos=+1)
        h_jit = torch.tensor([[[0.0, 0.0], [0.03, 0.0], [0.0, 0.0], [0.03, 0.0], [0.0, 0.0]]], dtype=torch.float64)
        h_emg = torch.tensor([[[0.0, 0.0], [0.05, 0.0], [0.13, 0.0], [0.24, 0.0], [0.38, 0.0]]], dtype=torch.float64)
        h = torch.cat([h_jit, h_emg], dim=0)
        out = compute_coherence_kalman_velocity(h, sigma_js=0.045, w_chord=0.65, gamma_neg=0.45, d_ref=0.55)
        self.assertLess(float(out["v_filt"][0].norm()), 0.01)  # 30x jitter completely suppressed!
        self.assertGreater(float(out["v_filt"][1, 0]), 1.35)   # 1.40 m/s emerging walk preserved (>96%)!

    def test_ekr6_o2_equivariance_and_stationary_zero_drift(self) -> None:
        torch.manual_seed(11)
        net = O2EmergenceKalmanRes6SlotNet(use_pose=True, use_interaction=True).eval()
        for p in net.parameters():
            if p.ndim >= 1:
                torch.nn.init.normal_(p, std=0.04)

        # Stationary isolated pedestrian -> Slot 0 must have exact 0.0m drift!
        h_zero = torch.zeros(2, 5, 2, dtype=torch.float64)
        be = torch.randn(2, 5, 11)
        pvw = torch.randn(2, 5, 2, 2, dtype=torch.float64) * 0.1
        sk = torch.randn(2, 5, 1)
        qv = torch.ones(2, 4)
        mp = torch.ones(2)
        oz = net(h_zero, be, pvw, sk, qv, mp)
        self.assertLess(float(oz["paths"][:, 0].detach().abs().max()), 1e-12)

        # O(2) reflection + rotation equivariance on active moving + neighbor + pose batch
        h = torch.randn(4, 5, 2, dtype=torch.float64) * 0.25
        ed = torch.zeros(4, 12, 8); ed[:, :2, :4] = torch.randn(4, 2, 4); ed[:, :2, 4] = 2.0; ed[:, :2, 5] = 0.7
        ic, nm = torch.ones(4, 12), torch.zeros(4, 12); nm[:, :2] = 1.0
        sc0 = compute_split_anatomical_covisibility(None, None, be[:1].expand(4, -1, -1), pvw[:1].expand(4, -1, -1, -1), sk[:1].expand(4, -1, -1), qv[:1].expand(4, -1))
        o0 = net(h, be[:1].expand(4, -1, -1), pvw[:1].expand(4, -1, -1, -1), sk[:1].expand(4, -1, -1), qv[:1].expand(4, -1), mp[:1].expand(4), sc0, None, ed, ic, nm)
        ang = torch.tensor(1.13, dtype=torch.float64)
        R = torch.tensor([[torch.cos(ang), -torch.sin(ang)], [torch.sin(ang), torch.cos(ang)]], dtype=torch.float64) @ torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.float64)
        pvw4 = pvw[:1].expand(4, -1, -1, -1)
        pvw_r = torch.einsum("btvk,jk->btvj", pvw4, R)
        sk4 = sk[:1].expand(4, -1, -1)
        ed_r = ed.clone()
        ed_r[:, :, 0:2] = (ed[:, :, 0:2].double() @ R.T).float()
        ed_r[:, :, 2:4] = (ed[:, :, 2:4].double() @ R.T).float()
        sc_r = compute_split_anatomical_covisibility(None, None, be[:1].expand(4, -1, -1), pvw_r, -sk4, qv[:1].expand(4, -1))
        o_r = net(h @ R.T, be[:1].expand(4, -1, -1), pvw_r, -sk4, qv[:1].expand(4, -1), mp[:1].expand(4), sc_r, None, ed_r, ic, nm)
        self.assertLess(float(((o0["paths"] @ R.T)[:, [0, 1, 2, 4, 3, 5]] - o_r["paths"]).abs().max()), 1e-6)

    def test_jaad_full17_covis_imputation_and_np_conformal_quantile(self) -> None:
        k5 = np.ones((5, 17, 3), dtype=np.float32) * 50.0
        k5[..., 2] = 0.85
        k5[:2, 13:, 0] = 999.0  # wild hallucinated leg coords when conf < 0.25
        k5[:2, 13:, 2] = 0.05
        f20, q4 = kpts5_to_full17_split_autopose(k5, 80.0)
        self.assertEqual(f20.shape, (5, 20))
        self.assertEqual(q4.shape, (4,))
        np.testing.assert_allclose(f20[0, 14:18], f20[2, 14:18], atol=1e-6)
        scores = np.linspace(0.01, 0.99, 39)[:, None]
        thr = calibrate_conformal_np_thresholds(scores, target_fpr=0.05)
        # ceil((39+1)*0.95) = 38 -> 1-based 38th out of 39 (0-based idx 37)
        self.assertAlmostEqual(float(thr[0]), float(scores[37, 0]), places=6)


if __name__ == "__main__":
    unittest.main()
