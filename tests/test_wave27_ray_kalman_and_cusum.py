"""Unit tests for Wave 27 Projective-Jacobian Anisotropic Ray-Kalman (PJ-IAKF) & Causal Post-Reappear CUSUM."""
from __future__ import annotations

import math
import unittest
import numpy as np
import torch

from scripts.ppt_mainline_o2_models import _map_uv_torch
from src.models.emergence_kalman_residual_6slot import compute_split_anatomical_covisibility
from src.models.ray_kalman_and_cusum import (
    O2RayKalmanRes6SlotNet,
    anisotropic_ray_kalman_filter,
    causal_post_reappear_cusum_scores,
    compute_homography_jacobian_world,
)


class TestWave27RayKalmanAndCUSUM(unittest.TestCase):
    def test_homography_jacobian_matches_finite_differences(self) -> None:
        hm = torch.tensor(
            [[[0.018, -0.004, -5.2], [0.002, 0.065, -18.0], [0.0001, 0.0022, 1.0]]],
            dtype=torch.float64,
        ).expand(4, 3, 3)
        ys = torch.full((4,), 12.0, dtype=torch.float64)
        bbox = torch.tensor(
            [[[420.0, 310.0, 36.0, 88.0], [425.0, 312.0, 36.0, 90.0]]],
            dtype=torch.float64,
        ).expand(4, 2, 4)
        jac = compute_homography_jacobian_world(bbox, hm, ys)
        uv = torch.stack((bbox[..., 0], bbox[..., 1] + 0.5 * bbox[..., 3]), dim=-1)
        eps = 1e-4
        du = (_map_uv_torch(uv + torch.tensor([eps, 0.0], dtype=torch.float64), hm, ys)[0] - _map_uv_torch(uv - torch.tensor([eps, 0.0], dtype=torch.float64), hm, ys)[0]) / (2 * eps)
        dv = (_map_uv_torch(uv + torch.tensor([0.0, eps], dtype=torch.float64), hm, ys)[0] - _map_uv_torch(uv - torch.tensor([0.0, eps], dtype=torch.float64), hm, ys)[0]) / (2 * eps)
        self.assertLess(float((jac[:, :, 0] - du).abs().max()), 1e-8)
        self.assertLess(float((jac[:, :, 1] - dv).abs().max()), 1e-8)

    def test_ray_kalman_exact_o2_equivariance_and_anisotropic_gain(self) -> None:
        torch.manual_seed(2026)
        bsz, t_obs = 8, 5
        t_grid = torch.arange(t_obs, dtype=torch.float64)[None, :] * 0.1
        true_p = torch.stack((0.6 * t_grid.expand(bsz, -1), torch.zeros((bsz, t_obs), dtype=torch.float64)), dim=-1)
        jac = torch.zeros((bsz, t_obs, 2, 2), dtype=torch.float64)
        jac[:, :, 0, 0] = 0.015
        jac[:, :, 1, 1] = 0.085
        bounce_v_px = torch.tensor([[0.0, 3.0, -3.5, 3.2, -3.4]], dtype=torch.float64).expand(bsz, -1)
        noisy_h = true_p.clone()
        noisy_h[:, :, 1] += bounce_v_px * 0.085
        bh = torch.tensor([[62.0, 68.0, 61.0, 67.0, 60.0]], dtype=torch.float64).expand(bsz, -1)

        out = anisotropic_ray_kalman_filter(noisy_h, jac_uv_world=jac, bbox_h=bh)
        fut_t = torch.arange(1, 31, dtype=torch.float64)[None, :, None] * 0.1
        gt_fut = torch.stack((0.6 * (0.4 + fut_t.squeeze(-1)).expand(bsz, -1), torch.zeros((bsz, 30), dtype=torch.float64)), dim=-1)
        pred_cv = noisy_h[:, -1:, :] + out["v_last"][:, None, :] * fut_t
        pred_rk = out["p_filt"][:, None, :] + out["v_filt"][:, None, :] * fut_t
        ade_cv = float(torch.linalg.vector_norm(pred_cv - gt_fut, dim=-1).mean())
        ade_rk = float(torch.linalg.vector_norm(pred_rk - gt_fut, dim=-1).mean())
        self.assertLess(ade_rk, 0.35 * ade_cv)

        th = 1.13
        q_ref = torch.tensor([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]], dtype=torch.float64) @ torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.float64)
        out_q = anisotropic_ray_kalman_filter(noisy_h @ q_ref.T, jac_uv_world=jac @ q_ref.T, bbox_h=bh)
        self.assertLess(float((out_q["p_filt"] - out["p_filt"] @ q_ref.T).abs().max()), 1e-12)
        self.assertLess(float((out_q["v_filt"] - out["v_filt"] @ q_ref.T).abs().max()), 1e-12)
        pvv_rot = q_ref[None] @ out["P_vv"] @ q_ref.T[None]
        self.assertLess(float((out_q["P_vv"] - pvv_rot).abs().max()), 1e-12)

    def test_causal_post_reappear_cusum_suppresses_flicker_and_heavy_occ(self) -> None:
        rng = np.random.default_rng(7)
        z_neg = rng.normal(-0.5, 0.25, size=(30, 3))
        occ_neg = np.zeros(30, dtype=np.int64); occ_neg[:6] = 2
        z_neg[3, 0] = 4.5
        z_neg[15, 0] = 1.8
        s_neg = causal_post_reappear_cusum_scores(z_neg, occ_neg, lam=0.70, drift_nu=0.20, w_cusum=0.65, gate_heavy_occ=True)
        self.assertTrue(bool(np.all(s_neg[:6] <= -1e3)))

        z_pos = np.full((15, 3), -0.4, dtype=np.float64)
        occ_pos = np.zeros(15, dtype=np.int64); occ_pos[:6] = 2
        z_pos[6:10, 0] = 1.45
        s_pos = causal_post_reappear_cusum_scores(z_pos, occ_pos, lam=0.70, drift_nu=0.20, w_cusum=0.65, gate_heavy_occ=True)
        self.assertGreater(float(s_pos[6:, 0].max()), float(s_neg[:, 0].max()) + 0.50)

    def test_rkr6_oblique_covariance_o2_equivariance_and_zero_stationary_drift(self) -> None:
        torch.manual_seed(27)
        net = O2RayKalmanRes6SlotNet(use_pose=True, use_interaction=True).eval()
        for p in net.parameters():
            if p.ndim >= 1:
                torch.nn.init.normal_(p, std=0.05)

        jac0 = torch.tensor([[[[0.014, 0.006], [-0.025, 0.082]]]], dtype=torch.float64).expand(3, 5, 2, 2)
        hz = torch.zeros(3, 5, 2, dtype=torch.float64)
        be = torch.randn(3, 5, 11)
        pvw = torch.randn(3, 5, 2, 2, dtype=torch.float64) * 0.15
        sk = torch.randn(3, 5, 1)
        qv, mp = torch.ones(3, 4), torch.ones(3)
        oz = net(hz, be, pvw, sk, qv, mp, jac_uv_world=jac0)
        self.assertLess(float(oz["paths"][:, 0].detach().abs().max()), 1e-12)

        h = torch.randn(4, 5, 2, dtype=torch.float64) * 0.30
        jac = jac0[:1].expand(4, 5, 2, 2).clone()
        bh = torch.full((4, 5), 64.0, dtype=torch.float64); bh[:, 2] = 71.0
        ac = torch.full((4, 5), 0.72, dtype=torch.float64); ac[:, 0] = 0.15
        ed = torch.zeros(4, 12, 8); ed[:, :2, :4] = torch.randn(4, 2, 4); ed[:, :2, 4] = 2.2; ed[:, :2, 5] = 0.65
        ic, nm = torch.ones(4, 12), torch.zeros(4, 12); nm[:, :2] = 1.0
        be4, pvw4, sk4, qv4, mp4 = be[:1].expand(4, -1, -1), pvw[:1].expand(4, -1, -1, -1), sk[:1].expand(4, -1, -1), qv[:1].expand(4, -1), mp[:1].expand(4)
        sc0 = compute_split_anatomical_covisibility(None, None, be4, pvw4, sk4, qv4)
        o0 = net(h, be4, pvw4, sk4, qv4, mp4, sc0, jac, bh, ac, ed, ic, nm)

        ang = 0.87
        q_m = torch.tensor([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]], dtype=torch.float64) @ torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.float64)
        pvw_r = torch.einsum("btvk,jk->btvj", pvw4, q_m)
        jac_r = torch.einsum("btvk,jk->btvj", jac, q_m)
        ed_r = ed.clone()
        ed_r[:, :, 0:2] = (ed[:, :, 0:2].double() @ q_m.T).float()
        ed_r[:, :, 2:4] = (ed[:, :, 2:4].double() @ q_m.T).float()
        sc_r = compute_split_anatomical_covisibility(None, None, be4, pvw_r, -sk4, qv4)
        o_r = net(h @ q_m.T, be4, pvw_r, -sk4, qv4, mp4, sc_r, jac_r, bh, ac, ed_r, ic, nm)
        self.assertLess(float(((o0["paths"] @ q_m.T)[:, [0, 1, 2, 4, 3, 5]] - o_r["paths"]).detach().abs().max()), 1e-6)


if __name__ == "__main__":
    unittest.main()
