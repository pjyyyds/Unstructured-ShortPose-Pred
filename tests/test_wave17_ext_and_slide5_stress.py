"""Unit tests for Wave 17 GPU2 external baselines (Raw/Frenet) and Slide 5 pose-noise/gate audit."""
from __future__ import annotations

import unittest
import numpy as np
import torch

from scripts.ppt_mainline_o2_models import PPTMainlineO2Forecaster
from scripts.run_wave17_ppt_ext_baselines_loso_5seed import (
    _make_ext_model,
    compute_so2_frenet_basis,
    rotate_local_to_world,
    rotate_world_to_local,
    weighted_shared_winner_loss,
)
from scripts.run_wave17_slide5_pose_noise_and_gate_response import eval_model_on_pack, summarize_gate_response
from src.losses.losses import JointPredictionLoss


class TestWave17ExtAndSlide5Stress(unittest.TestCase):
    def test_so2_frenet_roundtrip_and_loss_parity(self) -> None:
        torch.manual_seed(17)
        hist = torch.randn(12, 5, 2, dtype=torch.float64)
        hist[:3] = 0.0  # exact zero-displacement windows must still receive det=+1 SO(2) identity
        rot = compute_so2_frenet_basis(hist)
        eye = torch.eye(2, dtype=torch.float64)[None].expand(12, -1, -1)
        self.assertLess(float((rot @ rot.transpose(-1, -2) - eye).abs().max()), 1e-12)
        self.assertLess(float((torch.linalg.det(rot) - 1.0).abs().max()), 1e-12)
        vec = torch.randn(12, 6, 7, 2, dtype=torch.float64)
        rt = rotate_local_to_world(rotate_world_to_local(vec, rot), rot)
        self.assertLess(float((rt - vec).abs().max()), 1e-12)

        for seq_t, fut_t, dt in ((5, 30, 0.1), (2, 7, 0.4)):
            batch = {
                "x_m": torch.randn(8, seq_t, 9),
                "x_p": torch.randn(8, seq_t, 17, 6),
                "p_env": torch.randn(8, 16, 10, 11),
                "m_map": torch.ones(8, 1),
                "nbr_edges": torch.randn(8, 12, 9),
                "nbr_mask": (torch.rand(8, 12) > 0.3).float(),
                "nbr_type_oh": torch.zeros(8, 12, 8),
            }
            fut_xy = torch.randn(8, fut_t, 2)
            ref_loss_fn = JointPredictionLoss(use_action_loss=False)
            for kind in ("mcenet", "hst", "st"):
                net = _make_ext_model(kind, fut_steps=fut_t, dt=dt).eval()
                pred = net(batch)
                l_w = weighted_shared_winner_loss(pred, fut_xy, torch.ones(8))
                l_ref = ref_loss_fn(pred, fut_xy, torch.ones(8, fut_t))["loss"]
                self.assertAlmostEqual(float(l_w.detach()), float(l_ref.detach()), places=5)
                l_w.backward()

    def test_slide5_full_drop_shutoff_and_gate_summary(self) -> None:
        torch.manual_seed(23)
        n = 10
        hist = torch.randn(n, 5, 2, dtype=torch.float64) * 0.4
        truth = np.random.default_rng(5).normal(size=(n, 30, 2))
        pack_drop = {
            "bio_even": torch.zeros(n, 5, 11),
            "pose_vec_world": torch.zeros(n, 5, 2, 2, dtype=torch.float64),
            "signed_knee": torch.zeros(n, 5, 1),
            "quality": torch.zeros(n, 4),
            "m_pose": torch.zeros(n),
        }
        obs_dt = torch.full((n, 4), 0.1, dtype=torch.float64)
        fut_s = torch.arange(1, 31, dtype=torch.float64)[None].expand(n, -1) * 0.1
        ids = np.arange(n)
        for arm in ("M1a", "M1b", "M1"):
            net = PPTMainlineO2Forecaster(arm).eval()
            _, _, g0 = eval_model_on_pack(net, hist, pack_drop, obs_dt, fut_s, truth, ids, torch.device("cpu"))
            self.assertEqual(float(np.max(np.abs(g0))), 0.0)
        _, _, g1 = eval_model_on_pack(PPTMainlineO2Forecaster("M1a").eval(), hist, pack_drop, obs_dt, fut_s, truth, ids, torch.device("cpu"), force_mpose_one=True)
        self.assertTrue(np.allclose(g1, 1.0))
        mp_arr = np.array([0, 0] + [1] * (n - 2), dtype=np.float32)
        g_syn = np.tile(np.linspace(0.0, 0.8, n) * mp_arr, (5, 1))
        rep = summarize_gate_response(g_syn, np.column_stack([np.linspace(0.0, 1.0, n)] * 4), mp_arr, np.array([2, 2, 1, 1, 1, 0, 0, 0, 0, 0]))
        self.assertEqual(rep["max_abs_gate_when_mpose_zero"], 0.0)
        self.assertGreater(float(rep["spearman_gate_vs_q1_valid_joint_ratio"]), 0.99)

    def test_end_to_end_pinns_and_jrdb_evaluate_dataset_matrix_aggregation(self) -> None:
        import scripts.run_wave17_slide5_pose_noise_and_gate_response as s5
        from scripts.run_pinns_matched_o2_multiduration import SITES, paired_site_clip_ci

        old_seeds = s5.SEEDS
        s5.SEEDS = (42,)
        try:
            n = 16
            sites = np.array([SITES[i % 4] for i in range(n)], dtype=str)
            clips = np.array([f"{sites[i]}_c{i // 4}" for i in range(n)], dtype=str)
            hist = torch.randn(n, 5, 2, dtype=torch.float64) * 0.3
            truth = torch.randn(n, 30, 2, dtype=torch.float64) * 0.5
            pk = {
                "bio_even": torch.randn(n, 5, 11),
                "pose_vec_world": torch.randn(n, 5, 2, 2, dtype=torch.float64) * 0.2,
                "signed_knee": torch.randn(n, 5, 1) * 0.1,
                "quality": torch.full((n, 4), 0.9),
                "m_pose": torch.ones(n),
            }
            pk_drop = {**pk, "m_pose": torch.zeros(n), "quality": torch.zeros(n, 4)}
            packs = {"Test_Natural": pk, "Test_Jitter020": pk, "Test_LowerOcc": pk, "Test_FullDrop": pk_drop}
            qb = np.array([0 if i % 2 == 0 else 1 for i in range(n)])
            obs_dt = torch.full((n, 4), 0.1, dtype=torch.float64)
            fut_s = torch.arange(1, 31, dtype=torch.float64)[None].expand(n, -1) * 0.1
            res_p, _ = s5.evaluate_dataset_matrix("pinns", sites, clips, True, hist, truth, truth.numpy(), packs, qb, obs_dt, fut_s, 8, 1, 1, torch.device("cpu"))
            self.assertIn("ade_m", res_p["train_test_matrix"]["VarQTrain"]["Test_Jitter020"]["M1_minus_M1a"]["ci95"])
            res_j, _ = s5.evaluate_dataset_matrix("jrdb", sites, clips, False, hist, truth, truth.numpy(), packs, qb, obs_dt, fut_s, 8, 1, 1, torch.device("cpu"))
            self.assertIsNotNone(res_j["train_test_matrix"]["CleanTrain"]["Test_Jitter020"]["M1_minus_M1a"]["location_equal"]["ci95"])
            ci = paired_site_clip_ci(np.zeros((n, 2)), sites, clips)
            self.assertEqual(ci["mean"]["ade_m"], 0.0)
        finally:
            s5.SEEDS = old_seeds


if __name__ == "__main__":
    unittest.main()
