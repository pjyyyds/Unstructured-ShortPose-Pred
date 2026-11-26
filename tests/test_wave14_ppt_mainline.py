"""Unit tests for Wave 14 PPT mainline O(2) M0-M3 chain, fixed-predictor conformal, & JRDB RC1."""
from __future__ import annotations

import unittest
import numpy as np
import torch

from scripts.ppt_mainline_o2_models import (
    PPT_ARMS,
    PPTMainlineO2Forecaster,
    build_ground_pose_o2_inputs,
    polyline_crossing_rate_np,
)
from scripts.run_jrdb_rc1_pose_quality_and_initiation_5seed import (
    _loc_bootstrap_ci,
    build_jrdb_3d_pose_o2_features,
    eval_jrdb_arm,
    fit_jrdb_rc1_fold,
)
from scripts.run_ppt_mainline_o2_m0_m3_and_fixed_conformal import (
    evaluate_fixed_predictor_conformal,
    fit_ppt_chain_fold,
    predict_arm_numpy,
)


class TestWave14PPTMainline(unittest.TestCase):
    def test_o2_equivariance_and_zero_fallbacks(self) -> None:
        torch.manual_seed(11)
        b = 4
        hist = torch.randn(b, 5, 2, dtype=torch.float64) * 0.25
        hist[:, -1] = 0.0
        hist[0] = 0.0
        kpts = torch.rand(b, 5, 17, 3) * 80.0
        kpts[..., 2] = 0.9
        bbox = torch.tensor([[[40.0, 50.0, 25.0, 70.0]] * 5] * b)
        h_m = torch.eye(3, dtype=torch.float64)[None].repeat(b, 1, 1) * 0.01
        h_m[:, 2, 2] = 1.0
        gp = build_ground_pose_o2_inputs(hist, kpts, bbox, h_m, torch.zeros(b, dtype=torch.float64))
        qv = torch.full((b, 4), 0.8)
        pe = torch.zeros(b, 16, 10, 11); pe[:, 0, :, 0] = 0.4; pe[:, 0, :, 1] = torch.linspace(-1, 1, 10); pe[:, 0, :, 10] = 1.0
        mm = torch.ones(b)
        ed = torch.randn(b, 12, 9); ed[:, :, 4] = 3.0; ed[:, :, 6] = 1.0
        car, nmsk = torch.zeros(b, 12), torch.ones(b, 12)

        net = PPTMainlineO2Forecaster("M3").eval()
        # Zero-init fallback check: blackout & empty graph & m_map=0 equal M0 at init
        o_full = net(hist, gp["bio_even"], gp["pose_vec_world"], gp["signed_knee"], qv, torch.zeros(b), pe, torch.zeros(b), ed, car, torch.zeros_like(nmsk))
        o_clean = net(hist)
        self.assertLess(float((o_full["paths"] - o_clean["paths"]).detach().abs().max()), 1e-12)

        for p in net.parameters():
            if p.detach().abs().max() == 0:
                torch.nn.init.normal_(p, std=0.05)
        for arm in ("M0", "M1a", "M1b", "M1", "M2a", "M2b_no_vrad", "M2b", "M2", "M3a", "M3b", "M3"):
            na = PPTMainlineO2Forecaster(arm).eval()
            na.load_state_dict(net.state_dict())
            o0 = na(hist, gp["bio_even"], gp["pose_vec_world"], gp["signed_knee"], qv, gp["m_pose"], pe, mm, ed, car, nmsk)
            s = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.float64); sf = s.float()
            pe_s = pe.clone(); pe_s[:, :, :, 0:2] = pe[:, :, :, 0:2] @ sf.T
            ed_s = ed.clone(); ed_s[:, :, 0:2] = ed[:, :, 0:2] @ sf.T; ed_s[:, :, 2:4] = ed[:, :, 2:4] @ sf.T
            os_ = na(hist @ s.T, gp["bio_even"], gp["pose_vec_world"] @ s.T, -gp["signed_knee"], qv, gp["m_pose"], pe_s, mm, ed_s, car, nmsk)
            self.assertLess(float(((o0["paths"] @ s.T)[:, [0, 1, 2, 4, 3, 5]] - os_["paths"]).detach().abs().max()), 5e-6)
        self.assertGreaterEqual(polyline_crossing_rate_np(np.tile(np.linspace(0.1, 1.0, 30)[:, None], (b, 1, 2)), pe.numpy()), 0.0)
    def test_jrdb_rc1_and_fixed_conformal_end_to_end(self) -> None:
        rng = np.random.default_rng(9)
        n = 48
        j3d = rng.normal(0.0, 0.2, size=(n, 5, 33, 3))
        j3d[:8] = np.nan  # Heavy/Missing bucket
        j3d[8:16, :2] = np.nan  # Partial bucket
        pf = build_jrdb_3d_pose_o2_features(j3d)
        self.assertEqual(set(pf["q_buckets"].tolist()), {0, 1, 2})
        hist = torch.from_numpy(rng.normal(0.0, 0.2, size=(n, 2, 2)))
        hist[:, -1] = 0.0
        truth = torch.from_numpy(rng.normal(0.0, 0.5, size=(n, 7, 2)))
        locs = np.array(["locA"] * 16 + ["locB"] * 16 + ["locC"] * 16)
        data_j = {"history": hist, "truth": truth, "location": locs, **pf}
        mods_j = fit_jrdb_rc1_fold(42, data_j, np.arange(32), torch.device("cpu"))
        ev_m1 = eval_jrdb_arm(mods_j["M1"], data_j, np.arange(32, 48), torch.device("cpu"))
        self.assertEqual(ev_m1.shape, (16, 7))
        ci = _loc_bootstrap_ci(rng.normal(size=(n, 2)), locs, n_boot=200)
        self.assertIn("ci95", ci)
        from scripts.run_dair_o2_bioeq_external import person_bootstrap_ci
        pci = person_bootstrap_ci(rng.normal(size=(n, 2)), np.ones(n, dtype=bool), np.array([f"p{i // 2}" for i in range(n)]))
        self.assertIn("person_equal", pci)

        # Synthetic 4-site PINNS cohort for fixed-predictor split-conformal
        from scripts.run_pinns_matched_o2_multiduration import SITES
        sites = np.repeat(SITES, 16)
        clips = np.array([f"{s}_{k // 4}" for s in SITES for k in range(16)])
        n_p = len(sites)
        hp = torch.from_numpy(rng.normal(0.0, 0.3, size=(n_p, 5, 2))); hp[:, -1] = 0.0
        tp = torch.from_numpy(rng.normal(0.0, 0.6, size=(n_p, 30, 2)))
        data_p = {
            "site": sites, "clip": clips, "history": hp, "truth": tp,
            "bio_even": torch.zeros((n_p, 5, 11)), "pose_vec_world": torch.zeros((n_p, 5, 2, 2)),
            "signed_knee": torch.zeros((n_p, 5, 1)), "quality": torch.full((n_p, 4), 0.7),
            "m_pose": torch.ones(n_p), "p_env": torch.zeros((n_p, 16, 10, 11)), "m_map": torch.ones(n_p),
            "edges": torch.zeros((n_p, 12, 9)), "is_car": torch.zeros((n_p, 12)), "nbr_mask": torch.zeros((n_p, 12)),
            "y_act": torch.zeros((n_p, 5), dtype=torch.bool), "valid_act": torch.ones((n_p, 5), dtype=torch.bool),
        }
        oof_tb = rng.uniform(0.3, 1.5, size=(1, n_p, 3))
        cv_tb = rng.uniform(0.5, 2.0, size=(n_p,))
        conf = evaluate_fixed_predictor_conformal(oof_tb, cv_tb, data_p, [42], torch.device("cpu"))
        self.assertIn("within_site_clip_split_fixed_3site_predictor", conf)
        self.assertIn("cross_site_12pair_fixed_2site_predictor_no_retrain", conf)

        from scripts.run_pinns_hybrid_action_events_5seed import score_action_matrix
        ya_s = rng.integers(0, 2, size=(n_p, 5)).astype(bool)
        va_s = np.ones((n_p, 5), dtype=bool)
        pr_s = rng.uniform(0.1, 0.9, size=(n_p, 5))
        sc_a = score_action_matrix(ya_s, va_s, pr_s, sites, np.full(5, 0.5, dtype=np.float64))
        self.assertIn("site_equal", sc_a)

        from scripts.run_ppt_rc1_3x3_duration_quality_matrix import eval_rc1_dur, fit_rc1_dur_fold
        h20 = torch.from_numpy(rng.normal(0.0, 0.25, size=(n_p, 20, 2))); h20[:, -1] = 0.0
        d3x3 = {
            "site": sites, "clip": clips, "hist20": h20, "truth": tp,
            "per_dur": {
                10: {
                    "bio_even": torch.zeros((n_p, 10, 11)), "pose_vec_world": torch.zeros((n_p, 10, 2, 2)),
                    "signed_knee": torch.zeros((n_p, 10, 1)), "quality": torch.full((n_p, 4), 0.7), "m_pose": torch.ones(n_p),
                }
            },
        }
        m_d10 = fit_rc1_dur_fold(42, d3x3, 10, np.arange(48), torch.device("cpu"))
        ev10 = eval_rc1_dur(m_d10["M1"], d3x3, 10, np.arange(48, n_p), torch.device("cpu"))
        self.assertEqual(ev10.shape, (16, 7))

    def test_homography_ego_polylines_and_unconfounded_gate(self) -> None:
        from scripts.ppt_mainline_o2_models import fit_pinns_ground_homography, verify_pinns_ground_and_polyline_diagnostics
        from src.data.scene_graph import build_local_polylines

        calib = {
            "pts_2d": [[100.0, 100.0], [300.0, 100.0], [300.0, 300.0], [100.0, 300.0]],
            "pts_3d": [[10.0, 20.0], [30.0, 20.0], [30.0, 0.0], [10.0, 0.0]],
            "y_min": 0.0, "y_max": 20.0,
        }
        hm, ys = fit_pinns_ground_homography(calib)
        self.assertEqual(hm.shape, (3, 3))
        b = 8
        hist = torch.linspace(-0.4, 0.0, 5, dtype=torch.float64)[None, :, None].repeat(b, 1, 2)
        hist[:, :, 0] = 0.0  # Moving along +Y (towards camera: increasing v)
        kpts = torch.zeros((b, 5, 17, 3), dtype=torch.float32); kpts[..., 2] = 0.9
        bbox = torch.tensor([[[200.0, 165.0, 30.0, 70.0]] * 5] * b, dtype=torch.float32)  # foot at (200, 200) -> world (20, 10)
        # Half samples face +Y (du > 0), half face +X (du == 0, dv != 0)
        kpts[:4, :, 5, :2] = torch.tensor([205.5, 145.0]); kpts[:4, :, 6, :2] = torch.tensor([197.5, 145.0])
        kpts[4:, :, 5, :2] = torch.tensor([202.5, 142.0]); kpts[4:, :, 6, :2] = torch.tensor([200.5, 148.0])
        kpts[:, :, 11, :2] = torch.tensor([201.0, 170.0]); kpts[:, :, 12, :2] = torch.tensor([199.0, 170.0])

        with self.assertRaises(ValueError):
            build_ground_pose_o2_inputs(hist, kpts, bbox, torch.zeros((b, 3, 3), dtype=torch.float64), torch.full((b,), ys, dtype=torch.float64))

        hm_t = torch.from_numpy(hm)[None].repeat(b, 1, 1)
        calib2 = {
            "pts_2d": [[50.0, 80.0], [250.0, 80.0], [250.0, 280.0], [50.0, 280.0]],
            "pts_3d": [[-30.0, 10.0], [-10.0, 10.0], [-10.0, -10.0], [-30.0, -10.0]],
            "y_min": -10.0, "y_max": 10.0,
        }
        hm2, ys2 = fit_pinns_ground_homography(calib2)
        hm_t[4:] = torch.from_numpy(hm2)
        ys_t = torch.tensor([ys] * 4 + [ys2] * 4, dtype=torch.float64)
        gp = build_ground_pose_o2_inputs(hist, kpts, bbox, hm_t, ys_t)
        self.assertTrue(bool((torch.linalg.vector_norm(gp["pose_vec_world"][:, :, 0], dim=-1) > 0.05).all()))
        self.assertTrue(bool((torch.linalg.vector_norm(gp["pose_vec_world"][:, :, 1], dim=-1) > 0.01).all()))

        pt0_ego = np.vstack((np.tile([[20.0, 10.0]], (4, 1)), np.tile([[-15.0, 2.0]], (4, 1))))
        polys = [(np.array([[20.5, 9.0], [20.5, 11.0]], dtype=np.float64), np.ones(8, dtype=np.float32))]
        p_ego = np.stack([build_local_polylines(polys, pt0_ego[i], True)[0] for i in range(b)])
        p_bug = np.stack([build_local_polylines(polys, np.zeros(2, dtype=np.float32), True)[0] for i in range(b)])
        self.assertGreater(float(p_ego[:, :, :, 10].sum()), 0.0)
        self.assertEqual(float(p_bug[:, :, :, 10].sum()), 0.0)
        diag = verify_pinns_ground_and_polyline_diagnostics(hist, bbox, gp["pose_vec_world"], hm_t, ys_t, pt0_ego_all=pt0_ego)
        self.assertLess(diag["footpoint_to_published_xy_median_err_m"], 1e-9)
        self.assertGreater(diag["moving_coronal_velocity_mean_cos"], 0.4)

        # Single-factor M1 vs M1b check across slow and fast speeds when q_vec == 0
        m1b = PPTMainlineO2Forecaster("M1b").eval()
        m1 = PPTMainlineO2Forecaster("M1").eval()
        m1.load_state_dict(m1b.state_dict())
        with torch.no_grad():
            m1.gate_with_q[0].weight.zero_()
            m1.gate_with_q[0].weight[:, :72].copy_(m1b.gate_no_q[0].weight)
            m1.gate_with_q[0].bias.copy_(m1b.gate_no_q[0].bias)
            m1.gate_with_q[2].weight.copy_(m1b.gate_no_q[2].weight)
            m1.gate_with_q[2].bias.copy_(m1b.gate_no_q[2].bias)
        hist_fast = hist.clone(); hist_fast[0] *= 6.0  # 2.4 m in 0.4 s = 6.0 m/s
        ob = m1b(hist_fast, gp["bio_even"], gp["pose_vec_world"], gp["signed_knee"], torch.zeros((b, 4)), gp["m_pose"])
        oq = m1(hist_fast, gp["bio_even"], gp["pose_vec_world"], gp["signed_knee"], torch.zeros((b, 4)), gp["m_pose"])
        self.assertLess(float((ob["gate"] - oq["gate"]).abs().max()), 1e-6)
        self.assertLess(float((ob["paths"] - oq["paths"]).abs().max()), 1e-9)


if __name__ == "__main__":
    unittest.main()
