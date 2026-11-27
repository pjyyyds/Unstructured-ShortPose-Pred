"""Unit tests for Wave 23 PINNS homography occlusion, iso-Test-FPR lead-time, and O(2) precursor ranker."""
from __future__ import annotations

import unittest
import numpy as np
import torch

from src.data.pinns_camera_occlusion import (
    camera_box_occludes_pedestrian,
    fit_verified_ground_to_camera_homography,
    project_homog,
    project_vehicle_3d_to_camera_box,
)
from src.models.apa_precursor_leadtime import (
    evaluate_iso_test_fpr_and_unconflated_leadtime,
    paired_iso_fpr_comparison,
)
from src.models.cond_precursor_ranker import (
    ParityExactCondPrecursorNet,
    build_conditioned_precursor_features,
    select_medoid_expected_ade,
    summarize_6candidate_5seed,
)


class TestWave23Modules(unittest.TestCase):
    def test_camera_homography_vehicle_projection_and_depth_order(self) -> None:
        calib = {
            "pts_2d": [[200.0, 300.0], [1000.0, 300.0], [1150.0, 800.0], [50.0, 800.0]],
            "pts_3d": [[0.0, 20.0], [16.0, 20.0], [16.0, 2.0], [0.0, 2.0]],
            "y_min": 0.0,
            "y_max": 22.0,
        }
        h_g2i, y_sum, _ = fit_verified_ground_to_camera_homography(calib)
        # Pedestrian at (8m, 10m); conjugated y_label = 22 - 10 = 12
        ped_uv = project_homog(np.array([[8.0, 10.0]]), h_g2i)[0]
        ped_rows = np.array([[ped_uv[0], ped_uv[1], 42.0, 110.0, 8.0, 22.0 - 10.0]])
        h_g2i, y_conj, med_px = fit_verified_ground_to_camera_homography(calib, ped_rows)
        self.assertLess(med_px, 1e-4)
        self.assertAlmostEqual(float(y_conj), 22.0)

        ped_box = (ped_uv[0] - 21.0, ped_uv[1] - 110.0, ped_uv[0] + 21.0, ped_uv[1])
        # Car IN FRONT of pedestrian (closer to camera -> smaller ground Y = 8.5m -> label wy = 13.5m)
        car_front_box, _ = project_vehicle_3d_to_camera_box(8.0, 22.0 - 8.5, "Car", h_g2i, y_conj)
        self.assertTrue(camera_box_occludes_pedestrian(car_front_box, ped_box))
        # Car BEHIND pedestrian (farther from camera -> larger ground Y = 15.0m -> label wy = 7.0m)
        car_back_box, _ = project_vehicle_3d_to_camera_box(8.0, 22.0 - 15.0, "Car", h_g2i, y_conj)
        self.assertFalse(camera_box_occludes_pedestrian(car_back_box, ped_box))

    def test_iso_test_fpr_separates_recall_from_hit_lead_time(self) -> None:
        # 4 Start events (each 3 tau steps: -0.6, -0.2, +0.2) + 40 negative control windows
        probs_a = np.zeros((52, 3), dtype=np.float64)
        probs_b = np.zeros((52, 3), dtype=np.float64)
        rng = np.random.default_rng(7)
        probs_a[:40] = rng.uniform(0.0, 0.30, size=(40, 3))
        probs_b[:40] = rng.uniform(0.0, 0.60, size=(40, 3))  # Higher raw scores on negatives!
        rel_tau = np.concatenate([np.zeros(40), np.tile([-0.6, -0.2, 0.2], 4)])
        eid = np.concatenate([-np.ones(40, dtype=int), np.repeat([1, 2, 3, 4], 3)])
        ecls = np.zeros(52, dtype=int)
        is_neg = np.arange(52) < 40
        # Model A hits only event 1 at -0.2s; Model B hits event 1 at -0.6s and events 2,3 at -0.4s/-0.2s
        probs_a[41, 0] = 0.95
        probs_b[[40, 44, 47], 0] = 0.98
        ra = evaluate_iso_test_fpr_and_unconflated_leadtime(probs_a, rel_tau, eid, ecls, is_neg, target_test_fpr=0.05)
        rb = evaluate_iso_test_fpr_and_unconflated_leadtime(probs_b, rel_tau, eid, ecls, is_neg, target_test_fpr=0.05)
        self.assertAlmostEqual(ra["Start"]["empirical_test_fpr"], rb["Start"]["empirical_test_fpr"], places=6)
        self.assertEqual(ra["Start"]["pre_onset_hit_count"], 1)
        self.assertEqual(ra["Start"]["pre_onset_miss_count"], 3)
        self.assertEqual(rb["Start"]["pre_onset_hit_count"], 3)
        comp = paired_iso_fpr_comparison(rb, ra)
        self.assertAlmostEqual(comp["Start"]["recall_diff_mean"], 0.50)
        self.assertEqual(comp["Start"]["n_both_pre_hit"], 1)
        self.assertAlmostEqual(comp["Start"]["common_hit_lead_diff_ms_mean"], 400.0)

    def test_parity_exact_precursor_net_and_5seed_summary(self) -> None:
        torch.manual_seed(3)
        n = 6
        h = torch.randn(n, 5, 2, dtype=torch.float64)
        be = torch.randn(n, 5, 10)
        pw = torch.randn(n, 5, 2, 2, dtype=torch.float64)
        sk = torch.randn(n, 5)
        q = torch.rand(n, 4)
        mp = torch.ones(n)
        ed = torch.randn(n, 4, 9)
        ic = (torch.rand(n, 4) > 0.5).float()
        nm = torch.ones(n, 4)
        dt = torch.full((n, 4), 0.1, dtype=torch.float64)
        net = ParityExactCondPrecursorNet().eval()
        base = torch.zeros(n, 5)
        f0 = build_conditioned_precursor_features(h, be, pw, sk, q, mp, ed, ic, nm, dt)
        out0 = net(f0, base)
        # Apply Y-reflection: h[...,1]->-y, pw[...,1]->-y, sk->-sk
        hm = h.clone(); hm[..., 1] *= -1.0
        pwm = pw.clone(); pwm[..., 1] *= -1.0
        edm = ed.clone(); edm[..., 1] *= -1.0; edm[..., 3] *= -1.0
        fm = build_conditioned_precursor_features(hm, be, pwm, -sk, q, mp, edm, ic, nm, dt)
        outm = net(fm, base)
        self.assertLess(float((out0[:, [0, 1, 4]] - outm[:, [0, 1, 4]]).detach().abs().max()), 1e-6)
        self.assertLess(float((out0[:, 2] - outm[:, 3]).detach().abs().max()), 1e-6)
        idx = select_medoid_expected_ade(np.zeros((4, 6, 10, 2)), np.zeros((4, 6)), np.full((4, 5), 0.2), np.ones((4, 5), dtype=bool))
        self.assertEqual(idx.shape, (4,))
        s5 = summarize_6candidate_5seed([{"Start": {"n_true": 9, "n_neg_eligible": 39, "any6_generated_hits": 9, "top1_selected_hits": k, "oracle6_beats_cv_by_2cm": 5, "top1_neg_false_positives": 1}} for k in range(5)])
        self.assertAlmostEqual(s5["Start"]["top1_selected_hits_5seed_mean"], 2.0)

    def test_jaad_autopose_factorial_and_pinns_wave23_smoke(self) -> None:
        from scripts.run_wave23_jaad_autopose_and_isofpr_leadtime import (
            JAADModalityFactorialNet,
            _fit_and_predict_lead_arms,
            _kpts5_to_autopose,
            _score_lead_operating_points,
        )
        from scripts.run_wave23_pinns_video_audit_and_6slot_ranker import eval_cohort_wave23_5seed

        rng = np.random.default_rng(11)
        k5 = rng.uniform(10.0, 90.0, size=(5, 17, 3)).astype(np.float32)
        k5[:, :, 2] = 0.85
        ap5, aq2 = _kpts5_to_autopose(k5, 70.0)
        self.assertEqual(ap5.shape, (5, 12))
        self.assertEqual(aq2.shape, (2,))

        net_j = JAADModalityFactorialNet(True, True, True)
        oj = net_j(torch.zeros(4, 5, 2, dtype=torch.float64), torch.randn(4, 5, 12), torch.rand(4, 2), torch.randn(4, 5, 8), torch.randn(4, 5, 3))
        self.assertEqual(oj["paths"].shape, (4, 4, 20, 2))

        ds_s = {
            "kin": rng.normal(size=(24, 5, 5)).astype(np.float32),
            "auto_pose": rng.normal(size=(24, 5, 12)).astype(np.float32),
            "auto_q": rng.uniform(0.3, 0.9, size=(24, 2)).astype(np.float32),
            "veh": rng.normal(size=(24, 3)).astype(np.float32),
            "y": (rng.uniform(size=(24, 3)) > 0.7).astype(np.float32),
            "tau_s": np.tile([-0.4, 0.0, 0.2], 8).astype(np.float64),
            "event_id": np.repeat(np.arange(8), 3).astype(np.int64),
            "event_cls": np.repeat([0, 1, 2, 0, -1, -1, -1, -1], 3).astype(np.int64),
            "is_neg": np.repeat([False, False, False, False, True, True, True, True], 3),
            "split": np.repeat(["train", "train", "val", "val", "test", "test", "test", "test"], 3),
        }
        arms = [("M0", ds_s["auto_pose"], ds_s["auto_q"], False, False), ("Auto", ds_s["auto_pose"], ds_s["auto_q"], True, False)]
        p_va, p_te = _fit_and_predict_lead_arms(ds_s["kin"], arms, ds_s["veh"], ds_s["y"], ds_s["tau_s"], ds_s["is_neg"], ds_s["split"], [42], torch.device("cpu"), epochs=2)
        sc = _score_lead_operating_points(p_va, p_te, ds_s, {"all": ~ds_s["is_neg"][ds_s["split"] == "test"]}, [("auto_vs_m0", "Auto", "M0")])
        self.assertIn("fpr_0p050", sc["iso_test_fpr"])

        # Smoke test eval_cohort_wave23_5seed on 12 synthetic windows across 3 sites
        n = 12
        gd = {
            "history": torch.randn(n, 5, 2, dtype=torch.float64),
            "truth": torch.randn(n, 10, 2, dtype=torch.float64),
            "bio_even": torch.randn(n, 5, 11),
            "pose_vec_world": torch.randn(n, 5, 2, 2, dtype=torch.float64),
            "signed_knee": torch.randn(n, 5, 1),
            "quality": torch.rand(n, 4),
            "m_pose": torch.ones(n),
            "edges": torch.randn(n, 12, 9),
            "is_car": torch.zeros(n, 12),
            "nbr_mask": torch.ones(n, 12),
            "obs_dt": torch.full((n, 4), 0.1, dtype=torch.float64),
            "fut_s": torch.arange(1, 11, dtype=torch.float64)[None, :].expand(n, 10) * 0.1,
        }
        grp = np.array(["s0"] * 4 + ["s1"] * 4 + ["s2"] * 4)
        unit = np.array([f"c{k // 2}" for k in range(n)])
        ya = np.zeros((n, 5), dtype=bool); ya[[0, 5], 0] = True; ya[[1, 6], 1] = True; ya[2:, 4] = True
        va = np.ones((n, 5), dtype=bool)
        on = np.zeros((n, 5), dtype=np.int64)
        res = eval_cohort_wave23_5seed("SMOKE", grp, unit, gd, gd["history"].numpy(), gd["truth"].numpy(), ya, va, on, {"all": np.ones(n, dtype=bool)}, [42], torch.device("cpu"), ep_slot=2)
        self.assertIn("S6_Full_CondMedoid", res["strata"]["all"]["models"])

        # Also smoke-test 2-frame JRDB shape (T_obs=2, obs_dt shape [N, 1])
        gd2 = {
            "history": torch.randn(n, 2, 2, dtype=torch.float64),
            "truth": torch.randn(n, 7, 2, dtype=torch.float64),
            "bio_even": torch.randn(n, 2, 11),
            "pose_vec_world": torch.randn(n, 2, 2, 2, dtype=torch.float64),
            "signed_knee": torch.randn(n, 2, 1),
            "quality": torch.rand(n, 4),
            "m_pose": torch.ones(n),
            "edges": torch.randn(n, 12, 9),
            "is_car": torch.zeros(n, 12),
            "nbr_mask": torch.ones(n, 12),
            "obs_dt": torch.full((n, 1), 0.4, dtype=torch.float64),
            "fut_s": torch.arange(1, 8, dtype=torch.float64)[None, :].expand(n, 7) * 0.4,
        }
        res2 = eval_cohort_wave23_5seed("JRDB_SMOKE", grp, unit, gd2, gd2["history"].numpy(), gd2["truth"].numpy(), ya, va, on, {"all": np.ones(n, dtype=bool)}, [42], torch.device("cpu"), dt=0.4, ep_slot=2, is_pinns=False)
        self.assertIn("S6_Full_CondMedoid", res2["strata"]["all"]["models"])


if __name__ == "__main__":
    unittest.main()
