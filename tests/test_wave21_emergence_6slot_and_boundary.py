"""Unit tests for Wave 21: 6-slot O(2) precursor net, onset/6-cand audit, uncrop fix, & JAAD2 net."""
from __future__ import annotations

import unittest
import numpy as np
import torch

from scripts.run_wave21_jaad_deocclusion_and_eth_ucy_6dataset import JAADPrecursor2x2Net, eval_jaad2_5seed
from scripts.run_wave21_pinns_true_emergence_and_6slot_diag import run_2x2_6slot_and_baseline_eval
from scripts.run_wave21_polyline_crop_fix_and_dual_boundary import _dedup_world_polylines
from src.models.emergence_maneuver_6slot import (
    O2Maneuver6SlotPrecursorNet,
    audit_6candidate_generation_vs_selection,
    compute_action_labels_with_onset,
    horizon_and_post_onset_errors,
)


class TestWave21EmergenceAnd6Slot(unittest.TestCase):
    def test_o2_equivariance_onset_and_2x2_pipeline(self) -> None:
        torch.manual_seed(11)
        net = O2Maneuver6SlotPrecursorNet(use_pose=True, use_interaction=True).eval()
        hist = torch.zeros((4, 5, 2), dtype=torch.float64)
        hist[1:, :, 0] = torch.linspace(-0.4, 0.0, 5)
        bio = torch.randn(4, 5, 11)
        pvw = torch.randn(4, 5, 2, 2, dtype=torch.float64) * 0.25
        sk = torch.randn(4, 5, 1)
        qv = torch.rand(4, 4).clamp(0.4, 0.95)
        mp = torch.ones(4)
        ed = torch.zeros((4, 12, 9)); ed[:, 0, :6] = torch.tensor([2.5, 0.5, -1.2, -0.2, 2.55, 1.2])
        ic = torch.zeros((4, 12)); ic[:, 0] = 1.0
        nm = torch.zeros((4, 12)); nm[:, 0] = 1.0

        o0 = net(hist, bio, pvw, sk, qv, mp, ed, ic, nm)
        self.assertGreater(float(o0["paths"][0, 1, -1].detach().norm()), 1.0)
        for deg, refl in ((90.0, 1.0), (180.0, 1.0), (45.0, -1.0)):
            rad = np.deg2rad(deg)
            rm = torch.tensor([[np.cos(rad), -np.sin(rad)], [refl * np.sin(rad), refl * np.cos(rad)]], dtype=torch.float64)
            ed_r = ed.clone()
            ed_r[:, :, 0:2] = (ed[:, :, 0:2].double() @ rm.T).float()
            ed_r[:, :, 2:4] = (ed[:, :, 2:4].double() @ rm.T).float()
            o_r = net(hist @ rm.T, bio, pvw @ rm.T, sk * refl, qv, mp, ed_r, ic, nm)
            pr = (o0["paths"] @ rm.T)[:, [0, 1, 2, 4, 3, 5]] if refl < 0 else (o0["paths"] @ rm.T)
            self.assertLess(float((o_r["paths"] - pr).detach().abs().max()), 1e-6)

        # Synthetic Start & Stop onset + 6-candidate diagnostic + full 2x2 runner smoke test
        n = 24
        h_np = np.zeros((n, 5, 2), dtype=np.float64)
        h_np[12:, :, 0] = np.linspace(-0.45, 0.0, 5)[None, :]
        t_np = np.zeros((n, 30, 2), dtype=np.float64)
        for i in range(12):
            t_np[i, 5:, 0] = np.arange(1, 26) * 0.09  # Start at step 5!
        for i in range(12, n):
            t_np[i, :4, 0] = np.arange(1, 5) * 0.08
            t_np[i, 4:, 0] = 0.32  # Stop at step 4!
        hm, fm = np.ones(5, dtype=np.float32), np.ones(30, dtype=np.float32)
        ya, va, on = np.zeros((n, 5), dtype=bool), np.zeros((n, 5), dtype=bool), np.full((n, 5), -1, dtype=np.int64)
        for i in range(n):
            ya[i], va[i], on[i] = compute_action_labels_with_onset(h_np[i], hm, t_np[i], fm, dt=0.1)
        self.assertTrue(ya[:12, 0].all() and (on[:12, 0] >= 0).all())
        self.assertTrue(ya[12:, 1].all() and (on[12:, 1] >= 0).all())

        gd = {
            "history": torch.from_numpy(h_np), "truth": torch.from_numpy(t_np),
            "bio_even": torch.randn(n, 5, 11), "pose_vec_world": torch.randn(n, 5, 2, 2, dtype=torch.float64) * 0.2,
            "signed_knee": torch.randn(n, 5, 1), "quality": torch.full((n, 4), 0.85), "m_pose": torch.ones(n),
            "edges": torch.zeros((n, 12, 9)), "is_car": torch.zeros((n, 12)), "nbr_mask": torch.zeros((n, 12)),
            "y_act": torch.from_numpy(ya), "valid_act": torch.from_numpy(va),
            "obs_dt": torch.full((n, 4), 0.1, dtype=torch.float64),
            "fut_s": torch.arange(1, 31, dtype=torch.float64)[None].expand(n, -1) * 0.1,
        }
        grp = np.array([f"s{i % 3}" for i in range(n)])
        cv_r = ((h_np[:, -1:] - h_np[:, -2:-1]) / 0.1) * (np.arange(1, 31)[None, :, None] * 0.1)
        res = run_2x2_6slot_and_baseline_eval(
            "unit", grp, grp, True, gd, h_np, t_np, cv_r, ya, va, on,
            {"all": np.ones(n, dtype=bool)}, np.ones(n, dtype=bool),
            0.1, 12, (2, 2, 2, 2), 3, (2e-3, 1e-3, 1e-3, 1e-3), 2e-3, [42], torch.device("cpu"), True,
        )
        self.assertIn("S6_Full_pose_nbr", res["strata"]["all"]["models"])
        self.assertIn("Start", res["candidate_generation_vs_selection_seed0"]["S6_Full_pose_nbr"])

        # Polyline dedup & JAAD2 2x2 smoke test
        p1 = np.array([[0.0, 0.0], [2.0, 0.0]])
        self.assertEqual(len(_dedup_world_polylines([(p1, np.zeros(8)), (p1[::-1] + 0.01, np.zeros(8))])), 1)
        jd = {
            "hist": np.random.randn(20, 5, 2) * 0.05, "truth": np.random.randn(20, 20, 2) * 0.2,
            "pose5": np.random.rand(20, 5, 8).astype(np.float32), "veh5": np.random.rand(20, 5, 3).astype(np.float32),
            "y_act": np.ones((20, 4), dtype=bool), "v_act": np.ones((20, 4), dtype=bool),
            "onset": np.zeros((20, 5), dtype=np.int64), "split": np.array(["train"] * 10 + ["val"] * 5 + ["test"] * 5),
        }
        jr = eval_jaad2_5seed(jd, [42], torch.device("cpu"))
        self.assertEqual(jr["err_arms"]["J_Full_pose_veh"].shape, (5, 6))


if __name__ == "__main__":
    unittest.main()
