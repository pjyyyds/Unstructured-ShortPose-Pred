"""Unit tests for Wave 17 2x2 (q x speed) factorial, <0.5m O(2) regime, & JRDB 72-pair fixed conformal."""
from __future__ import annotations

import unittest
import numpy as np
import torch

from scripts.run_pinns_matched_o2_multiduration import SITES
from scripts.run_wave17_rc1_2x2_factorial_and_lowdisp_regime import (
    FACT_ARMS,
    FactorialPoseO2Net,
    evaluate_lowdisp_regime_loso,
    fit_eval_2x2_factorial_fold,
)
from scripts.run_wave17_rc3_clean_chain_and_jrdb_conformal import (
    evaluate_jrdb_fixed_conformal,
    fit_clean_pinns_nopoly_fold,
)


class TestWave17FactorialAndConformal(unittest.TestCase):
    def test_2x2_factorial_and_lowdisp_o2_regime(self) -> None:
        rng = np.random.default_rng(21)
        sites = np.repeat(SITES, 12)
        clips = np.array([f"{s}_{k // 4}" for s in SITES for k in range(12)])
        n = len(sites)
        hist = torch.from_numpy(rng.normal(0.0, 0.06, size=(n, 5, 2))); hist[:, -1] = 0.0
        truth = torch.from_numpy(rng.normal(0.0, 0.3, size=(n, 30, 2)))
        dp = {
            "site": sites, "clip": clips, "history": hist, "truth": truth,
            "bio_even": torch.randn((n, 5, 11)), "pose_vec_world": torch.randn((n, 5, 2, 2), dtype=torch.float64) * 0.2,
            "signed_knee": torch.randn((n, 5, 1)) * 0.1, "quality": torch.full((n, 4), 0.8),
            "m_pose": torch.ones(n), "qmask_clear_partial": torch.ones(n),
            "strata": {"first_short_323": np.ones(n, dtype=bool), "first_short_clear": np.ones(n, dtype=bool)},
        }
        res_low = evaluate_lowdisp_regime_loso(dp)
        self.assertIn("Regime_M1_ground", res_low["first_short_323"]["models"])

        # Exact O(2) mirror equivariance check on FactorialPoseO2Net("M1_qv")
        net = FactorialPoseO2Net("M1_qv").eval()
        for p in net.parameters():
            if p.detach().abs().max() == 0:
                torch.nn.init.normal_(p, std=0.05)
        s = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.float64)
        o0 = net(hist[:4], dp["bio_even"][:4], dp["pose_vec_world"][:4], dp["signed_knee"][:4], dp["quality"][:4], dp["m_pose"][:4])
        os = net(hist[:4] @ s.T, dp["bio_even"][:4], dp["pose_vec_world"][:4] @ s.T, -dp["signed_knee"][:4], dp["quality"][:4], dp["m_pose"][:4])
        self.assertLess(float(((o0["paths"] @ s.T)[:, [0, 1, 2, 4, 3, 5]] - os["paths"]).detach().abs().max()), 5e-6)

        gd = {k: dp[k] for k in ("history", "truth", "bio_even", "pose_vec_world", "signed_knee", "quality", "m_pose")}
        of = fit_eval_2x2_factorial_fold(42, sites, gd, np.arange(36), np.arange(36, n), torch.device("cpu"), is_jrdb=False)
        self.assertEqual(set(of), set(FACT_ARMS))

    def test_clean_pinns_chain_and_jrdb_conformal(self) -> None:
        rng = np.random.default_rng(33)
        n = 48
        locs = np.array(["locA"] * 16 + ["locB"] * 16 + ["locC"] * 16)
        scs = np.array(["scA"] * 24 + ["scB"] * 24)
        dj = {
            "scene": scs, "location": locs,
            "history": torch.from_numpy(rng.normal(0.0, 0.15, size=(n, 2, 2))),
            "truth": torch.from_numpy(rng.normal(0.0, 0.4, size=(n, 7, 2))),
            "bio_even": torch.zeros((n, 5, 11)), "pose_vec_world": torch.zeros((n, 5, 2, 2)),
            "signed_knee": torch.zeros((n, 5, 1)), "quality": torch.full((n, 4), 0.7), "m_pose": torch.ones(n),
            "edges": torch.zeros((n, 12, 9)), "is_car": torch.zeros((n, 12)), "nbr_mask": torch.zeros((n, 12)),
            "y_act": torch.zeros((n, 5), dtype=torch.bool), "valid_act": torch.ones((n, 5), dtype=torch.bool),
        }
        dj["history"][:, -1] = 0.0
        from scripts.run_jrdb_rc2_rc3_graph_and_actions_5seed import _stage_gpu_dict
        gdj = _stage_gpu_dict(dj, torch.device("cpu"))
        oof = {a: rng.uniform(0.2, 1.0, size=(1, n, 3)) for a in ("M0", "M1", "M2b", "M3")}
        cf = evaluate_jrdb_fixed_conformal(oof, rng.uniform(0.3, 1.2, size=(n,)), dj, gdj, [42], torch.device("cpu"))
        self.assertIn("cross_location_72pair_fixed_7loc_predictor_no_retrain", cf)

        sites = np.repeat(SITES, 8)
        np_p = len(sites)
        dp_c = {
            "site": sites, "history": torch.zeros((np_p, 5, 2), dtype=torch.float64), "truth": torch.zeros((np_p, 30, 2), dtype=torch.float64),
            "bio_even": torch.zeros((np_p, 5, 11)), "pose_vec_world": torch.zeros((np_p, 5, 2, 2)),
            "signed_knee": torch.zeros((np_p, 5, 1)), "quality": torch.ones((np_p, 4)), "m_pose": torch.ones(np_p),
            "edges": torch.zeros((np_p, 12, 9)), "is_car": torch.zeros((np_p, 12)), "nbr_mask": torch.zeros((np_p, 12)),
            "y_act": torch.zeros((np_p, 5), dtype=torch.bool), "valid_act": torch.ones((np_p, 5), dtype=torch.bool),
        }
        mods = fit_clean_pinns_nopoly_fold(42, dp_c, np.arange(24), torch.device("cpu"))
        self.assertIn("M3", mods)


if __name__ == "__main__":
    unittest.main()
