"""End-to-end unit tests for Wave 18 conflict-gated interaction & 5-action Platt calibration/reranking."""
from __future__ import annotations

import unittest
import numpy as np
import torch

from scripts.ppt_mainline_o2_models import PPTMainlineO2Forecaster
from scripts.run_ppt_mainline_o2_m0_m3_and_fixed_conformal import _copy_matching
import scripts.run_wave18_action_calibration_and_rerank_5seed as w18a
import scripts.run_wave18_conflict_gated_interaction_5seed as w18c


class TestWave18ConflictAndActionRerank(unittest.TestCase):
    def test_conflict_gate_zero_leakage_on_isolated_and_end_to_end(self) -> None:
        torch.manual_seed(31)
        m1 = PPTMainlineO2Forecaster("M1").eval()
        cg = w18c.ConflictGatedM2bForecaster(use_conflict_gate=True).eval()
        _copy_matching(cg, m1)
        with torch.no_grad():
            cg.graph_proj.weight.fill_(0.5)
            cg.edge_mlp[2].bias.fill_(3.5)
        n = 12
        hist = torch.randn(n, 5, 2, dtype=torch.float64) * 0.4
        be = torch.randn(n, 5, 11)
        pvw = torch.randn(n, 5, 2, 2, dtype=torch.float64) * 0.2
        sk = torch.randn(n, 5, 1) * 0.1
        qv = torch.full((n, 4), 0.85)
        mp = torch.ones(n)
        edg = torch.randn(n, 12, 9).abs()
        car = torch.zeros(n, 12)
        z_msk = torch.zeros(n, 12)
        o_m1 = m1(hist, be, pvw, sk, qv, mp)
        o_cg = cg(hist, be, pvw, sk, qv, mp, edges=edg, is_car=car, nbr_mask=z_msk)
        self.assertEqual(float(o_cg["w_risk"].detach().abs().max()), 0.0)
        self.assertLess(float((o_m1["paths"] - o_cg["paths"]).detach().abs().max()), 1e-12)
        self.assertLess(float((o_m1["logits"] - o_cg["logits"]).detach().abs().max()), 1e-6)

        old_s = w18c.SEEDS
        w18c.SEEDS = (42,)
        try:
            sites = np.array(["eth", "ind", "chn", "usa"] * 3, dtype=str)
            clips = np.array([f"{sites[i]}_c0" for i in range(n)], dtype=str)
            truth = torch.randn(n, 30, 2, dtype=torch.float64) * 0.5
            nm = torch.zeros(n, 12)
            nm[3, 0] = 1.0  # car conflict present in ONLY 1 site ('usa') to stress-test safe_site_clip_ci!
            car[3, 0] = 1.0
            edg[3, 0, 4], edg[3, 0, 5], edg[3, 0, 6] = 3.0, 1.2, 1.0
            gd = {"history": hist, "truth": truth, "bio_even": be, "pose_vec_world": pvw, "signed_knee": sk, "quality": qv, "m_pose": mp, "edges": edg, "is_car": car, "nbr_mask": nm, "obs_dt": torch.full((n, 4), 0.1, dtype=torch.float64), "fut_s": torch.arange(1, 31, dtype=torch.float64)[None].expand(n, -1) * 0.1, "p_env": torch.zeros(n, 16, 10, 11), "m_map": torch.zeros(n)}
            st = w18c.build_interaction_strata(edg.numpy(), car.numpy(), nm.numpy(), has_cars=True)
            res, _ = w18c.run_conflict_benchmark("pinns", sites, clips, True, gd, truth.numpy(), np.ones((n, 2)), st, 6, (1, 1, 1), (1e-3, 1e-3, 1e-3), torch.device("cpu"))
            self.assertEqual(res["strata"]["S4_approaching_car"]["n_windows"], 1)
        finally:
            w18c.SEEDS = old_s

    def test_action_platt_calibration_and_rerank_end_to_end(self) -> None:
        old_s = w18a.SEEDS
        w18a.SEEDS = (42,)
        try:
            n = 16
            locs = np.array([f"campus_loc_{i % 4}" for i in range(n)], dtype=str)
            actors = np.array([f"act_{i // 2}" for i in range(n)], dtype=str)
            hist = torch.randn(n, 2, 2, dtype=torch.float64) * 0.3
            truth = torch.randn(n, 7, 2, dtype=torch.float64) * 0.5
            ya = torch.tensor([[i % 2 == 0, i % 3 == 0, i % 4 == 0, i % 5 == 0, i % 2 == 1] for i in range(n)], dtype=torch.bool)
            va = torch.ones((n, 5), dtype=torch.bool)
            gd = {"history": hist, "truth": truth, "bio_even": torch.randn(n, 5, 11), "pose_vec_world": torch.randn(n, 5, 2, 2, dtype=torch.float64) * 0.2, "signed_knee": torch.randn(n, 5, 1), "quality": torch.full((n, 4), 0.9), "m_pose": torch.ones(n), "edges": torch.zeros(n, 12, 9), "is_car": torch.zeros(n, 12), "nbr_mask": torch.zeros(n, 12), "y_act": ya, "valid_act": va, "obs_dt": torch.full((n, 1), 0.4, dtype=torch.float64), "fut_s": torch.arange(1, 8, dtype=torch.float64)[None].expand(n, -1) * 0.4, "p_env": torch.zeros(n, 16, 10, 11), "m_map": torch.zeros(n)}
            res, _ = w18a.run_action_rerank_benchmark("jrdb", locs, actors, False, gd, hist.numpy(), truth.numpy(), ya.numpy(), va.numpy(), 0.4, (3, 5, 7), 8, (1, 1, 1, 1), (1e-3, 1e-3, 1e-3, 1e-3), torch.device("cpu"))
            self.assertIn("macro_f1", res["action_metrics_platt_inner_opt_thr"]["group_equal"])
            self.assertIsNotNone(res["horizon_rerank_strata"]["all"]["h3_full"]["rerank_inner_minus_prior_ci"]["location_equal"]["ci95"])
        finally:
            w18a.SEEDS = old_s


if __name__ == "__main__":
    unittest.main()
