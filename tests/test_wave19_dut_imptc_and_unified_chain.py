"""Unit tests for Wave 19: external DUT/IMPTC conflict+conformal & unified clean M0->M3 chain."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from scripts.run_wave18_conflict_gated_interaction_5seed import ConflictGatedM2bForecaster, PPTMainlineO2Forecaster
from scripts.run_wave19_dut_imptc_conflict_and_conformal_5seed import evaluate_external_conflict_and_conformal
from scripts.run_wave19_unified_clean_m0_to_m3_5seed import UnifiedCleanO2Forecaster, run_unified_benchmark_on_cohort


class TestWave19Scripts(unittest.TestCase):
    def test_external_conflict_equivariance_and_conformal(self) -> None:
        rng = np.random.default_rng(7)
        n = 48
        hist = rng.normal(0.0, 0.3, size=(n, 5, 2)).astype(np.float64)
        hist -= hist[:, -1:]
        truth = rng.normal(0.0, 0.8, size=(n, 30, 2)).astype(np.float64)
        obs_dt = np.full((n, 4), 0.1, dtype=np.float64)
        fut_s = np.broadcast_to(np.arange(1, 31, dtype=np.float64) * 0.1, (n, 30)).copy()
        edges = np.abs(rng.normal(2.5, 1.5, size=(n, 12, 9))).astype(np.float32)
        edges[:, :, 0:4] = rng.normal(0.0, 1.0, size=(n, 12, 4)).astype(np.float32)
        edges[:, :, 5] = rng.uniform(-0.5, 1.5, size=(n, 12)).astype(np.float32)
        edges[:, :, 6] = 1.0
        is_car = (rng.uniform(size=(n, 12)) > 0.7).astype(np.float32)
        nbr_mask = (rng.uniform(size=(n, 12)) > 0.3).astype(np.float32)
        nbr_mask[:8] = 0.0
        v_last = (hist[:, -1:] - hist[:, -2:-1]) / 0.1
        d_cv = np.linalg.norm(v_last * fut_s[:, :, None] - truth, axis=-1)
        cohort = {
            "group": np.array(["g0"] * 24 + ["g1"] * 24, dtype=str),
            "history": hist, "truth": truth, "obs_dt": obs_dt, "fut_s": fut_s,
            "edges": edges, "is_car": is_car, "nbr_mask": nbr_mask,
            "cv_err": np.column_stack((d_cv.mean(1), d_cv[:, -1])), "cv_tube": d_cv.max(1),
        }
        torch.manual_seed(42)
        ens = [[{
            "M1": PPTMainlineO2Forecaster("M1").eval(),
            "M2b_frozen_head": ConflictGatedM2bForecaster(use_conflict_gate=False).eval(),
            "M2b_ttc_conflict_gate": ConflictGatedM2bForecaster(use_conflict_gate=True).eval(),
        }]]
        res = evaluate_external_conflict_and_conformal(cohort, ens, torch.device("cpu"))
        self.assertLess(res["max_five_axis_ade1_swing_m"], 1e-6)
        self.assertIsNotNone(res["strata"]["all"]["paired_top1_ci"]["M2b_conflict_gate_minus_M1"]["ci95"])
        cf = res["zero_retrain_split_conformal"]
        self.assertEqual(cf["n_ordered_cross_pairs"], 2)
        self.assertGreater(cf["within_group_temporal_50_50"]["M2b_ttc_conflict_gate"]["90"]["rad_u6"], 0.0)

    def test_unified_clean_chain_monotonic_and_end_to_end(self) -> None:
        dev = torch.device("cpu")
        m = UnifiedCleanO2Forecaster("U_M3_mono_conflict").eval()
        q_lo = torch.tensor([[0.2, 0.2, 0.2, 0.2], [0.9, 0.9, 0.9, 0.9]])
        mp = torch.tensor([1.0, 1.0])
        g = m.compute_mono_gate(q_lo, mp).detach()
        self.assertLess(float(g[0]), float(g[1]))
        self.assertEqual(float(m.compute_mono_gate(q_lo, torch.zeros(2)).detach().abs().max()), 0.0)

        rng = np.random.default_rng(11)
        n = 36
        grp = np.array(["s0"] * 12 + ["s1"] * 12 + ["s2"] * 12, dtype=str)
        clips = np.array(["c0"] * 6 + ["c1"] * 6 + ["c2"] * 6 + ["c3"] * 6 + ["c4"] * 6 + ["c5"] * 6, dtype=str)
        hist = rng.normal(0.0, 0.3, size=(n, 5, 2)).astype(np.float64)
        hist -= hist[:, -1:]
        truth = rng.normal(0.0, 0.6, size=(n, 30, 2)).astype(np.float64)
        q_np = rng.uniform(0.1, 0.95, size=(n, 4)).astype(np.float32)
        mp_np = (rng.uniform(size=n) > 0.15).astype(np.float32)
        ya_np = (rng.uniform(size=(n, 5)) > 0.65)
        va_np = np.ones((n, 5), dtype=bool)
        gd = {
            "history": torch.from_numpy(hist), "truth": torch.from_numpy(truth),
            "bio_even": torch.randn(n, 5, 11), "pose_vec_world": torch.randn(n, 5, 2, 2, dtype=torch.float64) * 0.2,
            "signed_knee": torch.randn(n, 5, 1), "quality": torch.from_numpy(q_np), "m_pose": torch.from_numpy(mp_np),
            "edges": torch.randn(n, 12, 9).abs(), "is_car": torch.zeros(n, 12), "nbr_mask": torch.ones(n, 12),
            "y_act": torch.from_numpy(ya_np), "valid_act": torch.from_numpy(va_np),
            "obs_dt": torch.full((n, 4), 0.1, dtype=torch.float64),
            "fut_s": torch.arange(1, 31, dtype=torch.float64)[None].expand(n, -1) * 0.1,
        }
        d_cv = np.linalg.norm(((hist[:, -1:] - hist[:, -2:-1]) / 0.1) * (np.arange(1, 31)[None, :, None] * 0.1) - truth, axis=-1)
        out = run_unified_benchmark_on_cohort(
            "pinns_syn", grp, clips, True, gd, hist, truth, ya_np, va_np, q_np, mp_np,
            np.column_stack((d_cv.mean(1), d_cv[:, -1])), d_cv.max(1),
            {"all": np.ones(n, dtype=bool)}, 0.1, 16, (1, 1, 1, 1), (2e-3, 1e-3, 1e-3, 1e-3), [42], dev,
        )
        self.assertGreater(out["spearman_mono_gate_vs_q1"], 0.5)
        self.assertIsNotNone(out["strata"]["all"]["paired_top1_ci"]["U_M2b_conflict_minus_U_M1_mono"]["ci95"])

        hist_j = hist[:, :2] - hist[:, 1:2]
        truth_j = truth[:, :7]
        gd_j = {**gd, "history": torch.from_numpy(hist_j), "truth": torch.from_numpy(truth_j),
                "obs_dt": torch.full((n, 1), 0.4, dtype=torch.float64),
                "fut_s": torch.arange(1, 8, dtype=torch.float64)[None].expand(n, -1) * 0.4}
        d_cvj = np.linalg.norm(((hist_j[:, 1:2] - hist_j[:, 0:1]) / 0.4) * (np.arange(1, 8)[None, :, None] * 0.4) - truth_j, axis=-1)
        out_j = run_unified_benchmark_on_cohort(
            "jrdb_syn", grp, clips, False, gd_j, hist_j, truth_j, ya_np, va_np, q_np, mp_np,
            np.column_stack((d_cvj.mean(1), d_cvj[:, -1])), d_cvj.max(1),
            {"all": np.ones(n, dtype=bool)}, 0.4, 16, (1, 1, 1, 1), (2e-3, 1e-3, 1e-3, 1e-3), [42], dev,
        )
        self.assertIsNotNone(out_j["strata"]["all"]["paired_top1_ci"]["U_M3_rerank_minus_U_M3_conflict"]["ci95"])


if __name__ == "__main__":
    unittest.main()
