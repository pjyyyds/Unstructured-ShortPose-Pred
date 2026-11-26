"""Unit tests for Wave 18 GPU2 hardware profiler, structural monotonic gate, and JRDB _eval_gd unpack."""
from __future__ import annotations

import unittest
import numpy as np
import torch

from scripts.ppt_mainline_o2_models import PPTMainlineO2Forecaster
from scripts.run_jrdb_rc2_rc3_graph_and_actions_5seed import _eval_gd
import scripts.run_wave18_t4_latency_and_monotonic_gate as w18m


class TestWave18T4LatencyAndMonoGate(unittest.TestCase):
    def test_jrdb_eval_gd_unpack_and_monotonic_gate_property(self) -> None:
        n = 10
        gd = {
            "history": torch.randn(n, 2, 2, dtype=torch.float64) * 0.3,
            "bio_even": torch.randn(n, 5, 11),
            "pose_vec_world": torch.randn(n, 5, 2, 2, dtype=torch.float64) * 0.2,
            "signed_knee": torch.randn(n, 5, 1),
            "quality": torch.full((n, 4), 0.8),
            "m_pose": torch.ones(n),
            "p_env": torch.zeros(n, 16, 10, 11),
            "m_map": torch.zeros(n),
            "edges": torch.zeros(n, 12, 9),
            "is_car": torch.zeros(n, 12),
            "nbr_mask": torch.zeros(n, 12),
            "obs_dt": torch.full((n, 1), 0.4, dtype=torch.float64),
            "fut_s": torch.arange(1, 8, dtype=torch.float64)[None].expand(n, -1) * 0.4,
        }
        truth = np.zeros((n, 7, 2), dtype=np.float64)
        te_ids = np.arange(n)
        m7, ap = _eval_gd(PPTMainlineO2Forecaster("M3a").eval(), gd, truth, te_ids, torch.device("cpu"))
        self.assertEqual(m7[:, :2].shape, (n, 2))
        self.assertEqual(m7[:, 2:4].shape, (n, 2))
        self.assertEqual(ap.shape, (n, 5))

        mono = w18m.MonotonicQualityM1Forecaster().eval()
        grid = torch.linspace(0.05, 1.0, 20)
        q_seq = torch.stack((grid, grid, torch.full_like(grid, 0.5), torch.ones_like(grid)), dim=-1)
        g_vals = mono.compute_mono_gate(q_seq, torch.ones(20)).detach().numpy()
        self.assertTrue(np.all(np.diff(g_vals) > 0.0))

    def test_hardware_profiler_and_end_to_end_mono_eval(self) -> None:
        hw = w18m.profile_all_models_hardware(torch.device("cpu"), n_warmup=1, n_timed=2)
        self.assertEqual(len(hw["batch_sizes"]["B12"]), 14)
        self.assertGreater(hw["batch_sizes"]["B12"]["O2_M3"]["flops"], 0)
        self.assertGreater(hw["batch_sizes"]["B12"]["Ext2_HST"]["params"], 100000)

        old_s = w18m.SEEDS
        w18m.SEEDS = (42,)
        try:
            n = 16
            sites = np.array(["eth", "ind", "chn", "usa"] * 4, dtype=str)
            clips = np.array([f"{sites[i]}_c{i // 4}" for i in range(n)], dtype=str)
            hist = torch.randn(n, 5, 2, dtype=torch.float64) * 0.3
            truth = torch.randn(n, 30, 2, dtype=torch.float64) * 0.5
            q = torch.stack([torch.linspace(0.1, 0.95, n)] * 4, dim=-1)
            pk = {"bio_even": torch.randn(n, 5, 11), "pose_vec_world": torch.randn(n, 5, 2, 2, dtype=torch.float64), "signed_knee": torch.randn(n, 5, 1), "quality": q, "m_pose": torch.ones(n)}
            pks = {tc: pk for tc in w18m.TEST_CONDS}
            pks["Test_FullDrop"] = {**pk, "m_pose": torch.zeros(n), "quality": torch.zeros(n, 4)}
            qb = np.array([2] * 5 + [1] * 5 + [0] * 6)
            odt = torch.full((n, 4), 0.1, dtype=torch.float64)
            fs = torch.arange(1, 31, dtype=torch.float64)[None].expand(n, -1) * 0.1
            rp, _ = w18m.evaluate_monotonic_gate_remedy("pinns", sites, clips, True, hist, truth, truth.numpy(), pks, qb, odt, fs, 8, 1, 1, torch.device("cpu"))
            self.assertGreater(rp["gate_audit"]["M1_mono_q"]["spearman_gate_vs_q1_valid_joint_ratio"], 0.99)
            rj, _ = w18m.evaluate_monotonic_gate_remedy("jrdb", sites, clips, False, hist, truth, truth.numpy(), pks, qb, odt, fs, 8, 1, 1, torch.device("cpu"))
            self.assertIsNotNone(rj["test_conditions"]["Test_Jitter020"]["M1_mono_minus_M1_unconstrained"]["location_equal"]["ci95"])
        finally:
            w18m.SEEDS = old_s


if __name__ == "__main__":
    unittest.main()
