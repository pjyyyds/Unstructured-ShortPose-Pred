"""Unit tests for Wave 22 causal APA lead-time detector & delayed-onset Start/Stop candidate upgrade."""
from __future__ import annotations

import unittest
import numpy as np
import torch

from scripts.run_wave22_apa_leadtime_and_delayed_onset_4dataset import (
    eval_delayed_onset_apa_upgrade_cohort,
    eval_sliding_leadtime_5seed,
)
from src.models.apa_precursor_leadtime import build_delayed_onset_start_stop_candidates


class TestWave22APALeadTimeAndDelayedOnset(unittest.TestCase):
    def test_delayed_onset_start_beats_both_cv_and_immediate_ramp(self) -> None:
        # Stationary pedestrian in history who starts walking along +X at t_onset = 1.2 s
        hist = np.zeros((1, 5, 2), dtype=np.float64)
        facing = np.array([[1.0, 0.0]], dtype=np.float64)
        fut_s = (np.arange(1, 31, dtype=np.float64) * 0.1)[None, :]
        gt = np.zeros((1, 30, 2), dtype=np.float64)
        tau = np.maximum(fut_s[0] - 1.2, 0.0)
        gt[0, :, 0] = np.where(tau < 1.0, 0.55 * (tau ** 2), 0.55 + 1.1 * (tau - 1.0))

        imm_ramp = np.zeros((1, 30, 2), dtype=np.float64)
        imm_ramp[0, :, 0] = 0.80 * fut_s[0]  # Immediate linear 2.4m over 3s
        s_del, _ = build_delayed_onset_start_stop_candidates(hist, facing, np.array([1.2]), fut_s, dt_obs=0.1)

        ade_cv = float(np.linalg.norm(gt[0] - 0.0, axis=-1).mean())
        ade_imm = float(np.linalg.norm(gt[0] - imm_ramp[0], axis=-1).mean())
        ade_del = float(np.linalg.norm(gt[0] - s_del[0], axis=-1).mean())
        self.assertLess(ade_del, ade_cv - 0.25)
        self.assertLess(ade_del, ade_imm - 0.30)

    def test_sliding_leadtime_and_delayed_cohort_pipeline(self) -> None:
        rng = np.random.default_rng(7)
        ds = {
            "kin": rng.normal(0, 0.1, (48, 5, 5)).astype(np.float32),
            "pose": rng.normal(0, 0.2, (48, 5, 8)).astype(np.float32),
            "q": np.ones((48, 2), dtype=np.float32),
            "veh": rng.normal(0, 0.2, (48, 3)).astype(np.float32),
            "y": np.tile(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.float32), (12, 1)),
            "tau_s": np.tile(np.array([-0.4, -0.2, 0.0, 0.0]), 12),
            "event_id": np.arange(48, dtype=np.int64) // 2,
            "event_cls": np.tile(np.array([0, 1, 2, -1], dtype=np.int64), 12),
            "is_neg": np.tile(np.array([False, False, False, True]), 12),
            "is_deocc": np.ones(48, dtype=bool),
            "vid": np.array([f"v_{i % 6}" for i in range(48)]),
            "split": np.array(["train"] * 20 + ["val"] * 12 + ["test"] * 16),
        }
        lt = eval_sliding_leadtime_5seed(ds, [42], torch.device("cpu"), epochs=2)
        self.assertIn("matched_fpr_0p05", lt)

        n = 24
        h_np = rng.normal(0, 0.05, (n, 5, 2))
        t_np = rng.normal(0, 0.3, (n, 30, 2))
        ya = np.ones((n, 5), dtype=bool)
        va = np.ones((n, 5), dtype=bool)
        ya[n // 2 :, :4] = False
        on = np.full((n, 5), 6, dtype=np.int64)
        gd = {
            "history": torch.from_numpy(h_np), "truth": torch.from_numpy(t_np),
            "bio_even": torch.randn(n, 5, 10), "pose_vec_world": torch.randn(n, 5, 2, 2, dtype=torch.float64),
            "signed_knee": torch.randn(n, 5, 2), "quality": torch.ones(n, 4), "m_pose": torch.ones(n),
            "edges": torch.randn(n, 12, 9), "is_car": torch.zeros(n, 12), "nbr_mask": torch.ones(n, 12),
            "y_act": torch.from_numpy(ya), "valid_act": torch.from_numpy(va),
            "obs_dt": torch.full((n, 4), 0.1, dtype=torch.float64),
            "fut_s": torch.arange(1, 31, dtype=torch.float64)[None].expand(n, -1) * 0.1,
        }
        grp = np.array([f"g{i % 3}" for i in range(n)])
        fut_s_np = np.broadcast_to(np.arange(1, 31, dtype=np.float64)[None, :] * 0.1, (n, 30)).copy()
        res = eval_delayed_onset_apa_upgrade_cohort(
            "unit", grp, grp, True, gd, h_np, t_np, np.ones((n, 2)), ya, va, on,
            np.ones(n, dtype=bool), 0.1, fut_s_np, 12, 3, 2e-3, [42], torch.device("cpu"),
        )
        self.assertIn("S6_APA_Delayed_NetGain", res["overall_ade1_fde1_post"])

        # Also test JRDB mixed-horizon shape: history=(N, 2, 2), pose=(N, 5, ...), truth=(N, 7, 2)
        h2 = rng.normal(0, 0.05, (n, 2, 2))
        t7 = rng.normal(0, 0.3, (n, 7, 2))
        gd2 = {**gd, "history": torch.from_numpy(h2), "truth": torch.from_numpy(t7),
               "obs_dt": torch.full((n, 1), 0.4, dtype=torch.float64),
               "fut_s": torch.arange(1, 8, dtype=torch.float64)[None].expand(n, -1) * 0.4}
        fut7 = np.broadcast_to(np.arange(1, 8, dtype=np.float64)[None, :] * 0.4, (n, 7)).copy()
        res2 = eval_delayed_onset_apa_upgrade_cohort(
            "unit_jrdb", grp, grp, False, gd2, h2, t7, np.ones((n, 2)), ya, va, np.full((n, 5), 2, dtype=np.int64),
            np.ones(n, dtype=bool), 0.4, fut7, 12, 3, 2e-3, [42], torch.device("cpu"),
        )
        self.assertIn("S6_APA_Delayed_NP5pctFPR", res2["overall_ade1_fde1_post"])


if __name__ == "__main__":
    unittest.main()
