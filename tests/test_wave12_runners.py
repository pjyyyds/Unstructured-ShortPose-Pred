"""Unit tests for Wave 12 cloud experiment runners (emergence, IMPTC/DAIR unified, frame-offset retrain)."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from scripts.run_emergence_and_action_sensitivity import (
    _calibrate_hybrid_fold,
    _score_prob_and_bin,
    implied_probs_scaled,
    scaled_labels_and_valid,
    summarize_stratum,
)
from scripts.run_imptc_and_dair_o2_full_external import score_dair_unified, score_imptc_ensemble
from scripts.run_pinns_frame_offset_retrain_5seed import _eval_net, _pack_offset_results
from scripts.run_pinns_matched_o2_multiduration import SITES
from src.models.o2_unified_full import O2UnifiedFullForecaster


class TestWave12Runners(unittest.TestCase):
    def test_emergence_and_scaled_actions(self) -> None:
        rng = np.random.default_rng(7)
        hist = np.cumsum(rng.normal(0.0, 0.08, size=(12, 5, 2)), axis=1)
        hist -= hist[:, -1:]
        truth = np.cumsum(rng.normal(0.05, 0.04, size=(12, 30, 2)), axis=1)
        vh = np.linspace(0.05, 1.1, 12)
        ya, va = scaled_labels_and_valid(hist, truth, vh, 0.75)
        self.assertEqual(ya.shape, (12, 5))
        self.assertEqual(va.shape, (12, 5))
        paths = rng.normal(0.0, 0.5, size=(12, 6, 30, 2))
        probs = np.full((12, 6), 1.0 / 6.0)
        imp = implied_probs_scaled(hist, paths, probs, vh, 0.75)
        self.assertTrue(np.all((imp >= 0.0) & (imp <= 1.0)))
        site = np.array([SITES[i % 4] for i in range(12)], dtype=str)
        clip = np.array([f"{site[i]}_c{i % 2}" for i in range(12)], dtype=str)
        al, th = _calibrate_hybrid_fold(ya, va, imp * 0.8, imp, np.arange(9))
        self.assertEqual(al.shape, (5,))
        sc = _score_prob_and_bin(ya, va, imp, imp >= th[None, :], site)
        self.assertIn("site_equal", sc)
        errs = {k: rng.uniform(0.2, 1.2, size=(12, 3)) for k in ("cv", "u0_bayes", "full_bayes", "naive_pad_t20_bayes")}
        st = summarize_stratum(errs, np.ones(12, dtype=bool), site, clip)
        self.assertEqual(st["n_windows"], 12)

    def test_imptc_and_dair_5axis_equivariance(self) -> None:
        torch.manual_seed(11)
        net = O2UnifiedFullForecaster().eval()
        rng = np.random.default_rng(11)
        hist5 = np.cumsum(rng.normal(0.08, 0.03, size=(8, 5, 2)), axis=1)
        hist5 -= hist5[:, -1:]
        edges = rng.normal(0.0, 1.5, size=(8, 12, 9)).astype(np.float32)
        edges[:, :, 4] = np.linalg.norm(edges[:, :, 0:2], axis=-1)
        imptc = {
            "history": hist5,
            "truth": rng.normal(0.0, 1.0, size=(8, 30, 2)),
            "ts_us": np.arange(35, dtype=np.int64)[None, :] * 100000 + np.zeros((8, 1), dtype=np.int64),
            "seq": np.array(["0000_20230322_081506", "0001_20230523_105516", "0002_20230523_111106", "0003_20230523_111514"] * 2),
            "actors": np.array([f"p{i // 2}" for i in range(8)]),
            "edges": edges,
            "is_car": (rng.random((8, 12)) > 0.7).astype(np.float32),
            "nbr_mask": (rng.random((8, 12)) > 0.3).astype(np.float32),
        }
        im_res = score_imptc_ensemble(imptc, [(net, 0.5, 0.5)])
        self.assertLess(im_res["max_blackout_empty_algebraic_diff_m"], 1e-7)
        self.assertLess(im_res["five_axis_max_ade_swing_m"]["full_graph_clk_bayes"], 1e-6)

        dair = {
            "hist2": np.stack((-rng.uniform(0.05, 0.2, size=(6, 2)), np.zeros((6, 2))), axis=1),
            "truth": rng.normal(0.0, 1.0, size=(6, 30, 2)),
            "kpts": rng.uniform(50.0, 300.0, size=(6, 5, 17, 3)).astype(np.float32),
            "bbox": np.tile(np.array([0.0, 0.0, 80.0, 200.0], dtype=np.float32), (6, 5, 1)),
            "ts_us": np.arange(35, dtype=np.int64)[None, :] * 100000 + np.zeros((6, 1), dtype=np.int64),
            "actors": np.array(["a0", "a0", "a1", "a1", "a2", "a2"]),
            "m0_by_rot": rng.uniform(1.0, 2.0, size=(6, 4, 2)),
            "m1_by_rot": rng.uniform(1.0, 2.0, size=(6, 4, 2)),
        }
        dair["kpts"][..., 2] = 0.85
        da_res = score_dair_unified(dair, [(net, 0.5, 0.5)], [(net, 0.5, 0.5)])
        self.assertLess(da_res["rot4_ade_swing_m"], 1e-6)
        self.assertLess(da_res["five_axis_ade_swing_m"], 1e-5)

    def test_offset_eval_and_pack(self) -> None:
        torch.manual_seed(5)
        net = O2UnifiedFullForecaster().eval()
        site = np.array([SITES[i % 4] for i in range(8)], dtype=str)
        d = {
            "site": site,
            "clip": np.array([f"{site[i]}_0" for i in range(8)], dtype=str),
            "history": torch.randn(8, 5, 2, dtype=torch.float64) * 0.2,
            "truth": torch.randn(8, 30, 2, dtype=torch.float64),
            "bio_seq": torch.randn(8, 5, 11),
            "q_bio": torch.full((8,), 0.8),
            "edges": torch.randn(8, 12, 9),
            "is_car": torch.zeros(8, 12),
            "nbr_mask": torch.ones(8, 12),
            "y_act": torch.ones(8, 5, dtype=torch.bool),
            "valid_act": torch.ones(8, 5, dtype=torch.bool),
            "v_hist_mean": np.linspace(0.1, 0.9, 8),
        }
        d["history"][:, -1] = 0.0
        e, _, a = _eval_net(net, d, np.arange(8), 0.5, 0.5, torch.device("cpu"))
        errs = {k: e for k in ("train0_test0", "train0_test1", "train1_test1", "train1_test0")}
        aps = {k: a for k in errs}
        with tempfile.TemporaryDirectory() as td:
            out = _pack_offset_results(
                d,
                np.array([True, True, False, False, False, False, False, False]),
                np.array([False, False, True, True, False, False, False, False]),
                errs,
                aps,
                {"active_providers": ["CPUExecutionProvider"], "clips": {}},
                {"same_device_max_abs_m": 0.0, "cuda_minus_cpu_max_abs_m": 1e-7},
                Path(td),
                (42,),
            )
            self.assertIn("train1_test1", out["train_test_2x2_matrix"])


if __name__ == "__main__":
    unittest.main()
