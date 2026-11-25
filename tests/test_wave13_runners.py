"""Unit tests for Wave 13 cloud runners (DAIR neighbor graph, JRDB outdoor 1,677, Table 1 cluster bootstrap)."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from scripts.run_dair_roadside_neighbor_graph_o2 import score_dair_graph_vs_nograph
from scripts.run_jrdb_outdoor_o2_full_5seed import _jrdb_edges, score_jrdb_ensemble
from scripts.run_pinns_matched_o2_multiduration import SITES
from scripts.run_table1_cluster_bootstrap_o2_full import cluster_paired_ci
from src.models.o2_unified_full import O2UnifiedFullForecaster


class TestWave13Runners(unittest.TestCase):
    def test_dair_graph_equivariance(self) -> None:
        torch.manual_seed(19)
        net = O2UnifiedFullForecaster().eval()
        with torch.no_grad():
            net.graph_proj.weight.normal_(0.0, 0.05)
        rng = np.random.default_rng(19)
        dair = {
            "hist2": np.stack((-rng.uniform(0.05, 0.2, size=(6, 2)), np.zeros((6, 2))), axis=1),
            "truth": rng.normal(0.0, 1.0, size=(6, 30, 2)),
            "kpts": rng.uniform(50.0, 300.0, size=(6, 5, 17, 3)).astype(np.float32),
            "bbox": np.tile(np.array([0.0, 0.0, 80.0, 200.0], dtype=np.float32), (6, 5, 1)),
            "ts_us": np.arange(35, dtype=np.int64)[None, :] * 100000 + np.zeros((6, 1), dtype=np.int64),
            "actors": np.array(["a0", "a0", "a1", "a1", "a2", "a2"]),
        }
        dair["kpts"][..., 2] = 0.85
        edges = rng.normal(0.0, 2.0, size=(6, 12, 9)).astype(np.float32)
        edges[:, :, 4] = np.linalg.norm(edges[:, :, 0:2], axis=-1)
        cars = (rng.random((6, 12)) > 0.6).astype(np.float32)
        masks = (rng.random((6, 12)) > 0.3).astype(np.float32)
        unambig = np.array([True, True, True, True, True, False])
        res = score_dair_graph_vs_nograph(dair, edges, cars, masks, unambig, [(net, 0.5, 0.5)])
        self.assertLess(res["rot4_ade_swing_m"], 1e-5)

    def test_jrdb_edges_and_equivariance(self) -> None:
        torch.manual_seed(23)
        net = O2UnifiedFullForecaster().eval()
        with torch.no_grad():
            net.graph_proj.weight.normal_(0.0, 0.05)
        by_f = {0: {1: np.array([0.0, 0.0]), 2: np.array([1.2, 0.5])}, 6: {1: np.array([0.4, 0.0]), 2: np.array([1.0, 0.5])}}
        e, m = _jrdb_edges(by_f, 0, 6, 1)
        self.assertEqual(e.shape, (12, 9))
        self.assertEqual(float(m[0]), 1.0)
        rng = np.random.default_rng(23)
        jrdb = {
            "history": np.stack((-rng.uniform(0.1, 0.4, size=(6, 2)), np.zeros((6, 2))), axis=1),
            "truth": rng.normal(0.0, 0.8, size=(6, 7, 2)),
            "scene": np.array(["s0", "s0", "s1", "s1", "s2", "s2"]),
            "actors": np.array(["p0", "p0", "p1", "p1", "p2", "p2"]),
            "edges": np.tile(e[None], (6, 1, 1)),
            "is_car": np.zeros((6, 12), dtype=np.float32),
            "nbr_mask": np.tile(m[None], (6, 1)),
            "scenes_order": ["s0", "s1", "s2"],
        }
        res = score_jrdb_ensemble(jrdb, [(net, 0.5, 0.5)])
        self.assertLess(res["five_axis_max_ade_swing_m"], 1e-6)

    def test_cluster_paired_ci(self) -> None:
        diff = np.full(16, -0.05)
        cl = np.array([f"c{i // 4}" for i in range(16)])
        st = np.array([SITES[i % 4] for i in range(16)])
        ci = cluster_paired_ci(diff, cl, st, n_boot=500)
        self.assertTrue(ci["window_weighted"]["strictly_negative"])
        self.assertTrue(ci["cluster_equal"]["strictly_negative"])


if __name__ == "__main__":
    unittest.main()
