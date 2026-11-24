"""12-item rigorous mathematical invariant, gradient, FLOPs, and MPS smoke test suite."""

import unittest
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from src.data.actions import compute_action_labels
from src.data.features import assign_quality_bucket, build_motion_features, build_pose_and_quality_features
from src.data.scene_graph import build_local_polylines, build_neighbor_edges
from src.data.splits import temporal_buffer_split_indices, verify_zero_temporal_leakage
from src.losses.losses import JointPredictionLoss
from src.metrics.metrics_and_stats import (
    compute_action_metrics,
    compute_sample_trajectory_metrics,
    holm_bonferroni_adjust,
    paired_hypothesis_test,
)
from src.models.baselines import HSTBaseline, MCENETBaseline, SocialTransmotionBaseline
from src.models.predictor import MODEL_SPECS, UnstructuredShortPosePredictor


class TestMathematicalInvariants(unittest.TestCase):
    def test_01_anti_leakage_temporal_buffer(self) -> None:
        sp = temporal_buffer_split_indices(np.arange(0, 1200))
        meta = [{"seq_id": "s1", "split": k, "t0_frame_10hz": int(f)} for k, v in sp.items() for f in v]
        ok, msg = verify_zero_temporal_leakage(meta)
        self.assertTrue(ok, msg)
        # Inject deliberate 1-frame overlap -> must catch!
        bad_meta = meta + [{"seq_id": "s1", "split": "test", "t0_frame_10hz": int(sp["train"][-1]) + 20}]
        ok_bad, _ = verify_zero_temporal_leakage(bad_meta)
        self.assertFalse(ok_bad)

    def test_02_motion_gap_clearing_and_nan_safety(self) -> None:
        pos = np.full((10, 2), np.nan, dtype=np.float32)
        pos[0] = [1.0, 2.0]
        pos[9] = [4.0, 6.0]
        mask = np.zeros(10, dtype=np.float32)
        mask[0] = 1.0
        mask[9] = 1.0  # 0.9s > 0.5s gap!
        xm = build_motion_features(pos, mask)
        self.assertTrue(np.isfinite(xm).all())
        self.assertEqual(float(xm[9, 6]), 0.0)
        self.assertEqual(float(xm[9, 4]), 0.0)

    def test_03_pose_hip_fallback_and_quality_buckets(self) -> None:
        kpts = np.zeros((5, 17, 3), dtype=np.float32)
        kpts[:, :11, 2] = 0.80  # Upper body valid, hips (11,12) missing -> fallback r_hip=0
        bbox = np.tile(np.array([200.0, 300.0, 40.0, 120.0], dtype=np.float32), (5, 1))
        xp, q, qp, b = build_pose_and_quality_features(kpts, bbox, np.ones(5), img_h=1000.0)
        self.assertEqual(float(xp[0, 0, 4]), 0.0)
        self.assertEqual(b, 1)  # q1 = 11/17 = 0.647 -> Partial bucket
        self.assertGreater(float(qp[0]), 0.80)  # Head-shoulder clean
        self.assertEqual(float(qp[4]), 0.0)     # Lower-limb zero!
        self.assertEqual(assign_quality_bucket(np.array([0.90, 0.75, 0.12, 1.0])), 0)
        self.assertEqual(assign_quality_bucket(np.array([0.40, 0.75, 0.12, 1.0])), 2)

    def test_04_action_rotation_invariance_and_exclusivity(self) -> None:
        t_h = np.arange(5) * 0.1
        base_h = np.stack([t_h * 1.0, np.zeros_like(t_h)], axis=-1)
        ang = np.linspace(0.0, np.deg2rad(-42.0), 30)
        base_f = base_h[-1] + np.cumsum(np.stack([0.1 * np.cos(ang), 0.1 * np.sin(ang)], axis=-1), axis=0)
        for deg in (0, 60, 179.5, 180.0, 180.5, 270):
            th = np.deg2rad(deg)
            rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            y, m = compute_action_labels(base_h @ rot.T, np.ones(5), base_f @ rot.T, np.ones(30))
            self.assertEqual(m, 1.0)
            self.assertEqual(y.tolist(), [0.0, 0.0, 0.0, 1.0, 0.0])
        # Incomplete future (<0.80) -> m_act = 0, NOT Maintain!
        f_bad = np.zeros(30); f_bad[:20] = 1.0
        y_bad, m_bad = compute_action_labels(base_h, np.ones(5), base_f, f_bad)
        self.assertEqual(m_bad, 0.0)
        self.assertEqual(float(y_bad.sum()), 0.0)

    def test_05_neighbor_radial_approaching_speed(self) -> None:
        edges, _, nmask = build_neighbor_edges(
            ego_pos=np.array([0.0, 0.0]),
            ego_vel=np.array([0.0, 0.0]),
            ego_vel_valid=1.0,
            ego_obs_ratio=1.0,
            nbr_pos=np.array([[5.0, 0.0], [5.0, 0.0], [5.0, 0.0]]),
            nbr_vel=np.array([[-6.0, 0.0], [4.0, 0.0], [-9.0, 0.0]]),
            nbr_vel_valid=np.array([1.0, 1.0, 0.0]),
            nbr_obs_ratio=np.ones(3),
            nbr_types=np.array([4, 6, 4]),
        )
        self.assertEqual(float(nmask.sum()), 3.0)
        self.assertAlmostEqual(float(edges[0, 5]), 6.0, places=3)   # Approaching > 0
        self.assertAlmostEqual(float(edges[1, 5]), -4.0, places=3)  # Receding < 0
        self.assertEqual(float(edges[2, 5]), 0.0)                   # Missing vel -> 0.0!


    @staticmethod
    def _dummy_batch(bsz: int = 4, seq_t: int = 5, device: str = "cpu") -> dict:
        g = torch.Generator(device="cpu").manual_seed(1234)
        xm = torch.randn(bsz, seq_t, 9, generator=g).to(device)
        xm[:, :, 5] = 1.0
        xm[:, :, 6] = 1.0
        xp = torch.rand(bsz, seq_t, 17, 6, generator=g).to(device)
        xp[:, :, :, 2] = 0.85
        xp[:, :, :, 3] = 1.0
        penv = torch.randn(bsz, 16, 10, 11, generator=g).to(device)
        penv[:, :, :, 2:10] = 0.0
        penv[:, :4, :, 3] = 1.0
        penv[:, :4, :, 10] = 1.0
        penv[:, 4:, :, 10] = 0.0
        return {
            "x_m": xm,
            "x_p": xp,
            "q_vec": torch.full((bsz, 4), 0.85, device=device),
            "q_part": torch.full((bsz, 6), 0.85, device=device),
            "p_env": penv,
            "m_map": torch.ones((bsz, 1), device=device),
            "ego_type_oh": torch.nn.functional.one_hot(torch.zeros(bsz, dtype=torch.long, device=device), 8).float(),
            "nbr_edges": torch.randn(bsz, 12, 9, generator=g).to(device),
            "nbr_type_oh": torch.nn.functional.one_hot(torch.ones(bsz, 12, dtype=torch.long, device=device), 8).float(),
            "nbr_mask": torch.cat([torch.ones(bsz, 4, device=device), torch.zeros(bsz, 8, device=device)], dim=1),
            "fut_xy": torch.cumsum(0.12 * torch.randn(bsz, 30, 2, generator=g), dim=1).to(device),
            "fut_mask": torch.ones(bsz, 30, device=device),
            "y_act": torch.tensor([[1, 0, 1, 0, 0]] * bsz, dtype=torch.float32, device=device),
            "m_act": torch.ones(bsz, device=device),
        }

    def test_06_masked_gru_and_stgcn_occlusion_isolation(self) -> None:
        m = UnstructuredShortPosePredictor("Ours_Full").eval()
        b1 = self._dummy_batch()
        b1["x_m"][:, 1, 5] = 0.0
        b1["x_p"][:, :, 11:17, 3] = 0.0
        b2 = {k: v.clone() for k, v in b1.items()}
        b2["x_m"][:, 1, :4] = 99999.0
        b2["x_p"][:, :, 11:17, :2] = -77777.0
        b2["nbr_edges"][:, 4:, :] = 55555.0
        with torch.no_grad():
            o1, o2 = m(b1), m(b2)
        self.assertTrue(torch.allclose(o1["trajs"], o2["trajs"], atol=0.0))

    def test_07_hard_pose_fallback_and_param_parity(self) -> None:
        b = self._dummy_batch()
        b["x_p"].zero_()
        b["q_vec"][:, 0] = 0.0
        b["q_part"].zero_()
        for v in ("M1a", "M1b", "M1", "M1p", "Ours_Full"):
            mod = UnstructuredShortPosePredictor(v).eval()
            with torch.no_grad():
                out = mod(b)
            self.assertTrue(torch.all(out["gate_weights"] == 0.0))
        self.assertEqual(
            sum(p.numel() for p in UnstructuredShortPosePredictor("M1b").parameters()),
            sum(p.numel() for p in UnstructuredShortPosePredictor("M1").parameters()),
        )
        self.assertEqual(
            sum(p.numel() for p in UnstructuredShortPosePredictor("M3b").parameters()),
            sum(p.numel() for p in UnstructuredShortPosePredictor("M3").parameters()),
        )

    def test_08_isolated_pedestrian_and_unannotated_map_grad(self) -> None:
        mod = UnstructuredShortPosePredictor("Ours_Full")
        b = self._dummy_batch()
        b["nbr_mask"].zero_()
        b["m_map"][:2] = 0.0
        b["p_env"][2:, :, :, 10] = 0.0  # Open plaza
        out = mod(b)
        (out["trajs"].pow(2).sum() + out["mode_logits"].pow(2).sum()).backward()
        for p in mod.parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all())

    def test_09_stop_grad_bridge_and_m_act_zero_loss(self) -> None:
        b = self._dummy_batch()
        b["m_act"].zero_()  # All invalid action labels -> loss_act must be exact 0!
        crit = JointPredictionLoss(use_action_loss=True)
        for v, expect_zero in (("M3", True), ("M3p", True), ("M3b", False)):
            mod = UnstructuredShortPosePredictor(v)
            losses = crit(mod(b), b["fut_xy"], b["fut_mask"], b["y_act"], b["m_act"])
            self.assertEqual(float(losses["loss_act"].item()), 0.0)
            losses["loss"].backward()
            g_max = max(0.0 if p.grad is None else p.grad.abs().max().item() for p in mod.decoder.act_head.parameters())
            if expect_zero:
                self.assertEqual(g_max, 0.0)
            else:
                self.assertGreater(g_max, 1e-6)

    def test_10_parameter_and_flops_budget(self) -> None:
        b12 = self._dummy_batch(bsz=12, seq_t=20)
        for v in list(MODEL_SPECS.keys()):
            mod = UnstructuredShortPosePredictor(v).eval()
            n_params = sum(p.numel() for p in mod.parameters())
            self.assertLess(n_params, 2_500_000)
        mod_full = UnstructuredShortPosePredictor("Ours_Full").eval()
        with FlopCounterMode(display=False) as fc:
            _ = mod_full(b12)
        gflops = fc.get_total_flops() / 1e9
        self.assertLess(gflops, 0.80, f"Exceeded 0.8 GFLOPs budget: {gflops:.4f}")

    def test_11_metrics_and_holm_bonferroni(self) -> None:
        trajs = np.zeros((10, 6, 30, 2))
        probs = np.tile(np.array([0.8, 0.04, 0.04, 0.04, 0.04, 0.04]), (10, 1))
        gt = np.ones((10, 30, 2))
        fmask = np.ones((10, 30))
        fmask[:, 20:] = 0.0
        gt[:, 20:, :] = 999.0  # Padded garbage after frame 19 must be ignored!
        sm = compute_sample_trajectory_metrics(trajs, probs, gt, fmask)
        self.assertAlmostEqual(float(sm["FDE_1"].mean()), np.sqrt(2.0), places=5)
        r1 = paired_hypothesis_test(np.ones(20) * 0.5, np.ones(20) * 0.8 + np.linspace(-0.02, 0.02, 20))
        r2 = paired_hypothesis_test(np.ones(20) * 0.8, np.ones(20) * 0.8 + np.linspace(-0.01, 0.01, 20))
        adj = holm_bonferroni_adjust([r1, r2])
        self.assertTrue(bool(adj[0]["significant_005"]))
        self.assertFalse(bool(adj[1]["significant_005"]))
        am = compute_action_metrics(np.array([[1,0,0,0,0],[0,0,0,0,1]]), np.array([[0.9,0.1,0.1,0.1,0.1],[0.9,0.9,0.9,0.9,0.9]]), np.array([1.0, 0.0]))
        self.assertAlmostEqual(am["macro_f1"], 1.0, places=4)

    def test_12_mps_end_to_end_training_smoke(self) -> None:
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        torch.manual_seed(42)
        mod = UnstructuredShortPosePredictor("Ours_Full").to(dev)
        ext_models = [MCENETBaseline().to(dev), HSTBaseline().to(dev), SocialTransmotionBaseline().to(dev)]
        b = self._dummy_batch(bsz=8, seq_t=5, device=dev)
        for em in ext_models:
            eo = em(b)
            self.assertEqual(eo["trajs"].shape, (8, 6, 30, 2))
        opt = torch.optim.AdamW(mod.parameters(), lr=3e-3)
        crit = JointPredictionLoss(use_action_loss=True).to(dev)
        l_first = None
        l_last = None
        for _ in range(20):
            opt.zero_grad()
            out = mod(b)
            loss_d = crit(out, b["fut_xy"], b["fut_mask"], b["y_act"], b["m_act"])
            loss_d["loss"].backward()
            torch.nn.utils.clip_grad_norm_(mod.parameters(), 2.0)
            opt.step()
            val = float(loss_d["loss"].item())
            if l_first is None:
                l_first = val
            l_last = val
        self.assertLess(l_last, 0.65 * l_first)

    def test_13_short_clip_and_stratified_site_splits(self) -> None:
        from src.data.splits import assign_site_clip_splits

        # Short 22s (220-frame) sequence must retain >=15 future steps in test!
        sp_short = temporal_buffer_split_indices(np.arange(0, 220))
        for k in ("train", "val", "test"):
            self.assertGreater(len(sp_short[k]), 0, f"Empty {k} on 22s clip!")
        self.assertLessEqual(int(sp_short["test"][0]), 220 - 15)

        # Stratified 12-clip site split must assign all 3 splits with 60-step boundary guards
        cmap = assign_site_clip_splits(list(range(1, 13)))
        splits_seen = {v[0] for v in cmap.values()}
        self.assertEqual(splits_seen, {"train", "val", "test"})
        val_guards = [v[1] for _, v in sorted(cmap.items()) if v[0] == "val"]
        test_guards = [v[1] for _, v in sorted(cmap.items()) if v[0] == "test"]
        self.assertEqual(val_guards[0], 60)
        self.assertEqual(test_guards[0], 60)


if __name__ == "__main__":
    unittest.main(verbosity=2)
