"""Unit tests for Wave 25 explicit-241 audit, O2EmergenceSafe6SlotNet, McNemar p, & JAAD reappearance."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import unittest

import numpy as np
import torch

from scripts.run_wave25_jaad_reappearance_per_event_audit import (
    audit_per_event_timestamps_and_visible_alarms,
    exact_mcnemar_two_sided_p,
)
from src.models.emergence_safe_6slot import O2EmergenceSafe6SlotNet, compute_pose_covisibility_features


class TestWave25ExplicitAuditAndReappearance(unittest.TestCase):
    def test_explicit_241_visual_audit_completeness(self) -> None:
        root = Path(__file__).resolve().parents[1]
        exp = json.loads((root / "data/annotations/pinns_explicit_241_visual_audit.json").read_text())
        old = json.loads((root / "data/annotations/pinns_blind_raw_video_emergence_labels.json").read_text())
        self.assertEqual(set(exp["labels"].keys()), set(old.keys()))
        self.assertEqual(len(exp["labels"]), 241)
        self.assertEqual(len(exp["reclassified_from_wave24_default_pers_45"]), 45)
        cnt = Counter(exp["labels"].values())
        self.assertEqual(dict(cnt), {"veh": 32, "static": 66, "pers": 20, "border": 32, "open": 91})

    def test_es6_stationary_zero_drift_and_o2_equivariance(self) -> None:
        torch.manual_seed(7)
        net = O2EmergenceSafe6SlotNet(use_pose=True, use_interaction=True, low_disp_shrink=0.80).eval()
        hist = torch.zeros((2, 5, 2), dtype=torch.float64)
        hist[1] = torch.tensor([[0.0, 0.0], [0.10, 0.0], [0.22, 0.02], [0.35, 0.05], [0.48, 0.09]], dtype=torch.float64)
        kpts = torch.rand((2, 5, 17, 3)) * 20.0 + 40.0
        kpts[..., 2] = 0.85
        bbox = torch.tensor([[[50.0, 80.0, 28.0, 75.0]] * 5] * 2)
        bio = torch.randn((2, 5, 11))
        pvw = torch.randn((2, 5, 2, 2), dtype=torch.float64) * 0.25
        sk = torch.randn((2, 5, 1))
        qv, mp = torch.ones((2, 4)) * 0.9, torch.ones(2)
        cov = compute_pose_covisibility_features(kpts, bbox, bio, pvw, sk)
        out0 = net(hist, bio, pvw, sk, qv, mp, cov)
        self.assertLess(float(out0["paths"][0, [0, 2, 3, 4, 5]].detach().abs().max()), 1e-12)
        self.assertGreater(float(out0["paths"][0, 1, -1].detach().norm()), 0.5)

        ang = 0.93
        rot = torch.tensor([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]], dtype=torch.float64)
        out_r = net(hist @ rot.T, bio, torch.einsum("btvk,jk->btvj", pvw, rot), sk, qv, mp, cov)
        self.assertLess(float(((out0["paths"] @ rot.T) - out_r["paths"]).detach().abs().max()), 1e-6)

    def test_exact_mcnemar_and_reappearance_audit(self) -> None:
        hb = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
        ha = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
        mc = exact_mcnemar_two_sided_p(hb, ha)
        self.assertAlmostEqual(float(mc["exact_two_sided_p"]), 0.125, places=8)

        # Synthetic 2 strict Start events (one early-occluded alarm, one post-reappearance visible alarm) + 20 negs
        slide_ds = {
            "split": np.array(["val"] * 10 + ["test"] * 24),
            "is_neg": np.array([True] * 10 + [False, False, False, False] + [True] * 20),
            "tau_s": np.array([0.0] * 10 + [-0.6, -0.2, -0.6, -0.2] + [0.0] * 20),
            "event_id": np.array(list(range(10)) + [100, 100, 101, 101] + list(range(200, 220))),
            "event_cls": np.array([-1] * 10 + [0, 0, 0, 0] + [-1] * 20),
            "is_deocc_strict_recovered": np.array([False] * 10 + [True, True, True, True] + [False] * 20),
        }
        meta = {
            "vid": np.array(["v0"] * 34), "pid": np.array(["p1"] * 34),
            "event_id": slide_ds["event_id"], "event_cls": slide_ds["event_cls"],
            "t0_frame_30hz": np.array([30] * 10 + [42, 54, 42, 54] + [30] * 20),
            "f_onset_30hz": np.array([30] * 10 + [60, 60, 60, 60] + [30] * 20),
            "f_last_occ_30hz": np.array([-1] * 10 + [45, 45, 36, 36] + [-1] * 20),
            "f_reappear_30hz": np.array([30] * 10 + [48, 48, 39, 39] + [30] * 20),
            "occ_at_t0": np.array(["none"] * 10 + ["part", "none", "none", "none"] + ["none"] * 20),
            "all_5f_none": np.ones(34, dtype=bool),
        }
        names = ("Lead_M0_kin_only", "Lead_AutoPose_rtmpose17", "Lead_AutoPose_CovisSafe", "Lead_AutoPose_veh", "Lead_HumanAttr_orient_gaze", "Lead_M2_kin_veh")
        base_g = np.linspace(0.01, 0.25, 24)[:, None].repeat(3, axis=1)
        p_va = {nm: np.linspace(0.01, 0.25, 10)[:, None].repeat(3, axis=1) for nm in names}
        p_te = {nm: base_g.copy() for nm in names}
        p_te["Lead_AutoPose_rtmpose17"][0, 0] = 0.95  # event 100 fires at f=42 (occ='part'), silent at f=54
        p_te["Lead_AutoPose_rtmpose17"][2, 0] = 0.95  # event 101 fires at f=42 >= f_reappear=39 (occ='none')
        res = audit_per_event_timestamps_and_visible_alarms(p_va, p_te, slide_ds, meta)
        st = res["strict_recovered_summary_by_class"]["Start"]
        self.assertEqual(st["newly_hit_by_autopose_over_kin_count"], 2)
        self.assertEqual(st["newly_hit_first_alarm_strictly_after_reappearance"], 1)
        self.assertEqual(st["models"]["Lead_AutoPose_rtmpose17"]["protocol2_visible_only_hits"], 1)


if __name__ == "__main__":
    unittest.main()
