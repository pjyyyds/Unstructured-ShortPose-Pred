"""Regression checks for the paired held-site frame-offset comparison."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.evaluate_pinns_frame_offset import (
    CLIPS, check_cache_changes, expected_control_batches, summarize, verify_pose_controls, verify_replay,
)


class TestFrameOffsetEvaluation(unittest.TestCase):
    def test_only_selected_pose_fields_can_change_after_key_alignment(self):
        target = next(iter(CLIPS))
        keys = [(target, 7, 4), (target, 7, 39), ("unshifted_clip", 1, 8)]
        rows = {
            "clip": np.array([k[0] for k in keys]),
            "person_id": np.array([k[1] for k in keys]),
            "t0": np.array([k[2] for k in keys]),
            "hist_kpts": np.zeros((3, 5, 17, 3), dtype=np.float32),
            "pose_inferred": np.ones((3, 5), dtype=bool),
            "hist_bbox": np.ones((3, 5, 4), dtype=np.float32),
        }

        with tempfile.TemporaryDirectory() as tmp:
            base, shifted = Path(tmp) / "base.npz", Path(tmp) / "shifted.npz"
            np.savez(base, **rows, metadata_json="{}")
            changed = {k: v[::-1].copy() for k, v in rows.items()}
            changed["hist_kpts"][2, 1, 8, 0] = 2
            np.savez(shifted, **changed, metadata_json="{}")
            self.assertEqual(check_cache_changes(base, shifted, keys)[target], 1)
            changed["hist_bbox"][2, 1, 0] = 7
            np.savez(shifted, **changed, metadata_json="{}")
            with self.assertRaisesRegex(ValueError, "non-pose input"):
                check_cache_changes(base, shifted, keys)
            changed["hist_bbox"][2, 1, 0] = 1
            changed["pose_inferred"][0, 2] = False
            np.savez(shifted, **changed, metadata_json="{}")
            with self.assertRaisesRegex(ValueError, "outside the three"):
                check_cache_changes(base, shifted, keys)
            changed["pose_inferred"][0, 2] = True
            changed["t0"][2] = changed["t0"][1]
            np.savez(shifted, **changed, metadata_json="{}")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                check_cache_changes(base, shifted, keys)

    def test_pose_controls_require_matched_runtime_and_batches(self):
        clips = list(CLIPS) * 3
        with tempfile.TemporaryDirectory() as tmp:
            npz = Path(tmp) / "pose.npz"
            np.savez(npz, clip=np.array(clips), t0=np.array([4] * 3 + [6] * 3 + [12] * 3),
                     hist_bbox=np.ones((9, 5, 4)))
            batches = expected_control_batches(npz)
        self.assertEqual(batches, {c: {0: 1, 6: 2, 36: 1} for c in CLIPS})
        providers = ["CPUExecutionProvider"]
        control = {"sampled_controls_passed": True, "xy_atol_px": .001, "confidence_atol": .0001,
                   "allowed_confidence_0_2_crossings": 0}
        details = {c: {"zero_offset_controls": [
            {"raw_frame": f, "crops": n, "xy_max_abs_px": 0., "confidence_max_abs": 0.,
             "confidence_0_2_crossings": 0} for f, n in batches[c].items()]} for c in CLIPS}
        record = {"providers": providers, "zero_offset_control": control, "clips": details}
        verify_pose_controls(record, providers, batches)
        for invalid, reason in (({**record, "providers": ["other"]}, "providers"),
                                ({**record, "zero_offset_control": {}}, "controls or tolerances"),
                                ({**record, "zero_offset_control": {**control, "sampled_controls_passed": False}},
                                 "controls or tolerances"),
                                ({**record, "clips": {}}, "three clips")):
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                verify_pose_controls(invalid, providers, batches)
        bad = deepcopy(record)
        values = bad["clips"][clips[0]]["zero_offset_controls"]
        values[0]["confidence_0_2_crossings"] = 1
        with self.assertRaisesRegex(ValueError, "outside tolerance"):
            verify_pose_controls(bad, providers, batches)
        values.pop(0)
        with self.assertRaisesRegex(ValueError, "missing or incorrect"):
            verify_pose_controls(bad, providers, batches)

    def test_baseline_check_is_paired_before_averaging(self):
        saved = np.array([[1., 3.], [2., 4.]])
        with self.assertRaisesRegex(ValueError, "frozen replay differs"):
            verify_replay(saved[::-1], saved, "test")
        with self.assertRaisesRegex(ValueError, "invalid frozen replay"):
            verify_replay(saved[:1], saved, "test")
        verify_replay(saved + 1e-6, saved, "test")
        scores = summarize(saved, saved + [0.5, -1.])
        self.assertEqual(scores["shifted_minus_frozen_m"], {"ade_m": 0.5, "fde_m": -1.})


if __name__ == "__main__":
    unittest.main()
