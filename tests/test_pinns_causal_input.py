"""Small regression fixtures for the map join's motion-identity checks."""
import unittest

import numpy as np

from src.data.pinns_causal_input import full_motion_fingerprint, motion_fingerprint


def sample(person=1):
    hist = np.arange(40, dtype=np.float32).reshape(20, 2)
    future = np.arange(60, dtype=np.float32).reshape(30, 2)
    hmask = np.ones(20, dtype=np.float32)
    fmask = np.ones(30, dtype=np.float32)
    fmask[-1] = 0
    future[-1] = 0
    return dict(seq_id="clip", person_id=person, t0_frame_10hz=32, split="train",
                hist_pos_20=hist, hist_mask_20=hmask, p_t0=hist[-1].copy(),
                fut_pos_abs=future, fut_mask=fmask, fut_xy=(future-hist[-1])*fmask[:, None])


class CausalIdentityTests(unittest.TestCase):
    def test_fractional_identity_cannot_alias_an_original_window(self):
        a = sample(1.5)
        with self.assertRaisesRegex(ValueError, "integer"):
            full_motion_fingerprint([a])

    def test_full_history_and_absolute_ground_frame_are_registered(self):
        a, b = sample(1), sample(2)
        self.assertEqual(full_motion_fingerprint([a, b]), full_motion_fingerprint([b, a]))
        before = full_motion_fingerprint([a])
        short = motion_fingerprint([a])
        a["hist_pos_20"][0, 1] += 1
        self.assertEqual(short, motion_fingerprint([a]))
        self.assertNotEqual(before, full_motion_fingerprint([a]))
        a = sample()
        a["p_t0"] += 1
        a["hist_pos_20"] += 1
        a["fut_pos_abs"][:-1] += 1
        self.assertEqual(short, motion_fingerprint([a]))
        self.assertNotEqual(before, full_motion_fingerprint([a]))

    def test_both_time_axes_require_binary_masks_and_zero_padding(self):
        for field, index in (("fut_mask", 3), ("hist_mask_20", 0)):
            for value in (.25, np.nan, -1):
                with self.subTest(field=field, value=value):
                    a = sample()
                    a[field][index] = value
                    with self.assertRaisesRegex(ValueError, "binary"):
                        full_motion_fingerprint([a])
        a = sample()
        a["fut_pos_abs"][-1, 0] = 1
        with self.assertRaisesRegex(ValueError, "zero-padded"):
            full_motion_fingerprint([a])

    def test_anchor_and_both_supervision_coordinate_fields_must_agree(self):
        for field, index, message in (("p_t0", 0, "anchor"), ("fut_xy", (2, 1), "relative"),
                                      ("fut_pos_abs", (2, 1), "relative")):
            with self.subTest(field=field):
                a = sample()
                a[field][index] += 1
                with self.assertRaisesRegex(ValueError, message):
                    full_motion_fingerprint([a])


if __name__ == "__main__":
    unittest.main()
