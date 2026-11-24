"""Synthetic regression test for the PINNS held-out-site training entry point."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from scripts.benchmark_heldout_sites import HISTORY, WINDOW, TrackWindow
from scripts.train_pinns_heldout_sites import Protocol, fit_predict, tensorize


class TestHeldOutTraining(unittest.TestCase):
    def test_window_alignment_and_unseen_future(self):
        groups, rows = {}, []
        for i in range(8):
            site = f"synthetic_site_{i // 2}"
            step = np.array([0.01 if i == 0 else 0.08, i * 0.004])
            xy = (np.arange(WINDOW) - HISTORY + 1)[:, None] * step
            row = TrackWindow(site, i, 4, xy, int(i == 1))
            groups.setdefault(site, []).append(row)
            rows.append(row)
        reverse = list(reversed(rows))
        n = len(rows)
        pose = np.broadcast_to([10., 15., .9], (n, HISTORY, 17, 3)).astype("float32").copy()
        inferred = np.ones((n, HISTORY), dtype=bool)
        pose[0] = 0
        inferred[0] = False
        arrays = {
            "clip": np.array([r.clip for r in reverse]), "person_id": [r.person_id for r in reverse],
            "t0": np.full(n, 4), "img_h": np.full(n, 100.), "hist_kpts": pose,
            "hist_bbox": np.broadcast_to([10., 20., 12., 20.], (n, HISTORY, 4)),
            "hist_pos_relative": np.stack([r.xy[:HISTORY] for r in reverse]),
            "hist_mask": np.ones((n, HISTORY)), "pose_inferred": inferred,
            "ego_type": [2 if r.person_id in (0, 1) else 0 for r in reverse],
            "metadata_json": np.array("{}"),
        }
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "pose.npz"
            np.savez(cache, **arrays)
            data, index, keys, cv, coverage, _ = tensorize(groups, cache)
            np.testing.assert_allclose(cv, 0, atol=1e-12)
            self.assertEqual(keys, [(r.clip, r.person_id, r.t0) for r in rows])
            test_site = "synthetic_site_3"
            self.assertEqual(coverage[test_site]["windows_without_valid_joints"], 1)
            invalid = arrays["person_id"].copy()
            invalid[0] = invalid[1]
            np.savez(cache, **{**arrays, "person_id": invalid})
            with self.assertRaisesRegex(ValueError, "exactly one record"):
                tensorize(groups, cache)

        train_ids = np.concatenate([ids for s, ids in index.items() if s != test_site])
        test_ids = index[test_site]
        protocol = Protocol(epochs=1, batch_size=3)
        dev = torch.device("cpu")
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            for model in ("M0", "M1"):
                state, err = fit_predict(model, data, train_ids, test_ids, dev, protocol)
                self.assertEqual(err.shape, (2, 2))
                self.assertTrue(np.isfinite(err).all())
            other_future = data["fut_xy"].clone()
            other_future[test_ids] += 50
            altered = {**data, "fut_xy": other_future}
            repeat, other_err = fit_predict("M1", altered, train_ids, test_ids, dev, protocol)
            self.assertTrue(all(torch.equal(value, repeat[key]) for key, value in state.items()))
            self.assertTrue(np.all(other_err > err + 40))
        finally:
            torch.set_num_threads(threads)


if __name__ == "__main__":
    unittest.main()
