import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from scripts import run_pinns_matched_fusion as run
from scripts.train_pinns_heldout_sites import Protocol, fit_predict
from scripts.verify_heldout_bundle import run_cpu_forward
from src.models.predictor import UnstructuredShortPosePredictor


def synthetic():
    gen = torch.Generator().manual_seed(11)
    n, t = 8, 5
    motion = torch.randn(n, t, 9, generator=gen)
    motion[:, :, 5] = 1.0
    pose = torch.randn(n, t, 17, 6, generator=gen)
    pose[:, :, :, 2:5] = 1.0
    return {
        "x_m": motion,
        "x_p": pose,
        "q_vec": torch.ones(n, 4),
        "ego_type_oh": torch.eye(8),
        "fut_xy": torch.randn(n, 30, 2, generator=gen),
        "fut_mask": torch.ones(n, 30),
    }


class MatchedFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_concat_initialization_preserves_backbone_rng_and_missing_pose(self):
        torch.manual_seed(23)
        archived = UnstructuredShortPosePredictor("M1a")
        rng_after = torch.get_rng_state()
        torch.manual_seed(23)
        new = run.make_model("M1-concat")
        self.assertTrue(torch.equal(rng_after, torch.get_rng_state()))
        old, state = archived.state_dict(), new.state_dict()
        self.assertEqual(set(state) - set(old), {"quality_gate.weight"})
        self.assertEqual(set(old) - set(state), set())
        for k in old:
            self.assertTrue(torch.equal(state[k], old[k]), k)
        eye = torch.eye(128)
        torch.testing.assert_close(new.quality_gate.weight, torch.cat([eye, eye], dim=1), atol=0, rtol=0)
        motion, pose = torch.randn(3, 128), torch.randn(3, 128)
        q = torch.tensor([[0.65, 0.8, 0.1, 1.0], [0.0, 0.9, 0.9, 1.0], [0.4, 0.1, 0.1, 1.0]])
        fused, mask = new.quality_gate(motion, pose, q)
        old_fused, old_mask = archived.quality_gate(motion, pose, q)
        torch.testing.assert_close(fused, old_fused, atol=0, rtol=0)
        torch.testing.assert_close(mask, old_mask, atol=0, rtol=0)
        with torch.no_grad():
            new.quality_gate.weight[:, :128].mul_(0.25)
            new.quality_gate.weight[:, 128:].mul_(-2.0)
        pose[1] = float("nan")
        fused, mask = new.quality_gate(motion, pose, q)
        torch.testing.assert_close(fused[1], new.quality_gate.out_norm(motion)[1], atol=0, rtol=0)
        torch.testing.assert_close(
            fused[0], new.quality_gate.out_norm(0.25 * motion[0] - 2 * pose[0]), atol=1e-6, rtol=1e-6
        )
        self.assertTrue(torch.isfinite(fused).all())

    def test_original_implicit_gate_and_new_training_loop_are_bitwise_identical(self):
        tensors = synthetic()
        train, test = np.array([4, 1, 0, 2, 3, 7]), np.array([6, 5])
        config = Protocol(seed=19, epochs=2, batch_size=3)
        old, old_errors = fit_predict("M1b", tensors, train, test, torch.device("cpu"), config)
        state, detail = run.fit_variant("M1b", tensors, train, test, torch.device("cpu"), config)
        self.assertEqual(set(state), set(old))
        for k in old:
            self.assertTrue(torch.equal(old[k], state[k]), k)
        np.testing.assert_array_equal(detail["errors"].numpy(), old_errors)
        run.check_predictions(detail, tensors["fut_xy"][test].numpy())
        replay = run_cpu_forward(run.make_model, "M1b", state, tensors, test, config.batch_size)
        np.testing.assert_allclose(replay, old_errors, atol=run.ATOL_M, rtol=run.RTOL)

    def test_concat_trainable_projection_saves_and_replays(self):
        tensors = synthetic()
        train = np.arange(6)
        test = np.array([7, 6])
        conf = Protocol(seed=12, epochs=1, batch_size=3)
        state, details = run.fit_variant("M1-concat", tensors, train, test, torch.device("cpu"), conf)
        self.assertFalse(torch.equal(state["quality_gate.weight"], torch.cat([torch.eye(128)] * 2, dim=1)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.pt"
            torch.save({"model_state": state, "predictions": details}, path)
            saved = torch.load(path, map_location="cpu", weights_only=True)
            replay = run_cpu_forward(
                run.make_model, "M1-concat", saved["model_state"], tensors, test, conf.batch_size
            )
        measured = run.check_predictions(saved["predictions"], tensors["fut_xy"][test].numpy())
        np.testing.assert_allclose(replay, measured, atol=run.ATOL_M, rtol=run.RTOL)
        fresh = run.replay_saved("M1-concat", state, details, tensors, test, conf.batch_size)
        np.testing.assert_allclose(fresh, measured, atol=run.ATOL_M, rtol=run.RTOL)
        damaged = {k: v.clone() for k, v in details.items()}
        loser = int(damaged["probs"][0].argmin())
        damaged["trajs"][0, loser, 0, 0] += 0.5
        with self.assertRaises(AssertionError):
            run.replay_saved("M1-concat", state, damaged, tensors, test, conf.batch_size)
        for m in ("M0", "M1a", "M1"):
            with self.assertRaises(ValueError):
                run.make_model(m)
