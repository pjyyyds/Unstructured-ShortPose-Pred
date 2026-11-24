"""Synthetic tests for the registered two-arm M2a entry; no real predictions or fitting."""

import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import torch
from tests.test_invariants import TestMathematicalInvariants
from reviews.pinns_map_axis_comparison.neural import run_neural as r


def run_tests():
    torch.set_num_threads(1)
    splits = {}
    for split in ("train", "val", "test"):
        examples = TestMathematicalInvariants._dummy_batch(bsz=6)
        examples["bucket"] = torch.tensor([0, 1, 2, 0, 1, 2])
        examples["fut_mask"][0, -1] = 0
        splits[split] = examples
    original = splits["train"]["p_env"].clone()
    control = r.arm_tensors(splits["train"], "zero_xy")
    assert torch.all(control["p_env"][..., :2] == 0)
    assert torch.equal(control["p_env"][..., 2:], original[..., 2:])
    assert torch.equal(splits["train"]["p_env"], original)
    for k, val in splits["train"].items():
        if k != "p_env":
            assert control[k] is val
    # Exercise the exact unmodified trainer, capture the selected forward once and verify masks.
    cfg = dict(r.CFG, device_str="cpu", epochs=4, batch_size=6)
    fingerprints = []
    with patch.object(r, "CFG", cfg):
        for arm in r.ARMS:
            values = r.train_arm(splits, arm, "synthetic")
            meta = json.loads(values["fit.json"])
            fingerprints.append(
                (meta["initial_state_sha256"], meta["sampled_indices_sha256"], meta["parameters"])
            )
            assert meta["best_epoch"] == 4 and meta["sampled_windows"] == 24 and meta["minibatches"] == 4
            with np.load(io.BytesIO(values["predictions.npz"]), allow_pickle=False) as a:
                assert a["paths"].shape == (6, 6, 30, 2)
                scores = r.top_errors(
                    a["paths"],
                    a["probabilities"],
                    splits["test"]["fut_xy"].numpy(),
                    splits["test"]["fut_mask"].numpy().astype(bool),
                )
                assert np.isfinite(scores[:, :2]).all() and np.isnan(scores[0, -1])
            with tempfile.TemporaryDirectory() as d:
                path = Path(d) / "arm.zip"
                r.write_archive(path, values)
                assert r.read_or_quarantine(path, values.keys(), "synthetic", arm) is not None
                assert (
                    r.read_or_quarantine(path, values.keys(), "unrelated", arm) is None and not path.exists()
                )
        assert fingerprints[0] == fingerprints[1]
    pred = np.zeros((2, 6, 30, 2))
    pred[:, 1] = 100.0
    # An exact tie chooses candidate zero. The shorter row's FDE is at its actual last annotated point.
    prob = np.zeros((2, 6))
    prob[:, :2] = 0.5
    gt = np.zeros((2, 30, 2))
    gt[:, 0, 0] = 2
    gt[:, -1, 0] = 9
    mask = np.zeros((2, 30), bool)
    mask[:, 0] = True
    mask[1, -1] = True
    out = r.top_errors(pred, prob, gt, mask)
    np.testing.assert_allclose(out[:, :2], [[2, 2], [5.5, 9]])
    assert np.isnan(out[0, 2]) and out[1, 2] == 9
    inter = r.estimate(np.arange(4.0), np.array(["A", "A", "B", "B"]), np.array(["x", "x", "y", "y"]))
    assert inter["estimate_m"] == 1.5 and inter["clip_bootstrap_percentile_95_m"] == [1.5, 1.5]
    print(
        "Passed synthetic trainer replay, identical initialization and sampled minibatches, archive recovery, signed metric and mask checks."
    )


if __name__ == "__main__":
    run_tests()
