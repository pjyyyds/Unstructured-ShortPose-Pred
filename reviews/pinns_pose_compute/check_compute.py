"""Count the frozen predictor's operations and compare its two executions on synthetic inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.count_forward_ops import count_float_elements
from src.models.pose_inference import prepare_pose_inference
from src.models.predictor import UnstructuredShortPosePredictor
from tests.test_invariants import TestMathematicalInvariants

WEIGHT = ROOT / "checkpoints/PINNS_T5_Ours_Full_seed42.pt"
HISTORIES = (5, 10, 20)


def operation_counts(model, batch):
    arithmetic = (torch.ops.aten.add, torch.ops.aten.sub, torch.ops.aten.mul)
    result = {}
    for name, mapping in (("default", {}), ("with_float_add_sub_mul", dict.fromkeys(arithmetic, count_float_elements))):
        with FlopCounterMode(display=False, custom_mapping=mapping) as counter:
            model(batch)
        result[name] = {"total": counter.get_total_flops(), "by_op": {
            str(op): val for op, val in counter.flop_counts["Global"].items()}}
    return result


def expected_reduction(model, people, frames):
    layers = (model.pose_enc.block1, model.pose_enc.block2, model.pose_enc.block3)
    return sum(2 * people * frames * 17 ** 2 * (b.nbr_lin.out_features - b.nbr_lin.in_features)
               for b in layers)


def compare_outputs(original, optimized, batch):
    with torch.inference_mode():
        before, after = original(batch), optimized(batch)
    if before.keys() != after.keys():
        raise AssertionError("Different model outputs")
    worst = {}
    for key in before:
        x, y = before[key], after[key]
        tol = 0.0002 if key == "trajs" else 0.00002
        torch.testing.assert_close(x, y, atol=tol, rtol=0)
        worst[key] = float((x - y).abs().max())
    ranks = before["mode_probs"].argmax(1) != after["mode_probs"].argmax(1)
    return {"max_abs": worst, "top1_changes": int(ranks.sum()), "windows": len(ranks)}


def run():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    original = UnstructuredShortPosePredictor("Ours_Full").eval()
    original.load_state_dict(torch.load(WEIGHT, map_location="cpu", weights_only=True), strict=True)
    optimized = prepare_pose_inference(original)
    before, after = original.state_dict(), optimized.state_dict()
    assert set(before) == set(after) and all(torch.equal(val, after[key]) for key, val in before.items())
    report = {"checkpoint_sha256": hashlib.sha256(WEIGHT.read_bytes()).hexdigest(),
              "torch": str(torch.__version__), "input_source": "synthetic registered budget fixture",
              "parameters": sum(p.numel() for p in original.parameters()), "state_entries_identical": len(before),
              "by_history": {}}
    for t in HISTORIES:
        batch = TestMathematicalInvariants()._dummy_batch(bsz=12, seq_t=t)
        counts = {name: operation_counts(model, batch) for name, model in (("original", original), ("optimized", optimized))}
        saving = expected_reduction(original, 12, t)
        for method in counts["original"]:
            assert counts["original"][method]["total"] - counts["optimized"][method]["total"] == saving
        outputs = {}
        for case in ("original_fixture", "partial_pose", "missing_pose", "empty_context"):
            sample = {name: value.clone() for name, value in batch.items()}
            if case == "partial_pose":
                sample["x_p"][:, ::2, ::2, 3] = 0
            if case == "missing_pose":
                for name in ("x_p", "q_vec", "q_part"):
                    sample[name].zero_()
            if case == "empty_context":
                for name in ("m_map", "p_env", "nbr_mask", "nbr_edges", "nbr_type_oh"):
                    sample[name].zero_()
            outputs[case] = compare_outputs(original, optimized, sample)
        report["by_history"][str(t)] = {"batch_size": 12, "saved_flops": saving,
                                         "counts": counts, "synthetic_output_check": outputs}
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for t, v in result["by_history"].items():
        print(t, v["saved_flops"], {k: w["total"] for k, w in v["counts"]["optimized"].items()})
