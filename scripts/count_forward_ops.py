"""Count the V2 predictor's arithmetic on the same input as its budget test."""

import json

import torch
from torch.utils.flop_counter import FlopCounterMode

from src.models.predictor import UnstructuredShortPosePredictor
from tests.test_invariants import TestMathematicalInvariants


def count_float_elements(*args, out_val, **kwargs):
    """One arithmetic operation per floating output element of add, sub or mul."""
    if isinstance(out_val, torch.Tensor) and out_val.is_floating_point():
        return out_val.numel()
    return 0


# PyTorch normally passes shapes to a custom counter. The output dtype is
# needed here to exclude integer and boolean operations from the count.
count_float_elements._get_raw = True


def main():
    torch.manual_seed(42)
    batch = TestMathematicalInvariants()._dummy_batch(bsz=12, seq_t=20)
    model = UnstructuredShortPosePredictor("Ours_Full").eval()
    arithmetic = (torch.ops.aten.add, torch.ops.aten.sub, torch.ops.aten.mul)
    mappings = {"default": {}, "with_float_add_sub_mul": dict.fromkeys(arithmetic, count_float_elements)}
    counts = {}
    for name, mapping in mappings.items():
        # Match the test's forward path. Some torch versions use autograd
        # hooks for operator attribution even though no backward pass is run.
        with FlopCounterMode(display=False, custom_mapping=mapping) as counter:
            model(batch)
        counts[name] = {
            "total": counter.get_total_flops(),
            "by_operator": {str(op): value for op, value in counter.flop_counts["Global"].items()},
        }
    print(json.dumps({"torch": str(torch.__version__), "targets": 12, "history_frames": 20, **counts}, indent=2))


if __name__ == "__main__":
    main()
