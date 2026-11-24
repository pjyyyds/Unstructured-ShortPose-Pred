"""Replay frozen M1 with recorded, missing, or reverse-ordered five-frame pose inputs."""
import argparse
from pathlib import Path

import numpy as np
import torch

from scripts.analyze_heldout_pairs import metric_dict
from scripts.evaluate_pinns_frame_offset import M1_INPUTS, read_frozen_fold, verify_replay
from scripts.run_heldout_seed_sensitivity import pilot_inputs
from scripts.train_pinns_heldout_sites import Protocol, save_json
from scripts.verify_heldout_bundle import errors_by_window


CONDITIONS = ("recorded", "missing_joints", "reversed_pose_frames")


@torch.inference_mode()
def predict(model, tensors, test, condition):
    paths, modes, gates = [], [], []
    for ids in torch.as_tensor(test).split(Protocol().batch_size):
        batch = {key: tensors[key][ids] for key in M1_INPUTS}
        if condition == "missing_joints":
            batch["x_p"] = torch.zeros_like(batch["x_p"])
            # Joint validity and confidence vanish. Box height and motion availability remain observed.
            batch["q_vec"][:, :2] = 0
        elif condition == "reversed_pose_frames":
            batch["x_p"] = batch["x_p"].flip(1)
        elif condition != "recorded":
            raise ValueError(f"Unknown pose condition: {condition}")
        out = model(batch)
        mode = out["mode_probs"].argmax(1)
        paths.append(out["trajs"][torch.arange(len(ids)), mode].numpy())
        modes.append(mode.numpy())
        gates.append(out["gate_weights"].numpy().reshape(len(ids)))
    return np.concatenate(paths), np.concatenate(modes), np.concatenate(gates)


def evaluate(data_root, pilot_dir):
    tensors, index, keys, pilot, _, _ = pilot_inputs(data_root, pilot_dir)
    sites = {}
    for site, test in index.items():
        site_keys = [keys[i] for i in test]
        model, archived, _ = read_frozen_fold(pilot_dir / "folds" / site, site, site_keys, pilot["identity"])
        truth = tensors["fut_xy"][test].numpy()
        reference, ref_mode, ref_gate = predict(model, tensors, test, "recorded")
        original_errors = errors_by_window(reference, truth)
        verify_replay(original_errors, archived, site)
        if not np.isfinite(ref_gate).all() or np.any((ref_gate < 0) | (ref_gate > 1)):
            raise ValueError(f"Invalid recorded gates at {site}")
        controls = {}
        for name in CONDITIONS[1:]:
            paths, modes, gate = predict(model, tensors, test, name)
            if name == "missing_joints" and np.any(gate != 0):
                raise ValueError("A missing-pose sample retained a nonzero gate")
            errors = errors_by_window(paths, truth)
            if not np.isfinite(errors).all():
                raise ValueError(f"Nonfinite {name} predictions at {site}")
            controls[name] = {
                "errors_m": metric_dict(errors.mean(axis=0)),
                "minus_recorded_m": metric_dict((errors - original_errors).mean(axis=0)),
                "top_mode_changes": int((modes != ref_mode).sum()),
                "mean_distance_between_top_paths_m": float(np.linalg.norm(paths - reference, axis=-1).mean()),
                "mean_gate": float(gate.mean()),
            }
        sites[site] = {"windows": len(test), "clips": len({k[0] for k in site_keys}),
                       "recorded_errors_m": metric_dict(original_errors.mean(axis=0)),
                       "recorded_mean_gate": float(ref_gate.mean()), "controls": controls}
    macro = {name: {m: float(np.mean([s["controls"][name]["minus_recorded_m"][m] for s in sites.values()]))
                    for m in ("ade_m", "fde_m")} for name in CONDITIONS[1:]}
    return {
        "scope": "Post-hoc test-input diagnostics for four fixed, already examined sites and a single trained seed (42).",
        "conditions": {
            "recorded": "Original five-frame pose tensor, motion history, and quality vector.",
            "missing_joints": "All pose channels and joint validity/confidence zero; retain motion, observed-frame ratio and box-height quality.",
            "reversed_pose_frames": "Reverse all five observed pose frames together; the motion history keeps its order and aggregate quality stays fixed. This perturbs both pose order and its alignment with motion.",
        },
        "interpretation": "Positive control minus recorded errors favor the recorded pose input under the same weights. Different training procedures and a physically occluded camera are outside this comparison.",
        "weight_and_input_identity": pilot["identity"], "sites": sites, "equal_site_control_minus_recorded_m": macro,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("pilot_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    save_json(args.output, evaluate(args.data_root, args.pilot_dir))
