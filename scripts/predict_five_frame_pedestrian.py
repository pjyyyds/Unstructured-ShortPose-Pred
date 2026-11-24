"""Forecast one pedestrian from five measured, consecutive 10 Hz observations."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data.actions import _estimate_causal_velocities
from src.data.constants import ACTION_NAMES, DT, T_FUT
from src.data.features import build_motion_features, build_pose_and_quality_features
from src.models.action_and_decoder import ActionConditionedDecoder
from src.models.predictor import UnstructuredShortPosePredictor


def finite_array(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values with shape {shape}")
    return array


def observation_inputs(observation, use_pose):
    times = finite_array(observation["timestamps_s"], (5,), "timestamps_s")
    if not np.allclose(np.diff(times), DT, rtol=0, atol=0.001):
        raise ValueError("Expected five consecutive observations spaced 0.1 s apart (1 ms tolerance)")
    world = finite_array(observation["ground_xy_m"], (5, 2), "ground_xy_m")
    observed = np.ones(5, dtype=np.float32)
    motion = build_motion_features(world - world[-1], observed, dt=DT)
    ego = np.zeros(8, dtype=np.float32)
    ego[2 if motion[-1, 4] < 0.20 else 0] = 1
    inputs = {"x_m": motion, "ego_type_oh": ego}
    if use_pose:
        boxes = np.zeros((5, 4), dtype=np.float32)
        joints = np.zeros((5, 17, 3), dtype=np.float32)
        poses = observation["pose_frames"]
        height = float(observation["image_height_px"])
        if not np.isfinite(height) or height <= 0 or len(poses) != 5:
            raise ValueError("Pose inputs require a positive image height and five pose frames")
        for i, frame in enumerate(poses):
            if frame is None:
                continue
            box = finite_array(frame["bbox_center_xywh_px"], (4,), "bbox_center_xywh_px")
            if min(box[2:]) <= 0:
                raise ValueError("Measured bounding-box width and height must be positive")
            boxes[i] = box
            if frame.get("keypoints_xyc") is not None:
                value = finite_array(frame["keypoints_xyc"], (17, 3), "keypoints_xyc")
                joints[i] = value
        pose, quality, _, _ = build_pose_and_quality_features(joints, boxes, observed, height)
        inputs.update(x_p=pose, q_vec=quality)
    return {name: torch.from_numpy(value).unsqueeze(0) for name, value in inputs.items()}, world, times[-1]


def unpack_model(checkpoint):
    identity = checkpoint["identity"]
    if "arm" in identity:
        # The registered training-only dropout is the identity at inference.
        if identity["registration_sha256"] != "0eb60b827089822cffa096596848f6e0d0e8b47436c715afda9345a2f84ac1cc":
            raise ValueError("Expected the registered five-frame individual joint-action weights")
        family = identity["arm"]
        modes = {"no_head": "none", "auxiliary": "aux_only", "end_to_end": "e2e", "detached": "stop_grad"}
        if family not in modes or modes[family] != identity["mode"]:
            raise ValueError("Unknown action decoder")
        model = UnstructuredShortPosePredictor("M1")
        model.action_mode = modes[family]
        if family != "no_head":
            model.decoder = ActionConditionedDecoder(action_mode=model.action_mode)
        site = identity["heldout"]
        use_pose = True
        use_head = family != "no_head"
    else:
        family = identity.get("model")
        if family not in ("M0", "M1") or identity.get("ordered_ids_sha256") != (
            "f5ee511a54ad818499d3dde6213f7f3b49332943ff2fc93d93aed87d316dbb91"
        ):
            raise ValueError("Expected held-out-site five-frame M0/M1 weights or registered joint-action weights")
        model = UnstructuredShortPosePredictor(family)
        site = identity["test_site"]
        use_pose = family == "M1"
        use_head = False
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.eval(), family, site, use_pose, use_head


@torch.inference_mode()
def forecast(checkpoint, observation):
    model, family, site, use_pose, use_head = unpack_model(checkpoint)
    inputs, world, decision_time = observation_inputs(observation, use_pose)
    result = model(inputs)
    relative = result["trajs"][0].cpu().numpy().astype(np.float64)
    probability = result["mode_probs"][0].cpu().numpy()
    if (relative.shape != (6, T_FUT, 2) or not np.isfinite(relative).all()
            or probability.shape != (6,) or not np.isfinite(probability).all()
            or np.any(probability < 0) or not np.isclose(probability.sum(), 1., atol=1e-6)):
        raise ValueError("Invalid six-mode forecast")
    steps = np.arange(1, T_FUT + 1)
    output = {
        "model": family, "training_held_out_site": site,
        "decision_time_s": float(decision_time),
        "forecast_offsets_s": np.round(steps * DT, 3).tolist(),
        "ground_origin_m": world[-1].tolist(),
        "top_mode_zero_based": int(probability.argmax()),
        "mode_probabilities": probability.tolist(),
        "trajectories_ground_xy_m": (relative + world[-1]).tolist(),
        "last_pair_constant_velocity_ground_xy_m": (
            world[-1] + steps[:, None] * (world[-1] - world[-2])
        ).tolist(),
    }
    if use_head:
        scores = result["act_probs"][0].cpu().numpy()
        if scores.shape != (5,) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError("Invalid action-head probabilities")
        output["observable_action_probabilities"] = dict(zip(ACTION_NAMES, scores.tolist()))
        # These are necessary conditions in the published action definitions. Apply them
        # after decoding so inference still returns the original probabilities and paths.
        _, _, speed = _estimate_causal_velocities(world, np.ones(len(world), dtype=bool), dt=DT)
        pre_speed = float(np.mean(speed[-3:]))
        eligible = {"Start": pre_speed <= 0.20, "Stop": pre_speed >= 0.60}
        output["observable_action_eligibility"] = eligible
        output["pre_event_mean_speed_m_s"] = pre_speed
        output["eligible_action_probabilities"] = {
            name: value if eligible.get(name, True) else 0.0
            for name, value in output["observable_action_probabilities"].items()
        }
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("observation", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    output = json.dumps(forecast(checkpoint, json.loads(args.observation.read_text())), indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(output + "\n")
    else:
        print(output)
