"""Variable-clock O(2) parity-equivariant pedestrian trajectory & 5-action CLI predictor (`Ours_O2_Full`)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data.constants import ACTION_NAMES
from src.models.bioeq_neural import biomech_sequence_torch
from src.models.o2_unified_full import O2UnifiedFullForecaster


def forecast_observation(obs: dict, ckpt_path: Path | None = None) -> dict:
    times = np.asarray(obs["timestamps_s"], dtype=np.float64)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError("timestamps_s must be strictly increasing with >= 2 observations")
    world = np.asarray(obs["ground_xy_m"], dtype=np.float64)
    if world.shape != (len(times), 2) or not np.isfinite(world).all():
        raise ValueError("ground_xy_m must have finite shape (T, 2) matching timestamps_s")

    t_obs = len(times)
    origin = world[-1:].copy()
    hist_rel = torch.from_numpy(world - origin).unsqueeze(0)
    obs_dt = torch.from_numpy(np.diff(times)).unsqueeze(0)
    fut_dt = float(obs.get("future_step_s", 0.1))
    fut_steps = int(obs.get("future_steps", 30))
    fut_s = torch.arange(1, fut_steps + 1, dtype=torch.float64)[None, :] * fut_dt

    bio_seq, q_bio, bsc = None, None, 0.0
    if obs.get("pose_frames") is not None:
        poses = obs["pose_frames"]
        if len(poses) != t_obs:
            raise ValueError("pose_frames length must match timestamps_s")
        boxes = np.zeros((1, t_obs, 4), dtype=np.float32)
        kpts = np.zeros((1, t_obs, 17, 3), dtype=np.float32)
        for i, fr in enumerate(poses):
            if fr is not None:
                boxes[0, i] = np.asarray(fr["bbox_center_xywh_px"], dtype=np.float32)
                if fr.get("keypoints_xyc") is not None:
                    kpts[0, i] = np.asarray(fr["keypoints_xyc"], dtype=np.float32)
        bio_seq, q_bio = biomech_sequence_torch(torch.from_numpy(kpts), torch.from_numpy(boxes))
        bsc = 0.5

    edges = torch.zeros((1, 12, 9), dtype=torch.float32)
    is_car = torch.zeros((1, 12), dtype=torch.float32)
    nbr_mask = torch.zeros((1, 12), dtype=torch.float32)
    gsc = 0.0
    if obs.get("neighbors"):
        v_ego = (world[-1] - world[-2]) / max(float(times[-1] - times[-2]), 1e-4)
        cand = []
        for nb in obs["neighbors"]:
            dp = np.asarray(nb["ground_xy_m"], dtype=np.float64) - world[-1]
            d = float(np.hypot(dp[0], dp[1]))
            if 1e-5 < d <= 10.0:
                cand.append((d, dp, nb))
        cand.sort(key=lambda z: z[0])
        for j, (d, dp, nb) in enumerate(cand[:12]):
            has_v = nb.get("velocity_xy_mps") is not None
            dv = (np.asarray(nb["velocity_xy_mps"], dtype=np.float64) - v_ego) if has_v else np.zeros(2)
            rad = float(-(dp @ dv) / (d + 1e-6)) if has_v else 0.0
            edges[0, j] = torch.tensor([dp[0], dp[1], dv[0], dv[1], d, rad, float(has_v), 1.0, 1.0], dtype=torch.float32)
            is_car[0, j] = 1.0 if str(nb.get("kind", "Person")).lower() == "car" else 0.0
            nbr_mask[0, j] = 1.0
        gsc = 0.5

    if ckpt_path is None:
        torch.manual_seed(42)
    net = O2UnifiedFullForecaster().eval()
    if ckpt_path is not None:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        net.load_state_dict(ck["model_state"] if "model_state" in ck else ck, strict=True)
        bsc = float(ck.get("bio_scale", bsc))
        gsc = float(ck.get("graph_scale", gsc))

    with torch.inference_mode():
        out = net(hist_rel, bio_seq, q_bio, edges, is_car, nbr_mask, bio_scale=bsc, graph_scale=gsc, obs_dt=obs_dt, fut_times_s=fut_s)
        paths_w = (out["paths"][0].numpy() + origin[None, :, :])
        probs = out["logits"][0].softmax(dim=0).numpy()
        bayes_w = np.einsum("k,ktw->tw", probs, paths_w)
        pa = torch.sigmoid(out["act_logits"][0]).numpy()

    return {
        "model": "Ours_O2_Full",
        "origin_timestamp_s": float(times[-1]),
        "future_timestamps_s": (times[-1] + fut_s[0].numpy()).tolist(),
        "selected_mode": int(probs.argmax()),
        "mode_probabilities": probs.tolist(),
        "candidate_trajectories_xy_m": paths_w.tolist(),
        "bayesian_trajectory_xy_m": bayes_w.tolist(),
        "action_probabilities": {k: float(v) for k, v in zip(ACTION_NAMES, pa.tolist())},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("observation_json", type=Path)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    res = forecast_observation(json.loads(args.observation_json.read_text()), args.checkpoint)
    txt = json.dumps(res, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(txt)
    else:
        print(txt, end="")


if __name__ == "__main__":
    main()
