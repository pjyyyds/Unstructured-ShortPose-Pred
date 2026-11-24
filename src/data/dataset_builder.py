"""Dataset root runners (PINNS 10Hz & JTA 2.5Hz) and T-window tensor materializer."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import numpy as np
import torch

from .actions import compute_action_labels
from .constants import DT, NUM_JOINTS, T_FUT
from .features import build_motion_features, build_pose_and_quality_features
from .pinns_extractor import RTMPoseONNXExtractor, process_pinns_sequence
from .scene_graph import build_neighbor_edges
from .splits import verify_zero_temporal_leakage


def process_pinns_root(
    pinns_root: Path,
    onnx_path: Optional[Path] = None,
    window_stride: int = 3,
) -> Tuple[List[Dict], Dict]:
    """Process all PINNS sequences under `pinns_root` and verify zero temporal leakage."""
    labels_dir = pinns_root / "labels"
    videos_dir = pinns_root / "videos"
    pose_ext = RTMPoseONNXExtractor(onnx_path=onnx_path, batch_size=256)

    all_samples: List[Dict] = []
    bbox_is_tl: Optional[bool] = None
    seq_counts: Dict[str, int] = {}

    for lbl_file in sorted(labels_dir.glob("*.txt")):
        if lbl_file.name.lower().startswith("readme"):
            continue
        stem = lbl_file.stem
        if "_" not in stem:
            continue
        scene_folder, vid_idx = stem.rsplit("_", 1)
        scene_path = videos_dir / scene_folder
        ori_mp4 = scene_path / "videos" / f"ori_{vid_idx}.mp4"
        bev_mp4 = scene_path / "videos" / f"bev_{vid_idx}.mp4"
        calib_file = scene_path / "calib.json"
        if not ori_mp4.exists() or not calib_file.exists():
            continue
        with open(calib_file, "r", encoding="utf-8") as cf:
            calib = json.load(cf)

        seq_samps, bbox_is_tl = process_pinns_sequence(
            seq_stem=stem,
            lbl_path=lbl_file,
            ori_video_path=ori_mp4,
            bev_video_path=bev_mp4 if bev_mp4.exists() else None,
            calib=calib,
            pose_ext=pose_ext,
            bbox_is_topleft=bbox_is_tl,
            window_stride=window_stride,
        )
        seq_counts[stem] = len(seq_samps)
        all_samples.extend(seq_samps)

    leak_ok, leak_msg = verify_zero_temporal_leakage(all_samples, max_hist_steps=19, fut_steps=T_FUT)
    if not leak_ok:
        raise RuntimeError(f"CRITICAL LEAKAGE AUDIT FAILURE: {leak_msg}")

    split_dist = {sp: sum(1 for s in all_samples if s["split"] == sp) for sp in ("train", "val", "test")}
    audit = {
        "total_samples": len(all_samples),
        "num_sequences": len(seq_counts),
        "split_counts": split_dist,
        "leakage_check": leak_msg,
        "rtmpose_providers": pose_ext.active_providers,
        "bbox_is_topleft": bool(bbox_is_tl),
    }
    return all_samples, audit


def materialize_window_split(
    raw_samples: List[Dict],
    split_name: str,
    t_obs: int = 5,
    dt: float = DT,
    speed_scale: float = 1.0,
    angle_scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Slice trailing `t_obs` frames from raw samples of `split_name` into contiguous PyTorch tensors."""
    sub = [s for s in raw_samples if s["split"] == split_name]
    n = len(sub)
    if n == 0:
        raise ValueError(f"Zero samples found for split={split_name}")

    xm_arr = np.zeros((n, t_obs, 9), dtype=np.float32)
    xp_arr = np.zeros((n, t_obs, NUM_JOINTS, 6), dtype=np.float32)
    qv_arr = np.zeros((n, 4), dtype=np.float32)
    qp_arr = np.zeros((n, 6), dtype=np.float32)
    bk_arr = np.zeros(n, dtype=np.int64)
    fut_t = sub[0]["fut_xy"].shape[0]

    penv_arr = np.zeros((n, 16, 10, 11), dtype=np.float32)
    mmap_arr = np.zeros((n, 1), dtype=np.float32)
    ego_arr = np.zeros((n, 8), dtype=np.float32)
    edge_arr = np.zeros((n, 12, 9), dtype=np.float32)
    nbrt_arr = np.zeros((n, 12, 8), dtype=np.float32)
    nbrm_arr = np.zeros((n, 12), dtype=np.float32)
    fxy_arr = np.zeros((n, fut_t, 2), dtype=np.float32)
    fmsk_arr = np.zeros((n, fut_t), dtype=np.float32)
    yact_arr = np.zeros((n, 5), dtype=np.float32)
    mact_arr = np.zeros(n, dtype=np.float32)

    recomp_act = abs(speed_scale - 1.0) > 1e-6 or abs(angle_scale - 1.0) > 1e-6
    for i, s in enumerate(sub):
        hp = s["hist_pos_20"][-t_obs:]
        hm = s["hist_mask_20"][-t_obs:]
        hk = s["hist_kpts_20"][-t_obs:]
        hb = s["hist_bbox_20"][-t_obs:]

        xm_arr[i] = build_motion_features(hp, hm, dt=dt)
        xp_arr[i], qv_arr[i], qp_arr[i], bk_arr[i] = build_pose_and_quality_features(
            hk, hb, hm, img_h=float(s["img_h"])
        )
        penv_arr[i] = s["p_env"]
        mmap_arr[i] = s["m_map"]
        ego_arr[i] = s["ego_type_oh"]
        edge_arr[i] = s["nbr_edges"]
        nbrt_arr[i] = s["nbr_type_oh"]
        nbrm_arr[i] = s["nbr_mask"]
        fxy_arr[i] = s["fut_xy"]
        fmsk_arr[i] = s["fut_mask"]
        if recomp_act:
            yact_arr[i], mact_arr[i] = compute_action_labels(
                s["hist_pos_20"][-5:],
                s["hist_mask_20"][-5:],
                s["fut_pos_abs"],
                s["fut_mask"],
                dt=dt,
                speed_scale=speed_scale,
                angle_scale=angle_scale,
            )
        else:
            yact_arr[i] = s["y_act"]
            mact_arr[i] = s["m_act"]

    return {
        "x_m": torch.from_numpy(xm_arr),
        "x_p": torch.from_numpy(xp_arr),
        "q_vec": torch.from_numpy(qv_arr),
        "q_part": torch.from_numpy(qp_arr),
        "bucket": torch.from_numpy(bk_arr),
        "p_env": torch.from_numpy(penv_arr),
        "m_map": torch.from_numpy(mmap_arr),
        "ego_type_oh": torch.from_numpy(ego_arr),
        "nbr_edges": torch.from_numpy(edge_arr),
        "nbr_type_oh": torch.from_numpy(nbrt_arr),
        "nbr_mask": torch.from_numpy(nbrm_arr),
        "fut_xy": torch.from_numpy(fxy_arr),
        "fut_mask": torch.from_numpy(fmsk_arr),
        "y_act": torch.from_numpy(yact_arr),
        "m_act": torch.from_numpy(mact_arr),
    }
