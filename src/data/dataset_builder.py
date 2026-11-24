"""Dataset root runners (PINNS 10Hz & JTA 2.5Hz) and T-window tensor materializer."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import time
import numpy as np
import torch

from .actions import compute_action_labels
from .constants import DT, NUM_JOINTS, T_FUT
from .features import build_motion_features, build_pose_and_quality_features
from .pinns_extractor import (
    RTMPoseONNXExtractor,
    _check_right_handed_homography,
    _parse_pinns_label_file,
    extract_static_bev_and_train_polylines,
    process_pinns_sequence,
)
from .scene_graph import build_neighbor_edges
from .splits import assign_site_clip_splits, verify_zero_temporal_leakage


def _build_site_train_polylines(train_entries: List[Tuple]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build static BEV polylines for one camera site strictly from its train clips only."""
    if not train_entries:
        return []
    _, _, _, _, _, first_bev, calib = train_entries[0]
    flip_y = _check_right_handed_homography(calib)
    y_min, y_max = float(calib.get("y_min", 0.0)), float(calib.get("y_max", 30.0))
    train_ped_pts, parked_pts = [], []
    for _, _, _, lbl_file, _, _, _ in train_entries:
        by_f10 = _parse_pinns_label_file(lbl_file, flip_y, y_min, y_max)
        for objs in by_f10.values():
            for o in objs:
                if o["type"].lower().startswith("per"):
                    train_ped_pts.append([o["wx"], o["wy"]])
                elif o["fixed"] == 1:
                    parked_pts.append([o["wx"], o["wy"]])
    ped_arr = np.array(train_ped_pts, dtype=np.float32) if train_ped_pts else np.zeros((0, 2), dtype=np.float32)
    pk_arr = np.unique(np.round(parked_pts, 1), axis=0) if parked_pts else np.zeros((0, 2), dtype=np.float32)
    return extract_static_bev_and_train_polylines(first_bev, calib, ped_arr, pk_arr, flip_y)


def process_pinns_root(
    pinns_root: Path,
    onnx_path: Optional[Path] = None,
    window_stride: int = 3,
) -> Tuple[List[Dict], Dict]:
    """Process all PINNS sequences under `pinns_root` with stratified anti-leakage site splits."""
    import queue

    labels_dir = pinns_root / "labels"
    videos_dir = pinns_root / "videos"
    by_site: Dict[str, List[Tuple]] = {}

    for lbl_file in sorted(labels_dir.glob("*.txt")):
        if lbl_file.name.lower().startswith("readme") or "_" not in lbl_file.stem:
            continue
        stem = lbl_file.stem
        scene_folder, vid_str = stem.rsplit("_", 1)
        vid_num = int(vid_str) if vid_str.isdigit() else abs(hash(vid_str)) % 10000
        scene_path = videos_dir / scene_folder
        ori_mp4 = scene_path / "videos" / f"ori_{vid_str}.mp4"
        bev_mp4 = scene_path / "videos" / f"bev_{vid_str}.mp4"
        calib_file = scene_path / "calib.json"
        if not ori_mp4.exists() or not calib_file.exists():
            continue
        calib = json.loads(calib_file.read_text(encoding="utf-8"))
        by_site.setdefault(scene_folder, []).append(
            (scene_folder, vid_num, stem, lbl_file, ori_mp4, bev_mp4 if bev_mp4.exists() else None, calib)
        )

    jobs: List[Tuple] = []
    for site, entries in sorted(by_site.items()):
        entries.sort(key=lambda x: x[1])
        clip_map = assign_site_clip_splits([e[1] for e in entries])
        if clip_map:
            tr_entries = [e for e in entries if clip_map[e[1]][0] == "train"]
            site_polys = _build_site_train_polylines(tr_entries)
            for e in entries:
                sp_name, guard = clip_map[e[1]]
                jobs.append((*e, sp_name, guard, site_polys))
        else:
            for e in entries:
                jobs.append((*e, None, 0, None))

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    n_workers = max(1, min(2, n_gpus if n_gpus > 0 else (2 if onnx_path is not None else 1)))
    ext_pool: "queue.Queue[RTMPoseONNXExtractor]" = queue.Queue()
    active_provs: List[str] = []
    for w in range(n_workers):
        ext = RTMPoseONNXExtractor(onnx_path=onnx_path, batch_size=256, device_id=w)
        if w == 0:
            active_provs = ext.active_providers
        ext_pool.put(ext)
    print(f"[PINNS] Launching {len(jobs)} clips on {n_workers} worker(s), backend={active_provs}", flush=True)

    t_p0 = time.perf_counter()

    def _run_clip(idx_job: int, jb: Tuple) -> Tuple[int, str, List[Dict]]:
        _, _, stem, lbl_f, ori_f, bev_f, cal, f_sp, f_gd, o_poly = jb
        ext = ext_pool.get()
        try:
            samps, _ = process_pinns_sequence(
                seq_stem=stem, lbl_path=lbl_f, ori_video_path=ori_f, bev_video_path=bev_f,
                calib=cal, pose_ext=ext, window_stride=window_stride,
                forced_split=f_sp, min_t0_guard=f_gd, override_polys=o_poly,
            )
            return idx_job, stem, samps
        finally:
            ext_pool.put(ext)

    results_ordered: List[Tuple[int, str, List[Dict]]] = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(_run_clip, i, jb) for i, jb in enumerate(jobs)]
        for done_n, fut in enumerate(as_completed(futs), 1):
            idx_j, st_name, s_list = fut.result()
            results_ordered.append((idx_j, st_name, s_list))
            if done_n % 5 == 0 or done_n == len(jobs):
                print(f"  [PINNS {done_n}/{len(jobs)}] last={st_name} ({len(s_list)} win) elapsed={time.perf_counter()-t_p0:.1f}s", flush=True)

    results_ordered.sort(key=lambda x: x[0])
    all_samples = [s for _, _, s_list in results_ordered for s in s_list]
    seq_counts = {st_name: len(s_list) for _, st_name, s_list in results_ordered}

    leak_ok, leak_msg = verify_zero_temporal_leakage(all_samples, max_hist_steps=19, fut_steps=T_FUT)
    if not leak_ok:
        raise RuntimeError(f"CRITICAL LEAKAGE AUDIT FAILURE: {leak_msg}")

    split_dist = {sp: sum(1 for s in all_samples if s["split"] == sp) for sp in ("train", "val", "test")}
    audit = {
        "total_samples": len(all_samples),
        "num_sequences": len(seq_counts),
        "split_counts": split_dist,
        "leakage_check": leak_msg,
        "rtmpose_providers": active_provs,
        "bbox_is_topleft": False,
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
