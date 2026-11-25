"""Five-seed Ours_O2_Full +1 video-frame retraining sensitivity & CUDA/CPU replay audit on PINNS."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

import cv2
import numpy as np
import torch

from scripts.benchmark_heldout_sites import HISTORY
from scripts.cache_pinns_frame_offset import CLIPS, MODEL_SHA
from scripts.run_heldout_cloud import DRIVE_ID, MODEL_URL, PINNS_URL, download_zip, extract_member, pip_install
from scripts.run_pinns_matched_o2_multiduration import SITES, paired_site_clip_ci
from scripts.run_pinns_o2_joint_actions_5seed import ACTIONS, SEEDS
from scripts.run_unified_o2_full_gpu_benchmark import fit_unified_fold, load_unified_cohort, select_unified_scales
from src.data.pinns_extractor import RTMPoseONNXExtractor
from src.metrics.metrics_and_stats import _binary_average_precision
from src.models.bioeq_neural import biomech_sequence_torch


def stage_three_us_videos_and_model(scratch: Path) -> tuple[Path, Path]:
    scratch.mkdir(parents=True, exist_ok=True)
    if importlib.util.find_spec("onnxruntime") is None:
        pip_install("onnxruntime-gpu")
    pinns, model_zip = scratch / "pinns.zip", scratch / "rtmpose_m.zip"
    download_zip(PINNS_URL, pinns, 973657335, DRIVE_ID)
    download_zip(MODEL_URL, model_zip, 50799818)
    ds = scratch / "pinns_subset"
    wanted_mp4 = {f"videos/{c.rsplit('_', 1)[0]}/videos/ori_{c.rsplit('_', 1)[1]}.mp4" for c in CLIPS}
    with zipfile.ZipFile(pinns) as arc:
        for name in arc.namelist():
            if re.fullmatch(r"labels/[A-Za-z0-9_]+_[0-9]+\.txt", name) or name.endswith("/calib.json") or name in wanted_mp4:
                extract_member(arc, name, ds / name)
    onnx_p = scratch / "end2end.onnx"
    with zipfile.ZipFile(model_zip) as arc:
        hit = next(k for k in arc.namelist() if PurePosixPath(k).name == "end2end.onnx" and "__MACOSX" not in PurePosixPath(k).parts)
        extract_member(arc, hit, onnx_p)
    if hashlib.sha256(onnx_p.read_bytes()).hexdigest() != MODEL_SHA:
        raise ValueError("RTMPose-m ONNX SHA-256 mismatch")
    return ds, onnx_p


def build_matched_offset_cohorts(ds_root: Path, pose_npz: Path, joint_zip: Path, onnx_p: Path) -> tuple[dict, dict, dict]:
    base_data = load_unified_cohort(ds_root, pose_npz, joint_zip)
    with np.load(pose_npz, allow_pickle=False) as ar:
        p = {k: ar[k] for k in ar.files}
    kp_orig = p["hist_kpts"].copy()
    kp_same0 = p["hist_kpts"].copy()
    kp_plus1 = p["hist_kpts"].copy()
    bbox_all = p["hist_bbox"]

    cv2.setNumThreads(1)
    extractor = RTMPoseONNXExtractor(onnx_path=onnx_p, batch_size=32)
    clip_stats = {}
    for clip, exp_w in CLIPS.items():
        rows = np.flatnonzero(p["clip"] == clip)
        if len(rows) != exp_w:
            raise ValueError(f"Expected {exp_w} rows for {clip}, got {len(rows)}")
        req0, req1 = defaultdict(list), defaultdict(list)
        for r in rows:
            for ti, fr in enumerate(range(int(p["t0"][r]) - HISTORY + 1, int(p["t0"][r]) + 1)):
                bx = bbox_all[r, ti]
                if np.isfinite(bx).all() and min(bx[2:]) > 0:
                    req0[3 * fr].append((r, ti))
                    req1[3 * fr + 1].append((r, ti))
        cond, num = clip.rsplit("_", 1)
        vid = ds_root / "videos" / cond / "videos" / f"ori_{num}.mp4"
        cap = cv2.VideoCapture(str(vid))
        max_raw = max(max(req0), max(req1))
        try:
            for raw in range(max_raw + 1):
                if not cap.grab():
                    raise ValueError(f"Failed grab at {raw} in {vid}")
                if raw not in req0 and raw not in req1:
                    continue
                ok, img = cap.retrieve()
                if not ok:
                    raise ValueError(f"Failed retrieve at {raw} in {vid}")
                if raw in req0:
                    sel0 = tuple(np.array(req0[raw], dtype=np.int64).T)
                    kp_same0[sel0] = extractor.extract_crops(img, bbox_all[sel0])
                if raw in req1:
                    sel1 = tuple(np.array(req1[raw], dtype=np.int64).T)
                    kp_plus1[sel1] = extractor.extract_crops(img, bbox_all[sel1])
        finally:
            cap.release()
        d_vs_frozen = np.abs(kp_same0[rows] - kp_orig[rows])
        d_1_vs_0 = np.abs(kp_plus1[rows] - kp_same0[rows])
        clip_stats[clip] = {
            "windows": int(len(rows)),
            "same0_vs_frozen_xy_max_px": float(d_vs_frozen[..., :2].max()),
            "same0_vs_frozen_conf_max": float(d_vs_frozen[..., 2].max()),
            "plus1_vs_same0_xy_median_px": float(np.median(np.linalg.norm(kp_plus1[rows][..., :2] - kp_same0[rows][..., :2], axis=-1))),
            "plus1_vs_same0_xy_p95_px": float(np.percentile(np.linalg.norm(kp_plus1[rows][..., :2] - kp_same0[rows][..., :2], axis=-1), 95)),
            "changed_frames": int(np.any(d_1_vs_0 > 1e-6, axis=(-2, -1)).sum()),
        }

    from scripts.benchmark_heldout_sites import load_site_windows
    groups, _, _ = load_site_windows(ds_root)
    g_keys = [(str(r.clip), int(r.person_id), int(r.t0)) for s in sorted(groups) for r in groups[s]]
    p_map = {(str(c), int(pid), int(t)): i for i, (c, pid, t) in enumerate(zip(p["clip"], p["person_id"], p["t0"]))}
    order = np.array([p_map[k] for k in g_keys], dtype=np.int64)
    bx_ord = torch.from_numpy(bbox_all[order].astype(np.float32))
    bs0, qb0 = biomech_sequence_torch(torch.from_numpy(kp_same0[order].astype(np.float32)), bx_ord)
    bs1, qb1 = biomech_sequence_torch(torch.from_numpy(kp_plus1[order].astype(np.float32)), bx_ord)
    data0 = {**base_data, "bio_seq": bs0, "q_bio": qb0}
    data1 = {**base_data, "bio_seq": bs1, "q_bio": qb1}
    return data0, data1, {"active_providers": extractor.active_providers, "clips": clip_stats}


@torch.inference_mode()
def _eval_net(net, d: dict, ids: np.ndarray, bs: float, gs: float, dev: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    oo = net(
        d["history"][ids].to(dev),
        d["bio_seq"][ids].to(dev),
        d["q_bio"][ids].to(dev),
        d["edges"][ids].to(dev),
        d["is_car"][ids].to(dev),
        d["nbr_mask"][ids].to(dev),
        bio_scale=bs,
        graph_scale=gs,
        use_action_bridge=True,
    )
    pw = oo["paths"].cpu().numpy()
    pr = oo["logits"].softmax(1).cpu().numpy()
    ap = torch.sigmoid(oo["act_logits"]).cpu().numpy().astype(np.float64)
    bay = np.einsum("nk,nktw->ntw", pr, pw)
    tr = d["truth"][ids].numpy()
    db = np.linalg.norm(bay - tr, axis=-1)
    da = np.linalg.norm(pw - tr[:, None], axis=-1)
    return np.stack((db.mean(1), db[:, -1], da[:, :, -1].min(1)), axis=-1), bay, ap


def _macro_ap(d: dict, ap_pred: np.ndarray) -> float:
    vh = d["v_hist_mean"]
    pe = ap_pred.copy()
    pe[:, 0] = np.where(vh <= 0.20, pe[:, 0], 0.0)
    pe[:, 1] = np.where(vh >= 0.60, pe[:, 1], 0.0)
    ya, va = d["y_act"].numpy(), d["valid_act"].numpy()
    s_aps = []
    for s in SITES:
        sm = d["site"] == s
        s_aps.append(float(np.mean([_binary_average_precision(ya[sm & va[:, c], c].astype(np.float64), pe[sm & va[:, c], c]) for c in range(5)])))
    return float(np.mean(s_aps))


def run_offset_retrain_suite(scratch_dir: Path, pose_npz: Path, joint_zip: Path, out_dir: Path, seeds: tuple[int, ...] = SEEDS) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_root, onnx_p = stage_three_us_videos_and_model(scratch_dir)
    data0, data1, pose_prov = build_matched_offset_cohorts(ds_root, pose_npz, joint_zip, onnx_p)
    n_tot = len(data0["site"])
    affected_597 = np.isin(data0["clip"], list(CLIPS))
    unaffected_us_219 = (data0["site"] == "america_crossroad") & (~affected_597)
    if (int(affected_597.sum()), int(unaffected_us_219.sum())) != (597, 219):
        raise ValueError("Expected 597 affected and 219 unaffected US windows")

    cells = ("train0_test0", "train0_test1", "train1_test1", "train1_test0")
    errs = {k: np.zeros((n_tot, 3), dtype=np.float64) for k in cells}
    aps = {k: np.zeros((n_tot, 5), dtype=np.float64) for k in cells}
    cuda_cpu_audit = {}

    for s_hold in SITES:
        tr_ids = np.flatnonzero(data0["site"] != s_hold)
        te_ids = np.flatnonzero(data0["site"] == s_hold)
        for seed in seeds:
            bs0, gs0 = select_unified_scales(seed, data0, tr_ids, device)
            n0 = fit_unified_fold(seed, data0, tr_ids, device)
            bs1, gs1 = select_unified_scales(seed, data1, tr_ids, device)
            n1 = fit_unified_fold(seed, data1, tr_ids, device)
            for tag, net, td, b_s, g_s in (
                ("train0_test0", n0, data0, bs0, gs0),
                ("train0_test1", n0, data1, bs0, gs0),
                ("train1_test1", n1, data1, bs1, gs1),
                ("train1_test0", n1, data0, bs1, gs1),
            ):
                e_t, _, a_t = _eval_net(net, td, te_ids, b_s, g_s, device)
                errs[tag][te_ids] += e_t / len(seeds)
                aps[tag][te_ids] += a_t / len(seeds)
            if s_hold == "america_crossroad" and seed == seeds[0]:
                u_ids = np.flatnonzero(unaffected_us_219)
                _, bay_dev0, _ = _eval_net(n0, data0, u_ids, bs0, gs0, device)
                _, bay_dev1, _ = _eval_net(n0, data1, u_ids, bs0, gs0, device)
                from src.models.o2_unified_full import O2UnifiedFullForecaster
                n0_cpu = O2UnifiedFullForecaster().eval()
                n0_cpu.load_state_dict({k: v.cpu() for k, v in n0.state_dict().items()})
                _, bay_cpu0, _ = _eval_net(n0_cpu, data0, u_ids, bs0, gs0, torch.device("cpu"))
                diff_cc = np.abs(bay_dev0 - bay_cpu0)
                cuda_cpu_audit = {
                    "unaffected_us_windows": 219,
                    "same_device_offset1_minus_offset0_exact_zero": bool(np.array_equal(bay_dev0, bay_dev1)),
                    "same_device_max_abs_m": float(np.max(np.abs(bay_dev0 - bay_dev1))),
                    "cuda_minus_cpu_same_weights_exact_equal": bool(np.array_equal(bay_dev0, bay_cpu0)),
                    "cuda_minus_cpu_max_abs_m": float(diff_cc.max()),
                    "cuda_minus_cpu_p95_abs_m": float(np.percentile(diff_cc, 95)),
                    "runtime_device": str(device),
                }
    return _pack_offset_results(data0, affected_597, unaffected_us_219, errs, aps, pose_prov, cuda_cpu_audit, out_dir, seeds)


def _pack_offset_results(data0, affected_597, unaffected_219, errs, aps, pose_prov, cuda_cpu_audit, out_dir: Path, seeds) -> dict:
    site_arr, clip_arr = data0["site"], data0["clip"]
    matrix = {}
    for k, e in errs.items():
        by_s = {s: {"ade_m": float(e[site_arr == s, 0].mean()), "fde_m": float(e[site_arr == s, 1].mean()), "min_fde6_m": float(e[site_arr == s, 2].mean())} for s in SITES}
        matrix[k] = {
            "site_equal_ade_m": float(np.mean([v["ade_m"] for v in by_s.values()])),
            "site_equal_fde_m": float(np.mean([v["fde_m"] for v in by_s.values()])),
            "site_equal_min_fde6_m": float(np.mean([v["min_fde6_m"] for v in by_s.values()])),
            "site_equal_macro_ap": _macro_ap(data0, aps[k]),
            "affected_597_ade_m": float(e[affected_597, 0].mean()),
            "affected_597_fde_m": float(e[affected_597, 1].mean()),
            "unaffected_us_219_ade_m": float(e[unaffected_219, 0].mean()),
            "by_site": by_s,
        }
    agg = {
        "study": "Five-seed Ours_O2_Full +1 video-frame retraining sensitivity (2x2 train/test matrix) & CUDA/CPU replay audit",
        "seeds": list(seeds),
        "pose_extraction_provenance": pose_prov,
        "cuda_vs_cpu_unaffected_219_audit": cuda_cpu_audit,
        "train_test_2x2_matrix": matrix,
        "paired_site_clip_ci": {
            "retrained_11_minus_baseline_00": paired_site_clip_ci(errs["train1_test1"][:, :2] - errs["train0_test0"][:, :2], site_arr, clip_arr),
            "frozen_01_minus_baseline_00": paired_site_clip_ci(errs["train0_test1"][:, :2] - errs["train0_test0"][:, :2], site_arr, clip_arr),
        },
        "affected_597_window_delta_stats": {
            "retrained_11_minus_00_fde_p95_abs_m": float(np.percentile(np.abs(errs["train1_test1"][affected_597, 1] - errs["train0_test0"][affected_597, 1]), 95)),
            "frozen_01_minus_00_fde_p95_abs_m": float(np.percentile(np.abs(errs["train0_test1"][affected_597, 1] - errs["train0_test0"][affected_597, 1]), 95)),
        },
    }
    agg_p, md_p, man_p = out_dir / "report.json", out_dir / "RESULTS.md", out_dir / "sha256.json"
    agg_p.write_text(json.dumps(agg, indent=2) + "\n")
    m00, m11 = matrix["train0_test0"], matrix["train1_test1"]
    md_p.write_text(
        f"# `+1` Video-Frame Retraining Sensitivity & CUDA/CPU Audit (5-Seed `Ours_O2_Full`)\n\n"
        f"- **Baseline `Train(0)->Test(0)`**: `{m00['site_equal_ade_m']:.6f} / {m00['site_equal_fde_m']:.6f} m` (Macro-AP `{m00['site_equal_macro_ap']:.4f}`).\n"
        f"- **Retrained `Train(+1)->Test(+1)`**: `{m11['site_equal_ade_m']:.6f} / {m11['site_equal_fde_m']:.6f} m` (Macro-AP `{m11['site_equal_macro_ap']:.4f}`).\n"
        f"- **CUDA vs CPU 219 Unaffected US Windows**: same-device max diff `{cuda_cpu_audit['same_device_max_abs_m']:.3e} m`, CUDA-CPU max diff `{cuda_cpu_audit['cuda_minus_cpu_max_abs_m']:.3e} m`.\n"
    )
    mani = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (agg_p, md_p)}
    man_p.write_text(json.dumps(mani, indent=2) + "\n")
    with zipfile.ZipFile(out_dir / "pinns_frame_offset_retrain_5seed.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in (agg_p, md_p, man_p):
            zf.write(p, arcname=p.name)
    return agg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch-dir", type=Path, required=True)
    ap.add_argument("--pose-npz", type=Path, required=True)
    ap.add_argument("--joint-zip", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    run_offset_retrain_suite(args.scratch_dir, args.pose_npz, args.joint_zip, args.out_dir)
