"""Zero-shot 5-seed O(2) BioEq & clock-aware transfer on 54 DAIR-V2X Beijing roadside windows."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

import cv2
import numpy as np
import torch

from scripts.run_heldout_cloud import MODEL_SHA, MODEL_URL, download_zip
from scripts.run_pinns_bioeq_benchmark import prepare_cohort
from scripts.run_pinns_matched_o2_multiduration import SITES, fit_motion_and_bio
from src.data.pinns_extractor import RTMPoseONNXExtractor
from src.models.biomech_equivariant import causal_frenet_frame, fit_biomech_equivariant, parity_motion_features
from src.models.bioeq_neural import biomech_sequence_torch
from src.models.bioeq_o2_joint import O2BioEqJointForecaster

SEEDS = (42, 1024, 2025, 3407, 4096)
COCO_SWAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
SPLIT_URL = "https://raw.githubusercontent.com/AIR-THU/DAIR-V2X/c885c54af0c34bc515fa9ca8b5e8fda76a15462c/data/split_datas/cooperative-split-data-spd.json"
SPLIT_SHA = "0a59332221f3618639f34fb70678d03c5221ab80fd69f2cf41acffca0d8859fd"


def stage_dair_54(repo_root: Path, work_dir: Path, onnx_path: Path) -> dict:
    src_dir = repo_root / "reviews" / "dair_roadside_external"
    exp_dir = work_dir / "dair_exp"
    exp_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("full_range.py", "third_coverage.py", "third_cohort_freeze.json", "expansion_freeze.txt"):
        shutil.copy2(src_dir / fname, exp_dir / fname)
    raw_split = urllib.request.urlopen(SPLIT_URL, timeout=60).read()
    if hashlib.sha256(raw_split).hexdigest() != SPLIT_SHA:
        raise ValueError("Unexpected DAIR-V2X split SHA-256")
    (exp_dir / "full_split.json").write_bytes(raw_split)
    env = {**os.environ, "PYTHONPATH": f"{exp_dir}:{repo_root}", "CUDA_VISIBLE_DEVICES": ""}
    subprocess.run([sys.executable, "-u", str(exp_dir / "third_coverage.py"), "--stage-past"], cwd=exp_dir, env=env, check=True)

    cohort = json.loads((exp_dir / "private_third_cohort.json").read_text())
    rec = cohort["records"]
    if len(rec) != 54:
        raise ValueError(f"Expected 54 DAIR windows, found {len(rec)}")
    pose_ext = RTMPoseONNXExtractor(onnx_path=onnx_path, batch_size=1)
    hxy_l, truth_l, kpts_l, box_l, w_l, ts_l, actors, sites = [], [], [], [], [], [], [], []
    for r in rec:
        gxy = np.asarray(r["ground_xy_m"], dtype=np.float64)
        hxy = gxy[:5] - gxy[4:5]
        truth = gxy[5:] - gxy[4:5]
        boxes = np.asarray(r["boxes_cxcywh"], dtype=np.float32)
        j_frames = []
        for img_rel, bx, wh in zip(r["image_paths"], boxes, r["images_wh"]):
            img = cv2.imread(str(exp_dir / "private_expansion" / "images" / img_rel), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if img is None or (img.shape[1], img.shape[0]) != tuple(wh):
                raise ValueError(f"Image decode mismatch on {img_rel}")
            j_frames.append(pose_ext.extract_crops(img, bx[None])[0])
        hxy_l.append(hxy)
        truth_l.append(truth)
        kpts_l.append(np.stack(j_frames, axis=0))
        box_l.append(boxes)
        w_l.append(float(r["images_wh"][0][0]))
        ts_l.append(np.asarray(r["timestamps_us"], dtype=np.int64))
        actors.append(f"{r['key'][0]}:{r['key'][1]}")
        sites.append(str(r["site"]))
    return {
        "hist": np.stack(hxy_l, axis=0),
        "truth": np.stack(truth_l, axis=0),
        "kpts": np.stack(kpts_l, axis=0),
        "bbox": np.stack(box_l, axis=0),
        "img_w": np.array(w_l, dtype=np.float32),
        "ts_us": np.stack(ts_l, axis=0),
        "actors": np.array(actors, dtype=str),
        "sites": np.array(sites, dtype=str),
    }


def reduce_dair(err: np.ndarray, mask: np.ndarray, actors: np.ndarray) -> dict:
    ids = np.flatnonzero(mask)
    if len(ids) == 0:
        return {"windows": 0, "people": 0, "window_equal": None, "person_equal": None}
    uniq = sorted(set(actors[ids].tolist()))
    we = err[ids].mean(axis=0)
    pe = np.mean([err[ids[actors[ids] == p]].mean(axis=0) for p in uniq], axis=0)
    return {
        "windows": int(len(ids)),
        "people": int(len(uniq)),
        "window_equal": {"ade_m": float(we[0]), "fde_m": float(we[1])},
        "person_equal": {"ade_m": float(pe[0]), "fde_m": float(pe[1])},
    }


def person_bootstrap_ci(diff: np.ndarray, mask: np.ndarray, actors: np.ndarray, seed: int = 2026) -> dict:
    ids = np.flatnonzero(mask)
    uniq = sorted(set(actors[ids].tolist()))
    rows_by_p = {p: ids[actors[ids] == p] for p in uniq}
    rng = np.random.default_rng(seed)
    pt_w = diff[ids].mean(axis=0)
    pt_p = np.mean([diff[rows_by_p[p]].mean(axis=0) for p in uniq], axis=0)
    dw = np.empty((20000, 2), dtype=np.float64)
    dp = np.empty((20000, 2), dtype=np.float64)
    for r in range(20000):
        picked = [uniq[j] for j in rng.integers(0, len(uniq), size=len(uniq))]
        pooled = np.concatenate([rows_by_p[p] for p in picked])
        dw[r] = diff[pooled].mean(axis=0)
        dp[r] = np.mean([diff[rows_by_p[p]].mean(axis=0) for p in picked], axis=0)
    lw, hw = np.percentile(dw, [2.5, 97.5], axis=0)
    lp, hp = np.percentile(dp, [2.5, 97.5], axis=0)
    return {
        "window_equal": {"mean": {"ade_m": float(pt_w[0]), "fde_m": float(pt_w[1])}, "ci95": {"ade_m": [float(lw[0]), float(hw[0])], "fde_m": [float(lw[1]), float(hw[1])]}},
        "person_equal": {"mean": {"ade_m": float(pt_p[0]), "fde_m": float(pt_p[1])}, "ci95": {"ade_m": [float(lp[0]), float(hp[0])], "fde_m": [float(lp[1]), float(hp[1])]}},
    }


@torch.inference_mode()
def score_transform(
    dair: dict,
    ridge_models: list[dict],
    ensemble: list[tuple[O2BioEqJointForecaster, O2BioEqJointForecaster, float]],
    op: np.ndarray,
    reflect: bool,
) -> dict[str, np.ndarray]:
    hist_g = dair["hist"] @ op.T
    truth = dair["truth"]
    kp_g = dair["kpts"].copy()
    if reflect:
        kp_g[..., 0] = dair["img_w"][:, None, None] - kp_g[..., 0]
        kp_g = kp_g[:, :, COCO_SWAP, :]
    bs_t, qb_t = biomech_sequence_torch(torch.from_numpy(kp_g), torch.from_numpy(dair["bbox"]))
    hist_t = torch.from_numpy(hist_g)

    ts = dair["ts_us"]
    obs_dt = np.diff(ts[:, :5], axis=1) / 1e6
    fut_s = (ts[:, 5:] - ts[:, 4:5]) / 1e6
    obs_dt_t = torch.from_numpy(obs_dt)
    fut_s_t = torch.from_numpy(fut_s)

    v_last_meas = (hist_g[:, -1:] - hist_g[:, -2:-1]) / obs_dt[:, -1:, None]
    cv_meas = (v_last_meas * fut_s[:, :, None]) @ op
    cv_nom = ((hist_g[:, -1:] - hist_g[:, -2:-1]) * np.arange(1, 31, dtype=np.float64)[None, :, None]) @ op

    # Clock-scaled local features for O(2) ridge
    hist_clk = np.zeros_like(hist_g)
    steps_01 = (obs_dt / 0.1)
    for t in range(3, -1, -1):
        hist_clk[:, t] = hist_clk[:, t + 1] - (hist_g[:, t + 1] - hist_g[:, t]) / steps_01[:, t : t + 1]
    _, basis, valid = causal_frenet_frame(hist_clk)
    even, odd, _ = parity_motion_features(hist_clk)
    from src.models.biomech_equivariant import _pred_even, _pred_odd
    r_errs = []
    for rm in ridge_models:
        par = np.where(valid[:, None], rm["g_mot"] * _pred_even(rm["fe"], even), 0.0)
        perp = np.where(valid[:, None], rm["g_mot"] * _pred_odd(rm["fo"], odd), 0.0)
        loc = np.stack((par, perp), axis=-1)
        pred_r = cv_meas + (np.einsum("nts,nws->ntw", loc, basis) @ op)
        d = np.linalg.norm(pred_r - truth, axis=-1)
        r_errs.append(np.stack((d.mean(1), d[:, -1]), axis=-1))

    acc = {k: [] for k in ("mot_nom_top1", "bio_nom_top1", "mot_clk_top1", "bio_clk_top1", "mot_clk_bayes", "bio_clk_bayes")}
    for mot, bio, sc in ensemble:
        for prefix, dt_arg, ft_arg in (("nom", None, None), ("clk", obs_dt_t, fut_s_t)):
            for mtag, net, q_mult in (("mot", mot, 0.0), ("bio", bio, sc)):
                out = net(hist_t, bs_t, qb_t * q_mult, obs_dt=dt_arg, fut_times_s=ft_arg)
                paths_w = out["paths"].numpy() @ op
                pr = out["logits"].softmax(dim=1).numpy()
                top = paths_w[np.arange(len(hist_g)), pr.argmax(1)]
                d_top = np.linalg.norm(top - truth, axis=-1)
                acc[f"{mtag}_{prefix}_top1"].append(np.stack((d_top.mean(1), d_top[:, -1]), axis=-1))
                if prefix == "clk":
                    bay = np.einsum("nk,nktw->ntw", pr, paths_w)
                    d_bay = np.linalg.norm(bay - truth, axis=-1)
                    acc[f"{mtag}_clk_bayes"].append(np.stack((d_bay.mean(1), d_bay[:, -1]), axis=-1))

    def err2(pred: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(pred - truth, axis=-1)
        return np.stack((d.mean(1), d[:, -1]), axis=-1)

    return {
        "cv_measured": err2(cv_meas),
        "cv_nominal": err2(cv_nom),
        "o2_ridge_clock_aware": np.mean(r_errs, axis=0),
        **{k: np.mean(v, axis=0) for k, v in acc.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pinns_metadata", type=Path)
    ap.add_argument("pose_npz", type=Path)
    ap.add_argument("output_zip", type=Path)
    ap.add_argument("--onnx", type=Path, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()

    torch.set_num_threads(4)
    repo_root = Path(__file__).resolve().parents[1]
    tensors, idx_by_site, keys, _ = prepare_cohort(args.pinns_metadata, args.pose_npz)
    site_arr = np.empty(len(keys), dtype="<U24")
    for s, ids in idx_by_site.items():
        site_arr[ids] = s
    pinns_data = {
        "site": site_arr,
        "hist20": tensors["history"],
        "truth30": tensors["truth"],
        "bio_seq": tensors["bio_seq"],
        "q_bio": tensors["q_bio"],
    }

    ridge_models = []
    hist_np = tensors["history"].numpy()
    truth_np = tensors["truth"].numpy()
    for s in SITES:
        tr = site_arr != s
        ridge_models.append(fit_biomech_equivariant(hist_np[tr], truth_np[tr], site_arr[tr]))

    ensemble = []
    for seed in args.seeds:
        for s in SITES:
            tr_ids = np.flatnonzero(site_arr != s)
            ensemble.append(fit_motion_and_bio(seed, 5, pinns_data, tr_ids))
        print(f"Fitted PINNS 4-fold O(2) BioEq for seed={seed}", flush=True)

    with tempfile.TemporaryDirectory(prefix="dair_o2_") as tmp:
        tmp_p = Path(tmp)
        onnx_p = args.onnx
        if onnx_p is None or not onnx_p.is_file():
            mzip = tmp_p / "rtmpose.zip"
            download_zip(MODEL_URL, mzip, 50799818)
            with zipfile.ZipFile(mzip) as zf:
                cand = [n for n in zf.namelist() if Path(n).name == "end2end.onnx" and "__MACOSX" not in Path(n).parts]
                blob = zf.read(cand[0])
                if hashlib.sha256(blob).hexdigest() != MODEL_SHA:
                    raise ValueError("RTMPose ONNX SHA-256 mismatch")
                onnx_p = tmp_p / "end2end.onnx"
                onnx_p.write_bytes(blob)
        dair = stage_dair_54(repo_root, tmp_p, onnx_p)

    all_m = np.ones(len(dair["actors"]), dtype=bool)
    strict_m = np.max(np.abs(np.diff(dair["ts_us"][:, :5], axis=1) - 100000), axis=1) <= 1000
    transforms = [
        ("0", np.array([[1.0, 0.0], [0.0, 1.0]]), False),
        ("90", np.array([[0.0, -1.0], [1.0, 0.0]]), False),
        ("180", np.array([[-1.0, 0.0], [0.0, -1.0]]), False),
        ("270", np.array([[0.0, 1.0], [-1.0, 0.0]]), False),
        ("reflect_y", np.array([[1.0, 0.0], [0.0, -1.0]]), True),
    ]
    by_tf, zero_errs = {}, None
    for name, op, ref in transforms:
        errs = score_transform(dair, ridge_models, ensemble, op, ref)
        if name == "0":
            zero_errs = errs
        by_tf[name] = {
            "all_54": {k: reduce_dair(v, all_m, dair["actors"]) for k, v in errs.items()},
            "strict_history_1ms_31": {k: reduce_dair(v, strict_m, dair["actors"]) for k, v in errs.items()},
        }
    max_swing_ade = float(max(
        abs(by_tf[a]["all_54"]["bio_clk_top1"]["window_equal"]["ade_m"] - by_tf["0"]["all_54"]["bio_clk_top1"]["window_equal"]["ade_m"])
        for a, _, _ in transforms
    ))
    by_inter = {
        loc: {k: reduce_dair(v, dair["sites"] == loc, dair["actors"]) for k, v in zero_errs.items()}
        for loc in sorted(set(dair["sites"].tolist()))
    }
    report = {
        "windows": int(len(dair["actors"])),
        "strict_history_1ms_windows": int(strict_m.sum()),
        "seeds": list(args.seeds),
        "max_o2_transform_ade_swing_m": max_swing_ade,
        "transforms": by_tf,
        "by_intersection_deg0": by_inter,
        "paired_bootstrap_all54": {
            "bio_clk_minus_mot_clk_top1": person_bootstrap_ci(zero_errs["bio_clk_top1"] - zero_errs["mot_clk_top1"], all_m, dair["actors"]),
            "bio_clk_minus_cv_measured_top1": person_bootstrap_ci(zero_errs["bio_clk_top1"] - zero_errs["cv_measured"], all_m, dair["actors"]),
            "bio_clk_bayes_minus_cv_measured": person_bootstrap_ci(zero_errs["bio_clk_bayes"] - zero_errs["cv_measured"], all_m, dair["actors"]),
            "bio_clk_minus_bio_nom_top1": person_bootstrap_ci(zero_errs["bio_clk_top1"] - zero_errs["bio_nom_top1"], all_m, dair["actors"]),
            "o2_ridge_minus_cv_measured": person_bootstrap_ci(zero_errs["o2_ridge_clock_aware"] - zero_errs["cv_measured"], all_m, dair["actors"]),
        },
        "paired_bootstrap_strict31": {
            "bio_clk_minus_mot_clk_top1": person_bootstrap_ci(zero_errs["bio_clk_top1"] - zero_errs["mot_clk_top1"], strict_m, dair["actors"]),
            "bio_clk_minus_cv_measured_top1": person_bootstrap_ci(zero_errs["bio_clk_top1"] - zero_errs["cv_measured"], strict_m, dair["actors"]),
        },
    }
    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    members = {"report.json": (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()}
    members["sha256.json"] = (json.dumps({k: hashlib.sha256(v).hexdigest() for k, v in members.items()}, indent=2, sort_keys=True) + "\n").encode()
    with zipfile.ZipFile(args.output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for k, v in members.items():
            zf.writestr(k, v)
    print("Saved", args.output_zip, flush=True)


if __name__ == "__main__":
    main()
