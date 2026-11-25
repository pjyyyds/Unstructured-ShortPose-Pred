"""Five-seed 4-site LOSO pose degradation & occlusion stress-test on O(2) BioEq v2 vs Raw-Pose."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from scripts.run_pinns_bioeq_benchmark import SEEDS, prepare_cohort
from scripts.run_pinns_bioeq_v2_benchmark import eval_metrics, fit_backbone, fit_stage2_bio
from scripts.run_pinns_matched_o2_multiduration import SITES
from src.data.features import build_pose_and_quality_features
from src.models.bioeq_neural import biomech_sequence_torch

REGIMES = ("clean", "lower_limb_occluded", "frame_dropout_50pct", "heavy_jitter_025", "complete_blackout")


def corrupt_pose_bank(kpts: np.ndarray, bbox: np.ndarray, img_h: np.ndarray, regime: str, seed: int = 2026) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    kp = kpts.copy()
    bx = bbox.copy()
    rng = np.random.default_rng(seed)
    if regime == "clean":
        pass
    elif regime == "lower_limb_occluded":
        kp[:, :, [13, 14, 15, 16], :] = 0.0
    elif regime == "frame_dropout_50pct":
        drop = np.zeros((len(kp), 5), dtype=bool)
        drop[:, [1, 3]] = True
        kp[drop] = 0.0
        bx[drop] = 0.0
    elif regime == "heavy_jitter_025":
        h = np.maximum(bx[:, :, None, 3:4], 1.0)
        noise = rng.normal(0.0, 0.25, size=kp[..., :2].shape).astype(np.float32) * h
        kp[..., :2] = kp[..., :2] + noise
        kp[..., 2] = np.clip(kp[..., 2] * 0.55, 0.0, 1.0)
    elif regime == "complete_blackout":
        kp.fill(0.0)
        bx.fill(0.0)
    else:
        raise ValueError(regime)

    n = len(kp)
    xp_list, qv_list = [], []
    mask5 = np.ones(5, dtype=np.float32)
    for i in range(n):
        m_f = ((bx[i, :, 2] > 0) & (bx[i, :, 3] > 0)).astype(np.float32) if regime in ("frame_dropout_50pct", "complete_blackout") else mask5
        xp, qv, _, _ = build_pose_and_quality_features(kp[i], bx[i], m_f, float(img_h[i]))
        xp_list.append(xp)
        qv_list.append(qv)
    kp_t, bx_t = torch.from_numpy(kp), torch.from_numpy(bx)
    bs_t, qb_t = biomech_sequence_torch(kp_t, bx_t)
    return torch.from_numpy(np.stack(xp_list)), torch.from_numpy(np.stack(qv_list)), bs_t, qb_t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pinns_metadata", type=Path)
    ap.add_argument("pose_npz", type=Path)
    ap.add_argument("output_zip", type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()

    torch.set_num_threads(4)
    device = torch.device("cpu")
    tensors, idx, keys, _ = prepare_cohort(args.pinns_metadata, args.pose_npz)
    n = len(keys)
    site_arr = np.empty(n, dtype="<U24")
    clip_arr = np.array([k[0] for k in keys], dtype=str)
    for s, rows in idx.items():
        site_arr[rows] = s

    with np.load(args.pose_npz, allow_pickle=False) as pz:
        ck = list(zip(pz["clip"].tolist(), pz["person_id"].tolist(), pz["t0"].tolist()))
        order = {k: i for i, k in enumerate(ck)}
        p_ids = [order[k] for k in keys]
        kpts_raw = pz["hist_kpts"][p_ids].copy()
        bbox_raw = pz["hist_bbox"][p_ids].copy()
        img_h_raw = pz["img_h"][p_ids].copy()

    corrupted = {}
    for reg in REGIMES:
        xp_c, qv_c, bs_c, qb_c = corrupt_pose_bank(kpts_raw, bbox_raw, img_h_raw, reg)
        corrupted[reg] = {**tensors, "pose": xp_c, "quality": qv_c, "bio_seq": bs_c, "q_bio": qb_c}

    from scripts.run_pinns_bioeq_v2_benchmark import select_inner_bio_scale
    from scripts.run_pinns_matched_o2_multiduration import paired_site_clip_ci

    acc = {reg: {"motion": [], "raw_pose": [], "bio_pose": []} for reg in REGIMES}
    for seed in args.seeds:
        cur = {reg: {m: np.zeros((n, 7), dtype=np.float64) for m in ("motion", "raw_pose", "bio_pose")} for reg in REGIMES}
        for s in SITES:
            tr_ids = np.flatnonzero(site_arr != s)
            te_ids = np.flatnonzero(site_arr == s)
            mot_net = fit_backbone(seed, tensors, tr_ids, site_arr, device, arm="motion")
            raw_net = fit_backbone(seed, tensors, tr_ids, site_arr, device, arm="raw_pose")
            bio_net = fit_stage2_bio(mot_net, seed, tensors, tr_ids, site_arr, device)
            sc = select_inner_bio_scale(seed, tensors, tr_ids, site_arr, device)
            for reg in REGIMES:
                ct = corrupted[reg]
                cur[reg]["motion"][te_ids] = eval_metrics(mot_net, ct, te_ids, device, 0.0)
                cur[reg]["raw_pose"][te_ids] = eval_metrics(raw_net, ct, te_ids, device, 0.0)
                cur[reg]["bio_pose"][te_ids] = eval_metrics(bio_net, ct, te_ids, device, sc)
        for reg in REGIMES:
            for m in ("motion", "raw_pose", "bio_pose"):
                acc[reg][m].append(cur[reg][m])
        print(f"Done occlusion stress seed={seed}", flush=True)

    report = {"windows": n, "seeds": list(args.seeds), "regimes": {}}
    for reg in REGIMES:
        m_avg = np.mean(acc[reg]["motion"], axis=0)
        r_avg = np.mean(acc[reg]["raw_pose"], axis=0)
        b_avg = np.mean(acc[reg]["bio_pose"], axis=0)
        def eq(arr: np.ndarray) -> dict:
            v = np.mean([arr[site_arr == s].mean(0) for s in SITES], axis=0)
            return {"ade_m": float(v[0]), "fde_m": float(v[1]), "minfde6_m": float(v[3]), "bayes_fde_m": float(v[6])}
        report["regimes"][reg] = {
            "o2_motion": eq(m_avg),
            "o2_raw_pose": eq(r_avg),
            "o2_bioeq_v2": eq(b_avg),
            "paired_bioeq_minus_raw_pose": paired_site_clip_ci(b_avg[:, :2] - r_avg[:, :2], site_arr, clip_arr),
            "paired_bioeq_minus_motion": paired_site_clip_ci(b_avg[:, :2] - m_avg[:, :2], site_arr, clip_arr),
            "paired_raw_pose_minus_motion": paired_site_clip_ci(r_avg[:, :2] - m_avg[:, :2], site_arr, clip_arr),
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
